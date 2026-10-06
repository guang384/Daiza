//! LayerNorm (带 bias, ViT 风格)
//!
//! 与 RMSNorm 不同,LayerNorm 会减去均值并应用可学习 bias。
//! 公式: `y_i = (x_i - mean) / sqrt(var + eps) * w_i + b_i`
//!
//! 用于 Qwen3-VL ViT encoder 的 norm1/norm2 (eps=1e-6) 和 PatchMerger (eps=1e-6)。

use std::arch::x86_64::*;
use crate::math::simd_exp::{simd_available, hsum_ps};

/// AVX2 内核: 非原地 LayerNorm (src → dst)
///
/// Pass 1: sum + sum_sq 累加 (4x unroll)
/// Pass 2: dst[i] = (src[i] - mean) * inv_std * w[i] + b[i]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn layernorm_into_avx2(
    src: &[f32], dst: &mut [f32],
    w: &[f32], b: &[f32], eps: f32,
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), w.len());
    debug_assert_eq!(src.len(), b.len());
    let n = src.len();
    let nf = n as f32;

    // Pass 1: sum + sum_sq
    let mut sum_v = _mm256_setzero_ps();
    let mut sq_v = _mm256_setzero_ps();
    let mut i = 0;
    while i + 32 <= n {
        for off in (0..32).step_by(8) {
            let v = _mm256_loadu_ps(src.as_ptr().add(i + off));
            sum_v = _mm256_add_ps(sum_v, v);
            sq_v = _mm256_fmadd_ps(v, v, sq_v);
        }
        i += 32;
    }
    while i + 8 <= n {
        let v = _mm256_loadu_ps(src.as_ptr().add(i));
        sum_v = _mm256_add_ps(sum_v, v);
        sq_v = _mm256_fmadd_ps(v, v, sq_v);
        i += 8;
    }
    let mut sum = hsum_ps(sum_v);
    let mut sq = hsum_ps(sq_v);
    for &s in &src[i..n] {
        sum += s;
        sq += s * s;
    }
    let mean = sum / nf;
    let var = sq / nf - mean * mean;
    let inv_std = 1.0 / (var + eps).sqrt();
    // ★ 2-FMA 优化: (x - mean) * inv_std = FMA(x, inv_std, -mean*inv_std)
    //   原 sub+mul+fma = 3 op, 现 fmadd+fmadd = 2 op, 消除 sub/mul 中间步骤
    let mean_neg_inv = -mean * inv_std;
    let mean_v = _mm256_set1_ps(mean_neg_inv);
    let inv_std_v = _mm256_set1_ps(inv_std);

    // Pass 2: dst[i] = FMA(FMA(src[i], inv_std, -mean*inv_std), w[i], b[i])
    let mut i = 0;
    while i + 32 <= n {
        for off in (0..32).step_by(8) {
            let v = _mm256_loadu_ps(src.as_ptr().add(i + off));
            let wv = _mm256_loadu_ps(w.as_ptr().add(i + off));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i + off));
            let centered = _mm256_fmadd_ps(v, inv_std_v, mean_v);
            let r = _mm256_fmadd_ps(centered, wv, bv);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + off), r);
        }
        i += 32;
    }
    while i + 8 <= n {
        let v = _mm256_loadu_ps(src.as_ptr().add(i));
        let wv = _mm256_loadu_ps(w.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        let centered = _mm256_fmadd_ps(v, inv_std_v, mean_v);
        let r = _mm256_fmadd_ps(centered, wv, bv);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), r);
        i += 8;
    }
    for j in i..n {
        dst[j] = (src[j] * inv_std + mean_neg_inv) * w[j] + b[j];
    }
}

/// 非原地 LayerNorm: `dst[i] = (src[i] - mean) / sqrt(var + eps) * w[i] + b[i]`
pub fn layernorm_into(
    src: &[f32], dst: &mut [f32],
    w: &[f32], b: &[f32], eps: f32,
) {
    #[cfg(target_arch = "x86_64")]
    if simd_available() && src.len() >= 8 {
        #[allow(unsafe_code)]
        unsafe {
            layernorm_into_avx2(src, dst, w, b, eps);
            return;
        }
    }
    let n = src.len() as f32;
    let mut sum = 0.0f32;
    let mut sq = 0.0f32;
    for &s in src.iter() {
        sum += s;
        sq += s * s;
    }
    let mean = sum / n;
    let var = sq / n - mean * mean;
    let inv_std = 1.0 / (var + eps).sqrt();
    for (d, (&s, &wi)) in dst.iter_mut().zip(src.iter().zip(w.iter())) {
        *d = (s - mean) * inv_std * wi;
    }
    for (d, &bi) in dst.iter_mut().zip(b.iter()) {
        *d += bi;
    }
}



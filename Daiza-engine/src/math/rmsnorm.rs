//! RMSNorm(Qwen3 / LLaMA 风格的均方根归一化)
//!
//! 公式:`y_i = x_i * w_i / sqrt(mean(x^2) + eps)`
//!
//! - `eps = 1e-6`(从 `qwen35.attention.layer_norm_rms_epsilon` 读取)
//! - `w` 是可学习的 scale 向量,在 GGUF 中以 F32 存储

use std::arch::x86_64::*;
use crate::math::simd_exp::{simd_available, hsum_ps};

/// AVX2 内核: 原地 RMSNorm
///
/// Pass 1: ss = sum(x^2) — 4x unroll FMA,隐藏 latency
/// Pass 2: x[i] = x[i] * inv_rms * w[i] — 8-wide mul
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn rmsnorm_inplace_avx2(x: &mut [f32], w: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), w.len());
    let n = x.len();
    let nf = n as f32;

    // Pass 1: ss 累加 (4x unroll)
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    let mut i = 0;
    let n32 = (n / 32) * 32;
    while i < n32 {
        let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(x.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(x.as_ptr().add(i + 24));
        sum0 = _mm256_fmadd_ps(v0, v0, sum0);
        sum1 = _mm256_fmadd_ps(v1, v1, sum1);
        sum2 = _mm256_fmadd_ps(v2, v2, sum2);
        sum3 = _mm256_fmadd_ps(v3, v3, sum3);
        i += 32;
    }
    let n8 = (n / 8) * 8;
    while i < n8 {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(v, v, sum0);
        i += 8;
    }
    sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    let mut ss = hsum_ps(sum0);
    // tail (n 不是 8 的倍数时)
    for &v in &x[n8..n] {
        ss += v * v;
    }

    let inv_rms = 1.0 / ((ss / nf + eps).sqrt());
    let inv_rms_v = _mm256_set1_ps(inv_rms);

    // Pass 2: x[i] = x[i] * inv_rms * w[i]
    // 用 FMA: x * (inv_rms * w) 一次性, 但 w 需要预乘 inv_rms;
    // 或者两步 mul: (x * inv_rms) * w — 额外 mul 但避免 w 预乘
    // 选两步 mul (无 inv_rms*w 预计算 buffer)
    let mut i = 0;
    while i + 32 <= n {
        for off in (0..32).step_by(8) {
            let v = _mm256_loadu_ps(x.as_ptr().add(i + off));
            let wv = _mm256_loadu_ps(w.as_ptr().add(i + off));
            let r = _mm256_mul_ps(_mm256_mul_ps(v, inv_rms_v), wv);
            _mm256_storeu_ps(x.as_mut_ptr().add(i + off), r);
        }
        i += 32;
    }
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        let wv = _mm256_loadu_ps(w.as_ptr().add(i));
        let r = _mm256_mul_ps(_mm256_mul_ps(v, inv_rms_v), wv);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), r);
        i += 8;
    }
    for j in i..n {
        x[j] = x[j] * inv_rms * w[j];
    }
}

/// AVX2 内核: 非原地 RMSNorm (src → dst,省一次拷贝)
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn rmsnorm_into_avx2(src: &[f32], dst: &mut [f32], w: &[f32], eps: f32) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), w.len());
    let n = src.len();
    let nf = n as f32;

    // Pass 1: ss = sum(src^2), 同时不拷贝 (Pass 2 才写 dst)
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    let mut i = 0;
    let n32 = (n / 32) * 32;
    while i < n32 {
        let v0 = _mm256_loadu_ps(src.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(src.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(src.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(src.as_ptr().add(i + 24));
        sum0 = _mm256_fmadd_ps(v0, v0, sum0);
        sum1 = _mm256_fmadd_ps(v1, v1, sum1);
        sum2 = _mm256_fmadd_ps(v2, v2, sum2);
        sum3 = _mm256_fmadd_ps(v3, v3, sum3);
        i += 32;
    }
    let n8 = (n / 8) * 8;
    while i < n8 {
        let v = _mm256_loadu_ps(src.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(v, v, sum0);
        i += 8;
    }
    sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    let mut ss = hsum_ps(sum0);
    for &v in &src[n8..n] {
        ss += v * v;
    }

    let inv_rms = 1.0 / ((ss / nf + eps).sqrt());
    let inv_rms_v = _mm256_set1_ps(inv_rms);

    // Pass 2: dst[i] = src[i] * inv_rms * w[i] (单 pass 替代标量 rmsnorm_into 的两 pass)
    let mut i = 0;
    while i + 32 <= n {
        for off in (0..32).step_by(8) {
            let v = _mm256_loadu_ps(src.as_ptr().add(i + off));
            let wv = _mm256_loadu_ps(w.as_ptr().add(i + off));
            let r = _mm256_mul_ps(_mm256_mul_ps(v, inv_rms_v), wv);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + off), r);
        }
        i += 32;
    }
    while i + 8 <= n {
        let v = _mm256_loadu_ps(src.as_ptr().add(i));
        let wv = _mm256_loadu_ps(w.as_ptr().add(i));
        let r = _mm256_mul_ps(_mm256_mul_ps(v, inv_rms_v), wv);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), r);
        i += 8;
    }
    for j in i..n {
        dst[j] = src[j] * inv_rms * w[j];
    }
}

/// 原地 RMSNorm:x[i] 被替换为归一化后的值
///
/// - `x`: 长度 n 的输入向量(被原地修改)
/// - `w`: 长度 n 的权重
/// - `eps`: 小常数,默认 1e-6
pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), w.len());
    #[cfg(target_arch = "x86_64")]
    if simd_available() && x.len() >= 8 {
        #[allow(unsafe_code)]
        unsafe {
            rmsnorm_inplace_avx2(x, w, eps);
            return;
        }
    }
    let n = x.len() as f32;
    let mut ss = 0.0f32;
    for &xi in x.iter() {
        ss += xi * xi;
    }
    let mean = ss / n;
    let inv_rms = 1.0 / (mean + eps).sqrt();
    for (xi, &wi) in x.iter_mut().zip(w.iter()) {
        *xi = *xi * inv_rms * wi;
    }
}

/// 非原地版本:直接从 `src` 读、写入 `dst`,消除 caller 的 copy_from_slice。
///
/// 等价于 `dst.copy_from_slice(src); rmsnorm_inplace(dst, w, eps)`,
/// 但省去一次 20KB(hidden=5120)的内存拷贝。
///
/// `src` 与 `dst` 必须不重叠(典型场景:`src` = 主残差流 `h`,`dst` = workspace buffer)。
pub fn rmsnorm_into(src: &[f32], dst: &mut [f32], w: &[f32], eps: f32) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), w.len());
    #[cfg(target_arch = "x86_64")]
    if simd_available() && src.len() >= 8 {
        #[allow(unsafe_code)]
        unsafe {
            rmsnorm_into_avx2(src, dst, w, eps);
            return;
        }
    }
    let n = src.len() as f32;
    let mut ss = 0.0f32;
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = s;
        ss += s * s;
    }
    let mean = ss / n;
    let inv_rms = 1.0 / (mean + eps).sqrt();
    for (di, &wi) in dst.iter_mut().zip(w.iter()) {
        *di = *di * inv_rms * wi;
    }
}

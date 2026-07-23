//! GELU 激活函数 (Gaussian Error Linear Unit)
//!
//! 用于 Qwen3-VL ViT MLP (clip.use_gelu=true) 和 PatchMerger (clip.use_gelu=true)。
//!
//! 精确 GELU: `gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))`
//!   = `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` (tanh 近似)
//!
//! ViT 通常用 tanh 近似 (PyTorch nn.GELU(approximate='tanh') 默认)。
//! mmproj GGUF 中 clip.use_gelu=true 对应 nn.GELU() (精确 erf 版本)。
//!
//! 性能: ViT 调用频次相对低 (ffn 27 层 × 2304 patch × 4304 dim = 270M/call),
//! 但 vision encoder 只在每张图调用一次,所以用 SIMD exp 即可。

use std::arch::x86_64::*;
use crate::math::simd_exp::{simd_available, exp_ps};

const SQRT_2_OVER_PI: f32 = 0.7978845608028654; // sqrt(2/pi)
const GELU_CONST: f32 = 0.044715;

/// AVX2 tanh 近似 GELU 内核: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
///
/// tanh(y) = (exp(2y) - 1) / (exp(2y) + 1) = 1 - 2 / (exp(2y) + 1)
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn gelu_tanh_avx2(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let two = _mm256_set1_ps(2.0);
    let c = _mm256_set1_ps(SQRT_2_OVER_PI);
    let k = _mm256_set1_ps(GELU_CONST);

    let mut i = 0;
    let n8 = (x.len() / 8) * 8;
    while i < n8 {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        // x^3
        let x2 = _mm256_mul_ps(v, v);
        let x3 = _mm256_mul_ps(x2, v);
        // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
        let inner = _mm256_fmadd_ps(k, x3, v);
        let inner = _mm256_mul_ps(c, inner);
        // tanh(inner) = 1 - 2 / (exp(2*inner) + 1)
        let two_inner = _mm256_mul_ps(two, inner);
        let exp_2y = exp_ps(two_inner);
        let denom = _mm256_add_ps(exp_2y, one);
        let tanh_v = _mm256_sub_ps(one, _mm256_div_ps(two, denom));
        // 0.5 * x * (1 + tanh)
        let one_plus_tanh = _mm256_add_ps(one, tanh_v);
        let r = _mm256_mul_ps(_mm256_mul_ps(half, v), one_plus_tanh);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), r);
        i += 8;
    }
    for j in i..x.len() {
        let v = x[j];
        let x3 = v * v * v;
        let inner = SQRT_2_OVER_PI * (v + GELU_CONST * x3);
        let exp_2y = (2.0 * inner).exp();
        let tanh_v = 1.0 - 2.0 / (exp_2y + 1.0);
        out[j] = 0.5 * v * (1.0 + tanh_v);
    }
}

/// GELU 激活 (tanh 近似),写到 out
///
/// `out[i] = 0.5 * x[i] * (1 + tanh(sqrt(2/pi) * (x[i] + 0.044715 * x[i]^3)))`
pub fn gelu_into(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    if simd_available() && x.len() >= 8 {
        #[allow(unsafe_code)]
        unsafe {
            gelu_tanh_avx2(x, out);
            return;
        }
    }
    for (i, &v) in x.iter().enumerate() {
        let x3 = v * v * v;
        let inner = SQRT_2_OVER_PI * (v + GELU_CONST * x3);
        let exp_2y = (2.0 * inner).exp();
        let tanh_v = 1.0 - 2.0 / (exp_2y + 1.0);
        out[i] = 0.5 * v * (1.0 + tanh_v);
    }
}

/// GELU in-place (tanh 近似)
///
/// 安全性: gelu_tanh_avx2 每 8 元素 chunk 独立 load→compute→store, 无跨 chunk 依赖,
/// 因此 src==dst aliasing 安全 (每 chunk 先读后写同一地址, 不再读已写区域)。
/// 原实现需 tmp buffer + copy_from_slice (2 pass 读写), in-place 仅 1 pass。
pub fn gelu_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd_available() && x.len() >= 8 {
        #[allow(unsafe_code)]
        unsafe {
            let src = std::slice::from_raw_parts(x.as_ptr(), x.len());
            gelu_tanh_avx2(src, x);
            return;
        }
    }
    for v in x.iter_mut() {
        let val = *v;
        let x3 = val * val * val;
        let inner = SQRT_2_OVER_PI * (val + GELU_CONST * x3);
        let exp_2y = (2.0 * inner).exp();
        let tanh_v = 1.0 - 2.0 / (exp_2y + 1.0);
        *v = 0.5 * val * (1.0 + tanh_v);
    }
}



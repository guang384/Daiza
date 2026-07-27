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

// ============================================================================
// 精确 GELU (erf 版本): `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`
//
// ★ 用于 Qwen3-VL PatchMerger (HF: nn.GELU(approximate='none'))
//   ViT MLP 用 tanh 近似 (nn.GELU(approximate='tanh')), 两者不可混用。
//
// erf 用 Abramowitz-Stegun 7.1.26 近似 (最大误差 1.5e-7, 对 f32 足够):
//   t = 1 / (1 + p*|z|),  p = 0.3275911
//   erf(z) = sign(z) * (1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-z^2))
//   a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429
// ============================================================================

const ERF_P: f32 = 0.3275911;
const ERF_A1: f32 = 0.254829592;
const ERF_A2: f32 = -0.284496736;
const ERF_A3: f32 = 1.421413741;
const ERF_A4: f32 = -1.453152027;
const ERF_A5: f32 = 1.061405429;
const INV_SQRT_2: f32 = 0.7071067811865476; // 1/sqrt(2)

/// AVX2 精确 GELU 内核 (erf 版本)
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn gelu_erf_avx2(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let inv_sqrt2 = _mm256_set1_ps(INV_SQRT_2);
    let erf_p = _mm256_set1_ps(ERF_P);
    let a1 = _mm256_set1_ps(ERF_A1);
    let a2 = _mm256_set1_ps(ERF_A2);
    let a3 = _mm256_set1_ps(ERF_A3);
    let a4 = _mm256_set1_ps(ERF_A4);
    let a5 = _mm256_set1_ps(ERF_A5);
    let sign_mask = _mm256_set1_ps(-0.0); // 符号位

    let mut i = 0;
    let n8 = (x.len() / 8) * 8;
    while i < n8 {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        // z = x / sqrt(2)
        let z = _mm256_mul_ps(v, inv_sqrt2);
        // |z|
        let az = _mm256_andnot_ps(sign_mask, z);
        // t = 1 / (1 + p*|z|)
        let t = _mm256_rcp_ps(_mm256_fmadd_ps(erf_p, az, one));
        // poly = a1 + t*(a2 + t*(a3 + t*(a4 + t*a5)))  (Horner)
        let poly = _mm256_fmadd_ps(
            a1,
            one,
            _mm256_mul_ps(t, _mm256_fmadd_ps(
                a2, one,
                _mm256_mul_ps(t, _mm256_fmadd_ps(
                    a3, one,
                    _mm256_mul_ps(t, _mm256_fmadd_ps(a4, one, _mm256_mul_ps(t, a5)))
                ))
            ))
        );
        // 实际 poly = t * (a1 + t*(a2 + t*(...))), 上面少乘了一个 t, 修正:
        //   正确 Horner: poly = ((a5*t + a4)*t + a3)*t + a2)*t + a1, 然后整体 * t
        //   简化: poly_t = t * (a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
        //   上面计算的是 a1 + t*(a2 + ...), 需再 * t
        let poly_t = _mm256_mul_ps(t, poly);
        // exp(-z^2)
        let z2 = _mm256_mul_ps(z, z);
        let exp_neg_z2 = exp_ps(_mm256_xor_ps(z2, sign_mask)); // -z^2
        // erf = sign(z) * (1 - poly_t * exp(-z^2))
        let one_minus = _mm256_sub_ps(one, _mm256_mul_ps(poly_t, exp_neg_z2));
        let sign_z = _mm256_or_ps(_mm256_and_ps(z, sign_mask), one); // 提取符号位 → ±1.0
        let erf = _mm256_mul_ps(sign_z, one_minus);
        // gelu = 0.5 * x * (1 + erf)
        let r = _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, erf));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), r);
        i += 8;
    }
    for j in i..x.len() {
        out[j] = gelu_erf_scalar(x[j]);
    }
}

/// 标量精确 GELU (erf 版本)
#[inline]
fn gelu_erf_scalar(v: f32) -> f32 {
    let z = v * INV_SQRT_2;
    let az = z.abs();
    let t = 1.0 / (1.0 + ERF_P * az);
    // Horner: poly = a1 + t*(a2 + t*(a3 + t*(a4 + t*a5)))
    let poly = ERF_A1 + t * (ERF_A2 + t * (ERF_A3 + t * (ERF_A4 + t * ERF_A5)));
    let poly_t = t * poly;
    let exp_neg_z2 = (-z * z).exp();
    let erf = if z >= 0.0 { 1.0 - poly_t * exp_neg_z2 } else { poly_t * exp_neg_z2 - 1.0 };
    0.5 * v * (1.0 + erf)
}

/// 精确 GELU in-place (erf 版本)
///
/// ★ 用于 Qwen3-VL PatchMerger (HF: GELU(approximate='none'))
///   区别于 gelu_inplace (tanh 近似, 用于 ViT MLP)
pub fn gelu_erf_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd_available() && x.len() >= 8 {
        #[allow(unsafe_code)]
        unsafe {
            let src = std::slice::from_raw_parts(x.as_ptr(), x.len());
            gelu_erf_avx2(src, x);
            return;
        }
    }
    for v in x.iter_mut() {
        *v = gelu_erf_scalar(*v);
    }
}



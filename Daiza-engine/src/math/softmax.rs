//! Softmax(在线减最大值,数值稳定版本)

use crate::math::simd_exp::{exp_inplace_simd, simd_available, exp_ps};
use core::arch::x86_64::*;

/// AVX2 水平求和 __m256 → f32
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[inline]
unsafe fn hsum_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// AVX2 水平 max __m256 → f32
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[inline]
unsafe fn hmax_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let max128 = _mm_max_ps(hi, lo);
    let shuf = _mm_movehdup_ps(max128);
    let maxs = _mm_max_ps(max128, shuf);
    let shuf2 = _mm_movehl_ps(maxs, maxs);
    _mm_cvtss_f32(_mm_max_ss(maxs, shuf2))
}

/// 标准数值稳定 softmax:原地修改
///
/// ★ 优化:3 pass SIMD (原 4 pass, Pass 2+3a 融合)
///   Pass 1: AVX2 max reduce(含尾部标量)
///   Pass 2+3a: AVX2 (sub_max + exp + sum 累加) 融合 — 省 1 次完整读 x
///   Pass 3b: AVX2 mul inv (归一化)
///
/// attention 调用频次:16 层 × 24 head × n_cached 次/token
/// n_cached=512 时每 token ~200K 元素,4 pass → 3 pass 节省 1 次遍历
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let n = x.len();

    #[cfg(target_arch = "x86_64")]
    if simd_available() {
        #[allow(unsafe_code)]
        unsafe {
            let n8 = (n / 8) * 8;

            // Pass 1: AVX2 max reduce
            let mut max_v = _mm256_set1_ps(x[0]);
            let mut i = 0;
            while i < n8 {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                max_v = _mm256_max_ps(max_v, v);
                i += 8;
            }
            let mut max = hmax_ps(max_v);
            while i < n {
                if x[i] > max {
                    max = x[i];
                }
                i += 1;
            }

            // Pass 2+3a: (sub_max + exp + sum 累加) 融合
            // ★ 原 Pass 2 (write x[i]=exp) + Pass 3a (read x[i] for sum) = 2 reads + 1 write
            //   现 fused: 1 read + 1 write + 在线累加 sum (省 1 次完整读 x)
            let max_v = _mm256_set1_ps(max);
            let mut sum_v = _mm256_setzero_ps();
            i = 0;
            while i < n8 {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                let shifted = _mm256_sub_ps(v, max_v);
                let e = exp_ps(shifted);
                _mm256_storeu_ps(x.as_mut_ptr().add(i), e);
                sum_v = _mm256_add_ps(sum_v, e);
                i += 8;
            }
            let mut sum = hsum_ps(sum_v);
            for j in n8..n {
                let e = (x[j] - max).exp();
                x[j] = e;
                sum += e;
            }

            // Pass 3b: AVX2 mul inv (归一化)
            let inv = 1.0 / sum;
            let inv_v = _mm256_set1_ps(inv);
            i = 0;
            while i < n8 {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                let r = _mm256_mul_ps(v, inv_v);
                _mm256_storeu_ps(x.as_mut_ptr().add(i), r);
                i += 8;
            }
            for j in n8..n {
                x[j] *= inv;
            }
            return;
        }
    }

    // 标量 fallback
    let mut max = x[0];
    for &xi in x.iter() {
        if xi > max {
            max = xi;
        }
    }
    for xi in x.iter_mut() {
        *xi -= max;
    }
    exp_inplace_simd(x);
    let mut sum = 0.0f32;
    for &xi in x.iter() {
        sum += xi;
    }
    let inv = 1.0 / sum;
    for xi in x.iter_mut() {
        *xi *= inv;
    }
}

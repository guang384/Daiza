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
/// ★ 优化:5 次标量遍历 → 4 次 SIMD 遍历(3 阶段,Pass 3 含 sum+mul 两遍)
///   Pass 1: AVX2 max reduce(含尾部标量)
///   Pass 2: AVX2 (sub_max + exp) 融合(原标量 sub + 分离 SIMD exp)
///   Pass 3: AVX2 sum reduce + AVX2 mul inv
///
/// attention 调用频次:16 层 × 24 head × n_cached 次/token
/// n_cached=512 时每 token ~200K 元素,5 pass → 3 pass 节省 ~2 次遍历
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

            // Pass 2: (sub_max + exp) 融合
            let max_v = _mm256_set1_ps(max);
            i = 0;
            while i < n8 {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                let shifted = _mm256_sub_ps(v, max_v);
                let e = exp_ps(shifted);
                _mm256_storeu_ps(x.as_mut_ptr().add(i), e);
                i += 8;
            }
            for j in n8..n {
                x[j] = (x[j] - max).exp();
            }

            // Pass 3: AVX2 sum reduce + AVX2 mul inv
            let mut sum_v = _mm256_setzero_ps();
            i = 0;
            while i < n8 {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                sum_v = _mm256_add_ps(sum_v, v);
                i += 8;
            }
            let mut sum = hsum_ps(sum_v);
            for j in n8..n {
                sum += x[j];
            }
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

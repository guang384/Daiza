//! Softmax(在线减最大值,数值稳定版本)

use crate::math::simd_exp::exp_inplace_simd;

/// 标准数值稳定 softmax:原地修改
///
/// ★ 优化:用 SIMD exp(多项式 + 位运算 exp2)替代标量 libm expf
///   attention 调用频次:16 层 × 24 head × n_cached 次/token
///   T=4K 时 ~1.5M 次,标量 ~30c vs SIMD ~1.25c = 节省 ~2ms/token
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = x[0];
    for &xi in x.iter() {
        if xi > max {
            max = xi;
        }
    }
    // 减最大值(数值稳定)+ SIMD exp
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

/// 带 mask 的 softmax:对 mask=False 的位置置 -inf(常用于因果 attention)
pub fn softmax_masked_inplace(x: &mut [f32], mask: &[bool]) {
    debug_assert_eq!(x.len(), mask.len());
    if x.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for (i, &xi) in x.iter().enumerate() {
        if mask[i] && xi > max {
            max = xi;
        }
    }
    if max == f32::NEG_INFINITY {
        max = 0.0;
    }
    let mut sum = 0.0f32;
    for (i, xi) in x.iter_mut().enumerate() {
        if mask[i] {
            *xi = (*xi - max).exp();
            sum += *xi;
        } else {
            *xi = 0.0;
        }
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for (i, xi) in x.iter_mut().enumerate() {
        if mask[i] {
            *xi *= inv;
        }
    }
}

/// 在线 softmax(用于长上下文,避免二次内存)
/// v0 占位,真正实现见 https://arxiv.org/abs/1805.02867
pub fn _softmax_online_placeholder() {}

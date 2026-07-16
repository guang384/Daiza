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

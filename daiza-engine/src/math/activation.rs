//! 激活函数:SiLU/Swish + SwiGLU

use crate::math::simd_exp::{silu_fast, silu_inplace_simd, swiglu_inplace_simd};

/// SiLU / Swish / x * sigmoid(x)
///
/// ★ 优化:用 SIMD exp 内核(标量入口,内部走 AVX2)
#[inline]
pub fn silu(x: f32) -> f32 {
    silu_fast(x)
}

/// 原地 SiLU(批量 SIMD 版本)
pub fn silu_inplace(x: &mut [f32]) {
    silu_inplace_simd(x)
}

/// SwiGLU 激活:`y = silu(gate) * up`
///
/// Bonsai 的 FFN 是 SwiGLU MLP(来自 Qwen3):
/// - `ffn_gate.weight` [d, d_ff]:先做 `gate = x @ W_gate`
/// - `ffn_up.weight`   [d, d_ff]:再做 `up   = x @ W_up`
/// - 输出:`act = silu(gate) * up`
/// - 然后 `ffn_down.weight` [d_ff, d] 把结果降维回 d
///
/// ★ 优化:用 SIMD silu + 乘法,8-wide AVX2
///   调用频次:17408 × 64 层 = 1.1M 次/token
///   标量 ~30c vs SIMD ~1.5c = 节省 ~10ms/token
pub fn swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    swiglu_inplace_simd(gate, up)
}

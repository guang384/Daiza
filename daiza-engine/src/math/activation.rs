//! SwiGLU 激活函数

use crate::math::simd_exp::swiglu_inplace_simd;

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

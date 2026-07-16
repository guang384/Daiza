//! 激活函数:SiLU/Swish + SwiGLU

/// SiLU / Swish / x * sigmoid(x)
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// 原地 SiLU
pub fn silu_inplace(x: &mut [f32]) {
    for xi in x.iter_mut() {
        *xi = silu(*xi);
    }
}

/// SwiGLU 激活:`y = silu(gate) * up`
///
/// Bonsai 的 FFN 是 SwiGLU MLP(来自 Qwen3):
/// - `ffn_gate.weight` [d, d_ff]:先做 `gate = x @ W_gate`
/// - `ffn_up.weight`   [d, d_ff]:再做 `up   = x @ W_up`
/// - 输出:`act = silu(gate) * up`
/// - 然后 `ffn_down.weight` [d_ff, d] 把结果降维回 d
pub fn swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g = silu(*g) * u;
    }
}

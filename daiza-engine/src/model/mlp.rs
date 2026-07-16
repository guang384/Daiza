//! SwiGLU MLP(两种 block 共享)
//!
//! ```text
//! gate = x @ W_gate        # [hidden, d_ff]
//! up   = x @ W_up         # [hidden, d_ff]
//! h    = silu(gate) * up
//! y    = h @ W_down        # [d_ff, hidden]
//! ```
//!
//! W_gate / W_up / W_down 都是 Q1_0,通过 `Q1_0Matrix::matvec` 流式反量化 + GEMM。

use crate::math;
use crate::model::weights::Q1_0Matrix;

pub struct MlpOutput {
    pub out: Vec<f32>,
}

/// 单 token 前向,W_gate / W_up / W_down 都是 Q1_0Matrix
pub fn mlp_forward_single(
    x: &[f32],
    w_gate: &Q1_0Matrix,
    w_up: &Q1_0Matrix,
    w_down: &Q1_0Matrix,
) -> MlpOutput {
    // gate = x @ W_gate  (1×d_ff)
    let mut gate = w_gate.matvec(x);
    // up = x @ W_up
    let up = w_up.matvec(x);
    // silu(gate) * up
    math::swiglu_inplace(&mut gate, &up);
    // y = h @ W_down
    let out = w_down.matvec(&gate);
    MlpOutput { out }
}

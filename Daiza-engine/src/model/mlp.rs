//! SwiGLU MLP(两种 block 共享)
//!
//! ```text
//! gate = x @ W_gate        # [hidden, d_ff]
//! up   = x @ W_up         # [hidden, d_ff]
//! h    = silu(gate) * up
//! y    = h @ W_down        # [d_ff, hidden]
//! ```
//!
//! W_gate / W_up / W_down 都是 Q1_0,通过 `Q1_0Matrix::matvec_into_slice` 流式反量化 + GEMM。
//!
//! ## v2 优化(in-place workspace 复用)
//!
//! - `gate` / `up` 写入预分配的 caller-provided slice,跨 token 复用
//! - `W_down` 结果通过 `matvec_add_into_slice` 直接累加到主残差流 `h`,避免分配
//! - 调用方(`block.rs`)负责把 `ws.mlp_gate` / `ws.mlp_up` 传进来,
//!   这样可以在同一作用域内同时借用 `ws.block_normed`(输入,不可变)
//!   和 `ws.mlp_gate` / `ws.mlp_up`(中间 buffer,可变)—— Rust split borrow。

use crate::math;
use crate::model::weights::Q1_0Matrix;

/// 单 token 前向,W_gate / W_up / W_down 都是 Q1_0Matrix
///
/// - `x`:输入(norm 后的 h,长度 = hidden)
/// - `mlp_gate` / `mlp_up`:caller 提供的工作区(长度 = d_ff,跨 token 复用)
/// - `h`:主残差流,输出累加到此: `h += W_down @ (silu(W_gate @ x) * (W_up @ x))`
pub fn mlp_forward_into(
    x: &[f32],
    w_gate: &Q1_0Matrix,
    w_up: &Q1_0Matrix,
    w_down: &Q1_0Matrix,
    mlp_gate: &mut [f32],
    mlp_up: &mut [f32],
    h: &mut [f32],
) {
    // ★ P0-B: gate + up 合并为单次线程池 barrier (共享输入 x, 20KB 驻留 L2)
    //   原: 2 次 scatter_wait (各 17408 rows → 2 barriers)
    //   新: 1 次 scatter_wait (34816 rows → 1 barrier, -1 barrier/block × 64 blocks)
    let matrices: &[&Q1_0Matrix] = &[w_gate, w_up];
    let outputs: &mut [&mut [f32]] = &mut [mlp_gate, mlp_up];
    Q1_0Matrix::matvec_multi_into_slice(x, matrices, outputs);
    // silu(gate) * up (in-place on mlp_gate)
    math::swiglu_inplace(mlp_gate, mlp_up);
    // h += W_down @ mlp_gate
    w_down.matvec_add_into_slice(mlp_gate, h);
}

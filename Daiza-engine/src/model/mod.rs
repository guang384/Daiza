//! Bonsai 27B 模型架构
//!
//! 64 个 transformer block,两种类型由 `block_idx % 4 == 3` 决定:
//!
//! | 类型 | 数量 | 张量集 | 关键算子 |
//! |------|------|--------|----------|
//! | **SSM / 线性注意力块** | 48 | `attn_gate / attn_qkv / ssm_*` | Mamba2 selective scan |
//! | **全注意力块**       | 16 | `attn_q / k / v / output / q_norm / k_norm` | GQA + RoPE + softmax |
//!
//! 两种块都共享同一个 SwiGLU MLP。
//!
//! 残差结构:
//! ```text
//! x → attn_norm → [SSM 或 full attention] → + x (residual)
//!   → post_attention_norm → MLP → + x (residual)
//! ```
//!
//! 置信常数:
//! - hidden = 5120
//! - d_ff = 17408
//! - head_dim = 256(GQA,24 query heads,4 KV heads)
//! - rope_dim = 64
//! - rope_freq_base = 1e7
//! - rms_eps = 1e-6

pub mod config;
pub mod weights;
pub mod ssm;
pub mod attention;
pub mod mlp;
pub mod block;
pub mod forward;
pub mod workspace;
pub mod dspark;

pub use config::Config;
pub use forward::{ForwardContext, ModelState};
pub use weights::{BlockWeights, FullAttentionBlockWeights, GlobalWeights, SsmBlockWeights};
pub use workspace::Workspace;

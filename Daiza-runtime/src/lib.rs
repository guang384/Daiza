//! # daiza-runtime
//!
//! Daiza 引擎编排层:会话管理、工具调用、分词器、顶层 Engine。
//!
//! 依赖 `daiza-engine` 推理核心(forward / weights / quant / ssm / workspace)。
//! 热路径(forward → weights → quant AVX2 kernel)全部在 daiza-engine 内,
//! 本 crate 仅在每 token 调用一次 forward_batch,非紧凑循环。

pub mod engine;
pub mod session;
pub mod session_persist;
pub mod session_manager;
pub mod tool_call;
pub mod tokenizer;

// re-export daiza-engine 的常用类型,方便 binary crate 使用
pub use daiza_engine::{BonsaiError, Result};
pub use engine::Engine;

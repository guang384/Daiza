//! DSpark 推测解码实现
//!
//! 基于 DSpark 论文 (Confidence-Scheduled Speculative Decoding with
//! Semi-Autoregressive Generation) 的完整实现, 用于加速 Bonsai-27B 推理。
//!
//! ## 架构
//!
//! DSpark drafter 是一个 6 层 block-parallel transformer, 一次 forward 生成
//! k=4 个候选 token 的 base logits, 再通过 Markov head 注入块内依赖, 最后
//! 由 target model 验证 (Leviathan rejection sampling)。
//!
//! ## 模块组织
//!
//! - `config`: drafter 超参数 (从 GGUF metadata 解析)
//! - `weights`: drafter 权重加载 (Q4_1 + Q1_0 + Iq1M)
//! - `drafter`: block-parallel forward (非因果 attention + log_snr + target tap)
//! - `markov`: Markov head 顺序链式 resample
//! - `speculative`: verify + rejection sampling + KV/SSM cache rollback

pub mod config;
pub mod weights;
pub mod drafter;
pub mod markov;
pub mod speculative;

pub use config::DrafterConfig;
pub use weights::DrafterWeights;
pub use drafter::DrafterContext;

//! 推理时的运行时状态:KV cache(全注意力层)与 SSM state(Mamba2 层)
//!
//! ## 为什么有两种缓存
//!
//! Bonsai 27B 是 **混合注意力** 模型:
//! - **全注意力层(16/64):** KV cache 随 token 数线性增长(每 token 每层加一对 K/V 向量)
//! - **SSM 层(48/64):** 状态是**固定大小的 recurrent state**,不随 token 数增长
//!   → 这正是 Mamba/SSM 的关键优势,让长上下文(262K)可行
//!
//! 白皮书 §4.4:FP16 KV cache ≈ 64 KiB/token,因为只有 16 层需要。
//! KV cache 量化到 4-bit 后 ≈ 16 KiB/token,100K context 占 4.3 GB。

pub mod kv_cache;
pub mod ssm_state;

pub use kv_cache::KvCache;
pub use ssm_state::SsmState;

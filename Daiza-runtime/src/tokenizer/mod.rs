//! GPT-2 BPE tokenizer(Qwen35 预分词)
//!
//! ## 零依赖挑战
//!
//! 标准实现需要:
//! 1. **Unicode 处理**:Rust 标准库够用(`char::is_alphanumeric` 等)
//! 2. **正则表达式**:`tiktoken` / `transformers` 用 Python regex 的预分词 pattern
//!    → 学习项目 v0:手写一个简化版的预分词(只支持基本 ASCII 与中文),
//!    后续可换成更完整的实现
//! 3. **BPE merge**:从 `tokenizer.ggml.merges`(247587 条)构建 rank 表
//!
//! v0 实现策略:
//! - 简单 tokenizer:仅支持把单 token id 转换为 bytes / bytes 转回 id
//! - 完整 BPE 留作练习

pub mod bpe;
pub mod vocab;

pub use bpe::BpeTokenizer;
pub use vocab::Vocab;

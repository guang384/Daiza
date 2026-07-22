//! # daiza-engine
//!
//! 纯 CPU、零依赖的 Rust 推理引擎,目标模型为 **1-bit Bonsai 27B**
//! (Qwen3.6-27B 的二值化版本,PrismML 出品)。
//!
//! ## 设计依据
//!
//! 所有架构常数来自对 `Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf` 的二进制解析,
//! 已与白皮书 `bonsai-27b-whitepaper.pdf` 交叉验证。关键事实:
//!
//! - 64 个 transformer block,其中 **48 个 SSM 块**(Mamba2 风格)+ **16 个全注意力块**
//! - 节拍:`block_N % 4 == 3` 时为全注意力层(由 `qwen35.full_attention_interval = 4` 决定)
//! - 权重格式 **Q1_0**:每 128 权重 = 1 个 FP16 scale + 16 字节符号位 = 18 字节
//! - 1.125 bits/weight,语言模型总计 3.9 GB
//!
//! ## 模块组织
//!
//! | 模块 | 职责 |
//! |------|------|
//! | `gguf`     | GGUF v3 二进制格式解析(magic / metadata / tensor info) |
//! | `tensor`   | 张量抽象与量化类型(Q1_0 / F32 / F16 / BF16)的反量化 |
//! | `math`     | 纯 CPU 数学内核:GEMM、RMSNorm、RoPE、Softmax、采样 |
//! | `model`    | Bonsai 27B 架构:SSM 块、全注意力块、SwiGLU MLP、前向传播 |
//! | `cache`    | KV 缓存(仅 16 个全注意力层)+ SSM 循环状态(48 层) |
//!
//! 编排层 (engine / session / tokenizer / tool_call) 在独立的 `daiza-runtime` crate。

// unsafe 策略:全局 deny,只在 quant.rs 的 AVX2 SIMD 内核局部 allow。
// 这样既限制了 unsafe 的扩散,又允许手写 SIMD 突破 rustc 自动向量化的瓶颈。
#![deny(unsafe_code)]
// SSM/数学符号沿用论文记法 (U/V/D/K 等),允许 non_snake_case
#![allow(non_snake_case)]

pub mod gguf;
pub mod tensor;
pub mod math;
pub mod model;
pub mod cache;

/// 项目级错误类型,所有模块统一使用
#[derive(Debug)]
pub enum BonsaiError {
    /// GGUF 文件格式错误
    Gguf(String),
    /// 文件 IO 错误
    Io(String),
    /// 张量形状/类型不匹配
    Tensor(String),
    /// 模型架构错误
    Model(String),
    /// Tokenizer 错误
    Tokenizer(String),
    /// 不支持的功能(用于学习项目的明确占位)
    Unsupported(String),
}

impl std::fmt::Display for BonsaiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf(s) => write!(f, "GGUF error: {s}"),
            Self::Io(s) => write!(f, "IO error: {s}"),
            Self::Tensor(s) => write!(f, "Tensor error: {s}"),
            Self::Model(s) => write!(f, "Model error: {s}"),
            Self::Tokenizer(s) => write!(f, "Tokenizer error: {s}"),
            Self::Unsupported(s) => write!(f, "Unsupported: {s}"),
        }
    }
}

impl std::error::Error for BonsaiError {}

impl From<std::io::Error> for BonsaiError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, BonsaiError>;

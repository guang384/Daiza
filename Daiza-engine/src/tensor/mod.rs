//! 张量抽象与量化类型
//!
//! 引擎需要处理三种主要的张量数据类型:
//!
//! | 类型 | 用途 | 字节/元素 | 备注 |
//! |------|------|----------|------|
//! | `F32`    | norms / conv1d / ssm_a / ssm_dt.bias | 4 | 标准 |
//! | `F16`    | 部分 mmproj 权重 + Q1_0 的 group scale | 2 | IEEE 754 半精度 |
//! | `BF16`   | dspark 文件中的几个特殊 head | 2 | bfloat16 |
//! | `Q1_0`   | **Bonsai 主权重** — 每 128 权重 = 18 字节 | 1.125 bits | 自定义 |
//!
//! Q1_0 的反量化是这个引擎的核心创新点。

pub mod dtype;
pub mod iq1m;
pub mod quant;
pub mod tensor;

pub use dtype::{byte_size, TensorType};
pub use quant::{
    bf16_to_f32, dequantize_q1_0, dequantize_q1_0_row_into, f16_to_f32,
};
pub use tensor::Tensor;

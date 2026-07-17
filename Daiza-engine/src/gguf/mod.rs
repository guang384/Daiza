//! GGUF v3 二进制格式解析
//!
//! GGUF 是 llama.cpp 的模型文件格式,结构如下:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │ magic        : "GGUF" (4 bytes)                         │
//! │ version      : u32 LE                                  │
//! │ tensor_count : u64 LE                                 │
//! │ kv_count      : u64 LE                                 │
//! ├─────────────────────────────────────────────────────────┤
//! │ metadata KV pairs × kv_count                          │
//! │   每个 KV: key: gguf_string                          │
//! │            value_type: u32 LE                         │
//! │            value:    根据 value_type 解析               │
//! ├─────────────────────────────────────────────────────────┤
//! │ tensor info   × tensor_count                          │
//! │   每个 tensor: name: gguf_string                      │
//! │               n_dims: u32 LE                         │
//! │               dims:   u64[n_dims] LE                  │
//! │               dtype:  u32 LE  (ggml_type enum)        │
//! │               offset: u64 LE  (相对数据段起点)          │
//! ├─────────────────────────────────────────────────────────┤
//! │ padding to alignment                                  │
//! ├─────────────────────────────────────────────────────────┤
//! │ tensor data (按 offset 索引)                          │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! `gguf_string` = u64 LE length + UTF-8 bytes
//!
//! ## 已验证的事实(对 Bonsai-27B-Q1_0.gguf 实测)
//!
//! - version = 3
//! - tensor_count = 851
//! - kv_count = 37
//! - alignment = 32
//! - 数据段起始偏移 = 10,992,704 字节
//! - 自定义类型 **Q1_0 = ggml_type 枚举值 41**(上游 llama.cpp 最高到 38)

pub mod metadata;
pub mod parser;
pub mod reader;
pub mod tensor_info;

pub use metadata::{MetaValue, Metadata};
pub use parser::GgufFile;
pub use reader::ByteReader;
pub use tensor_info::{TensorInfo, TensorType, GGUF_MAGIC, GGUF_VERSION};

use crate::BonsaiError;

/// GGUF 文件解析错误
pub(crate) fn err(msg: impl Into<String>) -> BonsaiError {
    BonsaiError::Gguf(msg.into())
}

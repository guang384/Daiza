//! 词表元数据(token 字符串、token 类型、merges)
//!
//! 从 GGUF metadata 加载:
//! - `tokenizer.ggml.tokens`:STRING[248320]
//! - `tokenizer.ggml.token_type`:INT32[248320]
//! - `tokenizer.ggml.merges`:STRING[247587]
//! - `tokenizer.ggml.model` = "gpt2"

use daiza_engine::gguf::metadata::Metadata;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

pub struct Vocab {
    /// token 字符串(对 GPT-2 BPE,以 byte-level 形式存储,可能含前导空格 / GPT2_BPE 格式)
    pub tokens: Vec<String>,
    /// token 类型(1=normal, 2=unknown, ...)
    pub token_types: Vec<i32>,
    /// BPE merge rank:merge pair ("a b") → rank
    pub merges: std::collections::HashMap<String, u32>,
    /// bos / eos / pad token id
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl Vocab {
    pub fn from_metadata(meta: &Metadata) -> crate::Result<Self> {
        let tokens = match meta.kv.get("tokenizer.ggml.tokens") {
            Some(daiza_engine::gguf::metadata::MetaValue::Array(elems)) => elems
                .iter()
                .map(|v| match v {
                    daiza_engine::gguf::metadata::MetaValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
            _ => {
                return Err(crate::BonsaiError::Tokenizer(
                    "missing tokenizer.ggml.tokens".into(),
                ));
            }
        };
        let token_types = meta
            .get_i32_array("tokenizer.ggml.token_type")
            .unwrap_or_default();

        let merges = match meta.kv.get("tokenizer.ggml.merges") {
            Some(daiza_engine::gguf::metadata::MetaValue::Array(elems)) => elems
                .iter()
                .enumerate()
                .map(|(i, v)| match v {
                    daiza_engine::gguf::metadata::MetaValue::String(s) => (s.clone(), i as u32),
                    _ => (String::new(), i as u32),
                })
                .collect(),
            _ => std::collections::HashMap::new(),
        };

        Ok(Self {
            tokens,
            token_types,
            merges,
            bos_token_id: meta
                .get_u32("tokenizer.ggml.bos_token_id")
                .unwrap_or(0),
            eos_token_id: meta
                .get_u32("tokenizer.ggml.eos_token_id")
                .unwrap_or(0),
            pad_token_id: meta
                .get_u32("tokenizer.ggml.padding_token_id")
                .unwrap_or(0),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_text(&self, id: u32) -> &str {
        self.tokens.get(id as usize).map(|s| s.as_str()).unwrap_or("")
    }
}

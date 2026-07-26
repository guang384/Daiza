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
    /// BPE merge rank:merge pair ("a b") → rank(保留兼容)
    pub merges: std::collections::HashMap<String, u32>,
    /// BPE merge rank by token id pair:(id_a, id_b) → rank
    pub merges_by_id: std::collections::HashMap<(u32, u32), u32>,
    /// merge 结果 token id:(id_a, id_b) → merged_id
    pub merge_result: std::collections::HashMap<(u32, u32), u32>,
    /// bos / eos / pad token id
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl Vocab {
    pub fn from_metadata(meta: &Metadata) -> crate::Result<Self> {
        let tokens: Vec<String> = match meta.kv.get("tokenizer.ggml.tokens") {
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

        // 构建 token 文本 → token_id 的临时映射,用于把 merges 字符串 key 转 id pair key
        let mut token_to_id: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        // 构建 merges_by_id:(id_a, id_b) → rank
        // 以及 merge_result:(id_a, id_b) → merged_id
        let mut merges_by_id: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::new();
        let mut merge_result: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::new();
        for (merge_str, &rank) in &merges {
            // merges 格式为 "token_a token_b",用 rsplit_once(' ') 拆分更安全
            if let Some((a, b)) = merge_str.rsplit_once(' ') {
                let id_a = token_to_id.get(a).copied();
                let id_b = token_to_id.get(b).copied();
                if let (Some(id_a), Some(id_b)) = (id_a, id_b) {
                    // merge 结果是两 token 字符串拼接,查其在词表中的 id
                    let merged_text = format!("{}{}", a, b);
                    if let Some(&merged_id) = token_to_id.get(&merged_text) {
                        merges_by_id.insert((id_a, id_b), rank);
                        merge_result.insert((id_a, id_b), merged_id);
                    }
                }
            }
        }

        let eos_token_id = meta.get_u32("tokenizer.ggml.eos_token_id");
        let eos_token_id = match eos_token_id {
            Some(id) => id,
            None => {
                eprintln!("[tokenizer] warning: missing eos_token_id in GGUF metadata, defaulting to 0");
                0
            }
        };

        Ok(Self {
            tokens,
            token_types,
            merges,
            merges_by_id,
            merge_result,
            bos_token_id: meta
                .get_u32("tokenizer.ggml.bos_token_id")
                .unwrap_or(0),
            eos_token_id,
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

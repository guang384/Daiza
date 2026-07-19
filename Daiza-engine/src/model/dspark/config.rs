//! DSpark drafter 超参数 (从 GGUF metadata 解析)
//!
//! 关键 metadata (来自 Bonsai-27B-dspark-Q4_1.gguf):
//! - dspark.block_count = 6
//! - dspark.embedding_length = 5120
//! - dspark.attention.head_count = 40
//! - dspark.attention.head_count_kv = 4
//! - dspark.attention.key_length = 128
//! - dspark.attention.value_length = 128
//! - dspark.feed_forward_length = 5120
//! - dspark.dspark.block_size = 4
//! - dspark.dspark.markov_rank = 256
//! - dspark.dspark.log_snr_conditioning = true
//! - dspark.dspark.mask_token_id = 248319
//! - dspark.dspark.max_log_snr = 9
//! - dspark.dspark.min_log_snr = -9
//! - dspark.dspark.target_layers = [array len=5]
//! - dspark.rope.freq_base = 10000000
//! - dspark.vocab_size = 248320

use crate::gguf::metadata::{MetaValue, Metadata};

/// DSpark drafter 配置
#[derive(Debug, Clone)]
pub struct DrafterConfig {
    /// drafter transformer 层数 (6)
    pub block_count: usize,
    /// hidden dim (5120, 与 target 相同)
    pub embedding_length: usize,
    /// query head 数 (40)
    pub head_count: usize,
    /// KV head 数 (4, GQA)
    pub head_count_kv: usize,
    /// head dim (128)
    pub head_dim: usize,
    /// FFN 中间维度 (5120)
    pub feed_forward_length: usize,
    /// 一次 draft 生成的 token 数 (4)
    pub block_size: usize,
    /// Markov head 低秩维度 (256)
    pub markov_rank: usize,
    /// 是否启用 log-SNR conditioning
    pub log_snr_conditioning: bool,
    /// mask token id (248319)
    pub mask_token_id: u32,
    /// log-SNR 最大值 (9.0)
    pub max_log_snr: f32,
    /// log-SNR 最小值 (-9.0)
    pub min_log_snr: f32,
    /// 从 target 抽取 hidden states 的层索引 (5 层)
    pub target_layers: Vec<usize>,
    /// RoPE 频率基 (1e7)
    pub rope_freq_base: f32,
    /// 词表大小 (248320)
    pub vocab_size: usize,
    /// RMSNorm epsilon
    pub rms_eps: f32,
    /// log-SNR FC1 输出维度 (5120)
    pub log_snr_n_freq: usize,
    /// 是否启用 confidence head (GGUF: dspark.confidence_head)
    pub confidence_head: bool,
    /// confidence head 输入是否包含 markov embedding (GGUF: dspark.confidence_head_with_markov)
    /// true: input = [hidden(5120), markov_emb(256)] = 5376
    /// false: input = [hidden(5120)]
    pub confidence_head_with_markov: bool,
}

impl Default for DrafterConfig {
    fn default() -> Self {
        Self {
            block_count: 6,
            embedding_length: 5120,
            head_count: 40,
            head_count_kv: 4,
            head_dim: 128,
            feed_forward_length: 5120,
            block_size: 4,
            markov_rank: 256,
            log_snr_conditioning: true,
            mask_token_id: 248319,
            max_log_snr: 9.0,
            min_log_snr: -9.0,
            target_layers: vec![1, 16, 31, 46, 61],
            rope_freq_base: 1e7,
            vocab_size: 248320,
            rms_eps: 1e-6,
            log_snr_n_freq: 128,
            confidence_head: false,
            confidence_head_with_markov: false,
        }
    }
}

impl DrafterConfig {
    /// 从 GGUF metadata 解析 drafter 配置
    pub fn from_metadata(meta: &Metadata) -> crate::Result<Self> {
        let mut cfg = Self::default();

        let get_u32 = |key: &str| -> Option<u32> {
            match meta.kv.get(key) {
                Some(MetaValue::Uint32(v)) => Some(*v),
                Some(MetaValue::Int32(v)) => Some(*v as u32),
                _ => None,
            }
        };
        let get_f32 = |key: &str| -> Option<f32> {
            match meta.kv.get(key) {
                Some(MetaValue::Float32(v)) => Some(*v),
                Some(MetaValue::Int32(v)) => Some(*v as f32),
                _ => None,
            }
        };
        let get_bool = |key: &str| -> Option<bool> {
            match meta.kv.get(key) {
                Some(MetaValue::Bool(b)) => Some(*b),
                _ => None,
            }
        };

        if let Some(v) = get_u32("dspark.block_count") { cfg.block_count = v as usize; }
        if let Some(v) = get_u32("dspark.embedding_length") { cfg.embedding_length = v as usize; }
        if let Some(v) = get_u32("dspark.attention.head_count") { cfg.head_count = v as usize; }
        if let Some(v) = get_u32("dspark.attention.head_count_kv") { cfg.head_count_kv = v as usize; }
        if let Some(v) = get_u32("dspark.attention.key_length") { cfg.head_dim = v as usize; }
        if let Some(v) = get_u32("dspark.feed_forward_length") { cfg.feed_forward_length = v as usize; }
        if let Some(v) = get_u32("dspark.dspark.block_size") { cfg.block_size = v as usize; }
        if let Some(v) = get_u32("dspark.dspark.markov_rank") { cfg.markov_rank = v as usize; }
        if let Some(v) = get_u32("dspark.dspark.mask_token_id") { cfg.mask_token_id = v; }
        if let Some(v) = get_u32("dspark.vocab_size") { cfg.vocab_size = v as usize; }
        if let Some(v) = get_u32("dspark.dspark.max_log_snr") { cfg.max_log_snr = v as f32; }
        if let Some(v) = get_u32("dspark.dspark.min_log_snr") { cfg.min_log_snr = v as f32; }
        if let Some(v) = get_f32("dspark.dspark.max_log_snr") { cfg.max_log_snr = v; }
        if let Some(v) = get_f32("dspark.dspark.min_log_snr") { cfg.min_log_snr = v; }
        if let Some(v) = get_bool("dspark.dspark.log_snr_conditioning") { cfg.log_snr_conditioning = v; }
        if let Some(v) = get_f32("dspark.rope.freq_base") { cfg.rope_freq_base = v; }
        if let Some(v) = get_f32("dspark.attention.layer_norm_rms_epsilon") { cfg.rms_eps = v; }
        if let Some(v) = get_bool("dspark.dspark.confidence_head") { cfg.confidence_head = v; }
        if let Some(v) = get_bool("dspark.dspark.confidence_head_with_markov") { cfg.confidence_head_with_markov = v; }

        // target_layers 是 array
        if let Some(MetaValue::Array(arr)) = meta.kv.get("dspark.dspark.target_layers") {
            cfg.target_layers = arr.iter().filter_map(|v| {
                match v {
                    MetaValue::Uint32(n) => Some(*n as usize),
                    MetaValue::Int32(n) => Some(*n as usize),
                    _ => None,
                }
            }).collect();
        }

        Ok(cfg)
    }

    /// GQA group size (head_count / head_count_kv)
    pub fn group_size(&self) -> usize {
        self.head_count / self.head_count_kv
    }

    /// target tap 特征维度 (n_capture × hidden)
    pub fn n_embd_cap(&self) -> usize {
        self.target_layers.len() * self.embedding_length
    }

    /// confidence head 输入维度
    /// with_markov=true: hidden + markov_rank = 5120 + 256 = 5376
    /// with_markov=false: hidden = 5120
    pub fn confidence_input_dim(&self) -> usize {
        if self.confidence_head_with_markov {
            self.embedding_length + self.markov_rank
        } else {
            self.embedding_length
        }
    }
}

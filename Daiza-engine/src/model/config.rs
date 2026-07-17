//! 模型超参数,从 GGUF metadata 读取
//!
//! 这些字段名与 GGUF KV 表中的命名空间 `qwen35.*` 一一对应。
//! 引擎加载时按以下顺序读取:
//!   1. `general.architecture` 必须为 `"qwen35"`
//!   2. 然后读取 `qwen35.*` 字段

use crate::gguf::metadata::Metadata;
use crate::BonsaiError;

#[derive(Debug, Clone)]
pub struct Config {
    // === 基础维度 ===
    pub block_count: usize,
    pub context_length: usize,
    pub hidden: usize,
    pub feed_forward_length: usize,

    // === Attention ===
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,           // 由 key_length 推断
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rope_dim_sections: Vec<i32>, // Qwen3 部分旋转分段
    pub rms_eps: f32,

    // === Hybrid attention 节拍 ===
    /// 周期:`block_idx % full_attention_interval == (interval - 1)` 时为全注意力层
    /// Bonsai 默认 4
    pub full_attention_interval: usize,

    // === SSM 配置(Mamba2 风格) ===
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
    pub ssm_inner_size: usize,

    // === 采样 ===
    pub vocab_size: usize,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    pub padding_token_id: u32,
}

impl Default for Config {
    fn default() -> Self {
        // Bonsai 27B 默认值(已从 GGUF metadata 验证)
        Self {
            block_count: 64,
            context_length: 262144,
            hidden: 5120,
            feed_forward_length: 17408,
            head_count: 24,
            head_count_kv: 4,
            head_dim: 256,
            rope_dim: 64,
            rope_freq_base: 1.0e7,
            rope_dim_sections: vec![11, 11, 10, 0],
            rms_eps: 1e-6,
            full_attention_interval: 4,
            ssm_conv_kernel: 4,
            ssm_state_size: 128,
            ssm_group_count: 16,
            ssm_time_step_rank: 48,
            ssm_inner_size: 6144,
            vocab_size: 248320,
            eos_token_id: 248046,
            bos_token_id: 248044,
            padding_token_id: 248044,
        }
    }
}

impl Config {
    /// 从 GGUF metadata 构造配置
    pub fn from_metadata(meta: &Metadata) -> crate::Result<Self> {
        let arch = meta.get_str("general.architecture").ok_or_else(|| {
            BonsaiError::Model("missing general.architecture".into())
        })?;
        if arch != "qwen35" {
            return Err(BonsaiError::Model(format!(
                "expected architecture 'qwen35', got '{arch}'"
            )));
        }

        let mut cfg = Config::default();
        if let Some(v) = meta.get_u32("qwen35.block_count") {
            cfg.block_count = v as usize;
        }
        if let Some(v) = meta.get_u64("qwen35.context_length") {
            cfg.context_length = v as usize;
        }
        if let Some(v) = meta.get_u32("qwen35.embedding_length") {
            cfg.hidden = v as usize;
        }
        if let Some(v) = meta.get_u32("qwen35.feed_forward_length") {
            cfg.feed_forward_length = v as usize;
        }
        if let Some(v) = meta.get_u32("qwen35.attention.head_count") {
            cfg.head_count = v as usize;
        }
        if let Some(v) = meta.get_u32("qwen35.attention.head_count_kv") {
            cfg.head_count_kv = v as usize;
        }
        if let Some(v) = meta.get_u32("qwen35.attention.key_length") {
            cfg.head_dim = v as usize;
        }
        if let Some(v) = meta.get_f32("qwen35.attention.layer_norm_rms_epsilon") {
            cfg.rms_eps = v;
        }
        if let Some(v) = meta.get_u32("qwen35.rope.dimension_count") {
            cfg.rope_dim = v as usize;
        }
        if let Some(v) = meta.get_i32_array("qwen35.rope.dimension_sections") {
            cfg.rope_dim_sections = v;
        }
        if let Some(v) = meta.get_f32("qwen35.rope.freq_base") {
            cfg.rope_freq_base = v;
        }
        if let Some(v) = meta.get_u32("qwen35.full_attention_interval") {
            cfg.full_attention_interval = v as usize;
        }

        // SSM
        if let Some(v) = meta.get_u32("qwen35.ssm.conv_kernel") {
            cfg.ssm_conv_kernel = v as usize;
        }
        if let Some(v) = meta.get_u32("qwen35.ssm.state_size") {
            cfg.ssm_state_size = v as usize;
        }
        if let Some(v) = meta.get_u32("qwen35.ssm.group_count") {
            cfg.ssm_group_count = v as usize;
        }
        if let Some(v) = meta.get_u32("qwen35.ssm.time_step_rank") {
            cfg.ssm_time_step_rank = v as usize;
        }
        if let Some(v) = meta.get_u32("qwen35.ssm.inner_size") {
            cfg.ssm_inner_size = v as usize;
        }

        // Tokenizer
        if let Some(v) = meta.get_i32_array("tokenizer.ggml.tokens").map(|t| t.len()) {
            cfg.vocab_size = v;
        }
        if let Some(v) = meta.get_u32("tokenizer.ggml.eos_token_id") {
            cfg.eos_token_id = v;
        }
        if let Some(v) = meta.get_u32("tokenizer.ggml.bos_token_id") {
            cfg.bos_token_id = v;
        }
        if let Some(v) = meta.get_u32("tokenizer.ggml.padding_token_id") {
            cfg.padding_token_id = v;
        }

        Ok(cfg)
    }

    /// 第 `block_idx` 个 block 是否为全注意力层
    pub fn is_full_attention_block(&self, block_idx: usize) -> bool {
        if self.full_attention_interval == 0 {
            return true;
        }
        block_idx % self.full_attention_interval == self.full_attention_interval - 1
    }

    /// 全注意力层的索引列表(共 16 个)
    pub fn full_attention_layer_indices(&self) -> Vec<usize> {
        (0..self.block_count)
            .filter(|&i| self.is_full_attention_block(i))
            .collect()
    }

    /// SSM 层的索引列表(共 48 个)
    pub fn ssm_layer_indices(&self) -> Vec<usize> {
        (0..self.block_count)
            .filter(|&i| !self.is_full_attention_block(i))
            .collect()
    }
}

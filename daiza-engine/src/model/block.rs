//! 统一的 Block 调度器
//!
//! 根据 `block_idx % full_attention_interval` 选择 SSM 或全注意力分支,
//! 然后调用对应的 `forward_single`。
//!
//! 残差结构:
//! ```text
//! x -> attn_norm -> [SSM 或 full attention] -> + x
//!   -> post_attention_norm -> MLP -> + x
//! ```

use crate::math;
use crate::model::config::Config;
use crate::model::weights::BlockWeights;
use crate::cache::{KvCache, SsmState};

pub struct BlockOutput {
    pub out: Vec<f32>,
}

/// 单 token 前向通过一个 block
#[allow(clippy::too_many_arguments)]
pub fn forward_single(
    x: &[f32],
    block_idx: usize,
    block_w: &BlockWeights,
    cfg: &Config,
    kv_cache: Option<&mut KvCache>,
    ssm_state: Option<&mut SsmState>,
    pos: usize,
    cos_sin: (&[f32], &[f32]),
) -> BlockOutput {
    let hidden = cfg.hidden;

    // 1. attn_norm (norm a COPY, keep original x for residual)
    let norm_w = match block_w {
        BlockWeights::Ssm(w) => &w.attn_norm,
        BlockWeights::FullAttention(w) => &w.attn_norm,
    };
    let mut h = x.to_vec();
    math::rmsnorm_inplace(&mut h, &norm_w.data, cfg.rms_eps);

    // 2. attention / SSM (uses normed h)
    let attn_out = match (block_w, kv_cache, ssm_state) {
        (BlockWeights::FullAttention(w), Some(kv), None) => {
            crate::model::attention::attention_forward_single(
                &h, w, cfg, kv, pos, cos_sin,
            ).out
        }
        (BlockWeights::Ssm(w), None, Some(ssm)) => {
            crate::model::ssm::ssm_forward_single(&h, w, cfg, ssm, pos).out
        }
        _ => Vec::new(),
    };

    // 3. residual add: out = x + attn_out (use ORIGINAL x, not normed h)
    let mut h = x.to_vec();
    for i in 0..hidden {
        h[i] += if i < attn_out.len() { attn_out[i] } else { 0.0 };
    }

    // 4. post_attention_norm (norm a COPY, keep residual stream h)
    let post_norm_w = match block_w {
        BlockWeights::Ssm(w) => &w.post_attention_norm,
        BlockWeights::FullAttention(w) => &w.post_attention_norm,
    };
    let mut mlp_in = h.clone();
    math::rmsnorm_inplace(&mut mlp_in, &post_norm_w.data, cfg.rms_eps);

    // 5. MLP (uses normed input)
    let (w_gate, w_up, w_down) = match block_w {
        BlockWeights::Ssm(w) => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
        BlockWeights::FullAttention(w) => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
    };
    let mlp_out = crate::model::mlp::mlp_forward_single(
        &mlp_in,
        w_gate,
        w_up,
        w_down,
    );

    // 6. residual add: out = h + mlp_out (residual stream preserved)
    for i in 0..hidden {
        h[i] += if i < mlp_out.out.len() { mlp_out.out[i] } else { 0.0 };
    }

    BlockOutput { out: h }
}

pub struct Block;

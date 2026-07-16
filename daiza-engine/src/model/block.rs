//! 统一的 Block 调度器
//!
//! 根据 `block_idx % full_attention_interval` 选择 SSM 或全注意力分支,
//! 然后调用对应的 `forward_into`。
//!
//! 残差结构(in-place 版本):
//! ```text
//! h → attn_norm → [SSM 或 full attention] → += h   (residual 1)
//!   → post_attention_norm → MLP → += h             (residual 2)
//! ```
//! 主残差流 `h` 在整个 block 中 in-place 更新,避免任何 `to_vec()` / `clone()`。

use crate::math;
use crate::model::config::Config;
use crate::model::weights::BlockWeights;
use crate::model::workspace::Workspace;
use crate::cache::{KvCache, SsmState};

/// 单 token 前向通过一个 block(in-place,无堆分配)
///
/// - 输入:`h` = 上一 block 的输出(主残差流)
/// - 输出:`h` = 本 block 的输出(原地更新)
/// - 所有中间 buffer 复用 `ws` 中预分配的字段
#[allow(clippy::too_many_arguments)]
pub fn forward_single_inplace(
    h: &mut [f32],
    block_w: &BlockWeights,
    cfg: &Config,
    kv_cache: Option<&mut KvCache>,
    ssm_state: Option<&mut SsmState>,
    cos_sin: (&[f32], &[f32]),
    ws: &mut Workspace,
) {
    let hidden = cfg.hidden;

    // 1. attention / SSM block:
    //    子函数内部负责 attn_norm + forward + 残差累加(h += attn_out)
    match (block_w, kv_cache, ssm_state) {
        (BlockWeights::FullAttention(w), Some(kv), None) => {
            crate::model::attention::attention_forward_into(
                h, w, cfg, kv, cos_sin, ws,
            );
        }
        (BlockWeights::Ssm(w), None, Some(ssm)) => {
            crate::model::ssm::ssm_forward_into(h, w, cfg, ssm, ws);
        }
        _ => {}
    }

    // 2. post_attention_norm: ws.block_normed = norm(h)
    //    ★ P0-3: 复用 block_normed(attention/ssm 已完成,不再需要此 buffer)
    //      原 block_mlp_in 已删除,省 20KB workspace
    let post_norm_w = match block_w {
        BlockWeights::Ssm(w) => &w.post_attention_norm,
        BlockWeights::FullAttention(w) => &w.post_attention_norm,
    };
    math::rmsnorm_into(&h[..hidden], &mut ws.block_normed, &post_norm_w.data, cfg.rms_eps);

    // 3. MLP: h += W_down @ (silu(W_gate @ mlp_in) * (W_up @ mlp_in))
    let (w_gate, w_up, w_down) = match block_w {
        BlockWeights::Ssm(w) => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
        BlockWeights::FullAttention(w) => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
    };
    crate::model::mlp::mlp_forward_into(
        &ws.block_normed,
        w_gate,
        w_up,
        w_down,
        &mut ws.mlp_gate,
        &mut ws.mlp_up,
        h,
    );
}

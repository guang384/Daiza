//! 顶层前向传播:把所有 block 串起来

use crate::cache::{KvCache, SsmState};
use crate::math;
use crate::model::config::Config;
use crate::model::weights::LoadedWeights;

/// 每层的运行时状态
pub struct ModelState {
    pub kv_caches: Vec<Option<KvCache>>,
    pub ssm_states: Vec<Option<SsmState>>,
    pub pos: usize,
    pub rope_freqs: Vec<f32>,
    pub rope_sections: Vec<i32>,
}

impl ModelState {
    pub fn new(cfg: &Config) -> Self {
        let mut kv_caches = Vec::with_capacity(cfg.block_count);
        let mut ssm_states = Vec::with_capacity(cfg.block_count);
        for blk_idx in 0..cfg.block_count {
            if cfg.is_full_attention_block(blk_idx) {
                kv_caches.push(Some(KvCache::new(
                    cfg.head_count_kv,
                    cfg.head_dim,
                    cfg.context_length,
                )));
                ssm_states.push(None);
            } else {
                kv_caches.push(None);
                // SsmState 维度: [num_v_heads=48, state_size=128, state_size=128]
                // 注意: ssm_time_step_rank=48 实际是 num_v_heads (GGUF 命名误导)
                ssm_states.push(Some(SsmState::new(cfg.ssm_time_step_rank, cfg.ssm_state_size)));
            }
        }
        Self {
            kv_caches,
            ssm_states,
            pos: 0,
            rope_freqs: math::rope_freqs(cfg.rope_dim, cfg.rope_freq_base),
            rope_sections: cfg.rope_dim_sections.clone(),
        }
    }

    pub fn cos_sin_at(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        math::rope_cos_sin_mrope_text(pos, &self.rope_freqs, &self.rope_sections)
    }

    pub fn reset(&mut self) {
        for kv in self.kv_caches.iter_mut().flatten() {
            kv.reset();
        }
        for ssm in self.ssm_states.iter_mut().flatten() {
            ssm.reset();
        }
        self.pos = 0;
    }
}

/// 前向上下文:持有加载的权重 + 运行时状态
pub struct ForwardContext<'a> {
    pub cfg: &'a Config,
    pub weights: &'a LoadedWeights,
    pub state: ModelState,
}

/// 单 token 前向,返回 logits [vocab_size]
pub fn forward_single_token(
    ctx: &mut ForwardContext<'_>,
    token_id: u32,
) -> crate::Result<Vec<f32>> {
    let cfg = ctx.cfg;
    let hidden = cfg.hidden;

    // 1. embedding lookup(流式:只反量化 token_id 对应的一行)
    let x = ctx.weights.global.token_embd.row(token_id as usize);

    // 2. 预计算当前 pos 的 cos/sin
    let pos = ctx.state.pos;
    let (cos, sin) = ctx.state.cos_sin_at(pos);

    // 3. 逐 block 前向(权重已在内存中,无需重新加载)
    let mut h = x;
    let block_start_ts = std::time::Instant::now();
    for blk_idx in 0..cfg.block_count {
        let block_w = &ctx.weights.blocks[blk_idx];
        let is_full = cfg.is_full_attention_block(blk_idx);

        let kv = if is_full {
            ctx.state.kv_caches[blk_idx].as_mut()
        } else {
            None
        };
        let ssm = if !is_full {
            ctx.state.ssm_states[blk_idx].as_mut()
        } else {
            None
        };

        let block_ts = std::time::Instant::now();
        let block_out = crate::model::block::forward_single(
            &h,
            blk_idx,
            block_w,
            cfg,
            kv,
            ssm,
            pos,
            (&cos, &sin),
        );
        eprint!("\r[block {blk_idx:>2}] {}ms", block_ts.elapsed().as_millis());
        h = block_out.out;
    }
    eprintln!("\r[forward] 64 blocks in {}ms", block_start_ts.elapsed().as_millis());

    // 4. final norm
    math::rmsnorm_inplace(&mut h, &ctx.weights.global.output_norm.data, cfg.rms_eps);

    // 5. LM head:流式 GEMM(逐行反量化 output + 累加)
    let logits = ctx.weights.global.output.matvec(&h);

    // 6. 推进位置
    ctx.state.pos += 1;

    Ok(logits)
}

/// 方便构造
pub fn make_context<'a>(
    weights: &'a LoadedWeights,
    cfg: &'a Config,
) -> ForwardContext<'a> {
    ForwardContext {
        cfg,
        weights,
        state: ModelState::new(cfg),
    }
}

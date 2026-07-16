//! 顶层前向传播:把所有 block 串起来

use crate::cache::{KvCache, SsmState};
use crate::math;
use crate::model::config::Config;
use crate::model::weights::LoadedWeights;
use crate::model::workspace::Workspace;

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

/// 前向上下文:持有加载的权重 + 运行时状态 + 跨 token 复用的工作区
///
/// `h_buf` 与 `workspace` 分离,避免 `forward_single_token` 内部
/// `&mut ctx.h_buf` 和 `&mut ctx.workspace` 的借用冲突。
pub struct ForwardContext<'a> {
    pub cfg: &'a Config,
    pub weights: &'a LoadedWeights,
    pub state: ModelState,
    pub h_buf: Vec<f32>,         // [hidden] 主残差流,跨 block in-place 更新
    pub workspace: Workspace,    // block/attention/ssm/mlp 的中间 buffer
    pub logits_buf: Vec<f32>,    // [vocab_size] LM head 输出,跨 token 复用避免 vec!
    pub cos_buf: Vec<f32>,       // [rope_dim] RoPE cos,跨 token 复用
    pub sin_buf: Vec<f32>,       // [rope_dim] RoPE sin,跨 token 复用
}

/// 单 token 前向,返回 logits [vocab_size]
///
/// v2 优化:主残差流 `h` 在 `ctx.h_buf` 中跨 block in-place 更新,
/// 所有中间 buffer 复用 `ctx.workspace` 中预分配的字段。
/// 每 token 仅 1 次堆分配(final logits 输出),其余 0 alloc。
pub fn forward_single_token(
    ctx: &mut ForwardContext<'_>,
    token_id: u32,
) -> crate::Result<Vec<f32>> {
    let cfg = ctx.cfg;
    let hidden = cfg.hidden;

    // 1. embedding lookup → ctx.h_buf
    //    ★ 通过 row_into_slice 直接写入预分配 buffer,避免返回 Vec
    ctx.weights.global.token_embd
        .row_into_slice(token_id as usize, &mut ctx.h_buf[..hidden]);

    // 2. 预计算当前 pos 的 cos/sin(写入预分配 buffer,避免每 token 分配 Vec)
    let pos = ctx.state.pos;
    math::rope_cos_sin_mrope_text_into(
        pos,
        &ctx.state.rope_freqs,
        &ctx.state.rope_sections,
        &mut ctx.cos_buf,
        &mut ctx.sin_buf,
    );
    let cos = &ctx.cos_buf[..];
    let sin = &ctx.sin_buf[..];

    // 3. 逐 block 前向(in-place 更新 h_buf)
    // ★ 热路径优化:移除每块 Instant::now() + eprint! 的系统调用开销
    //   (每 token × 64 块 = 64 次 syscall,~1-3ms/token 损耗)
    //   改为只在外层测量一次,通过 env 变量控制是否打印明细
    let debug_blocks = std::env::var("DAIZA_DEBUG_BLOCKS").is_ok();
    let block_start_ts = std::time::Instant::now();
    for blk_idx in 0..cfg.block_count {
        let is_full = cfg.is_full_attention_block(blk_idx);

        let block_ts = if debug_blocks { Some(std::time::Instant::now()) } else { None };

        if is_full {
            // 全注意力块:借用 ctx.h_buf / ctx.workspace / ctx.state.kv_caches[blk_idx]
            let kv = ctx.state.kv_caches[blk_idx].as_mut().unwrap();
            crate::model::block::forward_single_inplace(
                &mut ctx.h_buf,
                blk_idx,
                &ctx.weights.blocks[blk_idx],
                cfg,
                Some(kv),
                None,
                pos,
                (&cos, &sin),
                &mut ctx.workspace,
            );
        } else {
            // SSM 块:借用 ctx.h_buf / ctx.workspace / ctx.state.ssm_states[blk_idx]
            let ssm = ctx.state.ssm_states[blk_idx].as_mut().unwrap();
            crate::model::block::forward_single_inplace(
                &mut ctx.h_buf,
                blk_idx,
                &ctx.weights.blocks[blk_idx],
                cfg,
                None,
                Some(ssm),
                pos,
                (&cos, &sin),
                &mut ctx.workspace,
            );
        }

        if let Some(ts) = block_ts {
            eprint!("\r[block {blk_idx:>2}] {}ms", ts.elapsed().as_millis());
        }
    }
    if debug_blocks {
        eprintln!("\r[forward] 64 blocks in {}ms", block_start_ts.elapsed().as_millis());
    }

    // 4. final norm(in-place on h_buf)
    math::rmsnorm_inplace(&mut ctx.h_buf, &ctx.weights.global.output_norm.data, cfg.rms_eps);

    // 5. LM head:流式 GEMM(逐行反量化 output + 累加)
    //    ★ 用预分配 logits_buf 替代每 token 的 vec![0.0; vocab_size]
    //      clone 比 vec! 快(memcpy vs VirtualAlloc+memset),~50μs vs ~500μs
    if ctx.logits_buf.len() != cfg.vocab_size {
        ctx.logits_buf = vec![0.0; cfg.vocab_size];
    }
    ctx.weights.global.output.matvec_into_slice(&ctx.h_buf, &mut ctx.logits_buf);

    // 6. 推进位置
    ctx.state.pos += 1;

    Ok(ctx.logits_buf.clone())
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
        h_buf: vec![0.0; cfg.hidden],
        workspace: Workspace::new(cfg),
        logits_buf: Vec::with_capacity(cfg.vocab_size),
        cos_buf: vec![0.0; cfg.rope_dim],
        sin_buf: vec![0.0; cfg.rope_dim],
    }
}

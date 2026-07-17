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

/// 剖析开关:DAIZA_PROFILE env var,OnceLock 缓存避免热路径 env::var 开销
fn profile_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("DAIZA_PROFILE").is_ok())
}

/// 单 token 前向,logits 写入 `ctx.logits_buf`(无 clone)
///
/// v2 优化:主残差流 `h` 在 `ctx.h_buf` 中跨 block in-place 更新,
/// 所有中间 buffer 复用 `ctx.workspace` 中预分配的字段。
/// 调用方直接读 `ctx.logits_buf` 进行采样,避免每 token 1MB clone。
pub fn forward_single_token(
    ctx: &mut ForwardContext<'_>,
    token_id: u32,
) -> crate::Result<()> {
    let cfg = ctx.cfg;
    let hidden = cfg.hidden;
    let profile = profile_enabled();

    // 1. embedding lookup → ctx.h_buf
    //    ★ 通过 row_into_slice 直接写入预分配 buffer,避免返回 Vec
    let t0 = std::time::Instant::now();
    ctx.weights.global.token_embd
        .row_into_slice(token_id as usize, &mut ctx.h_buf[..hidden]);
    let t_emb = t0.elapsed();

    // 2. 预计算当前 pos 的 cos/sin(写入预分配 buffer,避免每 token 分配 Vec)
    let t1 = std::time::Instant::now();
    let pos = ctx.state.pos;
    math::rope_cos_sin_mrope_text_into(
        pos,
        &ctx.state.rope_freqs,
        &ctx.state.rope_sections,
        &mut ctx.cos_buf,
        &mut ctx.sin_buf,
    );
    let t_rope = t1.elapsed();
    let cos = &ctx.cos_buf[..];
    let sin = &ctx.sin_buf[..];

    // 3. 逐 block 前向(in-place 更新 h_buf)
    // ★ 热路径优化:移除每块 Instant::now() + eprint! 的系统调用开销
    //   (每 token × 64 块 = 64 次 syscall,~1-3ms/token 损耗)
    //   改为只在外层测量一次,通过 env 变量控制是否打印明细
    let debug_blocks = std::env::var("DAIZA_DEBUG_BLOCKS").is_ok();
    let block_start_ts = std::time::Instant::now();
    let mut attn_total = std::time::Duration::ZERO;
    let mut ssm_total = std::time::Duration::ZERO;
    if profile {
        crate::model::block::reset_timings();
    }
    for blk_idx in 0..cfg.block_count {
        let is_full = cfg.is_full_attention_block(blk_idx);

        let block_ts = if debug_blocks || profile {
            Some(std::time::Instant::now())
        } else {
            None
        };

        if is_full {
            // 全注意力块:借用 ctx.h_buf / ctx.workspace / ctx.state.kv_caches[blk_idx]
            let kv = ctx.state.kv_caches[blk_idx].as_mut().unwrap();
            crate::model::block::forward_single_inplace(
                &mut ctx.h_buf,
                &ctx.weights.blocks[blk_idx],
                cfg,
                Some(kv),
                None,
                (&cos, &sin),
                &mut ctx.workspace,
            );
        } else {
            // SSM 块:借用 ctx.h_buf / ctx.workspace / ctx.state.ssm_states[blk_idx]
            let ssm = ctx.state.ssm_states[blk_idx].as_mut().unwrap();
            crate::model::block::forward_single_inplace(
                &mut ctx.h_buf,
                &ctx.weights.blocks[blk_idx],
                cfg,
                None,
                Some(ssm),
                (&cos, &sin),
                &mut ctx.workspace,
            );
        }

        if let Some(ts) = block_ts {
            let elapsed = ts.elapsed();
            if is_full {
                attn_total += elapsed;
            } else {
                ssm_total += elapsed;
            }
            if debug_blocks {
                eprint!("\r[block {blk_idx:>2}] {}ms", elapsed.as_millis());
            }
        }
    }
    if debug_blocks {
        eprintln!("\r[forward] 64 blocks in {}ms", block_start_ts.elapsed().as_millis());
    }

    // 4. final norm(in-place on h_buf)
    let t3 = std::time::Instant::now();
    math::rmsnorm_inplace(&mut ctx.h_buf, &ctx.weights.global.output_norm.data, cfg.rms_eps);
    let t_final_norm = t3.elapsed();

    // 5. LM head:流式 GEMM(逐行反量化 output + 累加)
    //    ★ 用预分配 logits_buf 替代每 token 的 vec![0.0; vocab_size]
    let t4 = std::time::Instant::now();
    if ctx.logits_buf.len() != cfg.vocab_size {
        ctx.logits_buf = vec![0.0; cfg.vocab_size];
    }
    ctx.weights.global.output.matvec_into_slice(&ctx.h_buf, &mut ctx.logits_buf);
    let t_lm_head = t4.elapsed();

    // 6. 推进位置
    ctx.state.pos += 1;

    if profile {
        let blocks_ms = block_start_ts.elapsed().as_secs_f64() * 1000.0;
        let attn_ms = attn_total.as_secs_f64() * 1000.0;
        let ssm_ms = ssm_total.as_secs_f64() * 1000.0;
        let bt = crate::model::block::get_timings();
        let bt_attn = bt.attn_fwd.as_secs_f64() * 1000.0;
        let bt_ssm = bt.ssm_fwd.as_secs_f64() * 1000.0;
        let bt_mlp = bt.mlp.as_secs_f64() * 1000.0;
        let bt_pn = bt.post_norm.as_secs_f64() * 1000.0;
        eprintln!(
            "[profile] emb={:.2}ms rope={:.2}ms blocks={:.2}ms [attn(16)={:.2} ssm(48)={:.2}] | block_detail: attn_fwd={:.2} ssm_fwd={:.2} mlp(64)={:.2} post_norm={:.2} | final_norm={:.2}ms lm_head={:.2}ms total={:.2}ms",
            t_emb.as_secs_f64() * 1000.0,
            t_rope.as_secs_f64() * 1000.0,
            blocks_ms,
            attn_ms,
            ssm_ms,
            bt_attn,
            bt_ssm,
            bt_mlp,
            bt_pn,
            t_final_norm.as_secs_f64() * 1000.0,
            t_lm_head.as_secs_f64() * 1000.0,
            t_emb.as_secs_f64() * 1000.0 + t_rope.as_secs_f64() * 1000.0 + blocks_ms
                + t_final_norm.as_secs_f64() * 1000.0 + t_lm_head.as_secs_f64() * 1000.0,
        );
    }

    Ok(())
}

/// 批量前向传播(用于 prefill 阶段加速)
///
/// **核心优化**: 对每个 block, 批量计算所有 token 的 matvec, 把 13GB 权重只读一次
/// 而非 N 次。attention 和 SSM 仍逐 token 顺序计算(状态依赖)。
///
/// `h_batch`: [n_batch, hidden] 行优先, 原地更新
/// `token_ids`: [n_batch] 输入 token IDs
/// `start_pos`: batch 起始位置
///
/// 最后一个 token 的 logits 写入 `ctx.logits_buf`(无 clone, 直接读)
pub fn forward_batch(
    ctx: &mut ForwardContext<'_>,
    token_ids: &[u32],
    start_pos: usize,
) -> crate::Result<()> {
    let cfg = ctx.cfg;
    let hidden = cfg.hidden;
    let n_batch = token_ids.len();
    if n_batch == 0 {
        return Ok(());
    }

    let profile = profile_enabled();
    let mut p_emb = std::time::Duration::ZERO;
    let mut p_cos_sin = std::time::Duration::ZERO;
    let mut p_batch_rmsnorm = std::time::Duration::ZERO;
    let mut p_batch_matvec = std::time::Duration::ZERO;
    let mut p_attn_serial = std::time::Duration::ZERO;
    let mut p_ssm_serial = std::time::Duration::ZERO;
    let mut p_swiglu = std::time::Duration::ZERO;
    let mut p_final = std::time::Duration::ZERO;

    // 1. embedding lookup: ctx.h_buf 需要扩展为 batch 大小
    let t0 = if profile { Some(std::time::Instant::now()) } else { None };
    ctx.h_buf.resize(n_batch * hidden, 0.0);
    for t in 0..n_batch {
        ctx.weights.global.token_embd
            .row_into_slice(token_ids[t] as usize, &mut ctx.h_buf[t * hidden..(t + 1) * hidden]);
    }
    if let Some(t) = t0 { p_emb = t.elapsed(); }

    // 2. 预分配 batch 临时 buffer(跨 block 复用)
    // 所有维度均从 cfg 直接派生,避免通过 block[1] 间接访问(脆弱依赖)
    let ffn_dim = cfg.feed_forward_length;
    let n_q_heads = cfg.head_count;
    let n_kv_heads = cfg.head_count_kv;
    let head_dim = cfg.head_dim;
    let qkv_total_dim = n_q_heads * head_dim * 2; // Q + gate interleaved
    let attn_out_dim = n_q_heads * head_dim;
    // SSM block 派生常量(从 cfg 直接计算,与 ssm.rs 内部公式一致)
    let ssm_state_size = cfg.ssm_state_size;
    let ssm_num_v_heads = cfg.ssm_time_step_rank; // 48
    let ssm_num_k_heads = cfg.ssm_group_count;    // 16
    let ssm_qkv_dim = 2 * ssm_num_k_heads * ssm_state_size + cfg.ssm_inner_size; // 10240
    let ssm_out_dim = hidden;                       // ssm_out 投影回 hidden
    let ssm_alpha_dim = ssm_num_v_heads;            // 48
    let ssm_gate_dim = cfg.ssm_inner_size;          // 6144

    let mut normed_batch = vec![0.0f32; n_batch * hidden];
    let mut qkv_buf = vec![0.0f32; n_batch * qkv_total_dim.max(ssm_qkv_dim).max(ffn_dim)];
    let mut attn_out_buf = vec![0.0f32; n_batch * attn_out_dim.max(ssm_out_dim)];
    let mut k_buf = vec![0.0f32; n_batch * n_kv_heads * head_dim];
    let mut v_buf = vec![0.0f32; n_batch * n_kv_heads * head_dim];
    let mut tmp_buf = vec![0.0f32; n_batch * ffn_dim]; // MLP gate/up
    let mut ssm_alpha_buf = vec![0.0f32; n_batch * ssm_alpha_dim];
    let mut ssm_beta_buf = vec![0.0f32; n_batch * ssm_alpha_dim];
    let mut ssm_gate_buf = vec![0.0f32; n_batch * ssm_gate_dim];

    // 3. 逐 block 前向
    let debug_blocks = std::env::var("DAIZA_DEBUG_BLOCKS").is_ok();
    let block_start_ts = std::time::Instant::now();

    // 预计算所有 batch token 的 cos/sin(避免与 kv cache 的 borrow 冲突)
    let t0 = if profile { Some(std::time::Instant::now()) } else { None };
    let cos_sin_batch: Vec<_> = (0..n_batch)
        .map(|t| ctx.state.cos_sin_at(start_pos + t))
        .collect();
    if let Some(t) = t0 { p_cos_sin = t.elapsed(); }

    for blk_idx in 0..cfg.block_count {
        let is_full = cfg.is_full_attention_block(blk_idx);
        let block_ts = if debug_blocks { Some(std::time::Instant::now()) } else { None };

        let block_w = &ctx.weights.blocks[blk_idx];

        if is_full {
            let kv = ctx.state.kv_caches[blk_idx].as_mut().unwrap();
            let w = block_w.as_full_attention();

            // 3a. Batch rmsnorm
            let ts = if profile { Some(std::time::Instant::now()) } else { None };
            for t in 0..n_batch {
                let src = &ctx.h_buf[t * hidden..(t + 1) * hidden];
                let dst = &mut normed_batch[t * hidden..(t + 1) * hidden];
                math::rmsnorm_into(src, dst, &w.attn_norm.data, cfg.rms_eps);
            }
            if let Some(ts) = ts { p_batch_rmsnorm += ts.elapsed(); }

            // 3b. Batch Q matvec: qkv_buf[t] = W_q @ normed[t]
            // 3c. Batch K and V matvecs
            let ts = if profile { Some(std::time::Instant::now()) } else { None };
            w.attn_q.matvec_batch_into_slice(&normed_batch, n_batch, &mut qkv_buf[..n_batch * qkv_total_dim]);
            w.attn_k.matvec_batch_into_slice(&normed_batch, n_batch, &mut k_buf);
            w.attn_v.matvec_batch_into_slice(&normed_batch, n_batch, &mut v_buf);
            if let Some(ts) = ts { p_batch_matvec += ts.elapsed(); }

            // 3d. Per-token attention (sequential)
            let ts_attn = if profile { Some(std::time::Instant::now()) } else { None };
            for t in 0..n_batch {
                let qkv_t = &qkv_buf[t * qkv_total_dim..(t + 1) * qkv_total_dim];

                // Deinterlace Q and gate → ws.attn_q / ws.attn_gate (Q 拆分必需)
                for h_i in 0..n_q_heads {
                    let src = h_i * (head_dim * 2);
                    let dst = h_i * head_dim;
                    ctx.workspace.attn_q[dst..dst + head_dim].copy_from_slice(&qkv_t[src..src + head_dim]);
                    ctx.workspace.attn_gate[dst..dst + head_dim].copy_from_slice(&qkv_t[src + head_dim..src + 2 * head_dim]);
                }

                // ★ P1-5: 不再 copy K/V 到 workspace, K norm+RoPE 直接在 k_buf 上 in-place

                // QK-norm + RoPE 融合 (减少循环开销)
                let (cos, sin) = &cos_sin_batch[t];
                let cos = &cos[..];
                let sin = &sin[..];
                let rope_dim = cfg.rope_dim;

                for h_i in 0..n_q_heads {
                    let hs = h_i * head_dim;
                    math::rmsnorm_inplace(&mut ctx.workspace.attn_q[hs..hs + head_dim], &w.attn_q_norm.data, cfg.rms_eps);
                    math::apply_rope_partial(&mut ctx.workspace.attn_q[hs..hs + head_dim], rope_dim, cos, sin);
                }
                // K norm + RoPE in-place on k_buf (省一次 K copy 到 workspace)
                let kv_off = t * n_kv_heads * head_dim;
                let k_t_mut = &mut k_buf[kv_off..kv_off + n_kv_heads * head_dim];
                for h_i in 0..n_kv_heads {
                    let hs = h_i * head_dim;
                    math::rmsnorm_inplace(&mut k_t_mut[hs..hs + head_dim], &w.attn_k_norm.data, cfg.rms_eps);
                    math::apply_rope_partial(&mut k_t_mut[hs..hs + head_dim], rope_dim, cos, sin);
                }

                // Append KV (直接读 k_buf/v_buf, 省 ws.attn_k/v copy)
                kv.append(
                    &k_buf[kv_off..kv_off + n_kv_heads * head_dim],
                    &v_buf[kv_off..kv_off + n_kv_heads * head_dim],
                );

                // Attention scores + V weighted sum
                let n_cached = kv.len;
                let scale = 1.0 / (head_dim as f32).sqrt();
                // ★ P1-5: 直接写 attn_out_buf[t..], 省末尾 attn_out copy
                let out_t = &mut attn_out_buf[t * attn_out_dim..(t + 1) * attn_out_dim];
                out_t[..attn_out_dim].fill(0.0);
                let scores = &mut ctx.workspace.attn_scores[..n_cached];
                let group_size = n_q_heads / n_kv_heads;

                for qh in 0..n_q_heads {
                    let kvh = qh / group_size;
                    let q_head = &ctx.workspace.attn_q[qh * head_dim..(qh + 1) * head_dim];
                    for c in 0..n_cached {
                        let k_t_c = kv.k_at(c);
                        let k_head = &k_t_c[kvh * head_dim..(kvh + 1) * head_dim];
                        scores[c] = crate::math::simd_exp::dot_product_avx2(q_head, k_head, head_dim) * scale;
                    }
                    math::softmax_inplace(scores);
                    let out_head = &mut out_t[qh * head_dim..(qh + 1) * head_dim];
                    for c in 0..n_cached {
                        let v_head = &kv.v_at(c)[kvh * head_dim..(kvh + 1) * head_dim];
                        crate::math::simd_exp::saxpy_avx2(scores[c], v_head, out_head, head_dim);
                    }
                }

                // Gate (直接 apply 到 out_t, 省一次 copy)
                math::sigmoid_inplace_simd(&mut ctx.workspace.attn_gate);
                math::mul_inplace_simd(&mut out_t[..attn_out_dim], &ctx.workspace.attn_gate[..attn_out_dim]);
            }
            if let Some(ts) = ts_attn { p_attn_serial += ts.elapsed(); }

            // 3e. Batch output projection: h += W_out @ attn_out
            let ts = if profile { Some(std::time::Instant::now()) } else { None };
            w.attn_output.matvec_add_batch_into_slice(&attn_out_buf[..n_batch * attn_out_dim], n_batch, &mut ctx.h_buf);
            if let Some(ts) = ts { p_batch_matvec += ts.elapsed(); }
        } else {
            let ssm = ctx.state.ssm_states[blk_idx].as_mut().unwrap();
            let w = block_w.as_ssm();
            let ssm_qkv_dim = w.attn_qkv.rows;

            // 3a. Batch rmsnorm
            let ts = if profile { Some(std::time::Instant::now()) } else { None };
            for t in 0..n_batch {
                let src = &ctx.h_buf[t * hidden..(t + 1) * hidden];
                let dst = &mut normed_batch[t * hidden..(t + 1) * hidden];
                math::rmsnorm_into(src, dst, &w.attn_norm.data, cfg.rms_eps);
            }
            if let Some(ts) = ts { p_batch_rmsnorm += ts.elapsed(); }

            // 3b. Batch SSM matvecs
            let ts = if profile { Some(std::time::Instant::now()) } else { None };
            w.attn_qkv.matvec_batch_into_slice(&normed_batch, n_batch, &mut qkv_buf[..n_batch * ssm_qkv_dim]);
            w.ssm_alpha.matvec_batch_into_slice(&normed_batch, n_batch, &mut ssm_alpha_buf);
            w.ssm_beta.matvec_batch_into_slice(&normed_batch, n_batch, &mut ssm_beta_buf);
            w.attn_gate.matvec_batch_into_slice(&normed_batch, n_batch, &mut ssm_gate_buf);
            if let Some(ts) = ts { p_batch_matvec += ts.elapsed(); }

            // 3c. Per-token SSM (conv1d + silu + L2 norm + q_scale + scan + output gate)
            //   所有 cfg 派生常量在外层已计算,这里只取本 block 权重引用
            let state_size = ssm_state_size;
            let num_v_heads = ssm_num_v_heads;
            let num_k_heads = ssm_num_k_heads;
            let v_heads_per_group = num_v_heads / num_k_heads;
            let conv_k = cfg.ssm_conv_kernel;
            let qkv_dim = num_k_heads * state_size;
            let inner = num_v_heads * state_size;
            let qkv_full_len = ssm_qkv_dim;
            let ssm_norm_w = &w.ssm_norm.data;
            let dt_bias = &w.ssm_dt_bias.data;
            let a = &w.ssm_a.data;

            // ★ P0-2: 不再把 alpha/beta/z/qkv copy 到 workspace
            //   - alpha/beta 只标量访问,直接读 batch buffer
            //   - z (gate) 在 output gate 里读,直接读 batch buffer
            //   - qkv 在 conv1d 里只用于 copy 进 conv_history,直接用 batch buffer
            let ts_ssm = if profile { Some(std::time::Instant::now()) } else { None };
            for t in 0..n_batch {
                let qkv_t = &qkv_buf[t * ssm_qkv_dim..(t + 1) * ssm_qkv_dim];
                let alpha_t = &ssm_alpha_buf[t * ssm_alpha_dim..(t + 1) * ssm_alpha_dim];
                let beta_t = &ssm_beta_buf[t * ssm_alpha_dim..(t + 1) * ssm_alpha_dim];
                let gate_t = &mut ssm_gate_buf[t * ssm_gate_dim..(t + 1) * ssm_gate_dim];

                // Conv1d (depthwise, causal, kernel=4) + silu
                if ssm.conv_history.is_empty() {
                    ssm.conv_history.resize(conv_k * qkv_full_len, 0.0);
                    ssm.conv_head = 0;
                }
                // ★ P2-2: 环形 buffer — 写入 conv_head 行, 然后 head 前进 (省滑窗左移 copy)
                let cur_off = ssm.conv_head * qkv_full_len;
                ssm.conv_history[cur_off..cur_off + qkv_full_len].copy_from_slice(qkv_t);
                ssm.conv_head = (ssm.conv_head + 1) % conv_k;

                let conv_w = &w.ssm_conv1d.data;
                ctx.workspace.ssm_conv_out.fill(0.0);
                // ★ P2-2: 环形读取 — 第 ct 个历史 token 在 (conv_head + ct) % conv_k 行
                for ct in 0..conv_k {
                    let row = (ssm.conv_head + ct) % conv_k;
                    let hist_row = &ssm.conv_history[row * qkv_full_len..(row + 1) * qkv_full_len];
                    for ch in 0..qkv_full_len {
                        ctx.workspace.ssm_conv_out[ch] += hist_row[ch] * conv_w[ch * conv_k + ct];
                    }
                }
                // ★ P2-9: silu 向量化 — 先原地 SIMD silu,再 memcpy 拆分
                use crate::math::simd_exp::silu_inplace_simd;
                silu_inplace_simd(&mut ctx.workspace.ssm_conv_out[..2 * qkv_dim + inner]);
                ctx.workspace.ssm_q[..qkv_dim].copy_from_slice(&ctx.workspace.ssm_conv_out[..qkv_dim]);
                ctx.workspace.ssm_k[..qkv_dim].copy_from_slice(&ctx.workspace.ssm_conv_out[qkv_dim..2 * qkv_dim]);
                ctx.workspace.ssm_v[..inner].copy_from_slice(&ctx.workspace.ssm_conv_out[2 * qkv_dim..2 * qkv_dim + inner]);

                // L2 norm q/k per head
                let l2norm_eps = 1e-6f32;
                for h_i in 0..num_k_heads {
                    let hs = h_i * state_size;
                    let he = hs + state_size;
                    let mut ss = 0.0f32;
                    for j in hs..he {
                        ss += ctx.workspace.ssm_q[j] * ctx.workspace.ssm_q[j];
                    }
                    let inv_norm = 1.0 / (ss + l2norm_eps).sqrt();
                    for j in hs..he {
                        ctx.workspace.ssm_q[j] *= inv_norm;
                    }
                    ss = 0.0;
                    for j in hs..he {
                        ss += ctx.workspace.ssm_k[j] * ctx.workspace.ssm_k[j];
                    }
                    let inv_norm = 1.0 / (ss + l2norm_eps).sqrt();
                    for j in hs..he {
                        ctx.workspace.ssm_k[j] *= inv_norm;
                    }
                }

                // q scale: q *= 1/sqrt(head_dim)
                let q_scale = 1.0 / (state_size as f32).sqrt();
                for qi in ctx.workspace.ssm_q.iter_mut() {
                    *qi *= q_scale;
                }

                // Gated Delta Rule scan per v_head
                // ★ alpha/beta 直接读 batch buffer(省 2 × 48 × 4B = 384B copy/token)
                // ★ P1-5: scan 直接写 attn_out_buf[t..], 省 ssm_y 末尾 copy
                let y_t = &mut attn_out_buf[t * inner..(t + 1) * inner];
                // ssm_scan_vhead 内部完全覆盖 y (不是累加), 无需 fill(0)
                for vh in 0..num_v_heads {
                    let kh = vh / v_heads_per_group;
                    let q_head = &ctx.workspace.ssm_q[kh * state_size..(kh + 1) * state_size];
                    let k_head = &ctx.workspace.ssm_k[kh * state_size..(kh + 1) * state_size];
                    let v_head = &ctx.workspace.ssm_v[vh * state_size..(vh + 1) * state_size];
                    let s_off = vh * state_size * state_size;
                    let s = &mut ssm.state[s_off..s_off + state_size * state_size];
                    let y_off = vh * state_size;
                    let y = &mut y_t[y_off..y_off + state_size];
                    crate::model::ssm::ssm_scan_vhead(
                        s, y, q_head, k_head, v_head,
                        a[vh], alpha_t[vh], beta_t[vh], dt_bias[vh],
                        state_size,
                    );
                }

                // Output gate: y = rmsnorm(y) * ssm_norm_w * silu(z)
                // ★ P2-9: 预先对 gate_t 做原地 SIMD silu (省 6144 次标量 silu_fast)
                // ★ z 直接读 batch buffer 的 gate_t(省 6144 × 4B = 24KB copy/token)
                // ★ P1-5: 直接 in-place 修改 y_t (省 ssm_y 末尾 copy)
                crate::math::simd_exp::silu_inplace_simd(gate_t);
                for vh in 0..num_v_heads {
                    let y_off = vh * state_size;
                    let mut ss = 0.0f32;
                    for i in 0..state_size {
                        ss += y_t[y_off + i] * y_t[y_off + i];
                    }
                    let inv_rms = 1.0 / (ss / state_size as f32 + l2norm_eps).sqrt();
                    for i in 0..state_size {
                        let normed = y_t[y_off + i] * inv_rms;
                        y_t[y_off + i] = normed * ssm_norm_w[i] * gate_t[y_off + i];
                    }
                }
            }
            if let Some(ts) = ts_ssm { p_ssm_serial += ts.elapsed(); }

            // 3d. Batch output projection: h += W_out @ ssm_y_batch
            let ts = if profile { Some(std::time::Instant::now()) } else { None };
            w.ssm_out.matvec_add_batch_into_slice(&attn_out_buf[..n_batch * inner], n_batch, &mut ctx.h_buf);
            if let Some(ts) = ts { p_batch_matvec += ts.elapsed(); }
        }

        // 4. Post-attention norm + MLP (batch)
        let (post_norm, w_gate, w_up, w_down) = block_w.post_norm_and_ffn();

        // 4a. Batch post-attention norm
        let ts = if profile { Some(std::time::Instant::now()) } else { None };
        for t in 0..n_batch {
            let src = &ctx.h_buf[t * hidden..(t + 1) * hidden];
            let dst = &mut normed_batch[t * hidden..(t + 1) * hidden];
            math::rmsnorm_into(src, dst, &post_norm.data, cfg.rms_eps);
        }
        if let Some(ts) = ts { p_batch_rmsnorm += ts.elapsed(); }

        // 4b. Batch MLP matvecs (gate → qkv_buf, up → tmp_buf, 直接复用无需 take)
        let ts = if profile { Some(std::time::Instant::now()) } else { None };
        w_gate.matvec_batch_into_slice(&normed_batch, n_batch, &mut qkv_buf[..n_batch * ffn_dim]);
        w_up.matvec_batch_into_slice(&normed_batch, n_batch, &mut tmp_buf[..n_batch * ffn_dim]);
        if let Some(ts) = ts { p_batch_matvec += ts.elapsed(); }

        // 4c. Per-token SwiGLU: gate = silu(gate) * up (in-place on qkv_buf, 读 tmp_buf)
        //    qkv_buf 和 tmp_buf 是不同 Vec,可同时 &mut qkv_buf[..] 和 &tmp_buf[..]
        let ts = if profile { Some(std::time::Instant::now()) } else { None };
        for t in 0..n_batch {
            math::swiglu_inplace(
                &mut qkv_buf[t * ffn_dim..(t + 1) * ffn_dim],
                &tmp_buf[t * ffn_dim..(t + 1) * ffn_dim],
            );
        }
        if let Some(ts) = ts { p_swiglu += ts.elapsed(); }

        // 4d. Batch down projection: h += W_down @ gate
        let ts = if profile { Some(std::time::Instant::now()) } else { None };
        w_down.matvec_add_batch_into_slice(&qkv_buf[..n_batch * ffn_dim], n_batch, &mut ctx.h_buf);
        if let Some(ts) = ts { p_batch_matvec += ts.elapsed(); }

        if let Some(ts) = block_ts {
            eprint!("\r[block {blk_idx:>2}] {}ms", ts.elapsed().as_millis());
        }
    }

    if debug_blocks {
        eprintln!("\r[batch forward] {} blocks in {}ms", cfg.block_count, block_start_ts.elapsed().as_millis());
    }

    // 5. Final norm + LM head — 只算最后一个 token (省 (n_batch-1) × 179MB 权重读取)
    //    logits 写入 ctx.logits_buf,decode 阶段直接读
    let t0 = if profile { Some(std::time::Instant::now()) } else { None };
    let last_h = &mut ctx.h_buf[(n_batch - 1) * hidden..n_batch * hidden];
    math::rmsnorm_inplace(last_h, &ctx.weights.global.output_norm.data, cfg.rms_eps);
    if ctx.logits_buf.len() != cfg.vocab_size {
        ctx.logits_buf = vec![0.0; cfg.vocab_size];
    }
    ctx.weights.global.output.matvec_into_slice(last_h, &mut ctx.logits_buf);
    if let Some(t) = t0 { p_final = t.elapsed(); }

    // 6. Restore h_buf to single-token size for decode phase
    ctx.h_buf.truncate(hidden);

    // 7. Advance position
    ctx.state.pos = start_pos + n_batch;

    if profile {
        let blocks_ms = block_start_ts.elapsed().as_secs_f64() * 1000.0;
        let serial_ms = p_attn_serial.as_secs_f64() * 1000.0
            + p_ssm_serial.as_secs_f64() * 1000.0
            + p_swiglu.as_secs_f64() * 1000.0;
        let parallel_ms = p_batch_matvec.as_secs_f64() * 1000.0
            + p_batch_rmsnorm.as_secs_f64() * 1000.0;
        let total_ms = p_emb.as_secs_f64() * 1000.0
            + p_cos_sin.as_secs_f64() * 1000.0
            + blocks_ms
            + p_final.as_secs_f64() * 1000.0;
        let ratio = if total_ms > 0.0 { serial_ms / total_ms * 100.0 } else { 0.0 };
        eprintln!(
            "[prefill-profile] n_batch={n_batch} blocks={block_count} | total={total_ms:.1}ms",
            block_count = cfg.block_count,
        );
        eprintln!(
            "  emb={emb_ms:.2}ms cos_sin={cos_ms:.2}ms final={final_ms:.2}ms",
            emb_ms = p_emb.as_secs_f64() * 1000.0,
            cos_ms = p_cos_sin.as_secs_f64() * 1000.0,
            final_ms = p_final.as_secs_f64() * 1000.0,
        );
        eprintln!(
            "  blocks={blocks_ms:.1}ms | batch_parallel={parallel_ms:.1}ms [matvec={matvec_ms:.1} rmsnorm={rmsnorm_ms:.1}]",
            matvec_ms = p_batch_matvec.as_secs_f64() * 1000.0,
            rmsnorm_ms = p_batch_rmsnorm.as_secs_f64() * 1000.0,
        );
        eprintln!(
            "  per_token_serial={serial_ms:.1}ms [attn={attn_ms:.1} ssm={ssm_ms:.1} swiglu={swiglu_ms:.1}] serial_ratio={ratio:.1}%",
            attn_ms = p_attn_serial.as_secs_f64() * 1000.0,
            ssm_ms = p_ssm_serial.as_secs_f64() * 1000.0,
            swiglu_ms = p_swiglu.as_secs_f64() * 1000.0,
        );
    }

    Ok(())
}
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

//! DSpark drafter block-parallel forward
//!
//! 一次 forward 处理 block_size=4 个 draft 位置 (非自回归, 非因果 attention)。
//!
//! ## 算子序列
//!
//! 1. Embedding: anchor (真实 token) + mask_token × (block_size-1)
//! 2. Log-SNR conditioning: sinusoidal feat → FC1+SiLU → FC2 → 加到 embedding
//! 3. 6 层 transformer (非因果 attention):
//!    - attn_in = concat(target_ctx, cur)  // context 行只贡献 K/V
//!    - Q/K/V proj, Q/K norm, RoPE
//!    - non-causal attention (全开 mask)
//!    - 切掉 context 行的 output
//!    - 残差 + SwiGLU FFN
//! 4. Output norm + LM head → base logits [block_size, vocab]
//!
//! 注意: drafter context 行 (来自 target tap) 只贡献 K/V, 不进 query/FFN/残差。

use crate::math;
use crate::tensor::quant::dequantize_q1_0_row_into;

use super::weights::DrafterWeights;

// ---------------------------------------------------------------------------
// env var 缓存 (drafter 每次 draft_forward 调用, 避免重复 env::var syscall)
// ---------------------------------------------------------------------------
fn no_kv_cache() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("DAIZA_DSPARK_NO_KV_CACHE").is_ok())
}

fn no_snr() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("DAIZA_DSPARK_NO_SNR").is_ok())
}

fn no_fc_cache() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("DAIZA_DSPARK_NO_FC_CACHE").is_ok())
}

/// DSpark drafter 运行时上下文 (持有 workspace buffers)
pub struct DrafterContext {
    pub weights: DrafterWeights,
    /// RoPE cos/sin 缓冲 (跨 token 复用)
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
    /// workspace buffers (预分配, 避免 draft 时堆分配)
    pub ws_embd: Vec<f32>,          // [block_size * hidden]
    pub ws_snr_feat: Vec<f32>,      // [block_size * n_freq]
    pub ws_snr_hidden: Vec<f32>,    // [block_size * hidden]
    pub ws_snr_embed: Vec<f32>,     // [block_size * hidden]
    pub ws_target_ctx: Vec<f32>,    // [ctx_len * hidden] (投影后的 target context, 跨 call 缓存)
    /// fc 缓存: 已投影+normed 的 target_ctx 行数 (跨 draft_forward call 复用)
    /// 每次 draft_forward 只投影新增行 [cached_ctx_len..ctx_len], 避免全量重投影
    pub cached_ctx_len: usize,
    /// K/V cache: 已计算 (proj + norm + RoPE) 的 context 行数 (跨 draft_forward call 复用)
    /// context 行 (前 ctx_len 行) 的 attn_in 来自 ws_target_ctx, 跨 cycle 稳定,
    /// 故 K/V proj+norm+RoPE 结果可缓存。draft 行 (后 bs 行) 每 cycle 不同, 不缓存。
    /// 每 cycle 平均新增 ctx_len ≈ 2.23 行, 缓存命中率接近 100%。
    pub cached_kv_len: usize,
    /// per-layer K cache: [block_count][cached_kv_len * n_kvh * hd]
    pub ws_k_cache: Vec<Vec<f32>>,
    /// per-layer V cache: [block_count][cached_kv_len * n_kvh * hd]
    pub ws_v_cache: Vec<Vec<f32>>,
    pub ws_attn_in: Vec<f32>,       // [(ctx_len + block_size) * hidden]
    pub ws_q: Vec<f32>,             // [(ctx_len + block_size) * n_q_heads * head_dim]
    pub ws_k: Vec<f32>,             // [(ctx_len + block_size) * n_kv_heads * head_dim]
    pub ws_v: Vec<f32>,             // [(ctx_len + block_size) * n_kv_heads * head_dim]
    pub ws_attn_out: Vec<f32>,      // [block_size * hidden]
    pub ws_ffn_gate: Vec<f32>,      // [block_size * ffn]
    pub ws_ffn_up: Vec<f32>,        // [block_size * ffn]
    pub ws_hidden: Vec<f32>,        // [block_size * hidden] (每层输出)
    pub ws_logits: Vec<f32>,        // [block_size * vocab] (base logits)
    pub ws_kq_scale: f32,           // 1/sqrt(head_dim)
    // ★ P2: 热路径复用 buffer (消除 forward_block 内 vec![] 分配)
    pub ws_attn_tmp: Vec<f32>,      // [bs * hidden] attn output proj 中间结果
    pub ws_ffn_normed: Vec<f32>,    // [bs * hidden] FFN normed
    pub ws_ffn_out: Vec<f32>,       // [bs * hidden] FFN output
    /// 预计算的 RoPE 频率 [head_dim/2], 避免每次 compute_rope_into 重复 powf
    pub rope_freqs: Vec<f32>,
    /// ★ log_snr 缓存标志: ws_snr_embed 只依赖 cfg (max/min_log_snr, fc1/fc2 权重),
    /// 跨 draft_forward call 完全不变。首次 compute_log_snr 后置 true, 后续直接复用。
    /// FC2 是 BF16 标量 matvec [5120,5120]×4 = 105M FMA, 缓存后节省 ~30-50ms/call。
    pub snr_computed: bool,
    /// ★ Confidence head 输入: RMSNorm 之前的 drafter hidden state [bs * hidden]
    /// 在 output RMSNorm 之前 copy ws_hidden → ws_confidence_hidden, 供 confidence head 用。
    /// DeepSpec proposal_hidden_states = drafter 最后一层 block 输出 (norm 前)。
    pub ws_confidence_hidden: Vec<f32>,
}

impl DrafterContext {
    /// 加载 drafter 并初始化 workspace
    pub fn new(weights: DrafterWeights) -> Self {
        let cfg = &weights.cfg;
        let bs = cfg.block_size;
        let h = cfg.embedding_length;
        let n_freq = cfg.log_snr_n_freq;
        let n_qh = cfg.head_count;
        let n_kvh = cfg.head_count_kv;
        let hd = cfg.head_dim;
        let ffn = cfg.feed_forward_length;
        let vocab = cfg.vocab_size;
        let block_count = cfg.block_count;
        // 预计算 RoPE 频率: freq[i] = 1 / (freq_base ^ (2i/hd)), i ∈ [0, hd/2)
        // ★ drafter 使用全维度 RoPE (head_dim=128, 非 partial M-RoPE)。
        //   实测验证: 用 target 的 M-RoPE partial (rope_dim=64, sections=[11,11,10,0])
        //   接受率从 36% 暴跌到 3%, 说明 drafter 训练时就是全维度标准 RoPE。
        let rope_freqs: Vec<f32> = (0..hd / 2)
            .map(|i| 1.0 / cfg.rope_freq_base.powf((2 * i) as f32 / hd as f32))
            .collect();

        Self {
            weights,
            rope_cos: vec![0.0; hd],  // drafter RoPE 全维度 (非 partial)
            rope_sin: vec![0.0; hd],
            ws_embd: vec![0.0; bs * h],
            ws_snr_feat: vec![0.0; bs * n_freq],
            ws_snr_hidden: vec![0.0; bs * h],
            ws_snr_embed: vec![0.0; bs * h],
            ws_target_ctx: Vec::new(),  // 动态大小, 取决于 ctx_len
            cached_ctx_len: 0,          // fc 缓存: 初始无缓存
            cached_kv_len: 0,           // K/V cache: 初始无缓存
            ws_k_cache: vec![Vec::new(); block_count],
            ws_v_cache: vec![Vec::new(); block_count],
            ws_attn_in: Vec::new(),
            ws_q: vec![0.0; bs * n_qh * hd],  // 初始 bs, 实际用 (ctx+bs)
            ws_k: vec![0.0; bs * n_kvh * hd],
            ws_v: vec![0.0; bs * n_kvh * hd],
            ws_attn_out: vec![0.0; bs * h],
            ws_ffn_gate: vec![0.0; bs * ffn],
            ws_ffn_up: vec![0.0; bs * ffn],
            ws_hidden: vec![0.0; bs * h],
            ws_logits: vec![0.0; bs * vocab],
            ws_kq_scale: 1.0 / (hd as f32).sqrt(),
            // P2: 热路径复用 buffer (消除 forward_block 内 vec![] 分配)
            ws_attn_tmp: vec![0.0; bs * h],
            ws_ffn_normed: vec![0.0; bs * h],
            ws_ffn_out: vec![0.0; bs * h],
            rope_freqs,
            snr_computed: false,
            ws_confidence_hidden: vec![0.0; bs * h],
        }
    }

    /// Log-SNR conditioning: 计算 sinusoidal feat + FC1+SiLU + FC2 → snr_embed
    ///
    /// feat[pos][i] = sin(t * freq_i)  for i < 64
    /// feat[pos][i] = cos(t * freq_i)  for i >= 64
    /// 其中 t = (log_snr - min) / (max - min) * 1000
    /// log_snr: pos=0 (anchor) 用 max_log_snr, pos>0 (mask) 用 min_log_snr
    ///
    /// ★ 缓存优化: 输出 ws_snr_embed 只依赖 cfg (固定), 跨 draft_forward call 不变。
    ///   首次调用计算并缓存, 后续直接返回。FC2 是 BF16 标量 [5120,5120]×4 = 105M FMA,
    ///   缓存后节省 ~30-50ms/call (70 calls = 2-3.5s 总节省)。
    fn compute_log_snr(&mut self) {
        // ★ 缓存: ws_snr_embed 只依赖 cfg (max/min_log_snr, fc1/fc2 权重, bias),
        // 跨 draft_forward call 完全不变。首次计算后直接复用。
        // (已 A/B 验证: 缓存不影响接受率, 因为输出完全确定)
        if self.snr_computed { return; }
        let cfg = &self.weights.cfg;
        let bs = cfg.block_size;
        let n_freq = cfg.log_snr_n_freq;
        let half = n_freq / 2;

        // Step 1: sinusoidal feat [bs, n_freq]
        // t = (log_snr - min) / (max - min) * 1000
        //   pos=0 (anchor, 高SNR=9): t=1000
        //   pos>0 (draft, 低SNR=-9): t=0
        // (A/B 验证: 翻转 t 映射对 greedy 接受率无影响 — snr_embed 不足以翻转 argmax)
        for pos in 0..bs {
            let log_snr = if pos == 0 { cfg.max_log_snr } else { cfg.min_log_snr };
            let t = (log_snr - cfg.min_log_snr) / (cfg.max_log_snr - cfg.min_log_snr) * 1000.0;
            for i in 0..half {
                let freq = (-(10000.0f32.ln()) * i as f32 / half as f32).exp();
                let angle = t * freq;
                self.ws_snr_feat[pos * n_freq + i] = angle.sin();
                self.ws_snr_feat[pos * n_freq + half + i] = angle.cos();
            }
        }

        // Step 2: FC1 + SiLU → snr_hidden [bs, hidden]
        // log_snr_fc1_w: [hidden, n_freq] (rows=hidden, cols=n_freq)
        // 但 GGUF dims[0]=n_freq, dims[1]=hidden → 实际 rows=hidden, cols=n_freq
        // matvec: hidden[i] = sum_j fc1_w[i][j] * feat[j] + bias[i]
        for pos in 0..bs {
            let feat = &self.ws_snr_feat[pos * n_freq..(pos + 1) * n_freq];
            // fc1_w @ feat → [hidden]
            let w = &self.weights.log_snr_fc1_w;
            debug_assert_eq!(w.rows, cfg.embedding_length);
            debug_assert_eq!(w.cols, n_freq);
            w.matvec_into_slice(feat, &mut self.ws_snr_hidden[pos * cfg.embedding_length..(pos + 1) * cfg.embedding_length]);
            // + bias
            for i in 0..cfg.embedding_length {
                self.ws_snr_hidden[pos * cfg.embedding_length + i] += self.weights.log_snr_fc1_b.data[i];
            }
            // SiLU: x * sigmoid(x) — AVX2 向量化
            let silu_slice = &mut self.ws_snr_hidden[pos * cfg.embedding_length..(pos + 1) * cfg.embedding_length];
            crate::math::simd_exp::silu_inplace_simd(silu_slice);
        }

        // Step 3: FC2 → snr_embed [bs, hidden]
        for pos in 0..bs {
            let hidden_in = &self.ws_snr_hidden[pos * cfg.embedding_length..(pos + 1) * cfg.embedding_length];
            let w = &self.weights.log_snr_fc2_w;
            debug_assert_eq!(w.rows, cfg.embedding_length);
            debug_assert_eq!(w.cols, cfg.embedding_length);
            w.matvec_into_slice(hidden_in, &mut self.ws_snr_embed[pos * cfg.embedding_length..(pos + 1) * cfg.embedding_length]);
            // + bias
            for i in 0..cfg.embedding_length {
                self.ws_snr_embed[pos * cfg.embedding_length + i] += self.weights.log_snr_fc2_b.data[i];
            }
        }

        // ★ 缓存标志: 后续 draft_forward call 直接复用 ws_snr_embed
        self.snr_computed = true;
    }

    /// 一次 drafter forward, 生成 block_size 个候选 token 的 base logits
    ///
    /// 输入:
    /// - anchor_token: 上一个被接受的 token id
    /// - target_ctx_feat: target tap 特征 [ctx_len, n_embd_cap=25600] (flat)
    /// - ctx_len: context 行数 (已 commit 的 target tap 行)
    /// - start_pos: anchor 的绝对位置
    ///
    /// 输出: base_logits [block_size, vocab] (写入 self.ws_logits)
    /// 同时返回 base_logits 的引用 (供 markov resample 用)
    pub fn draft_forward(
        &mut self,
        anchor_token: u32,
        target_ctx_feat: &[f32],
        ctx_len: usize,
        start_pos: usize,
    ) {
        // ★ P2: 不 clone cfg, 直接读出需要的字段 (避免 Vec<usize> 堆分配)
        let (bs, h, block_count, log_snr_conditioning, mask_token_id, n_embd_cap, rms_eps, vocab_size) = {
            let c = &self.weights.cfg;
            (c.block_size, c.embedding_length, c.block_count,
             c.log_snr_conditioning, c.mask_token_id, c.n_embd_cap(), c.rms_eps, c.vocab_size)
        };
        let n_total = ctx_len + bs;
        let profile = crate::model::forward::profile_enabled();
        let t0 = if profile { Some(std::time::Instant::now()) } else { None };

        // ★ K/V cache 回退保护: 若 ctx_len < cached_kv_len (异常, 理论不发生),
        // 重置缓存标志, 全量重算 (与 fc cache 的回退保护一致)。
        // ★ DAIZA_DSPARK_NO_KV_CACHE=1: 强制禁用 K/V cache 复用 (A/B 测试用)
        if ctx_len < self.cached_kv_len || no_kv_cache() {
            self.cached_kv_len = 0;
        }

        // 1. Embedding: pos 0 = anchor, pos 1..bs-1 = mask_token
        // ws_embd: [bs, hidden]
        for pos in 0..bs {
            let tok = if pos == 0 { anchor_token } else { mask_token_id };
            // token_embd: [vocab, hidden], row = tok
            dequantize_q1_0_row_into(
                &self.weights.token_embd.bytes,
                tok as usize,
                h,
                &mut self.ws_embd[pos * h..(pos + 1) * h],
            );
        }
        let t_emb = t0.map(|t| t.elapsed().as_millis());

        // 2. Log-SNR conditioning: snr_embed 加到 embedding (AVX2 saxpy)
        //    DAIZA_DSPARK_NO_SNR=1 禁用 (调试用, 隔离 snr 是否有 bug)
        let t_snr = if log_snr_conditioning && !no_snr() {
            let t = if profile { Some(std::time::Instant::now()) } else { None };
            self.compute_log_snr();
            crate::math::simd_exp::saxpy_avx2(1.0, &self.ws_snr_embed[..bs * h], &mut self.ws_embd[..bs * h], bs * h);
            t.map(|t| t.elapsed().as_millis())
        } else {
            None
        };

        // 3. 投影 target context: fc @ ctx_feat → target_ctx [ctx_len, hidden]
        //    fc: [hidden, n_embd_cap=25600], ctx_feat[ctx_len, 25600]
        //    target_ctx[row] = fc @ ctx_feat[row]
        //
        //    ★ fc 缓存: 跨 draft_forward call 复用已投影行, 只投影新增行。
        //    ctx_len 单调递增 (每 cycle 增 n_acc+1 行), 缓存命中率接近 100%。
        //    若 ctx_len 回退 (理论上不会发生), 重置缓存全量重投影。
        let t_fc = if ctx_len > 0 {
            let t = if profile { Some(std::time::Instant::now()) } else { None };
            let fc = &self.weights.fc;
            debug_assert_eq!(fc.cols, n_embd_cap);
            debug_assert_eq!(fc.rows, h);
            let hidden_norm = &self.weights.hidden_norm.data;

            if ctx_len < self.cached_ctx_len || no_fc_cache() {
                // ctx_len 回退 (异常): 重置缓存, 全量重投影
                self.cached_ctx_len = 0;
            }
            let need = ctx_len * h;
            if self.ws_target_ctx.len() < need {
                self.ws_target_ctx.resize(need, 0.0);
            }
            // 增量投影: 只处理 [cached_ctx_len..ctx_len]
            let new_rows = ctx_len - self.cached_ctx_len;
            if new_rows > 0 {
                let start_row = self.cached_ctx_len;
                fc.matvec_batch_into_slice(
                    &target_ctx_feat[start_row * n_embd_cap..ctx_len * n_embd_cap],
                    new_rows,
                    &mut self.ws_target_ctx[start_row * h..ctx_len * h],
                );
                // hidden_norm (RMSNorm) 只对新增行
                for row in start_row..ctx_len {
                    math::rmsnorm_inplace(
                        &mut self.ws_target_ctx[row * h..(row + 1) * h],
                        hidden_norm,
                        rms_eps,
                    );
                }
                self.cached_ctx_len = ctx_len;
            }
            t.map(|t| t.elapsed().as_millis())
        } else {
            None
        };

        // 4. 构造 attn_in = concat(target_ctx, draft_embd) [(ctx+bs), hidden]
        //    context 行只贡献 K/V, draft 行进 query/FFN/残差
        //
        //    ★ 增量复制: K/V cache 有效时 (cached_kv_len > 0), 前 cached_kv_len 行的
        //    attn_in 不被读取 (K/V 从 cache 复制), 只需复制新增 context 行
        //    [cached_kv_len..ctx_len]。节省 ~98% context 复制 (3MB → 60KB)。
        //    ★ draft 行 (后 bs 行) 不复制: forward_block Step 1 会用 RMSNorm(ws_hidden)
        //    覆盖 ws_attn_in[ctx_len..n_total], 故此处复制会被立即覆盖, 是非必要计算。
        self.ws_attn_in.resize(n_total * h, 0.0);
        let cached_kv_len = self.cached_kv_len;
        let kv_cache_valid = cached_kv_len > 0 && cached_kv_len <= ctx_len;
        if ctx_len > 0 {
            let start_row = if kv_cache_valid { cached_kv_len } else { 0 };
            if start_row < ctx_len {
                let start = start_row * h;
                let end = ctx_len * h;
                self.ws_attn_in[start..end].copy_from_slice(&self.ws_target_ctx[start..end]);
            }
        }

        // ws_hidden 初始化为 draft embedding (残差流, forward_block Step 1 读取)
        self.ws_hidden[..bs * h].copy_from_slice(&self.ws_embd[..bs * h]);

        // 5. 6 层 transformer (非因果 attention)
        let t_blk = if profile { Some(std::time::Instant::now()) } else { None };
        for il in 0..block_count {
            self.forward_block(il, ctx_len, start_pos);
        }
        let t_blk = t_blk.map(|t| t.elapsed().as_millis());

        // ★ K/V cache: 更新 cached_kv_len 为本 cycle 的 ctx_len, 供下一 cycle 复用
        // (forward_block 内部已将 [0..ctx_len] 行的 K/V 写入 ws_k_cache/ws_v_cache)
        self.cached_kv_len = ctx_len;

        // 6. Output norm + LM head → base logits [bs, vocab]
        let t_lm = if profile { Some(std::time::Instant::now()) } else { None };
        let output_norm = &self.weights.output_norm.data;
        for pos in 0..bs {
            math::rmsnorm_inplace(
                &mut self.ws_hidden[pos * h..(pos + 1) * h],
                output_norm,
                rms_eps,
            );
        }
        // ★ Confidence head 输入: RMSNorm 之后的 hidden state
        // 实测 RMSNorm 之前 logit 值 100-330 (sigmoid 全 1.0, 无区分能力);
        // RMSNorm 之后值范围归一化, logit 落在合理区间。
        // 仅当 confidence_head 启用时才 copy (避免无 confidence head 时的 80KB 复制开销)。
        if self.weights.confidence_head_w.is_some() {
            self.ws_confidence_hidden[..bs * h]
                .copy_from_slice(&self.ws_hidden[..bs * h]);
        }
        // output: [vocab, hidden], batched 计算 logits (W 只读一次)
        let h_in = &self.ws_hidden[..bs * h];
        let logits = &mut self.ws_logits[..bs * vocab_size];
        self.weights.output.matvec_batch_into_slice(h_in, bs, logits);
        let t_lm = t_lm.map(|t| t.elapsed().as_millis());

        if profile {
            eprintln!("[drafter-profile] emb={:?}ms snr={:?}ms fc={:?}ms blocks={:?}ms lm_head={:?}ms (ctx_len={ctx_len})",
                t_emb, t_snr, t_fc, t_blk, t_lm);
        }
    }

    /// 单层 transformer forward (非因果 attention)
    fn forward_block(&mut self, il: usize, ctx_len: usize, start_pos: usize) {
        // ★ P2: 不 clone cfg, 直接读出需要的字段
        let (bs, h, n_qh, n_kvh, hd, rms_eps, group_size, ffn_len) = {
            let c = &self.weights.cfg;
            (c.block_size, c.embedding_length, c.head_count, c.head_count_kv,
             c.head_dim, c.rms_eps, c.group_size(), c.feed_forward_length)
        };
        let n_total = ctx_len + bs;
        let blk = &self.weights.blocks[il];

        // attn_in = [ctx_len × target_ctx; bs × hidden]
        // Step 1: RMSNorm on draft hidden → ws_attn_in[ctx_len..]
        for pos in 0..bs {
            let src = &self.ws_hidden[pos * h..(pos + 1) * h];
            let dst = &mut self.ws_attn_in[(ctx_len + pos) * h..(ctx_len + pos + 1) * h];
            math::rmsnorm_into(src, dst, &blk.attn_norm.data, rms_eps);
        }
        // 前 ctx_len 行保持 target_ctx (不变)

        // Step 2: Q/K/V proj
        // ★ 优化 1: Q proj 只对 draft 行 (后 bs 行) — context 行的 Q 不被 attention 使用
        //   (attention 只读 q_row = ctx_len + pos 的 Q)。K/V proj 仍对所有行 (context
        //   行的 K/V 是 attention 的 key/value 来源)。
        //   ctx_len=60, bs=4 → 节省 60/64 = 94% Q proj + Q norm + Q RoPE 计算。
        //   Q proj 占 block 时间 ~70% (5120×5120 矩阵), 这是 drafter 主要瓶颈。
        // ★ 优化 2: K/V cache 复用 — context 行 (前 ctx_len 行) 的 attn_in 来自
        //   ws_target_ctx (跨 cycle 稳定), 故 K/V proj+norm+RoPE 结果可跨 cycle 缓存。
        //   每 cycle 平均新增 ctx_len ≈ 2.23 行, 缓存命中率接近 100%, 节省 ~90% K/V proj。
        self.ws_q.resize(n_total * n_qh * hd, 0.0);
        self.ws_k.resize(n_total * n_kvh * hd, 0.0);
        self.ws_v.resize(n_total * n_kvh * hd, 0.0);
        let attn_in = &self.ws_attn_in[..n_total * h];
        // Q: 只投影 draft 行 (attn_in[ctx_len..n_total] → ws_q[ctx_len..n_total])
        let draft_attn_in = &self.ws_attn_in[ctx_len * h..n_total * h];
        let draft_q_out = &mut self.ws_q[ctx_len * n_qh * hd..n_total * n_qh * hd];
        blk.attn_q.matvec_batch_into_slice(draft_attn_in, bs, draft_q_out);

        // K/V: 检查 cache 是否可用 (cached_kv_len <= ctx_len, 即上一 cycle 的 context 行)
        // 若 cached_kv_len > ctx_len (异常, 理论不发生), 重置 cache 全量重算。
        let cached_kv_len = self.cached_kv_len;
        let kv_cache_valid = cached_kv_len > 0 && cached_kv_len <= ctx_len;
        if kv_cache_valid {
            // ★ P1 零拷贝: 不再把 cache 复制到 ws_k/ws_v, 直接在 attention 循环里
            //   分段读 (k < cached_kv_len 读 ws_k_cache[il], k >= cached_kv_len 读 ws_k)。
            //   节省 ctx_len × n_kvh × hd × 4B × 2(K+V) ≈ 60×512×8B = 245KB/层 内存拷贝。
            //   只投影 [cached_kv_len..n_total] 行 (新增 context 行 + draft 行)
            let start_row = cached_kv_len;
            let new_rows = n_total - start_row;
            let new_attn_in = &self.ws_attn_in[start_row * h..n_total * h];
            let new_k_out = &mut self.ws_k[start_row * n_kvh * hd..n_total * n_kvh * hd];
            let new_v_out = &mut self.ws_v[start_row * n_kvh * hd..n_total * n_kvh * hd];
            blk.attn_k.matvec_batch_into_slice(new_attn_in, new_rows, new_k_out);
            blk.attn_v.matvec_batch_into_slice(new_attn_in, new_rows, new_v_out);
        } else {
            // 首次或异常: 全量 K/V proj
            let k_out = &mut self.ws_k[..n_total * n_kvh * hd];
            let v_out = &mut self.ws_v[..n_total * n_kvh * hd];
            blk.attn_k.matvec_batch_into_slice(attn_in, n_total, k_out);
            blk.attn_v.matvec_batch_into_slice(attn_in, n_total, v_out);
        }

        // Step 3: Q/K per-head RMSNorm + RoPE
        // 位置语义 (对齐 llama.cpp speculative.cpp L1183-1191):
        //   context 行 i 位置 = (start_pos - ctx_len) + i  (= L + i, 绝对位置)
        //   draft[k] 位置 = start_pos + k
        // ★ BUG 修复 (2026-07-19): 之前 context 行用相对位置 i, 与 llama.cpp 不一致。
        //   llama.cpp 用 L+i (L = 上一 cycle 的 start = start_pos - ctx_len), 绝对位置。
        //   第一个 cycle L=0 两者一致, 后续 cycle L>0 导致 RoPE 旋转错误, 接受率 75% vs 95%。
        // ★ drafter 使用全维度 RoPE (head_dim=128, GPT-NeoX style), 非 target 的 M-RoPE partial。
        //   实测: 用 M-RoPE partial 接受率 36%→3%, 说明 drafter 训练时就是全维度标准 RoPE。
        // ★ 优化: Q norm + RoPE 只对 draft 行 (前 ctx_len 行的 Q 未计算, 是 garbage)
        // ★ 优化: K norm + RoPE 只对 [cached_kv_len..n_total] (前 cached_kv_len 行已缓存)
        // ★ P3: Q norm+RoPE 合并到 K 循环的 draft 行分支, 共享 cos/sin
        let q_norm = &blk.attn_q_norm.data;
        let k_norm = &blk.attn_k_norm.data;
        let rope_freqs = &self.rope_freqs;
        let rope_cos = &mut self.rope_cos;
        let rope_sin = &mut self.rope_sin;
        let k_start = if kv_cache_valid { cached_kv_len } else { 0 };
        // ★ context 行 position = L + row = (start_pos - ctx_len) + row (绝对位置, 对齐 llama.cpp)
        //   L = start_pos - ctx_len = 上一 cycle 的 start (已 commit 的 KV cache 长度)
        let ctx_pos_base = start_pos.wrapping_sub(ctx_len);
        // K norm + RoPE: 只对 [k_start..n_total] 行; draft 行同时算 Q norm+RoPE
        for row in k_start..n_total {
            let pos = if row < ctx_len { ctx_pos_base + row } else { start_pos + (row - ctx_len) };
            compute_rope_into(rope_cos, rope_sin, rope_freqs, pos);

            for kvh in 0..n_kvh {
                let k_off = row * n_kvh * hd + kvh * hd;
                math::rmsnorm_inplace(&mut self.ws_k[k_off..k_off + hd], k_norm, rms_eps);
                apply_rope_full(&mut self.ws_k[k_off..k_off + hd], rope_cos, rope_sin);
            }

            // Q norm + RoPE: 仅 draft 行 (row >= ctx_len), 复用本行 cos/sin
            if row >= ctx_len {
                for qh in 0..n_qh {
                    let q_off = row * n_qh * hd + qh * hd;
                    math::rmsnorm_inplace(&mut self.ws_q[q_off..q_off + hd], q_norm, rms_eps);
                    apply_rope_full(&mut self.ws_q[q_off..q_off + hd], rope_cos, rope_sin);
                }
            }
        }

        // Step 3.5: 保存新增 context 行的 K/V 到 cache (增量, 跨 cycle 复用)
        // ★ 增量保存: 前 cached_kv_len 行已在上 cycle 保存 (且本 cycle 未修改,
        //   K norm+RoPE 只处理 [k_start..n_total]), 只需保存新增行 [cached_kv_len..ctx_len]。
        //   节省 ~98% cache 写入 (303KB → 6KB per layer)。
        if ctx_len > cached_kv_len {
            let start = cached_kv_len * n_kvh * hd;
            let end = ctx_len * n_kvh * hd;
            if self.ws_k_cache[il].len() < end {
                self.ws_k_cache[il].resize(end, 0.0);
                self.ws_v_cache[il].resize(end, 0.0);
            }
            self.ws_k_cache[il][start..end].copy_from_slice(&self.ws_k[start..end]);
            self.ws_v_cache[il][start..end].copy_from_slice(&self.ws_v[start..end]);
        }

        // Step 4: 非因果 attention (全开 mask, GQA) — online softmax + V-update 融合
        // ★ 优化: 改用 online softmax (与 target decode path attention.rs 算法一致),
        //   完全消除 ws_scores buffer (原 group_size × n_total × 4B × 6 pass 流量)
        //   每个 kvh 内 group_size 个 qh 各维护 running max m / running sum s / running out
        //   逐 k 读取一份 K[k][kvh] + V[k][kvh], 服务 group_size 个 qh
        self.ws_attn_out.resize(bs * h, 0.0);
        let kq_scale = self.ws_kq_scale;
        // 提前借用 K/V 数据源 slice (避免循环内重复借用 self)
        let (k_src, v_src, cache_split): (&[f32], &[f32], usize) = if kv_cache_valid {
            (&self.ws_k_cache[il], &self.ws_v_cache[il], cached_kv_len)
        } else {
            (&self.ws_k, &self.ws_v, 0)
        };
        let ws_k = &self.ws_k;
        let ws_v = &self.ws_v;
        // ★ GQA K/V 复用: 外层 kvh, 内层 group_size 个 qh 共享 K/V 读取
        //   K/V 每个 k 只读一次, 服务 group_size 个 qh (与 target decode path 一致)
        // ★ drafter group_size=10, hd=128; stack 数组容量 16 (对齐 2^N)
        debug_assert!(hd <= 128, "drafter online softmax stack buffer requires hd<=128");
        debug_assert!(group_size <= 16, "drafter online softmax stack buffer requires group_size<=16");
        for pos in 0..bs {
            let q_row = ctx_len + pos;
            for kvh in 0..n_kvh {
                let q_base = q_row * n_qh * hd + kvh * group_size * hd;

                // group_size 个 qh 的 running state (stack 数组, group_size=10, 容量 16)
                let mut m = [f32::NEG_INFINITY; 16];      // running max
                let mut s = [0.0f32; 16];                  // running sum
                let mut out = [[0.0f32; 128]; 16];         // running output (hd=128)

                // K/V 数据源分两段: [0..cache_split) 读 k_src/v_src (cache),
                //                  [cache_split..n_total) 读 ws_k/ws_v (新算)
                for k in 0..cache_split {
                    let kv_ptr = k * n_kvh * hd + kvh * hd;
                    let k_head = &k_src[kv_ptr..kv_ptr + hd];
                    let v_head = &v_src[kv_ptr..kv_ptr + hd];
                    for qh_in_group in 0..group_size {
                        let q_ptr = q_base + qh_in_group * hd;
                        let q_head = &self.ws_q[q_ptr..q_ptr + hd];
                        let score = crate::math::simd_exp::dot_product_avx2(q_head, k_head, hd) * kq_scale;

                        let m_old = m[qh_in_group];
                        let m_new = m_old.max(score);
                        let alpha = crate::math::simd_exp::exp_fast(m_old - m_new);
                        let beta = crate::math::simd_exp::exp_fast(score - m_new);

                        s[qh_in_group] = s[qh_in_group] * alpha + beta;
                        crate::math::simd_exp::online_softmax_v_update_avx2(
                            &mut out[qh_in_group], alpha, beta, v_head, hd,
                        );
                        m[qh_in_group] = m_new;
                    }
                }
                for k in cache_split..n_total {
                    let kv_ptr = k * n_kvh * hd + kvh * hd;
                    let k_head = &ws_k[kv_ptr..kv_ptr + hd];
                    let v_head = &ws_v[kv_ptr..kv_ptr + hd];
                    for qh_in_group in 0..group_size {
                        let q_ptr = q_base + qh_in_group * hd;
                        let q_head = &self.ws_q[q_ptr..q_ptr + hd];
                        let score = crate::math::simd_exp::dot_product_avx2(q_head, k_head, hd) * kq_scale;

                        let m_old = m[qh_in_group];
                        let m_new = m_old.max(score);
                        let alpha = crate::math::simd_exp::exp_fast(m_old - m_new);
                        let beta = crate::math::simd_exp::exp_fast(score - m_new);

                        s[qh_in_group] = s[qh_in_group] * alpha + beta;
                        crate::math::simd_exp::online_softmax_v_update_avx2(
                            &mut out[qh_in_group], alpha, beta, v_head, hd,
                        );
                        m[qh_in_group] = m_new;
                    }
                }

                // 归一化并写入 ws_attn_out
                let out_base = pos * h + kvh * group_size * hd;
                for qh_in_group in 0..group_size {
                    let out_off = out_base + qh_in_group * hd;
                    let inv_s = 1.0 / s[qh_in_group];
                    crate::math::simd_exp::scale_avx2(
                        &out[qh_in_group], inv_s, &mut self.ws_attn_out[out_off..out_off + hd], hd,
                    );
                }
            }
        }

        // Step 5: output proj + 残差 (batched, 复用 ws_attn_tmp)
        // ★ AVX2 saxpy: ws_hidden += tmp (bs*h = 4*5120 = 20480 元素, 6 layers = 122880 次/call)
        let tmp = &mut self.ws_attn_tmp[..bs * h];
        let attn_out = &self.ws_attn_out[..bs * h];
        blk.attn_output.matvec_batch_into_slice(attn_out, bs, tmp);
        crate::math::simd_exp::saxpy_avx2(1.0, tmp, &mut self.ws_hidden[..bs * h], bs * h);

        // Step 6: FFN (SwiGLU)
        // gate = W_gate @ norm(hidden), up = W_up @ norm(hidden)
        // ffn_out = W_down @ (silu(gate) * up)
        let ffn_normed = &mut self.ws_ffn_normed[..bs * h];
        for pos in 0..bs {
            math::rmsnorm_into(
                &self.ws_hidden[pos * h..(pos + 1) * h],
                &mut ffn_normed[pos * h..(pos + 1) * h],
                &blk.ffn_norm.data,
                rms_eps,
            );
        }
        self.ws_ffn_gate.resize(bs * ffn_len, 0.0);
        self.ws_ffn_up.resize(bs * ffn_len, 0.0);
        let gate_out = &mut self.ws_ffn_gate[..bs * ffn_len];
        let up_out = &mut self.ws_ffn_up[..bs * ffn_len];
        blk.ffn_gate.matvec_batch_into_slice(ffn_normed, bs, gate_out);
        blk.ffn_up.matvec_batch_into_slice(ffn_normed, bs, up_out);
        // SwiGLU: silu(gate) * up (原地写入 ws_ffn_gate)
        // ★ AVX2 优化: swiglu_inplace_simd (融合 silu+mul, 8-wide), 替代标量循环
        //   bs*ffn_len = 4*5120 = 20480 次/call, 6 layers = 122880 次/call
        math::swiglu_inplace_simd(&mut self.ws_ffn_gate[..bs * ffn_len], &self.ws_ffn_up[..bs * ffn_len]);
        // down proj + 残差 (batched, 复用 ws_ffn_out, AVX2 saxpy)
        let ffn_out = &mut self.ws_ffn_out[..bs * h];
        let gate = &self.ws_ffn_gate[..bs * ffn_len];
        blk.ffn_down.matvec_batch_into_slice(gate, bs, ffn_out);
        crate::math::simd_exp::saxpy_avx2(1.0, ffn_out, &mut self.ws_hidden[..bs * h], bs * h);
    }
}

/// 全维度 RoPE (drafter 用, 非 partial)
/// x[i] = x[i] * cos[i] - x[i+hd/2] * sin[i]
/// x[i+hd/2] = x[i+hd/2] * cos[i] + x[i] * sin[i]
fn apply_rope_full(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let hd = x.len();
    let half = hd / 2;
    // ★ P4: AVX2 8-wide, hd=128 → half=64 = 8 iter, 无尾处理
    #[cfg(target_arch = "x86_64")]
    if crate::math::simd_exp::simd_available() && half >= 8 {
        #[allow(unsafe_code)]
        unsafe {
            use std::arch::x86_64::*;
            let mut i = 0;
            let n8 = (half / 8) * 8;
            while i < n8 {
                let x1 = _mm256_loadu_ps(x.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(x.as_ptr().add(i + half));
                let c = _mm256_loadu_ps(cos.as_ptr().add(i));
                let s = _mm256_loadu_ps(sin.as_ptr().add(i));
                // out1 = x1*c - x2*s
                let out1 = _mm256_fmsub_ps(x1, c, _mm256_mul_ps(x2, s));
                // out2 = x2*c + x1*s
                let out2 = _mm256_fmadd_ps(x2, c, _mm256_mul_ps(x1, s));
                _mm256_storeu_ps(x.as_mut_ptr().add(i), out1);
                _mm256_storeu_ps(x.as_mut_ptr().add(i + half), out2);
                i += 8;
            }
            for j in i..half {
                let x1 = x[j];
                let x2 = x[j + half];
                x[j] = x1 * cos[j] - x2 * sin[j];
                x[j + half] = x2 * cos[j] + x1 * sin[j];
            }
            return;
        }
    }
    for i in 0..half {
        let x1 = x[i];
        let x2 = x[i + half];
        x[i] = x1 * cos[i] - x2 * sin[i];
        x[i + half] = x2 * cos[i] + x1 * sin[i];
    }
}

/// 计算 RoPE cos/sin 写入提供的缓冲 (drafter 全维度 RoPE)
/// 使用预计算的 freqs (避免重复 powf)
/// cos[i+hd/2] = cos[i], sin[i+hd/2] = sin[i] (对称)
fn compute_rope_into(
    cos: &mut [f32],
    sin: &mut [f32],
    freqs: &[f32],
    pos: usize,
) {
    let half = freqs.len();
    for i in 0..half {
        let angle = pos as f32 * freqs[i];
        let c = angle.cos();
        let s = angle.sin();
        cos[i] = c;
        sin[i] = s;
        cos[i + half] = c;
        sin[i + half] = s;
    }
}

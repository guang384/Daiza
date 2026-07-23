//! DSpark 推测解码: draft 调度 + target tap 缓冲
//!
//! ## 实际流程 (engine.rs 内联实现)
//!
//! 1. **Draft**: `drafter.draft_forward(anchor, target_tap, ctx_len, pos)` → base_logits
//!    + `markov.resample_chain(base_logits, anchor)` → block_size 个候选 token
//! 2. **Verify**: engine 顺序 forward 这 block_size 个 token, 收集每位置的 logits
//! 3. **Reject**: engine 内联 Leviathan rejection sampling, 决定接受前 n 个 token + 1 个 bonus
//! 4. **Capture**: target forward 过程中, 在 layers [1,16,31,46,61] tap hidden states
//!    作为下一轮 draft 的 context 特征
//!
//! ## Cache 管理
//!
//! - **Target KV/SSM cache**: 无需 rollback — verify 期间 forward 的 accepted tokens 已
//!   正确更新状态 (delta rule 下 SSM state 是增量的); reject 时 accepted prefix 仍
//!   是 target 真正生成的 token, 状态保持正确。
//! - **Drafter state**: block-parallel forward 无状态副作用, 无需回滚。

use super::config::DrafterConfig;
use super::drafter::DrafterContext;
use super::markov::MarkovContext;
use super::weights::DrafterWeights;

/// DSpark 推测解码运行时上下文
///
/// 持有 drafter + markov 工作区, 以及 target hidden state tap 缓冲。
/// 由 engine.rs 持有, 每次 draft-verify 循环调用 `draft` (verify 在 engine 内联)。
pub struct SpeculativeContext {
    pub drafter: DrafterContext,
    pub markov: MarkovContext,
    /// target tap 特征 [ctx_len, n_embd_cap=25600] (flat)
    /// 由 engine 在每次 verify 后从 target layers [1,16,31,46,61] 抽取并拼接
    pub target_tap_feat: Vec<f32>,
    /// 当前 target tap 行数 (通常 = 1, 即上一个被接受 token 的 hidden state)
    pub target_tap_len: usize,
    /// block_size 个 draft token IDs (markov resample 输出)
    pub draft_tokens: Vec<u32>,
    /// block_size × vocab 的 step_logits (markov resample 输出, 作为 draft q)
    pub draft_logits: Vec<f32>,
    /// ★ block_size × rank 的 prev_embd (markov resample 输出, 供 confidence head 用)
    /// 每位置 k 的 prev_embd = W1[prev_token_k]
    pub draft_prev_embds: Vec<f32>,
    /// ★ confidence head 每位置 logit (drafter forward + markov 后计算)
    /// 长度 = block_size, 调用方据此决定早停位置
    pub confidence_logits: Vec<f32>,
    /// confidence head 输入缓冲 (跨 call 复用, 避免 5376*4=21KB 堆分配)
    pub ws_conf_input: Vec<f32>,
}

impl SpeculativeContext {
    /// 加载 drafter 权重并初始化所有 workspace
    pub fn new(weights: DrafterWeights) -> Self {
        let cfg = weights.cfg.clone();
        let conf_input_dim = cfg.confidence_input_dim();
        Self {
            drafter: DrafterContext::new(weights),
            markov: MarkovContext::new(&cfg),
            // ★ 初始 target_tap_len = 0: 第一次 set_target_tap 会全量复制 [0..n_rows]
            //   (增量复制方案下, old_len=0 → 复制 [0..n_rows] 全部)
            target_tap_feat: Vec::new(),
            target_tap_len: 0,
            draft_tokens: Vec::with_capacity(cfg.block_size),
            draft_logits: Vec::with_capacity(cfg.block_size * cfg.vocab_size),
            draft_prev_embds: Vec::with_capacity(cfg.block_size * cfg.markov_rank),
            confidence_logits: Vec::with_capacity(cfg.block_size),
            ws_conf_input: vec![0.0; conf_input_dim],
        }
    }

    /// 返回 drafter config 引用
    pub fn cfg(&self) -> &DrafterConfig {
        &self.drafter.weights.cfg
    }

    /// 设置 target tap 特征 (engine 在每次 verify 后调用)
    ///
    /// `feat`: [n_rows, n_embd_cap] flat 切片, 行优先
    ///
    /// ★ 增量复制优化: target_tap_history 是累积的 (旧行不变), 且 drafter fc cache
    ///   (cached_ctx_len) 只读新增行 [cached..n_rows]。所以只需复制新增行, 避免
    ///   O(N²) 全量复制 (200 tokens 时 ~1.7GB 复制 → ~170ms 总节省)。
    pub fn set_target_tap(&mut self, feat: &[f32], n_rows: usize) {
        let cap = self.cfg().n_embd_cap();
        debug_assert_eq!(feat.len(), n_rows * cap);

        // 回退保护: n_rows < target_tap_len (异常, 理论不发生) → 重置全量
        if n_rows < self.target_tap_len {
            self.target_tap_feat.clear();
            self.target_tap_len = 0;
        }

        // 确保 buffer 容量足够
        let need = n_rows * cap;
        if self.target_tap_feat.len() < need {
            self.target_tap_feat.resize(need, 0.0);
        }

        // 只复制新增行 [target_tap_len..n_rows]
        let old_len = self.target_tap_len;
        if n_rows > old_len {
            let src_start = old_len * cap;
            let src_end = n_rows * cap;
            self.target_tap_feat[old_len * cap..n_rows * cap]
                .copy_from_slice(&feat[src_start..src_end]);
        }
        self.target_tap_len = n_rows;
    }

    /// Phase 1: 生成 block_size 个 draft tokens
    ///
    /// 输入:
    /// - `anchor_token`: 上一个被 target 接受的 token
    /// - `start_pos`: anchor 的绝对位置 (drafter RoPE 用)
    /// - `need_logits`: 是否填充 self.draft_logits (greedy 模式下采样不需要 q, 省大块 copy)
    ///
    /// 输出: `&[u32]` 长度 = block_size, 当 need_logits=true 时 self.draft_logits 被填充。
    /// 若 confidence_head 启用, 同时填充 self.confidence_logits (调用方据此早停)。
    pub fn draft(&mut self, anchor_token: u32, start_pos: usize, need_logits: bool) -> &[u32] {
        let bs = self.drafter.weights.cfg.block_size;
        let vocab = self.drafter.weights.cfg.vocab_size;
        let has_conf = self.drafter.weights.confidence_head_w.is_some();
        let profile = crate::model::forward::profile_enabled();

        // 1. drafter forward → base_logits [block_size, vocab] (写入 self.drafter.ws_logits)
        //    若 confidence_head 启用, 同时填充 ws_confidence_hidden [bs, hidden] (norm 前)
        let t0 = if profile { Some(std::time::Instant::now()) } else { None };
        self.drafter.draft_forward(
            anchor_token,
            &self.target_tap_feat,
            self.target_tap_len,
            start_pos,
        );
        let t_drafter = t0.map(|t| t.elapsed().as_millis());

        // 2. markov resample → draft_tokens + step_logits + prev_embds
        //    base_logits 借用 self.drafter.ws_logits, weights 借用 self.drafter.weights,
        //    二者都是 self.drafter 的不可变借用, 与 &mut self.markov 不冲突
        let t1 = if profile { Some(std::time::Instant::now()) } else { None };
        let base_logits = &self.drafter.ws_logits[..bs * vocab];
        let weights = &self.drafter.weights;
        let prev_embds_buf: Option<&mut Vec<f32>> = if has_conf {
            Some(&mut self.draft_prev_embds)
        } else {
            None
        };
        let step_logits_buf: Option<&mut Vec<f32>> = if need_logits {
            Some(&mut self.draft_logits)
        } else {
            self.draft_logits.clear();
            None
        };
        self.markov.resample_chain(
            weights,
            base_logits,
            anchor_token,
            &mut self.draft_tokens,
            step_logits_buf,
            prev_embds_buf,
        );
        let t_markov = t1.map(|t| t.elapsed().as_millis());

        // 3. confidence head (若启用): 每位置 logit = W @ [hidden, prev_embd] + bias
        let t_conf = if has_conf {
            let t = if profile { Some(std::time::Instant::now()) } else { None };
            self.predict_confidence();
            t.map(|t| t.elapsed().as_millis())
        } else {
            None
        };

        if let (Some(td), Some(tm)) = (t_drafter, t_markov) {
            if let Some(tc) = t_conf {
                eprintln!("[draft-profile] drafter={td}ms markov={tm}ms conf={tc}ms (ctx_len={}, bs={bs})",
                    self.target_tap_len);
            } else {
                eprintln!("[draft-profile] drafter={td}ms markov={tm}ms (ctx_len={}, bs={bs})",
                    self.target_tap_len);
            }
        }

        &self.draft_tokens
    }

    /// ★ Confidence head: 对每位置 k 计算 logit = W @ [hidden_k, prev_embd_k] + bias
    ///
    /// 输入 (每位置 k):
    /// - hidden_k = ws_confidence_hidden[k*hidden..(k+1)*hidden] (drafter 最后一层输出, norm 前)
    /// - prev_embd_k = draft_prev_embds[k*rank..(k+1)*rank] (W1[prev_token_k])
    ///
    /// 输出: self.confidence_logits[k] = logit (调用方对 sigmoid(logit) 与 threshold 比较)
    ///
    /// 实现: W 是 [1, 5376] Q4_1, bias 是 [1] F32。
    ///   单次 matvec (rows=1, cols=5376) 走 Q4_1 单行 AVX2 kernel (~5K FMA, <0.1ms)。
    pub fn predict_confidence(&mut self) {
        let cfg = &self.drafter.weights.cfg;
        let bs = cfg.block_size;
        let h = cfg.embedding_length;
        let rank = cfg.markov_rank;
        let with_markov = cfg.confidence_head_with_markov;
        let w = match self.drafter.weights.confidence_head_w.as_ref() {
            Some(w) => w,
            None => return,
        };
        let b = match self.drafter.weights.confidence_head_b.as_ref() {
            Some(b) => b,
            None => return,
        };
        debug_assert_eq!(w.rows, 1);
        debug_assert_eq!(w.cols, cfg.confidence_input_dim());
        debug_assert_eq!(b.data.len(), 1);

        self.confidence_logits.clear();
        self.confidence_logits.reserve(bs);

        let input_dim = w.cols;
        for k in 0..bs {
            // 拼接 [hidden_k, prev_embd_k] → ws_conf_input[..input_dim]
            let hidden_k = &self.drafter.ws_confidence_hidden[k * h..(k + 1) * h];
            // hidden 部分直接 copy
            self.ws_conf_input[..h].copy_from_slice(hidden_k);
            // markov 部分 (若启用)
            if with_markov {
                let prev_embd_k = &self.draft_prev_embds[k * rank..(k + 1) * rank];
                self.ws_conf_input[h..h + rank].copy_from_slice(prev_embd_k);
            }
            debug_assert_eq!(input_dim, if with_markov { h + rank } else { h });

            // matvec: y[0] = dot(W_row_0, x), w.rows=1
            let mut logit = 0.0f32;
            w.matvec_into_slice(&self.ws_conf_input[..input_dim], std::slice::from_mut(&mut logit));
            // + bias
            logit += b.data[0];
            self.confidence_logits.push(logit);
        }
    }

    /// 根据 confidence_logits 与 threshold 计算早停位置 (DeepSpec _confident_prefix_length)
    ///
    /// 返回值: 实际应 verify 的 draft token 数量 (0..=block_size)
    /// - threshold <= 0.0: 返回 block_size (不截断, 当前默认行为)
    /// - 否则: 返回第一个 sigmoid(logit) < threshold 的位置; 若全部 >= threshold 返回 block_size
    ///
    /// 注: 若 draft[i] 被截断, draft[0..i] 仍正常 verify, draft[i..] 跳过。
    ///     bonus token 仍由 target 在 i 位置采样 (与 reject 路径一致)。
    ///
    /// ★ A/B 验证结论 (Q1_0 target): confidence head 预测不准, threshold=0.75 截断后
    ///   accepted 58→35 (-40%), 吞吐量 5.20→4.49 tok/s (-14%)。被截断的位置中 23 个
    ///   实际会被接受, 说明 confidence head 输出与实际接受率不相关。
    ///   原因: Q1_0 量化严重破坏 tap 特征质量, drafter 是 target-specific。
    ///   默认 threshold=0.0 不截断, 保留代码供未来 target 升级 (bf16/fp16) 后重新验证。
    pub fn confident_prefix_length(&self, threshold: f32) -> usize {
        let bs = self.drafter.weights.cfg.block_size;
        if threshold <= 0.0 || self.confidence_logits.is_empty() {
            return bs;
        }
        for k in 0..bs {
            if k >= self.confidence_logits.len() {
                return k;
            }
            let logit = self.confidence_logits[k];
            // sigmoid(logit) = 1 / (1 + exp(-logit))
            let sig = 1.0 / (1.0 + (-logit).exp());
            if sig < threshold {
                return k;
            }
        }
        bs
    }
}

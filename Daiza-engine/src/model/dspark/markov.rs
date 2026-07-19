//! DSpark Markov head: 顺序链式 resample
//!
//! ## 算法 (来自 dspark-markov.h / 论文 §3.2)
//!
//! Drafter 一次 forward 生成 block_size 个候选 token 的 base logits
//! (非因果 attention, 各位置独立)。为了让位置 k 的预测依赖位置 k-1 的采样
//! 结果, 在 base logits 上加一个 Markov bias B(x_{k-1}):
//!
//! ```text
//! B(x_{k-1})[v] = ⟨W1[x_{k-1}], W2[v]⟩   (rank=256 低秩分解)
//! step_logit[k][v] = base_logit[k][v] + B(x_{k-1})[v]
//! out[k] = argmax_v step_logit[k][v]
//! prev_token = out[k]   // 链式: 下一位置用本位置采样结果作为 prev
//! ```
//!
//! `W1` (markov_head_a, [vocab, rank=256] Iq1M) 提供 prev-token 的 256 维
//! embedding; `W2` (markov_head_b, [vocab, rank=256] Q4_1) 把它投影回 vocab
//! 空间。加法 bias 一次 resample 一个位置, 顺序链式。
//!
//! ## 调用方
//!
//! 由 `speculative.rs` 在 `draft_forward` 后调用, 接收 base_logits 并返回
//! resampled 的 block_size 个 token IDs + 每位置的 confidence (供调度用)。

use crate::tensor::iq1m::dequantize_iq1m_row_into;

use super::config::DrafterConfig;
use super::weights::DrafterWeights;

/// Markov resample 的工作区 (跨调用复用, 避免堆分配)
pub struct MarkovContext {
    /// prev-token 的 256 维 embedding (从 W1 行抽取)
    pub prev_embd: Vec<f32>,
    /// B(x_prev)[v] = W2 @ prev_embd,  [vocab]
    pub markov_bias: Vec<f32>,
    /// 临时: 单步 step_logit = base_logit + markov_bias, [vocab]
    pub step_logit: Vec<f32>,
}

impl MarkovContext {
    pub fn new(cfg: &DrafterConfig) -> Self {
        Self {
            prev_embd: vec![0.0; cfg.markov_rank],
            markov_bias: vec![0.0; cfg.vocab_size],
            step_logit: vec![0.0; cfg.vocab_size],
        }
    }

    /// 顺序链式 resample: 在 base_logits 上注入 Markov bias, 逐位置 argmax
    ///
    /// 输入:
    /// - `base_logits`: [block_size, vocab] (来自 drafter.draft_forward)
    /// - `anchor_token`: 位置 0 的 prev token (上一个被 target 接受的 token)
    ///
    /// 输出:
    /// - `out_tokens`: 长度 block_size 的 resampled token IDs
    /// - `out_logits`: 长度 block_size * vocab 的 step_logits (供 confidence head 用)
    ///   (写入 self.step_logit 复用, 每位置覆盖前先 copy 到 caller buffer)
    /// - `out_prev_embds`: 长度 block_size * rank 的 prev_embd 副本 (供 confidence head 用)
    ///   每位置 k 的 prev_embd = W1[prev_token_k], prev_token_k = (k==0 ? anchor : out[k-1])
    ///   与 DeepSpec prev_token_ids = [anchor, sampled[:-1]] 一致。
    ///   若传入 None 则跳过 (无 confidence head 时省 1KB copy)。
    pub fn resample_chain(
        &mut self,
        weights: &DrafterWeights,
        base_logits: &[f32],
        anchor_token: u32,
        out_tokens: &mut Vec<u32>,
        out_step_logits: &mut Vec<f32>,
        mut out_prev_embds: Option<&mut Vec<f32>>,
    ) {
        let cfg = &weights.cfg;
        let bs = cfg.block_size;
        let vocab = cfg.vocab_size;
        let rank = cfg.markov_rank;
        debug_assert_eq!(base_logits.len(), bs * vocab);
        out_tokens.clear();
        out_tokens.reserve(bs);
        out_step_logits.clear();
        out_step_logits.reserve(bs * vocab);
        if let Some(buf) = out_prev_embds.as_mut() {
            buf.clear();
            buf.reserve(bs * rank);
        }

        // prev_token 初始为 anchor
        let mut prev_token = anchor_token;

        for k in 0..bs {
            // Step 1: 抽取 prev-token 的 256 维 embedding (W1[prev_token])
            // markov_w1: [vocab, rank] Iq1M, 行 = prev_token
            dequantize_iq1m_row_into(
                &weights.markov_w1.bytes,
                prev_token as usize,
                rank,
                &mut self.prev_embd[..rank],
            );

            // Step 2: markov_bias = W2 @ prev_embd, [vocab]
            // markov_w2: [vocab, rank] Q4_1. 用 matvec_into_slice 走 AVX2 + 多线程路径
            // (vs 原 scalar 逐行: 248320 行 × 4 位置 ≈ 1M scalar dot/call, 是 draft 主要瓶颈)
            weights.markov_w2.matvec_into_slice(
                &self.prev_embd[..rank],
                &mut self.markov_bias[..vocab],
            );

            // Step 3: step_logit = base_logit[k] + markov_bias (AVX2 add, 单 pass)
            // ★ P2: 用 add_avx2 单 pass 替代 copy_from_slice + saxpy_avx2 两 pass
            //   vocab=248320, 4 位置/cycle, 内存 traffic 从 4MB 降到 3MB (-25%)
            let base_row = &base_logits[k * vocab..(k + 1) * vocab];
            crate::math::simd_exp::add_avx2(
                base_row,
                &self.markov_bias[..vocab],
                &mut self.step_logit[..vocab],
                vocab,
            );

            // Step 4: argmax → best_id
            // ★ P6: AVX2 向量化 argmax
            //   每 8 元素: AVX2 compare + movemask 检测是否有 > best_val
            //   若有, 标量扫描 8 个找 max (避免 horizontal reduce + index 跟踪复杂度)
            //   大多数块 (best_val 已接近 max) 直接跳过, 节省 ~70% 标量比较
            let mut best_id = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            #[cfg(target_arch = "x86_64")]
            if crate::math::simd_exp::simd_available() && vocab >= 8 {
                #[allow(unsafe_code)]
                unsafe {
                    use std::arch::x86_64::*;
                    let ptr = self.step_logit.as_ptr();
                    let mut i = 0;
                    let n8 = (vocab / 8) * 8;
                    while i < n8 {
                        let v = _mm256_loadu_ps(ptr.add(i));
                        let bv = _mm256_set1_ps(best_val);
                        let cmp = _mm256_cmp_ps(v, bv, _CMP_GT_OQ);
                        if _mm256_movemask_ps(cmp) != 0 {
                            // 本 8 元素块内有 > best_val, 标量扫描找 max
                            for j in 0..8 {
                                let val = *self.step_logit.get_unchecked(i + j);
                                if val > best_val {
                                    best_val = val;
                                    best_id = (i + j) as u32;
                                }
                            }
                        }
                        i += 8;
                    }
                    for v in n8..vocab {
                        if self.step_logit[v] > best_val {
                            best_val = self.step_logit[v];
                            best_id = v as u32;
                        }
                    }
                }
            } else {
                for v in 0..vocab {
                    if self.step_logit[v] > best_val {
                        best_val = self.step_logit[v];
                        best_id = v as u32;
                    }
                }
            }

            // 保存 step_logit 副本 (供 confidence head 用)
            out_step_logits.extend_from_slice(&self.step_logit[..vocab]);
            // 保存 prev_embd 副本 (供 confidence head 用, 与 step_logit 对齐位置 k)
            if let Some(buf) = out_prev_embds.as_mut() {
                buf.extend_from_slice(&self.prev_embd[..rank]);
            }

            // 链式: 下一位置的 prev = 本位置采样结果
            prev_token = best_id;
            out_tokens.push(best_id);
        }
    }
}

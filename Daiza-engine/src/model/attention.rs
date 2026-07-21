//! Gated Attention(全注意力块,16 层之一)
//!
//! ## 关键架构(来自调研 + llama.cpp qwen35.cpp)
//!
//! Qwen3-Next 的 Gated Attention:`attn_q.weight` 输出 12288 维 = Q(6144) + gate(6144)。
//!
//! ```text
//! qkv = x @ W_q   # [hidden, 12288]  → 拆成 q[6144] 与 gate[6144]
//! k   = x @ W_k   # [hidden, 1024]   (4 KV head × 256)
//! v   = x @ W_v   # [hidden, 1024]
//!
//! q = reshape(q, [24, 256])           # 24 query heads, head_dim=256
//! k = reshape(k, [4, 256])            # 4 KV heads
//! v = reshape(v, [4, 256])
//!
//! # QK-norm(在 RoPE 之前,标准 RMSNorm: y = x * inv_rms * w)
//! # 参考 qwen35.cpp build_norm(..., LLM_NORM_RMS, ...) — 不是 1+w
//! q = rmsnorm(q, q_norm)              # per-head
//! k = rmsnorm(k, k_norm)
//!
//! # Partial RoPE(前 64 维旋转)
//! apply_rope_partial(q, rope_dim=64, cos, sin)
//! apply_rope_partial(k, rope_dim=64, cos, sin)
//!
//! # GQA:把 4 个 KV head 广播到 24 个 Q head(每 6 个 Q head 共享 1 个 KV head)
//! # attention scores
//! scores = q @ k_cache.T / sqrt(256)
//! attn = softmax(scores, mask=causal)
//! y = attn @ v_cache
//!
//! # ★ Gated:用 gate 调制输出
//! gate = reshape(gate, [24, 256])
//! y = y * sigmoid(gate)
//!
//! # 输出投影
//! y = reshape(y, [6144])
//! out = y @ W_output    # [6144, 5120]
//! ```
//!
//! ## v2 优化(workspace 复用)
//!
//! 所有中间 buffer(q/gate/k/v/attn_out/scores)写入预分配的 `Workspace`,
//! 跨 token 复用。最终输出通过 `matvec_add_into_slice` 直接累加到主残差流 h。

use crate::math;
use crate::model::weights::{FullAttentionBlockWeights, Q1_0Matrix};
use crate::model::workspace::Workspace;
use crate::cache::KvCache;

/// 单 token 前向(decode 阶段,batch=1)
///
/// 输入:`h` 为主残差流(上一 block 输出)
/// 输出:`h += W_output @ (attn(h_normed) * sigmoid(gate(h_normed)))`
/// 所有中间量写入 `ws`,跨 token 复用。
#[allow(clippy::too_many_arguments)]
pub fn attention_forward_into(
    h: &mut [f32],
    w: &FullAttentionBlockWeights,
    cfg: &crate::model::Config,
    kv_cache: &mut KvCache,
    cos_sin: (&[f32], &[f32]),
    ws: &mut Workspace,
) {
    let hidden = cfg.hidden;
    let head_dim = cfg.head_dim;
    let rope_dim = cfg.rope_dim;
    let n_q_heads = cfg.head_count;
    let n_kv_heads = cfg.head_count_kv;
    let group_size = n_q_heads / n_kv_heads;
    let rms_eps = cfg.rms_eps;

    // 1. attn_norm: ws.block_normed = norm(h)
    //    ★ 用 rmsnorm_into 直接从 h 读、写入 block_normed,消除 copy_from_slice
    math::rmsnorm_into(&h[..hidden], &mut ws.block_normed, &w.attn_norm.data, rms_eps);

    // 2. ★ P0-B: Q + K + V 合并为单次线程池 barrier (共享输入 ws.block_normed)
    //    原: 3 次 scatter_wait (Q 12288 rows + K 1024 rows + V 1024 rows = 3 barriers)
    //    新: 1 次 scatter_wait (14336 rows → 1 barrier, -2 barriers/block × 16 blocks)
    //    额外收益: ws.block_normed (20KB) 在 Q/K/V 间自然驻留 L2
    {
        let matrices: &[&Q1_0Matrix] = &[&w.attn_q, &w.attn_k, &w.attn_v];
        let outputs: &mut [&mut [f32]] = &mut [
            &mut ws.attn_q_total,
            &mut ws.attn_k,
            &mut ws.attn_v,
        ];
        Q1_0Matrix::matvec_multi_into_slice(&ws.block_normed, matrices, outputs);
    }

    // 3. 解交错 Q 和 gate
    //    [Q_head0(256) | gate_head0(256) | Q_head1(256) | gate_head1(256) | ...]
    for h_i in 0..n_q_heads {
        let src = h_i * (head_dim * 2);
        let dst = h_i * head_dim;
        ws.attn_q[dst..dst + head_dim].copy_from_slice(&ws.attn_q_total[src..src + head_dim]);
        ws.attn_gate[dst..dst + head_dim].copy_from_slice(&ws.attn_q_total[src + head_dim..src + 2 * head_dim]);
    }

    // 5+6. QK-norm + RoPE 融合 (减少循环开销)
    let (cos, sin) = cos_sin;
    for h_i in 0..n_q_heads {
        let hs = h_i * head_dim;
        let he = hs + head_dim;
        math::rmsnorm_inplace(&mut ws.attn_q[hs..he], &w.attn_q_norm.data, rms_eps);
        math::apply_rope_partial(&mut ws.attn_q[hs..hs + head_dim], rope_dim, cos, sin);
    }
    for h_i in 0..n_kv_heads {
        let hs = h_i * head_dim;
        let he = hs + head_dim;
        math::rmsnorm_inplace(&mut ws.attn_k[hs..he], &w.attn_k_norm.data, rms_eps);
        math::apply_rope_partial(&mut ws.attn_k[hs..hs + head_dim], rope_dim, cos, sin);
    }

    // 7. 写入 KV cache
    kv_cache.append(&ws.attn_k, &ws.attn_v);

    // 8. Attention scores + softmax + V 加权(★ GQA 复用)
    //
    // 原实现:每 qh 独立读 K/V cache,24 qh × n_cached 次 K 读取 + 24 × n_cached 次 V 读取
    //   每 6 个共享 kvh 的 qh 重复读同一份 K/V cache → 6× 冗余读取
    //
    // GQA 复用:外层循环 kvh,内层一次性算 group_size 个 qh 的 scores + V 加权
    //   K/V cache 每 kvh 只读 1 次,服务 group_size 个 qh → 消除 6× 冗余
    //   scores 布局:[group_size × n_cached] flat, scores[qh_in_group * n_cached + c]
    let n_cached = kv_cache.len;
    let scale = 1.0 / (head_dim as f32).sqrt();
    // 清零 attn_out(全部 n_q_heads * head_dim 长度,fill 更易被识别为 memset)
    ws.attn_out.fill(0.0);
    // ★ GQA 复用:attn_scores 容量为 group_size × context_length,持有 group_size 个 qh 的 scores
    debug_assert!(ws.attn_scores.len() >= group_size * n_cached);

    for kvh in 0..n_kv_heads {
        // 阶段 1: K 复用 — 一次读 K[c][kvh],算 group_size 个 qh 的 scores
        for c in 0..n_cached {
            let k_head = &kv_cache.k_at(c)[kvh * head_dim..(kvh + 1) * head_dim];
            for qh_in_group in 0..group_size {
                let qh = kvh * group_size + qh_in_group;
                let q_head = &ws.attn_q[qh * head_dim..(qh + 1) * head_dim];
                ws.attn_scores[qh_in_group * n_cached + c] =
                    crate::math::simd_exp::dot_product_avx2(q_head, k_head, head_dim) * scale;
            }
        }

        // 阶段 2: group_size 个 qh 各自 softmax(独立行,无依赖)
        for qh_in_group in 0..group_size {
            let s = &mut ws.attn_scores[qh_in_group * n_cached..(qh_in_group + 1) * n_cached];
            math::softmax_inplace(s);
        }

        // 阶段 3: V 复用 — 一次读 V[c][kvh],做 group_size 个 qh 的 V 加权
        for c in 0..n_cached {
            let v_head = &kv_cache.v_at(c)[kvh * head_dim..(kvh + 1) * head_dim];
            for qh_in_group in 0..group_size {
                let qh = kvh * group_size + qh_in_group;
                let out_head = &mut ws.attn_out[qh * head_dim..(qh + 1) * head_dim];
                let s = ws.attn_scores[qh_in_group * n_cached + c];
                crate::math::simd_exp::saxpy_avx2(s, v_head, out_head, head_dim);
            }
        }
    }

    // 9. Gated: attn_out *= sigmoid(gate) — 融合 sigmoid + multiply
    math::sigmoid_inplace_simd(&mut ws.attn_gate);
    math::mul_inplace_simd(&mut ws.attn_out, &ws.attn_gate);

    // 10. Output projection + residual: h += W_output @ attn_out
    //     ★ &ws.attn_out (不可变) + &mut h (可变) 不冲突(h 不是 ws 字段)
    w.attn_output.matvec_add_into_slice(&ws.attn_out, h);
}

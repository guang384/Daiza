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
use crate::model::weights::FullAttentionBlockWeights;
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

    // 2. Q projection: ws.attn_q_total = W_q @ ws.block_normed
    //    ★ Split borrow: 同时 &ws.block_normed (不可变) 和 &mut ws.attn_q_total (可变)
    w.attn_q.matvec_into_slice(&ws.block_normed, &mut ws.attn_q_total);

    // 3. 解交错 Q 和 gate
    //    [Q_head0(256) | gate_head0(256) | Q_head1(256) | gate_head1(256) | ...]
    for h_i in 0..n_q_heads {
        let src = h_i * (head_dim * 2);
        let dst = h_i * head_dim;
        ws.attn_q[dst..dst + head_dim].copy_from_slice(&ws.attn_q_total[src..src + head_dim]);
        ws.attn_gate[dst..dst + head_dim].copy_from_slice(&ws.attn_q_total[src + head_dim..src + 2 * head_dim]);
    }

    // 4. K / V projection
    w.attn_k.matvec_into_slice(&ws.block_normed, &mut ws.attn_k);
    w.attn_v.matvec_into_slice(&ws.block_normed, &mut ws.attn_v);

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

    // 8. Attention scores + softmax + V 加权
    let n_cached = kv_cache.len;
    let scale = 1.0 / (head_dim as f32).sqrt();
    // 清零 attn_out(全部 n_q_heads * head_dim 长度,fill 更易被识别为 memset)
    ws.attn_out.fill(0.0);
    // attn_scores 已在 Workspace::new 中预分配到 context_length,无需热路径 resize
    debug_assert!(ws.attn_scores.len() >= n_cached);
    let scores = &mut ws.attn_scores[..n_cached];

    for qh in 0..n_q_heads {
        let kvh = qh / group_size;
        let q_head = &ws.attn_q[qh * head_dim..(qh + 1) * head_dim];

        // scores[t] = q · k_cache[t]
        // ★ AVX2 向量化: dot_product_avx2, head_dim=256 = 32 × 8-wide FMA
        for t in 0..n_cached {
            let k_t = kv_cache.k_at(t);
            let k_head = &k_t[kvh * head_dim..(kvh + 1) * head_dim];
            scores[t] = crate::math::simd_exp::dot_product_avx2(q_head, k_head, head_dim) * scale;
        }

        // softmax(全部可见,因为 decode 阶段只看历史 + 当前)
        math::softmax_inplace(scores);

        // ★ AVX2 向量化 V 加权: saxpy_avx2, 8-wide FMA
        let out_head = &mut ws.attn_out[qh * head_dim..(qh + 1) * head_dim];
        for t in 0..n_cached {
            let v_head = &kv_cache.v_at(t)[kvh * head_dim..(kvh + 1) * head_dim];
            crate::math::simd_exp::saxpy_avx2(scores[t], v_head, out_head, head_dim);
        }
    }

    // 9. Gated: attn_out *= sigmoid(gate) — 融合 sigmoid + multiply
    math::sigmoid_inplace_simd(&mut ws.attn_gate);
    math::mul_inplace_simd(&mut ws.attn_out, &ws.attn_gate);

    // 10. Output projection + residual: h += W_output @ attn_out
    //     ★ &ws.attn_out (不可变) + &mut h (可变) 不冲突(h 不是 ws 字段)
    w.attn_output.matvec_add_into_slice(&ws.attn_out, h);
}

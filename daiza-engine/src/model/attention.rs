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
    pos: usize,
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

    // 5. QK-norm(per-head 标准 RMSNorm: y = x * inv_rms * w)
    for h_i in 0..n_q_heads {
        let hs = h_i * head_dim;
        let he = hs + head_dim;
        math::rmsnorm_inplace(&mut ws.attn_q[hs..he], &w.attn_q_norm.data, rms_eps);
    }
    for h_i in 0..n_kv_heads {
        let hs = h_i * head_dim;
        let he = hs + head_dim;
        math::rmsnorm_inplace(&mut ws.attn_k[hs..he], &w.attn_k_norm.data, rms_eps);
    }

    // 6. Partial RoPE:对每个 head 的前 rope_dim 维应用旋转
    let (cos, sin) = cos_sin;
    for h_i in 0..n_q_heads {
        let hs = h_i * head_dim;
        math::apply_rope_partial(
            &mut ws.attn_q[hs..hs + head_dim],
            rope_dim,
            cos,
            sin,
        );
    }
    for h_i in 0..n_kv_heads {
        let hs = h_i * head_dim;
        math::apply_rope_partial(
            &mut ws.attn_k[hs..hs + head_dim],
            rope_dim,
            cos,
            sin,
        );
    }

    // 7. 写入 KV cache
    kv_cache.append(pos, &ws.attn_k, &ws.attn_v);

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
        for t in 0..n_cached {
            let k_t = kv_cache.k_at(t);
            let k_head = &k_t[kvh * head_dim..(kvh + 1) * head_dim];
            let mut s = 0.0f32;
            for d in 0..head_dim {
                s += q_head[d] * k_head[d];
            }
            scores[t] = s * scale;
        }

        // softmax(全部可见,因为 decode 阶段只看历史 + 当前)
        math::softmax_inplace(scores);

        // weighted sum of V
        // ★ 循环顺序交换:外层 t(顺序读 V),内层 d(连续 head_dim)
        // V cache 按 [seq, head_kv, head_dim] 存储,新顺序让 V 连续读取 head_dim 个 f32,
        // 完美命中 cache line。
        let out_head = &mut ws.attn_out[qh * head_dim..(qh + 1) * head_dim];
        for t in 0..n_cached {
            let s = scores[t];
            let v_head = &kv_cache.v_at(t)[kvh * head_dim..(kvh + 1) * head_dim];
            for d in 0..head_dim {
                out_head[d] += s * v_head[d];
            }
        }
    }

    // 9. Gated:用 gate 调制 attn_out
    //    attn_out[i] *= sigmoid(gate[i])
    //    ★ 用 SIMD sigmoid(8-wide AVX2)替代标量 expf
    //      6144 次/层 × 16 层 = 98304 次/token,标量 ~30c vs SIMD ~1.5c
    {
        // 先把 sigmoid(gate) 算到 attn_gate 原地(后续不再用 attn_gate)
        math::sigmoid_inplace_simd(&mut ws.attn_gate);
        // attn_out *= sigmoid(gate)
        for i in 0..ws.attn_out.len() {
            ws.attn_out[i] *= ws.attn_gate[i];
        }
    }

    // 10. Output projection + residual: h += W_output @ attn_out
    //     ★ &ws.attn_out (不可变) + &mut h (可变) 不冲突(h 不是 ws 字段)
    w.attn_output.matvec_add_into_slice(&ws.attn_out, h);
}

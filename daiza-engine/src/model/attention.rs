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

use crate::math;
use crate::model::weights::FullAttentionBlockWeights;
use crate::cache::KvCache;

pub struct AttentionOutput {
    pub out: Vec<f32>, // [hidden]
}

/// 单 token 前向(decode 阶段,batch=1)
#[allow(clippy::too_many_arguments)]
pub fn attention_forward_single(
    x: &[f32],
    w: &FullAttentionBlockWeights,
    cfg: &crate::model::Config,
    kv_cache: &mut KvCache,
    pos: usize,
    cos_sin: (&[f32], &[f32]),
) -> AttentionOutput {
    let hidden = cfg.hidden;
    let head_dim = cfg.head_dim;
    let rope_dim = cfg.rope_dim;
    let n_q_heads = cfg.head_count;
    let n_kv_heads = cfg.head_count_kv;
    let group_size = n_q_heads / n_kv_heads;

    // 1. Q projection:输出 12288 = 24 * (256 Q + 256 gate)
    //    ★ Q 和 gate 按 head 交错排列(参考 qwen35.cpp L272-297):
    //      [Q_head0(256) | gate_head0(256) | Q_head1(256) | gate_head1(256) | ...]
    //    不是 [全部Q(6144) | 全部gate(6144)]
    let q_total = w.attn_q.matvec(x);
    let mut q = vec![0.0f32; n_q_heads * head_dim];
    let mut gate = vec![0.0f32; n_q_heads * head_dim];
    for h in 0..n_q_heads {
        let src = h * (head_dim * 2);
        let dst = h * head_dim;
        q[dst..dst + head_dim].copy_from_slice(&q_total[src..src + head_dim]);
        gate[dst..dst + head_dim].copy_from_slice(&q_total[src + head_dim..src + 2 * head_dim]);
    }

    // 2. K / V projection
    let mut k = w.attn_k.matvec(x);
    let v = w.attn_v.matvec(x);

    // 3. QK-norm(per-head 标准 RMSNorm: y = x * inv_rms * w)
    //    参考 qwen35.cpp: build_norm(Qcur, attn_q_norm, nullptr, LLM_NORM_RMS, il)
    //    注意: 不是 (1+w) 缩放,是标准 RMSNorm
    let rms_eps = cfg.rms_eps;
    for h in 0..n_q_heads {
        let head_start = h * head_dim;
        let head_end = head_start + head_dim;
        math::rmsnorm_inplace(&mut q[head_start..head_end], &w.attn_q_norm.data, rms_eps);
    }
    for h in 0..n_kv_heads {
        let head_start = h * head_dim;
        let head_end = head_start + head_dim;
        math::rmsnorm_inplace(&mut k[head_start..head_end], &w.attn_k_norm.data, rms_eps);
    }

    // 4. Partial RoPE:对每个 head 的前 rope_dim 维应用旋转
    let (cos, sin) = cos_sin;
    for h in 0..n_q_heads {
        let head_start = h * head_dim;
        math::apply_rope_partial(
            &mut q[head_start..head_start + head_dim],
            rope_dim,
            cos,
            sin,
        );
    }
    for h in 0..n_kv_heads {
        let head_start = h * head_dim;
        math::apply_rope_partial(
            &mut k[head_start..head_start + head_dim],
            rope_dim,
            cos,
            sin,
        );
    }

    // 5. 写入 KV cache(把当前 token 的 K/V 追加到 cache)
    kv_cache.append(pos, &k, &v);

    // 6. Attention scores:对每个 Q head,与所有历史 K 计算点积
    let n_cached = kv_cache.len; // 当前已缓存多少 token(含刚写入的)
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; n_q_heads * head_dim]; // [24, 256]

    for qh in 0..n_q_heads {
        let kvh = qh / group_size; // 对应的 KV head 索引
        let q_head = &q[qh * head_dim..(qh + 1) * head_dim];

        // scores[t] = q · k_cache[t]
        let mut scores = vec![0.0f32; n_cached];
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
        math::softmax_inplace(&mut scores);

        // weighted sum of V
        let out_head = &mut attn_out[qh * head_dim..(qh + 1) * head_dim];
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for t in 0..n_cached {
                let v_t = kv_cache.v_at(t);
                let v_val = v_t[kvh * head_dim + d];
                acc += scores[t] * v_val;
            }
            out_head[d] = acc;
        }
    }

    // 7. Gated:用 gate 调制 attn_out
    //    attn_out[i] *= sigmoid(gate[i])
    for i in 0..attn_out.len() {
        let sigmoid_g = 1.0 / (1.0 + (-gate[i]).exp());
        attn_out[i] *= sigmoid_g;
    }

    // 8. Output projection:y @ W_output  ([6144, 5120])
    let out = w.attn_output.matvec(&attn_out);

    AttentionOutput { out }
}

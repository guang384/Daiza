//! Gated DeltaNet (SSM 块, 48 层之一)
//!
//! ## 架构 (Qwen3NextGatedDeltaNet)
//!
//! 参考: HuggingFace `modular_qwen3_next.py` + CSDN 深度拆解
//!
//! - `num_v_heads` = 48 (value heads, GGUF 中误导命名为 `ssm_time_step_rank`)
//! - `num_k_heads` = 16 (key/query heads, GGUF 中命名为 `ssm_group_count`)
//! - `head_k_dim` = `head_v_dim` = 128 (= `ssm_state_size`)
//! - 每 3 个 v_head 共享 1 个 k_head (GQA repeat_kv: 48/16=3)
//! - 状态: `[num_v_heads, 128, 128]` = 48 * 16384 = 786432 floats
//!
//! ## Gated Delta Rule (来自 HF 参考 + CSDN + llama.cpp qwen35.cpp)
//!
//! ```text
//! // ssm_a 张量存储的是 A = -exp(A_log) (已取负号和 exp), 不是 A_log 本身
//! g    = A * softplus(a + dt_bias)             # [48] 衰减, A = -exp(A_log)
//! beta = sigmoid(b)                            # [48] 输入门
//!
//! # Per v_head (48):
//!   decay = exp(g[vh])
//!   S = decay * S                               # 1) 衰减 FIRST
//!   kv_mem = S @ k                              # 2) 使用衰减后的 S
//!   delta = (v - kv_mem) * beta                 # 3) delta 校正
//!   S += k ⊗ delta                              # 4) 外积更新
//!   y = S @ q                                   # 5) 使用更新后的 S
//! ```
//!
//! ## q/k 处理
//!
//! - q, k 投影到 16 头 × 128 维
//! - L2 norm (无权重, `use_qk_l2norm_in_kernel`): `x * rsqrt(sum(x²) + eps)`
//! - q 预缩放: `q *= 1/sqrt(head_dim)`
//! - q, k 重复 16→48 头 (GQA repeat_kv)
//!
//! ## 输出 (Qwen3NextRMSNormGated)
//!
//! `y = rmsnorm(y) * ssm_norm_weight * silu(z)`
//! 其中 `z = attn_gate @ x` (输出门), 对每个 v_head (head_v_dim=128) 单独应用

use crate::math;
use crate::model::weights::SsmBlockWeights;
use crate::cache::SsmState;

pub struct SsmOutput {
    pub out: Vec<f32>, // [hidden]
}

const SSM_EPS: f32 = 1e-6;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// softplus(x) = ln(1 + exp(x)), 数值稳定
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// L2 normalization (无权重, use_qk_l2norm_in_kernel)
/// `x * rsqrt(sum(x²) + eps)` —— 单位向量归一化
#[inline]
fn l2norm_inplace(x: &mut [f32], eps: f32) {
    let mut ss = 0.0f32;
    for &xi in x.iter() {
        ss += xi * xi;
    }
    let inv_norm = 1.0 / (ss + eps).sqrt();
    for xi in x.iter_mut() {
        *xi *= inv_norm;
    }
}

/// 单 token 前向 (decode)
pub fn ssm_forward_single(
    x: &[f32],
    w: &SsmBlockWeights,
    cfg: &crate::model::Config,
    state: &mut SsmState,
    pos: usize,
) -> SsmOutput {
    let _ = pos;
    let inner = cfg.ssm_inner_size;              // 6144 = num_v_heads * head_v_dim
    let num_k_heads = cfg.ssm_group_count;        // 16  (GGUF 命名误导, 实际 num_k_heads)
    let state_size = cfg.ssm_state_size;          // 128 = head_k_dim = head_v_dim
    let num_v_heads = cfg.ssm_time_step_rank;     // 48  (GGUF 命名误导, 实际 num_v_heads)
    let conv_k = cfg.ssm_conv_kernel;             // 4
    let head_dim = state_size;                    // 128
    let v_heads_per_group = num_v_heads / num_k_heads; // 3
    let qkv_dim = num_k_heads * head_dim;         // 2048 (q/k 维度)
    let qkv_full_len = 2 * qkv_dim + inner;       // 10240

    // 1. attn_qkv 投影: [q(2048), k(2048), v(6144)] = 10240
    let qkv = w.attn_qkv.matvec(x);
    let mut q = qkv[..qkv_dim].to_vec();                    // [2048] = 16 heads × 128
    let mut k = qkv[qkv_dim..2 * qkv_dim].to_vec();         // [2048] = 16 heads × 128
    let mut v = qkv[2 * qkv_dim..2 * qkv_dim + inner].to_vec(); // [6144] = 48 heads × 128

    // 2. Conv1d (depthwise, causal, kernel=4) + silu on cat(q,k,v)
    if state.conv_history.is_empty() {
        state.conv_history.resize(conv_k * qkv_full_len, 0.0);
    }
    // 滑窗左移
    for t in 0..conv_k - 1 {
        let src = (t + 1) * qkv_full_len;
        let dst = t * qkv_full_len;
        state.conv_history.copy_within(src..src + qkv_full_len, dst);
    }
    // 末位写当前 qkv
    let cur_offset = (conv_k - 1) * qkv_full_len;
    state.conv_history[cur_offset..cur_offset + qkv_full_len].copy_from_slice(&qkv);

    // depthwise conv1d + silu
    // ssm_conv1d.weight GGUF dims=[conv_k=4, qkv_full_len=10240]
    //   行优先存储: (channel=ch, kernel_pos=t) 偏移 = ch * conv_k + t
    let conv_w = &w.ssm_conv1d.data;
    let mut conv_out = vec![0.0f32; qkv_full_len];
    for ch in 0..qkv_full_len {
        let mut acc = 0.0f32;
        for t in 0..conv_k {
            acc += state.conv_history[t * qkv_full_len + ch] * conv_w[ch * conv_k + t];
        }
        conv_out[ch] = math::silu(acc);
    }
    q.copy_from_slice(&conv_out[..qkv_dim]);
    k.copy_from_slice(&conv_out[qkv_dim..2 * qkv_dim]);
    v.copy_from_slice(&conv_out[2 * qkv_dim..2 * qkv_dim + inner]);

    // 3. q/k per-head L2 normalization (无权重, use_qk_l2norm_in_kernel)
    for h in 0..num_k_heads {
        let hs = h * head_dim;
        l2norm_inplace(&mut q[hs..hs + head_dim], SSM_EPS);
        l2norm_inplace(&mut k[hs..hs + head_dim], SSM_EPS);
    }

    // 4. q 预缩放: q *= 1/sqrt(head_dim)
    let q_scale = 1.0 / (head_dim as f32).sqrt();
    for qi in q.iter_mut() {
        *qi *= q_scale;
    }

    // 5. Alpha / Beta / dt / A 投影 (per v_head, [48])
    let alpha = w.ssm_alpha.matvec(x);   // [48] -> a
    let beta_raw = w.ssm_beta.matvec(x); // [48] -> b
    let dt_bias = &w.ssm_dt_bias.data;   // [48]
    // 注意: ssm_a 张量存储的是 A = -exp(A_log) (已取负号和 exp), 不是 A_log 本身
    // 参考实现 qwen35.cpp: gate = alpha_softplus * ssm_a  (注释: -A_log.exp() * softplus)
    let a = &w.ssm_a.data;               // [48] -> A = -exp(A_log)

    // 6. Gated Delta Rule scan (per v_head, 48 heads)
    //    状态: [48, 128, 128]
    let state_buf = &mut state.state;
    let mut y = vec![0.0f32; inner]; // [48 * 128] = 6144

    for vh in 0..num_v_heads {
        // v_head vh 映射到 k_head kh (GQA: 每 3 个 v_head 共享 1 个 k_head)
        let kh = vh / v_heads_per_group;
        let q_head = &q[kh * head_dim..(kh + 1) * head_dim];   // [128]
        let k_head = &k[kh * head_dim..(kh + 1) * head_dim];   // [128]
        let v_head = &v[vh * head_dim..(vh + 1) * head_dim];   // [128]

        // g = A * softplus(a + dt_bias), 其中 A = -exp(A_log) 已存储在 ssm_a 中
        // 参考实现: gate = alpha_softplus * ssm_a  (注释: -A_log.exp() * softplus)
        let g = a[vh] * softplus(alpha[vh] + dt_bias[vh]);
        let decay = g.exp();
        let beta = sigmoid(beta_raw[vh]);

        // 该 v_head 的状态 [128, 128]
        let s_off = vh * state_size * state_size;
        let s = &mut state_buf[s_off..s_off + state_size * state_size];

        // Delta Rule:
        // 1) 衰减 FIRST: S = decay * S
        for si in s.iter_mut() {
            *si *= decay;
        }
        // 2) kv_mem = S @ k  (使用衰减后的 S)
        //    kv_mem[i] = sum_j s[i, j] * k[j]
        let mut kv_mem = vec![0.0f32; head_dim];
        for i in 0..head_dim {
            let mut acc = 0.0f32;
            let srow = &s[i * head_dim..(i + 1) * head_dim];
            for j in 0..head_dim {
                acc += srow[j] * k_head[j];
            }
            kv_mem[i] = acc;
        }
        // 3) delta = (v - kv_mem) * beta
        let mut delta = vec![0.0f32; head_dim];
        for i in 0..head_dim {
            delta[i] = (v_head[i] - kv_mem[i]) * beta;
        }
        // 4) S += k ⊗ delta  (外积: s[i, j] += delta[i] * k[j])
        for i in 0..head_dim {
            let di = delta[i];
            let srow = &mut s[i * head_dim..(i + 1) * head_dim];
            for j in 0..head_dim {
                srow[j] += di * k_head[j];
            }
        }
        // 5) y = S @ q  (使用更新后的 S)
        //    y[i] = sum_j s[i, j] * q[j]
        let y_off = vh * head_dim;
        for i in 0..head_dim {
            let mut acc = 0.0f32;
            let srow = &s[i * head_dim..(i + 1) * head_dim];
            for j in 0..head_dim {
                acc += srow[j] * q_head[j];
            }
            y[y_off + i] = acc;
        }
    }

    // 7. Output gate: Qwen3NextRMSNormGated
    //    y = rmsnorm(y) * ssm_norm_weight * silu(z)
    //    其中 z = attn_gate @ x (输出门)
    //    对每个 v_head (head_v_dim=128) 单独应用 RMSNorm + weight + silu(gate)
    let z = w.attn_gate.matvec(x);     // [6144]
    let ssm_norm_w = &w.ssm_norm.data; // [128]
    for vh in 0..num_v_heads {
        let y_off = vh * head_dim;
        // RMSNorm per v_head: variance = mean(x²)
        let mut ss = 0.0f32;
        for i in 0..head_dim {
            ss += y[y_off + i] * y[y_off + i];
        }
        let inv_rms = 1.0 / (ss / head_dim as f32 + SSM_EPS).sqrt();
        // y = rmsnorm(y) * weight * silu(z)
        for i in 0..head_dim {
            let normed = y[y_off + i] * inv_rms;
            y[y_off + i] = normed * ssm_norm_w[i] * math::silu(z[y_off + i]);
        }
    }

    // 8. Output projection: [6144] -> [5120]
    let out = w.ssm_out.matvec(&y);

    SsmOutput { out }
}

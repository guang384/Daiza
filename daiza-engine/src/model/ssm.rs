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
//! 其中 `z = attn_gate @ x` (输出门, 对每个 v_head (head_v_dim=128) 单独应用)
//!
//! ## v2 优化(workspace 复用)
//!
//! 所有中间 buffer(qkv/conv_out/q/k/v/y/alpha/beta/z)写入预分配的 `Workspace`,
//! 跨 token 复用。最终输出通过 `matvec_add_into_slice` 直接累加到主残差流 h。

use crate::math;
use crate::model::weights::SsmBlockWeights;
use crate::model::workspace::Workspace;
use crate::cache::SsmState;

const SSM_EPS: f32 = 1e-6;

#[inline]
fn sigmoid(x: f32) -> f32 {
    crate::math::simd_exp::sigmoid_fast(x)
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

/// 单个 v_head 的 Gated Delta Rule scan(可并行,无副作用)
///
/// 输入:`s` = [state_size, state_size], `y` = [head_dim]
/// 输出:原地更新 `s` 和 `y`
///
/// 融合 5 个原步骤为 2 pass(减少 S 流量 50%):
/// - Pass 1: s *= decay 同时累加 kv_mem[i] = sum_j S[i,j] * k[j]
/// - Pass 2: S += delta ⊗ k 同时计算 y[i] = sum_j S_new[i,j] * q[j]
#[inline]
fn ssm_scan_vhead(
    s: &mut [f32],
    y: &mut [f32],
    q_head: &[f32],
    k_head: &[f32],
    v_head: &[f32],
    a_vh: f32,
    alpha_vh: f32,
    beta_vh: f32,
    dt_bias_vh: f32,
    head_dim: usize,
) {
    let g = a_vh * softplus(alpha_vh + dt_bias_vh);
    let decay = g.exp();
    let beta = sigmoid(beta_vh);

    let mut kv_mem = [0.0f32; 128];

    // Pass 1: s *= decay 同时累加 kv_mem
    for i in 0..head_dim {
        let srow = &mut s[i * head_dim..(i + 1) * head_dim];
        let mut acc = 0.0f32;
        for j in 0..head_dim {
            let s_new = srow[j] * decay;
            srow[j] = s_new;
            acc += s_new * k_head[j];
        }
        kv_mem[i] = acc;
    }

    // Pass 2: S += delta ⊗ k 同时计算 y
    for i in 0..head_dim {
        let di = (v_head[i] - kv_mem[i]) * beta;
        let srow = &mut s[i * head_dim..(i + 1) * head_dim];
        let mut acc = 0.0f32;
        for j in 0..head_dim {
            srow[j] += di * k_head[j];
            acc += srow[j] * q_head[j];
        }
        y[i] = acc;
    }
}

/// 单 token 前向 (decode)
///
/// 输入:`h` 为主残差流(上一 block 输出)
/// 输出:`h += W_out @ (rmsnorm(y) * ssm_norm_w * silu(z))`
/// 所有中间量写入 `ws`,跨 token 复用。
pub fn ssm_forward_into(
    h: &mut [f32],
    w: &SsmBlockWeights,
    cfg: &crate::model::Config,
    state: &mut SsmState,
    pos: usize,
    ws: &mut Workspace,
) {
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

    // 1. attn_norm: ws.block_normed = norm(h)
    //    ★ 用 rmsnorm_into 直接从 h 读、写入 block_normed,消除 copy_from_slice
    math::rmsnorm_into(&h[..cfg.hidden], &mut ws.block_normed, &w.attn_norm.data, cfg.rms_eps);

    // 2. attn_qkv 投影: ws.ssm_qkv = W_qkv @ ws.block_normed
    //    [q(2048), k(2048), v(6144)] = 10240
    w.attn_qkv.matvec_into_slice(&ws.block_normed, &mut ws.ssm_qkv);

    // 3. Conv1d (depthwise, causal, kernel=4) + silu on cat(q,k,v)
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
    state.conv_history[cur_offset..cur_offset + qkv_full_len].copy_from_slice(&ws.ssm_qkv);

    // depthwise conv1d + silu
    // ssm_conv1d.weight GGUF dims=[conv_k=4, qkv_full_len=10240]
    //   行优先存储: (channel=ch, kernel_pos=t) 偏移 = ch * conv_k + t
    //
    // ★ 循环顺序交换(零成本优化):外层 t,内层 ch
    //   原顺序:外层 ch(10240),内层 t(4) — conv_history 访问 stride=10240,
    //   每 t 读取 40KB,4 次全部 L1 cache miss
    //   新顺序:外层 t(4),内层 ch(10240) — conv_history 连续读取,
    //   rustc 自动 AVX2 向量化 saxpy,cache 完美命中
    let conv_w = &w.ssm_conv1d.data;
    // 先清零 conv_out(fill 更易被识别为 memset)
    ws.ssm_conv_out.fill(0.0);
    // 外层 t,内层 ch:hist_row[ch] 连续,conv_w[ch*conv_k+t] stride=4 但总 size 仅 160KB
    // (放得下 L2,硬件预取器能识别 stride 模式)
    for t in 0..conv_k {
        let hist_row = &state.conv_history[t * qkv_full_len..(t + 1) * qkv_full_len];
        for ch in 0..qkv_full_len {
            ws.ssm_conv_out[ch] += hist_row[ch] * conv_w[ch * conv_k + t];
        }
    }
    // ★ silu 融合进拆分 copy + SIMD silu_fast
    //   10240 次/SSM 块 × 48 SSM 块 = 491520 次/token
    //   silu_fast 内部走 AVX2 8-way exp,标量 ~30c vs SIMD ~1.5c
    use crate::math::simd_exp::silu_fast;
    for i in 0..qkv_dim {
        ws.ssm_q[i] = silu_fast(ws.ssm_conv_out[i]);
        ws.ssm_k[i] = silu_fast(ws.ssm_conv_out[qkv_dim + i]);
    }
    for i in 0..inner {
        ws.ssm_v[i] = silu_fast(ws.ssm_conv_out[2 * qkv_dim + i]);
    }

    // 4. q/k per-head L2 normalization (无权重, use_qk_l2norm_in_kernel)
    for h_i in 0..num_k_heads {
        let hs = h_i * head_dim;
        l2norm_inplace(&mut ws.ssm_q[hs..hs + head_dim], SSM_EPS);
        l2norm_inplace(&mut ws.ssm_k[hs..hs + head_dim], SSM_EPS);
    }

    // 5. q 预缩放: q *= 1/sqrt(head_dim)
    let q_scale = 1.0 / (head_dim as f32).sqrt();
    for qi in ws.ssm_q.iter_mut() {
        *qi *= q_scale;
    }

    // 6. Alpha / Beta / dt / A 投影 (per v_head, [48])
    w.ssm_alpha.matvec_into_slice(&ws.block_normed, &mut ws.ssm_alpha);
    w.ssm_beta.matvec_into_slice(&ws.block_normed, &mut ws.ssm_beta);
    let dt_bias = &w.ssm_dt_bias.data;   // [48]
    // 注意: ssm_a 张量存储的是 A = -exp(A_log) (已取负号和 exp), 不是 A_log 本身
    // 参考实现 qwen35.cpp: gate = alpha_softplus * ssm_a  (注释: -A_log.exp() * softplus)
    let a = &w.ssm_a.data;               // [48] -> A = -exp(A_log)

    // 7. Gated Delta Rule scan (per v_head, 48 heads)
    //    状态: [48, 128, 128]
    //
    // ★ 融合 v2:把原本 5 个独立循环合并为 2 个 pass
    //   原:1) s*=decay  2) kv_mem=S@k  3) delta=(v-kv_mem)*beta  4) S+=delta⊗k  5) y=S_new@q
    //      共 4 遍 S 流量 = 4 * 16384 * 4B = 256KB/v_head × 48 = 12.3MB/token
    //   融合:
    //      Pass 1: s *= decay 同时累加 kv_mem[i] = sum_j S[i,j] * k[j]
    //      Pass 2: S += delta ⊗ k 同时计算 y[i] = sum_j S_new[i,j] * q[j]
    //   流量降至 2 遍 = 6.1MB/token(50%↓),且消除 2 个 Vec 堆分配/v_head
    //
    // 注:SSM scan 计算量仅 75M FMA/token(vs matvec 20G FMA/token,占 0.3%),
    // 并行化经实测无收益(spawn 开销 > 并行收益),保持串行。
    let state_buf = &mut state.state;
    // 清零 y(fill 更易被识别为 memset)
    ws.ssm_y.fill(0.0);

    for vh in 0..num_v_heads {
        let kh = vh / v_heads_per_group;
        let q_head = &ws.ssm_q[kh * head_dim..(kh + 1) * head_dim];
        let k_head = &ws.ssm_k[kh * head_dim..(kh + 1) * head_dim];
        let v_head = &ws.ssm_v[vh * head_dim..(vh + 1) * head_dim];
        let s_off = vh * state_size * state_size;
        let s = &mut state_buf[s_off..s_off + state_size * state_size];
        let y_off = vh * head_dim;
        let y = &mut ws.ssm_y[y_off..y_off + head_dim];
        ssm_scan_vhead(
            s, y, q_head, k_head, v_head,
            a[vh], ws.ssm_alpha[vh], ws.ssm_beta[vh], dt_bias[vh],
            head_dim,
        );
    }

    // 8. Output gate: Qwen3NextRMSNormGated
    //    y = rmsnorm(y) * ssm_norm_weight * silu(z)
    //    其中 z = attn_gate @ x (输出门)
    //    对每个 v_head (head_v_dim=128) 单独应用 RMSNorm + weight + silu(gate)
    w.attn_gate.matvec_into_slice(&ws.block_normed, &mut ws.ssm_z);
    let ssm_norm_w = &w.ssm_norm.data; // [128]
    for vh in 0..num_v_heads {
        let y_off = vh * head_dim;
        // RMSNorm per v_head: variance = mean(x²)
        let mut ss = 0.0f32;
        for i in 0..head_dim {
            ss += ws.ssm_y[y_off + i] * ws.ssm_y[y_off + i];
        }
        let inv_rms = 1.0 / (ss / head_dim as f32 + SSM_EPS).sqrt();
        // y = rmsnorm(y) * weight * silu(z)
        // ★ SIMD silu(48 v_head × 128 = 6144 次/层 × 48 SSM 块 = 294912 次/token)
        for i in 0..head_dim {
            let normed = ws.ssm_y[y_off + i] * inv_rms;
            ws.ssm_y[y_off + i] = normed * ssm_norm_w[i] * crate::math::simd_exp::silu_fast(ws.ssm_z[y_off + i]);
        }
    }

    // 9. Output projection + residual: h += W_out @ y
    //    ★ &ws.ssm_y (不可变) + &mut h (可变) 不冲突
    w.ssm_out.matvec_add_into_slice(&ws.ssm_y, h);
}

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
use crate::model::weights::{Q1_0Matrix, SsmBlockWeights};
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
///
/// ★ P0-2 优化: 内层 j 循环手写 AVX2 (head_dim=128 = 16 × 8-wide FMA)
///   原标量循环有 read-after-write 依赖,rustc 无法自动向量化
#[inline]
pub(crate) fn ssm_scan_vhead(
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

    // Pass 1 + Pass 2 (AVX2 向量化的内层循环)
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        #[allow(unsafe_code)]
        unsafe {
            ssm_scan_pass1_avx2(s, k_head, decay, &mut kv_mem, head_dim);
            ssm_scan_pass2_avx2(s, q_head, k_head, v_head, &kv_mem, beta, y, head_dim);
        }
        return;
    }
    // Fallback: 标量实现(非 x86_64 或无 AVX2)
    ssm_scan_scalar(s, y, q_head, k_head, v_head, decay, beta, &mut kv_mem, head_dim);
}

/// 标量 fallback(与 AVX2 版本逻辑一致)
#[inline(never)]
fn ssm_scan_scalar(
    s: &mut [f32],
    y: &mut [f32],
    q_head: &[f32],
    k_head: &[f32],
    v_head: &[f32],
    decay: f32,
    beta: f32,
    kv_mem: &mut [f32; 128],
    head_dim: usize,
) {
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

/// AVX2 向量化的 Pass 1: s *= decay 同时累加 kv_mem[i] = sum_j s[i,j] * k[j]
///
/// 内层 128 元素循环 = 16 次 8-wide FMA,完全打破标量依赖链
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
#[inline]
unsafe fn ssm_scan_pass1_avx2(
    s: &mut [f32],
    k_head: &[f32],
    decay: f32,
    kv_mem: &mut [f32; 128],
    head_dim: usize,
) {
    use core::arch::x86_64::*;
    let decay_v = _mm256_set1_ps(decay);
    for i in 0..head_dim {
        let srow = &mut s[i * head_dim..(i + 1) * head_dim];
        let mut acc = _mm256_setzero_ps();
        for j in (0..head_dim).step_by(8) {
            let s_old = _mm256_loadu_ps(srow.as_ptr().add(j));
            let k = _mm256_loadu_ps(k_head.as_ptr().add(j));
            let s_new = _mm256_mul_ps(s_old, decay_v);
            _mm256_storeu_ps(srow.as_mut_ptr().add(j), s_new);
            acc = _mm256_fmadd_ps(s_new, k, acc);
        }
        kv_mem[i] = horizontal_sum_ps(acc);
    }
}

/// AVX2 向量化的 Pass 2: S += delta ⊗ k 同时计算 y[i] = sum_j S_new[i,j] * q[j]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
#[inline]
unsafe fn ssm_scan_pass2_avx2(
    s: &mut [f32],
    q_head: &[f32],
    k_head: &[f32],
    v_head: &[f32],
    kv_mem: &[f32; 128],
    beta: f32,
    y: &mut [f32],
    head_dim: usize,
) {
    use core::arch::x86_64::*;
    for i in 0..head_dim {
        let di = (v_head[i] - kv_mem[i]) * beta;
        let di_v = _mm256_set1_ps(di);
        let srow = &mut s[i * head_dim..(i + 1) * head_dim];
        let mut acc = _mm256_setzero_ps();
        for j in (0..head_dim).step_by(8) {
            let s_old = _mm256_loadu_ps(srow.as_ptr().add(j));
            let k = _mm256_loadu_ps(k_head.as_ptr().add(j));
            let q = _mm256_loadu_ps(q_head.as_ptr().add(j));
            let s_new = _mm256_fmadd_ps(di_v, k, s_old);
            _mm256_storeu_ps(srow.as_mut_ptr().add(j), s_new);
            acc = _mm256_fmadd_ps(s_new, q, acc);
        }
        y[i] = horizontal_sum_ps(acc);
    }
}

/// __m256 → f32 横向求和(纯寄存器内,无 store)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[inline]
unsafe fn horizontal_sum_ps(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
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
    ws: &mut Workspace,
) {
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

    // 2. ★ P0-B: qkv + alpha + beta + gate 合并为单次线程池 barrier (共享输入 ws.block_normed)
    //    原: qkv(10240 rows, barrier) + alpha(48 rows, single-thread) + beta(48, single-thread)
    //        + gate(6144 rows, barrier) = 2 barriers
    //    新: 1 次 scatter_wait (16480 rows → 1 barrier, -1 barrier/block × 48 blocks)
    //    alpha/beta/gate 的值提前计算但延后使用 (均只依赖 ws.block_normed, 无数据依赖)
    {
        let matrices: &[&Q1_0Matrix] = &[
            &w.attn_qkv,
            &w.ssm_alpha,
            &w.ssm_beta,
            &w.attn_gate,
        ];
        let outputs: &mut [&mut [f32]] = &mut [
            &mut ws.ssm_qkv,
            &mut ws.ssm_alpha,
            &mut ws.ssm_beta,
            &mut ws.ssm_z,
        ];
        Q1_0Matrix::matvec_multi_into_slice(&ws.block_normed, matrices, outputs);
    }

    // 3. Conv1d (depthwise, causal, kernel=4) + silu on cat(q,k,v)
    if state.conv_history.is_empty() {
        state.conv_history.resize(conv_k * qkv_full_len, 0.0);
        state.conv_head = 0;
    }
    // ★ P2-2: 环形 buffer — 写入 conv_head 行(最旧位置), 然后 head 前进
    //   原实现滑窗左移 O((conv_k-1)*qkv_full_len) copy, 现只写 1 行
    let cur_offset = state.conv_head * qkv_full_len;
    state.conv_history[cur_offset..cur_offset + qkv_full_len].copy_from_slice(&ws.ssm_qkv);
    state.conv_head = (state.conv_head + 1) % conv_k;

    // depthwise conv1d + silu
    // ssm_conv1d.weight GGUF dims=[conv_k=4, qkv_full_len=10240]
    //   行优先存储: (channel=ch, kernel_pos=t) 偏移 = ch * conv_k + t
    //
    // ★ P2-2: 环形读取 — 第 t 个历史 token 在 (conv_head + t) % conv_k 行
    //   conv_head 现指向最旧 token, t=0 最旧, t=conv_k-1 最新
    //   conv_k=4 行共 160KB 全在 L2, 环形访问顺序不影响 cache 命中
    let conv_w = &w.ssm_conv1d.data;
    // 先清零 conv_out(fill 更易被识别为 memset)
    ws.ssm_conv_out.fill(0.0);
    for t in 0..conv_k {
        let row = (state.conv_head + t) % conv_k;
        let hist_row = &state.conv_history[row * qkv_full_len..(row + 1) * qkv_full_len];
        for ch in 0..qkv_full_len {
            ws.ssm_conv_out[ch] += hist_row[ch] * conv_w[ch * conv_k + t];
        }
    }
    // ★ P2-9: silu 向量化 — 原 10240 次 silu_fast (broadcast+extract 浪费 7 lane)
    //   改为先原地 SIMD silu (8-way),再 memcpy 拆分
    //   10240 次/SSM 块 × 48 SSM 块 = 491520 次/token
    use crate::math::simd_exp::silu_inplace_simd;
    silu_inplace_simd(&mut ws.ssm_conv_out[..2 * qkv_dim + inner]);
    ws.ssm_q[..qkv_dim].copy_from_slice(&ws.ssm_conv_out[..qkv_dim]);
    ws.ssm_k[..qkv_dim].copy_from_slice(&ws.ssm_conv_out[qkv_dim..2 * qkv_dim]);
    ws.ssm_v[..inner].copy_from_slice(&ws.ssm_conv_out[2 * qkv_dim..2 * qkv_dim + inner]);

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

    // 6. Alpha / Beta / dt / A (per v_head, [48])
    //    ★ P0-B: alpha/beta 已在步骤 2 与 qkv/gate 一起计算, 这里直接读取
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
    //    其中 z = attn_gate @ x (输出门, 已在步骤 2 计算)
    //    对每个 v_head (head_v_dim=128) 单独应用 RMSNorm + weight + silu(gate)
    let ssm_norm_w = &w.ssm_norm.data; // [128]
    // ★ P1-2: 批量 silu(z) 一次(6144 元素 = 768 × 8-wide SIMD)
    //   原标量 silu_fast 294912 次/token,每次 ~5c(broadcast+extract 浪费 7 lane)
    //   批量后 768 次 SIMD,每次 ~10c 处理 8 元素 → ~1.25c/element
    math::silu_inplace_simd(&mut ws.ssm_z[..num_v_heads * head_dim]);
    for vh in 0..num_v_heads {
        let y_off = vh * head_dim;
        // RMSNorm per v_head: variance = mean(x²)
        let mut ss = 0.0f32;
        for i in 0..head_dim {
            ss += ws.ssm_y[y_off + i] * ws.ssm_y[y_off + i];
        }
        let inv_rms = 1.0 / (ss / head_dim as f32 + SSM_EPS).sqrt();
        // y = rmsnorm(y) * weight * silu(z)  (z 已批量 silu,这里直接乘)
        for i in 0..head_dim {
            ws.ssm_y[y_off + i] = ws.ssm_y[y_off + i] * inv_rms * ssm_norm_w[i] * ws.ssm_z[y_off + i];
        }
    }

    // 9. Output projection + residual: h += W_out @ y
    //    ★ &ws.ssm_y (不可变) + &mut h (可变) 不冲突
    w.ssm_out.matvec_add_into_slice(&ws.ssm_y, h);
}

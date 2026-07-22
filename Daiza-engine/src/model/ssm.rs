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
/// ★ 按 i 融合 Pass 1 + Pass 2(每行在 L1 中完成两 phase,减少 L2 重新加载):
///   对每个 i:
///     Phase 1: s[i,:] *= decay, 同时累加 kv_mem_i = sum_j s[i,j] * k[j]
///     Phase 2: di = (v[i] - kv_mem_i) * beta, s[i,:] += di * k, y[i] = sum_j s_new[i,j] * q[j]
///
///   原分离实现: Pass 1 遍历全部 128 行后 Pass 2 再遍历, S 矩阵 64KB > L1(32KB),
///   Pass 2 需从 L2 重新加载每行。融合后每行只从 L2 加载 1 次, L2 流量减半。
///
/// ★ P0-2 优化: 内层 j 循环手写 AVX2 (head_dim=128 = 16 × 8-wide FMA)
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

    // AVX2 融合路径
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        #[allow(unsafe_code)]
        unsafe {
            ssm_scan_fused_avx2(s, y, q_head, k_head, v_head, decay, beta, head_dim);
        }
        return;
    }
    // Fallback: 标量融合实现(非 x86_64 或无 AVX2)
    ssm_scan_fused_scalar(s, y, q_head, k_head, v_head, decay, beta, head_dim);
}

/// SSM output gate 标量实现 — 与 AVX2 版本逻辑一致
/// y = rmsnorm(y) * ssm_norm_weight * silu(z), per v_head
#[inline(never)]
pub(crate) fn ssm_output_gate_scalar(
    y: &mut [f32],
    z: &[f32],
    norm_w: &[f32],
    num_v_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    let inv_hd = 1.0 / head_dim as f32;
    for vh in 0..num_v_heads {
        let off = vh * head_dim;
        let mut ss = 0.0f32;
        for i in 0..head_dim {
            ss += y[off + i] * y[off + i];
        }
        let inv_rms = 1.0 / (ss * inv_hd + eps).sqrt();
        for i in 0..head_dim {
            y[off + i] = y[off + i] * inv_rms * norm_w[i] * z[off + i];
        }
    }
}

/// SSM output gate AVX2 向量化: sum-of-squares + 4-way mul
///
/// y 布局: [num_v_heads * head_dim] flat 连续
/// z 布局: 同 y
/// norm_w 布局: [head_dim],per v_head 共享
///
/// head_dim=128 = 16×8-wide,无尾处理。
/// sum-of-squares: AVX2 FMA + 水平 reduce (16 iter → 1 个 __m256 累加)
/// 4-way mul: y = y * inv_rms * norm_w * z (2 次 FMUL,或 1 次 FMA + 1 次 FMUL)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
#[inline]
pub(crate) unsafe fn ssm_output_gate_avx2(
    y: &mut [f32],
    z: &[f32],
    norm_w: &[f32],
    num_v_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    use core::arch::x86_64::*;
    debug_assert_eq!(head_dim % 8, 0);
    let inv_hd = 1.0 / head_dim as f32;
    let inv_hd_v = _mm256_set1_ps(inv_hd);
    let eps_v = _mm256_set1_ps(eps);
    for vh in 0..num_v_heads {
        let off = vh * head_dim;
        // Pass 1: sum of squares (16 iter FMA → 1 __m256)
        let mut ss_v = _mm256_setzero_ps();
        for i in (0..head_dim).step_by(8) {
            let y_v = _mm256_loadu_ps(y.as_ptr().add(off + i));
            ss_v = _mm256_fmadd_ps(y_v, y_v, ss_v);
        }
        // 水平 reduce: sum → rsqrt(sum*inv_hd + eps)
        let ss = horizontal_sum_ps(ss_v);
        let inv_rms = 1.0 / (ss * inv_hd + eps).sqrt();
        // Pass 2: y = y * inv_rms * norm_w * z
        let inv_rms_v = _mm256_set1_ps(inv_rms);
        for i in (0..head_dim).step_by(8) {
            let y_v = _mm256_loadu_ps(y.as_ptr().add(off + i));
            let nw_v = _mm256_loadu_ps(norm_w.as_ptr().add(i));
            let z_v = _mm256_loadu_ps(z.as_ptr().add(off + i));
            // y = (y * inv_rms) * norm_w * z,先 FMA: tmp = norm_w * inv_rms * y,
            // 再 mul z。但更精确的写法:y * inv_rms → FMA(norm_w, ·, 0) → * z
            let scaled = _mm256_mul_ps(y_v, inv_rms_v);
            let gated = _mm256_mul_ps(scaled, nw_v);
            let result = _mm256_mul_ps(gated, z_v);
            _mm256_storeu_ps(y.as_mut_ptr().add(off + i), result);
        }
    }
    // 抑制 unused warning(inv_hd_v/eps_v 在 rsqrt 走标量,但保留语义清晰)
    let _ = (inv_hd_v, eps_v);
}

/// 标量融合 fallback(与 AVX2 版本逻辑一致)
#[inline(never)]
fn ssm_scan_fused_scalar(
    s: &mut [f32],
    y: &mut [f32],
    q_head: &[f32],
    k_head: &[f32],
    v_head: &[f32],
    decay: f32,
    beta: f32,
    head_dim: usize,
) {
    for i in 0..head_dim {
        let srow = &mut s[i * head_dim..(i + 1) * head_dim];
        // Phase 1: s *= decay, 累加 kv_mem_i
        let mut kv_mem_i = 0.0f32;
        for j in 0..head_dim {
            let s_new = srow[j] * decay;
            srow[j] = s_new;
            kv_mem_i += s_new * k_head[j];
        }
        // Phase 2: di = (v[i] - kv_mem_i) * beta, s += di * k, 累加 y[i]
        let di = (v_head[i] - kv_mem_i) * beta;
        let mut acc = 0.0f32;
        for j in 0..head_dim {
            srow[j] += di * k_head[j];
            acc += srow[j] * q_head[j];
        }
        y[i] = acc;
    }
}

/// AVX2 向量化的融合 scan: 按 i 融合 Pass 1 + Pass 2
///
/// S 矩阵每行 512B (128×4B), 在 L1 中完成 decay + delta 两 phase,
/// 消除分离实现中 Pass 2 的 L2 重新加载 (S 64KB > L1 32KB)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
#[inline]
unsafe fn ssm_scan_fused_avx2(
    s: &mut [f32],
    y: &mut [f32],
    q_head: &[f32],
    k_head: &[f32],
    v_head: &[f32],
    decay: f32,
    beta: f32,
    head_dim: usize,
) {
    use core::arch::x86_64::*;
    let decay_v = _mm256_set1_ps(decay);
    for i in 0..head_dim {
        let srow = &mut s[i * head_dim..(i + 1) * head_dim];

        // Phase 1: s *= decay, 同时累加 kv_mem_i = sum_j s[i,j] * k[j]
        let mut kv_acc = _mm256_setzero_ps();
        for j in (0..head_dim).step_by(8) {
            let s_old = _mm256_loadu_ps(srow.as_ptr().add(j));
            let k = _mm256_loadu_ps(k_head.as_ptr().add(j));
            let s_new = _mm256_mul_ps(s_old, decay_v);
            _mm256_storeu_ps(srow.as_mut_ptr().add(j), s_new);
            kv_acc = _mm256_fmadd_ps(s_new, k, kv_acc);
        }
        let kv_mem_i = horizontal_sum_ps(kv_acc);

        // Phase 2: di = (v[i] - kv_mem_i) * beta, s += di * k, 同时累加 y[i]
        let di = (v_head[i] - kv_mem_i) * beta;
        let di_v = _mm256_set1_ps(di);
        let mut y_acc = _mm256_setzero_ps();
        for j in (0..head_dim).step_by(8) {
            let s_cur = _mm256_loadu_ps(srow.as_ptr().add(j));
            let k = _mm256_loadu_ps(k_head.as_ptr().add(j));
            let q = _mm256_loadu_ps(q_head.as_ptr().add(j));
            let s_new = _mm256_fmadd_ps(di_v, k, s_cur);
            _mm256_storeu_ps(srow.as_mut_ptr().add(j), s_new);
            y_acc = _mm256_fmadd_ps(s_new, q, y_acc);
        }
        y[i] = horizontal_sum_ps(y_acc);
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
        // ★ GQA 映射: llama.cpp 用 ggml_repeat (mod/tiled 布局), 非 repeat_interleave (div/grouped)
        //   conversion/qwen.py _LinearAttentionVReorderBase 将 V heads 重排为 tiled 布局
        //   k_head i 对应 v_head [i, i+num_k_heads, i+2*num_k_heads]
        let kh = vh % num_k_heads;
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
    // ★ AVX2 output gate: sum-of-squares + 4-way mul 向量化
    //   head_dim=128 = 16×8-wide,无尾处理
    //   原: 48 v_heads × 2 × 128 iter = 12288 标量 iter/block × 48 blocks = 589824 iter/token
    //   新: 48 v_heads × 16 iter = 768 SIMD iter/block × 48 = 36864 iter/token (~16x 减少)
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        #[allow(unsafe_code)]
        unsafe {
            ssm_output_gate_avx2(
                &mut ws.ssm_y[..num_v_heads * head_dim],
                &ws.ssm_z[..num_v_heads * head_dim],
                ssm_norm_w,
                num_v_heads,
                head_dim,
                SSM_EPS,
            );
        }
    } else {
        ssm_output_gate_scalar(
            &mut ws.ssm_y[..num_v_heads * head_dim],
            &ws.ssm_z[..num_v_heads * head_dim],
            ssm_norm_w,
            num_v_heads,
            head_dim,
            SSM_EPS,
        );
    }

    // 9. Output projection + residual: h += W_out @ y
    //    ★ &ws.ssm_y (不可变) + &mut h (可变) 不冲突
    w.ssm_out.matvec_add_into_slice(&ws.ssm_y, h);
}

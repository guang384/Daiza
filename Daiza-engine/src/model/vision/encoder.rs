//! ViT encoder: 27 层 Transformer 视觉编码器
//!
//! 流程 (对齐 llama.cpp qwen3vl.cpp + HF Qwen3VLVisionModel):
//! 1. 门控 patch embedding: `Conv2D(W, x) + Conv2D(W.1, x) + bias` → [n_patches, n_embd]
//! 2. 加 learned position embedding (broadcast add)
//! 3. 27 层 ViT block:
//!    ln1 → QKV proj → M-RoPE(Q,K) → bidirectional attention → out_proj → +residual
//!    → ln2 → ffn_up → GELU → ffn_down → +residual
//! 4. post_ln
//!
//! 注意:
//! - attention 是 bidirectional (is_causal=false), 所有 patch 互相可见, 无 causal mask
//! - M-RoPE 只应用在 Q 和 K 上, 不应用在 V 上
//! - FFN 是普通 MLP (Linear → GELU → Linear), 不是 SwiGLU
//! - LayerNorm 带 bias (ViT 风格, 非 RMSNorm)
//! - 所有 per-patch / per-head 循环并行化 (使用全局线程池), 14 核 ~10x 加速
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use crate::math::{layernorm_into, gelu_inplace};
use crate::math::softmax_inplace;
use crate::math::simd_exp::simd_available;
use crate::model::workspace::get_thread_pool;

use super::config::VisionConfig;
use super::weights::{VisionWeights, ViTBlockWeights};
use super::rope::{vision_mrope_cos_sin, apply_vision_rope};

// ---------------------------------------------------------------------------
// AVX2 attention kernel (head_dim=72 fast path)
// ---------------------------------------------------------------------------
// Q·K^T: hoist q_i 到 9 YMM 寄存器 (j 循环外), 2-acc unroll 填满 FMA pipeline.
// 原始: 每 j 循环 reload q_i 9 次 (编译器可能不 hoist 9 YMM).
// AVX2: q_i load 9 次/i (vs 9×2304 次/i), load 减少 2304×.

/// AVX2 水平求和 __m256 → f32
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
#[inline]
unsafe fn hsum_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// AVX2 attention kernel for single head (head_dim=72 fast path)
///
/// 计算: out[i, d] = sum_j softmax(scale * dot(q[i], k[j])) * v[j, d]
///
/// # Safety
/// - q, k, v: [n_patches, head_dim] head-major, len >= n_patches * head_dim
/// - out: [n_patches, head_dim], 已初始化 (累加 +=)
/// - scores: [n_patches] scratch buffer
/// - head_dim == 72
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn attention_head_avx2(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    scores: *mut f32,
    n_patches: usize,
    scale: f32,
) {
    const HD: usize = 72;
    for i in 0..n_patches {
        let q_i = q.add(i * HD);
        // ★ Hoist q_i 到 9 YMM 寄存器 (j 循环外, 减少 2304× q_i load)
        let q0 = _mm256_loadu_ps(q_i);
        let q1 = _mm256_loadu_ps(q_i.add(8));
        let q2 = _mm256_loadu_ps(q_i.add(16));
        let q3 = _mm256_loadu_ps(q_i.add(24));
        let q4 = _mm256_loadu_ps(q_i.add(32));
        let q5 = _mm256_loadu_ps(q_i.add(40));
        let q6 = _mm256_loadu_ps(q_i.add(48));
        let q7 = _mm256_loadu_ps(q_i.add(56));
        let q8 = _mm256_loadu_ps(q_i.add(64));

        // Q·K^T scores
        for j in 0..n_patches {
            let k_j = k.add(j * HD);
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            acc0 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(k_j), acc0);
            acc1 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(k_j.add(8)), acc1);
            acc0 = _mm256_fmadd_ps(q2, _mm256_loadu_ps(k_j.add(16)), acc0);
            acc1 = _mm256_fmadd_ps(q3, _mm256_loadu_ps(k_j.add(24)), acc1);
            acc0 = _mm256_fmadd_ps(q4, _mm256_loadu_ps(k_j.add(32)), acc0);
            acc1 = _mm256_fmadd_ps(q5, _mm256_loadu_ps(k_j.add(40)), acc1);
            acc0 = _mm256_fmadd_ps(q6, _mm256_loadu_ps(k_j.add(48)), acc0);
            acc1 = _mm256_fmadd_ps(q7, _mm256_loadu_ps(k_j.add(56)), acc1);
            acc0 = _mm256_fmadd_ps(q8, _mm256_loadu_ps(k_j.add(64)), acc0);
            *scores.add(j) = hsum_ps(_mm256_add_ps(acc0, acc1)) * scale;
        }

        // softmax (scalar fallback, n_patches=2304 element-wise)
        let scores_slice = std::slice::from_raw_parts_mut(scores, n_patches);
        softmax_inplace(scores_slice);

        // attn_out += scores @ V (4-acc unroll for store throughput)
        let out_i = out.add(i * HD);
        for j in 0..n_patches {
            let w = _mm256_set1_ps(*scores.add(j));
            let v_j = v.add(j * HD);
            // 72 = 4×16 + 8, 用 4-acc unroll 处理前 64, 单 acc 处理最后 8
            let o0 = _mm256_loadu_ps(out_i);
            let o1 = _mm256_loadu_ps(out_i.add(8));
            let o2 = _mm256_loadu_ps(out_i.add(16));
            let o3 = _mm256_loadu_ps(out_i.add(24));
            let o4 = _mm256_loadu_ps(out_i.add(32));
            let o5 = _mm256_loadu_ps(out_i.add(40));
            let o6 = _mm256_loadu_ps(out_i.add(48));
            let o7 = _mm256_loadu_ps(out_i.add(56));
            let o8 = _mm256_loadu_ps(out_i.add(64));
            let v0 = _mm256_loadu_ps(v_j);
            let v1 = _mm256_loadu_ps(v_j.add(8));
            let v2 = _mm256_loadu_ps(v_j.add(16));
            let v3 = _mm256_loadu_ps(v_j.add(24));
            let v4 = _mm256_loadu_ps(v_j.add(32));
            let v5 = _mm256_loadu_ps(v_j.add(40));
            let v6 = _mm256_loadu_ps(v_j.add(48));
            let v7 = _mm256_loadu_ps(v_j.add(56));
            let v8 = _mm256_loadu_ps(v_j.add(64));
            _mm256_storeu_ps(out_i, _mm256_fmadd_ps(w, v0, o0));
            _mm256_storeu_ps(out_i.add(8), _mm256_fmadd_ps(w, v1, o1));
            _mm256_storeu_ps(out_i.add(16), _mm256_fmadd_ps(w, v2, o2));
            _mm256_storeu_ps(out_i.add(24), _mm256_fmadd_ps(w, v3, o3));
            _mm256_storeu_ps(out_i.add(32), _mm256_fmadd_ps(w, v4, o4));
            _mm256_storeu_ps(out_i.add(40), _mm256_fmadd_ps(w, v5, o5));
            _mm256_storeu_ps(out_i.add(48), _mm256_fmadd_ps(w, v6, o6));
            _mm256_storeu_ps(out_i.add(56), _mm256_fmadd_ps(w, v7, o7));
            _mm256_storeu_ps(out_i.add(64), _mm256_fmadd_ps(w, v8, o8));
        }
    }
}

// ---------------------------------------------------------------------------
// Send+Sync raw pointer wrappers
// ---------------------------------------------------------------------------
// `scatter_wait` 要求 `F: Fn(usize) + Send + Sync`. 闭包需要捕获 raw pointer
// 跨线程访问 ctx 缓冲区, 但 `*const T` / `*mut T` 默认 !Send + !Sync.
// 用 newtype 包裹 + unsafe impl Send+Sync 让闭包满足约束.
//
// ★ Rust 2021 disjoint capture: 闭包若通过 `wrapper.0` 访问字段, 会捕获字段
//   (raw pointer) 而非 wrapper, 导致 Send+Sync impl 失效. 用方法 (`.as_ptr()`)
//   强制闭包捕获整个 wrapper.
//
// # Safety
// 调用方 (encode_image / forward_vit_block) 必须保证:
// 1. 持有 buffer 的 `&mut Vec` 借用直到 scatter_wait 返回 (scatter_wait 阻塞)
// 2. 不同 worker 写入的内存范围不重叠 (per-patch / per-head 切分)
// 3. 只读 buffer (CPtr) 跨 worker 共享安全

#[derive(Copy, Clone)]
struct CPtr<T>(*const T);
unsafe impl<T> Send for CPtr<T> {}
unsafe impl<T> Sync for CPtr<T> {}
impl<T> CPtr<T> {
    #[inline(always)]
    fn as_ptr(self) -> *const T { self.0 }
}

#[derive(Copy, Clone)]
struct MPtr<T>(*mut T);
unsafe impl<T> Send for MPtr<T> {}
unsafe impl<T> Sync for MPtr<T> {}
impl<T> MPtr<T> {
    #[inline(always)]
    fn as_ptr(self) -> *mut T { self.0 }
}

/// 并行执行 per-patch 任务 (n_patches 个独立工作单元)
fn par_for_patches<F>(n_patches: usize, f: F)
where
    F: Fn(usize) + Send + Sync,
{
    if let Some(pool) = get_thread_pool() {
        pool.scatter_wait(n_patches, f);
    } else {
        for i in 0..n_patches {
            f(i);
        }
    }
}

/// 并行执行 per-head 任务 (n_heads 个独立工作单元)
fn par_for_heads<F>(n_heads: usize, f: F)
where
    F: Fn(usize) + Send + Sync,
{
    if let Some(pool) = get_thread_pool() {
        pool.scatter_wait(n_heads, f);
    } else {
        for i in 0..n_heads {
            f(i);
        }
    }
}

/// ViT 前向上下文: 持有工作缓冲区, 跨多次 encode_image 调用复用
pub struct ViTContext {
    pub hidden: Vec<f32>,
    pub ln1_out: Vec<f32>,
    pub qkv: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_proj: Vec<f32>,
    pub ln2_out: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
    /// per-head 独立 scores buffer: [n_heads * n_patches]
    /// (并行 per-head attention 时, 每 head 写入自己的 [h*n_patches..(h+1)*n_patches])
    pub attn_scores: Vec<f32>,
    pub q_head: Vec<f32>,
    pub k_head: Vec<f32>,
    pub v_head: Vec<f32>,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl ViTContext {
    pub fn new(cfg: &VisionConfig) -> Self {
        let n_patches = cfg.n_patches;
        let n_embd = cfg.embedding_length;
        let ffn_dim = cfg.feed_forward_length;
        let n_heads = cfg.head_count;
        let head_dim = cfg.head_dim;

        let (cos, sin) = vision_mrope_cos_sin(cfg);

        Self {
            hidden: vec![0.0; n_patches * n_embd],
            ln1_out: vec![0.0; n_patches * n_embd],
            qkv: vec![0.0; n_patches * 3 * n_embd],
            attn_out: vec![0.0; n_patches * n_embd],
            attn_proj: vec![0.0; n_patches * n_embd],
            ln2_out: vec![0.0; n_patches * n_embd],
            ffn_up: vec![0.0; n_patches * ffn_dim],
            ffn_down: vec![0.0; n_patches * n_embd],
            // ★ per-head 独立 buffer (n_heads * n_patches), 避免数据竞争
            attn_scores: vec![0.0; n_heads * n_patches],
            // head-major: [n_heads * n_patches * head_dim]
            q_head: vec![0.0; n_heads * n_patches * head_dim],
            k_head: vec![0.0; n_heads * n_patches * head_dim],
            v_head: vec![0.0; n_heads * n_patches * head_dim],
            cos,
            sin,
        }
    }
}

/// 完整 ViT 前向传播
pub fn encode_image(
    weights: &VisionWeights,
    cfg: &VisionConfig,
    ctx: &mut ViTContext,
    patches: &[f32],
) -> crate::Result<()> {
    let n_patches = cfg.n_patches;
    let n_embd = cfg.embedding_length;
    let patch_dim = 3 * cfg.patch_size * cfg.patch_size;
    debug_assert_eq!(patches.len(), n_patches * patch_dim);

    // ───── 1. 门控 patch embedding ───── (batched matmul + per-patch bias)
    // ★ batched: 每行权重只读一次, 对所有 2304 patch 做 dot product
    //   per-patch 时每 patch 读一次完整 W (768*1152*4=3.4MB), 2304 patch = 7.6GB DRAM 读
    //   batched 时 W 只读一次 (3.4MB), X=patches (7.5MB) 放 L3 复用
    let t0 = std::time::Instant::now();
    let patch_embd_w = &weights.patch_embd_w;
    let patch_embd_w1 = &weights.patch_embd_w1;
    let patch_embd_b = weights.patch_embd_b.as_slice();

    // hidden = W0 @ patches + W1 @ patches (batched)
    patch_embd_w.matmat_into_slice(patches, &mut ctx.hidden, n_patches);
    patch_embd_w1.matmat_add_into_slice(patches, &mut ctx.hidden, n_patches);

    // + bias (per-patch 并行, AVX2 saxpy; n_embd=1152=8×144)
    let hidden_ptr = MPtr(ctx.hidden.as_mut_ptr());
    let pb_ptr = CPtr(patch_embd_b.as_ptr());
    par_for_patches(n_patches, move |p| {
        unsafe {
            let h = std::slice::from_raw_parts_mut(hidden_ptr.as_ptr().add(p * n_embd), n_embd);
            let b = std::slice::from_raw_parts(pb_ptr.as_ptr(), n_embd);
            crate::math::simd_exp::saxpy_avx2(1.0, b, h, n_embd);
        }
    });
    let t_patch = t0.elapsed();

    // ───── 2. 加 learned position embedding ───── (per-patch 并行, AVX2 saxpy)
    let pos_embd = weights.position_embd.as_slice();
    let hidden_ptr = MPtr(ctx.hidden.as_mut_ptr());
    let pos_ptr = CPtr(pos_embd.as_ptr());
    par_for_patches(n_patches, move |p| {
        unsafe {
            let h = std::slice::from_raw_parts_mut(hidden_ptr.as_ptr().add(p * n_embd), n_embd);
            let pos = std::slice::from_raw_parts(pos_ptr.as_ptr().add(p * n_embd), n_embd);
            crate::math::simd_exp::saxpy_avx2(1.0, pos, h, n_embd);
        }
    });

    // ───── 3. 27 层 ViT block ─────
    let t_blocks = std::time::Instant::now();
    let profile = std::env::var("DAIZA_VISION_PROFILE").is_ok();
    let mut p_ln1 = std::time::Duration::ZERO;
    let mut p_qkv = std::time::Duration::ZERO;
    let mut p_attn = std::time::Duration::ZERO;
    let mut p_outproj = std::time::Duration::ZERO;
    let mut p_ln2 = std::time::Duration::ZERO;
    let mut p_ffn = std::time::Duration::ZERO;
    for blk_idx in 0..cfg.block_count {
        let blk = &weights.blocks[blk_idx];
        forward_vit_block(blk, cfg, ctx, profile, &mut p_ln1, &mut p_qkv, &mut p_attn, &mut p_outproj, &mut p_ln2, &mut p_ffn)?;
    }
    let t_blocks_elapsed = t_blocks.elapsed();
    if profile {
        eprintln!("[vision-blocks] ln1={:.1}ms qkv={:.1}ms attn={:.1}ms outproj={:.1}ms ln2={:.1}ms ffn={:.1}ms",
            p_ln1.as_secs_f64()*1000.0, p_qkv.as_secs_f64()*1000.0,
            p_attn.as_secs_f64()*1000.0, p_outproj.as_secs_f64()*1000.0,
            p_ln2.as_secs_f64()*1000.0, p_ffn.as_secs_f64()*1000.0);
    }

    // ───── 4. post_ln ───── (并行 per-patch, 与 block 内 LN1/LN2 一致)
    let t_post = std::time::Instant::now();
    let post_w = weights.post_ln_w.as_slice();
    let post_b = weights.post_ln_b.as_slice();
    let eps = cfg.layer_norm_eps;
    let hidden_ptr = CPtr(ctx.hidden.as_ptr());
    let ln1_out_ptr = MPtr(ctx.ln1_out.as_mut_ptr());
    let postw_ptr = CPtr(post_w.as_ptr());
    let postb_ptr = CPtr(post_b.as_ptr());
    par_for_patches(n_patches, move |p| {
        unsafe {
            let src = std::slice::from_raw_parts(hidden_ptr.as_ptr().add(p * n_embd), n_embd);
            let dst = std::slice::from_raw_parts_mut(ln1_out_ptr.as_ptr().add(p * n_embd), n_embd);
            let w = std::slice::from_raw_parts(postw_ptr.as_ptr(), n_embd);
            let b = std::slice::from_raw_parts(postb_ptr.as_ptr(), n_embd);
            layernorm_into(src, dst, w, b, eps);
        }
    });
    std::mem::swap(&mut ctx.hidden, &mut ctx.ln1_out);
    let t_post_elapsed = t_post.elapsed();

    eprintln!("[vision] encode_image: patch_embd={:.1}ms blocks({})={:.1}ms post_ln={:.1}ms total={:.1}ms",
        t_patch.as_secs_f64() * 1000.0,
        cfg.block_count,
        t_blocks_elapsed.as_secs_f64() * 1000.0,
        t_post_elapsed.as_secs_f64() * 1000.0,
        t_patch.as_secs_f64() * 1000.0 + t_blocks_elapsed.as_secs_f64() * 1000.0 + t_post_elapsed.as_secs_f64() * 1000.0,
    );

    Ok(())
}

/// 单个 ViT block 前向 (bidirectional attention + MLP, 残差连接)
/// 所有 per-patch / per-head 循环并行化
fn forward_vit_block(
    blk: &ViTBlockWeights,
    cfg: &VisionConfig,
    ctx: &mut ViTContext,
    profile: bool,
    p_ln1: &mut std::time::Duration,
    p_qkv: &mut std::time::Duration,
    p_attn: &mut std::time::Duration,
    p_outproj: &mut std::time::Duration,
    p_ln2: &mut std::time::Duration,
    p_ffn: &mut std::time::Duration,
) -> crate::Result<()> {
    let n_patches = cfg.n_patches;
    let n_embd = cfg.embedding_length;
    let n_heads = cfg.head_count;
    let head_dim = cfg.head_dim;
    let ffn_dim = cfg.feed_forward_length;
    let eps = cfg.layer_norm_eps;

    // ───── 3a. LN1 ───── (并行 per-patch)
    let t = if profile { Some(std::time::Instant::now()) } else { None };
    let ln1_w = blk.ln1_w.as_slice();
    let ln1_b = blk.ln1_b.as_slice();
    let hidden_ptr = CPtr(ctx.hidden.as_ptr());
    let ln1_out_ptr = MPtr(ctx.ln1_out.as_mut_ptr());
    let ln1w_ptr = CPtr(ln1_w.as_ptr());
    let ln1b_ptr = CPtr(ln1_b.as_ptr());
    par_for_patches(n_patches, move |p| {
        unsafe {
            let src = std::slice::from_raw_parts(hidden_ptr.as_ptr().add(p * n_embd), n_embd);
            let dst = std::slice::from_raw_parts_mut(ln1_out_ptr.as_ptr().add(p * n_embd), n_embd);
            let w = std::slice::from_raw_parts(ln1w_ptr.as_ptr(), n_embd);
            let b = std::slice::from_raw_parts(ln1b_ptr.as_ptr(), n_embd);
            layernorm_into(src, dst, w, b, eps);
        }
    });
    if let Some(t) = t { *p_ln1 += t.elapsed(); }

    // ───── 3b. QKV projection ───── (batched matmul + per-patch bias)
    // ★ batched: X=ln1_out (10.6MB) 放 L3, W_qkv (15.9MB) 只读一次
    //   per-patch 时每 patch 读一次 W_qkv, 2304 patch = 36.6GB DRAM 读
    let t = if profile { Some(std::time::Instant::now()) } else { None };
    let qkv_w = &blk.attn_qkv_w;
    let qkv_b = blk.attn_qkv_b.as_slice();
    debug_assert_eq!(qkv_w.rows, 3 * n_embd);
    debug_assert_eq!(qkv_w.cols, n_embd);

    // qkv = W_qkv @ ln1_out (batched, [n_patches, 3*n_embd])
    qkv_w.matmat_into_slice(&ctx.ln1_out, &mut ctx.qkv, n_patches);

    // + bias (per-patch 并行, AVX2 saxpy; qkv_dim=3456=8×432)
    let qkv_ptr = MPtr(ctx.qkv.as_mut_ptr());
    let qkv_b_ptr = CPtr(qkv_b.as_ptr());
    let qkv_dim = 3 * n_embd;
    par_for_patches(n_patches, move |p| {
        unsafe {
            let y = std::slice::from_raw_parts_mut(qkv_ptr.as_ptr().add(p * qkv_dim), qkv_dim);
            let b = std::slice::from_raw_parts(qkv_b_ptr.as_ptr(), qkv_dim);
            crate::math::simd_exp::saxpy_avx2(1.0, b, y, qkv_dim);
        }
    });
    if let Some(t) = t { *p_qkv += t.elapsed(); }

    // ───── 3c. 拆分 Q/K/V + M-RoPE ───── (并行 per-head)
    let qkv_ptr = CPtr(ctx.qkv.as_ptr());
    let q_head_ptr = MPtr(ctx.q_head.as_mut_ptr());
    let k_head_ptr = MPtr(ctx.k_head.as_mut_ptr());
    let v_head_ptr = MPtr(ctx.v_head.as_mut_ptr());
    let cos_ptr = CPtr(ctx.cos.as_ptr());
    let sin_ptr = CPtr(ctx.sin.as_ptr());
    par_for_heads(n_heads, move |h| {
        unsafe {
            for p in 0..n_patches {
                let q_src = std::slice::from_raw_parts(
                    qkv_ptr.as_ptr().add(p * 3 * n_embd + h * head_dim), head_dim);
                let k_src = std::slice::from_raw_parts(
                    qkv_ptr.as_ptr().add(p * 3 * n_embd + n_embd + h * head_dim), head_dim);
                let v_src = std::slice::from_raw_parts(
                    qkv_ptr.as_ptr().add(p * 3 * n_embd + 2 * n_embd + h * head_dim), head_dim);
                let dst_off = (h * n_patches + p) * head_dim;
                std::ptr::copy_nonoverlapping(q_src.as_ptr(), q_head_ptr.as_ptr().add(dst_off), head_dim);
                std::ptr::copy_nonoverlapping(k_src.as_ptr(), k_head_ptr.as_ptr().add(dst_off), head_dim);
                std::ptr::copy_nonoverlapping(v_src.as_ptr(), v_head_ptr.as_ptr().add(dst_off), head_dim);

                // M-RoPE on Q and K
                let cos_p = std::slice::from_raw_parts(cos_ptr.as_ptr().add(p * head_dim), head_dim);
                let sin_p = std::slice::from_raw_parts(sin_ptr.as_ptr().add(p * head_dim), head_dim);
                {
                    let q = std::slice::from_raw_parts_mut(q_head_ptr.as_ptr().add(dst_off), head_dim);
                    apply_vision_rope(q, cos_p, sin_p);
                }
                {
                    let k = std::slice::from_raw_parts_mut(k_head_ptr.as_ptr().add(dst_off), head_dim);
                    apply_vision_rope(k, cos_p, sin_p);
                }
            }
        }
    });

    // ───── 3d. Bidirectional attention ───── (并行 per-head)
    let t = if profile { Some(std::time::Instant::now()) } else { None };
    let scale = 1.0 / (head_dim as f32).sqrt();
    ctx.attn_out.fill(0.0);

    let q_head_ptr = CPtr(ctx.q_head.as_ptr());
    let k_head_ptr = CPtr(ctx.k_head.as_ptr());
    let v_head_ptr = CPtr(ctx.v_head.as_ptr());
    let attn_out_ptr = MPtr(ctx.attn_out.as_mut_ptr());
    // ★ per-head 独立 scores buffer: attn_scores[h*n_patches..(h+1)*n_patches]
    let scores_ptr = MPtr(ctx.attn_scores.as_mut_ptr());
    let use_avx2_attn = simd_available() && head_dim == 72;
    par_for_heads(n_heads, move |h| {
        unsafe {
            let scores_h = scores_ptr.as_ptr().add(h * n_patches);
            let q_base = h * n_patches * head_dim;
            let q_h = q_head_ptr.as_ptr().add(q_base);
            let k_h = k_head_ptr.as_ptr().add(q_base);
            let v_h = v_head_ptr.as_ptr().add(q_base);
            let out_h = attn_out_ptr.as_ptr().add(q_base);

            if use_avx2_attn {
                // ★ AVX2 fast path: hoist q_i 到 9 YMM, 2-acc unroll Q·K^T, 9-acc axpy
                attention_head_avx2(q_h, k_h, v_h, out_h, scores_h, n_patches, scale);
            } else {
                // scalar fallback (head_dim != 72 或 AVX2 不可用)
                let scores = std::slice::from_raw_parts_mut(scores_h, n_patches);
                for i in 0..n_patches {
                    let q_i = std::slice::from_raw_parts(q_h.add(i * head_dim), head_dim);
                    for j in 0..n_patches {
                        let k_j = std::slice::from_raw_parts(k_h.add(j * head_dim), head_dim);
                        let mut s = 0.0f32;
                        for d in 0..head_dim {
                            s += q_i[d] * k_j[d];
                        }
                        scores[j] = s * scale;
                    }
                    softmax_inplace(&mut scores[..n_patches]);
                    let out_i_base = (h * n_patches + i) * head_dim;
                    for j in 0..n_patches {
                        let w = scores[j];
                        let v_j = std::slice::from_raw_parts(v_h.add(j * head_dim), head_dim);
                        let out_i = std::slice::from_raw_parts_mut(attn_out_ptr.as_ptr().add(out_i_base), head_dim);
                        for d in 0..head_dim {
                            out_i[d] += w * v_j[d];
                        }
                    }
                }
            }
        }
    });
    if let Some(t) = t { *p_attn += t.elapsed(); }

    // ───── 3e. Output projection ───── (并行 per-patch)
    let t = if profile { Some(std::time::Instant::now()) } else { None };
    // 重组 attn_out: head-major → patch-major (per-patch 并行)
    let attn_out_ptr = CPtr(ctx.attn_out.as_ptr());
    let attn_proj_ptr = MPtr(ctx.attn_proj.as_mut_ptr());
    par_for_patches(n_patches, move |p| {
        unsafe {
            for h in 0..n_heads {
                let src_off = (h * n_patches + p) * head_dim;
                let dst_off = p * n_embd + h * head_dim;
                std::ptr::copy_nonoverlapping(
                    attn_out_ptr.as_ptr().add(src_off),
                    attn_proj_ptr.as_ptr().add(dst_off),
                    head_dim);
            }
        }
    });

    // attn_proj = W_out @ attn_out (patch-major) + bias (batched matmul)
    // ★ batched: W_out (5.3MB) 只读一次, X=attn_proj (10.6MB) 放 L3
    let out_w = &blk.attn_out_w;
    let out_b = blk.attn_out_b.as_slice();
    debug_assert_eq!(out_w.rows, n_embd);
    debug_assert_eq!(out_w.cols, n_embd);

    out_w.matmat_into_slice(&ctx.attn_proj, &mut ctx.attn_out, n_patches);

    // ★ V-3: 残差 + bias 融合 (hidden += attn_out + bias), 单 par_for_patches + AVX2 saxpy
    //   原实现: bias 写 attn_out (1 pass W attn_out) + 残差读 attn_out (1 pass R) = 2 pass over attn_out
    //   融合后: 只读 attn_out 一次, bias 从 L1 复用, 省一次 10.6MB attn_out 写 pass
    let hidden_ptr = MPtr(ctx.hidden.as_mut_ptr());
    let attn_out_ptr = CPtr(ctx.attn_out.as_ptr());
    let out_b_ptr = CPtr(out_b.as_ptr());
    par_for_patches(n_patches, move |p| {
        unsafe {
            let h = std::slice::from_raw_parts_mut(hidden_ptr.as_ptr().add(p * n_embd), n_embd);
            let a = std::slice::from_raw_parts(attn_out_ptr.as_ptr().add(p * n_embd), n_embd);
            let b = std::slice::from_raw_parts(out_b_ptr.as_ptr(), n_embd);
            crate::math::simd_exp::saxpy_avx2(1.0, a, h, n_embd);
            crate::math::simd_exp::saxpy_avx2(1.0, b, h, n_embd);
        }
    });
    if let Some(t) = t { *p_outproj += t.elapsed(); }

    // ───── 3g. LN2 ───── (并行 per-patch)
    let t = if profile { Some(std::time::Instant::now()) } else { None };
    let ln2_w = blk.ln2_w.as_slice();
    let ln2_b = blk.ln2_b.as_slice();
    let hidden_ptr = CPtr(ctx.hidden.as_ptr());
    let ln2_out_ptr = MPtr(ctx.ln2_out.as_mut_ptr());
    let ln2w_ptr = CPtr(ln2_w.as_ptr());
    let ln2b_ptr = CPtr(ln2_b.as_ptr());
    par_for_patches(n_patches, move |p| {
        unsafe {
            let src = std::slice::from_raw_parts(hidden_ptr.as_ptr().add(p * n_embd), n_embd);
            let dst = std::slice::from_raw_parts_mut(ln2_out_ptr.as_ptr().add(p * n_embd), n_embd);
            let w = std::slice::from_raw_parts(ln2w_ptr.as_ptr(), n_embd);
            let b = std::slice::from_raw_parts(ln2b_ptr.as_ptr(), n_embd);
            layernorm_into(src, dst, w, b, eps);
        }
    });
    if let Some(t) = t { *p_ln2 += t.elapsed(); }

    // ───── 3h+3i. FFN (up + GELU + down) ───── (batched matmul + per-patch GELU)
    // ★ batched: FFN up W_up (19.8MB) 只读一次, X=ln2_out (10.6MB) 放 L3
    //   FFN down W_down (19.8MB) 只读一次, X=ffn_up (39.7MB) 超过 L3 但仍比 per-patch 好
    //   per-patch 时每 patch 读 W_up+W_down (39.6MB), 2304 patch = 91GB DRAM 读
    //   batched 时 W_up+W_down 各读一次 (39.6MB), X 从 L3 或 DRAM 读
    let t = if profile { Some(std::time::Instant::now()) } else { None };
    let ffn_up_w = &blk.ffn_up_w;
    let ffn_up_b = blk.ffn_up_b.as_slice();
    debug_assert_eq!(ffn_up_w.rows, ffn_dim);
    debug_assert_eq!(ffn_up_w.cols, n_embd);
    let ffn_down_w = &blk.ffn_down_w;
    let ffn_down_b = blk.ffn_down_b.as_slice();
    debug_assert_eq!(ffn_down_w.rows, n_embd);
    debug_assert_eq!(ffn_down_w.cols, ffn_dim);

    // ffn_up = W_up @ ln2_out (batched, [n_patches, ffn_dim])
    ffn_up_w.matmat_into_slice(&ctx.ln2_out, &mut ctx.ffn_up, n_patches);

    // + bias + GELU (per-patch 并行, AVX2 saxpy + in-place GELU; ffn_dim=4304=8×538)
    let ffn_up_ptr = MPtr(ctx.ffn_up.as_mut_ptr());
    let ffn_up_b_ptr = CPtr(ffn_up_b.as_ptr());
    par_for_patches(n_patches, move |p| {
        unsafe {
            let y = std::slice::from_raw_parts_mut(ffn_up_ptr.as_ptr().add(p * ffn_dim), ffn_dim);
            let b = std::slice::from_raw_parts(ffn_up_b_ptr.as_ptr(), ffn_dim);
            crate::math::simd_exp::saxpy_avx2(1.0, b, y, ffn_dim);
            // ★ V-6: in-place GELU (无需 tmp buffer, 1 pass 替代 2 pass)
            gelu_inplace(y);
        }
    });

    // ffn_down = W_down @ ffn_up (batched, [n_patches, n_embd])
    ffn_down_w.matmat_into_slice(&ctx.ffn_up, &mut ctx.ffn_down, n_patches);

    // ★ V-3: 残差 + bias 融合 (hidden += ffn_down + bias), 单 par_for_patches + AVX2 saxpy
    let hidden_ptr = MPtr(ctx.hidden.as_mut_ptr());
    let ffn_down_ptr = CPtr(ctx.ffn_down.as_ptr());
    let ffn_down_b_ptr = CPtr(ffn_down_b.as_ptr());
    par_for_patches(n_patches, move |p| {
        unsafe {
            let h = std::slice::from_raw_parts_mut(hidden_ptr.as_ptr().add(p * n_embd), n_embd);
            let f = std::slice::from_raw_parts(ffn_down_ptr.as_ptr().add(p * n_embd), n_embd);
            let b = std::slice::from_raw_parts(ffn_down_b_ptr.as_ptr(), n_embd);
            crate::math::simd_exp::saxpy_avx2(1.0, f, h, n_embd);
            crate::math::simd_exp::saxpy_avx2(1.0, b, h, n_embd);
        }
    });
    if let Some(t) = t { *p_ffn += t.elapsed(); }

    Ok(())
}

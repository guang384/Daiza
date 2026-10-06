//! qwen3vl_merger 投影器: ViT 输出 [n_patches, n_embd] → text hidden [n_patches_merged, projection_dim]
//!
//! 流程 (对齐 llama.cpp qwen3vl.cpp `build_vision_projector` + HF Qwen3VLPatchMerger):
//!
//! 1. **Spatial merge reshape**: 把 2×2 空间相邻 patch 拼接
//!    - 输入: `vit_out` [n_patches=2304, n_embd=1152] 行优先 (patch-major)
//!    - 输入 patch 的二维索引 (py, px), n_per_side=48
//!    - 合并后: n_patches_merged = 2304 / 4 = 576, merged_hidden = 4 * 1152 = 4608
//!    - 重排: merged[m] = concat(vit_out[py_0, px_0], vit_out[py_0, px_1],
//!      vit_out[py_1, px_0], vit_out[py_1, px_1])
//!      其中 (py_0, py_1) = (2*my, 2*my+1), (px_0, px_1) = (2*mx, 2*mx+1)
//!    - 输出: `merged` [n_patches_merged=576, merged_hidden=4608] 行优先
//!
//! 2. **mm.0**: Linear(4608 → 4608) + bias
//! 3. **GELU** (clip.use_gelu=true)
//! 4. **mm.2**: Linear(4608 → projection_dim=5120) + bias
//!
//! 输出 576 个 vision embeddings, 每个维度 = text model hidden dim (5120)
//! 这些 embeddings 替换 text model prefill 中的 image_token 位置的 token_embd

use crate::math::gelu_erf_inplace;

use super::config::VisionConfig;
use super::weights::VisionWeights;

/// 投影器工作缓冲区 (跨多次 project_vision 调用复用)
pub struct ProjectorContext {
    /// n_patches_merged * merged_hidden, spatial merge 输出
    pub merged: Vec<f32>,
    /// n_patches_merged * merged_hidden, mm.0 输出 (GELU 后)
    pub mm_0_out: Vec<f32>,
    /// n_patches_merged * projection_dim, mm.2 输出 (最终 vision embeddings)
    pub projected: Vec<f32>,
}

impl ProjectorContext {
    pub fn new(cfg: &VisionConfig) -> Self {
        let n_merged = cfg.n_patches_merged();
        let merged_hidden = cfg.merged_hidden();
        let proj_dim = cfg.projection_dim;
        Self {
            merged: vec![0.0; n_merged * merged_hidden],
            mm_0_out: vec![0.0; n_merged * merged_hidden],
            projected: vec![0.0; n_merged * proj_dim],
        }
    }
}

/// 投影 ViT 输出到 text embedding 空间
///
/// 输入:
/// - `vit_out`: [n_patches, n_embd] 行优先 (来自 encoder.rs 的 ctx.hidden)
/// - `weights`: 完整 mmproj 权重
/// - `cfg`: VisionConfig
/// - `pctx`: 投影器工作缓冲
///
/// 输出: 写入 `pctx.projected` [n_patches_merged, projection_dim] 行优先
pub fn project_vision(
    vit_out: &[f32],
    weights: &VisionWeights,
    cfg: &VisionConfig,
    pctx: &mut ProjectorContext,
) -> crate::Result<()> {
    let n_patches = cfg.n_patches;
    let n_embd = cfg.embedding_length;
    let n_per_side = cfg.n_patches_per_side;
    let merge = cfg.spatial_merge_size; // 2
    let n_per_side_merged = n_per_side / merge; // 24
    let n_merged = cfg.n_patches_merged(); // 576
    let merged_hidden = cfg.merged_hidden(); // 4608
    let proj_dim = cfg.projection_dim; // 5120

    debug_assert_eq!(vit_out.len(), n_patches * n_embd);
    debug_assert_eq!(pctx.merged.len(), n_merged * merged_hidden);
    debug_assert_eq!(pctx.mm_0_out.len(), n_merged * merged_hidden);
    debug_assert_eq!(pctx.projected.len(), n_merged * proj_dim);

    // ───── 1. Spatial merge reshape ─────
    // merged[m * merged_hidden + k * n_embd + d] = vit_out[patch_idx * n_embd + d]
    //   其中 m = my * n_per_side_merged + mx
    //         k ∈ 0..4: 0=(py_0,px_0), 1=(py_0,px_1), 2=(py_1,px_0), 3=(py_1,px_1)
    //         patch_idx = (2*my + (k/2)) * n_per_side + (2*mx + (k%2))
    for my in 0..n_per_side_merged {
        for mx in 0..n_per_side_merged {
            let m = my * n_per_side_merged + mx;
            let merged_off = m * merged_hidden;
            for k in 0..4 {
                let py = 2 * my + k / 2;
                let px = 2 * mx + k % 2;
                let patch_idx = py * n_per_side + px;
                let src_off = patch_idx * n_embd;
                let dst_off = merged_off + k * n_embd;
                pctx.merged[dst_off..dst_off + n_embd]
                    .copy_from_slice(&vit_out[src_off..src_off + n_embd]);
            }
        }
    }

    // ───── 2. mm.0: Linear(4608 → 4608) + bias, batched matmul ─────
    // ★ batched: W_mm0 (84.7MB) 只读一次, 576 patch per-patch 时读 48.8GB
    let mm_0_w = &weights.mm_0_w;
    let mm_0_b = weights.mm_0_b.as_slice();
    debug_assert_eq!(mm_0_w.rows, merged_hidden);
    debug_assert_eq!(mm_0_w.cols, merged_hidden);
    debug_assert_eq!(mm_0_b.len(), merged_hidden);

    mm_0_w.matmat_into_slice(&pctx.merged, &mut pctx.mm_0_out, n_merged);

    // + bias (per-patch 并行, 仅 element-wise add)
    // ★ P2: 内层标量循环改 AVX2 saxpy (y = 1.0*bias + y)
    //   merged_hidden=4608, 576 patch × 576c 标量 ≈ 33Kc → AVX2 72c × 576 = 41Kc
    //   注: saxpy_avx2 要求 len 为 8 的倍数 (merged_hidden=4608=8×576 ✓)
    debug_assert!(merged_hidden % 8 == 0, "merged_hidden must be 8-aligned for saxpy_avx2");
    for m in 0..n_merged {
        let y = &mut pctx.mm_0_out[m * merged_hidden..(m + 1) * merged_hidden];
        crate::math::simd_exp::saxpy_avx2(1.0, mm_0_b, y, merged_hidden);
    }

    // ───── 3. GELU (in-place, 精确 erf 版本) ─────
    // ★ Qwen3-VL PatchMerger 用 nn.GELU(approximate='none') 即精确 erf 版本,
    //   区别于 ViT MLP 的 tanh 近似。混用会导致 vision embedding 数值偏差,
    //   进而影响模型对图像内容的判别 (虽量级小, 但 PatchMerger 是 vision→text
    //   的关键投影层, 数值精度直接影响 text model 对图像 token 的解读)。
    for m in 0..n_merged {
        gelu_erf_inplace(&mut pctx.mm_0_out[m * merged_hidden..(m + 1) * merged_hidden]);
    }

    // ───── 4. mm.2: Linear(4608 → 5120) + bias, batched matmul ─────
    // ★ batched: W_mm2 (94.4MB) 只读一次, 576 patch per-patch 时读 54.4GB
    let mm_2_w = &weights.mm_2_w;
    let mm_2_b = weights.mm_2_b.as_slice();
    debug_assert_eq!(mm_2_w.rows, proj_dim);
    debug_assert_eq!(mm_2_w.cols, merged_hidden);
    debug_assert_eq!(mm_2_b.len(), proj_dim);

    mm_2_w.matmat_into_slice(&pctx.mm_0_out, &mut pctx.projected, n_merged);

    // + bias (per-patch)
    // ★ P2: 内层标量循环改 AVX2 saxpy (proj_dim=5120=8×640 ✓)
    debug_assert!(proj_dim % 8 == 0, "proj_dim must be 8-aligned for saxpy_avx2");
    for m in 0..n_merged {
        let y = &mut pctx.projected[m * proj_dim..(m + 1) * proj_dim];
        crate::math::simd_exp::saxpy_avx2(1.0, mm_2_b, y, proj_dim);
    }

    Ok(())
}

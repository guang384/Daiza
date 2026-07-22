//! 图像预处理: 加载 → resize → normalize → patchify
//!
//! 流程 (对齐 llama.cpp qwen3vl + HF Qwen3VLVisionPatchEmbed):
//! 1. `image::open` 加载图像 (PNG/JPEG)
//! 2. resize 到 image_size × image_size (768×768) using Lanczos3
//! 3. 转 RGB, normalize: `(x/255 - mean) / std`
//! 4. patchify: 按 patch_size (16) 切块, 每个 patch flatten 为 [in_ch=3, kh=16, kw=16] = 768 dim
//!    顺序与 Conv2D weight [out_ch, in_ch, kh, kw] 行优先一致

use image::imageops::FilterType;
use crate::BonsaiError;

use super::config::VisionConfig;

/// 预处理图像: 返回 [n_patches, patch_dim] 的 patches 张量 (F32)
///
/// patch_dim = in_channels * patch_size * patch_size = 3 * 16 * 16 = 768
/// n_patches = n_patches_per_side^2 = (image_size / patch_size)^2 = 2304
pub fn preprocess_image(path: &std::path::Path, cfg: &VisionConfig) -> crate::Result<Vec<f32>> {
    let img = image::open(path)
        .map_err(|e| BonsaiError::Io(format!("open image {}: {e}", path.display())))?;

    eprintln!("[vision] image loaded: {} (original {}x{})",
        path.display(), img.width(), img.height());

    // Resize 到 image_size × image_size
    let resized = img.resize_exact(
        cfg.image_size as u32,
        cfg.image_size as u32,
        FilterType::Lanczos3,
    );
    let rgb = resized.to_rgb8();

    // 转 normalized F32 HWC: [H, W, 3]
    let h = cfg.image_size;
    let w = cfg.image_size;
    let mut normalized: Vec<f32> = Vec::with_capacity(h * w * 3);
    for pixel in rgb.pixels() {
        for c in 0..3 {
            let v = pixel.0[c] as f32 / 255.0;
            let n = (v - cfg.image_mean[c]) / cfg.image_std[c];
            normalized.push(n);
        }
    }

    // patchify: 每个 patch 是 [patch_size, patch_size, 3] 像素块
    // flatten 顺序: [in_ch, kh, kw] (与 Conv2D weight 行优先布局一致)
    //   patch[c * (ps*ps) + kh * ps + kw] = normalized[(py*ps+kh) * w * 3 + (px*ps+kw) * 3 + c]
    let ps = cfg.patch_size;
    let n_per_side = cfg.n_patches_per_side;
    let n_patches = cfg.n_patches;
    let patch_dim = 3 * ps * ps; // 768

    let mut patches: Vec<f32> = vec![0.0; n_patches * patch_dim];
    for py in 0..n_per_side {
        for px in 0..n_per_side {
            let patch_idx = py * n_per_side + px;
            let patch_off = patch_idx * patch_dim;
            for kh in 0..ps {
                for kw in 0..ps {
                    let y = py * ps + kh;
                    let x = px * ps + kw;
                    let pixel_off = (y * w + x) * 3;
                    for c in 0..3 {
                        // patch[in_ch * (ps*ps) + kh * ps + kw]
                        let patch_inner = c * (ps * ps) + kh * ps + kw;
                        patches[patch_off + patch_inner] = normalized[pixel_off + c];
                    }
                }
            }
        }
    }

    eprintln!("[vision] patchified: {} patches × {} dim", n_patches, patch_dim);
    Ok(patches)
}

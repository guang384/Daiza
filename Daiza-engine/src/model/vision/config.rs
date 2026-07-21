//! Qwen3-VL 视觉编码器配置 (从 mmproj GGUF metadata 解析)
//!
//! 关键 metadata (实测 Bonsai-27B-mmproj-Q8_0.gguf):
//! ```
//! clip.has_vision_encoder = true
//! clip.projector_type = qwen3vl_merger
//! clip.use_gelu = true
//! clip.vision.attention.head_count = 16
//! clip.vision.attention.layer_norm_epsilon = 0.000001
//! clip.vision.block_count = 27
//! clip.vision.embedding_length = 1152
//! clip.vision.feed_forward_length = 4304
//! clip.vision.image_mean = [0.5, 0.5, 0.5]
//! clip.vision.image_size = 768
//! clip.vision.image_std = [0.5, 0.5, 0.5]
//! clip.vision.patch_size = 16
//! clip.vision.projection_dim = 5120
//! clip.vision.spatial_merge_size = 2
//! ```

use crate::gguf::metadata::Metadata;
use crate::BonsaiError;

#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// ViT embedding dim (n_embd) = 1152
    pub embedding_length: usize,
    /// ViT FFN intermediate dim = 4304
    pub feed_forward_length: usize,
    /// ViT block count = 27
    pub block_count: usize,
    /// attention head count = 16
    pub head_count: usize,
    /// head_dim = embedding_length / head_count = 72
    pub head_dim: usize,
    /// LayerNorm epsilon = 1e-6
    pub layer_norm_eps: f32,
    /// image size (square) = 768
    pub image_size: usize,
    /// patch size (square) = 16
    pub patch_size: usize,
    /// patches per side = image_size / patch_size = 48
    pub n_patches_per_side: usize,
    /// total patches = n_patches_per_side^2 = 2304
    pub n_patches: usize,
    /// spatial merge size = 2 (2×2 patch 合并)
    pub spatial_merge_size: usize,
    /// projection dim (text hidden) = 5120
    pub projection_dim: usize,
    /// image normalization mean [R, G, B]
    pub image_mean: [f32; 3],
    /// image normalization std [R, G, B]
    pub image_std: [f32; 3],
    /// use GELU activation (vs QuickGELU)
    pub use_gelu: bool,
    /// projector type name (qwen3vl_merger)
    pub projector_type: String,
    /// M-RoPE sections for vision: [head_dim/4] × 4 = [18, 18, 18, 18]
    pub mrope_sections: [usize; 4],
    /// vision rope_dim = head_dim = 72
    pub rope_dim: usize,
}

impl VisionConfig {
    /// 从 mmproj GGUF metadata 解析 VisionConfig
    pub fn from_metadata(meta: &Metadata) -> crate::Result<Self> {
        // 检查 architecture
        let arch = meta.get_str("general.architecture")
            .ok_or_else(|| BonsaiError::Model("mmproj: general.architecture not found".into()))?;
        if arch != "clip" {
            return Err(BonsaiError::Model(format!(
                "mmproj: expected architecture=clip, got {arch}"
            )));
        }

        // 验证 has_vision_encoder
        let has_vision = meta.get_bool("clip.has_vision_encoder").unwrap_or(false);
        if !has_vision {
            return Err(BonsaiError::Model(
                "mmproj: clip.has_vision_encoder is false".into()
            ));
        }

        let embedding_length = meta.get_u32("clip.vision.embedding_length")
            .ok_or_else(|| BonsaiError::Model("clip.vision.embedding_length not found".into()))?
            as usize;
        let feed_forward_length = meta.get_u32("clip.vision.feed_forward_length")
            .ok_or_else(|| BonsaiError::Model("clip.vision.feed_forward_length not found".into()))?
            as usize;
        let block_count = meta.get_u32("clip.vision.block_count")
            .ok_or_else(|| BonsaiError::Model("clip.vision.block_count not found".into()))?
            as usize;
        let head_count = meta.get_u32("clip.vision.attention.head_count")
            .ok_or_else(|| BonsaiError::Model("clip.vision.attention.head_count not found".into()))?
            as usize;
        let layer_norm_eps = meta.get_f32("clip.vision.attention.layer_norm_epsilon")
            .unwrap_or(1e-6);
        let image_size = meta.get_u32("clip.vision.image_size")
            .ok_or_else(|| BonsaiError::Model("clip.vision.image_size not found".into()))?
            as usize;
        let patch_size = meta.get_u32("clip.vision.patch_size")
            .ok_or_else(|| BonsaiError::Model("clip.vision.patch_size not found".into()))?
            as usize;
        let spatial_merge_size = meta.get_u32("clip.vision.spatial_merge_size")
            .unwrap_or(1) as usize;
        let projection_dim = meta.get_u32("clip.vision.projection_dim")
            .ok_or_else(|| BonsaiError::Model("clip.vision.projection_dim not found".into()))?
            as usize;

        let image_mean = meta.get_f32_array("clip.vision.image_mean")
            .ok_or_else(|| BonsaiError::Model("clip.vision.image_mean not found".into()))?;
        let image_std = meta.get_f32_array("clip.vision.image_std")
            .ok_or_else(|| BonsaiError::Model("clip.vision.image_std not found".into()))?;
        if image_mean.len() != 3 || image_std.len() != 3 {
            return Err(BonsaiError::Model(format!(
                "image_mean/std must have 3 elements, got {}/{}",
                image_mean.len(), image_std.len()
            )));
        }
        let use_gelu = meta.get_bool("clip.use_gelu").unwrap_or(false);
        let projector_type = meta.get_str("clip.projector_type")
            .ok_or_else(|| BonsaiError::Model("clip.projector_type not found".into()))?
            .to_string();
        if projector_type != "qwen3vl_merger" {
            return Err(BonsaiError::Model(format!(
                "unsupported projector_type: {projector_type} (only qwen3vl_merger)"
            )));
        }

        let head_dim = embedding_length / head_count;
        let n_patches_per_side = image_size / patch_size;
        let n_patches = n_patches_per_side * n_patches_per_side;
        // M-RoPE: sections = [head_dim/4] × 4
        let mrope_section_size = head_dim / 4;
        let mrope_sections = [mrope_section_size, mrope_section_size, mrope_section_size, mrope_section_size];
        let rope_dim = head_dim; // vision rope 旋转全部 head_dim

        Ok(Self {
            embedding_length,
            feed_forward_length,
            block_count,
            head_count,
            head_dim,
            layer_norm_eps,
            image_size,
            patch_size,
            n_patches_per_side,
            n_patches,
            spatial_merge_size,
            projection_dim,
            image_mean: [image_mean[0], image_mean[1], image_mean[2]],
            image_std: [image_std[0], image_std[1], image_std[2]],
            use_gelu,
            projector_type,
            mrope_sections,
            rope_dim,
        })
    }

    /// 派生: spatial merge 后的 patch 数 = n_patches / (spatial_merge_size^2)
    pub fn n_patches_merged(&self) -> usize {
        let merge = self.spatial_merge_size * self.spatial_merge_size;
        self.n_patches / merge
    }

    /// 派生: spatial merge 后的 hidden dim = embedding_length × (spatial_merge_size^2)
    pub fn merged_hidden(&self) -> usize {
        let merge = self.spatial_merge_size * self.spatial_merge_size;
        self.embedding_length * merge
    }
}

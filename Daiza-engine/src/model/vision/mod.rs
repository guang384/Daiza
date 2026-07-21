//! Qwen3-VL 多模态视觉编码器 (mmproj)
//!
//! 实现 Bonsai-27B 的多模态能力。从独立 mmproj GGUF 文件加载 ViT 编码器 +
//! qwen3vl_merger 投影器, 把图像编码为 text embedding 空间的 vision embeddings,
//! 在 prefill 阶段注入到 text model 的 input embedding 中。
//!
//! ## 架构 (来自 llama.cpp qwen3vl.cpp + mmproj GGUF dump)
//!
//! 1. **图像预处理**: resize 到 768×768, normalize (mean=std=0.5), 按 patch_size=16 切块
//!    → 48×48 = 2304 patches, 每个 patch 用 gated Conv2D 编码为 1152 维向量
//! 2. **门控 patch embedding**:
//!    `out = Conv2D(W, x) + Conv2D(W.1, x) + bias`
//!    (两个相同形状的 Conv2D 权重相加, 来自 qwen2vl build_inp_with_temporal_merge)
//! 3. **Learned position embedding**: 加 `v.position_embd.weight` [1152, 2304]
//! 4. **27 层 ViT encoder** (LayerNorm + QKV + M-RoPE + Attention + MLP)
//!    - bidirectional attention (no causal mask)
//!    - M-RoPE: mrope_sections = [head_dim/4] × 4 = [18, 18, 18, 18]
//!    - position_ids 为 4D (t=0, h, w, extra=0)
//! 5. **post_ln**: LayerNorm
//! 6. **qwen3vl_merger 投影器**: reshape [n_embd*4, n_pos/4] → mm.0 → GELU → mm.2
//!    - spatial_merge_size=2: 把 2×2 相邻 patch 合并 (2304 → 576 patches)
//!    - 输出 [576, 5120] (text hidden dim)
//! 7. **注入到 text model**: 把 576 个 vision embeddings 替换 input_ids 中的
//!    image_token 位置的 token_embd
//!
//! ## 性能策略
//!
//! - vision encoder 只在每张图调用一次, 不在 decode 路径
//! - 加载时一次性反量化 Q8_0 → F32 (mmproj 0.4GB → ~1.6GB F32)
//! - F32 matvec 复用现有线程池 (scatter_wait_stealing)
//! - text-only decode 零退化: forward_single_token 不修改

pub mod config;
pub mod weights;
pub mod preprocess;
pub mod rope;
pub mod encoder;
pub mod projector;

pub use config::VisionConfig;
pub use weights::{VisionWeights, VisionMatrix, VisionTensor};
pub use preprocess::preprocess_image;
pub use encoder::{ViTContext, encode_image};
pub use projector::project_vision;

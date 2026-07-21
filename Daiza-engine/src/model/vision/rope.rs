//! Vision M-RoPE (Multi-dimensional Rotary Position Embedding)
//!
//! ViT 使用 4D M-RoPE, mrope_sections = [head_dim/4] × 4 = [18, 18, 18, 18]
//! 对应 4 个位置维度: (t, h, w, extra)
//! - section 0 (18 pairs = 36 dim): 时间维 (vision: t=0)
//! - section 1 (18 pairs = 36 dim): 高度维 h
//! - section 2 (18 pairs = 36 dim): 宽度维 w
//! - section 3 (18 pairs = 36 dim): 额外维 (vision: 0)
//!
//! 总 rope_dim = 72 = 4 × 18 pairs (pair = cos+sin)
//!
//! 旋转风格: GPT-NeoX rotate_half (非 interleave)
//!   - 前半 (36 dim) 与后半 (36 dim) 配对
//!   - x_rotated = x * cos + rotate_half(x) * sin
//!
//! 位置 IDs: 每个 patch (py, px) 的位置为 (0, py, px, 0)
//!   即 section 0 用 0, section 1 用 py, section 2 用 px, section 3 用 0
//!
//! 参考: llama.cpp qwen3vl.cpp + transformers modeling_qwen3_vl.py
//!   `get_vision_position_ids` 函数

use super::config::VisionConfig;

/// 预计算所有 patches 的 cos/sin 向量
///
/// 返回:
/// - `cos`: [n_patches * rope_dim] 行优先, 每个 patch 一个 rope_dim=72 的 cos 向量
/// - `sin`: 同上
pub fn vision_mrope_cos_sin(cfg: &VisionConfig) -> (Vec<f32>, Vec<f32>) {
    let n_patches = cfg.n_patches;
    let rope_dim = cfg.rope_dim;  // 72
    let half = rope_dim / 2;       // 36 pairs
    let sections = &cfg.mrope_sections;  // [18, 18, 18, 18]
    let n_per_side = cfg.n_patches_per_side; // 48

    // 频率向量: freqs[i] = 1 / base^(2i/rope_dim), i = 0..half
    let freq_base = 10000.0f32;
    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / freq_base.powf((2 * i) as f32 / rope_dim as f32))
        .collect();

    let mut cos = vec![1.0f32; n_patches * rope_dim];
    let mut sin = vec![0.0f32; n_patches * rope_dim];

    // 对每个 patch 计算 4D M-RoPE
    for py in 0..n_per_side {
        for px in 0..n_per_side {
            let patch_idx = py * n_per_side + px;
            let base_off = patch_idx * rope_dim;

            // 4 个位置值: t=0, h=py, w=px, extra=0
            let pos = [0.0f32, py as f32, px as f32, 0.0f32];

            // 在 half 向量 (36 pairs) 中, sections[0]=18 占 [0,18), sections[1]=18 占 [18,36)
            // 但 rope_dim=72, half=36, 4 sections × 9 pairs = 36? No, 4 × 18 = 72 不对
            //
            // 重新理解: mrope_sections 是 cos/sin 向量 (length=rope_dim=72) 的分段
            // 而不是 pair 数。每个 section 是 dims 不是 pairs。
            // 实际: sections[0]=18 表示 section 0 占 18 dims (即 9 pairs)
            // 但 4 × 18 = 72 = rope_dim ✓
            //
            // 但 GPT-NeoX rotate_half 风格是 cos = cat([freqs_half, freqs_half])
            // 即前 36 dim 是 freqs[0..36] 应用 cos, 后 36 dim 是同样 freqs 应用 cos (镜像)
            //
            // ★ 正确理解 (transformers modeling_qwen3_vl apply_rotary_pos_emb_vision):
            //   cos/sin 长度 = rope_dim = 72
            //   freqs 长度 = half = 36 (一对 cos+sin)
            //   cos[i] = cos(position * freqs[i mod half])? 不对
            //
            // ★ 实际算法 (llama.cpp rope_multi):
            //   sections 划分的是 rope_dim (72), 每个 section 是 18 dims = 9 pairs
            //   freqs[i] = 1/base^(2i/rope_dim) for i in 0..half (=36)
            //   第 i 个 pair (i=0..half=36) 的 position 取决于 i 属于哪个 section:
            //     - i in [0, 9): section 0 (time), pos = pos[0]
            //     - i in [9, 18): section 1 (height), pos = pos[1]
            //     - i in [18, 27): section 2 (width), pos = pos[2]
            //     - i in [27, 36): section 3 (extra), pos = pos[3]
            //
            //   sections=[18, 18, 18, 18] dims, 即 9 pairs per section
            //   section 累积边界 (pair 索引): [0, 9, 18, 27, 36]
            //
            //   cos[i] for i in 0..half:
            //     section_idx = i / 9  (即 pair i 属于哪个 section)
            //     cos[i] = cos(pos[section_idx] * freqs[i])
            //
            //   然后镜像: cos[i + half] = cos[i], sin[i + half] = sin[i] (GPT-NeoX rotate_half)

            // 计算 section 边界 (pair 索引)
            // sections 是 dims, 每个 pair = 2 dims, 所以 section_i_pairs = sections[i] / 2
            let section_pairs: Vec<usize> = sections.iter().map(|&s| s / 2).collect();
            let section_bounds: Vec<usize> = {
                let mut b = Vec::with_capacity(5);
                b.push(0);
                let mut acc = 0;
                for &p in &section_pairs {
                    acc += p;
                    b.push(acc);
                }
                b
            };

            for i in 0..half {
                // 找到 i 属于哪个 section
                let sec = section_bounds.iter().position(|&b| i < b).unwrap_or(section_bounds.len()) - 1;
                let p = pos[sec.min(3)];
                let theta = p * freqs[i];
                let c = theta.cos();
                let s = theta.sin();
                cos[base_off + i] = c;
                sin[base_off + i] = s;
                cos[base_off + i + half] = c;  // GPT-NeoX 镜像
                sin[base_off + i + half] = s;
            }
        }
    }

    (cos, sin)
}

/// 对单个 head 的向量应用 vision M-RoPE (原地修改)
///
/// - `x`: 完整 head 向量 (长度 = head_dim = 72)
/// - `cos`/`sin`: 长度 = rope_dim = 72 的预计算值
///
/// GPT-NeoX rotate_half 风格:
///   x_rotated = x * cos + rotate_half(x) * sin
///   rotate_half(x) = [-x[half:], x[:half]]
pub fn apply_vision_rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let n = x.len();
    let half = n / 2;
    debug_assert_eq!(cos.len(), n);
    debug_assert_eq!(sin.len(), n);

    // rotate_half(x) = [-x[half:], x[:half]]
    // x_new[i] = x[i] * cos[i] + rotate_half[i] * sin[i]
    //   for i in 0..half:    x_new[i] = x[i] * cos[i] + (-x[i + half]) * sin[i]
    //   for i in half..n:    x_new[i] = x[i] * cos[i] + x[i - half] * sin[i]
    let mut new_x = vec![0.0f32; n];
    for i in 0..half {
        new_x[i] = x[i] * cos[i] - x[i + half] * sin[i];
    }
    for i in half..n {
        new_x[i] = x[i] * cos[i] + x[i - half] * sin[i];
    }
    x.copy_from_slice(&new_x);
}

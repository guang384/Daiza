//! RoPE 旋转位置编码(Qwen3 partial RoPE,GPT-NeoX 风格)
//!
//! ## 关键事实(来自调研 + llama.cpp qwen35.cpp)
//!
//! - 每头 256 维,但**只有前 64 维做 RoPE**,后 192 维直通(pass-through)
//! - `rope.dimension_count = 64`(旋转维数)
//! - `rope.dimension_sections = [11, 11, 10, 0]` 是 **M-RoPE** 多模态分段
//!   - section 0 (11 对 = 22 维): 时间维, position = token_pos
//!   - section 1 (11 对 = 22 维): 高度维, position = 0 (纯文本)
//!   - section 2 (10 对 = 20 维): 宽度维, position = 0 (纯文本)
//!   - section 3 (0 对): 无
//!   - **纯文本推理时,只有前 22 维被实际旋转**,其余 42 维 cos=1, sin=0 (不变)
//! - 旋转风格:**GPT-NeoX rotate_half**(非 interleave)
//!   - 把前 32 维与后 32 维配对:`rotate_half(x) = [-x[d/2:], x[:d/2]]`
//!   - `freqs[i] = 1 / base^(2i/d)`,i = 0..d/2
//!   - `cos = cat([freqs, freqs])`(长度 d),`sin` 同
//!   - `x_rotated = x * cos + rotate_half(x) * sin`
//! - **Q 与 K 都做 RoPE**(部分旋转)
//! - RoPE 在 **QK-norm 之后**施加(调研结论 B)

/// 预计算 RoPE 频率向量,长度 = rope_dim(64)
///
/// 返回 cat([freqs, freqs]) 形式(GPT-NeoX 风格)
pub fn rope_freqs(rope_dim: usize, freq_base: f32) -> Vec<f32> {
    let half = rope_dim / 2; // 32
    let mut freqs = Vec::with_capacity(rope_dim);
    for i in 0..half {
        let freq = 1.0 / freq_base.powf((2 * i) as f32 / rope_dim as f32);
        freqs.push(freq);
    }
    // cat([freqs, freqs])
    let mut doubled = freqs.clone();
    doubled.extend(freqs);
    doubled
}

/// 给定 position,计算 cos/sin 向量(长度 = rope_dim)
///
/// **M-RoPE 纯文本模式**:只有 section 0 用 position,其余 sections 用 position=0
/// (cos=1, sin=0, 不旋转)
pub fn rope_cos_sin_mrope_text(
    pos: usize,
    freqs: &[f32],
    sections: &[i32],
) -> (Vec<f32>, Vec<f32>) {
    let n = freqs.len();      // 64 = cat([freqs_half, freqs_half])
    let half = n / 2;         // 32 pairs
    // 默认: cos=1, sin=0 (不旋转)
    let mut cos = vec![1.0f32; n];
    let mut sin = vec![0.0f32; n];

    // Section 0: 时间维, position = pos (token position)
    let sec_0_pairs = sections[0] as usize;  // 11
    for i in 0..sec_0_pairs {
        let theta = pos as f32 * freqs[i];
        let c = theta.cos();
        let s = theta.sin();
        cos[i] = c;
        sin[i] = s;
        // GPT-NeoX: 第二半镜像第一半
        cos[i + half] = c;
        sin[i + half] = s;
    }
    // Sections 1, 2, 3: position = 0 for 纯文本 → cos=1, sin=0 (already default)

    (cos, sin)
}

/// GPT-NeoX rotate_half:`[-x[d/2:], x[:d/2]]`
#[inline]
fn rotate_half(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let half = n / 2;
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&x[half..]);
    out.extend_from_slice(&x[..half]);
    out
}

/// 对单个 head 的向量应用 partial RoPE(原地修改)
///
/// - `x`: 完整的 head 向量(长度 = head_dim,如 256)
/// - `rope_dim`: 旋转维数(如 64),只旋转前 rope_dim 维,其余不动
/// - `cos`/`sin`: 长度 = rope_dim 的预计算值
///
/// 公式(GPT-NeoX 风格,仅作用于前 rope_dim 维):
/// ```text
/// x_rot = x[:rope_dim]
/// x_rot_new = x_rot * cos + rotate_half(x_rot) * sin
/// x[:rope_dim] = x_rot_new
/// ```
pub fn apply_rope_partial(x: &mut [f32], rope_dim: usize, cos: &[f32], sin: &[f32]) {
    debug_assert!(x.len() >= rope_dim);
    debug_assert_eq!(cos.len(), rope_dim);
    debug_assert_eq!(sin.len(), rope_dim);

    // 提取前 rope_dim 维
    let x_rot = x[..rope_dim].to_vec();
    let rotated = rotate_half(&x_rot);

    // x_rot * cos + rotated * sin
    for i in 0..rope_dim {
        x[i] = x_rot[i] * cos[i] + rotated[i] * sin[i];
    }
    // 后 (head_dim - rope_dim) 维保持不变
}

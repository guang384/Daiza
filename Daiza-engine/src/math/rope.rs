//! RoPE 旋转位置编码(Qwen3 partial RoPE,GPT-NeoX 风格)
//!
//! ## 关键事实(来自调研 + llama.cpp qwen35.cpp + transformers modeling_qwen3_vl.py)
//!
//! - 每头 256 维,但**只有前 64 维做 RoPE**,后 192 维直通(pass-through)
//! - `rope.dimension_count = 64`(旋转维数)
//! - `rope.dimension_sections = [11, 11, 10, 0]` 是 **M-RoPE** 多模态分段
//!   - section 0 (11 对 = 22 维): 时间维 (T)
//!   - section 1 (11 对 = 22 维): 高度维 (H)
//!   - section 2 (10 对 = 20 维): 宽度维 (W)
//!   - section 3 (0 对): 无
//! - **纯文本模式**: position = (t, t, t), 所有 section 都用序列位置 t
//!   (等价于标准 RoPE; 与 Qwen3-VL `get_rope_index` text token 一致)
//! - **多模态模式**: text token 用 (t, t, t), vision token 用 (t, h, w)
//!   其中 h/w 是 patch 在 spatial-merge 后的行列索引
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
/// **M-RoPE 纯文本模式**: position = (t, t, t), 所有 section 都用序列位置 t
///
/// ★ 与 Qwen3-VL `get_rope_index` 一致: text token 的三个位置维度 (T, H, W)
///   都用同一个序列位置 t。这等价于标准 RoPE (所有 32 pairs 用同一 position)。
///
/// 原实现只旋转 section 0 (11 pairs = 22 dims), 其余 21 pairs cos=1 sin=0 不旋转,
/// 导致 text token 的 H/W 维 (42 dims) 位置信息完全丢失。text token 与 vision token
/// (用三维 M-RoPE) 的位置编码不匹配, attention 计算时相对位置关系错乱,
/// 模型无法正确建立 text-vision 空间对应关系 → 图像内容混叠、重复计数。
pub fn rope_cos_sin_mrope_text(
    pos: usize,
    freqs: &[f32],
    _sections: &[i32],
) -> (Vec<f32>, Vec<f32>) {
    let n = freqs.len();      // 64 = cat([freqs_half, freqs_half])
    let half = n / 2;         // 32 pairs
    let mut cos = vec![0.0f32; n];
    let mut sin = vec![0.0f32; n];

    // 所有 section 用同一个 pos (纯文本模式 position = (t, t, t))
    for i in 0..half {
        let theta = pos as f32 * freqs[i];
        let c = theta.cos();
        let s = theta.sin();
        cos[i] = c;
        sin[i] = s;
        // GPT-NeoX: 第二半镜像第一半
        cos[i + half] = c;
        sin[i + half] = s;
    }

    (cos, sin)
}

/// 给定 position,计算 cos/sin 写入预分配 buffer(避免每 token 分配 Vec)
///
/// `cos`/`sin` 长度必须 >= `freqs.len()`
///
/// **纯文本模式**: position = (t, t, t), 所有 section 都用 pos
/// (与 Qwen3-VL `get_rope_index` text token 一致)
pub fn rope_cos_sin_mrope_text_into(
    pos: usize,
    freqs: &[f32],
    _sections: &[i32],
    cos: &mut [f32],
    sin: &mut [f32],
) {
    let n = freqs.len();
    let half = n / 2;
    // 所有 section 用同一个 pos (纯文本模式 position = (t, t, t))
    for i in 0..half {
        let theta = pos as f32 * freqs[i];
        let c = theta.cos();
        let s = theta.sin();
        cos[i] = c;
        sin[i] = s;
        cos[i + half] = c;
        sin[i + half] = s;
    }
}

/// Vision M-RoPE: 三段分别用 (pos_t, pos_h, pos_w) 位置编码
///
/// ★ Qwen3-VL 多模态关键: vision token 的 M-RoPE 必须用三维位置:
///   - section 0 (T维, 11对): pos_t = token 在序列中的位置 (与 text token 连续递增)
///   - section 1 (H维, 11对): pos_h = patch 在 spatial-merge 后的行索引 (0..n_per_side_merged)
///   - section 2 (W维, 10对): pos_w = patch 在 spatial-merge 后的列索引 (0..n_per_side_merged)
///
/// 若 vision token 误用 text 模式 (H/W 维 position=0), 模型无法区分 576 个 patch
/// 的空间位置, 导致图像内容混叠、重复计数 (如 "几个文件夹" 被描述为 "成百上千个").
///
/// `freqs` 布局: cat([freqs_half, freqs_half]) (GPT-NeoX 风格, 长度 = rope_dim)
/// `sections`: [sec_0_pairs, sec_1_pairs, sec_2_pairs, sec_3_pairs] = [11, 11, 10, 0]
/// `cos`/`sin` 长度必须 >= `freqs.len()`
pub fn rope_cos_sin_mrope_vision_into(
    pos_t: usize,
    pos_h: usize,
    pos_w: usize,
    freqs: &[f32],
    sections: &[i32],
    cos: &mut [f32],
    sin: &mut [f32],
) {
    let n = freqs.len();
    let half = n / 2;
    // 默认: cos=1, sin=0 (覆盖 section 3 及未使用维度)
    for c in cos[..n].iter_mut() { *c = 1.0; }
    for s in sin[..n].iter_mut() { *s = 0.0; }

    let sec_0_pairs = sections[0] as usize; // 11 (T)
    let sec_1_pairs = sections[1] as usize; // 11 (H)
    let sec_2_pairs = sections[2] as usize; // 10 (W)

    // Section 0 (T维): pos_t
    for i in 0..sec_0_pairs {
        let theta = pos_t as f32 * freqs[i];
        let c = theta.cos();
        let s = theta.sin();
        cos[i] = c;
        sin[i] = s;
        cos[i + half] = c;
        sin[i + half] = s;
    }
    // Section 1 (H维): pos_h
    for i in 0..sec_1_pairs {
        let j = sec_0_pairs + i;
        let theta = pos_h as f32 * freqs[j];
        let c = theta.cos();
        let s = theta.sin();
        cos[j] = c;
        sin[j] = s;
        cos[j + half] = c;
        sin[j + half] = s;
    }
    // Section 2 (W维): pos_w
    for i in 0..sec_2_pairs {
        let j = sec_0_pairs + sec_1_pairs + i;
        let theta = pos_w as f32 * freqs[j];
        let c = theta.cos();
        let s = theta.sin();
        cos[j] = c;
        sin[j] = s;
        cos[j + half] = c;
        sin[j + half] = s;
    }
}

/// 对单个 head 的向量应用 partial RoPE(原地修改)
///
/// - `x`: 完整的 head 向量(长度 = head_dim,如 256)
/// - `rope_dim`: 旋转维数(如 64),只旋转前 rope_dim 维,其余不动
/// - `cos`/`sin`: 长度 = rope_dim 的预计算值,布局为 cat([freqs_half, freqs_half])
///   即 `cos[i] == cos[i+half]`,`sin[i] == sin[i+half]`(half = rope_dim/2)
///
/// 公式(GPT-NeoX 风格,仅作用于前 rope_dim 维):
/// ```text
/// out[i]      = x[i] * cos[i] + x[i+half] * sin[i]
/// out[i+half] = x[i+half] * cos[i] + x[i] * sin[i]
/// ```
///
/// ★ AVX2 优化: 与 drafter 的 apply_rope_full 同算法, 只是 half=32 (4×8-wide, 无尾处理)
///   原: 3 次 copy_from_slice + 64 标量 mul+add = ~200 cycle
///   新: 4 次 SIMD FMA = ~20 cycle (~10x 减少)
///   调用频次: 16 attn blocks × (24 Q + 4 K) heads = 448 次/token
///   原 28672 标量 iter → 新 1792 SIMD iter
pub fn apply_rope_partial(x: &mut [f32], rope_dim: usize, cos: &[f32], sin: &[f32]) {
    debug_assert!(x.len() >= rope_dim);
    debug_assert_eq!(cos.len(), rope_dim);
    debug_assert_eq!(sin.len(), rope_dim);
    let half = rope_dim / 2;

    #[cfg(target_arch = "x86_64")]
    if crate::math::simd_exp::simd_available() && half >= 8 {
        #[allow(unsafe_code)]
        unsafe {
            use std::arch::x86_64::*;
            let mut i = 0;
            let n8 = (half / 8) * 8;
            while i < n8 {
                let x1 = _mm256_loadu_ps(x.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(x.as_ptr().add(i + half));
                let c = _mm256_loadu_ps(cos.as_ptr().add(i));
                let s = _mm256_loadu_ps(sin.as_ptr().add(i));
                // out1 = x1*c + x2*s (rotate_half 无取负, 用 +)
                let out1 = _mm256_fmadd_ps(x1, c, _mm256_mul_ps(x2, s));
                // out2 = x2*c + x1*s
                let out2 = _mm256_fmadd_ps(x2, c, _mm256_mul_ps(x1, s));
                _mm256_storeu_ps(x.as_mut_ptr().add(i), out1);
                _mm256_storeu_ps(x.as_mut_ptr().add(i + half), out2);
                i += 8;
            }
            for j in i..half {
                let x1 = x[j];
                let x2 = x[j + half];
                x[j] = x1 * cos[j] + x2 * sin[j];
                x[j + half] = x2 * cos[j] + x1 * sin[j];
            }
            return;
        }
    }

    // 标量 fallback
    for i in 0..half {
        let x1 = x[i];
        let x2 = x[i + half];
        x[i] = x1 * cos[i] + x2 * sin[i];
        x[i + half] = x2 * cos[i] + x1 * sin[i];
    }
    // 后 (head_dim - rope_dim) 维保持不变
}

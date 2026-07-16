//! 反量化与浮点格式转换
//!
//! ## Q1_0 反量化(核心)
//!
//! 每组 128 权重的二进制布局:
//!
//! ```text
//! ┌──────────────┬──────────────────────────────────────────┐
//! │ FP16 scale    │ 16 字节符号位(每 bit = 一个 ±1 权重)       │
//! │  2 字节       │  128 bits total                          │
//! └──────────────┴──────────────────────────────────────────┘
//!     共 18 字节
//! ```
//!
//! 反量化规则(来自白皮书 §4.2):
//! - `bit = 0  →  weight = -scale`
//! - `bit = 1  →  weight = +scale`
//!
//! ## Q1_0 GEMM 的位运算加速思路(留给后续优化)
//!
//! 计算 `y = W x`,其中 W 是 Q1_0 矩阵。直接展开成 F32 会失去 Q1_0 的核心带宽优势。
//! 利用 ±1 权重的性质:
//!
//! ```text
//! y_i = sum_k s_g * b_ik * x_k
//!     = s_g * (2 * popcount(b_ik bits where x_k is weighted) - sum_k x_k)
//! ```
//!
//! 学习项目的 v0 先做正确性(反量化到 F32 再 GEMM),v1 再做位运算融合。

use crate::tensor::dtype::{Q1_0_BLOCK_BYTES, Q1_0_GROUP_SIZE};

/// IEEE 754 半精度 (binary16) → f32 转换
///
/// 半精度布局:`sign(1) | exponent(5, bias=15) | mantissa(10)`
/// f32 布局:`sign(1) | exponent(8, bias=127) | mantissa(23)`
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;

    let sign32 = (sign as u32) << 31;

    match exp {
        0 => {
            // Subnormal or zero
            if mant == 0 {
                f32::from_bits(sign32)
            } else {
                // 转为 normal:subnormal half → normal float
                let mut m = mant as u32;
                let mut e = 1u32;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e += 1;
                }
                m &= 0x3FF;
                let exp32 = 127 - 15 + e;
                f32::from_bits(sign32 | (exp32 << 23) | (m << 13))
            }
        }
        0x1F => {
            // Inf or NaN
            f32::from_bits(sign32 | 0x7F800000 | ((mant as u32) << 13))
        }
        _ => {
            // Normal:重新对齐指数
            let exp32 = (exp as u32) + (127 - 15);
            f32::from_bits(sign32 | (exp32 << 23) | ((mant as u32) << 13))
        }
    }
}

/// bfloat16 → f32 转换(简单:高 16 位就是 f32 的高 16 位)
///
/// BF16 布局:`sign(1) | exponent(8, bias=127) | mantissa(7)`
/// 等于 f32 截断低 16 位。直接左移 16 位即可。
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// 把 Q1_0 的 raw bytes 反量化为 F32 vec
///
/// - `data`:完整张量的字节(Q1_0 编码)
/// - `n_elements`:权重元素总数(必须为 128 的倍数;末尾不足补零,本引擎假设齐整)
///
/// 返回长度为 `n_elements` 的 F32 向量
pub fn dequantize_q1_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_groups = n_elements.div_ceil(Q1_0_GROUP_SIZE);
    let mut out = Vec::with_capacity(n_elements);
    for g in 0..n_groups {
        let block_start = g * Q1_0_BLOCK_BYTES;
        if block_start + 2 > data.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = f16_to_f32(scale_bits);
        let sign_bytes = &data[block_start + 2..block_start + Q1_0_BLOCK_BYTES];
        // 16 字节 = 128 bits,逐 bit 展开
        for byte_idx in 0..16 {
            let b = sign_bytes[byte_idx];
            for bit_idx in 0..8 {
                // bit=1 → +scale,bit=0 → -scale
                let bit = (b >> bit_idx) & 1;
                let w = if bit == 1 { scale } else { -scale };
                out.push(w);
                if out.len() >= n_elements {
                    return out;
                }
            }
        }
    }
    // 补齐末尾不足一组的零权重(若 n_elements 不是 128 倍数)
    while out.len() < n_elements {
        out.push(0.0);
    }
    out
}

/// 反量化单行(用于 GEMM 中按需反量化,避免一次性展开整个大矩阵)
///
/// - `data`:整个 Q1_0 张量的字节
/// - `row_idx`:第几行(0-based)
/// - `n_cols`:权重列数(输入维度)
pub fn dequantize_q1_0_row(data: &[u8], row_idx: usize, n_cols: usize) -> Vec<f32> {
    // Q1_0 在 GGUF 中按行连续存储:第 i 行从第 i*n_cols 字节偏移开始(按权重)
    // 每 128 权重占 18 字节,所以第 i 行的字节偏移 = i * n_cols / 128 * 18
    let row_byte_offset = row_idx * (n_cols.div_ceil(Q1_0_GROUP_SIZE) * Q1_0_BLOCK_BYTES);
    let row_bytes = &data[row_byte_offset..];
    dequantize_q1_0(row_bytes, n_cols)
}

/// 反量化单行,写入提供的缓冲区(避免每行分配 Vec)
///
/// - `data`:整个 Q1_0 张量的字节
/// - `row_idx`:第几行(0-based)
/// - `n_cols`:权重列数(输入维度)
/// - `out`:输出缓冲,长度必须 >= n_cols
pub fn dequantize_q1_0_row_into(data: &[u8], row_idx: usize, n_cols: usize, out: &mut [f32]) {
    debug_assert!(out.len() >= n_cols);
    let groups_per_row = n_cols.div_ceil(Q1_0_GROUP_SIZE);
    let row_byte_offset = row_idx * (groups_per_row * Q1_0_BLOCK_BYTES);
    let row_bytes = &data[row_byte_offset..];

    let mut out_idx = 0;
    for g in 0..groups_per_row {
        let block_start = g * Q1_0_BLOCK_BYTES;
        if block_start + Q1_0_BLOCK_BYTES > row_bytes.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([row_bytes[block_start], row_bytes[block_start + 1]]);
        let scale = f16_to_f32(scale_bits);
        let sign_bytes = &row_bytes[block_start + 2..block_start + Q1_0_BLOCK_BYTES];
        for byte_idx in 0..16 {
            let b = sign_bytes[byte_idx];
            for bit_idx in 0..8 {
                let bit = ((b >> bit_idx) & 1) as f32;
                // 无分支:bit=1 → +scale,bit=0 → -scale
                let w = (bit * 2.0 - 1.0) * scale;
                if out_idx < n_cols {
                    out[out_idx] = w;
                    out_idx += 1;
                } else {
                    return;
                }
            }
        }
    }
    while out_idx < n_cols {
        out[out_idx] = 0.0;
        out_idx += 1;
    }
}

/// 直接计算 Q1_0 一行与 x 的点积,不分配中间 F32 缓冲
///
/// 利用 Q1_0 的二值性质:`w_k = bit_k ? scale : -scale`
/// - `dot = sum_k w_k * x_k = sum_g scale_g * (2 * sum_pos_g - sum_x_g)`
///   其中 `sum_pos_g` 是该组中 bit=1 对应的 x 之和,`sum_x_g` 是该组全部 x 之和
///
/// **关键优化**:内层循环使用 `sum_pos += bit_f * xv` (无分支 FMA),
/// 而非 `if bit == 1 { sum_pos += xv }`,使编译器能自动向量化(AVX2)。
///
/// - `data`:整个 Q1_0 张量的字节
/// - `row_idx`:第几行(0-based)
/// - `n_cols`:权重列数(输入维度)
/// - `x`:输入向量,长度必须等于 `n_cols`
pub fn dot_q1_0_row(data: &[u8], row_idx: usize, n_cols: usize, x: &[f32]) -> f32 {
    debug_assert!(x.len() >= n_cols);
    // Bonsai 所有权重矩阵列数均为 128 的倍数,直接整除避免余数处理
    let groups_per_row = n_cols / Q1_0_GROUP_SIZE;
    let row_byte_offset = row_idx * (groups_per_row * Q1_0_BLOCK_BYTES);
    let row_bytes = &data[row_byte_offset..];

    let mut acc = 0.0f32;
    for g in 0..groups_per_row {
        let block_start = g * Q1_0_BLOCK_BYTES;
        let scale_bits = u16::from_le_bytes([row_bytes[block_start], row_bytes[block_start + 1]]);
        let scale = f16_to_f32(scale_bits);
        let sign_bytes = &row_bytes[block_start + 2..block_start + Q1_0_BLOCK_BYTES];

        let x_off = g * Q1_0_GROUP_SIZE;
        // 无分支:bit=1 → sum_pos += xv;bit=0 → sum_pos += 0
        // 编译器可向量化此 8 次内层循环
        let mut sum_x = 0.0f32;
        let mut sum_pos = 0.0f32;
        for byte_idx in 0..16 {
            let b = sign_bytes[byte_idx];
            for bit_idx in 0..8 {
                let bit = ((b >> bit_idx) & 1) as f32;
                let xv = x[x_off + byte_idx * 8 + bit_idx];
                sum_x += xv;
                sum_pos += bit * xv;
            }
        }
        acc += scale * (2.0 * sum_pos - sum_x);
    }
    acc
}

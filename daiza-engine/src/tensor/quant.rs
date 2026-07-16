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
///
/// 完整实现(含 subnormal/inf 边界)。Q1_0 scale 走 [`f16_to_f32_fast`] 路径。
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

/// Q1_0 scale 专用 fast path(无分支,无 subnormal 处理)
///
/// Bonsai 的 Q1_0 group scale 几乎都是 normal range(exp ∈ [1, 30]),
/// subnormal/inf 情况极少出现且 Bonsai 训练已规约。此 fast path 假设:
/// - exp == 0 → 返回 ±0.0(subnormal 当 0 处理,误差可忽略)
/// - exp == 0x1F → 返回 ±inf(正确)
/// - 其他 → 标准 normal 转换
///
/// 关键:无 match 分支,可被内联到 GEMM 内层循环,助力向量化。
#[inline(always)]
pub fn f16_to_f32_fast(bits: u16) -> f32 {
    let t32 = bits as u32;
    let sign = (t32 & 0x8000) << 16;                       // sign 移到 f32 位 31
    let exp = (t32 >> 10) & 0x1F;                          // 5-bit exp
    let mant = t32 & 0x3FF;                               // 10-bit mantissa

    // 真无分支:用 bool→u32 转换(setcc 指令,非分支)
    // - exp == 0  → exp32 = 0       (subnormal → 0,误差可忽略)
    // - exp == 31 → exp32 = 0xFF     (inf/nan)
    // - 其他      → exp32 = exp+112  (normal range)
    // 公式:exp32 = (exp != 0) as u32 * (exp + 112) + (exp == 31) as u32 * (0xFF - 31 - 112)
    //      化简:exp32 = ((exp != 0) as u32) * (exp + 112) + ((exp == 0x1F) as u32) * 112
    // 验证:
    //   exp=0:  0 * 112 + 0 * 112   = 0     ✓
    //   exp=1:  1 * 113 + 0 * 112    = 113   ✓
    //   exp=30: 1 * 142 + 0 * 112    = 142   ✓
    //   exp=31: 1 * 143 + 1 * 112    = 255   ✓ (= 0xFF)
    let nz = (exp != 0) as u32;
    let inf_flag = (exp == 0x1F) as u32;
    let exp32 = nz * (exp + (127 - 15)) + inf_flag * 112;

    f32::from_bits(sign | (exp32 << 23) | (mant << 13))
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
/// **关键优化 v4**:AVX2 intrinsics 手写内核 + scalar fallback
///
/// v2/v3 的 safe Rust 版本经汇编验证完全未向量化(全是 `vmulss`/`vaddss` 标量),
/// 即使 `RUSTFLAGS="-C target-feature=+avx2,+fma"` 也无效 —— rustc 无法把
/// `(b>>n)&1` + `as f32` 这种位运算+类型转换组合识别为可向量化模式。
///
/// v4 直接手写 AVX2 内核:
/// - 每 byte_idx (16 次) 处理 1 个 sign byte + 8 个连续 x (单 __m256)
/// - sign byte → 8 个 ±1.0 f32 (单 __m256):`broadcast b` + `and masks`
///   + `cmpgt zero` + `blendv ±1.0`
/// - `vfmadd231ps` 累加 → 单 group 16 次 FMA
/// - 对比标量 128 次 vmulss/vaddss,理论 8x 加速
///
/// 仍保留 scalar fallback 供非 x86_64 / 无 AVX2 环境使用。
///
/// - `data`:整个 Q1_0 张量的字节
/// - `row_idx`:第几行(0-based)
/// - `n_cols`:权重列数(输入维度)
/// - `x`:输入向量,长度必须等于 `n_cols`
pub fn dot_q1_0_row(data: &[u8], row_idx: usize, n_cols: usize, x: &[f32]) -> f32 {
    debug_assert!(x.len() >= n_cols);
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
        && std::is_x86_feature_detected!("f16c")
    {
        // 调用 AVX2 + F16C 内核 —— unsafe 在此处用 deny(unsafe_code) 全局豁免
        #[allow(unsafe_code)]
        unsafe {
            return dot_q1_0_row_avx2(data, row_idx, n_cols, x);
        }
    }
    dot_q1_0_row_scalar(data, row_idx, n_cols, x)
}

/// Sign bit LUT(查表优化):每个 sign byte (0..256) 对应 8 个 ±1.0 的固定组合。
///
/// 原实现:AND(bit_mask) → CMPGT → BLENDV → FMA,4-cycle 依赖链,且 BLENDV 占用
/// 稀缺的 port 5。改用 256 项 LUT(每项 `__m256` = 32 字节,共 8KB,完美驻留 L1):
///
/// ```text
/// 原: load byte → AND(1c) → CMPGT(1c) → BLENDV(2c) → FMA(4c)   链长 4c
/// 新: load byte → loadu_ps(LUT[b])  → FMA(4c)                  链长 1c
/// ```
///
/// LUT 在编译期用 `const fn` 生成,放在 `.rodata` 段,程序启动后由 hardware
/// prefetcher 一次性拉入 L1,后续 4.15M 次/token 全部 L1 命中(1c 吞吐)。
#[repr(C, align(32))]
#[derive(Copy, Clone)]
struct SignLutEntry([f32; 8]);

const SIGN_LUT: [SignLutEntry; 256] = {
    let mut lut = [SignLutEntry([0.0f32; 8]); 256];
    let mut idx = 0;
    while idx < 256 {
        let mut bit = 0;
        while bit < 8 {
            // bit=1 → +1.0,bit=0 → -1.0(与 Q1_0 规约一致)
            lut[idx].0[bit] = if (idx >> bit) & 1 == 1 { 1.0 } else { -1.0 };
            bit += 1;
        }
        idx += 1;
    }
    lut
};

/// AVX2 内联版本(unsafe,需 runtime feature detect)
///
/// 两项关键优化(相比 v7 baseline):
/// 1. **Sign bit LUT 查表**:256 项 `__m256` LUT(8KB,驻 L1),用一条 `loadu_ps`
///    替代 AND+CMPGT+BLENDV 三步依赖链,消除了 BLENDV 对 port 5 的占用。
/// 2. **向量累加器**:把每 group 的横向求和(5-7 cycle)延迟到行末只做一次。
///    原:40 groups/行 × 5 cycle × 4.15M 行/token = 830M cycle/token ≈ 275ms/token
///    新:每行只做 1 次横向求和,节省 ~200-300ms/tok。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
unsafe fn dot_q1_0_row_avx2(
    data: &[u8],
    row_idx: usize,
    n_cols: usize,
    x: &[f32],
) -> f32 {
    use std::arch::x86_64::*;

    let groups_per_row = n_cols / Q1_0_GROUP_SIZE;
    let row_byte_offset = row_idx * (groups_per_row * Q1_0_BLOCK_BYTES);

    // 向量累加器:跨 group 累加,行末只做一次横向求和
    // (原实现每 group 横向求和再标量累加,~5c/group × 40 groups = 200c/行 浪费)
    let mut acc_vec = _mm256_setzero_ps();

    for g in 0..groups_per_row {
        let block_start = row_byte_offset + g * Q1_0_BLOCK_BYTES;
        let scale_bits = u16::from_le_bytes([
            *data.get_unchecked(block_start),
            *data.get_unchecked(block_start + 1),
        ]);

        // ★ F16C 指令优化:用 _mm_cvtph_ps 一条指令转 4 个 f16→f32,取 lane 0
        let scale_xmm = _mm_cvtph_ps(_mm_set1_epi16(scale_bits as i16));
        let scale_v = _mm256_broadcastss_ps(scale_xmm);

        let sign_ptr = data.as_ptr().add(block_start + 2);
        let x_ptr = x.as_ptr().add(g * Q1_0_GROUP_SIZE);

        // ★ 4 路独立累加器:打破 FMA 依赖链
        //   原:16 次串行 FMA (每次依赖前一次结果) → 16c/group
        //   新:4 路并行,每路 4 深 → 2 FMA units 同时执行 → ~4c/group (4x 加速)
        //   Intel Haswell+ 有 2 个独立 FMA unit (port 0 + port 1)
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for byte_idx in (0..16).step_by(4) {
            let b0 = *sign_ptr.add(byte_idx) as usize;
            let b1 = *sign_ptr.add(byte_idx + 1) as usize;
            let b2 = *sign_ptr.add(byte_idx + 2) as usize;
            let b3 = *sign_ptr.add(byte_idx + 3) as usize;

            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[b0].0.as_ptr()),
                _mm256_loadu_ps(x_ptr.add(byte_idx * 8)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[b1].0.as_ptr()),
                _mm256_loadu_ps(x_ptr.add((byte_idx + 1) * 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[b2].0.as_ptr()),
                _mm256_loadu_ps(x_ptr.add((byte_idx + 2) * 8)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[b3].0.as_ptr()),
                _mm256_loadu_ps(x_ptr.add((byte_idx + 3) * 8)),
                acc3,
            );
        }

        // 合并 4 路累加器
        let group_acc = _mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        );

        // 向量 FMA 累加到全局 acc_vec
        acc_vec = _mm256_fmadd_ps(scale_v, group_acc, acc_vec);
    }

    // 行末一次性横向求和 __m256 → f32
    let hi = _mm256_extractf128_ps(acc_vec, 1);
    let lo = _mm256_castps256_ps128(acc_vec);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// scalar fallback(非 AVX2 平台用,逻辑与 v3 一致)
pub fn dot_q1_0_row_scalar(data: &[u8], row_idx: usize, n_cols: usize, x: &[f32]) -> f32 {
    debug_assert!(x.len() >= n_cols);
    let groups_per_row = n_cols / Q1_0_GROUP_SIZE;
    let row_byte_offset = row_idx * (groups_per_row * Q1_0_BLOCK_BYTES);
    let row_bytes = &data[row_byte_offset..];

    let mut acc = 0.0f32;
    let mut signs_buf = [0.0f32; 128];

    for g in 0..groups_per_row {
        let block_start = g * Q1_0_BLOCK_BYTES;
        let scale_bits = u16::from_le_bytes([row_bytes[block_start], row_bytes[block_start + 1]]);
        let scale = f16_to_f32_fast(scale_bits);
        let sign_bytes = &row_bytes[block_start + 2..block_start + Q1_0_BLOCK_BYTES];
        let x_off = g * Q1_0_GROUP_SIZE;

        for byte_idx in 0..16 {
            let b = sign_bytes[byte_idx];
            let dst_off = byte_idx * 8;
            signs_buf[dst_off + 0] = 2.0 * (((b >> 0) & 1) as f32) - 1.0;
            signs_buf[dst_off + 1] = 2.0 * (((b >> 1) & 1) as f32) - 1.0;
            signs_buf[dst_off + 2] = 2.0 * (((b >> 2) & 1) as f32) - 1.0;
            signs_buf[dst_off + 3] = 2.0 * (((b >> 3) & 1) as f32) - 1.0;
            signs_buf[dst_off + 4] = 2.0 * (((b >> 4) & 1) as f32) - 1.0;
            signs_buf[dst_off + 5] = 2.0 * (((b >> 5) & 1) as f32) - 1.0;
            signs_buf[dst_off + 6] = 2.0 * (((b >> 6) & 1) as f32) - 1.0;
            signs_buf[dst_off + 7] = 2.0 * (((b >> 7) & 1) as f32) - 1.0;
        }

        let xs = &x[x_off..x_off + 128];
        let mut group_acc = 0.0f32;
        for i in 0..128 {
            group_acc += signs_buf[i] * xs[i];
        }
        acc += scale * group_acc;
    }
    acc
}

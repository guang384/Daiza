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
//! ## Q1_0 GEMM 当前方案: SIGN_LUT 查表 + AVX2 FMA
//!
//! 计算 `y = W x`,其中 W 是 Q1_0 矩阵。直接展开成 F32 会失去 Q1_0 的核心带宽优势。
//! 当前采用 SIGN_LUT 查表方案: 16 字节符号位 → 2 次 `_mm256_shuffle_epi8` 查表
//! 得到 8 个 ±1.0 f32,再 FMA 累加 x。详见 `SIGN_LUT` 注释与 `dot_q1_0_row_avx2`。
//!
//! 历史上曾尝试 BLENDV/sign-mask 方案,但 port 5 压力 + 4c 依赖链导致退化,已放弃。
//! AVX-512 上的 popcount 位运算方案在本项目目标 CPU (Meteor Lake, 无 AVX-512) 不可用。

use crate::tensor::dtype::{Q1_0_BLOCK_BYTES, Q1_0_GROUP_SIZE, Q4_1_BLOCK_BYTES, Q4_1_GROUP_SIZE};
#[cfg(target_arch = "x86_64")]
use crate::math::simd_exp::hsum_ps;

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
        for &b in sign_bytes {
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
        for &b in sign_bytes {
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

/// 返回 SIGN_LUT[byte] 的 8-lane ±1.0 向量 (供 gemm.rs 的块反量化复用)
#[inline]
pub fn sign_lut_entry(byte: u8) -> &'static [f32; 8] {
    &SIGN_LUT[byte as usize].0
}

/// 一次性 runtime AVX2+FMA+F16C feature 检测 (P2-1)
///
/// 供 `weights.rs` 在循环外调用一次, 避免每行 `dot_q1_0_row` 内部的
/// `is_x86_feature_detected!` atomic load + branch 开销 (~3c × 4.15M 行/token)。
#[cfg(target_arch = "x86_64")]
pub fn avx2_q1_0_available() -> bool {
    std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
        && std::is_x86_feature_detected!("f16c")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_q1_0_available() -> bool {
    false
}

/// AVX2 内联版本(unsafe,需 runtime feature detect)
///
/// 两项关键优化(相比 v7 baseline):
/// 1. **Sign bit LUT 查表**:256 项 `__m256` LUT(8KB,驻 L1),用一条 `loadu_ps`
///    替代 AND+CMPGT+BLENDV 三步依赖链。
/// 2. **向量累加器**:横向求和延迟到行末只做一次。
///
/// # Safety
///
/// - 仅可在支持 `avx2+fma+f16c` 的 CPU 上调用 (见 `avx2_q1_0_available`), 否则触发 SIGILL。
/// - `n_cols` 须为 `Q1_0_GROUP_SIZE` (128) 的倍数 (整除截断, 余数列被忽略)。
/// - `data` 须至少覆盖 `(row_idx + 1) * groups_per_row * Q1_0_BLOCK_BYTES` 字节
///   (`groups_per_row = n_cols / 128`; 内部 `get_unchecked` 读 scale/sign, 无边界检查)。
/// - `x` 须至少 `n_cols` 个元素 (全部 unaligned load, 无对齐要求)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
#[inline]
pub unsafe fn dot_q1_0_row_avx2(
    data: &[u8],
    row_idx: usize,
    n_cols: usize,
    x: &[f32],
) -> f32 {
    use std::arch::x86_64::*;

    let groups_per_row = n_cols / Q1_0_GROUP_SIZE;
    let row_byte_offset = row_idx * (groups_per_row * Q1_0_BLOCK_BYTES);

    let mut acc_vec = _mm256_setzero_ps();

    for g in 0..groups_per_row {
        let block_start = row_byte_offset + g * Q1_0_BLOCK_BYTES;
        let scale_bits = u16::from_le_bytes([
            *data.get_unchecked(block_start),
            *data.get_unchecked(block_start + 1),
        ]);

        // ★ _mm256_cvtph_ps 直接 8-lane f16→f32 (set1 填充 8 个相同 f16, 结果天然 broadcast)
        //   原 _mm_cvtph_ps(4 lane) + _mm256_broadcastss_ps(扩 8 lane) = 2 条指令
        let scale_v = _mm256_cvtph_ps(_mm_set1_epi16(scale_bits as i16));

        let sign_ptr = data.as_ptr().add(block_start + 2);
        let x_ptr = x.as_ptr().add(g * Q1_0_GROUP_SIZE);

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

        let group_acc = _mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        );

        acc_vec = _mm256_fmadd_ps(scale_v, group_acc, acc_vec);
    }

    // 行末一次性横向求和 __m256 → f32
    hsum_ps(acc_vec)
}

/// ★ P0-A: 双行并行 kernel — 同时计算 2 行的点积, 共享 x 向量 load
///
/// **瓶颈分析**: 单行 kernel 每 FMA 需 2 次 32B load (LUT + x),
///   load port (2 ports × 32B/cycle = 64B/cycle) 限制吞吐为 ~1 FMA/cycle。
///
/// **优化**: 2 行共享同一 x 向量, 每 2 FMA 只需 3 次 load (2 LUT + 1 x):
///   - 单行: 68B/FMA → 0.94 FMA/cycle
///   - 双行: 52B/FMA → 1.23 FMA/cycle (+31%)
///
/// 寄存器: 8 group_acc (4/row × 2) + 2 row_acc + 2 scale = 12/16 YMM
///
/// # Safety
///
/// - 仅可在支持 `avx2+fma+f16c` 的 CPU 上调用 (见 `avx2_q1_0_available`), 否则触发 SIGILL。
/// - `n_cols` 须为 128 的倍数 (整除截断, 余数列被忽略)。
/// - `data` 须至少覆盖 `(max(row_idx0, row_idx1) + 1) * groups_per_row * Q1_0_BLOCK_BYTES`
///   字节 (内部 `get_unchecked` 读 scale/sign, 无边界检查)。
/// - `x` 须至少 `n_cols` 个元素 (全部 unaligned load, 无对齐要求)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
#[inline]
pub unsafe fn dot_q1_0_row_dual_avx2(
    data: &[u8],
    row_idx0: usize,
    row_idx1: usize,
    n_cols: usize,
    x: &[f32],
) -> (f32, f32) {
    use std::arch::x86_64::*;

    let groups_per_row = n_cols / Q1_0_GROUP_SIZE;
    let row_off0 = row_idx0 * (groups_per_row * Q1_0_BLOCK_BYTES);
    let row_off1 = row_idx1 * (groups_per_row * Q1_0_BLOCK_BYTES);

    let mut acc_vec0 = _mm256_setzero_ps();
    let mut acc_vec1 = _mm256_setzero_ps();

    for g in 0..groups_per_row {
        let bs0 = row_off0 + g * Q1_0_BLOCK_BYTES;
        let bs1 = row_off1 + g * Q1_0_BLOCK_BYTES;

        let scale0 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*data.get_unchecked(bs0), *data.get_unchecked(bs0 + 1)]) as i16,
        ));
        let scale1 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*data.get_unchecked(bs1), *data.get_unchecked(bs1 + 1)]) as i16,
        ));

        let sp0 = data.as_ptr().add(bs0 + 2);
        let sp1 = data.as_ptr().add(bs1 + 2);
        let xp = x.as_ptr().add(g * Q1_0_GROUP_SIZE);

        // 4 路/行 × 2 行 = 8 路独立 FMA 链
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut b2 = _mm256_setzero_ps();
        let mut b3 = _mm256_setzero_ps();

        for byte_idx in (0..16).step_by(4) {
            // ★ x 只 load 一次, 2 行共享 (省 1 次 load/FMA pair)
            let x0 = _mm256_loadu_ps(xp.add(byte_idx * 8));
            let x1 = _mm256_loadu_ps(xp.add((byte_idx + 1) * 8));
            let x2 = _mm256_loadu_ps(xp.add((byte_idx + 2) * 8));
            let x3 = _mm256_loadu_ps(xp.add((byte_idx + 3) * 8));

            // row 0
            a0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp0.add(byte_idx) as usize].0.as_ptr()),
                x0, a0,
            );
            a1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp0.add(byte_idx + 1) as usize].0.as_ptr()),
                x1, a1,
            );
            a2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp0.add(byte_idx + 2) as usize].0.as_ptr()),
                x2, a2,
            );
            a3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp0.add(byte_idx + 3) as usize].0.as_ptr()),
                x3, a3,
            );

            // row 1 (复用 x0-x3)
            b0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp1.add(byte_idx) as usize].0.as_ptr()),
                x0, b0,
            );
            b1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp1.add(byte_idx + 1) as usize].0.as_ptr()),
                x1, b1,
            );
            b2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp1.add(byte_idx + 2) as usize].0.as_ptr()),
                x2, b2,
            );
            b3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp1.add(byte_idx + 3) as usize].0.as_ptr()),
                x3, b3,
            );
        }

        let g0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let g1 = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
        acc_vec0 = _mm256_fmadd_ps(scale0, g0, acc_vec0);
        acc_vec1 = _mm256_fmadd_ps(scale1, g1, acc_vec1);
    }

    let hi0 = _mm256_extractf128_ps(acc_vec0, 1);
    let lo0 = _mm256_castps256_ps128(acc_vec0);
    let s0 = _mm_add_ps(hi0, lo0);
    let sh0 = _mm_movehdup_ps(s0);
    let sm0 = _mm_add_ps(s0, sh0);
    let r0 = _mm_cvtss_f32(_mm_add_ss(sm0, _mm_movehl_ps(sm0, sm0)));

    let hi1 = _mm256_extractf128_ps(acc_vec1, 1);
    let lo1 = _mm256_castps256_ps128(acc_vec1);
    let s1 = _mm_add_ps(hi1, lo1);
    let sh1 = _mm_movehdup_ps(s1);
    let sm1 = _mm_add_ps(s1, sh1);
    let r1 = _mm_cvtss_f32(_mm_add_ss(sm1, _mm_movehl_ps(sm1, sm1)));

    (r0, r1)
}

/// ★ P0-E: Triple-row kernel — 3 行共享 x load, load/FMA 比 1.33 (vs dual 1.50)
///
/// **瓶颈分析**: Dual-row 每 2 FMA 需 3 loads (2 LUT + 1 x), load port (2 ports) 限制吞吐。
///
/// **优化**: 3 行共享同一 x 向量, 每 3 FMA 需 4 loads (3 LUT + 1 x):
///   - Dual:  1.50 loads/FMA → 1.33 FMA/cycle
///   - Triple: 1.33 loads/FMA → 1.50 FMA/cycle (+12.5%)
///
/// **寄存器**: 6 group_acc (2/row × 3) + 3 row_acc + 3 scale = 12/16 YMM
/// 内层循环: 2 sign bytes/iter (8 iter/group), 8 loads + 6 FMAs per iter
/// load port: 8 loads / 2 ports = 4 cycles, 6 FMAs / 2 ports = 3 cycles → load-bound 4c/6 FMA
///
/// # Safety
///
/// - 仅可在支持 `avx2+fma+f16c` 的 CPU 上调用 (见 `avx2_q1_0_available`), 否则触发 SIGILL。
/// - `n_cols` 须为 128 的倍数 (整除截断, 余数列被忽略)。
/// - `data` 须至少覆盖 `(max(row_idx0, row_idx1, row_idx2) + 1) * groups_per_row *
///   Q1_0_BLOCK_BYTES` 字节 (内部 `get_unchecked` 读 scale/sign, 无边界检查)。
/// - `x` 须至少 `n_cols` 个元素 (全部 unaligned load, 无对齐要求)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
#[inline]
pub unsafe fn dot_q1_0_row_triple_avx2(
    data: &[u8],
    row_idx0: usize,
    row_idx1: usize,
    row_idx2: usize,
    n_cols: usize,
    x: &[f32],
) -> (f32, f32, f32) {
    use std::arch::x86_64::*;

    let groups_per_row = n_cols / Q1_0_GROUP_SIZE;
    let row_off0 = row_idx0 * (groups_per_row * Q1_0_BLOCK_BYTES);
    let row_off1 = row_idx1 * (groups_per_row * Q1_0_BLOCK_BYTES);
    let row_off2 = row_idx2 * (groups_per_row * Q1_0_BLOCK_BYTES);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();

    for g in 0..groups_per_row {
        let bs0 = row_off0 + g * Q1_0_BLOCK_BYTES;
        let bs1 = row_off1 + g * Q1_0_BLOCK_BYTES;
        let bs2 = row_off2 + g * Q1_0_BLOCK_BYTES;

        let scale0 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*data.get_unchecked(bs0), *data.get_unchecked(bs0 + 1)]) as i16,
        ));
        let scale1 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*data.get_unchecked(bs1), *data.get_unchecked(bs1 + 1)]) as i16,
        ));
        let scale2 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*data.get_unchecked(bs2), *data.get_unchecked(bs2 + 1)]) as i16,
        ));

        let sp0 = data.as_ptr().add(bs0 + 2);
        let sp1 = data.as_ptr().add(bs1 + 2);
        let sp2 = data.as_ptr().add(bs2 + 2);
        let xp = x.as_ptr().add(g * Q1_0_GROUP_SIZE);

        // 2 路/行 × 3 行 = 6 路独立 FMA 链
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut c0 = _mm256_setzero_ps();
        let mut c1 = _mm256_setzero_ps();

        for byte_idx in (0..16).step_by(2) {
            // ★ x 只 load 一次, 3 行共享
            let x0 = _mm256_loadu_ps(xp.add(byte_idx * 8));
            let x1 = _mm256_loadu_ps(xp.add((byte_idx + 1) * 8));

            // row 0
            a0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp0.add(byte_idx) as usize].0.as_ptr()),
                x0, a0,
            );
            a1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp0.add(byte_idx + 1) as usize].0.as_ptr()),
                x1, a1,
            );

            // row 1 (复用 x0, x1)
            b0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp1.add(byte_idx) as usize].0.as_ptr()),
                x0, b0,
            );
            b1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp1.add(byte_idx + 1) as usize].0.as_ptr()),
                x1, b1,
            );

            // row 2 (复用 x0, x1)
            c0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp2.add(byte_idx) as usize].0.as_ptr()),
                x0, c0,
            );
            c1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp2.add(byte_idx + 1) as usize].0.as_ptr()),
                x1, c1,
            );
        }

        // Merge 2-way → group_acc, then FMA scale → row_acc
        let g0 = _mm256_add_ps(a0, a1);
        let g1 = _mm256_add_ps(b0, b1);
        let g2 = _mm256_add_ps(c0, c1);
        acc0 = _mm256_fmadd_ps(scale0, g0, acc0);
        acc1 = _mm256_fmadd_ps(scale1, g1, acc1);
        acc2 = _mm256_fmadd_ps(scale2, g2, acc2);
    }

    let r0 = hsum_ps(acc0);
    let r1 = hsum_ps(acc1);
    let r2 = hsum_ps(acc2);

    (r0, r1, r2)
}

/// ★ P0-H: Quad-row kernel — 4 行共享 x load, load/FMA 比 1.25 (vs triple 1.33)
///
/// **瓶颈分析**: Triple-row 每 3 FMA 需 4 loads (3 LUT + 1 x), load port (2 ports) 限制吞吐。
///
/// **优化**: 4 行共享同一 x 向量, 每 4 FMA 需 5 loads (4 LUT + 1 x):
///   - Triple: 1.33 loads/FMA → 1.50 FMA/cycle
///   - Quad:   1.25 loads/FMA → 1.60 FMA/cycle (+6.7%)
///
/// **寄存器**: 8 group_acc (2/row × 4) + 4 row_acc + 4 scale = 16/16 YMM (满)
/// 内层循环: 2 sign bytes/iter (8 iter/group), 10 loads + 8 FMAs per iter
/// load port: 10 loads / 2 ports = 5 cycles, 8 FMAs / 2 ports = 4 cycles → load-bound 5c/8 FMA
///
/// # Safety
///
/// - 仅可在支持 `avx2+fma+f16c` 的 CPU 上调用 (见 `avx2_q1_0_available`), 否则触发 SIGILL。
/// - `n_cols` 须为 128 的倍数 (整除截断, 余数列被忽略)。
/// - `data` 须至少覆盖 `(max(row_idx0..=row_idx3) + 1) * groups_per_row * Q1_0_BLOCK_BYTES`
///   字节 (内部 `get_unchecked` 读 scale/sign, 无边界检查)。
/// - `x` 须至少 `n_cols` 个元素 (全部 unaligned load, 无对齐要求)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
#[inline]
pub unsafe fn dot_q1_0_row_quad_avx2(
    data: &[u8],
    row_idx0: usize,
    row_idx1: usize,
    row_idx2: usize,
    row_idx3: usize,
    n_cols: usize,
    x: &[f32],
) -> (f32, f32, f32, f32) {
    use std::arch::x86_64::*;

    let groups_per_row = n_cols / Q1_0_GROUP_SIZE;
    let row_off0 = row_idx0 * (groups_per_row * Q1_0_BLOCK_BYTES);
    let row_off1 = row_idx1 * (groups_per_row * Q1_0_BLOCK_BYTES);
    let row_off2 = row_idx2 * (groups_per_row * Q1_0_BLOCK_BYTES);
    let row_off3 = row_idx3 * (groups_per_row * Q1_0_BLOCK_BYTES);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    // ★ V5 边界折叠: 链首以零常量起步 (省每组 8×vxorps), 组末 merge 折叠为
    //   双 scale-FMA (省 8×vaddps, 缩短组边界串行依赖)。
    //   bench_klab 交错实测 +2.5%; 数值与原版差 ~1e-6 (求和结构变化,
    //   与 k-lane GEMM 同级, 8-token greedy 逐字节验证通过)。
    let zero = _mm256_setzero_ps();

    for g in 0..groups_per_row {
        let bs0 = row_off0 + g * Q1_0_BLOCK_BYTES;
        let bs1 = row_off1 + g * Q1_0_BLOCK_BYTES;
        let bs2 = row_off2 + g * Q1_0_BLOCK_BYTES;
        let bs3 = row_off3 + g * Q1_0_BLOCK_BYTES;

        let scale0 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*data.get_unchecked(bs0), *data.get_unchecked(bs0 + 1)]) as i16,
        ));
        let scale1 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*data.get_unchecked(bs1), *data.get_unchecked(bs1 + 1)]) as i16,
        ));
        let scale2 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*data.get_unchecked(bs2), *data.get_unchecked(bs2 + 1)]) as i16,
        ));
        let scale3 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*data.get_unchecked(bs3), *data.get_unchecked(bs3 + 1)]) as i16,
        ));

        let sp0 = data.as_ptr().add(bs0 + 2);
        let sp1 = data.as_ptr().add(bs1 + 2);
        let sp2 = data.as_ptr().add(bs2 + 2);
        let sp3 = data.as_ptr().add(bs3 + 2);
        let xp = x.as_ptr().add(g * Q1_0_GROUP_SIZE);

        // 2 路/行 × 4 行 = 8 路独立 FMA 链
        let mut a0 = zero;
        let mut a1 = zero;
        let mut b0 = zero;
        let mut b1 = zero;
        let mut c0 = zero;
        let mut c1 = zero;
        let mut d0 = zero;
        let mut d1 = zero;

        for byte_idx in (0..16).step_by(2) {
            // ★ x 只 load 一次, 4 行共享
            let x0 = _mm256_loadu_ps(xp.add(byte_idx * 8));
            let x1 = _mm256_loadu_ps(xp.add((byte_idx + 1) * 8));

            // row 0
            a0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp0.add(byte_idx) as usize].0.as_ptr()),
                x0, a0,
            );
            a1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp0.add(byte_idx + 1) as usize].0.as_ptr()),
                x1, a1,
            );

            // row 1 (复用 x0, x1)
            b0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp1.add(byte_idx) as usize].0.as_ptr()),
                x0, b0,
            );
            b1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp1.add(byte_idx + 1) as usize].0.as_ptr()),
                x1, b1,
            );

            // row 2 (复用 x0, x1)
            c0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp2.add(byte_idx) as usize].0.as_ptr()),
                x0, c0,
            );
            c1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp2.add(byte_idx + 1) as usize].0.as_ptr()),
                x1, c1,
            );

            // row 3 (复用 x0, x1)
            d0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp3.add(byte_idx) as usize].0.as_ptr()),
                x0, d0,
            );
            d1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(SIGN_LUT[*sp3.add(byte_idx + 1) as usize].0.as_ptr()),
                x1, d1,
            );
        }

        // ★ V5: merge 折叠 — 直接以 scale 把两路链 FMA 进持久 acc
        //   (原: g0=add(a0,a1); acc=fma(scale,g0,acc); 省 8×vadd/组, 边界依赖更短)
        acc0 = _mm256_fmadd_ps(scale0, a0, acc0);
        acc0 = _mm256_fmadd_ps(scale0, a1, acc0);
        acc1 = _mm256_fmadd_ps(scale1, b0, acc1);
        acc1 = _mm256_fmadd_ps(scale1, b1, acc1);
        acc2 = _mm256_fmadd_ps(scale2, c0, acc2);
        acc2 = _mm256_fmadd_ps(scale2, c1, acc2);
        acc3 = _mm256_fmadd_ps(scale3, d0, acc3);
        acc3 = _mm256_fmadd_ps(scale3, d1, acc3);
    }

    let r0 = hsum_ps(acc0);
    let r1 = hsum_ps(acc1);
    let r2 = hsum_ps(acc2);
    let r3 = hsum_ps(acc3);

    (r0, r1, r2, r3)
}

/// scalar fallback(非 AVX2 平台用,逻辑与 v3 一致)
// 8 行展开的 bit-0 项刻意保留 `+ 0`/`>> 0` 前缀, 与 bit1..7 保持对称可读
#[allow(clippy::identity_op)]
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

        for (byte_idx, &b) in sign_bytes.iter().enumerate() {
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

// ===========================================================================
// ★ 整数乘加 kernel (模仿 llama.cpp): Q1_0 × Q8_0 用 maddubs + madd
// 降低浮点 FMA 数量 (16→5), 降低功耗密度, 避免热降频
// ===========================================================================

/// 将 F32 向量量化为简化 Q8_0 字节流
/// 格式: 每 32 值一组, 4 字节 f32 scale + 32 字节 int8 values = 36 字节/block
pub fn quantize_f32_to_q8_0_simple(x: &[f32]) -> Vec<u8> {
    debug_assert!(x.len().is_multiple_of(32), "Q8_0 requires len % 32 == 0, got {}", x.len());
    let n_blocks = x.len() / 32;
    let mut out = vec![0u8; n_blocks * 36];
    for b in 0..n_blocks {
        let off = b * 32;
        let mut amax = 0.0f32;
        for j in 0..32 {
            let ax = x[off + j].abs();
            if ax > amax { amax = ax; }
        }
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        // f32 scale (little-endian)
        let d_bits = d.to_bits();
        out[b * 36] = (d_bits & 0xFF) as u8;
        out[b * 36 + 1] = ((d_bits >> 8) & 0xFF) as u8;
        out[b * 36 + 2] = ((d_bits >> 16) & 0xFF) as u8;
        out[b * 36 + 3] = ((d_bits >> 24) & 0xFF) as u8;
        // int8 values
        for j in 0..32 {
            let q = (x[off + j] * id).round() as i32;
            let q = q.clamp(-128, 127) as i8;
            out[b * 36 + 4 + j] = q as u8;
        }
    }
    out
}

/// Q1_0 × Q8_0 整数乘加 kernel (模仿 llama.cpp)
///
/// 算法: 每 block (128 值) 分 4 个 K-iteration:
///   1. sign-mask: shuffle + and + cmpeq + xor + sub → signed int8 sy
///   2. 整数乘加: maddubs(ones, sy) → 16-bit, madd(16-bit, ones) → 32-bit
///   3. 浮点累加: cvtepi32_ps + fmadd(y_d, sum32, acc_block)
///
/// 浮点 FMA 数量: 5/block (vs LUT 方案 16/block) — 降低功耗密度 70%
///
/// # Safety
///
/// - 仅可在支持 `avx2+fma` 的 CPU 上调用 (f16 scale 走 `f16_to_f32_fast` 位运算, 无需 f16c),
///   否则触发 SIGILL。
/// - `n_cols` 须为 128 的倍数 (整除截断, 余数列被忽略)。
/// - `data` 须至少覆盖 `(row_idx + 1) * groups_per_row * Q1_0_BLOCK_BYTES` 字节
///   (内部 `get_unchecked` + `read_unaligned`, 无边界检查、无对齐要求)。
/// - `x_q8` 须为简化 Q8_0 (36 字节/block), 至少 `groups_per_row * 4 * 36` 字节
///   (每 group 128 值 = 4 个 Q8 block)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
#[inline]
pub unsafe fn dot_q1_0_q8_0_row_avx2(
    data: &[u8],
    row_idx: usize,
    n_cols: usize,
    x_q8: &[u8],  // 简化 Q8_0: 36 字节/block
) -> f32 {
    use std::arch::x86_64::*;
    let groups_per_row = n_cols / Q1_0_GROUP_SIZE;
    let row_byte_offset = row_idx * (groups_per_row * Q1_0_BLOCK_BYTES);

    let ones_8 = _mm256_set1_epi8(1);
    let ones_16 = _mm256_set1_epi16(1);
    let byte_shuf = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,
    );
    let bit_masks = _mm256_setr_epi8(
        1,2,4,8,16,32,64,-128i8 as u8 as i8, 1,2,4,8,16,32,64,-128i8 as u8 as i8,
        1,2,4,8,16,32,64,-128i8 as u8 as i8, 1,2,4,8,16,32,64,-128i8 as u8 as i8,
    );
    let zero = _mm256_setzero_si256();
    let mut acc = _mm256_setzero_ps();

    for g in 0..groups_per_row {
        let block_start = row_byte_offset + g * Q1_0_BLOCK_BYTES;
        let scale_bits = u16::from_le_bytes([
            *data.get_unchecked(block_start),
            *data.get_unchecked(block_start + 1),
        ]);
        let d0 = _mm256_set1_ps(f16_to_f32_fast(scale_bits));
        let qs_ptr = data.as_ptr().add(block_start + 2);
        // 每 group 128 值 = 4 个 Q8_0 block (32 值 each, 36 字节 each)
        let x_q8_off = g * 4 * 36;

        let mut acc_block = _mm256_setzero_ps();
        for K in 0..4 {
            // Q1_0 sign bits: 4 字节 (uint32)
            // ★ Q1_0 block=18 字节, 地址几乎总非 4 对齐, 必须用 read_unaligned 避免 UB
            let qs32 = std::ptr::read_unaligned(qs_ptr.add(K * 4) as *const u32) as i32;
            // Q8_0 int8 values: 32 字节
            let qy = _mm256_loadu_si256(x_q8.as_ptr().add(x_q8_off + K * 36 + 4) as *const __m256i);
            // Q8_0 f32 scale
            let y_d_bits = u32::from_le_bytes([
                *x_q8.get_unchecked(x_q8_off + K * 36),
                *x_q8.get_unchecked(x_q8_off + K * 36 + 1),
                *x_q8.get_unchecked(x_q8_off + K * 36 + 2),
                *x_q8.get_unchecked(x_q8_off + K * 36 + 3),
            ]);
            let y_d = _mm256_set1_ps(f32::from_bits(y_d_bits));

            // sign-mask: sy = (qy XOR sm) - sm, sm = cmpeq(and(shuffle(qs32), bit_masks), zero)
            let sm = _mm256_cmpeq_epi8(
                _mm256_and_si256(
                    _mm256_shuffle_epi8(_mm256_set1_epi32(qs32), byte_shuf),
                    bit_masks,
                ),
                zero,
            );
            let sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            // 整数乘加: ones × sy → 16-bit, ones × 16-bit → 32-bit
            let s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);

            if K == 0 {
                acc_block = _mm256_mul_ps(y_d, _mm256_cvtepi32_ps(s32));
            } else {
                acc_block = _mm256_fmadd_ps(y_d, _mm256_cvtepi32_ps(s32), acc_block);
            }
        }
        acc = _mm256_fmadd_ps(d0, acc_block, acc);
    }

    // hsum
    hsum_ps(acc)
}

/// 批量计算 Q1_0 一行与多个 x 的点积 (P1-4 优化)
///
/// 固定 `row_idx`, 对 `n_batch` 个 `x[t]` 同时计算点积, 写入 `y[t * y_stride]`。
///
/// **核心优化**: scale 广播和 LUT 查表在 group 内只做一次, 对 batch 内所有 token 复用,
/// 消除原 `matvec_batch_into_slice` 中 `dot_q1_0_row` 重复加载 LUT 和 scale 的开销。
///
/// - `x`: `[n_batch * x_stride]` 行优先 (通常 `x_stride = n_cols`)
/// - `y`: `[n_batch * y_stride]` 输出 (通常 `y_stride = 1`, 即 `y[t]` 是第 t 个输出)
#[allow(clippy::too_many_arguments)]
pub fn dot_q1_0_row_batch(
    data: &[u8],
    row_idx: usize,
    n_cols: usize,
    x: &[f32],
    x_stride: usize,
    n_batch: usize,
    y: &mut [f32],
    y_stride: usize,
) {
    debug_assert!(x.len() >= n_batch * x_stride);
    debug_assert!(y.len() >= n_batch * y_stride);
    debug_assert!(x_stride >= n_cols, "x_stride {x_stride} < n_cols {n_cols}");
    if n_batch == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
        && std::is_x86_feature_detected!("f16c")
    {
        #[allow(unsafe_code)]
        unsafe {
            dot_q1_0_row_batch_avx2(data, row_idx, n_cols, x, x_stride, n_batch, y, y_stride);
        }
        return;
    }
    // Fallback: 逐 token 调用 scalar
    for t in 0..n_batch {
        let xt = &x[t * x_stride..t * x_stride + n_cols];
        y[t * y_stride] = dot_q1_0_row_scalar(data, row_idx, n_cols, xt);
    }
}

/// AVX2 batched kernel
///
/// **设计**: 分块 2 token, 每 group 内 4 路 FMA 并行 × 2 token = 8 个 group-level acc,
/// 加上 4 个 LUT temp + 1 个 scale_v = 13 寄存器, 不溢出。
///
/// 相比原 `dot_q1_0_row_avx2` 调用 n_batch 次:
/// - LUT load 次数从 n_batch × 64 降到 64 (节省 ~75%)
/// - scale F16C + broadcast 从 n_batch 降到 1 (节省 ~96%)
/// - hsum 次数不变 (n_batch, 每 token 行末一次)
///
/// # Safety
///
/// - 仅可在支持 `avx2+fma+f16c` 的 CPU 上调用 (见 `avx2_q1_0_available`), 否则触发 SIGILL。
/// - `n_cols` 须为 128 的倍数 (整除截断, 余数列被忽略)。
/// - `data` 须至少覆盖 `(row_idx + 1) * groups_per_row * Q1_0_BLOCK_BYTES` 字节。
/// - `x` 行优先: 第 t 行始于 `x[t * x_stride]`, 每行须可读 `n_cols` 个元素
///   (与安全包装 `dot_q1_0_row_batch` 的 debug 断言一致: `x.len() >= n_batch * x_stride`、
///   `x_stride >= n_cols`)。
/// - `y` 须满足 `y.len() >= n_batch * y_stride`, 写入 `y[t * y_stride]`。
/// - `x` 与 `y` 不得重叠 (pair 内先读 x 后写 y, 重叠会读到被改写的数据)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn dot_q1_0_row_batch_avx2(
    data: &[u8],
    row_idx: usize,
    n_cols: usize,
    x: &[f32],
    x_stride: usize,
    n_batch: usize,
    y: &mut [f32],
    y_stride: usize,
) {
    use std::arch::x86_64::*;
    let groups_per_row = n_cols / Q1_0_GROUP_SIZE;
    let row_byte_offset = row_idx * (groups_per_row * Q1_0_BLOCK_BYTES);

    // ★ 分块 2 token: 寄存器分配 = 2 row_acc + 8 group_acc + 4 LUT + 1 scale = 15 寄存器
    let mut t_start = 0usize;
    while t_start < n_batch {
        let has_pair = t_start + 1 < n_batch;
        // ★ 第二 token 行步长: 无配对时置 0 (= x_base0, 界内且不参与任何 load)。
        //   ptr::add 计算越过 one-past-end 的指针本身即 UB (即使从不解引用),
        //   奇数 n_batch 的最后一个单 token 会越界; has_pair 为循环不变量, 分支
        //   完全可预测, 且 GEP 数学与原式逐字节一致 (x_base0 + x_stride == 原 x_base1)。
        let x_step1 = if has_pair { x_stride } else { 0 };

        // 行级累加器 (跨 group 累加, 行末一次 hsum)
        let mut row_acc0 = _mm256_setzero_ps();
        let mut row_acc1 = _mm256_setzero_ps();

        for g in 0..groups_per_row {
            let block_start = row_byte_offset + g * Q1_0_BLOCK_BYTES;
            let scale_bits = u16::from_le_bytes([
                *data.get_unchecked(block_start),
                *data.get_unchecked(block_start + 1),
            ]);
            // ★ scale F16C + broadcast 每 group 只做一次, 2 token 共享
            //   _mm256_cvtph_ps 直接 8-lane (set1 填充 8 个相同 f16 → 8 个相同 f32)
            let scale_v = _mm256_cvtph_ps(_mm_set1_epi16(scale_bits as i16));
            let sign_ptr = data.as_ptr().add(block_start + 2);
            let x_base0 = x.as_ptr().add(t_start * x_stride + g * Q1_0_GROUP_SIZE);
            let x_base1 = x_base0.add(x_step1);

            // group 级 4 路并行 acc (每 token 一组)
            let mut g0a = _mm256_setzero_ps();
            let mut g1a = _mm256_setzero_ps();
            let mut g2a = _mm256_setzero_ps();
            let mut g3a = _mm256_setzero_ps();
            let mut g0b = _mm256_setzero_ps();
            let mut g1b = _mm256_setzero_ps();
            let mut g2b = _mm256_setzero_ps();
            let mut g3b = _mm256_setzero_ps();

            for byte_idx in (0..16).step_by(4) {
                // ★ LUT 查表提到 token 循环外, 2 token 共享 (省一半 LUT load)
                let b0 = *sign_ptr.add(byte_idx) as usize;
                let b1 = *sign_ptr.add(byte_idx + 1) as usize;
                let b2 = *sign_ptr.add(byte_idx + 2) as usize;
                let b3 = *sign_ptr.add(byte_idx + 3) as usize;
                let lut0 = _mm256_loadu_ps(SIGN_LUT[b0].0.as_ptr());
                let lut1 = _mm256_loadu_ps(SIGN_LUT[b1].0.as_ptr());
                let lut2 = _mm256_loadu_ps(SIGN_LUT[b2].0.as_ptr());
                let lut3 = _mm256_loadu_ps(SIGN_LUT[b3].0.as_ptr());

                // token 0
                g0a = _mm256_fmadd_ps(lut0, _mm256_loadu_ps(x_base0.add(byte_idx * 8)), g0a);
                g1a = _mm256_fmadd_ps(lut1, _mm256_loadu_ps(x_base0.add((byte_idx + 1) * 8)), g1a);
                g2a = _mm256_fmadd_ps(lut2, _mm256_loadu_ps(x_base0.add((byte_idx + 2) * 8)), g2a);
                g3a = _mm256_fmadd_ps(lut3, _mm256_loadu_ps(x_base0.add((byte_idx + 3) * 8)), g3a);

                if has_pair {
                    g0b = _mm256_fmadd_ps(lut0, _mm256_loadu_ps(x_base1.add(byte_idx * 8)), g0b);
                    g1b = _mm256_fmadd_ps(lut1, _mm256_loadu_ps(x_base1.add((byte_idx + 1) * 8)), g1b);
                    g2b = _mm256_fmadd_ps(lut2, _mm256_loadu_ps(x_base1.add((byte_idx + 2) * 8)), g2b);
                    g3b = _mm256_fmadd_ps(lut3, _mm256_loadu_ps(x_base1.add((byte_idx + 3) * 8)), g3b);
                }
            }

            // 4 路 merge → group_acc, 然后 FMA scale 累加到 row_acc
            let group_acc0 = _mm256_add_ps(
                _mm256_add_ps(g0a, g1a),
                _mm256_add_ps(g2a, g3a),
            );
            row_acc0 = _mm256_fmadd_ps(scale_v, group_acc0, row_acc0);
            if has_pair {
                let group_acc1 = _mm256_add_ps(
                    _mm256_add_ps(g0b, g1b),
                    _mm256_add_ps(g2b, g3b),
                );
                row_acc1 = _mm256_fmadd_ps(scale_v, group_acc1, row_acc1);
            }
        }

        // 行末一次性横向求和
        *y.get_unchecked_mut(t_start * y_stride) = hsum_ps(row_acc0);
        if has_pair {
            *y.get_unchecked_mut((t_start + 1) * y_stride) = hsum_ps(row_acc1);
        }

        t_start += if has_pair { 2 } else { 1 };
    }
}

/// AVX2 batch4 kernel — 4 个 x 单次 pass, 权重只读 1 次
///
/// **设计动机**: 原 `dot_q1_0_row_batch_avx2` 对 n_batch=4 分 2 pair 处理,
/// 每 pair 重读权重; 在 14 线程下 L2 thrashing 导致 pair 2 也走 DRAM,
/// 实测 forward_batch(k=4) ≈ 4× single forward (无权重复用)。
///
/// **新设计**: 4 个 x 在单次 weight pass 内全部处理完。
/// - 寄存器: 4 row_acc + 8 group_acc (2/x × 4 x) + 2 LUT + 1 scale = 15 YMM
/// - step_by(2) 拆 16 字节为 8 次迭代, 每次 2 LUT + 8 FMA
/// - 权重 bytes 每行只读 1 次 (DRAM 带宽 = single forward)
///
/// 相比 batch_avx2 (n_batch=4):
/// - 权重 DRAM 读取: 1× vs 2-4× (主要收益)
/// - LUT loads: 16 vs 32 (pair-based 重复)
/// - FMA 数: 64 vs 64 (相同)
///
/// # Safety
///
/// - 仅可在支持 `avx2+fma+f16c` 的 CPU 上调用 (见 `avx2_q1_0_available`), 否则触发 SIGILL。
/// - `n_cols` 须为 128 的倍数; `x.len() >= 4 * n_cols`、`y.len() >= 4` (debug 断言)。
/// - `data` 须至少覆盖 `(row_idx + 1) * groups_per_row * Q1_0_BLOCK_BYTES` 字节。
/// - `x` 与 `y` 不得重叠 (末尾一次性写 `y[0..4]`)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
// 4 行展开的 row-0 项刻意保留 `0 *`/`1 *` 前缀, 与 row1/2/3 保持对称可读
#[allow(clippy::erasing_op, clippy::identity_op)]
#[inline]
pub unsafe fn dot_q1_0_row_batch4_avx2(
    data: &[u8],
    row_idx: usize,
    n_cols: usize,
    x: &[f32],  // [4 * n_cols], 行优先
    y: &mut [f32],  // [4]
) {
    use std::arch::x86_64::*;
    debug_assert!(x.len() >= 4 * n_cols);
    debug_assert!(y.len() >= 4);

    let groups_per_row = n_cols / Q1_0_GROUP_SIZE;
    let row_byte_offset = row_idx * (groups_per_row * Q1_0_BLOCK_BYTES);

    // 4 个 row 级累加器 (每 x 一个, 跨 group 累加, 行末一次 hsum)
    let mut row_acc0 = _mm256_setzero_ps();
    let mut row_acc1 = _mm256_setzero_ps();
    let mut row_acc2 = _mm256_setzero_ps();
    let mut row_acc3 = _mm256_setzero_ps();

    for g in 0..groups_per_row {
        let block_start = row_byte_offset + g * Q1_0_BLOCK_BYTES;
        let scale_bits = u16::from_le_bytes([
            *data.get_unchecked(block_start),
            *data.get_unchecked(block_start + 1),
        ]);
        // scale F16C + broadcast 每 group 只做一次, 4 x 共享
        //   _mm256_cvtph_ps 直接 8-lane (set1 填充 8 个相同 f16 → 8 个相同 f32)
        let scale_v = _mm256_cvtph_ps(_mm_set1_epi16(scale_bits as i16));
        let sign_ptr = data.as_ptr().add(block_start + 2);

        // 4 个 x base (不同 x, 同一 group)
        let x_base0 = x.as_ptr().add(0 * n_cols + g * Q1_0_GROUP_SIZE);
        let x_base1 = x.as_ptr().add(1 * n_cols + g * Q1_0_GROUP_SIZE);
        let x_base2 = x.as_ptr().add(2 * n_cols + g * Q1_0_GROUP_SIZE);
        let x_base3 = x.as_ptr().add(3 * n_cols + g * Q1_0_GROUP_SIZE);

        // 8 个 group 级累加器 (2/x × 4 x), step_by(2) 下两路独立 FMA 链
        let mut g0a = _mm256_setzero_ps();
        let mut g1a = _mm256_setzero_ps();
        let mut g0b = _mm256_setzero_ps();
        let mut g1b = _mm256_setzero_ps();
        let mut g0c = _mm256_setzero_ps();
        let mut g1c = _mm256_setzero_ps();
        let mut g0d = _mm256_setzero_ps();
        let mut g1d = _mm256_setzero_ps();

        for byte_idx in (0..16).step_by(2) {
            // 2 LUT loads (4 x 共享, 相比 batch_avx2 省 50% LUT)
            let lut0 = _mm256_loadu_ps(SIGN_LUT[*sign_ptr.add(byte_idx) as usize].0.as_ptr());
            let lut1 = _mm256_loadu_ps(SIGN_LUT[*sign_ptr.add(byte_idx + 1) as usize].0.as_ptr());

            // x0, x1 for this byte_idx (4 x's)
            let x0a = _mm256_loadu_ps(x_base0.add(byte_idx * 8));
            let x1a = _mm256_loadu_ps(x_base0.add((byte_idx + 1) * 8));
            let x0b = _mm256_loadu_ps(x_base1.add(byte_idx * 8));
            let x1b = _mm256_loadu_ps(x_base1.add((byte_idx + 1) * 8));
            let x0c = _mm256_loadu_ps(x_base2.add(byte_idx * 8));
            let x1c = _mm256_loadu_ps(x_base2.add((byte_idx + 1) * 8));
            let x0d = _mm256_loadu_ps(x_base3.add(byte_idx * 8));
            let x1d = _mm256_loadu_ps(x_base3.add((byte_idx + 1) * 8));

            // 8 FMA (2/x × 4 x), 两路独立链
            g0a = _mm256_fmadd_ps(lut0, x0a, g0a);
            g1a = _mm256_fmadd_ps(lut1, x1a, g1a);
            g0b = _mm256_fmadd_ps(lut0, x0b, g0b);
            g1b = _mm256_fmadd_ps(lut1, x1b, g1b);
            g0c = _mm256_fmadd_ps(lut0, x0c, g0c);
            g1c = _mm256_fmadd_ps(lut1, x1c, g1c);
            g0d = _mm256_fmadd_ps(lut0, x0d, g0d);
            g1d = _mm256_fmadd_ps(lut1, x1d, g1d);
        }

        // 2 路 merge → group_acc, FMA scale → row_acc
        let group_acc0 = _mm256_add_ps(g0a, g1a);
        let group_acc1 = _mm256_add_ps(g0b, g1b);
        let group_acc2 = _mm256_add_ps(g0c, g1c);
        let group_acc3 = _mm256_add_ps(g0d, g1d);
        row_acc0 = _mm256_fmadd_ps(scale_v, group_acc0, row_acc0);
        row_acc1 = _mm256_fmadd_ps(scale_v, group_acc1, row_acc1);
        row_acc2 = _mm256_fmadd_ps(scale_v, group_acc2, row_acc2);
        row_acc3 = _mm256_fmadd_ps(scale_v, group_acc3, row_acc3);
    }

    // 行末一次性横向求和
    *y.get_unchecked_mut(0) = hsum_ps(row_acc0);
    *y.get_unchecked_mut(1) = hsum_ps(row_acc1);
    *y.get_unchecked_mut(2) = hsum_ps(row_acc2);
    *y.get_unchecked_mut(3) = hsum_ps(row_acc3);
}

/// ★ verify4 kernel: 推测解码 batch-verify 专用 (1 权重行 × 4 token)
///
/// 与 `dot_q1_0_row_batch4_avx2` 的差异 (性能修复):
///
/// | | batch4_avx2 (现状) | verify4 (本内核) |
/// |---|---|---|
/// | 长寿命 ymm | 8 group_acc + 4 row_acc = 12 | 4 group_acc + 4 row_acc = 8 |
/// | x 临时 | 8 (x0a..x1d 同时活跃) | 2-4 (xa..xd FMA 后即死, 编译器复用) |
/// | 寄存器总量 | ~20 ymm → 必然 spill | ~13 ymm → 零 spill |
/// | 实测 | ~169 cycles/group | 目标 ~45c/group |
///
/// 每 group (128 权重 = 16 sign bytes):
/// - 16 LUT loads (4 token 共享, 每 byte 1 次)
/// - 64 x loads (每 token 每 byte 1 次 32B)
/// - 64 FMA (FMA 端口下限 64/2 = 32 cycles)
/// - 4 scale FMA (group 末)
///
/// 依赖链: 每 token 单 group_acc 链 (16 FMA × ~4c = 64c 延迟),
/// 4 条独立链交错发射填满 2 FMA/cycle, 延迟被完全遮盖。
///
/// 用途: speculative decode 的 batch verify — 一次权重 DRAM 流产出
/// 4 个 token 的该行点积, 将 257 barriers/token 的同步开销摊薄 4 倍。
///
/// # Safety
///
/// - 仅可在支持 `avx2+fma+f16c` 的 CPU 上调用 (见 `avx2_q1_0_available`), 否则触发 SIGILL。
/// - `n_cols` 须为 128 的倍数; `x.len() >= 4 * n_cols`、`y.len() >= 4` (debug 断言)。
/// - `data` 须至少覆盖 `(row_idx + 1) * groups_per_row * Q1_0_BLOCK_BYTES` 字节。
/// - `x` 与 `y` 不得重叠 (末尾一次性写 `y[0..4]`)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
#[inline]
pub unsafe fn dot_q1_0_row_verify4_avx2(
    data: &[u8],
    row_idx: usize,
    n_cols: usize,
    x: &[f32],   // [4 * n_cols] 行优先: token t 在 x[t*n_cols..]
    y: &mut [f32], // [4] 输出
) {
    use std::arch::x86_64::*;
    debug_assert!(x.len() >= 4 * n_cols);
    debug_assert!(y.len() >= 4);

    let groups_per_row = n_cols / Q1_0_GROUP_SIZE;
    let row_byte_offset = row_idx * (groups_per_row * Q1_0_BLOCK_BYTES);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let x_base = x.as_ptr();
    let stride = n_cols;

    for g in 0..groups_per_row {
        let block_start = row_byte_offset + g * Q1_0_BLOCK_BYTES;
        let scale_bits = u16::from_le_bytes([
            *data.get_unchecked(block_start),
            *data.get_unchecked(block_start + 1),
        ]);
        let scale_v = _mm256_cvtph_ps(_mm_set1_epi16(scale_bits as i16));
        let sign_ptr = data.as_ptr().add(block_start + 2);
        let xg = x_base.add(g * Q1_0_GROUP_SIZE);

        // 每 byte: 1 LUT load (4 token 共享) + 4 x load + 4 FMA
        let mut g0 = _mm256_setzero_ps();
        let mut g1 = _mm256_setzero_ps();
        let mut g2 = _mm256_setzero_ps();
        let mut g3 = _mm256_setzero_ps();
        for byte_idx in 0..16 {
            let lut = _mm256_loadu_ps(SIGN_LUT[*sign_ptr.add(byte_idx) as usize].0.as_ptr());
            let xa = _mm256_loadu_ps(xg.add(byte_idx * 8));
            let xb = _mm256_loadu_ps(xg.add(stride + byte_idx * 8));
            let xc = _mm256_loadu_ps(xg.add(2 * stride + byte_idx * 8));
            let xd = _mm256_loadu_ps(xg.add(3 * stride + byte_idx * 8));
            g0 = _mm256_fmadd_ps(lut, xa, g0);
            g1 = _mm256_fmadd_ps(lut, xb, g1);
            g2 = _mm256_fmadd_ps(lut, xc, g2);
            g3 = _mm256_fmadd_ps(lut, xd, g3);
        }
        acc0 = _mm256_fmadd_ps(scale_v, g0, acc0);
        acc1 = _mm256_fmadd_ps(scale_v, g1, acc1);
        acc2 = _mm256_fmadd_ps(scale_v, g2, acc2);
        acc3 = _mm256_fmadd_ps(scale_v, g3, acc3);
    }

    *y.get_unchecked_mut(0) = hsum_ps(acc0);
    *y.get_unchecked_mut(1) = hsum_ps(acc1);
    *y.get_unchecked_mut(2) = hsum_ps(acc2);
    *y.get_unchecked_mut(3) = hsum_ps(acc3);
}

/// ★ verify4 x 交错布局: [group][token][128 列]
///
/// x_int[(g * 4 + t) * 128 .. +128] = x4[t * n_cols + g * 128 .. +128]
///
/// kernel 每 group 只触碰 512B 连续区域 (8 cache lines), 64 次 x loads 全部
/// L1 命中 —— 消除 verify4 行优先布局下 4 路跨 n_cols*4B stride 的 L2 访问
/// (x 总量 4×20KB=80KB > L1 32KB, 行优先时每次 group 都 miss)。
///
/// 交错成本 ~10μs/调用 (80KB×2 读写), 相对 verify 的 ~160ms 权重流可忽略。
pub fn interleave_x4_verify(x4: &[f32], n_cols: usize, out: &mut [f32]) {
    debug_assert_eq!(x4.len(), 4 * n_cols);
    debug_assert!(out.len() >= 4 * n_cols);
    let groups = n_cols / Q1_0_GROUP_SIZE;
    for g in 0..groups {
        for t in 0..4 {
            let src = t * n_cols + g * Q1_0_GROUP_SIZE;
            let dst = (g * 4 + t) * Q1_0_GROUP_SIZE;
            out[dst..dst + Q1_0_GROUP_SIZE]
                .copy_from_slice(&x4[src..src + Q1_0_GROUP_SIZE]);
        }
    }
}

/// ★ verify4t kernel: 交错布局版 (L1 命中优化)
///
/// x_int 必须由 `interleave_x4_verify` 生成: [group][token][128] 交错,
/// 每 group 512B 连续 —— 与 verify4 相比 x loads 从 L2 降为 L1。
///
/// # Safety
///
/// - 仅可在支持 `avx2+fma+f16c` 的 CPU 上调用 (见 `avx2_q1_0_available`), 否则触发 SIGILL。
/// - `n_cols` 须为 128 的倍数; `x_int` 为交错布局且 `x_int.len() >= 4 * n_cols`、
///   `y.len() >= 4` (debug 断言)。
/// - `data` 须至少覆盖 `(row_idx + 1) * groups_per_row * Q1_0_BLOCK_BYTES` 字节
///   (含对下一 group sign bytes 的 `_mm_prefetch` — prefetch 越界仅为提示, 不构成 UB)。
/// - `x_int` 与 `y` 不得重叠 (末尾一次性写 `y[0..4]`)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
#[inline]
pub unsafe fn dot_q1_0_row_verify4t_avx2(
    data: &[u8],
    row_idx: usize,
    n_cols: usize,
    x_int: &[f32],  // [4 * n_cols] 交错布局
    y: &mut [f32],  // [4]
) {
    use std::arch::x86_64::*;
    debug_assert!(x_int.len() >= 4 * n_cols);
    debug_assert!(y.len() >= 4);

    let groups_per_row = n_cols / Q1_0_GROUP_SIZE;
    let row_byte_offset = row_idx * (groups_per_row * Q1_0_BLOCK_BYTES);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let x_base = x_int.as_ptr();
    let gstride = 4 * Q1_0_GROUP_SIZE; // 512B per group

    for g in 0..groups_per_row {
        let block_start = row_byte_offset + g * Q1_0_BLOCK_BYTES;
        // ★ 软件预取: 下一个 group 的 sign bytes 提前 ~200c 拉入 L1,
        //   遮盖 DRAM 延迟 (权重流是本 kernel 唯一的 DRAM 访问)
        if g + 1 < groups_per_row {
            _mm_prefetch::<_MM_HINT_T0>(
                data.as_ptr().add(block_start + Q1_0_BLOCK_BYTES) as *const i8,
            );
        }
        let scale_bits = u16::from_le_bytes([
            *data.get_unchecked(block_start),
            *data.get_unchecked(block_start + 1),
        ]);
        let scale_v = _mm256_cvtph_ps(_mm_set1_epi16(scale_bits as i16));
        let sign_ptr = data.as_ptr().add(block_start + 2);
        let xg = x_base.add(g * gstride); // token t 在 xg + t*128

        // ★ 8 条独立 FMA 链 (每 token 2 链: 偶/奇 byte) + 手动全展开
        //   16 bytes 平铺消除循环开销, 编译器静态调度 loads/FMA 交错。
        let mut g0a = _mm256_setzero_ps();
        let mut g1a = _mm256_setzero_ps();
        let mut g0b = _mm256_setzero_ps();
        let mut g1b = _mm256_setzero_ps();
        let mut g0c = _mm256_setzero_ps();
        let mut g1c = _mm256_setzero_ps();
        let mut g0d = _mm256_setzero_ps();
        let mut g1d = _mm256_setzero_ps();
        // 手动展开的 8 轮 (原 (0..16).step_by(2))
        let mut r = 0usize;
        while r < 16 {
            let lut0 = _mm256_loadu_ps(SIGN_LUT[*sign_ptr.add(r) as usize].0.as_ptr());
            let lut1 = _mm256_loadu_ps(SIGN_LUT[*sign_ptr.add(r + 1) as usize].0.as_ptr());
            let off0 = r * 8;
            let off1 = (r + 1) * 8;
            let xa0 = _mm256_loadu_ps(xg.add(off0));
            let xa1 = _mm256_loadu_ps(xg.add(off1));
            let xb0 = _mm256_loadu_ps(xg.add(Q1_0_GROUP_SIZE + off0));
            let xb1 = _mm256_loadu_ps(xg.add(Q1_0_GROUP_SIZE + off1));
            let xc0 = _mm256_loadu_ps(xg.add(2 * Q1_0_GROUP_SIZE + off0));
            let xc1 = _mm256_loadu_ps(xg.add(2 * Q1_0_GROUP_SIZE + off1));
            let xd0 = _mm256_loadu_ps(xg.add(3 * Q1_0_GROUP_SIZE + off0));
            let xd1 = _mm256_loadu_ps(xg.add(3 * Q1_0_GROUP_SIZE + off1));
            g0a = _mm256_fmadd_ps(lut0, xa0, g0a);
            g1a = _mm256_fmadd_ps(lut1, xa1, g1a);
            g0b = _mm256_fmadd_ps(lut0, xb0, g0b);
            g1b = _mm256_fmadd_ps(lut1, xb1, g1b);
            g0c = _mm256_fmadd_ps(lut0, xc0, g0c);
            g1c = _mm256_fmadd_ps(lut1, xc1, g1c);
            g0d = _mm256_fmadd_ps(lut0, xd0, g0d);
            g1d = _mm256_fmadd_ps(lut1, xd1, g1d);
            r += 2;
        }
        // 8 链 merge → 4 token group acc → scale FMA → row acc
        acc0 = _mm256_fmadd_ps(scale_v, _mm256_add_ps(g0a, g1a), acc0);
        acc1 = _mm256_fmadd_ps(scale_v, _mm256_add_ps(g0b, g1b), acc1);
        acc2 = _mm256_fmadd_ps(scale_v, _mm256_add_ps(g0c, g1c), acc2);
        acc3 = _mm256_fmadd_ps(scale_v, _mm256_add_ps(g0d, g1d), acc3);
    }

    *y.get_unchecked_mut(0) = hsum_ps(acc0);
    *y.get_unchecked_mut(1) = hsum_ps(acc1);
    *y.get_unchecked_mut(2) = hsum_ps(acc2);
    *y.get_unchecked_mut(3) = hsum_ps(acc3);
}

// ============================================================================
// Q4_1 反量化与 GEMM (用于 DSpark drafter)
// ============================================================================
//
// Q4_1 布局 (每 32 权重 = 20 字节):
//   ┌──────────┬──────────┬─────────────────────────────┐
//   │ FP16 d   │ FP16 m   │ 16 字节 packed (32 × 4 bit) │
//   │ 2 字节    │ 2 字节    │                             │
//   └──────────┴──────────┴─────────────────────────────┘
// 反量化: w = m + d * q,  q ∈ [0, 15] (4-bit 无符号, 无 -8 偏移; -8 是 Q4_0)
//
// 相比 Q1_0 (1.125 bit/weight), Q4_1 是 5 bit/weight, 精度更高但带宽需求 4.4×。
// DSpark drafter 仅 6 层 × 5120 hidden, 总权重 ~200MB, 可接受。

/// 反量化 Q4_1 单行, 写入 caller 提供的 slice
pub fn dequantize_q4_1_row_into(data: &[u8], row_idx: usize, n_cols: usize, out: &mut [f32]) {
    debug_assert!(out.len() >= n_cols);
    let groups_per_row = n_cols.div_ceil(Q4_1_GROUP_SIZE);
    let row_byte_offset = row_idx * (groups_per_row * Q4_1_BLOCK_BYTES);
    let row_bytes = &data[row_byte_offset..];
    let mut out_idx = 0;
    for g in 0..groups_per_row {
        let bs = g * Q4_1_BLOCK_BYTES;
        if bs + Q4_1_BLOCK_BYTES > row_bytes.len() {
            break;
        }
        let d_bits = u16::from_le_bytes([row_bytes[bs], row_bytes[bs + 1]]);
        let m_bits = u16::from_le_bytes([row_bytes[bs + 2], row_bytes[bs + 3]]);
        let d = f16_to_f32_fast(d_bits);
        let m = f16_to_f32_fast(m_bits);
        let packed = &row_bytes[bs + 4..bs + Q4_1_BLOCK_BYTES];
        // llama.cpp Q4_1 nibble 布局: 低 nibble → 前半 0..15, 高 nibble → 后半 16..31
        for &b in packed {
            if out_idx < n_cols {
                out[out_idx] = m + d * (b & 0x0F) as f32;
                out_idx += 1;
            }
        }
        for &b in packed {
            if out_idx < n_cols {
                out[out_idx] = m + d * ((b >> 4) & 0x0F) as f32;
                out_idx += 1;
            }
        }
    }
    while out_idx < n_cols {
        out[out_idx] = 0.0;
        out_idx += 1;
    }
}

/// BF16 按行反量化: 读取 row_idx 行的 n_cols 个 BF16 元素, 转为 F32 写入 out
///
/// BF16 布局: sign(1) | exponent(8, bias=127) | mantissa(7) — 等于 f32 的高 16 位
pub fn dequantize_bf16_row_into(data: &[u8], row_idx: usize, n_cols: usize, out: &mut [f32]) {
    debug_assert!(out.len() >= n_cols);
    let row_byte_offset = row_idx * n_cols * 2;
    if row_byte_offset + n_cols * 2 > data.len() {
        for v in out[..n_cols].iter_mut() {
            *v = 0.0;
        }
        return;
    }
    for j in 0..n_cols {
        let bits = u16::from_le_bytes([
            data[row_byte_offset + j * 2],
            data[row_byte_offset + j * 2 + 1],
        ]);
        out[j] = f32::from_bits((bits as u32) << 16);
    }
}

/// BF16 matvec 标量实现: y = W_row · x
///
/// 用于 drafter 的 BF16 矩阵 (markov_w1, log_snr_fc1_w, log_snr_fc2_w)。
/// 大矩阵的多线程并行由 `DrafterMatrix::matvec_into_slice` 负责。
#[inline]
pub fn dot_bf16_row_scalar(data: &[u8], row_idx: usize, n_cols: usize, x: &[f32]) -> f32 {
    debug_assert!(x.len() >= n_cols);
    let row_byte_offset = row_idx * n_cols * 2;
    let mut acc = 0.0f32;
    if row_byte_offset + n_cols * 2 > data.len() {
        return 0.0;
    }
    for j in 0..n_cols {
        let bits = u16::from_le_bytes([
            data[row_byte_offset + j * 2],
            data[row_byte_offset + j * 2 + 1],
        ]);
        let w = f32::from_bits((bits as u32) << 16);
        acc += w * x[j];
    }
    acc
}

/// Q4_1 matvec 标量实现 (正确性优先, drafter 权重小性能不敏感)
///
/// y[i] = sum_g sum_{j=0..32} (m_g + d_g * q_j) * x[g*32 + j],  q_j ∈ [0, 15]
pub fn dot_q4_1_row_scalar(data: &[u8], row_idx: usize, n_cols: usize, x: &[f32]) -> f32 {
    debug_assert!(x.len() >= n_cols);
    let groups_per_row = n_cols / Q4_1_GROUP_SIZE;
    let row_byte_offset = row_idx * (groups_per_row * Q4_1_BLOCK_BYTES);

    let mut acc = 0.0f32;
    let mut w_buf = [0.0f32; 32];

    for g in 0..groups_per_row {
        let bs = row_byte_offset + g * Q4_1_BLOCK_BYTES;
        let d_bits = u16::from_le_bytes([data[bs], data[bs + 1]]);
        let m_bits = u16::from_le_bytes([data[bs + 2], data[bs + 3]]);
        let d = f16_to_f32_fast(d_bits);
        let m = f16_to_f32_fast(m_bits);
        let packed = &data[bs + 4..bs + Q4_1_BLOCK_BYTES];
        // llama.cpp Q4_1 nibble 布局: 低 nibble → 前半 0..15, 高 nibble → 后半 16..31
        // Q4_1 公式: w = d * q + m  (q 为 4-bit 值 0..15, 无 -8 偏移; -8 是 Q4_0 的)
        for byte_idx in 0..16 {
            let b = packed[byte_idx];
            w_buf[byte_idx] = m + d * (b & 0x0F) as f32;           // 低 nibble → 前半
            w_buf[byte_idx + 16] = m + d * ((b >> 4) & 0x0F) as f32;  // 高 nibble → 后半
        }
        let x_off = g * Q4_1_GROUP_SIZE;
        let mut group_acc = 0.0f32;
        for j in 0..32 {
            group_acc += w_buf[j] * x[x_off + j];
        }
        acc += group_acc;
    }
    acc
}

/// Q4_1 runtime AVX2 feature 检测
#[cfg(target_arch = "x86_64")]
pub fn avx2_q4_1_available() -> bool {
    std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
        && std::is_x86_feature_detected!("f16c")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_q4_1_available() -> bool {
    false
}

/// ★ Q4_1 AVX2 kernel — 8x 加速 vs scalar
///
/// 每个 Q4_1 block = 32 weights = 4 × __m256:
///   1. Load 16 bytes nibbles as __m128i
///   2. Low nibbles (AND 0x0F) → weights 0..15
///   3. High nibbles (SHR 4 + AND 0x0F) → weights 16..31
///   4. cvtepi8_epi32 (8 bytes → 8 × i32, 0..15 安全当 i8) → cvtepi32_ps → __m256 f32
///   5. w = m + d * nibble (FMA)
///   6. acc = w * x + acc (FMA)
///
/// 注: nibble 值 0..15 < 128, 用有符号 cvtepi8_epi32 安全 (u8 当 i8 解读不变)
///
/// # Safety
///
/// - 仅可在支持 `avx2+fma+f16c` 的 CPU 上调用 (见 `avx2_q4_1_available`), 否则触发 SIGILL。
/// - `n_cols` 须为 `Q4_1_GROUP_SIZE` (32) 的倍数 (整除截断, 余数列被忽略)。
/// - `data` 须至少覆盖 `(row_idx + 1) * groups_per_row * Q4_1_BLOCK_BYTES` 字节
///   (`groups_per_row = n_cols / 32`; 内部 `get_unchecked`/`loadu_si128`, 无边界检查)。
/// - `x` 须至少 `n_cols` 个元素 (全部 unaligned load, 无对齐要求)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
#[inline]
pub unsafe fn dot_q4_1_row_avx2(
    data: &[u8],
    row_idx: usize,
    n_cols: usize,
    x: &[f32],
) -> f32 {
    use std::arch::x86_64::*;

    let groups_per_row = n_cols / Q4_1_GROUP_SIZE;
    let row_byte_offset = row_idx * (groups_per_row * Q4_1_BLOCK_BYTES);

    let mut acc_vec = _mm256_setzero_ps();
    let nibble_mask = _mm_set1_epi8(0x0F);

    for g in 0..groups_per_row {
        let bs = row_byte_offset + g * Q4_1_BLOCK_BYTES;

        // Load d, m (f16) → broadcast to f32
        let d_bits = u16::from_le_bytes([*data.get_unchecked(bs), *data.get_unchecked(bs + 1)]);
        let m_bits = u16::from_le_bytes([*data.get_unchecked(bs + 2), *data.get_unchecked(bs + 3)]);
        let d_v = _mm256_cvtph_ps(_mm_set1_epi16(d_bits as i16));
        let m_v = _mm256_cvtph_ps(_mm_set1_epi16(m_bits as i16));

        // Load 16 bytes nibbles
        let nibbles = _mm_loadu_si128(data.as_ptr().add(bs + 4) as *const __m128i);
        // Low nibbles (bytes 0..15 → weights 0..15)
        let low = _mm_and_si128(nibbles, nibble_mask);
        // High nibbles (bytes 0..15 → weights 16..31)
        let high = _mm_and_si128(_mm_srli_epi16(nibbles, 4), nibble_mask);

        // Convert 16 bytes → 16 × i32 → 16 × f32 (4 个 __m256)
        // low[0..7] → weights 0..7
        let w0_n = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low));
        // low[8..15] → weights 8..15
        let low_hi = _mm_srli_si128(low, 8);
        let w1_n = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low_hi));
        // high[0..7] → weights 16..23
        let w2_n = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high));
        // high[8..15] → weights 24..31
        let high_hi = _mm_srli_si128(high, 8);
        let w3_n = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high_hi));

        // w = m + d * nibble (FMA: d * nibble + m)
        let w0 = _mm256_fmadd_ps(d_v, w0_n, m_v);
        let w1 = _mm256_fmadd_ps(d_v, w1_n, m_v);
        let w2 = _mm256_fmadd_ps(d_v, w2_n, m_v);
        let w3 = _mm256_fmadd_ps(d_v, w3_n, m_v);

        // Load x[32] as 4 × __m256
        let x_ptr = x.as_ptr().add(g * Q4_1_GROUP_SIZE);
        let x0 = _mm256_loadu_ps(x_ptr);
        let x1 = _mm256_loadu_ps(x_ptr.add(8));
        let x2 = _mm256_loadu_ps(x_ptr.add(16));
        let x3 = _mm256_loadu_ps(x_ptr.add(24));

        // acc += w * x (FMA)
        acc_vec = _mm256_fmadd_ps(w0, x0, acc_vec);
        acc_vec = _mm256_fmadd_ps(w1, x1, acc_vec);
        acc_vec = _mm256_fmadd_ps(w2, x2, acc_vec);
        acc_vec = _mm256_fmadd_ps(w3, x3, acc_vec);
    }

    // Horizontal sum __m256 → f32 (SSE)
    hsum_ps(acc_vec)
}

/// ★ Q4_1 batched AVX2 kernel — 同一 W[row] 与 n_batch 个 x 向量做点积
///
/// **设计** (P7 优化):
/// - 主循环 4 token 分块, 每 group 内 4 个 w 向量 (32 weights) 在 4 token 间共享,
///   节省 nibble unpack + d/m broadcast 各 75% (vs 逐 token)。
/// - 余数走 2-token + 1-token fallback。
///
/// 相比 `dot_q4_1_row_avx2` 调用 n_batch 次:
/// - nibble unpack (load+AND+SHR+cvtepi8+cvtepi32) 从 n_batch 降到 n_batch/4
/// - d/m F16C+broadcast 从 n_batch 降到 n_batch/4
/// - hsum 次数不变 (n_batch, 每 token 行末一次)
///
/// 寄存器分配 (4-token path): 4 w (shared) + 4 row_acc + 2 (d_v, m_v) + 4 x (transient) = 14/16 YMM
///
/// # Safety
///
/// - 仅可在支持 `avx2+fma+f16c` 的 CPU 上调用 (见 `avx2_q4_1_available`), 否则触发 SIGILL。
/// - `n_cols` 须为 32 的倍数 (整除截断, 余数列被忽略)。
/// - `data` 须至少覆盖 `(row_idx + 1) * groups_per_row * Q4_1_BLOCK_BYTES` 字节。
/// - `x` 行优先: 第 t 行始于 `x[t * x_stride]`, 每行须可读 `n_cols` 个元素
///   (`x.len() >= n_batch * x_stride`、`x_stride >= n_cols`)。
/// - `y` 写入 `y[t * y_stride + row_idx]`, 须满足 `y.len() >= n_batch * y_stride` 且
///   `row_idx < y_stride` (与安全包装 `dot_q4_1_row_batch` 的 debug 断言一致)。
/// - `x` 与 `y` 不得重叠。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)]
#[inline]
pub unsafe fn dot_q4_1_row_batch_avx2(
    data: &[u8],
    row_idx: usize,
    n_cols: usize,
    x: &[f32],
    x_stride: usize,
    n_batch: usize,
    y: &mut [f32],
    y_stride: usize,
) {
    use std::arch::x86_64::*;
    let groups_per_row = n_cols / Q4_1_GROUP_SIZE;
    let row_byte_offset = row_idx * (groups_per_row * Q4_1_BLOCK_BYTES);
    let nibble_mask = _mm_set1_epi8(0x0F);

    let mut t_start = 0usize;

    // ★ P7: 4-token 主循环 — 每 group 内 w unpack 只做 1 次, 共享给 4 个 token
    // 节省 50% unpack (vs 2-token), instruction count -28%, 权重带宽 -3.5%
    while t_start + 4 <= n_batch {
        let mut row_acc0 = _mm256_setzero_ps();
        let mut row_acc1 = _mm256_setzero_ps();
        let mut row_acc2 = _mm256_setzero_ps();
        let mut row_acc3 = _mm256_setzero_ps();

        for g in 0..groups_per_row {
            let bs = row_byte_offset + g * Q4_1_BLOCK_BYTES;

            // Load d, m (f16) → broadcast (shared across 4 tokens)
            let d_bits = u16::from_le_bytes([*data.get_unchecked(bs), *data.get_unchecked(bs + 1)]);
            let m_bits = u16::from_le_bytes([*data.get_unchecked(bs + 2), *data.get_unchecked(bs + 3)]);
            let d_v = _mm256_cvtph_ps(_mm_set1_epi16(d_bits as i16));
            let m_v = _mm256_cvtph_ps(_mm_set1_epi16(m_bits as i16));

            // Unpack nibbles → 4 w vectors (shared across 4 tokens)
            let nibbles = _mm_loadu_si128(data.as_ptr().add(bs + 4) as *const __m128i);
            let low = _mm_and_si128(nibbles, nibble_mask);
            let high = _mm_and_si128(_mm_srli_epi16(nibbles, 4), nibble_mask);

            let w0 = _mm256_fmadd_ps(d_v, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low)), m_v);
            let w1 = _mm256_fmadd_ps(d_v, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(low, 8))), m_v);
            let w2 = _mm256_fmadd_ps(d_v, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high)), m_v);
            let w3 = _mm256_fmadd_ps(d_v, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(high, 8))), m_v);

            // Token 0: load x[32], 4 FMA into row_acc0
            let x0_ptr = x.as_ptr().add(t_start * x_stride + g * Q4_1_GROUP_SIZE);
            row_acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x0_ptr), row_acc0);
            row_acc0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x0_ptr.add(8)), row_acc0);
            row_acc0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x0_ptr.add(16)), row_acc0);
            row_acc0 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x0_ptr.add(24)), row_acc0);

            // Token 1: same w, different x
            let x1_ptr = x.as_ptr().add((t_start + 1) * x_stride + g * Q4_1_GROUP_SIZE);
            row_acc1 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x1_ptr), row_acc1);
            row_acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x1_ptr.add(8)), row_acc1);
            row_acc1 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x1_ptr.add(16)), row_acc1);
            row_acc1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x1_ptr.add(24)), row_acc1);

            // Token 2
            let x2_ptr = x.as_ptr().add((t_start + 2) * x_stride + g * Q4_1_GROUP_SIZE);
            row_acc2 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x2_ptr), row_acc2);
            row_acc2 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x2_ptr.add(8)), row_acc2);
            row_acc2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x2_ptr.add(16)), row_acc2);
            row_acc2 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x2_ptr.add(24)), row_acc2);

            // Token 3
            let x3_ptr = x.as_ptr().add((t_start + 3) * x_stride + g * Q4_1_GROUP_SIZE);
            row_acc3 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x3_ptr), row_acc3);
            row_acc3 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x3_ptr.add(8)), row_acc3);
            row_acc3 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x3_ptr.add(16)), row_acc3);
            row_acc3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x3_ptr.add(24)), row_acc3);
        }

        *y.get_unchecked_mut(t_start * y_stride + row_idx) = hsum_ps(row_acc0);
        *y.get_unchecked_mut((t_start + 1) * y_stride + row_idx) = hsum_ps(row_acc1);
        *y.get_unchecked_mut((t_start + 2) * y_stride + row_idx) = hsum_ps(row_acc2);
        *y.get_unchecked_mut((t_start + 3) * y_stride + row_idx) = hsum_ps(row_acc3);

        t_start += 4;
    }

    // 余数: 2-token / 1-token fallback (n_batch % 4 ∈ {1, 2, 3})
    while t_start < n_batch {
        let has_pair = t_start + 1 < n_batch;

        let mut row_acc0 = _mm256_setzero_ps();
        let mut row_acc1 = _mm256_setzero_ps();

        for g in 0..groups_per_row {
            let bs = row_byte_offset + g * Q4_1_BLOCK_BYTES;

            let d_bits = u16::from_le_bytes([*data.get_unchecked(bs), *data.get_unchecked(bs + 1)]);
            let m_bits = u16::from_le_bytes([*data.get_unchecked(bs + 2), *data.get_unchecked(bs + 3)]);
            let d_v = _mm256_cvtph_ps(_mm_set1_epi16(d_bits as i16));
            let m_v = _mm256_cvtph_ps(_mm_set1_epi16(m_bits as i16));

            let nibbles = _mm_loadu_si128(data.as_ptr().add(bs + 4) as *const __m128i);
            let low = _mm_and_si128(nibbles, nibble_mask);
            let high = _mm_and_si128(_mm_srli_epi16(nibbles, 4), nibble_mask);

            let w0 = _mm256_fmadd_ps(d_v, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low)), m_v);
            let w1 = _mm256_fmadd_ps(d_v, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(low, 8))), m_v);
            let w2 = _mm256_fmadd_ps(d_v, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high)), m_v);
            let w3 = _mm256_fmadd_ps(d_v, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(high, 8))), m_v);

            let x0_ptr = x.as_ptr().add(t_start * x_stride + g * Q4_1_GROUP_SIZE);
            row_acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x0_ptr), row_acc0);
            row_acc0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x0_ptr.add(8)), row_acc0);
            row_acc0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x0_ptr.add(16)), row_acc0);
            row_acc0 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x0_ptr.add(24)), row_acc0);

            if has_pair {
                let x1_ptr = x.as_ptr().add((t_start + 1) * x_stride + g * Q4_1_GROUP_SIZE);
                row_acc1 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x1_ptr), row_acc1);
                row_acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x1_ptr.add(8)), row_acc1);
                row_acc1 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x1_ptr.add(16)), row_acc1);
                row_acc1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x1_ptr.add(24)), row_acc1);
            }
        }

        *y.get_unchecked_mut(t_start * y_stride + row_idx) = hsum_ps(row_acc0);
        if has_pair {
            *y.get_unchecked_mut((t_start + 1) * y_stride + row_idx) = hsum_ps(row_acc1);
        }

        t_start += if has_pair { 2 } else { 1 };
    }
}

/// Q4_1 batched matvec 入口 (runtime AVX2 检测 + scalar fallback)
///
#[allow(clippy::too_many_arguments)]
/// 计算 `y[t * y_stride + row_idx] = dot(W[row_idx], x[t * x_stride..t * x_stride + n_cols])`
/// 对 t ∈ 0..n_batch。同一 W 行被所有 token 共享 (只 unpack 一次)。
#[allow(unsafe_code)]
pub fn dot_q4_1_row_batch(
    data: &[u8],
    row_idx: usize,
    n_cols: usize,
    x: &[f32],
    x_stride: usize,
    n_batch: usize,
    y: &mut [f32],
    y_stride: usize,
) {
    debug_assert!(x.len() >= n_batch * x_stride);
    debug_assert!(y.len() >= n_batch * y_stride);
    debug_assert!(x_stride >= n_cols, "x_stride {x_stride} < n_cols {n_cols}");
    debug_assert!(row_idx < y_stride, "row_idx {row_idx} >= y_stride {y_stride}");
    if n_batch == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if avx2_q4_1_available() {
        #[allow(unsafe_code)]
        unsafe {
            dot_q4_1_row_batch_avx2(data, row_idx, n_cols, x, x_stride, n_batch, y, y_stride);
        }
        return;
    }
    // Fallback: 逐 token 调用 scalar
    for t in 0..n_batch {
        let xt = &x[t * x_stride..t * x_stride + n_cols];
        y[t * y_stride + row_idx] = dot_q4_1_row_scalar(data, row_idx, n_cols, xt);
    }
}

/// ★ Q8_0 量化-反量化: 模拟 llama.cpp 的 F32→Q8_0→F32 路径
///
/// llama.cpp 在 Q1_0/Q4_0/Q4_1 等量化 weight × F32 input 时, 会自动调用
/// `quantize_row_q8_0` 把 F32 input 量化为 Q8_0 (per-32-element block),
/// 然后用量化后的 Q8_0 做点积。这引入 ~0.5% 的量化误差。
///
/// Daiza 原本直接用 F32 input 做点积 (无量化误差), 导致 target hidden state
/// 比 llama.cpp "更精确", 但 drafter 是用 llama.cpp (Q8_0 path) 训练的,
/// 看到的 tap 分布与训练时不匹配, 导致接受率从 95% 降到 75%。
///
/// 本函数把 F32 x 量化为 Q8_0 再反量化回 F32, 引入与 llama.cpp 相同的量化误差,
/// 使 target tap 与 drafter 训练分布一致。
///
/// Q8_0 格式: 32 元素/block, `d = amax / 127`, `qs[i] = round(x[i] / d)`,
/// 反量化 `x' = qs[i] * d`。
pub fn quantize_dequantize_q8_0_into(x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert!(x.len().is_multiple_of(32), "Q8_0 requires len % 32 == 0, got {}", x.len());
    let n_blocks = x.len() / 32;
    for b in 0..n_blocks {
        let off = b * 32;
        let mut amax = 0.0f32;
        for j in 0..32 {
            let ax = x[off + j].abs();
            if ax > amax {
                amax = ax;
            }
        }
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for j in 0..32 {
            y[off + j] = (x[off + j] * id).round() * d;
        }
    }
}

/// 反量化 Q8_0 raw bytes → F32 vec (用于 mmproj ViT 权重加载)
///
/// Q8_0 格式 (per 32-element block):
/// ```text
/// ┌─────────────┬──────────────────────────┐
/// │ FP16 scale  │  32 个 int8 quantized值    │
/// │  2 字节     │  32 字节                  │
/// └─────────────┴──────────────────────────┘
/// 共 34 字节 / 32 元素
/// ```
///
/// 反量化: `x[i] = qs[i] * d` (d 从 FP16 scale 转换)
pub fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    const Q8_0_BLOCK_BYTES: usize = 34;
    const Q8_0_BLOCK_SIZE: usize = 32;
    let n_blocks = n_elements.div_ceil(Q8_0_BLOCK_SIZE);
    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let block_start = b * Q8_0_BLOCK_BYTES;
        if block_start + 2 > data.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let d = f16_to_f32(scale_bits);
        let qs_start = block_start + 2;
        for j in 0..Q8_0_BLOCK_SIZE {
            if out.len() >= n_elements {
                return out;
            }
            if qs_start + j >= data.len() {
                out.push(0.0);
                continue;
            }
            let q = data[qs_start + j] as i8 as f32; // int8 是 signed
            out.push(q * d);
        }
    }
    while out.len() < n_elements {
        out.push(0.0);
    }
    out
}

/// 反量化单行 Q8_0,写入 caller 提供的 slice
pub fn dequantize_q8_0_row_into(data: &[u8], row_idx: usize, n_cols: usize, out: &mut [f32]) {
    const Q8_0_BLOCK_BYTES: usize = 34;
    const Q8_0_BLOCK_SIZE: usize = 32;
    debug_assert!(out.len() >= n_cols);
    let blocks_per_row = n_cols.div_ceil(Q8_0_BLOCK_SIZE);
    let row_byte_offset = row_idx * (blocks_per_row * Q8_0_BLOCK_BYTES);
    let row_bytes = &data[row_byte_offset..];

    let mut out_idx = 0;
    for g in 0..blocks_per_row {
        let bs = g * Q8_0_BLOCK_BYTES;
        if bs + 2 > row_bytes.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([row_bytes[bs], row_bytes[bs + 1]]);
        let d = f16_to_f32(scale_bits);
        for j in 0..Q8_0_BLOCK_SIZE {
            if out_idx >= n_cols {
                return;
            }
            let off = bs + 2 + j;
            if off >= row_bytes.len() {
                out[out_idx] = 0.0;
            } else {
                let q = row_bytes[off] as i8 as f32;
                out[out_idx] = q * d;
            }
            out_idx += 1;
        }
    }
    while out_idx < n_cols {
        out[out_idx] = 0.0;
        out_idx += 1;
    }
}

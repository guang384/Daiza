//! 内核实验室: Q1_0 quad matvec 的 LUT 寻址变体 A/B
//!
//! 汇编实测 (dumpbin): 当前内核每 8 权重 = movzx(加载) + shl 5(port0, 与 FMA 抢端口)
//! + vmovups(LUT)。x86 寻址 scale 只有 {1,2,4,8}，32B 步长 LUT 必然带 1 个 shift。
//!
//! 两个零 ALU 逃逸口:
//!   V1 ptr-table: 2KB 指针表 [ptrs + b*8] (scale-8 免费) → 3 加载 0 ALU / 8 权重
//!   V4 u16 预移位: 加载时把符号字节预算成 (b<<3) 的 u16 (= LUT float 索引),
//!      内核 movzx word + [LUT + s*4] (scale-4 免费) → 2 加载 0 ALU / 8 权重
//!      代价: 符号区 RAM 翻倍 (3.1→6.2GB), 引擎侧可做加载时转换
//!
//! V0 = 引擎真实内核 dot_q1_0_row_quad_avx2 (基线)。
//! 所有变体 FMA 顺序与 V0 完全一致 → 结果应逐位相等 (correctness check)。
//!
//! Run: cargo build --release --example bench_klab && target/release/examples/bench_klab

use daiza_engine::model::workspace;
use daiza_engine::model::weights::Q1_0Matrix;
use std::time::Instant;

const ROWS: usize = 17408;
const COLS: usize = 5120;
const GROUP: usize = 128;
const BLOCK_BYTES: usize = 18;
const N_MATRICES: usize = 12;
const N_THREADS: usize = 14;
const N_ACTIVE: usize = 9; // E2E decode 工作点
const N_VARIANTS: usize = 5; // V0, V1, V4, V5, V5b

/// steal_chunk 尺寸 (env DAIZA_KLAB_CHUNK 可覆盖; 引擎 multi-matrix 路径用 256)
fn lab_chunk() -> usize {
    std::env::var("DAIZA_KLAB_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64)
}

fn make_random_q10(rows: usize, cols: usize) -> Q1_0Matrix {
    let groups = cols / GROUP;
    let mut bytes = vec![0u8; rows * groups * BLOCK_BYTES];
    let mut s: u64 = 0x9E3779B97F4A7C15;
    for chunk in bytes.chunks_exact_mut(BLOCK_BYTES) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let scale_f: f32 = 0.001 + (s & 0xFF) as f32 / 255.0 * 0.049;
        let scale_bits = scale_f.to_bits();
        let f16_bits: u16 = ((scale_bits >> 16) & 0x7FFF) as u16
            | (if scale_bits & 0x8000_0000 != 0 { 0x8000 } else { 0 });
        chunk[0] = (f16_bits & 0xFF) as u8;
        chunk[1] = (f16_bits >> 8) as u8;
        for b in chunk[2..18].iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *b = (s >> 32) as u8;
        }
    }
    Q1_0Matrix { bytes, rows, cols }
}

/// 符号字节 → u16 预移位数组: (b << 3) = LUT8K 的 float 索引 (b*8)
/// 每行 groups*16 项, 布局与 bytes 的符号区一一对应
fn make_signs16(bytes: &[u8], rows: usize, cols: usize) -> Vec<u16> {
    let groups = cols / GROUP;
    let mut out = vec![0u16; rows * groups * 16];
    for r in 0..rows {
        for g in 0..groups {
            let base = r * groups * BLOCK_BYTES + g * BLOCK_BYTES + 2;
            let dst = (r * groups + g) * 16;
            for b in 0..16 {
                out[dst + b] = (bytes[base + b] as u16) << 3;
            }
        }
    }
    out
}

/// 平铺 LUT: [f32; 256*8], entry b 在 float 偏移 b*8 (= byte 偏移 b*32)
/// 值与引擎 SIGN_LUT 完全一致: bit=1 → +1.0, bit=0 → -1.0
static LUT8K: [f32; 2048] = {
    let mut lut = [0.0f32; 2048];
    let mut b = 0;
    while b < 256 {
        let mut bit = 0;
        while bit < 8 {
            lut[b * 8 + bit] = if (b >> bit) & 1 == 1 { 1.0 } else { -1.0 };
            bit += 1;
        }
        b += 1;
    }
    lut
};

// ---------------------------------------------------------------------------
// V1: ptr-table quad (零 ALU, 3 加载/8 权重)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
unsafe fn quad_v1(
    bytes: &[u8],
    ptrs: &[*const f32; 256],
    r0: usize,
    r1: usize,
    r2: usize,
    r3: usize,
    n_cols: usize,
    x: &[f32],
) -> (f32, f32, f32, f32) {
    use std::arch::x86_64::*;
    let groups = n_cols / GROUP;
    let stride = groups * BLOCK_BYTES;
    let (off0, off1, off2, off3) = (r0 * stride, r1 * stride, r2 * stride, r3 * stride);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    for g in 0..groups {
        let bs0 = off0 + g * BLOCK_BYTES;
        let bs1 = off1 + g * BLOCK_BYTES;
        let bs2 = off2 + g * BLOCK_BYTES;
        let bs3 = off3 + g * BLOCK_BYTES;

        let scale0 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs0), *bytes.get_unchecked(bs0 + 1)]) as i16,
        ));
        let scale1 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs1), *bytes.get_unchecked(bs1 + 1)]) as i16,
        ));
        let scale2 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs2), *bytes.get_unchecked(bs2 + 1)]) as i16,
        ));
        let scale3 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs3), *bytes.get_unchecked(bs3 + 1)]) as i16,
        ));

        let sp0 = bytes.as_ptr().add(bs0 + 2);
        let sp1 = bytes.as_ptr().add(bs1 + 2);
        let sp2 = bytes.as_ptr().add(bs2 + 2);
        let sp3 = bytes.as_ptr().add(bs3 + 2);
        let xp = x.as_ptr().add(g * GROUP);

        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut c0 = _mm256_setzero_ps();
        let mut c1 = _mm256_setzero_ps();
        let mut d0 = _mm256_setzero_ps();
        let mut d1 = _mm256_setzero_ps();

        for byte_idx in (0..16).step_by(2) {
            let x0 = _mm256_loadu_ps(xp.add(byte_idx * 8));
            let x1 = _mm256_loadu_ps(xp.add((byte_idx + 1) * 8));

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(*ptrs.get_unchecked(*sp0.add(byte_idx) as usize), ), x0, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(*ptrs.get_unchecked(*sp0.add(byte_idx + 1) as usize)), x1, a1);
            b0 = _mm256_fmadd_ps(_mm256_loadu_ps(*ptrs.get_unchecked(*sp1.add(byte_idx) as usize)), x0, b0);
            b1 = _mm256_fmadd_ps(_mm256_loadu_ps(*ptrs.get_unchecked(*sp1.add(byte_idx + 1) as usize)), x1, b1);
            c0 = _mm256_fmadd_ps(_mm256_loadu_ps(*ptrs.get_unchecked(*sp2.add(byte_idx) as usize)), x0, c0);
            c1 = _mm256_fmadd_ps(_mm256_loadu_ps(*ptrs.get_unchecked(*sp2.add(byte_idx + 1) as usize)), x1, c1);
            d0 = _mm256_fmadd_ps(_mm256_loadu_ps(*ptrs.get_unchecked(*sp3.add(byte_idx) as usize)), x0, d0);
            d1 = _mm256_fmadd_ps(_mm256_loadu_ps(*ptrs.get_unchecked(*sp3.add(byte_idx + 1) as usize)), x1, d1);
        }

        let g0 = _mm256_add_ps(a0, a1);
        let g1 = _mm256_add_ps(b0, b1);
        let g2 = _mm256_add_ps(c0, c1);
        let g3 = _mm256_add_ps(d0, d1);
        acc0 = _mm256_fmadd_ps(scale0, g0, acc0);
        acc1 = _mm256_fmadd_ps(scale1, g1, acc1);
        acc2 = _mm256_fmadd_ps(scale2, g2, acc2);
        acc3 = _mm256_fmadd_ps(scale3, g3, acc3);
    }

    (
        hsum(acc0),
        hsum(acc1),
        hsum(acc2),
        hsum(acc3),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
#[inline]
unsafe fn hsum(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s2 = _mm_movehdup_ps(s);
    let s = _mm_add_ps(s, s2);
    let s3 = _mm_movehl_ps(s, s);
    let s = _mm_add_ss(s, s3);
    _mm_cvtss_f32(s)
}

// ---------------------------------------------------------------------------
// V4: u16 预移位符号 quad (零 ALU, 2 加载/8 权重)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
unsafe fn quad_v4(
    bytes: &[u8],
    signs16: &[u16],
    r0: usize,
    r1: usize,
    r2: usize,
    r3: usize,
    n_cols: usize,
    x: &[f32],
) -> (f32, f32, f32, f32) {
    use std::arch::x86_64::*;
    let groups = n_cols / GROUP;
    let stride = groups * BLOCK_BYTES;
    let s16_stride = groups * 16;
    let (off0, off1, off2, off3) = (r0 * stride, r1 * stride, r2 * stride, r3 * stride);
    let (s0, s1, s2, s3) = (r0 * s16_stride, r1 * s16_stride, r2 * s16_stride, r3 * s16_stride);

    let lut = LUT8K.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    for g in 0..groups {
        let bs0 = off0 + g * BLOCK_BYTES;
        let bs1 = off1 + g * BLOCK_BYTES;
        let bs2 = off2 + g * BLOCK_BYTES;
        let bs3 = off3 + g * BLOCK_BYTES;

        let scale0 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs0), *bytes.get_unchecked(bs0 + 1)]) as i16,
        ));
        let scale1 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs1), *bytes.get_unchecked(bs1 + 1)]) as i16,
        ));
        let scale2 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs2), *bytes.get_unchecked(bs2 + 1)]) as i16,
        ));
        let scale3 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs3), *bytes.get_unchecked(bs3 + 1)]) as i16,
        ));

        let q0 = signs16.as_ptr().add(s0 + g * 16);
        let q1 = signs16.as_ptr().add(s1 + g * 16);
        let q2 = signs16.as_ptr().add(s2 + g * 16);
        let q3 = signs16.as_ptr().add(s3 + g * 16);
        let xp = x.as_ptr().add(g * GROUP);

        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut c0 = _mm256_setzero_ps();
        let mut c1 = _mm256_setzero_ps();
        let mut d0 = _mm256_setzero_ps();
        let mut d1 = _mm256_setzero_ps();

        for byte_idx in (0..16).step_by(2) {
            let x0 = _mm256_loadu_ps(xp.add(byte_idx * 8));
            let x1 = _mm256_loadu_ps(xp.add((byte_idx + 1) * 8));

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(lut.add(*q0.add(byte_idx) as usize)), x0, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(lut.add(*q0.add(byte_idx + 1) as usize)), x1, a1);
            b0 = _mm256_fmadd_ps(_mm256_loadu_ps(lut.add(*q1.add(byte_idx) as usize)), x0, b0);
            b1 = _mm256_fmadd_ps(_mm256_loadu_ps(lut.add(*q1.add(byte_idx + 1) as usize)), x1, b1);
            c0 = _mm256_fmadd_ps(_mm256_loadu_ps(lut.add(*q2.add(byte_idx) as usize)), x0, c0);
            c1 = _mm256_fmadd_ps(_mm256_loadu_ps(lut.add(*q2.add(byte_idx + 1) as usize)), x1, c1);
            d0 = _mm256_fmadd_ps(_mm256_loadu_ps(lut.add(*q3.add(byte_idx) as usize)), x0, d0);
            d1 = _mm256_fmadd_ps(_mm256_loadu_ps(lut.add(*q3.add(byte_idx + 1) as usize)), x1, d1);
        }

        let g0 = _mm256_add_ps(a0, a1);
        let g1 = _mm256_add_ps(b0, b1);
        let g2 = _mm256_add_ps(c0, c1);
        let g3 = _mm256_add_ps(d0, d1);
        acc0 = _mm256_fmadd_ps(scale0, g0, acc0);
        acc1 = _mm256_fmadd_ps(scale1, g1, acc1);
        acc2 = _mm256_fmadd_ps(scale2, g2, acc2);
        acc3 = _mm256_fmadd_ps(scale3, g3, acc3);
    }

    (hsum(acc0), hsum(acc1), hsum(acc2), hsum(acc3))
}

// ---------------------------------------------------------------------------
// V5: 边界折叠 quad (setzero 消除 + merge 折叠)
//   每组: 8 setzero + 8 vadd(merge) + 4 scale-FMA → 0 setzero + 0 merge + 8 scale-FMA
//   链首 FMA 以零常量为 addend, 边界串行依赖缩短
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
unsafe fn quad_v5(
    bytes: &[u8],
    r0: usize,
    r1: usize,
    r2: usize,
    r3: usize,
    n_cols: usize,
    x: &[f32],
) -> (f32, f32, f32, f32) {
    use std::arch::x86_64::*;
    let groups = n_cols / GROUP;
    let stride = groups * BLOCK_BYTES;
    let (off0, off1, off2, off3) = (r0 * stride, r1 * stride, r2 * stride, r3 * stride);

    let zero = _mm256_setzero_ps();
    let mut acc0 = zero;
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;

    for g in 0..groups {
        let bs0 = off0 + g * BLOCK_BYTES;
        let bs1 = off1 + g * BLOCK_BYTES;
        let bs2 = off2 + g * BLOCK_BYTES;
        let bs3 = off3 + g * BLOCK_BYTES;

        let scale0 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs0), *bytes.get_unchecked(bs0 + 1)]) as i16,
        ));
        let scale1 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs1), *bytes.get_unchecked(bs1 + 1)]) as i16,
        ));
        let scale2 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs2), *bytes.get_unchecked(bs2 + 1)]) as i16,
        ));
        let scale3 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs3), *bytes.get_unchecked(bs3 + 1)]) as i16,
        ));

        let sp0 = bytes.as_ptr().add(bs0 + 2);
        let sp1 = bytes.as_ptr().add(bs1 + 2);
        let sp2 = bytes.as_ptr().add(bs2 + 2);
        let sp3 = bytes.as_ptr().add(bs3 + 2);
        let xp = x.as_ptr().add(g * GROUP);

        // 链以 zero 常量起步; 每链 8 次 FMA; 组末直接双 FMA 折叠 (无 merge-add)
        let mut a0 = zero;
        let mut a1 = zero;
        let mut b0 = zero;
        let mut b1 = zero;
        let mut c0 = zero;
        let mut c1 = zero;
        let mut d0 = zero;
        let mut d1 = zero;

        for byte_idx in (0..16).step_by(2) {
            let x0 = _mm256_loadu_ps(xp.add(byte_idx * 8));
            let x1 = _mm256_loadu_ps(xp.add((byte_idx + 1) * 8));

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp0.add(byte_idx) as usize) * 8)), x0, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp0.add(byte_idx + 1) as usize) * 8)), x1, a1);
            b0 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp1.add(byte_idx) as usize) * 8)), x0, b0);
            b1 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp1.add(byte_idx + 1) as usize) * 8)), x1, b1);
            c0 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp2.add(byte_idx) as usize) * 8)), x0, c0);
            c1 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp2.add(byte_idx + 1) as usize) * 8)), x1, c1);
            d0 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp3.add(byte_idx) as usize) * 8)), x0, d0);
            d1 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp3.add(byte_idx + 1) as usize) * 8)), x1, d1);
        }

        // 组末: 直接以 scale 折叠进持久 acc (省 8 vadd + 8 setzero)
        acc0 = _mm256_fmadd_ps(scale0, a0, acc0);
        acc0 = _mm256_fmadd_ps(scale0, a1, acc0);
        acc1 = _mm256_fmadd_ps(scale1, b0, acc1);
        acc1 = _mm256_fmadd_ps(scale1, b1, acc1);
        acc2 = _mm256_fmadd_ps(scale2, c0, acc2);
        acc2 = _mm256_fmadd_ps(scale2, c1, acc2);
        acc3 = _mm256_fmadd_ps(scale3, d0, acc3);
        acc3 = _mm256_fmadd_ps(scale3, d1, acc3);
    }

    (hsum(acc0), hsum(acc1), hsum(acc2), hsum(acc3))
}

// ---------------------------------------------------------------------------
// V5b: 仅消除 setzero (链首以零常量起步), 保留 merge 结构 → 逐位一致
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
unsafe fn quad_v5b(
    bytes: &[u8],
    r0: usize,
    r1: usize,
    r2: usize,
    r3: usize,
    n_cols: usize,
    x: &[f32],
) -> (f32, f32, f32, f32) {
    use std::arch::x86_64::*;
    let groups = n_cols / GROUP;
    let stride = groups * BLOCK_BYTES;
    let (off0, off1, off2, off3) = (r0 * stride, r1 * stride, r2 * stride, r3 * stride);

    let zero = _mm256_setzero_ps();
    let mut acc0 = zero;
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;

    for g in 0..groups {
        let bs0 = off0 + g * BLOCK_BYTES;
        let bs1 = off1 + g * BLOCK_BYTES;
        let bs2 = off2 + g * BLOCK_BYTES;
        let bs3 = off3 + g * BLOCK_BYTES;

        let scale0 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs0), *bytes.get_unchecked(bs0 + 1)]) as i16,
        ));
        let scale1 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs1), *bytes.get_unchecked(bs1 + 1)]) as i16,
        ));
        let scale2 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs2), *bytes.get_unchecked(bs2 + 1)]) as i16,
        ));
        let scale3 = _mm256_cvtph_ps(_mm_set1_epi16(
            u16::from_le_bytes([*bytes.get_unchecked(bs3), *bytes.get_unchecked(bs3 + 1)]) as i16,
        ));

        let sp0 = bytes.as_ptr().add(bs0 + 2);
        let sp1 = bytes.as_ptr().add(bs1 + 2);
        let sp2 = bytes.as_ptr().add(bs2 + 2);
        let sp3 = bytes.as_ptr().add(bs3 + 2);
        let xp = x.as_ptr().add(g * GROUP);

        let mut a0 = zero;
        let mut a1 = zero;
        let mut b0 = zero;
        let mut b1 = zero;
        let mut c0 = zero;
        let mut c1 = zero;
        let mut d0 = zero;
        let mut d1 = zero;

        for byte_idx in (0..16).step_by(2) {
            let x0 = _mm256_loadu_ps(xp.add(byte_idx * 8));
            let x1 = _mm256_loadu_ps(xp.add((byte_idx + 1) * 8));

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp0.add(byte_idx) as usize) * 8)), x0, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp0.add(byte_idx + 1) as usize) * 8)), x1, a1);
            b0 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp1.add(byte_idx) as usize) * 8)), x0, b0);
            b1 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp1.add(byte_idx + 1) as usize) * 8)), x1, b1);
            c0 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp2.add(byte_idx) as usize) * 8)), x0, c0);
            c1 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp2.add(byte_idx + 1) as usize) * 8)), x1, c1);
            d0 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp3.add(byte_idx) as usize) * 8)), x0, d0);
            d1 = _mm256_fmadd_ps(_mm256_loadu_ps(LUT8K.as_ptr().add((*sp3.add(byte_idx + 1) as usize) * 8)), x1, d1);
        }

        let g0 = _mm256_add_ps(a0, a1);
        let g1 = _mm256_add_ps(b0, b1);
        let g2 = _mm256_add_ps(c0, c1);
        let g3 = _mm256_add_ps(d0, d1);
        acc0 = _mm256_fmadd_ps(scale0, g0, acc0);
        acc1 = _mm256_fmadd_ps(scale1, g1, acc1);
        acc2 = _mm256_fmadd_ps(scale2, g2, acc2);
        acc3 = _mm256_fmadd_ps(scale3, g3, acc3);
    }

    (hsum(acc0), hsum(acc1), hsum(acc2), hsum(acc3))
}

fn main() {
    workspace::init_thread_pool(N_THREADS);
    if let Some(p) = workspace::get_thread_pool() {
        p.set_active_workers(N_ACTIVE);
    }

    let mats: Vec<Q1_0Matrix> = (0..N_MATRICES)
        .map(|i| make_random_q10(ROWS, COLS + i))
        .collect();
    let signs: Vec<Vec<u16>> = mats
        .iter()
        .map(|m| make_signs16(&m.bytes, m.rows, m.cols))
        .collect();
    let ptrs_table: Vec<[*const f32; 256]> = mats
        .iter()
        .map(|_| {
            let mut t: [*const f32; 256] = [LUT8K.as_ptr(); 256];
            for (b, slot) in t.iter_mut().enumerate() {
                *slot = unsafe { LUT8K.as_ptr().add(b * 8) };
            }
            t
        })
        .collect();

    let x = vec![0.017f32; COLS + N_MATRICES];
    let mut y = vec![0f32; ROWS];

    let n_active = workspace::active_workers();
    println!("lab: {N_MATRICES} mats x {ROWS}x{COLS}, workers={n_active} (pool {N_THREADS}), steal_chunk={}", lab_chunk());

    // ---------- correctness: V1/V4 vs V0 (引擎真实内核) ----------
    unsafe {
        let m = &mats[0];
        let s16 = &signs[0];
        let pt = &ptrs_table[0];
        let k = m.cols;
        let xm = &x[..k];
        let mut max_e1 = 0f32;
        let mut max_e4 = 0f32;
        let mut max_e5 = 0f32;
        let mut max_e5b = 0f32;
        for r in (0..m.rows).step_by(4) {
            let v0 = daiza_engine::tensor::quant::dot_q1_0_row_quad_avx2(&m.bytes, r, r + 1, r + 2, r + 3, k, xm);
            let v1 = quad_v1(&m.bytes, pt, r, r + 1, r + 2, r + 3, k, xm);
            let v4 = quad_v4(&m.bytes, s16, r, r + 1, r + 2, r + 3, k, xm);
            let v5 = quad_v5(&m.bytes, r, r + 1, r + 2, r + 3, k, xm);
            let v5b = quad_v5b(&m.bytes, r, r + 1, r + 2, r + 3, k, xm);
            for i in 0..4 {
                let (t0, t1, t2, t3) = v0;
                let (u0, u1, u2, u3) = v1;
                let (w0, w1, w2, w3) = v4;
                let r0 = [t0, t1, t2, t3];
                let r1 = [u0, u1, u2, u3];
                let r4 = [w0, w1, w2, w3];
                let (z0, z1, z2, z3) = v5;
                let r5 = [z0, z1, z2, z3];
                let (q0, q1, q2, q3) = v5b;
                let r6 = [q0, q1, q2, q3];
                max_e1 = max_e1.max((r0[i] - r1[i]).abs());
                max_e4 = max_e4.max((r0[i] - r4[i]).abs());
                max_e5 = max_e5.max((r0[i] - r5[i]).abs());
                max_e5b = max_e5b.max((r0[i] - r6[i]).abs());
            }
        }
        println!("correctness: V1={max_e1:.8} V4={max_e4:.8} V5={max_e5:.8} V5b={max_e5b:.8} (V5b expect 0)");
    }

    // ---------- interleaved timing ----------
    let rounds = 15;
    let warmup = 3;
    let mut times: Vec<[f64; N_VARIANTS]> = Vec::new();

    for round in 0..rounds {
        let mut row_t = [0f64; N_VARIANTS];
        for (v, slot) in row_t.iter_mut().enumerate() {
            let t = Instant::now();
            for mi in 0..N_MATRICES {
                dispatch_variant(v, &mats, &signs, &ptrs_table, mi, &x, &mut y, lab_chunk());
            }
            *slot = t.elapsed().as_secs_f64() * 1000.0;
        }
        if round >= warmup {
            times.push(row_t);
        }
    }

    let names = ["V0 engine-quad", "V1 ptr-table ", "V4 u16-presign", "V5 boundary-fold", "V5b zero-start "];
    let mut med = [0f64; N_VARIANTS];
    for (v, m) in med.iter_mut().enumerate() {
        let mut ts: Vec<f64> = times.iter().map(|t| t[v]).collect();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        *m = ts[ts.len() / 2];
    }
    println!("\n{:<16} {:>10} {:>12} {:>10}", "variant", "ms/rot", "ms/matvec", "GMAC/s");
    for (v, &m) in med.iter().enumerate() {
        let mm = m / N_MATRICES as f64;
        let gmacs = (ROWS as f64 * COLS as f64) / mm / 1000.0;
        println!("{:<16} {:>10.2} {:>12.3} {:>10.0}", names[v], m, mm, gmacs);
    }
    let base = med[0];
    for v in 1..N_VARIANTS {
        println!("{}: {:+.1}% vs V0", names[v].trim(), (base / med[v] - 1.0) * 100.0);
    }

    // ---------- 同进程交错 chunk A/B (消除热漂移) ----------
    // 只测 V0 (引擎真实内核), 交替 chunk=256/128, 每轮配对取差异
    println!("\n── interleaved chunk A/B (V0 only, same-process) ──");
    let ab_rounds = 20;
    let ab_warmup = 4;
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for round in 0..ab_rounds {
        let mut t256 = 0f64;
        let mut t128 = 0f64;
        // 交替: 偶数轮先 256, 奇数轮先 128 (消除"前一配置残留"偏向)
        let order = if round % 2 == 0 { [256, 128] } else { [128, 256] };
        for &ch in &order {
            let t = Instant::now();
            for mi in 0..N_MATRICES {
                dispatch_variant(0, &mats, &signs, &ptrs_table, mi, &x, &mut y, ch);
            }
            let dt = t.elapsed().as_secs_f64() * 1000.0;
            if ch == 256 { t256 = dt; } else { t128 = dt; }
        }
        if round >= ab_warmup {
            pairs.push((t256, t128));
        }
    }
    // 配对差异 (每轮 t256 - t128), 取中位
    let mut diffs: Vec<f64> = pairs.iter().map(|(a, b)| a - b).collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_diff = diffs[diffs.len() / 2];
    let mut t256s: Vec<f64> = pairs.iter().map(|(a, _)| *a).collect();
    t256s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut t128s: Vec<f64> = pairs.iter().map(|(_, b)| *b).collect();
    t128s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med256 = t256s[t256s.len() / 2];
    let med128 = t128s[t128s.len() / 2];
    println!("chunk=256: median {med256:.2} ms/rot ({:.3} ms/matvec)", med256 / N_MATRICES as f64);
    println!("chunk=128: median {med128:.2} ms/rot ({:.3} ms/matvec)", med128 / N_MATRICES as f64);
    println!("paired diff (256-128) median: {med_diff:.3} ms ({:+.1}% per-matvec)",
        med_diff / med256 * 100.0);
    println!("({} paired rounds)", pairs.len());
}

#[allow(clippy::too_many_arguments)]
fn dispatch_variant(
    v: usize,
    mats: &[Q1_0Matrix],
    signs: &[Vec<u16>],
    ptrs: &[[*const f32; 256]],
    mi: usize,
    x: &[f32],
    y: &mut [f32],
    chunk: usize,
) {
    let m = &mats[mi];
    let k = m.cols;
    let rows = m.rows;
    let bytes_addr = m.bytes.as_ptr() as usize;
    let x_addr = x.as_ptr() as usize;
    let y_addr = y.as_mut_ptr() as usize;
    let s16_addr = signs[mi].as_ptr() as usize;
    let ptrs_addr = ptrs[mi].as_ptr() as usize;

    if let Some(pool) = workspace::get_thread_pool() {
        pool.scatter_wait_stealing(rows, chunk, move |lo, hi| {
            let bytes = unsafe { std::slice::from_raw_parts(bytes_addr as *const u8, rows * (k / GROUP) * BLOCK_BYTES) };
            let x = unsafe { std::slice::from_raw_parts(x_addr as *const f32, k) };
            let y = unsafe { std::slice::from_raw_parts_mut(y_addr as *mut f32, rows) };
            let s16: &[u16] = unsafe { std::slice::from_raw_parts(s16_addr as *const u16, rows * (k / GROUP) * 16) };
            let pt: &[*const f32; 256] = unsafe { &*(ptrs_addr as *const [*const f32; 256]) };
            let mut i = lo;
            while i + 3 < hi {
                unsafe {
                    let (y0, y1, y2, y3) = match v {
                        0 => daiza_engine::tensor::quant::dot_q1_0_row_quad_avx2(bytes, i, i + 1, i + 2, i + 3, k, x),
                        1 => quad_v1(bytes, pt, i, i + 1, i + 2, i + 3, k, x),
                        2 => quad_v4(bytes, s16, i, i + 1, i + 2, i + 3, k, x),
                        3 => quad_v5(bytes, i, i + 1, i + 2, i + 3, k, x),
                        _ => quad_v5b(bytes, i, i + 1, i + 2, i + 3, k, x),
                    };
                    y[i] = y0;
                    y[i + 1] = y1;
                    y[i + 2] = y2;
                    y[i + 3] = y3;
                }
                i += 4;
            }
            while i < hi {
                // 尾行: 逐行调用对应变体 (单行走 V0 引擎内核保持简单, 仅影响 <0.1%)
                unsafe {
                    y[i] = daiza_engine::tensor::quant::dot_q1_0_row_avx2(bytes, i, k, x);
                }
                i += 1;
            }
        });
    }
}

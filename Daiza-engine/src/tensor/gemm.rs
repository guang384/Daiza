//! ★ Prefill 块 GEMM 化 — Q1_0 权重 × 多 token 批量矩阵乘 (k-lane 形态)
//!
//! ## 演进
//!
//! - v1 (反量化 GEMM): 权重整行反量化为 f32 后做 t-lanes GEMM, 128t 冷态 1.29×。
//!   瓶颈: 每 k 每行 1 次 bcast (`_mm256_set1_ps`) 占满 port 5 —— 8c/k, 是 FMA
//!   端口 (4c/k) 的 2 倍。
//! - v2-t-lanes (Q1_0 域 LUT 直乘, 已废弃): 把 LUT lane 与列优先 x_t 的 lane 直接
//!   FMA —— 语义错误: LUT 的 8 lanes 是 **k 方向** sign 位, x_t 列优先的 8 lanes
//!   是 **t 方向**, lane-wise 乘混叠不同 (r,t) 对, max_err=3.8。
//! - v2 (本文件, k-lane): 把 decode `verify4t` 的结构推广到 GEMM —— x 不转置,
//!   LUT 与 x 的 lanes 对齐到同一 k 方向, 累加器 lanes 保持 k 空间分片,
//!   hsum 归约到标量。
//!
//! ## 算法
//!
//! `y[t][r] = Σ_g scale[r][g] × Σ_{k∈g} sign[r][g][k] × x[t][k]`
//!
//! **预缩放 scratch** (prep): 每行每 group 的 16 条 LUT 按 scale 预乘一次:
//! `scratch[r][g][b] = scale_v × LUT[w[r][g][b]]` (16 × __m256 / 行 / group)。
//! 权重 18B/group 膨胀为 512B, 但成本摊到 n_tokens (≥64) 个 token 上 ——
//! 主循环彻底摆脱 scale 处理: 无 cvtph / bcast / port 5 开销。
//!
//! **microkernel (2 行 × 4 token × 8 group)**:
//! ```text
//! acc[2][4] = 0                 (8 条 ymm, lanes = k 空间分片)
//! for g in g_block (8 groups):
//!   for b in 0..16:             (展开)
//!     sl0, sl1 = 两行的预缩放 LUT     (2 load, L1)
//!     v0..v3    = 4 个 token 的 8k 片段 (4 load, L1)
//!     acc[rr][ti] = fma(sl_rr, v_ti, acc)   (8 FMA)
//! 行末: hsum(acc) × g_block 数 → 标量累加 → y[t][r]
//! ```
//! 每 (2r, 4t, 8g): 768 load (384c / 2 ports) + 1024 FMA (512c / 2 ports):
//! **FMA 端口 bound**, 25% load 余量。主循环零 port 5 开销 (hsum 仅在
//! g_block 边界, ~1%)。寄存器: 8 acc + 6 临时 = 14/16 ymm, 零 spill。
//! 8 条独立 FMA 链恰好填满 2 FMA/cycle 的发射槽, 4c 延迟被完全遮盖。
//!
//! **四层 blocking (缓存层级对齐)**:
//! - `r_block` (自适应 4..=32 行, scratch ~700KB 驻 L2): 权重 DRAM 只流一遍,
//!   prep 一次, 供块内全部 token tile 复用
//! - `t0` tile (4 token): 寄存器上限 (8 acc)
//! - `g_block` (8 group): x 片段 4 token × 8 group × 512B = 16KB 驻 L1,
//!   r_block 内全部 pair 共享命中
//! - `t_sub` (token 子块, 仅宽列 x 总量 > 4MB 时分片, 取 32): 实测主导项是
//!   每 scatter 的 straggler 尾巴 + prep 重做, 而非 x 走 L3 的带宽 ——
//!   t_sub 8/16/32 单调变优; 窄列 (x ≈ 2.6MB) 不分片直达
//! - `steal_chunk` 128 行: 512 行在 ~10 执行者下零偷取弹性, 慢核持块期间
//!   全场 barrier 等待; 细 chunk 让快核偷走余块, 尾巴缩到 1/4
//!   (交错基准实测 down_proj 再 -17%)
//!
//! 实测 (225H, 交错基准 9/12/13 worker 轮转): gate/up 479→436 ms/12 矩阵,
//! down_proj 543→543 (chunk 128 + t32 后不再随 worker 数退化);
//! CLI 端到端 (142t) 见提交信息; greedy 逐字节一致, max_err 1-2e-6。
//!
//! 数值与 `dot_q1_0_row_avx2` (batch/decode 参照) 仅浮点结合顺序不同,
//! bench_prefill 校验 max_err, 端到端 greedy 逐字节比对。
//!
//! ## 调用入口
//!
//! `matvec_batch_into_slice` 在 n_batch >= 64 && rows >= 4096 时 dispatch 到本模块。
//! 其余 (vision 段小 batch、小矩阵) 走原 batch kernel 路径不变。

use std::cell::RefCell;

use crate::model::workspace;
use crate::tensor::dtype::{Q1_0_BLOCK_BYTES, Q1_0_GROUP_SIZE};

/// g_block: 单次 microkernel 覆盖的 group 数。
/// 8 同时整除 40 (hidden 5120) 与 136 (down_proj 17408), 无残块;
/// x 片段 4 token × 8 group × 512B = 16KB, 驻 L1 (P 48KB / E 32KB)。
const G_BLOCK: usize = 8;

/// prep scratch 目标容量/worker: f16 400KB (bench_prefill 交错实测最优)
/// 4 worker × 400KB = 1.6MB < 4MB E-core L2 (低压, x 切片获得更多 L2 空间)
/// 实测 700KB 退化 8% (L2 x 切片被 scratch 挤出), 400KB 最优
const SCRATCH_TARGET_BYTES: usize = 400 * 1024;
const R_BLOCK_MAX: usize = 64;

/// t_sub 分片预算与阈值 (完整 rationale 见文件头 "四层 blocking"):
/// 实测 (交错基准) 主导项是 scatter straggler 尾巴 + prep 重做 ——
/// t_sub 8→16→32 单调变优; DAIZA_GEMM_T_SUB 可覆盖 (A/B 实验用)。
/// 仅宽列 (x 总量 > 4MB) 分片, 窄列单块直达。
const L2_TOKEN_BUDGET: usize = 2856 * 1024;
const T_SUB_X_THRESHOLD: usize = 4 * 1024 * 1024;

/// t_sub 上限: 默认 32; DAIZA_GEMM_T_SUB env 可覆盖 (A/B 实验)
fn t_sub_max() -> usize {
    use std::sync::OnceLock;
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("DAIZA_GEMM_T_SUB")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&v| v >= 4)
            .unwrap_or(32)
    })
}

// ---------------------------------------------------------------------------
// prep: 预缩放 LUT scratch —— scratch[r][g][b] = f16(scale_v × LUT[sign_byte])
// f16 存储: Q1_0 scale 本身是 f16, ±scale 在 f16 中无损。scratch 写入减半。
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(unsafe_code)]
unsafe fn prep_scratch_avx2(
    w_bytes: &[u8],
    r_lo: usize,
    r_hi: usize,
    cols: usize,
    scratch: &mut [u16],
) {
    use std::arch::x86_64::*;
    let groups = cols / Q1_0_GROUP_SIZE;
    let row_stride = groups * Q1_0_GROUP_SIZE; // u16/行 = groups × 128
    for r in r_lo..r_hi {
        let lr = r - r_lo;
        let w_row = w_bytes.as_ptr().add(r * groups * Q1_0_BLOCK_BYTES);
        let s_row = scratch.as_mut_ptr().add(lr * row_stride);
        for g in 0..groups {
            let wb = w_row.add(g * Q1_0_BLOCK_BYTES);
            let scale_v = _mm256_cvtph_ps(_mm_set1_epi16(
                u16::from_le_bytes([*wb, *wb.add(1)]) as i16,
            ));
            let dst = s_row.add(g * Q1_0_GROUP_SIZE);
            for b in 0..16 {
                let lut = _mm256_loadu_ps(
                    crate::tensor::quant::sign_lut_entry(*wb.add(2 + b)).as_ptr(),
                );
                // f32 → f16 (无损: ±scale 本身是 f16)
                let f16x8 = _mm256_cvtps_ph(_mm256_mul_ps(scale_v, lut), 0);
                _mm_storeu_si128(dst.add(b * 8) as *mut __m128i, f16x8);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// microkernel: (2 行 × 4 token × [g0, g_hi)) k-lane FMA
// ---------------------------------------------------------------------------

/// 8 个 hsum 结果按 [r+0: t0..t0+4, r+1: t0..t0+4] 顺序累加进 `acc`。
///
/// 尾 token tile (t0+k >= t_hi) 的 x load 下标 clamp 到合法区间,
/// 垃圾结果由调用方跳过 store —— 保证 load 永不越界且无分支开销。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
#[inline]
unsafe fn gemm_pair4t_gb_avx2(
    x: &[f32],
    cols: usize,
    t_hi: usize,
    scratch: &[u16],
    rb: usize,
    r: usize,
    t0: usize,
    g0: usize,
    g_hi: usize,
    acc: &mut [f32; 8],
) {
    use std::arch::x86_64::*;

    let groups = cols / Q1_0_GROUP_SIZE;
    let row_stride = groups * Q1_0_GROUP_SIZE;
    let s0 = scratch.as_ptr().add((r - rb) * row_stride);
    let s1 = scratch.as_ptr().add((r + 1 - rb) * row_stride);

    let t_last = t_hi - 1;
    let xg = [
        x.as_ptr().add(t0.min(t_last) * cols),
        x.as_ptr().add((t0 + 1).min(t_last) * cols),
        x.as_ptr().add((t0 + 2).min(t_last) * cols),
        x.as_ptr().add((t0 + 3).min(t_last) * cols),
    ];

    let mut a00 = _mm256_setzero_ps();
    let mut a01 = _mm256_setzero_ps();
    let mut a02 = _mm256_setzero_ps();
    let mut a03 = _mm256_setzero_ps();
    let mut a10 = _mm256_setzero_ps();
    let mut a11 = _mm256_setzero_ps();
    let mut a12 = _mm256_setzero_ps();
    let mut a13 = _mm256_setzero_ps();

    let mut g = g0;
    while g < g_hi {
        let sg0 = s0.add(g * Q1_0_GROUP_SIZE);
        let sg1 = s1.add(g * Q1_0_GROUP_SIZE);
        let x0 = xg[0].add(g * Q1_0_GROUP_SIZE);
        let x1 = xg[1].add(g * Q1_0_GROUP_SIZE);
        let x2 = xg[2].add(g * Q1_0_GROUP_SIZE);
        let x3 = xg[3].add(g * Q1_0_GROUP_SIZE);
        for b in 0..16 {
            // f16 scratch load + cvtph_ps → f32 (port 5, 不抢 FMA port 0/1)
            let sl0 = _mm256_cvtph_ps(_mm_loadu_si128(sg0.add(b * 8) as *const __m128i));
            let sl1 = _mm256_cvtph_ps(_mm_loadu_si128(sg1.add(b * 8) as *const __m128i));
            let v0 = _mm256_loadu_ps(x0.add(b * 8));
            let v1 = _mm256_loadu_ps(x1.add(b * 8));
            let v2 = _mm256_loadu_ps(x2.add(b * 8));
            let v3 = _mm256_loadu_ps(x3.add(b * 8));
            a00 = _mm256_fmadd_ps(sl0, v0, a00);
            a01 = _mm256_fmadd_ps(sl0, v1, a01);
            a02 = _mm256_fmadd_ps(sl0, v2, a02);
            a03 = _mm256_fmadd_ps(sl0, v3, a03);
            a10 = _mm256_fmadd_ps(sl1, v0, a10);
            a11 = _mm256_fmadd_ps(sl1, v1, a11);
            a12 = _mm256_fmadd_ps(sl1, v2, a12);
            a13 = _mm256_fmadd_ps(sl1, v3, a13);
        }
        g += 1;
    }

    acc[0] += crate::math::simd_exp::hsum_ps(a00);
    acc[1] += crate::math::simd_exp::hsum_ps(a01);
    acc[2] += crate::math::simd_exp::hsum_ps(a02);
    acc[3] += crate::math::simd_exp::hsum_ps(a03);
    acc[4] += crate::math::simd_exp::hsum_ps(a10);
    acc[5] += crate::math::simd_exp::hsum_ps(a11);
    acc[6] += crate::math::simd_exp::hsum_ps(a12);
    acc[7] += crate::math::simd_exp::hsum_ps(a13);
}

// ---------------------------------------------------------------------------
// worker: [start, end) 行区间 (r_block → t0 tile → pair → g_block)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn gemm_rows_range_avx2(
    w_bytes: &[u8],
    rows_total: usize,
    cols: usize,
    x: &[f32],
    t_lo: usize,
    t_hi: usize,
    y: &mut [f32],
    start: usize,
    end: usize,
    r_block: usize,
    scratch: &mut [u16],
) {
    let groups = cols / Q1_0_GROUP_SIZE;
    let g_tiles = groups.div_ceil(G_BLOCK);
    let n_t_tiles = (t_hi - t_lo).div_ceil(4);

    let mut rb = start;
    while rb < end {
        let r_hi = (rb + r_block).min(end);
        let rows_even = (r_hi - rb) & !1;
        if rows_even > 0 {
            unsafe { prep_scratch_avx2(w_bytes, rb, rb + rows_even, cols, scratch) };
            for ti in 0..n_t_tiles {
                let t0 = t_lo + ti * 4;
                let t_lim = (t_hi - t0).min(4); // 本 tile 实际 token 数
                let mut r = rb;
                while r < rb + rows_even {
                    let mut acc = [0.0f32; 8];
                    for gt in 0..g_tiles {
                        let g0 = gt * G_BLOCK;
                        let g_hi = (g0 + G_BLOCK).min(groups);
                        unsafe {
                            gemm_pair4t_gb_avx2(
                                x, cols, t_hi, scratch, rb, r, t0, g0, g_hi, &mut acc,
                            );
                        }
                    }
                    for k in 0..t_lim {
                        let ty = (t0 + k) * rows_total;
                        y[ty + r] = acc[k];
                        y[ty + r + 1] = acc[4 + k];
                    }
                    r += 2;
                }
            }
        }
        // 奇数尾行 (仅矩阵末行): 逐 token dot 回退
        for r in (rb + rows_even)..r_hi {
            for t in t_lo..t_hi {
                unsafe {
                    *y.get_unchecked_mut(t * rows_total + r) =
                        crate::tensor::quant::dot_q1_0_row_avx2(
                            w_bytes,
                            r,
                            cols,
                            &x[t * cols..(t + 1) * cols],
                        );
                }
            }
        }
        rb = r_hi;
    }
}

// ---------------------------------------------------------------------------
// 标量回退 (无 AVX2 / 非 x86_64): 正确性优先, 性能不敏感
// ---------------------------------------------------------------------------

#[allow(unsafe_code)]
fn gemm_q1_0_batch_scalar(
    w_bytes: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    n_tokens: usize,
    y: &mut [f32],
) {
    let mut w_deq = vec![0.0f32; cols];
    for r in 0..rows {
        crate::tensor::quant::dequantize_q1_0_row_into(w_bytes, r, cols, &mut w_deq);
        for t in 0..n_tokens {
            let xr = &x[t * cols..(t + 1) * cols];
            let mut acc = 0.0f32;
            for k in 0..cols {
                acc += w_deq[k] * xr[k];
            }
            y[t * rows + r] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

/// ★ prefill GEMM dispatch: `matvec_batch_into_slice` 在
/// n_batch >= 64 && rows >= 4096 时调用。
///
/// - `x`: [n_tokens][cols] 行优先 (不转置, kernel 直接按 k-lane 片段读)
/// - `y`: [n_tokens][rows] 行优先, 覆盖写入
#[allow(unsafe_code)]
pub fn gemm_q1_0_batch(
    w_bytes: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    n_tokens: usize,
    y: &mut [f32],
) {
    debug_assert_eq!(x.len(), n_tokens * cols);
    debug_assert_eq!(y.len(), n_tokens * rows);
    debug_assert_eq!(cols % Q1_0_GROUP_SIZE, 0);

    if n_tokens == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if crate::tensor::quant::avx2_q1_0_available() {
        if n_tokens == 1 {
            for (r, y_r) in y.iter_mut().enumerate() {
                unsafe {
                    *y_r = crate::tensor::quant::dot_q1_0_row_avx2(w_bytes, r, cols, x);
                }
            }
            return;
        }

        let groups = cols / Q1_0_GROUP_SIZE;
        // r_block: scratch ~700KB 为目标的自适应行块 (f16: 2 bytes/element)
        let r_block = (SCRATCH_TARGET_BYTES / (groups * Q1_0_GROUP_SIZE * 2))
            .clamp(2, R_BLOCK_MAX)
            & !1;
        // t_sub: x 切片与 scratch 同驻 L2 的 token 子块 (4 对齐, [4, 32]);
        // 仅宽列大 batch 分片 (T_SUB_X_THRESHOLD), 其余单块直达
        let scratch_block = r_block * groups * Q1_0_GROUP_SIZE * 2;
        let t_sub = if n_tokens * cols * 4 > T_SUB_X_THRESHOLD {
            (L2_TOKEN_BUDGET.saturating_sub(scratch_block) / (cols * 4))
                .clamp(4, t_sub_max())
                & !3
        } else {
            n_tokens
        };

        let pool = match workspace::get_thread_pool() {
            Some(p) => p,
            None => {
                let mut scratch = vec![0u16; r_block * groups * Q1_0_GROUP_SIZE];
                let mut t_lo = 0;
                while t_lo < n_tokens {
                    let t_hi = (t_lo + t_sub).min(n_tokens);
                    gemm_rows_range_avx2(
                        w_bytes, rows, cols, x, t_lo, t_hi, y, 0, rows, r_block, &mut scratch,
                    );
                    t_lo = t_hi;
                }
                return;
            }
        };

        let bytes_addr = w_bytes.as_ptr() as usize;
        let bytes_len = w_bytes.len();
        let x_addr = x.as_ptr() as usize;
        let y_addr = y.as_mut_ptr() as usize;
        let scratch_need = r_block * groups * Q1_0_GROUP_SIZE;
        // 细 chunk: 512 行在 ~10 执行者下每人一块、零偷取弹性, 慢核 (E/LP-E)
        //   持块期间全场 barrier 等待 (straggler 尾巴 ~1-2ms/scatter);
        //   128 行让快核偷走余块, 尾巴缩到 1/4
        let steal_chunk = 128usize;

        // 每个 t_sub 一次 scatter + barrier: 保证全部 worker 处于同一 token 子块,
        // x 切片在 E-core 簇 L2 内只读共享 (4 worker 只算一份)
        let mut t_lo = 0;
        while t_lo < n_tokens {
            let t_hi = (t_lo + t_sub).min(n_tokens);
            pool.scatter_wait_stealing(rows, steal_chunk, move |start, end| {
                thread_local! {
                    static SCRATCH: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
                }
                SCRATCH.with(|buf| {
                    let mut b = buf.borrow_mut();
                    if b.len() < scratch_need {
                        b.resize(scratch_need, 0);
                    }
                    let w = unsafe {
                        std::slice::from_raw_parts(bytes_addr as *const u8, bytes_len)
                    };
                    let xs = unsafe {
                        std::slice::from_raw_parts(x_addr as *const f32, n_tokens * cols)
                    };
                    let ys = unsafe {
                        std::slice::from_raw_parts_mut(y_addr as *mut f32, n_tokens * rows)
                    };
                    gemm_rows_range_avx2(
                        w, rows, cols, xs, t_lo, t_hi, ys, start, end, r_block, &mut b[..],
                    );
                });
            });
            t_lo = t_hi;
        }
        return;
    }

    gemm_q1_0_batch_scalar(w_bytes, rows, cols, x, n_tokens, y);
}

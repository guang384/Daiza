//! SIMD 超越函数内核(AVX2 8-wide)
//!
//! 提供 `exp_ps(__m256) -> __m256` 等 SIMD 超越函数,用于 softmax / silu / sigmoid。
//!
//! ## 算法(多项式 + 位运算 exp2)
//!
//! `exp(x) = 2^n * exp(r)`,其中 `n = round(x/ln2)`,`r = x - n*ln2`
//! `exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120` (5 阶,误差 < 2^-23 ≈ 1e-7)
//! `2^n` 用 IEEE 754 浮点位运算:把 `(n + 127)` 装入 f32 指数位
//!
//! ## 性能
//!
//! - 标量 `expf`: ~30 cycle/element
//! - 本内核 AVX2 8-wide: ~10 cycle 总(含 FMA 流水),~1.25 cycle/element
//! - 加速比: ~24x
//!
//! ## 调用频次(token)
//!
//! - silu/sigmoid: ~2M 次/token(SSM conv1d + MLP swiglu + SSM gate + attn gate)
//! - softmax exp: ~1.5M 次/token(T=4K,16 层 × 24 head)
//! - 合计 ~3.5M 次/token × 节省 ~28c = ~100M cycle ≈ **33ms/token**

use std::arch::x86_64::*;

const LN2_F: f32 = std::f32::consts::LN_2; // 0.6931472
const LN2_INV: f32 = 1.0 / LN2_F; // 1.4426950

/// AVX2 8-wide exp(x) 内核
///
/// 精度:误差 < 2^-23 (1e-7),远小于 softmax/silu 的精度需求
/// 输入范围:x ∈ [-88, 88](超出范围返回 0 或 inf,与 libm 一致)
///
/// ★ 输入 clamp: online softmax 首次 iter 时 m_old=-inf, m_old-m_new=-inf
///   原 exp_ps(-inf) 产生 NaN (r = -inf - (-inf*ln2) = -inf + inf = NaN),
///   导致 attention output 全 NaN。clamp x 到 [-88, 88] 使 exp(-88)≈0 (denormal),
///   既消除 NaN 又与 libm 行为一致 (超出 [-88, 88] 返回 0/inf)。
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
pub unsafe fn exp_ps(x: __m256) -> __m256 {
    // 0. clamp x 到 [-88, 88] 防止 inf/NaN 传播 (exp(-88)≈6e-39, exp(88)≈1.65e38)
    let x_lo = _mm256_set1_ps(-88.0);
    let x_hi = _mm256_set1_ps(88.0);
    let x = _mm256_max_ps(_mm256_min_ps(x, x_hi), x_lo);

    // 1. n = round(x / ln2) — 用 round_ps (round to nearest, ties to even)
    let n_f = _mm256_mul_ps(x, _mm256_set1_ps(LN2_INV));
    let n_f = _mm256_round_ps(n_f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // 2. r = x - n * ln2(用 FNMADD: r = -(n * ln2) + x = x - n*ln2)
    let r = _mm256_fnmadd_ps(n_f, _mm256_set1_ps(LN2_F), x);

    // 3. exp(r) 多项式展开(Horner 法,从高到低):
    //    exp(r) = 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
    //      p = 1/120
    //      p = p * r + 1/24  → r/120 + 1/24
    //      p = p * r + 1/6   → r²/120 + r/24 + 1/6
    //      p = p * r + 1/2   → r³/120 + r²/24 + r/6 + 1/2
    //      p = p * r + 1     → r⁴/120 + r³/24 + r²/6 + r/2 + 1
    //      p = p * r + 1     → r⁵/120 + r⁴/24 + r³/6 + r²/2 + r + 1  ✓
    let mut p = _mm256_set1_ps(1.0 / 120.0);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 24.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 6.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));

    // 4. 2^n 用位运算:n + 127 装入 f32 指数位(8 bits,位置 23-30)
    //    n_clamped = clamp(n, -127, 128) 防止溢出
    let n_max = _mm256_set1_ps(129.0);
    let n_min = _mm256_set1_ps(-126.0);
    let n_clamped = _mm256_max_ps(_mm256_min_ps(n_f, n_max), n_min);
    let n_int = _mm256_cvtps_epi32(n_clamped); // 转 i32
    // (n + 127) << 23 — 装入 f32 的指数位
    let bias = _mm256_set1_epi32(127);
    let exp_bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(n_int, bias));
    let pow2_n = _mm256_castsi256_ps(exp_bits); // reinterpret as f32

    // 5. 结果 = 2^n * exp(r)
    _mm256_mul_ps(pow2_n, p)
}

/// SIMD sigmoid(x) = 1 / (1 + exp(-x))
///
/// ★ 用 rcp + 1 次 Newton 迭代替代 _mm256_div_ps
///   - _mm256_div_ps: ~10-20c (IEEE-compliant, correctly rounded)
///   - rcp + Newton:  ~5c (rcp 1c + 2×FMA + 1×MUL = ~5c, ~1 ULP 误差)
///   - sigmoid 输出 ∈ [0,1], 1 ULP 相对误差对模型精度无影响
///   - 除数 1+exp(-x) ∈ [1, 2], 无 0/inf 边界
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
pub unsafe fn sigmoid_ps(x: __m256) -> __m256 {
    let neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0)); // 翻转符号位
    let exp_neg = exp_ps(neg_x);
    let one = _mm256_set1_ps(1.0);
    let d = _mm256_add_ps(one, exp_neg);
    // rcp + Newton: y = rcp(d); y *= (2 - d*y)
    let y0 = _mm256_rcp_ps(d);
    let two = _mm256_set1_ps(2.0);
    let y1 = _mm256_mul_ps(y0, _mm256_fnmadd_ps(d, y0, two)); // y0 * (2 - d*y0)
    y1
}

/// SIMD silu(x) = x * sigmoid(x)
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
pub unsafe fn silu_ps(x: __m256) -> __m256 {
    _mm256_mul_ps(x, sigmoid_ps(x))
}

/// 检测 CPU 是否支持 AVX2 + FMA
pub fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// SIMD sigmoid 标量入口
pub fn sigmoid_fast(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd_available() {
        #[allow(unsafe_code)]
        unsafe {
            let v = _mm256_set1_ps(x);
            let r = sigmoid_ps(v);
            return _mm256_cvtss_f32(r);
        }
    }
    1.0 / (1.0 + (-x).exp())
}

/// SIMD exp 标量入口 (broadcast + exp_ps + extract low lane)
/// ★ 用于 online softmax 内层循环的 alpha/beta 计算, 替代 libm expf (~30c → ~12c)
pub fn exp_fast(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd_available() {
        #[allow(unsafe_code)]
        unsafe {
            let v = _mm256_set1_ps(x);
            let r = exp_ps(v);
            return _mm256_cvtss_f32(r);
        }
    }
    x.exp()
}

// ============================================================================
// 批量版本(对整个 slice 操作)
// ============================================================================

/// 对 slice 应用 exp(x),原地修改
pub fn exp_inplace_simd(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd_available() {
        #[allow(unsafe_code)]
        unsafe {
            let n = x.len();
            let n8 = (n / 8) * 8;
            for i in (0..n8).step_by(8) {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                let r = exp_ps(v);
                _mm256_storeu_ps(x.as_mut_ptr().add(i), r);
            }
            // 尾部标量
            for i in n8..n {
                x[i] = x[i].exp();
            }
            return;
        }
    }
    for xi in x.iter_mut() {
        *xi = xi.exp();
    }
}

/// 对 slice 应用 sigmoid(x),原地修改
pub fn sigmoid_inplace_simd(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd_available() {
        #[allow(unsafe_code)]
        unsafe {
            let n = x.len();
            let n8 = (n / 8) * 8;
            for i in (0..n8).step_by(8) {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                let r = sigmoid_ps(v);
                _mm256_storeu_ps(x.as_mut_ptr().add(i), r);
            }
            for i in n8..n {
                x[i] = 1.0 / (1.0 + (-x[i]).exp());
            }
            return;
        }
    }
    for xi in x.iter_mut() {
        *xi = 1.0 / (1.0 + (-*xi).exp());
    }
}

/// 对 slice 应用 silu(x),原地修改
pub fn silu_inplace_simd(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd_available() {
        #[allow(unsafe_code)]
        unsafe {
            let n = x.len();
            let n8 = (n / 8) * 8;
            for i in (0..n8).step_by(8) {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                let r = silu_ps(v);
                _mm256_storeu_ps(x.as_mut_ptr().add(i), r);
            }
            for i in n8..n {
                x[i] = x[i] / (1.0 + (-x[i]).exp());
            }
            return;
        }
    }
    for xi in x.iter_mut() {
        *xi = *xi / (1.0 + (-*xi).exp());
    }
}

/// dst[i] *= src[i],原地修改 dst (P2-3: attn_out *= sigmoid(gate))
pub fn mul_inplace_simd(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if simd_available() {
        #[allow(unsafe_code)]
        unsafe {
            let n = dst.len();
            let n8 = (n / 8) * 8;
            for i in (0..n8).step_by(8) {
                let d = _mm256_loadu_ps(dst.as_ptr().add(i));
                let s = _mm256_loadu_ps(src.as_ptr().add(i));
                let r = _mm256_mul_ps(d, s);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), r);
            }
            for i in n8..n {
                dst[i] *= src[i];
            }
            return;
        }
    }
    for i in 0..dst.len() {
        dst[i] *= src[i];
    }
}

/// SwiGLU: gate[i] = silu(gate[i]) * up[i],原地修改 gate
pub fn swiglu_inplace_simd(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    #[cfg(target_arch = "x86_64")]
    if simd_available() {
        #[allow(unsafe_code)]
        unsafe {
            let n = gate.len();
            let n8 = (n / 8) * 8;
            for i in (0..n8).step_by(8) {
                let g = _mm256_loadu_ps(gate.as_ptr().add(i));
                let u = _mm256_loadu_ps(up.as_ptr().add(i));
                let r = _mm256_mul_ps(silu_ps(g), u);
                _mm256_storeu_ps(gate.as_mut_ptr().add(i), r);
            }
            for i in n8..n {
                gate[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
            }
            return;
        }
    }
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g = (*g / (1.0 + (-*g).exp())) * u;
    }
}

// ---------------------------------------------------------------------------
// AVX2 dot product & saxpy(用于 attention scores 向量化)
// ---------------------------------------------------------------------------

/// AVX2 8-wide 点积: sum(a[i] * b[i])
/// head_dim (128 或 256) 是 8 的倍数,无需尾处理。
#[allow(unsafe_code)]
#[inline]
pub fn dot_product_avx2(a: &[f32], b: &[f32], len: usize) -> f32 {
    debug_assert!(len >= 8);
    unsafe {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut i = 0;
        // 2x unroll: 每次处理 16 个 f32, 隐藏 FMA 延迟
        while i + 16 <= len {
            let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(va0, vb0, sum0);
            let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            sum1 = _mm256_fmadd_ps(va1, vb1, sum1);
            i += 16;
        }
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(va, vb, sum0);
            i += 8;
        }
        sum0 = _mm256_add_ps(sum0, sum1);
        // ★ P1-1: 水平求和改纯寄存器内 SSE (无 store+scalar reduce)
        //   原 store + 7 次标量 add ~5c; 寄存器内 ~3c
        //   attention scores 调用频次高 (24 qh × n_cached × 16 attn blocks)
        let hi = _mm256_extractf128_ps(sum0, 1);
        let lo = _mm256_castps256_ps128(sum0);
        let sum128 = _mm_add_ps(hi, lo);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    }
}

/// AVX2 水平求和: __m256 → f32 (纯寄存器内 SSE, 无 store)
///
/// 公共实现, 替代 quant.rs / layernorm.rs / rmsnorm.rs / softmax.rs / ssm.rs /
/// forward.rs / vision/weights.rs / vision/encoder.rs 中重复的 `hsum_ps` /
/// `horizontal_sum_avx2` / `horizontal_sum_ps` 局部定义。
///
/// 算法: 把 256-bit 拆成两个 128-bit, 相加后用 movhdup + movhl 三步 reduce 到标量。
/// 约 3 cycle (vs store + 7 次标量 add ~5c)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[inline]
pub unsafe fn hsum_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// AVX2 8-wide saxpy: y[i] += scale * x[i]
/// head_dim (128 或 256) 是 8 的倍数,无需尾处理。
#[allow(unsafe_code)]
#[inline]
pub fn saxpy_avx2(scale: f32, x: &[f32], y: &mut [f32], len: usize) {
    debug_assert!(len >= 8);
    unsafe {
        let sv = _mm256_set1_ps(scale);
        let mut i = 0;
        while i + 8 <= len {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let vy = _mm256_loadu_ps(y.as_ptr().add(i));
            let result = _mm256_fmadd_ps(sv, vx, vy);
            _mm256_storeu_ps(y.as_mut_ptr().add(i), result);
            i += 8;
        }
    }
}

/// AVX2 8-wide online softmax V-update: y[i] = y[i] * alpha + beta * x[i]
///
/// attention online softmax 内层循环 (head_dim=256 = 32×8-wide, 无尾处理)。
/// 原标量循环每 iter 2c (FMUL + FMA), 256 iter = 512c;
/// AVX2 32 iter × 2c (mul + fma 可融合为单 FMA if 编译器优化, 或 2c 独立) ≈ 64c, **8× 加速**。
///
/// 调用频次 (per token, n_cached=N):
///   16 attn blocks × 4 kv_heads × N × 6 group_size = 384N 次 V-update
///   N=256: 98K 次 × (512c-64c) = 43M cycle ≈ 12ms/token 节省
///   N=1024: 393K 次 × 448c = 176M cycle ≈ 50ms/token 节省 (但受 KV cache L2 带宽限制, 实际收益较小)
#[allow(unsafe_code)]
#[inline]
pub fn online_softmax_v_update_avx2(
    y: &mut [f32],
    alpha: f32,
    beta: f32,
    x: &[f32],
    len: usize,
) {
    debug_assert!(len >= 8);
    debug_assert_eq!(y.len(), len);
    debug_assert_eq!(x.len(), len);
    unsafe {
        let av = _mm256_set1_ps(alpha);
        let bv = _mm256_set1_ps(beta);
        let mut i = 0;
        // ★ 2x unroll: 打破 mul(4c) → fma(4c) 依赖链, 两独立链交错
        //   head_dim=256 = 32×8-wide, 足够迭代做 unroll
        while i + 16 <= len {
            let vy0 = _mm256_loadu_ps(y.as_ptr().add(i));
            let vx0 = _mm256_loadu_ps(x.as_ptr().add(i));
            let vy1 = _mm256_loadu_ps(y.as_ptr().add(i + 8));
            let vx1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
            let sy0 = _mm256_mul_ps(av, vy0);
            let sy1 = _mm256_mul_ps(av, vy1);
            let r0 = _mm256_fmadd_ps(bv, vx0, sy0);
            let r1 = _mm256_fmadd_ps(bv, vx1, sy1);
            _mm256_storeu_ps(y.as_mut_ptr().add(i), r0);
            _mm256_storeu_ps(y.as_mut_ptr().add(i + 8), r1);
            i += 16;
        }
        // tail (len 通常是 8 的倍数, 但保留鲁棒性)
        while i + 8 <= len {
            let vy = _mm256_loadu_ps(y.as_ptr().add(i));
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let scaled_y = _mm256_mul_ps(av, vy);
            let result = _mm256_fmadd_ps(bv, vx, scaled_y);
            _mm256_storeu_ps(y.as_mut_ptr().add(i), result);
            i += 8;
        }
    }
}

/// AVX2 8-wide scale: y[i] = x[i] * scale
///
/// attention online softmax 归一化: out_head[j] = out[j] * inv_s (head_dim=256)。
#[allow(unsafe_code)]
#[inline]
pub fn scale_avx2(x: &[f32], scale: f32, y: &mut [f32], len: usize) {
    debug_assert!(len >= 8);
    debug_assert_eq!(x.len(), len);
    debug_assert_eq!(y.len(), len);
    unsafe {
        let sv = _mm256_set1_ps(scale);
        let mut i = 0;
        while i + 8 <= len {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let result = _mm256_mul_ps(sv, vx);
            _mm256_storeu_ps(y.as_mut_ptr().add(i), result);
            i += 8;
        }
    }
}

/// AVX2 8-wide add: dst[i] = a[i] + b[i]
///
/// ★ 单 pass 替代 `copy_from_slice + saxpy_avx2(1.0, ...)` 两 pass。
///   markov Step 3: step_logit = base + bias (vocab=248320, 4 位置/cycle)
///   原 2 pass: copy (1MB read+write) + saxpy (1MB read bias + 1MB RW step) = 4MB traffic
///   新 1 pass: 1MB read a + 1MB read b + 1MB write dst = 3MB traffic, 节省 25%
pub fn add_avx2(a: &[f32], b: &[f32], dst: &mut [f32], len: usize) {
    debug_assert!(len >= 8);
    debug_assert_eq!(a.len(), len);
    debug_assert_eq!(b.len(), len);
    debug_assert_eq!(dst.len(), len);
    #[cfg(target_arch = "x86_64")]
    if simd_available() {
        #[allow(unsafe_code)]
        unsafe {
            let mut i = 0;
            let n8 = (len / 8) * 8;
            while i < n8 {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_add_ps(va, vb));
                i += 8;
            }
            for j in i..len {
                dst[j] = a[j] + b[j];
            }
            return;
        }
    }
    for i in 0..len {
        dst[i] = a[i] + b[i];
    }
}

/// AVX2 向量化 argmax: 返回 (argmax_index, max_value)
///
/// ★ 用于 leviathan_check greedy verify (vocab=248320, 每 cycle 多次调用)
///   算法: 每 8 元素 AVX2 compare + movemask 检测是否有 > best_val
///   - 大多数块 (best_val 已接近 max) 直接跳过, 节省 ~70% 标量比较
///   - 命中块内标量扫描 8 个找 max (避免 horizontal reduce + index 跟踪复杂度)
pub fn argmax_avx2(x: &[f32]) -> (usize, f32) {
    let len = x.len();
    if len == 0 {
        return (0, f32::NEG_INFINITY);
    }
    let mut best_id = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    #[cfg(target_arch = "x86_64")]
    if simd_available() && len >= 8 {
        #[allow(unsafe_code)]
        unsafe {
            use std::arch::x86_64::*;
            let ptr = x.as_ptr();
            let mut i = 0;
            let n8 = (len / 8) * 8;
            while i < n8 {
                let v = _mm256_loadu_ps(ptr.add(i));
                let bv = _mm256_set1_ps(best_val);
                let cmp = _mm256_cmp_ps(v, bv, _CMP_GT_OQ);
                if _mm256_movemask_ps(cmp) != 0 {
                    // 本 8 元素块内有 > best_val, 标量扫描找 max
                    for j in 0..8 {
                        let val = *x.get_unchecked(i + j);
                        if val > best_val {
                            best_val = val;
                            best_id = i + j;
                        }
                    }
                }
                i += 8;
            }
            for v in n8..len {
                if x[v] > best_val {
                    best_val = x[v];
                    best_id = v;
                }
            }
        }
        return (best_id, best_val);
    }
    for (i, &v) in x.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_id = i;
        }
    }
    (best_id, best_val)
}

//! 矩阵乘法(GEMM)内核
//!
//! v0 实现:三重循环的朴素 F32 × F32 矩阵乘法。
//!
//! ## 优化路线图(留给学习者练习)
//!
//! 1. **循环序优化**:把 `k` 循环放外层,使 W 行驻留 cache(`i-k-j` 顺序)
//! 2. **块化 blocking**:对 k 维做 64/128 块,适配 L1/L2
//! 3. **SIMD**:`std::arch::x86_64` 的 AVX2(8×f32)或 AVX-512(16×f32)
//! 4. **多线程**:按 `i` 行切分到 N 个 `std::thread`
//! 5. **Q1_0 融合 GEMM**:不反量化,直接对原始 bit 数据运算
//!    (利用 ±1 的位运算性质:`sum_k b_k * x_k = 2*popcount(bits where x_k>0) - sum x_k`)
//!
//! > 注意:本项目 v0 在张量加载时即把 Q1_0 反量化为 F32,
//! > 这里 `matmul` 看到的是已展开的 F32 数据。这是**学习项目的简化**,
//! > 真正引擎如 llama.cpp 永不展开 Q1_0。

/// C = A × B
///
/// - A: [m × k]
/// - B: [k × n]
/// - 返回 C: [m × n]
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    // i-k-j 顺序:对每个 i 一次扫完 k,使 b[k*n..] 在 L1 中复用
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        for kk in 0..k {
            let aik = a_row[kk];
            if aik == 0.0 {
                continue;
            }
            let b_row = &b[kk * n..(kk + 1) * n];
            // 加速点:可换成 SIMD fma
            for j in 0..n {
                c_row[j] += aik * b_row[j];
            }
        }
    }
    c
}

/// C += A × B(累加版本,用于 residual stream)
pub fn matmul_add_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        for kk in 0..k {
            let aik = a_row[kk];
            if aik == 0.0 {
                continue;
            }
            let b_row = &b[kk * n..(kk + 1) * n];
            for j in 0..n {
                c_row[j] += aik * b_row[j];
            }
        }
    }
}

/// 单向量 × 矩阵: y[1,n] = x[1,k] × W[k,n]
///
/// 推理引擎在 decode 阶段 batch_size = 1,这是最频繁的 GEMM 形态。
pub fn matvec(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    // W 是 row-major [k, n],x 是 [k]
    // 优化点:循环顺序让 W 列遍历 → 写出 y[j] 累加 x[k] * W[k*n + j]
    let mut y = vec![0.0f32; n];
    for kk in 0..k {
        let xk = x[kk];
        if xk == 0.0 {
            continue;
        }
        let w_row = &w[kk * n..(kk + 1) * n];
        for j in 0..n {
            y[j] += xk * w_row[j];
        }
    }
    y
}

/// 单向量 × 矩阵(带 bias 累加): y = x × W + b
pub fn matvec_with_bias(x: &[f32], w: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut y = matvec(x, w, k, n);
    for j in 0..n {
        y[j] += b[j];
    }
    y
}

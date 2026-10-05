//! 模型权重容器(优化版:Q1_0 张量保持原始字节)
//!
//! ## 内存策略
//!
//! - **Q1_0 张量保持原始字节**(不展开为 F32):
//!   - 单 block ~50MB,64 个 block = 3.2GB(可接受)
//!   - GEMM 时逐行反量化 + 累加,内存峰值仅 20KB(单行)
//! - **F32 张量直接加载**(norms、conv1d、ssm_a 等小张量):
//!   - 64 个 block × 几 KB = 几 MB

use crate::gguf::parser::GgufFile;
use crate::gguf::tensor_info::TensorType;
use crate::tensor::tensor::Tensor;
use crate::tensor::tensor::load_as_f32;
#[cfg(not(target_arch = "x86_64"))]
use crate::tensor::quant::{
    avx2_q1_0_available, dot_q1_0_row_batch, dot_q1_0_row_scalar,
    quantize_dequantize_q8_0_into, quantize_f32_to_q8_0_simple, dot_q1_0_q8_0_row_avx2,
};
#[cfg(target_arch = "x86_64")]
use crate::tensor::quant::{
    avx2_q1_0_available, dot_q1_0_q8_0_row_avx2, dot_q1_0_row_avx2, dot_q1_0_row_batch4_avx2,
    dot_q1_0_row_batch_avx2, dot_q1_0_row_batch, dot_q1_0_row_dual_avx2, dot_q1_0_row_quad_avx2,
    dot_q1_0_row_scalar, dot_q1_0_row_triple_avx2,
    quantize_dequantize_q8_0_into, quantize_f32_to_q8_0_simple,
};
use crate::BonsaiError;

/// ★ Q8_0 量化路径开关: 模拟 llama.cpp 的 F32→Q8_0 量化误差
///
/// llama.cpp 在 Q1_0 weight × F32 input 时, 自动把 input 量化为 Q8_0 (per-32-block),
/// 引入 ~0.5% 量化误差。Daiza 原本直接用 F32 input, 导致 target tap 比 llama.cpp "更精确",
/// 但 drafter 是用 llama.cpp (Q8_0 path) 训练的, 看到的 tap 分布与训练时不匹配,
/// 接受率从 95% 降到 75%。启用 DAIZA_Q8_PATH=1 后, Daiza 也走 Q8_0 量化路径, 使分布一致。
fn q8_path_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("DAIZA_Q8_PATH").is_ok())
}

/// DAIZA_KERNEL_MODE 缓存 (避免热路径 matvec 每次调用 env::var)
fn kernel_mode() -> &'static str {
    use std::sync::OnceLock;
    static FLAG: OnceLock<String> = OnceLock::new();
    FLAG.get_or_init(|| std::env::var("DAIZA_KERNEL_MODE").unwrap_or_default())
}

/// DAIZA_INT_KERNEL 缓存 (避免热路径 matvec 每次调用 env::var)
fn int_kernel_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("DAIZA_INT_KERNEL").is_ok())
}

/// ★ DAIZA_MATVEC_CHUNK: work-stealing chunk 尺寸 (默认 128)
///
/// bench_klab 交错实测: 128 比 256 快 ~15%@9w (E 核 straggler 尾巴更短)。
/// 仅影响 decode matvec 路径; prefill 走 GEMM/batch 不经过此参数。
fn matvec_steal_chunk() -> usize {
    use std::sync::OnceLock;
    static CHUNK: OnceLock<usize> = OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("DAIZA_MATVEC_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&v: &usize| v > 0)
            .unwrap_or(128)
    })
}

/// 如果 Q8_PATH 启用, 把 x 量化为 Q8_0 再反量化回 F32 (引入量化误差), 返回 Cow::Owned
/// 否则返回 Cow::Borrowed(x), 零开销
pub(crate) fn maybe_quantize_x_q8<'a>(x: &'a [f32]) -> std::borrow::Cow<'a, [f32]> {
    if !q8_path_enabled() {
        return std::borrow::Cow::Borrowed(x);
    }
    let mut x_q = vec![0.0f32; x.len()];
    quantize_dequantize_q8_0_into(x, &mut x_q);
    std::borrow::Cow::Owned(x_q)
}

/// Q1_0 编码的矩阵(保留原始字节,按需反量化单行)
pub struct Q1_0Matrix {
    pub bytes: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}

impl Q1_0Matrix {
    pub fn from_gguf(gguf: &GgufFile, name: &str) -> crate::Result<Self> {
        let info = gguf
            .find_tensor(name)
            .ok_or_else(|| BonsaiError::Model(format!("tensor {name} not found")))?;
        let data = gguf.tensor_data(info)?.to_vec();
        let rows = info.rows() as usize;
        let cols = info.cols() as usize;
        Ok(Self {
            bytes: data,
            rows,
            cols,
        })
    }

    /// 反量化单行,写入 caller 提供的 slice(避免堆分配)
    #[inline]
    pub fn row_into_slice(&self, row_idx: usize, y: &mut [f32]) {
        debug_assert!(row_idx < self.rows, "row_idx {} >= rows {}", row_idx, self.rows);
        debug_assert!(y.len() >= self.cols, "y.len() {} < cols {}", y.len(), self.cols);
        crate::tensor::quant::dequantize_q1_0_row_into(&self.bytes, row_idx, self.cols, y);
    }

    /// 流式 GEMM into caller-provided slice(避免堆分配)
    ///
    /// `y[i] = dot_q1_0_row(W, i, k, x)`,覆盖写入 `y`(不是累加)。
    /// `y.len()` 必须等于 `self.rows`。
    ///
    /// **多线程并行**:当 `self.rows >= 1024` 时按行切分到 N 个 OS thread,
    /// 每 thread 处理 rows/N 行,各自独立累加。threads 数由 `DAIZA_THREADS` env var 控制。
    ///
    /// 优先使用全局持久线程池(若已初始化),否则回退到 `std::thread::scope`。
    #[allow(unsafe_code)]
    #[inline]
    pub fn matvec_into_slice(&self, x: &[f32], y: &mut [f32]) {
        let k = self.cols;
        let n = self.rows;
        debug_assert_eq!(x.len(), k);
        debug_assert_eq!(y.len(), n);

        // ★ Q8_PATH: 模拟 llama.cpp 的 F32→Q8_0 量化误差, 使 target tap 与 drafter 训练分布一致
        let x_q8 = maybe_quantize_x_q8(x);
        let x = x_q8.as_ref();

        let n_threads = crate::model::workspace::thread_count();
        // ★ P2-1: runtime AVX2 check 提到循环外, 避免每行 dot_q1_0_row 内部重复检测
        let use_avx2 = avx2_q1_0_available();
        // ★ 实验: DAIZA_KERNEL_MODE 控制并行度 (quad=4行/dual=2行/single=1行)
        // quad: 8 FMA 链/group (默认, 最高功耗密度)
        // dual: 4 FMA 链/group (降功耗密度 50%)
        // single: 2 FMA 链/group (降功耗密度 75%)
        let kernel_mode = kernel_mode();
        // ★ 实验: DAIZA_INT_KERNEL=1 走整数乘加 kernel (模仿 llama.cpp, 降低功耗密度)
        let use_int_kernel = use_avx2 && int_kernel_enabled();
        // ★ P2-5: 阈值从 4096 降到 1024, 让 attn_k/v (1024 rows) 也走线程池
        if n_threads <= 1 || n < 1024 {
            #[cfg(target_arch = "x86_64")]
            if use_int_kernel {
                // ★ 整数乘加 kernel (模仿 llama.cpp)
                let x_q8_bytes = quantize_f32_to_q8_0_simple(x);
                for i in 0..n {
                    y[i] = unsafe { dot_q1_0_q8_0_row_avx2(&self.bytes, i, k, &x_q8_bytes) };
                }
                return;
            }
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                // ★ 实验: kernel_mode 切换并行度
                if kernel_mode == "dual" {
                    let mut i = 0;
                    while i + 1 < n {
                        let (y0, y1) = unsafe { dot_q1_0_row_dual_avx2(&self.bytes, i, i + 1, k, x) };
                        y[i] = y0; y[i + 1] = y1;
                        i += 2;
                    }
                    if i < n { y[i] = unsafe { dot_q1_0_row_avx2(&self.bytes, i, k, x) }; }
                    return;
                } else if kernel_mode == "single" {
                    for i in 0..n { y[i] = unsafe { dot_q1_0_row_avx2(&self.bytes, i, k, x) }; }
                    return;
                }
                // 默认: quad (P0-H 四行并行, 共享 x load)
                let mut i = 0;
                while i + 3 < n {
                    let (y0, y1, y2, y3) = unsafe { dot_q1_0_row_quad_avx2(&self.bytes, i, i + 1, i + 2, i + 3, k, x) };
                    y[i] = y0;
                    y[i + 1] = y1;
                    y[i + 2] = y2;
                    y[i + 3] = y3;
                    i += 4;
                }
                // 余数: 3 行用 triple, 2 行用 dual, 1 行用 single
                if i + 2 < n {
                    let (y0, y1, y2) = unsafe { dot_q1_0_row_triple_avx2(&self.bytes, i, i + 1, i + 2, k, x) };
                    y[i] = y0;
                    y[i + 1] = y1;
                    y[i + 2] = y2;
                } else if i + 1 < n {
                    let (y0, y1) = unsafe { dot_q1_0_row_dual_avx2(&self.bytes, i, i + 1, k, x) };
                    y[i] = y0;
                    y[i + 1] = y1;
                } else if i < n {
                    y[i] = unsafe { dot_q1_0_row_avx2(&self.bytes, i, k, x) };
                }
                return;
            }
            for i in 0..n {
                y[i] = dot_q1_0_row_scalar(&self.bytes, i, k, x);
            }
            return;
        }

        // 优先使用全局持久线程池(消除 thread spawn 开销)
        if let Some(pool) = crate::model::workspace::get_thread_pool() {
            let bytes_addr = self.bytes.as_ptr() as usize;
            let bytes_len = self.bytes.len();
            let x_addr = x.as_ptr() as usize;
            let y_addr = y.as_mut_ptr() as usize;
            // ★ Work-stealing: chunk_size=256 改善负载均衡
            let steal_chunk = 256;
            let km = kernel_mode;

            // ★ 整数乘加 kernel 路径 (模仿 llama.cpp)
            if use_int_kernel {
                let x_q8_bytes = quantize_f32_to_q8_0_simple(x);
                let xq8_addr = x_q8_bytes.as_ptr() as usize;
                let xq8_len = x_q8_bytes.len();
                pool.scatter_wait_stealing(n, steal_chunk, move |start, end| {
                    let bytes = unsafe { std::slice::from_raw_parts(bytes_addr as *const u8, bytes_len) };
                    let xq8 = unsafe { std::slice::from_raw_parts(xq8_addr as *const u8, xq8_len) };
                    for i in start..end {
                        unsafe {
                            *((y_addr as *mut f32).add(i)) = dot_q1_0_q8_0_row_avx2(bytes, i, k, xq8);
                        }
                    }
                });
                return;
            }

            pool.scatter_wait_stealing(n, steal_chunk, move |start, end| {
                let bytes = unsafe { std::slice::from_raw_parts(bytes_addr as *const u8, bytes_len) };
                let x = unsafe { std::slice::from_raw_parts(x_addr as *const f32, k) };
                #[cfg(target_arch = "x86_64")]
                if use_avx2 {
                    // ★ 实验: kernel_mode 切换并行度
                    if km == "dual" {
                        let mut i = start;
                        while i + 1 < end {
                            let (y0, y1) = unsafe { dot_q1_0_row_dual_avx2(bytes, i, i + 1, k, x) };
                            unsafe {
                                *((y_addr as *mut f32).add(i)) = y0;
                                *((y_addr as *mut f32).add(i + 1)) = y1;
                            }
                            i += 2;
                        }
                        if i < end {
                            unsafe { *((y_addr as *mut f32).add(i)) = dot_q1_0_row_avx2(bytes, i, k, x); }
                        }
                        return;
                    } else if km == "single" {
                        for i in start..end {
                            unsafe { *((y_addr as *mut f32).add(i)) = dot_q1_0_row_avx2(bytes, i, k, x); }
                        }
                        return;
                    }
                    // 默认: quad (P0-H 四行并行)
                    let mut i = start;
                    while i + 3 < end {
                        let (y0, y1, y2, y3) = unsafe { dot_q1_0_row_quad_avx2(bytes, i, i + 1, i + 2, i + 3, k, x) };
                        unsafe {
                            *((y_addr as *mut f32).add(i)) = y0;
                            *((y_addr as *mut f32).add(i + 1)) = y1;
                            *((y_addr as *mut f32).add(i + 2)) = y2;
                            *((y_addr as *mut f32).add(i + 3)) = y3;
                        }
                        i += 4;
                    }
                    if i + 2 < end {
                        let (y0, y1, y2) = unsafe { dot_q1_0_row_triple_avx2(bytes, i, i + 1, i + 2, k, x) };
                        unsafe {
                            *((y_addr as *mut f32).add(i)) = y0;
                            *((y_addr as *mut f32).add(i + 1)) = y1;
                            *((y_addr as *mut f32).add(i + 2)) = y2;
                        }
                    } else if i + 1 < end {
                        let (y0, y1) = unsafe { dot_q1_0_row_dual_avx2(bytes, i, i + 1, k, x) };
                        unsafe {
                            *((y_addr as *mut f32).add(i)) = y0;
                            *((y_addr as *mut f32).add(i + 1)) = y1;
                        }
                    } else if i < end {
                        unsafe {
                            *((y_addr as *mut f32).add(i)) = dot_q1_0_row_avx2(bytes, i, k, x);
                        }
                    }
                    return;
                }
                for i in start..end {
                    unsafe {
                        *((y_addr as *mut f32).add(i)) = dot_q1_0_row_scalar(bytes, i, k, x);
                    }
                }
            });
            return;
        }

        // 回退: std::thread::scope
        let bytes = &self.bytes;
        std::thread::scope(|s| {
            let chunk = (n + n_threads - 1) / n_threads;
            let mut handles = Vec::with_capacity(n_threads);
            let mut row_start = 0usize;
            for y_chunk in y.chunks_mut(chunk) {
                let chunk_len = y_chunk.len();
                let start = row_start;
                row_start += chunk_len;
                let h = s.spawn(move || {
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        for (i, y_i) in y_chunk.iter_mut().enumerate() {
                            *y_i = unsafe { dot_q1_0_row_avx2(bytes, start + i, k, x) };
                        }
                        return;
                    }
                    for (i, y_i) in y_chunk.iter_mut().enumerate() {
                        *y_i = dot_q1_0_row_scalar(bytes, start + i, k, x);
                    }
                });
                handles.push(h);
            }
            for h in handles {
                h.join().unwrap();
            }
        });
    }

    /// 流式 GEMM 累加到 caller-provided slice(用于残差合并)
    ///
    /// `y[i] += dot_q1_0_row(W, i, k, x)`,常用于 `h += W_down @ mlp_hidden`。
    ///
    /// 优先使用全局持久线程池(若已初始化),否则回退到 `std::thread::scope`。
    #[allow(unsafe_code)]
    #[inline]
    pub fn matvec_add_into_slice(&self, x: &[f32], y: &mut [f32]) {
        let k = self.cols;
        let n = self.rows;
        debug_assert_eq!(x.len(), k);
        debug_assert_eq!(y.len(), n);

        // ★ Q8_PATH: 模拟 llama.cpp 的 F32→Q8_0 量化误差, 使 target tap 与 drafter 训练分布一致
        let x_q8 = maybe_quantize_x_q8(x);
        let x = x_q8.as_ref();

        let n_threads = crate::model::workspace::thread_count();
        // ★ P2-1: runtime AVX2 check 提到循环外
        let use_avx2 = avx2_q1_0_available();
        // ★ P2-5: 阈值从 4096 降到 1024, 让 attn_k/v (1024 rows) 也走线程池
        if n_threads <= 1 || n < 1024 {
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                // ★ P0-H: 四行并行
                let mut i = 0;
                while i + 3 < n {
                    let (y0, y1, y2, y3) = unsafe { dot_q1_0_row_quad_avx2(&self.bytes, i, i + 1, i + 2, i + 3, k, x) };
                    y[i] += y0;
                    y[i + 1] += y1;
                    y[i + 2] += y2;
                    y[i + 3] += y3;
                    i += 4;
                }
                if i + 2 < n {
                    let (y0, y1, y2) = unsafe { dot_q1_0_row_triple_avx2(&self.bytes, i, i + 1, i + 2, k, x) };
                    y[i] += y0;
                    y[i + 1] += y1;
                    y[i + 2] += y2;
                } else if i + 1 < n {
                    let (y0, y1) = unsafe { dot_q1_0_row_dual_avx2(&self.bytes, i, i + 1, k, x) };
                    y[i] += y0;
                    y[i + 1] += y1;
                } else if i < n {
                    y[i] += unsafe { dot_q1_0_row_avx2(&self.bytes, i, k, x) };
                }
                return;
            }
            for i in 0..n {
                y[i] += dot_q1_0_row_scalar(&self.bytes, i, k, x);
            }
            return;
        }

        // 优先使用全局持久线程池
        if let Some(pool) = crate::model::workspace::get_thread_pool() {
            let bytes_addr = self.bytes.as_ptr() as usize;
            let bytes_len = self.bytes.len();
            let x_addr = x.as_ptr() as usize;
            let y_addr = y.as_mut_ptr() as usize;
            // ★ Work-stealing: chunk_size=256 改善负载均衡
            let steal_chunk = 256;

            pool.scatter_wait_stealing(n, steal_chunk, move |start, end| {
                let bytes = unsafe { std::slice::from_raw_parts(bytes_addr as *const u8, bytes_len) };
                let x = unsafe { std::slice::from_raw_parts(x_addr as *const f32, k) };
                #[cfg(target_arch = "x86_64")]
                if use_avx2 {
                    // ★ P0-H: 四行并行
                    let mut i = start;
                    while i + 3 < end {
                        let (y0, y1, y2, y3) = unsafe { dot_q1_0_row_quad_avx2(bytes, i, i + 1, i + 2, i + 3, k, x) };
                        unsafe {
                            *((y_addr as *mut f32).add(i)) += y0;
                            *((y_addr as *mut f32).add(i + 1)) += y1;
                            *((y_addr as *mut f32).add(i + 2)) += y2;
                            *((y_addr as *mut f32).add(i + 3)) += y3;
                        }
                        i += 4;
                    }
                    if i + 2 < end {
                        let (y0, y1, y2) = unsafe { dot_q1_0_row_triple_avx2(bytes, i, i + 1, i + 2, k, x) };
                        unsafe {
                            *((y_addr as *mut f32).add(i)) += y0;
                            *((y_addr as *mut f32).add(i + 1)) += y1;
                            *((y_addr as *mut f32).add(i + 2)) += y2;
                        }
                    } else if i + 1 < end {
                        let (y0, y1) = unsafe { dot_q1_0_row_dual_avx2(bytes, i, i + 1, k, x) };
                        unsafe {
                            *((y_addr as *mut f32).add(i)) += y0;
                            *((y_addr as *mut f32).add(i + 1)) += y1;
                        }
                    } else if i < end {
                        unsafe {
                            *((y_addr as *mut f32).add(i)) += dot_q1_0_row_avx2(bytes, i, k, x);
                        }
                    }
                    return;
                }
                for i in start..end {
                    unsafe {
                        *((y_addr as *mut f32).add(i)) += dot_q1_0_row_scalar(bytes, i, k, x);
                    }
                }
            });
            return;
        }

        // 回退: std::thread::scope
        let bytes = &self.bytes;
        std::thread::scope(|s| {
            let chunk = (n + n_threads - 1) / n_threads;
            let mut handles = Vec::with_capacity(n_threads);
            let mut row_start = 0usize;
            for y_chunk in y.chunks_mut(chunk) {
                let chunk_len = y_chunk.len();
                let start = row_start;
                row_start += chunk_len;
                let h = s.spawn(move || {
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        for (i, y_i) in y_chunk.iter_mut().enumerate() {
                            *y_i += unsafe { dot_q1_0_row_avx2(bytes, start + i, k, x) };
                        }
                        return;
                    }
                    for (i, y_i) in y_chunk.iter_mut().enumerate() {
                        *y_i += dot_q1_0_row_scalar(bytes, start + i, k, x);
                    }
                });
                handles.push(h);
            }
            for h in handles {
                h.join().unwrap();
            }
        });
    }

    /// 批量 matvec: y[t][i] = dot(W_row_i, x[t])
    ///
    /// `x`: [n_batch * cols] 行优先
    /// `y`: [n_batch * rows] 行优先, 覆盖写入
    ///
    /// **核心优化**: 每行权重只读一次, 对 batch 内所有 token 复用。
    /// 原 prefill 每 token 读 13GB 权重 → batch 化后只读一次 13GB。
    #[allow(unsafe_code)]
    pub fn matvec_batch_into_slice(&self, x: &[f32], n_batch: usize, y: &mut [f32]) {
        let n_cols = self.cols;
        let n = self.rows;
        debug_assert_eq!(x.len(), n_batch * n_cols);
        debug_assert_eq!(y.len(), n_batch * n);

        if n_batch <= 1 {
            self.matvec_into_slice(x, &mut y[..n]);
            return;
        }

        // ★ Prefill 块 GEMM 化: 大矩阵 (rows >= 4096) + 大 batch (>= 64) 时走块 GEMM
        // 收益来源: batch kernel 的 LUT 每 token 每组重复 + 行优先 x 跨 stride 访存灾难
        // 阈值: 小矩阵 (alpha/beta 48 rows, k/v 1024 rows) 权重驻缓存, batch kernel
        //   已接近带宽上限, GEMM 的 FMA 优势小于 dispatch 开销 (prep+barrier);
        //   小 batch 时计算量不足以摊薄 prep 成本
        #[cfg(target_arch = "x86_64")]
        if n_batch >= 64 && n >= 4096 && crate::tensor::quant::avx2_q1_0_available() {
            crate::tensor::gemm::gemm_q1_0_batch(&self.bytes, n, n_cols, x, n_batch, y);
            return;
        }

        // ★ Q8_PATH: 模拟 llama.cpp 的 F32→Q8_0 量化误差, 使 target tap 与 drafter 训练分布一致
        // Q8_0 是 per-32-element block, n_cols=5120 是 32 的倍数, 可直接对整个 x 量化
        let x_q8 = maybe_quantize_x_q8(x);
        let x = x_q8.as_ref();

        let n_threads = crate::model::workspace::thread_count();
        // ★ P2-1: runtime AVX2 check 提到循环外
        let use_avx2 = avx2_q1_0_available();
        // ★ P2-5: 阈值从 4096 降到 1024, 让 attn_k/v (1024 rows) 也走线程池
        if n_threads <= 1 || n < 1024 {
            // ★ batch4 专用 kernel: 4 x 单次 pass, 权重只读 1 次 (vs batch_avx2 的 2×)
            #[cfg(target_arch = "x86_64")]
            if use_avx2 && n_batch == 4 {
                let mut tmp = [0.0f32; 4];
                for i in 0..n {
                    unsafe {
                        dot_q1_0_row_batch4_avx2(&self.bytes, i, n_cols, x, &mut tmp);
                    }
                    y[i] = tmp[0];
                    y[n + i] = tmp[1];
                    y[2 * n + i] = tmp[2];
                    y[3 * n + i] = tmp[3];
                }
                return;
            }
            // ★ P1-4: 用 batched kernel 复用 scale/LUT, tmp buffer 避免堆分配
            // ★ vision 路径 n_batch 可能 > 64 (如 576 vision tokens), 栈数组不够用
            //   正常文本 prefill n_batch <= 64, 用栈数组; vision prefill 用 Vec (少见路径)
            let mut tmp_stack = [0.0f32; 64];
            let mut tmp_vec: Vec<f32> = if n_batch > 64 { vec![0.0; n_batch] } else { Vec::new() };
            let tmp: &mut [f32] = if n_batch > 64 { &mut tmp_vec } else { &mut tmp_stack[..n_batch] };
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                for i in 0..n {
                    unsafe {
                        dot_q1_0_row_batch_avx2(&self.bytes, i, n_cols, x, n_cols, n_batch, tmp, 1);
                    }
                    for t in 0..n_batch {
                        y[t * n + i] = tmp[t];
                    }
                }
                return;
            }
            for i in 0..n {
                dot_q1_0_row_batch(&self.bytes, i, n_cols, x, n_cols, n_batch, tmp, 1);
                for t in 0..n_batch {
                    y[t * n + i] = tmp[t];
                }
            }
            return;
        }

        if let Some(pool) = crate::model::workspace::get_thread_pool() {
            let bytes_addr = self.bytes.as_ptr() as usize;
            let bytes_len = self.bytes.len();
            let x_addr = x.as_ptr() as usize;
            let y_addr = y.as_mut_ptr() as usize;
            let chunk = (n + n_threads - 1) / n_threads;

            pool.scatter_wait(n_threads, move |tid| {
                let start = tid * chunk;
                let end = (start + chunk).min(n);
                let bytes = unsafe { std::slice::from_raw_parts(bytes_addr as *const u8, bytes_len) };
                let x = unsafe { std::slice::from_raw_parts(x_addr as *const f32, n_batch * n_cols) };
                // ★ batch4 专用 kernel (线程池路径)
                #[cfg(target_arch = "x86_64")]
                if use_avx2 && n_batch == 4 {
                    let mut tmp = [0.0f32; 4];
                    for i in start..end {
                        unsafe {
                            dot_q1_0_row_batch4_avx2(bytes, i, n_cols, x, &mut tmp);
                        }
                        unsafe {
                            *((y_addr as *mut f32).add(i)) = tmp[0];
                            *((y_addr as *mut f32).add(n + i)) = tmp[1];
                            *((y_addr as *mut f32).add(2 * n + i)) = tmp[2];
                            *((y_addr as *mut f32).add(3 * n + i)) = tmp[3];
                        }
                    }
                    return;
                }
                // ★ vision 路径 n_batch 可能 > 64 (如 576), 栈数组不够用
                let mut tmp_stack = [0.0f32; 64];
                let mut tmp_vec: Vec<f32> = if n_batch > 64 { vec![0.0; n_batch] } else { Vec::new() };
                let tmp: &mut [f32] = if n_batch > 64 { &mut tmp_vec } else { &mut tmp_stack[..n_batch] };
                #[cfg(target_arch = "x86_64")]
                if use_avx2 {
                    // ★ batch4 分块: 4-token 共享权重读取 (vs 2-token 的 n_batch/2 次)
                    //   权重 DRAM 读取次数: ceil(n_batch/4) vs ceil(n_batch/2) (减半)
                    //   对 prefill (n_batch=26, DRAM bound) 显著减少带宽压力
                    let mut t_start = 0usize;
                    while t_start + 4 <= n_batch {
                        let mut tmp4 = [0.0f32; 4];
                        for i in start..end {
                            unsafe {
                                let x_off = t_start * n_cols;
                                let x_slice = std::slice::from_raw_parts(x.as_ptr().add(x_off), 4 * n_cols);
                                dot_q1_0_row_batch4_avx2(bytes, i, n_cols, x_slice, &mut tmp4);
                            }
                            for (j, &v) in tmp4.iter().enumerate() {
                                unsafe { *((y_addr as *mut f32).add((t_start + j) * n + i)) = v; }
                            }
                        }
                        t_start += 4;
                    }
                    // 余数走 batch_avx2 (2-token 分块)
                    if t_start < n_batch {
                        let rem = n_batch - t_start;
                        for i in start..end {
                            unsafe {
                                let x_off = t_start * n_cols;
                                let x_slice = std::slice::from_raw_parts(x.as_ptr().add(x_off), rem * n_cols);
                                dot_q1_0_row_batch_avx2(bytes, i, n_cols, x_slice, n_cols, rem, &mut tmp[..rem], 1);
                            }
                            for t in 0..rem {
                                unsafe { *((y_addr as *mut f32).add((t_start + t) * n + i)) = tmp[t]; }
                            }
                        }
                    }
                    return;
                }
                for i in start..end {
                    dot_q1_0_row_batch(bytes, i, n_cols, x, n_cols, n_batch, tmp, 1);
                    for t in 0..n_batch {
                        unsafe { *((y_addr as *mut f32).add(t * n + i)) = tmp[t]; }
                    }
                }
            });
            return;
        }

        // 回退: std::thread::scope
        let bytes = &self.bytes;
        let y_addr = y.as_mut_ptr() as usize;
        std::thread::scope(|s| {
            let chunk = (n + n_threads - 1) / n_threads;
            let mut handles = Vec::with_capacity(n_threads);
            for tid in 0..n_threads {
                let start = tid * chunk;
                let end = (start + chunk).min(n);
                let h = s.spawn(move || {
                    // ★ batch4 专用 kernel (scope 路径)
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 && n_batch == 4 {
                        let mut tmp = [0.0f32; 4];
                        for i in start..end {
                            unsafe {
                                dot_q1_0_row_batch4_avx2(bytes, i, n_cols, x, &mut tmp);
                            }
                            unsafe {
                                *((y_addr as *mut f32).add(i)) = tmp[0];
                                *((y_addr as *mut f32).add(n + i)) = tmp[1];
                                *((y_addr as *mut f32).add(2 * n + i)) = tmp[2];
                                *((y_addr as *mut f32).add(3 * n + i)) = tmp[3];
                            }
                        }
                        return;
                    }
                    // ★ vision 路径 n_batch 可能 > 64 (如 576), 栈数组不够用
                    let mut tmp_stack = [0.0f32; 64];
                    let mut tmp_vec: Vec<f32> = if n_batch > 64 { vec![0.0; n_batch] } else { Vec::new() };
                    let tmp: &mut [f32] = if n_batch > 64 { &mut tmp_vec } else { &mut tmp_stack[..n_batch] };
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        for i in start..end {
                            unsafe {
                                dot_q1_0_row_batch_avx2(bytes, i, n_cols, x, n_cols, n_batch, tmp, 1);
                            }
                            for t in 0..n_batch {
                                unsafe { *((y_addr as *mut f32).add(t * n + i)) = tmp[t]; }
                            }
                        }
                        return;
                    }
                    for i in start..end {
                        dot_q1_0_row_batch(bytes, i, n_cols, x, n_cols, n_batch, tmp, 1);
                        for t in 0..n_batch {
                            unsafe { *((y_addr as *mut f32).add(t * n + i)) = tmp[t]; }
                        }
                    }
                });
                handles.push(h);
            }
            for h in handles { h.join().unwrap(); }
        });
    }

    /// ★ verify4 dispatch: 推测解码 batch-verify 专用 (k=4 硬编码)
    ///
    /// `x4`: [4 * cols] 行优先 (token t 在 x4[t*cols..])
    /// `y`:  [4 * rows] 行优先 (y[t*rows + i] = dot(W_row_i, x4[t]))
    ///
    /// 与 `matvec_batch_into_slice(n_batch=4)` 的区别:
    /// - 使用 `dot_q1_0_row_verify4_avx2` (零 spill, ~45c/group vs 现状 ~169c)
    /// - 权重 DRAM 流 1 次, 4 token 的 logits 一次产出
    /// - 单次 scatter_wait_stealing barrier (与 decode 路径一致)
    #[allow(unsafe_code)]
    pub fn matvec_verify4_into_slice(&self, x4: &[f32], y: &mut [f32]) {
        let n_cols = self.cols;
        let n = self.rows;
        debug_assert_eq!(x4.len(), 4 * n_cols);
        debug_assert_eq!(y.len(), 4 * n);

        #[cfg(target_arch = "x86_64")]
        if !avx2_q1_0_available() {
            // AVX2 不可用: 回退 batch 通用路径 (正确但慢, 仅老 CPU 触发)
            self.matvec_batch_into_slice(x4, 4, y);
            return;
        }

        if let Some(pool) = crate::model::workspace::get_thread_pool() {
            let bytes_addr = self.bytes.as_ptr() as usize;
            let bytes_len = self.bytes.len();
            let x_addr = x4.as_ptr() as usize;
            let y_addr = y.as_mut_ptr() as usize;
            let steal_chunk = 256;
            pool.scatter_wait_stealing(n, steal_chunk, move |start, end| {
                let bytes = unsafe { std::slice::from_raw_parts(bytes_addr as *const u8, bytes_len) };
                let x = unsafe { std::slice::from_raw_parts(x_addr as *const f32, 4 * n_cols) };
                let mut tmp = [0.0f32; 4];
                for i in start..end {
                    unsafe {
                        crate::tensor::quant::dot_q1_0_row_verify4_avx2(bytes, i, n_cols, x, &mut tmp);
                        *((y_addr as *mut f32).add(i)) = tmp[0];
                        *((y_addr as *mut f32).add(n + i)) = tmp[1];
                        *((y_addr as *mut f32).add(2 * n + i)) = tmp[2];
                        *((y_addr as *mut f32).add(3 * n + i)) = tmp[3];
                    }
                }
            });
            return;
        }

        // 无线程池回退: 串行 verify4
        let mut tmp = [0.0f32; 4];
        for i in 0..n {
            unsafe {
                crate::tensor::quant::dot_q1_0_row_verify4_avx2(&self.bytes, i, n_cols, x4, &mut tmp);
            }
            y[i] = tmp[0];
            y[n + i] = tmp[1];
            y[2 * n + i] = tmp[2];
            y[3 * n + i] = tmp[3];
        }
    }

    /// ★ verify4t dispatch: 交错布局版 verify4 (x 预交错, 64 x loads 全 L1 命中)
    ///
    /// `x4`: [4 * cols] 行优先 (输入, 任意布局)
    /// `y`:  [4 * rows] 行优先输出
    ///
    /// 内部先把 x4 交错到 workspace 缓冲 (每次调用 ~10μs), 再 scatter kernel。
    /// 交错缓冲跨调用复用 (thread_local, 避免每层 80KB 堆分配)。
    #[allow(unsafe_code)]
    pub fn matvec_verify4t_into_slice(&self, x4: &[f32], y: &mut [f32]) {
        let n_cols = self.cols;
        let n = self.rows;
        debug_assert_eq!(x4.len(), 4 * n_cols);
        debug_assert_eq!(y.len(), 4 * n);

        #[cfg(target_arch = "x86_64")]
        if !avx2_q1_0_available() {
            self.matvec_batch_into_slice(x4, 4, y);
            return;
        }

        // x 交错缓冲: thread_local 复用 (每 worker 独立, scatter 内部按行不跨 worker)
        use std::cell::RefCell;
        thread_local! {
            static X_INT: RefCell<Vec<f32>> = RefCell::new(Vec::new());
        }
        let interleaved = X_INT.with(|buf| {
            let mut b = buf.borrow_mut();
            if b.len() < 4 * n_cols {
                b.resize(4 * n_cols, 0.0);
            }
            crate::tensor::quant::interleave_x4_verify(x4, n_cols, &mut b);
            // 返回裸指针避免 RefCell 跨 await/作用域借用问题; scatter_wait 阻塞至完成,
            // 期间本线程不会再 borrow 此 thread_local
            b.as_ptr() as usize
        });

        if let Some(pool) = crate::model::workspace::get_thread_pool() {
            let bytes_addr = self.bytes.as_ptr() as usize;
            let bytes_len = self.bytes.len();
            let y_addr = y.as_mut_ptr() as usize;
            let steal_chunk = 256;
            pool.scatter_wait_stealing(n, steal_chunk, move |start, end| {
                let bytes = unsafe { std::slice::from_raw_parts(bytes_addr as *const u8, bytes_len) };
                let x_int = unsafe { std::slice::from_raw_parts(interleaved as *const f32, 4 * n_cols) };
                let mut tmp = [0.0f32; 4];
                for i in start..end {
                    unsafe {
                        crate::tensor::quant::dot_q1_0_row_verify4t_avx2(bytes, i, n_cols, x_int, &mut tmp);
                        *((y_addr as *mut f32).add(i)) = tmp[0];
                        *((y_addr as *mut f32).add(n + i)) = tmp[1];
                        *((y_addr as *mut f32).add(2 * n + i)) = tmp[2];
                        *((y_addr as *mut f32).add(3 * n + i)) = tmp[3];
                    }
                }
            });
            return;
        }

        // 无线程池回退: 串行 verify4t
        let x_int = {
            let mut buf = vec![0.0f32; 4 * n_cols];
            crate::tensor::quant::interleave_x4_verify(x4, n_cols, &mut buf);
            buf
        };
        let mut tmp = [0.0f32; 4];
        for i in 0..n {
            unsafe {
                crate::tensor::quant::dot_q1_0_row_verify4t_avx2(&self.bytes, i, n_cols, &x_int, &mut tmp);
            }
            y[i] = tmp[0];
            y[n + i] = tmp[1];
            y[2 * n + i] = tmp[2];
            y[3 * n + i] = tmp[3];
        }
    }

    /// ★ P0-B: 多矩阵合并 matvec — 多个共享同一输入 x 的矩阵在单次线程池 barrier 内完成
    ///
    /// **动机**: 原 attention Q/K/V、MLP gate/up、SSM qkv/gate 各自独立调用 matvec_into_slice,
    ///   每次 trigger 一次 scatter_wait barrier (Arc alloc + N×Box alloc + N×channel send + spin)。
    ///   合并后 N 个矩阵的 total_rows 一次性分发, barrier 数从 3 降到 1。
    ///
    /// **额外收益**:
    ///   - x 向量 (20KB) 在多个矩阵间自然驻留 L2, 消除跨 dispatch 的 L2 eviction
    ///   - total_rows 更大 → chunk 更大 → 相对 dispatch 开销更低, load balancing 更好
    ///
    /// `matrices[i].cols` 必须全等于 `x.len()`; `outputs[i].len()` 必须等于 `matrices[i].rows`。
    #[allow(unsafe_code)]
    pub fn matvec_multi_into_slice(
        x: &[f32],
        matrices: &[&Q1_0Matrix],
        outputs: &mut [&mut [f32]],
    ) {
        let n_entries = matrices.len();
        debug_assert_eq!(n_entries, outputs.len());
        if n_entries == 0 {
            return;
        }
        let k = matrices[0].cols;
        debug_assert_eq!(x.len(), k);
        for i in 0..n_entries {
            debug_assert_eq!(matrices[i].cols, k);
            debug_assert_eq!(outputs[i].len(), matrices[i].rows);
        }

        // 单矩阵直接走原路径 (避免 entry table 开销)
        if n_entries == 1 {
            matrices[0].matvec_into_slice(x, outputs[0]);
            return;
        }

        let n_threads = crate::model::workspace::thread_count();
        let use_avx2 = avx2_q1_0_available();

        // 构建 entry table: 每个 entry 记录矩阵的 raw 指针 + 全局行偏移
        // ★ 栈分配 [Entry; 8] 避免 Vec heap alloc (每 token 128 次调用)
        //   (闭包需 'static, 不能持有借用; 用 raw pointer 绕过生命周期)
        #[derive(Clone, Copy)]
        struct Entry {
            bytes_addr: usize,
            bytes_len: usize,
            rows: usize,
            y_addr: usize,
            row_start: usize, // 在全局虚拟行空间中的起点
        }
        let mut entries: [Entry; 8] = [Entry {
            bytes_addr: 0, bytes_len: 0, rows: 0, y_addr: 0, row_start: 0,
        }; 8];
        debug_assert!(n_entries <= 8);
        let mut total_rows = 0usize;
        for i in 0..n_entries {
            entries[i] = Entry {
                bytes_addr: matrices[i].bytes.as_ptr() as usize,
                bytes_len: matrices[i].bytes.len(),
                rows: matrices[i].rows,
                y_addr: outputs[i].as_mut_ptr() as usize,
                row_start: total_rows,
            };
            total_rows += matrices[i].rows;
        }

        // 小矩阵或单线程: 顺序调用各矩阵 (仍用 dual-row AVX2 kernel)
        if n_threads <= 1 || total_rows < 1024 {
            for i in 0..n_entries {
                matrices[i].matvec_into_slice(x, outputs[i]);
            }
            return;
        }

        // 持久线程池: 单次 barrier 分发所有矩阵的行
        if let Some(pool) = crate::model::workspace::get_thread_pool() {
            let x_addr = x.as_ptr() as usize;
            // ★ Work-stealing: chunk_size=128 让快线程多抢 chunk, 改善负载均衡
            //   (原 static chunk: tid × (total/N), 慢线程拖整 barrier)
            //   bench_klab 交错实测 decode matvec 128 比 256 快 ~15%@9w (E 核尾巴更短)
            let steal_chunk = matvec_steal_chunk();

            pool.scatter_wait_stealing(total_rows, steal_chunk, move |start, end| {
                if start >= end {
                    return;
                }
                let x = unsafe { std::slice::from_raw_parts(x_addr as *const f32, k) };

                // 在虚拟行空间中迭代, 找到每个 entry 的连续行段
                let mut virt = start;
                let mut entry_idx = 0;
                while virt < end {
                    // 找到 virt 所属的 entry (entries 按 row_start 升序, 线性扫描即可)
                    while entry_idx + 1 < n_entries && virt >= entries[entry_idx + 1].row_start {
                        entry_idx += 1;
                    }
                    let e = &entries[entry_idx];
                    let local_start = virt - e.row_start;
                    let local_end = (end - e.row_start).min(e.rows);
                    if local_start >= local_end {
                        break;
                    }
                    let bytes = unsafe { std::slice::from_raw_parts(e.bytes_addr as *const u8, e.bytes_len) };

                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        // ★ P0-H: 四行并行
                        let mut i = local_start;
                        while i + 3 < local_end {
                            let (y0, y1, y2, y3) = unsafe { dot_q1_0_row_quad_avx2(bytes, i, i + 1, i + 2, i + 3, k, x) };
                            unsafe {
                                *((e.y_addr as *mut f32).add(i)) = y0;
                                *((e.y_addr as *mut f32).add(i + 1)) = y1;
                                *((e.y_addr as *mut f32).add(i + 2)) = y2;
                                *((e.y_addr as *mut f32).add(i + 3)) = y3;
                            }
                            i += 4;
                        }
                        if i + 2 < local_end {
                            let (y0, y1, y2) = unsafe { dot_q1_0_row_triple_avx2(bytes, i, i + 1, i + 2, k, x) };
                            unsafe {
                                *((e.y_addr as *mut f32).add(i)) = y0;
                                *((e.y_addr as *mut f32).add(i + 1)) = y1;
                                *((e.y_addr as *mut f32).add(i + 2)) = y2;
                            }
                        } else if i + 1 < local_end {
                            let (y0, y1) = unsafe { dot_q1_0_row_dual_avx2(bytes, i, i + 1, k, x) };
                            unsafe {
                                *((e.y_addr as *mut f32).add(i)) = y0;
                                *((e.y_addr as *mut f32).add(i + 1)) = y1;
                            }
                        } else if i < local_end {
                            unsafe {
                                *((e.y_addr as *mut f32).add(i)) = dot_q1_0_row_avx2(bytes, i, k, x);
                            }
                        }
                    } else {
                        for i in local_start..local_end {
                            unsafe {
                                *((e.y_addr as *mut f32).add(i)) = dot_q1_0_row_scalar(bytes, i, k, x);
                            }
                        }
                    }
                    virt = e.row_start + local_end;
                    entry_idx += 1;
                }
            });
            return;
        }

        // 回退: 顺序调用
        for i in 0..n_entries {
            matrices[i].matvec_into_slice(x, outputs[i]);
        }
    }

    /// 批量 matvec 累加: y[t][i] += dot(W_row_i, x[t])
    #[allow(unsafe_code)]
    pub fn matvec_add_batch_into_slice(&self, x: &[f32], n_batch: usize, y: &mut [f32]) {
        let n_cols = self.cols;
        let n = self.rows;
        debug_assert_eq!(x.len(), n_batch * n_cols);
        debug_assert_eq!(y.len(), n_batch * n);

        if n_batch <= 1 {
            self.matvec_add_into_slice(x, &mut y[..n]);
            return;
        }

        // ★ GEMM dispatch (与 matvec_batch_into_slice L489 条件一致):
        //   大矩阵 (rows >= 4096) + 大 batch (>= 64) + 宽列 (cols >= 8192) 时走 k-lane GEMM。
        //   down_proj (rows=5120, cols=17408) 宽列受益于 GEMM 预缩放 scratch;
        //   attn_output/ssm_out (cols=6144) 窄列 GEMM prep 开销 > FMA 收益, 不 dispatch。
        //   GEMM 覆盖写入临时 buffer, 再逐元素加到 y (保持 += 语义)。
        //   临时 buffer = n_batch × n × 4B (down_proj 64×5120×4 = 1.28MB, 可接受)。
        #[cfg(target_arch = "x86_64")]
        if n_batch >= 64 && n >= 4096 && n_cols >= 8192 && crate::tensor::quant::avx2_q1_0_available() {
            let mut tmp = vec![0.0f32; n_batch * n];
            crate::tensor::gemm::gemm_q1_0_batch(&self.bytes, n, n_cols, x, n_batch, &mut tmp);
            // y[t][i] += tmp[t][i]
            for t in 0..n_batch {
                let dst = t * n;
                let src = t * n;
                for i in 0..n {
                    y[dst + i] += tmp[src + i];
                }
            }
            return;
        }

        let n_threads = crate::model::workspace::thread_count();
        // ★ P2-1: runtime AVX2 check 提到循环外
        let use_avx2 = avx2_q1_0_available();
        // ★ P2-5: 阈值从 4096 降到 1024, 让 attn_k/v (1024 rows) 也走线程池
        if n_threads <= 1 || n < 1024 {
            // ★ batch4 专用 kernel: 4 x 单次 pass, 权重只读 1 次
            #[cfg(target_arch = "x86_64")]
            if use_avx2 && n_batch == 4 {
                let mut tmp = [0.0f32; 4];
                for i in 0..n {
                    unsafe {
                        dot_q1_0_row_batch4_avx2(&self.bytes, i, n_cols, x, &mut tmp);
                    }
                    y[i] += tmp[0];
                    y[n + i] += tmp[1];
                    y[2 * n + i] += tmp[2];
                    y[3 * n + i] += tmp[3];
                }
                return;
            }
            // ★ P1-4: batched kernel
            // ★ vision 路径 n_batch 可能 > 64 (如 576), 栈数组不够用
            let mut tmp_stack = [0.0f32; 64];
            let mut tmp_vec: Vec<f32> = if n_batch > 64 { vec![0.0; n_batch] } else { Vec::new() };
            let tmp: &mut [f32] = if n_batch > 64 { &mut tmp_vec } else { &mut tmp_stack[..n_batch] };
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                for i in 0..n {
                    unsafe {
                        dot_q1_0_row_batch_avx2(&self.bytes, i, n_cols, x, n_cols, n_batch, tmp, 1);
                    }
                    for t in 0..n_batch {
                        y[t * n + i] += tmp[t];
                    }
                }
                return;
            }
            for i in 0..n {
                dot_q1_0_row_batch(&self.bytes, i, n_cols, x, n_cols, n_batch, tmp, 1);
                for t in 0..n_batch {
                    y[t * n + i] += tmp[t];
                }
            }
            return;
        }

        if let Some(pool) = crate::model::workspace::get_thread_pool() {
            let bytes_addr = self.bytes.as_ptr() as usize;
            let bytes_len = self.bytes.len();
            let x_addr = x.as_ptr() as usize;
            let y_addr = y.as_mut_ptr() as usize;
            let chunk = (n + n_threads - 1) / n_threads;

            pool.scatter_wait(n_threads, move |tid| {
                let start = tid * chunk;
                let end = (start + chunk).min(n);
                let bytes = unsafe { std::slice::from_raw_parts(bytes_addr as *const u8, bytes_len) };
                let x = unsafe { std::slice::from_raw_parts(x_addr as *const f32, n_batch * n_cols) };
                // ★ batch4 专用 kernel (线程池路径)
                #[cfg(target_arch = "x86_64")]
                if use_avx2 && n_batch == 4 {
                    let mut tmp = [0.0f32; 4];
                    for i in start..end {
                        unsafe {
                            dot_q1_0_row_batch4_avx2(bytes, i, n_cols, x, &mut tmp);
                        }
                        unsafe {
                            *((y_addr as *mut f32).add(i)) += tmp[0];
                            *((y_addr as *mut f32).add(n + i)) += tmp[1];
                            *((y_addr as *mut f32).add(2 * n + i)) += tmp[2];
                            *((y_addr as *mut f32).add(3 * n + i)) += tmp[3];
                        }
                    }
                    return;
                }
                // ★ vision 路径 n_batch 可能 > 64 (如 576), 栈数组不够用
                let mut tmp_stack = [0.0f32; 64];
                let mut tmp_vec: Vec<f32> = if n_batch > 64 { vec![0.0; n_batch] } else { Vec::new() };
                let tmp: &mut [f32] = if n_batch > 64 { &mut tmp_vec } else { &mut tmp_stack[..n_batch] };
                #[cfg(target_arch = "x86_64")]
                if use_avx2 {
                    // ★ batch4 分块: 4-token 共享权重读取 (vs 2-token 的 n_batch/2 次)
                    let mut t_start = 0usize;
                    while t_start + 4 <= n_batch {
                        let mut tmp4 = [0.0f32; 4];
                        for i in start..end {
                            unsafe {
                                let x_off = t_start * n_cols;
                                let x_slice = std::slice::from_raw_parts(x.as_ptr().add(x_off), 4 * n_cols);
                                dot_q1_0_row_batch4_avx2(bytes, i, n_cols, x_slice, &mut tmp4);
                            }
                            for (j, &v) in tmp4.iter().enumerate() {
                                unsafe { *((y_addr as *mut f32).add((t_start + j) * n + i)) += v; }
                            }
                        }
                        t_start += 4;
                    }
                    // 余数走 batch_avx2 (2-token 分块)
                    if t_start < n_batch {
                        let rem = n_batch - t_start;
                        for i in start..end {
                            unsafe {
                                let x_off = t_start * n_cols;
                                let x_slice = std::slice::from_raw_parts(x.as_ptr().add(x_off), rem * n_cols);
                                dot_q1_0_row_batch_avx2(bytes, i, n_cols, x_slice, n_cols, rem, &mut tmp[..rem], 1);
                            }
                            for t in 0..rem {
                                unsafe { *((y_addr as *mut f32).add((t_start + t) * n + i)) += tmp[t]; }
                            }
                        }
                    }
                    return;
                }
                for i in start..end {
                    dot_q1_0_row_batch(bytes, i, n_cols, x, n_cols, n_batch, tmp, 1);
                    for t in 0..n_batch {
                        unsafe { *((y_addr as *mut f32).add(t * n + i)) += tmp[t]; }
                    }
                }
            });
            return;
        }

        // 回退: std::thread::scope
        let bytes = &self.bytes;
        let y_addr = y.as_mut_ptr() as usize;
        std::thread::scope(|s| {
            let chunk = (n + n_threads - 1) / n_threads;
            let mut handles = Vec::with_capacity(n_threads);
            for tid in 0..n_threads {
                let start = tid * chunk;
                let end = (start + chunk).min(n);
                let h = s.spawn(move || {
                    // ★ batch4 专用 kernel (scope 路径)
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 && n_batch == 4 {
                        let mut tmp = [0.0f32; 4];
                        for i in start..end {
                            unsafe {
                                dot_q1_0_row_batch4_avx2(bytes, i, n_cols, x, &mut tmp);
                            }
                            unsafe {
                                *((y_addr as *mut f32).add(i)) += tmp[0];
                                *((y_addr as *mut f32).add(n + i)) += tmp[1];
                                *((y_addr as *mut f32).add(2 * n + i)) += tmp[2];
                                *((y_addr as *mut f32).add(3 * n + i)) += tmp[3];
                            }
                        }
                        return;
                    }
                    // ★ vision 路径 n_batch 可能 > 64 (如 576), 栈数组不够用
                    let mut tmp_stack = [0.0f32; 64];
                    let mut tmp_vec: Vec<f32> = if n_batch > 64 { vec![0.0; n_batch] } else { Vec::new() };
                    let tmp: &mut [f32] = if n_batch > 64 { &mut tmp_vec } else { &mut tmp_stack[..n_batch] };
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        for i in start..end {
                            unsafe {
                                dot_q1_0_row_batch_avx2(bytes, i, n_cols, x, n_cols, n_batch, tmp, 1);
                            }
                            for t in 0..n_batch {
                                unsafe { *((y_addr as *mut f32).add(t * n + i)) += tmp[t]; }
                            }
                        }
                        return;
                    }
                    for i in start..end {
                        dot_q1_0_row_batch(bytes, i, n_cols, x, n_cols, n_batch, tmp, 1);
                        for t in 0..n_batch {
                            unsafe { *((y_addr as *mut f32).add(t * n + i)) += tmp[t]; }
                        }
                    }
                });
                handles.push(h);
            }
            for h in handles { h.join().unwrap(); }
        });
    }
}

impl std::fmt::Debug for Q1_0Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Q1_0Matrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("bytes_len", &self.bytes.len())
            .finish()
    }
}

/// 非块级权重(全局)
pub struct GlobalWeights {
    pub token_embd: Q1_0Matrix,
    pub output_norm: Tensor,
    pub output: Q1_0Matrix,
}

/// SSM 块的权重
pub struct SsmBlockWeights {
    pub attn_gate: Q1_0Matrix,    // [hidden, ssm_inner]
    pub attn_qkv: Q1_0Matrix,     // [hidden, qkv_dim]
    pub ssm_alpha: Q1_0Matrix,    // [hidden, ts_rank]
    pub ssm_beta: Q1_0Matrix,     // [hidden, ts_rank]
    pub ssm_out: Q1_0Matrix,      // [ssm_inner, hidden]
    pub ssm_conv1d: Tensor,       // [conv_kernel, qkv_dim]  F32
    pub ssm_a: Tensor,            // [ts_rank]               F32
    pub ssm_dt_bias: Tensor,      // [ts_rank]               F32
    pub ssm_norm: Tensor,         // [state_size]            F32
    pub ffn_gate: Q1_0Matrix,
    pub ffn_up: Q1_0Matrix,
    pub ffn_down: Q1_0Matrix,
    pub attn_norm: Tensor,
    pub post_attention_norm: Tensor,
}

/// 全注意力块的权重
pub struct FullAttentionBlockWeights {
    pub attn_q: Q1_0Matrix,
    pub attn_k: Q1_0Matrix,
    pub attn_v: Q1_0Matrix,
    pub attn_output: Q1_0Matrix,
    pub attn_q_norm: Tensor,
    pub attn_k_norm: Tensor,
    pub ffn_gate: Q1_0Matrix,
    pub ffn_up: Q1_0Matrix,
    pub ffn_down: Q1_0Matrix,
    pub attn_norm: Tensor,
    pub post_attention_norm: Tensor,
}

pub enum BlockWeights {
    Ssm(SsmBlockWeights),
    FullAttention(FullAttentionBlockWeights),
}

impl BlockWeights {
    pub fn as_full_attention(&self) -> &FullAttentionBlockWeights {
        match self {
            BlockWeights::FullAttention(w) => w,
            _ => panic!("expected FullAttention block"),
        }
    }

    pub fn as_ssm(&self) -> &SsmBlockWeights {
        match self {
            BlockWeights::Ssm(w) => w,
            _ => panic!("expected SSM block"),
        }
    }

    pub fn post_norm_and_ffn(
        &self,
    ) -> (&Tensor, &Q1_0Matrix, &Q1_0Matrix, &Q1_0Matrix) {
        match self {
            BlockWeights::Ssm(w) => (
                &w.post_attention_norm,
                &w.ffn_gate,
                &w.ffn_up,
                &w.ffn_down,
            ),
            BlockWeights::FullAttention(w) => (
                &w.post_attention_norm,
                &w.ffn_gate,
                &w.ffn_up,
                &w.ffn_down,
            ),
        }
    }
}

/// 一次性加载所有 block(原始字节,~3.2GB)
pub struct LoadedWeights {
    pub global: GlobalWeights,
    pub blocks: Vec<BlockWeights>,
}

impl LoadedWeights {
    pub fn load_all(gguf: &GgufFile, cfg: &crate::model::Config) -> crate::Result<Self> {
        let token_embd = Q1_0Matrix::from_gguf(gguf, "token_embd.weight")?;
        let output = Q1_0Matrix::from_gguf(gguf, "output.weight")?;
        let output_norm = Self::load_tensor(gguf, "output_norm.weight")?;

        let mut blocks = Vec::with_capacity(cfg.block_count);
        for blk_idx in 0..cfg.block_count {
            if blk_idx % 4 == 0 {
                eprint!("\r[load] block {blk_idx}/{}", cfg.block_count);
            }
            let w = if cfg.is_full_attention_block(blk_idx) {
                BlockWeights::FullAttention(FullAttentionBlockWeights {
                    attn_q: Self::load_q1_0_block(gguf, blk_idx, "attn_q.weight")?,
                    attn_k: Self::load_q1_0_block(gguf, blk_idx, "attn_k.weight")?,
                    attn_v: Self::load_q1_0_block(gguf, blk_idx, "attn_v.weight")?,
                    attn_output: Self::load_q1_0_block(gguf, blk_idx, "attn_output.weight")?,
                    attn_q_norm: Self::load_block_tensor(gguf, blk_idx, "attn_q_norm.weight")?,
                    attn_k_norm: Self::load_block_tensor(gguf, blk_idx, "attn_k_norm.weight")?,
                    ffn_gate: Self::load_q1_0_block(gguf, blk_idx, "ffn_gate.weight")?,
                    ffn_up: Self::load_q1_0_block(gguf, blk_idx, "ffn_up.weight")?,
                    ffn_down: Self::load_q1_0_block(gguf, blk_idx, "ffn_down.weight")?,
                    attn_norm: Self::load_block_tensor(gguf, blk_idx, "attn_norm.weight")?,
                    post_attention_norm: Self::load_block_tensor(gguf, blk_idx, "post_attention_norm.weight")?,
                })
            } else {
                BlockWeights::Ssm(SsmBlockWeights {
                    attn_gate: Self::load_q1_0_block(gguf, blk_idx, "attn_gate.weight")?,
                    attn_qkv: Self::load_q1_0_block(gguf, blk_idx, "attn_qkv.weight")?,
                    ssm_alpha: Self::load_q1_0_block(gguf, blk_idx, "ssm_alpha.weight")?,
                    ssm_beta: Self::load_q1_0_block(gguf, blk_idx, "ssm_beta.weight")?,
                    ssm_out: Self::load_q1_0_block(gguf, blk_idx, "ssm_out.weight")?,
                    ssm_conv1d: {
                        // ★ P1-5: 转置 ssm_conv1d 权重 [qkv_full_len, conv_k] → [conv_k, qkv_full_len]
                        //   原布局 data[ch * conv_k + t] (stride=conv_k 访问, 无法 SIMD)
                        //   转置后 data[t * qkv + ch] (每个 t 连续, 可 AVX2 FMA)
                        //   一次性成本: 48 blocks × 40960 元素 ≈ 2ms, 运行时每 token 省 stride 访问
                        let mut t = Self::load_block_tensor(gguf, blk_idx, "ssm_conv1d.weight")?;
                        let conv_k = cfg.ssm_conv_kernel;
                        let qkv = t.data.len() / conv_k;
                        let mut transposed = vec![0.0f32; t.data.len()];
                        for ch in 0..qkv {
                            for t_idx in 0..conv_k {
                                transposed[t_idx * qkv + ch] = t.data[ch * conv_k + t_idx];
                            }
                        }
                        t.data = transposed;
                        t.dims = vec![conv_k, qkv];
                        t
                    },
                    ssm_a: Self::load_block_tensor(gguf, blk_idx, "ssm_a")?,
                    ssm_dt_bias: Self::load_block_tensor(gguf, blk_idx, "ssm_dt.bias")?,
                    ssm_norm: Self::load_block_tensor(gguf, blk_idx, "ssm_norm.weight")?,
                    ffn_gate: Self::load_q1_0_block(gguf, blk_idx, "ffn_gate.weight")?,
                    ffn_up: Self::load_q1_0_block(gguf, blk_idx, "ffn_up.weight")?,
                    ffn_down: Self::load_q1_0_block(gguf, blk_idx, "ffn_down.weight")?,
                    attn_norm: Self::load_block_tensor(gguf, blk_idx, "attn_norm.weight")?,
                    post_attention_norm: Self::load_block_tensor(gguf, blk_idx, "post_attention_norm.weight")?,
                })
            };
            blocks.push(w);
        }
        eprintln!("\r[load] all {} blocks loaded", cfg.block_count);

        Ok(Self {
            global: GlobalWeights {
                token_embd,
                output_norm,
                output,
            },
            blocks,
        })
    }

    fn load_tensor(gguf: &GgufFile, name: &str) -> crate::Result<Tensor> {
        let info = gguf
            .find_tensor(name)
            .ok_or_else(|| BonsaiError::Model(format!("tensor {name} not found")))?;
        let data = gguf.tensor_data(info)?;
        load_as_f32(data, &info.dims, info.dtype)
    }

    fn load_block_tensor(
        gguf: &GgufFile,
        block_idx: usize,
        tensor_suffix: &str,
    ) -> crate::Result<Tensor> {
        let name = format!("blk.{block_idx}.{tensor_suffix}");
        Self::load_tensor(gguf, &name)
    }

    fn load_q1_0_block(
        gguf: &GgufFile,
        block_idx: usize,
        tensor_suffix: &str,
    ) -> crate::Result<Q1_0Matrix> {
        let name = format!("blk.{block_idx}.{tensor_suffix}");
        Q1_0Matrix::from_gguf(gguf, &name)
    }

    pub fn print_dtype_summary(gguf: &GgufFile) {
        let mut q1_0 = 0;
        let mut f32 = 0;
        let mut other: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
        for t in &gguf.tensors {
            match t.dtype {
                TensorType::Q1_0 => q1_0 += 1,
                TensorType::F32 => f32 += 1,
                other_type => {
                    *other.entry(other_type.name().to_string()).or_insert(0) += 1;
                }
            }
        }
        println!("[weights] Q1_0 tensors: {q1_0}");
        println!("[weights] F32 tensors : {f32}");
        for (k, v) in other {
            println!("[weights] {k} tensors: {v}");
        }
    }
}

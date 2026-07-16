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
use crate::gguf::tensor_info::{TensorInfo, TensorType};
use crate::tensor::tensor::Tensor;
use crate::tensor::tensor::load_as_f32;
use crate::tensor::quant::{dequantize_q1_0_row, dot_q1_0_row};
use crate::BonsaiError;

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

    /// 反量化单行,返回长度 = cols 的 F32 向量(用于 embedding 查找)
    #[inline]
    pub fn row(&self, row_idx: usize) -> Vec<f32> {
        dequantize_q1_0_row(&self.bytes, row_idx, self.cols)
    }

    /// 反量化单行,写入 caller 提供的 slice(避免堆分配)
    #[inline]
    pub fn row_into_slice(&self, row_idx: usize, y: &mut [f32]) {
        crate::tensor::quant::dequantize_q1_0_row_into(&self.bytes, row_idx, self.cols, y);
    }

    /// 流式 GEMM:`y[i] = sum_j W[i, j] * x[j]`
    ///
    /// 使用融合点积 `dot_q1_0_row`:直接在 Q1_0 原始字节上计算点积,
    /// 无需中间 F32 缓冲。内层循环是无分支 FMA,可被编译器自动向量化(AVX2)。
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        let k = self.cols;
        let n = self.rows;
        debug_assert_eq!(x.len(), k);
        let mut y = vec![0.0f32; n];
        self.matvec_into_slice(x, &mut y);
        y
    }

    /// 流式 GEMM into caller-provided slice(避免堆分配)
    ///
    /// `y[i] = dot_q1_0_row(W, i, k, x)`,覆盖写入 `y`(不是累加)。
    /// `y.len()` 必须等于 `self.rows`。
    ///
    /// **多线程并行**:当 `self.rows >= 2 * n_threads` 时按行切分到 N 个 OS thread,
    /// 每 thread 处理 rows/N 行,各自独立累加。对小矩阵(rows < 256)走串行路径
    /// 避免 spawn 开销。threads 数由 `DAIZA_THREADS` env var 控制,默认物理核数。
    #[inline]
    pub fn matvec_into_slice(&self, x: &[f32], y: &mut [f32]) {
        let k = self.cols;
        let n = self.rows;
        debug_assert_eq!(x.len(), k);
        debug_assert_eq!(y.len(), n);

        let n_threads = crate::model::workspace::thread_count();
        // 阈值:行数太少时 spawn 开销超过并行收益
        // 4096 阈值:attn_k/attn_v(1024)、ssm_alpha/beta(48)等小矩阵走串行,
        // 避免 400+ 次 spawn/join 开销(~10-40ms/token on Windows)
        if n_threads <= 1 || n < 4096 {
            for i in 0..n {
                y[i] = dot_q1_0_row(&self.bytes, i, k, x);
            }
            return;
        }

        let bytes = &self.bytes;
        std::thread::scope(|s| {
            // 用 chunks_mut 自动拆分 y 为不重叠 mut slice(Sync + Send 都 OK)
            let chunk = (n + n_threads - 1) / n_threads;
            let mut handles = Vec::with_capacity(n_threads);
            let mut row_start = 0usize;
            for y_chunk in y.chunks_mut(chunk) {
                let chunk_len = y_chunk.len();
                let start = row_start;
                row_start += chunk_len;
                let h = s.spawn(move || {
                    for (i, y_i) in y_chunk.iter_mut().enumerate() {
                        *y_i = dot_q1_0_row(bytes, start + i, k, x);
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
    #[inline]
    pub fn matvec_add_into_slice(&self, x: &[f32], y: &mut [f32]) {
        let k = self.cols;
        let n = self.rows;
        debug_assert_eq!(x.len(), k);
        debug_assert_eq!(y.len(), n);

        let n_threads = crate::model::workspace::thread_count();
        if n_threads <= 1 || n < 4096 {
            for i in 0..n {
                y[i] += dot_q1_0_row(&self.bytes, i, k, x);
            }
            return;
        }

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
                    for (i, y_i) in y_chunk.iter_mut().enumerate() {
                        *y_i += dot_q1_0_row(bytes, start + i, k, x);
                    }
                });
                handles.push(h);
            }
            for h in handles {
                h.join().unwrap();
            }
        });
    }

    /// matvec 后加 bias
    pub fn matvec_add_bias(&self, x: &[f32], bias: &[f32]) -> Vec<f32> {
        let mut y = self.matvec(x);
        for i in 0..y.len() {
            if i < bias.len() {
                y[i] += bias[i];
            }
        }
        y
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
                    ssm_conv1d: Self::load_block_tensor(gguf, blk_idx, "ssm_conv1d.weight")?,
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

/// 兼容旧 API
pub struct WeightLoader<'a> {
    pub gguf: &'a GgufFile,
}

impl<'a> WeightLoader<'a> {
    pub fn new(gguf: &'a GgufFile) -> Self {
        Self { gguf }
    }
    pub fn print_dtype_summary(gguf: &GgufFile) {
        LoadedWeights::print_dtype_summary(gguf);
    }
}

pub fn expect_tensor(
    gguf: &GgufFile,
    name: &str,
    expected_dtype: TensorType,
) -> crate::Result<TensorInfo> {
    let info = gguf
        .find_tensor(name)
        .ok_or_else(|| BonsaiError::Model(format!("missing tensor {name}")))?;
    if info.dtype != expected_dtype {
        return Err(BonsaiError::Model(format!(
            "tensor {name}: expected dtype {:?} but got {:?}",
            expected_dtype, info.dtype
        )));
    }
    Ok(info.clone())
}

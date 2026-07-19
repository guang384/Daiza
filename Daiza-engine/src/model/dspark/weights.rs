//! DSpark drafter 权重加载
//!
//! 权重量化格式:
//! - Q4_1: attn_q/k/v/output, ffn_gate/up/down, fc, markov_head_b, output (LM head)
//! - Q1_0: token_embd (与 target 共享)
//! - Iq1M: markov_head_a, log_snr_fc1, log_snr_fc2
//! - F32: norms, bias, output_norm
//!
//! 内存策略: Q4_1 和 Iq1M 权重保持原始字节, matvec 时逐行反量化。
//! drafter 总权重 ~200MB (Q4_1) + ~80MB (Iq1M) = ~280MB。

use crate::gguf::parser::GgufFile;
use crate::gguf::tensor_info::TensorType;
use crate::tensor::quant::{
    dot_q4_1_row_scalar, dot_q1_0_row_scalar,
    avx2_q4_1_available,
};
#[cfg(target_arch = "x86_64")]
use crate::tensor::quant::{dot_q4_1_row_avx2, dot_q4_1_row_batch_avx2};
use crate::tensor::iq1m::dot_iq1m_row_scalar;
use crate::model::workspace::get_thread_pool;
use crate::BonsaiError;

use super::config::DrafterConfig;

/// 通用量化矩阵 (支持 Q4_1 / Q1_0 / Iq1M / F32)
pub struct DrafterMatrix {
    pub bytes: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
    pub dtype: TensorType,
}

impl DrafterMatrix {
    pub fn from_gguf(gguf: &GgufFile, name: &str) -> crate::Result<Self> {
        let info = gguf.find_tensor(name)
            .ok_or_else(|| BonsaiError::Model(format!("drafter tensor {name} not found")))?;
        let data = gguf.tensor_data(info)?.to_vec();
        // GGUF dims: dims[0] = cols (内层), dims[1] = rows (外层)
        let cols = info.dims[0] as usize;
        let rows = if info.dims.len() > 1 { info.dims[1] as usize } else { 1 };
        Ok(Self {
            bytes: data,
            rows,
            cols,
            dtype: info.dtype,
        })
    }

    /// matvec: y[i] = dot(W_row_i, x), 覆盖写入 y
    ///
    /// ★ Q4_1 走 AVX2 kernel + 多线程并行 (drafter 权重大, 单线程标量是 95.9% 瓶颈)
    /// 小矩阵 (rows < 64) 走单线程避免 thread pool 开销
    #[allow(unsafe_code)]
    pub fn matvec_into_slice(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(y.len(), self.rows);
        let n = self.rows;
        let k = self.cols;

        match self.dtype {
            TensorType::Q4_1 => {
                #[cfg(target_arch = "x86_64")]
                let use_avx2 = avx2_q4_1_available();
                #[cfg(not(target_arch = "x86_64"))]
                let use_avx2 = false;

                // 小矩阵单线程 (避免 thread pool 开销)
                if n < 64 || get_thread_pool().is_none() {
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        for i in 0..n {
                            unsafe { y[i] = dot_q4_1_row_avx2(&self.bytes, i, k, x); }
                        }
                        return;
                    }
                    for i in 0..n {
                        y[i] = dot_q4_1_row_scalar(&self.bytes, i, k, x);
                    }
                    return;
                }

                // 大矩阵多线程并行
                let pool = get_thread_pool().unwrap();
                let n_threads = pool.n_threads();
                let chunk = (n + n_threads - 1) / n_threads;
                let bytes_addr = self.bytes.as_ptr() as usize;
                let bytes_len = self.bytes.len();
                let x_addr = x.as_ptr() as usize;
                let y_addr = y.as_mut_ptr() as usize;
                pool.scatter_wait(n_threads, move |tid| {
                    let start = tid * chunk;
                    let end = (start + chunk).min(n);
                    if start >= end { return; }
                    let bytes = unsafe { std::slice::from_raw_parts(bytes_addr as *const u8, bytes_len) };
                    let x = unsafe { std::slice::from_raw_parts(x_addr as *const f32, k) };
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        for i in start..end {
                            unsafe {
                                *((y_addr as *mut f32).add(i)) = dot_q4_1_row_avx2(bytes, i, k, x);
                            }
                        }
                        return;
                    }
                    for i in start..end {
                        let v = dot_q4_1_row_scalar(bytes, i, k, x);
                        unsafe { *((y_addr as *mut f32).add(i)) = v; }
                    }
                });
            }
            TensorType::Q1_0 => {
                for i in 0..n {
                    y[i] = dot_q1_0_row_scalar(&self.bytes, i, k, x);
                }
            }
            TensorType::Iq1M => {
                for i in 0..n {
                    y[i] = dot_iq1m_row_scalar(&self.bytes, i, k, x);
                }
            }
            TensorType::F32 => {
                // F32 行优先: bytes 当 f32 读
                for i in 0..n {
                    let row_off = i * k * 4;
                    let mut acc = 0.0f32;
                    for j in 0..k {
                        let b = [
                            self.bytes[row_off + j * 4],
                            self.bytes[row_off + j * 4 + 1],
                            self.bytes[row_off + j * 4 + 2],
                            self.bytes[row_off + j * 4 + 3],
                        ];
                        acc += f32::from_le_bytes(b) * x[j];
                    }
                    y[i] = acc;
                }
            }
            _ => panic!("unsupported drafter dtype: {:?}", self.dtype),
        }
    }

    /// matvec 累加: y[i] += dot(W_row_i, x)
    #[allow(unsafe_code)]
    pub fn matvec_add_into_slice(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(y.len(), self.rows);
        let n = self.rows;
        let k = self.cols;

        match self.dtype {
            TensorType::Q4_1 => {
                #[cfg(target_arch = "x86_64")]
                let use_avx2 = avx2_q4_1_available();
                #[cfg(not(target_arch = "x86_64"))]
                let use_avx2 = false;

                if n < 64 || get_thread_pool().is_none() {
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        for i in 0..n {
                            unsafe { y[i] += dot_q4_1_row_avx2(&self.bytes, i, k, x); }
                        }
                        return;
                    }
                    for i in 0..n {
                        y[i] += dot_q4_1_row_scalar(&self.bytes, i, k, x);
                    }
                    return;
                }

                let pool = get_thread_pool().unwrap();
                let n_threads = pool.n_threads();
                let chunk = (n + n_threads - 1) / n_threads;
                let bytes_addr = self.bytes.as_ptr() as usize;
                let bytes_len = self.bytes.len();
                let x_addr = x.as_ptr() as usize;
                let y_addr = y.as_mut_ptr() as usize;
                pool.scatter_wait(n_threads, move |tid| {
                    let start = tid * chunk;
                    let end = (start + chunk).min(n);
                    if start >= end { return; }
                    let bytes = unsafe { std::slice::from_raw_parts(bytes_addr as *const u8, bytes_len) };
                    let x = unsafe { std::slice::from_raw_parts(x_addr as *const f32, k) };
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        for i in start..end {
                            unsafe {
                                let v = dot_q4_1_row_avx2(bytes, i, k, x);
                                *((y_addr as *mut f32).add(i)) += v;
                            }
                        }
                        return;
                    }
                    for i in start..end {
                        let v = dot_q4_1_row_scalar(bytes, i, k, x);
                        unsafe { *((y_addr as *mut f32).add(i)) += v; }
                    }
                });
            }
            TensorType::Q1_0 => {
                for i in 0..n {
                    y[i] += dot_q1_0_row_scalar(&self.bytes, i, k, x);
                }
            }
            TensorType::Iq1M => {
                for i in 0..n {
                    y[i] += dot_iq1m_row_scalar(&self.bytes, i, k, x);
                }
            }
            TensorType::F32 => {
                for i in 0..n {
                    let row_off = i * k * 4;
                    let mut acc = 0.0f32;
                    for j in 0..k {
                        let b = [
                            self.bytes[row_off + j * 4],
                            self.bytes[row_off + j * 4 + 1],
                            self.bytes[row_off + j * 4 + 2],
                            self.bytes[row_off + j * 4 + 3],
                        ];
                        acc += f32::from_le_bytes(b) * x[j];
                    }
                    y[i] += acc;
                }
            }
            _ => panic!("unsupported drafter dtype: {:?}", self.dtype),
        }
    }

    /// Batched matvec: `y[t*rows + i] = dot(W_row_i, x[t*cols..(t+1)*cols])` 对 t ∈ 0..n_batch。
    ///
    /// ★ 同一 W 行被所有 n_batch 个 token 共享 (只 unpack 一次), 节省 (n_batch-1)/n_batch
    /// 的权重读取带宽。Q4_1 走 batched AVX2 kernel (2-token 分块共享 nibble unpack +
    /// d/m broadcast), 大矩阵多线程并行。n_batch=1 退化为普通 matvec。
    ///
    /// 布局: x = [n_batch, cols] row-major, y = [n_batch, rows] row-major。
    #[allow(unsafe_code)]
    pub fn matvec_batch_into_slice(&self, x: &[f32], n_batch: usize, y: &mut [f32]) {
        debug_assert_eq!(x.len(), n_batch * self.cols, "x len mismatch");
        debug_assert_eq!(y.len(), n_batch * self.rows, "y len mismatch");
        if n_batch == 0 { return; }
        if n_batch == 1 {
            self.matvec_into_slice(x, y);
            return;
        }
        let n = self.rows;
        let k = self.cols;

        match self.dtype {
            TensorType::Q4_1 => {
                #[cfg(target_arch = "x86_64")]
                let use_avx2 = avx2_q4_1_available();
                #[cfg(not(target_arch = "x86_64"))]
                let use_avx2 = false;

                // 小矩阵单线程 (避免 thread pool 开销)
                if n < 64 || get_thread_pool().is_none() {
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        for i in 0..n {
                            unsafe {
                                dot_q4_1_row_batch_avx2(
                                    &self.bytes, i, k, x, k, n_batch, y, n,
                                );
                            }
                        }
                        return;
                    }
                    for i in 0..n {
                        for t in 0..n_batch {
                            let xt = &x[t * k..t * k + k];
                            y[t * n + i] = dot_q4_1_row_scalar(&self.bytes, i, k, xt);
                        }
                    }
                    return;
                }

                // 大矩阵多线程并行 (按行分块, 每行计算 n_batch 个输出)
                let pool = get_thread_pool().unwrap();
                let n_threads = pool.n_threads();
                let chunk = (n + n_threads - 1) / n_threads;
                let bytes_addr = self.bytes.as_ptr() as usize;
                let bytes_len = self.bytes.len();
                let x_addr = x.as_ptr() as usize;
                let y_addr = y.as_mut_ptr() as usize;
                pool.scatter_wait(n_threads, move |tid| {
                    let start = tid * chunk;
                    let end = (start + chunk).min(n);
                    if start >= end { return; }
                    let bytes = unsafe { std::slice::from_raw_parts(bytes_addr as *const u8, bytes_len) };
                    let x = unsafe { std::slice::from_raw_parts(x_addr as *const f32, n_batch * k) };
                    let y = unsafe { std::slice::from_raw_parts_mut(y_addr as *mut f32, n_batch * n) };
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        for i in start..end {
                            unsafe {
                                dot_q4_1_row_batch_avx2(
                                    bytes, i, k, x, k, n_batch, y, n,
                                );
                            }
                        }
                        return;
                    }
                    for i in start..end {
                        for t in 0..n_batch {
                            let xt = &x[t * k..t * k + k];
                            y[t * n + i] = dot_q4_1_row_scalar(bytes, i, k, xt);
                        }
                    }
                });
            }
            // 非 Q4_1 dtype: 逐 token 调用 matvec_into_slice (复用现有路径)
            _ => {
                for t in 0..n_batch {
                    let xt = &x[t * k..(t + 1) * k];
                    let yt = &mut y[t * n..(t + 1) * n];
                    self.matvec_into_slice(xt, yt);
                }
            }
        }
    }
}

/// F32 向量 (norms, bias 等)
pub struct F32Vec {
    pub data: Vec<f32>,
}

impl F32Vec {
    pub fn from_gguf(gguf: &GgufFile, name: &str) -> crate::Result<Self> {
        let info = gguf.find_tensor(name)
            .ok_or_else(|| BonsaiError::Model(format!("drafter tensor {name} not found")))?;
        let bytes = gguf.tensor_data(info)?;
        let n = info.n_elements() as usize;
        let expected_bytes = n * 4;
        if bytes.len() < expected_bytes {
            return Err(BonsaiError::Model(format!(
                "drafter F32Vec {name}: n_elements={n} expect {expected_bytes} bytes, got {}",
                bytes.len()
            )));
        }
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * 4;
            data.push(f32::from_le_bytes([
                bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3],
            ]));
        }
        Ok(Self { data })
    }

    pub fn len(&self) -> usize { self.data.len() }
}

/// 单层 drafter transformer 权重
pub struct DrafterBlock {
    pub attn_norm: F32Vec,       // [5120]
    pub attn_q: DrafterMatrix,   // [5120, 5120] Q4_1
    pub attn_k: DrafterMatrix,   // [512, 5120]  Q4_1 (4 kv heads × 128)
    pub attn_v: DrafterMatrix,   // [512, 5120]  Q4_1
    pub attn_output: DrafterMatrix, // [5120, 5120] Q4_1
    pub attn_q_norm: F32Vec,     // [128]
    pub attn_k_norm: F32Vec,     // [128]
    pub ffn_norm: F32Vec,        // [5120]
    pub ffn_gate: DrafterMatrix, // [5120, 5120] Q4_1
    pub ffn_up: DrafterMatrix,   // [5120, 5120] Q4_1
    pub ffn_down: DrafterMatrix, // [5120, 5120] Q4_1
}

/// DSpark drafter 完整权重
pub struct DrafterWeights {
    pub cfg: DrafterConfig,
    /// token embedding [vocab, hidden] Q1_0 (与 target 共享)
    pub token_embd: DrafterMatrix,
    /// 6 层 transformer block
    pub blocks: Vec<DrafterBlock>,
    /// output norm [hidden] F32
    pub output_norm: F32Vec,
    /// LM head [vocab, hidden] Q4_1
    pub output: DrafterMatrix,
    /// target hidden state 投影 [hidden, n_embd_cap=25600] Q4_1
    pub fc: DrafterMatrix,
    /// target hidden state norm [hidden] F32
    pub hidden_norm: F32Vec,
    /// log-SNR FC1 weight [hidden, n_freq=128] Iq1M
    pub log_snr_fc1_w: DrafterMatrix,
    /// log-SNR FC1 bias [hidden] F32
    pub log_snr_fc1_b: F32Vec,
    /// log-SNR FC2 weight [hidden, hidden] Iq1M
    pub log_snr_fc2_w: DrafterMatrix,
    /// log-SNR FC2 bias [hidden] F32
    pub log_snr_fc2_b: F32Vec,
    /// Markov head W1 [vocab, rank=256] Iq1M (prev-token embedding)
    pub markov_w1: DrafterMatrix,
    /// Markov head W2 [vocab, rank=256] Q4_1 (output projection)
    pub markov_w2: DrafterMatrix,
    /// Confidence head weight [1, hidden+markov_rank=5376] Q4_1 (AcceptRatePredictor)
    /// 输入 = [drafter hidden, markov prev_embd] 拼接, 输出 = 单个 logit
    pub confidence_head_w: Option<DrafterMatrix>,
    /// Confidence head bias [1] F32
    pub confidence_head_b: Option<F32Vec>,
}

impl DrafterWeights {
    /// 从 GGUF 文件加载 drafter 权重
    pub fn load(gguf: &GgufFile) -> crate::Result<Self> {
        let cfg = DrafterConfig::from_metadata(&gguf.metadata)?;
        eprintln!("[dspark] Loading drafter: {} blocks, hidden={}, block_size={}",
            cfg.block_count, cfg.embedding_length, cfg.block_size);

        let token_embd = DrafterMatrix::from_gguf(gguf, "token_embd.weight")?;
        let output_norm = F32Vec::from_gguf(gguf, "output_norm.weight")?;
        let output = DrafterMatrix::from_gguf(gguf, "output.weight")?;
        let fc = DrafterMatrix::from_gguf(gguf, "dspark.fc.weight")?;
        let hidden_norm = F32Vec::from_gguf(gguf, "dspark.hidden_norm.weight")?;
        let log_snr_fc1_w = DrafterMatrix::from_gguf(gguf, "dspark.log_snr_fc1.weight")?;
        let log_snr_fc1_b = F32Vec::from_gguf(gguf, "dspark.log_snr_fc1.bias")?;
        let log_snr_fc2_w = DrafterMatrix::from_gguf(gguf, "dspark.log_snr_fc2.weight")?;
        let log_snr_fc2_b = F32Vec::from_gguf(gguf, "dspark.log_snr_fc2.bias")?;
        let markov_w1 = DrafterMatrix::from_gguf(gguf, "dspark.markov_head_a.weight")?;
        let markov_w2 = DrafterMatrix::from_gguf(gguf, "dspark.markov_head_b.weight")?;

        // Confidence head (可选, 由 cfg.confidence_head 控制)
        // weight: [1, hidden+markov_rank=5376] Q4_1, bias: [1] F32
        let (confidence_head_w, confidence_head_b) = if cfg.confidence_head {
            let w = DrafterMatrix::from_gguf(gguf, "dspark.confidence_head.weight")?;
            let b = F32Vec::from_gguf(gguf, "dspark.confidence_head.bias")?;
            // 校验维度
            let expected_in = cfg.confidence_input_dim();
            if w.cols != expected_in || w.rows != 1 {
                return Err(crate::BonsaiError::Model(format!(
                    "confidence_head.weight dims mismatch: got [{}x{}], expected [1x{}]",
                    w.cols, w.rows, expected_in
                )));
            }
            eprintln!("[dspark] Confidence head loaded: input_dim={expected_in}, with_markov={}",
                cfg.confidence_head_with_markov);
            (Some(w), Some(b))
        } else {
            (None, None)
        };

        let mut blocks = Vec::with_capacity(cfg.block_count);
        for i in 0..cfg.block_count {
            let blk = DrafterBlock {
                attn_norm: F32Vec::from_gguf(gguf, &format!("blk.{i}.attn_norm.weight"))?,
                attn_q: DrafterMatrix::from_gguf(gguf, &format!("blk.{i}.attn_q.weight"))?,
                attn_k: DrafterMatrix::from_gguf(gguf, &format!("blk.{i}.attn_k.weight"))?,
                attn_v: DrafterMatrix::from_gguf(gguf, &format!("blk.{i}.attn_v.weight"))?,
                attn_output: DrafterMatrix::from_gguf(gguf, &format!("blk.{i}.attn_output.weight"))?,
                attn_q_norm: F32Vec::from_gguf(gguf, &format!("blk.{i}.attn_q_norm.weight"))?,
                attn_k_norm: F32Vec::from_gguf(gguf, &format!("blk.{i}.attn_k_norm.weight"))?,
                ffn_norm: F32Vec::from_gguf(gguf, &format!("blk.{i}.ffn_norm.weight"))?,
                ffn_gate: DrafterMatrix::from_gguf(gguf, &format!("blk.{i}.ffn_gate.weight"))?,
                ffn_up: DrafterMatrix::from_gguf(gguf, &format!("blk.{i}.ffn_up.weight"))?,
                ffn_down: DrafterMatrix::from_gguf(gguf, &format!("blk.{i}.ffn_down.weight"))?,
            };
            blocks.push(blk);
        }

        eprintln!("[dspark] Drafter loaded: {} blocks, markov_rank={}, target_layers={:?}",
            blocks.len(), cfg.markov_rank, cfg.target_layers);

        Ok(Self {
            cfg,
            token_embd,
            blocks,
            output_norm,
            output,
            fc,
            hidden_norm,
            log_snr_fc1_w,
            log_snr_fc1_b,
            log_snr_fc2_w,
            log_snr_fc2_b,
            markov_w1,
            markov_w2,
            confidence_head_w,
            confidence_head_b,
        })
    }
}

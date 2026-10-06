//! Qwen3-VL 视觉编码器权重加载
//!
//! 从 mmproj GGUF 加载所有张量, **加载时一次性反量化为 F32**:
//! - Q8_0 权重 → F32 (0.4GB → ~1.6GB)
//! - F16 权重 → F32 (ffn_down.weight)
//! - F32 张量直接保留 (norms, biases, patch_embd)
//!
//! 策略: vision encoder 只在每张图调用一次, 不在热路径, 不需要 AVX2 量化 kernel。
//! 用通用 F32 matvec + 线程池即可。F32 内存占用 ~1.6GB 可接受 (主权重 13GB)。

use std::arch::x86_64::*;

use crate::gguf::parser::GgufFile;
use crate::gguf::tensor_info::TensorType;
use crate::tensor::quant::{dequantize_q8_0, f16_to_f32};
use crate::model::workspace::{get_thread_pool, in_parallel_region};
use crate::math::simd_exp::simd_available;
use crate::BonsaiError;

use super::config::VisionConfig;

// ---------------------------------------------------------------------------
// AVX2 8-row dot product kernel (register tiling)
// ---------------------------------------------------------------------------
// 8 output rows × 1 acc per row = 8 YMM acc + 1 x load + 1 row load = 10/16 YMM.
// 8 个独立 acc 填满 FMA pipeline (latency 4c × throughput 0.5c = 8 acc needed).
// 8 行共享 x_v load, load 带宽减少 8× (vs 每 row 独立 load x).
//
// 理论加速: load ops 从 2304 → 1296 (144 x + 1152 row), FMA 1152 不变.
// Load-bound 1152c → 648c = 1.78x.

/// AVX2 水平求和 __m256 → f32
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
#[inline]
unsafe fn hsum_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// AVX2 8-row dot product kernel: y[i] = dot(rows[i*k..(i+1)*k], x[0..k]) for i in 0..8
///
/// # Safety
/// - rows.len() >= 8 * k
/// - x.len() >= k
/// - y.len() >= 8
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
// 8 行展开的 row-0 项刻意保留 `0 *`/`1 *` 前缀, 与 row1..7 保持对称可读
#[allow(clippy::erasing_op, clippy::identity_op)]
#[inline]
unsafe fn dot_8rows_avx2(rows: *const f32, x: *const f32, y: *mut f32, k: usize) {
    let mut acc = [_mm256_setzero_ps(); 8];
    let n8 = (k / 8) * 8;
    let mut j = 0;
    while j < n8 {
        let x_v = _mm256_loadu_ps(x.add(j));
        // 展开 8 个 row 的 FMA (8 个独立 acc, 填满 FMA pipeline)
        acc[0] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(0 * k + j)), x_v, acc[0]);
        acc[1] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(1 * k + j)), x_v, acc[1]);
        acc[2] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(2 * k + j)), x_v, acc[2]);
        acc[3] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(3 * k + j)), x_v, acc[3]);
        acc[4] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(4 * k + j)), x_v, acc[4]);
        acc[5] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(5 * k + j)), x_v, acc[5]);
        acc[6] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(6 * k + j)), x_v, acc[6]);
        acc[7] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(7 * k + j)), x_v, acc[7]);
        j += 8;
    }
    // horizontal sum + tail (k 不是 8 的倍数时)
    for (i, &acc_i) in acc.iter().enumerate() {
        let mut s = hsum_ps(acc_i);
        let mut jj = n8;
        while jj < k {
            s += *rows.add(i * k + jj) * *x.add(jj);
            jj += 1;
        }
        *y.add(i) = s;
    }
}

/// AVX2 8-row dot product kernel (累加版): y[i] += dot(rows[i*k..], x[0..k])
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
#[allow(clippy::erasing_op, clippy::identity_op)] // 8 行展开的 row-0/1 项刻意保留 `0 *`/`1 *` 前缀 (同 dot_8rows_avx2)
#[inline]
unsafe fn dot_8rows_add_avx2(rows: *const f32, x: *const f32, y: *mut f32, k: usize) {
    let mut acc = [_mm256_setzero_ps(); 8];
    let n8 = (k / 8) * 8;
    let mut j = 0;
    while j < n8 {
        let x_v = _mm256_loadu_ps(x.add(j));
        acc[0] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(0 * k + j)), x_v, acc[0]);
        acc[1] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(1 * k + j)), x_v, acc[1]);
        acc[2] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(2 * k + j)), x_v, acc[2]);
        acc[3] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(3 * k + j)), x_v, acc[3]);
        acc[4] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(4 * k + j)), x_v, acc[4]);
        acc[5] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(5 * k + j)), x_v, acc[5]);
        acc[6] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(6 * k + j)), x_v, acc[6]);
        acc[7] = _mm256_fmadd_ps(_mm256_loadu_ps(rows.add(7 * k + j)), x_v, acc[7]);
        j += 8;
    }
    for (i, &acc_i) in acc.iter().enumerate() {
        let mut s = hsum_ps(acc_i);
        let mut jj = n8;
        while jj < k {
            s += *rows.add(i * k + jj) * *x.add(jj);
            jj += 1;
        }
        *y.add(i) += s;
    }
}

/// F32 矩阵 (行优先: rows × cols, 行内连续)
///
/// 加载时一次性反量化, runtime matvec 走 F32 通用路径。
pub struct VisionMatrix {
    pub data: Vec<f32>,  // rows × cols, row-major
    pub rows: usize,
    pub cols: usize,
}

impl VisionMatrix {
    /// 从 GGUF 加载并反量化为 F32 矩阵
    ///
    /// GGUF dims (反向存储): dims[0]=最内层 (快变), dims[N-1]=最外层 (慢变)
    /// 当作 2D 矩阵处理: cols = dims[0..N-1] 之积 (内层), rows = dims[N-1] (外层)
    /// 例如:
    /// - 2D [1152, 3456] → cols=1152, rows=3456
    /// - 4D [16, 16, 3, 1152] → cols=16*16*3=768 (in_ch*kh*kw), rows=1152 (out_ch)
    pub fn from_gguf(gguf: &GgufFile, name: &str) -> crate::Result<Self> {
        let info = gguf.find_tensor(name)
            .ok_or_else(|| BonsaiError::Model(format!("vision tensor {name} not found")))?;
        let raw = gguf.tensor_data(info)?;
        let n_dims = info.dims.len();
        if n_dims == 0 {
            return Err(BonsaiError::Model(format!("{name}: 0-dim tensor")));
        }
        // cols = dims[0..N-1] 之积, rows = dims[N-1]
        let cols: usize = info.dims[..n_dims - 1].iter().product::<u64>() as usize;
        let rows = info.dims[n_dims - 1] as usize;
        let n_elements = rows * cols;

        let data = match info.dtype {
            TensorType::F32 => {
                if raw.len() < n_elements * 4 {
                    return Err(BonsaiError::Tensor(format!(
                        "{name}: F32 data too short: {} bytes for {} elements",
                        raw.len(), n_elements
                    )));
                }
                (0..n_elements)
                    .map(|i| {
                        let b = &raw[i * 4..i * 4 + 4];
                        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                    })
                    .collect()
            }
            TensorType::F16 => {
                if raw.len() < n_elements * 2 {
                    return Err(BonsaiError::Tensor(format!(
                        "{name}: F16 data too short"
                    )));
                }
                (0..n_elements)
                    .map(|i| {
                        let lo = raw[i * 2];
                        let hi = raw[i * 2 + 1];
                        f16_to_f32(u16::from_le_bytes([lo, hi]))
                    })
                    .collect()
            }
            TensorType::Q8_0 => dequantize_q8_0(raw, n_elements),
            other => return Err(BonsaiError::Unsupported(format!(
                "{name}: unsupported dtype {:?} (expected F32/F16/Q8_0)", other
            ))),
        };

        Ok(Self { data, rows, cols })
    }

    /// 批量矩阵乘: Y[p, i] = sum_k W[i, k] * X[p, k]
    ///
    /// - W = self: [n (rows), k (cols)], row-major
    /// - x: [n_batch, k]  (n_batch 个输入向量, 每个维度 k)
    /// - y: [n_batch, n]  (n_batch 个输出向量, 每个维度 n = self.rows)
    ///
    /// ★ 2D tiling: 处理 (I_TILE, P_TILE) 输出 tile, 让 W/X/Y tile 各自驻留 L1/L2/L3:
    ///   - W tile: I_TILE * k * 4 = 8 * 1152 * 4 = 36 KB  → L2
    ///   - X tile: P_TILE * k * 4 = 64 * 1152 * 4 = 288 KB → L2
    ///   - Y tile: P_TILE * I_TILE * 4 = 64 * 8 * 4 = 2 KB → L1 (不污染 L3)
    ///
    /// ★ 关键优化: tiling 让 Y writes (39.7MB) 只经过 L1 (2KB/tile), 不挤出 L3 中的 X.
    ///   无 tiling 时 Y writes 走 L3 write-allocate, 把 X (10.6MB) 从 L3 挤出 → X 从 DRAM 读.
    ///
    /// ★ X reads 减少 I_TILE× : 每 X 元素被读 n/I_TILE 次 (vs 无 tiling 的 n 次)
    ///   I_TILE=8: X reads 45.6GB → 5.7GB, 从 L3 (100GB/s) 读 = 57ms
    #[allow(unsafe_code)]
    pub fn matmat_into_slice(&self, x: &[f32], y: &mut [f32], n_batch: usize) {
        let n = self.rows;
        let k = self.cols;
        debug_assert_eq!(x.len(), n_batch * k);
        debug_assert_eq!(y.len(), n_batch * n);

        // 小矩阵 / 无线程池 / 嵌套并行 → 单线程
        if n < 64 || get_thread_pool().is_none() || in_parallel_region() {
            for i in 0..n {
                let row = &self.data[i * k..(i + 1) * k];
                for p in 0..n_batch {
                    let x_p = &x[p * k..(p + 1) * k];
                    let mut acc = 0.0f32;
                    for j in 0..k {
                        acc += row[j] * x_p[j];
                    }
                    y[p * n + i] = acc;
                }
            }
            return;
        }

        // 2D tiling 参数 (针对 Meteor Lake L1=32KB, L2=1.25MB, L3=18MB)
        const I_TILE: usize = 8;   // 8 output rows per tile (匹配 AVX2 8-row kernel)
        const P_TILE: usize = 64;  // 64 patches per tile

        let n_i_tiles = n.div_ceil(I_TILE);
        let n_p_tiles = n_batch.div_ceil(P_TILE);
        let total_tiles = n_i_tiles * n_p_tiles;

        let pool = get_thread_pool().unwrap();
        let steal_chunk = 4; // 4 tiles per steal chunk
        let data_addr = self.data.as_ptr() as usize;
        let x_addr = x.as_ptr() as usize;
        let y_addr = y.as_mut_ptr() as usize;
        let use_avx2 = simd_available();
        pool.scatter_wait_stealing(total_tiles, steal_chunk, move |start, end| {
            let data = unsafe { std::slice::from_raw_parts(data_addr as *const f32, n * k) };
            let x = unsafe { std::slice::from_raw_parts(x_addr as *const f32, n_batch * k) };
            for tile_idx in start..end {
                let i_tile = tile_idx / n_p_tiles;
                let p_tile = tile_idx % n_p_tiles;
                let i_start = i_tile * I_TILE;
                let i_end = (i_start + I_TILE).min(n);
                let p_start = p_tile * P_TILE;
                let p_end = (p_start + P_TILE).min(n_batch);

                // ★ AVX2 8-row kernel: 8 行共享 x load, 8 个独立 acc 填满 FMA pipeline
                if use_avx2 && i_end - i_start == 8 {
                    let rows_ptr = data.as_ptr().wrapping_add(i_start * k);
                    for p in p_start..p_end {
                        let x_p = &x[p * k..(p + 1) * k];
                        unsafe {
                            dot_8rows_avx2(
                                rows_ptr,
                                x_p.as_ptr(),
                                (y_addr as *mut f32).add(p * n + i_start),
                                k,
                            );
                        }
                    }
                } else {
                    // scalar fallback (n 不是 8 的倍数 或 AVX2 不可用)
                    for i in i_start..i_end {
                        let row = &data[i * k..(i + 1) * k];
                        for p in p_start..p_end {
                            let x_p = &x[p * k..(p + 1) * k];
                            let mut acc = 0.0f32;
                            for j in 0..k {
                                acc += row[j] * x_p[j];
                            }
                            unsafe { *((y_addr as *mut f32).add(p * n + i)) = acc; }
                        }
                    }
                }
            }
        });
    }

    /// 批量矩阵乘累加: Y[p, i] += sum_k W[i, k] * X[p, k]
    #[allow(unsafe_code)]
    pub fn matmat_add_into_slice(&self, x: &[f32], y: &mut [f32], n_batch: usize) {
        let n = self.rows;
        let k = self.cols;
        debug_assert_eq!(x.len(), n_batch * k);
        debug_assert_eq!(y.len(), n_batch * n);

        if n < 64 || get_thread_pool().is_none() || in_parallel_region() {
            for i in 0..n {
                let row = &self.data[i * k..(i + 1) * k];
                for p in 0..n_batch {
                    let x_p = &x[p * k..(p + 1) * k];
                    let mut acc = 0.0f32;
                    for j in 0..k {
                        acc += row[j] * x_p[j];
                    }
                    y[p * n + i] += acc;
                }
            }
            return;
        }

        const I_TILE: usize = 8;
        const P_TILE: usize = 64;

        let n_i_tiles = n.div_ceil(I_TILE);
        let n_p_tiles = n_batch.div_ceil(P_TILE);
        let total_tiles = n_i_tiles * n_p_tiles;

        let pool = get_thread_pool().unwrap();
        let steal_chunk = 4;
        let data_addr = self.data.as_ptr() as usize;
        let x_addr = x.as_ptr() as usize;
        let y_addr = y.as_mut_ptr() as usize;
        let use_avx2 = simd_available();
        pool.scatter_wait_stealing(total_tiles, steal_chunk, move |start, end| {
            let data = unsafe { std::slice::from_raw_parts(data_addr as *const f32, n * k) };
            let x = unsafe { std::slice::from_raw_parts(x_addr as *const f32, n_batch * k) };
            for tile_idx in start..end {
                let i_tile = tile_idx / n_p_tiles;
                let p_tile = tile_idx % n_p_tiles;
                let i_start = i_tile * I_TILE;
                let i_end = (i_start + I_TILE).min(n);
                let p_start = p_tile * P_TILE;
                let p_end = (p_start + P_TILE).min(n_batch);

                if use_avx2 && i_end - i_start == 8 {
                    let rows_ptr = data.as_ptr().wrapping_add(i_start * k);
                    for p in p_start..p_end {
                        let x_p = &x[p * k..(p + 1) * k];
                        unsafe {
                            dot_8rows_add_avx2(
                                rows_ptr,
                                x_p.as_ptr(),
                                (y_addr as *mut f32).add(p * n + i_start),
                                k,
                            );
                        }
                    }
                } else {
                    for i in i_start..i_end {
                        let row = &data[i * k..(i + 1) * k];
                        for p in p_start..p_end {
                            let x_p = &x[p * k..(p + 1) * k];
                            let mut acc = 0.0f32;
                            for j in 0..k {
                                acc += row[j] * x_p[j];
                            }
                            unsafe { *((y_addr as *mut f32).add(p * n + i)) += acc; }
                        }
                    }
                }
            }
        });
    }

}

/// F32 向量 (1D 张量, bias / norm weight 等)
pub struct VisionTensor {
    pub data: Vec<f32>,
    pub dims: Vec<usize>,
}

impl VisionTensor {
    /// 从 GGUF 加载 1D/2D 张量为 F32 (Q8_0 / F16 / F32 / F32 反量化)
    pub fn from_gguf(gguf: &GgufFile, name: &str) -> crate::Result<Self> {
        let info = gguf.find_tensor(name)
            .ok_or_else(|| BonsaiError::Model(format!("vision tensor {name} not found")))?;
        let raw = gguf.tensor_data(info)?;
        let n_elements = info.n_elements() as usize;
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();

        let data = match info.dtype {
            TensorType::F32 => {
                (0..n_elements)
                    .map(|i| {
                        let b = &raw[i * 4..i * 4 + 4];
                        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                    })
                    .collect()
            }
            TensorType::F16 => {
                (0..n_elements)
                    .map(|i| {
                        let lo = raw[i * 2];
                        let hi = raw[i * 2 + 1];
                        f16_to_f32(u16::from_le_bytes([lo, hi]))
                    })
                    .collect()
            }
            TensorType::Q8_0 => dequantize_q8_0(raw, n_elements),
            other => return Err(BonsaiError::Unsupported(format!(
                "{name}: unsupported dtype {:?} (expected F32/F16/Q8_0)", other
            ))),
        };

        Ok(Self { data, dims })
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// 单个 ViT block 的权重
pub struct ViTBlockWeights {
    pub ln1_w: VisionTensor,    // [1152]
    pub ln1_b: VisionTensor,    // [1152]
    pub attn_qkv_w: VisionMatrix,  // [1152, 3456] (3*1152 fused QKV)
    pub attn_qkv_b: VisionTensor,  // [3456]
    pub attn_out_w: VisionMatrix,  // [1152, 1152]
    pub attn_out_b: VisionTensor,  // [1152]
    pub ln2_w: VisionTensor,    // [1152]
    pub ln2_b: VisionTensor,    // [1152]
    pub ffn_up_w: VisionMatrix,   // [1152, 4304]
    pub ffn_up_b: VisionTensor,   // [4304]
    pub ffn_down_w: VisionMatrix, // [4304, 1152] (F16 in GGUF, 反量化为 F32)
    pub ffn_down_b: VisionTensor, // [1152]
}

/// 完整 mmproj 权重容器
pub struct VisionWeights {
    pub cfg: VisionConfig,

    // Patch embedding (gated Conv2D)
    pub patch_embd_w: VisionMatrix,    // [16, 16, 3, 1152] reshape to [16*16*3, 1152]
    pub patch_embd_w1: VisionMatrix,   // [16, 16, 3, 1152] (gating weight, 相同形状)
    pub patch_embd_b: VisionTensor,    // [1152]
    pub position_embd: VisionTensor,   // [1152, 2304] (learned absolute position embd)
    pub post_ln_w: VisionTensor,       // [1152]
    pub post_ln_b: VisionTensor,       // [1152]

    // 27 个 ViT blocks
    pub blocks: Vec<ViTBlockWeights>,

    // Projector (qwen3vl_merger)
    // mm.0: [merged_hidden=4608, merged_hidden=4608] (linear_fc1)
    // mm.2: [merged_hidden=4608, projection_dim=5120] (linear_fc2)
    pub mm_0_w: VisionMatrix,  // [4608, 4608]
    pub mm_0_b: VisionTensor,  // [4608]
    pub mm_2_w: VisionMatrix,  // [4608, 5120]
    pub mm_2_b: VisionTensor,  // [5120]
}

impl VisionWeights {
    /// 从 mmproj GGUF 加载所有权重
    ///
    /// 总加载时间 ~5-10s (主要是 Q8_0 反量化, 0.4GB → 1.6GB F32)
    pub fn load(gguf: &GgufFile, cfg: &VisionConfig) -> crate::Result<Self> {
        eprintln!("[vision] Loading mmproj weights (this may take a few seconds for Q8_0 dequant)...");

        // Patch embedding: GGUF dims=[16, 16, 3, 1152], 总元素 884736
        // 直接作为 4D Conv2D 权重访问, 用 VisionMatrix 存储 [in_ch*kh*kw=768, out_ch=1152]
        let patch_embd_w = VisionMatrix::from_gguf(gguf, "v.patch_embd.weight")?;
        let patch_embd_w1 = VisionMatrix::from_gguf(gguf, "v.patch_embd.weight.1")?;
        let patch_embd_b = VisionTensor::from_gguf(gguf, "v.patch_embd.bias")?;
        let position_embd = VisionTensor::from_gguf(gguf, "v.position_embd.weight")?;
        let post_ln_w = VisionTensor::from_gguf(gguf, "v.post_ln.weight")?;
        let post_ln_b = VisionTensor::from_gguf(gguf, "v.post_ln.bias")?;

        // 验证 patch_embd 形状
        // GGUF 存储为 [16, 16, 3, 1152], 我们的 VisionMatrix 视为 [cols=16*16*3=768, rows=1152]
        // 即 rows=out_channels=1152, cols=in_channels*kh*kw=3*16*16=768
        let expected_patch_cols = 3 * cfg.patch_size * cfg.patch_size;
        let expected_patch_rows = cfg.embedding_length;
        if patch_embd_w.cols != expected_patch_cols || patch_embd_w.rows != expected_patch_rows {
            return Err(BonsaiError::Model(format!(
                "patch_embd.weight shape mismatch: got [{}x{}], expected [{}x{}] (cols x rows)",
                patch_embd_w.cols, patch_embd_w.rows,
                expected_patch_cols, expected_patch_rows
            )));
        }

        // ViT blocks
        let mut blocks = Vec::with_capacity(cfg.block_count);
        for blk_idx in 0..cfg.block_count {
            let prefix = format!("v.blk.{blk_idx}");
            let w = ViTBlockWeights {
                ln1_w: VisionTensor::from_gguf(gguf, &format!("{prefix}.ln1.weight"))?,
                ln1_b: VisionTensor::from_gguf(gguf, &format!("{prefix}.ln1.bias"))?,
                attn_qkv_w: VisionMatrix::from_gguf(gguf, &format!("{prefix}.attn_qkv.weight"))?,
                attn_qkv_b: VisionTensor::from_gguf(gguf, &format!("{prefix}.attn_qkv.bias"))?,
                attn_out_w: VisionMatrix::from_gguf(gguf, &format!("{prefix}.attn_out.weight"))?,
                attn_out_b: VisionTensor::from_gguf(gguf, &format!("{prefix}.attn_out.bias"))?,
                ln2_w: VisionTensor::from_gguf(gguf, &format!("{prefix}.ln2.weight"))?,
                ln2_b: VisionTensor::from_gguf(gguf, &format!("{prefix}.ln2.bias"))?,
                ffn_up_w: VisionMatrix::from_gguf(gguf, &format!("{prefix}.ffn_up.weight"))?,
                ffn_up_b: VisionTensor::from_gguf(gguf, &format!("{prefix}.ffn_up.bias"))?,
                ffn_down_w: VisionMatrix::from_gguf(gguf, &format!("{prefix}.ffn_down.weight"))?,
                ffn_down_b: VisionTensor::from_gguf(gguf, &format!("{prefix}.ffn_down.bias"))?,
            };
            blocks.push(w);
        }

        // Projector
        let mm_0_w = VisionMatrix::from_gguf(gguf, "mm.0.weight")?;
        let mm_0_b = VisionTensor::from_gguf(gguf, "mm.0.bias")?;
        let mm_2_w = VisionMatrix::from_gguf(gguf, "mm.2.weight")?;
        let mm_2_b = VisionTensor::from_gguf(gguf, "mm.2.bias")?;

        eprintln!("[vision] mmproj weights loaded: {} blocks, patch=[{}x{}], mm_0=[{}x{}], mm_2=[{}x{}]",
            blocks.len(),
            patch_embd_w.cols, patch_embd_w.rows,
            mm_0_w.cols, mm_0_w.rows,
            mm_2_w.cols, mm_2_w.rows);

        Ok(Self {
            cfg: cfg.clone(),
            patch_embd_w,
            patch_embd_w1,
            patch_embd_b,
            position_embd,
            post_ln_w,
            post_ln_b,
            blocks,
            mm_0_w,
            mm_0_b,
            mm_2_w,
            mm_2_b,
        })
    }
}

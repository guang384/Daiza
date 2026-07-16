//! 张量抽象(学习项目版)
//!
//! 一个 `Tensor` 持有自己的 F32 数据 + 形状信息。
//! 量化权重(Q1_0 等)在加载时即反量化为 F32 —— **学习项目的简化路径**:
//!
//! > 真正生产引擎(如 llama.cpp fork)直接把 Q1_0 数据喂给融合 GEMM 内核,
//! > 永不展开成 F32。本引擎 v0 先做正确性,优化留作后续练习
//! > (见 `math::gemm` 中的位运算加速思路)。
//!
//! `TensorView` 是对 borrowed 切片的只读视图,用于零拷贝传递。

use crate::tensor::TensorType;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub dims: Vec<usize>,
}

impl Tensor {
    pub fn new(dims: Vec<usize>) -> Self {
        let n: usize = dims.iter().product();
        Self {
            data: vec![0.0; n],
            dims,
        }
    }

    pub fn from_data(dims: Vec<usize>, data: Vec<f32>) -> Self {
        Self { data, dims }
    }

    pub fn n_elements(&self) -> usize {
        self.data.len()
    }

    /// 行数(输出维度,外层维度) = dims[1] 或 dims[0](1D 时)
    pub fn rows(&self) -> usize {
        self.dims.get(1).copied().unwrap_or_else(|| self.dims.first().copied().unwrap_or(1))
    }

    /// 列数(输入维度,行内元素数,内层维度) = dims[0]
    pub fn cols(&self) -> usize {
        self.dims.first().copied().unwrap_or(1)
    }

    pub fn as_view(&self) -> TensorView<'_> {
        TensorView {
            data: &self.data,
            dims: &self.dims,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub data: &'a [f32],
    pub dims: &'a [usize],
}

impl<'a> TensorView<'a> {
    pub fn rows(&self) -> usize {
        self.dims.get(1).copied().unwrap_or_else(|| self.dims.first().copied().unwrap_or(1))
    }
    pub fn cols(&self) -> usize {
        self.dims.first().copied().unwrap_or(1)
    }
}

/// 从 GGUF 张量数据反量化为 F32 Tensor
///
/// dtype 决定走哪条反量化路径:
/// - F32:直接重新解释
/// - F16:逐元素转换
/// - BF16:逐元素转换
/// - Q1_0:按 128 权重一组反量化
pub fn load_as_f32(
    data: &[u8],
    dims: &[u64],
    dtype: TensorType,
) -> crate::Result<Tensor> {
    let n_elements: usize = dims.iter().product::<u64>() as usize;
    let dims_usize: Vec<usize> = dims.iter().map(|&d| d as usize).collect();

    let f32_data = match dtype {
        TensorType::F32 => {
            if data.len() < n_elements * 4 {
                return Err(crate::BonsaiError::Tensor(format!(
                    "F32 data too short: {} bytes for {} elements",
                    data.len(),
                    n_elements
                )));
            }
            // 把 [u8] 重新解释为 [f32](借用 → owned)
            (0..n_elements)
                .map(|i| {
                    let b = &data[i * 4..i * 4 + 4];
                    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                })
                .collect()
        }
        TensorType::F16 => {
            if data.len() < n_elements * 2 {
                return Err(crate::BonsaiError::Tensor(format!(
                    "F16 data too short: {} bytes for {} elements",
                    data.len(),
                    n_elements
                )));
            }
            (0..n_elements)
                .map(|i| {
                    let lo = data[i * 2];
                    let hi = data[i * 2 + 1];
                    crate::tensor::quant::f16_to_f32(u16::from_le_bytes([lo, hi]))
                })
                .collect()
        }
        TensorType::Bf16 => {
            if data.len() < n_elements * 2 {
                return Err(crate::BonsaiError::Tensor(format!(
                    "BF16 data too short: {} bytes for {} elements",
                    data.len(),
                    n_elements
                )));
            }
            (0..n_elements)
                .map(|i| {
                    let lo = data[i * 2];
                    let hi = data[i * 2 + 1];
                    crate::tensor::quant::bf16_to_f32(u16::from_le_bytes([lo, hi]))
                })
                .collect()
        }
        TensorType::Q1_0 => crate::tensor::quant::dequantize_q1_0(data, n_elements),
        other => {
            return Err(crate::BonsaiError::Unsupported(format!(
                "dtype {} not supported by load_as_f32",
                other.name()
            )));
        }
    };

    Ok(Tensor::from_data(dims_usize, f32_data))
}

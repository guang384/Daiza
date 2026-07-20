//! 张量元信息 + ggml 张量类型枚举
//!
//! 张量信息在 GGUF 头部后存储,记录:名字、形状、dtype、数据偏移。
//! 引擎实际加载权重时按 dtype 选择反量化路径,按 offset 索引数据。

use crate::gguf::err;
use crate::gguf::reader::ByteReader;
use crate::Result;

/// GGUF magic:"GGUF" = `[0x47, 0x47, 0x55, 0x46]`
pub const GGUF_MAGIC: u32 = 0x46554747; // 注意 from_le_bytes 后的小端值

/// GGUF 版本(实测 Bonsai 文件均为 v3)
pub const GGUF_VERSION: u32 = 3;

/// ggml 张量数据类型枚举
///
/// 上游 llama.cpp 的 `ggml_type` 枚举最高到 38(IQ4_XS 等)。
/// Bonsai 使用自定义类型 **Q1_0 = 41**(PrismML fork 新增)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32,    // 0  - 标准 32 位浮点
    F16,    // 1  - 半精度
    Q4_0,   // 2  - 4-bit 量化
    Q4_1,   // 3  - 4-bit 量化(带 min/max)
    Q5_0,   // 6
    Q5_1,   // 7
    Q8_0,   // 8  - 8-bit 量化
    Q2K,    // 10
    Q3K,    // 11
    Q4K,    // 12
    Q5K,    // 13
    Q6K,    // 14
    Q8K,    // 15
    Iq2Xxs, // 24
    Iq1S,   // 28
    Iq1M,   // 29
    Bf16,   // 30  - bfloat16(标准 ggml: GGML_TYPE_BF16=30)
    /// Bonsai 自定义二值格式:128 权重 = 2 字节 FP16 scale + 16 字节符号位
    /// = 18 字节 / 128 权重 = 1.125 bits/weight
    Q1_0,   // 41
    /// 未识别的类型(对学习项目足够:在解析阶段即报错而非静默误用)
    Unknown(u32),
}

impl TensorType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            24 => Self::Iq2Xxs,
            28 => Self::Iq1S,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            41 => Self::Q1_0,
            _ => Self::Unknown(v),
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::Iq2Xxs => 24,
            Self::Iq1S => 28,
            Self::Iq1M => 29,
            Self::Bf16 => 30,
            Self::Q1_0 => 41,
            Self::Unknown(v) => v,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq1S => "IQ1_S",
            Self::Iq1M => "IQ1_M",
            Self::Q1_0 => "Q1_0",
            Self::Unknown(_) => "unknown",
        }
    }

    /// 是否为 Bonsai 的二值格式
    pub fn is_q1_0(self) -> bool {
        matches!(self, Self::Q1_0)
    }
}

/// 单个张量的元信息(头部)
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// 维度,GGUF 按反向存储(与 PyTorch 相反):
    /// - `dims[0]` 是**列数**(输入维度,行内元素数,内层维度)
    /// - `dims[1]` 是**行数**(输出维度,行数,外层维度)
    pub dims: Vec<u64>,
    pub dtype: TensorType,
    /// 相对张量数据段起点的偏移(字节)
    pub offset: u64,
}

impl TensorInfo {
    /// 解析 `tensor_count` 个张量信息
    pub fn parse_all(reader: &mut ByteReader<'_>, tensor_count: u64) -> Result<Vec<Self>> {
        let mut out = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            let n_dims = reader.read_u32()? as usize;
            if n_dims > 4 {
                return Err(err(format!(
                    "tensor '{name}' has {n_dims} dims, only <=4 supported"
                )));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(reader.read_u64()?);
            }
            let dtype_tag = reader.read_u32()?;
            let dtype = TensorType::from_u32(dtype_tag);
            if matches!(dtype, TensorType::Unknown(_)) {
                // 学习项目允许枚举外的值先通过,实际加载时再拒绝
            }
            let offset = reader.read_u64()?;
            out.push(Self {
                name,
                dims,
                dtype,
                offset,
            });
        }
        Ok(out)
    }

    /// 元素总数 = dims 之积
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// 行数(输出维度,外层维度) = dims[1] 或 dims[0](1D 时)
    pub fn rows(&self) -> u64 {
        self.dims.get(1).copied().unwrap_or_else(|| self.dims.first().copied().unwrap_or(1))
    }

    /// 列数(输入维度,行内元素数,内层维度) = dims[0]
    pub fn cols(&self) -> u64 {
        self.dims.first().copied().unwrap_or(1)
    }

    /// 简单显示:`name [rows x cols] dtype`
    pub fn display(&self) -> String {
        let shape = match self.dims.len() {
            0 => "[]".to_string(),
            1 => format!("[{}]", self.dims[0]),
            2 => format!("[{}x{}]", self.dims[0], self.dims[1]),
            _ => format!(
                "[{}]",
                self.dims
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x")
            ),
        };
        format!("{:<40} {:<20} {}", self.name, shape, self.dtype.name())
    }
}

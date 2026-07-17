//! 量化类型与字节大小计算
//!
//! 反量化、GEMM 等内核需要知道每种 dtype 的实际字节占用。

use crate::gguf::tensor_info::TensorType as GgufTensorType;

// 重新导出,简化模块路径
pub use crate::gguf::tensor_info::TensorType;

/// Q1_0 一组的元素数:128 权重共享一个 FP16 scale
pub const Q1_0_GROUP_SIZE: usize = 128;

/// Q1_0 一组的字节数:2 字节 scale + 16 字节符号位(128 bit / 8)
pub const Q1_0_BLOCK_BYTES: usize = 18;

/// 给定 dtype 和元素总数,返回该张量在文件中的字节数
pub fn byte_size(dtype: TensorType, n_elements: u64) -> usize {
    match dtype {
        TensorType::F32 => (n_elements * 4) as usize,
        TensorType::F16 | TensorType::Bf16 => (n_elements * 2) as usize,
        TensorType::Q1_0 => {
            let groups = n_elements.div_ceil(Q1_0_GROUP_SIZE as u64);
            (groups as usize) * Q1_0_BLOCK_BYTES
        }
        TensorType::Q8_0 => {
            // 32 weights/block: 2-byte scale + 32 int8 = 34 bytes
            let groups = n_elements.div_ceil(32);
            (groups as usize) * 34
        }
        TensorType::Q4_0 | TensorType::Q4_1 => {
            // 32 weights/block: 2-byte scale (+ 2-byte min for Q4_1) + 16 bytes packed
            let groups = n_elements.div_ceil(32);
            let per_block = match dtype {
                TensorType::Q4_0 => 18,
                TensorType::Q4_1 => 20,
                _ => unreachable!(),
            };
            (groups as usize) * per_block
        }
        _ => {
            // 学习项目:暂只支持上面几种,其他在加载时报错
            // (避免静默返回错误字节数)
            0
        }
    }
}

/// 将 GGUF 的 TensorType 转为本模块的(同名)TensorType —— 两者目前是同一个枚举
pub fn from_gguf(t: GgufTensorType) -> TensorType {
    t
}

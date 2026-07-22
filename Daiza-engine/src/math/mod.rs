//! 纯 CPU 数学内核集合
//!
//! 每个子模块对应推理流程中的一个算子。所有实现都是参考版本(无 SIMD / 无多线程),
//! 优先正确性 —— 优化路径在每处注释中标注为 TODO。
//!
//! ## 性能瓶颈(来自白皮书 §2.2 与 §5)
//!
//! Token 生成阶段是 **内存带宽受限**(memory-bandwidth bound),不是算力受限。
//! 这正是 Q1_0 的价值:把 3.9 GB 的权重流量降到 1/14.2。
//! 因此 math 模块的优化方向是:
//!
//! 1. **GEMM**:让 Q1_0 权重**不展开**直接参与运算(位运算加速思路)
//! 2. **算子融合**:RMSNorm + GEMM、RoPE + QK 等可融合
//! 3. **多线程**:用 `std::thread`(零依赖)按行切分 GEMM
//!
//! v0 先做正确,后续按上述方向逐步优化。

// v0 占位模块已删除(gemm/conv1d/ssm_scan),实际实现见:
//   - GEMM: weights.rs::Q1_0Matrix::matvec_into_slice + tensor/quant.rs::dot_q1_0_row_avx2
//   - Conv1d: model/ssm.rs 内联实现(已做循环顺序交换优化)
//   - SSM scan: model/ssm.rs::ssm_scan_vhead(Gated Delta Rule, 2-pass 融合)
pub mod rmsnorm;
pub mod rope;
pub mod softmax;
pub mod activation;
pub mod sampling;
pub mod simd_exp;
pub mod layernorm;
pub mod gelu;

pub use rmsnorm::{rmsnorm_inplace, rmsnorm_into};
pub use rope::{apply_rope_partial, rope_cos_sin_mrope_text, rope_cos_sin_mrope_text_into, rope_freqs};
pub use softmax::softmax_inplace;
pub use activation::swiglu_inplace;
pub use sampling::{sample_top_k_top_p_into, LcgRng, SamplingBuffers, SamplingParams};
pub use simd_exp::{
    exp_inplace_simd, mul_inplace_simd, sigmoid_fast, sigmoid_inplace_simd,
    silu_inplace_simd, swiglu_inplace_simd,
};
pub use layernorm::layernorm_into;
pub use gelu::gelu_into;

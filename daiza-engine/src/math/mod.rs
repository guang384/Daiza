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

pub mod gemm;
pub mod rmsnorm;
pub mod rope;
pub mod softmax;
pub mod activation;
pub mod sampling;
pub mod conv1d;
pub mod ssm_scan;

pub use gemm::{matmul, matmul_add_into, matvec, matvec_with_bias};
pub use rmsnorm::rmsnorm_inplace;
pub use rope::{apply_rope_partial, rope_cos_sin_mrope_text, rope_freqs};
pub use softmax::{softmax_inplace, softmax_masked_inplace};
pub use activation::{silu, silu_inplace, swiglu_inplace};
pub use sampling::{sample_top_k_top_p, SamplingParams};

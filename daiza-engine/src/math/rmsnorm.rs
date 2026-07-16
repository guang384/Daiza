//! RMSNorm(Qwen3 / LLaMA 风格的均方根归一化)
//!
//! 公式:`y_i = x_i * w_i / sqrt(mean(x^2) + eps)`
//!
//! - `eps = 1e-6`(从 `qwen35.attention.layer_norm_rms_epsilon` 读取)
//! - `w` 是可学习的 scale 向量,在 GGUF 中以 F32 存储

/// 原地 RMSNorm:x[i] 被替换为归一化后的值
///
/// - `x`: 长度 n 的输入向量(被原地修改)
/// - `w`: 长度 n 的权重
/// - `eps`: 小常数,默认 1e-6
pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), w.len());
    let n = x.len() as f32;

    // 计算平方均值
    let mut ss = 0.0f32;
    for &xi in x.iter() {
        ss += xi * xi;
    }
    let mean = ss / n;
    let inv_rms = 1.0 / (mean + eps).sqrt();

    // 应用权重
    for (xi, &wi) in x.iter_mut().zip(w.iter()) {
        *xi = *xi * inv_rms * wi;
    }
}

/// 非原地版本:返回归一化后的向量
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    rmsnorm_inplace(&mut out, w, eps);
    out
}

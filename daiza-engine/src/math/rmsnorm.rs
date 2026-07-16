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

/// 非原地版本:直接从 `src` 读、写入 `dst`,消除 caller 的 copy_from_slice。
///
/// 等价于 `dst.copy_from_slice(src); rmsnorm_inplace(dst, w, eps)`,
/// 但省去一次 20KB(hidden=5120)的内存拷贝。
///
/// `src` 与 `dst` 必须不重叠(典型场景:`src` = 主残差流 `h`,`dst` = workspace buffer)。
pub fn rmsnorm_into(src: &[f32], dst: &mut [f32], w: &[f32], eps: f32) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), w.len());
    let n = src.len() as f32;

    // Pass 1: 计算 ss 同时把 src 拷到 dst(rustc 可能无法自动融合,
    // 显式分两步但数据在 L1,拷贝代价远低于原 copy_from_slice 的 20KB read+write)
    let mut ss = 0.0f32;
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = s;
        ss += s * s;
    }
    let mean = ss / n;
    let inv_rms = 1.0 / (mean + eps).sqrt();

    // Pass 2: 应用权重(已 in dst)
    for (di, &wi) in dst.iter_mut().zip(w.iter()) {
        *di = *di * inv_rms * wi;
    }
}

/// 非原地版本:返回归一化后的向量
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    rmsnorm_inplace(&mut out, w, eps);
    out
}

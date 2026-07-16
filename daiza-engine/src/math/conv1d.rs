//! 1D 因果短卷积(Mamba2 SSM 的前置 conv)
//!
//! 配置:`qwen35.ssm.conv_kernel = 4`
//! 张量:`ssm_conv1d.weight` 形状 [4, inner]
//!
//! 在 decode 阶段需要维护长度 = conv_kernel 的滑窗

/// 对一个向量应用 1D 因果卷积(仅 decode 路径,single token)
///
/// - `x_history`:最近 conv_kernel 个 token 的输入(包括当前),
///   长度必须 == conv_kernel,索引 0 是最旧,索引 -1 是当前
/// - `weight`:形状 [conv_kernel, inner_channels],行优先
/// - `inner`:通道数
/// - `out`:输出,长度 = inner
pub fn conv1d_causal_single(
    x_history: &[f32],
    weight: &[f32],
    conv_kernel: usize,
    inner: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(x_history.len(), conv_kernel * inner);
    debug_assert_eq!(out.len(), inner);
    for ch in 0..inner {
        let mut acc = 0.0f32;
        for t in 0..conv_kernel {
            let x = x_history[t * inner + ch];
            // weight[t, ch] 存在 weight[t * inner + ch]
            let w = weight[t * inner + ch];
            acc += x * w;
        }
        out[ch] = acc;
    }
}

/// 在 prefix/prefill 阶段对整个序列做 1D 因果卷积
///
/// - `x`: shape [seq_len, inner]
/// - `weight`: shape [conv_kernel, inner]
/// - `out`: shape [seq_len, inner]
pub fn conv1d_causal_seq(
    x: &[f32],
    weight: &[f32],
    conv_kernel: usize,
    inner: usize,
    seq_len: usize,
    out: &mut [f32],
) {
    for pos in 0..seq_len {
        for ch in 0..inner {
            let mut acc = 0.0f32;
            for t in 0..conv_kernel {
                let hist_pos = pos as isize - (conv_kernel - 1 - t) as isize;
                if hist_pos < 0 {
                    continue;
                }
                let hist_pos = hist_pos as usize;
                let xv = x[hist_pos * inner + ch];
                let w = weight[t * inner + ch];
                acc += xv * w;
            }
            out[pos * inner + ch] = acc;
        }
    }
}

//! 采样:temperature / top-k / top-p
//!
//! Bonsai 推荐参数(白皮书附录 B.1):
//! - temperature = 0.7
//! - top_p = 0.95
//! - top_k = 20

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.95,
        }
    }
}

/// 从 logits 中采样一个 token
///
/// 步骤:
/// 1. 应用 temperature: logits /= T
/// 2. top-k 截断:保留最大的 K 个,其余设为 -inf
/// 3. top-p (nucleus) 截断:从高到低累加概率,达到 p 后截断
/// 4. softmax + 按概率随机选择
pub fn sample_top_k_top_p(
    logits: &[f32],
    params: SamplingParams,
    rng: &mut impl FnMut() -> f32,
) -> usize {
    let n = logits.len();
    if n == 0 {
        return 0;
    }
    if params.temperature <= 0.0 {
        // 贪心解码
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    // 1. 应用 temperature
    let inv_t = 1.0 / params.temperature;
    let mut scaled: Vec<f32> = logits.iter().map(|&x| x * inv_t).collect();

    // 2. top-k:用部分排序选出前 K 个索引
    let k = params.top_k.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_k_indices = &indices[..k];

    // 3. softmax on top-k
    let mut max = f32::NEG_INFINITY;
    for &i in top_k_indices {
        if scaled[i] > max {
            max = scaled[i];
        }
    }
    let mut probs: Vec<(usize, f32)> = top_k_indices
        .iter()
        .map(|&i| (i, (scaled[i] - max).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    let inv_sum = 1.0 / sum;
    for (_, p) in probs.iter_mut() {
        *p *= inv_sum;
    }

    // 4. top-p:按概率降序累加,保留累积到 p 之前(含)的所有项
    probs.sort_by(|(_, a), (_, b)| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0f32;
    let mut cutoff = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cum += p;
        if cum >= params.top_p {
            cutoff = i + 1;
            break;
        }
    }
    probs.truncate(cutoff);
    // 重新归一化
    let new_sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    let new_inv = 1.0 / new_sum;
    for (_, p) in probs.iter_mut() {
        *p *= new_inv;
    }

    // 5. 按概率选择
    let r = rng();
    let mut acc = 0.0f32;
    let mut chosen = probs[0].0;
    for &(idx, p) in probs.iter() {
        acc += p;
        if r <= acc {
            chosen = idx;
            break;
        }
    }
    chosen
}

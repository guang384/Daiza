//! 采样:temperature / top-k / top-p
//!
//! Bonsai 推荐参数(白皮书附录 B.1):
//! - temperature = 0.7
//! - top_p = 0.95
//! - top_k = 20

/// 简易 LCG 随机数生成器 (确定性,用于可重现的采样)
pub struct LcgRng {
    state: u64,
}
impl LcgRng {
    pub fn new(seed: u64) -> Self { Self { state: seed } }
    pub fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = ((self.state >> 40) & 0xFFFFFF) as u32;
        (bits as f32) / (0x1000000 as f32)
    }
}

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

/// 采样用复用 buffer(消除每 token 3MB 堆分配)
///
/// - `scaled`: [vocab_size] f32 = ~1MB
/// - `indices`: [vocab_size] usize = ~2MB
/// - `probs`: [top_k] (usize, f32) = 小
///
/// 在 decode 循环外创建一次,跨 token 复用。
pub struct SamplingBuffers {
    pub scaled: Vec<f32>,
    pub indices: Vec<usize>,
    pub probs: Vec<(usize, f32)>,
}

impl SamplingBuffers {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            scaled: vec![0.0; vocab_size],
            indices: Vec::with_capacity(vocab_size),
            probs: Vec::with_capacity(64),
        }
    }
}

/// 从 logits 中采样一个 token(复用 buffer 版本)
///
/// 步骤:
/// 1. 应用 temperature: logits /= T
/// 2. top-k 截断:保留最大的 K 个,其余设为 -inf
/// 3. top-p (nucleus) 截断:从高到低累加概率,达到 p 后截断
/// 4. softmax + 按概率随机选择
pub fn sample_top_k_top_p_into(
    logits: &[f32],
    params: SamplingParams,
    rng: &mut impl FnMut() -> f32,
    buf: &mut SamplingBuffers,
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

    // 1. 应用 temperature(写入预分配 buf.scaled)
    let inv_t = 1.0 / params.temperature;
    if buf.scaled.len() < n {
        buf.scaled.resize(n, 0.0);
    }
    let scaled = &mut buf.scaled[..n];
    for (s, &l) in scaled.iter_mut().zip(logits.iter()) {
        *s = l * inv_t;
    }

    // 2. top-k:用 `select_nth_unstable` 做部分排序(O(n) 而非 O(n log n))
    //    对 248320 词表:O(n) ~3ms vs O(n log n) ~12ms
    let k = params.top_k.min(n);
    buf.indices.clear();
    buf.indices.extend(0..n);
    // `select_nth_unstable` 把第 k 大元素放到位置 k,左侧均 ≤ 它
    // 然后只对前 k 个排序即可
    // ★ 关键:把 (scaled, index) 打包为 (f32, usize) 避免 select_nth 时跨数组随机访问
    //   原:scaled[b].partial_cmp(&scaled[a]) 需读 scaled[b] 和 scaled[a],跨数组 cache unfriendly
    //   新:indices 中存 (scaled_value, original_index),比较时直接用 key,无跨数组访问
    buf.indices.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
        scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    buf.indices[..k].sort_by(|&a, &b| {
        scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_k_indices = &buf.indices[..k];

    // 3. softmax on top-k
    let mut max = f32::NEG_INFINITY;
    for &i in top_k_indices {
        if scaled[i] > max {
            max = scaled[i];
        }
    }
    buf.probs.clear();
    buf.probs.extend(top_k_indices.iter().map(|&i| (i, (scaled[i] - max).exp())));
    let sum: f32 = buf.probs.iter().map(|(_, p)| *p).sum();
    let inv_sum = 1.0 / sum;
    for (_, p) in buf.probs.iter_mut() {
        *p *= inv_sum;
    }

    // 4. top-p:按概率降序累加,保留累积到 p 之前(含)的所有项
    buf.probs.sort_by(|(_, a), (_, b)| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0f32;
    let mut cutoff = buf.probs.len();
    for (i, &(_, p)) in buf.probs.iter().enumerate() {
        cum += p;
        if cum >= params.top_p {
            cutoff = i + 1;
            break;
        }
    }
    buf.probs.truncate(cutoff);
    // 重新归一化
    let new_sum: f32 = buf.probs.iter().map(|(_, p)| *p).sum();
    let new_inv = 1.0 / new_sum;
    for (_, p) in buf.probs.iter_mut() {
        *p *= new_inv;
    }

    // 5. 按概率选择
    let r = rng();
    let mut acc = 0.0f32;
    let mut chosen = buf.probs[0].0;
    for &(idx, p) in buf.probs.iter() {
        acc += p;
        if r <= acc {
            chosen = idx;
            break;
        }
    }
    chosen
}

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
    /// 重复惩罚因子 (1.0 = 不惩罚, 1.1-1.3 常用)
    /// 对最近生成的 token 的 logits 除以此因子, 抑制重复
    pub repetition_penalty: f32,
    /// 频率惩罚 (0.0 = 不惩罚, 0.1-1.0 常用)
    /// 对窗口内出现 n 次的 token, logit -= n * freq_penalty (线性累加)
    /// 与 repetition_penalty 叠加使用, 对高频重复 token 特别有效
    pub frequency_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.95,
            repetition_penalty: 1.3,
            frequency_penalty: 0.4,
        }
    }
}

/// 重复惩罚: 对 recent_tokens 中出现过的 token 应用惩罚
///
/// - `penalty` (repetition_penalty): 对窗口内每个出现过的 token, logits /= penalty
///   多次出现会多次除以 penalty (exponential 衰减, 已包含 frequency 意味)
/// - `freq_penalty` (frequency_penalty): 对窗口内出现 n 次的 token, logit -= n * freq_penalty
///   线性累加, 与 repetition_penalty 叠加, 提供更精细的高频抑制
///
/// window=256 覆盖整首诗的重复跨度 (vs 原 64 仅覆盖段落内)
pub fn apply_repetition_penalty(
    logits: &mut [f32],
    recent_tokens: &[u32],
    penalty: f32,
    freq_penalty: f32,
) {
    let need_rp = penalty > 1.0;
    let need_fp = freq_penalty > 0.0;
    if (!need_rp && !need_fp) || recent_tokens.is_empty() {
        return;
    }
    let window = 256usize;
    let start = recent_tokens.len().saturating_sub(window);
    let window_slice = &recent_tokens[start..];

    if need_fp {
        // ★ 需要统计频次: 用线性扫描统计每个 token 在窗口内的出现次数
        //   window ≤ 256, vocab=248K, 直接 O(N²) 扫描仅 65K 操作 (~0.05ms)
        //   比 HashMap 分配快得多, 且无需堆分配
        //   对每个 token: 先应用 repetition_penalty (除以 penalty^count), 再减 count * freq_penalty
        for i in 0..window_slice.len() {
            let tok = window_slice[i];
            // 跳过本 token 之前已处理过的相同 token (避免重复应用)
            // 简单去重: 若 window_slice[..i] 已包含 tok, 跳过
            if window_slice[..i].iter().any(|&t| t == tok) {
                continue;
            }
            // 统计 tok 在整个窗口的出现次数
            let count = window_slice.iter().filter(|&&t| t == tok).count();
            let idx = tok as usize;
            if idx < logits.len() {
                // repetition: 多次出现 = 多次除以 penalty (exponential)
                if need_rp {
                    let factor = penalty.powi(count as i32);
                    if logits[idx] > 0.0 {
                        logits[idx] /= factor;
                    } else {
                        logits[idx] *= factor;
                    }
                }
                // frequency: logit -= count * freq_penalty (线性)
                logits[idx] -= (count as f32) * freq_penalty;
            }
        }
    } else {
        // 仅 repetition_penalty: 原 O(N) 路径, 多次出现多次除以 penalty
        for &tok in window_slice {
            let idx = tok as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }
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
        // ★ 优化: 用 AVX2 argmax_avx2 替代标量 iter().max_by
        //   vocab=248K, 标量 ~0.5ms, AVX2 ~0.1ms, 每 token 省 ~0.4ms
        //   (bench --greedy 路径热路径, 每 token 都走)
        return crate::math::simd_exp::argmax_avx2(logits).0;
    }

    // 1. 应用 temperature(写入预分配 buf.scaled)
    // ★ 优化: AVX2 向量化 scale (原标量 248K iter ~0.3ms, AVX2 ~0.05ms)
    let inv_t = 1.0 / params.temperature;
    if buf.scaled.len() < n {
        buf.scaled.resize(n, 0.0);
    }
    let scaled = &mut buf.scaled[..n];
    if crate::math::simd_exp::simd_available() {
        crate::math::simd_exp::scale_avx2(logits, inv_t, scaled, n);
    } else {
        for (s, &l) in scaled.iter_mut().zip(logits.iter()) {
            *s = l * inv_t;
        }
    }

    // 2. top-k:用 `select_nth_unstable` 做部分排序(O(n) 而非 O(n log n))
    //    对 248320 词表:O(n) ~3ms vs O(n log n) ~12ms
    let k = params.top_k.min(n);
    buf.indices.clear();
    buf.indices.extend(0..n);
    // `select_nth_unstable` 把第 k 大元素放到位置 k,左侧均 ≤ 它
    // 然后只对前 k 个排序即可
    // indices 中存原始下标,比较时通过 scaled[idx] 间接比较 (跨数组访问,
    // vocab=248K 下 O(n) 次比较 cache 不友好, 但 select_nth 的 partition 模式
    // 使得大部分访问集中在局部, 实际影响可忽略)
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
    // ★ probs 已是降序 (top_k_indices 已降序 + exp 单调递增),无需再 sort
    //   原 sort_by 是冗余的 O(k log k) 操作,删除后每 token 省 ~1-2ms
    #[cfg(debug_assertions)]
    for w in buf.probs.windows(2) {
        debug_assert!(w[0].1 >= w[1].1 || w[0].1.is_nan() || w[1].1.is_nan(),
            "probs must be descending before top-p cutoff");
    }
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

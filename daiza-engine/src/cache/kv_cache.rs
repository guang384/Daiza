//! KV Cache: 存储每个 attention block 的 key/value 中间结果
//!
//! ## 布局: [seq, kv_head, dim]
//!
//! 每个 token 的 K/V 按 [kv_head, dim] 连续存储, head_dim=256 连续,
//! 在 attention 循环中顺序读取 head_dim 个 f32,完美命中 cache line。

#[derive(Clone)]
pub struct KvCache {
    /// K: [seq, kv_head, dim] — 每个 token 的 K 连续存储
    pub k: Vec<f32>,
    /// V: [seq, kv_head, dim]
    pub v: Vec<f32>,
    /// 当前已缓存的 token 数量
    pub len: usize,
    per_token: usize,
}

impl KvCache {
    pub fn new(n_kv_heads: usize, head_dim: usize, context_length: usize) -> Self {
        let per_token = n_kv_heads * head_dim;
        Self {
            k: Vec::with_capacity(context_length * per_token),
            v: Vec::with_capacity(context_length * per_token),
            len: 0,
            per_token,
        }
    }

    /// 追加一个新 token 的 K/V 向量
    pub fn append(&mut self, _pos: usize, k: &[f32], v: &[f32]) {
        self.k.extend_from_slice(k);
        self.v.extend_from_slice(v);
        self.len += 1;
    }

    /// 获取第 `seq_idx` 个 token 的 K slice: [n_kv_heads * head_dim]
    #[inline]
    pub fn k_at(&self, seq_idx: usize) -> &[f32] {
        let start = seq_idx * self.per_token;
        &self.k[start..start + self.per_token]
    }

    /// 获取第 `seq_idx` 个 token 的 V slice: [n_kv_heads * head_dim]
    #[inline]
    pub fn v_at(&self, seq_idx: usize) -> &[f32] {
        let start = seq_idx * self.per_token;
        &self.v[start..start + self.per_token]
    }

    /// 重置缓存(保留已分配内存,仅清零 len)
    pub fn reset(&mut self) {
        self.len = 0;
    }
}
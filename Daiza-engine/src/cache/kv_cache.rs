//! KV Cache: 存储每个 attention block 的 key/value 中间结果
//!
//! ## 布局: per-kvh [seq, dim]
//!
//! ★ 优化 (vs 原 [seq, kvh, dim] 单一 Vec): attention 内层循环对固定 kvh 遍历 c,
//!   新布局下同一 kvh 的所有 token 连续存储, cache line 利用率 100%。
//!   原布局下相邻 c 相距 per_token=4KB, 只读 head_dim=1KB, 利用率 25%。
//!   长 context (n_cached > 256, 工作集 > L2) 时收益显著。
//!
//! ## append 复杂度
//!
//! per-kvh Vec 方案: append O(n_kv_heads * head_dim) = O(per_token),
//! 与原 [seq, kvh, dim] 单一 Vec 的 extend_from_slice 相同。
//! (原 [kvh, seq, dim] 单一 Vec 方案 append 需中间插入, O(n) 不可行, 已放弃)

#[derive(Clone)]
pub struct KvCache {
    /// K: per-kvh, 每个 Vec 是 [seq, dim] 连续
    pub k: Vec<Vec<f32>>,
    /// V: per-kvh, 每个 Vec 是 [seq, dim] 连续
    pub v: Vec<Vec<f32>>,
    /// 当前已缓存的 token 数量
    pub len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// per_token = n_kv_heads * head_dim (用于 session_persist 兼容)
    per_token: usize,
}

impl KvCache {
    /// 初始预留 4096 token 容量 (避免频繁 realloc)
    const INITIAL_TOKEN_CAPACITY: usize = 4096;

    pub fn new(n_kv_heads: usize, head_dim: usize, _context_length: usize) -> Self {
        let per_token = n_kv_heads * head_dim;
        let cap_per_kvh = Self::INITIAL_TOKEN_CAPACITY * head_dim;
        let k = (0..n_kv_heads).map(|_| Vec::with_capacity(cap_per_kvh)).collect();
        let v = (0..n_kv_heads).map(|_| Vec::with_capacity(cap_per_kvh)).collect();
        Self { k, v, len: 0, n_kv_heads, head_dim, per_token }
    }

    /// 从扁平 [seq, kvh, dim] 数据构造 KvCache (用于从 SSD 恢复 session)
    ///
    /// 输入 k_flat/v_flat 是 session 文件格式的 [seq, kvh, dim] 交错布局,
    /// 内部按 kvh 分散存储到 per-kvh Vec (scatter)。
    pub fn from_raw(
        k_flat: &[f32],
        v_flat: &[f32],
        len: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let per_token = n_kv_heads * head_dim;
        debug_assert_eq!(k_flat.len(), len * per_token);
        debug_assert_eq!(v_flat.len(), len * per_token);
        let cap_per_kvh = len.max(Self::INITIAL_TOKEN_CAPACITY) * head_dim;
        let mut k: Vec<Vec<f32>> = (0..n_kv_heads).map(|_| Vec::with_capacity(cap_per_kvh)).collect();
        let mut v: Vec<Vec<f32>> = (0..n_kv_heads).map(|_| Vec::with_capacity(cap_per_kvh)).collect();
        for kvh in 0..n_kv_heads {
            for seq in 0..len {
                let src = seq * per_token + kvh * head_dim;
                k[kvh].extend_from_slice(&k_flat[src..src + head_dim]);
                v[kvh].extend_from_slice(&v_flat[src..src + head_dim]);
            }
        }
        Self { k, v, len, n_kv_heads, head_dim, per_token }
    }

    /// 每个 token 的 K/V 向量长度 (n_kv_heads * head_dim)
    #[inline]
    pub fn per_token(&self) -> usize {
        self.per_token
    }

    /// 追加一个新 token 的 K/V 向量
    ///
    /// 输入 k/v 布局: [n_kv_heads * head_dim] 连续 (投影输出)
    /// 内部按 kvh 分散写入到 per-kvh Vec
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.per_token);
        debug_assert_eq!(v.len(), self.per_token);
        for kvh in 0..self.n_kv_heads {
            let src = kvh * self.head_dim;
            self.k[kvh].extend_from_slice(&k[src..src + self.head_dim]);
            self.v[kvh].extend_from_slice(&v[src..src + self.head_dim]);
        }
        self.len += 1;
    }

    /// 获取第 kvh 个 head 的第 seq_idx 个 token 的 K slice: [head_dim]
    #[inline]
    pub fn k_head_at(&self, kvh: usize, seq_idx: usize) -> &[f32] {
        let start = seq_idx * self.head_dim;
        &self.k[kvh][start..start + self.head_dim]
    }

    /// 获取第 kvh 个 head 的第 seq_idx 个 token 的 V slice: [head_dim]
    #[inline]
    pub fn v_head_at(&self, kvh: usize, seq_idx: usize) -> &[f32] {
        let start = seq_idx * self.head_dim;
        &self.v[kvh][start..start + self.head_dim]
    }

    /// 重置缓存 (保留已分配内存, 仅清零 len)
    pub fn reset(&mut self) {
        for kvh in 0..self.n_kv_heads {
            self.k[kvh].clear();
            self.v[kvh].clear();
        }
        self.len = 0;
    }

    /// 截断缓存到 new_len 个 token (用于 speculative decoding rollback)
    /// 保留前 new_len 个 token 的 K/V, 多余的被逻辑丢弃 (Vec 不缩容, 避免 realloc)
    pub fn truncate(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.len);
        let new_bytes = new_len * self.head_dim;
        for kvh in 0..self.n_kv_heads {
            self.k[kvh].truncate(new_bytes);
            self.v[kvh].truncate(new_bytes);
        }
        self.len = new_len;
    }
}

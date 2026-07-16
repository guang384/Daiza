//! 全注意力层的 KV cache
//!
//! 学习项目 v0:FP32 存储。
//! 后续优化:4-bit KV 量化(白皮书 §4.4 显示 Bonsai 对此几乎无损)

pub struct KvCache {
    /// K 向量缓存:[max_seq_len, head_count_kv, head_dim]
    pub k: Vec<f32>,
    /// V 向量缓存:[max_seq_len, head_count_kv, head_dim]
    pub v: Vec<f32>,
    /// 已写入的 token 数(下一个写入位置 = len)
    pub len: usize,
    /// 每层一个 KV head 的元素数 = head_count_kv * head_dim
    pub per_token: usize,
}

impl KvCache {
    pub fn new(head_count_kv: usize, head_dim: usize, _max_seq_len: usize) -> Self {
        let per_token = head_count_kv * head_dim;
        // 不再按 context_length 预分配(262144 * 1024 * 4B = 1GB/层,16 层 = 32GB)
        // 改为按需增长;预分配 64 token 容量以减少早期重分配
        let cap = 64 * per_token;
        Self {
            k: Vec::with_capacity(cap),
            v: Vec::with_capacity(cap),
            len: 0,
            per_token,
        }
    }

    /// 追加一个 token 的 K 和 V 向量(每个长度 = per_token)
    pub fn append(&mut self, pos: usize, k: &[f32], v: &[f32]) {
        let _ = pos;
        if k.len() >= self.per_token {
            self.k.extend_from_slice(&k[..self.per_token]);
            self.v.extend_from_slice(&v[..self.per_token]);
            self.len += 1;
        }
    }

    /// 取第 `i` 个历史 token 的 K 向量(长度 = per_token)
    pub fn k_at(&self, i: usize) -> &[f32] {
        let start = i * self.per_token;
        &self.k[start..start + self.per_token]
    }

    /// 取第 `i` 个历史 token 的 V 向量
    pub fn v_at(&self, i: usize) -> &[f32] {
        let start = i * self.per_token;
        &self.v[start..start + self.per_token]
    }

    /// 重置到 0
    pub fn reset(&mut self) {
        self.k.clear();
        self.v.clear();
        self.len = 0;
    }
}

//! SSM (Gated DeltaNet) 层的 recurrent state
//!
//! 与 KV cache 不同,**SSM state 是固定大小的**:
//!   `[num_v_heads, state_size, state_size]` = 48 * 128 * 128 = 786432 floats ≈ 3 MB / 层
//!
//! 在 decode 阶段每 token 用 delta rule 原地更新,不增长。
//! 这是混合注意力能在 262K context 上可行的核心原因。

pub struct SsmState {
    /// recurrent state: `[num_v_heads, state_size, state_size]`
    /// 每个 v_head 维护一个 [128, 128] 矩阵作为线性 attention 的"记忆"
    pub state: Vec<f32>,
    /// Conv1d 滑窗历史:长度 = conv_kernel 个 token 的 qkv 缓存
    /// 形状: `[conv_kernel, qkv_dim]` (qkv_dim = 2 * num_k_heads * state_size + inner)
    pub conv_history: Vec<f32>,
}

impl SsmState {
    pub fn new(num_v_heads: usize, state_size: usize) -> Self {
        // Gated DeltaNet 的 state 是 [num_v_heads, state_size, state_size] 矩阵
        let n = num_v_heads * state_size * state_size;
        Self {
            state: vec![0.0; n],
            conv_history: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.conv_history.clear();
    }
}

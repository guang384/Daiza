//! 跨 token 复用的工作区缓冲区(消除热路径堆分配)
//!
//! ## 设计动机
//!
//! 原实现每 token 在 block/attention/ssm/mlp 内部产生 ~1000+ 次 `vec!`/`to_vec()` 堆分配,
//! 每次 alloc ~20μs,合计 ~20ms/token,占单 token 时延的显著比例。
//!
//! 本 Workspace 在 `ForwardContext` 创建时一次性预分配所有热路径 buffer,
//! 跨 token 复用 —— 同一个 buffer 在每次 `forward_single_token` 中被覆盖写入。
//!
//! 主残差流 `h_buf` 不在本 Workspace 中,而在 `ForwardContext::h_buf`,
//! 避免与 `&mut Workspace` 的借用冲突(两者需同时传给 `block::forward_single_inplace`)。
//!
//! ## Buffer 划分原则
//!
//! - 同一阶段不会同时使用的 buffer 可以共享(本实现保守,全部独立)
//! - 大小固定(由 Config 决定),`attn_scores` 例外(随序列长度增长)
//! - 最终输出仍返回 owned `Vec`(每 token 1 个 alloc,可接受),内部 buffer 全部复用

use crate::model::config::Config;

/// 工作线程数,由 `DAIZA_THREADS` env var 控制,默认物理核数
///
/// 缓存结果避免每次调用 `std::env::var`。第一次调用时初始化。
pub fn thread_count() -> usize {
    use std::sync::OnceLock;
    static N_THREADS: OnceLock<usize> = OnceLock::new();
    *N_THREADS.get_or_init(|| {
        if let Ok(s) = std::env::var("DAIZA_THREADS") {
            if let Ok(n) = s.parse::<usize>() {
                if n > 0 {
                    return n;
                }
            }
        }
        // 默认:物理核数(避开 SMT 超线程,避免 false sharing)
        // std::thread::available_parallelism 返回逻辑核数,SMT 下通常 = 物理×2
        // 取一半更接近物理核数,但避免 0
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        (logical / 2).max(1)
    })
}

pub struct Workspace {
    // === block.rs 用 ===
    /// `[hidden]` attn_norm 输入(norm 后的 h,送入 attention/ssm)
    pub block_normed: Vec<f32>,
    /// `[hidden]` post_attention_norm 输入(norm 后的 h,送入 mlp)
    pub block_mlp_in: Vec<f32>,

    // === attention.rs 用 ===
    /// `[n_q_heads * head_dim * 2]` = 12288,attn_q matvec 输出(Q + gate 交错)
    pub attn_q_total: Vec<f32>,
    /// `[n_q_heads * head_dim]` = 6144,解交错后的 Q
    pub attn_q: Vec<f32>,
    /// `[n_q_heads * head_dim]` = 6144,解交错后的 gate
    pub attn_gate: Vec<f32>,
    /// `[n_kv_heads * head_dim]` = 1024,K 投影
    pub attn_k: Vec<f32>,
    /// `[n_kv_heads * head_dim]` = 1024,V 投影
    pub attn_v: Vec<f32>,
    /// `[n_q_heads * head_dim]` = 6144,attention 输出(被 gate 调制后)
    pub attn_out: Vec<f32>,
    /// `[max_seq_len]` softmax scores(动态增长,与 KV cache 同步)
    pub attn_scores: Vec<f32>,

    // === ssm.rs 用 ===
    /// `[qkv_full_len]` = 10240,attn_qkv matvec 输出
    pub ssm_qkv: Vec<f32>,
    /// `[qkv_dim]` = 2048,conv 后的 q
    pub ssm_q: Vec<f32>,
    /// `[qkv_dim]` = 2048,conv 后的 k
    pub ssm_k: Vec<f32>,
    /// `[ssm_inner]` = 6144,conv 后的 v
    pub ssm_v: Vec<f32>,
    /// `[qkv_full_len]` = 10240,conv1d 输出
    pub ssm_conv_out: Vec<f32>,
    /// `[ssm_inner]` = 6144,scan 输出 y
    pub ssm_y: Vec<f32>,
    /// `[num_v_heads]` = 48,alpha 投影
    pub ssm_alpha: Vec<f32>,
    /// `[num_v_heads]` = 48,beta 投影
    pub ssm_beta: Vec<f32>,
    /// `[ssm_inner]` = 6144,output gate z = attn_gate @ x
    pub ssm_z: Vec<f32>,

    // === mlp.rs 用 ===
    /// `[d_ff]` = 17408,gate 投影
    pub mlp_gate: Vec<f32>,
    /// `[d_ff]` = 17408,up 投影
    pub mlp_up: Vec<f32>,
    // mlp_down 结果通过 `matvec_add_into_slice` 直接累加到 h_buf,无需独立 buffer
}

impl Workspace {
    pub fn new(cfg: &Config) -> Self {
        let hidden = cfg.hidden;
        let head_dim = cfg.head_dim;
        let n_q_heads = cfg.head_count;
        let n_kv_heads = cfg.head_count_kv;
        let d_ff = cfg.feed_forward_length;
        let ssm_inner = cfg.ssm_inner_size;
        let qkv_dim = cfg.ssm_group_count * cfg.ssm_state_size;
        let qkv_full_len = 2 * qkv_dim + ssm_inner;
        let num_v_heads = cfg.ssm_time_step_rank;

        Self {
            block_normed: vec![0.0; hidden],
            block_mlp_in: vec![0.0; hidden],

            attn_q_total: vec![0.0; n_q_heads * head_dim * 2],
            attn_q: vec![0.0; n_q_heads * head_dim],
            attn_gate: vec![0.0; n_q_heads * head_dim],
            attn_k: vec![0.0; n_kv_heads * head_dim],
            attn_v: vec![0.0; n_kv_heads * head_dim],
            attn_out: vec![0.0; n_q_heads * head_dim],
            // 预分配到 context_length,消除热路径 realloc
            // (长上下文下 n_cached 会持续增长,with_capacity 会多次 realloc)
            attn_scores: vec![0.0; cfg.context_length],

            ssm_qkv: vec![0.0; qkv_full_len],
            ssm_q: vec![0.0; qkv_dim],
            ssm_k: vec![0.0; qkv_dim],
            ssm_v: vec![0.0; ssm_inner],
            ssm_conv_out: vec![0.0; qkv_full_len],
            ssm_y: vec![0.0; ssm_inner],
            ssm_alpha: vec![0.0; num_v_heads],
            ssm_beta: vec![0.0; num_v_heads],
            ssm_z: vec![0.0; ssm_inner],

            mlp_gate: vec![0.0; d_ff],
            mlp_up: vec![0.0; d_ff],
        }
    }
}

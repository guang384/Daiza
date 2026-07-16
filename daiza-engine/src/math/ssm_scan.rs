//! Mamba2 选择性扫描(selective scan)
//!
//! 这是整个引擎最复杂的算子。Mamba2 的 SSM 块在每个时间步:
//!
//! ```text
//! Δ_t = softplus(W_Δ @ x_t + b_Δ)         # 时间步长
//! B_t, C_t = split(W_BC @ x_t)            # 输入门 / 输出门
//! A_t = -exp(A_log)                       # 衰减(对数参数化)
//!
//! # 离散化(zero-order hold)
//! Ā_t = exp(A_t * Δ_t)
//! B̄_t = B_t * Δ_t
//!
//! # 状态更新
//! h_t = Ā_t * h_{t-1} + B̄_t * x_t
//! y_t = C_t @ h_t
//! ```
//!
//! `state_size = 128`,`group_count = 16`(A/B/C 按 group 共享:`A.ndim=16,128`),
//! `time_step_rank = 48`(Δ 投影到 48 维),`inner_size = 6144`。
//!
//! ## 学习项目实现策略
//!
//! 完整的 Mamba2 实现需要严格对照参考代码(推荐 Mamba2 原论文的官方实现)。
//! v0 先实现最朴素的逐时间步循环,正确性优先。组共享(grouped)是 Mamba2 vs Mamba1 的关键差异,
//! 必须在 `a / b / c` 的 reshape 上正确分组。

/// SSM 单时间步的状态更新(decode 阶段,single token)
///
/// 输入:
/// - `x_t`: 当前 token 的输入向量,长度 = inner
/// - `state`: 当前 SSM 状态,长度 = group_count * state_size(被原地修改)
/// - `dt`: 时间步长向量,长度 = time_step_rank(已 softplus 后)
/// - `a_log`: 对数参数化的 A,长度 = group_count * state_size
/// - `b_proj`: B 投影结果,长度 = group_count * state_size
/// - `c_proj`: C 投影结果,长度 = group_count * state_size
///
/// 输出:
/// - `y_out`: 长度 = inner
pub fn ssm_step(
    state: &mut [f32],
    dt: &[f32],
    a_log: &[f32],
    b_proj: &[f32],
    c_proj: &[f32],
    group_count: usize,
    state_size: usize,
    time_step_rank: usize,
    inner: usize,
    y_out: &mut [f32],
) {
    // 实现待补充(参考 Mamba2 官方 CUDA kernel 的语义)
    // 关键点:
    //   1. dt 经过 softplus
    //   2. a_log 取负 exp → A < 0
    //   3. B、C 由 ssm_alpha / ssm_beta 投影得到(参考 Nemotron-H / Qwen3.6 SSM block)
    //   4. 状态按 group 切分:A[group_idx, state_idx] 与对应 B/C 维度匹配
    //
    // 学习项目 v0 占位:返回零向量以让前向传播链路编译通过
    for y in y_out.iter_mut() {
        *y = 0.0;
    }
    let _ = (dt, a_log, b_proj, c_proj, group_count, state_size, time_step_rank, inner);
}

/// SSM prefill(对整个 prompt 一次扫描)
///
/// - 比 decode 复杂:需要对 seq_len 个时间步连续更新状态,并输出每步的 y
/// - 学习项目 v0 占位
pub fn ssm_scan_prefill(
    _x_seq: &[f32],
    _state: &mut [f32],
    _seq_len: usize,
    _inner: usize,
    _group_count: usize,
    _state_size: usize,
    _y_out: &mut [f32],
) {
    // TODO: 逐位置调用 ssm_step
}

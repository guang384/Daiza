//! 会话管理:跨多轮对话复用 KV cache + SSM state
//!
//! 核心思想:Session 持有 ModelState + Workspace + buffers,
//! 不持有 weights 引用。session_reply 时临时构造 ForwardContext,
//! 用所有权移交避免生命周期冲突。
//!
//! 相关模块:
//! - `session_persist`: SSD 持久化 (dump/load)
//! - `session_manager`: 多 session LRU 管理 (active 在 DRAM, inactive 在 SSD)
//! - `tool_call`: 工具调用 (tools/tool_call/tool_response 模板渲染 + 解析)

use daiza_engine::math::{sample_top_k_top_p_into, LcgRng, SamplingBuffers, SamplingParams};
use daiza_engine::model::forward::{
    forward_batch, forward_single_token, ForwardContext, ModelState,
};
use daiza_engine::model::workspace::Workspace;
use crate::tokenizer::BpeTokenizer;
use daiza_engine::model::config::Config;
use daiza_engine::model::weights::LoadedWeights;
use crate::Result;
use crate::tool_call::{ToolDef, ToolMessage, ToolResponse};

/// 一次会话的运行时状态,跨多轮 reply 复用
///
/// 持有 KV cache + SSM state + 各种 buffer,避免每轮重新分配。
/// 不持有 weights 引用 → 无生命周期参数。
pub struct Session {
    pub state: ModelState,
    pub workspace: Workspace,
    pub history_tokens: Vec<u32>,
    pub h_buf: Vec<f32>,
    pub logits_buf: Vec<f32>,
    pub cos_buf: Vec<f32>,
    pub sin_buf: Vec<f32>,
    /// 是否启用思考模式(控制 increment 末尾是否追加 <think>\n)
    pub think_enabled: bool,
    /// 系统提示词(首轮 prefill 时使用)
    pub system_prompt: Option<String>,
    /// 待发送的图片路径 (REPL /image 收集, 下次 session_reply 时消费并清空)
    pub pending_images: Vec<std::path::PathBuf>,
    /// 已激活的工具列表 (session_reply 时渲染到 system prompt)
    pub tools: Vec<ToolDef>,
    /// 历史消息 (按 role 记录, 用于 tool_call 场景的 multi-step 模板渲染)
    /// 普通对话不会填充此字段 (走 history_tokens 增量路径)
    /// tool_call 模式下,每次 reply 后追加 Assistant 消息, ToolResponse 时追加 Tool 消息
    pub messages: Vec<ToolMessage>,
    /// DSpark 跨轮 KV cache 复用: 累积所有已 forward token 的 hidden tap
    /// 每个 token 一行 [n_tap_layers * hidden], 由 engine.generate_with_dspark_stream 维护
    pub dspark_tap_history: Vec<f32>,
    /// DSpark target tap 层配置 (首次 prefill 时从 spec_ctx 读, 跨轮复用)
    pub dspark_tap_layers: Vec<usize>,
}

impl Session {
    /// 创建新会话(state 全零, pos=0)
    pub fn new(cfg: &Config, think_enabled: bool, system_prompt: Option<String>) -> Self {
        Self {
            state: ModelState::new(cfg),
            workspace: Workspace::new(cfg),
            history_tokens: Vec::new(),
            h_buf: vec![0.0; cfg.hidden],
            logits_buf: Vec::with_capacity(cfg.vocab_size),
            cos_buf: vec![0.0; cfg.rope_dim],
            sin_buf: vec![0.0; cfg.rope_dim],
            think_enabled,
            system_prompt,
            pending_images: Vec::new(),
            tools: Vec::new(),
            messages: Vec::new(),
            dspark_tap_history: Vec::new(),
            dspark_tap_layers: Vec::new(),
        }
    }

    /// 重置会话(KV/SSM 全部清零, pos 归零, 历史清空)
    pub fn reset(&mut self) {
        self.state.reset();
        self.history_tokens.clear();
        self.pending_images.clear();
        self.messages.clear();
        self.dspark_tap_history.clear();
    }
}

/// 将 ForwardContext 的状态移回 Session
///
/// ★ 错误安全的状态管理: 无论 prefill/decode 成功或失败, 都必须调用此函数,
///   否则 ForwardContext 被 drop 时 ModelState (KV cache + SSM state + pos) 会随之丢失,
///   导致下一轮对话从空状态开始 (表现为"多轮对话上下文丢失")。
///
/// 成功路径: 在 history_tokens.extend 之前调用;
/// 错误路径: 在 `?` 传播错误之前调用 (本文件通过 IIFE 模式统一处理)。
fn restore_ctx_state(session: &mut Session, ctx: &mut ForwardContext<'_>) {
    session.state = std::mem::take(&mut ctx.state);
    session.h_buf = std::mem::take(&mut ctx.h_buf);
    session.workspace = std::mem::take(&mut ctx.workspace);
    session.logits_buf = std::mem::take(&mut ctx.logits_buf);
    session.cos_buf = std::mem::take(&mut ctx.cos_buf);
    session.sin_buf = std::mem::take(&mut ctx.sin_buf);
}

/// 构造增量 token 文本 (普通对话路径)
///
/// - 首轮:`<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n` [+ `<think>\n`]
/// - 后续轮:`<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n` [+ `<think>\n`]
///
/// 后续轮开头的 `<|im_end|>\n` 补上上一轮 assistant 回复的结束标记
/// (decode 遇到 EOS 时 break 不 forward, EOS 的 KV 在下一轮 prefill 时补入)
fn build_increment(session: &Session, user_msg: &str) -> String {
    let mut s = String::new();
    if session.history_tokens.is_empty() {
        // 首轮: 带 system prompt (tools 非空时注入 tools 块)
        if !session.tools.is_empty() {
            s.push_str("<|im_start|>system\n");
            s.push_str(&render_tools_block(&session.tools));
            if let Some(sys) = &session.system_prompt {
                s.push_str(sys);
                s.push('\n');
            }
            s.push_str("<|im_end|>\n");
        } else if let Some(sys) = &session.system_prompt {
            s.push_str("<|im_start|>system\n");
            s.push_str(sys);
            s.push_str("<|im_end|>\n");
        }
    } else {
        // 后续轮:补上上一轮的结束标记 (decode 时 EOS 未 forward, 这里补入)
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>user\n");
    s.push_str(user_msg);
    s.push_str("<|im_end|>\n");
    s.push_str("<|im_start|>assistant\n");
    if session.think_enabled {
        // <think>\n (换行符) — <think> 作为 special token 被 tokenizer 识别,
        // \n 作为普通 token 编码, 模型看到 <think> special token 后进入 thinking 模式
        s.push_str("<think>\n");
    } else {
        // ★ think 关闭: 追加空 think 块 <think></think>\n, 告诉模型 think 阶段已结束,
        //   直接生成正式回答 (Qwen3 标准 enable_thinking=false 处理)
        //   Bonsai-27B 训练时 assistant 总是以 <think> 开头, 若只写 assistant\n,
        //   模型会自发生成 <think>...长篇think内容</think>, 浪费 token 和算力
        s.push_str("<think></think>\n");
    }
    s
}

/// 渲染 tools 块 (注入到 system prompt)
/// 格式参考 tool_call::render_chat_template 中的 tools 部分
pub fn render_tools_block(tools: &[ToolDef]) -> String {
    let mut s = String::new();
    s.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
    for tool in tools {
        s.push_str(&format_tool_def_inline(tool));
        s.push('\n');
    }
    s.push_str("</tools>\n\n");
    s.push_str("If you choose to call a function ONLY reply in the following format with NO suffix:\n\n");
    s.push_str("<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n");
    s.push_str("<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n");
    s.push_str("<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n");
    s.push_str("- Required parameters MUST be specified\n");
    s.push_str("- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n");
    s.push_str("- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n");
    s.push_str("</IMPORTANT>\n");
    s
}

/// 渲染单个 tool 定义为 JSON schema 格式 (与 tool_call::format_tool_def 一致, 内联避免可见性冲突)
fn format_tool_def_inline(tool: &ToolDef) -> String {
    let mut s = String::new();
    s.push_str("{\"type\": \"function\", \"function\": {");
    s.push_str(&format!("\"name\": \"{}\", ", escape_json_inline(&tool.name)));
    s.push_str(&format!("\"description\": \"{}\", ", escape_json_inline(&tool.description)));
    s.push_str("\"parameters\": {\"type\": \"object\", \"properties\": {");
    for (i, p) in tool.parameters.iter().enumerate() {
        if i > 0 { s.push_str(", "); }
        s.push_str(&format!("\"{}\": {{\"type\": \"{}\", \"description\": \"{}\"}}",
            escape_json_inline(&p.name), escape_json_inline(&p.param_type), escape_json_inline(&p.description)));
    }
    s.push_str("}, \"required\": [");
    let mut first_req = true;
    for p in tool.parameters.iter() {
        if !p.required { continue; }
        if !first_req { s.push_str(", "); }
        first_req = false;
        s.push_str(&format!("\"{}\"", escape_json_inline(&p.name)));
    }
    s.push_str("]}}");
    s
}

fn escape_json_inline(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// 分批 prefill: forward_batch 内部 tmp 数组限制 n_batch ≤ 64,
/// 超过时需分批调用, 每批 ≤ MAX_PREFILL_BATCH 个 token, 从 ctx.state.pos 继续。
const MAX_PREFILL_BATCH: usize = 64;

fn prefill_batched(ctx: &mut ForwardContext<'_>, input_ids: &[u32], start_pos: usize) -> Result<()> {
    prefill_batched_with_progress(ctx, input_ids, start_pos, None)
}

/// 分批 prefill + 进度上报
///
/// `on_progress`: 若 Some, 每完成一批 forward_batch 调用一次 (done_tokens, total_tokens)
fn prefill_batched_with_progress(
    ctx: &mut ForwardContext<'_>,
    input_ids: &[u32],
    start_pos: usize,
    mut on_progress: Option<&mut dyn FnMut(usize, usize)>,
) -> Result<()> {
    let n = input_ids.len();
    if n == 0 {
        return Ok(());
    }
    if n <= MAX_PREFILL_BATCH {
        forward_batch(ctx, input_ids, start_pos, None)?;
        if let Some(cb) = on_progress {
            cb(n, n);
        }
        return Ok(());
    }
    // 分批: 每批 MAX_PREFILL_BATCH 个 token, pos 自动累加
    let mut offset = 0usize;
    while offset < n {
        let end = (offset + MAX_PREFILL_BATCH).min(n);
        let batch = &input_ids[offset..end];
        forward_batch(ctx, batch, start_pos + offset, None)?;
        offset = end;
        if let Some(cb) = on_progress.as_deref_mut() {
            cb(offset, n);
        }
    }
    Ok(())
}

/// 执行一轮对话:增量 prefill + decode
///
/// 内部流程:
/// 1. take session 出来(避免借用冲突)
/// 2. 构造增量 token,从 state.pos 继续 prefill
/// 3. decode 循环(遇到 EOS 停止, 不 forward EOS)
/// 4. state 移回 session
///
/// 多模态: 若 session.pending_images 非空, 调用方需通过 session_reply_with_vision
/// 走 vision 注入路径 (本函数不处理 image)
pub fn session_reply(
    cfg: &Config,
    weights: &LoadedWeights,
    tokenizer: &BpeTokenizer,
    session: &mut Session,
    user_msg: &str,
    max_tokens: usize,
    params: SamplingParams,
) -> Result<String> {
    // tool_call 模式: 走增量 prefill 路径 (build_increment 首轮注入 tools 块)
    // 完整重渲染路径 session_reply_with_tools 保留供 REPL 或 tool_response 回传使用
    // web 层通过流式 delta 检测 <tool_call> 标签, 不走 session_reply_with_tools

    // 1. 构造增量 token
    let increment = build_increment(session, user_msg);
    let input_ids = tokenizer.encode(&increment);
    if input_ids.is_empty() {
        return Err(crate::BonsaiError::Tokenizer(
            "increment encode returned empty".into(),
        ));
    }

    // 2. 构造临时 ForwardContext,移交 session 的 state/buffers 所有权
    let start_pos = session.state.pos;
    let mut ctx = ForwardContext {
        cfg,
        weights,
        state: std::mem::take(&mut session.state),
        h_buf: std::mem::take(&mut session.h_buf),
        workspace: std::mem::take(&mut session.workspace),
        logits_buf: std::mem::take(&mut session.logits_buf),
        cos_buf: std::mem::take(&mut session.cos_buf),
        sin_buf: std::mem::take(&mut session.sin_buf),
        hidden_tap_buf: Vec::new(),
        hidden_tap_layers: Vec::new(),
        hidden_tap_batch_buf: Vec::new(),
    };

    // 3-4. prefill + decode 封装到 IIFE
    // ★ 错误安全: inner 内任何 `?` 失败都会让闭包返回 Err, 但 ctx 不会被 drop
    //   (闭包只持有 &mut ctx, 所有权在外层), 外层 restore_ctx_state 仍会执行,
    //   保证 KV cache + pos 正确回移到 session, 下一轮对话可复用历史上下文。
    let inner_result: Result<Vec<u32>> = (|| {
        let n_input = input_ids.len();
        if n_input == 1 {
            forward_single_token(&mut ctx, input_ids[0])?;
        } else if n_input > 1 {
            prefill_batched(&mut ctx, &input_ids, start_pos)?;
        }

        let mut rng = LcgRng::new(0xC0FFEE);
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut sampling_buf = SamplingBuffers::new(ctx.logits_buf.len());

        for _step in 0..max_tokens {
            let next_id = sample_top_k_top_p_into(
                &ctx.logits_buf, params, &mut || rng.next_f32(), &mut sampling_buf,
            );
            // EOS:停止生成,不 forward EOS(EOS 的 KV 在下一轮 increment 补入)
            if next_id as u32 == cfg.eos_token_id {
                break;
            }
            generated_ids.push(next_id as u32);
            forward_single_token(&mut ctx, next_id as u32)?;
        }
        Ok(generated_ids)
    })();

    // 5. state/buffers 移回 session(关键:无论成功失败都执行, 不 drop!)
    restore_ctx_state(session, &mut ctx);

    let generated_ids = inner_result?;
    session.history_tokens.extend(&input_ids);
    session.history_tokens.extend(&generated_ids);

    // 6. 正确性验证: dump token IDs (env DAIZA_DUMP_TOKENS=path)
    crate::session_persist::dump_tokens_if_enabled(&input_ids, &generated_ids)?;

    // 7. decode token ids 为字符串
    Ok(tokenizer.decode(&generated_ids))
}

/// 解码单个 token 为原始字节 (与 BpeTokenizer::decode 的字节收集逻辑一致,
/// 但不做 UTF-8 lossy 转换, 保留跨 token 的不完整字节供增量对齐)。
pub fn decode_token_bytes(tokenizer: &BpeTokenizer, id: u32, buf: &mut Vec<u8>) {
    if let Some(token_text) = tokenizer.vocab.tokens.get(id as usize) {
        for ch in token_text.chars() {
            if let Some(&b) = tokenizer.unicode_char_to_byte.get(&ch) {
                buf.push(b);
            }
        }
    }
}

/// 从 `pending` 中解码出所有完整的 UTF-8 字符并返回, 不完整的尾部字节保留到下次。
/// 遇到真正的非法字节时用 U+FFFD 替换 (与 `String::from_utf8_lossy` 行为一致)。
pub fn drain_complete_utf8(pending: &mut Vec<u8>) -> String {
    if pending.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let mut consumed = 0usize;
    let n = pending.len();
    while consumed < n {
        match std::str::from_utf8(&pending[consumed..]) {
            Ok(s) => {
                out.push_str(s);
                consumed = n;
                break;
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid > 0 {
                    out.push_str(std::str::from_utf8(&pending[consumed..consumed + valid]).unwrap());
                    consumed += valid;
                }
                match e.error_len() {
                    None => break, // 尾部不完整多字节序列, 保留到下次
                    Some(len) => {
                        // 非法字节, 用 U+FFFD 替换 (与 from_utf8_lossy 一致)
                        out.push('\u{FFFD}');
                        consumed += len;
                    }
                }
            }
        }
    }
    if consumed > 0 {
        pending.drain(..consumed);
    }
    out
}

/// 流式版 session_reply: 每生成一个 token 就回调一次增量文本
///
/// 与 session_reply 的区别仅在于 decode 循环中每步通过 `on_delta`
/// 上报自上次以来新增的文本 (增量解码, 处理跨 token UTF-8 边界),
/// 其余逻辑 (增量 prefill / EOS 处理 / state 移交 / history 维护) 完全一致。
///
/// `on_delta(&str)` 返回 false 可中断生成 (已生成部分照常入库)。
/// 若 session.tools 非空 (tool_call 模式), 回调只触发一次 (完整输出)。
pub fn session_reply_stream(
    cfg: &Config,
    weights: &LoadedWeights,
    tokenizer: &BpeTokenizer,
    session: &mut Session,
    user_msg: &str,
    max_tokens: usize,
    params: SamplingParams,
    on_delta: &mut dyn FnMut(&str) -> bool,
) -> Result<String> {
    // tool_call 模式: 走增量 prefill 流式路径 (build_increment 首轮注入 tools 块)
    // web 层通过 on_delta 回调累积文本, 检测 <tool_call> 标签
    let increment = build_increment(session, user_msg);
    reply_with_increment_stream(cfg, weights, tokenizer, session, &increment, max_tokens, params, on_delta)
}

/// tool_response 回传后的流式生成
///
/// 构造 `<|im_end|>\n<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n<|im_start|>assistant\n` 增量 prompt,
/// 复用 reply_with_increment_stream 核心逻辑。
pub fn session_reply_tool_response_stream(
    cfg: &Config,
    weights: &LoadedWeights,
    tokenizer: &BpeTokenizer,
    session: &mut Session,
    tool_content: &str,
    max_tokens: usize,
    params: SamplingParams,
    on_delta: &mut dyn FnMut(&str) -> bool,
) -> Result<String> {
    let increment = build_tool_response_increment(session, tool_content);
    reply_with_increment_stream(cfg, weights, tokenizer, session, &increment, max_tokens, params, on_delta)
}

/// 构造 tool_response 增量 prompt
///
/// 格式: `<|im_end|>\n<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n<|im_start|>assistant\n` [+ think 头]
fn build_tool_response_increment(session: &Session, content: &str) -> String {
    let mut s = String::new();
    // tool_response 一定在后续轮 (首轮不可能有 tool_call), 补上上一轮 EOS
    s.push_str("<|im_end|>\n");
    s.push_str("<|im_start|>user\n<tool_response>\n");
    s.push_str(content);
    s.push_str("\n</tool_response><|im_end|>\n");
    s.push_str("<|im_start|>assistant\n");
    if session.think_enabled {
        s.push_str("<think>\n");
    } else {
        s.push_str("<think></think>\n");
    }
    s
}

/// 核心流式生成: 接收已构造好的 increment, 执行 prefill + decode + on_delta 回调
fn reply_with_increment_stream(
    cfg: &Config,
    weights: &LoadedWeights,
    tokenizer: &BpeTokenizer,
    session: &mut Session,
    increment: &str,
    max_tokens: usize,
    params: SamplingParams,
    on_delta: &mut dyn FnMut(&str) -> bool,
) -> Result<String> {
    let input_ids = tokenizer.encode(increment);
    if input_ids.is_empty() {
        return Err(crate::BonsaiError::Tokenizer(
            "increment encode returned empty".into(),
        ));
    }

    let start_pos = session.state.pos;
    let mut ctx = ForwardContext {
        cfg,
        weights,
        state: std::mem::take(&mut session.state),
        h_buf: std::mem::take(&mut session.h_buf),
        workspace: std::mem::take(&mut session.workspace),
        logits_buf: std::mem::take(&mut session.logits_buf),
        cos_buf: std::mem::take(&mut session.cos_buf),
        sin_buf: std::mem::take(&mut session.sin_buf),
        hidden_tap_buf: Vec::new(),
        hidden_tap_layers: Vec::new(),
        hidden_tap_batch_buf: Vec::new(),
    };

    let n_input = input_ids.len();

    // ★ 错误安全: prefill + decode 封装到 IIFE, 任何 `?` 失败都不会 drop ctx,
    //   外层 restore_ctx_state 保证 KV cache + pos 回移到 session。
    let inner_result: Result<Vec<u32>> = (|| {
        let prefill_start = std::time::Instant::now();
        if n_input == 1 {
            forward_single_token(&mut ctx, input_ids[0])?;
        } else if n_input > 1 {
            prefill_batched(&mut ctx, &input_ids, start_pos)?;
        }
        let prefill_ms = prefill_start.elapsed().as_millis();

        let mut rng = LcgRng::new(0xC0FFEE);
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut sampling_buf = SamplingBuffers::new(ctx.logits_buf.len());
        let decode_start = std::time::Instant::now();
        // 增量解码缓冲: 跨 token 的不完整 UTF-8 字节暂存于此。
        // 每 token 只追加本 token 的 bytes 并对齐 UTF-8 边界, 不再重解整个历史 (O(N) 而非 O(N²))。
        let mut pending_bytes: Vec<u8> = Vec::new();
        let mut broke = false;

        for _step in 0..max_tokens {
            let next_id = sample_top_k_top_p_into(
                &ctx.logits_buf, params, &mut || rng.next_f32(), &mut sampling_buf,
            );
            if next_id as u32 == cfg.eos_token_id {
                break;
            }
            generated_ids.push(next_id as u32);
            forward_single_token(&mut ctx, next_id as u32)?;

            // 增量解码: 仅追加本 token 的 bytes, 再对齐 UTF-8 边界输出 delta
            decode_token_bytes(tokenizer, next_id as u32, &mut pending_bytes);
            let delta = drain_complete_utf8(&mut pending_bytes);
            if !delta.is_empty() {
                if !on_delta(&delta) {
                    broke = true;
                    break;
                }
            }
        }
        // 收尾: 未提前中断时, 把残留的不完整字节以 lossy 形式上报
        // (与旧实现最终 from_utf8_lossy 一致; 完整文本通常无残留)
        if !broke && !pending_bytes.is_empty() {
            let tail = String::from_utf8_lossy(&pending_bytes);
            if !tail.is_empty() {
                on_delta(&tail);
            }
        }

        let decode_ms = decode_start.elapsed().as_millis();
        let n_gen = generated_ids.len();
        eprintln!("[bench] session_reply: prefill({n_input}t)={prefill_ms}ms, decode({n_gen}t)={decode_ms}ms (~{}ms/tok ~{:.2} tok/s)",
            if n_gen > 0 { decode_ms / n_gen as u128 } else { 0 },
            if decode_ms > 0 { n_gen as f64 * 1000.0 / decode_ms as f64 } else { 0.0 });

        Ok(generated_ids)
    })();

    // 无论成功失败都移回状态 (关键: 避免 KV cache 丢失)
    restore_ctx_state(session, &mut ctx);

    let generated_ids = inner_result?;
    session.history_tokens.extend(&input_ids);
    session.history_tokens.extend(&generated_ids);

    crate::session_persist::dump_tokens_if_enabled(&input_ids, &generated_ids)?;
    Ok(tokenizer.decode(&generated_ids))
}

/// 多模态对话: 带 vision 注入的 session_reply
///
/// 与 session_reply 的区别:
/// - prefill 阶段用 forward_batch_with_vision, image_token 位置展开为 vision embeddings
/// - 调用方需提供 vision_ctx (从 engine.vision 借用)
/// - 消费 session.pending_images 并清空
///
/// `vision_embeddings`: 扁平 [n_total_patch * hidden] 行优先
/// `n_vision_per_image`: 单张图展开后的 patch 数
/// `image_token_id`: image_token 的 token id
pub fn session_reply_with_vision(
    cfg: &Config,
    weights: &LoadedWeights,
    tokenizer: &BpeTokenizer,
    session: &mut Session,
    user_msg: &str,
    max_tokens: usize,
    params: SamplingParams,
    vision_embeddings: &[f32],
    n_vision_per_image: usize,
    image_token_id: u32,
) -> Result<String> {
    // 文本路径无进度回调 (兼容旧调用方)
    session_reply_with_vision_inner(
        cfg, weights, tokenizer, session, user_msg, max_tokens, params,
        vision_embeddings, n_vision_per_image, image_token_id,
        None, None,
    )
}

/// 流式版 session_reply_with_vision: 支持 on_delta 增量文本 + on_progress 进度上报
///
/// 与 session_reply_with_vision 的区别:
/// - decode 循环中每步通过 on_delta 上报增量文本 (UTF-8 边界对齐)
/// - prefill 阶段每批 forward_batch 完成后通过 on_progress 上报 (done_tokens, total_tokens)
pub fn session_reply_with_vision_stream(
    cfg: &Config,
    weights: &LoadedWeights,
    tokenizer: &BpeTokenizer,
    session: &mut Session,
    user_msg: &str,
    max_tokens: usize,
    params: SamplingParams,
    vision_embeddings: &[f32],
    n_vision_per_image: usize,
    image_token_id: u32,
    on_delta: &mut dyn FnMut(&str) -> bool,
    on_progress: &mut dyn FnMut(crate::engine::ProgressEvent),
) -> Result<String> {
    use crate::engine::ProgressEvent;
    session_reply_with_vision_inner(
        cfg, weights, tokenizer, session, user_msg, max_tokens, params,
        vision_embeddings, n_vision_per_image, image_token_id,
        Some(on_delta),
        Some(on_progress as &mut dyn FnMut(ProgressEvent)),
    )
}

fn session_reply_with_vision_inner(
    cfg: &Config,
    weights: &LoadedWeights,
    tokenizer: &BpeTokenizer,
    session: &mut Session,
    user_msg: &str,
    max_tokens: usize,
    params: SamplingParams,
    vision_embeddings: &[f32],
    n_vision_per_image: usize,
    image_token_id: u32,
    mut on_delta: Option<&mut dyn FnMut(&str) -> bool>,
    on_progress: Option<&mut dyn FnMut(crate::engine::ProgressEvent)>,
) -> Result<String> {
    use daiza_engine::model::forward::{forward_batch_with_vision, VisionInject};
    use crate::engine::ProgressEvent;

    // 1. 构造带 image_token 占位符的增量文本
    // n_images 从 vision_embeddings 反推 (engine.rs 已 drain pending_images, 不能依赖其长度)
    let hidden = cfg.hidden;
    let n_images = if n_vision_per_image > 0 && hidden > 0 {
        vision_embeddings.len() / (n_vision_per_image * hidden)
    } else { 0 };
    let image_token_str = tokenizer.vocab.tokens.get(image_token_id as usize)
        .cloned().unwrap_or_else(|| "<|image_pad|>".to_string());
    let mut image_section = String::new();
    for _ in 0..n_images {
        image_section.push_str(&image_token_str);
    }

    // 构造增量: 首轮带 sys prompt, 后续轮补 <|im_end|>\n
    let mut increment = String::new();
    if session.history_tokens.is_empty() {
        if let Some(sys) = &session.system_prompt {
            increment.push_str("<|im_start|>system\n");
            increment.push_str(sys);
            increment.push_str("<|im_end|>\n");
        }
    } else {
        increment.push_str("<|im_end|>\n");
    }
    increment.push_str("<|im_start|>user\n");
    increment.push_str(&image_section);
    increment.push_str(user_msg);
    increment.push_str("<|im_end|>\n");
    increment.push_str("<|im_start|>assistant\n");
    if session.think_enabled {
        // <think>\n (换行符) — <think> special token + \n 普通 token
        increment.push_str("<think>\n");
    } else {
        // ★ think 关闭: 追加空 think 块 <think></think>\n (与普通 session_reply 一致)
        //   避免模型自发生成 <think>...长篇think内容</think> 浪费算力
        increment.push_str("<think></think>\n");
    }

    let input_ids = tokenizer.encode(&increment);
    if input_ids.is_empty() {
        return Err(crate::BonsaiError::Tokenizer(
            "vision increment encode returned empty".into(),
        ));
    }

    // 2. 构造临时 ForwardContext
    let mut ctx = ForwardContext {
        cfg,
        weights,
        state: std::mem::take(&mut session.state),
        h_buf: std::mem::take(&mut session.h_buf),
        workspace: std::mem::take(&mut session.workspace),
        logits_buf: std::mem::take(&mut session.logits_buf),
        cos_buf: std::mem::take(&mut session.cos_buf),
        sin_buf: std::mem::take(&mut session.sin_buf),
        hidden_tap_buf: Vec::new(),
        hidden_tap_layers: Vec::new(),
        hidden_tap_batch_buf: Vec::new(),
    };

    // 3-4. prefill + decode 封装到 IIFE
    // ★ 错误安全: vision prefill 中 flush_text/flush_vision 的 `?` 失败, 或 decode 中
    //   forward_single_token 的 `?` 失败, 都会让闭包返回 Err, 但 ctx 不会被 drop,
    //   外层 restore_ctx_state 保证 KV cache + pos 回移到 session。
    //   这对 vision 模式尤其重要: vision embeddings 已注入 KV cache, 若错误路径丢失,
    //   下一轮普通模式将完全失去图片上下文 (用户反馈"每次对话都是独立的"的根因之一)。
    let inner_result: Result<Vec<u32>> = (|| {
        // 3. prefill: text + vision 混合注入 + 进度上报
        //    总 token 数 = input_ids.len() + n_images * (n_vision_per_image - 1)
        //    (每个 image_token 占位符展开为 n_vision_per_image 个 vision token)
        let total_prefill_tokens = input_ids.len() + n_images * n_vision_per_image.saturating_sub(1);
        let mut done_prefill_tokens = 0usize;
        const MAX_TEXT_BATCH: usize = 32;
        const MAX_VISION_BATCH: usize = 64;
        let mut text_batch: Vec<u32> = Vec::with_capacity(MAX_TEXT_BATCH);
        let mut vision_offset = 0usize;

        // flush_text + flush_vision: 内联闭包, 完成后上报 prefill 进度
        // (text batch: +batch.len(); vision batch: +bs)
        let flush_text = |batch: &mut Vec<u32>, ctx: &mut ForwardContext<'_>,
                          done: &mut usize, prog: &mut Option<&mut dyn FnMut(ProgressEvent)>| -> crate::Result<()> {
            if batch.is_empty() { return Ok(()); }
            let n = batch.len();
            if n == 1 {
                forward_single_token(ctx, batch[0])?;
            } else {
                forward_batch(ctx, batch, ctx.state.pos, None)?;
            }
            batch.clear();
            *done += n;
            if let Some(cb) = prog.as_deref_mut() {
                cb(ProgressEvent::Prefill { done_tokens: *done, total_tokens: total_prefill_tokens });
            }
            Ok(())
        };
        let flush_vision = |ctx: &mut ForwardContext<'_>, emb: &[f32],
                            done: &mut usize, prog: &mut Option<&mut dyn FnMut(ProgressEvent)>| -> crate::Result<()> {
            let n = emb.len() / hidden;
            debug_assert_eq!(emb.len(), n * hidden);
            let token_ids: Vec<u32> = vec![image_token_id; n];
            let inject = VisionInject {
                image_token_id,
                vision_embeddings: emb,
                n_vision_per_image: 1,
            };
            forward_batch_with_vision(ctx, &token_ids, ctx.state.pos, None, Some(inject), None)?;
            *done += n;
            if let Some(cb) = prog.as_deref_mut() {
                cb(ProgressEvent::Prefill { done_tokens: *done, total_tokens: total_prefill_tokens });
            }
            Ok(())
        };

        let mut prog = on_progress;
        for &tid in &input_ids {
            if tid == image_token_id {
                flush_text(&mut text_batch, &mut ctx, &mut done_prefill_tokens, &mut prog)?;
                let mut vi = 0;
                while vi < n_vision_per_image {
                    let bs = MAX_VISION_BATCH.min(n_vision_per_image - vi);
                    let s = (vision_offset + vi) * hidden;
                    let e = s + bs * hidden;
                    flush_vision(&mut ctx, &vision_embeddings[s..e], &mut done_prefill_tokens, &mut prog)?;
                    vi += bs;
                }
                vision_offset += n_vision_per_image;
            } else {
                text_batch.push(tid);
                if text_batch.len() >= MAX_TEXT_BATCH {
                    flush_text(&mut text_batch, &mut ctx, &mut done_prefill_tokens, &mut prog)?;
                }
            }
        }
        flush_text(&mut text_batch, &mut ctx, &mut done_prefill_tokens, &mut prog)?;

        // 4. decode 循环 + 增量回调
        let mut rng = LcgRng::new(0xC0FFEE);
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut sampling_buf = SamplingBuffers::new(ctx.logits_buf.len());
        let mut pending_bytes: Vec<u8> = Vec::new();
        let mut broke = false;

        for _step in 0..max_tokens {
            let next_id = sample_top_k_top_p_into(
                &ctx.logits_buf, params, &mut || rng.next_f32(), &mut sampling_buf,
            );
            if next_id as u32 == cfg.eos_token_id {
                break;
            }
            generated_ids.push(next_id as u32);
            forward_single_token(&mut ctx, next_id as u32)?;

            if let Some(cb) = on_delta.as_deref_mut() {
                decode_token_bytes(tokenizer, next_id as u32, &mut pending_bytes);
                let delta = drain_complete_utf8(&mut pending_bytes);
                if !delta.is_empty() && !cb(&delta) {
                    broke = true;
                    break;
                }
            }
        }
        if !broke {
            if let Some(cb) = on_delta.as_deref_mut() {
                if !pending_bytes.is_empty() {
                    let tail = String::from_utf8_lossy(&pending_bytes);
                    if !tail.is_empty() { cb(&tail); }
                }
            }
        }
        Ok(generated_ids)
    })();

    // 5. state/buffers 移回 session (无论成功失败都执行, 避免 KV cache 丢失)
    restore_ctx_state(session, &mut ctx);

    let generated_ids = inner_result?;
    session.history_tokens.extend(&input_ids);
    session.history_tokens.extend(&generated_ids);

    // 6. 消费 pending_images (engine.rs 已 drain, 这里 clear 兜底)
    session.pending_images.clear();

    crate::session_persist::dump_tokens_if_enabled(&input_ids, &generated_ids)?;
    Ok(tokenizer.decode(&generated_ids))
}

/// tool_call 模式: 完整 messages 重新渲染路径
///
/// 与普通 session_reply 的区别:
/// - 每次都从 0 开始 prefill 完整 messages (tool_call 需要 multi-step 完整重渲染)
///   TODO(perf): 多轮 tool_call 下每轮都从 pos=0 重 prefill 整段历史 → 跨轮 O(N²)。
///               后续可考虑增量 prefill + KV cache 复用 (需解决 tool_call 模板
///               增量拼接的复杂性, 当前为正确性优先而完整重渲染)。
/// - 不走增量 prefill (messages 历史复杂, 增量拼接易出错)
/// - 渲染 system prompt 时注入 tools 定义
/// - decode 后解析 <tool_call> 标签, 返回结构化 ToolCall 结果
#[allow(dead_code)]
fn session_reply_with_tools(
    cfg: &Config,
    weights: &LoadedWeights,
    tokenizer: &BpeTokenizer,
    session: &mut Session,
    user_msg: &str,
    max_tokens: usize,
    params: SamplingParams,
) -> Result<String> {
    // 1. 追加 user 消息到 messages 历史
    session.messages.push(ToolMessage::user(user_msg.to_string()));

    // 2. 从 0 开始完整渲染 chat 模板 (tools + messages)
    //    ★ 不走增量 prefill, 因为 tool_call multi-step 模板需要完整重渲染
    session.state.reset();
    session.history_tokens.clear();

    let chat_text = crate::tool_call::render_chat_template(
        &session.tools,
        &session.messages,
        session.system_prompt.as_deref(),
        session.think_enabled,
    );
    let input_ids = tokenizer.encode(&chat_text);
    if input_ids.is_empty() {
        return Err(crate::BonsaiError::Tokenizer(
            "tool_call chat_template encode returned empty".into(),
        ));
    }

    // 3. 构造临时 ForwardContext
    let mut ctx = ForwardContext {
        cfg,
        weights,
        state: std::mem::take(&mut session.state),
        h_buf: std::mem::take(&mut session.h_buf),
        workspace: std::mem::take(&mut session.workspace),
        logits_buf: std::mem::take(&mut session.logits_buf),
        cos_buf: std::mem::take(&mut session.cos_buf),
        sin_buf: std::mem::take(&mut session.sin_buf),
        hidden_tap_buf: Vec::new(),
        hidden_tap_layers: Vec::new(),
        hidden_tap_batch_buf: Vec::new(),
    };

    // 4. 完整 prefill (从 pos=0 开始, 分批避免 tmp[64] 越界)
    let n_input = input_ids.len();
    if n_input == 1 {
        forward_single_token(&mut ctx, input_ids[0])?;
    } else {
        prefill_batched(&mut ctx, &input_ids, 0)?;
    }

    // 5. decode 循环
    let mut rng = LcgRng::new(0xC0FFEE);
    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut sampling_buf = SamplingBuffers::new(ctx.logits_buf.len());

    for _step in 0..max_tokens {
        let next_id = sample_top_k_top_p_into(
            &ctx.logits_buf, params, &mut || rng.next_f32(), &mut sampling_buf,
        );
        if next_id as u32 == cfg.eos_token_id {
            break;
        }
        generated_ids.push(next_id as u32);
        forward_single_token(&mut ctx, next_id as u32)?;
    }

    // 6. state/buffers 移回 session
    session.state = std::mem::take(&mut ctx.state);
    session.h_buf = std::mem::take(&mut ctx.h_buf);
    session.workspace = std::mem::take(&mut ctx.workspace);
    session.logits_buf = std::mem::take(&mut ctx.logits_buf);
    session.cos_buf = std::mem::take(&mut ctx.cos_buf);
    session.sin_buf = std::mem::take(&mut ctx.sin_buf);
    session.history_tokens.extend(&input_ids);
    session.history_tokens.extend(&generated_ids);

    // 7. decode 生成的 token 为文本
    let generated_text = tokenizer.decode(&generated_ids);

    // 8. 解析 <tool_call> 标签
    let tool_calls = crate::tool_call::parse_tool_calls(&generated_text);
    if !tool_calls.is_empty() {
        // 有 tool_call: 追加 assistant 消息 (带 tool_calls)
        session.messages.push(ToolMessage::assistant_with_tool_calls(
            generated_text.clone(),
            tool_calls.clone(),
        ));
        // 返回格式化字符串, 调用方 (REPL) 可解析并执行
        let mut out = String::new();
        for tc in &tool_calls {
            out.push_str(&format!("[tool_call] function={}\n", tc.name));
            for (k, v) in &tc.arguments {
                out.push_str(&format!("  {k} = {v}\n"));
            }
            out.push('\n');
        }
        return Ok(out);
    }

    // 无 tool_call: 追加普通 assistant 消息
    session.messages.push(ToolMessage::assistant(generated_text.clone()));
    Ok(generated_text)
}

/// tool_call 模式: 注入 tool response 后继续生成
///
/// 调用方执行 tool 后, 用本函数把结果送回模型, 模型继续生成下一轮回复。
/// 本函数会:
/// 1. 把 ToolResponse 追加到 session.messages
/// 2. 从 0 开始完整 prefill + decode (与 session_reply_with_tools 一致)
///    TODO(perf): 与 session_reply_with_tools 相同的跨轮 O(N²) 重 prefill 问题;
///                每次 tool response 都重跑整段历史。后续可与 tools 路径一并优化。
pub fn session_reply_with_tool_response(
    cfg: &Config,
    weights: &LoadedWeights,
    tokenizer: &BpeTokenizer,
    session: &mut Session,
    responses: &[ToolResponse],
    max_tokens: usize,
    params: SamplingParams,
) -> Result<String> {
    // 追加 tool response 消息
    for r in responses {
        session.messages.push(ToolMessage::tool(r.clone()));
    }

    // 走与 session_reply_with_tools 相同的完整重渲染路径
    session.state.reset();
    session.history_tokens.clear();

    let chat_text = crate::tool_call::render_chat_template(
        &session.tools,
        &session.messages,
        session.system_prompt.as_deref(),
        session.think_enabled,
    );
    let input_ids = tokenizer.encode(&chat_text);
    if input_ids.is_empty() {
        return Err(crate::BonsaiError::Tokenizer(
            "tool_response chat_template encode returned empty".into(),
        ));
    }

    let mut ctx = ForwardContext {
        cfg,
        weights,
        state: std::mem::take(&mut session.state),
        h_buf: std::mem::take(&mut session.h_buf),
        workspace: std::mem::take(&mut session.workspace),
        logits_buf: std::mem::take(&mut session.logits_buf),
        cos_buf: std::mem::take(&mut session.cos_buf),
        sin_buf: std::mem::take(&mut session.sin_buf),
        hidden_tap_buf: Vec::new(),
        hidden_tap_layers: Vec::new(),
        hidden_tap_batch_buf: Vec::new(),
    };

    let n_input = input_ids.len();
    if n_input == 1 {
        forward_single_token(&mut ctx, input_ids[0])?;
    } else {
        prefill_batched(&mut ctx, &input_ids, 0)?;
    }

    let mut rng = LcgRng::new(0xC0FFEE);
    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut sampling_buf = SamplingBuffers::new(ctx.logits_buf.len());

    for _step in 0..max_tokens {
        let next_id = sample_top_k_top_p_into(
            &ctx.logits_buf, params, &mut || rng.next_f32(), &mut sampling_buf,
        );
        if next_id as u32 == cfg.eos_token_id {
            break;
        }
        generated_ids.push(next_id as u32);
        forward_single_token(&mut ctx, next_id as u32)?;
    }

    session.state = std::mem::take(&mut ctx.state);
    session.h_buf = std::mem::take(&mut ctx.h_buf);
    session.workspace = std::mem::take(&mut ctx.workspace);
    session.logits_buf = std::mem::take(&mut ctx.logits_buf);
    session.cos_buf = std::mem::take(&mut ctx.cos_buf);
    session.sin_buf = std::mem::take(&mut ctx.sin_buf);
    session.history_tokens.extend(&input_ids);
    session.history_tokens.extend(&generated_ids);

    let generated_text = tokenizer.decode(&generated_ids);

    // 解析 <tool_call>: 可能继续调 tool, 也可能直接回复
    let tool_calls = crate::tool_call::parse_tool_calls(&generated_text);
    if !tool_calls.is_empty() {
        session.messages.push(ToolMessage::assistant_with_tool_calls(
            generated_text.clone(),
            tool_calls.clone(),
        ));
        let mut out = String::new();
        for tc in &tool_calls {
            out.push_str(&format!("[tool_call] function={}\n", tc.name));
            for (k, v) in &tc.arguments {
                out.push_str(&format!("  {k} = {v}\n"));
            }
            out.push('\n');
        }
        return Ok(out);
    }

    session.messages.push(ToolMessage::assistant(generated_text.clone()));
    Ok(generated_text)
}

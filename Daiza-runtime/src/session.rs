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
        }
    }

    /// 重置会话(KV/SSM 全部清零, pos 归零, 历史清空)
    pub fn reset(&mut self) {
        self.state.reset();
        self.history_tokens.clear();
        self.pending_images.clear();
        self.messages.clear();
    }
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
        // 首轮:带 system prompt
        if let Some(sys) = &session.system_prompt {
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
        // <think>\n 作为 byte sequence (memory 约束: 3c 74 68 69 6e 6b 3e 5c 6e)
        // tokenizer 会编码成正确的 special token
        s.push_str("<think>\n");
    }
    s
}

/// 分批 prefill: forward_batch 内部 tmp 数组限制 n_batch ≤ 64,
/// 超过时需分批调用, 每批 ≤ MAX_PREFILL_BATCH 个 token, 从 ctx.state.pos 继续。
const MAX_PREFILL_BATCH: usize = 64;

fn prefill_batched(ctx: &mut ForwardContext<'_>, input_ids: &[u32], start_pos: usize) -> Result<()> {
    let n = input_ids.len();
    if n == 0 {
        return Ok(());
    }
    if n <= MAX_PREFILL_BATCH {
        return forward_batch(ctx, input_ids, start_pos, None);
    }
    // 分批: 每批 MAX_PREFILL_BATCH 个 token, pos 自动累加
    let mut offset = 0usize;
    while offset < n {
        let end = (offset + MAX_PREFILL_BATCH).min(n);
        let batch = &input_ids[offset..end];
        forward_batch(ctx, batch, start_pos + offset, None)?;
        offset = end;
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
    // tool_call 模式: 走完整 messages 重新渲染路径
    if !session.tools.is_empty() {
        return session_reply_with_tools(cfg, weights, tokenizer, session, user_msg, max_tokens, params);
    }

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

    // 3. 增量 prefill(从 start_pos 继续, KV/SSM 已持有历史)
    let n_input = input_ids.len();
    if n_input == 1 {
        forward_single_token(&mut ctx, input_ids[0])?;
    } else if n_input > 1 {
        prefill_batched(&mut ctx, &input_ids, start_pos)?;
    }

    // 4. decode 循环
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

    // 5. state/buffers 移回 session(关键:不 drop!)
    session.state = std::mem::take(&mut ctx.state);
    session.h_buf = std::mem::take(&mut ctx.h_buf);
    session.workspace = std::mem::take(&mut ctx.workspace);
    session.logits_buf = std::mem::take(&mut ctx.logits_buf);
    session.cos_buf = std::mem::take(&mut ctx.cos_buf);
    session.sin_buf = std::mem::take(&mut ctx.sin_buf);
    session.history_tokens.extend(&input_ids);
    session.history_tokens.extend(&generated_ids);

    // 6. 正确性验证: dump token IDs (env DAIZA_DUMP_TOKENS=path)
    crate::session_persist::dump_tokens_if_enabled(&input_ids, &generated_ids)?;

    // 7. decode token ids 为字符串
    Ok(tokenizer.decode(&generated_ids))
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
    // tool_call 模式: 不支持流式 (需要完整文本解析 <tool_call>), 退化为一次性回调
    if !session.tools.is_empty() {
        let out = session_reply_with_tools(cfg, weights, tokenizer, session, user_msg, max_tokens, params)?;
        on_delta(&out);
        return Ok(out);
    }

    let increment = build_increment(session, user_msg);
    let input_ids = tokenizer.encode(&increment);
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
    if n_input == 1 {
        forward_single_token(&mut ctx, input_ids[0])?;
    } else if n_input > 1 {
        prefill_batched(&mut ctx, &input_ids, start_pos)?;
    }

    let mut rng = LcgRng::new(0xC0FFEE);
    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut sampling_buf = SamplingBuffers::new(ctx.logits_buf.len());
    let mut emitted_len = 0usize; // 已上报的文本字节数

    for _step in 0..max_tokens {
        let next_id = sample_top_k_top_p_into(
            &ctx.logits_buf, params, &mut || rng.next_f32(), &mut sampling_buf,
        );
        if next_id as u32 == cfg.eos_token_id {
            break;
        }
        generated_ids.push(next_id as u32);
        forward_single_token(&mut ctx, next_id as u32)?;

        // 增量解码: 只上报新增的字节 (from_utf8_lossy 对跨 token 的
        // 多字节字符会暂存半个字符, 等下一个 token 补齐后自然上报完整字符)
        let full = tokenizer.decode(&generated_ids);
        if full.len() > emitted_len {
            let delta = &full[emitted_len..];
            emitted_len = full.len();
            if !on_delta(delta) {
                break;
            }
        }
    }

    session.state = std::mem::take(&mut ctx.state);
    session.h_buf = std::mem::take(&mut ctx.h_buf);
    session.workspace = std::mem::take(&mut ctx.workspace);
    session.logits_buf = std::mem::take(&mut ctx.logits_buf);
    session.cos_buf = std::mem::take(&mut ctx.cos_buf);
    session.sin_buf = std::mem::take(&mut ctx.sin_buf);
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
    use daiza_engine::model::forward::{forward_batch_with_vision, VisionInject};

    // 1. 构造带 image_token 占位符的增量文本
    let image_token_str = tokenizer.vocab.tokens.get(image_token_id as usize)
        .cloned().unwrap_or_else(|| "<|image_pad|>".to_string());
    let mut image_section = String::new();
    for _ in 0..session.pending_images.len() {
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
        increment.push_str("<think>\n");
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

    // 3. prefill: text + vision 混合注入
    //    策略: 逐 token 扫描, image_token 位置批量注入 vision embeddings,
    //    其他 token 累积成 text batch 走 forward_batch
    let hidden = cfg.hidden;
    const MAX_TEXT_BATCH: usize = 32;
    const MAX_VISION_BATCH: usize = 64;
    let mut text_batch: Vec<u32> = Vec::with_capacity(MAX_TEXT_BATCH);
    let mut vision_offset = 0usize;

    let flush_text = |batch: &mut Vec<u32>, ctx: &mut ForwardContext<'_>| -> crate::Result<()> {
        if batch.is_empty() { return Ok(()); }
        if batch.len() == 1 {
            forward_single_token(ctx, batch[0])?;
        } else {
            forward_batch(ctx, batch, ctx.state.pos, None)?;
        }
        batch.clear();
        Ok(())
    };
    let flush_vision = |ctx: &mut ForwardContext<'_>, emb: &[f32]| -> crate::Result<()> {
        let n = emb.len() / hidden;
        debug_assert_eq!(emb.len(), n * hidden);
        let token_ids: Vec<u32> = vec![image_token_id; n];
        let inject = VisionInject {
            image_token_id,
            vision_embeddings: emb,
            n_vision_per_image: 1,
        };
        forward_batch_with_vision(ctx, &token_ids, ctx.state.pos, None, Some(inject))
    };

    for &tid in &input_ids {
        if tid == image_token_id {
            flush_text(&mut text_batch, &mut ctx)?;
            let mut vi = 0;
            while vi < n_vision_per_image {
                let bs = MAX_VISION_BATCH.min(n_vision_per_image - vi);
                let s = (vision_offset + vi) * hidden;
                let e = s + bs * hidden;
                flush_vision(&mut ctx, &vision_embeddings[s..e])?;
                vi += bs;
            }
            vision_offset += n_vision_per_image;
        } else {
            text_batch.push(tid);
            if text_batch.len() >= MAX_TEXT_BATCH {
                flush_text(&mut text_batch, &mut ctx)?;
            }
        }
    }
    flush_text(&mut text_batch, &mut ctx)?;

    // 4. decode 循环 (与 session_reply 一致)
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

    // 5. state/buffers 移回 session
    session.state = std::mem::take(&mut ctx.state);
    session.h_buf = std::mem::take(&mut ctx.h_buf);
    session.workspace = std::mem::take(&mut ctx.workspace);
    session.logits_buf = std::mem::take(&mut ctx.logits_buf);
    session.cos_buf = std::mem::take(&mut ctx.cos_buf);
    session.sin_buf = std::mem::take(&mut ctx.sin_buf);
    session.history_tokens.extend(&input_ids);
    session.history_tokens.extend(&generated_ids);

    // 6. 消费 pending_images
    session.pending_images.clear();

    crate::session_persist::dump_tokens_if_enabled(&input_ids, &generated_ids)?;
    Ok(tokenizer.decode(&generated_ids))
}

/// tool_call 模式: 完整 messages 重新渲染路径
///
/// 与普通 session_reply 的区别:
/// - 每次都从 0 开始 prefill 完整 messages (tool_call 需要 multi-step 完整重渲染)
/// - 不走增量 prefill (messages 历史复杂, 增量拼接易出错)
/// - 渲染 system prompt 时注入 tools 定义
/// - decode 后解析 <tool_call> 标签, 返回结构化 ToolCall 结果
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

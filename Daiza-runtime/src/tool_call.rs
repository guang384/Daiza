//! 工具调用 (tool calling) 支持
//!
//! 基于 GGUF chat_template 提取的 Qwen3 tool_call 格式:
//!
//! ```text
//! <tool_call>
//! <function=function_name>
//! <parameter=param1>
//! value1
//! </parameter>
//! <parameter=param2>
//! value2
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! 模板渲染规则 (从 GGUF chat_template 提取):
//! - tools 非空时, system prompt 注入 `<tools>...</tools>` 块 + 格式说明
//! - assistant 消息若带 tool_calls, 渲染为 `<tool_call>...` 块
//! - tool role 消息渲染为 `<tool_response>content</tool_response>`

use std::collections::BTreeMap;

/// 工具定义 (调用方注册)
#[derive(Clone, Debug)]
pub struct ToolDef {
    /// 函数名
    pub name: String,
    /// 函数描述
    pub description: String,
    /// 参数 JSON schema (简化为 BTreeMap<参数名, 类型描述>)
    pub parameters: Vec<ToolParam>,
}

#[derive(Clone, Debug)]
pub struct ToolParam {
    pub name: String,
    pub description: String,
    pub param_type: String,
    pub required: bool,
}

/// 一次工具调用 (从模型输出解析)
#[derive(Clone, Debug)]
pub struct ToolCall {
    pub name: String,
    /// 参数 (有序, 保留模型输出的顺序)
    pub arguments: BTreeMap<String, String>,
}

/// 工具响应 (调用方执行后回送)
#[derive(Clone, Debug)]
pub struct ToolResponse {
    /// 对应的 tool_call name (用于关联, 模型模板不使用此字段)
    pub name: String,
    /// 执行结果文本
    pub content: String,
}

/// 历史消息 (用于 tool_call 模式的完整 messages 渲染)
#[derive(Clone, Debug)]
pub struct ToolMessage {
    pub role: MessageRole,
    /// 文本内容 (user/assistant 的普通文本, 或 tool role 的 response content)
    pub content: String,
    /// 若为 assistant 且发起 tool_call, 这里存解析出的 calls
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ToolMessage {
    pub fn user(content: String) -> Self {
        Self { role: MessageRole::User, content, tool_calls: Vec::new() }
    }
    pub fn assistant(content: String) -> Self {
        Self { role: MessageRole::Assistant, content, tool_calls: Vec::new() }
    }
    pub fn assistant_with_tool_calls(content: String, tool_calls: Vec<ToolCall>) -> Self {
        Self { role: MessageRole::Assistant, content, tool_calls }
    }
    pub fn tool(response: ToolResponse) -> Self {
        Self {
            role: MessageRole::Tool,
            content: response.content,
            tool_calls: Vec::new(),
        }
    }
}

/// 渲染完整 chat 模板 (含 tools + messages)
///
/// 简化版 Jinja2 模板, 严格按 GGUF chat_template 提取的格式拼接。
///
/// - `tools`: 工具列表 (空则不注入 tools 块)
/// - `messages`: 完整对话历史
/// - `system_prompt`: 可选系统提示词 (与 tools 互斥, tools 优先)
/// - `think_enabled`: 是否在最后 assistant 消息后追加 `<think>\n`
pub fn render_chat_template(
    tools: &[ToolDef],
    messages: &[ToolMessage],
    system_prompt: Option<&str>,
    think_enabled: bool,
) -> String {
    let mut s = String::new();

    // 1. system 块: tools 优先, 否则用 system_prompt
    if !tools.is_empty() {
        s.push_str("<|im_start|>system\n");
        s.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
        for tool in tools {
            s.push_str(&format_tool_def(tool));
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
        // tools 模式下若有 system_prompt, 追加在 tools 块后
        if let Some(sys) = system_prompt {
            s.push_str(sys);
            s.push('\n');
        }
        s.push_str("<|im_end|>\n");
    } else if let Some(sys) = system_prompt {
        s.push_str("<|im_start|>system\n");
        s.push_str(sys);
        s.push_str("<|im_end|>\n");
    }

    // 2. 逐消息渲染
    let n = messages.len();
    for (i, msg) in messages.iter().enumerate() {
        match msg.role {
            MessageRole::System => {
                // 已在前面处理, 跳过
            }
            MessageRole::User => {
                s.push_str("<|im_start|>user\n");
                s.push_str(&msg.content);
                s.push_str("<|im_end|>\n");
            }
            MessageRole::Assistant => {
                s.push_str("<|im_start|>assistant\n");
                s.push_str(&msg.content);
                // 若有 tool_calls, 渲染 (content 末尾应已包含模型输出的 <tool_call> 文本)
                // 这里不重复渲染 (parse_tool_calls 已在 content 中)
                s.push_str("<|im_end|>\n");
            }
            MessageRole::Tool => {
                // tool role 消息: <tool_response>content</tool_response>
                s.push_str("<|im_start|>user\n<tool_response>\n");
                s.push_str(&msg.content);
                s.push_str("\n</tool_response><|im_end|>\n");
            }
        }
        // 最后一条 assistant 消息: 追加 <think>\n
        if i == n - 1 && msg.role == MessageRole::User && think_enabled {
            // 注意: 上面 user 消息后紧接 assistant, 这里处理的是 messages 末尾是 user 的情况
            // 实际上 chat_template 末尾是 <|im_start|>assistant\n + <think>\n
        }
    }

    // 3. 末尾: 若最后一条是 user 或 tool (tool 渲染为 user 格式), 追加 assistant 开头
    if matches!(messages.last().map(|m| &m.role), Some(MessageRole::User) | Some(MessageRole::Tool)) {
        s.push_str("<|im_start|>assistant\n");
        if think_enabled {
            s.push_str("<think>\n");
        }
    }

    s
}

/// 渲染单个 tool 定义为 JSON schema 格式
fn format_tool_def(tool: &ToolDef) -> String {
    let mut s = String::new();
    s.push_str("{\"type\": \"function\", \"function\": {");
    s.push_str(&format!("\"name\": \"{}\", ", escape_json(&tool.name)));
    s.push_str(&format!("\"description\": \"{}\", ", escape_json(&tool.description)));
    s.push_str("\"parameters\": {\"type\": \"object\", \"properties\": {");
    for (i, p) in tool.parameters.iter().enumerate() {
        if i > 0 { s.push_str(", "); }
        s.push_str(&format!("\"{}\": {{\"type\": \"{}\", \"description\": \"{}\"}}",
            escape_json(&p.name), escape_json(&p.param_type), escape_json(&p.description)));
    }
    s.push_str("}, \"required\": [");
    for (i, p) in tool.parameters.iter().enumerate() {
        if !p.required { continue; }
        if i > 0 { s.push_str(", "); }
        s.push_str(&format!("\"{}\"", escape_json(&p.name)));
    }
    s.push_str("]}}");
    s
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// 从模型输出文本中解析 <tool_call> 标签
///
/// 格式:
/// ```text
/// <tool_call>
/// <function=name>
/// <parameter=key>
/// value
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// 返回所有解析到的 tool_calls (可能多个)
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut idx = 0usize;

    while idx < text.len() {
        // 找 <tool_call>
        let start = match text[idx..].find("<tool_call>") {
            Some(s) => idx + s,
            None => break,
        };
        let content_start = start + "<tool_call>".len();
        // 找配对的 </tool_call>
        let end = match text[content_start..].find("</tool_call>") {
            Some(e) => content_start + e,
            None => break,
        };
        let block = &text[content_start..end];
        if let Some(tc) = parse_single_tool_call(block) {
            calls.push(tc);
        }
        idx = end + "</tool_call>".len();
    }
    calls
}

/// 解析单个 <tool_call>...</tool_call> 内部内容
fn parse_single_tool_call(block: &str) -> Option<ToolCall> {
    // 找 <function=NAME>
    let fn_start = block.find("<function=")? + "<function=".len();
    let fn_end = block[fn_start..].find('>')? + fn_start;
    let name = block[fn_start..fn_end].trim().to_string();

    // 找 </function>
    let fn_body_end = block.find("</function>")?;
    let body = &block[fn_end + 1..fn_body_end];

    // 解析 <parameter=key>value</parameter>
    let mut arguments = BTreeMap::new();
    let mut p_idx = 0;
    while p_idx < body.len() {
        let p_start = match body[p_idx..].find("<parameter=") {
            Some(s) => p_idx + s,
            None => break,
        };
        let key_start = p_start + "<parameter=".len();
        let key_end = match body[key_start..].find('>') {
            Some(e) => key_start + e,
            None => break,
        };
        let key = body[key_start..key_end].trim().to_string();

        let val_start = key_end + 1;
        let val_end = match body[val_start..].find("</parameter>") {
            Some(e) => val_start + e,
            None => break,
        };
        let value = body[val_start..val_end].trim().to_string();
        arguments.insert(key, value);
        p_idx = val_end + "</parameter>".len();
    }

    Some(ToolCall { name, arguments })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_tool_call() {
        let block = "<function=get_weather>\n<parameter=location>\nBeijing\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>";
        let tc = parse_single_tool_call(block).unwrap();
        assert_eq!(tc.name, "get_weather");
        assert_eq!(tc.arguments.get("location"), Some(&"Beijing".to_string()));
        assert_eq!(tc.arguments.get("unit"), Some(&"celsius".to_string()));
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let text = r#"Some reasoning text

<tool_call>
<function=get_weather>
<parameter=location>
Beijing
</parameter>
</function>
</tool_call>

<tool_call>
<function=get_time>
<parameter=timezone>
UTC
</parameter>
</function>
</tool_call>"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_time");
    }

    #[test]
    fn test_render_chat_template_basic() {
        let tools = vec![ToolDef {
            name: "get_weather".into(),
            description: "Get weather".into(),
            parameters: vec![ToolParam {
                name: "location".into(),
                description: "City name".into(),
                param_type: "string".into(),
                required: true,
            }],
        }];
        let messages = vec![ToolMessage::user("What's the weather?".into())];
        let rendered = render_chat_template(&tools, &messages, None, false);
        assert!(rendered.contains("<tools>"));
        assert!(rendered.contains("\"name\": \"get_weather\""));
        assert!(rendered.contains("<|im_start|>user\nWhat's the weather?<|im_end|>"));
        assert!(rendered.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_render_tool_response() {
        let messages = vec![
            ToolMessage::user("weather?".into()),
            ToolMessage::assistant_with_tool_calls(
                "text".into(),
                vec![ToolCall { name: "get_weather".into(), arguments: BTreeMap::new() }],
            ),
            ToolMessage::tool(ToolResponse { name: "get_weather".into(), content: "Sunny 25C".into() }),
        ];
        let rendered = render_chat_template(&[], &messages, None, false);
        assert!(rendered.contains("<tool_response>\nSunny 25C\n</tool_response>"));
    }
}

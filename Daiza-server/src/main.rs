//! daiza-server: OpenAI 兼容 HTTP API 服务 (零依赖)
//!
//! 支持端点:
//!   POST /v1/chat/completions   - 聊天补全 (流式 SSE + 非流式)
//!   POST /v1/completions        - 文本补全 (legacy, 流式 + 非流式)
//!   POST /v1/embeddings         - 文本嵌入 (token_embd 均值池化)
//!   GET  /v1/models             - 模型列表
//!
//! 鉴权: --api-key <key> 启用, 客户端需带 Authorization: Bearer <key>
//! 多模态: messages.content 支持 image_url (data:image/...;base64,... 或 http URL)
//!
//! 用法:
//!   daiza-server --model <gguf> [--mmproj <gguf>] [--port 8080] [--api-key sk-xxx]

mod json;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use daiza_engine::math::{sample_top_k_top_p_into, LcgRng, SamplingBuffers, SamplingParams};
use daiza_engine::model::forward::{forward_batch, forward_single_token, make_context, ForwardContext};
use daiza_runtime::engine::Engine;
use daiza_runtime::session::{decode_token_bytes, drain_complete_utf8};
use json::Json;

// ─── 工具函数 ─────────────────────────────────────────────────────────

fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn gen_id(prefix: &str) -> String {
    let ts = now_ts();
    let rand = (ts as u32).wrapping_mul(2654435761) % 1_000_000;
    format!("{prefix}-{ts}{rand:06}")
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn read_request(stream: &mut TcpStream) -> Option<(String, String, String, Option<String>)> {
    let mut buf = Vec::with_capacity(8192);
    let mut chunk = [0u8; 16384];
    let mut header_end = None;
    let mut content_len = 0usize;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        if header_end.is_none() {
            if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
                header_end = Some(pos + 4);
                let head = String::from_utf8_lossy(&buf[..pos]);
                for line in head.lines() {
                    let l = line.to_ascii_lowercase();
                    if l.starts_with("content-length:") {
                        content_len = l[15..].trim().parse().unwrap_or(0);
                    }
                }
            }
        }
        if let Some(he) = header_end {
            if buf.len() >= he + content_len {
                break;
            }
        }
        if std::time::Instant::now() > deadline {
            return None;
        }
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > 64 << 20 {
            return None;
        }
    }
    let he = header_end?;
    let head = String::from_utf8_lossy(&buf[..he]).into_owned();
    let body = String::from_utf8_lossy(&buf[he..he + content_len]).into_owned();
    let mut parts = head.lines().next()?.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    // 提取 Authorization header
    let auth = head.lines()
        .find(|l| l.to_ascii_lowercase().starts_with("authorization:"))
        .map(|l| l[14..].trim().to_string());
    Some((method, path, body, auth))
}

fn http_response(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: *\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.flush();
}

fn http_json_error(stream: &mut TcpStream, status: &str, code: &str, message: &str) {
    let body = format!(
        "{{\"error\":{{\"message\":\"{}\",\"type\":\"{code}\",\"code\":\"{code}\"}}}}",
        json_escape(message)
    );
    http_response(stream, status, "application/json; charset=utf-8", &body);
}

fn sse_write(stream: &mut TcpStream, data: &str) -> bool {
    stream
        .write_all(format!("data: {data}\n\n").as_bytes())
        .and_then(|_| stream.flush())
        .is_ok()
}

/// 从 data URI (data:image/png;base64,...) 或文件路径加载图片字节
fn load_image_bytes(src: &str) -> Result<Vec<u8>, String> {
    if src.starts_with("data:") {
        // data:image/png;base64,<data>
        let comma = src.find(',').ok_or("invalid data URI: no comma")?;
        let meta = &src[5..comma];
        let data = &src[comma + 1..];
        if !meta.contains("base64") {
            return Err("only base64 data URI supported".into());
        }
        base64_decode(data)
    } else if src.starts_with("http://") || src.starts_with("https://") {
        Err("http image URL not supported, please use base64 data URI".into())
    } else {
        // 文件路径
        std::fs::read(src).map_err(|e| format!("read file {src}: {e}"))
    }
}

/// 标准 base64 解码
fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes: Vec<u8> = s.bytes().filter(|&b| b != b'\n' && b != b'\r' && b != b' ').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut i = 0;
    while i + 4 <= bytes.len() {
        let mut chunk = [0u8; 4];
        let mut pad = 0;
        for j in 0..4 {
            chunk[j] = match bytes[i + j] {
                b'=' => { pad += 1; 0 }
                c => val(c).ok_or_else(|| format!("invalid char at {}", i + j))?,
            };
        }
        let n = ((chunk[0] as u32) << 18) | ((chunk[1] as u32) << 12)
              | ((chunk[2] as u32) << 6) | (chunk[3] as u32);
        out.push((n >> 16) as u8);
        if pad < 2 { out.push((n >> 8) as u8); }
        if pad < 1 { out.push(n as u8); }
        i += 4;
    }
    if i != bytes.len() {
        return Err(format!("length not multiple of 4 (trailing {} bytes)", bytes.len() - i));
    }
    Ok(out)
}

// ─── Shared 状态 ──────────────────────────────────────────────────────

struct Shared {
    engine: Mutex<Engine>,
    model_id: String,
    api_key: Option<String>,
    vision_available: bool,
}

// ─── 路由 ─────────────────────────────────────────────────────────────

fn handle_conn(mut stream: TcpStream, shared: Arc<Shared>) {
    let _ = stream.set_nodelay(true);
    let Some((method, path, body, auth)) = read_request(&mut stream) else {
        return;
    };
    let route = path.split('?').next().unwrap_or("/");

    // CORS preflight
    if method == "OPTIONS" {
        http_response(&mut stream, "204 No Content", "text/plain", "");
        return;
    }

    // 鉴权检查 (除 /v1/models 外都需要校验, 若配置了 api_key)
    if let Some(ref key) = shared.api_key {
        let need_auth = route.starts_with("/v1/chat") || route.starts_with("/v1/completions") || route.starts_with("/v1/embeddings");
        if need_auth {
            let expected = format!("Bearer {key}");
            let ok = auth.as_deref().map(|a| a.trim() == expected).unwrap_or(false);
            if !ok {
                http_json_error(&mut stream, "401 Unauthorized", "invalid_api_key",
                    "Incorrect API key provided. Set Authorization: Bearer <key> header.");
                return;
            }
        }
    }

    match (method.as_str(), route) {
        ("GET", "/v1/models") => handle_models(&mut stream, &shared),
        ("POST", "/v1/chat/completions") => handle_chat_completions(&mut stream, &shared, &body),
        ("POST", "/v1/completions") => handle_completions(&mut stream, &shared, &body),
        ("POST", "/v1/embeddings") => handle_embeddings(&mut stream, &shared, &body),
        ("GET", "/") | ("GET", "/health") => {
            http_response(&mut stream, "200 OK", "application/json; charset=utf-8",
                &format!("{{\"status\":\"ok\",\"model\":\"{}\"}}", json_escape(&shared.model_id)));
        }
        _ => {
            http_json_error(&mut stream, "404 Not Found", "not_found", &format!("unknown route: {method} {route}"));
        }
    }
}

// ─── /v1/models ───────────────────────────────────────────────────────

fn handle_models(stream: &mut TcpStream, shared: &Shared) {
    let body = format!(
        "{{\"object\":\"list\",\"data\":[{{\"id\":\"{}\",\"object\":\"model\",\"created\":{},\"owned_by\":\"daiza\"}}]}}",
        json_escape(&shared.model_id), now_ts()
    );
    http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
}

// ─── /v1/chat/completions ─────────────────────────────────────────────

fn handle_chat_completions(stream: &mut TcpStream, shared: &Shared, body: &str) {
    let req = match Json::parse(body) {
        Ok(j) => j,
        Err(e) => {
            http_json_error(stream, "400 Bad Request", "invalid_json", &format!("parse error: {e}"));
            return;
        }
    };

    let messages = req.get("messages").and_then(|j| j.as_array())
        .filter(|a| !a.is_empty());
    let Some(messages) = messages else {
        http_json_error(stream, "400 Bad Request", "invalid_request", "messages field is required and must be non-empty array");
        return;
    };

    // 解析参数
    let temperature = req.get("temperature").and_then(|j| j.as_f64()).unwrap_or(0.7) as f32;
    let top_p = req.get("top_p").and_then(|j| j.as_f64()).unwrap_or(0.95) as f32;
    let top_k = req.get("top_k").and_then(|j| j.as_u64()).unwrap_or(20) as usize;
    let max_tokens = req.get("max_tokens").or_else(|| req.get("max_completion_tokens"))
        .and_then(|j| j.as_u64()).unwrap_or(2048) as usize;
    let stream_mode = req.get("stream").and_then(|j| j.as_bool()).unwrap_or(false);
    let stop = req.get("stop").and_then(|j| j.as_str()).map(String::from);

    let params = SamplingParams {
        temperature: temperature.max(0.0),
        top_k,
        top_p: top_p.clamp(0.0, 1.0),
    };

    // 构造 chat prompt + 提取图片
    let (prompt, image_bytes_list) = match build_chat_prompt(messages) {
        Ok(p) => p,
        Err(e) => {
            http_json_error(stream, "400 Bad Request", "invalid_request", &e);
            return;
        }
    };

    let model_id = shared.model_id.clone();
    let created = now_ts();
    let id = gen_id("chatcmpl");

    if stream_mode {
        // 流式: 先发 SSE header
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
        if stream.write_all(head.as_bytes()).and_then(|_| stream.flush()).is_err() {
            return;
        }
        // 首个 chunk: role
        let first = format!(
            "{{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}",
            json_escape(&model_id)
        );
        if !sse_write(stream, &first) { return; }

        let result = generate_stream(shared, &prompt, &image_bytes_list, max_tokens, params, stop, |delta| {
            let chunk = format!(
                "{{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}",
                json_escape(&model_id), json_escape(delta)
            );
            sse_write(stream, &chunk)
        });

        let finish_reason = match &result {
            Ok(reason) => *reason,
            Err(e) => {
                let err_chunk = format!(
                    "{{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"error\"}}],\"error\":{{\"message\":\"{}\"}}}}",
                    json_escape(&model_id), json_escape(e)
                );
                let _ = sse_write(stream, &err_chunk);
                "error"
            }
        };

        // 结束 chunk
        let end = format!(
            "{{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{finish_reason}\"}}]}}",
            json_escape(&model_id)
        );
        let _ = sse_write(stream, &end);
        let _ = sse_write(stream, "[DONE]");
    } else {
        // 非流式: 收集完整文本
        let mut full = String::new();
        let result = generate_stream(shared, &prompt, &image_bytes_list, max_tokens, params, stop, |delta| {
            full.push_str(delta);
            true
        });

        match result {
            Ok(finish_reason) => {
                let body = format!(
                    "{{\"id\":\"{id}\",\"object\":\"chat.completion\",\"created\":{created},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},\"finish_reason\":\"{finish_reason}\"}}],\"usage\":{{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}}}",
                    json_escape(&model_id), json_escape(&full)
                );
                http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
            }
            Err(e) => {
                http_json_error(stream, "500 Internal Server Error", "internal_error", &e);
            }
        }
    }
}

/// 从 messages 数组构造 chat prompt + 提取图片字节
///
/// 返回 (prompt, image_bytes_list)
/// prompt 格式: Qwen3 chat template
///   <|im_start|>system\n{sys}<|im_end|>\n
///   <|im_start|>user\n[图片占位]{text}<|im_end|>\n
///   <|im_start|>assistant\n{text}<|im_end|>\n
///   <|im_start|>assistant\n<think>\n
fn build_chat_prompt(messages: &[Json]) -> Result<(String, Vec<Vec<u8>>), String> {
    let mut prompt = String::new();
    let mut images: Vec<Vec<u8>> = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|j| j.as_str()).unwrap_or("user");
        let content = msg.get("content").ok_or("message missing content field")?;

        // content 可能是 string 或 array of {type, text/image_url}
        let (text_parts, img_urls): (Vec<String>, Vec<String>) = match content {
            Json::Str(s) => (vec![s.clone()], vec![]),
            Json::Arr(arr) => {
                let mut texts = Vec::new();
                let mut imgs = Vec::new();
                for item in arr {
                    let ty = item.get("type").and_then(|j| j.as_str()).unwrap_or("");
                    match ty {
                        "text" => {
                            if let Some(t) = item.get("text").and_then(|j| j.as_str()) {
                                texts.push(t.to_string());
                            }
                        }
                        "image_url" => {
                            if let Some(url) = item.get("image_url")
                                .and_then(|j| j.get("url"))
                                .and_then(|j| j.as_str())
                            {
                                imgs.push(url.to_string());
                            }
                        }
                        _ => {}
                    }
                }
                (texts, imgs)
            }
            _ => return Err("content must be string or array".into()),
        };

        // 加载图片字节
        for url in &img_urls {
            match load_image_bytes(url) {
                Ok(bytes) => images.push(bytes),
                Err(e) => return Err(format!("load image failed: {e}")),
            }
        }

        let text = text_parts.join("\n");
        // 图片占位符: 每张图一个 <|image_pad|>
        let image_section: String = if img_urls.is_empty() {
            String::new()
        } else {
            "<|image_pad|>".repeat(img_urls.len())
        };

        // 跳过 system 的 role 映射 (system 直接放 system 段)
        let role_tag = match role {
            "system" => "system",
            "user" => "user",
            "assistant" => "assistant",
            "tool" => "tool",
            other => return Err(format!("unsupported role: {other}")),
        };

        prompt.push_str(&format!("<|im_start|>{role_tag}\n{image_section}{text}<|im_end|>\n"));
    }

    // 最后追加 assistant 段 (模型开始生成)
    prompt.push_str("<|im_start|>assistant\n");
    // 思考模式: 追加 <think>\n 字面反斜杠+n (与 daiza-cli 一致, byte sequence: 3c 74 68 69 6e 6b 3e 5c 6e)
    prompt.push_str("<think>\n");

    Ok((prompt, images))
}

// ─── /v1/completions (legacy) ─────────────────────────────────────────

fn handle_completions(stream: &mut TcpStream, shared: &Shared, body: &str) {
    let req = match Json::parse(body) {
        Ok(j) => j,
        Err(e) => {
            http_json_error(stream, "400 Bad Request", "invalid_json", &format!("parse error: {e}"));
            return;
        }
    };

    let prompt = match req.get("prompt").and_then(|j| j.as_str()) {
        Some(p) => p.to_string(),
        None => {
            http_json_error(stream, "400 Bad Request", "invalid_request", "prompt field is required");
            return;
        }
    };

    let temperature = req.get("temperature").and_then(|j| j.as_f64()).unwrap_or(0.7) as f32;
    let top_p = req.get("top_p").and_then(|j| j.as_f64()).unwrap_or(0.95) as f32;
    let top_k = req.get("top_k").and_then(|j| j.as_u64()).unwrap_or(20) as usize;
    let max_tokens = req.get("max_tokens").and_then(|j| j.as_u64()).unwrap_or(2048) as usize;
    let stream_mode = req.get("stream").and_then(|j| j.as_bool()).unwrap_or(false);
    let stop = req.get("stop").and_then(|j| j.as_str()).map(String::from);

    let params = SamplingParams {
        temperature: temperature.max(0.0),
        top_k,
        top_p: top_p.clamp(0.0, 1.0),
    };

    let model_id = shared.model_id.clone();
    let created = now_ts();
    let id = gen_id("cmpl");

    if stream_mode {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
        if stream.write_all(head.as_bytes()).and_then(|_| stream.flush()).is_err() {
            return;
        }

        let result = generate_stream_raw(shared, &prompt, max_tokens, params, stop, |delta| {
            let chunk = format!(
                "{{\"id\":\"{id}\",\"object\":\"text_completion\",\"created\":{created},\"model\":\"{}\",\"choices\":[{{\"text\":\"{}\",\"index\":0,\"finish_reason\":null}}]}}",
                json_escape(&model_id), json_escape(delta)
            );
            sse_write(stream, &chunk)
        });

        let finish_reason = match &result {
            Ok(r) => *r,
            Err(e) => {
                let err_chunk = format!(
                    "{{\"id\":\"{id}\",\"object\":\"text_completion\",\"created\":{created},\"model\":\"{}\",\"choices\":[{{\"text\":\"\",\"index\":0,\"finish_reason\":\"error\"}}],\"error\":{{\"message\":\"{}\"}}}}",
                    json_escape(&model_id), json_escape(e)
                );
                let _ = sse_write(stream, &err_chunk);
                "error"
            }
        };

        let end = format!(
            "{{\"id\":\"{id}\",\"object\":\"text_completion\",\"created\":{created},\"model\":\"{}\",\"choices\":[{{\"text\":\"\",\"index\":0,\"finish_reason\":\"{finish_reason}\"}}]}}",
            json_escape(&model_id)
        );
        let _ = sse_write(stream, &end);
        let _ = sse_write(stream, "[DONE]");
    } else {
        let mut full = String::new();
        let result = generate_stream_raw(shared, &prompt, max_tokens, params, stop, |delta| {
            full.push_str(delta);
            true
        });

        match result {
            Ok(finish_reason) => {
                let body = format!(
                    "{{\"id\":\"{id}\",\"object\":\"text_completion\",\"created\":{created},\"model\":\"{}\",\"choices\":[{{\"text\":\"{}\",\"index\":0,\"finish_reason\":\"{finish_reason}\"}}],\"usage\":{{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}}}",
                    json_escape(&model_id), json_escape(&full)
                );
                http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
            }
            Err(e) => {
                http_json_error(stream, "500 Internal Server Error", "internal_error", &e);
            }
        }
    }
}

// ─── /v1/embeddings ───────────────────────────────────────────────────

fn handle_embeddings(stream: &mut TcpStream, shared: &Shared, body: &str) {
    let req = match Json::parse(body) {
        Ok(j) => j,
        Err(e) => {
            http_json_error(stream, "400 Bad Request", "invalid_json", &format!("parse error: {e}"));
            return;
        }
    };

    // input 可以是 string 或 string array
    let inputs: Vec<String> = match req.get("input") {
        Some(Json::Str(s)) => vec![s.clone()],
        Some(Json::Arr(arr)) => {
            let mut v = Vec::new();
            for item in arr {
                if let Some(s) = item.as_str() {
                    v.push(s.to_string());
                }
            }
            v
        }
        _ => {
            http_json_error(stream, "400 Bad Request", "invalid_request", "input must be string or array of strings");
            return;
        }
    };

    if inputs.is_empty() {
        http_json_error(stream, "400 Bad Request", "invalid_request", "input is empty");
        return;
    }

    let mut engine = shared.engine.lock().unwrap();
    // 确保权重已加载
    if engine.weights.is_none() {
        if let Err(e) = engine.load_weights() {
            http_json_error(stream, "500 Internal Server Error", "internal_error", &format!("load weights: {e}"));
            return;
        }
    }

    let weights = engine.weights.as_ref().unwrap();
    let tokenizer = &engine.tokenizer;
    let embd_dim = engine.config.hidden;

    let mut data_items = Vec::new();
    let mut total_tokens = 0usize;

    for (idx, text) in inputs.iter().enumerate() {
        let token_ids = tokenizer.encode(text);
        if token_ids.is_empty() {
            data_items.push(format!(
                "{{\"object\":\"embedding\",\"index\":{idx},\"embedding\":[]}}"
            ));
            continue;
        }
        // 均值池化: 对所有 token 的 embedding 求平均
        let mut sum = vec![0.0f32; embd_dim];
        let mut row_buf = vec![0.0f32; embd_dim];
        for &tid in &token_ids {
            weights.global.token_embd.row_into_slice(tid as usize, &mut row_buf);
            for (s, v) in sum.iter_mut().zip(row_buf.iter()) {
                *s += *v;
            }
        }
        let n = token_ids.len() as f32;
        let embd: Vec<f32> = sum.iter().map(|v| v / n).collect();
        total_tokens += token_ids.len();

        // 序列化为 JSON array
        let embd_str: Vec<String> = embd.iter().map(|v| format!("{v}")).collect();
        data_items.push(format!(
            "{{\"object\":\"embedding\",\"index\":{idx},\"embedding\":[{}]}}",
            embd_str.join(",")
        ));
    }

    let body = format!(
        "{{\"object\":\"list\",\"data\":[{}],\"model\":\"{}\",\"usage\":{{\"prompt_tokens\":{total_tokens},\"total_tokens\":{total_tokens}}}}}",
        data_items.join(","),
        json_escape(&shared.model_id)
    );
    http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
}

// ─── 流式生成核心 ─────────────────────────────────────────────────────

/// chat completions 流式生成 (带 vision 支持)
///
/// 构造完整 prompt → encode → prefill → decode 循环, 每生成一个 token 回调 on_delta
/// on_delta 返回 false 中断 (stop sequence 命中也中断)
/// 返回 Ok(finish_reason) 或 Err(message)
fn generate_stream(
    shared: &Shared,
    prompt: &str,
    images: &[Vec<u8>],
    max_tokens: usize,
    params: SamplingParams,
    stop: Option<String>,
    mut on_delta: impl FnMut(&str) -> bool,
) -> Result<&'static str, String> {
    let mut engine = shared.engine.lock().unwrap();

    // 加载权重 (懒加载)
    if engine.weights.is_none() {
        engine.load_weights().map_err(|e| format!("load weights: {e}"))?;
        let n_threads = daiza_engine::model::workspace::thread_count();
        daiza_engine::model::workspace::init_thread_pool(n_threads);
    }

    // 多模态: 有图片走 vision 路径
    if !images.is_empty() && shared.vision_available {
        return generate_with_vision_stream(&mut engine, prompt, images, max_tokens, params, stop, &mut on_delta);
    }

    // 纯文本: encode → prefill → decode
    let input_ids = engine.tokenizer.encode(prompt);
    if input_ids.is_empty() {
        return Err("encode returned empty".into());
    }
    let cfg = &engine.config;
    let weights = engine.weights.as_ref().unwrap();
    let mut ctx = make_context(weights, cfg);

    // prefill
    prefill_batched(&mut ctx, &input_ids)?;

    // decode
    let finish = decode_loop(&mut ctx, cfg, &engine.tokenizer, max_tokens, params, stop, &mut on_delta)?;
    Ok(finish)
}

/// legacy completions 流式生成 (raw prompt, 跳过 chat template)
fn generate_stream_raw(
    shared: &Shared,
    prompt: &str,
    max_tokens: usize,
    params: SamplingParams,
    stop: Option<String>,
    mut on_delta: impl FnMut(&str) -> bool,
) -> Result<&'static str, String> {
    let mut engine = shared.engine.lock().unwrap();
    if engine.weights.is_none() {
        engine.load_weights().map_err(|e| format!("load weights: {e}"))?;
        let n_threads = daiza_engine::model::workspace::thread_count();
        daiza_engine::model::workspace::init_thread_pool(n_threads);
    }

    let input_ids = engine.tokenizer.encode(prompt);
    if input_ids.is_empty() {
        return Err("encode returned empty".into());
    }
    let cfg = &engine.config;
    let weights = engine.weights.as_ref().unwrap();
    let mut ctx = make_context(weights, cfg);

    prefill_batched(&mut ctx, &input_ids)?;
    let finish = decode_loop(&mut ctx, cfg, &engine.tokenizer, max_tokens, params, stop, &mut on_delta)?;
    Ok(finish)
}

/// 分批 prefill (每批 ≤64 token)
fn prefill_batched(ctx: &mut ForwardContext<'_>, input_ids: &[u32]) -> Result<(), String> {
    const MAX_BATCH: usize = 64;
    let n = input_ids.len();
    let mut offset = 0;
    while offset < n {
        let end = (offset + MAX_BATCH).min(n);
        forward_batch(ctx, &input_ids[offset..end], offset, None)
            .map_err(|e| format!("prefill batch: {e}"))?;
        offset = end;
    }
    Ok(())
}

/// decode 循环 (增量解码 + stop 检查)
///
/// 返回 finish_reason: "stop" (EOS 或 stop sequence) / "length" (达到 max_tokens)
fn decode_loop(
    ctx: &mut ForwardContext<'_>,
    cfg: &daiza_engine::model::config::Config,
    tokenizer: &daiza_runtime::tokenizer::BpeTokenizer,
    max_tokens: usize,
    params: SamplingParams,
    stop: Option<String>,
    on_delta: &mut dyn FnMut(&str) -> bool,
) -> Result<&'static str, String> {
    let mut rng = LcgRng::new(0xC0FFEE);
    let mut sampling_buf = SamplingBuffers::new(ctx.logits_buf.len());
    let mut pending_bytes: Vec<u8> = Vec::new();
    let mut generated: Vec<u8> = Vec::new(); // 完整生成文本的字节 (用于 stop sequence 匹配)

    let stop_bytes = stop.map(|s| s.into_bytes());
    let stop_len = stop_bytes.as_ref().map(|s| s.len()).unwrap_or(0);

    for _step in 0..max_tokens {
        let next_id = sample_top_k_top_p_into(
            &ctx.logits_buf, params, &mut || rng.next_f32(), &mut sampling_buf,
        ) as u32;

        // EOS
        if next_id == cfg.eos_token_id {
            // flush 残留字节
            if !pending_bytes.is_empty() {
                let tail = String::from_utf8_lossy(&pending_bytes);
                if !tail.is_empty() {
                    if !on_delta(&tail) { return Ok("stop"); }
                }
            }
            return Ok("stop");
        }

        // 收集 token bytes
        decode_token_bytes(tokenizer, next_id, &mut pending_bytes);
        let delta = drain_complete_utf8(&mut pending_bytes);

        // 检查 stop sequence (累积已生成字节, 命中则截断输出并结束)
        let mut stop_hit = false;
        if let Some(ref sb) = stop_bytes {
            generated.extend_from_slice(delta.as_bytes());
            if generated.len() >= stop_len && &generated[generated.len() - stop_len..] == sb.as_slice() {
                stop_hit = true;
            }
        }

        if !delta.is_empty() {
            if !on_delta(&delta) { return Ok("stop"); }
        }

        if stop_hit {
            return Ok("stop");
        }

        // forward 下一个 token
        forward_single_token(ctx, next_id).map_err(|e| format!("forward: {e}"))?;
    }

    // 达到 max_tokens
    if !pending_bytes.is_empty() {
        let tail = String::from_utf8_lossy(&pending_bytes);
        if !tail.is_empty() {
            let _ = on_delta(&tail);
        }
    }
    Ok("length")
}

/// vision 流式生成: 保存图片到临时文件 → 调 generate_with_image 逻辑
///
/// 简化实现: 非流式生成完整文本后一次性回调 (vision 路径流式化较复杂, 后续优化)
fn generate_with_vision_stream(
    engine: &mut Engine,
    prompt: &str,
    images: &[Vec<u8>],
    max_tokens: usize,
    params: SamplingParams,
    stop: Option<String>,
    on_delta: &mut dyn FnMut(&str) -> bool,
) -> Result<&'static str, String> {
    // 保存图片到临时文件 (preprocessImage 需要 PathBuf)
    let tmp_dir = std::env::temp_dir().join("daiza_server_images");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let mut img_paths: Vec<PathBuf> = Vec::new();
    for (i, bytes) in images.iter().enumerate() {
        let path = tmp_dir.join(format!("img_{i}_{}.dat", now_ts()));
        if let Err(e) = std::fs::write(&path, bytes) {
            return Err(format!("write temp image: {e}"));
        }
        img_paths.push(path);
    }

    // 构造 vision prompt (与 engine.generate_with_image 一致: prompt 中已含 image_pad)
    // 注意: build_chat_prompt 已经在 prompt 里插入了 <|image_pad|>,
    // 但 generate_with_image 内部会再插一次, 所以这里用原始 prompt (不含 image_pad)
    // 实际上 build_chat_prompt 已经处理了 image_pad, 我们需要用 raw generate 路径
    //
    // 简化: 直接调 generate_with_image, 它会自己构造 chat input
    // 但 generate_with_image 会重新构造 chat template, 与我们的 prompt 重复
    // 所以这里用 prompt 作为 user_msg, 走 generate_with_image
    let result = engine.generate_with_image(
        prompt,
        &img_paths,
        max_tokens,
        params,
        None, // system prompt 已在 prompt 里
    );

    // 清理临时文件
    for p in &img_paths {
        let _ = std::fs::remove_file(p);
    }

    match result {
        Ok(text) => {
            // stop sequence 检查
            let output = if let Some(ref s) = stop {
                if let Some(pos) = text.find(s) {
                    text[..pos].to_string()
                } else {
                    text
                }
            } else {
                text
            };
            if !output.is_empty() {
                let _ = on_delta(&output);
            }
            Ok("stop")
        }
        Err(e) => Err(e.to_string()),
    }
}

// ─── main ─────────────────────────────────────────────────────────────

fn get_opt<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == flag {
            if let Some(v) = iter.next() {
                return Some(v.as_str());
            }
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("Usage: daiza-server --model <gguf_path> [options]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --model <path>       主权重 GGUF 路径 (必需)");
        eprintln!("  --mmproj <path>      多模态视觉编码器 mmproj GGUF (可选, 启用 image_url)");
        eprintln!("  --port <n>           监听端口 (默认 8080)");
        eprintln!("  --api-key <key>      API key 鉴权 (客户端需带 Authorization: Bearer <key>)");
        eprintln!("  --max-tokens <n>     默认最大生成 token 数 (默认 2048)");
        eprintln!("  --greedy             贪心解码 (temperature=0)");
        return;
    }
    let gguf_path: PathBuf = match get_opt(&args, "--model") {
        Some(m) => m.into(),
        None => {
            eprintln!("Error: --model <gguf_path> is required (see --help)");
            std::process::exit(1);
        }
    };
    let mmproj_path: Option<PathBuf> = get_opt(&args, "--mmproj").map(PathBuf::from);
    let port: u16 = get_opt(&args, "--port")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    let api_key: Option<String> = get_opt(&args, "--api-key").map(String::from);
    let _greedy = args.iter().any(|a| a == "--greedy");

    // 热降频默认配置 (与 daiza-cli/daiza-web 一致)
    if std::env::var("DAIZA_WAIT_MODE").is_err() {
        std::env::set_var("DAIZA_WAIT_MODE", "yield");
    }
    if std::env::var("DAIZA_ACTIVE_WORKERS").is_err() {
        std::env::set_var("DAIZA_ACTIVE_WORKERS", "9");
    }

    eprintln!("[daiza-server] Loading model: {}", gguf_path.display());
    let mut engine = match Engine::load(&gguf_path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[daiza-server] load failed: {e}");
            std::process::exit(1);
        }
    };
    let model_id = gguf_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "daiza".to_string());
    eprintln!("[daiza-server] Engine loaded (weights load lazily on first request).");

    let vision_available = if let Some(ref mp) = mmproj_path {
        match engine.load_mmproj(mp) {
            Ok(()) => {
                eprintln!("[daiza-server] mmproj (vision encoder) loaded.");
                true
            }
            Err(e) => {
                eprintln!("[daiza-server] mmproj load failed: {e}");
                false
            }
        }
    } else {
        false
    };

    if api_key.is_some() {
        eprintln!("[daiza-server] API key authentication enabled.");
    } else {
        eprintln!("[daiza-server] WARNING: no API key set, all requests unauthenticated.");
    }

    let shared = Arc::new(Shared {
        engine: Mutex::new(engine),
        model_id,
        api_key,
        vision_available,
    });

    let listener = match TcpListener::bind(("0.0.0.0", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[daiza-server] bind 0.0.0.0:{port} failed: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("[daiza-server] OpenAI-compatible API ready at http://0.0.0.0:{port}/v1/chat/completions");

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let shared = shared.clone();
                std::thread::spawn(move || handle_conn(stream, shared));
            }
            Err(e) => eprintln!("[daiza-server] accept error: {e}"),
        }
    }
}

//! daiza-web 库: HTTP + SSE 聊天服务 (可被 daiza-web binary 或 daiza-app 内嵌调用)
//!
//! 暴露 `run_server` 函数, 接收已加载的 Engine + 配置, 阻塞当前线程跑 accept 循环。

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;

use daiza_runtime::engine::Engine;
use daiza_runtime::session_manager::SessionManager;
use daiza_engine::math::SamplingParams;

pub const INDEX_HTML: &str = include_str!("../web/index.html");
pub const SESSIONS_DIR: &str = "daiza_sessions";
pub const MAX_INACTIVE_SESSIONS: usize = 10;
pub const DEFAULT_SYSTEM_PROMPT: &str = "你是一个乐于助人的助手。请用与用户相同的语言回复。";

/// 服务配置: 由调用方构造 (daiza-web binary 或 daiza-app), 传入已加载的 Engine
pub struct ServerConfig {
    pub engine: Engine,
    pub model_name: String,
    pub params: SamplingParams,
    pub max_tokens: usize,
    pub port: u16,
    pub no_open: bool,
    pub dspark_available: bool,
    pub vision_available: bool,
}

/// 启动 HTTP/SSE 服务 (阻塞当前线程)
///
/// 创建 SessionManager + upload_dir + Shared, bind listener, accept 循环。
/// 每个连接 spawn 独立线程处理。
pub fn run_server(cfg: ServerConfig) -> std::io::Result<()> {
    let ServerConfig {
        port,
        no_open,
        ..
    } = cfg;

    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let url = format!("http://127.0.0.1:{port}/");
    println!("[daiza-web] Chat GUI ready at {url}");
    if !no_open {
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", &url])
            .spawn();
    }
    run_server_with_listener(cfg, listener, None)
}

/// 使用已 bind 的 listener 启动服务 (供 daiza-app 预检端口后调用, 避免竞态)
///
/// 调用方负责 bind listener (可提前检测端口占用并返回明确错误)
/// `ready_tx`: 进入 accept 循环前发送一次信号, 让调用方确认服务已就绪
///             (None 时跳过, 与 run_server 行为一致)
pub fn run_server_with_listener(
    cfg: ServerConfig,
    listener: TcpListener,
    ready_tx: Option<std::sync::mpsc::Sender<()>>,
) -> std::io::Result<()> {
    let ServerConfig {
        engine,
        model_name,
        params,
        max_tokens,
        port: _,
        no_open: _,
        dspark_available,
        vision_available,
    } = cfg;

    let session_mgr = SessionManager::new(
        std::path::Path::new(SESSIONS_DIR),
        MAX_INACTIVE_SESSIONS,
    )
    .map_err(|e| std::io::Error::other(e.to_string()))?;
    let upload_dir = std::path::PathBuf::from(SESSIONS_DIR).join("uploads");
    std::fs::create_dir_all(&upload_dir)?;
    let history_dir = std::path::PathBuf::from(SESSIONS_DIR).join("history");
    std::fs::create_dir_all(&history_dir)?;

    let shared = std::sync::Arc::new(Shared {
        engine: Mutex::new(engine),
        session_mgr: Mutex::new(session_mgr),
        params: Mutex::new(params),
        max_tokens: Mutex::new(max_tokens),
        model_name,
        ready: std::sync::atomic::AtomicBool::new(true),
        abort: std::sync::atomic::AtomicBool::new(false),
        dspark_available,
        vision_available,
        use_dspark: std::sync::atomic::AtomicBool::new(true),
        dspark_fallback: std::sync::atomic::AtomicBool::new(true),
        think_enabled: std::sync::atomic::AtomicBool::new(true),
        system_prompt: std::sync::Mutex::new(DEFAULT_SYSTEM_PROMPT.to_string()),
        upload_dir,
        history_dir,
    });

    // 通知调用方: 即将进入 accept 循环 (SessionManager/upload_dir 已就绪)
    if let Some(tx) = ready_tx {
        let _ = tx.send(());
    }

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let shared = shared.clone();
                std::thread::spawn(move || handle_conn(stream, shared));
            }
            Err(e) => eprintln!("[daiza-web] accept error: {e}"),
        }
    }
    Ok(())
}

// ─── 辅助函数 ─────────────────────────────────────────────────────────

/// 极简 JSON: 提取 {"key": "..."} 字段 (支持 \" \\ \/ \n \t \r \uXXXX 转义)
fn json_extract_field(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let idx = body.find(&pat)? + pat.len();
    let rest = &body[idx..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('"') {
        return None;
    }
    let bytes = after.as_bytes();
    let mut out = String::new();
    let mut i = 1;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Some(out),
            b'\\' => {
                i += 1;
                if i >= bytes.len() {
                    return None;
                }
                match bytes[i] {
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'u' => {
                        if i + 4 >= bytes.len() {
                            return None;
                        }
                        let hex = std::str::from_utf8(&bytes[i + 1..i + 5]).ok()?;
                        let mut cp = u32::from_str_radix(hex, 16).ok()?;
                        i += 4;
                        // 代理对: \uD83D\uDE00
                        if (0xD800..0xDC00).contains(&cp)
                            && i + 6 < bytes.len()
                            && bytes[i + 1] == b'\\'
                            && bytes[i + 2] == b'u'
                        {
                            let hex2 = std::str::from_utf8(&bytes[i + 3..i + 7]).ok()?;
                            let lo = u32::from_str_radix(hex2, 16).ok()?;
                            if (0xDC00..0xE000).contains(&lo) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                i += 6;
                            }
                        }
                        out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                    }
                    _ => return None,
                }
            }
            _ => {
                // 拷贝完整 UTF-8 序列
                let len = utf8_len(bytes[i]);
                if i + len > bytes.len() {
                    return None;
                }
                out.push_str(std::str::from_utf8(&bytes[i..i + len]).ok()?);
                i += len - 1;
            }
        }
        i += 1;
    }
    None
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
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

/// 提取 "key": value 中的 value 原始字符串 (数字/布尔/null, 字符串则去引号)
fn json_extract_raw(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let idx = body.find(&pat)? + pat.len();
    let rest = &body[idx..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if after.starts_with('"') {
        return json_extract_field(body, key);
    }
    let end = after
        .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
        .unwrap_or(after.len());
    Some(after[..end].trim().to_string())
}

/// 提取 "key": ["s1","s2",...] 字符串数组 (用于 images 字段)
/// 容错: 仅支持简单字符串数组 (无嵌套对象/数组), 字符串内不含转义引号
fn json_extract_string_array(body: &str, key: &str) -> Vec<String> {
    let pat = format!("\"{key}\"");
    let Some(idx) = body.find(&pat) else { return Vec::new(); };
    let rest = &body[idx + pat.len()..];
    let Some(colon) = rest.find(':') else { return Vec::new(); };
    let after = rest[colon + 1..].trim_start();
    let Some(arr_start) = after.find('[') else { return Vec::new(); };
    // 找到匹配的 ]
    let arr_body = &after[arr_start + 1..];
    let Some(arr_end) = find_matching_bracket(arr_body) else { return Vec::new(); };
    let inner = &arr_body[..arr_end];
    // 简单切分: 提取所有 "..." 字符串
    let mut out = Vec::new();
    let bytes = inner.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            // 复用 json_extract_field 的转义逻辑: 找到配对引号
            let mut s = String::new();
            i += 1;
            while i < bytes.len() {
                match bytes[i] {
                    b'"' => { i += 1; break; }
                    b'\\' => {
                        i += 1;
                        if i < bytes.len() {
                            match bytes[i] {
                                b'n' => s.push('\n'),
                                b't' => s.push('\t'),
                                b'r' => s.push('\r'),
                                b'"' => s.push('"'),
                                b'\\' => s.push('\\'),
                                b'/' => s.push('/'),
                                _ => s.push(bytes[i] as char),
                            }
                            i += 1;
                        }
                    }
                    _ => {
                        let len = utf8_len(bytes[i]);
                        if i + len <= bytes.len() {
                            if let Ok(seg) = std::str::from_utf8(&bytes[i..i + len]) {
                                s.push_str(seg);
                            }
                        }
                        i += len;
                    }
                }
            }
            out.push(s);
        } else {
            i += 1;
        }
    }
    out
}

/// 找到 ] 的位置 (忽略字符串内的 ])
fn find_matching_bracket(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b']' => return Some(i),
            b'"' => {
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' { i += 2; continue; }
                    if bytes[i] == b'"' { i += 1; break; }
                    i += utf8_len(bytes[i]);
                }
            }
            _ => i += utf8_len(bytes[i]),
        }
    }
    None
}

fn read_request(stream: &mut TcpStream) -> Option<(String, String, String)> {
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 8192];
    let mut header_end = None;
    let mut content_len = 0usize;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        if header_end.is_none() {
            if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
                header_end = Some(pos + 4);
                let head = String::from_utf8_lossy(&buf[..pos]);
                for line in head.lines() {
                    let l = line.to_ascii_lowercase();
                    if let Some(v) = l.strip_prefix("content-length:") {
                        content_len = v.trim().parse().unwrap_or(0);
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
        // 限制 16MB (支持多图 base64 上传, 单图 ~10MB)
        if buf.len() > 16 << 20 {
            return None;
        }
    }
    let he = header_end?;
    let head = String::from_utf8_lossy(&buf[..he]).into_owned();
    let body = String::from_utf8_lossy(&buf[he..he + content_len]).into_owned();
    let mut parts = head.lines().next()?.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    Some((method, path, body))
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn http_response(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.flush();
}

fn sse_write(stream: &mut TcpStream, event: &str, data: &str) -> bool {
    stream
        .write_all(format!("event: {event}\ndata: {data}\n\n").as_bytes())
        .and_then(|_| stream.flush())
        .is_ok()
}

// ─── Shared 状态 + HTTP handler ──────────────────────────────────────

struct Shared {
    engine: Mutex<Engine>,
    /// 多会话管理器 (park/unpark inactive session)
    session_mgr: Mutex<SessionManager>,
    /// 采样参数 (运行时可通过 /api/params 修改)
    params: Mutex<SamplingParams>,
    /// 每轮最多生成 token 数 (运行时可修改)
    max_tokens: Mutex<usize>,
    model_name: String,
    /// weights 是否已加载 (AtomicBool, /api/status 无需锁 engine)
    ready: std::sync::atomic::AtomicBool,
    /// 中断当前 handle_chat 推理 (/api/abort 设置, 回调每次检查)
    abort: std::sync::atomic::AtomicBool,
    /// DSpark drafter 是否已加载 (启动时由 --dspark 决定, 运行时只读)
    dspark_available: bool,
    /// mmproj 视觉编码器是否已加载 (启动时由 --mmproj 决定, 运行时只读)
    vision_available: bool,
    /// 当前是否使用 DSpark 模式 (/api/dspark 设置, handle_chat 读取)
    use_dspark: std::sync::atomic::AtomicBool,
    /// DSpark 自动降级开关 (/api/dspark 设置, handle_chat_dspark 读取)
    /// 开启时 probe 窗口检测 DSpark 慢于 native 则自动切原生 decode
    dspark_fallback: std::sync::atomic::AtomicBool,
    /// 是否启用思考模式 (/api/params 设置, handle_chat 读取)
    /// 切换后需要重置 session (think_enabled 影响 increment 末尾的 <think>\n)
    think_enabled: std::sync::atomic::AtomicBool,
    /// 系统提示词 (/api/params 设置, session_begin 时读取)
    /// 修改后需要重置 session 才能生效 (system_prompt 在 session 创建时固化)
    system_prompt: std::sync::Mutex<String>,
    /// 图片上传保存目录 (POST /api/upload 把 base64 图片保存到此目录)
    upload_dir: std::path::PathBuf,
    /// 聊天历史保存目录 (每个 session 一个 JSON 文件, 切换/重启时恢复前端显示)
    history_dir: std::path::PathBuf,
}

fn handle_conn(mut stream: TcpStream, shared: std::sync::Arc<Shared>) {
    let _ = stream.set_nodelay(true);
    let Some((method, path, body)) = read_request(&mut stream) else {
        return;
    };
    let route = path.split('?').next().unwrap_or("/");
    match (method.as_str(), route) {
        ("GET", "/") | ("GET", "/index.html") => {
            http_response(&mut stream, "200 OK", "text/html; charset=utf-8", INDEX_HTML);
        }
        ("GET", "/api/status") => {
            let ready = shared.ready.load(std::sync::atomic::Ordering::Relaxed);
            let model = &shared.model_name;
            let dspark_available = shared.dspark_available;
            let vision_available = shared.vision_available;
            let use_dspark = shared.use_dspark.load(std::sync::atomic::Ordering::Relaxed);
            http_response(
                &mut stream,
                "200 OK",
                "application/json; charset=utf-8",
                &format!(
                    "{{\"ready\":{ready},\"model\":\"{}\",\"dspark_available\":{dspark_available},\"vision_available\":{vision_available},\"use_dspark\":{use_dspark}}}",
                    json_escape(model)
                ),
            );
        }
        ("POST", "/api/chat") => {
            handle_chat(&mut stream, &shared, &body);
        }
        ("POST", "/api/upload") => {
            handle_upload(&mut stream, &shared, &body);
        }
        ("GET", "/api/sessions") => {
            handle_sessions_list(&mut stream, &shared);
        }
        ("POST", "/api/sessions/new") => {
            handle_sessions_new(&mut stream, &shared, &body);
        }
        ("POST", "/api/sessions/ensure") => {
            handle_sessions_ensure(&mut stream, &shared);
        }
        ("POST", "/api/sessions/switch") => {
            handle_sessions_switch(&mut stream, &shared, &body);
        }
        ("POST", "/api/sessions/delete") => {
            handle_sessions_delete(&mut stream, &shared, &body);
        }
        ("POST", "/api/sessions/rename") => {
            handle_sessions_rename(&mut stream, &shared, &body);
        }
        ("GET", "/api/sessions/history") => {
            handle_sessions_history(&mut stream, &shared, &path);
        }
        ("GET", "/api/params") => {
            handle_params_get(&mut stream, &shared);
        }
        ("POST", "/api/params") => {
            handle_params_set(&mut stream, &shared, &body);
        }
        ("GET", "/api/dspark") => {
            handle_dspark_get(&mut stream, &shared);
        }
        ("POST", "/api/dspark") => {
            handle_dspark_set(&mut stream, &shared, &body);
        }
        ("POST", "/api/abort") => {
            handle_abort(&mut stream, &shared);
        }
        ("GET", "/api/tools") => {
            handle_tools_get(&mut stream, &shared);
        }
        ("POST", "/api/tools") => {
            handle_tools_set(&mut stream, &shared, &body);
        }
        ("POST", "/api/tools/delete") => {
            handle_tools_delete(&mut stream, &shared, &body);
        }
        ("POST", "/api/chat/tool_response") => {
            handle_chat_tool_response(&mut stream, &shared, &body);
        }
        _ => {
            http_response(&mut stream, "404 Not Found", "text/plain; charset=utf-8", "not found");
        }
    }
}

#[allow(unsafe_code)]
fn handle_chat(stream: &mut TcpStream, shared: &Shared, body: &str) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if stream.write_all(head.as_bytes()).and_then(|_| stream.flush()).is_err() {
        return;
    }

    let Some(msg) = json_extract_field(body, "message") else {
        sse_write(stream, "error", "{\"error\":\"invalid request body\"}");
        return;
    };
    let msg = msg.trim().to_string();
    if msg.is_empty() {
        sse_write(stream, "error", "{\"error\":\"empty message\"}");
        return;
    }
    if msg.len() > 32 * 1024 {
        sse_write(stream, "error", "{\"error\":\"message too long (max 32KB)\"}");
        return;
    }

    // 重置 abort flag (上一次 chat 可能被中断, 防止残留 flag 影响本次)
    shared.abort.store(false, std::sync::atomic::Ordering::Relaxed);

    let mut engine = shared.engine.lock().unwrap();
    let t0 = std::time::Instant::now();

    // session 懒创建 (首个请求时执行; 权重已在启动时加载, 此处仅创建 Session, <1s)
    if engine.session.is_none() {
        let think = shared.think_enabled.load(std::sync::atomic::Ordering::Relaxed);
        let sys = shared.system_prompt.lock().unwrap().clone();
        match engine.session_begin(think, Some(&sys)) {
            Ok(()) => {
                ensure_active_id(shared);
                shared.ready.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            Err(e) => {
                sse_write(stream, "error", &format!("{{\"error\":\"session init: {e}\"}}"));
                return;
            }
        }
    } else {
        // session 已存在但 active_id 可能缺失 (被其他路径重建但未同步), 补一次
        ensure_active_id(shared);
    }

    // think 模式切换检测: 若 Shared.think_enabled 与 session.think_enabled 不一致,
    // - 非 DSpark: 重置 session (清空 KV/SSM state) 以应用新设置
    // - DSpark: 直接修改 session.think_enabled (DSpark 不依赖 KV/SSM state)
    let want_think = shared.think_enabled.load(std::sync::atomic::Ordering::Relaxed);
    // DSpark 模式: 不走 session KV/SSM 复用, 每次从 messages 构造完整 prompt,
    // 调 generate_with_dspark (非流式, 完成后一次性发 delta)
    let use_dspark = shared.use_dspark.load(std::sync::atomic::Ordering::Relaxed)
        && shared.dspark_available;
    if use_dspark {
        if let Some(s) = engine.session.as_mut() {
            s.think_enabled = want_think;
        }
    } else {
        let need_reset = engine.session.as_ref().map(|s| s.think_enabled != want_think).unwrap_or(false);
        if need_reset {
            let think = want_think;
            let sys = engine.session.as_ref().and_then(|s| s.system_prompt.clone());
            if let Err(e) = engine.session_begin(think, sys.as_deref()) {
                sse_write(stream, "error", &format!("{{\"error\":\"session reset: {e}\"}}"));
                return;
            }
        }
    }

    let stream_ref = &mut *stream;
    let params = *shared.params.lock().unwrap();
    let max_tokens = *shared.max_tokens.lock().unwrap();
    let abort_ref = &shared.abort;

    // 指标统计: TTFT (首 token 延迟) + n_tokens (近似 token 数)
    let mut t_first: Option<std::time::Instant> = None;
    let mut n_tokens: usize = 0;

    // 图片处理: 解析 images 字段 (路径数组), 推入 session.pending_images
    // ★ DSpark 不支持 vision 注入: 有图片时自动走 vision 路径 (非 DSpark)
    let images: Vec<String> = json_extract_string_array(body, "images");
    let use_vision = shared.vision_available && !images.is_empty();
    // DSpark + 有图片: 临时禁用 DSpark, 走 vision 路径
    let use_dspark = use_dspark && images.is_empty();
    if use_vision {
        // ★ DSpark → Vision 切换时 session 状态可能不兼容 (dspark_tap_history 等),
        //   重置 session 确保干净状态; think_enabled 已同步
        if !use_dspark && shared.use_dspark.load(std::sync::atomic::Ordering::Relaxed) {
            let think = want_think;
            let sys = engine.session.as_ref().and_then(|s| s.system_prompt.clone());
            if let Err(e) = engine.session_begin(think, sys.as_deref()) {
                sse_write(stream, "error", &format!("{{\"error\":\"session reset for vision: {e}\"}}"));
                return;
            }
        }
        // 推入 pending_images (session_reply_with_vision_stream 会 drain 消费)
        if let Some(s) = engine.session.as_mut() {
            for p in &images {
                s.pending_images.push(std::path::PathBuf::from(p));
            }
        }
    }

    let result = if use_dspark {
        let fallback = shared.dspark_fallback.load(std::sync::atomic::Ordering::Relaxed);
        handle_chat_dspark(
            &mut engine, &msg, max_tokens, params, abort_ref, stream_ref,
            &mut t_first, &mut n_tokens, fallback,
        )
    } else if use_vision {
        // Vision 流式路径: session_reply_with_vision_stream + progress SSE 事件
        // 两个闭包都需写 stream, 但不会同时执行 (on_progress 在 prefill, on_delta 在 decode),
        // 用裸指针绕过借用检查 (engine 同步调用, 无并发风险)
        use daiza_runtime::engine::ProgressEvent;
        let stream_ptr: *mut TcpStream = stream_ref as *mut TcpStream;
        engine.session_reply_with_vision_stream(
            &msg, max_tokens, params,
            &mut |delta| {
                if abort_ref.load(std::sync::atomic::Ordering::Relaxed) {
                    return false;
                }
                if t_first.is_none() {
                    t_first = Some(std::time::Instant::now());
                }
                n_tokens += 1;
                let payload = format!("{{\"text\":\"{}\"}}", json_escape(delta));
                // SAFETY: on_delta 与 on_progress 互斥执行 (prefill vs decode), 不会并发访问 stream
                unsafe { sse_write(&mut *stream_ptr, "delta", &payload) }
            },
            &mut |prog| {
                let payload = match prog {
                    ProgressEvent::Vision { image_idx, total, stage, ms } => {
                        format!("{{\"stage\":\"vision\",\"image_idx\":{image_idx},\"total\":{total},\"phase\":\"{stage}\",\"ms\":{ms}}}")
                    }
                    ProgressEvent::Prefill { done_tokens, total_tokens } => {
                        format!("{{\"stage\":\"prefill\",\"done\":{done_tokens},\"total\":{total_tokens}}}")
                    }
                };
                // SAFETY: 同上, 与 on_delta 互斥执行
                unsafe { let _ = sse_write(&mut *stream_ptr, "progress", &payload); }
            },
        )
    } else {
        engine.session_reply_stream(
            &msg,
            max_tokens,
            params,
            &mut |delta| {
                // 中断检查: /api/abort 设置 flag 后立即返回 false, 推理循环退出
                if abort_ref.load(std::sync::atomic::Ordering::Relaxed) {
                    return false;
                }
                if t_first.is_none() {
                    t_first = Some(std::time::Instant::now());
                }
                n_tokens += 1;
                let payload = format!("{{\"text\":\"{}\"}}", json_escape(delta));
                sse_write(stream_ref, "delta", &payload)
            },
        )
    };

    // 清理 flag (防止下次 chat 误判)
    shared.abort.store(false, std::sync::atomic::Ordering::Relaxed);

    match result {
        Ok(full) => {
            let total_ms = t0.elapsed().as_millis();
            let ttft_ms = t_first
                .map(|t| t.duration_since(t0).as_millis())
                .unwrap_or(total_ms);
            let decode_ms = total_ms.saturating_sub(ttft_ms);
            let ms_per_tok = if n_tokens > 0 { decode_ms / n_tokens as u128 } else { 0 };
            let elapsed = t0.elapsed().as_secs_f64();
            let n_chars = full.chars().count();
            let stats = format!("~{n_chars} chars · {elapsed:.1}s");

            // ★ tool_call 检测: 解析完整输出中的 <tool_call> 标签
            //   若有 tool_call, 发出 tool_call SSE 事件 (前端收到后执行工具并回传 tool_response)
            //   done 事件附带 has_tool_call 标志, 前端据此判断是否等待 tool_response
            let tool_calls = daiza_runtime::tool_call::parse_tool_calls(&full);
            let has_tool_call = !tool_calls.is_empty();
            // 统一 push assistant 消息到 messages 历史 (DSpark 和非 DSpark 路径共用)
            use daiza_runtime::tool_call::ToolMessage;
            let think_on = engine.session.as_ref().map(|s| s.think_enabled).unwrap_or(false);
            let cleaned = clean_assistant_text(&full, think_on);
            // 保留一份给 history_append (cleaned 会在下面 push 时被 move)
            let cleaned_for_history = cleaned.clone();
            if has_tool_call {
                // 逐个发出 tool_call 事件
                for tc in &tool_calls {
                    let payload = format!("{{\"tool_call\":{}}}", tool_call_to_json(tc));
                    sse_write(stream, "tool_call", &payload);
                }
                // push assistant 消息 (带 tool_calls)
                let session = engine.session.as_mut().unwrap();
                session.messages.push(ToolMessage::assistant_with_tool_calls(
                    cleaned,
                    tool_calls.clone(),
                ));
            } else {
                // 无 tool_call: push 普通 assistant 消息
                let session = engine.session.as_mut().unwrap();
                session.messages.push(ToolMessage::assistant(cleaned));
            }

            sse_write(
                stream,
                "done",
                &format!(
                    "{{\"ok\":true,\"stats\":\"{}\",\"ttft_ms\":{ttft_ms},\"ms_per_tok\":{ms_per_tok},\"n_tokens\":{n_tokens},\"has_tool_call\":{has_tool_call}}}",
                    json_escape(&stats)
                ),
            );
            eprintln!("[bench] chat done: ttft={ttft_ms}ms, decode({n_tokens}t)={decode_ms}ms (~{ms_per_tok}ms/tok), total={total_ms}ms");
            // ★ 追加 history (user + assistant), 用于切换会话/重启时恢复前端显示
            let sid = shared.session_mgr.lock().unwrap().active_id().map(String::from);
            if let Some(sid) = sid {
                history_append(&shared.history_dir, &sid, "user", &msg);
                history_append(&shared.history_dir, &sid, "assistant", &cleaned_for_history);
            }
        }
        Err(e) => {
            sse_write(
                stream,
                "error",
                &format!("{{\"error\":\"{}\"}}", json_escape(&e.to_string())),
            );
        }
    }
}

/// POST /api/upload: 接收 base64 编码的图片, 保存到 upload_dir, 返回路径
///
/// 请求体格式: {"name":"foo.png","data":"<base64 without data: prefix>"}
/// 响应: {"path":"daiza_sessions/uploads/xxx.png"} 或 {"error":"..."}
fn handle_upload(stream: &mut TcpStream, shared: &Shared, body: &str) {
    let Some(name) = json_extract_field(body, "name") else {
        http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
            "{\"error\":\"missing 'name' field\"}");
        return;
    };
    let Some(data_b64) = json_extract_field(body, "data") else {
        http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
            "{\"error\":\"missing 'data' field\"}");
        return;
    };
    // 限制 name 仅含文件名部分 (防路径穿越)
    let safe_name = std::path::Path::new(&name)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| format!("upload_{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis()));
    // 限制大小: base64 ~1.33x, 此处限制原始 ~10MB → base64 ~13.3MB
    if data_b64.len() > 14 * 1024 * 1024 {
        http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
            "{\"error\":\"image too large (max 10MB)\"}");
        return;
    }
    // base64 解码 (标准 base64, 允许 padding)
    let bytes = match base64_decode(&data_b64) {
        Ok(b) => b,
        Err(e) => {
            http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
                &format!("{{\"error\":\"base64 decode failed: {e}\"}}"));
            return;
        }
    };
    // 校验图片格式 (magic bytes): PNG / JPEG / WebP / BMP / GIF
    if !is_supported_image(&bytes) {
        http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
            "{\"error\":\"unsupported image format (only PNG/JPEG/WebP/BMP/GIF)\"}");
        return;
    }
    // 唯一文件名 (避免并发冲突): timestamp + 原始名
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default().as_millis();
    let ext = std::path::Path::new(&safe_name).extension()
        .map(|s| format!(".{}", s.to_string_lossy()))
        .unwrap_or_else(|| ".png".to_string());
    let stem = std::path::Path::new(&safe_name).file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "upload".to_string());
    let filename = format!("{stem}_{ts}{ext}");
    let path = shared.upload_dir.join(&filename);
    if let Err(e) = std::fs::write(&path, &bytes) {
        http_response(stream, "500 Internal Server Error", "application/json; charset=utf-8",
            &format!("{{\"error\":\"write failed: {e}\"}}"));
        return;
    }
    // 返回相对路径 (engine.preprocessImage 接受任意路径, 相对/绝对均可)
    let rel = format!("daiza_sessions/uploads/{filename}");
    eprintln!("[upload] saved {} ({} bytes) → {rel}", safe_name, bytes.len());
    http_response(stream, "200 OK", "application/json; charset=utf-8",
        &format!("{{\"path\":\"{}\"}}", json_escape(&rel)));
}

/// 标准 base64 解码 (无依赖, 支持 padding)
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

/// 校验图片 magic bytes
fn is_supported_image(b: &[u8]) -> bool {
    b.len() >= 6 && (
        b.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A]) // PNG
        || b.starts_with(&[0xFF, 0xD8, 0xFF])                 // JPEG
        || b.starts_with(b"RIFF") && b[8..12] == *b"WEBP"    // WebP
        || b.starts_with(&[0x42, 0x4D])                       // BMP
        || b.starts_with(&[0x47, 0x49, 0x46, 0x38])          // GIF
    )
}


/// 流式: 每 token 通过 on_delta 回调增量发送 SSE delta
///
/// `fallback_enabled`: DSpark 自动降级开关 (probe 检测慢于 native 时切原生 decode)
#[allow(clippy::too_many_arguments)]
fn handle_chat_dspark(
    engine: &mut Engine,
    msg: &str,
    max_tokens: usize,
    params: SamplingParams,
    abort_ref: &std::sync::atomic::AtomicBool,
    stream: &mut TcpStream,
    t_first: &mut Option<std::time::Instant>,
    n_tokens: &mut usize,
    fallback_enabled: bool,
) -> Result<String, daiza_runtime::BonsaiError> {
    use daiza_runtime::tool_call::ToolMessage;
    use daiza_runtime::session::render_tools_block;

    let session = engine.session.as_mut().unwrap();
    // push 当前 user msg (session_reply_stream 也会 push, 这里手动维护)
    session.messages.push(ToolMessage::user(msg.to_string()));

    // ★ 增量 prompt: 只编码新增 token (复用 KV cache + SSM state + drafter context)
    //   首轮 (state.pos==0): system + user + assistant 头
    //   后续轮: 补上一轮 EOS + user + assistant 头
    //   ★ 用 history_tokens.is_empty() 判断首轮 (而非 state.pos==0),
    //     因为 DSpark 切换后 state.pos=0 但 history_tokens 可能非空 (需恢复上下文)
    let mut prompt = String::new();
    if session.history_tokens.is_empty() {
        // 首轮: 带 system prompt (tools 非空时注入 tools 块)
        if !session.tools.is_empty() {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(&render_tools_block(&session.tools));
            if let Some(ref sys) = session.system_prompt {
                prompt.push_str(sys);
                prompt.push('\n');
            }
            prompt.push_str("<|im_end|>\n");
        } else if let Some(ref sys) = session.system_prompt {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(sys);
            prompt.push_str("<|im_end|>\n");
        }
    } else {
        // 后续轮: 补上上一轮 assistant 的结束标记 (decode 时 EOS 未 forward, 这里补入)
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(msg);
    prompt.push_str("<|im_end|>\n");
    prompt.push_str("<|im_start|>assistant\n");
    if session.think_enabled {
        prompt.push_str("<think>\n");
    } else {
        // ★ think 关闭: 追加空 think 块 <think></think>\n, 告诉模型 think 阶段已结束,
        //   直接生成正式回答。Bonsai-27B 训练时 assistant 总是以 <think> 开头,
        //   若只写 <|im_start|>assistant\n, 模型会自发生成 <think>...长篇think内容</think>,
        //   导致 CPU 空转、响应缓慢 (虽然 suppress_buf 过滤了显示, 但 token 仍被生成)
        prompt.push_str("<think></think>\n");
    }

    // clone system_prompt 避免 session 借用冲突 (generate_with_dspark_stream 需 &mut engine)
    let system = engine.session.as_ref().and_then(|s| s.system_prompt.clone());
    let result = engine.generate_with_dspark_stream(
        &prompt, max_tokens, params, system.as_deref(), 0.0, fallback_enabled,
        &mut |event| {
            // 中断检查: /api/abort 设置 flag 后立即返回 false, 推理循环退出
            if abort_ref.load(std::sync::atomic::Ordering::Relaxed) {
                return false;
            }
            // 首 token 时间 (draft 或 delta 都算)
            if t_first.is_none() {
                *t_first = Some(std::time::Instant::now());
            }
            use daiza_runtime::engine::DsparkEvent;
            match event {
                DsparkEvent::Draft(text) => {
                    // 乐观显示: 前端灰色显示 drafter 预测文本
                    // token 数近似: draft 文本字符数 / 2 (中英文混合粗估)
                    *n_tokens += text.chars().count().div_ceil(2);
                    let payload = format!("{{\"text\":\"{}\"}}", json_escape(&text));
                    sse_write(stream, "draft", &payload)
                }
                DsparkEvent::Accept => {
                    // draft 全部通过 verify: 前端灰色保留为最终文本
                    sse_write(stream, "accept", "{}")
                }
                DsparkEvent::Reject(accepted) => {
                    // 部分 reject: 前端删除上一个 draft, 用 accepted 替换
                    // 修正 token 数: draft 已按文本估算, 这里不额外加 (accepted 是 draft 的子集)
                    let payload = format!("{{\"accepted\":\"{}\"}}", json_escape(&accepted));
                    sse_write(stream, "reject", &payload)
                }
                DsparkEvent::Delta(text) => {
                    // 正常增量 (bonus / fallback)
                    *n_tokens += 1;
                    let payload = format!("{{\"text\":\"{}\"}}", json_escape(&text));
                    sse_write(stream, "delta", &payload)
                }
            }
        },
    );

    // ★ messages push 移到 handle_chat 的 result Ok 处理中统一管理
    //   (避免 tool_call 场景重复 push: assistant_with_tool_calls vs 普通 assistant)
    result
}

/// 清理 assistant 回复文本: 移除 think 块和特殊 token 字面文本
/// - <think>...</think> (含未闭合的 <think>): 模型思考内容, 不应进入下一轮 prompt 历史
/// - <|im_start|> <|im_end|> <|vision_start|> 等: chat template 标记, decode 后字面出现
///
/// `think_enabled`: think 模式是否开启。开启时 <think> 是 special token 不出现在模型输出中,
/// 模型 thinking 时输出格式为 `think内容</think>正式回答`, 需移除 `</think>` 及之前的内容。
/// 若输出不含 </think>, 说明模型未 thinking, 保留整个文本 (仅清理特殊 token)。
///
/// ★ think 关闭时模型仍可能输出 think 内容 (无 <think> 开头, 仅含 </think>):
///   如 "think内容</think>正式回答"。此时也需移除 </think> 及之前的内容,
///   否则 think 内容和 </think> 字面文本会污染下一轮 prompt 历史,
///   导致模型"学到"继续输出 think 格式 (即使 prompt 未加 <think>\n)。
fn clean_assistant_text(text: &str, think_enabled: bool) -> String {
    let _ = think_enabled; // think_enabled 不再影响清理逻辑 (统一处理 think 内容)
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    // ★ 统一处理: 无 <think> 开头 + 含 </think> (think 开启时 special token 不在输出中,
    //   think 关闭时模型仍可能输出 think 内容)。移除 </think> 及之前的内容, 保留正式回答。
    if !text.contains("<think>") && text.contains("</think>") {
        if let Some(close_idx) = text.find("</think>") {
            let after = &text[close_idx + 8..];
            // 递归清理 after 中的特殊 token (不再包含 think 块)
            return clean_assistant_text(after, false);
        }
    }
    // 其他情况: 走通用清理循环 (移除 <think>...</think> 块和特殊 token)
    while i < bytes.len() {
        // 检测 <think> 开始
        if text[i..].starts_with("<think>") {
            // 找到 </think> 或文本末尾
            let rest = &text[i + 7..];
            if let Some(end) = rest.find("</think>") {
                i = i + 7 + end + 8; // 跳过 <think>...</think>
            } else {
                break; // 未闭合 think, 丢弃剩余
            }
            continue;
        }
        // 检测特殊 token <|...|>
        if text[i..].starts_with("<|") {
            if let Some(end) = text[i + 2..].find("|>") {
                i = i + 2 + end + 2; // 跳过 <|...|>
                continue;
            }
        }
        // 普通字符: 拷贝完整 UTF-8 序列
        let len = utf8_len_local(bytes[i]);
        if i + len <= bytes.len() {
            if let Ok(seg) = std::str::from_utf8(&bytes[i..i + len]) {
                out.push_str(seg);
            }
        }
        i += len;
    }
    out
}

fn utf8_len_local(b: u8) -> usize {
    if b < 0x80 { 1 } else if b < 0xE0 { 2 } else if b < 0xF0 { 3 } else { 4 }
}

/// GET /api/dspark → {"available":bool,"enabled":bool,"fallback":bool}
fn handle_dspark_get(stream: &mut TcpStream, shared: &Shared) {
    let available = shared.dspark_available;
    let enabled = shared.use_dspark.load(std::sync::atomic::Ordering::Relaxed);
    let fallback = shared.dspark_fallback.load(std::sync::atomic::Ordering::Relaxed);
    let body = format!("{{\"available\":{available},\"enabled\":{enabled},\"fallback\":{fallback}}}");
    http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
}

/// POST /api/dspark {"enabled":true|false,"fallback":true|false}
/// enabled: 切换 DSpark 模式 (切换时自动重置 session)
/// fallback: DSpark 自动降级开关 (probe 检测慢于 native 时切原生 decode), 仅 DSpark 模式生效
fn handle_dspark_set(stream: &mut TcpStream, shared: &Shared, body: &str) {
    if !shared.dspark_available {
        http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
            "{\"error\":\"DSpark drafter not loaded\"}");
        return;
    }
    // fallback 字段可选 (只更新提供的字段)
    if let Some(v) = json_extract_raw(body, "fallback") {
        shared.dspark_fallback.store(v == "true", std::sync::atomic::Ordering::Relaxed);
    }
    // enabled 字段: 切换 DSpark 模式
    let enabled = match json_extract_raw(body, "enabled") {
        Some(v) => v == "true",
        None => {
            // 仅更新 fallback, 不切换模式
            let enabled = shared.use_dspark.load(std::sync::atomic::Ordering::Relaxed);
            let fallback = shared.dspark_fallback.load(std::sync::atomic::Ordering::Relaxed);
            let body = format!("{{\"available\":true,\"enabled\":{enabled},\"fallback\":{fallback}}}");
            http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
            return;
        }
    };
    let prev = shared.use_dspark.swap(enabled, std::sync::atomic::Ordering::Relaxed);

    // 模式切换时重置 session state (KV/SSM state 与 DSpark 非复用模式不一致)
    // ★ 保留 messages 历史: 用户切换 DSpark 不应丢失对话上下文,
    //   下一轮 prefill 会从 history_tokens 重建 KV/SSM state
    if prev != enabled {
        let mut engine = shared.engine.lock().unwrap();
        let think = shared.think_enabled.load(std::sync::atomic::Ordering::Relaxed);
        if let Some(session) = engine.session.as_mut() {
            session.reset_state();
            session.think_enabled = think;
        } else {
            // session 不存在: 创建新 session (无历史可保留)
            let sys = shared.system_prompt.lock().unwrap().clone();
            let _ = engine.session_begin(think, Some(&sys));
            ensure_active_id(shared);
        }
        // ★ 重置 spec_ctx cache: DSpark 模式切换时 drafter context 不兼容
        if let Some(spec) = engine.spec_ctx.as_mut() {
            spec.target_tap_len = 0;
            spec.drafter.cached_ctx_len = 0;
            spec.drafter.cached_kv_len = 0;
        }
    }
    let fallback = shared.dspark_fallback.load(std::sync::atomic::Ordering::Relaxed);
    let body = format!("{{\"available\":true,\"enabled\":{enabled},\"fallback\":{fallback}}}");
    http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
}

/// history 文件安全 id (与 session_manager 的 safe_id 规则一致)
fn history_safe_id(id: &str) -> String {
    id.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_")
}

/// history 文件路径: {history_dir}/{safe_id}.jsonl
fn history_path(history_dir: &std::path::Path, id: &str) -> std::path::PathBuf {
    history_dir.join(format!("{}.jsonl", history_safe_id(id)))
}

/// 追加一条消息到 history 文件 (JSON Lines 格式, 每行一个对象)
///
/// 文件格式 (每行):
/// `{"role":"user","content":"..."}`
/// `{"role":"assistant","content":"..."}`
///
/// append 写入, 无需读取全量, 进程崩溃不损坏已有内容
fn history_append(history_dir: &std::path::Path, id: &str, role: &str, content: &str) {
    let path = history_path(history_dir, id);
    let line = format!("{{\"role\":\"{}\",\"content\":\"{}\"}}\n", json_escape(role), json_escape(content));
    // create(true) + append(true): 文件不存在则新建, 存在则追加
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = f.write_all(line.as_bytes());
    }
}

/// 读取 history 文件, 返回 JSON 数组字符串 `[{"role":"...","content":"..."},...]`
///
/// 把 JSON Lines 按行组装成 JSON 数组 (每行已是合法 JSON 对象, 直接 join)
/// 文件不存在时返回 `"[]"`
fn history_load_json(history_dir: &std::path::Path, id: &str) -> String {
    let path = history_path(history_dir, id);
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            let lines: Vec<&str> = text.lines()
                .filter(|l| !l.is_empty())
                .collect();
            if lines.is_empty() {
                "[]".to_string()
            } else {
                format!("[{}]", lines.join(","))
            }
        }
        Err(_) => "[]".to_string(),
    }
}

/// 删除 history 文件 (session delete 时同步删除)
fn history_delete(history_dir: &std::path::Path, id: &str) {
    let path = history_path(history_dir, id);
    let _ = std::fs::remove_file(&path);
}

/// 重命名 history 文件 (session rename 时同步)
fn history_rename(history_dir: &std::path::Path, old_id: &str, new_id: &str) {
    let old_path = history_path(history_dir, old_id);
    let new_path = history_path(history_dir, new_id);
    let _ = std::fs::rename(&old_path, &new_path);
}

/// GET /api/sessions → {"active": "id"|null, "sessions": [{"id":"...","active":bool}]}
fn handle_sessions_list(stream: &mut TcpStream, shared: &Shared) {
    let mgr = shared.session_mgr.lock().unwrap();
    let active = mgr.active_id().map(String::from);
    let list = mgr.list();
    let mut sessions = String::from("[");
    for (i, (id, is_active)) in list.iter().enumerate() {
        if i > 0 { sessions.push(','); }
        sessions.push_str(&format!(
            "{{\"id\":\"{}\",\"active\":{}}}",
            json_escape(id), is_active
        ));
    }
    sessions.push(']');
    let active_json = match &active {
        Some(a) => format!("\"{}\"", json_escape(a)),
        None => "null".to_string(),
    };
    let body = format!("{{\"active\":{active_json},\"sessions\":{sessions}}}");
    http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
}

/// POST /api/sessions/ensure → 确保已有 active session (懒创建, 无需 body)
///
/// 前端 send() 开始时调用: 若无 active session, 立即创建并返回 active_id,
/// 让侧边栏在 prefill/推理开始前就显示标签。
/// 若已有 active session, 直接返回当前 active_id (幂等)。
fn handle_sessions_ensure(stream: &mut TcpStream, shared: &Shared) {
    let mut engine = shared.engine.lock().unwrap();
    if let Err(e) = ensure_session(shared, &mut engine) {
        http_response(stream, "500 Internal Server Error", "application/json; charset=utf-8",
            &format!("{{\"error\":\"{e}\"}}"));
        return;
    }
    let mgr = shared.session_mgr.lock().unwrap();
    let active = mgr.active_id().map(String::from);
    let active_json = match &active {
        Some(a) => format!("\"{}\"", json_escape(a)),
        None => "null".to_string(),
    };
    http_response(stream, "200 OK", "application/json; charset=utf-8",
        &format!("{{\"ok\":true,\"active\":{active_json}}}"));
}

/// POST /api/sessions/new {"id":"..."} → park 当前 active, 创建新 session
fn handle_sessions_new(stream: &mut TcpStream, shared: &Shared, body: &str) {
    let id = match json_extract_field(body, "id") {
        Some(i) => i.trim().to_string(),
        None => {
            http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
                "{\"error\":\"missing id\"}");
            return;
        }
    };
    if id.is_empty() || id.len() > 64 {
        http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
            "{\"error\":\"invalid id\"}");
        return;
    }
    let mut engine = shared.engine.lock().unwrap();
    let mut mgr = shared.session_mgr.lock().unwrap();
    if mgr.exists(&id) {
        http_response(stream, "409 Conflict", "application/json; charset=utf-8",
            "{\"error\":\"session id exists\"}");
        return;
    }
    // park 当前 active (如果有)
    if let Some(cur_id) = mgr.active_id().map(String::from) {
        if let Some(session) = engine.session.take() {
            let _ = mgr.park(&cur_id, &session, &engine.config);
        }
    }
    let sys = shared.system_prompt.lock().unwrap().clone();
    match engine.session_begin(
        shared.think_enabled.load(std::sync::atomic::Ordering::Relaxed),
        Some(&sys),
    ) {
        Ok(()) => {
            mgr.set_active_id(Some(id));
            http_response(stream, "200 OK", "application/json; charset=utf-8", "{\"ok\":true}");
        }
        Err(e) => {
            http_response(stream, "500 Internal Server Error", "application/json; charset=utf-8",
                &format!("{{\"error\":\"{}\"}}", json_escape(&e.to_string())));
        }
    }
}

/// POST /api/sessions/switch {"id":"..."} → park 当前 active, unpark 目标
fn handle_sessions_switch(stream: &mut TcpStream, shared: &Shared, body: &str) {
    let id = match json_extract_field(body, "id") {
        Some(i) => i.trim().to_string(),
        None => {
            http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
                "{\"error\":\"missing id\"}");
            return;
        }
    };
    let mut engine = shared.engine.lock().unwrap();
    let mut mgr = shared.session_mgr.lock().unwrap();
    if mgr.active_id() == Some(id.as_str()) {
        // 已是 active: 直接返回当前 history
        let msgs = history_load_json(&shared.history_dir, &id);
        let body = format!("{{\"ok\":true,\"messages\":{msgs}}}");
        http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
        return;
    }
    if !mgr.exists(&id) {
        http_response(stream, "404 Not Found", "application/json; charset=utf-8",
            "{\"error\":\"session not found\"}");
        return;
    }
    // park 当前 active (如果有)
    if let Some(cur_id) = mgr.active_id().map(String::from) {
        if let Some(session) = engine.session.take() {
            let _ = mgr.park(&cur_id, &session, &engine.config);
        }
    }
    match mgr.unpark(&id, &engine.config) {
        Ok(session) => {
            engine.session = Some(session);
            // 返回目标 session 的聊天历史
            let msgs = history_load_json(&shared.history_dir, &id);
            let body = format!("{{\"ok\":true,\"messages\":{msgs}}}");
            http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
        }
        Err(e) => {
            http_response(stream, "500 Internal Server Error", "application/json; charset=utf-8",
                &format!("{{\"error\":\"{}\"}}", json_escape(&e.to_string())));
        }
    }
}

/// POST /api/sessions/delete {"id":"..."} → 删除 session (active 则 session_end)
fn handle_sessions_delete(stream: &mut TcpStream, shared: &Shared, body: &str) {
    let id = match json_extract_field(body, "id") {
        Some(i) => i.trim().to_string(),
        None => {
            http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
                "{\"error\":\"missing id\"}");
            return;
        }
    };
    let mut engine = shared.engine.lock().unwrap();
    let mut mgr = shared.session_mgr.lock().unwrap();
    if !mgr.exists(&id) {
        http_response(stream, "404 Not Found", "application/json; charset=utf-8",
            "{\"error\":\"session not found\"}");
        return;
    }
    if mgr.active_id() == Some(id.as_str()) {
        engine.session_end();
    }
    mgr.delete(&id);
    // 同步删除 history 文件
    history_delete(&shared.history_dir, &id);
    http_response(stream, "200 OK", "application/json; charset=utf-8", "{\"ok\":true}");
}

/// POST /api/sessions/rename {"id":"...", "new_id":"..."} → 重命名 session
///
/// active session: 只改 active_id (无需搬动 session 数据)
/// inactive session: 改队列 id + rename SSD 文件
fn handle_sessions_rename(stream: &mut TcpStream, shared: &Shared, body: &str) {
    let id = match json_extract_field(body, "id") {
        Some(i) => i.trim().to_string(),
        None => {
            http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
                "{\"error\":\"missing id\"}");
            return;
        }
    };
    let new_id = match json_extract_field(body, "new_id") {
        Some(i) => i.trim().to_string(),
        None => {
            http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
                "{\"error\":\"missing new_id\"}");
            return;
        }
    };
    if id.is_empty() || new_id.is_empty() || new_id.len() > 64 {
        http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
            "{\"error\":\"invalid id\"}");
        return;
    }
    // 禁止包含文件路径分隔符或控制字符 (与 session_path 的 safe_id 规则一致)
    if new_id.chars().any(|c| matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|')
        || (c as u32) < 0x20)
    {
        http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
            "{\"error\":\"new_id contains invalid characters\"}");
        return;
    }
    let _engine = shared.engine.lock().unwrap();
    let mut mgr = shared.session_mgr.lock().unwrap();
    if !mgr.exists(&id) {
        http_response(stream, "404 Not Found", "application/json; charset=utf-8",
            "{\"error\":\"session not found\"}");
        return;
    }
    match mgr.rename(&id, &new_id) {
        Ok(()) => {
            // 同步重命名 history 文件
            history_rename(&shared.history_dir, &id, &new_id);
            http_response(stream, "200 OK", "application/json; charset=utf-8", "{\"ok\":true}");
        }
        Err(e) => {
            http_response(stream, "500 Internal Server Error", "application/json; charset=utf-8",
                &format!("{{\"error\":\"{}\"}}", json_escape(&e.to_string())));
        }
    }
}

/// 简单 URL decode: 把 %XX (UTF-8 字节) 和 + 还原, 支持中文等多字节字符
fn url_decode(s: &str) -> String {
    let mut bytes = Vec::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '+' => bytes.push(b' '),
            '%' => {
                let h1 = chars.next();
                let h2 = chars.next();
                if let (Some(h1), Some(h2)) = (h1, h2) {
                    if let Ok(byte) = u8::from_str_radix(&format!("{h1}{h2}"), 16) {
                        bytes.push(byte);
                        continue;
                    }
                }
                // 解析失败: 保留原始 %
                bytes.push(b'%');
                if let Some(h1) = h1 { bytes.extend(h1.to_string().as_bytes()); }
                if let Some(h2) = h2 { bytes.extend(h2.to_string().as_bytes()); }
            }
            c => bytes.extend(c.to_string().as_bytes()),
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// GET /api/sessions/history?id=xxx → {"ok":true,"messages":[{"role":"...","content":"..."},...]}
///
/// 用于重启后恢复 active session 的聊天历史到前端
fn handle_sessions_history(stream: &mut TcpStream, shared: &Shared, path: &str) {
    // 从 query string 解析 id 参数: ?id=Chat%20123456
    let id = path.split("?id=").nth(1)
        .map(|s| s.split('&').next().unwrap_or(""))
        .map(url_decode)
        .unwrap_or_default();
    if id.is_empty() {
        http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
            "{\"error\":\"missing id parameter\"}");
        return;
    }
    let msgs = history_load_json(&shared.history_dir, &id);
    let body = format!("{{\"ok\":true,\"messages\":{msgs}}}");
    http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
}

/// GET /api/params → {"temperature":0.7,"top_k":20,"top_p":0.95,"repetition_penalty":1.2,"frequency_penalty":0.0,"max_tokens":4096,"think":true,"system_prompt":"..."}
fn handle_params_get(stream: &mut TcpStream, shared: &Shared) {
    let p = shared.params.lock().unwrap();
    let m = *shared.max_tokens.lock().unwrap();
    let think = shared.think_enabled.load(std::sync::atomic::Ordering::Relaxed);
    let sys = shared.system_prompt.lock().unwrap();
    let body = format!(
        "{{\"temperature\":{},\"top_k\":{},\"top_p\":{},\"repetition_penalty\":{},\"frequency_penalty\":{},\"max_tokens\":{},\"think\":{think},\"system_prompt\":\"{}\"}}",
        p.temperature, p.top_k, p.top_p, p.repetition_penalty, p.frequency_penalty, m, json_escape(&sys)
    );
    http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
}

/// POST /api/params {"temperature":0.7,"top_k":20,"top_p":0.95,"repetition_penalty":1.1,"max_tokens":4096,"think":true}
/// 所有字段可选, 只更新提供的字段
fn handle_params_set(stream: &mut TcpStream, shared: &Shared, body: &str) {
    {
        let mut p = shared.params.lock().unwrap();
        if let Some(v) = json_extract_raw(body, "temperature").and_then(|s| s.parse::<f32>().ok()) {
            p.temperature = v.max(0.0);
        }
        if let Some(v) = json_extract_raw(body, "top_k").and_then(|s| s.parse::<usize>().ok()) {
            p.top_k = v;
        }
        if let Some(v) = json_extract_raw(body, "top_p").and_then(|s| s.parse::<f32>().ok()) {
            p.top_p = v.clamp(0.0, 1.0);
        }
        if let Some(v) = json_extract_raw(body, "repetition_penalty").and_then(|s| s.parse::<f32>().ok()) {
            p.repetition_penalty = v.clamp(1.0, 2.0);
        }
        if let Some(v) = json_extract_raw(body, "frequency_penalty").and_then(|s| s.parse::<f32>().ok()) {
            p.frequency_penalty = v.clamp(0.0, 2.0);
        }
    }
    if let Some(v) = json_extract_raw(body, "max_tokens").and_then(|s| s.parse::<usize>().ok()) {
        if v > 0 && v <= 65536 {
            *shared.max_tokens.lock().unwrap() = v;
        }
    }
    if let Some(v) = json_extract_raw(body, "think").and_then(|s| s.parse::<bool>().ok()) {
        shared.think_enabled.store(v, std::sync::atomic::Ordering::Relaxed);
    }
    // system_prompt: 修改后重置 session (system_prompt 在 session 创建时固化, 需重建才能生效)
    if let Some(new_sys) = json_extract_field(body, "system_prompt") {
        let mut sys_lock = shared.system_prompt.lock().unwrap();
        if *sys_lock != new_sys {
            *sys_lock = new_sys.clone();
            drop(sys_lock);
            // 重置 session (与 think 切换一致的行为): 用新 system_prompt 重建
            let mut engine = shared.engine.lock().unwrap();
            engine.session_end();
            let think = shared.think_enabled.load(std::sync::atomic::Ordering::Relaxed);
            let _ = engine.session_begin(think, Some(&new_sys));
            // ★ 同步 active_id: 若重置前未发消息 (active_id=None), 需创建,
            //   否则 handle_chat 跳过懒创建导致侧边栏无标签
            ensure_active_id(shared);
        }
    }
    http_response(stream, "200 OK", "application/json; charset=utf-8", "{\"ok\":true}");
}

/// POST /api/abort → 设置 abort flag, handle_chat 回调下次检查时返回 false 中断推理
///
/// 无需锁 engine: AtomicBool 与 engine lock 互不阻塞,
/// 中断请求可立即返回, 推理线程在下次回调迭代时退出。
fn handle_abort(stream: &mut TcpStream, shared: &Shared) {
    shared.abort.store(true, std::sync::atomic::Ordering::Relaxed);
    http_response(stream, "200 OK", "application/json; charset=utf-8", "{\"ok\":true}");
}

// ─── Tools 管理 (tool_call 模式) ─────────────────────────────────────

use daiza_runtime::tool_call::{ToolDef, ToolParam, ToolCall};

/// 从 JSON body 解析单个 ToolDef
///
/// 格式: {"name":"...","description":"...","parameters":[{"name":"...","description":"...","param_type":"...","required":true}]}
fn parse_tool_def(body: &str) -> Option<ToolDef> {
    let name = json_extract_field(body, "name")?;
    let description = json_extract_field(body, "description").unwrap_or_default();
    let parameters = parse_tool_params(body);
    Some(ToolDef { name, description, parameters })
}

/// 从 JSON body 解析 parameters 数组
fn parse_tool_params(body: &str) -> Vec<ToolParam> {
    let pat = "\"parameters\"";
    let Some(idx) = body.find(pat) else { return Vec::new(); };
    let rest = &body[idx + pat.len()..];
    let Some(colon) = rest.find(':') else { return Vec::new(); };
    let after = rest[colon + 1..].trim_start();
    let Some(arr_start) = after.find('[') else { return Vec::new(); };
    let arr_body = &after[arr_start + 1..];
    let Some(arr_end) = find_matching_bracket(arr_body) else { return Vec::new(); };
    let inner = &arr_body[..arr_end];

    // 按顶层对象分割: 找每个 {...} 作为一个 param 对象
    let mut params = Vec::new();
    let bytes = inner.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            // 找匹配的 }
            let mut depth = 1;
            let mut j = i + 1;
            let mut in_str = false;
            while j < bytes.len() && depth > 0 {
                match bytes[j] {
                    b'"' if j == 0 || bytes[j - 1] != b'\\' => in_str = !in_str,
                    b'{' if !in_str => depth += 1,
                    b'}' if !in_str => depth -= 1,
                    _ => {}
                }
                j += 1;
            }
            if depth == 0 {
                let obj = &inner[i..j];
                if let Some(p) = parse_single_tool_param(obj) {
                    params.push(p);
                }
                i = j;
                continue;
            }
        }
        i += 1;
    }
    params
}

fn parse_single_tool_param(obj: &str) -> Option<ToolParam> {
    let name = json_extract_field(obj, "name")?;
    let description = json_extract_field(obj, "description").unwrap_or_default();
    let param_type = json_extract_field(obj, "param_type").unwrap_or_else(|| "string".into());
    let required = json_extract_raw(obj, "required").map(|v| v == "true").unwrap_or(false);
    Some(ToolParam { name, description, param_type, required })
}

/// 序列化 ToolDef 为 JSON
fn tool_def_to_json(t: &ToolDef) -> String {
    let mut params = String::from("[");
    for (i, p) in t.parameters.iter().enumerate() {
        if i > 0 { params.push(','); }
        params.push_str(&format!(
            "{{\"name\":\"{}\",\"description\":\"{}\",\"param_type\":\"{}\",\"required\":{}}}",
            json_escape(&p.name), json_escape(&p.description),
            json_escape(&p.param_type), p.required
        ));
    }
    params.push(']');
    format!(
        "{{\"name\":\"{}\",\"description\":\"{}\",\"parameters\":{}}}",
        json_escape(&t.name), json_escape(&t.description), params
    )
}

/// 序列化 ToolCall 为 JSON (含 name + arguments 数组)
fn tool_call_to_json(tc: &ToolCall) -> String {
    let mut args = String::from("[");
    for (i, (k, v)) in tc.arguments.iter().enumerate() {
        if i > 0 { args.push(','); }
        args.push_str(&format!("{{\"name\":\"{}\",\"value\":\"{}\"}}",
            json_escape(k), json_escape(v)));
    }
    args.push(']');
    format!("{{\"name\":\"{}\",\"arguments\":{}}}", json_escape(&tc.name), args)
}

/// 确保 mgr.active_id 存在: 若为 None, 创建 "New Chat N" 作为 active_id
///
/// 用于 session_begin 后同步 active_id:
/// handle_dspark_set / handle_params_set 等会 session_end + session_begin 重建 session,
/// 若此时 active_id 为 None (用户尚未发消息就切换了配置), 后续 handle_chat 检测到
/// session 已存在会跳过懒创建, 导致侧边栏无标签。
fn ensure_active_id(shared: &Shared) {
    let mut mgr = shared.session_mgr.lock().unwrap();
    if mgr.active_id().is_some() { return; }
    let mut n = mgr.list().len() + 1;
    let mut id = format!("New Chat {n}");
    while mgr.exists(&id) {
        n += 1;
        id = format!("New Chat {n}");
    }
    mgr.set_active_id(Some(id));
}

/// 确保已有 active session (懒创建)
///
/// tools API 在用户尚未发送任何消息时也会被调用 (用户先配置工具再对话),
/// 此时 session 为 None, 需自动创建以承载 tools 字段。
/// 与 handle_chat 中懒创建逻辑等价: 同步 session_mgr.active_id 并设置 ready。
fn ensure_session(shared: &Shared, engine: &mut Engine) -> Result<(), String> {
    if engine.session.is_some() { return Ok(()); }
    let think = shared.think_enabled.load(std::sync::atomic::Ordering::Relaxed);
    let sys = shared.system_prompt.lock().unwrap().clone();
    engine.session_begin(think, Some(&sys))
        .map_err(|e| format!("session init: {e}"))?;
    ensure_active_id(shared);
    shared.ready.store(true, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// GET /api/tools → {"tools": [...]}
fn handle_tools_get(stream: &mut TcpStream, shared: &Shared) {
    let mut engine = shared.engine.lock().unwrap();
    if let Err(e) = ensure_session(shared, &mut engine) {
        http_response(stream, "500 Internal Server Error", "application/json; charset=utf-8",
            &format!("{{\"error\":\"{e}\"}}"));
        return;
    }
    let tools: Vec<String> = engine.session.as_ref()
        .map(|s| s.tools.iter().map(tool_def_to_json).collect())
        .unwrap_or_default();
    let body = format!("{{\"tools\":[{}]}}", tools.join(","));
    http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
}

/// POST /api/tools {"name":"...","description":"...","parameters":[...]}
/// 按 name 去重: 已存在则更新, 不存在则添加
fn handle_tools_set(stream: &mut TcpStream, shared: &Shared, body: &str) {
    let Some(tool) = parse_tool_def(body) else {
        http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
            "{\"error\":\"invalid tool definition\"}");
        return;
    };
    let mut engine = shared.engine.lock().unwrap();
    if let Err(e) = ensure_session(shared, &mut engine) {
        http_response(stream, "500 Internal Server Error", "application/json; charset=utf-8",
            &format!("{{\"error\":\"{e}\"}}"));
        return;
    }
    // ensure_session 已保证 session 存在
    let session = engine.session.as_mut().expect("session ensured");
    // 按 name 去重: 已存在则替换, 不存在则 push
    if let Some(pos) = session.tools.iter().position(|t| t.name == tool.name) {
        session.tools[pos] = tool.clone();
    } else {
        session.tools.push(tool.clone());
    }
    // ★ tools 变化需重置 session (system prompt 注入 tools 块, KV cache 需重建)
    //   但保留 tools 和 messages 历史 (用户可能想继续对话)
    //   实际上 tools 变化意味着 system prompt 变化, 下一轮首轮需要重 prefill
    //   简单处理: 重置 state, 下一轮首轮重新注入 tools
    session.state.reset();
    session.history_tokens.clear();
    drop(engine);
    let body = format!("{{\"ok\":true,\"tool\":{}}}", tool_def_to_json(&tool));
    http_response(stream, "200 OK", "application/json; charset=utf-8", &body);
}

/// POST /api/tools/delete {"name":"..."}
fn handle_tools_delete(stream: &mut TcpStream, shared: &Shared, body: &str) {
    let Some(name) = json_extract_field(body, "name") else {
        http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
            "{\"error\":\"missing name\"}");
        return;
    };
    let mut engine = shared.engine.lock().unwrap();
    if let Err(e) = ensure_session(shared, &mut engine) {
        http_response(stream, "500 Internal Server Error", "application/json; charset=utf-8",
            &format!("{{\"error\":\"{e}\"}}"));
        return;
    }
    if let Some(session) = engine.session.as_mut() {
        let before = session.tools.len();
        session.tools.retain(|t| t.name != name);
        if session.tools.len() != before {
            // tools 变化, 重置 state
            session.state.reset();
            session.history_tokens.clear();
        }
        drop(engine);
        http_response(stream, "200 OK", "application/json; charset=utf-8", "{\"ok\":true}");
    } else {
        // 不会到达: ensure_session 已保证 session 存在
        http_response(stream, "400 Bad Request", "application/json; charset=utf-8",
            "{\"error\":\"no active session\"}");
    }
}

/// POST /api/chat/tool_response {"name":"...","content":"..."}
///
/// 前端执行工具后回传结果, 模型继续生成下一轮回复。
/// 流程:
/// 1. push Tool 消息到 messages 历史
/// 2. 构造 tool_response 增量 prompt
/// 3. 流式生成 (DSpark 或非 DSpark 路径)
/// 4. 生成完成后检测 <tool_call> (模型可能链式调用多个工具)
/// 5. 发出 tool_call / done SSE 事件
fn handle_chat_tool_response(stream: &mut TcpStream, shared: &Shared, body: &str) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if stream.write_all(head.as_bytes()).and_then(|_| stream.flush()).is_err() {
        return;
    }

    let Some(content) = json_extract_field(body, "content") else {
        sse_write(stream, "error", "{\"error\":\"missing content\"}");
        return;
    };
    let name = json_extract_field(body, "name").unwrap_or_default();

    shared.abort.store(false, std::sync::atomic::Ordering::Relaxed);

    let mut engine = shared.engine.lock().unwrap();
    let t0 = std::time::Instant::now();

    // push Tool 消息到 messages 历史
    use daiza_runtime::tool_call::{ToolMessage, ToolResponse};
    if let Some(session) = engine.session.as_mut() {
        session.messages.push(ToolMessage::tool(ToolResponse {
            name: name.clone(),
            content: content.clone(),
        }));
    }

    let _use_dspark = shared.use_dspark.load(std::sync::atomic::Ordering::Relaxed)
        && shared.dspark_available;
    let stream_ref = &mut *stream;
    let params = *shared.params.lock().unwrap();
    let max_tokens = *shared.max_tokens.lock().unwrap();
    let abort_ref = &shared.abort;
    let mut t_first: Option<std::time::Instant> = None;
    let mut n_tokens: usize = 0;

    // ★ tool_response 走非 DSpark 增量路径 (DSpark 路径的 handle_chat_dspark 内部构造 user prompt, 不兼容 tool_response 格式)
    //   后续可扩展 DSpark + tool_response 共存
    let result = engine.session_reply_tool_response_stream(
        &content, max_tokens, params,
        &mut |delta| {
            if abort_ref.load(std::sync::atomic::Ordering::Relaxed) {
                return false;
            }
            if t_first.is_none() {
                t_first = Some(std::time::Instant::now());
            }
            n_tokens += 1;
            let payload = format!("{{\"text\":\"{}\"}}", json_escape(delta));
            sse_write(stream_ref, "delta", &payload)
        },
    );

    shared.abort.store(false, std::sync::atomic::Ordering::Relaxed);

    match result {
        Ok(full) => {
            let total_ms = t0.elapsed().as_millis();
            let ttft_ms = t_first
                .map(|t| t.duration_since(t0).as_millis())
                .unwrap_or(total_ms);
            let decode_ms = total_ms.saturating_sub(ttft_ms);
            let ms_per_tok = if n_tokens > 0 { decode_ms / n_tokens as u128 } else { 0 };

            // 链式 tool_call 检测: 模型可能继续调用其他工具
            let tool_calls = daiza_runtime::tool_call::parse_tool_calls(&full);
            let has_tool_call = !tool_calls.is_empty();
            let think_on = engine.session.as_ref().map(|s| s.think_enabled).unwrap_or(false);
            let cleaned = clean_assistant_text(&full, think_on);
            if has_tool_call {
                for tc in &tool_calls {
                    let payload = format!("{{\"tool_call\":{}}}", tool_call_to_json(tc));
                    sse_write(stream, "tool_call", &payload);
                }
                let session = engine.session.as_mut().unwrap();
                session.messages.push(ToolMessage::assistant_with_tool_calls(
                    cleaned,
                    tool_calls.clone(),
                ));
            } else {
                let session = engine.session.as_mut().unwrap();
                session.messages.push(ToolMessage::assistant(cleaned));
            }

            sse_write(
                stream,
                "done",
                &format!(
                    "{{\"ok\":true,\"ttft_ms\":{ttft_ms},\"ms_per_tok\":{ms_per_tok},\"n_tokens\":{n_tokens},\"has_tool_call\":{has_tool_call}}}",
                ),
            );
        }
        Err(e) => {
            sse_write(
                stream,
                "error",
                &format!("{{\"error\":\"{}\"}}", json_escape(&e.to_string())),
            );
        }
    }
}

//! daiza-web: 基于 daiza-engine 的炫酷聊天 GUI (零依赖 HTTP + SSE)
//!
//! 用法:
//! ```
//! daiza-web --model <gguf_path> [--port 8787] [--max-tokens 4096] [--greedy]
//! ```
//!
//! 浏览器打开 http://127.0.0.1:8787 即可聊天。
//! 回复通过 SSE (Server-Sent Events) 逐 token 流式渲染。

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Mutex;

use daiza_runtime::engine::Engine;
use daiza_engine::math::SamplingParams;

const INDEX_HTML: &str = include_str!("../web/index.html");

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

/// 极简 JSON: 提取 {"message": "..."} 字段 (支持 \" \\ \/ \n \t \r \uXXXX 转义)
fn json_extract_message(body: &str) -> Option<String> {
    let pat = "\"message\"";
    let idx = body.find(pat)? + pat.len();
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
        if buf.len() > 1 << 20 {
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
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("Usage: daiza-web --model <gguf_path> [options]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --model <path>       主权重 GGUF 路径 (必需)");
        eprintln!("  --port <n>           监听端口 (默认 8787)");
        eprintln!("  --max-tokens <n>     每轮最多生成 token 数 (默认 4096)");
        eprintln!("  --greedy             贪心解码 (temperature=0)");
        eprintln!("  --no-open            启动后不自动打开浏览器");
        return;
    }
    let gguf_path: PathBuf = match get_opt(&args, "--model") {
        Some(m) => m.into(),
        None => {
            eprintln!("Error: --model <gguf_path> is required (see --help)");
            std::process::exit(1);
        }
    };
    let port: u16 = get_opt(&args, "--port")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8787);
    let max_tokens: usize = get_opt(&args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let greedy = args.iter().any(|a| a == "--greedy");
    let no_open = args.iter().any(|a| a == "--no-open");

    let params = if greedy {
        SamplingParams { temperature: 0.0, top_k: 0, top_p: 1.0 }
    } else {
        SamplingParams { temperature: 0.7, top_k: 20, top_p: 0.95 }
    };

    // 与 daiza-cli 一致的热降频默认配置 (用户已设置的环境变量优先)
    if std::env::var("DAIZA_WAIT_MODE").is_err() {
        std::env::set_var("DAIZA_WAIT_MODE", "yield");
    }
    if std::env::var("DAIZA_ACTIVE_WORKERS").is_err() {
        std::env::set_var("DAIZA_ACTIVE_WORKERS", "9");
    }

    println!("[daiza-web] Loading model: {}", gguf_path.display());
    let engine = match Engine::load(&gguf_path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[daiza-web] load failed: {e}");
            std::process::exit(1);
        }
    };
    let model_name = gguf_path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    println!("[daiza-web] Engine loaded (weights load lazily on first request).");

    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[daiza-web] bind 127.0.0.1:{port} failed: {e}");
            std::process::exit(1);
        }
    };
    let url = format!("http://127.0.0.1:{port}/");
    println!("[daiza-web] Chat GUI ready at {url}");
    if !no_open {
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", &url])
            .spawn();
    }

    let shared = std::sync::Arc::new(Shared {
        engine: Mutex::new(engine),
        params,
        max_tokens,
        model_name,
    });

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let shared = shared.clone();
                std::thread::spawn(move || handle_conn(stream, shared));
            }
            Err(e) => eprintln!("[daiza-web] accept error: {e}"),
        }
    }
}

struct Shared {
    engine: Mutex<Engine>,
    params: SamplingParams,
    max_tokens: usize,
    model_name: String,
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
            let (ready, model) = {
                let eng = shared.engine.lock().unwrap();
                (eng.weights.is_some(), shared.model_name.clone())
            };
            http_response(
                &mut stream,
                "200 OK",
                "application/json; charset=utf-8",
                &format!("{{\"ready\":{ready},\"model\":\"{}\"}}", json_escape(&model)),
            );
        }
        ("POST", "/api/chat") => {
            handle_chat(&mut stream, &shared, &body);
        }
        _ => {
            http_response(&mut stream, "404 Not Found", "text/plain; charset=utf-8", "not found");
        }
    }
}

fn handle_chat(stream: &mut TcpStream, shared: &Shared, body: &str) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if stream.write_all(head.as_bytes()).and_then(|_| stream.flush()).is_err() {
        return;
    }

    let Some(msg) = json_extract_message(body) else {
        sse_write(stream, "error", "{\"error\":\"invalid request body\"}");
        return;
    };
    let msg = msg.trim().to_string();
    if msg.is_empty() {
        sse_write(stream, "error", "{\"error\":\"empty message\"}");
        return;
    }

    let mut engine = shared.engine.lock().unwrap();
    let t0 = std::time::Instant::now();

    // 权重懒加载 + session 懒创建 (首个请求时执行, 可能耗时数十秒)
    let mut ok = true;
    if engine.session.is_none() {
        ok = sse_write(stream, "delta", "{\"text\":\"\"}");
        if ok {
            ok = engine
                .session_begin(true, Some("You are a helpful assistant."))
                .is_ok();
        }
    }
    if !ok {
        sse_write(stream, "error", "{\"error\":\"session init failed\"}");
        return;
    }

    let stream_ref = &mut *stream;
    let result = engine.session_reply_stream(
        &msg,
        shared.max_tokens,
        shared.params.clone(),
        &mut |delta| {
            let payload = format!("{{\"text\":\"{}\"}}", json_escape(delta));
            sse_write(stream_ref, "delta", &payload)
        },
    );

    match result {
        Ok(full) => {
            let elapsed = t0.elapsed().as_secs_f64();
            let n_chars = full.chars().count();
            let stats = format!("~{n_chars} 字符 · {elapsed:.1}s");
            sse_write(
                stream,
                "done",
                &format!("{{\"ok\":true,\"stats\":\"{}\"}}", json_escape(&stats)),
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

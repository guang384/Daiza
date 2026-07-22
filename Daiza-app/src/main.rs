//! daiza-app: Tauri 桌面客户端 (内嵌 WebView2 渲染 daiza-web 聊天页)
//!
//! 架构:
//!   后台线程跑 daiza-web HTTP/SSE 服务 (127.0.0.1:8787),
//!   Tauri 窗口先显示启动页, 等服务就绪后自动切换到聊天页。
//!   聊天请求走 Tauri command (Rust 侧 ureq 流式转发 SSE → emit 到前端),
//!   避开 WebView2 fetch 对 127.0.0.1 的 mixed-content 拦截。
//!
//! 构建 (需要 feature app):
//! ```text
//! cargo tauri dev -- --features app          # 开发模式
//! cargo tauri build -- --features app        # 发布 (dist/daiza-app.exe + NSIS 安装包)
//! ```

use std::io::Read;
use std::sync::Mutex;

use base64::Engine as _;
use tauri::{Emitter, Manager};

const BOOT_HTML: &str = include_str!("../web/boot.html");

struct Backend {
    child: Mutex<Option<std::process::Child>>,
}

#[derive(Clone, serde::Serialize)]
struct ChatEvent {
    kind: String,
    text: String,
}

/// 转发一次聊天请求: ureq POST → 逐字节读 SSE → 解析事件 → emit 到前端
#[tauri::command]
fn chat_send(app: tauri::AppHandle, message: String) -> Result<(), String> {
    let body = format!("{{\"message\":\"{}\"}}", json_escape(&message));
    let resp = ureq::post("http://127.0.0.1:8787/api/chat")
        .header("Content-Type", "application/json")
        .send(body.as_bytes())
        .map_err(|e| format!("backend request failed: {e}"))?;
    let mut reader = resp.into_body().into_reader();
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let mut byte = [0u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => break,
            Ok(_) => {
                buf.push(byte[0]);
                if buf.ends_with(b"\n\n") {
                    handle_sse_block(&app, &buf);
                    buf.clear();
                }
                if buf.len() > 1 << 20 {
                    return Err("SSE frame too large".into());
                }
            }
            Err(e) => return Err(format!("stream read failed: {e}")),
        }
    }
    if !buf.is_empty() {
        handle_sse_block(&app, &buf);
    }
    Ok(())
}

/// 解析一个 SSE 块 ("event: x\ndata: {...}\n\n") 并 emit
fn handle_sse_block(app: &tauri::AppHandle, block: &[u8]) {
    let text = String::from_utf8_lossy(block);
    let mut kind = "delta".to_string();
    let mut data = String::new();
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("event:") {
            kind = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("data:") {
            data.push_str(v.trim());
        }
    }
    if data.is_empty() {
        return;
    }
    let payload = if kind == "delta" {
        // {"text":"..."} → 提取 text (后端已 JSON 转义, 这里反转义)
        json_extract_field(&data, "text").unwrap_or_default()
    } else if kind == "done" {
        json_extract_field(&data, "stats").unwrap_or_default()
    } else {
        json_extract_field(&data, "error").unwrap_or_else(|| "unknown error".into())
    };
    let _ = app.emit("chat", ChatEvent { kind, text: payload });
}

/// 极简 JSON 字符串字段提取 ({"key":"value"}, 支持 \" \\ \n \t \r \uXXXX 转义)
fn json_extract_field(json: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let idx = json.find(&pat)? + pat.len();
    let rest = &json[idx..];
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
    if b < 0x80 { 1 } else if b < 0xE0 { 2 } else if b < 0xF0 { 3 } else { 4 }
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

/// 启动后台 daiza-web 进程 (带 15s 超时, 失败返回 Err)
#[tauri::command]
fn start_backend(app: tauri::AppHandle) -> Result<(), String> {
    {
        let state = app.state::<Backend>();
        if state.child.lock().unwrap().is_some() {
            return Ok(());
        }
    }
    let exe = std::env::current_exe()
        .map_err(|e| e.to_string())?
        .parent()
        .ok_or("no exe dir")?
        .join("daiza-web.exe");
    if !exe.exists() {
        return Err(format!("daiza-web.exe not found at {}", exe.display()));
    }
    // 模型路径解析顺序: --model 参数 > DAIZA_MODEL 环境变量 > 默认相对路径
    let args: Vec<String> = std::env::args().collect();
    let model = args
        .windows(2)
        .find(|w| w[0] == "--model")
        .map(|w| w[1].clone())
        .or_else(|| std::env::var("DAIZA_MODEL").ok())
        .unwrap_or_else(|| "../../Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf".to_string());
    let child = std::process::Command::new(exe)
        .args(["--model", &model, "--no-open"])
        .current_dir("..")
        .spawn()
        .map_err(|e| format!("spawn daiza-web failed: {e}"))?;
    let state = app.state::<Backend>();
    *state.child.lock().unwrap() = Some(child);
    Ok(())
}

/// 窗口退出时杀掉后台 daiza-web
fn kill_backend(app: &tauri::AppHandle) {
    let state = app.state::<Backend>();
    let child = state.child.lock().unwrap().take();
    if let Some(mut c) = child {
        let _ = c.kill();
    }
}

fn main() {
    tauri::Builder::default()
        .manage(Backend { child: Mutex::new(None) })
        .invoke_handler(tauri::generate_handler![chat_send, start_backend])
        .setup(|app| {
            let window = app.get_webview_window("main").unwrap();
            // 启动页: base64 内联, 服务就绪后前端 JS 调 start_backend + location 切换
            let b64 = base64::engine::general_purpose::STANDARD.encode(BOOT_HTML);
            window.navigate(format!("data:text/html;base64,{b64}").parse().unwrap()).unwrap();
            let w = window.clone();
            window.on_window_event(move |ev| {
                if let tauri::WindowEvent::Destroyed = ev {
                    kill_backend(&w.app_handle());
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running daiza-app");
}

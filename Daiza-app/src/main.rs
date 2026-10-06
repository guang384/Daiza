//! daiza-app: Tauri 桌面客户端 (内嵌 daiza-web HTTP/SSE 服务, 单 exe 便携版)
//!
//! 架构:
//!   start_backend 同步加载 Engine (mmap, <1s), 然后 spawn 后台线程跑 run_server
//!   (load_drafter + load_mmproj + accept 循环, 阻塞该线程)。
//!   Tauri 窗口先显示启动页, 等服务就绪后自动切换到聊天页。
//!   聊天请求走 Tauri command (Rust 侧 ureq 流式转发 SSE → emit 到前端),
//!   避开 WebView2 fetch 对 127.0.0.1 的 mixed-content 拦截。
//!
//! 构建 (需要 feature app):
//! ```text
//! cargo tauri dev -- --features app          # 开发模式
//! cargo tauri build -- --features app        # 发布 (单 exe 便携版)
//! ```

// 在 release 模式下隐藏 Windows 控制台窗口 (debug 模式保留控制台方便看日志)
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};

use tauri::{Emitter, Manager};

use daiza_engine::math::SamplingParams;
use daiza_runtime::engine::Engine;
use daiza_web::{run_server_with_listener, ServerConfig};

struct Backend {
    /// 后台 HTTP server 是否已启动 (防止重复 start_backend)
    running: AtomicBool,
    /// chat 中断标志: abort_chat 设置后, chat_send 循环下次迭代时退出
    abort: AtomicBool,
}

#[derive(Clone, serde::Serialize)]
struct ChatEvent {
    kind: String,
    /// 原始 JSON data (前端按 kind 解析: delta/draft→text, reject→accepted, done→stats, error→error)
    data: String,
}

#[derive(Clone, serde::Serialize)]
struct BackendError {
    error: String,
}

#[derive(Clone, serde::Serialize)]
struct BootProgress {
    stage: String,
    message: String,
}

fn boot_progress(app: &tauri::AppHandle, stage: &str, message: &str) {
    let _ = app.emit("boot_progress", BootProgress {
        stage: stage.into(),
        message: message.into(),
    });
}

/// 转发一次聊天请求: ureq POST → BufReader 读 SSE → 解析事件 → emit 到前端
///
/// `images`: 图片路径数组 (前端已通过 /api/upload 上传, 这里只传路径给后端)
#[tauri::command]
fn chat_send(app: tauri::AppHandle, message: String, images: Option<Vec<String>>) -> Result<(), String> {
    let state = app.state::<Backend>();
    state.abort.store(false, Ordering::SeqCst);
    // 构造请求体: {"message":"...","images":["p1","p2",...]}
    // images 为空时省略 images 字段 (后端 json_extract_string_array 返回空 Vec)
    let body = if let Some(imgs) = images.as_ref().filter(|v| !v.is_empty()) {
        let arr: Vec<String> = imgs.iter().map(|p| format!("\"{}\"", json_escape(p))).collect();
        format!("{{\"message\":\"{}\",\"images\":[{}]}}", json_escape(&message), arr.join(","))
    } else {
        format!("{{\"message\":\"{}\"}}", json_escape(&message))
    };
    let resp = ureq::post("http://127.0.0.1:8787/api/chat")
        .header("Content-Type", "application/json")
        .send(body.as_bytes())
        .map_err(|e| format!("backend request failed: {e}"))?;
    let reader = resp.into_body().into_reader();
    let mut buf_reader = std::io::BufReader::new(reader);
    let mut event_buf: Vec<u8> = Vec::with_capacity(4096);
    let mut line: Vec<u8> = Vec::with_capacity(512);
    loop {
        if state.abort.load(Ordering::SeqCst) {
            let _ = app.emit("chat", ChatEvent { kind: "aborted".into(), data: String::new() });
            return Ok(());
        }
        line.clear();
        let n = buf_reader.read_until(b'\n', &mut line)
            .map_err(|e| format!("stream read failed: {e}"))?;
        if n == 0 {
            if !event_buf.is_empty() {
                handle_sse_block(&app, &event_buf);
            }
            break;
        }
        // 空行 = SSE 事件边界
        if line == b"\n" || line == b"\r\n" {
            if !event_buf.is_empty() {
                handle_sse_block(&app, &event_buf);
                event_buf.clear();
            }
        } else {
            event_buf.extend_from_slice(&line);
        }
        if event_buf.len() > 1 << 22 {
            return Err("SSE frame too large".into());
        }
    }
    Ok(())
}

/// 弹出原生文件选择框 (过滤 .gguf 文件)
///
/// `title`: 对话框标题 (如 "选择主权重 GGUF" / "选择 DSpark drafter GGUF")
/// 返回选中文件的完整路径, 用户取消则返回 None
///
/// 用 spawn_blocking 在独立线程运行同步 FileDialog (避免阻塞 Tauri async runtime,
/// Win32 模态对话框自带消息泵, 不依赖主线程)
#[tauri::command]
async fn pick_file(title: String) -> Option<String> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut dialog = rfd::FileDialog::new()
            .set_title(&title)
            .add_filter("GGUF model files (*.gguf)", &["gguf"]);
        // 默认目录: exe 所在目录 (便携版用户通常把 gguf 放在 exe 旁边)
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                dialog = dialog.set_directory(dir);
            }
        }
        dialog.pick_file().map(|p| p.to_string_lossy().into_owned())
    })
    .await
    .ok()
    .flatten()
}

/// 中断当前 chat_send (设置本地 abort flag + 通知后端中断推理)
///
/// 两步中断:
///   1. 本地 flag: chat_send SSE 读取循环下次迭代退出
///   2. POST /api/abort: 后端 handle_chat 回调下次检查时返回 false, 推理循环退出
#[tauri::command]
fn abort_chat(app: tauri::AppHandle) -> Result<(), String> {
    let state = app.state::<Backend>();
    state.abort.store(true, Ordering::SeqCst);
    // 通知后端立即中断推理 (后端 handle_abort 立即返回, 无阻塞风险)
    let _ = ureq::post("http://127.0.0.1:8787/api/abort").send(b"");
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
    // 直接传原始 JSON data, 前端按 kind 解析 (delta/draft→text, reject→accepted, done→stats, error→error)
    let _ = app.emit("chat", ChatEvent { kind, data });
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

/// 启动内嵌 HTTP server (单 exe 便携版)
///
/// 模型路径解析顺序: 前端传入 modelPath > DAIZA_MODEL 环境变量
/// DSpark/mmproj 路径: 前端传入 (可选)
///
/// 流程:
///   1. 同步 Engine::load (mmap + metadata, <1s), 失败立即返回 Err
///   2. spawn 后台线程: load_drafter + load_mmproj + run_server (阻塞)
///   3. 立即返回 Ok, 前端轮询 /api/status 直到 ready
#[tauri::command]
fn start_backend(
    app: tauri::AppHandle,
    model_path: Option<String>,
    dspark_path: Option<String>,
    mmproj_path: Option<String>,
) -> Result<(), String> {
    {
        let state = app.state::<Backend>();
        if state.running.load(Ordering::SeqCst) {
            return Ok(());
        }
    }

    let model = model_path
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .or_else(|| std::env::var("DAIZA_MODEL").ok())
        .ok_or_else(|| "model path not provided".to_string())?;

    // 与 daiza-cli 一致的热降频默认配置 (用户已设置的环境变量优先)
    if std::env::var("DAIZA_WAIT_MODE").is_err() {
        std::env::set_var("DAIZA_WAIT_MODE", "yield");
    }
    if std::env::var("DAIZA_ACTIVE_WORKERS").is_err() {
        std::env::set_var("DAIZA_ACTIVE_WORKERS", "9");
    }

    eprintln!("[daiza-app] Loading model: {model}");
    boot_progress(&app, "loading_metadata", "Parsing model metadata…");
    // 同步加载 Engine (mmap + metadata 解析, <1s)
    let mut engine = Engine::load(std::path::Path::new(&model))
        .map_err(|e| format!("engine load failed: {e}"))?;
    let model_name = std::path::Path::new(&model)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    eprintln!("[daiza-app] Engine metadata loaded. Loading weights (~13GB)…");
    boot_progress(&app, "loading_weights", "Loading model weights (~13GB)…");
    // 同步加载权重 + 初始化线程池 (在 bind listener 之前完成, 避免首次 chat 卡顿)
    engine.load_weights()
        .map_err(|e| format!("load_weights failed: {e}"))?;
    let n_threads = daiza_engine::model::workspace::thread_count();
    daiza_engine::model::workspace::init_thread_pool(n_threads);
    eprintln!("[daiza-app] Weights loaded ({n_threads} workers). Engine ready.");

    // 同步加载 DSpark drafter (在 bind listener 之前完成, 避免 run_server 启动延迟)
    let dspark_available = if let Some(dp) = dspark_path.as_ref().filter(|s| !s.trim().is_empty()) {
        boot_progress(&app, "loading_dspark", "Loading DSpark drafter…");
        match engine.load_drafter(std::path::Path::new(dp.trim())) {
            Ok(()) => { eprintln!("[daiza-app] DSpark drafter loaded."); true }
            Err(e) => { eprintln!("[daiza-app] DSpark drafter load failed: {e}"); false }
        }
    } else {
        false
    };

    // 同步加载多模态视觉编码器
    let vision_available = if let Some(mp) = mmproj_path.as_ref().filter(|s| !s.trim().is_empty()) {
        boot_progress(&app, "loading_mmproj", "Loading vision encoder…");
        match engine.load_mmproj(std::path::Path::new(mp.trim())) {
            Ok(()) => { eprintln!("[daiza-app] mmproj (vision encoder) loaded."); true }
            Err(e) => { eprintln!("[daiza-app] mmproj load failed: {e}"); false }
        }
    } else {
        false
    };

    boot_progress(&app, "starting_server", "Starting HTTP server…");

    // 预检 8787 端口: 被占用时立即返回明确错误, 避免前端轮询 30s 超时
    // listener 不 drop, 直接传给后台线程 run_server_with_listener, 无竞态
    let port = 8787;
    let listener = std::net::TcpListener::bind(("127.0.0.1", port)).map_err(|e| {
        format!(
            "Port {port} is in use or cannot be bound ({e}).\nPlease use Task Manager to end the process occupying port {port} and retry."
        )
    })?;

    let params = SamplingParams { temperature: 0.7, top_k: 20, top_p: 0.95, repetition_penalty: 1.3, frequency_penalty: 0.4 };

    app.state::<Backend>().running.store(true, Ordering::SeqCst);

    // 后台线程: 立即 run_server_with_listener 开始 accept (加载已在主线程完成)
    // ready_rx: 后台线程进入 accept 循环前发送 (), 主线程等待此信号才返回 Ok
    // err_rx: 后台线程出错时发送错误信息, 主线程立即返回 Err
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
    let (err_tx, err_rx) = std::sync::mpsc::channel::<String>();
    let app_handle = app.clone();
    std::thread::spawn(move || {
        let cfg = ServerConfig {
            engine,
            model_name,
            params,
            max_tokens: 4096,
            port: 8787,
            no_open: true, // Tauri 内嵌不自动打开浏览器
            dspark_available,
            vision_available,
        };

        eprintln!("[daiza-web] Chat GUI ready at http://127.0.0.1:8787/");
        if let Err(e) = run_server_with_listener(cfg, listener, Some(ready_tx)) {
            eprintln!("[daiza-app] server error: {e}");
            let _ = err_tx.send(e.to_string());
            let _ = app_handle.emit("backend_error", BackendError { error: e.to_string() });
        }
    });

    // 等待后台线程确认进入 accept 循环 (ready_rx) 或出错 (err_rx) — 单次等待, 无重试
    // 这是唯一可靠的就绪检测: TCP connect 成功只代表 listener 已 bind,
    // 不代表后端在 accept (backlog 会自动完成三次握手)
    match ready_rx.recv_timeout(std::time::Duration::from_secs(10)) {
        Ok(()) => {
            eprintln!("[daiza-app] Backend accept loop ready.");
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            return Err("Backend service startup timed out (10s), please check logs".to_string());
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            // 后台线程已退出, 检查是否留下了错误信息
            return Err(match err_rx.try_recv() {
                Ok(err) => format!("Backend service failed to start: {err}"),
                Err(_) => "Backend thread exited unexpectedly".to_string(),
            });
        }
    }

    // 额外验证: 用 ureq 实际请求 /api/status, 确认后端能处理 HTTP 请求
    // (ready signal 只确认进入 accept 循环, 不确认 handle_conn 正常工作)
    match ureq::get("http://127.0.0.1:8787/api/status").call() {
        Ok(resp) => {
            let body = resp.into_body().read_to_string()
                .map_err(|e| format!("Failed to read response: {e}"))?;
            eprintln!("[daiza-app] Backend verification OK: {body}");
        }
        Err(e) => {
            return Err(format!(
                "Backend verification failed: cannot request /api/status ({e}).\nBackend may be running but unable to handle requests."
            ));
        }
    }

    // 主窗口直接导航到聊天页 http://127.0.0.1:8787/
    // (放弃 iframe 方案: WebView2 对 iframe 加载 127.0.0.1 有限制)
    // (放弃新窗口方案: 用户体验差, 会出现两个窗口)
    // 主窗口导航后 window.__TAURI__ 消失, 聊天页自动走浏览器模式 (直接 HTTP fetch + SSE)
    if let Some(main_window) = app.get_webview_window("main") {
        match main_window.eval("window.location.replace('http://127.0.0.1:8787/')") {
            Ok(_) => eprintln!("[daiza-app] Main window navigating to chat UI."),
            Err(e) => return Err(format!("Failed to navigate to chat UI: {e}")),
        }
    }

    Ok(())
}

/// 安装 panic hook: 将 panic 信息 + backtrace 写入 exe 同目录的 panic.log
///
/// ★ panic = "abort" 下, panic 直接终止进程 (Tauri 窗口闪退, 无任何输出)。
///   此 hook 在 abort 前将 panic 详情写入文件, 便于事后定位崩溃根因。
fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = info.payload();
        let msg = payload.downcast_ref::<&str>().copied()
            .or_else(|| payload.downcast_ref::<String>().map(|s| s.as_str()))
            .unwrap_or("<non-string panic payload>");
        let location = info.location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown location>".to_string());
        let bt = std::backtrace::Backtrace::force_capture();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let log = format!(
            "[daiza-app panic]\nTime: epoch={now}s\nMessage: {msg}\nLocation: {location}\nThread: {:?}\n\nBacktrace:\n{bt}\n",
            std::thread::current().name().unwrap_or("<unnamed>"),
        );
        // 写入 exe 同目录下的 panic.log (追加模式, 保留多次崩溃记录)
        let log_path = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("panic.log")))
            .unwrap_or_else(|| std::path::PathBuf::from("panic.log"));
        let _ = std::fs::OpenOptions::new()
            .create(true).append(true)
            .open(&log_path)
            .and_then(|mut f| std::io::Write::write_all(&mut f, log.as_bytes()));
        // 也写到 stderr (debug 模式下控制台可见)
        eprintln!("{log}");
        // 调用之前的 hook (保留默认行为)
        prev(info);
    }));
}

fn main() {
    install_panic_hook();
    tauri::Builder::default()
        .manage(Backend {
            running: AtomicBool::new(false),
            abort: AtomicBool::new(false),
        })
        .invoke_handler(tauri::generate_handler![
            chat_send,
            abort_chat,
            start_backend,
            pick_file,
        ])
        .run(tauri::generate_context!())
        .expect("error while running daiza-app");
}

//! daiza-web: 基于 daiza-engine 的炫酷聊天 GUI (零依赖 HTTP + SSE)
//!
//! 用法:
//! ```
//! daiza-web --model <gguf_path> [--port 8787] [--max-tokens 4096] [--greedy]
//! ```
//!
//! 浏览器打开 http://127.0.0.1:8787 即可聊天。
//! 回复通过 SSE (Server-Sent Events) 逐 token 流式渲染。

use std::path::PathBuf;

use daiza_engine::math::SamplingParams;
use daiza_runtime::engine::Engine;
use daiza_web::{run_server, ServerConfig};

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
        eprintln!("Usage: daiza-web --model <gguf_path> [options]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --model <path>       主权重 GGUF 路径 (必需)");
        eprintln!("  --dspark <path>      DSpark drafter GGUF 路径 (可选, 启用投机解码)");
        eprintln!("  --mmproj <path>      多模态视觉编码器 mmproj GGUF (可选, 启用 Vision)");
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
    let dspark_path: Option<PathBuf> = get_opt(&args, "--dspark").map(PathBuf::from);
    let mmproj_path: Option<PathBuf> = get_opt(&args, "--mmproj").map(PathBuf::from);
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
    let mut engine = match Engine::load(&gguf_path) {
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
    println!("[daiza-web] Engine metadata loaded. Loading weights (~13GB)…");
    // 启动时同步加载权重 + 初始化线程池 (避免首次 chat 请求卡顿数十秒)
    if let Err(e) = engine.load_weights() {
        eprintln!("[daiza-web] load_weights failed: {e}");
        std::process::exit(1);
    }
    let n_threads = daiza_engine::model::workspace::thread_count();
    daiza_engine::model::workspace::init_thread_pool(n_threads);
    println!("[daiza-web] Weights loaded ({n_threads} workers). Engine ready.");

    // 可选: 加载 DSpark drafter
    let dspark_available = if let Some(ref dp) = dspark_path {
        match engine.load_drafter(dp) {
            Ok(()) => {
                println!("[daiza-web] DSpark drafter loaded.");
                true
            }
            Err(e) => {
                eprintln!("[daiza-web] DSpark drafter load failed: {e}");
                false
            }
        }
    } else {
        false
    };

    // 可选: 加载多模态视觉编码器
    let vision_available = if let Some(ref mp) = mmproj_path {
        match engine.load_mmproj(mp) {
            Ok(()) => {
                println!("[daiza-web] mmproj (vision encoder) loaded.");
                true
            }
            Err(e) => {
                eprintln!("[daiza-web] mmproj load failed: {e}");
                false
            }
        }
    } else {
        false
    };

    let cfg = ServerConfig {
        engine,
        model_name,
        params,
        max_tokens,
        port,
        no_open,
        dspark_available,
        vision_available,
    };
    if let Err(e) = run_server(cfg) {
        eprintln!("[daiza-web] server error: {e}");
        std::process::exit(1);
    }
}

//! CLI 入口:加载 GGUF 权重并执行一次完整推理
//!
//! 用法:
//! ```
//! daiza-cli --model <gguf_path> [prompt] [max_tokens] [options...]
//! # 或向后兼容位置参数:
//! daiza-cli <gguf_path> <prompt> [max_tokens] [options...]
//! ```

use std::path::PathBuf;

use daiza_engine::engine::Engine;
use daiza_engine::math::SamplingParams;
use daiza_engine::Result;

/// 从 args 中查找 `--<flag> <value>` 并返回 value (单值, 后出现者覆盖)
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

/// 从 args 中查找 `--<flag> <value>` 的所有 value (多值, 如 --image)
fn get_opt_multi(args: &[String], flag: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == flag {
            if let Some(v) = iter.next() {
                out.push(PathBuf::from(v));
            }
        }
    }
    out
}

/// 已知选项名 (其后跟一个 value 参数),用于区分位置参数与选项值
const VALUE_OPTS: &[&str] = &[
    "--model", "--dspark", "--mmproj", "--image",
    "--confidence-threshold",
];

/// 从位置参数中提取 gguf_path / prompt / max_tokens
/// (跳过所有以 `--` 开头的 token 及其跟随的 value)
fn parse_positional(args: &[String]) -> (Option<String>, Option<String>, Option<usize>) {
    let mut pos: Vec<&str> = Vec::new();
    let mut iter = args.iter().skip(1).peekable();
    while let Some(a) = iter.next() {
        if a.starts_with("--") {
            // 若是 value-option, 跳过其 value
            if VALUE_OPTS.contains(&a.as_str()) {
                iter.next();
            }
            continue;
        }
        pos.push(a.as_str());
    }
    let gguf = pos.get(0).map(|s| s.to_string());
    let prompt = pos.get(1).map(|s| s.to_string());
    let max_tokens = pos.get(2).and_then(|s| s.parse().ok());
    (gguf, prompt, max_tokens)
}

fn print_usage() {
    eprintln!("Usage: daiza-cli --model <gguf_path> [prompt] [max_tokens] [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model <path>            主权重 GGUF 路径 (或第一个位置参数)");
    eprintln!("  --dspark <path>           DSpark drafter GGUF 路径");
    eprintln!("  --mmproj <path>           多模态视觉编码器 (mmproj GGUF)");
    eprintln!("  --image <path>            输入图像路径 (可多次指定, 需 --mmproj)");
    eprintln!("  --raw                     跳过 chat 模板, 直接编码 prompt (调试用)");
    eprintln!("  --greedy                  贪心解码 (temperature=0, 确定性输出)");
    eprintln!("  --confidence-threshold <f>  DSpark confidence 早停阈值 (默认 0.0)");
    eprintln!("  --inspect                 只显示模型元信息");
    eprintln!("  --dump-template           输出 chat_template 原始字节 (调试用)");
    eprintln!("  --help, -h                显示本帮助");
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // --help / -h
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }

    // 优先 --model, 回退到第一个位置参数, 最后用默认值
    let gguf_path: PathBuf = if let Some(m) = get_opt(&args, "--model") {
        m.into()
    } else if let (Some(p), _, _) = parse_positional(&args) {
        p.into()
    } else {
        "../Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf".into()
    };

    // 位置参数中的 prompt / max_tokens
    let (_, prompt_pos, max_tokens_pos) = parse_positional(&args);
    // 检查 --inspect / --dump-template 模式标记 (它们可以替代 prompt 位置)
    let is_inspect = args.iter().any(|a| a == "--inspect");
    let is_dump_template = args.iter().any(|a| a == "--dump-template");
    let prompt = if is_inspect {
        "--inspect".to_string()
    } else if is_dump_template {
        "--dump-template".to_string()
    } else {
        prompt_pos.unwrap_or_else(|| "你好".to_string())
    };
    let max_tokens: usize = max_tokens_pos.unwrap_or(64);

    let raw_mode = args.iter().any(|a| a == "--raw");
    let greedy = args.iter().any(|a| a == "--greedy");
    let dspark_path: Option<String> = get_opt(&args, "--dspark").map(String::from);
    let mmproj_path: Option<String> = get_opt(&args, "--mmproj").map(String::from);
    let image_paths: Vec<PathBuf> = get_opt_multi(&args, "--image");
    let confidence_threshold: f32 = get_opt(&args, "--confidence-threshold")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    println!("[daiza-cli] Loading: {}", gguf_path.display());

    // --inspect 模式:只显示元信息
    if is_inspect {
        let engine = Engine::load(&gguf_path)?;
        engine.print_summary();
        return Ok(());
    }

    // --dump-template 模式:输出 chat_template 的原始字节(用于调试 emoji 问题)
    if is_dump_template {
        let engine = Engine::load(&gguf_path)?;
        if let Some(tmpl) = engine.gguf.metadata.get_str("tokenizer.chat_template") {
            // 输出原始字节(十六进制)以便区分 brain emoji 与字面量 "mind"
            let bytes: Vec<u8> = tmpl.bytes().collect();
            println!("chat_template byte length: {}", bytes.len());
            // 找到所有 "mind" 或 brain emoji (F0 9F A7 A0) 的位置
            for i in 0..bytes.len() {
                if bytes[i..].starts_with(b"mind") {
                    println!("  found literal 'mind' at byte offset {i}");
                }
                if i + 3 < bytes.len() && bytes[i] == 0xF0 && bytes[i + 1] == 0x9F
                    && bytes[i + 2] == 0xA7 && bytes[i + 3] == 0xA0
                {
                    println!("  found brain emoji U+1F9E0 at byte offset {i}");
                }
            }
            // 输出最后 200 字节的十六进制 + 可打印 ASCII
            let tail_start = bytes.len().saturating_sub(200);
            println!("\n--- last 200 bytes (hex) ---");
            for i in tail_start..bytes.len() {
                print!("{:02x} ", bytes[i]);
                if (i - tail_start + 1) % 16 == 0 {
                    println!();
                }
            }
            println!();
            println!("\n--- last 200 bytes (ASCII, non-printable as .) ---");
            for &b in &bytes[tail_start..] {
                if (32..=126).contains(&b) {
                    print!("{}", b as char);
                } else {
                    print!(".");
                }
            }
            println!();
        } else {
            println!("chat_template not found");
        }
        return Ok(());
    }

    // --generate 模式:完整推理
    //
    // ★ 热降频根治: 9 active workers + yield 模式 (10 核活跃, 接近 llama.cpp 性能)
    //
    // 核心发现 (Meteor Lake 14 核实测, 256t):
    //   - 14 核 park: 功耗波动触发 turbo → 热降频 (235ms)
    //   - 7 active + park (8核): 212ms 稳定 (±1ms, 但算力不足)
    //   - 9 active + yield (10核): 147-165ms (冷启动147, 热稳定165, ±4ms) ← 当前
    //   - 10 active + yield (11核): 167-186ms (热积累退化)
    //   - llama.cpp 14 核: 154ms (稳定, 参考)
    //
    // 关键洞察: yield 模式 duty cycle ~40% (spin 4096 + yield), 类似 llama.cpp 的
    //   spin 6.5M + cond_wait, 平滑功耗避免热降频, 同时允许更多核心活跃
    //
    // 用户已手动设置的环境变量优先 (不覆盖)
    if std::env::var("DAIZA_WAIT_MODE").is_err() {
        // yield 模式: spin 4096 + yield, duty cycle ~40% (类似 llama.cpp spin+cond_wait)
        std::env::set_var("DAIZA_WAIT_MODE", "yield");
    }
    if std::env::var("DAIZA_ACTIVE_WORKERS").is_err() {
        // 9 active workers (10核活跃含main, 4核空闲散热, 避免热降频)
        // set_active_workers 会 min(9, n_workers), 小线程数自动全核
        std::env::set_var("DAIZA_ACTIVE_WORKERS", "9");
    }

    let mut engine = Engine::load(&gguf_path)?;
    println!("[daiza-cli] Engine loaded.");

    // 若指定 --dspark, 加载 drafter
    if let Some(dp) = &dspark_path {
        engine.load_drafter(std::path::Path::new(dp))?;
        println!("[daiza-cli] DSpark drafter loaded.");
    }

    // 若指定 --mmproj, 加载多模态视觉编码器
    if let Some(mp) = &mmproj_path {
        engine.load_mmproj(std::path::Path::new(mp))?;
        println!("[daiza-cli] mmproj (vision encoder) loaded.");
    }

    println!("[daiza-cli] Prompt: {prompt:?}");
    println!("[daiza-cli] Max tokens: {max_tokens}");
    println!("[daiza-cli] Raw mode: {raw_mode}");
    if dspark_path.is_some() {
        println!("[daiza-cli] DSpark: ENABLED");
        if confidence_threshold > 0.0 {
            println!("[daiza-cli] Confidence threshold: {confidence_threshold}");
        }
    }
    if mmproj_path.is_some() {
        println!("[daiza-cli] Vision: ENABLED ({} image(s))", image_paths.len());
    }
    println!();
    println!("=== Generating ===");

    let params = if greedy {
        // 贪心解码: temperature=0, 确定性 argmax 采样, 用于正确性验证
        SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    } else {
        SamplingParams {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.95,
        }
    };

    let output = if mmproj_path.is_some() && !image_paths.is_empty() {
        // 多模态路径: --mmproj + --image
        let system = "You are a helpful assistant.";
        engine.generate_with_image(&prompt, &image_paths, max_tokens, params, Some(system))?
    } else if dspark_path.is_some() {
        let system = "You are a helpful assistant.";
        engine.generate_with_dspark(&prompt, max_tokens, params, Some(system), confidence_threshold)?
    } else if raw_mode {
        engine.generate_raw(&prompt, max_tokens, params)?
    } else {
        let system = "You are a helpful assistant.";
        engine.generate_with_params(&prompt, max_tokens, params, Some(system))?
    };

    println!();
    println!("=== Output ===");
    println!("{output}");
    Ok(())
}

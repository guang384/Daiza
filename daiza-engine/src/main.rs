//! CLI 入口:加载 GGUF 权重并执行一次完整推理
//!
//! 用法:
//! ```
//! daiza-cli [gguf_path] [prompt]
//! ```

use std::path::PathBuf;

use daiza_engine::engine::Engine;
use daiza_engine::math::SamplingParams;
use daiza_engine::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let gguf_path: PathBuf = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "../Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf".to_string())
        .into();
    let prompt = args.get(2).cloned().unwrap_or_else(|| "你好".to_string());
    let max_tokens: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let raw_mode = args.iter().any(|a| a == "--raw");

    println!("[daiza-cli] Loading: {}", gguf_path.display());

    // --inspect 模式:只显示元信息
    if prompt == "--inspect" {
        let engine = Engine::load_metadata_only(&gguf_path)?;
        engine.print_summary();
        return Ok(());
    }

    // --dump-template 模式:输出 chat_template 的原始字节(用于调试 emoji 问题)
    if prompt == "--dump-template" {
        let engine = Engine::load_metadata_only(&gguf_path)?;
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
    let mut engine = Engine::load(&gguf_path)?;
    println!("[daiza-cli] Engine loaded.");
    println!("[daiza-cli] Prompt: {prompt:?}");
    println!("[daiza-cli] Max tokens: {max_tokens}");
    println!("[daiza-cli] Raw mode: {raw_mode}");
    println!();
    println!("=== Generating ===");

    let params = SamplingParams {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.95,
    };

    let output = if raw_mode {
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

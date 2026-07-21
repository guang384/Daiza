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
    // --greedy: 贪心解码 (temperature=0), 用于正确性验证 (确定性输出)
    let greedy = args.iter().any(|a| a == "--greedy");
    // --dspark <path>: 启用 DSpark 推测解码, 指定 drafter GGUF 路径
    let dspark_path: Option<String> = {
        let mut iter = args.iter().skip(1);
        let mut p = None;
        while let Some(a) = iter.next() {
            if a == "--dspark" {
                if let Some(v) = iter.next() {
                    p = Some(v.clone());
                }
            }
        }
        p
    };
    // --mmproj <path>: 加载多模态视觉编码器 (mmproj GGUF)
    let mmproj_path: Option<String> = {
        let mut iter = args.iter().skip(1);
        let mut p = None;
        while let Some(a) = iter.next() {
            if a == "--mmproj" {
                if let Some(v) = iter.next() {
                    p = Some(v.clone());
                }
            }
        }
        p
    };
    // --image <path>: 输入图像路径 (可多次指定, 与 --mmproj 配合)
    //   若指定至少一张图, 走 generate_with_image 多模态路径
    let image_paths: Vec<PathBuf> = {
        let mut paths = Vec::new();
        let mut iter = args.iter().skip(1);
        while let Some(a) = iter.next() {
            if a == "--image" {
                if let Some(v) = iter.next() {
                    paths.push(PathBuf::from(v));
                }
            }
        }
        paths
    };
    // --confidence-threshold <f32>: DSpark confidence head 早停阈值 (默认 0.0 = 不截断)
    // sigmoid(confidence_logit) < threshold 的位置起, draft token 不再 verify
    // 典型值: 0.5 (中等置信), 0.8 (高置信才 verify)
    let confidence_threshold: f32 = {
        let mut iter = args.iter().skip(1);
        let mut t = 0.0f32;
        while let Some(a) = iter.next() {
            if a == "--confidence-threshold" {
                if let Some(v) = iter.next() {
                    t = v.parse().unwrap_or(0.0);
                }
            }
        }
        t
    };

    println!("[daiza-cli] Loading: {}", gguf_path.display());

    // --inspect 模式:只显示元信息
    if prompt == "--inspect" {
        let engine = Engine::load(&gguf_path)?;
        engine.print_summary();
        return Ok(());
    }

    // --dump-template 模式:输出 chat_template 的原始字节(用于调试 emoji 问题)
    if prompt == "--dump-template" {
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

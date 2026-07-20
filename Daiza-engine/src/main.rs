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
    // ★ 热降频根治: 统一使用 7 active workers + park 模式
    //
    // 核心发现 (Meteor Lake 14 核实测):
    //   - 14 核 park: 功耗波动触发 turbo → 热降频 (256t: 235ms, 64t: 223ms)
    //   - 14 核 yield: 不降频但调度开销大 (229ms/tok)
    //   - 9 核+ park: 仍降频 (223ms/tok)
    //   - 8 核 park (7 active + main): 不降频, 零调度开销, 最优 (256t: 207ms, 64t: 202ms)
    //
    // 策略: 所有场景统一 7 active + park
    //   - 短跑 64t 全核降频 (223ms) vs 7 active 不降频 (202ms) → 7 active 更优
    //   - 长跑 256t 7 active 稳定 207ms, 不降频
    //   - 8 核活跃 (7 workers + main) + 6 核空闲散热 = 不触发 turbo, 持续基准频率
    //
    // 用户已手动设置的环境变量优先 (不覆盖)
    if std::env::var("DAIZA_WAIT_MODE").is_err() {
        // park 模式 (零调度开销, yield 有 ~40ms/token 调度开销)
        std::env::set_var("DAIZA_WAIT_MODE", "park");
    }
    if std::env::var("DAIZA_ACTIVE_WORKERS").is_err() {
        // 限制 7 active workers (8核活跃含main, 6核空闲散热, 避免热降频)
        // set_active_workers 会 min(7, n_workers), 小线程数自动全核
        std::env::set_var("DAIZA_ACTIVE_WORKERS", "7");
    }

    let mut engine = Engine::load(&gguf_path)?;
    println!("[daiza-cli] Engine loaded.");

    // 若指定 --dspark, 加载 drafter
    if let Some(dp) = &dspark_path {
        engine.load_drafter(std::path::Path::new(dp))?;
        println!("[daiza-cli] DSpark drafter loaded.");
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

    let output = if dspark_path.is_some() {
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

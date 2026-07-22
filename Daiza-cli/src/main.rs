//! CLI 入口:加载 GGUF 权重并执行一次完整推理
//!
//! 用法:
//! ```
//! daiza-cli --model <gguf_path> [options]
//! ```

use std::path::PathBuf;

use daiza_runtime::engine::Engine;
use daiza_engine::math::SamplingParams;
use daiza_runtime::session::Session;
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

fn print_usage() {
    eprintln!("Usage: daiza-cli --model <gguf_path> [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model <path>              主权重 GGUF 路径 (必需)");
    eprintln!("  --prompt <text>             输入提示词 (默认 \"你好\")");
    eprintln!("  --max-tokens <n>            最多生成 token 数 (默认 4096)");
    eprintln!("  --dspark <path>             DSpark drafter GGUF 路径");
    eprintln!("  --mmproj <path>             多模态视觉编码器 (mmproj GGUF)");
    eprintln!("  --image <path>              输入图像路径 (可多次指定, 需 --mmproj)");
    eprintln!("  --raw                       跳过 chat 模板, 直接编码 prompt (调试用)");
    eprintln!("  --greedy                    贪心解码 (temperature=0, 确定性输出)");
    eprintln!("  --confidence-threshold <f>  DSpark confidence 早停阈值 (默认 0.0)");
    eprintln!("  --inspect                   只显示模型元信息");
    eprintln!("  --dump-template             输出 chat_template 原始字节 (调试用)");
    eprintln!("  --interactive               启动交互式 REPL (多轮对话, 复用 KV/SSM state)");
    eprintln!("  --self-test                 自动化验证: 对比 generate_with_params vs session_reply");
    eprintln!("  --help, -h                  显示本帮助");
}

/// 交互式 REPL:引擎加载一次常驻, 循环读取 stdin
///
/// 阶段 1.1+:使用 session_reply (增量 prefill, 复用 KV + SSM state)
/// 阶段 3+:SessionManager 多 session LRU 管理 (active 在 DRAM, inactive 在 SSD)
fn run_repl(
    mut engine: Engine,
    params: SamplingParams,
    system_prompt: &str,
    default_max_tokens: usize,
) -> Result<()> {
    use std::io::{BufRead, Write};

    // 自动开启 session (默认 think_enabled=true, 与 generate_with_params 行为一致)
    engine.session_begin(true, Some(system_prompt))?;

    // 多 session 管理器:active session 在 engine.session 中,
    // inactive session dump 到 ./daiza_sessions/ 目录 (LRU, 最多 8 个)
    let ssd_dir = std::path::Path::new("./daiza_sessions");
    let mut mgr = daiza_runtime::session_manager::SessionManager::new(ssd_dir, 8)?;
    mgr.set_active_id(Some("default".to_string()));

    println!();
    println!("=== Daiza REPL ===");
    println!("Type /help for commands, /quit to exit.");
    println!("Active session: default");
    println!();

    let stdin = std::io::stdin();
    let mut line = String::new();
    let mut max_tokens = default_max_tokens;

    loop {
        // 提示符显示当前 think 状态 + session id
        let think_on = engine.session.as_ref().map(|s| s.think_enabled).unwrap_or(false);
        let sid = mgr.active_id().unwrap_or("none");
        print!("[{sid}]{}> ", if think_on { "[think]" } else { "" });
        std::io::stdout().flush()?;
        line.clear();
        if stdin.lock().read_line(&mut line)? == 0 {
            // EOF (Ctrl+D)
            println!();
            break;
        }
        let msg = line.trim();
        if msg.is_empty() {
            continue;
        }

        // 斜杠命令
        if let Some(cmd) = msg.strip_prefix('/') {
            let mut parts = cmd.split_whitespace();
            let name = parts.next().unwrap_or("");
            match name {
                "help" | "h" | "?" => {
                    println!("Commands:");
                    println!("  /help              Show this help");
                    println!("  /quit              Exit REPL (also Ctrl+D)");
                    println!("  /reset             Reset conversation (clear KV/SSM state)");
                    println!("  /tokens <n>        Set max tokens per reply (current: {max_tokens})");
                    println!("  /think [on|off]    Toggle or set thinking mode (current: {})", if think_on { "on" } else { "off" });
                    println!("  /save <path>       Dump session (KV+SSM) to SSD");
                    println!("  /load <path>       Load session from SSD (replace current)");
                    println!("  /raw <text>        Send raw text (skip chat template, debug)");
                    println!();
                    println!("Multi-session (LRU, inactive → {}):", ssd_dir.display());
                    println!("  /sessions          List all sessions (active + inactive)");
                    println!("  /create <id>       Create new session (current parks to SSD)");
                    println!("  /switch <id>       Switch to session (current parks, target unparks)");
                    println!("  /delete <id>       Delete session (from DRAM or SSD)");
                    println!();
                    println!("Multimodal (requires --mmproj):");
                    println!("  /image <path>      Add image to pending list (sent with next message)");
                    println!("  /images            List pending images (cleared after each reply)");
                    println!();
                    println!("Tool calling:");
                    println!("  /tools             List registered tools");
                    println!("  /tool add <json>   Register a tool (JSON: name/description/parameters)");
                    println!("  /tool clear        Clear all tools");
                    println!("  /tool resp <json>  Send tool response back to model");
                    println!();
                    println!("Tip: just type a message to chat.");
                }
                "quit" | "exit" | "q" => {
                    println!("[repl] bye.");
                    break;
                }
                "reset" => {
                    if let Some(s) = &mut engine.session {
                        s.reset();
                        println!("[repl] conversation reset (KV/SSM state cleared)");
                    }
                }
                "tokens" => {
                    if let Some(arg) = parts.next() {
                        match arg.parse::<usize>() {
                            Ok(n) if n > 0 => {
                                max_tokens = n;
                                println!("[repl] max tokens set to {max_tokens}");
                            }
                            _ => println!("[repl] invalid value: {arg}"),
                        }
                    } else {
                        println!("[repl] current max tokens: {max_tokens}");
                    }
                }
                "think" => {
                    if let Some(s) = &mut engine.session {
                        match parts.next() {
                            Some("on") => {
                                s.think_enabled = true;
                                println!("[repl] think: on");
                            }
                            Some("off") => {
                                s.think_enabled = false;
                                println!("[repl] think: off");
                            }
                            _ => {
                                // toggle
                                s.think_enabled = !s.think_enabled;
                                println!("[repl] think: {} (toggled)", if s.think_enabled { "on" } else { "off" });
                            }
                        }
                    }
                }
                "raw" => {
                    let rest: String = parts.collect::<Vec<_>>().join(" ");
                    if rest.is_empty() {
                        println!("[repl] usage: /raw <text>");
                        continue;
                    }
                    print_repl_reply_header();
                    match engine.generate_raw(&rest, max_tokens, params.clone()) {
                        Ok(out) => println!("{out}"),
                        Err(e) => eprintln!("[repl] error: {e}"),
                    }
                }
                "save" => {
                    let path = parts.next();
                    let path = match path {
                        Some(p) => p,
                        None => {
                            println!("[repl] usage: /save <path>");
                            continue;
                        }
                    };
                    let session = match engine.session.as_ref() {
                        Some(s) => s,
                        None => {
                            eprintln!("[repl] no active session");
                            continue;
                        }
                    };
                    let t = std::time::Instant::now();
                    match session.save_to_disk(std::path::Path::new(path), &engine.config) {
                        Ok(()) => {
                            let ms = t.elapsed().as_millis();
                            let pos = session.state.pos;
                            let n_hist = session.history_tokens.len();
                            println!("[repl] session saved to {path} (pos={pos}, {n_hist} tokens, {ms}ms)");
                        }
                        Err(e) => eprintln!("[repl] save failed: {e}"),
                    }
                }
                "load" => {
                    let path = parts.next();
                    let path = match path {
                        Some(p) => p,
                        None => {
                            println!("[repl] usage: /load <path>");
                            continue;
                        }
                    };
                    let t = std::time::Instant::now();
                    match Session::load_from_disk(std::path::Path::new(path), &engine.config) {
                        Ok(session) => {
                            let ms = t.elapsed().as_millis();
                            let pos = session.state.pos;
                            let n_hist = session.history_tokens.len();
                            engine.session = Some(session);
                            println!("[repl] session loaded from {path} (pos={pos}, {n_hist} tokens, {ms}ms)");
                        }
                        Err(e) => eprintln!("[repl] load failed: {e}"),
                    }
                }
                "sessions" => {
                    let list = mgr.list();
                    if list.is_empty() {
                        println!("[repl] no sessions");
                    } else {
                        println!("[repl] sessions ({}):", list.len());
                        for (id, is_active) in &list {
                            println!("  {} {}",
                                if *is_active { "*" } else { " " },
                                id);
                        }
                    }
                }
                "create" => {
                    let id = match parts.next() {
                        Some(id) => id,
                        None => {
                            println!("[repl] usage: /create <id>");
                            continue;
                        }
                    };
                    if mgr.exists(id) {
                        eprintln!("[repl] session '{id}' already exists");
                        continue;
                    }
                    // park 当前 active session 到 SSD
                    if let Some(active_id) = mgr.active_id().map(String::from) {
                        if let Some(session) = engine.session.take() {
                            let t = std::time::Instant::now();
                            match mgr.park(&active_id, &session, &engine.config) {
                                Ok(()) => {
                                    let ms = t.elapsed().as_millis();
                                    println!("[repl] parked '{active_id}' to SSD ({ms}ms)");
                                }
                                Err(e) => {
                                    eprintln!("[repl] park '{active_id}' failed: {e}");
                                    engine.session = Some(session); // 恢复
                                    continue;
                                }
                            }
                        }
                    }
                    // 创建新 session (engine.session 已 take, 使用 REPL 默认参数)
                    // 默认 think_enabled=true (与 run_repl 启动时一致)
                    match engine.session_begin(true, Some(system_prompt)) {
                        Ok(()) => {
                            mgr.set_active_id(Some(id.to_string()));
                            println!("[repl] created and switched to '{id}'");
                        }
                        Err(e) => eprintln!("[repl] create '{id}' failed: {e}"),
                    }
                }
                "switch" => {
                    let id = match parts.next() {
                        Some(id) => id,
                        None => {
                            println!("[repl] usage: /switch <id>");
                            continue;
                        }
                    };
                    if mgr.active_id() == Some(id) {
                        println!("[repl] already on '{id}'");
                        continue;
                    }
                    if !mgr.exists(id) {
                        eprintln!("[repl] session '{id}' not found");
                        continue;
                    }
                    // park 当前 active
                    if let Some(active_id) = mgr.active_id().map(String::from) {
                        if let Some(session) = engine.session.take() {
                            let t = std::time::Instant::now();
                            match mgr.park(&active_id, &session, &engine.config) {
                                Ok(()) => {
                                    let ms = t.elapsed().as_millis();
                                    println!("[repl] parked '{active_id}' to SSD ({ms}ms)");
                                }
                                Err(e) => {
                                    eprintln!("[repl] park '{active_id}' failed: {e}");
                                    engine.session = Some(session);
                                    continue;
                                }
                            }
                        }
                    }
                    // unpark 目标
                    let t = std::time::Instant::now();
                    match mgr.unpark(id, &engine.config) {
                        Ok(session) => {
                            let ms = t.elapsed().as_millis();
                            let pos = session.state.pos;
                            let n_hist = session.history_tokens.len();
                            engine.session = Some(session);
                            println!("[repl] switched to '{id}' (pos={pos}, {n_hist} tokens, load {ms}ms)");
                        }
                        Err(e) => eprintln!("[repl] switch to '{id}' failed: {e}"),
                    }
                }
                "delete" => {
                    let id = match parts.next() {
                        Some(id) => id,
                        None => {
                            println!("[repl] usage: /delete <id>");
                            continue;
                        }
                    };
                    let is_active = mgr.active_id() == Some(id);
                    mgr.delete(id);
                    if is_active {
                        engine.session = None;
                        println!("[repl] deleted active session '{id}' (engine.session cleared)");
                    } else {
                        println!("[repl] deleted inactive session '{id}' (SSD file removed)");
                    }
                }
                "image" => {
                    let path = match parts.next() {
                        Some(p) => p,
                        None => {
                            println!("[repl] usage: /image <path>");
                            continue;
                        }
                    };
                    let p = std::path::PathBuf::from(path);
                    if !p.exists() {
                        eprintln!("[repl] image not found: {path}");
                        continue;
                    }
                    if engine.vision.is_none() {
                        eprintln!("[repl] mmproj not loaded (start with --mmproj <path>)");
                        continue;
                    }
                    if let Some(s) = &mut engine.session {
                        s.pending_images.push(p.clone());
                        println!("[repl] added image: {} ({} pending)",
                            p.display(), s.pending_images.len());
                    } else {
                        eprintln!("[repl] no active session");
                    }
                }
                "images" => {
                    if let Some(s) = &engine.session {
                        if s.pending_images.is_empty() {
                            println!("[repl] no pending images");
                        } else {
                            println!("[repl] pending images ({}):", s.pending_images.len());
                            for (i, p) in s.pending_images.iter().enumerate() {
                                println!("  [{i}] {}", p.display());
                            }
                            println!("(will be sent with next message, then cleared)");
                        }
                    } else {
                        eprintln!("[repl] no active session");
                    }
                }
                "tools" => {
                    if let Some(s) = &engine.session {
                        if s.tools.is_empty() {
                            println!("[repl] no tools registered (use /tool add <json>)");
                        } else {
                            println!("[repl] registered tools ({}):", s.tools.len());
                            for (i, t) in s.tools.iter().enumerate() {
                                println!("  [{i}] {} - {}", t.name, t.description);
                                for p in &t.parameters {
                                    println!("        param {} ({}){}: {}",
                                        p.name, p.param_type,
                                        if p.required { " [required]" } else { "" },
                                        p.description);
                                }
                            }
                        }
                    } else {
                        eprintln!("[repl] no active session");
                    }
                }
                "tool" => {
                    let sub = parts.next().unwrap_or("");
                    match sub {
                        "add" => {
                            // JSON 格式: {"name":"...","description":"...","parameters":[{"name":"...","type":"string","description":"...","required":true}]}
                            let json_str: String = parts.collect::<Vec<_>>().join(" ");
                            if json_str.is_empty() {
                                println!("[repl] usage: /tool add <json>");
                                println!("        JSON format: {{\"name\":\"fn\",\"description\":\"...\",\"parameters\":[{{\"name\":\"p\",\"type\":\"string\",\"description\":\"...\",\"required\":true}}]}}");
                                continue;
                            }
                            match parse_tool_def_json(&json_str) {
                                Ok(td) => {
                                    if let Some(s) = &mut engine.session {
                                        let name = td.name.clone();
                                        s.tools.push(td);
                                        println!("[repl] tool '{name}' added ({} tools total)", s.tools.len());
                                    } else {
                                        eprintln!("[repl] no active session");
                                    }
                                }
                                Err(e) => eprintln!("[repl] parse tool JSON failed: {e}"),
                            }
                        }
                        "clear" => {
                            if let Some(s) = &mut engine.session {
                                let n = s.tools.len();
                                s.tools.clear();
                                s.messages.clear();
                                println!("[repl] cleared {n} tools (messages history also cleared)");
                            } else {
                                eprintln!("[repl] no active session");
                            }
                        }
                        "resp" => {
                            // 发送 tool response: /tool resp <name> <content>
                            let name = match parts.next() {
                                Some(n) => n.to_string(),
                                None => {
                                    println!("[repl] usage: /tool resp <name> <content>");
                                    continue;
                                }
                            };
                            let content: String = parts.collect::<Vec<_>>().join(" ");
                            if content.is_empty() {
                                println!("[repl] usage: /tool resp <name> <content>");
                                continue;
                            }
                            let resp = daiza_runtime::tool_call::ToolResponse { name, content };
                            print_repl_reply_header();
                            match engine.session_reply_with_tool_response(&[resp], max_tokens, params.clone()) {
                                Ok(out) => println!("{out}"),
                                Err(e) => eprintln!("[repl] tool_response error: {e}"),
                            }
                        }
                        _ => {
                            println!("[repl] usage: /tool add|clear|resp ...");
                        }
                    }
                }
                _ => {
                    println!("[repl] unknown command: /{name} (try /help)");
                }
            }
            continue;
        }

        // 普通对话:session_reply (增量 prefill, 复用 KV + SSM state)
        // 若 pending_images 非空: 走 vision 注入路径
        if engine.session.is_none() {
            eprintln!("[repl] no active session (use /create <id> to create one)");
            continue;
        }
        let has_images = engine.session.as_ref()
            .map(|s| !s.pending_images.is_empty())
            .unwrap_or(false);
        print_repl_reply_header();
        let result = if has_images {
            engine.session_reply_with_vision(msg, max_tokens, params.clone())
        } else {
            engine.session_reply(msg, max_tokens, params.clone())
        };
        match result {
            Ok(out) => println!("{out}"),
            Err(e) => eprintln!("[repl] error: {e}"),
        }
    }

    Ok(())
}

/// 解析 tool 定义 JSON (REPL /tool add 使用)
///
/// 简化 JSON 解析 (不引入 serde_json 依赖)
/// 格式: {"name":"fn","description":"...","parameters":[{"name":"p","type":"string","description":"...","required":true}]}
fn parse_tool_def_json(json: &str) -> daiza_engine::Result<daiza_runtime::tool_call::ToolDef> {
    use daiza_runtime::tool_call::{ToolDef, ToolParam};
    // 极简 JSON 解析: 提取 name, description, parameters
    let name = extract_json_string(json, "name")
        .ok_or_else(|| daiza_engine::BonsaiError::Model("missing 'name' field".into()))?;
    let description = extract_json_string(json, "description")
        .ok_or_else(|| daiza_engine::BonsaiError::Model("missing 'description' field".into()))?;

    // 解析 parameters 数组 (简化: 逐个提取 name/type/description/required)
    let mut parameters = Vec::new();
    let params_str = extract_json_array(json, "parameters").unwrap_or_default();
    let mut idx = 0;
    while idx < params_str.len() {
        // 找下一个 { 开始的对象
        let obj_start = match params_str[idx..].find('{') {
            Some(s) => idx + s,
            None => break,
        };
        // 配对 }
        let mut depth = 0;
        let mut obj_end = obj_start;
        for (i, c) in params_str[obj_start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        obj_end = obj_start + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let obj = &params_str[obj_start..=obj_end];
        let pname = extract_json_string(obj, "name").unwrap_or_default();
        let ptype = extract_json_string(obj, "type").unwrap_or("string".into());
        let pdesc = extract_json_string(obj, "description").unwrap_or_default();
        let prequired = extract_json_bool(obj, "required").unwrap_or(false);
        if !pname.is_empty() {
            parameters.push(ToolParam {
                name: pname,
                description: pdesc,
                param_type: ptype,
                required: prequired,
            });
        }
        idx = obj_end + 1;
    }

    Ok(ToolDef { name, description, parameters })
}

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)? + pattern.len();
    let rest = &json[idx..];
    // 跳过 :
    let colon = rest.find(':')?;
    let after = &rest[colon + 1..];
    // 跳过空白
    let after = after.trim_start();
    if !after.starts_with('"') {
        return None;
    }
    let val_start = 1;
    let mut val_end = val_start;
    let bytes = after.as_bytes();
    let mut i = val_start;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2;
            continue;
        }
        if bytes[i] == b'"' {
            val_end = i;
            break;
        }
        i += 1;
    }
    Some(after[val_start..val_end].to_string())
}

fn extract_json_bool(json: &str, key: &str) -> Option<bool> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)? + pattern.len();
    let rest = &json[idx..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if after.starts_with("true") {
        Some(true)
    } else if after.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn extract_json_array<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)? + pattern.len();
    let rest = &json[idx..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('[') {
        return None;
    }
    let start = 1;
    let mut depth = 1;
    let bytes = after.as_bytes();
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&after[start..i]);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

#[inline]
fn print_repl_reply_header() {
    println!();
    println!("--- assistant ---");
}

/// 自动化验证:对比 generate_with_params vs session_reply
///
/// 测试 1: 单轮对比 - 相同 prompt, 对比输出 token 序列 (应 bit-exact 一致)
/// 测试 2: 多轮验证 - session_reply 三轮对话, 检查第二轮能"记住"第一轮内容
/// 测试 3: think toggle - 验证 /think on/off 切换不影响已 cache 的历史
fn run_self_test(mut engine: Engine, max_tokens: usize) -> Result<()> {
    // 强制 greedy (确定性输出)
    let params = daiza_engine::math::SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
    };
    let sys = "You are a helpful assistant.";

    // 关闭 stream 输出, 减少干扰
    std::env::set_var("DAIZA_STREAM", "0");

    // 所有 self-test 临时文件统一写到 selftest_tmp/ 目录, 便于清理 + 不污染项目根
    // .gitignore 已忽略该目录
    let tmp_dir = std::path::Path::new("./selftest_tmp");
    let _ = std::fs::remove_dir_all(tmp_dir); // 清理上次残留
    std::fs::create_dir_all(tmp_dir)?;

    println!("=== Daiza Self-Test ===");
    println!();

    // ========== 测试 1: 单轮对比 ==========
    println!("[test 1] single-turn: generate_with_params vs session_reply");
    let prompt1 = "你好";

    // 方式 A: generate_with_params (全量重算, 无 session)
    let dump_a_path = tmp_dir.join("selftest_a.txt");
    std::env::set_var("DAIZA_DUMP_TOKENS", dump_a_path.to_str().unwrap());
    let out_a = engine.generate_with_params(prompt1, max_tokens, params.clone(), Some(sys))?;
    std::env::remove_var("DAIZA_DUMP_TOKENS");

    // 方式 B: session_reply (增量 prefill, 首轮与 A 输入完全一致)
    engine.session_begin(true, Some(sys))?;
    let dump_b_path = tmp_dir.join("selftest_b.txt");
    std::env::set_var("DAIZA_DUMP_TOKENS", dump_b_path.to_str().unwrap());
    let out_b = engine.session_reply(prompt1, max_tokens, params.clone())?;
    std::env::remove_var("DAIZA_DUMP_TOKENS");
    engine.session_end();

    // 对比 token 序列
    let dump_a = std::fs::read_to_string(&dump_a_path)
        .map_err(|e| daiza_engine::BonsaiError::Io(format!("read {} failed: {e}", dump_a_path.display())))?;
    let dump_b = std::fs::read_to_string(&dump_b_path)
        .map_err(|e| daiza_engine::BonsaiError::Io(format!("read {} failed: {e}", dump_b_path.display())))?;

    // 提取 generated 行 (第二行)
    let gen_a = dump_a.lines().nth(1).unwrap_or("").strip_prefix("generated:").unwrap_or("");
    let gen_b = dump_b.lines().nth(1).unwrap_or("").strip_prefix("generated:").unwrap_or("");

    let pass1 = gen_a == gen_b;
    println!("  A tokens: {} (len={})", &gen_a[..gen_a.len().min(80)], gen_a.split(',').count());
    println!("  B tokens: {} (len={})", &gen_b[..gen_b.len().min(80)], gen_b.split(',').count());
    println!("  output A: {:?}", &out_a[..out_a.len().min(60)]);
    println!("  output B: {:?}", &out_b[..out_b.len().min(60)]);
    println!("  result: {}", if pass1 { "PASS" } else { "FAIL" });
    println!();

    // ========== 测试 2: 多轮验证 (上下文记忆) ==========
    println!("[test 2] multi-turn: session_reply 5 rounds (context memory + speedup)");
    engine.session_begin(true, Some(sys))?;

    // 关闭 think 模式, 让模型直接回答 (避免 think 占满 max_tokens)
    if let Some(s) = &mut engine.session {
        s.think_enabled = false;
    }

    // 用较大的 max_tokens 让模型能完成回答
    let mt_tokens = max_tokens.max(64);
    let prompts = ["我叫小明", "我叫什么名字?", "1+1等于几?", "我刚才问了你什么?", "用一句话介绍我"];
    let mut times = Vec::new();
    let mut outputs = Vec::new();
    let mut token_counts = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let dump_path = tmp_dir.join(format!("selftest_r{}.txt", i + 1));
        std::env::set_var("DAIZA_DUMP_TOKENS", dump_path.to_str().unwrap());
        let t = std::time::Instant::now();
        let out = engine.session_reply(p, mt_tokens, params.clone())?;
        let ms = t.elapsed().as_millis();
        std::env::remove_var("DAIZA_DUMP_TOKENS");
        times.push(ms);
        outputs.push(out);

        // 从 dump 文件读取 token 数
        let n_gen = std::fs::read_to_string(&dump_path)
            .ok()
            .and_then(|s| s.lines().nth(1).map(|l| l.trim_start_matches("generated:").split(',').filter(|s| !s.is_empty()).count()))
            .unwrap_or(0);
        token_counts.push(n_gen);

        let eos_stopped = n_gen < mt_tokens;
        let think_closed = outputs[i].contains("</think>");
        let has_think_open = outputs[i].contains("<think>");

        println!("  R{}: {}ms, {} tokens, EOS={}, think={}{}",
            i + 1, ms, n_gen,
            if eos_stopped { "YES" } else { "MAX" },
            if !has_think_open { "none" } else if think_closed { "closed" } else { "OPEN(unclosed)" },
            if eos_stopped { "" } else { " ← hit max_tokens" }
        );
        // 写完整输出到文件
        let full_path = tmp_dir.join(format!("selftest_r{}_full.txt", i + 1));
        std::fs::write(&full_path, &outputs[i]).ok();
        println!("       full output → {}", full_path.display());
    }

    // 检查上下文记忆
    let r2_has_name = outputs[1].contains("小明");
    let r4_has_recall = outputs[3].contains("1+1") || outputs[3].contains("等于") || outputs[3].contains("加");
    println!("  R2 mentions name: {}", if r2_has_name { "PASS" } else { "FAIL" });
    println!("  R4 recalls prev Q: {}", if r4_has_recall { "PASS" } else { "FAIL" });
    println!();

    // ========== 测试 3: prefill 加速比 ==========
    println!("[test 3] prefill speedup: full vs incremental");
    for (i, &ms) in times.iter().enumerate() {
        if i == 0 {
            println!("  R1 (full prefill, sys+user1):    {ms}ms");
        } else {
            println!("  R{} (incremental):                {ms}ms ({:.1}x vs R1)", i + 1, times[0] as f64 / ms.max(1) as f64);
        }
    }
    println!();

    // ========== 测试 4: SSD 持久化 (dump → load → 继续对话) ==========
    println!("[test 4] SSD persistence: dump → load → continue");
    // 4a: dump 当前 session
    let dump_path = tmp_dir.join("selftest_session.dzss");
    let session = engine.session.as_ref().unwrap();
    let t = std::time::Instant::now();
    session.save_to_disk(&dump_path, &engine.config)?;
    let dump_ms = t.elapsed().as_millis();
    let dump_pos = session.state.pos;
    let dump_hist = session.history_tokens.len();
    let file_size = std::fs::metadata(&dump_path)?.len();
    println!("  dump: {dump_ms}ms, pos={dump_pos}, {dump_hist} tokens, {:.1} MB",
        file_size as f64 / 1024.0 / 1024.0);

    // 4b: load 到新 session (模拟跨进程恢复)
    let t = std::time::Instant::now();
    use daiza_runtime::session::Session;
    let loaded_session = Session::load_from_disk(&dump_path, &engine.config)?;
    let load_ms = t.elapsed().as_millis();
    let load_pos = loaded_session.state.pos;
    let load_hist = loaded_session.history_tokens.len();
    println!("  load: {load_ms}ms, pos={load_pos}, {load_hist} tokens");

    // 4c: 用 loaded session 继续 1 轮对话, 对比连续 session 的输出
    engine.session = Some(loaded_session);
    let verify_prompt = "我叫什么名字?";
    let t = std::time::Instant::now();
    let out_loaded = engine.session_reply(verify_prompt, max_tokens, params.clone())?;
    let loaded_reply_ms = t.elapsed().as_millis();

    // 4d: 重新跑一个连续 session 到相同状态, 对比输出
    engine.session_end();
    engine.session_begin(false, Some(sys))?; // think off, 与 test2 一致
    for p in &prompts {
        engine.session_reply(p, max_tokens, params.clone())?;
    }
    let t = std::time::Instant::now();
    let out_continuous = engine.session_reply(verify_prompt, max_tokens, params.clone())?;
    let continuous_reply_ms = t.elapsed().as_millis();

    let pass4 = out_loaded == out_continuous;
    println!("  loaded reply ({loaded_reply_ms}ms): {:?}", &out_loaded[..out_loaded.char_indices().take(60).last().map(|(i,_)| i).unwrap_or(out_loaded.len())]);
    println!("  continuous reply ({continuous_reply_ms}ms): {:?}", &out_continuous[..out_continuous.char_indices().take(60).last().map(|(i,_)| i).unwrap_or(out_continuous.len())]);
    println!("  dump→load→reply == continuous: {}", if pass4 { "PASS" } else { "FAIL" });
    println!();

    // ========== 测试 5: SessionManager 多 session LRU 管理 ==========
    // 场景: 创建 A (告诉名字) → park → 创建 B (告诉另一个名字) → unpark A → 验证状态守恒
    //
    // SessionManager 的核心职责是正确 park/unpark session (state 守恒),
    // 模型回答质量 (是否记住名字) 依赖 max_tokens + think 模式, 仅作 info 不作 pass/fail
    println!("[test 5] SessionManager: create A → park → create B → switch A → verify state");
    use daiza_runtime::session_manager::SessionManager;

    let ssd_dir = tmp_dir.join("mgr_sessions");
    // 清理上次遗留的 session 文件
    let _ = std::fs::remove_dir_all(&ssd_dir);
    let mut mgr = SessionManager::new(&ssd_dir, 8)?;

    // 5a: 创建全新 session_a (不复用 test4 的 session, 避免历史污染)
    engine.session_end();
    engine.session_begin(false, Some(sys))?; // think off
    mgr.set_active_id(Some("session_a".to_string()));
    let _ = engine.session_reply("请记住我的名字叫张三丰", max_tokens, params.clone())?;
    let a_pos_before_park = engine.session.as_ref().map(|s| s.state.pos).unwrap_or(0);
    let a_hist_before_park = engine.session.as_ref().map(|s| s.history_tokens.len()).unwrap_or(0);
    println!("  5a: session_a replied, pos={a_pos_before_park}, hist={a_hist_before_park}");

    // 5b: park session_a 到 SSD
    let t = std::time::Instant::now();
    let session_a = engine.session.take().unwrap();
    mgr.park("session_a", &session_a, &engine.config)?;
    let park_ms = t.elapsed().as_millis();
    println!("  5b: parked session_a ({park_ms}ms)");

    // 5c: 创建 session_b, 告知不同名字
    engine.session_begin(false, Some(sys))?;
    mgr.set_active_id(Some("session_b".to_string()));
    let _ = engine.session_reply("请记住我的名字叫李四光", max_tokens, params.clone())?;
    let b_pos = engine.session.as_ref().map(|s| s.state.pos).unwrap_or(0);
    println!("  5c: session_b replied, pos={b_pos}");

    // 5d: 列出 sessions, 应有 2 个 (session_a inactive + session_b active)
    let list = mgr.list();
    println!("  5d: sessions list: {:?}", list);
    let list_has_2 = list.len() == 2;
    let list_a_inactive = list.iter().any(|(id, act)| id == "session_a" && !act);
    let list_b_active = list.iter().any(|(id, act)| id == "session_b" && *act);

    // 5e: switch 回 session_a (park session_b, unpark session_a)
    let session_b = engine.session.take().unwrap();
    mgr.park("session_b", &session_b, &engine.config)?;
    let t = std::time::Instant::now();
    let session_a_loaded = mgr.unpark("session_a", &engine.config)?;
    let unpark_ms = t.elapsed().as_millis();
    let a_pos_after_unpark = session_a_loaded.state.pos;
    let a_hist_after_unpark = session_a_loaded.history_tokens.len();
    engine.session = Some(session_a_loaded);
    println!("  5e: switched to session_a (unpark {unpark_ms}ms, pos={a_pos_after_unpark}, hist={a_hist_after_unpark})");

    // 5f: 验证 session_a 状态守恒 (pos + history 完全一致 = park/unpark 无损)
    let pos_conserved = a_pos_before_park == a_pos_after_unpark;
    let hist_conserved = a_hist_before_park == a_hist_after_unpark;
    println!("  5f: state conservation: pos {} ({a_pos_before_park}→{a_pos_after_unpark}), hist {} ({a_hist_before_park}→{a_hist_after_unpark})",
        if pos_conserved { "PASS" } else { "FAIL" },
        if hist_conserved { "PASS" } else { "FAIL" });

    // 5g: 回复验证 (info only — 依赖模型质量 + max_tokens, 不影响 pass/fail)
    let out_a_recall = engine.session_reply("我叫什么名字?", max_tokens, params.clone())?;
    let a_remembered = out_a_recall.contains("张三丰");
    let a_not_confused = !out_a_recall.contains("李四光");
    let recall_preview: String = out_a_recall.char_indices().take(60).last().map(|(i,_)| out_a_recall[..i].to_string()).unwrap_or_else(|| out_a_recall.clone());
    println!("  5g: session_a recall (info): {:?}", recall_preview);
    println!("      remembered 张三丰: {}, not confused with 李四光: {}",
        if a_remembered { "YES" } else { "NO" },
        if a_not_confused { "YES" } else { "NO" });

    // pass5 标准: 结构正确性 (list + state 守恒), 不含模型回答质量
    let pass5 = list_has_2 && list_a_inactive && list_b_active && pos_conserved && hist_conserved;
    println!("  result: {}", if pass5 { "PASS" } else { "FAIL" });
    println!();

    // ========== 测试 6: 异步后台 dump (park_async) ==========
    println!("[test 6] async park: park_async → join → verify file exists");
    let session_a = engine.session.take().unwrap();
    let t = std::time::Instant::now();
    let handle = mgr.park_async("session_a", session_a, &engine.config);
    let spawn_ms = t.elapsed().as_millis();
    println!("  6a: park_async spawned ({spawn_ms}ms, non-blocking)");

    // join 等待写盘完成
    let t = std::time::Instant::now();
    let join_ok = match handle.join() {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            eprintln!("  6b: async park returned error: {e}");
            false
        }
        Err(_) => {
            eprintln!("  6b: async park thread panicked");
            false
        }
    };
    let join_ms = t.elapsed().as_millis();
    println!("  6b: join completed ({join_ms}ms): ok={join_ok}");

    // 验证文件存在
    let a_path = ssd_dir.join("session_a.dzss");
    let file_exists = a_path.exists();
    let file_size = std::fs::metadata(&a_path).map(|m| m.len()).unwrap_or(0);
    println!("  6c: file {} exists={}, size={:.1} MB",
        a_path.display(), file_exists, file_size as f64 / 1024.0 / 1024.0);

    let pass6 = file_exists && file_size > 0 && join_ok;
    println!("  result: {}", if pass6 { "PASS" } else { "FAIL" });
    println!();

    // ========== 总结 ==========
    let total_pass = pass1 && r2_has_name && pass4 && pass5 && pass6;
    println!("=== Summary ===");
    println!("  test 1 (single-turn token match):    {}", if pass1 { "PASS" } else { "FAIL" });
    println!("  test 2a (R2 mentions name):          {}", if r2_has_name { "PASS" } else { "FAIL" });
    println!("  test 2b (R4 recalls prev Q):         {}", if r4_has_recall { "PASS" } else { "FAIL" });
    println!("  test 3 (prefill speedup):            info only");
    println!("  test 4 (SSD dump→load→reply):        {}", if pass4 { "PASS" } else { "FAIL" });
    println!("  test 5 (SessionManager LRU):         {}", if pass5 { "PASS" } else { "FAIL" });
    println!("  test 6 (async park):                 {}", if pass6 { "PASS" } else { "FAIL" });
    println!();
    println!("Overall: {}", if total_pass { "PASS" } else { "FAIL" });

    // 清理临时文件: 整个 selftest_tmp/ 目录一次性删除 (包含所有 selftest_*.txt / .dzss / mgr_sessions/)
    let _ = std::fs::remove_dir_all(&tmp_dir);

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // --help / -h
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }

    // --model 必需
    let gguf_path: PathBuf = match get_opt(&args, "--model") {
        Some(m) => m.into(),
        None => {
            eprintln!("Error: --model <gguf_path> is required");
            eprintln!();
            print_usage();
            return Err(daiza_engine::BonsaiError::Unsupported(
                "missing --model flag".into()
            ));
        }
    };

    let prompt = get_opt(&args, "--prompt").unwrap_or("你好").to_string();
    let max_tokens: usize = get_opt(&args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let is_inspect = args.iter().any(|a| a == "--inspect");
    let is_dump_template = args.iter().any(|a| a == "--dump-template");
    let is_interactive = args.iter().any(|a| a == "--interactive");
    let is_self_test = args.iter().any(|a| a == "--self-test");
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

    // --interactive 模式:引擎常驻, REPL 循环读取 stdin
    // 阶段 1.0:内部仍用 generate_with_params (全量重算), 验证交互流程
    // 阶段 1.1+ 将切换到 session_reply (增量 prefill, 复用 KV + SSM state)
    if is_interactive {
        println!("[daiza-cli] Interactive mode.");
        if dspark_path.is_some() {
            println!("[daiza-cli] DSpark: ENABLED");
        }
        if mmproj_path.is_some() {
            println!("[daiza-cli] Vision: ENABLED ({} image(s))", image_paths.len());
        }
        return run_repl(engine, params, "You are a helpful assistant.", max_tokens);
    }

    // --self-test 模式:自动化验证 session_reply 正确性
    if is_self_test {
        return run_self_test(engine, max_tokens);
    }

    // 默认单次生成模式
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

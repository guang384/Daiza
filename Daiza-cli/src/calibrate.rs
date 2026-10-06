//! `--calibrate` 自调优: 交错子进程 sweep 性能旋钮 + 确认门限, 输出推荐环境变量。
//!
//! ## 为什么用子进程
//! 三个旋钮 (`DAIZA_GEMM_T_SUB` / `DAIZA_ACTIVE_WORKERS` / `DAIZA_MATVEC_CHUNK`)
//! 均经 OnceLock 进程级缓存 (消除热路径 env::var 开销) → 进程内不可改写,
//! sweep 必须 spawn 子进程 (每个子进程全新 env + 全新 OnceLock)。
//!
//! ## 方法论 (交错 A/B 纪律)
//! 热耦合机器 (笔记本) 上短跑基准的槽位噪声可达 ±5%~50%, 单次 sweep 不可信。
//! 因此分两层:
//! 1. **sweep**: 每候选 × 2 轮蛇形交错 (各占早/晚热态槽位), 取 min — 只做初筛;
//! 2. **确认门限**: sweep 赢家若非默认值, 与默认值头对头 4 轮交错 [赢,默,默,赢],
//!    双方各取 min; 仅当赢家 min 比默认 min 好 ≥5% 才推荐覆盖, 否则保持默认。
//!    (225H 实测: t_sub 32 vs 64 E2E min 差 0.1% — 无门限会把槽位噪声当发现)
//! - 子进程先 `env_remove` 全部被 sweep 旋钮再显式设置, 用户既有配置不污染基准;
//! - 指标: prefill ms (GEMM t_sub); decode ms/tok (ACTIVE_WORKERS 热平衡 /
//!   MATVEC_CHUNK)。确认阶段 ACTIVE_WORKERS 用 128t (长跑稳态才是热平衡目标场景)。
//!   prefill 对 ACTIVE_WORKERS 不敏感 (n_batch ≥ 32 自动 boost 全核)。
//!
//! ## 用法
//!   daiza-cli --model <gguf> --calibrate [--prompt <text>]
//!
//! 一次性操作, 耗时 ~8-15 分钟 (取决于多少阶段触发确认)。

use std::path::Path;
use std::process::Command;
use std::time::Duration;

/// sweep 时需要在子进程中清除的旋钮 (避免用户既有配置污染基准)
const SWEPT_VARS: &[&str] = &[
    "DAIZA_GEMM_T_SUB",
    "DAIZA_ACTIVE_WORKERS",
    "DAIZA_MATVEC_CHUNK",
    "DAIZA_PREFILL_WORKERS",
    "DAIZA_WAIT_MODE",
    "DAIZA_SPIN_ROUNDS",
    "DAIZA_KERNEL_MODE",
    "DAIZA_INT_KERNEL",
    "DAIZA_PROFILE",
    "DAIZA_PLD",
    "DAIZA_STREAM",
    "DAIZA_DUMP_TOKENS",
];

/// 内置基准 prompt (~142 token; GEMM dispatch 需 n_batch ≥ 64, 太短无法校准 t_sub)
const DEFAULT_PROMPT: &str = "The quick brown fox jumps over the lazy dog. Artificial intelligence has transformed how modern software systems process language and reason about complex problems. Researchers continue to push the boundaries of what neural networks can achieve across vision, speech and structured reasoning. Scaling laws suggest that larger models trained on more data keep improving in predictable ways. Efficient inference on consumer hardware remains one of the most important engineering challenges today. Quantization reduces memory bandwidth requirements while preserving most of the model quality. Sparse and dense architectures continue to evolve in complementary directions. Future systems will combine specialized hardware with algorithmic advances to deliver real time intelligence on ordinary devices.";

/// 推荐覆盖所需的最小优势 (低于此视为槽位噪声, 保持默认)
const CONFIRM_GAIN_MIN: f64 = 0.05;

/// 一个 sweep 候选: `env_value = None` 表示清除该旋钮 (走编译期默认值)
#[derive(Clone)]
struct Candidate {
    label: String,
    env_value: Option<String>,
}

impl Candidate {
    fn explicit(v: String) -> Self {
        let label = v.clone();
        Self { label, env_value: Some(v) }
    }
    /// 编译期默认 (子进程 env_remove; 与硬编码默认值等价但不随代码漂移)
    fn default() -> Self {
        Self { label: "默认".into(), env_value: None }
    }
    fn envs_for<'a>(&'a self, env_key: &'a str, base_envs: &'a [(&'a str, &'a str)]) -> Vec<(&'a str, Option<&'a str>)> {
        let mut envs: Vec<(&str, Option<&str>)> =
            base_envs.iter().map(|(k, v)| (*k, Some(*v))).collect();
        match &self.env_value {
            Some(v) => envs.push((env_key, Some(v))),
            None => envs.push((env_key, None)),
        }
        envs
    }
}

/// 子进程 [bench] 解析结果
struct ChildBench {
    prefill_ms: f64,
    /// decode 段 ms/tok。max_tokens ≥ 1 时 [bench] 总会输出 decode 行
    /// (`decode(1t)=145ms (~145ms/tok)`), 但单 token 均值无统计意义 —
    /// Stage 1 (max-tokens 1) 只用 prefill 指标, 不读此字段
    decode_ms_per_tok: f64,
}

/// 运行一个子进程基准并解析 [bench] 行 (stderr)
fn run_child(
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    envs: &[(&str, Option<&str>)],
) -> Result<ChildBench, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut cmd = Command::new(exe);
    cmd.arg("--model").arg(model)
        .arg("--prompt").arg(prompt)
        .arg("--max-tokens").arg(max_tokens.to_string())
        .arg("--greedy")
        .stdout(std::process::Stdio::null())   // 生成文本不进终端
        .stderr(std::process::Stdio::piped()); // [bench] 行在 stderr
    for v in SWEPT_VARS {
        cmd.env_remove(v);
    }
    cmd.env("DAIZA_STREAM", "0");
    for (k, v) in envs {
        match v {
            Some(val) => {
                cmd.env(k, val);
            }
            None => {
                cmd.env_remove(k);
            }
        }
    }
    let out = cmd.output().map_err(|e| e.to_string())?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        return Err(format!(
            "子进程退出码 {:?}, stderr 尾部: {:?}",
            out.status,
            stderr.lines().last()
        ));
    }
    let mut prefill_ms = f64::NAN;
    let mut decode_ms = f64::NAN;
    for line in stderr.lines() {
        if let Some(p) = parse_prefill_ms(line) {
            prefill_ms = p;
        }
        if let Some(d) = parse_decode_ms_per_tok(line) {
            decode_ms = d;
        }
    }
    if prefill_ms.is_nan() {
        return Err("未找到 [bench] prefill 行".into());
    }
    Ok(ChildBench { prefill_ms, decode_ms_per_tok: decode_ms })
}

/// `[bench] load=2239ms prefill(142t)=10715ms (~75ms/tok)` → 10715
fn parse_prefill_ms(line: &str) -> Option<f64> {
    let i = line.find("prefill(")?;
    let rest = &line[i..];
    let j = rest.find(")=")?;
    parse_leading_float(&rest[j + 2..])
}

/// `[bench] decode(32t)=4832ms (~151.0ms/tok ~6.62 tok/s)` → 151.0
fn parse_decode_ms_per_tok(line: &str) -> Option<f64> {
    let i = line.find("decode(")?;
    let rest = &line[i..];
    let j = rest.find("(~")?;
    parse_leading_float(&rest[j + 2..])
}

fn parse_leading_float(s: &str) -> Option<f64> {
    let end = s.find("ms").unwrap_or(s.len());
    s[..end].trim().parse().ok()
}

#[derive(Clone, Copy)]
enum Metric {
    PrefillMs,
    DecodeMsPerTok,
}

impl Metric {
    fn unit(self) -> &'static str {
        match self {
            Metric::PrefillMs => "ms",
            Metric::DecodeMsPerTok => "ms/tok",
        }
    }
    fn pick(self, b: &ChildBench) -> f64 {
        match self {
            Metric::PrefillMs => b.prefill_ms,
            Metric::DecodeMsPerTok => b.decode_ms_per_tok,
        }
    }
}

/// 运行一个候选一次, 返回指标值 (失败返回 None)
fn bench_once(
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    env_key: &str,
    cand: &Candidate,
    base_envs: &[(&str, &str)],
    metric: Metric,
) -> Option<f64> {
    let envs = cand.envs_for(env_key, base_envs);
    match run_child(model, prompt, max_tokens, &envs) {
        Ok(b) => {
            let m = metric.pick(&b);
            if m.is_nan() {
                eprintln!("  {label:<12} 解析失败 (无对应指标段)", label = cand.label);
                None
            } else {
                println!(
                    "  {label:<12} {m:8.1} {unit}  (prefill {p:.0}ms)",
                    label = cand.label,
                    unit = metric.unit(),
                    p = b.prefill_ms,
                );
                Some(m)
            }
        }
        Err(e) => {
            eprintln!("  {label:<12} 运行失败: {e}", label = cand.label);
            None
        }
    }
}

/// 单阶段 sweep: 每候选 × 2 轮蛇形交错 (pass 0 正序, pass 1 逆序)。
/// 返回 (最优候选下标, 每候选样本列表)。
#[allow(clippy::too_many_arguments)]
fn sweep_stage(
    title: &str,
    env_key: &str,
    candidates: &[Candidate],
    base_envs: &[(&str, &str)],
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    cooldown: Duration,
    metric: Metric,
) -> Result<(usize, Vec<Vec<f64>>), String> {
    println!("[calibrate] {title} (env: {env_key}, max-tokens {max_tokens}, 交错×2)");
    let n = candidates.len();
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); n];
    let order: Vec<usize> = (0..n).chain((0..n).rev()).collect();
    for &ci in &order {
        if let Some(m) = bench_once(
            model, prompt, max_tokens, env_key, &candidates[ci], base_envs, metric,
        ) {
            samples[ci].push(m);
        }
        std::thread::sleep(cooldown);
    }
    for (ci, s) in samples.iter().enumerate() {
        println!(
            "  汇总 {label:<10} 样本 [{vals}] → min {min:.1} {unit}",
            label = candidates[ci].label,
            vals = s.iter().map(|v| format!("{v:.0}")).collect::<Vec<_>>().join("/"),
            min = s.iter().cloned().reduce(f64::min).unwrap_or(f64::INFINITY),
            unit = metric.unit(),
        );
    }
    let winner = samples
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_empty())
        .min_by(|a, b| {
            a.1.iter().cloned().reduce(f64::min).unwrap()
                .partial_cmp(&b.1.iter().cloned().reduce(f64::min).unwrap())
                .unwrap()
        })
        .map(|(i, _)| i)
        .ok_or_else(|| format!("{title}: 所有候选均失败"))?;
    println!("  → sweep 初筛最优: {label}", label = candidates[winner].label);
    Ok((winner, samples))
}

/// 确认门限: sweep 赢家 vs 默认, 4 轮交错 [赢,默,默,赢], 各取 min。
/// 赢家 min 比默认 min 好 ≥ CONFIRM_GAIN_MIN 时返回 Ok(true) (覆盖默认), 否则 Ok(false);
/// 任一侧两轮均运行失败时保守返回 Ok(false)。
#[allow(clippy::too_many_arguments)]
fn confirm_override(
    env_key: &str,
    default: &Candidate,
    winner: &Candidate,
    base_envs: &[(&str, &str)],
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    cooldown: Duration,
    metric: Metric,
) -> Result<bool, String> {
    println!(
        "[calibrate] 确认门限: {w} vs {d} (4 轮交错, ≥{pct:.0}% 优势才覆盖默认, max-tokens {max_tokens})",
        w = winner.label,
        d = default.label,
        pct = CONFIRM_GAIN_MIN * 100.0,
    );
    // [赢家, 默认, 默认, 赢家]: 双方各占一个早/晚槽位
    let order: Vec<(&Candidate, bool)> = vec![
        (winner, true),
        (default, false),
        (default, false),
        (winner, true),
    ];
    let mut w_min = f64::INFINITY;
    let mut d_min = f64::INFINITY;
    for (cand, is_winner) in &order {
        if let Some(m) = bench_once(model, prompt, max_tokens, env_key, cand, base_envs, metric) {
            if *is_winner {
                w_min = w_min.min(m);
            } else {
                d_min = d_min.min(m);
            }
        }
        std::thread::sleep(cooldown);
    }
    if !w_min.is_finite() || !d_min.is_finite() {
        println!("  确认轮运行失败, 保守保持默认");
        return Ok(false);
    }
    let gain = (d_min - w_min) / d_min;
    println!(
        "  {w_label} min {w_min:.1} vs {d_label} min {d_min:.1} {unit} → 优势 {gain_pct:+.1}%",
        w_label = winner.label,
        d_label = default.label,
        unit = metric.unit(),
        gain_pct = gain * 100.0,
    );
    let confirmed = gain >= CONFIRM_GAIN_MIN;
    if confirmed {
        println!("  → 确认: 覆盖默认");
    } else {
        println!("  → 未达门限, 保持默认 (差异在槽位噪声内)");
    }
    Ok(confirmed)
}

/// sweep + 确认的组合流程, 返回最终推荐的候选 (克隆, 避免借用纠缠)
#[allow(clippy::too_many_arguments)]
fn calibrate_knob(
    title: &str,
    env_key: &str,
    candidates: &[Candidate],
    default_idx: usize,
    base_envs: &[(&str, &str)],
    model: &Path,
    prompt: &str,
    sweep_tokens: usize,
    confirm_tokens: usize,
    cooldown: Duration,
    metric: Metric,
) -> Result<Candidate, String> {
    let (winner_idx, _) = sweep_stage(
        title, env_key, candidates, base_envs, model, prompt, sweep_tokens, cooldown, metric,
    )?;
    if winner_idx == default_idx {
        println!("  → 与默认一致, 无需覆盖");
        return Ok(candidates[default_idx].clone());
    }
    let confirmed = confirm_override(
        env_key,
        &candidates[default_idx],
        &candidates[winner_idx],
        base_envs,
        model,
        prompt,
        confirm_tokens,
        cooldown,
        metric,
    )?;
    Ok(if confirmed { candidates[winner_idx].clone() } else { candidates[default_idx].clone() })
}

/// `--calibrate` 入口
pub fn run(model: &Path, prompt: Option<&str>) -> Result<(), String> {
    let prompt = prompt.unwrap_or(DEFAULT_PROMPT);
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let n_workers = n_threads.saturating_sub(1).max(1);
    println!("[calibrate] 逻辑核 {n_threads} ({n_workers} worker + main)");
    println!(
        "[calibrate] 方法: sweep 初筛 (交错×2 min) + 确认门限 (4 轮头对头, ≥{pct:.0}% 才覆盖)",
        pct = CONFIRM_GAIN_MIN * 100.0,
    );
    println!("[calibrate] 一次性操作, 耗时 ~8-15 分钟; 中途可 Ctrl+C 中止");

    // ---- Stage 1: DAIZA_GEMM_T_SUB (prefill GEMM 形状参数) ----
    let tsub_cands = vec![
        Candidate::default(), // idx 0 = 编译默认 (32)
        Candidate::explicit("64".into()),
        Candidate::explicit("128".into()),
    ];
    let tsub = calibrate_knob(
        "Stage 1/3: GEMM_T_SUB (prefill)",
        "DAIZA_GEMM_T_SUB",
        &tsub_cands,
        0,
        &[],
        model,
        prompt,
        1,
        1,
        Duration::from_secs(8),
        Metric::PrefillMs,
    )?;

    // ---- Stage 2: DAIZA_ACTIVE_WORKERS (decode 长跑热平衡) ----
    // 出厂默认 min(9, n_workers); 候选 = 默认 + 全核/-2/-4/-6 档位 (去重)
    let aw_default = n_workers.min(9);
    let mut aw_cands: Vec<Candidate> = vec![Candidate {
        label: format!("{aw_default} (默认)"),
        env_value: Some(aw_default.to_string()),
    }];
    for off in [0usize, 2, 4, 6] {
        let v = n_workers.saturating_sub(off).max(1);
        if aw_cands.iter().any(|c| c.env_value.as_deref() == Some(v.to_string().as_str())) {
            continue;
        }
        let label = if off == 0 { format!("{v} (全核)") } else { v.to_string() };
        aw_cands.push(Candidate { label, env_value: Some(v.to_string()) });
    }
    let aw = calibrate_knob(
        "Stage 2/3: ACTIVE_WORKERS (decode 热平衡)",
        "DAIZA_ACTIVE_WORKERS",
        &aw_cands,
        0,
        &[],
        model,
        prompt,
        64,
        128, // 确认用 128t 长跑 — 热平衡的目标场景
        Duration::from_secs(3),
        Metric::DecodeMsPerTok,
    )?;
    let aw_env: (&str, &str) = (
        "DAIZA_ACTIVE_WORKERS",
        aw.env_value.as_deref().unwrap_or("9"),
    );

    // ---- Stage 3: DAIZA_MATVEC_CHUNK (decode matvec work-stealing) ----
    let chunk_cands = vec![
        Candidate::explicit("64".into()),
        Candidate::default(), // idx 1 = 编译默认 (128)
        Candidate::explicit("256".into()),
    ];
    let chunk = calibrate_knob(
        "Stage 3/3: MATVEC_CHUNK (decode matvec 窃取粒度)",
        "DAIZA_MATVEC_CHUNK",
        &chunk_cands,
        1,
        &[aw_env],
        model,
        prompt,
        64,
        64,
        Duration::from_secs(3),
        Metric::DecodeMsPerTok,
    )?;

    // ---- 报告 ----
    println!();
    println!("[calibrate] ─────────── 推荐配置 ───────────");
    print_env_line("DAIZA_GEMM_T_SUB", tsub.env_value.as_ref(), "prefill GEMM t-subdivision");
    print_env_line("DAIZA_ACTIVE_WORKERS", aw.env_value.as_ref(), "长跑热平衡 (worker 数, 不含 main)");
    print_env_line("DAIZA_MATVEC_CHUNK", chunk.env_value.as_ref(), "decode matvec work-stealing 粒度");
    println!();
    println!("[calibrate] 与默认一致时无需设置; 上述 $env: 行可直接粘贴到当前会话, 写入 $PROFILE 可持久化。");
    println!("[calibrate] 注意: decode 为短跑近似指标, 长跑稳态以实际 workload 验证为准;");
    println!("[calibrate] 热耦合机器 (笔记本) 槽位噪声大, 已用交错+min+确认门限抑制; 换硬件/散热变化后重跑。");
    Ok(())
}

fn print_env_line(key: &str, value: Option<&String>, desc: &str) {
    match value {
        Some(v) => println!("  $env:{key:<22} = '{v}'   # {desc}"),
        None => println!("  $env:{key:<22}  <默认即可>         # {desc}"),
    }
}

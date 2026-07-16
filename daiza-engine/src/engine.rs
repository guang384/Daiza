//! 顶层推理引擎入口
//!
//! 把 GGUF 解析 → 配置 → 权重加载 → 前向传播 → 采样 → 解码 串起来。
//!
//! ## 使用方式
//!
//! ```no_run
//! use daiza_engine::engine::Engine;
//! let mut engine = Engine::load("../Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf").unwrap();
//! let out = engine.generate("你好", 64).unwrap();
//! println!("{out}");
//! ```

use std::path::Path;

use crate::gguf::parser::GgufFile;
use crate::math::{sample_top_k_top_p_into, SamplingBuffers, SamplingParams};
use crate::model::config::Config;
use crate::model::forward::{forward_single_token, make_context};
use crate::model::weights::LoadedWeights;
use crate::tokenizer::vocab::Vocab;
use crate::tokenizer::BpeTokenizer;
use crate::Result;

/// 简单的伪随机数(零依赖)
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }
    fn next_f32(&mut self) -> f32 {
        // Numerical Recipes LCG
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = (self.state >> 33) as u32;
        (x as f32) / (u32::MAX as f32)
    }
}

pub struct Engine {
    pub gguf: GgufFile,
    pub config: Config,
    pub tokenizer: BpeTokenizer,
    pub weights: Option<LoadedWeights>,
}

impl Engine {
    /// 加载 GGUF + tokenizer(不预加载权重)
    pub fn load(path: &Path) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let config = Config::from_metadata(&gguf.metadata)?;
        let vocab = Vocab::from_metadata(&gguf.metadata)?;
        let tokenizer = BpeTokenizer::new(vocab);
        Ok(Self {
            gguf,
            config,
            tokenizer,
            weights: None,
        })
    }

    /// 只解析 GGUF 头部(用于调试)
    pub fn load_metadata_only(path: &Path) -> Result<Self> {
        Self::load(path)
    }

    /// 一次性加载所有 block 权重到内存(约 13GB)
    pub fn load_weights(&mut self) -> Result<()> {
        let w = LoadedWeights::load_all(&self.gguf, &self.config)?;
        self.weights = Some(w);
        Ok(())
    }

    /// 完整生成循环
    ///
    /// - `prompt`:输入文本
    /// - `max_tokens`:最多生成多少 token(不含 prompt)
    /// - 返回:生成的文本(不含 prompt)
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate_with_params(prompt, max_tokens, SamplingParams::default(), None)
    }

    /// 完整生成循环(带可配置参数与系统 prompt)
    /// raw=true 时跳过 chat 模板,直接编码 prompt(用于调试)
    pub fn generate_with_params(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        params: SamplingParams,
        system_prompt: Option<&str>,
    ) -> Result<String> {
        self.generate_inner(prompt, max_tokens, params, system_prompt, false)
    }

    /// 原始模式:跳过 chat 模板,直接编码 prompt(调试用)
    pub fn generate_raw(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        params: SamplingParams,
    ) -> Result<String> {
        self.generate_inner(prompt, max_tokens, params, None, true)
    }

    fn generate_inner(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        params: SamplingParams,
        system_prompt: Option<&str>,
        raw: bool,
    ) -> Result<String> {
        // 1. 构造输入
        let chat_text = if raw {
            eprintln!("[debug] raw mode, prompt as-is");
            prompt.to_string()
        } else {
            build_chat_input(prompt, system_prompt)
        };
        eprintln!("[debug] input text: {chat_text:?}");
        let input_ids = self.tokenizer.encode(&chat_text);
        eprintln!("[debug] input_ids count: {}", input_ids.len());
        eprintln!("[debug] first 10 ids: {:?}", &input_ids[..input_ids.len().min(10)]);
        eprintln!("[debug] last 10 ids:  {:?}", &input_ids[input_ids.len().saturating_sub(10)..]);
        // 打印特殊 token 数量
        eprintln!("[debug] special tokens found: {}", self.tokenizer.special_tokens.len());
        if input_ids.is_empty() {
            return Err(crate::BonsaiError::Tokenizer(
                "encode returned empty".into(),
            ));
        }

        // 2. 加载权重(若未加载)
        let load_start = std::time::Instant::now();
        if self.weights.is_none() {
            eprintln!("[engine] loading weights (one-shot, ~13GB)...");
            self.load_weights()?;
        }
        let load_ms = load_start.elapsed().as_millis();

        // 3. 构造前向上下文
        let cfg = &self.config;
        let weights = self.weights.as_ref().unwrap();
        let mut ctx = make_context(weights, cfg);

        // 4. prefill:逐 token 前向(只填充 KV/SSM 状态)
        let n_input = input_ids.len();
        let mut last_logits: Option<Vec<f32>> = None;
        let prefill_start = std::time::Instant::now();
        for (i, &tid) in input_ids.iter().enumerate() {
            last_logits = Some(forward_single_token(&mut ctx, tid)?);
            if i % 4 == 0 {
                eprint!("\r[prefill] {i}/{n_input}");
            }
        }
        eprintln!("\r[prefill] {n_input}/{n_input} done");
        let prefill_ms = prefill_start.elapsed().as_millis();
        eprintln!("[bench] load={load_ms}ms prefill({n_input}t)={prefill_ms}ms (~{}ms/tok)",
            if n_input > 0 { prefill_ms / n_input as u128 } else { 0 });

        // 4. decode:采样 → 前向 → 重复
        let mut rng = LcgRng::new(0xC0FFEE);
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut current_logits = last_logits.ok_or_else(|| {
            crate::BonsaiError::Model("no logits from prefill".into())
        })?;

        // [debug] 打印 prefill 后的 top-5 logits
        {
            eprintln!("[debug] logits len: {}, any NaN: {}",
                current_logits.len(),
                current_logits.iter().any(|x| x.is_nan()));
            let mut indexed: Vec<(usize, f32)> = current_logits.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[debug] top-10 logits after prefill:");
            for (tid, logit) in indexed.iter().take(10) {
                let tok_str = self.tokenizer.vocab.tokens.get(*tid).cloned().unwrap_or_default();
                eprintln!("  id={tid:>6} logit={logit:>10.4} token={tok_str:?}");
            }
            // 检查 logit 范围
            let max_l = indexed[0].1;
            let min_l = indexed.last().unwrap().1;
            eprintln!("[debug] logit range: [{min_l:.4}, {max_l:.4}]");
        }

        let decode_start = std::time::Instant::now();
        // 预分配采样 buffer(复用 scaled[248320] + indices[248320] = ~3MB),
        // 避免每 token 重新分配
        let mut sampling_buf = SamplingBuffers::new(current_logits.len());
        for step in 0..max_tokens {
            // 采样一个 token
            let next_id = sample_top_k_top_p_into(&current_logits, params, &mut || rng.next_f32(), &mut sampling_buf);
            // 遇到 EOS 提前停止
            if next_id as u32 == self.config.eos_token_id {
                break;
            }
            generated_ids.push(next_id as u32);

            // 前向得到新 logits
            current_logits = forward_single_token(&mut ctx, next_id as u32)?;

            eprint!("\r[decode] {step}/{max_tokens}");
            // 流式输出当前生成的 token 文本
            if let Some(s) = self.tokenizer.vocab.tokens.get(next_id as usize) {
                eprint!(" -> {s}");
            }
        }
        eprintln!();
        let decode_ms = decode_start.elapsed().as_millis();
        let n_gen = generated_ids.len();
        if n_gen > 0 {
            eprintln!("[bench] decode({n_gen}t)={decode_ms}ms (~{}ms/tok ~{:.2} tok/s)",
                decode_ms / n_gen as u128,
                n_gen as f64 * 1000.0 / decode_ms as f64);
        }

        // 5. decode token ids 为字符串
        Ok(self.tokenizer.decode(&generated_ids))
    }

    /// 打印引擎概要(用于验证加载是否正确)
    pub fn print_summary(&self) {
        println!("=== Bonsai Engine Summary ===");
        println!();
        println!("[GGUF]");
        println!("  version        : {}", self.gguf.version);
        println!("  alignment      : {}", self.gguf.alignment);
        println!("  tensor_count   : {}", self.gguf.tensors.len());
        println!("  metadata_kv    : {}", self.gguf.metadata.kv.len());
        println!("  data_section_offset : {}", self.gguf.data_section_offset);
        println!();

        println!("[Config]");
        println!("  block_count    : {}", self.config.block_count);
        println!("  context_length : {}", self.config.context_length);
        println!("  hidden         : {}", self.config.hidden);
        println!("  feed_forward   : {}", self.config.feed_forward_length);
        println!(
            "  head_count     : {} (KV: {})",
            self.config.head_count, self.config.head_count_kv
        );
        println!("  head_dim       : {}", self.config.head_dim);
        println!(
            "  rope_dim       : {} (base: {})",
            self.config.rope_dim, self.config.rope_freq_base
        );
        println!("  rope_sections  : {:?}", self.config.rope_dim_sections);
        println!("  rms_eps        : {}", self.config.rms_eps);
        println!(
            "  full_attn_interval : {} (full attn layers: {}, ssm layers: {})",
            self.config.full_attention_interval,
            self.config.full_attention_layer_indices().len(),
            self.config.ssm_layer_indices().len(),
        );
        println!();
        println!("[SSM]");
        println!("  conv_kernel    : {}", self.config.ssm_conv_kernel);
        println!("  state_size    : {}", self.config.ssm_state_size);
        println!("  group_count   : {}", self.config.ssm_group_count);
        println!("  time_step_rank : {}", self.config.ssm_time_step_rank);
        println!("  inner_size     : {}", self.config.ssm_inner_size);
        println!();
        println!("[Tokenizer]");
        println!("  vocab_size    : {}", self.tokenizer.vocab.vocab_size());
        println!("  merges count  : {}", self.tokenizer.vocab.merges.len());
        println!("  eos_token_id  : {}", self.tokenizer.vocab.eos_token_id);
        println!("  bos_token_id  : {}", self.tokenizer.vocab.bos_token_id);
        println!("  pad_token_id  : {}", self.tokenizer.vocab.pad_token_id);
        println!();
        println!("[Chat Template]");
        if let Some(tmpl) = self.gguf.metadata.get_str("tokenizer.chat_template") {
            println!("{tmpl}");
        } else {
            println!("  (not found)");
        }
        println!();
        println!(
            "[Full attention layers] {:?}",
            self.config.full_attention_layer_indices()
        );
        println!();
        println!("[Dtype histogram]");
        crate::model::weights::WeightLoader::print_dtype_summary(&self.gguf);
        println!();
        println!("[Sample tensor shapes]");
        for name in [
            "token_embd.weight",
            "output.weight",
            "output_norm.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ssm_conv1d.weight",
            "blk.0.ssm_a",
            "blk.0.ssm_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.3.attn_q.weight",
            "blk.3.attn_k.weight",
            "blk.3.attn_v.weight",
            "blk.3.attn_output.weight",
            "blk.3.attn_q_norm.weight",
        ] {
            if let Some(t) = self.gguf.find_tensor(name) {
                println!("  {:<32} dims={:?} dtype={}", t.name, t.dims, t.dtype.name());
            }
        }
        println!();
        println!("=== ready ===");
    }
}

/// 构造 Qwen3 chat 输入
///
/// 模板(从 GGUF chat_template 提取):
/// ```text
/// <|im_start|>system
/// {system}<|im_end|>
/// <|im_start|>user
/// {user}<|im_end|>
/// <|im_start|>assistant
/// mind
/// ```
/// 注意:此模型用 `mind\n` 标记思考模式开始(GGUF chat_template 原文),
/// 与标准 Qwen3 的 `<think>` 不同,但模型就是用这个模板训练的。
fn build_chat_input(user_prompt: &str, system_prompt: Option<&str>) -> String {
    let mut s = String::new();
    if let Some(sys) = system_prompt {
        s.push_str("<|im_start|>system\n");
        s.push_str(sys);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>user\n");
    s.push_str(user_prompt);
    s.push_str("<|im_end|>\n");
    s.push_str("<|im_start|>assistant\n");
    // 思考模式:chat_template 的 else 分支(默认 enable_thinking=true)
    // 输出 `<think>\n` 作为思考模式开始标记(从 GGUF 原始字节确认:3c 74 68 69 6e 6b 3e)
    s.push_str("<think>\n");
    s
}

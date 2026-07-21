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
use crate::model::dspark::{
    weights::DrafterWeights,
    speculative::SpeculativeContext,
};
use crate::model::forward::{forward_batch, forward_batch_with_vision, forward_single_token, forward_single_token_with_embedding, make_context, VisionInject, ForwardContext};
use crate::model::vision::{
    VisionConfig, VisionWeights,
    encoder::{ViTContext, encode_image},
    projector::{ProjectorContext, project_vision},
    preprocess_image,
};
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

/// DSpark 性能分析 helper
#[inline]
fn pct(part: u128, total: u128) -> f64 {
    if total == 0 { 0.0 } else { part as f64 * 100.0 / total as f64 }
}
#[inline]
fn ms_per(total_ms: u128, n: usize) -> u128 {
    if n == 0 { 0 } else { total_ms / n as u128 }
}

/// 多模态视觉上下文 (mmproj 加载后填充)
pub struct VisionContext {
    pub cfg: VisionConfig,
    pub weights: VisionWeights,
    pub vit_ctx: ViTContext,
    pub proj_ctx: ProjectorContext,
}

pub struct Engine {
    pub gguf: GgufFile,
    pub config: Config,
    pub tokenizer: BpeTokenizer,
    pub weights: Option<LoadedWeights>,
    /// DSpark drafter (可选, 由 load_drafter 加载)
    pub spec_ctx: Option<SpeculativeContext>,
    /// 多模态视觉编码器 (可选, 由 load_mmproj 加载)
    pub vision: Option<VisionContext>,
    /// 缓存的 image_token_id (从 tokenizer.special_tokens 查找, 懒初始化)
    pub image_token_id: Option<u32>,
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
            spec_ctx: None,
            vision: None,
            image_token_id: None,
        })
    }

    /// 加载 DSpark drafter (独立 GGUF 文件)
    pub fn load_drafter(&mut self, path: &Path) -> Result<()> {
        eprintln!("[dspark] Loading drafter GGUF: {}", path.display());
        let drafter_gguf = GgufFile::open(path)?;
        let weights = DrafterWeights::load(&drafter_gguf)?;
        let cfg = weights.cfg.clone();
        eprintln!("[dspark] Drafter config: {} blocks, hidden={}, block_size={}, markov_rank={}",
            cfg.block_count, cfg.embedding_length, cfg.block_size, cfg.markov_rank);
        self.spec_ctx = Some(SpeculativeContext::new(weights));
        Ok(())
    }

    /// 加载多模态视觉编码器 (mmproj GGUF)
    ///
    /// 加载后 engine 可执行 generate_with_image 多模态推理
    pub fn load_mmproj(&mut self, path: &Path) -> Result<()> {
        eprintln!("[vision] Loading mmproj GGUF: {}", path.display());
        let mmproj_gguf = GgufFile::open(path)?;
        let cfg = VisionConfig::from_metadata(&mmproj_gguf.metadata)?;
        eprintln!("[vision] mmproj config: {} blocks, n_embd={}, image={}x{}, patch={}, n_patches={}, n_merged={}, proj_dim={}",
            cfg.block_count, cfg.embedding_length,
            cfg.image_size, cfg.image_size, cfg.patch_size,
            cfg.n_patches, cfg.n_patches_merged(), cfg.projection_dim);
        let weights = VisionWeights::load(&mmproj_gguf, &cfg)?;
        let vit_ctx = ViTContext::new(&cfg);
        let proj_ctx = ProjectorContext::new(&cfg);
        eprintln!("[vision] mmproj loaded successfully");
        self.vision = Some(VisionContext { cfg, weights, vit_ctx, proj_ctx });
        Ok(())
    }

    /// 查找 image_token 的 token id (Qwen3-VL 标准命名 <|image_pad|>)
    /// 优先查 tokenizer.ggml.image_token_id metadata, 否则从 special_tokens map 查找
    pub fn image_token_id(&mut self) -> Result<u32> {
        if let Some(id) = self.image_token_id {
            return Ok(id);
        }
        // 1. 尝试 GGUF metadata
        if let Some(id) = self.gguf.metadata.get_u32("tokenizer.ggml.image_token_id") {
            self.image_token_id = Some(id);
            return Ok(id);
        }
        // 2. 从 special_tokens 查找标准命名
        for name in ["<|image_pad|>", "<|vision_pad|>", "<|image_start|>"] {
            if let Some(&id) = self.tokenizer.special_tokens.get(name) {
                eprintln!("[vision] image_token: {name} → id={id}");
                self.image_token_id = Some(id);
                return Ok(id);
            }
        }
        Err(crate::BonsaiError::Unsupported(
            "image_token_id not found: GGUF has neither tokenizer.ggml.image_token_id metadata nor <|image_pad|>/<|vision_pad|> special token".into()
        ))
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

    /// 多模态生成:输入文本 prompt + 一张或多张图像路径
    ///
    /// 流程:
    /// 1. 对每张图: preprocess → ViT encode → project → 得到 n_vision_per_image 个 vision embeddings (hidden dim)
    /// 2. 在 prompt 文本中插入 image_token 占位符 (每个图像一个 <|image_pad|>)
    /// 3. 用 forward_batch_with_vision 做 prefill: image_token 位置展开为对应 vision embeddings
    /// 4. 后续 decode 与 text-only generate 一致
    ///
    /// ★ text-only decode 零退化: vision 注入只在 prefill 阶段, decode 走 forward_single_token 不修改
    pub fn generate_with_image(
        &mut self,
        prompt: &str,
        image_paths: &[std::path::PathBuf],
        max_tokens: usize,
        params: SamplingParams,
        system_prompt: Option<&str>,
    ) -> Result<String> {
        if self.vision.is_none() {
            return Err(crate::BonsaiError::Unsupported(
                "mmproj not loaded; call load_mmproj() first".into()
            ));
        }
        if image_paths.is_empty() {
            // 退化到纯文本
            return self.generate_with_params(prompt, max_tokens, params, system_prompt);
        }

        // 1. 加载 target 权重 + 线程池 (若未加载)
        if self.weights.is_none() {
            eprintln!("[engine] loading target weights...");
            self.load_weights()?;
            let n_threads = crate::model::workspace::thread_count();
            crate::model::workspace::init_thread_pool(n_threads);
            eprintln!("[engine] thread pool ({n_threads} workers) initialized");
        }

        // 2. 查找 image_token_id
        let image_token_id = self.image_token_id()?;

        // 3. 对每张图做 preprocess → encode → project, 拼接 vision_embeddings
        let n_vision_per_image;
        let vision_embeddings: Vec<f32> = {
            let vision = self.vision.as_mut().unwrap();
            n_vision_per_image = vision.cfg.n_patches_merged();
            let proj_dim = vision.cfg.projection_dim;
            let hidden = self.config.hidden;
            if proj_dim != hidden {
                return Err(crate::BonsaiError::Model(format!(
                    "mmproj projection_dim ({proj_dim}) != text model hidden ({hidden}), vision embeddings 无法直接注入"
                )));
            }
            let mut all_emb = Vec::with_capacity(image_paths.len() * n_vision_per_image * hidden);
            for img_path in image_paths {
                let t0 = std::time::Instant::now();
                let patches = preprocess_image(img_path, &vision.cfg)?;
                let t_pre = t0.elapsed();

                let t1 = std::time::Instant::now();
                encode_image(&vision.weights, &vision.cfg, &mut vision.vit_ctx, &patches)?;
                let t_enc = t1.elapsed();

                let t2 = std::time::Instant::now();
                let vit_out = vision.vit_ctx.hidden.clone(); // [n_patches, n_embd]
                project_vision(&vit_out, &vision.weights, &vision.cfg, &mut vision.proj_ctx)?;
                let t_proj = t2.elapsed();

                let proj_out = &vision.proj_ctx.projected;
                all_emb.extend_from_slice(proj_out);
                eprintln!("[vision] image {}: pre={:.1}ms enc={:.1}ms proj={:.1}ms total={:.1}ms ({} patches → {} merged)",
                    img_path.display(),
                    t_pre.as_secs_f64() * 1000.0,
                    t_enc.as_secs_f64() * 1000.0,
                    t_proj.as_secs_f64() * 1000.0,
                    (t_pre + t_enc + t_proj).as_secs_f64() * 1000.0,
                    vision.cfg.n_patches, n_vision_per_image);
            }
            all_emb
        };

        // 4. 构造输入文本: 在 prompt 中插入 image_token
        //    简单策略: 在 user 消息开头插入 N 个 image_token (N = 图像数)
        //    每个 image_token 会被 forward_batch_with_vision 展开为 n_vision_per_image 个 vision embeddings
        let image_token_str = self.tokenizer.vocab.tokens.get(image_token_id as usize)
            .cloned().unwrap_or_else(|| "<|image_pad|>".to_string());
        let mut image_section = String::new();
        for _ in 0..image_paths.len() {
            image_section.push_str(&image_token_str);
        }
        let chat_text = build_chat_input_with_image(&image_section, prompt, system_prompt);
        eprintln!("[debug] input text: {chat_text:?}");
        let input_ids = self.tokenizer.encode(&chat_text);
        eprintln!("[debug] input_ids count: {} (含 {} 个 image_token, 展开为 {} 个 vision embeddings)",
            input_ids.len(),
            image_paths.len(),
            image_paths.len() * n_vision_per_image);
        if input_ids.is_empty() {
            return Err(crate::BonsaiError::Tokenizer("encode returned empty".into()));
        }

        // 5. 构造前向上下文
        let cfg = &self.config;
        let weights = self.weights.as_ref().unwrap();
        let mut ctx = make_context(weights, cfg);

        // 6. prefill: text tokens 用 forward_batch (batched), vision embeddings 逐个注入
        //    ★ 策略:
        //      - text tokens: 收集成 batch (≤32), 用 forward_batch 一次读 13GB 权重
        //      - image_token: 展开为 n_vision_per_image 个 vision embeddings,
        //        用 forward_single_token_with_embedding 逐个注入 (不在热路径, 一次性成本)
        //    ★ 不使用 forward_batch_with_vision, 因 batched Q1_0 kernel tmp buffer 上限 64,
        //      而 n_vision_per_image=576 远超上限
        let n_input = input_ids.len();
        let prefill_start = std::time::Instant::now();
        const MAX_TEXT_BATCH: usize = 32;
        let mut text_batch: Vec<u32> = Vec::with_capacity(MAX_TEXT_BATCH);
        let mut vision_offset = 0usize; // 以 hidden-dim 为单位的偏移
        let hidden = self.config.hidden;

        // flush 当前 text batch
        let flush_text_batch = |batch: &mut Vec<u32>, ctx: &mut ForwardContext<'_>| -> crate::Result<()> {
            if batch.is_empty() { return Ok(()); }
            if batch.len() == 1 {
                forward_single_token(ctx, batch[0])?;
            } else {
                forward_batch(ctx, batch, ctx.state.pos, None)?;
            }
            batch.clear();
            Ok(())
        };

        for &tid in &input_ids {
            if tid == image_token_id {
                // 先 flush 累积的 text batch
                flush_text_batch(&mut text_batch, &mut ctx)?;
                // 逐个注入 n_vision_per_image 个 vision embeddings
                for vi in 0..n_vision_per_image {
                    let emb_start = (vision_offset + vi) * hidden;
                    let emb_end = emb_start + hidden;
                    let emb = &vision_embeddings[emb_start..emb_end];
                    forward_single_token_with_embedding(&mut ctx, emb)?;
                }
                vision_offset += n_vision_per_image;
            } else {
                text_batch.push(tid);
                if text_batch.len() >= MAX_TEXT_BATCH {
                    flush_text_batch(&mut text_batch, &mut ctx)?;
                }
            }
        }
        flush_text_batch(&mut text_batch, &mut ctx)?;

        let prefill_ms = prefill_start.elapsed().as_millis();
        let n_image_tokens = input_ids.iter().filter(|&&t| t == image_token_id).count();
        let n_batch_expanded = n_input - n_image_tokens + n_image_tokens * n_vision_per_image;
        eprintln!("\r[prefill] {n_input} input tokens (expanded to {n_batch_expanded} with vision) done in {prefill_ms}ms");

        // 7. decode: 与 text-only generate 完全一致
        let mut rng = LcgRng::new(0xC0FFEE);
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);
        let stream_output = !matches!(std::env::var("DAIZA_STREAM").as_deref(),
            Ok("0") | Ok("false") | Ok("no"));
        let mut sampling_buf = SamplingBuffers::new(ctx.logits_buf.len());

        let decode_start = std::time::Instant::now();
        for step in 0..max_tokens {
            let next_id = sample_top_k_top_p_into(
                &ctx.logits_buf, params, &mut || rng.next_f32(), &mut sampling_buf,
            );
            if next_id as u32 == self.config.eos_token_id {
                break;
            }
            generated_ids.push(next_id as u32);
            forward_single_token(&mut ctx, next_id as u32)?;
            if stream_output {
                eprint!("\r[decode] {step}/{max_tokens}");
                if let Some(s) = self.tokenizer.vocab.tokens.get(next_id as usize) {
                    eprint!(" -> {s}");
                }
            }
        }
        if stream_output {
            eprintln!();
        }
        let decode_ms = decode_start.elapsed().as_millis();
        let n_gen = generated_ids.len();
        if n_gen > 0 {
            eprintln!("[bench] vision decode({n_gen}t)={decode_ms}ms (~{}ms/tok ~{:.2} tok/s)",
                decode_ms / n_gen as u128,
                n_gen as f64 * 1000.0 / decode_ms as f64);
        }

        Ok(self.tokenizer.decode(&generated_ids))
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
            // 初始化持久线程池(消除每 token ~369 次 scope 创建 + ~2952 次 thread spawn)
            let n_threads = crate::model::workspace::thread_count();
            crate::model::workspace::init_thread_pool(n_threads);
            eprintln!("[engine] thread pool ({n_threads} workers) initialized");
            // GGUF 文件已通过 mmap 映射, 权重加载时 to_vec() 复制到独立缓冲,
            // mmap 区域由内核按需 page-in, 物理内存占用远小于文件大小。
            // 加载完成后 mmap 仍保留(用于 --inspect 等场景), 但未访问的 page
            // 不占物理内存。
        }
        let load_ms = load_start.elapsed().as_millis();

        // 3. 构造前向上下文
        let cfg = &self.config;
        let weights = self.weights.as_ref().unwrap();
        let mut ctx = make_context(weights, cfg);

        // 4. prefill: 批量前向(一次读权重, 13GB 只读一次而非 N 次)
        //    logits 直接写入 ctx.logits_buf,decode 阶段复用同一 buffer(无 clone)
        let n_input = input_ids.len();
        let prefill_start = std::time::Instant::now();
        if n_input == 1 {
            forward_single_token(&mut ctx, input_ids[0])?;
        } else if n_input > 1 {
            forward_batch(&mut ctx, &input_ids, 0, None)?;
        }
        let prefill_ms = prefill_start.elapsed().as_millis();
        eprintln!("\r[prefill] {n_input}/{n_input} done");
        eprintln!("[bench] load={load_ms}ms prefill({n_input}t)={prefill_ms}ms (~{}ms/tok)",
            if n_input > 0 { prefill_ms / n_input as u128 } else { 0 });

        if n_input == 0 {
            return Ok(String::new());
        }

        // 5. decode:采样 → 前向 → 重复
        //    每 token 直接读 ctx.logits_buf 采样,前向覆盖 ctx.logits_buf(无 clone)
        let mut rng = LcgRng::new(0xC0FFEE);
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);

        // [debug] 打印 prefill 后的 top-K logits(用 DAIZA_DEBUG_LOGITS env var 控制,
        //   避免每次生成都做一次 O(n log n) sort on 248320 元素,~15-20ms 损耗)
        if std::env::var("DAIZA_DEBUG_LOGITS").is_ok() {
            let current_logits = &ctx.logits_buf;
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
            let max_l = indexed[0].1;
            let min_l = indexed.last().unwrap().1;
            eprintln!("[debug] logit range: [{min_l:.4}, {max_l:.4}]");
        }

        let decode_start = std::time::Instant::now();
        // ★ Benchmark 模式: DAIZA_STREAM=0 关闭每 token 的 eprint! 输出
        //   Windows stderr 未缓冲, 每 token eprint! 触发 WriteFile syscall (~0.5-3ms)
        //   benchmark 时设 DAIZA_STREAM=0 可消除 I/O 开销, 让 ms/tok 反映纯计算
        let stream_output = !matches!(std::env::var("DAIZA_STREAM").as_deref(),
            Ok("0") | Ok("false") | Ok("no"));
        // 预分配采样 buffer(复用 scaled[248320] + indices[248320] = ~3MB),
        // 避免每 token 重新分配
        let mut sampling_buf = SamplingBuffers::new(ctx.logits_buf.len());
        for step in 0..max_tokens {
            // 采样一个 token(直接读 ctx.logits_buf,无需 clone)
            let next_id = sample_top_k_top_p_into(
                &ctx.logits_buf, params, &mut || rng.next_f32(), &mut sampling_buf,
            );
            // 遇到 EOS 提前停止
            if next_id as u32 == self.config.eos_token_id {
                break;
            }
            generated_ids.push(next_id as u32);

            // 前向(覆盖 ctx.logits_buf,无 clone)
            forward_single_token(&mut ctx, next_id as u32)?;

            if stream_output {
                eprint!("\r[decode] {step}/{max_tokens}");
                // 流式输出当前生成的 token 文本
                if let Some(s) = self.tokenizer.vocab.tokens.get(next_id as usize) {
                    eprint!(" -> {s}");
                }
            }
        }
        if stream_output {
            eprintln!();
        }
        let decode_ms = decode_start.elapsed().as_millis();
        let n_gen = generated_ids.len();
        if n_gen > 0 {
            eprintln!("[bench] decode({n_gen}t)={decode_ms}ms (~{}ms/tok ~{:.2} tok/s)",
                decode_ms / n_gen as u128,
                n_gen as f64 * 1000.0 / decode_ms as f64);
        }

        // 正确性验证: 把生成的 token IDs dump 到文件 (env DAIZA_DUMP_TOKENS=path)
        // 用于 baseline vs optimized 的逐 token 对比 (FP 重排可能让 argmax 翻转)
        if let Ok(path) = std::env::var("DAIZA_DUMP_TOKENS") {
            let mut content = String::new();
            // 第一行: prompt token IDs
            content.push_str("prompt:");
            for (i, &id) in input_ids.iter().enumerate() {
                if i > 0 { content.push(','); }
                content.push_str(&id.to_string());
            }
            content.push('\n');
            // 第二行: generated token IDs
            content.push_str("generated:");
            for (i, &id) in generated_ids.iter().enumerate() {
                if i > 0 { content.push(','); }
                content.push_str(&id.to_string());
            }
            content.push('\n');
            std::fs::write(&path, content)
                .map_err(|e| crate::BonsaiError::Io(format!("dump_tokens write failed: {e}")))?;
            eprintln!("[dump_tokens] wrote {n_gen} generated ids to {path}");
        }

        // 5. decode token ids 为字符串
        Ok(self.tokenizer.decode(&generated_ids))
    }

    /// DSpark 推测解码生成循环
    ///
    /// 流程: prefill → 循环 { draft k 个 → batched verify (forward_batch 一次) →
    ///   逐 token Leviathan check → KV truncate + bonus forward }
    ///
    /// **Batched verify**: 一次 forward_batch(k) 读完 13GB 权重 (batch4 AVX2 kernel),
    /// 逐 token 用 per_pos_logits 做 Leviathan rejection sampling。reject 时 KV cache
    /// truncate 到 pos_before + n_accepted (attention 层正确), SSM state 保持
    /// pos_before + k (no rollback, Gated DeltaNet gate 衰减 rejected tokens)。
    pub fn generate_with_dspark(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        params: SamplingParams,
        system_prompt: Option<&str>,
        confidence_threshold: f32,
    ) -> Result<String> {
        if self.spec_ctx.is_none() {
            return Err(crate::BonsaiError::Unsupported(
                "DSpark not loaded; call load_drafter() first".into(),
            ));
        }

        // 1. 构造输入
        // ★ DAIZA_DSPARK_RAW=1: 用 raw prompt (不走 chat template), 对齐 llama.cpp test-dspark-real-eval
        let chat_text = if std::env::var("DAIZA_DSPARK_RAW").is_ok() {
            prompt.to_string()
        } else {
            build_chat_input(prompt, system_prompt)
        };
        eprintln!("[debug] input text: {chat_text:?}");
        let input_ids = self.tokenizer.encode(&chat_text);
        eprintln!("[debug] input_ids count: {}", input_ids.len());
        if input_ids.is_empty() {
            return Err(crate::BonsaiError::Tokenizer("encode returned empty".into()));
        }

        // 2. 加载 target 权重 + 初始化线程池
        if self.weights.is_none() {
            eprintln!("[engine] loading target weights...");
            self.load_weights()?;
            let n_threads = crate::model::workspace::thread_count();
            crate::model::workspace::init_thread_pool(n_threads);
            eprintln!("[engine] thread pool ({n_threads} workers) initialized");
        }

        // 3. 构造前向上下文 + 启用 hidden tap
        let cfg = &self.config;
        let weights = self.weights.as_ref().unwrap();
        let mut ctx = make_context(weights, cfg);
        // 启用 hidden tap: 从 spec_ctx 读 target_layers
        {
            let spec = self.spec_ctx.as_ref().unwrap();
            ctx.hidden_tap_layers = spec.cfg().target_layers.clone();
            eprintln!("[dspark] target tap layers: {:?}", ctx.hidden_tap_layers);
        }

        // 4. prefill (对齐 llama.cpp speculative-simple.cpp L210-216:
        //    forward N-1 tokens, 最后一个 token 作为 anchor/id_last, 不 forward)
        //    Daiza 之前是 forward N tokens + sample generated anchor + forward anchor,
        //    与 llama.cpp 语义不一致, 导致 context 多 1 行 + start_pos 偏移 1, 接受率降。
        let n_input = input_ids.len();
        let prefill_start = std::time::Instant::now();
        if n_input >= 2 {
            // prefill 前 N-1 tokens (去掉最后一个作为 anchor)
            let prefill_ids = &input_ids[..n_input - 1];
            forward_batch(&mut ctx, prefill_ids, 0, None)?;
        }
        // n_input == 1: 无 prefill, anchor = input_ids[0], 稍后 forward
        let prefill_ms = prefill_start.elapsed().as_millis();
        let n_prefill = if n_input >= 2 { n_input - 1 } else { 0 };
        eprintln!("\r[prefill] {n_prefill}/{n_input} done in {prefill_ms}ms");

        // 5. DSpark decode 循环
        let mut rng = LcgRng::new(0xC0FFEE);
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut sampling_buf = SamplingBuffers::new(ctx.logits_buf.len());

        // 累积 target tap history: 每个已 forward token 一行 [n_embd_cap]
        // draft 时传整个 history 作为 drafter context (对齐 llama.cpp ctx_feat 累积语义)
        let n_tap_layers = ctx.hidden_tap_layers.len();
        let hidden = cfg.hidden;
        let n_embd_cap = n_tap_layers * hidden;
        let mut target_tap_history: Vec<f32> = Vec::new();

        // prefill 阶段: forward_batch 已捕获 hidden_tap_batch_buf [n_batch, n_tap, hidden]
        // 累积到 history (行优先 token-major, 与 set_target_tap 期望一致)
        if n_input >= 2 {
            target_tap_history.extend_from_slice(&ctx.hidden_tap_batch_buf);
        }

        // anchor = 最后一个 prefill token (对齐 llama.cpp: id_last = inp.back())
        // 不 sample, 不加入 generated_ids (anchor 不是生成的 token)
        let mut anchor_token = input_ids[n_input - 1];
        // forward anchor 以获取其 hidden tap + 更新 cache + 产生 logits for draft[0]
        forward_single_token(&mut ctx, anchor_token)?;
        // 累积 anchor 的 hidden tap
        target_tap_history.extend_from_slice(&ctx.hidden_tap_buf);

        let decode_start = std::time::Instant::now();
        let stream_output = !matches!(std::env::var("DAIZA_STREAM").as_deref(),
            Ok("0") | Ok("false") | Ok("no"));

        let block_size = self.spec_ctx.as_ref().unwrap().cfg().block_size;
        let mut total_draft_calls = 0usize;
        let mut total_accepted = 0usize;
        let mut total_bonus = 0usize;
        let mut total_draft_truncated = 0usize;  // confidence head 截断的 token 数
        // ★ 性能分析: 各阶段累计耗时 (DAIZA_PROFILE 控制)
        let profile_dspark = std::env::var("DAIZA_PROFILE").is_ok();
        let mut t_draft = 0u128;       // Phase 1: drafter forward
        let mut t_verify = 0u128;      // Phase 2: sequential verify forwards
        let mut t_bonus = 0u128;       // Phase 3: bonus forward
        let mut n_target_forwards = 0usize;  // 总 target forward 次数

        // bonus 采样复用 buffer (p, q, residual, 各 vocab_size = ~1MB, 跨 cycle 复用)
        let mut bonus_buf = BonusBuffers::new();

        // ★ 自动降级 (DAIZA_DSPARK_FALLBACK, 默认开启; =0 关闭)
        // 探测期收集若干 cycle 的真实数据, 判定 DSpark 是否慢于原生 decode。
        // 判定依据: probe 窗口内 dspark ms/token 是否 > target 单 forward 实测耗时
        // (同进程同热状态对比, 免疫系统波动)。sequential verify 下 forwards/token 恒 = 1.0,
        // 故 DSpark 比原生慢 ⟺ draft overhead > 0 (恒成立)。
        let fallback_enabled = !matches!(
            std::env::var("DAIZA_DSPARK_FALLBACK").as_deref(),
            Ok("0") | Ok("false") | Ok("no")
        );
        const PROBE_CYCLES: usize = 6;
        let mut probe_done = !fallback_enabled;
        let mut probe_cycles = 0usize;
        let mut probe_gen_start = 0usize;
        let mut probe_time_start = 0u128;
        let mut fell_back = false;

        while generated_ids.len() < max_tokens {
            // --- Phase 1: Draft ---
            let draft_start = std::time::Instant::now();
            let draft_tokens: Vec<u32>;
            {
                let spec = self.spec_ctx.as_mut().unwrap();
                // 位置语义 (对齐 llama.cpp dspark speculative.cpp):
                //   context 行 = target hidden tap [L, ..., start-1] (不含 anchor)
                //   draft[0] = anchor token at position `start` (绝对位置)
                //   draft[k] = mask token at position start + k
                // 因此:
                //   ctx_len   = history_rows - 1 (排除最后 1 行 anchor 的 hidden tap)
                //   start_pos = anchor 的绝对位置 = ctx.state.pos - 1
                //     (anchor forward 后 ctx.state.pos 已递增到 anchor_pos + 1)
                // ★ 对齐 llama.cpp: drafter RoPE 用绝对位置 (L+row for context, start+k for draft)
                //   其中 L = start - ctx_len = 上一 cycle 的 start。Daiza history 滑动窗口后
                //   ctx_len 变小 (3-5),若 start_pos = ctx_len (相对位置) 会与 target 绝对位置
                //   严重偏移,导致 RoPE 频率错误 → 接受率下降。
                let history_rows = target_tap_history.len() / n_embd_cap;
            let ctx_len = history_rows - 1;
            let start_pos = ctx.state.pos - 1;
            spec.set_target_tap(
                &target_tap_history[..ctx_len * n_embd_cap],
                ctx_len,
            );
                let dt = spec.draft(anchor_token, start_pos).to_vec();
                draft_tokens = dt;
            }
            t_draft += draft_start.elapsed().as_millis();
            total_draft_calls += 1;

            // ★ Confidence head: 根据 confidence_logits 与 threshold 截断 draft tokens
            // 只 verify 前 n_draft_to_verify 个, 后面的 draft token 直接跳过 (省 target forward)
            // n_draft_to_verify == 0 → 跳过 Phase 2, 直接 bonus (与 all-reject 等价)
            // n_draft_to_verify == block_size → 不截断, 原行为
            let n_draft_to_verify = self.spec_ctx.as_ref().unwrap()
                .confident_prefix_length(confidence_threshold);
            total_draft_truncated += block_size - n_draft_to_verify;

            // --- Phase 2: Sequential Verify with early stop ---
            // 逐 token forward + Leviathan check, reject 时立即停止。
            //
            // target_logits 语义: ctx.logits_buf 始终持有 "预测下一个 token" 的 target 分布。
            //   - 进入 Phase 2 前, ctx.logits_buf = 上一 cycle bonus forward 的输出 = 预测 draft[0]
            //   - 接受 draft[i] 后 forward_single_token(draft[i]) → ctx.logits_buf = 预测 draft[i+1]
            //   - reject draft[i] 时, ctx.logits_buf = 预测 draft[i] (直接用于 bonus 采样)
            //
            // SSM state 处理 (no rollback):
            // - 只 forward 接受的 draft token, SSM state 推进到 pos_before + n_accepted (正确)
            // - 不需要 KV truncate (只 forward 了接受的 token, KV cache 自然正确)
            // - 不需要 SSM undo (未 forward 的 draft token 不影响 state)
            //
            // ★ Batched verify (forward_batch 一次处理 k drafts) 已尝试并回退:
            //   - 方案 A (SSM state 不 rollback): Gated DeltaNet gate 不能有效衰减 rejected tokens,
            //     conv_history 滑窗含 rejected drafts 的 qkv, 导致 SSM 输出错乱 (输出重复 "the user's
            //     perspective..."), 接受率下降, 性能退化 +91% (414ms/tok vs 217ms)
            //   - forward_batch 为 prefill 设计, 小批量 (k=4) 时开销超过 batched matvec 收益
            //   - 方案 B (SSM state snapshot/restore) 可行但复杂度高, 收益不确定, 暂不实施
            let verify_start = std::time::Instant::now();
            let pos_before = ctx.state.pos;

            let mut n_accepted = 0usize;
            let mut bonus_token: Option<usize> = None;
            for (i, &dt) in draft_tokens[..n_draft_to_verify].iter().enumerate() {
                // ctx.logits_buf = 预测 draft[i] 的 target 分布 (前一个 forward 的输出)
                let target_logits_i = &ctx.logits_buf[..cfg.vocab_size];
                let draft_logits_i = &self.spec_ctx.as_ref().unwrap().draft_logits
                    [i * cfg.vocab_size..(i + 1) * cfg.vocab_size];

                let accepted = leviathan_check(
                    target_logits_i, draft_logits_i, dt, params, &mut || rng.next_f32(),
                );

                if !accepted {
                    // Reject: 用 ctx.logits_buf 采样 bonus (预测 draft[i] 的分布)
                    bonus_token = Some(sample_bonus(
                        target_logits_i, draft_logits_i, params,
                        &mut || rng.next_f32(), &mut sampling_buf, &mut bonus_buf,
                    ));
                    break;
                }

                // Accept: forward draft[i] (更新 KV/SSM, 产生 draft[i+1] 的 logits)
                n_accepted += 1;
                generated_ids.push(dt);
                if dt == self.config.eos_token_id {
                    eprintln!("[dspark] EOS accepted at draft pos {i}");
                    return Ok(self.tokenizer.decode(&generated_ids));
                }
                forward_single_token(&mut ctx, dt)?;
                n_target_forwards += 1;
                target_tap_history.extend_from_slice(&ctx.hidden_tap_buf);

                if stream_output {
                    if let Some(s) = self.tokenizer.vocab.tokens.get(dt as usize) {
                        eprint!("\r[dspark] accepted draft[{i}] -> {s}    ");
                    }
                }
            }
            t_verify += verify_start.elapsed().as_millis();
            // 注: sequential verify 只 forward 接受的 token,
            // ctx.state.pos = pos_before + n_accepted (forward_single_token 自然推进)
            // KV cache 也只含接受的 token, 无需 truncate

            // --- Phase 3: Bonus forward ---
            // reject: bonus = sample_bonus(target_logits_i, draft_logits_i) (已采样)
            // all-accept: bonus = sample from ctx.logits_buf (最后接受 token 的 forward 输出)
            let bonus_start = std::time::Instant::now();
            let bt_raw = if let Some(bt) = bonus_token {
                bt
            } else {
                // All-accept: 从 ctx.logits_buf 采样 (最后一个 forward 的输出, 预测 pos_before+k)
                sample_top_k_top_p_into(
                    &ctx.logits_buf, params,
                    &mut || rng.next_f32(), &mut sampling_buf,
                )
            };
            let bt = bt_raw as u32;
            if bt == self.config.eos_token_id {
                eprintln!("[dspark] EOS from bonus");
                return Ok(self.tokenizer.decode(&generated_ids));
            }
            generated_ids.push(bt);
            anchor_token = bt;
            forward_single_token(&mut ctx, bt)?;
            n_target_forwards += 1;
            target_tap_history.extend_from_slice(&ctx.hidden_tap_buf);
            total_bonus += 1;
            if stream_output {
                if let Some(s) = self.tokenizer.vocab.tokens.get(bt as usize) {
                    eprint!("\r[dspark] bonus -> {s}    ");
                }
            }
            t_bonus += bonus_start.elapsed().as_millis();

            total_accepted += n_accepted;

            // ★ 滑动窗口 drain 已禁用: 实测 drain (keep n_accepted+2) 接受率 71.2%
            //   反而比不 drain 77.1% 更差 (drain 后某些 cycle draft 全被拒, rounds 反而更多)。
            //   Daiza 的累积 history 语义与 llama.cpp ctx_feat 滑动窗口不等价,
            //   需要 drain + 绝对位置 + 完整 start_pos 对齐才能匹配, 当前不 drain 更优。
            //   详见 project_memory lessons learned。

            // ★ 自动降级判定: 探测窗口满后, 比较 DSpark 实测 ms/token vs 原生 decode 下界
            if !probe_done {
                if probe_cycles == 0 {
                    probe_gen_start = generated_ids.len();
                    probe_time_start = decode_start.elapsed().as_millis();
                }
                probe_cycles += 1;
                if probe_cycles >= PROBE_CYCLES {
                    let win_gen = generated_ids.len() - probe_gen_start;
                    let win_ms = decode_start.elapsed().as_millis() - probe_time_start;
                    let dspark_ms_per_tok = if win_gen > 0 { win_ms as f64 / win_gen as f64 } else { f64::MAX };
                    let native_ms_per_tok = if n_target_forwards > 0 {
                        (t_verify + t_bonus) as f64 / n_target_forwards as f64
                    } else { f64::MAX };
                    eprintln!("[dspark-fallback] probe: dspark={dspark_ms_per_tok:.1}ms/tok \
                        vs native~{native_ms_per_tok:.1}ms/tok (window {win_gen}t/{win_ms}ms)");
                    if dspark_ms_per_tok > native_ms_per_tok {
                        eprintln!("[dspark-fallback] DSpark slower than native decode → \
                            falling back to greedy single-token decode for remaining tokens");
                        fell_back = true;
                        break;
                    }
                    probe_done = true;
                }
            }

            if generated_ids.len() >= max_tokens {
                break;
            }
        }

        // ★ 降级路径: 从当前 ctx 状态 (KV/SSM 已推进, logits_buf 持有下一 token 分布)
        //   无缝切换为原生 greedy decode, 继续生成剩余 token。零重复计算。
        if fell_back {
            while generated_ids.len() < max_tokens {
                let next_id = sample_top_k_top_p_into(
                    &ctx.logits_buf, params, &mut || rng.next_f32(), &mut sampling_buf,
                );
                if next_id as u32 == self.config.eos_token_id {
                    break;
                }
                generated_ids.push(next_id as u32);
                forward_single_token(&mut ctx, next_id as u32)?;
                if stream_output {
                    if let Some(s) = self.tokenizer.vocab.tokens.get(next_id as usize) {
                        eprint!("\r[dspark→native] -> {s}    ");
                    }
                }
            }
        }
        if stream_output {
            eprintln!();
        }
        let decode_ms = decode_start.elapsed().as_millis();
        let n_gen = generated_ids.len();
        eprintln!("[bench] dspark decode({n_gen}t)={decode_ms}ms (~{}ms/tok ~{:.2} tok/s)",
            if n_gen > 0 { decode_ms / n_gen as u128 } else { 0 },
            if decode_ms > 0 { n_gen as f64 * 1000.0 / decode_ms as f64 } else { 0.0 });
        eprintln!("[dspark] draft_calls={total_draft_calls} accepted={total_accepted} \
            (avg {:.2}/{block_size}) bonus={total_bonus} truncated={total_draft_truncated} \
            (conf_threshold={confidence_threshold})",
            if total_draft_calls > 0 { total_accepted as f64 / total_draft_calls as f64 }
            else { 0.0 });

        // ★ 性能分析: 各阶段耗时分解 (DAIZA_PROFILE)
        if profile_dspark {
            let total_phase = t_draft + t_verify + t_bonus;
            let n_verify_forwards = n_target_forwards - total_bonus;
            eprintln!("───────── [dspark-profile] phase breakdown ─────────");
            eprintln!("  draft    (drafter fwd): {:>6}ms  ({:>5.1}%)  — {} calls, ~{}ms/call",
                t_draft, pct(t_draft, total_phase),
                total_draft_calls, ms_per(t_draft, total_draft_calls));
            eprintln!("  verify   (sequential):  {:>6}ms  ({:>5.1}%)  — {} forwards (avg {:.2}/cycle)",
                t_verify, pct(t_verify, total_phase),
                n_verify_forwards,
                if total_draft_calls > 0 { n_verify_forwards as f64 / total_draft_calls as f64 }
                else { 0.0 });
            eprintln!("  bonus    (bonus fwd):   {:>6}ms  ({:>5.1}%)  — {} forwards",
                t_bonus, pct(t_bonus, total_phase), total_bonus);
            eprintln!("  ────────────────────────────────────────────────");
            eprintln!("  total phase:            {:>6}ms", total_phase);
            eprintln!("  total decode wall:      {:>6}ms  (diff = overhead/sched)", decode_ms);
            eprintln!("  target forwards:        {} total = {} verify + {} bonus",
                n_target_forwards, n_verify_forwards, total_bonus);
            eprintln!("  per-token cost:  ~{}ms target + ~{}ms draft overhead",
                ms_per(t_verify + t_bonus, n_target_forwards),
                ms_per(t_draft, n_gen));
        }

        if let Ok(path) = std::env::var("DAIZA_DUMP_TOKENS") {
            let mut content = String::new();
            content.push_str("prompt:");
            for (i, &id) in input_ids.iter().enumerate() {
                if i > 0 { content.push(','); }
                content.push_str(&id.to_string());
            }
            content.push('\n');
            content.push_str("generated:");
            for (i, &id) in generated_ids.iter().enumerate() {
                if i > 0 { content.push(','); }
                content.push_str(&id.to_string());
            }
            content.push('\n');
            std::fs::write(&path, content)
                .map_err(|e| crate::BonsaiError::Io(format!("dump_tokens: {e}")))?;
            eprintln!("[dump_tokens] wrote {n_gen} ids to {path}");
        }

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
        crate::model::weights::LoadedWeights::print_dtype_summary(&self.gguf);
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

/// 构造带图像占位符的 chat 输入
///
/// `image_section`: N 个 image_token 字符串 (如 "<|image_pad|><|image_pad|>")
/// `user_prompt`: 用户文本 prompt
/// `system_prompt`: 可选系统 prompt
///
/// 输出: `<|im_start|>user\n{image_section}{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n`
fn build_chat_input_with_image(
    image_section: &str,
    user_prompt: &str,
    system_prompt: Option<&str>,
) -> String {
    let mut s = String::new();
    if let Some(sys) = system_prompt {
        s.push_str("<|im_start|>system\n");
        s.push_str(sys);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>user\n");
    s.push_str(image_section);
    s.push_str(user_prompt);
    s.push_str("<|im_end|>\n");
    s.push_str("<|im_start|>assistant\n");
    s.push_str("<think>\n");
    s
}

/// Leviathan rejection sampling: 单 token 接受/拒绝判定
///
/// - `target_logits`: target 模型在该位置的 logits [vocab]
/// - `draft_logits`: drafter (markov resample 后) 的 logits [vocab]
/// - `draft_token`: drafter 采样的 token ID
/// - `rng`: 均匀随机数 [0, 1)
///
/// 返回 true = 接受 draft_token, false = 拒绝
///
/// **greedy 模式** (params.temperature == 0): 直接比较 argmax,
///   draft_token == target_argmax 则接受, 否则拒绝。
///   与 llama.cpp dspark (target verify temp=0) 行为一致。
///   Q1_0 量化下 target logits 峰值被压低, softmax(p[dt]/q[dt]) 随机接受
///   会错误拒绝正确 draft (Daiza 41.5% vs llama.cpp 95.83% 接受率的根因)。
///
/// **采样模式** (params.temperature > 0): 标准 Leviathan 随机接受
///   `u < p[dt] / q[dt]`, 接受后 bonus 从 residual=max(0, p-q) 采样。
///
/// ★ 优化: 只计算 p[dt] 和 q[dt], 避免全量 softmax 分配 [vocab] 两次 (~2MB/call)
fn leviathan_check(
    target_logits: &[f32],
    draft_logits: &[f32],
    draft_token: u32,
    params: SamplingParams,
    rng: &mut dyn FnMut() -> f32,
) -> bool {
    debug_assert_eq!(draft_logits.len(), target_logits.len());
    let dt = draft_token as usize;

    // greedy 模式: argmax 比较 (与 llama.cpp dspark target verify temp=0 一致)
    if params.temperature <= 0.0 {
        let mut target_argmax = 0usize;
        let mut target_max = f32::NEG_INFINITY;
        for (i, &l) in target_logits.iter().enumerate() {
            if l > target_max {
                target_max = l;
                target_argmax = i;
            }
        }
        return target_argmax == dt;
    }

    // 采样模式: 标准 Leviathan 随机接受 u < p[dt]/q[dt]
    let (p_dt, _) = softmax_single(target_logits, dt);
    let (q_dt, _) = softmax_single(draft_logits, dt);

    let r = if q_dt > 1e-12 { p_dt / q_dt } else { 0.0 };
    let u = rng();
    u < r
}

/// 从 residual 分布 norm(max(0, p - q)) 采样 bonus token
/// p = target prob, q = draft prob (已归一化)
///
/// ★ 优化: 复用预分配 buffer (p, q, residual), 避免每 cycle ~4MB 分配
fn sample_bonus(
    target_logits: &[f32],
    draft_logits: &[f32],
    params: SamplingParams,
    rng: &mut impl FnMut() -> f32,
    sampling_buf: &mut SamplingBuffers,
    bonus_buf: &mut BonusBuffers,
) -> usize {
    let vocab = target_logits.len();
    if bonus_buf.p.len() != vocab {
        bonus_buf.p = vec![0.0; vocab];
        bonus_buf.q = vec![0.0; vocab];
        bonus_buf.residual = vec![0.0; vocab];
    }
    let p = &mut bonus_buf.p;
    let q = &mut bonus_buf.q;
    softmax_inplace(target_logits, p);
    softmax_inplace(draft_logits, q);

    // residual = max(0, p - q), 归一化为概率分布, 转为 logits 供 top_k/top_p 采样
    let residual = &mut bonus_buf.residual;
    let mut sum = 0.0f32;
    for i in 0..vocab {
        let r = (p[i] - q[i]).max(0.0);
        residual[i] = r;
        sum += r;
    }
    if sum <= 1e-12 {
        // 退化为 target 分布采样
        return sample_top_k_top_p_into(target_logits, params, rng, sampling_buf);
    }
    // 转为 logits (log of residual prob) 供 top_k_top_p 采样
    let inv = 1.0 / sum;
    for i in 0..vocab {
        residual[i] = (residual[i] * inv).ln();
    }
    sample_top_k_top_p_into(residual, params, rng, sampling_buf)
}

/// 数值稳定的 softmax: logits → prob (写入 out)
fn softmax_inplace(logits: &[f32], out: &mut [f32]) {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max).exp();
        out[i] = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in out.iter_mut() {
        *v *= inv;
    }
}

/// 只计算 softmax 在 `idx` 位置的概率值, 避免全量分配
/// 返回 (prob[idx], sum)
fn softmax_single(logits: &[f32], idx: usize) -> (f32, f32) {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    let mut idx_exp = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max).exp();
        sum += e;
        if i == idx { idx_exp = e; }
    }
    (idx_exp / sum, sum)
}

/// DSpark bonus 采样复用 buffer (跨 cycle 复用, 避免每 cycle ~4MB 分配)
struct BonusBuffers {
    p: Vec<f32>,
    q: Vec<f32>,
    residual: Vec<f32>,
}

impl BonusBuffers {
    fn new() -> Self {
        Self { p: Vec::new(), q: Vec::new(), residual: Vec::new() }
    }
}

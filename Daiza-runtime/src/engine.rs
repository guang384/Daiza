//! 顶层推理引擎入口
//!
//! 把 GGUF 解析 → 配置 → 权重加载 → 前向传播 → 采样 → 解码 串起来。
//!
//! ## 使用方式
//!
//! ```no_run
//! use daiza_runtime::engine::Engine;
//! let mut engine = Engine::load(std::path::Path::new("../Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf")).unwrap();
//! let out = engine.generate("你好", 64).unwrap();
//! println!("{out}");
//! ```

use std::path::Path;

use daiza_engine::gguf::parser::GgufFile;
use daiza_engine::math::{sample_top_k_top_p_into, apply_repetition_penalty, LcgRng, SamplingBuffers, SamplingParams};
use daiza_engine::model::config::Config;
use daiza_engine::model::dspark::{
    weights::DrafterWeights,
    speculative::SpeculativeContext,
    ngram::NgramDrafter,
};
use daiza_engine::model::forward::{forward_batch, forward_batch_with_vision, forward_single_token, make_context, VisionInject, ForwardContext};
use daiza_engine::model::vision::{
    VisionConfig, VisionWeights,
    encoder::{ViTContext, encode_image},
    projector::{ProjectorContext, project_vision},
    preprocess_image,
};
use daiza_engine::model::weights::LoadedWeights;
use crate::session::{Session, session_reply};
use crate::tokenizer::vocab::Vocab;
use crate::tokenizer::BpeTokenizer;
use crate::Result;

/// DSpark 性能分析 helper
#[inline]
fn pct(part: u128, total: u128) -> f64 {
    if total == 0 { 0.0 } else { part as f64 * 100.0 / total as f64 }
}
#[inline]
fn ms_per(total_ms: u128, n: usize) -> u128 {
    if n == 0 { 0 } else { total_ms / n as u128 }
}

/// DSpark 流式事件: 用于前端实现"乐观显示 draft + verify 后修正"的 UX
///
/// - Draft: drafter 预测的文本 (前端灰色乐观显示)
/// - Accept: draft 全部通过 verify (前端保留灰色文本, 可选转黑)
/// - Reject: draft 部分被拒, text 为通过 verify 的 accepted 部分
///           (前端删除上一个 Draft 的全部文本, 用 text 替换, 黑色)
/// - Delta: 正常增量 (bonus token / fallback greedy, 黑色)
pub enum DsparkEvent {
    /// drafter 预测的文本 (乐观显示)
    Draft(String),
    /// draft 全部通过 verify
    Accept,
    /// draft 部分被拒, text = accepted 部分 (前端删除 draft, 用 text 替换)
    Reject(String),
    /// 正常增量 (bonus / fallback greedy)
    Delta(String),
}

/// 推理进度事件: 用于前端显示 prefill / vision 处理进度条
///
/// - Vision: 图像处理阶段 (preprocess / encode / project), 进度按 image_idx / total
/// - Prefill: 文本+图像 token prefill 阶段, 进度按 done_tokens / total_tokens
pub enum ProgressEvent {
    /// 图像处理进度: (image_idx 0-based, total_images, stage_name, elapsed_ms)
    Vision { image_idx: usize, total: usize, stage: &'static str, ms: u128 },
    /// Prefill 进度: (已处理 token 数, 总 token 数)
    /// n_total 包含 text tokens + vision tokens (每张图展开为 n_vision_per_image 个 token)
    Prefill { done_tokens: usize, total_tokens: usize },
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
    /// 活跃会话 (可选, 由 session_begin 创建, 跨多轮 reply 复用 KV + SSM state)
    pub session: Option<Session>,
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
            session: None,
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
        // ★ 验证 Qwen3-VL chat template 必需的 vision 边界 token 在 tokenizer vocab 中存在
        //   缺失会导致 <|vision_start|>/<|vision_end|> 被当普通文本字节编码, 模型无法识别图像
        self.verify_vision_boundary_tokens()?;
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

    /// 验证 Qwen3-VL vision 边界 token (<|vision_start|>/<|vision_end|>) 在 vocab 中存在
    ///
    /// ★ 必需: chat template 要求 image_pad 用 <|vision_start|>...<|vision_end|> 包裹,
    ///   若这些 token 不在 special_tokens 中, BPE 会把它们当普通文本字节编码,
    ///   产生大量乱码 token, 模型完全无法识别图像边界。
    pub fn verify_vision_boundary_tokens(&self) -> Result<()> {
        for name in ["<|vision_start|>", "<|vision_end|>"] {
            if !self.tokenizer.special_tokens.contains_key(name) {
                return Err(crate::BonsaiError::Unsupported(format!(
                    "vision boundary token {name} not found in tokenizer special_tokens; \
                     Qwen3-VL chat template requires <|vision_start|>/<|vision_end|> to wrap image_pad"
                )));
            }
        }
        Ok(())
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

    /// 开始新会话(创建 Session, state 全零)
    ///
    /// `think_enabled`:是否启用思考模式(控制 increment 末尾是否追加 <think>\n)
    /// `system_prompt`:可选系统提示词(首轮 prefill 时使用)
    pub fn session_begin(
        &mut self,
        think_enabled: bool,
        system_prompt: Option<&str>,
    ) -> Result<()> {
        // 确保权重已加载
        if self.weights.is_none() {
            self.load_weights()?;
            let n_threads = daiza_engine::model::workspace::thread_count();
            daiza_engine::model::workspace::init_thread_pool(n_threads);
            eprintln!("[engine] thread pool ({n_threads} workers) initialized");
        }
        // ★ 重置 DSpark spec_ctx cache: 新 session 的 dspark_tap_history 为空,
        //   spec_ctx 的 target_tap_len / cached_ctx_len / cached_kv_len 也需重置,
        //   否则跨 session 复用旧 cache 会导致 drafter context 不一致
        if let Some(spec) = self.spec_ctx.as_mut() {
            spec.target_tap_len = 0;
            spec.drafter.cached_ctx_len = 0;
            spec.drafter.cached_kv_len = 0;
        }
        let sys = system_prompt.map(String::from);
        self.session = Some(Session::new(&self.config, think_enabled, sys));
        Ok(())
    }

    /// 多轮对话:增量 prefill + decode(复用 KV cache + SSM state)
    ///
    /// 只编码增量 token(不含历史),从 state.pos 继续 prefill。
    /// 无损复用:KV cache 和 SSM state 跨调用持久化。
    pub fn session_reply(
        &mut self,
        user_msg: &str,
        max_tokens: usize,
        params: SamplingParams,
    ) -> Result<String> {
        let session = self.session.as_mut().ok_or(crate::BonsaiError::Unsupported(
            "no active session; call session_begin() first".into()
        ))?;
        let cfg = &self.config;
        let weights = self.weights.as_ref().ok_or(crate::BonsaiError::Unsupported(
            "weights not loaded".into()
        ))?;
        session_reply(cfg, weights, &self.tokenizer, session, user_msg, max_tokens, params)
    }

    /// 结束会话(释放 DRAM)
    pub fn session_end(&mut self) {
        self.session = None;
    }

    /// 流式多轮对话: 与 session_reply 行为一致, 但每生成一个 token
    /// 就通过 `on_delta` 回调增量文本 (供 GUI / Web 流式渲染)。
    /// 回调返回 false 可中断生成。
    pub fn session_reply_stream(
        &mut self,
        user_msg: &str,
        max_tokens: usize,
        params: SamplingParams,
        on_delta: &mut dyn FnMut(&str) -> bool,
    ) -> Result<String> {
        let session = self.session.as_mut().ok_or(crate::BonsaiError::Unsupported(
            "no active session; call session_begin() first".into()
        ))?;
        let cfg = &self.config;
        let weights = self.weights.as_ref().ok_or(crate::BonsaiError::Unsupported(
            "weights not loaded".into()
        ))?;
        crate::session::session_reply_stream(
            cfg, weights, &self.tokenizer, session, user_msg, max_tokens, params, on_delta,
        )
    }

    /// tool_response 回传后的流式生成
    ///
    /// 前端执行工具后, 把结果通过本函数送回模型, 模型继续生成下一轮回复。
    /// 构造 tool_response 格式的增量 prompt, 复用 session_reply_stream 的核心逻辑。
    pub fn session_reply_tool_response_stream(
        &mut self,
        tool_content: &str,
        max_tokens: usize,
        params: SamplingParams,
        on_delta: &mut dyn FnMut(&str) -> bool,
    ) -> Result<String> {
        let session = self.session.as_mut().ok_or(crate::BonsaiError::Unsupported(
            "no active session; call session_begin() first".into()
        ))?;
        let cfg = &self.config;
        let weights = self.weights.as_ref().ok_or(crate::BonsaiError::Unsupported(
            "weights not loaded".into()
        ))?;
        crate::session::session_reply_tool_response_stream(
            cfg, weights, &self.tokenizer, session, tool_content, max_tokens, params, on_delta,
        )
    }

    /// 多轮对话 + 多模态: 带 vision 注入的 session_reply
    ///
    /// 与 session_reply 区别: 若 session.pending_images 非空, 对每张图做
    /// preprocess → encode → project, 然后走 forward_batch_with_vision 路径。
    /// 图片处理完后清空 pending_images。
    ///
    /// 若 pending_images 为空, 退化为普通 session_reply (无额外开销)。
    pub fn session_reply_with_vision(
        &mut self,
        user_msg: &str,
        max_tokens: usize,
        params: SamplingParams,
    ) -> Result<String> {
        // 无 pending_images: 退化为普通 session_reply
        let has_images = self.session.as_ref()
            .map(|s| !s.pending_images.is_empty())
            .unwrap_or(false);
        if !has_images {
            return self.session_reply(user_msg, max_tokens, params);
        }

        // 检查 mmproj 已加载
        if self.vision.is_none() {
            return Err(crate::BonsaiError::Unsupported(
                "mmproj not loaded; call load_mmproj() first".into()
            ));
        }

        if self.weights.is_none() {
            return Err(crate::BonsaiError::Unsupported(
                "weights not loaded".into()
            ));
        }

        // 1. 对每张图做 preprocess → encode → project, 拼接 vision_embeddings
        let image_token_id = self.image_token_id()?;
        // 先把 pending_images 从 session 中取出, 避免后续借用冲突
        let images: Vec<_> = {
            let session = self.session.as_mut().ok_or(crate::BonsaiError::Unsupported(
                "no active session; call session_begin() first".into()
            ))?;
            session.pending_images.drain(..).collect()
        };
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
            let mut all_emb = Vec::with_capacity(images.len() * n_vision_per_image * hidden);
            for img_path in &images {
                let t0 = std::time::Instant::now();
                let patches = preprocess_image(img_path, &vision.cfg)?;
                let t_pre = t0.elapsed();

                let t1 = std::time::Instant::now();
                encode_image(&vision.weights, &vision.cfg, &mut vision.vit_ctx, &patches)?;
                let t_enc = t1.elapsed();

                let t2 = std::time::Instant::now();
                // ★ V-5: 消除 10.6MB clone (disjoint field borrow: vit_ctx.hidden + proj_ctx 互不重叠)
                project_vision(&vision.vit_ctx.hidden, &vision.weights, &vision.cfg, &mut vision.proj_ctx)?;
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

        // 2. 调用 session_reply_with_vision (在 session.rs 中实现)
        // 借出顺序: weights → session → tokenizer (无冲突)
        let weights = self.weights.as_ref().unwrap();
        let session = self.session.as_mut().unwrap();
        crate::session::session_reply_with_vision(
            &self.config, weights, &self.tokenizer, session,
            user_msg, max_tokens, params,
            &vision_embeddings, n_vision_per_image, image_token_id,
        )
    }

    /// 流式多模态对话: 与 session_reply_with_vision 行为一致, 但
    /// 1. 每生成一个 token 通过 `on_delta` 回调增量文本 (供 GUI / Web 流式渲染)
    /// 2. vision 三阶段 (preprocess / encode / project) 和 prefill 阶段
    ///    通过 `on_progress` 回调上报进度 (供前端显示进度条)
    ///
    /// 回调返回 false 可中断生成 (仅 on_delta 生效; on_progress 返回值忽略)
    pub fn session_reply_with_vision_stream(
        &mut self,
        user_msg: &str,
        max_tokens: usize,
        params: SamplingParams,
        on_delta: &mut dyn FnMut(&str) -> bool,
        on_progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<String> {
        // 无 pending_images: 退化为普通 session_reply_stream
        // 文本路径 prefill 通常 < 200ms, 进度条无意义, 不上报
        let has_images = self.session.as_ref()
            .map(|s| !s.pending_images.is_empty())
            .unwrap_or(false);
        if !has_images {
            return self.session_reply_stream(user_msg, max_tokens, params, on_delta);
        }

        if self.vision.is_none() {
            return Err(crate::BonsaiError::Unsupported(
                "mmproj not loaded; call load_mmproj() first".into()
            ));
        }
        if self.weights.is_none() {
            return Err(crate::BonsaiError::Unsupported(
                "weights not loaded".into()
            ));
        }

        // 1. 对每张图做 preprocess → encode → project, 拼接 vision_embeddings + 上报进度
        let image_token_id = self.image_token_id()?;
        let images: Vec<_> = {
            let session = self.session.as_mut().ok_or(crate::BonsaiError::Unsupported(
                "no active session; call session_begin() first".into()
            ))?;
            session.pending_images.drain(..).collect()
        };
        let n_images = images.len();
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
            let mut all_emb = Vec::with_capacity(n_images * n_vision_per_image * hidden);
            for (img_idx, img_path) in images.iter().enumerate() {
                let t0 = std::time::Instant::now();
                let patches = preprocess_image(img_path, &vision.cfg)?;
                let t_pre = t0.elapsed();
                on_progress(ProgressEvent::Vision { image_idx: img_idx, total: n_images, stage: "preprocess", ms: t_pre.as_millis() });

                let t1 = std::time::Instant::now();
                encode_image(&vision.weights, &vision.cfg, &mut vision.vit_ctx, &patches)?;
                let t_enc = t1.elapsed();
                on_progress(ProgressEvent::Vision { image_idx: img_idx, total: n_images, stage: "encode", ms: t_enc.as_millis() });

                let t2 = std::time::Instant::now();
                project_vision(&vision.vit_ctx.hidden, &vision.weights, &vision.cfg, &mut vision.proj_ctx)?;
                let t_proj = t2.elapsed();
                on_progress(ProgressEvent::Vision { image_idx: img_idx, total: n_images, stage: "project", ms: t_proj.as_millis() });

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

        // 2. 调用 session_reply_with_vision_stream (session.rs 中实现)
        let weights = self.weights.as_ref().unwrap();
        let session = self.session.as_mut().unwrap();
        crate::session::session_reply_with_vision_stream(
            &self.config, weights, &self.tokenizer, session,
            user_msg, max_tokens, params,
            &vision_embeddings, n_vision_per_image, image_token_id,
            on_delta, on_progress,
        )
    }

    /// tool_call: 注入 tool response 后继续生成
    ///
    /// 调用方执行 tool 后, 用本函数把结果送回模型, 模型继续生成下一轮回复。
    pub fn session_reply_with_tool_response(
        &mut self,
        responses: &[crate::tool_call::ToolResponse],
        max_tokens: usize,
        params: SamplingParams,
    ) -> Result<String> {
        let session = self.session.as_mut().ok_or(crate::BonsaiError::Unsupported(
            "no active session; call session_begin() first".into()
        ))?;
        let cfg = &self.config;
        let weights = self.weights.as_ref().ok_or(crate::BonsaiError::Unsupported(
            "weights not loaded".into()
        ))?;
        crate::session::session_reply_with_tool_response(
            cfg, weights, &self.tokenizer, session, responses, max_tokens, params,
        )
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
            let n_threads = daiza_engine::model::workspace::thread_count();
            daiza_engine::model::workspace::init_thread_pool(n_threads);
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
                // ★ V-5: 消除 10.6MB clone (disjoint field borrow: vit_ctx.hidden + proj_ctx 互不重叠)
                project_vision(&vision.vit_ctx.hidden, &vision.weights, &vision.cfg, &mut vision.proj_ctx)?;
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
        // ★ Qwen3-VL chat template: 每张图必须用 <|vision_start|><|image_pad|><|vision_end|> 包裹
        let mut image_section = String::new();
        for _ in 0..image_paths.len() {
            image_section.push_str("<|vision_start|>");
            image_section.push_str(&image_token_str);
            image_section.push_str("<|vision_end|>");
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

        // 6. prefill: text tokens 用 forward_batch (batched), vision embeddings 一次性注入
        //    ★ 策略:
        //      - text tokens: 收集成 batch (≤32), 用 forward_batch 一次读 13GB 权重
        //      - image_token: 1 个占位, 展开为 n_vision_per_image 个 vision embeddings,
        //        一次性注入 (用 forward_batch_with_vision)
        //    ★ 一次性注入原因: vision M-RoPE 需要 n_vision_per_image 计算
        //      n_per_side_merged = sqrt(n), 分批模式下 n=1 会导致 sqrt(1)=1,
        //      所有 patch 的 (h,w) 退化为 (idx,0), 模型无法区分空间位置
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

        // 一次性注入整张图的 vision embeddings
        let flush_vision_batch = |ctx: &mut ForwardContext<'_>, emb: &[f32]| -> crate::Result<()> {
            let n = emb.len() / hidden;
            debug_assert_eq!(emb.len(), n * hidden);
            // 1 个 image_token 占位, 展开为 n 个 vision token
            let token_ids: Vec<u32> = vec![image_token_id];
            let inject = VisionInject {
                image_token_id,
                vision_embeddings: emb,
                n_vision_per_image: n,
            };
            forward_batch_with_vision(ctx, &token_ids, ctx.state.pos, None, Some(inject), None)
        };

        for &tid in &input_ids {
            if tid == image_token_id {
                // 先 flush 累积的 text batch
                flush_text_batch(&mut text_batch, &mut ctx)?;
                // 一次性注入整张图的 vision embeddings
                let emb_start = vision_offset * hidden;
                let emb_end = emb_start + n_vision_per_image * hidden;
                let emb = &vision_embeddings[emb_start..emb_end];
                flush_vision_batch(&mut ctx, emb)?;
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
            let n_threads = daiza_engine::model::workspace::thread_count();
            daiza_engine::model::workspace::init_thread_pool(n_threads);
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

        // 正确性验证: dump token IDs (env DAIZA_DUMP_TOKENS=path)
        // 用于 baseline vs optimized 的逐 token 对比 (FP 重排可能让 argmax 翻转)
        crate::session_persist::dump_tokens_if_enabled(&input_ids, &generated_ids)?;

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
        fallback_enabled: bool,
    ) -> Result<String> {
        // 无回调包装: 忽略所有流式事件, 最后一次性 decode (CLI 用)
        let mut noop = |_e: DsparkEvent| -> bool { true };
        self.generate_with_dspark_stream(
            prompt, max_tokens, params, system_prompt, confidence_threshold, fallback_enabled, &mut noop,
        )
    }

    /// DSpark 流式版本: 通过 on_event 回调发出 Draft/Accept/Reject/Delta 事件
    ///
    /// 前端可实现"乐观显示": drafter 预测的文本先灰色显示, verify 通过保留,
    /// verify 拒绝则用 accepted 部分替换。回调返回 false 可中断生成。
    pub fn generate_with_dspark_stream(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        params: SamplingParams,
        system_prompt: Option<&str>,
        confidence_threshold: f32,
        fallback_enabled: bool,
        on_event: &mut dyn FnMut(DsparkEvent) -> bool,
    ) -> Result<String> {
        if self.spec_ctx.is_none() {
            return Err(crate::BonsaiError::Unsupported(
                "DSpark not loaded; call load_drafter() first".into(),
            ));
        }

        // ★ PLD (Prompt Lookup Decoding) 模式: DAIZA_PLD 未设为 0/false/no 即开启。
        //   Phase 1 用 2-gram 查表 (~100ns) 替代神经 drafter forward (~53ms/call),
        //   draft overhead → 0 → DSpark 路径与原生 decode 持平 (不触发 probe fallback)。
        //   仍要求 load_drafter: hidden tap 层表来自 spec 配置, 且 tap history 累积
        //   保证 session 在 PLD/神经 DSpark 间切换时 drafter context 一致。
        //   注: sequential verify 下 forwards/token 恒 = 1.0, PLD 不加速 decode;
        //   价值 = 零成本保留 DSpark (Draft 乐观 UI) + 实测接受率 p 供 batched
        //   verify 方向决策。greedy 下输出与原生 decode 逐字节一致: draft 接受 ⟺
        //   argmax 匹配, 拒绝时 bonus = argmax, 均为原生下一个 token。
        let pld_mode = match std::env::var("DAIZA_PLD") {
            Ok(v) => !matches!(v.as_str(), "0" | "false" | "no"),
            Err(_) => false,
        };

        // 1. 构造输入
        // ★ DAIZA_DSPARK_RAW=1: 用 raw prompt (不走 chat template), 对齐 llama.cpp test-dspark-real-eval
        // ★ 若 prompt 已包含 <|im_start|> (Web 端 handle_chat_dspark 构造的完整 chat 格式,
        //   含/不含 <think>\n 由 session.think_enabled 控制), 直接使用, 不再 build_chat_input
        //   包装 (否则会无条件追加 <think>\n, 导致 thinking 开关失效)
        let chat_text = if std::env::var("DAIZA_DSPARK_RAW").is_ok() || prompt.contains("<|im_start|>") {
            prompt.to_string()
        } else {
            build_chat_input(prompt, system_prompt)
        };
        let input_ids = self.tokenizer.encode(&chat_text);
        if input_ids.is_empty() {
            return Err(crate::BonsaiError::Tokenizer("encode returned empty".into()));
        }

        // 2. 加载 target 权重 + 初始化线程池
        if self.weights.is_none() {
            eprintln!("[engine] loading target weights...");
            self.load_weights()?;
            let n_threads = daiza_engine::model::workspace::thread_count();
            daiza_engine::model::workspace::init_thread_pool(n_threads);
            eprintln!("[engine] thread pool ({n_threads} workers) initialized");
        }

        // 3. 构造前向上下文 + 启用 hidden tap
        // ★ 增量模式: 若 self.session 存在, 从 session 取 state/buffers/tap_history,
        //   增量 prefill (只编码新增 token, 复用 KV cache + SSM state + drafter context)
        //   跨轮 KV cache 复用: 第二轮 prefill 从 M tokens (新增) 而非 N tokens (全历史)
        // ★ 全量模式: 无 session (CLI 单次生成), 创建新 ctx, 全量 prefill (原行为)
        let cfg = &self.config;
        let weights = self.weights.as_ref().unwrap();
        let n_tap_layers;
        let mut start_pos;
        // session_take: 从 engine 取出的 session (增量模式), 函数末尾放回
        let mut session_take: Option<crate::session::Session> = None;
        // ★ DSpark 切换后 state.pos=0 但 history_tokens 非空: 需先 prefill 整个历史恢复 KV/SSM
        let mut replay_history: Vec<u32> = Vec::new();
        let mut ctx = if self.session.is_some() {
            // ★ 增量模式: 从 session 取状态
            let mut session = self.session.take().unwrap();
            start_pos = session.state.pos;
            // ★ 检测 DSpark 切换恢复: state.pos=0 但 history_tokens 非空
            if start_pos == 0 && !session.history_tokens.is_empty() {
                replay_history = session.history_tokens.clone();
            }
            n_tap_layers = if session.dspark_tap_layers.is_empty() {
                let spec = self.spec_ctx.as_ref().unwrap();
                let layers = spec.cfg().target_layers.clone();
                session.dspark_tap_layers = layers.clone();
                layers.len()
            } else {
                session.dspark_tap_layers.len()
            };
            let hidden_tap_layers = session.dspark_tap_layers.clone();
            eprintln!("[dspark] incremental mode: start_pos={start_pos}, tap_layers={n_tap_layers}");
            // ★ 不重置 spec_ctx cache (跨轮复用 target_tap_len / cached_ctx_len / cached_kv_len)
            session_take = Some(session);
            ForwardContext {
                cfg, weights,
                state: std::mem::take(&mut session_take.as_mut().unwrap().state),
                h_buf: std::mem::take(&mut session_take.as_mut().unwrap().h_buf),
                workspace: std::mem::take(&mut session_take.as_mut().unwrap().workspace),
                logits_buf: std::mem::take(&mut session_take.as_mut().unwrap().logits_buf),
                cos_buf: std::mem::take(&mut session_take.as_mut().unwrap().cos_buf),
                sin_buf: std::mem::take(&mut session_take.as_mut().unwrap().sin_buf),
                hidden_tap_buf: Vec::new(),
                hidden_tap_layers,
                hidden_tap_batch_buf: Vec::new(),
                batch_normed: Vec::new(),
                batch_qkv: Vec::new(),
                batch_out: Vec::new(),
                batch_k: Vec::new(),
                batch_v: Vec::new(),
                batch_tmp: Vec::new(),
                batch_ssm_alpha: Vec::new(),
                batch_ssm_beta: Vec::new(),
                batch_ssm_gate: Vec::new(),
                batch_cos_sin: Vec::new(),
            }
        } else {
            // ★ 全量模式: 创建新 ctx (CLI 单次生成)
            start_pos = 0;
            let mut ctx = make_context(weights, cfg);
            {
                let spec = self.spec_ctx.as_mut().unwrap();
                ctx.hidden_tap_layers = spec.cfg().target_layers.clone();
                n_tap_layers = ctx.hidden_tap_layers.len();
                eprintln!("[dspark] full mode: tap_layers={n_tap_layers}");
                // 重置 spec_ctx cache (全量 prefill, 新 target_tap_history)
                spec.target_tap_len = 0;
                spec.drafter.cached_ctx_len = 0;
                spec.drafter.cached_kv_len = 0;
            }
            ctx
        };

        // 4. prefill (对齐 llama.cpp speculative-simple.cpp L210-216:
        //    forward N-1 tokens, 最后一个 token 作为 anchor/id_last, 不 forward)
        //    ★ 增量模式: start_pos = session.state.pos (从 KV cache 末尾继续)
        //    ★ 全量模式: start_pos = 0 (从头 prefill)
        let n_input = input_ids.len();
        let prefill_start = std::time::Instant::now();
        // ★ 累积所有 prefill batch 的 hidden tap (而非只保留最后一个 batch)
        let mut prefill_tap_acc: Vec<f32> = Vec::new();
        // ★ DSpark 切换恢复: 先 prefill 整个 history_tokens 重建 KV/SSM state
        //   (不累积 hidden tap, drafter context 从增量 token 开始)
        if !replay_history.is_empty() {
            const MAX_BATCH: usize = 64;
            let np = replay_history.len();
            let mut off = 0usize;
            while off < np {
                let end = (off + MAX_BATCH).min(np);
                forward_batch(&mut ctx, &replay_history[off..end], off, None)?;
                off = end;
            }
            start_pos = ctx.state.pos;
            eprintln!("[dspark] replayed {} history tokens, start_pos now {}", np, start_pos);
        }
        if n_input >= 2 {
            // ★ 与原生 generate_inner (L806) 逐位对齐: anchor (末位 token) 走 batch matvec
            //   内核而非 forward_single_token 的单 token 内核 — 消除 ~1e-6 数值差导致的
            //   greedy argmax 翻转 (先前 DSpark 与原生 greedy 在近平局处分歧的根因)。
            //   hidden_tap_batch_buf 覆盖 batch 内全部 token (含 anchor 末行),
            //   logits_buf = batch 末位 = anchor 的下一 token 分布 (draft[0] 的 target 分布)。
            let prefill_ids = &input_ids[..];   // 全部 token (含 anchor)
            const MAX_BATCH: usize = 64;
            let np = prefill_ids.len();
            let mut off = 0usize;
            while off < np {
                let end = (off + MAX_BATCH).min(np);
                forward_batch(&mut ctx, &prefill_ids[off..end], start_pos + off, None)?;
                prefill_tap_acc.extend_from_slice(&ctx.hidden_tap_batch_buf);
                off = end;
            }
        }
        let prefill_ms = prefill_start.elapsed().as_millis();
        let n_prefill = n_input;   // 含 anchor (batch 已 forward)
        eprintln!("\r[prefill] {n_prefill}/{n_input} done in {prefill_ms}ms (start_pos={start_pos})");

        // 5. DSpark decode 循环
        let mut rng = LcgRng::new(0xC0FFEE);
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut sampling_buf = SamplingBuffers::new(ctx.logits_buf.len());

        // 累积 target tap history: 每个已 forward token 一行 [n_embd_cap]
        // draft 时传整个 history 作为 drafter context (对齐 llama.cpp ctx_feat 累积语义)
        let hidden = cfg.hidden;
        let n_embd_cap = n_tap_layers * hidden;
        // ★ 增量模式: 从 session 取已累积的 target_tap_history (跨轮复用)
        // ★ 全量模式: 新建空 target_tap_history
        let mut target_tap_history: Vec<f32> = if let Some(session) = session_take.as_mut() {
            std::mem::take(&mut session.dspark_tap_history)
        } else {
            let est_rows = n_input.saturating_add(max_tokens);
            Vec::with_capacity(est_rows * n_embd_cap)
        };

        // prefill 阶段: 已逐批累积到 prefill_tap_acc (行优先 token-major)
        // 直接追加到 target_tap_history (增量模式下 history 已含历史行)
        target_tap_history.extend_from_slice(&prefill_tap_acc);

        // anchor = 最后一个 input token (对齐 llama.cpp: id_last = inp.back())
        // 不 sample, 不加入 generated_ids (anchor 不是生成的 token)
        let mut anchor_token = input_ids[n_input - 1];
        if n_input == 1 {
            // n_input==1: batch 无意义, 单 token forward (与原生 generate_inner L803 对齐)
            // —— 原生 n_input==1 也走 forward_single_token, 这里保持一致
            forward_single_token(&mut ctx, anchor_token)?;
            target_tap_history.extend_from_slice(&ctx.hidden_tap_buf);
        }
        // n_input>=2: anchor 已在 batch prefill 中 forward (末行), hidden tap 由
        //   prefill_tap_acc 末行提供 (已 extend 进 target_tap_history),
        //   logits_buf = batch 末位 = draft[0] 的 target 分布。不再单独 forward。

        // ★ think 关闭时, build_increment / handle_chat_dspark 已在 prompt 末尾预填空 think 块
        //   <think></think>\n, 模型不再生成 <think>...</think> 内容, 直接输出正式回答。
        //   因此 think_suppress 恒为 false, 不再需要 suppress_buf 过滤逻辑。
        //   (保留 think_suppress 变量仅为 emit_delta! / Accept/Reject 条件分支兼容)
        let _think_enabled = session_take.as_ref().map(|s| s.think_enabled).unwrap_or(false);
        let mut think_suppress = false;
        let mut suppress_buf: String = String::new();

        // ★ PLD drafter: 2-gram 表从完整 token 流 (session 历史 + 本轮输入) 构建
        let mut ngram = if pld_mode {
            let mut stream: Vec<u32> = Vec::new();
            if let Some(s) = session_take.as_ref() {
                stream.extend_from_slice(&s.history_tokens);
            }
            stream.extend_from_slice(&input_ids);
            let d = NgramDrafter::from_stream(&stream);
            eprintln!("[pld] 2-gram table: {} entries from {} tokens", d.len(), stream.len());
            d
        } else {
            NgramDrafter::from_stream(&[])
        };

        let decode_start = std::time::Instant::now();
        let stream_output = !matches!(std::env::var("DAIZA_STREAM").as_deref(),
            Ok("0") | Ok("false") | Ok("no"));
        // 增量解码缓冲 (跨 token 不完整 UTF-8 字节暂存, 与 session_reply_stream 一致)
        let mut pending_bytes: Vec<u8> = Vec::new();
        let mut broke = false;
        // 增量回调 helper: decode token → drain UTF-8 → emit Delta, 返回 false 则中断
        // ★ think_suppress 模式下: 累积文本到 suppress_buf, 检测 </think>, 过滤 think 内容
        macro_rules! emit_delta {
            ($tid:expr) => {{
                use crate::session::{decode_token_bytes, drain_complete_utf8};
                decode_token_bytes(&self.tokenizer, $tid as u32, &mut pending_bytes);
                let delta = drain_complete_utf8(&mut pending_bytes);
                if !delta.is_empty() {
                    if think_suppress {
                        suppress_buf.push_str(&delta);
                        if let Some(idx) = suppress_buf.find("</think>") {
                            let after = suppress_buf[idx + 8..].to_string();
                            suppress_buf.clear();
                            think_suppress = false;
                            if !after.is_empty() && !on_event(DsparkEvent::Delta(after)) {
                                broke = true;
                                break;
                            }
                        }
                    } else {
                        if !on_event(DsparkEvent::Delta(delta)) {
                            broke = true;
                            break;
                        }
                    }
                }
            }};
        }

        let block_size = self.spec_ctx.as_ref().unwrap().cfg().block_size;
        let mut total_draft_calls = 0usize;
        let mut total_accepted = 0usize;
        let mut total_bonus = 0usize;
        let mut total_draft_truncated = 0usize;  // confidence head 截断的 token 数
        let mut total_pld_hits = 0usize;         // PLD: 2-gram 表命中的 cycle 数
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
        // fallback_enabled 由调用方传入 (Web API 开关 / CLI 环境变量)
        const PROBE_CYCLES: usize = 6;
        let mut probe_done = !fallback_enabled;
        let mut probe_cycles = 0usize;
        let mut probe_gen_start = 0usize;
        let mut probe_time_start = 0u128;
        let mut fell_back = false;

        while generated_ids.len() < max_tokens {
            // --- Phase 1: Draft ---
            // ★ PLD 模式: 2-gram 查表 (k=1, ~100ns), miss → 空 draft → 跳过 verify
            //   直接 bonus (与 all-reject 等价, 1 forward/1 token, 零损失)。
            //   神经 drafter 路径的 set_target_tap / confidence head 不参与。
            let draft_start = std::time::Instant::now();
            let draft_tokens: Vec<u32> = if pld_mode {
                match ngram.lookup_next() {
                    Some(t) => { total_pld_hits += 1; vec![t] }
                    None => Vec::new(),
                }
            } else {
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
                spec.draft(anchor_token, start_pos, params.temperature > 0.0).to_vec()
            };
            t_draft += draft_start.elapsed().as_millis();
            total_draft_calls += 1;

            // ★ Confidence head: 根据 confidence_logits 与 threshold 截断 draft tokens
            // 只 verify 前 n_draft_to_verify 个, 后面的 draft token 直接跳过 (省 target forward)
            // n_draft_to_verify == 0 → 跳过 Phase 2, 直接 bonus (与 all-reject 等价)
            // n_draft_to_verify == block_size → 不截断, 原行为
            // PLD 模式: k=1 无 confidence head, n = draft 长度 (0 或 1)
            let n_draft_to_verify = if pld_mode {
                draft_tokens.len()
            } else {
                self.spec_ctx.as_ref().unwrap()
                    .confident_prefix_length(confidence_threshold)
            };
            if !pld_mode {
                total_draft_truncated += block_size - n_draft_to_verify;
            }

            // === 乐观显示: emit Draft 事件 (前端灰色显示预测文本) ===
            // ★ think_suppress 时不发送 Draft 事件: 避免 drafter 预测的 think 内容
            //   通过 Draft 事件发给前端显示。accept 的 token 后续通过 Delta 事件发送 (带过滤)
            if n_draft_to_verify > 0 && !think_suppress {
                let mut draft_buf: Vec<u8> = Vec::new();
                for &dt in &draft_tokens[..n_draft_to_verify] {
                    if dt == self.config.eos_token_id {
                        break;
                    }
                    use crate::session::decode_token_bytes;
                    decode_token_bytes(&self.tokenizer, dt, &mut draft_buf);
                }
                // lossy decode: draft 可能含不完整 UTF-8 (跨 token 字符), verify 后会修正
                let draft_text = String::from_utf8_lossy(&draft_buf).into_owned();
                if !draft_text.is_empty() {
                    if !on_event(DsparkEvent::Draft(draft_text)) {
                        broke = true;
                        break;
                    }
                }
            }

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
            let _pos_before = ctx.state.pos;

            let mut n_accepted = 0usize;
            let mut bonus_token: Option<usize> = None;
            for (i, &dt) in draft_tokens[..n_draft_to_verify].iter().enumerate() {
                // ctx.logits_buf = 预测 draft[i] 的 target 分布 (前一个 forward 的输出)
                let target_logits_i = &mut ctx.logits_buf[..cfg.vocab_size];

                // ★ greedy 模式跳过 step_logits 复制 (draft_logits 为空)
                //   leviathan_check greedy 路径只需 target argmax, 不读 draft_logits
                //   reject 时 greedy 直接 argmax target 作为 bonus, 无需 q
                let accepted = if params.temperature <= 0.0 {
                    leviathan_check_greedy(target_logits_i, dt)
                } else {
                    let draft_logits_i = &self.spec_ctx.as_ref().unwrap().draft_logits
                        [i * cfg.vocab_size..(i + 1) * cfg.vocab_size];
                    leviathan_check(
                        target_logits_i, draft_logits_i, dt, params, &mut || rng.next_f32(),
                    )
                };

                if !accepted {
                    // Reject: 用 ctx.logits_buf 采样 bonus (预测 draft[i] 的分布)
                    // ★ 重复惩罚: 对 target_logits_i 应用 (reject 路径)
                    apply_repetition_penalty(target_logits_i, &generated_ids, params.repetition_penalty, params.frequency_penalty);
                    bonus_token = Some(if params.temperature <= 0.0 {
                        // greedy: 直接 argmax target (省 sample_bonus 的 p/q softmax)
                        sample_top_k_top_p_into(
                            target_logits_i, params,
                            &mut || rng.next_f32(), &mut sampling_buf,
                        )
                    } else {
                        let draft_logits_i = &self.spec_ctx.as_ref().unwrap().draft_logits
                            [i * cfg.vocab_size..(i + 1) * cfg.vocab_size];
                        sample_bonus(
                            target_logits_i, draft_logits_i, params,
                            &mut || rng.next_f32(), &mut sampling_buf, &mut bonus_buf,
                        )
                    });
                    break;
                }

                // Accept: forward draft[i] (更新 KV/SSM, 产生 draft[i+1] 的 logits)
                n_accepted += 1;
                if dt == self.config.eos_token_id {
                    eprintln!("[dspark] EOS accepted at draft pos {i}");
                    // ★ EOS 不 push 到 generated_ids (与 session.rs decode 循环一致:
                    //   EOS 终止生成且不加入输出文本, 否则 decode 会产生字面 "<|im_end|>" 文本)
                    // ★ think_suppress 模式: 前端无 Draft 灰色文本, 不需发 Accept/Reject 事件
                    //   EOS 之前的 accept token 已在循环中通过 emit_delta! 发送 (带 think 过滤)
                    // ★ think 开启模式: 必须发 Accept/Reject 事件清除前端 Draft 灰色文本
                    //   否则前端 Draft 乐观显示的文本永远不会被 commit (无后续 Delta/Reject)
                    //   注: EOS token 不 decode 到文本, accepted_text 只含 EOS 之前的 token
                    if !think_suppress && n_draft_to_verify > 0 {
                        if n_accepted == n_draft_to_verify {
                            if !on_event(DsparkEvent::Accept) {
                                broke = true;
                                break;
                            }
                        } else {
                            let mut acc_buf: Vec<u8> = Vec::new();
                            // 只 decode EOS 之前的 token (n_accepted-1, 排除 EOS 自身)
                            for &at in &draft_tokens[..n_accepted - 1] {
                                use crate::session::decode_token_bytes;
                                decode_token_bytes(&self.tokenizer, at, &mut acc_buf);
                            }
                            let accepted_text = String::from_utf8_lossy(&acc_buf).into_owned();
                            if !on_event(DsparkEvent::Reject(accepted_text)) {
                                broke = true;
                                break;
                            }
                        }
                    }
                    broke = true;
                    break;
                }
                generated_ids.push(dt);
                if pld_mode {
                    ngram.commit(dt);
                }
                forward_single_token(&mut ctx, dt)?;
                n_target_forwards += 1;
                // ★ think_suppress 模式: accept 的 token 通过 emit_delta! 发送 (带 think 过滤)
                //   不发 Accept/Reject 事件 (前端无 Draft 灰色文本需要修正)
                // ★ think 开启模式: 不 emit_delta, 通过 Accept/Reject 事件修正前端 Draft 灰色文本
                if think_suppress {
                    emit_delta!(dt);
                }
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

            // === 乐观显示修正: 根据 verify 结果发 Accept / Reject 事件 ===
            // - 全部 accept (n_accepted == n_draft_to_verify): emit Accept (前端灰色保留为最终文本)
            // - 部分/全部 reject: emit Reject(accepted_text)
            //   前端删除上一个 Draft 的全部文本, 用 accepted_text 替换 (黑色)
            //   accepted_text = decode(draft_tokens[..n_accepted]) (lossy, 与 Draft 的 UTF-8 drain 一致)
            // ★ think_suppress 模式: 不发 Accept/Reject 事件 (accept 的 token 已通过 emit_delta! 发送,
            //   经过 suppress_buf 过滤 think 内容; 前端无 Draft 灰色文本需要修正)
            if !think_suppress && n_draft_to_verify > 0 && !broke {
                if n_accepted == n_draft_to_verify {
                    if !on_event(DsparkEvent::Accept) {
                        broke = true;
                        break;
                    }
                } else {
                    // decode accepted 部分
                    let mut acc_buf: Vec<u8> = Vec::new();
                    for &dt in &draft_tokens[..n_accepted] {
                        use crate::session::decode_token_bytes;
                        decode_token_bytes(&self.tokenizer, dt, &mut acc_buf);
                    }
                    let accepted_text = String::from_utf8_lossy(&acc_buf).into_owned();
                    if !on_event(DsparkEvent::Reject(accepted_text)) {
                        broke = true;
                        break;
                    }
                }
            }

            // 增量回调中断: 跳过 Phase 3 + 退出 decode 循环
            if broke {
                break;
            }

            // --- Phase 3: Bonus forward ---
            // reject: bonus = sample_bonus(target_logits_i, draft_logits_i) (已采样)
            // all-accept: bonus = sample from ctx.logits_buf (最后接受 token 的 forward 输出)
            let bonus_start = std::time::Instant::now();
            let bt_raw = if let Some(bt) = bonus_token {
                bt
            } else {
                // All-accept: 从 ctx.logits_buf 采样 (最后一个 forward 的输出, 预测 pos_before+k)
                // ★ 重复惩罚: bonus 路径 (all-accept 时 bonus_token 为 None)
                apply_repetition_penalty(&mut ctx.logits_buf, &generated_ids, params.repetition_penalty, params.frequency_penalty);
                sample_top_k_top_p_into(
                    &ctx.logits_buf, params,
                    &mut || rng.next_f32(), &mut sampling_buf,
                )
            };
            let bt = bt_raw as u32;
            if bt == self.config.eos_token_id {
                eprintln!("[dspark] EOS from bonus");
                break;
            }
            generated_ids.push(bt);
            anchor_token = bt;
            if pld_mode {
                ngram.commit(bt);
            }
            forward_single_token(&mut ctx, bt)?;
            emit_delta!(bt);
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
                emit_delta!(next_id);
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
        // 收尾: 未提前中断时, 把残留的不完整字节以 lossy 形式上报 (与 session_reply_stream 一致)
        if !broke && !pending_bytes.is_empty() {
            let tail = String::from_utf8_lossy(&pending_bytes);
            if !tail.is_empty() {
                let _ = on_event(DsparkEvent::Delta(tail.into_owned()));
            }
        }
        let decode_ms = decode_start.elapsed().as_millis();
        let n_gen = generated_ids.len();
        eprintln!("[bench] dspark decode({n_gen}t)={decode_ms}ms (~{}ms/tok ~{:.2} tok/s)",
            if n_gen > 0 { decode_ms / n_gen as u128 } else { 0 },
            if decode_ms > 0 { n_gen as f64 * 1000.0 / decode_ms as f64 } else { 0.0 });
        if pld_mode {
            // PLD k=1: p = 接受率 (每 cycle 至多接受 1 个 draft);
            // hit = 2-gram 表命中 cycle 数 (hit 率低 → 文本无重复; hit 高但 p 低 → 表预测不准)
            let p = if total_draft_calls > 0 {
                total_accepted as f64 / total_draft_calls as f64
            } else { 0.0 };
            let hit_rate = if total_draft_calls > 0 {
                total_pld_hits as f64 / total_draft_calls as f64
            } else { 0.0 };
            eprintln!("[pld] cycles={total_draft_calls} hits={total_pld_hits} ({:.0}%) \
                accepted={total_accepted} (p={p:.2}) bonus={total_bonus} table_entries={}",
                hit_rate * 100.0, ngram.len());
        } else {
            eprintln!("[dspark] draft_calls={total_draft_calls} accepted={total_accepted} \
                (avg {:.2}/{block_size}) bonus={total_bonus} truncated={total_draft_truncated} \
                (conf_threshold={confidence_threshold})",
                if total_draft_calls > 0 { total_accepted as f64 / total_draft_calls as f64 }
                else { 0.0 });
        }

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

        // ★ 增量模式: 状态移回 session (KV cache + SSM state + target_tap_history)
        //   确保下一轮 DSpark 对话复用 KV cache, 避免全量 reprefill
        if let Some(mut session) = session_take {
            session.state = std::mem::take(&mut ctx.state);
            session.h_buf = std::mem::take(&mut ctx.h_buf);
            session.workspace = std::mem::take(&mut ctx.workspace);
            session.logits_buf = std::mem::take(&mut ctx.logits_buf);
            session.cos_buf = std::mem::take(&mut ctx.cos_buf);
            session.sin_buf = std::mem::take(&mut ctx.sin_buf);
            session.dspark_tap_layers = std::mem::take(&mut ctx.hidden_tap_layers);
            session.dspark_tap_history = target_tap_history;
            // ★ 维护 history_tokens (与非 DSpark 模式一致),
            //   DSpark 切换到非 DSpark 时可从 history_tokens 恢复上下文
            session.history_tokens.extend(&input_ids);
            session.history_tokens.extend(&generated_ids);
            self.session = Some(session);
        }

        crate::session_persist::dump_tokens_if_enabled(&input_ids, &generated_ids)?;

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
        daiza_engine::model::weights::LoadedWeights::print_dtype_summary(&self.gguf);
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
    // <think> 作为 special token 被 tokenizer 识别, \n 作为普通 token
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
    // ★ AVX2 向量化 argmax (vocab=248320, 每 cycle 多次调用)
    if params.temperature <= 0.0 {
        let (target_argmax, _) = daiza_engine::math::simd_exp::argmax_avx2(target_logits);
        return target_argmax == dt;
    }

    // 采样模式: 标准 Leviathan 随机接受 u < p[dt]/q[dt]
    // ★ 优化: AVX2 融合 softmax_pair_single, target+draft 并行遍历
    //   原 softmax_single 标量两次调用 = 4 次 vocab 遍历 (~16ms/cycle)
    //   现融合为 2 次 AVX2 遍历 (~4ms/cycle), 省 ~12ms/cycle
    let (p_dt, q_dt) = daiza_engine::math::simd_exp::softmax_pair_single_avx2(
        target_logits, draft_logits, dt,
    );

    let r = if q_dt > 1e-12 { p_dt / q_dt } else { 0.0 };
    let u = rng();
    u < r
}

/// Greedy 专用 Leviathan check: 只比较 target argmax 与 draft_token
/// ★ 跳过 draft_logits 参数 (greedy 模式下未填充, 省全 vocab copy)
#[inline]
fn leviathan_check_greedy(target_logits: &[f32], draft_token: u32) -> bool {
    let (target_argmax, _) = daiza_engine::math::simd_exp::argmax_avx2(target_logits);
    target_argmax == draft_token as usize
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
    softmax_into(target_logits, p);
    softmax_into(draft_logits, q);

    // residual = max(0, p - q), 归一化为概率分布, 转为 logits 供 top_k/top_p 采样
    // ★ AVX2 向量化: max(0, p-q) + sum 累加 (原标量 248K iter, 现 31K×8-wide)
    let residual = &mut bonus_buf.residual;
    let sum = daiza_engine::math::simd_exp::residual_max_zero_sum_avx2(p, q, residual);
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
/// ★ 复用 daiza_engine::math::softmax_inplace 的 3-pass SIMD 实现
///   原标量实现 ~6ms (vocab=248K), SIMD ~1.5ms
fn softmax_into(logits: &[f32], out: &mut [f32]) {
    out[..logits.len()].copy_from_slice(logits);
    daiza_engine::math::softmax_inplace(&mut out[..logits.len()]);
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

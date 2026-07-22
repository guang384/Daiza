//! Session 持久化:dump 到 SSD / 从 SSD 恢复
//!
//! 文件格式 (v1):
//! ```text
//! [Header 128B]
//!   magic: "DZSS" (4B)
//!   version: u32 (=1)
//!   cfg_hash: u64 (Config 关键字段哈希, 加载时校验)
//!   pos: usize
//!   n_history: usize
//!   n_blocks: u32 (=block_count)
//!   think_enabled: u8
//!   reserved: [u8; 83]  ← 预留扩展位
//!
//! [History Tokens]
//!   [u32; n_history]
//!
//! [Per-block state × n_blocks]
//!   每个 block:
//!     kind: u8  (0=attention KV, 1=SSM)
//!     if kind == 0 (KV):
//!       per_token: usize
//!       len: usize
//!       k: [f32; len * per_token]
//!       v: [f32; len * per_token]
//!     if kind == 1 (SSM):
//!       state_len: usize
//!       state: [f32; state_len]
//!       conv_history_len: usize
//!       conv_history: [f32; conv_history_len]
//!       conv_head: usize
//! ```
//!
//! 典型大小:SSM state 固定 ~150MB (48 layers × 128×128 × 4B) +
//! KV cache 按 pos 线性增长 (~128KB/token) + history 几 KB。
//! 实测 306 token session ≈ 178MB, dump ~80ms (2.2 GB/s), load ~80ms。

use daiza_engine::cache::{KvCache, SsmState};
use daiza_engine::model::config::Config;
use daiza_engine::model::forward::ModelState;
use crate::session::Session;
use crate::Result;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"DZSS";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 128;

/// Config 关键字段哈希 (用于加载时校验模型一致性)
fn cfg_hash(cfg: &Config) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    cfg.block_count.hash(&mut h);
    cfg.hidden.hash(&mut h);
    cfg.head_count.hash(&mut h);
    cfg.head_count_kv.hash(&mut h);
    cfg.head_dim.hash(&mut h);
    cfg.full_attention_interval.hash(&mut h);
    cfg.ssm_state_size.hash(&mut h);
    cfg.ssm_time_step_rank.hash(&mut h);
    cfg.ssm_conv_kernel.hash(&mut h);
    cfg.context_length.hash(&mut h);
    h.finish()
}

impl Session {
    /// dump 当前 session 到 SSD
    ///
    /// 顺序写:Header → history → per-block state
    pub fn save_to_disk(&self, path: &Path, cfg: &Config) -> Result<()> {
        let mut file = std::fs::File::create(path)
            .map_err(|e| crate::BonsaiError::Io(format!("create {}: {e}", path.display())))?;

        // === Header ===
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(MAGIC);
        header[4..8].copy_from_slice(&VERSION.to_le_bytes());
        header[8..16].copy_from_slice(&cfg_hash(cfg).to_le_bytes());
        header[16..24].copy_from_slice(&(self.state.pos as u64).to_le_bytes());
        header[24..32].copy_from_slice(&(self.history_tokens.len() as u64).to_le_bytes());
        header[32..36].copy_from_slice(&(cfg.block_count as u32).to_le_bytes());
        header[36] = self.think_enabled as u8;
        // reserved: header[37..128] 已为零
        file.write_all(&header)
            .map_err(|e| crate::BonsaiError::Io(format!("write header: {e}")))?;

        // === History tokens ===
        let hist_bytes = bytemuck::cast_slice::<u32, u8>(&self.history_tokens);
        file.write_all(hist_bytes)
            .map_err(|e| crate::BonsaiError::Io(format!("write history: {e}")))?;

        // === Per-block state ===
        for blk_idx in 0..cfg.block_count {
            if cfg.is_full_attention_block(blk_idx) {
                // KV cache
                let kv = self.state.kv_caches[blk_idx].as_ref()
                    .ok_or_else(|| crate::BonsaiError::Io(format!("block {blk_idx}: KV cache missing")))?;
                file.write_all(&[0u8])?; // kind = 0 (KV)
                let per_token = kv.per_token() as u64;
                let len = kv.len as u64;
                file.write_all(&per_token.to_le_bytes())?;
                file.write_all(&len.to_le_bytes())?;
                // 只写 len * per_token 个 f32 (不写 capacity)
                let n = kv.len * kv.per_token();
                let k_bytes = bytemuck::cast_slice(&kv.k[..n]);
                let v_bytes = bytemuck::cast_slice(&kv.v[..n]);
                file.write_all(k_bytes)?;
                file.write_all(v_bytes)?;
            } else {
                // SSM state
                let ssm = self.state.ssm_states[blk_idx].as_ref()
                    .ok_or_else(|| crate::BonsaiError::Io(format!("block {blk_idx}: SSM state missing")))?;
                file.write_all(&[1u8])?; // kind = 1 (SSM)
                let state_len = ssm.state.len() as u64;
                file.write_all(&state_len.to_le_bytes())?;
                file.write_all(bytemuck::cast_slice(&ssm.state))?;
                let conv_len = ssm.conv_history.len() as u64;
                file.write_all(&conv_len.to_le_bytes())?;
                file.write_all(bytemuck::cast_slice(&ssm.conv_history))?;
                file.write_all(&(ssm.conv_head as u64).to_le_bytes())?;
            }
        }

        file.flush()?;
        Ok(())
    }

    /// 异步后台 dump:clone state 后 spawn 线程写盘,不阻塞下一轮对话
    ///
    /// 代价:临时多占 ~2.8GB DRAM (state clone)
    /// 收益:下一轮对话不被 dump 阻塞
    /// 注意:返回 JoinHandle,调用方可选择 join 或忽略
    pub fn save_to_disk_async(
        &self,
        path: &Path,
        cfg: &Config,
    ) -> std::thread::JoinHandle<Result<()>> {
        // 只 clone 序列化需要的字段 (workspace/buffers 不需要)
        let state = self.state.clone();
        let history_tokens = self.history_tokens.clone();
        let think_enabled = self.think_enabled;
        let system_prompt = self.system_prompt.clone();
        let path = path.to_path_buf();
        let cfg = cfg.clone();

        std::thread::spawn(move || {
            let session = Session {
                state,
                workspace: daiza_engine::model::workspace::Workspace::new(&cfg),
                history_tokens,
                h_buf: Vec::new(),
                logits_buf: Vec::new(),
                cos_buf: Vec::new(),
                sin_buf: Vec::new(),
                think_enabled,
                system_prompt,
                pending_images: Vec::new(),
                tools: Vec::new(),
                messages: Vec::new(),
            };
            session.save_to_disk(&path, &cfg)
        })
    }

    /// 从 SSD 恢复 session
    ///
    /// 顺序读, 跳过 prefill, 直接重建 ModelState
    pub fn load_from_disk(path: &Path, cfg: &Config) -> Result<Self> {

        let mut file = std::fs::File::open(path)
            .map_err(|e| crate::BonsaiError::Io(format!("open {}: {e}", path.display())))?;

        // === Header ===
        let mut header = [0u8; HEADER_SIZE];
        file.read_exact(&mut header)
            .map_err(|e| crate::BonsaiError::Io(format!("read header: {e}")))?;

        if &header[0..4] != MAGIC {
            return Err(crate::BonsaiError::Io("invalid magic (not a Daiza session file)".into()));
        }
        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(crate::BonsaiError::Io(format!(
                "unsupported version {version}, expected {VERSION}"
            )));
        }
        let file_cfg_hash = u64::from_le_bytes(header[8..16].try_into().unwrap());
        let expected_hash = cfg_hash(cfg);
        if file_cfg_hash != expected_hash {
            return Err(crate::BonsaiError::Io(format!(
                "cfg hash mismatch: file={file_cfg_hash:016x} expected={expected_hash:016x} (model mismatch?)"
            )));
        }
        let pos = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;
        let n_history = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;
        let n_blocks = u32::from_le_bytes(header[32..36].try_into().unwrap()) as usize;
        let think_enabled = header[36] != 0;

        if n_blocks != cfg.block_count {
            return Err(crate::BonsaiError::Io(format!(
                "block_count mismatch: file={n_blocks} expected={}", cfg.block_count
            )));
        }

        // === History tokens ===
        let mut history_tokens = vec![0u32; n_history];
        file.read_exact(bytemuck::cast_slice_mut(&mut history_tokens))
            .map_err(|e| crate::BonsaiError::Io(format!("read history: {e}")))?;

        // === Per-block state ===
        let mut kv_caches = Vec::with_capacity(cfg.block_count);
        let mut ssm_states = Vec::with_capacity(cfg.block_count);

        for blk_idx in 0..cfg.block_count {
            let mut kind_byte = [0u8; 1];
            file.read_exact(&mut kind_byte)?;
            let kind = kind_byte[0];

            if cfg.is_full_attention_block(blk_idx) {
                // 期望 KV
                if kind != 0 {
                    return Err(crate::BonsaiError::Io(format!(
                        "block {blk_idx}: expected KV (kind=0), got kind={kind}"
                    )));
                }
                let per_token = u64::from_le_bytes(read_u8x8(&mut file)? as [u8; 8]) as usize;
                let len = u64::from_le_bytes(read_u8x8(&mut file)? as [u8; 8]) as usize;
                let n = len * per_token;
                let mut k = vec![0.0f32; n];
                let mut v = vec![0.0f32; n];
                file.read_exact(bytemuck::cast_slice_mut(&mut k))?;
                file.read_exact(bytemuck::cast_slice_mut(&mut v))?;
                kv_caches.push(Some(KvCache::from_raw(k, v, len, per_token)));
                ssm_states.push(None);
            } else {
                // 期望 SSM
                if kind != 1 {
                    return Err(crate::BonsaiError::Io(format!(
                        "block {blk_idx}: expected SSM (kind=1), got kind={kind}"
                    )));
                }
                let state_len = u64::from_le_bytes(read_u8x8(&mut file)? as [u8; 8]) as usize;
                let mut state = vec![0.0f32; state_len];
                file.read_exact(bytemuck::cast_slice_mut(&mut state))?;
                let conv_len = u64::from_le_bytes(read_u8x8(&mut file)? as [u8; 8]) as usize;
                let mut conv_history = vec![0.0f32; conv_len];
                file.read_exact(bytemuck::cast_slice_mut(&mut conv_history))?;
                let conv_head = u64::from_le_bytes(read_u8x8(&mut file)? as [u8; 8]) as usize;
                ssm_states.push(Some(SsmState { state, conv_history, conv_head }));
                kv_caches.push(None);
            }
        }

        // === 重建 ModelState + Session ===
        let mut state = ModelState::new(cfg);
        state.kv_caches = kv_caches;
        state.ssm_states = ssm_states;
        state.pos = pos;
        // rope_freqs / rope_sections 已由 ModelState::new 初始化

        Ok(Self {
            state,
            workspace: daiza_engine::model::workspace::Workspace::new(cfg),
            history_tokens,
            h_buf: vec![0.0; cfg.hidden],
            logits_buf: Vec::with_capacity(cfg.vocab_size),
            cos_buf: vec![0.0; cfg.rope_dim],
            sin_buf: vec![0.0; cfg.rope_dim],
            think_enabled,
            system_prompt: None,
            pending_images: Vec::new(),
            tools: Vec::new(),
            messages: Vec::new(),
        })
    }
}

/// 辅助:读取 8 字节
fn read_u8x8<R: Read>(r: &mut R) -> Result<[u8; 8]> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// dump token IDs 到文件 (env DAIZA_DUMP_TOKENS=path 控制)
///
/// 格式:
/// ```text
/// prompt:id1,id2,...
/// generated:id1,id2,...
/// ```
/// engine.rs (generate_inner) 与 session.rs (session_reply) 共用,
/// 便于 baseline vs optimized 的逐 token A/B 对比。
pub fn dump_tokens_if_enabled(prompt_ids: &[u32], generated_ids: &[u32]) -> Result<()> {
    let Ok(path) = std::env::var("DAIZA_DUMP_TOKENS") else {
        return Ok(());
    };
    let mut content = String::new();
    content.push_str("prompt:");
    for (i, &id) in prompt_ids.iter().enumerate() {
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
        .map_err(|e| crate::BonsaiError::Io(format!("dump_tokens write failed: {e}")))?;
    eprintln!("[dump_tokens] wrote {} generated ids to {}", generated_ids.len(), path);
    Ok(())
}

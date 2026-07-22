//! 多 session 管理:active session 在 engine.session 中,inactive 在 SSD
//!
//! LRU 策略:当前活跃 session 切换时自动 dump 到 SSD;
//! 切回时从 SSD load 回 DRAM。
//!
//! 设计:SessionManager 不持有 active Session (避免与 Engine.session 双重持有),
//! 只跟踪 active_id + inactive 队列。REPL 负责在 engine.session 和 SSD 之间搬运。
//!
//! 适用场景:多用户多对话、长时间运行的服务进程

use daiza_engine::model::config::Config;
use crate::session::Session;
use crate::Result;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// 多 session 管理器
///
/// - `active_id`: 当前活跃 session 的 id (session 本身在调用方 Engine.session 中)
/// - `inactive`: 已 dump 到 SSD 的非活跃 session 路径 (LRU 顺序,队首=最久未用)
/// - `ssd_dir`: session 文件存储目录
/// - `max_inactive`: 最大 inactive 数量 (超过时淘汰最旧的)
pub struct SessionManager {
    /// 当前活跃 session 的 id (session 本身在调用方 Engine.session 中)
    active_id: Option<String>,
    /// 非活跃 session 的 LRU 队列 (id, path),队首=最久未用
    inactive: VecDeque<(String, PathBuf)>,
    /// session 文件存储目录
    ssd_dir: PathBuf,
    /// 最大 inactive 数量 (超过时淘汰最旧的)
    max_inactive: usize,
}

impl SessionManager {
    /// 创建 SessionManager,指定 SSD 存储目录和最大 inactive 数量
    pub fn new(ssd_dir: &Path, max_inactive: usize) -> Result<Self> {
        std::fs::create_dir_all(ssd_dir)
            .map_err(|e| crate::BonsaiError::Io(format!("create ssd_dir: {e}")))?;
        Ok(Self {
            active_id: None,
            inactive: VecDeque::new(),
            ssd_dir: ssd_dir.to_path_buf(),
            max_inactive,
        })
    }

    /// 当前活跃 session 的 id
    pub fn active_id(&self) -> Option<&str> {
        self.active_id.as_deref()
    }

    /// 设置当前活跃 session 的 id (调用方在 engine.session_begin / load 后调用)
    pub fn set_active_id(&mut self, id: Option<String>) {
        self.active_id = id;
    }

    /// 列出所有 session id (active + inactive)
    ///
    /// 返回 (id, is_active) 列表,active 在前
    pub fn list(&self) -> Vec<(String, bool)> {
        let mut result = Vec::new();
        if let Some(id) = &self.active_id {
            result.push((id.clone(), true));
        }
        for (id, _) in &self.inactive {
            result.push((id.clone(), false));
        }
        result
    }

    /// 把一个 session dump 到 SSD 并加入 inactive 队列
    ///
    /// 调用方应先 `engine.session.take()` 取出 session, 再调用本方法
    /// 本方法会清除 active_id (调用方应确保 engine.session 已 take)
    pub fn park(&mut self, id: &str, session: &Session, cfg: &Config) -> Result<()> {
        let path = self.session_path(id);
        session.save_to_disk(&path, cfg)?;
        self.add_to_inactive(id, path);
        Ok(())
    }

    /// 异步 park:clone state 后 spawn 线程写盘,不阻塞下一轮对话
    ///
    /// 代价:临时多占 ~2.8GB DRAM (state clone)
    /// 返回 JoinHandle,调用方可选择 join 或忽略
    pub fn park_async(
        &mut self,
        id: &str,
        session: Session,
        cfg: &Config,
    ) -> std::thread::JoinHandle<Result<()>> {
        let path = self.session_path(id);
        let handle = session.save_to_disk_async(&path, cfg);
        self.add_to_inactive(id, path);
        handle
    }

    /// 内部:把 (id, path) 加入 inactive 队列, 处理去重 + LRU 淘汰 + 清除 active_id
    fn add_to_inactive(&mut self, id: &str, path: PathBuf) {
        // 同名 id 先移除旧路径 (覆盖)
        if let Some(pos) = self.inactive.iter().position(|(sid, _)| sid == id) {
            let (_, old_path) = self.inactive.remove(pos).unwrap();
            let _ = std::fs::remove_file(&old_path);
        }
        self.inactive.push_back((id.to_string(), path));
        // 淘汰最旧的 (队首)
        while self.inactive.len() > self.max_inactive {
            if let Some((_, old_path)) = self.inactive.pop_front() {
                let _ = std::fs::remove_file(&old_path);
            }
        }
        if self.active_id.as_deref() == Some(id) {
            self.active_id = None;
        }
    }

    /// 从 SSD load 一个 session 并从 inactive 队列移除
    ///
    /// 返回 Session, 调用方应 `engine.session = Some(returned)`
    /// 同时设置 active_id = Some(id)
    pub fn unpark(&mut self, id: &str, cfg: &Config) -> Result<Session> {
        let pos = self.inactive.iter().position(|(sid, _)| sid == id);
        let Some(pos) = pos else {
            return Err(crate::BonsaiError::Io(format!(
                "session '{id}' not found in inactive list"
            )));
        };

        let (_, path) = self.inactive.remove(pos).unwrap();
        let session = Session::load_from_disk(&path, cfg)?;
        self.active_id = Some(id.to_string());
        Ok(session)
    }

    /// 删除指定 session
    ///
    /// - 如果是 active: 只清除 active_id (调用方应 `engine.session = None`)
    /// - 如果是 inactive: 从队列移除并删除 SSD 文件
    pub fn delete(&mut self, id: &str) {
        if self.active_id.as_deref() == Some(id) {
            self.active_id = None;
            return;
        }

        let pos = self.inactive.iter().position(|(sid, _)| sid == id);
        if let Some(pos) = pos {
            let (_, path) = self.inactive.remove(pos).unwrap();
            let _ = std::fs::remove_file(&path);
        }
    }

    /// 检查 id 是否已存在 (active 或 inactive)
    pub fn exists(&self, id: &str) -> bool {
        self.active_id.as_deref() == Some(id)
            || self.inactive.iter().any(|(sid, _)| sid == id)
    }

    /// 生成 session 文件路径
    fn session_path(&self, id: &str) -> PathBuf {
        // 替换文件名中不合法的字符
        let safe_id = id.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
        self.ssd_dir.join(format!("{safe_id}.dzss"))
    }
}

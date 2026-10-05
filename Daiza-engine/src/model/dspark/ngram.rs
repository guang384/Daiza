//! N-gram PLD (Prompt Lookup Decoding) drafter
//!
//! 零成本 drafter: 用 2-gram 查表 ((prev, cur) → next) 替代神经 drafter forward,
//! 查表 ~100ns vs 神经 drafter ~53ms/call。draft 来源 = prompt / 已生成文本中的
//! n-gram 重复 (复述、引用、列表、代码块等场景命中率最高)。
//!
//! ## 性能语义 (sequential verify 下)
//!
//! forwards/token 恒 = 1.0: verify 逐 token forward, 接受的 draft 各付各的
//! forward, bonus 付自己的 forward (与原生 decode 的 1 forward/token 相同)。
//! 因此 PLD 不加速 decode, 但把 DSpark 的 draft overhead 从 ~53ms/call 降到
//! ~0 → DSpark 路径与原生 decode 持平 (不再触发 probe fallback), Draft 乐观
//! 显示 UI 以零成本常驻。
//!
//! 真正的加速需要 batched verify (k>1 个 draft 共享一次 13GB 权重遍历),
//! 见 engine.rs Phase 2 注释; PLD 的实测接受率 p 是该方向的决策数据。
//!
//! ## 语义约定
//!
//! - 表项后插入覆盖先插入 (most-recent 语义): prompt 尾部的重复模式优先于头部,
//!   与 llama.cpp lookup decode 从后向前扫描的行为一致。
//! - k=1: 每次 cycle 至多 draft 1 个 token; miss → 空 draft → 跳过 verify 直接
//!   bonus 路径 (等价 all-reject, 1 forward / 1 token, 零损失)。

use std::collections::HashMap;

/// 2-gram PLD drafter: (prev, cur) → next 查表 + 已提交 token 流
pub struct NgramDrafter {
    map: HashMap<(u32, u32), u32>,
    /// 已提交 token 流 (prompt + 历史 + 已生成), 尾部 2-gram 作为 lookup key
    ctx: Vec<u32>,
}

impl NgramDrafter {
    /// 从完整 token 流 (session 历史 + 本轮输入) 构建
    pub fn from_stream(stream: &[u32]) -> Self {
        let mut map = HashMap::with_capacity(stream.len());
        for i in 2..stream.len() {
            map.insert((stream[i - 2], stream[i - 1]), stream[i]);
        }
        Self { map, ctx: stream.to_vec() }
    }

    /// 提交一个 token (accepted draft / bonus): 插入最新 2-gram 表项并推进 ctx
    #[inline]
    pub fn commit(&mut self, t: u32) {
        let n = self.ctx.len();
        if n >= 2 {
            self.map.insert((self.ctx[n - 2], self.ctx[n - 1]), t);
        }
        self.ctx.push(t);
    }

    /// 用最近 2-gram 预测下一 token (miss → None → 纯 bonus 路径)
    #[inline]
    pub fn lookup_next(&self) -> Option<u32> {
        let n = self.ctx.len();
        if n < 2 {
            return None;
        }
        self.map.get(&(self.ctx[n - 2], self.ctx[n - 1])).copied()
    }

    /// 表项数 (统计用)
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_hit_after_chain() {
        // 表: (1,2)→3。ctx 尾部推进到 (1,2) 时应命中
        let mut d = NgramDrafter::from_stream(&[1, 2, 3]);
        assert_eq!(d.lookup_next(), None); // tail (2,3) 无表项
        d.commit(1);                       // 插入 (2,3)→1
        assert_eq!(d.lookup_next(), None); // tail (3,1) 无表项
        d.commit(2);                       // 插入 (3,1)→2
        assert_eq!(d.lookup_next(), Some(3)); // tail (1,2) → 3
    }

    #[test]
    fn most_recent_pair_wins() {
        // (1,2) 出现两次: 先 →3 后 →5, 后插入覆盖 (most-recent 语义)
        let d = NgramDrafter::from_stream(&[1, 2, 3, 9, 9, 1, 2, 5, 1, 2]);
        assert_eq!(d.lookup_next(), Some(5)); // tail (1,2) → 5
    }

    #[test]
    fn short_stream_safe() {
        let mut d = NgramDrafter::from_stream(&[7]);
        assert!(d.is_empty());
        assert_eq!(d.lookup_next(), None); // ctx < 2, 无 key
        d.commit(8); // n=1 不插入, 仅推进 ctx
        assert!(d.is_empty());
        assert_eq!(d.lookup_next(), None); // tail (7,8) 无表项
        d.commit(9); // 插入 (7,8)→9
        assert_eq!(d.len(), 1);
        assert_eq!(d.lookup_next(), None); // tail (8,9) 仍无表项
    }

    #[test]
    fn empty_stream_safe() {
        let mut d = NgramDrafter::from_stream(&[]);
        assert!(d.is_empty());
        assert_eq!(d.lookup_next(), None);
        d.commit(1);
        d.commit(2);
        d.commit(3); // 插入 (1,2)→3
        assert_eq!(d.len(), 1);
    }
}

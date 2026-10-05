//! GPT-2 byte-level BPE tokenizer(Qwen35 预分词)
//!
//! ## 工作流程
//!
//! 1. **byte-level 映射**:UTF-8 字节 → unicode 字符(避免 whitespace 控制字符问题)
//! 2. **预分词**:Qwen35 regex pattern 把文本切成段
//! 3. **BPE merge**:每段内重复应用 rank 最低的 pair,直到不能 merge
//! 4. **查 vocab**:每个最终 token 字符串 → token_id
//!
//! ## Qwen35 预分词 pattern
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! Rust 标准库无 regex,这里手写一个简化版:按字符类型分段(L/N/空白/标点)。
//! 完整 regex 留作后续练习(可考虑用 `regex` crate,但破坏零依赖)。

use crate::tokenizer::vocab::Vocab;

/// 编码段:特殊 token 或普通文本
enum EncodeSegment {
    Special(u32),
    Text(String),
}

pub struct BpeTokenizer {
    pub vocab: Vocab,
    /// token 文本 → token_id(快速查找)
    pub token_to_id: std::collections::HashMap<String, u32>,
    /// byte → unicode 字符的映射表(GPT-2 byte-level)
    pub byte_to_unicode_char: Vec<char>,
    /// unicode 字符 → byte 的反向映射
    pub unicode_char_to_byte: std::collections::HashMap<char, u8>,
    /// 特殊 token(如 <|im_start|>)→ token_id,编码时直接匹配,不走 BPE
    pub special_tokens: std::collections::HashMap<String, u32>,
}

impl BpeTokenizer {
    pub fn new(vocab: Vocab) -> Self {
        let mut token_to_id = std::collections::HashMap::new();
        for (i, t) in vocab.tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        let (byte_to_unicode_char, unicode_char_to_byte) = build_byte_to_unicode();

        // 收集特殊 token:token_type 为 Control(3) 或 UserDefined(4),
        // 或 token 文本以 <| 开头 |> 结尾(如 <|im_start|>,<|im_end|>)
        let mut special_tokens = std::collections::HashMap::new();
        for (i, t) in vocab.tokens.iter().enumerate() {
            let ttype = vocab.token_types.get(i).copied().unwrap_or(1);
            let is_special = ttype == 3 || ttype == 4 || (t.starts_with("<|") && t.ends_with("|>"));
            if is_special {
                special_tokens.insert(t.clone(), i as u32);
            }
        }

        Self {
            vocab,
            token_to_id,
            byte_to_unicode_char,
            unicode_char_to_byte,
            special_tokens,
        }
    }

    /// 把文本编码为 token_id 序列
    ///
    /// 先扫描特殊 token(如 `<|im_start|>`),直接替换为对应 token_id;
    /// 非特殊段走标准 byte-level BPE。
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut all_ids = Vec::new();

        // 1. 按特殊 token 切分文本
        let segments = self.split_by_special_tokens(text);

        for seg in segments {
            match seg {
                EncodeSegment::Special(id) => {
                    all_ids.push(id);
                }
                EncodeSegment::Text(t) => {
                    // 2. 对普通文本段走 byte-level BPE
                    for pre_token in pre_tokenize(&t) {
                        let bytes = pre_token.as_bytes();
                        let mut s = String::new();
                        for &b in bytes {
                            s.push(self.byte_to_unicode_char[b as usize]);
                        }
                        // bpe_merge 直接返回 token id 序列
                        let merged = self.bpe_merge(&s);
                        all_ids.extend(merged);
                    }
                }
            }
        }
        all_ids
    }

    /// 扫描文本,按特殊 token 切分
    fn split_by_special_tokens(&self, text: &str) -> Vec<EncodeSegment> {
        if self.special_tokens.is_empty() {
            return vec![EncodeSegment::Text(text.to_string())];
        }

        let mut segments = Vec::new();
        let mut remaining = text;

        'outer: loop {
            if remaining.is_empty() {
                break;
            }
            // 在剩余文本中找最早出现的特殊 token
            let mut earliest: Option<(usize, &String, u32)> = None;
            for (tok_str, &tok_id) in &self.special_tokens {
                if let Some(pos) = remaining.find(tok_str.as_str()) {
                    if earliest.is_none() || pos < earliest.unwrap().0 {
                        earliest = Some((pos, tok_str, tok_id));
                    }
                }
            }
            match earliest {
                Some((pos, tok_str, tok_id)) => {
                    // pos 之前的普通文本
                    if pos > 0 {
                        segments.push(EncodeSegment::Text(remaining[..pos].to_string()));
                    }
                    // 特殊 token 本身
                    segments.push(EncodeSegment::Special(tok_id));
                    remaining = &remaining[pos + tok_str.len()..];
                }
                None => {
                    // 没有更多特殊 token,剩余都是普通文本
                    if !remaining.is_empty() {
                        segments.push(EncodeSegment::Text(remaining.to_string()));
                    }
                    break 'outer;
                }
            }
        }
        segments
    }

    /// 把 token_id 序列解码为字符串
    pub fn decode(&self, ids: &[u32]) -> String {
        // 反向:每个 token_id → unicode 字符串 → bytes → UTF-8 文本
        let mut byte_seq: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(token_text) = self.vocab.tokens.get(id as usize) {
                for ch in token_text.chars() {
                    if let Some(&b) = self.unicode_char_to_byte.get(&ch) {
                        byte_seq.push(b);
                    } else {
                        // 特殊 token(如 <|im_start|>)原样输出
                        // (实际 GPT-2 BPE 仅处理 byte-level 映射内的字符)
                    }
                }
            }
        }
        String::from_utf8_lossy(&byte_seq).into_owned()
    }

    /// 对单个预分词段做 BPE merge
    ///
    /// 返回 token id 序列。使用 (id_a, id_b) pair 查 merges_by_id,
    /// 避免每次 pair 检查都分配新 String。
    fn bpe_merge(&self, s: &str) -> Vec<u32> {
        // 把字符串切成单字符,每个字符查 token_to_id 得到 token id
        let mut word: Vec<u32> = Vec::with_capacity(s.chars().count());
        for c in s.chars() {
            let cs = c.to_string();
            match self.token_to_id.get(cs.as_str()) {
                Some(&id) => word.push(id),
                None => {
                    eprintln!(
                        "[tokenizer] warning: char not in vocab, skipping: {:?}",
                        cs
                    );
                }
            }
        }
        if word.len() < 2 {
            return word;
        }

        // 重复应用 rank 最低的 merge pair
        loop {
            let mut min_rank: Option<u32> = None;
            let mut min_idx: Option<usize> = None;
            for i in 0..word.len() - 1 {
                let pair = (word[i], word[i + 1]);
                if let Some(&rank) = self.vocab.merges_by_id.get(&pair) {
                    if min_rank.is_none() || rank < min_rank.unwrap() {
                        min_rank = Some(rank);
                        min_idx = Some(i);
                    }
                }
            }
            match (min_rank, min_idx) {
                (Some(_), Some(idx)) => {
                    // merge word[idx] + word[idx+1] → 新 token id
                    let pair = (word[idx], word[idx + 1]);
                    if let Some(&merged_id) = self.vocab.merge_result.get(&pair) {
                        word[idx] = merged_id;
                        word.remove(idx + 1);
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        word
    }
}

/// GPT-2 的 byte-level 映射表
///
/// 把 256 个字节映射到 unicode 字符,规则:
/// - 可见 ASCII (33-126) 与 161-172 / 174-255 范围 → 字节本身
/// - 其他(0-32, 127-160, 173) → 从 256 开始的 unicode 字符
fn build_byte_to_unicode() -> (Vec<char>, std::collections::HashMap<char, u8>) {
    let mut bs: Vec<u32> = Vec::new();
    for b in b'!'..=b'~' {
        bs.push(b as u32);
    }
    for b in 0xA1u32..=0xACu32 {
        bs.push(b);
    }
    for b in 0xAEu32..=0xFFu32 {
        bs.push(b);
    }
    let mut cs: Vec<u32> = bs.clone();
    let mut n: u32 = 0;
    for b in 0u32..=255u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    // 现在两两配对:b ↔ c
    let mut byte_to_char = vec!['\0'; 256];
    let mut char_to_byte = std::collections::HashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(*c).unwrap_or('\u{FFFD}');
        byte_to_char[*b as usize] = ch;
        char_to_byte.insert(ch, *b as u8);
    }
    (byte_to_char, char_to_byte)
}

/// GPT-2 / Qwen 预分词(手写简化版,零依赖)
///
/// 模拟 pattern: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
///
/// 关键规则:**前导空格与后续字母合并为一个 segment**(如 " capital" → 一个预分词段)
/// 这样 byte-level 映射后 "Ġcapital" 才能被 BPE 正确 merge。
fn pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    let n = chars.len();
    while i < n {
        let c = chars[i];

        // 规则 b: [^\r\n\p{L}\p{N}]?\p{L}+  —— 可选前导非字母数字非换行 + 连续字母
        // 注意:前导字符只消费一个,且不能是 \r \n
        // 例: " capital" / ".word" / "The"
        if c.is_alphabetic()
            || (i + 1 < n
                && !c.is_alphabetic()
                && !c.is_numeric()
                && c != '\r'
                && c != '\n'
                && chars[i + 1].is_alphabetic())
        {
            let mut s = String::new();
            // 可选前导字符(空格或标点,但非 \r \n)
            if !c.is_alphabetic() {
                s.push(c);
                i += 1;
            }
            // 连续字母
            while i < n && chars[i].is_alphabetic() {
                s.push(chars[i]);
                i += 1;
            }
            tokens.push(s);
            continue;
        }

        // 规则 c: \p{N}  —— 单个数字
        if c.is_numeric() {
            tokens.push(c.to_string());
            i += 1;
            continue;
        }

        // 规则 d: ?[^\s\p{L}\p{N}]+[\r\n]*  —— 可选空格 + 连续标点 + 换行
        if !c.is_whitespace() {
            let mut s = String::new();
            // 可选前导空格(已在上面规则 b 处理过带字母的情况,这里只处理标点)
            while i < n && !chars[i].is_whitespace() && !chars[i].is_alphabetic() && !chars[i].is_numeric() {
                s.push(chars[i]);
                i += 1;
            }
            // 后续换行
            while i < n && (chars[i] == '\r' || chars[i] == '\n') {
                s.push(chars[i]);
                i += 1;
            }
            tokens.push(s);
            continue;
        }

        // 规则 e/f/g: 空白处理
        // \s*[\r\n]+ : 前导空白 + 换行
        // \s+(?!\S)  : 末尾空白(后无非空白)
        // \s+        : 其他空白
        if c.is_whitespace() {
            let start = i;
            // 消费连续空白
            while i < n && chars[i].is_whitespace() {
                i += 1;
            }
            // 如果包含换行,整段是一个 token
            let seg = &chars[start..i];
            let has_newline = seg.iter().any(|&ch| ch == '\r' || ch == '\n');
            if has_newline {
                tokens.push(seg.iter().collect());
            } else if i < n {
                // 纯空白后还有非空白字符 → 空白归入下一个 segment
                // (但下一个字母段已在规则 b 处理,这里实际上不会走到)
                // 回退:把空白作为单独 token
                tokens.push(seg.iter().collect());
            } else {
                // 末尾空白
                tokens.push(seg.iter().collect());
            }
            continue;
        }

        // fallback:单字符
        tokens.push(c.to_string());
        i += 1;
    }
    tokens
}

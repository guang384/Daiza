//! 极简 JSON 解析器 (零依赖)
//!
//! 支持: null / bool / number / string / array / object
//! 不支持: 注释 / 尾随逗号 (严格 JSON)
//!
//! 用 Vec<(String, Json)> 而非 HashMap 保持顺序 + 减少分配,
//! get() 线性查找 (OpenAI 请求字段数少, 性能足够)。

#[derive(Debug, Clone)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut p = Parser { bytes: s.as_bytes(), pos: 0 };
        p.skip_ws();
        let v = p.parse_value()?;
        p.skip_ws();
        if p.pos < p.bytes.len() {
            return Err(format!("trailing data at pos {}", p.pos));
        }
        Ok(v)
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self { Some(s) } else { None }
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Json::Bool(b) = self { Some(*b) } else { None }
    }

    pub fn as_f64(&self) -> Option<f64> {
        if let Json::Num(n) = self { Some(*n) } else { None }
    }

    pub fn as_u64(&self) -> Option<u64> {
        if let Json::Num(n) = self {
            if *n >= 0.0 { Some(*n as u64) } else { None }
        } else { None }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        if let Json::Arr(a) = self { Some(a) } else { None }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        if let Json::Obj(o) = self {
            o.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        } else {
            None
        }
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => self.parse_string().map(Json::Str),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(format!("unexpected char '{}' at pos {}", c as char, self.pos)),
            None => Err("unexpected end of input".into()),
        }
    }

    fn parse_object(&mut self) -> Result<Json, String> {
        self.pos += 1; // skip {
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(items));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at pos {}", self.pos));
            }
            self.pos += 1; // skip :
            let val = self.parse_value()?;
            items.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => { self.pos += 1; }
                Some(b'}') => { self.pos += 1; break; }
                _ => return Err(format!("expected ',' or '}}' at pos {}", self.pos)),
            }
        }
        Ok(Json::Obj(items))
    }

    fn parse_array(&mut self) -> Result<Json, String> {
        self.pos += 1; // skip [
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            let val = self.parse_value()?;
            items.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => { self.pos += 1; }
                Some(b']') => { self.pos += 1; break; }
                _ => return Err(format!("expected ',' or ']' at pos {}", self.pos)),
            }
        }
        Ok(Json::Arr(items))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err(format!("expected '\"' at pos {}", self.pos));
        }
        self.pos += 1; // skip "
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b'"' => { self.pos += 1; return Ok(out); }
                b'\\' => {
                    self.pos += 1;
                    if self.pos >= self.bytes.len() {
                        return Err("unterminated string escape".into());
                    }
                    match self.bytes[self.pos] {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'r' => out.push('\r'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0C}'),
                        b'u' => {
                            if self.pos + 4 >= self.bytes.len() {
                                return Err("invalid \\u escape".into());
                            }
                            let hex = std::str::from_utf8(&self.bytes[self.pos + 1..self.pos + 5])
                                .map_err(|_| "invalid \\u hex")?;
                            let mut cp = u32::from_str_radix(hex, 16)
                                .map_err(|_| "invalid \\u hex")?;
                            self.pos += 4;
                            // 代理对
                            if (0xD800..0xDC00).contains(&cp)
                                && self.pos + 6 < self.bytes.len()
                                && self.bytes[self.pos + 1] == b'\\'
                                && self.bytes[self.pos + 2] == b'u'
                            {
                                let hex2 = std::str::from_utf8(&self.bytes[self.pos + 3..self.pos + 7])
                                    .map_err(|_| "invalid surrogate \\u hex")?;
                                let lo = u32::from_str_radix(hex2, 16)
                                    .map_err(|_| "invalid surrogate \\u hex")?;
                                if (0xDC00..0xE000).contains(&lo) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                    self.pos += 6;
                                }
                            }
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                        c => return Err(format!("invalid escape '\\{}' at pos {}", c as char, self.pos)),
                    }
                    self.pos += 1;
                }
                _ => {
                    // 拷贝完整 UTF-8 序列
                    let len = utf8_len(self.bytes[self.pos]);
                    if self.pos + len > self.bytes.len() {
                        return Err("unterminated string".into());
                    }
                    let seg = std::str::from_utf8(&self.bytes[self.pos..self.pos + len])
                        .map_err(|_| "invalid UTF-8 in string")?;
                    out.push_str(seg);
                    self.pos += len;
                }
            }
        }
        Err("unterminated string".into())
    }

    fn parse_number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') { self.pos += 1; }
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-' => self.pos += 1,
                _ => break,
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid number")?;
        s.parse::<f64>().map(Json::Num).map_err(|e| format!("invalid number '{s}': {e}"))
    }

    fn parse_bool(&mut self) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Json::Bool(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Json::Bool(false))
        } else {
            Err(format!("invalid bool at pos {}", self.pos))
        }
    }

    fn parse_null(&mut self) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(Json::Null)
        } else {
            Err(format!("invalid null at pos {}", self.pos))
        }
    }
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 { 1 } else if b < 0xE0 { 2 } else if b < 0xF0 { 3 } else { 4 }
}

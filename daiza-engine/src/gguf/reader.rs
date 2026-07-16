//! 极简字节读取器,封装小端读取与边界检查
//!
//! 所有 GGUF 字段都以小端字节序存储。Rust 的 `std` 没有零依赖的
//! `ReadBytesExt`,所以这里手写一组明确的帮助函数。

use crate::gguf::err;
use crate::Result;

/// 字节切片读取器,跟踪当前游标位置
pub struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// 读取 N 个字节,推进游标
    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(err(format!(
                "read_bytes: need {n} bytes at pos {} but only {} remain",
                self.pos,
                self.remaining()
            )));
        }
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// 读取 u8
    #[inline]
    pub fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    /// 读取 i8
    #[inline]
    pub fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    /// 读取 u16 LE
    #[inline]
    pub fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    /// 读取 i16 LE
    #[inline]
    pub fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    /// 读取 u32 LE
    #[inline]
    pub fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// 读取 i32 LE
    #[inline]
    pub fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    /// 读取 u64 LE
    #[inline]
    pub fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// 读取 i64 LE
    #[inline]
    pub fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    /// 读取 f32 LE
    #[inline]
    pub fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    /// 读取 f64 LE
    #[inline]
    pub fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    /// 读取 bool(1 字节,0=false)
    #[inline]
    pub fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    /// 读取 GGUF string:u64 LE length + UTF-8 bytes
    pub fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| err(format!("invalid UTF-8 in gguf string: {e}")))
    }

    /// 跳过 N 字节
    pub fn skip(&mut self, n: usize) -> Result<()> {
        if self.pos + n > self.buf.len() {
            return Err(err(format!("skip: overflow at pos {}", self.pos)));
        }
        self.pos += n;
        Ok(())
    }

    /// 对齐到指定字节边界(默认 32)
    pub fn align_to(&mut self, alignment: u64) -> Result<()> {
        if alignment == 0 {
            return Ok(());
        }
        let mask = alignment - 1;
        let cur = self.pos as u64;
        let aligned = (cur + mask) & !mask;
        let skip = (aligned - cur) as usize;
        if skip > 0 {
            self.skip(skip)?;
        }
        Ok(())
    }
}

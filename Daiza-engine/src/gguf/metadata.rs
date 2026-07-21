//! GGUF metadata KV 表
//!
//! GGUF 把模型超参数存为一组 `key → typed value` 键值对。
//! 不同的架构(qwen2 / qwen35 / dspark / clip / ...)使用各自的 `xxx.*` 命名空间,
//! 引擎通过 `general.architecture` 选择对应的 schema。

use crate::gguf::err;
use crate::gguf::reader::ByteReader;
use crate::Result;

/// Metadata 值的枚举类型。对应 GGUF 的 `gguf_metadata_value_type`
#[derive(Debug, Clone)]
pub enum MetaValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    /// 数组:元素类型 + 元素值(均为同一标量枚举)
    Array(Vec<MetaValue>),
}

impl MetaValue {
    /// 从 reader 解析单个标量值(数组元素也走这条路径)
    fn read_scalar(reader: &mut ByteReader<'_>, type_tag: u32) -> Result<Self> {
        Ok(match type_tag {
            0 => Self::Uint8(reader.read_u8()?),
            1 => Self::Int8(reader.read_i8()?),
            2 => Self::Uint16(reader.read_u16()?),
            3 => Self::Int16(reader.read_i16()?),
            4 => Self::Uint32(reader.read_u32()?),
            5 => Self::Int32(reader.read_i32()?),
            6 => Self::Float32(reader.read_f32()?),
            7 => Self::Bool(reader.read_bool()?),
            8 => Self::String(reader.read_string()?),
            10 => Self::Uint64(reader.read_u64()?),
            11 => Self::Int64(reader.read_i64()?),
            12 => Self::Float64(reader.read_f64()?),
            other => return Err(err(format!("unknown scalar value type {other}"))),
        })
    }
}

/// 整个 metadata KV 表,key = metadata 字段名(如 `qwen35.block_count`)
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    pub kv: std::collections::BTreeMap<String, MetaValue>,
}

impl Metadata {
    /// 解析全部 `kv_count` 个 KV 对
    pub fn parse(reader: &mut ByteReader<'_>, kv_count: u64) -> Result<Self> {
        let mut kv = std::collections::BTreeMap::new();
        for _ in 0..kv_count {
            let key = reader.read_string()?;
            let type_tag = reader.read_u32()?;
            let value = Self::read_value(reader, type_tag)?;
            kv.insert(key, value);
        }
        Ok(Self { kv })
    }

    fn read_value(reader: &mut ByteReader<'_>, type_tag: u32) -> Result<MetaValue> {
        if type_tag == 9 {
            // ARRAY: 读元素类型 + 长度 + 元素
            let elem_tag = reader.read_u32()?;
            let len = reader.read_u64()? as usize;
            let mut elems = Vec::with_capacity(len);
            for _ in 0..len {
                elems.push(MetaValue::read_scalar(reader, elem_tag)?);
            }
            return Ok(MetaValue::Array(elems));
        }
        MetaValue::read_scalar(reader, type_tag)
    }

    // ===== 类型化访问器(只暴露实际需要用到的几种) =====

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.kv.get(key)? {
            MetaValue::Uint32(v) => Some(*v),
            MetaValue::Int32(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.kv.get(key)? {
            MetaValue::Uint64(v) => Some(*v),
            MetaValue::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn get_i32(&self, key: &str) -> Option<i32> {
        match self.kv.get(key)? {
            MetaValue::Int32(v) => Some(*v),
            MetaValue::Uint32(v) => Some(*v as i32),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.kv.get(key)? {
            MetaValue::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.kv.get(key)? {
            MetaValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.kv.get(key)? {
            MetaValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn get_i32_array(&self, key: &str) -> Option<Vec<i32>> {
        match self.kv.get(key)? {
            MetaValue::Array(elems) => elems
                .iter()
                .map(|v| match v {
                    MetaValue::Int32(x) => Ok(*x),
                    MetaValue::Uint32(x) => Ok(*x as i32),
                    _ => Err(()),
                })
                .collect::<std::result::Result<Vec<_>, _>>()
                .ok(),
            _ => None,
        }
    }

    pub fn get_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        match self.kv.get(key)? {
            MetaValue::Array(elems) => elems
                .iter()
                .map(|v| match v {
                    MetaValue::Float32(x) => Ok(*x),
                    _ => Err(()),
                })
                .collect::<std::result::Result<Vec<_>, _>>()
                .ok(),
            _ => None,
        }
    }

    /// 读取 bool 数组 (mmproj clip.vision.is_deepstack_layers 等)
    pub fn get_bool_array(&self, key: &str) -> Option<Vec<bool>> {
        match self.kv.get(key)? {
            MetaValue::Array(elems) => elems
                .iter()
                .map(|v| match v {
                    MetaValue::Bool(b) => Ok(*b),
                    _ => Err(()),
                })
                .collect::<std::result::Result<Vec<_>, _>>()
                .ok(),
            _ => None,
        }
    }

    /// 读取字符串数组 (mmproj clip.vision.patch_bias 等)
    pub fn get_string_array(&self, key: &str) -> Option<Vec<String>> {
        match self.kv.get(key)? {
            MetaValue::Array(elems) => elems
                .iter()
                .map(|v| match v {
                    MetaValue::String(s) => Ok(s.clone()),
                    _ => Err(()),
                })
                .collect::<std::result::Result<Vec<_>, _>>()
                .ok(),
            _ => None,
        }
    }
}

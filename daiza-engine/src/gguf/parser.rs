//! GGUF 文件顶层解析器
//!
//! 只读取头部(magic / metadata / tensor info),不加载权重数据。
//! 引擎在需要某张量时再按 `offset` 随机读取(GGUF 数据段顺序与头部张量顺序无关)。

use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::path::Path;

use crate::gguf::err;
use crate::gguf::metadata::Metadata;
use crate::gguf::reader::ByteReader;
use crate::gguf::tensor_info::{TensorInfo, GGUF_MAGIC, GGUF_VERSION};
use crate::Result;

/// 解析后的 GGUF 文件结构(头部信息 + 整个文件的字节视图)
pub struct GgufFile {
    pub version: u32,
    pub alignment: u64,
    pub metadata: Metadata,
    pub tensors: Vec<TensorInfo>,
    /// 张量数据段在文件中的绝对偏移(用于按 tensor.offset 随机读取)
    pub data_section_offset: u64,
    /// 整个文件的字节缓冲(mmap 的替代方案;3.9GB 一次性读入对 64GB+ 内存是可接受的,
    /// 真正部署时可换成 `memmap2`,但那违反"零依赖"。学习项目优先简单正确)
    pub bytes: Vec<u8>,
}

impl GgufFile {
    /// 打开并解析 GGUF 文件头部
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(&path).map_err(|e| {
            crate::BonsaiError::Io(format!("open {}: {e}", path.as_ref().display()))
        })?;

        // 读头部预览(前 256 KB 足够覆盖全部 metadata + tensor info)
        // 之后再决定是否读全文件
        let mut header_buf = Vec::with_capacity(256 * 1024);
        let n = file
            .read_to_end(&mut header_buf)
            .map_err(|e| crate::BonsaiError::Io(format!("read header: {e}")))?;

        // 判断是头部还是整文件
        let mut reader = ByteReader::new(&header_buf);
        let magic = reader.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(err(format!(
                "bad GGUF magic: 0x{:08X} (expected 0x{:08X})",
                magic, GGUF_MAGIC
            )));
        }
        let version = reader.read_u32()?;
        if version != GGUF_VERSION {
            return Err(err(format!(
                "unsupported GGUF version {version} (only {GGUF_VERSION} supported)"
            )));
        }
        let tensor_count = reader.read_u64()?;
        let kv_count = reader.read_u64()?;

        // 对齐默认 32;部分文件会在 metadata 中显式给出 `general.alignment`
        let alignment = 32u64;

        let metadata = Metadata::parse(&mut reader, kv_count)?;
        let tensors = TensorInfo::parse_all(&mut reader, tensor_count)?;
        let data_section_offset = reader.pos() as u64;
        // 对齐数据段起点
        let mask = alignment - 1;
        let aligned = (data_section_offset + mask) & !mask;

        // 如果 header_buf 包含整文件,直接复用;否则再读一次整文件
        // 简单路径:对于学习项目,直接读全文件
        let bytes = if n == file.metadata().map(|m| m.len() as usize).unwrap_or(0) {
            // 已经把整文件读进来了
            header_buf
        } else {
            // 重新读整个文件
            let _ = aligned; // 对齐后的偏移在 bytes 切片中同样适用
            let mut all = Vec::new();
            file.seek(std::io::SeekFrom::Start(0))
                .map_err(|e| crate::BonsaiError::Io(format!("seek: {e}")))?;
            file.read_to_end(&mut all)
                .map_err(|e| crate::BonsaiError::Io(format!("read full: {e}")))?;
            all
        };

        Ok(Self {
            version,
            alignment,
            metadata,
            tensors,
            data_section_offset: aligned,
            bytes,
        })
    }

    /// 按 tensor 名字查找
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// 取某个张量的数据切片(只读视图)
    pub fn tensor_data(&self, info: &TensorInfo) -> Result<&[u8]> {
        let start = self.data_section_offset as usize + info.offset as usize;
        let nbytes = crate::tensor::dtype::byte_size(info.dtype, info.n_elements());
        let end = start + nbytes;
        if end > self.bytes.len() {
            return Err(err(format!(
                "tensor '{}' data out of range: {}..{} > {}",
                info.name, start, end, self.bytes.len()
            )));
        }
        Ok(&self.bytes[start..end])
    }

    /// 按 dtype 统计张量数(用于核验解析正确性)
    pub fn dtype_histogram(&self) -> Vec<(String, usize)> {
        use std::collections::BTreeMap;
        use crate::gguf::tensor_info::TensorType;
        let mut h: BTreeMap<u32, usize> = BTreeMap::new();
        for t in &self.tensors {
            *h.entry(t.dtype.as_u32()).or_default() += 1;
        }
        h.into_iter()
            .map(|(k, v)| (TensorType::from_u32(k).name().to_string(), v))
            .collect()
    }
}

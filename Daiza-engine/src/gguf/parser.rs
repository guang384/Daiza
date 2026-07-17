//! GGUF 文件顶层解析器
//!
//! 只读取头部(magic / metadata / tensor info),不加载权重数据。
//! 引擎在需要某张量时再按 `offset` 从 mmap 区域切片访问(GGUF 数据段顺序与头部张量顺序无关)。
//!
//! 内存策略: 整个文件通过 memmap2 映射为只读, 内核按需 page-in,
//! 避免一次性 read_to_end 导致 ~3.9GB 内存峰值。tensor_data 返回
//! &[u8] 切片到 mmap 区域, 零拷贝。

use std::fs::File;
use std::path::Path;
use std::path::PathBuf;

use crate::gguf::err;
use crate::gguf::metadata::Metadata;
use crate::gguf::reader::ByteReader;
use crate::gguf::tensor_info::{TensorInfo, GGUF_MAGIC, GGUF_VERSION};
use crate::Result;

/// 解析后的 GGUF 文件结构(头部信息 + mmap 映射的文件字节视图)
///
/// mmap 由内核按需 page-in, 不实际占用物理内存直到访问。tensor_data
/// 返回切片到 mmap 区域的 &[u8], 零拷贝。
pub struct GgufFile {
    pub version: u32,
    pub alignment: u64,
    pub metadata: Metadata,
    pub tensors: Vec<TensorInfo>,
    /// 张量数据段在文件中的绝对偏移(用于按 tensor.offset 随机读取)
    pub data_section_offset: u64,
    /// 文件路径(保留用于错误信息)
    #[allow(dead_code)]
    pub path: PathBuf,
    /// mmap 映射的整个文件字节
    mmap: memmap2::Mmap,
}

impl GgufFile {
    /// 打开并解析 GGUF 文件( mmap 整个文件, 几乎 0 成本)
    #[allow(unsafe_code)]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let file = File::open(&path_buf).map_err(|e| {
            crate::BonsaiError::Io(format!("open {}: {e}", path_buf.display()))
        })?;

        // mmap 整个文件为只读 (内核按需 page-in, 不实际占物理内存)
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| crate::BonsaiError::Io(format!("mmap {}: {e}", path_buf.display())))?
        };
        let bytes = &mmap[..];

        let mut reader = ByteReader::new(bytes);
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

        Ok(Self {
            version,
            alignment,
            metadata,
            tensors,
            data_section_offset: aligned,
            path: path_buf,
            mmap,
        })
    }

    /// 按 tensor 名字查找
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// 取某个张量的数据切片(零拷贝, 直接切片到 mmap 区域)
    pub fn tensor_data(&self, info: &TensorInfo) -> Result<&[u8]> {
        let start = self.data_section_offset as usize + info.offset as usize;
        let nbytes = crate::tensor::dtype::byte_size(info.dtype, info.n_elements());
        let end = start + nbytes;
        let bytes = &self.mmap[..];
        if end > bytes.len() {
            return Err(err(format!(
                "tensor '{}' data out of range: {}..{} > {}",
                info.name, start, end, bytes.len()
            )));
        }
        Ok(&bytes[start..end])
    }
}

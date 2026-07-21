# Daiza: A Pure-CPU Rust Inference Engine for 1-bit Bonsai 27B

> **Daiza**(台座) cradles **Bonsai**(盆栽) — a from-scratch Rust engine running the [1-bit Bonsai 27B](https://huggingface.co/prism-ml/Bonsai-27B-gguf) model.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-1-green.svg)](#)

一个学习项目:从零实现 GGUF 解析、Q1_0 反量化、混合注意力(SSM + Full Attention)、GPT-2 BPE 分词,最终在纯 CPU 上完成 Bonsai 27B 的完整推理。仅依赖 `memmap2` 用于权重文件按需 page-in,其余全部从零实现。

---

## ✨ 特性

- **最小依赖**:仅依赖 `memmap2`(GGUF 权重按需 page-in)+ `image`(PNG/JPEG 解码),其余全部(GGUF 解析、FP16/BF16、BPE、GEMV、SSM、RoPE、AVX2 内核)从零实现
- **纯 CPU**:AVX2 + FMA 手写向量化内核(`target-cpu=native`)
- **Q1_0 反量化**:1.125 bits/weight 二值化格式,每 128 权重共享一个 FP16 scale
- **混合注意力架构**:64 层 = 48 SSM 块 + 16 全注意力块(节拍 `(i+1) % 4 == 0`)
- **M-RoPE**:多模态 RoPE,文本推理时仅旋转时间维(22/64 维)
- **Gated DeltaNet**:SSM 层使用 Gated Delta Rule 循环更新
- **Qwen3.6 chat 模板**:支持 `<|im_start|>` 格式与 `mind` 思考模式标记
- **DSpark 推测解码**:6 层 block-parallel drafter + Markov head + Leviathan rejection sampling,~5.5 tok/s
- **多线程并行**:持久线程池 (park/unpark 零分配),14 线程 GEMM 并行
- **Qwen3-VL 多模态**:CLIP ViT (27 层) + qwen3vl_merger 投影器,支持图像输入,text-only decode 零退化

## 📦 目录结构

```
Daiza/
├── .gitignore
├── README.md
├── Bonsai-27B-gguf/          # 模型权重(外部,不入库)
│   ├── Bonsai-27B-Q1_0.gguf       # 主权重(必需)
│   ├── Bonsai-27B-dspark-Q4_1.gguf  # DSpark 投机解码 drafter
│   ├── Bonsai-27B-mmproj-Q8_0.gguf  # 视觉塔(多模态)
│   └── bonsai-27b-whitepaper.pdf
└── Daiza-engine/             # Rust 推理引擎
    ├── Cargo.toml
    ├── .cargo/config.toml    # target-cpu=native
    └── src/
        ├── lib.rs            # 模块入口 + BonsaiError
        ├── main.rs           # CLI 入口(--dspark / --mmproj / --image)
        ├── engine.rs         # 顶层 Engine:加载→前向→采样→解码 + DSpark + Vision 调度
        ├── gguf/             # GGUF v3 二进制格式解析(parser/metadata/tensor_info)
        ├── tensor/           # 张量类型 + Q1_0/Q4_1/Q8_0/Iq1M/F32/F16/BF16 反量化 + AVX2 GEMM 内核
        ├── math/             # RMSNorm/LayerNorm/RoPE/Softmax/GELU/SIMD 超越函数/采样
        ├── model/            # Bonsai 27B 架构
        │   ├── config.rs / weights.rs / block.rs / attention.rs / ssm.rs / mlp.rs
        │   ├── forward.rs    # 单 token 前向 + prefill batch + vision embedding 注入
        │   ├── workspace.rs  # 持久线程池(park/unpark 零分配)+ Workspace 复用
        │   ├── dspark/       # DSpark 推测解码(drafter/markov/speculative/weights/config)
        │   └── vision/       # Qwen3-VL 多模态(config/weights/preprocess/rope/encoder/projector)
        ├── cache/            # KV cache(16 层)+ SSM state(48 层)
        └── tokenizer/        # GPT-2 BPE + Qwen35 预分词
```

## 📥 模型权重下载

> 权重文件较大(~6.3 GB 共三件),不入库,需手动下载。

### 前置条件

安装 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
pip install -U "huggingface_hub[cli]"
```

### 下载权重(三件套)

本引擎目标模型为 `prism-ml/Bonsai-27B-gguf`,需下载以下三个文件:

| 文件 | 大小 | 用途 | 状态 |
|------|------|------|------|
| `Bonsai-27B-Q1_0.gguf` | 3.9 GB | **主权重**(1.125 bits/weight,语言模型) | ✅ 已支持 |
| `Bonsai-27B-dspark-Q4_1.gguf` | 1.8 GB | DSpark 投机解码 drafter | ✅ 已支持 |
| `Bonsai-27B-mmproj-Q8_0.gguf` | 0.63 GB | 视觉塔(多模态输入) | ✅ 已支持 |

**一键下载全部**:

```powershell
# 在仓库根目录执行
hf download prism-ml/Bonsai-27B-gguf `
    Bonsai-27B-Q1_0.gguf `
    Bonsai-27B-dspark-Q4_1.gguf `
    Bonsai-27B-mmproj-Q8_0.gguf `
    --local-dir ./Bonsai-27B-gguf
```

或单独下载主权重:

```powershell
hf download prism-ml/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf --local-dir ./Bonsai-27B-gguf
```

### 可选:白皮书

```powershell
hf download PrismML-Eng/Bonsai-demo bonsai-27b-whitepaper.pdf --local-dir ./Bonsai-27B-gguf
```

### 镜像源(Hugging Face 不稳定时)

```powershell
# 国内镜像
$env:HF_ENDPOINT="https://hf-mirror.com"
hf download prism-ml/Bonsai-27B-gguf `
    Bonsai-27B-Q1_0.gguf `
    Bonsai-27B-dspark-Q4_1.gguf `
    Bonsai-27B-mmproj-Q8_0.gguf `
    --local-dir ./Bonsai-27B-gguf
```

### 验证下载

```powershell
# 检查文件大小
Get-ChildItem .\Bonsai-27B-gguf\*.gguf | Select-Object Name, @{N='Size(GB)';E={[math]::Round($_.Length/1GB,2)}}

# 使用引擎 inspect 模式验证 GGUF 完整性
cd Daiza-engine
.\target\release\daiza-cli.exe "..\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" --inspect
```

预期输出包含 `tensor_count : 851`、`block_count : 64`、`Q1_0 tensors: 498`。

## 🚀 快速开始

### 环境要求

- Rust 2021 edition(推荐 1.75+)
- 约 13 GB 可用内存(权重加载)
- x86_64 CPU(AVX2/AVX-512 加速)

### 构建与运行

```powershell
cd Daiza-engine

# Release 构建(推荐,启用 LTO + 自动向量化)
cargo build --release --bin daiza-cli

# 运行推理(chat 模式,默认 64 token)
.\target\release\daiza-cli.exe "..\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" "你好" 64

# Raw 模式(跳过 chat 模板,用于调试)
.\target\release\daiza-cli.exe "..\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" "The capital of China is" 8 --raw

# 启用 DSpark 投机解码(需先下载 drafter 权重)
.\target\release\daiza-cli.exe "..\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" "请用中文写一首关于春天的诗,8句" 200 `
    --dspark "..\Bonsai-27B-gguf\Bonsai-27B-dspark-Q4_1.gguf"

# Greedy 模式(temperature=0, 用于正确性验证)
.\target\release\daiza-cli.exe "..\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" "你好" 100 `
    --dspark "..\Bonsai-27B-gguf\Bonsai-27B-dspark-Q4_1.gguf" --greedy

# 多模态:加载 mmproj 视觉塔并对图像问答 (--image 可多次指定)
.\target\release\daiza-cli.exe "..\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" "描述这张图" 128 `
    --mmproj "..\Bonsai-27B-gguf\Bonsai-27B-mmproj-Q8_0.gguf" --image my_image.jpg

# 检查模型元信息
.\target\release\daiza-cli.exe "..\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" --inspect
```

### 使用示例

```rust
use daiza_engine::engine::Engine;
use daiza_engine::math::SamplingParams;

let mut engine = Engine::load("../Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf")?;
let params = SamplingParams {
    temperature: 0.7,
    top_k: 20,
    top_p: 0.95,
};
let output = engine.generate_with_params(
    "你好",
    256,
    params,
    Some("You are a helpful assistant."),
)?;
println!("{output}");
```

多模态推理示例:

```rust
use std::path::PathBuf;
use daiza_engine::engine::Engine;
use daiza_engine::math::SamplingParams;

let mut engine = Engine::load("../Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf")?;
engine.load_mmproj(std::path::Path::new("../Bonsai-27B-gguf/Bonsai-27B-mmproj-Q8_0.gguf"))?;

let output = engine.generate_with_image(
    "描述这张图",
    &[PathBuf::from("my_image.jpg")],
    128,
    SamplingParams::default(),
    Some("You are a helpful assistant."),
)?;
println!("{output}");
```

## 📐 架构关键点

### Q1_0 量化格式

每 128 权重 = 1 个 FP16 scale + 16 字节符号位 = 18 字节(1.125 bits/weight)。
反量化公式(白皮书 §4.2):

```
w_i = sg * b_i,  b_i ∈ {-1, +1}
```

其中 `sg` 是 FP16 组内 scale,`b_i` 由单个 bit 决定:`bit=0 → -scale`,`bit=1 → +scale`。
点积核使用 branchless FMA:`acc += scale * (2 * sum_pos_bits - 128)`。

### 混合注意力

| 层类型 | 数量 | 索引 |
|--------|------|------|
| SSM(Gated DeltaNet) | 48 | `i % 4 != 3` |
| Full Attention | 16 | `3, 7, 11, ..., 63` |

- **Full Attention**:24 query head × 4 KV head(GQA group=6),head_dim=256,使用 M-RoPE
- **SSM**:48 value head × 128 state size,Gated Delta Rule 循环更新,前置 Conv1d(kernel=4)

### SSM 层前向流程

```
x → Q/K/V 投影 → Conv1d(因果)→ SiLU → SSM 循环 → 门控 RMSNorm → 输出
```

Gated Delta Rule(autoregressive decode):
- `g = A * softplus(alpha + dt_bias)`,A = ssm_a(已存为 -exp(A_log))
- `decay = exp(g)` → `s *= decay` → `kv = S^T @ k` → `d = (v - kv) * beta` → `S += k ⊗ d` → `y = S^T @ q`

### Qwen3.5 Gated Attention

**关键**:`attn_q` 输出 12288 维按 head 交错排列:
```
[Q_head0(256) | gate_head0(256) | Q_head1(256) | gate_head1(256) | ...]
```
**不是** `[全部 Q(6144) | 全部 gate(6144)]`。此交错格式是 Bonsai 的核心实现细节。

### Chat 模板

```text
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
mind
```

模型进入思考模式,输出 `mind ... </mind>` 包裹的思考内容,然后给出最终回复。EOS token(`<|im_end|>`, id=248046)终止生成。

### Qwen3-VL 多模态管线

参考 [llama.cpp qwen3vl.cpp](https://github.com/PrismML-Eng/llama.cpp) 实现,加载时一次性把 Q8_0/F16 反量化为 F32 (~1.6GB),运行时纯 F32 路径。

```
image → resize(768×768,Lanczos3) → normalize → patchify(16×16, 2304 patches × 768 dim)
       ↓
门控 patch embedding: Conv2D(W) + Conv2D(W.1) + bias → [2304, 1152]
       ↓
+ learned position_embd [1152, 2304]
       ↓
27 层 ViT block:
  LN1(bias) → QKV proj (fused 3456) → M-RoPE(Q,K) → bidirectional attn → out_proj → +residual
  → LN2(bias) → ffn_up(1152→4304) → GELU → ffn_down(4304→1152) → +residual
       ↓
post_ln → [2304, 1152]
       ↓
qwen3vl_merger 投影器:
  spatial_merge(2×2 块合并, 2304→576 patches, reshape 到 [4608, 576])
  → Linear(4608→4608) → GELU → Linear(4608→5120)
       ↓
[576, 5120] vision embeddings (与 text model hidden_dim=5120 一致)
```

**关键细节**:
- M-RoPE sections `[head_dim/4]×4 = [18,18,18,18]`,位置 IDs 为 4D `(t=0, h=py, w=px, extra=0)`,只应用在 Q/K
- 门控 patch embedding:两个相同形状 `[16,16,3,1152]` 的 Conv2D 权重相加
- attention bidirectional(无 causal mask),所有 patch 互相可见
- LayerNorm 带 bias(ViT 风格,非 RMSNorm),GELU 用 tanh 近似
- **vision 注入只在 prefill 阶段**,text-only decode 走 `forward_single_token` 零退化

**性能** (768×768 测试图,Intel Core Ultra 5 225H):

| 阶段 | 时间 | 说明 |
|------|------|------|
| preprocess | ~9ms | resize + patchify |
| ViT encode (27 层) | ~7.8s | AVX2 batched matmul + 8-row dot product kernel + AVX2 attention (hoist q_i) |
| projector | ~0.13s | batched matmul + AVX2 |
| prefill (600 expanded tokens) | ~100s | ~165ms/tok,vision 分批 batched 注入 (MAX_VISION_BATCH=64) |
| decode (32 tokens) | ~5.7s | 178ms/tok,与 text-only 一致 |

ViT encode 优化路径: 173s (标量) → 102s (线程池并行) → 60.2s (2D tiled batched matmul) → 14.8s (AVX2 8-row dot product kernel) → 10.2s (AVX2 attention kernel) → 7.92s (proj batched). 总加速 **22×**.

Vision prefill 优化路径: 132s (逐 token 注入,576 × forward_single_token) → 100s (分批 batched 注入,9 × forward_batch_with_vision, MAX_VISION_BATCH=64). 每批读 13GB 权重 1 次 vs 逐 token 读 576 次. 总加速 **1.32×**,text-only decode 零退化.

text-only 推理在加载 mmproj 后零退化(5 轮交错 benchmark 验证)。

## ⚙️ 生成参数(白皮书建议)

| 参数 | 建议值 |
|------|--------|
| temperature | 0.7 |
| top_k | 20 |
| top_p | 0.95 |

这些设置用于 Bonsai 27B 所有 benchmark 结果(thinking mode)。

## 🧪 已验证输出

| 模式 | 输入 | 输出 | 备注 |
|------|------|------|------|
| Raw | `The capital of China is` | ` Beijing` | ✅ top-1 logit 12.58 |
| Raw | `The capital of France is` (16t) | ` the capital of France is Paris. The capital of France is Paris...` | ✅ 答案正确,但有重复倾向 |
| Raw | `1, 2, 3, 4,` (16t) | ` 5, 6, 7, 8, 9,` | ✅ 序列补全完美 |
| Raw | `1+1=` (greedy 50t) | `2, 1+2=3, 1+3=4, 1+4=5, 1+5=6, 1+6=7, 1+7=8, 1+8=9` | ✅ 正确答案 + 加法序列补全 |
| Raw | `2+3=` (16t) | `5` | ✅ 正确 |
| Raw | `10+20=` (16t) | `30` | ✅ 正确 |
| Raw | `12+7=` / `5*6=` / `100-23=` (30t) | `19` / `30` / `77` | ✅ 全部正确 |
| Raw | `What is 2 plus 2?` (16t) | (空白) | ❌ 模型为 chat 模式训练, raw 模式缺思考标记无法回答 |
| Chat | `你好` (32 tok) | `Here's a thinking process: 1. **Analyze the user's input:** User says: "你好" (Hello)` | ✅ 正确进入思考 |
| Chat (greedy) | `1+1等于几` / `池塘鱼` | 正确进入 thinking 步骤, 完成 arithmetic/interpretation 分析 | ✅ thinking 推理正常 |

### ✅ 与官方 llama.cpp (PrismML-Eng 分支) 对比 — Daiza 实现 bug 已定位并修复

相同 prompt `1+1=`, 相同 Q1_0 模型, 官方推荐参数 T=0.7/top_k=20/top_p=0.95:

| 实现 | 输出 | 结果 |
|------|------|------|
| **官方 llama.cpp** (prism 分支, build b1-79697f2) | `[Start thinking] Here's a thinking process: 1. Analyze User Input... 6. Self-Correction: "2" is perfect.✅ [End thinking] 2` | ✅ 完整 6 步 thinking + 正确答案 `2` |
| **Daiza** (修复前) | `Here's a thinking process: 1. **Analyze the user's input:** * The user's input is just "1+1=1=1=1=1=1=1=...` | ❌ thinking 步骤 1 重复 collapse |
| **Daiza** (修复后) | `2, 1+2=3, 1+3=4, ...` (raw greedy) / thinking 推理正常 (chat) | ✅ 正确 |

**根因**: SSM (Gated DeltaNet) 块的 GQA v_head→k_head 映射错误。Bonsai-27B GGUF 由 llama.cpp prism 分支转换工具生成, V heads 经 `conversion/qwen.py` 的 `_LinearAttentionVReorderBase` 重排为 **tiled 布局** (k_head i 对应 v_head `[i, i+num_k_heads, i+2*num_k_heads]`)。原实现误用 **div 映射** (`kh = vh / 3`, grouped 布局), 与 GGUF 实际布局不匹配, 导致 SSM scan 中 q/k 与错误的 v_head 配对, 累积后输出 collapse。

**修复**: 改为 **mod 映射** (`kh = vh % num_k_heads`), 与 llama.cpp `ggml_repeat` 一致。修改 3 处: `ssm.rs` (decode path) + `forward.rs` (batch path 串行 + 并行)。commit `234c956`。

## 📊 性能参考(纯 CPU,单 token decode)

测试硬件:Intel Core Ultra 5 225H (Meteor Lake, 14 核, AVX2 + FMA, LPDDR5X-7467)

| 模式 | 吞吐量 | 接受率 | 说明 |
|------|--------|--------|------|
| 纯 target (无 DSpark) | ~4.4 tok/s | — | 64 层前向 ~227ms/tok |
| DSpark 推测解码 | ~5.5 tok/s | ~36% | drafter 53ms/call + target verify,~1.25× 加速 |

- 单 block 前向:~220ms(SSM 层)/ ~530ms(全注意力层)
- 完整 64 层前向:~14s(纯 SSM 层)/ ~34s(包含全注意力)
- 内存占用:~13 GB(权重)+ ~1.3 GB(KV/SSM/激活)
- DSpark 加速比:target forward 197ms/tok → DSpark 179ms/tok (~10% 加速)

**说明**:这是学习项目。当前性能已接近 LPDDR5X 单通道带宽极限 (~22 GB/s 实测 vs 60 GB/s 理论),
MLP 层占 58% 时间已饱和。商业部署请使用 [llama.cpp PrismML fork](https://github.com/PrismML-Eng/llama.cpp)。

## 🗺️ 路线图

- [x] Q1_0 主权重文本推理
- [x] 混合注意力(SSM + Full Attention)
- [x] Chat 模板与思考模式
- [x] 多线程并行(持久线程池 + AVX2 手写内核)
- [x] DSpark 投机解码(`Bonsai-27B-dspark-Q4_1.gguf`)
- [x] 多模态视觉输入(`Bonsai-27B-mmproj-Q8_0.gguf`)
- [x] ViT encoder AVX2 向量化 + 线程池并行(173s/图 → 7.9s/图,22× 加速)
- [x] Vision prefill batched(逐 token 注入 → 分批 64 个,132s → 100s,text-only 零退化)
- [ ] KV cache 量化(4-bit)

## 📚 参考资料

- [1-bit Bonsai 27B 白皮书](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf)
- [Bonsai-27B-gguf Hugging Face 仓库](https://huggingface.co/prism-ml/Bonsai-27B-gguf)
- [GGUF 格式规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Qwen3 模型架构](https://github.com/QwenLM/Qwen3)
- [llama.cpp PrismML fork (qwen3vl 参考)](https://github.com/PrismML-Eng/llama.cpp)
- [Qwen3-VL 多模态架构](https://github.com/QwenLM/Qwen3-VL)

## 📄 许可证

Apache-2.0(与上游 Bonsai 27B 模型一致)

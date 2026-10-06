**Languages:** [中文](README.zh.md) | [English](README.md)

# Daiza: A Pure-CPU Rust Inference Engine for 1-bit Bonsai 27B

> **Daiza** (台座, "pedestal") cradles **Bonsai** (盆栽, "bonsai") — a from-scratch Rust engine running the [1-bit Bonsai 27B](https://huggingface.co/prism-ml/Bonsai-27B-gguf) model.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-2-green.svg)](#)

A learning project: from-scratch implementation of GGUF parsing, Q1_0 dequantization, hybrid attention (SSM + Full Attention), and GPT-2 BPE tokenization, completing full inference of Bonsai 27B on pure CPU.

---

## ✨ Features

- **Minimal dependencies**: only `memmap2` (on-demand page-in of GGUF weights) + `image` (PNG/JPEG decoding). Everything else (GGUF parsing, FP16/BF16, BPE, GEMV, SSM, RoPE, AVX2 kernels) is implemented from scratch.
- **Pure CPU**: hand-written AVX2 + FMA vectorized kernels (`target-cpu=native`).
- **Q1_0 dequantization**: 1.125 bits/weight binarized format, one FP16 scale shared per 128 weights.
- **Hybrid attention architecture**: 64 layers = 48 SSM blocks + 16 full-attention blocks (beat `(i+1) % 4 == 0`).
- **M-RoPE**: multimodal RoPE; for text inference, only the time dimension (22/64 dims) is rotated.
- **Gated DeltaNet**: SSM layers use Gated Delta Rule recurrent updates.
- **Qwen3.6 chat template**: supports `<|im_start|>` format and `mind` thinking-mode marker.
- **DSpark speculative decoding**: 6-layer block-parallel drafter + Markov head + Leviathan rejection sampling; the 2-gram PLD lookup drafter replaces the neural drafter at zero cost, decode on par with the pure target (~7 tok/s).
- **Multi-threaded parallelism**: persistent thread pool (park/unpark zero-alloc), 14-thread GEMM parallelism.
- **Qwen3-VL multimodal**: CLIP ViT (27 layers) + qwen3vl_merger projector, supports image input, zero-degradation text-only decode.

## 🖼️ UI Preview

<img src="docs/screenshots/ScreenShot_2026-07-27_231037_345.png" width="720" alt="Daiza main UI" />

<img src="docs/screenshots/ScreenShot_2026-07-27_231235_359.png" width="720" alt="Chat demo" />

### Streaming chat demo

![Streaming chat](docs/screenshots/chating.gif)

### 🎵 Bonus: From lyrics to song

During the GIF demo above, the model improvised a set of lyrics. We casually fed those lyrics into an AI music tool, which composed and sang the following song:

<audio controls src="docs/echoes_of_us.mp3">
  Your browser does not support the audio element. You can download <a href="docs/echoes_of_us.mp3">echoes_of_us.mp3</a> directly.
</audio>

## 📦 Directory Structure

```
Daiza/
├── .cargo/config.toml       # target-cpu=native + rsproxy mirror
├── Cargo.toml               # workspace root (lto="fat")
├── README.md
├── Bonsai-27B-gguf/         # model weights (external, not in repo)
└── sota-baseline/           # benchmark scripts (bench/compare/create-baseline)
```

### Crate dependency layers

```
                    ┌─────────────────┐
                    │  Daiza-engine   │  Layer 0  Inference core (lib, zero internal deps)
                    │  gguf/tensor/   │
                    │  math/model/    │
                    │  cache          │
                    └────────┬────────┘
                             │ Cargo dependency
                    ┌────────▼────────┐
                    │  Daiza-runtime  │  Layer 1  Orchestration (lib, depends on engine)
                    │  engine/session │
                    │  tokenizer/     │
                    │  tool_call      │
                    └────────┬────────┘
                             │ Cargo dependency
                   ┌─────────┴─────────┐
                   │                   │
            ┌──────▼──────┐     ┌──────▼──────┐
            │  Daiza-cli  │     │  Daiza-web  │  Layer 2  Inference binary entry points
            │  (CLI bin)  │     │  (HTTP bin) │
            └─────────────┘     └──────▲──────┘
                                       │ Runtime spawns daiza-web.exe subprocess
                                ┌──────┴──────┐
                                │  Daiza-app  │  Layer 3  Tauri desktop shell
                                │  (Tauri bin)│  (only depends on tauri/ureq/serde/base64,
                                └─────────────┘   not on engine/runtime)
```

- **Daiza-engine** (Layer 0): Inference core, zero internal dependencies. GGUF parsing, Q1_0 dequantization, AVX2 GEMM kernels, SSM/Attention/MLP forward, thread pool.
- **Daiza-runtime** (Layer 1): Orchestration layer, depends on engine. Top-level Engine wrapper, session management, GPT-2 BPE tokenizer, tool calling, SSD persistence.
- **Daiza-cli / Daiza-web** (Layer 2): Inference binary entry points, Cargo depends on engine+runtime.
  - cli: command-line interaction
  - web: HTTP+SSE chat service, can run standalone (`daiza-web --model xxx.gguf`, browse to 127.0.0.1:8787)
- **Daiza-app** (Layer 3): Tauri desktop shell, **does not depend on engine/runtime**, only on tauri/ureq/serde/base64. At runtime it spawns `daiza-web.exe` as a subprocess to provide inference, while itself handling window management + SSE forwarding (bypassing WebView2's mixed-content blocking of 127.0.0.1).

> `lto = "fat"` + `codegen-units = 1` merges all crate IR into a single compilation unit, ensuring cross-crate inlining is equivalent to same-crate inlining (hot-path runtime → engine forward/matvec/AVX2 kernel calls can be inlined).

### Daiza-engine module layout

```
Daiza-engine/src/
├── lib.rs            # module entry + BonsaiError
├── gguf/             # GGUF v3 binary format parsing (parser/metadata/tensor_info)
├── tensor/           # tensor types + Q1_0/Q4_1/Q8_0/Iq1M/F32/F16/BF16 dequant + AVX2 GEMM kernels
├── math/             # RMSNorm/LayerNorm/RoPE/Softmax/GELU/SIMD transcendental functions/sampling
├── model/            # Bonsai 27B architecture
│   ├── config.rs / weights.rs / block.rs / attention.rs / ssm.rs / mlp.rs
│   ├── forward.rs    # single-token forward + prefill batch + vision embedding injection
│   ├── workspace.rs  # persistent thread pool (park/unpark zero-alloc) + Workspace reuse
│   ├── dspark/       # DSpark speculative decoding (drafter/markov/speculative/weights/config)
│   └── vision/       # Qwen3-VL multimodal (config/weights/preprocess/rope/encoder/projector)
└── cache/            # KV cache (16 layers) + SSM state (48 layers)
```

### Daiza-runtime module layout

```
Daiza-runtime/src/
├── lib.rs            # re-export engine::Engine + daiza_engine::{BonsaiError, Result}
├── engine.rs         # top-level Engine: load → forward → sample → decode + DSpark + Vision dispatch
├── session.rs        # multi-turn session state
├── session_persist.rs# session SSD persistence (.dzss)
├── session_manager.rs# multi-session management
├── tool_call.rs      # tool-call parsing
└── tokenizer/        # GPT-2 BPE + Qwen35 pre-tokenizer
```

## 📥 Model Weight Download

> Weights are large (~6.3 GB total, three files), not included in the repo, and must be downloaded manually.

### Prerequisites

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
pip install -U "huggingface_hub[cli]"
```

### Download weights (three-piece set)

The target model for this engine is `prism-ml/Bonsai-27B-gguf`. The following three files are required:

| File | Size | Purpose | Status |
|------|------|---------|--------|
| `Bonsai-27B-Q1_0.gguf` | 3.9 GB | **Main weights** (1.125 bits/weight, language model) | ✅ Supported |
| `Bonsai-27B-dspark-Q4_1.gguf` | 1.8 GB | DSpark speculative-decoding drafter | ✅ Supported |
| `Bonsai-27B-mmproj-Q8_0.gguf` | 0.63 GB | Vision tower (multimodal input) | ✅ Supported |

**One-click download of all files**:

```powershell
# Run from the repo root
hf download prism-ml/Bonsai-27B-gguf `
    Bonsai-27B-Q1_0.gguf `
    Bonsai-27B-dspark-Q4_1.gguf `
    Bonsai-27B-mmproj-Q8_0.gguf `
    --local-dir ./Bonsai-27B-gguf
```

Or download only the main weights:

```powershell
hf download prism-ml/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf --local-dir ./Bonsai-27B-gguf
```

### Optional: whitepaper

```powershell
hf download PrismML-Eng/Bonsai-demo bonsai-27b-whitepaper.pdf --local-dir ./Bonsai-27B-gguf
```

### Mirror source (when Hugging Face is unstable)

```powershell
# China mirror
$env:HF_ENDPOINT="https://hf-mirror.com"
hf download prism-ml/Bonsai-27B-gguf `
    Bonsai-27B-Q1_0.gguf `
    Bonsai-27B-dspark-Q4_1.gguf `
    Bonsai-27B-mmproj-Q8_0.gguf `
    --local-dir ./Bonsai-27B-gguf
```

### Verify the download

```powershell
# Check file sizes
Get-ChildItem .\Bonsai-27B-gguf\*.gguf | Select-Object Name, @{N='Size(GB)';E={[math]::Round($_.Length/1GB,2)}}

# Verify GGUF integrity via engine inspect mode
.\target\release\daiza-cli.exe --model ".\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" --inspect
```

Expected output includes `tensor_count : 851`, `block_count : 64`, `Q1_0 tensors: 498`.

## 🚀 Quick Start

### Requirements

- Rust 2021 edition (1.75+ recommended)
- ~13 GB available memory (for weight loading)
- x86_64 CPU (AVX2/AVX-512 acceleration)

### Build & Run

```powershell
# Release build (recommended, run from workspace root, enables fat LTO + cross-crate inlining)
cargo build --release --bin daiza-cli

# Run inference (chat mode, default 64 tokens)
.\target\release\daiza-cli.exe --model ".\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" --prompt "Hello" --max-tokens 64

# Raw mode (skips chat template, for debugging)
.\target\release\daiza-cli.exe --model ".\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" --prompt "The capital of China is" --max-tokens 8 --raw

# Enable DSpark speculative decoding (requires drafter weights)
.\target\release\daiza-cli.exe --model ".\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" `
    --prompt "Write a poem about spring, 8 lines" --max-tokens 200 `
    --dspark ".\Bonsai-27B-gguf\Bonsai-27B-dspark-Q4_1.gguf"

# Greedy mode (temperature=0, for correctness verification)
.\target\release\daiza-cli.exe --model ".\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" `
    --prompt "Hello" --max-tokens 100 `
    --dspark ".\Bonsai-27B-gguf\Bonsai-27B-dspark-Q4_1.gguf" --greedy

# Multimodal: load mmproj vision tower and answer questions about an image (--image may be repeated)
.\target\release\daiza-cli.exe --model ".\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" `
    --prompt "Describe this image" --max-tokens 128 `
    --mmproj ".\Bonsai-27B-gguf\Bonsai-27B-mmproj-Q8_0.gguf" --image my_image.jpg

# Inspect model metadata
.\target\release\daiza-cli.exe --model ".\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf" --inspect

# Show help
.\target\release\daiza-cli.exe --help
```

### Usage examples

```rust
use daiza_runtime::Engine;
use daiza_engine::math::SamplingParams;

let mut engine = Engine::load("../Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf")?;
let params = SamplingParams {
    temperature: 0.7,
    top_k: 20,
    top_p: 0.95,
    repetition_penalty: 1.3,
    frequency_penalty: 0.4,
};
let output = engine.generate_with_params(
    "Hello",
    256,
    params,
    Some("You are a helpful assistant."),
)?;
println!("{output}");
```

Multimodal inference example:

```rust
use std::path::PathBuf;
use daiza_runtime::Engine;
use daiza_engine::math::SamplingParams;

let mut engine = Engine::load("../Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf")?;
engine.load_mmproj(std::path::Path::new("../Bonsai-27B-gguf/Bonsai-27B-mmproj-Q8_0.gguf"))?;

let output = engine.generate_with_image(
    "Describe this image",
    &[PathBuf::from("my_image.jpg")],
    128,
    SamplingParams::default(),
    Some("You are a helpful assistant."),
)?;
println!("{output}");
```

DSpark speculative decoding example:

```rust
use daiza_runtime::Engine;
use daiza_engine::math::SamplingParams;

let mut engine = Engine::load("../Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf")?;
// Load the DSpark drafter (6-layer block-parallel drafter + Markov head)
engine.load_drafter(std::path::Path::new("../Bonsai-27B-gguf/Bonsai-27B-dspark-Q4_1.gguf"))?;

let output = engine.generate_with_params(
    "Write a poem about spring, 8 lines",
    200,
    SamplingParams::default(),
    Some("You are a helpful assistant."),
)?;
println!("{output}");
```

## 📐 Architecture Key Points

### Q1_0 quantization format

Every 128 weights = 1 FP16 scale + 16 bytes of sign bits = 18 bytes (1.125 bits/weight).
Dequantization formula (whitepaper §4.2):

```
w_i = sg * b_i,  b_i ∈ {-1, +1}
```

where `sg` is the FP16 group scale and `b_i` is determined by a single bit: `bit=0 → -scale`, `bit=1 → +scale`.
The dot-product kernel uses branchless FMA: `acc += scale * (2 * sum_pos_bits - 128)`.

### Hybrid attention

| Layer type | Count | Indices |
|------------|-------|---------|
| SSM (Gated DeltaNet) | 48 | `i % 4 != 3` |
| Full Attention | 16 | `3, 7, 11, ..., 63` |

- **Full Attention**: 24 query heads × 4 KV heads (GQA group=6), head_dim=256, uses M-RoPE.
- **SSM**: 48 value heads × 128 state size, Gated Delta Rule recurrent update, preceded by Conv1d (kernel=4).

### SSM layer forward flow

```
x → Q/K/V projection → Conv1d (causal) → SiLU → SSM recurrence → gated RMSNorm → output
```

Gated Delta Rule (autoregressive decode):
- `g = A * softplus(alpha + dt_bias)`, where A = ssm_a (stored as -exp(A_log))
- `decay = exp(g)` → `s *= decay` → `kv = S^T @ k` → `d = (v - kv) * beta` → `S += k ⊗ d` → `y = S^T @ q`

### Qwen3.5 Gated Attention

**Key point**: `attn_q` outputs 12288 dims arranged interleaved by head:
```
[Q_head0(256) | gate_head0(256) | Q_head1(256) | gate_head1(256) | ...]
```
**Not** `[all Q(6144) | all gate(6144)]`. This interleaved layout is a core implementation detail of Bonsai.

### Chat template

```text
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
mind
```

The model enters thinking mode, outputs `mind ... </mind>` wrapped thinking content, then gives the final reply. The EOS token (`<|im_end|>`, id=248046) terminates generation.

### Qwen3-VL multimodal pipeline

Implemented with reference to [llama.cpp qwen3vl.cpp](https://github.com/PrismML-Eng/llama.cpp). At load time, Q8_0/F16 are dequantized to F32 once (~1.6 GB); at runtime it is a pure F32 path.

```
image → resize(768×768, Lanczos3) → normalize → patchify(16×16, 2304 patches × 768 dim)
       ↓
gated patch embedding: Conv2D(W) + Conv2D(W.1) + bias → [2304, 1152]
       ↓
+ learned position_embd [1152, 2304]
       ↓
27 ViT blocks:
  LN1(bias) → QKV proj (fused 3456) → M-RoPE(Q,K) → bidirectional attn → out_proj → +residual
  → LN2(bias) → ffn_up(1152→4304) → GELU → ffn_down(4304→1152) → +residual
       ↓
post_ln → [2304, 1152]
       ↓
qwen3vl_merger projector:
  spatial_merge (2×2 block merge, 2304→576 patches, reshape to [4608, 576])
  → Linear(4608→4608) → GELU → Linear(4608→5120)
       ↓
[576, 5120] vision embeddings (matches text model hidden_dim=5120)
```

**Key details**:
- M-RoPE sections `[head_dim/4]×4 = [18,18,18,18]`, position IDs are 4D `(t=0, h=py, w=px, extra=0)`, applied only to Q/K.
- Gated patch embedding: two Conv2D weights of the same shape `[16,16,3,1152]` are added together.
- Attention is bidirectional (no causal mask); all patches are mutually visible.
- LayerNorm has bias (ViT style, not RMSNorm); GELU uses tanh approximation.
- **Vision injection happens only at the prefill stage**; text-only decode goes through `forward_single_token` with zero degradation.

**Performance** (768×768 test image, Intel Core Ultra 5 225H):

| Stage | Time | Notes |
|-------|------|-------|
| preprocess | ~9ms | resize + patchify |
| ViT encode (27 layers) | ~8.2s | AVX2 batched matmul + 8-row dot-product kernel + AVX2 attention (hoist q_i) |
| projector | ~0.19s | batched matmul + AVX2 |
| prefill (602 expanded tokens) | ~103s | ~165ms/tok, vision batched injection (MAX_VISION_BATCH=64) |
| decode (48 tokens) | ~8.3s | 173ms/tok, identical to text-only |

ViT encode optimized from scalar 173s to 8.2s (**22× speedup**): thread-pool parallelism → 2D tiled batched matmul → AVX2 8-row dot-product kernel → AVX2 attention kernel → projector batched.

Vision prefill changed from per-token injection (132s) to batched injection (MAX_VISION_BATCH=64, 100s), with zero text-only decode degradation.

## ⚙️ Generation Parameters (whitepaper recommendations)

| Parameter | Recommended value |
|-----------|-------------------|
| temperature | 0.7 |
| top_k | 20 |
| top_p | 0.95 |

These settings are used for all Bonsai 27B benchmark results (thinking mode).

## 🏎️ Performance Tuning

Performance knobs are configured via environment variables (process-level OnceLock cache, read once at startup). When switching hardware/thermal conditions, use the built-in auto-calibration:

```powershell
# Interleaved subprocess sweep + confirmation gate (≥5% advantage required to override the default), takes ~8-15 minutes
.\target\release\daiza-cli.exe --model <gguf_path> --calibrate
```

Core knobs:

| Env variable | Purpose | Default | Measured on 225H |
|--------------|---------|---------|------------------|
| `DAIZA_GEMM_T_SUB` | Prefill GEMM t-subdivision (×-slice granularity: balance between L2 residency and prep re-computation) | 32 | 32 optimal (64/128 degrade 2%/9% in isolated bench, on par E2E) |
| `DAIZA_ACTIVE_WORKERS` | Active worker count for sustained decode (thermal balance: exceeding the thermal budget backfires via throttling) | min(9, all cores) | 9 (13 threads: 4P+8E+2LP-E) |
| `DAIZA_MATVEC_CHUNK` | Decode matvec work-stealing granularity | 128 | 128 is ~15% faster than 256 (shorter E-core straggler tail) |

> On thermally coupled devices (laptops), slot noise in short benchmark runs can reach ±5%~50%.
> For manual tuning use interleaved A/B + min, or simply trust the `--calibrate` confirmation gate.

## 📊 Performance Reference (pure CPU)

Test hardware: Intel Core Ultra 5 225H (Meteor Lake, 14 cores, AVX2 + FMA, LPDDR5X-7467).
Test conditions: greedy sampling, 142-token prompt.

| Stage | Performance | Notes |
|-------|-------------|-------|
| Prefill (142 tokens) | ~10.0s (~71ms/token) | ~95% of the FMA roofline (~715 GFLOPS measured vs 755 peak) |
| Decode | ~143ms/token (7.0 tok/s) | Q1_0 LUT kernel saturated at FMA ports; V0-V6 variant space exhausted |

- Decode breakdown (DAIZA_PROFILE, per token): attention-forward ~11.5ms + SSM-forward ~38.5ms + MLP ~83.5ms (64 blocks, ~60%) + post_norm ~0.2ms + lm_head ~5.6ms ≈ 143ms/token; the MLP is dominated by Q1_0 LUT compute and is saturated at FMA ports after the V0-V6 kernel-variant search.
- Measured memory bandwidth ~120 GB/s; decode is compute-bound (bandwidth floor ~29ms vs 143ms measured) — the bottleneck is FMA throughput of the LUT lookups, not bandwidth.
- Memory footprint: ~13 GB (Q1_0 weights) + ~1.3 GB (KV/SSM/activations) + ~1.6 GB (mmproj, optional).
- Prefill optimization campaign (17 perf commits): GEMM dispatch fixes + f16 scratch + L2 layout tuning + serial-segment parallelization (attention online-softmax / SSM scan / swiglu across tokens) + heterogeneous work-stealing, 16.4s → 10.0s (-39%); all optimizations byte-identical under greedy, zero quality loss.
- DSpark: the n-gram PLD drafter eliminates the 53ms neural-drafter overhead, decode on par with the pure target; batched verify was rejected by data (no amortization headroom with a compute-bound Q1_0 LUT decoder; efficiency 0.62× at acceptance rate p=0.08).
- Vision: zero degradation for text-only with mmproj loaded; with-image previously measured ~173ms/token decode, one-time cost (ViT encoding ~8s + vision prefill ~100s) — pre-campaign measurements, for reference only.

**Note**: This is a learning project. Under the pure-CPU + Q1_0 + byte-identical constraints, the current implementation has reached this machine's limits (prefill at ~95% of the FMA roofline, decode at LUT-kernel FMA-port saturation); further breakthroughs require AVX-512/AMX (not available on this machine). For commercial deployment please use the [llama.cpp PrismML fork](https://github.com/PrismML-Eng/llama.cpp).

## 🗺️ Roadmap

- [x] Q1_0 main-weight text inference
- [x] Hybrid attention (SSM + Full Attention)
- [x] Chat template and thinking mode
- [x] Multi-threaded parallelism (persistent thread pool + hand-written AVX2 kernels)
- [x] DSpark speculative decoding (`Bonsai-27B-dspark-Q4_1.gguf`)
- [x] Multimodal vision input (`Bonsai-27B-mmproj-Q8_0.gguf`)
- [x] ViT encoder AVX2 vectorization + thread-pool parallelism (173s/image → 7.9s/image, 22× speedup)
- [x] Vision prefill batched (per-token injection → batches of 64, 132s → 100s, zero text-only degradation)
- [x] Prefill GEMM optimization campaign (17 perf commits, 16.4s → 10.0s, -39%, byte-identical): GEMM dispatch fixes, f16 scratch, L2 layout tuning, serial-segment parallelization across tokens, P/E/LP-E work-stealing
- [x] `--calibrate` auto-calibration subcommand (interleaved sweep + confirmation gate, one-shot recommended env output)

> The roadmap is complete. Under the pure-CPU + Q1_0 + byte-identical constraints the implementation has reached this machine's limits: prefill at ~95% of the FMA roofline, decode at LUT-kernel FMA-port saturation (further breakthroughs require AVX-512/AMX). After investigation, KV cache 4-bit quantization was deemed not worth implementing (KV cache reads account for <0.01% of bandwidth; 4-bit quantization yields <1% and would violate the "must not reduce model accuracy" hard constraint).

## 📚 References

- [1-bit Bonsai 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf)
- [Bonsai-27B-gguf Hugging Face repo](https://huggingface.co/prism-ml/Bonsai-27B-gguf)
- [GGUF format specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Qwen3 model architecture](https://github.com/QwenLM/Qwen3)
- [llama.cpp PrismML fork (qwen3vl reference)](https://github.com/PrismML-Eng/llama.cpp)
- [Qwen3-VL multimodal architecture](https://github.com/QwenLM/Qwen3-VL)

## 📄 License

Apache-2.0 (consistent with the upstream Bonsai 27B model).

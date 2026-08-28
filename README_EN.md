# C.C.C.P.

<p align="center">
  <img src="assets/cccp-icon.png" alt="C.C.C.P. project icon" width="96">
</p>

<p align="center">
  <a href="README.md">简体中文</a> · <strong>English</strong> · <a href="README_RU.md">Русский</a>
</p>

## ⬇️ Download the complete Windows offline package (v0.9.15)

> [!IMPORTANT]
> **First-time users must download the complete offline package, not only `CCCP-Launcher.exe`.** Python, Miniconda, CPU/CUDA/AMD runtimes, common prebuilt NVIDIA operators, and dependencies are bundled.

### [👉 Open the GitHub Release download page](https://github.com/Value99/CCCP/releases/tag/v0.9.15)

Download these **6 files** into the same folder:

1. `CCCP-Launcher-0.9.15-Offline-Setup.exe`
2. `CCCP-Launcher-v0.9.15-offline.parts.json`
3. `CCCP-Launcher-v0.9.15-win-x64-offline.zip.001`
4. `CCCP-Launcher-v0.9.15-win-x64-offline.zip.002`
5. `CCCP-Launcher-v0.9.15-win-x64-offline.zip.003`
6. `CCCP-Launcher-v0.9.15-win-x64-offline.zip.004`

Run `CCCP-Launcher-0.9.15-Offline-Setup.exe`. It verifies, joins, extracts, and starts the launcher. Model weights are distributed separately.

<!-- CCCP_AI_MAINTENANCE_START -->
## AI maintenance prompt (copyable)

> Thank you to every tester and contributor, and sorry that model/configuration differences can still expose compatibility issues. Give the following block to a coding AI together with your logs:

```text
Maintain CCCP Launcher 0.9.15. First read the bundled user-facing manuals docs/中文使用手册.md and docs/依赖与离线环境说明.md, then inspect the launcher terminal log and engine/CCCP-Engine. Reproduce with the bundled CPU/CUDA/AMD runtime and portable compiler only; do not use host Python, host CUDA, model-name branches, silent slow fallbacks, or the private quantization framework. Treat model configuration differences as data from cccp.json. Repair or simplify the public VQ, tensor-parallel, MLA, MoE, memory-planning, or launcher layer that owns the fault. Require finite logits, actual generation, clean process shutdown, and no regression against these H20-3e / bundled CUDA 13 baselines (Prefill/Decode): Qwen3.8 27B CCCP-L TP1 2462.68/56.94 token/s, DeepSeek V4 CCCP-S TP1 2786.87/46.23 token/s, GLM-5.2 CCCP TP4 932.36/25.48 token/s, GLM-5.3-Flash CCCP-S TP1 1504.67/30.02 token/s, GLM-5.3-Flash CCCP-M TP2 1282.44/20.22 token/s, Kimi K3 CCCP-470 TP4 672.04/15.69 token/s. H20 is only an approximate RTX 3090-class compute reference; VRAM capacity and memory bandwidth are not equivalent. Preserve CPU and existing Qwen, DeepSeek V4, GLM-5.2, GLM-5.3-Flash, and Kimi paths. Prefer deleting the broken duplicate branch after the shared path passes tests. Report exact commands, logs, finite-value checks, throughput, changed files, and package verification.
```
<!-- CCCP_AI_MAINTENANCE_END -->

### Highlights in 0.9.15

- All codebook models share one compact CUDA Decode algorithm: packed indices + Q8 codebooks + fused DP4A. Prefill selects the shared packed-VQ or E4M3/FP8 grouped path by capability.
- Linux/CUDA prefers FlashInfer MLA; native Windows/CUDA explicitly uses the bundled CCCP paged latent CUDA operator and never silently falls back to ordinary BF16 Attention.
- DSV4, Qwen, GLM, and Kimi declare structure through manifests and reuse the same codebook math, residency planning, and caching without pruning experts.
- Removes the single-token E4M3 MoE Decode and other legacy branches, and fixes Kimi/MTP workspace lifetime and CUDA builds from non-ASCII install paths.
- Measured on H20-3e with the bundled CUDA 13 runtime (Prefill/Decode, token/s): Qwen 2462.68/56.94 (TP1, 4096-token Prefill)、DSV4 2786.87/46.23 (TP1, 4096-token Prefill)、GLM-5.2 932.36/25.48 (TP4, 4096-token Prefill)、GLM Flash S 1504.67/30.02 (TP1, 4096-token Prefill)、GLM Flash M 1282.44/20.22 (TP2, 4096-token Prefill)、Kimi 672.04/15.69 (TP4, 4096-token Prefill). Every model passes its performance and finite-logits gates.

<!-- CCCP_PERFORMANCE_START -->
## Measured performance

These release-gated measurements use H20-3e and the bundled CUDA 13 runtime: 4096-token Prefill and Decode from a 4096-token context.

![CCCP 0.9.15 H20-3e performance](assets/cccp-performance-v0.9.15.svg)

| Full model name | Parallelism | Prefill token/s | Decode token/s |
|---|---:|---:|---:|
| Qwen3.8 27B CCCP-L | TP1 | 2462.68 | 56.94 |
| DeepSeek V4 CCCP-S | TP1 | 2786.87 | 46.23 |
| GLM-5.2 CCCP | TP4 | 932.36 | 25.48 |
| GLM-5.3-Flash CCCP-S | TP1 | 1504.67 | 30.02 |
| GLM-5.3-Flash CCCP-M | TP2 | 1282.44 | 20.22 |
| Kimi K3 CCCP-470 | TP4 | 672.04 | 15.69 |
<!-- CCCP_PERFORMANCE_END -->

## Core advantages

| Direction | Capability | Practical benefit |
| --- | --- | --- |
| **CCCP Quant** | Projection VQ + packed direct execution | Gate, Up, and Down use independent codebooks and precision layouts, while fused operators compute directly from compact indices. Public evaluation reaches approximately 2.31 bit/parameter with strong size–fidelity efficiency. |
| **More Useful** | One package, open and run | The package bundles the launcher, isolated Python runtime, inference dependencies, and operators. |
| **More Saving** | Quantization + task expert sets | Advanced quantization first reduces the complete model footprint, then task profiles define resident dynamic experts. Duplicate experts across combined profiles are counted once. |
| **More Smart** | Expert pre-selection | Real task corpora produce layer-by-layer expert heatmaps, allowing the most relevant experts to stay resident first. |
| **More Rapid** | Resident experts + format/operator co-design | Decoding routes through experts resident in RAM/VRAM, supported by codebook caches and CPU/CUDA/HIP fused operators. |

## CCCP quantization technology

CCCP advances MoE quantization as a system-level design. Weight representation, expert routing, memory residency, and execution operators share one coordinated format:

- **Projection-level vector quantization:** Gate, Up, and Down can use independent codebooks and precision layouts. The quantization granularity follows the internal structure of each expert and extends beyond one scalar bit width per layer.
- **Compact-index direct execution:** p8–p16 indices remain compact on disk, in RAM, and in VRAM. Fused operators consume codebooks and packed indices directly, reducing the resident footprint of expanded expert matrices.
- **Complete expert capacity:** storage savings come from a more efficient weight representation while preserving the expert count and per-token top-k activation count. Task profiles then narrow the active working set.
- **Structure-aware precision allocation:** `cccp.json` can describe per-projection, per-layer, and per-expert layouts, assigning higher precision to sensitive components and compact representations to more redundant components.
- **Quantization/runtime co-design:** CPU codebook caches, L2/L3-aware scheduling, CUDA/HIP fused operators, and multi-GPU parallelism execute directly around the CCCP format.

In the public evaluation, CCCP-S reaches an effective width of approximately **2.31 bit/parameter**, a **6.92× theoretical compression ratio**, and an **85.54% weight-byte reduction** relative to the theoretical BF16 weights of a 284B-parameter model. At a similar size, it reduces Mean KLD by approximately **54.90%** and raises same-top by **9.2736 percentage points** over UD-IQ1_S. See “Standardized evaluation” below for the full protocol and results.

<p align="center">
  <img src="assets/cccp-compression-chart.svg" alt="CCCP-S compression efficiency against the theoretical BF16 weight baseline" width="100%">
</p>

<p align="center">
  <img src="assets/cccp-quality-chart-v2.svg" alt="KLD and same-top comparison between CCCP-S and UD near 77 GiB" width="100%">
</p>

## Task expert profiles

“Training” in the launcher creates expert configuration files while model weights remain unchanged. The workflow:

- uses the existing model and its complete expert weights;
- preserves the number of experts activated per token while limiting the routing candidates for a task;
- each profile records the model name, version, fingerprint, and per-layer expert IDs;
- profiles can be exported, shared, combined, and deduplicated.

Long-context corpora are processed with pure prefill in `4096-token` blocks while expert hits are recorded throughout the complete context. For long-context workloads, start with a scan budget of roughly `500,000 tokens`, then adjust it based on the heatmap coverage curve and target profile size.

### Define a style with your own data

Users can upload preferred character, writing-style, or business corpora. The launcher scans which existing experts are actually used by that content and includes them in a new candidate/residency profile. In subsequent inference, experts related to that data distribution participate more often, biasing behavior toward the target style within the model's existing capabilities.

CCCP uses routing configuration: the corpus drives expert detection while weights and model knowledge remain unchanged. The core operation **selects a better combination from the model's existing experts**. Profiles can be named, described, exported, shared, and combined, giving users room to extend beyond bundled presets.

### Resident-expert fast path

The CCCP fast path loads the selected experts completely into RAM/VRAM before generation and builds codebook and execution caches. Normal decoding stays inside this resident pool, with disk access concentrated in the loading stage. Model size, working-set focus, and CPU/CUDA/HIP fused operators together target sustained interactive inference.

When device capacity runs short, mapped memory or disk offload keeps the model available and the launcher displays an explicit performance warning. This path covers capacity constraints; the fast path works best when the selected profile fits fully in RAM/VRAM.

## Standardized evaluation

The evaluation uses DeepSeek-V4-Flash-0731 on WikiText-2 with a fixed `ctx=512`: `573 chunks / 146,115` scored tokens. The reference distribution comes from BF16 logits produced by the official runtime.

| Method | Size (GiB) | Mean KLD ↓ | same-top ↑ | Execution |
| --- | ---: | ---: | ---: | --- |
| **CCCP-S** | **76.473** | **0.291115** | **83.0846%** | full-resident TP1×6 |
| UD-IQ1_S | 76.871 | 0.645514 | 73.8110% | llama.cpp |
| UD-IQ3_XXS | 97.051 | 0.306343 | 82.0150% | llama.cpp |
| UD-IQ4_NL | 127.277 | 0.180695 | 86.0750% | llama.cpp |

Using the theoretical BF16 weight bytes for the official `284B` parameter count as the baseline:

| Theoretical BF16 | CCCP-S | Theoretical compression | Weight-byte reduction | Effective width |
| ---: | ---: | ---: | ---: | ---: |
| 528.991 GiB | 76.473 GiB | **6.92×** | **85.54%** | **≈2.31 bit/parameter** |

At nearly the same size, CCCP-S reduces Mean KLD by approximately `54.90%` and raises same-top by `9.2736` percentage points compared with UD-IQ1_S.

> Measurement basis: 6.92× and 85.54% use the theoretical BF16 bytes for 284B parameters. The official mixed-precision download package follows a separate storage convention. This table compares size–fidelity efficiency; throughput is measured per execution path.

Sources:

- [CCCP project page](https://github.com/Value99/CCCP)
- [CCCP size and fidelity notes](https://github.com/Value99/CCCP#standardized-evaluation)
- [Official DeepSeek-V4-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

## Launcher interface

<p align="center">
  <img src="assets/launcher-home.jpg" alt="CCCP launcher home" width="82%">
</p>

<p align="center">
  <img src="assets/launcher-training.jpg" alt="CCCP expert-profile training page" width="82%">
</p>

## Quick start

1. Download the Offline Setup, parts manifest, and all four parts from [GitHub Release v0.9.15](https://github.com/Value99/CCCP/releases/tag/v0.9.15).
2. Run the Offline Setup from a writable directory and wait for verification and extraction.
3. Put a compatible model containing `cccp.json` in the `models` directory next to the application.
4. Double-click `CCCP-Launcher.exe`.
5. Select a model and an expert profile. A model can also be fully loaded when no profile is available.
6. Run preflight checks, review the estimated RAM/VRAM requirement, and start the model.

When available RAM or VRAM is insufficient, the launcher gradually falls back to mapped memory or disk offload. Inference can continue, but performance may drop substantially.

## OpenAI-compatible API

Once the model starts, CCCP exposes a complete OpenAI-compatible chat API path. Existing SDKs, chat frontends, and automation tools need no rewrite—change only the Base URL:

```text
http://127.0.0.1:8801/v1
```

| Endpoint | Capability |
| --- | --- |
| `GET /v1/models` | List the model currently being served |
| `GET /v1/models/{model_id}` | Read model identity and context metadata |
| `POST /v1/chat/completions` | Synchronous responses and SSE streaming |

The API accepts common message roles, `temperature`, `top_p`, `max_tokens`, `stop`, presence and repetition penalties, reasoning controls, tool calls, structured response formats, and streamed usage. Exact capabilities still depend on the active model architecture.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8801/v1",
    api_key="not-needed",
)

model = client.models.list().data[0].id
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

If API authentication is enabled in the launcher, replace `not-needed` with the generated or configured key. When authentication is disabled, any non-empty value works.

## Build a task-specific expert profile

1. Select a detected CCCP model on the Training page.
2. Import a UTF-8 `JSONL` or `TXT` corpus.
3. Set the token scan budget and start pure-prefill route scanning.
4. Inspect the per-layer expert heatmap and cumulative coverage curve.
5. Adjust coverage and review the estimated profile size.
6. Enter a profile name and description, then save or export it.

Corpora are stored locally by the application and can be reused after a restart or removed from the corpus library.

## Runtime environment

- Windows x64;
- CPU is the default and most portable inference path;
- isolated NVIDIA CUDA environments can be detected automatically;
- compatible AMD ROCm/HIP environments can be detected automatically;
- the package supplies its own Python runtime;
- the release package includes the runtime, dependencies, and precompiled or automatically compiled acceleration operators.

## Roadmap (TODO)

- [x] DeepSeek-V4 / DSpark CCCP support;
- [x] Kimi K3 CCCP support, including CPU-only, single-GPU+RAM, and multi-GPU tensor-parallel paths;
- [x] GLM-5.2 CCCP support, including RAM and multi-GPU inference;
- [x] Windows x64 offline launcher and OpenAI-compatible API;
- [ ] vision input with image URLs, Base64 images, and multimodal preprocessing;
- [ ] English and Russian launcher UI; multilingual documentation and the website are already available;
- [ ] macOS launcher, runtime, and native acceleration backend.

Vision input is scheduled for a later release and will open after image preprocessing and end-to-end regression are complete. The macOS runtime and release package are also on the roadmap; current packages target Windows x64.

## Tested platforms

| Platform / hardware | Validation scope | Status |
| --- | --- | --- |
| Windows 11 x64 · Core i9-13900H · 31.59 GiB RAM | Launcher, EXE, CPU inference, OpenAI API, profile scanning, real generation, full automated suite, and package smoke test | **CPU functionality passed end to end**; huge-model CPU speed is memory-bandwidth dependent and is not mixed with H20 scores |
| Windows 11 · NVIDIA RTX 3090 (20 GiB process limit) · CUDA 13 | DeepSeek-V4 constrained-VRAM CUDA/RAM, direct pinned transfer, strict LRU, fused decode, and multi-turn generation | **0.9.2 hardware passed** |
| NVIDIA RTX 5090 | Real CUDA/RAM runs with DeepSeek-V4 and GLM-5.2 | **Engine tested** |
| Linux · NVIDIA H20-3e · TP1 | Bundled CUDA 13; 4096-token Prefill / Decode from 4096-token context | **Qwen 2462.68/56.94, DSV4 2786.87/46.23 token/s**; both pass the 0.9.15 performance and finite-logits gates |
| NVIDIA H20-3e · release protocols | GLM-5.2 TP4, GLM Flash S TP1, GLM Flash M TP2, Kimi TP4 (GPU 2/3/4/5); 4096-token Prefill / Decode from 4096-token context | **GLM-5.2 932.36/25.48, GLM S 1504.67/30.02, GLM M 1282.44/20.22, Kimi 672.04/15.69 token/s**; all pass the 0.9.15 performance and finite-logits gates |
| Dual-socket CPU server (96 physical cores) | Qwen3.5 27B Dense VQ, Q4 NUMA shards, and 64-token Decode | **Historical 0.9.4 result: 9.77 token/s**; not a current-run result and no 30 token/s promise |
| Windows CUDA 13.0 / `sm_120` | Full NVCC compilation, linking, and module loading | **Toolchain passed**; validation scope ends at module loading |
| Windows ROCm 7.2.1 / `gfx1151` | HIPIFY, device-code generation, linking, and module loading on a build machine without an AMD GPU | **Toolchain passed**; AMD hardware end-to-end validation is still pending |
| macOS | Runtime and release package are on the roadmap | **Planned** |

Results apply only to the hardware, model, and execution path explicitly listed above. Performance also depends on model revision, expert profile, context length, RAM/VRAM bandwidth, SSD, and offload ratio.

## Reporting problems

If you encounter launcher, model-detection, operator-compilation, profile-training, or OpenAI API problems, please open a [GitHub Issue](https://github.com/Value99/CCCP/issues/new). Include:

- CCCP launcher/engine version and model name;
- operating system, CPU, GPU, RAM/VRAM, and driver version;
- reproducible steps, preflight result, and terminal logs;
- whether you used full loading, an expert profile, or disk offload.

Remove API keys, ModelScope tokens, private corpora, and other sensitive information before posting.

## Automatic update check

Stable-channel machine-readable manifest:

```text
https://raw.githubusercontent.com/Value99/CCCP/main/latest.json
```

PowerShell example:

```powershell
$release = Invoke-RestMethod 'https://raw.githubusercontent.com/Value99/CCCP/main/latest.json'
$release.version
$release.download.url
$release.launcher.sha256
```

`latest.json` uses the fixed schema identifier `cccp-launcher-update-v1`. The launcher should validate `schema`, compare semantic versions, and verify the downloaded file with SHA-256.

## Acknowledgements

Thanks to GitHub users [tmzncty](https://github.com/tmzncty) and [Zenon-Chen](https://github.com/Zenon-Chen) for supporting the development and progress of CCCP.

## Links

- Download: [GitHub Release v0.9.15](https://github.com/Value99/CCCP/releases/tag/v0.9.15)
- Community: [Discord](https://discord.gg/eNnwmAUY4M)
- Models: [ModelScope · ValueFX](https://www.modelscope.cn/profile/ValueFX)
- Project page: [GitHub · Value99/CCCP](https://github.com/Value99/CCCP)

---

If this project is useful to you, please click **Star** in the upper-right corner so more people can discover a practical way to run large MoE models on everyday hardware.

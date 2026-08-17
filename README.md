# C.C.C.P.

<p align="center">
  <img src="assets/cccp-icon.png" alt="C.C.C.P. 项目图标" width="96">
</p>

<p align="center">
  <strong>简体中文</strong> · <a href="README_EN.md">English</a> · <a href="README_RU.md">Русский</a>
</p>

## ⬇️ 下载 Windows 完整离线版（v0.9.6）

> [!IMPORTANT]
> **第一次使用请下载完整离线包，不要只下载单独的 `CCCP-Launcher.exe`。** 完整包已内置 Python、Miniconda、CPU/CUDA/AMD 推理环境、常见 NVIDIA 架构预编译算子及全部依赖。

### [👉 GitHub Release 下载页（推荐）](https://github.com/Value99/CCCP/releases/tag/v0.9.6)

打开下载页后，将下面 **6 个文件**全部下载到同一个文件夹：

1. `CCCP-Launcher-0.9.6-Offline-Setup.exe`
2. `CCCP-Launcher-v0.9.6-offline.parts.json`
3. `CCCP-Launcher-v0.9.6-win-x64-offline.zip.001`
4. `CCCP-Launcher-v0.9.6-win-x64-offline.zip.002`
5. `CCCP-Launcher-v0.9.6-win-x64-offline.zip.003`
6. `CCCP-Launcher-v0.9.6-win-x64-offline.zip.004`

然后双击 `CCCP-Launcher-0.9.6-Offline-Setup.exe`。安装器会自动校验、合并、解压并启动。模型不包含在启动器发行包内，需要单独下载并放入解压目录的 `models` 文件夹。

> Release 页面底部的 `Source code (zip/tar.gz)` 是 GitHub 自动生成的公开资料快照，不包含启动器源码、推理引擎源码、模型或 CCCP 量化框架。普通用户请下载上面列出的离线安装器和分卷。

### 0.9.6 重点更新

- 修复消费级 NVIDIA 显卡 Prefill 期间的 `cudaErrorIllegalAddress`：钳制 2 GiB 有符号偏移回绕、整块复用 Prefill 工作区、Windows 批量 H2D 拷贝限每组 ≤8 份。
- Kimi Prefill 与 MTP 的专家展开工作区统一块级收尾，不再泄漏进 Decode 阶段。
- 新增 `data/runtime/debug_env.txt` 诊断通道，GUI 下即可注入一次性环境变量。

## 核心优势

| 方向 | 能力 | 实际作用 |
| --- | --- | --- |
| **CCCP Quant · 先进量化** | 投影级 VQ + packed 直算 | Gate、Up、Down 使用独立码本和精度布局，融合算子直接计算紧凑索引；公开评测达到约 2.31 bit/parameter，并保持优秀的体积—保真度。 |
| **More Useful · 更加易用** | 一键整合包，打开就用 | 启动器、独立 Python 环境、推理依赖与算子随包提供，用户无需另装 Python。 |
| **More Saving · 更节省** | 量化 + 任务专家集合 | 先进量化先降低完整模型体积，再按任务配置常驻动态专家；多配置组合时重复专家只计算一次。 |
| **More Smart · 更智能** | 前置专家筛选 | 使用真实任务语料生成逐层专家热力图，让最匹配任务的专家优先常驻。 |
| **More Rapid · 更迅速** | 常驻专家 + 格式/算子协同 | 生成阶段在 RAM/显存常驻专家中路由，配合码本缓存与 CPU/CUDA/HIP 融合算子持续推理。 |

## CCCP 量化技术

CCCP 的先进性来自一套专门面向 MoE 的系统级量化设计。权重表示、专家路由、内存驻留和执行算子在同一套格式中协同工作：

- **投影级向量量化**：Gate、Up、Down 可以使用独立码本和精度布局。量化粒度贴合专家内部结构，比整层套用单一标量位宽更灵活。
- **紧凑索引全程直算**：p8–p16 索引在磁盘、RAM 和显存中持续保持紧凑形式。融合算子直接消费码本与 packed 索引，省去完整专家矩阵的常驻展开。
- **保留完整专家能力**：CCCP 通过更高效的权重表达压缩体积，模型的专家数量和每 token 的 top-k 激活数保持完整。任务配置在此基础上进一步收紧当前工作集。
- **精度按结构分配**：逐投影、逐层和逐专家布局都可以写入 `cccp.json`，重要部分保留更高精度，重复性更强的部分使用更紧凑的表示。
- **量化格式与算子协同**：CPU 码本缓存、L2/L3 友好调度、CUDA/HIP 融合算子和多卡并行直接围绕 CCCP 格式实现，量化后的模型可以进入高效执行路径。

公开评测中，CCCP-S 的有效位宽约为 **2.31 bit/parameter**，相对 284B 参数 BF16 理论权重达到 **6.92× 压缩**、字节减少 **85.54%**。在近似体积对比中，它相对 UD-IQ1_S 将 Mean KLD 降低约 **54.90%**，same-top 提高 **9.2736 个百分点**。完整协议与数据见下方“标准化实验数据”。

<p align="center">
  <img src="assets/cccp-compression-chart.svg" alt="CCCP-S 相对 BF16 理论权重的压缩效率图" width="100%">
</p>

<p align="center">
  <img src="assets/cccp-quality-chart-v2.svg" alt="约 77 GiB 档位下 CCCP-S 与 UD 的 KLD 和 same-top 对比图" width="100%">
</p>

## 任务专家配置

启动器中的“训练”负责生成专家配置文件，模型权重始终保持原样。配置流程具备以下特征：

- 直接使用现有模型及其完整专家权重；
- 保留每个 token 激活的 top-k 数量，并根据任务限定可参与路由的候选专家集合；
- 配置记录模型名、版本、指纹和逐层专家编号；
- 配置可以导出、分享、组合，并对重叠专家自动去重。

超长上下文语料按 `4096 token` 分块执行纯 prefill，并在整个过程中记录专家命中。长上下文任务建议从约 `500,000 token` 的扫描预算开始，再根据热力图覆盖率和目标体积调整。

### 用自己的数据定义风格

用户可以上传喜欢的角色、文风或业务数据集。启动器会扫描这些内容实际调用了哪些专家，并将它们纳入新的候选/常驻配置。之后推理会更多地让与该数据风格相关的专家参与工作，从而在模型原有能力范围内偏向目标风格。

CCCP 采用路由配置方式：语料用于探测专家，权重和模型知识保持不变。核心过程是**从模型已有的全部专家中找出更适合当前内容的组合**。配置支持重命名、说明、导出、分享和多方向组合，用户可以在预设之外继续扩展自己的任务风格。

### 专家常驻的高速路径

CCCP 的高速路径会在生成前把所选专家完整加载到 RAM/显存，同时建立码本和执行缓存。正常生成始终在这组常驻专家中路由，磁盘读取集中在加载阶段。模型体积、工作集和 CPU/CUDA/HIP 融合算子共同决定吞吐，设计目标是可交互、可持续运行的推理。

设备容量不足时，启动器会启用内存映射或磁盘卸载，并在界面中明确标注性能风险。该路径用于容量兜底；高性能运行建议让所选配置完整常驻 RAM/显存，并优先使用高速内存。

## 标准化实验数据

评测对象为 DeepSeek-V4-Flash-0731，数据集为 WikiText-2，固定 `ctx=512`，共计 `573 chunks / 146,115` 个计分 token。参考分布来自官方 runtime 的 BF16 logits。

| 方案 | 体积 (GiB) | Mean KLD ↓ | same-top ↑ | 执行方式 |
| --- | ---: | ---: | ---: | --- |
| **CCCP-S** | **76.473** | **0.291115** | **83.0846%** | full-resident TP1×6 |
| UD-IQ1_S | 76.871 | 0.645514 | 73.8110% | llama.cpp |
| UD-IQ3_XXS | 97.051 | 0.306343 | 82.0150% | llama.cpp |
| UD-IQ4_NL | 127.277 | 0.180695 | 86.0750% | llama.cpp |

以官方 `284B` 参数量的 BF16 理论权重字节数为基线：

| BF16 理论体积 | CCCP-S | 理论压缩比 | 理论字节减少 | 有效位宽 |
| ---: | ---: | ---: | ---: | ---: |
| 528.991 GiB | 76.473 GiB | **6.92×** | **85.54%** | **约 2.31 bit/parameter** |

同等体积附近，CCCP-S 相对 UD-IQ1_S 的 Mean KLD 降低约 `54.90%`，same-top 提高 `9.2736` 个百分点。

> 口径：6.92× 与 85.54% 以 284B 参数的 BF16 理论字节数为基准；官方混合精度下载包采用另一套存储口径。表格用于比较“体积—保真度”效率，吞吐速度按各运行路径单独测试。

CCCP 与模型资料：

- [CCCP 项目主页](https://github.com/Value99/CCCP)
- [CCCP 体积与保真度说明](https://github.com/Value99/CCCP#标准化实验数据)
- [DeepSeek-V4-Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

## 启动器界面

<p align="center">
  <img src="assets/launcher-home.jpg" alt="CCCP 启动器首页" width="82%">
</p>

<p align="center">
  <img src="assets/launcher-training.jpg" alt="CCCP 专家配置训练页" width="82%">
</p>

## 快速开始

1. 从上方 [GitHub Release](https://github.com/Value99/CCCP/releases/tag/v0.9.6) 下载全部 6 个离线安装文件。
2. 双击 `CCCP-Launcher-0.9.6-Offline-Setup.exe`，等待校验、合并和解压完成；不要直接在分卷内运行。
3. 将带有 `cccp.json` 的兼容模型放入程序同级 `models` 目录。
4. 双击 `CCCP-Launcher.exe`。
5. 选择模型和专家配置；初次使用也可直接选择全量加载。
6. 先执行预检，确认 RAM/显存估算，再启动模型。

当可用 RAM 或显存不足时，启动器会逐步使用内存映射或磁盘卸载。功能仍可继续，但推理速度可能明显降低。

## OpenAI 兼容 API

模型启动后会提供完整的 OpenAI 兼容聊天 API 链路。现有 SDK、聊天前端和自动化工具无需重写，只需把 Base URL 改为：

```text
http://127.0.0.1:8801/v1
```

| 接口 | 能力 |
| --- | --- |
| `GET /v1/models` | 列出当前已经启动的模型 |
| `GET /v1/models/{model_id}` | 获取模型标识与上下文信息 |
| `POST /v1/chat/completions` | 同步响应与 SSE 流式生成 |

兼容常用的 `messages` 角色、`temperature`、`top_p`、`max_tokens`、`stop`、存在惩罚、重复惩罚、思考强度、工具调用、结构化响应和流式 usage。具体能力仍取决于当前模型架构。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8801/v1",
    api_key="not-needed",
)

model = client.models.list().data[0].id
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

如果在启动器中启用了 API 鉴权，请把 `not-needed` 替换为启动器生成或设置的 Key。未启用鉴权时使用任意非空值即可。

## 生成任务专用专家配置

1. 在训练页面选择已经识别的 CCCP 模型。
2. 导入 UTF-8 编码的 `JSONL` 或 `TXT` 语料。
3. 设置 token 扫描预算并开始纯 prefill 路由扫描。
4. 查看逐层专家热力图和累计覆盖曲线。
5. 调整覆盖范围，查看预计配置体积。
6. 填写配置名称和说明并保存或导出。

语料会保存到应用本地目录，可在下次启动时继续复用，也可以在语料库中删除。

## 运行环境

- Windows x64；
- CPU 为默认且最通用的推理路径；
- 可自动探测独立的 NVIDIA CUDA 环境；
- 可自动探测兼容的 AMD ROCm/HIP 环境；
- 无需用户安装 Python；
- 运行时、依赖与已编译/可自动编译的加速算子随发布包提供。

## 路线图（TODO）

- [x] 支持 DeepSeek-V4 / DSpark CCCP 模型；
- [x] 支持 Kimi K3 CCCP，包括纯 CPU、单卡 GPU+RAM 和多卡张量并行路径；
- [x] 支持 GLM-5.2 CCCP，包括 RAM 模式和多卡推理；
- [x] 提供 Windows x64 离线启动器和 OpenAI 兼容 API；
- [ ] 支持视觉输入，接入图片 URL、Base64 图片和多模态前处理；
- [ ] 启动器界面增加 English / Русский，多语言文档和官网现已提供；
- [ ] 支持 macOS 启动器、运行环境和原生加速后端。

视觉输入已列入后续版本，完成图片前处理和端到端回归后开放。macOS 的运行环境与发布包也在路线图中，当前发布包面向 Windows x64。

## 已测试的平台

| 平台/硬件 | 当前验证情况 | 结论 |
| --- | --- | --- |
| Windows 11 x64 · Core i9-13900H · 31.59 GiB RAM | 启动器、EXE、CPU 推理、OpenAI API、配置扫描、118.47 GiB DeepSeek-V4 模型磁盘映射回归；完整自动化测试 `324 passed`、`11 skipped` | **CPU 端到端功能通过**；吞吐未达到原性能目标，GPU 路径见下方独立实测 |
| Windows 11 · NVIDIA RTX 3090（20 GiB 进程限额）· CUDA 13 | DeepSeek-V4 受限显存 CUDA/RAM、直接锁页传输、严格 LRU、融合 Decode 与多轮生成 | **0.9.2 真机通过** |
| Linux · NVIDIA H20 单卡 | Qwen3.5 27B Dense VQ 完整加载、有限值、MTP Top-3 与性能 | **0.9.3 真机通过：有效 Decode 93.21 token/s，完整请求 82.39 token/s；0.9.6 未改该路径** |
| NVIDIA RTX 5090 | DeepSeek-V4 和 GLM-5.2 的 CUDA/RAM 路径实机回归 | **引擎实测通过** |
| NVIDIA H20-3e（单卡与多卡） | DeepSeek-V4 TP1/TP4、GLM-5.2 TP2、Kimi K3 GPU+RAM/TP8 | **引擎实测通过** |
| 双路 CPU 服务器（96 物理核） | Qwen3.5 27B Dense VQ、Q4 NUMA 分片与 64-token Decode | **0.9.6 实测 9.77 token/s；较稳定基线约提升 38.6%，不承诺 30 token/s** |
| Windows CUDA 13.0 / `sm_120` | 完整 NVCC 编译、链接和模块加载 | **编译链通过**；验证范围到模块加载 |
| Windows ROCm 7.2.1 / `gfx1151` | 无 AMD GPU 构建机上的 HIPIFY、设备代码生成、链接和模块加载 | **编译链通过**，AMD 硬件端到端验证仍待补充 |
| macOS | 运行环境与发布包处于路线图阶段 | **计划中** |

测试结果只代表表中明确列出的硬件、模型和路径。速度会受模型版本、配置、上下文、RAM/显存带宽、SSD 和卸载比例影响。

## 问题反馈

如果启动、模型识别、算子编译、配置训练或 OpenAI API 运行有问题，欢迎提交 [GitHub Issue](https://github.com/Value99/CCCP/issues/new)。建议附上：

- CCCP 启动器/引擎版本和模型名称；
- 操作系统、CPU、GPU、RAM/显存和驱动版本；
- 可复现步骤、预检结果和终端日志；
- 是否使用全量加载、专家配置或磁盘卸载。

提交前请删除 API Key、ModelScope Token、私人语料和其他敏感信息。

## 自动更新检测

稳定版机器可读清单：

```text
https://raw.githubusercontent.com/Value99/CCCP/main/latest.json
```

PowerShell 检测示例：

```powershell
$release = Invoke-RestMethod 'https://raw.githubusercontent.com/Value99/CCCP/main/latest.json'
$release.version
$release.download.url
$release.launcher.sha256
```

`latest.json` 使用固定架构标识 `cccp-launcher-update-v1`。启动器应先校验 `schema`，再比较语义化版本号，最后使用 SHA-256 验证下载文件。

## 鸣谢

感谢 GitHub 用户 [tmzncty](https://github.com/tmzncty) 与 [Zenon-Chen](https://github.com/Zenon-Chen) 对 CCCP 项目开发与推进的支持。

## 链接

- 下载：[GitHub Release v0.9.6](https://github.com/Value99/CCCP/releases/tag/v0.9.6)
- 社区：[Discord](https://discord.gg/eNnwmAUY4M)
- 模型：[ModelScope · ValueFX](https://www.modelscope.cn/profile/ValueFX)
- 项目主页：[GitHub · Value99/CCCP](https://github.com/Value99/CCCP)

## English summary

C.C.C.P. is a Windows desktop launcher and dynamic-expert inference framework for large MoE models. It bundles its Python runtime and dependencies, measures real expert routes from task corpora, builds portable expert-residency profiles, deduplicates overlapping experts, and falls back to mapped memory or disk when device memory is insufficient. Training produces expert configuration files while leaving the original model and weights intact.

---

如果这个项目对你有帮助，欢迎点击右上角 **Star**，让更多需要在普通电脑上运行大型 MoE 模型的人看到它。

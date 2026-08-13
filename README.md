# C.C.C.P.

<p align="center">
  <strong>简体中文</strong> · <a href="README_EN.md">English</a> · <a href="README_RU.md">Русский</a>
</p>

<p align="center">
  <img src="assets/cccp-banner.jpg" alt="C.C.C.P. 动态专家推理框架" width="100%">
</p>

<p align="center">
  <strong>Collective Codebook Compression Pipeline</strong><br>
  量化、任务专家探测与多后端推理的一体化大型 MoE 模型运行框架
</p>

<p align="center">
  <img alt="Release" src="https://img.shields.io/badge/release-v0.9.0-a52f25">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%20x64-7a1e18">
  <img alt="Python" src="https://img.shields.io/badge/Python-%E6%97%A0%E9%9C%80%E5%AE%89%E8%A3%85-c49543">
  <img alt="Device" src="https://img.shields.io/badge/default-CPU-3a2118">
</p>

> C.C.C.P. 的目标是让大型 MoE 模型更容易在普通电脑上运行：不删除模型专家、不修改原始权重，通过量化、任务语料路由探测、专家常驻配置和优化算子，让有限内存优先服务真正会被当前任务调用的专家。

## 核心优势

| 方向 | 能力 | 实际作用 |
| --- | --- | --- |
| **More Useful · 更加易用** | 一键整合包，打开就用 | 启动器、独立 Python 环境、推理依赖与算子随包提供，用户无需另装 Python。 |
| **More Saving · 更节省** | 量化 + 专家探测 | 先扣除 Dense 与共享专家体积，再按配置预算选择动态专家；多配置组合时重复专家只计算一次。 |
| **More Smart · 更智能** | 前置专家筛选 | 使用真实任务语料生成逐层专家热力图，让最匹配任务的专家优先常驻。 |
| **More Rapid · 更迅速** | 更小模型 + 海量优化 | 使用码本缓存、CPU 加速算子和独立后端环境；具体吞吐取决于模型、内存带宽与硬件。 |

## 这不是模型训练

启动器中的“训练”只生成专家配置文件：

- 不生成新模型；
- 不修改或裁剪模型权重；
- 不改变模型原有 top-k 路由规则；
- 配置记录模型名、版本、指纹和逐层专家编号；
- 配置可以导出、分享、组合，并对重叠专家自动去重。

超长上下文语料按 `4096 token` 分块执行纯 prefill，并在整个过程中记录专家命中。长上下文任务建议从约 `500,000 token` 的扫描预算开始，再根据热力图覆盖率和目标体积调整。

## 标准化实验数据

评测对象为 DeepSeek-V4-Flash-0731，数据集为 WikiText-2，固定 `ctx=512`，共计 `573 chunks / 146,115` 个计分 token。参考分布来自官方 runtime 的 BF16 logits。

| 方案 | 体积 (GiB) | Mean KLD ↓ | same-top ↑ | 执行方式 |
| --- | ---: | ---: | ---: | --- |
| **CCCP-S** | **76.473** | **0.291115** | **83.0846%** | full-resident TP1×6 |
| MFQ EW-V2-S | 77.519 | 0.313576 | 82.2913% | streamed TP1×6, B4 |
| UD-IQ1_S | 76.871 | 0.645514 | 73.8110% | llama.cpp |
| UD-IQ3_XXS | 97.051 | 0.306343 | 82.0150% | llama.cpp |
| UD-IQ4_NL | 127.277 | 0.180695 | 86.0750% | llama.cpp |

以官方 `284B` 参数量的 BF16 理论权重字节数为基线：

| BF16 理论体积 | CCCP-S | 理论压缩比 | 理论字节减少 | 有效位宽 |
| ---: | ---: | ---: | ---: | ---: |
| 528.991 GiB | 76.473 GiB | **6.92×** | **85.54%** | **约 2.31 bit/parameter** |

同等体积附近，CCCP-S 相对 UD-IQ1_S 的 Mean KLD 降低约 `54.90%`，same-top 提高 `9.2736` 个百分点；相对 MFQ EW-V2-S 少 `1.046 GiB`，Mean KLD 降低约 `7.16%`，same-top 提高 `0.7933` 个百分点。

> 说明：6.92× 与 85.54% 是相对 284B 参数 BF16 理论字节数的计算值，不是相对官方混合精度下载包的压缩率。不同方案的执行路径不同，因此这组数据用于说明“体积—保真度”效率，不作为吞吐速度结论。

数据与模型资料：

- [原始实验仓库](https://github.com/Tylogi/TyloQuant)
- [实验协议与完整结果](https://github.com/Tylogi/TyloQuant/blob/master/README.zh-CN.md)
- [DeepSeek-V4-Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

## 启动器界面

<p align="center">
  <img src="assets/launcher-home.jpg" alt="CCCP 启动器首页" width="82%">
</p>

<p align="center">
  <img src="assets/launcher-training.jpg" alt="CCCP 专家配置训练页" width="82%">
</p>

## 快速开始

1. 从[百度网盘](https://pan.baidu.com/s/14ichCAsXKZMUQInIwIfQcA?pwd=cccp)下载完整 Windows 包，提取码：`cccp`。
2. 完整解压到可读写目录，避免直接在压缩包内运行。
3. 将带有 `cccp.json` 的兼容模型放入程序同级 `models` 目录。
4. 双击 `CCCP-Launcher.exe`。
5. 选择模型和专家配置；没有配置时也可以选择全量加载。
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

## 链接

- 下载：[百度网盘 · 提取码 cccp](https://pan.baidu.com/s/14ichCAsXKZMUQInIwIfQcA?pwd=cccp)
- 社区：[Discord](https://discord.gg/eNnwmAUY4M)
- 模型：[ModelScope · ValueFX](https://www.modelscope.cn/profile/ValueFX)
- 源码：[GitHub · Value99/CCCP](https://github.com/Value99/CCCP)

## English summary

C.C.C.P. is a Windows desktop launcher and dynamic-expert inference framework for large MoE models. It bundles its Python runtime and dependencies, measures real expert routes from task corpora, builds portable expert-residency profiles, deduplicates overlapping experts, and falls back to mapped memory or disk when device memory is insufficient. Training creates configuration files only—it does not generate a new model or alter the original weights.

---

如果这个项目对你有帮助，欢迎点击右上角 **Star**，让更多需要在普通电脑上运行大型 MoE 模型的人看到它。

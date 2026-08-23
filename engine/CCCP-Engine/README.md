# CCCP 最终推理框架

这是 CCCP 唯一的正式推理引擎仓库。生产源码、稳定回归测试、工程文档和可复跑的
验证摘要都在这里；模型权重、一次性探针、远端启动脚本和原始大日志不进入仓库。

当前支持：

| 模型 | 自动识别条件 | RAM 模式 | 多卡模式 |
|---|---|---:|---:|
| GLM MoE CCCP | `cccp.json` 不含 DeepSeek 特征字段 | 支持 | 支持专家并行 |
| DeepSeek-V4 / DSpark CCCP | 含 `hc_mult` 或 `compress_ratios` | 支持 | 三投影归档支持 no-owner 全层真 TP；TP4 已回归 |
| Kimi K3 CCCP | `model_family=kimi_k3` 或 Kimi 架构字段 | 支持 | 支持算子级张量并行 |

启动器读取模型目录内的 `cccp.json`，不依赖目录名称。Kimi、GLM 和 DeepSeek
分别使用独立配置文件，但共享 `cccp/ops/` 中按数学能力注册的算子与同一套运行时。
DeepSeek-V4 的旧双投影归档继续使用稳定兼容配置；带 Gate/Up/Down 独立
`projection_layouts` 或 `heterogeneous_expert_tiering` 的新归档自动选择
`dsv4_projection.json`，不靠目录名分支。同一异构 layout 解析也可用于
Kimi 逐层/逐专家分级归档。

Kimi 标准归档既支持源生 BF16 dense，也支持审计驱动的
source/FP8-block128/d3-p12 混合 Dense。存储层按 `dense.audit.json` 暴露逻辑
权重名，容量规划使用解码后的真实驻留字节；专家仍直接以 VQ 码本/索引计算，
不反量化成完整矩阵。p8–p16 专家索引在磁盘、RAM、VRAM 中都保持紧凑
格式，由与 GLM/DeepSeek 共用的原生 VQ 后端现场解析。

当前发布版本为 **1.2.0**。当前 Kimi K3-S 实测目录表观约 551GiB、含 82,432 个
专家，已通过 `cccp.json`/projection packing 读取、纯 CPU 和单 GPU+RAM 公共 CLI。
它不是旧 530GiB TP4 候选：当前标准清单的 TP4 静态需求约 150.8GiB/卡，超过
H20-3e 的 139.8GiB 物理容量；TP6/TP7 又不满足公共算子分片整除，完整 resident
只能使用约 82.7GiB/卡的 TP8。CLI 会在 CUDA 分配前完成这些检查。

纯 CPU 公共后端支持按真实形状自动选择紧凑 block-major32/NUMA-local block-FP8
布局，并可在 KDA recurrence 内融合 output gate 与 RMSNorm。默认选择、公共 CLI、
精确微基准和被拒绝方案见 [CPU block-major/NUMA 与 KDA 融合记录](docs/CPU-block-major与KDA融合-v118.md)。

单卡 RAM+VRAM 合计容量刚好覆盖模型时，统一启动器会根据 Manifest 实际字节、
可用 RAM/VRAM 和上下文预留自动进入极限模式，无需额外参数。它把紧凑专家分到
RAM/VRAM，并强制 Dense GPU-only；`--extreme/--no-extreme` 仅用于强制复现或排错。
完整限制和 CLI 见 [单卡 RAM+VRAM 极限模式](docs/极限模式.md)。

当前公开实机中位数：

| 模型 | 硬件/模式 | Decode |
|---|---|---:|
| DeepSeek-V4 | RTX 5090，TP1 RAM | 17.299 token/s |
| DeepSeek-V4 三投影 64G | H20-3e，TP1 RAM | 7.979 token/s |
| DeepSeek-V4 异构三投影 | H20-3e，严格64G RAM+24G VRAM极限模式 | 8.042 token/s |
| DeepSeek-V4 异构三投影 | H20-3e，TP1 全显存，2304-token SplitKV | 31.390 token/s |
| DeepSeek-V4 异构三投影 | H20-3e，TP1 全显存，6004-token SplitKV | 31.108 token/s |
| DeepSeek-V4 异构三投影 | H20-3e，TP1 全显存，当前发布复测 | 32.037 token/s |
| DeepSeek-V4 异构三投影 | H20-3e，TP4 全显存，当前发布复测 | 11.877 token/s |
| DeepSeek-V4 异构三投影 | RTX 5090/SM120，32GiB GPU+RAM（修复前 RAM 路径基线） | 15.413 token/s |
| DeepSeek-V4 异构三投影 | 双路 96 物理核，Q4 运行时镜像 | 12.232 token/s |
| DeepSeek-V4 异构三投影 | 单路 48 物理核，Q4 运行时镜像 | 10.981 token/s |
| GLM-5.2 | RTX 5090，TP1 RAM | 8.010 token/s |
| GLM-5.2 | 2×H20-3e，全显存 TP2 | 40.489 token/s |
| DeepSeek-V4 旧双投影归档 | 2×Xeon Gold 6530，纯 CPU VQ-INT8（历史） | 13.655 token/s |
| Kimi K3 480G | 8×H20-3e，全显存 TP8，短上下文 | 20.811 token/s |
| Kimi K3 480G | 8×H20-3e，全显存 TP8，32K 上下文 | 20.250 token/s |
| Kimi K3 700G | 8×H20-3e，全显存 TP8，三轮长程 | 20.149–20.791 token/s |
| 当前 Kimi K3-S | 1×H20-3e，GPU+RAM，当前 main | 4.218 token/s |
| 当前 Kimi K3-S | 双路 96 物理核，u16 运行时镜像 | 1.577 token/s |
| 当前 Kimi K3-S | 单路 48 物理核，compact | 1.306 token/s |

测量口径、硬件限制和证据等级见 [`BENCHMARKS.md`](BENCHMARKS.md)；全部 CLI、
模型预设和运行时变量见 [`PARAMETERS.md`](PARAMETERS.md)。完整安装、CLI、单卡、
多卡和 API 步骤见 [`docs/使用与部署.md`](docs/使用与部署.md)。所有可直接复制的
Chat、OpenAI API、CPU、单卡 RAM+GPU、极限模式和 TP2/TP4/TP8 命令统一见
[`docs/统一启动手册-Chat与OpenAI-API.md`](docs/统一启动手册-Chat与OpenAI-API.md)。
DeepSeek-V4 新三投影格式与公共算子验证见
[`docs/DeepSeek-V4三投影适配记录.md`](docs/DeepSeek-V4三投影适配记录.md)。
逐专家异构 VQ 76G 的格式、全 CLI 命令和 CPU/GPU/TP4 验收见
[`docs/DSV4异构专家VQ适配记录.md`](docs/DSV4异构专家VQ适配记录.md)。
正式 CLI 的实机命令、失败经验和新模型接入步骤见
[`docs/CLI经验与验收记录.md`](docs/CLI经验与验收记录.md)。
当前发布候选的完整结果、容量边界与可复制命令见
[`docs/2026-08-05最终验收报告.md`](docs/2026-08-05最终验收报告.md)。
公共 FP8 Indexer、精确 Top-K、分页 SplitKV 接口及非 SM90 后端移植方法见
[`docs/公共分页稀疏Attention接口-v166.md`](docs/公共分页稀疏Attention接口-v166.md)。

## 最快启动

环境需要 Python 3.10+、匹配显卡与 CUDA 的 PyTorch、CUDA Toolkit 和 Ninja。
首次安装：

```bash
cd /path/to/CCCP-Engine
chmod +x install.sh cccp.sh
./install.sh
```

CLI 对话只需给模型和要使用的物理显卡。启动器会自动识别架构、由显卡数量
推导 TP，并按实时容量选择单卡全显存、RAM offload 或极限 RAM+VRAM：

```bash
./cccp.sh chat --model /ssd/GLM-5.2-CCCP-L --gpus 7
./cccp.sh chat --model /ssd/DeepSeek-V4-Flash-DSpark-cccp-m --gpus 7
./cccp.sh chat --model /models/kimi-k3-cccp-480-standard --device cpu
```

其中纯 CPU 会自动隐藏 CUDA、启用公共融合后端，并在 RAM 足够时在线建立精确
CPU 执行镜像；空间不足时自动保留紧凑格式。`--profile`、`--tp`、缓存预算和
CPU 环境变量只用于强制复现，不是正常启动必填项。

CUDA 模式默认使用 `--dense-residency auto`：显存满足 Dense、上下文和运行时
安全余量时，Dense 常驻 GPU，启动期主机读取缓冲随即释放；专家仍按所选 profile
驻 RAM 或 VRAM。需要禁止静默回退时使用 `--dense-residency gpu`，容量不足会在加载
数百 GiB 专家前明确失败。显存上限低于 Dense 体积时可用
`--dense-residency ram --vram-limit-gb N`：CPU 计算紧凑 Dense，公共调度器只把
实际路由命中的 packed 专家字节和 Attention 动态状态交给 GPU。

GLM 多卡专家并行（两张卡自动推导 TP2）：

```bash
./cccp.sh chat \
  --model /ssd/GLM-5.2-CCCP-L \
  --gpus 0,1
```

TP2、TP4 会优先使用 intermediate tensor 分片；不能整除专家中间维的 TP3、
TP5、TP6、TP7 自动切换为 expert-ID 分片。若每卡显存不足以全量常驻专家，
当前引擎会明确提示并回退单卡 RAM 专家路径。

Kimi K3 真张量并行（全部 rank 共同持有 hidden，非 owner-rank 主从模式）：

```bash
./cccp.sh chat \
  --model /models/kimi-k3-cccp-700-standard \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-ctx 32768
```

Kimi 的 Attention、Dense、共享专家和 packed routed experts 都跨 rank 分片；
TPHidden 直接进入下一层。TP8 默认让小投影使用 TP4 子组、packed MoE 使用 TP8。
700G 长测的逐卡峰值最高 94.48 GiB，实际使用每卡 139.8 GiB 的 H20，并额外保留
3 GiB 运行余量；不能仅按模型目录表观大小估算可部署性。

Windows：

```powershell
.\install.ps1
.\cccp.ps1 chat --model "D:\models\GLM-CCCP" --gpus 0
```

## 工程目录

| 目录 | 职责 |
|---|---|
| `cccp/` | 模型无关引擎、模型实现、调度与公开 Python API |
| `cccp/ops/` | 按设备、量化格式和数学能力注册的通用算子 |
| `cccp/csrc/` | CPU/CUDA 原生融合算子 |
| `cccp/chat_adapters/` | 各模型的提示词协议与流式解析 |
| `cccp/configs/` | 可发布的静态架构配置 |
| `tests/` | 不依赖私有大模型权重的稳定回归测试 |
| `scripts/` | 历史开发实验与微基准；正式推理/验收统一走 `python -m cccp` |
| `results/` | 小体积、可公开的验收摘要与生成结果 |
| `docs/` | 架构决策、优化结论与可复现记录 |

主要子目录都有自己的 `README.md`；完整文件职责索引见
[`docs/文件与目录说明.md`](docs/文件与目录说明.md)。临时探针和远端执行脚本应
放在仓库外；验证后只把通用实现、稳定测试和结论写回本仓库。

## 启动 API

默认关闭 Think：

```bash
./cccp.sh serve \
  --model /ssd/GLM-5.2-CCCP-L \
  --gpus 7 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name cccp \
  --reasoning chat \
  --metrics-jsonl results/serve.jsonl
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

服务器启动后可使用标准库 CLI：

```bash
python -m cccp.api_cli_chat
```

OpenAI 兼容请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CCCP_API_KEY" \
  -d '{
    "model": "cccp",
    "messages": [{"role": "user", "content": "用中文解释张量并行。"}],
    "stream": true,
    "temperature": 0,
    "max_tokens": 512
  }'
```

已实现 `/health`、`/v1/models`、`/v1/chat/completions`，支持流式/非流式文本、
多轮消息、Think 模式，以及模型适配器允许的函数工具调用。当前请求中的
`message.content` 只接受文本；
图片 URL、Base64 图片、音频和视频尚未接入模型前处理，不能把保留了视觉权重的
模型归档误写为已经支持视觉推理。完整协议边界见
[`docs/OpenAI协议与视觉输入.md`](docs/OpenAI协议与视觉输入.md)。

## 启动前检查

无模型基础测试：

```bash
python -m cccp check --self-test
```

检查模型文件、RAM、GPU、P2P，并计算 GLM TP2–TP7 容量：

```bash
python -m cccp check \
  --model /ssd/GLM-5.2-CCCP-L \
  --profile parallel \
  --tp 2 \
  --gpus 0,1 \
  --matrix
```

只查看最终启动配置，不加载模型：

```bash
./cccp.sh serve \
  --model /ssd/DeepSeek-V4-Flash-DSpark-cccp-m \
  --gpus 7 \
  --dry-run
```

## 性能基准

统一短上下文稳态基准：

```bash
python -m cccp benchmark \
  --model /ssd/GLM-5.2-CCCP-L \
  --profile ram \
  --gpus 7 \
  --warmup 8 \
  --steps 32 \
  --repeat 3 \
  --json /tmp/cccp-benchmark.json
```

RTX 5090、H20-3e、4 卡容量、SM 支持等级、RTX 2080、PCIe/P2P/NUMA 和 SSD
机制的完整公开口径见 [`BENCHMARKS.md`](BENCHMARKS.md)。不要把容量规划标成实测
token/s，也不要跨不同模型和缓存模式直接比较数字。

单卡 `--profile auto` 会先做容量判断：完整模型可驻显存时选择 `resident`；模型
发布了稳定地址映射能力且专家需留在 RAM 时选择 `mapped + TokenGraph`；其余模型
继续使用安全的 `ram`。CUDA warp 数由设备能力选择，不把 H20 实测参数写成 SM120
默认值。5090 修复边界与复测命令见
[`docs/SM120单卡性能路径修复.md`](docs/SM120单卡性能路径修复.md)。

## 两种运行模式

### RAM

`--profile ram --tp 1`

- 这里的 RAM 模式仍由 GPU 计算，不等于纯 CPU 模式；
- dense、Attention、KV 和专家显存缓存位于单张 GPU；
- 压缩专家尽量全量常驻 RAM；
- 缓存未命中时由 RAM 上传 GPU；
- RAM 不足时回退文件读取；
- GLM 与 DeepSeek 均支持；DeepSeek 三投影归档还可选择全显存多卡模式。

### Parallel

`--profile parallel --tp N`

- GLM 支持 TP2–TP8 routed experts 多卡常驻与并行计算；
- Kimi 支持已验证的 TP4/TP8 no-owner 真张量并行；
- 上述 TP4 是较小历史归档的能力结论；当前约 551GiB 的 Kimi K3-S 逐卡容量不足以
  使用 TP4，且 TP6 不满足 routed hidden 分片整除，必须使用 TP8 或单卡 GPU+RAM；
- Kimi 的 Attention Column→Row、Dense/共享专家 Column→Row、Router、
  packed MoE 与跨层 TPHidden 都由所有 rank 共同计算；
- DeepSeek-V4 三投影 CCCP 的 Attention Head-TP、共享 Dense Column→Row、Router
  小 logits 规约、packed routed expert 与跨层 TPHidden 均由全部 rank 参与；
- DSV4 TP4 已做容量、数值和在线 CLI 回归；TP8 保留兼容配置但本轮没有复测性能；
  旧双投影归档仍只支持单卡；
- GLM 会自动选择可执行的 tensor/expert-ID 专家布局；
- 启动前应运行 `cccp check --matrix`，不要只按总显存相加估算。

### 纯 CPU 边界

> 以下 8.003/13.655 token/s 是旧双投影连续 VQ 归档
> `/ssd/DeepSeek-V4-Flash-DSpark-cccp-m` 的历史成绩。该目录已经清理；新的
> `projection-vq` Gate/Up/Down 三投影归档不能复用这个成绩。

DeepSeek-V4 已支持 x86 纯 CPU Decode。2×Xeon Gold 6530（64 个物理核）以
48 个计算线程、双路 NUMA 交错内存运行，高内存加速模式的正式复测中位数为
**13.655 token/s**，达到 12 token/s 目标。三轮分别测量 32 token，结果为
14.502 / 13.269 / 13.655 token/s；测量位置延伸至第 78 个 token 时仍为
13.655 token/s。

该模式会在约 64.5 GiB 常驻专家之外建立适合 CPU 连续访问的转置布局，实测进程
峰值 RSS 为 137.38 GiB，因此建议至少准备 160 GiB 可用 RAM。首次启动还会 JIT
编译 OpenMP/AVX-512 融合内核：

```bash
CUDA_VISIBLE_DEVICES='' \
CCCP_CPU_THREADS=48 \
CCCP_PREFETCH=0 \
CCCP_CPU_VQ_INT8=1 \
python3 -m cccp benchmark \
  --model /ssd/DeepSeek-V4-Flash-DSpark-cccp-m \
  --profile ram \
  --device cpu \
  --cache-gb 96 \
  --max-ctx 256 \
  --prompt cpu-test \
  --warmup 8 \
  --steps 32 \
  --repeat 3
```

`CCCP_CPU_VQ_INT8=1` 不修改模型权重和路由专家编号，但会把每个 token 的 VQ
查表分数按 32 个局部区间临时量化为 INT8，因此属于额外的近似计算模式。固定
短序列回归中，它与精确路径生成了相同的 12 个 token；这不等于对所有提示词
数学无损，正式部署仍应使用自身数据做质量回归。不启用该变量即可走精度保守路径。
高速结果依赖 AVX-512F/BW/VBMI；缺少 VBMI 时不能预期上述吞吐。

当前三投影测试模型 `<MODEL_DIR>` 的精确 CPU 路径
使用公共 compact block-FP8 与 p14/p16 AVX-512 稀疏行 gather，96 物理核、双 NUMA
交错、4-token 实测为 **3.662 token/s**，固定输出 token 与优化前一致，
`expanded_index_bytes=0`。相对该格式最初的 0.145 token/s 提升约 25.2×，但尚未达到
8 token/s；详细命令、阶段耗时和失败实验见
[`docs/CPU推理优化记录.md`](docs/CPU推理优化记录.md)。

更新的三投影模型 `<MODEL_DIR>` 会把不同码本的
Gate/Up、激活、Down 和 Top-K 归并合成一个公共原生调用。默认紧凑 row-tile8
路径为 **8.762 token/s**，不建立索引执行镜像。显式 `--cpu-compile q4` 会在进程
RAM 内建立额外的 Q4 BlockMajor 运行时执行镜像：双路 96 物理核正式 100-token×3
轮为 **12.232 token/s**，单路 node0 的 48 物理核为 **10.981 token/s**；两者均
`expanded_index_bytes=0`、43 层原生 packed 命中、无回退。Q4 是附加近似格式，
必须用部署数据单独验证质量，不能冒充无损路径。32 GiB H20 mapped 模式历史三轮
中位数为 **16.415 token/s**；最新完整验收见
[`docs/2026-08-05最终验收报告.md`](docs/2026-08-05最终验收报告.md)，实现演进见
[`docs/多码本CPU融合与32GiB弹性缓存-v168.md`](docs/多码本CPU融合与32GiB弹性缓存-v168.md)
、[`docs/llama.cpp式CPU执行布局-v169.md`](docs/llama.cpp式CPU执行布局-v169.md)
和 [`docs/CPU码本缓存与融合调度-v172.md`](docs/CPU码本缓存与融合调度-v172.md)。

Kimi K3-S Tail48 混合归档复用同一公共 CPU 后端：compact block-FP8、BF16
AVX-512、ProjectionGroup、原生 Router、SiTU/归并和 p8–p16 packed VQ 都按设备与
格式注册，不含模型名。当前双路 96 物理核、`--cpu-compile u16`、全量 82,432 个
专家常驻 RAM 的正式公共 CLI 为 **1.577 token/s**，RSS 905.0GiB；源 packed 索引
没有改写，执行镜像仅存在于进程 RAM。较省内存的 compact 路径历史最佳为
1.447667 token/s。Gate/Up 联合 score page 和 16 行硬件 gather 均经完整模型 A/B
证实退化并已删除。该结果不代表 4 token/s 目标已经完成；精确命令、物理下界和
失败实验见同一 CPU 优化记录及最终验收报告。

发布默认值按普通单路机器设置：`CCCP_CPU_THREADS=auto` 只选物理核，
`CCCP_CPU_NUMA=auto` 仅在系统实际报告多个 NUMA 节点时启用交错，
`CCCP_CPU_VQ_INT8=0` 保持精确路径。双路 NUMA 与 AVX-512/VBMI 都不是启动前提；
上面的服务器参数只用于复现实测成绩，不会自动套到家庭电脑。RAM profile 的默认
方式是 CPU RAM 保存专家、GPU 执行计算，不会把专家算子拆分给 CPU。

GLM 的旧 CPU 通用路径历史数据为 0.0478 token/s，加入 CPU 融合内核后尚未重新做
完整验收，因此不在这里填写新数字。没有 AVX-512 或缺少 C++/OpenMP 工具链时会
回退到兼容路径，不能预期上面的 DeepSeek 吞吐。

## 内存预留

命令行预留值优先于模型配置：

```bash
./cccp.sh chat \
  --model /ssd/GLM-5.2-CCCP-L \
  --gpus 7 \
  --ram-reserve-gb 48 \
  --vram-reserve-gb 4 \
  --dense-residency auto \
  --pin-gb 64
```

- `--ram-reserve-gb`：全量 RAM 常驻和 RAM 镜像都必须留下的系统余量；
- `--vram-reserve-gb`：启动阶段 CUDA allocator 硬上限之外的显存；
- `--vram-runtime-gb`：专家 arena 外保留给 KV 和临时张量的显存；
- `--pin-gb`：RAM 模式锁页热专家预算；
- `--dense-residency auto|gpu|ram`：自动/强制 GPU-only，或让紧凑 Dense 在 RAM 计算；
- `--vram-limit-gb`：进程 CUDA 分配硬上限，包含专家 arena、KV、Graph 和 workspace；
- `--cache-gb`、`--vram-gb`：需要人工固定缓存预算时再使用，默认自动计算。
- `--prefill-block-tokens`：外层预填充分块；全显存三投影 DSV4 默认 8192。
- `--prefill-moe-batch`：公共 packed MoE 固定微批，范围 1–256，默认 256。

对应底层环境变量：

```text
CCCP_RAM_RESERVE_GB
CCCP_RESIDENT_RESERVE_GB
CCCP_VRAM_RESERVE_GB
CCCP_VRAM_RUNTIME_GB
CCCP_PIN_GB
CCCP_HOST_PIN_GB
```

## 模型专属配置

每种架构只维护一个稳定预设文件：

```text
cccp/configs/glm.json
cccp/configs/dsv4.json
cccp/configs/dsv4_projection.json
cccp/configs/kimi_k3.json
```

配置包含上下文、采样、Think、RAM/VRAM 预留以及 RAM/parallel profile。
命令行参数覆盖配置，用户已显式设置的 `CCCP_*` 环境变量覆盖内置默认值。
完整参数表见 [`PARAMETERS.md`](PARAMETERS.md)。

DeepSeek 旧双投影归档传入 `--profile parallel` 或 `--tp 2` 会在加载权重前报错，
不会静默退化；新三投影归档由独立配置启用全层真 TP。Kimi TP4/TP8 使用
no-owner TPHidden；GLM 多卡显存不足时则由现有引擎打印原因并回退 RAM 路径。

## 直接使用底层入口

统一启动器之外，仍保留：

```bash
python -m cccp chat --help
python -m cccp serve --help
python -m cccp check --help
python -m cccp benchmark --help
python -m cccp launch --help
```

验证、性能与硬件边界统一记录在 [`BENCHMARKS.md`](BENCHMARKS.md)。
8192-token 批量 prefill 的精确 A/B、公共算子约束和复现命令见
[`docs/DSV4-8192批量预填充-v186.md`](docs/DSV4-8192批量预填充-v186.md)。

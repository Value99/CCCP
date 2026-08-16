# CCCP 参数手册

更新日期：2026-08-05

本文记录当前发布仓库的命令行参数、模型预设和全部 `CCCP_*` 运行时变量。
推荐只使用统一启动器和“稳定参数”；高级算子开关主要用于回归、二分与开发诊断。

## 配置优先级

从高到低：

1. 命令行显式参数；
2. 用户启动前显式设置的环境变量；
3. `cccp/configs/glm.json`、`cccp/configs/dsv4.json`、
   `cccp/configs/dsv4_projection.json` 或 `cccp/configs/kimi_k3.json`；
4. 源码中的兼容默认值。

统一启动器用 `setdefault` 应用模型环境，因此不会覆盖用户已经设置的 `CCCP_*`。
`--ram-reserve-gb` 同时设置 `CCCP_RAM_RESERVE_GB` 和
`CCCP_RESIDENT_RESERVE_GB`。启动阶段 CUDA allocator 硬上限由
`--vram-reserve-gb` 控制；专家 arena 外的上下文和临时工作区余量由
`--vram-runtime-gb` 独立控制。

## 模型预设

| 项目 | GLM | DeepSeek-V4 / DSpark | Kimi K3 |
|---|---|---|---|
| 自动识别 | `cccp.json` 不含其他架构特征 | 含 `hc_mult` 或 `compress_ratios` | `model_family=kimi_k3` 或 Kimi 架构字段 |
| `device` | `cuda` | `cuda` | `cuda` |
| `max_ctx` | 4096 | 4096 | 4096 |
| `max_new` | 512 | 512 | 512 |
| `temperature` / `top_p` | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |
| `spec` | 0 | 0 | 0 |
| `reasoning` | `chat`（关闭 Think） | `chat`（关闭 Think） | `chat`（关闭 Think） |
| API | `0.0.0.0:8000`，队列 16 | 同左 | 同左 |
| RAM 预留 | 32 GiB | 32 GiB | 32 GiB |
| 启动 allocator 预留 | 3 GiB | 旧格式 2 GiB；三投影 3 GiB | 1.25 GiB |
| 运行时 VRAM 余量 | 3 GiB | 旧格式 1.5 GiB；三投影 3 GiB | 3 GiB |
| CPU 默认 | 物理核自动、NUMA 自动、精确 VQ | 同左 | 48 线程、NUMA 自动、精确 packed VQ |
| RAM profile | TP1，关闭启动 RAM 镜像 | TP1，关闭启动 RAM 镜像 | TP1 packed RAM+GPU |
| Parallel profile | TP2–TP8 GLM EP | 三投影 no-owner 全层真 TP；TP4 已回归，旧格式不支持 | TP4/TP8 no-owner 真 TP |

这里的 RAM profile 是“专家权重保存在主机 RAM，算子由 GPU 执行”，不是把专家
计算拆给 CPU。纯 CPU 必须显式使用 `--device cpu`。

## 统一启动器

入口：

```bash
python -m cccp launch [chat|serve] --model MODEL [参数]
./cccp.sh [chat|serve] --model MODEL [参数]
```

| 参数 | 取值/默认 | 说明 |
|---|---|---|
| `action` | `chat` / `serve`；默认 `chat` | CLI 对话或 OpenAI 兼容 API |
| `--model` | 必填目录 | 含 `cccp.json` 的模型目录 |
| `--profile` | `auto` / `ram` / `resident` / `parallel` | 正常保持 `auto`；单卡按容量选全显存/RAM/极限，多卡选 parallel |
| `--tp` | 正整数 | 通常省略，由 `--gpus` 数量推导；显式值必须与卡数一致 |
| `--gpus` | 如 `7`、`0,1` | 设置物理可见卡；多卡时同时自动确定 TP |
| `--device` | `cpu` / `cuda` | 覆盖模型预设 |
| `--cpu-compile` | `off` / `auto` / `u16` | 纯 CPU 默认 `auto`；按可用 RAM 在线选择精确执行镜像 |
| `--max-ctx` | token 数 | 最大上下文 |
| `--max-new` | token 数 | 最大生成长度；小于等于 0 表示不设人工上限 |
| `--prefill-block-tokens` | 正整数；三投影 DSV4 默认 8192 | 外层 prefill 块；不会改变上下文数学 |
| `--prefill-moe-batch` | 1–256；默认 256 | 公共 packed MoE 固定微批；只控制复用 scratch 与提交数 |
| `--cache-gb` | GiB | 主机专家缓存；默认自动 |
| `--vram-gb` | GiB | 主卡专家显存缓存；默认自动 |
| `--dense-residency` | `auto` / `gpu` / `ram`；默认 `auto` | `auto` 容量足够时 Dense GPU-only、否则回退 CPU；`gpu` 容量不足立即失败；`ram` 让紧凑 Dense 在 CPU 计算、GPU 只执行适合加速的异构算子 |
| `--extreme` / `--no-extreme` | 可选覆盖 | 默认按 Manifest 与实际 RAM/VRAM 自动判断；仅用于强制启用或禁用 |
| `--extreme-placement` | `auto/layer/precision` | 同构默认整层；异构 `auto` 按每层 packed bit 预算选择 GPU 热专家 |
| `--extreme-score-file` | JSON 路径 | 可选 CCCP expert-preference/公共常驻分数，用 route mass 替代 bit 代理 |
| `--extreme-load-workspace-gb` | GiB | 加载峰值额外 RAM 余量，至少 0.25；仅在目标机器完成峰值验收后缩小 |
| `--ram-reserve-gb` | GiB | 系统 RAM 余量 |
| `--vram-reserve-gb` | GiB | 启动阶段 CUDA allocator 硬上限外的显存 |
| `--vram-runtime-gb` | GiB | 专家 arena 外为上下文和临时 workspace 保留的显存 |
| `--vram-limit-gb` | GiB | 本进程 CUDA 分配硬上限，适合在大显存卡上复现 16/24/32 GiB 部署；不裁剪主机 RAM 锁页预算 |
| `--pin-gb` | GiB | RAM 模式锁页热专家预算 |
| `--spec` | 整数；默认 0 | 投机解码草稿数；0 关闭 |
| `--temp` | 浮点数 | 采样温度；0 为贪心 |
| `--top-p` | 0–1 | nucleus sampling |
| `--think` | 开关 | CLI 开启 Think，等价于 `--reasoning max` |
| `--prompt` | 文本 | 单轮运行后退出 |
| `--host` | 地址 | API 监听地址 |
| `--port` | 端口 | API 监听端口 |
| `--served-model-name` | 文本 | API 对外模型名 |
| `--reasoning` | `chat` / `low` / `medium` / `high` / `max` | CLI Think 级别；Kimi 支持全部档位，API 当前支持 `chat/high/max` |
| `--max-queue` | 正整数 | API 最大排队请求数 |
| `--api-key` | 文本 | API Bearer Key |
| `--metrics-jsonl` | 路径 | 每请求指标 JSONL |
| `--cors-allow-origin` | 可重复 | 允许的 CORS Origin |
| `--dry-run` | 开关 | 只输出识别和最终配置，不加载模型 |

## 其他 CLI

### 基准

```bash
python -m cccp benchmark --model MODEL [参数]
```

| 参数 | 默认/说明 |
|---|---|
| `--profile`、`--tp`、`--gpus`、`--device`、`--dense-residency`、`--extreme` | 与启动器一致 |
| `--max-ctx` | 模型预设 |
| `--prompt` | 固定中文生产聊天提示 |
| `--prompt-repeat` | 1；在 CLI 内重复 prompt，构造可复现长上下文 |
| `--prompt-separator` | 一个空格；重复段之间的分隔符 |
| `--prefill-block-tokens` | 三投影 DSV4 默认 8192；显式设置用于 A/B |
| `--prefill-moe-batch` | 1–256；默认 256；`1` 是精确逐行对照 |
| `--warmup` | 预热 decode 步数 |
| `--steps` | 每轮计时 token 数 |
| `--repeat` | 重复轮数 |
| `--window` | 每轮起始位置间隔；用于延伸测量位置 |
| `--cache-gb`、`--vram-gb` | 人工缓存预算 |
| `--json` | 保存包含硬件、源码、环境、token 和每轮结果的 JSON |
| `--probe-stages` | 测量后额外执行 1 token 分阶段探针，不计入吞吐 |

### 启动前检查

```bash
python -m cccp check [参数]
```

| 参数 | 说明 |
|---|---|
| `--model` | 可选模型目录 |
| `--profile`、`--tp`、`--gpus`、`--max-ctx` | 待检查配置 |
| `--ram-reserve-gb`、`--vram-reserve-gb` | 容量检查余量 |
| `--matrix` | 输出所选模型各 TP 档容量矩阵；GLM 覆盖 TP2–TP8 |
| `--self-test` | 无模型基础测试 |
| `--cuda-ops` | 恰好一张可见 GPU 上编译并验证公共 packed CUDA 算子 |

### 底层 chat

`python -m cccp chat` 另提供 `--no-max-new`、`--rep-penalty` 和
`--no-repeat-ngram`。交互中可用
`/think off|low|medium|high|max` 动态调整；`low/medium` 仅适用于 Kimi。
通常应使用统一启动器，以免绕过模型预设。

### 底层 serve

`python -m cccp serve` 提供 `--model`、`--served-model-name`、`--host`、
`--port`、`--device`、`--cache-gb`、`--vram-gb`、`--dense-residency`、
`--tp`、`--max-ctx`、
`--default-reasoning`、`--spec`、`--max-queue`、`--api-key`、
`--cors-allow-origin` 和 `--metrics-jsonl`。API 依赖需安装：

```bash
pip install -e '.[api]'
```

### API CLI 客户端

```bash
python -m cccp.api_cli_chat [参数]
```

| 参数 | 默认/说明 |
|---|---|
| `--base-url` | `http://127.0.0.1:8000/v1` |
| `--model` | 默认从 `/v1/models` 获取 |
| `--api-key` | 可选 |
| `--system` | 系统提示词 |
| `--max-tokens` | 每轮生成上限 |
| `--no-stream` | 关闭流式输出 |
| `--timeout` | 单次请求超时秒数 |
| `--prompt` | 单轮消息，发送后退出 |

## 稳定环境变量

### 入口、精度与内存

| 变量 | 默认 | 说明 |
|---|---|---|
| `CCCP_API_KEY` | 未设置 | API 服务和客户端 Key |
| `CCCP_BASE_URL` | `http://127.0.0.1:8000/v1` | API CLI 地址 |
| `CCCP_MODEL` | 未设置 | API CLI 模型名 |
| `CCCP_PYTHON` | 当前 Python | Windows 包装脚本解释器 |
| `CCCP_COMPUTE_DTYPE` | `auto`；DSV4 预设 `bf16` | CUDA 计算精度：`auto/fp32/fp16/bf16` |
| `CCCP_DENSE_BF16` | 架构预设；DSV4 为 `all` | dense BF16 常驻集合或 `all/none` |
| `CCCP_FUSED` | 1 | 启用 CUDA 融合扩展 |
| `CCCP_FULL_RESIDENT` | 1 | RAM 足够时全量常驻专家 |
| `CCCP_RAM_RESERVE_GB` | 预设 2 | RAM 镜像与系统余量 |
| `CCCP_RESIDENT_RESERVE_GB` | 预设 2 | 全量专家常驻判定余量 |
| `CCCP_VRAM_RESERVE_GB` | 1 | 专用显存物理安全线；阶段工作区不会重复扣减 |
| `CCCP_VRAM_RUNTIME_GB` | GLM/Kimi 3；DSV4 1.5 | 专家 arena 外运行时余量 |
| `CCCP_PIN_GB` | 0 | 用户指定锁页热专家预算 |
| `CCCP_HOST_PIN_GB` | `auto` | 全量 RAM 常驻后的自动锁页预算 |
| `CCCP_HOST_PIN_VRAM_MULTIPLIER` | `0` | 默认不按显存裁剪锁页；只按专家总量、可用 RAM 与 2 GiB 系统余量判断。非零值仅用于手工限制 CUDA 主机映射。 |
| `CCCP_WDDM_DIRECT_PIN` | `auto` | 诊断开关；自动模式直接提交已锁页专家 RAM，只有未锁页来源才使用连续中转环。正常 GUI 不写入此项 |
| `CCCP_H2D_BATCH` | `auto` | 每个路由层把全部缺失专家一次提交给编译扩展；Windows/WDDM 在 C++ 内连续入队 `cudaMemcpyAsync`，Linux/TCC 使用 `cudaMemcpyBatchAsync`。运行时不支持时回退 Python 异步 copy stream；正常 GUI 不写入此项 |
| `CCCP_H2D_BATCH_MIN_COPIES` | `2` | 自动启用原生批量 H2D 的最小逻辑拷贝数 |
| `CCCP_KIMI_ASYNC_STAGE` | `1` | Kimi 单卡 RAM+GPU 缓存缺失时，以 CUDA event 异步提交紧凑专家 DMA；设为 `0` 可回退同步等待 |
| `CCCP_KIMI_OVERLAP_SHARED` | `1` | Kimi 单卡将 routed 专家 DMA 与同层共享专家计算重叠；逐层计时探针开启时自动停用 |
| `CCCP_RAM_MIRROR` | profile 决定 | 多卡加载前建立一次性 RAM 文件镜像 |
| `CCCP_PROFILE_JSON` | `MODEL/profile.json` | 专家热度档案路径 |
| `CCCP_LOAD_WORKERS` | 12 | 专家磁盘读取线程 |
| `CCCP_READ_BUF_MB` | 2 | 每文件读取缓冲 MiB |

`--dense-residency` 是 CLI/Engine 策略，不是新的权重格式。CUDA 成功加载后，
GLM、DeepSeek 和 Kimi 都只保留运行期 GPU Dense；`SafeFile` 的 Dense RAM 镜像引用
会单独释放，packed 专家 RAM 视图不受影响。`auto` 回退的是完整 CPU 计算模式，
不会形成一半 Dense 在 CPU、一半在 GPU 的隐式混算。
| `CCCP_PREFETCH` | GLM 1；DSV4 `auto` | 上一 token 专家预取 |
| `CCCP_PREFETCH_STAGE` | 1 | 预取同时执行 RAM→VRAM staging |
| `CCCP_VRAM_WATCH` | 1 | 动态显存保护 |
| `CCCP_VRAM_WATCH_LOW_GB` | 0.8 | 低于此空闲显存时收紧缓存 |
| `CCCP_VRAM_WATCH_HIGH_GB` | 3.0 | 高于此空闲显存时放宽缓存 |
| `CCCP_VRAM_WATCH_SEC` | 3 | 显存监测周期（秒） |

### CPU

| 变量 | 默认 | 说明 |
|---|---|---|
| `CCCP_CPU_THREADS` | `auto` | 自动选物理核；可设正整数 |
| `CCCP_CPU_NUMA` | `auto` | Linux 多 NUMA 节点时自动交错 |
| `CCCP_CPU_FUSED` | 1 | 启用 C++/OpenMP CPU 融合扩展 |
| `CCCP_CPU_BF16` | 0 | 实验性 CPU BF16 主路径；默认 FP32 |
| `CCCP_CPU_ATTN_MANY` | 1 | CPU 多头 Attention 融合 |
| `CCCP_CPU_QKV_POST` | 1 | CPU QKV 后处理融合 |
| `CCCP_CPU_DN_BLOCK` | 0 | 实验性 DN 索引转置/分块布局 |
| `CCCP_CPU_VQ_INT8` | 0 | 近似 VQ-INT8/VBMI 查表；非默认 |
| `CCCP_CPU_W4A8` | 0 | 实验性 INT4 权重、INT8 activation |
| `CCCP_CPU_W4ABF16` | 0 | 实验性 INT4 权重、BF16 activation |
| `CCCP_CPU_EXPAND_BF16` | 0 | 实验性 INT4→BF16 常驻展开 |
| `CCCP_CPU_MOE_PROFILE` | 0 | 记录 CPU MoE 分阶段耗时 |

普通家庭机默认不要求双路、NUMA 或 AVX-512。CPU 扩展使用本机 JIT 和条件编译；
`CCCP_CPU_VQ_INT8=1` 的公开高速数据依赖 AVX-512 VBMI，且属于额外近似。

### 通用 CUDA、Attention 与缓存

| 变量 | 默认 | 说明 |
|---|---|---|
| `CCCP_GROUPED` | 1 | 单 token MoE 分组计算 |
| `CCCP_SLOT_VQ` | 1 | 固定专家槽 VQ kernel |
| `CCCP_VQ_D4_SPECIALIZED` | 1 | D4 专用 VQ kernel |
| `CCCP_DECODE_WORKSPACES` | 1 | 复用 decode 工作区 |
| `CCCP_RMSNORM_WORKSPACES` | 1 | 复用 RMSNorm 工作区 |
| `CCCP_ATTENTION_GRAPH` | 1 | 单 token Attention CUDA Graph/稳定工作区路径 |
| `CCCP_ATTENTION_TENSOR_WORKSPACES` | 0 | 强制 tensor 工作区诊断路径 |
| `CCCP_PAGED_KV_FUSED` | 1 | 融合 paged KV |
| `CCCP_PAGED_KV_STRICT` | 0 | paged KV 不可用时禁止回退 |
| `CCCP_FLASHINFER_MLA` | 1 | 可用时启用 FlashInfer MLA |
| `CCCP_FLASHINFER_BACKEND` | `auto` | FlashInfer backend 选择 |
| `CCCP_FLASHINFER_GPU_PLAN` | 1 | 在 GPU 构造 batch-1 MLA plan |
| `CCCP_SPARSE_SPLITKV` | `auto` | DSV4 分页稀疏 Attention：`auto` 在上下文上限超过 4K 且后端可用时启用；另有 `force/off` |
| `CCCP_FLASHMLA_ROOT` | 空 | 可选 FlashMLA 源码/构建目录；正常安装为 Python 包时不需要 |
| `CCCP_GREEDY_DEVICE_WINDOW` | 8 | 贪心生成的设备侧窗口 |
| `CCCP_INT4_HALF` | 0 | INT4 反量化/计算半精度路径 |
| `CCCP_INT4_GEMV_FUSED` | 1 | 融合 INT4 GEMV |
| `CCCP_INT4_EMBEDDING_FUSED` | 1 | 融合 INT4 embedding |
| `CCCP_INT4_SWIGLU_FUSED` | 1 | 融合 INT4 SwiGLU |
| `CCCP_INT4_GROUP_VECTOR` | 0 | INT4 group-vector 实验路径 |
| `CCCP_INT4_LM_HEAD_VECTOR` | 1 | LM head vector kernel |
| `CCCP_INT4_SWIGLU_GROUP_VECTOR` | 0 | SwiGLU group-vector 实验路径 |
| `CCCP_LM_HEAD_INT4` | 1 | 允许 INT4 LM head |
| `CCCP_LM_HEAD_KEEP_F32` | 0 | LM head 保留 FP32 |
| `CCCP_STATIC_LM_OUTPUT` | 0；benchmark 为 1 | 复用静态 LM 输出缓冲 |

### DeepSeek-V4 / DSpark

| 变量 | 默认 | 说明 |
|---|---|---|
| `CCCP_SINGLE_TOKEN_ATTN_FAST` | 1 | DSV4 单 token Attention 快路径 |
| `CCCP_PROJECTION_WARPS` | 三投影预设 16 | 三投影 packed Gate/Up/Down CUDA kernel 的 warp 配置；由模型能力预设选择，不建议手工覆盖 |
| `CCCP_PREFILL_BLOCK_TOKENS` | 8192 | 全显存三投影 DSV4 的外层 prefill 块；其他路径保持安全默认 |
| `CCCP_PREFILL_MOE_BATCH` | 256 | 三投影 packed MoE 的固定行微批，范围 1–256 |
| `CCCP_DIRECT_KV_PREFIX` | 1 | DSV4 KV 前缀直接写入 |
| `CCCP_DSPARK_EXPERIMENTAL` | 0 | 启用非严格 DSpark 实验路径 |
| `CCCP_DSPARK_GB` | 1.5 | DSpark 主机专家/草稿预算 |
| `CCCP_DSPARK_VRAM_GB` | 2.75 | DSpark 显存预算 |
| `CCCP_SPEC` | 0 | 引擎投机解码默认草稿数 |

DSpark 严格模式默认回退主模型贪心。打开 `CCCP_DSPARK_EXPERIMENTAL` 后不保证与
`spec=0` token 等价，公开部署不应默认开启。

### GLM

| 变量 | 默认 | 说明 |
|---|---|---|
| `CCCP_LATENT_KV` | 1 | GLM MLA latent KV |
| `CCCP_LATENT_KV_INITIAL` | 2048 | latent KV 初始容量 |
| `CCCP_GLM_DIRECT_BMM` | 1 | GLM 直接 BMM 路径 |
| `CCCP_GLM_QB_SPLIT` | 1 | Q/B 分裂融合路径 |
| `CCCP_GLM_QB_GROUP_VECTOR` | 1 | Q/B group-vector kernel |
| `CCCP_GLM_SEQUENTIAL_PREFILL` | 1 | 大 prompt 顺序 prefill |
| `CCCP_GLM_SEQUENTIAL_PREFILL_MAX` | 512 | 顺序 prefill 分块上限 |
| `CCCP_GLM_CUBLAS_Q` | 0 | Q 路径强制 cuBLAS |
| `CCCP_GLM_CUBLAS_VALUE` | 1 | Value 路径使用 cuBLAS |
| `CCCP_GLM_CUBLAS_DECODE` | 0 | decode Q/Value 全部强制 cuBLAS |
| `CCCP_GLM_ROPE_FUSED` | 1 | RoPE 融合 |
| `CCCP_GLM_LATENT_PREP_FUSED` | 1 | latent KV 准备融合 |
| `CCCP_GLM_SCORE_FUSED` | 1 | Attention score 融合 |
| `CCCP_GLM_ROUTE_FUSED` | 1 | MoE router 融合 |
| `CCCP_GLM_NORM_QKV_FUSED` | 1 | Norm+QKV 融合 |
| `CCCP_GLM_RESIDUAL_NORM_QKV` | 1 | Residual+Norm+QKV 融合 |
| `CCCP_GLM_RESIDUAL_NORM_ROUTER` | 1 | Residual+Norm+Router 融合 |
| `CCCP_GLM_MOE_RESIDUAL_ADD` | 1 | MoE residual add 融合 |
| `CCCP_INDEXER_HADAMARD_FUSED` | 1 | Indexer Hadamard 融合 |

### GLM 多卡专家并行

| 变量 | 默认 | 说明 |
|---|---|---|
| `CCCP_EP_LAYOUT` | 自动 `tensor/expert` | 专家并行布局 |
| `CCCP_EP_DEVICE_ROUTE` | 1 | 路由结果保留在设备侧 |
| `CCCP_EP_FUSED_DISPATCH` | 1 | 专家分发融合 |
| `CCCP_EP_DIRECT_RETURN` | 1 | P2P 可用时从副卡直接返回 |
| `CCCP_EP_OVERLAP_SHARED` | 1 | 副卡 routed 与主卡 shared 重叠 |
| `CCCP_GLM_EP_FINAL_FUSED` | 1 | 多卡归并、shared、residual 最终融合 |
| `CCCP_CODEGEMM_VQ` | 1 | 全显存专家 CodeGEMM 布局 |
| `CCCP_CODEGEMM_GRAPH` | 1 | CodeGEMM CUDA Graph |
| `CCCP_CODEGEMM_DISPATCH_GRAPH` | 1 | 含 dispatch 的 CodeGEMM Graph |

这些变量只作用于 GLM 全显存 parallel profile。DeepSeek 三投影归档通过
`dsv4_projection.json` 组合公共 Attention Head-TP、Dense/Router Row-TP、TPHidden
与 packed expert Column/Row-TP；旧格式仍不支持。

### Kimi K3 RAM 与张量并行

Kimi 的 `parallel_tp4` 和 `parallel_tp8` profile 使用真正的 all-rank 数据流：
TPHidden 不存在 hidden owner，Attention、Dense/共享专家和 routed packed MoE
均按张量维分片。以下变量已进入稳定模型配置；一般用户不应逐项手工设置。

| 变量 | RAM / Parallel 默认 | 说明 |
|---|---|---|
| `CCCP_KIMI_PACKED_HYBRID` | RAM 为 1 | 单卡 packed 专家 RAM+VRAM 路径 |
| `CCCP_TP_GRAPH` | 1 | 启用张量并行固定地址 Graph |
| `CCCP_TP_DIRECT_INPUT` | 1 | 全 rank 直接读取输入状态 |
| `CCCP_TP_HIDDEN` | RAM 0；Parallel 1 | 启用分片 hidden 表示 |
| `CCCP_TP_HIDDEN_STATE` | RAM 0；Parallel 1 | 跨层延续 TPHidden |
| `CCCP_TP_NO_OWNER` | Parallel 1 | 禁止核心计算退化为 owner-rank 主从 |
| `CCCP_SMALL_OP_TP` | 4 | TP8 时小投影使用 TP4 子组 |
| `CCCP_DENSE_TP` | 1 | Dense Column→Row TP |
| `CCCP_FIRST_DENSE_TP` | 1 | 首层 dense TP |
| `CCCP_SHARED_MLP_TP` | 1 | 共享专家 Column→Row TP |
| `CCCP_ATTENTION_TP` | 1 | Q/K/V Column-TP、本地 head、O Row-TP |
| `CCCP_MLA_TP` | 1 | MLA head TP |
| `CCCP_PAGED_LATENT_ATTENTION` | 1 | paged latent KV |
| `CCCP_FLASHINFER_MLA` | 1 | 可用时使用 split-KV MLA |
| `CCCP_MOE_PARALLELISM` | Parallel 为 `tensor` | routed experts 按张量分片 |
| `CCCP_MOE_ROUTE_DOWN_TP` | Parallel 为 1 | Router/Down 公共 TP 路径 |
| `CCCP_TP_LAYER_GRAPH` | Parallel 为 1 | 每层固定地址父 Graph |
| `CCCP_TP_MOE_PLAN` | Parallel 为 1 | shared/router/packed MoE 合并提交 |
| `CCCP_TP_DECODE_LAYER_PLAN` | Parallel 为 1 | Attention→MoE 整层执行计划 |
| `CCCP_TP_PARALLEL_LAUNCH` | TP4 为 0；TP8 为 1 | TP8 全 rank 并行主机提交 |
| `CCCP_TP_FUSED_MOE_FINALIZE` | TP8 为 1 | 融合 MoE 最终归并 |
| `CCCP_TP_EVENT_BARRIER` | TP8 为 1 | 用 CUDA event 约束 rank 依赖 |
| `CCCP_ROUTED_WARPS` | 8 | packed routed kernel warp 配置 |
| `CCCP_P12_SHARED` | `direct` | packed 12-bit 索引直接读取 |

这些开关描述通用 TP 能力，实现在 `cccp/ops/`；Kimi 配置只组合所需能力，不复制
另一套推理系统。只有旧双投影 DeepSeek-V4 会在加载权重前拒绝 `tp>1`；带完整
`projection_layouts/index_packing` 的三投影归档由 `dsv4_projection.json` 启用真 TP。

### 诊断与元数据

| 变量 | 默认 | 说明 |
|---|---|---|
| `CCCP_STAGE_SYNC` | 0 | 专家上传改为同步诊断路径 |
| `CCCP_STAGE_VERIFY` | 0 | 逐字节校验 staging DMA |
| `CCCP_KV_TRACE` | 0 | 输出 KV 路径跟踪 |
| `CCCP_SOURCE_COMMIT` | 自动 Git commit | 覆盖 benchmark 源码版本元数据 |

## 不属于运行时参数的名称

`cccp/csrc/codegemm_vq.cuh` 中的 `CCCP_CODEGEMM_CODES`、
`CCCP_CODEGEMM_VECTOR`、`CCCP_CODEGEMM_K_TILE`、`CCCP_CODEGEMM_M_TILE` 和
`CCCP_CODEGEMM_THREADS` 是 C++ 编译期常量，不读取环境变量，不能在启动命令中调节。

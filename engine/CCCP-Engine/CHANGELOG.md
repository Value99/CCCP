# 变更记录

## 未发布

- H20 单卡 CUDA 分支以显式 merge 固化到主线；统一启动器现在由 `--gpus`
  数量自动推导 TP，单卡按实时容量选择 resident/RAM/extreme，纯 CPU 自动启用
  容量安全的执行镜像和公共融合参数。
- 纯 CPU 公共后端新增 p8～p16 索引在线编译、Q4 BlockMajor/NUMA-local 执行镜像
  和完整 Latent-MoE 单团队融合；DSV4 单/双路为 10.981/12.232 token/s，当前
  Kimi K3-S compact/u16 为 1.306/1.577 token/s，Kimi 输出 token 一致。
- DeepSeek-V4 新三投影 CCCP 增加独立配置、源生 block-FP8 Dense、p8–p16
  公共融合 MoE、单卡 RAM/resident 与 Attention/Dense/Router/packed MoE no-owner
  全层真 TP；TP4 已做实机数值和在线 CLI 回归，旧格式行为不变。
- 公共 CUDA 后端新增 grouped block-FP8 GEMV，以及 p8/p10/p16 混合 projection-VQ
  Top-K MoE 精确融合；注册键只描述设备、packed 格式、码本和激活，不绑定模型名。
- 单卡 Kimi RAM+GPU 完成 82,432 专家完整归档回归；热计算最高 6.204 token/s，
  最后 16-token 4.322 token/s。8 token/s 目标未达到，文档保留真实边界。
- 当前 551GiB Kimi K3-S 在最终 main 上的单 H20 GPU+RAM 正式三轮中位为
  4.218 token/s，prefill 7.625s，`expanded_index_bytes=0`。
- 统一 CLI 增加 `--dense-residency auto|gpu`。CUDA 成功加载后 Dense 仅驻 GPU，
  主机源镜像单独释放；packed 专家 RAM offload 不受影响。
- 增加 no-drop 专家清单收敛检查、异步路由元数据、固定地址 route plan 与相关回归。
- DSV4 启动检查新增 BOS/EOS/Think token、2048 Indexer 边界和官方
  `chat/high/max` 档位校验；新增全部模型的正式 CLI 命令与经验档案。
- Search-I01 新归档引入 p11/p13/p15；公共 CPU/CUDA 算子已支持连续
  p8–p16，六组 CUDA 布局数值回归和单卡 resident CLI 通过。
- CUDA 批量 H2D 按 CUDA 12.8/13 的 `cudaMemcpyBatchAsync` ABI 编译，H20 与
  RTX 5090/SM120 均从空缓存通过公共算子自检。
- DSV4 TokenGraph 的 Indexer Key 支持跨 1024 项分页 gather；8194-token prefill
  在 `max_ctx=12288` 后可继续精确 Top-512 decode。旧的无收益外层块实验未合并。
- 公共三投影 packed MoE 新增 1–256 行 CUDA prefill 批量能力，p8/p12/p14/p16
  批量与逐行位级一致；DSV4 默认使用 8192-token 外层块和 256 行 MoE 微批，
  最终 main 同一 8192-token 提示由 15.774 提升到 53.637 tok/s（3.400×），
  索引 0 展开；多 rank 也直接消费显式路由，不再回退旧组合图。
- Kimi 多卡启动新增公共真 TP 形状门禁：Head、Dense、共享专家、Router、
  routed latent 与 packed Down 的整除/字节边界在 CUDA 分配前检查。标准嵌套
  `routed_experts.layer_files` 也进入完整文件与容量审计；当前 S 的 TP4/TP8
  每卡需求分别为 150.8/82.7GiB，TP7 会在启动前明确拒绝。

## 1.2.0 - 2026-07-31

- Kimi K3 增加 TPHidden、Attention/Dense/共享专家 Column→Row、packed MoE
  all-rank TP4/TP8 和固定地址层级 Graph。
- Kimi 480G TP8 达到 20 token/s，并完成 32K 上下文衰减测试。
- Kimi 700G TP8 完成 HTML/BOSS/数学三轮长程稳定性与 KDA/MLA 缓存复用验收。
- GLM 200G FP8 dense 在同一通用框架完成 TP4 32K 与完整 HTML 回归。
- 补齐 Kimi 并行参数、OpenAI 文本协议、视觉输入边界和逐文件职责说明。
- 合并 GLM/DeepSeek 历史一键聊天入口的重复实现，保留原文件名作为薄兼容包装。
- 公共 CPU/CUDA VQ 与融合 MoE 增加 p9/p10/p16 及分组码本支持；磁盘、RAM、
  VRAM 始终保留紧凑索引，不生成完整反量化专家矩阵。
- 新 Kimi S（530.20 GiB）完成 82,432 个专家、十种 projection packing 的完整
  哈希审计，并通过 TP4 全显存短对话与 p9 CUDA 数值回归。

## 1.1.0

- 增加 Kimi K3 标准 CCCP 归档、KDA/MLA、SiTU、tokenizer 与聊天协议支持。
- 公共 CPU VQ 后端支持 `uint16` 索引，覆盖 12/14-bit 码本。
- 增加 Kimi 与 CPU VQ 稳定回归测试。
- 明确正式源码、测试、文档和临时实验文件的目录边界。

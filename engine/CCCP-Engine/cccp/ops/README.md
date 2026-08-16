# 通用算子注册层

本目录把“设备、量化格式、张量形状和数学能力”与具体模型解耦。Kimi、GLM 和
DeepSeek 通过同一注册表选择 CPU/CUDA VQ、MoE、Attention 和张量并行能力；
模型文件只保留架构数学差异。

- `spec.py`：设备、packed 格式、码本和算子能力描述。
- `registry.py`：注册、选择和能力查询。
- `api.py`：模型调用的稳定公共入口。
- `config.py`：注册层配置与环境开关。
- `cpu_backend.py`：CPU packed VQ/MoE 注册。
- `cuda_backend.py`：CUDA VQ/MoE/Attention 注册。
- `moe.py`：Top-K MoE 公共调度。
- `hidden.py`：跨 rank hidden 状态抽象。
- `tensor_parallel.py`：Column/Row-TP、collective 和固定地址 Graph。
- `profiling.py`：通用算子耗时探针。
- `selftest.py`：由 `cccp check --cuda-ops` 调用的公共 CUDA 数值验收。

约束：

1. 注册键描述数学能力，不使用模型名称作为算子分叉。
2. packed 8–16-bit 索引在磁盘、RAM、VRAM 中保持紧凑，不创建完整
   反量化矩阵；分组码本由专家 ID 选择，不能先展开成模型级统一 dtype。
3. owner 只能用于元数据，不能成为核心 hidden 或专家计算的数据 owner。
4. 新快路径必须保留正确性回退并增加无私有权重的数值测试。

长序列 KDA prefill 统一调用公共 `ordered_recurrent_scan`。CUDA/Hopper 环境在
安装 `fla-core` 与 Triton 后优先选择 64-token chunk scan；依赖或形状不满足时，
自动回退单次提交的 ordered CUDA scan，再回退注册的逐 token 参考实现。8192 是
外层 prefill block，不是递推 kernel 的内部 chunk。模型适配器只传 Q/K/V、gate、
beta 与 V-first state，后端名、chunk 大小和回退策略不得写入模型文件。

三投影 CUDA packed MoE 的同一个注册项同时覆盖 batch 1–256。Decode 传入
`[1,D] + [TopK]`，prefill 传入 `[N,D] + [N,TopK]`；内核以二维
`[row, expert]` 网格完成 Gate/Up、激活、Down 和 FP32 路由归并。微批只分配固定
activation scratch，packed 索引、码本和元数据地址保持不变。当前多行能力仅声明给
三投影 metadata，旧双投影或 grouped-prefix 请求会在公共 API 层明确拒绝并进入调用方
安全回退，不能误选一个形状看似兼容的内核。

三投影 packed MoE 的注册项覆盖 SiTU、SiLU/SwiGLU；具体模型只提交 projection
能力元组、激活名和 clamp 参数。H/C/S、front/tail 等层型属于配置数据，不能进入
算子注册名称。

DSV4 新增的是数学能力键 `cuda.route_topk.sqrtsoftplus.decode`、
`cuda.linear_route_topk.sqrtsoftplus.decode` 和
`cuda.attention.sliding_compressed_mqa.decode`。Kimi 既有的 SiTU、KDA、MLA、
Front44/Tail48 注册键不改名；`cccp check --cuda-ops` 会在一次测试中同时回归两类
布局，防止新增格式覆盖旧注册项。

Hyper-Connection CUDA 能力支持调用方固定输出缓冲：HC pre 复用 `y/post/comb`，HC post
复用互不别名的 hidden 结果；公共能力 `hyper_connection:post_moe` 还可把 FP32 routed、
BF16 shared 合并与 HC post 合成一次提交。注册名只有 `hyper_connection:pre_norm/post/post_moe`，
按 dtype、形状和数学能力选择，任何采用相同 H/C 数学的配置都能复用，不按模型目录名分叉。
`cccp check --cuda-ops` 的 `hyper_connection_workspace` 会同时验证公共注册选择、逐元素一致性和
输出地址复用。Kimi 不使用 H/C 数学，但它的 `residual_mix:attention`、TPHidden 固定 workspace、
packed MoE 固定输出遵循同一公共接口和“decode 不分配临时结果”的原则。

单卡 RAM+GPU 的动态路由通过公共能力
`cuda.packed_route_slots.fixed_metadata.decode` 完成。输入是 GPU 上的 Top-K
专家 ID 和固定槽目录，输出是调用方提供的指针/形状元数据与命中掩码；注册键不含
模型名。正常命中路径不调用 `.tolist()`，也不创建索引或反量化权重副本。目录未命中
时只回读 Top-K ID，再进入原有紧凑 RAM→VRAM 搬运回退。该算子和 FlashInfer MLA
动态 plan 都由 `python -m cccp check --cuda-ops` 做固定地址 CUDA Graph 验收；
MLA 同时逐字段核对单 CTA `1×78` 与双 CTA `2×39` 的官方调度。

CPU 多 token 验证使用两个同样不含模型名的公共能力：

- `cpu.block_scaled_gemm.e4m3fn.b128.verify`：一次扫描一个紧凑 block-FP8
  权重，为 2–16 行激活计算输出；row-major 与 block-major32 都直接读取原字节。
- `cpu.block_scaled_grouped_gemm.e4m3fn.b128.verify`：逻辑拼接多个投影，成员
  可以分成不同布局段；输出允许带行 stride，因而不需要为布局段复制整块结果。

两个能力都只展开一个 FP8 字节到寄存器并复用于候选行，不创建 BF16/FP32 权重，
注册选择依据仍是设备、格式、block size 和 batch。模型适配层只负责保持 Attention、
MoE 和 recurrent state 的数学顺序。

多卡 `Router → Top-K → packed expert` 的父图由公共
`TensorParallelRoutePackedPlan` 统一绑定。其能力键只包含 scoring、Top-K、
分组参数和 TP rank 数，不包含模型名；Kimi 与 DSV4 都只提交固定 logits、
correction、mask 和输出缓冲。模型自身的 token-hash 等特殊数学留在模型适配层，
普通分层专家配置可以直接复用该公共计划。

共享同一 token 输入的多个 block-FP8 projection 使用公共 `ProjectionGroup`。
该对象只建立固定地址 pointer/row-offset 元数据，源 FP8 payload 与独立 scale 原点
保持不变；执行统一进入注册项 `block_scaled_grouped_gemv`。Q/KV、Gate/Up、
Compressor/Indexer 等名称不进入注册键，Kimi、DSV4 或后续按层分级模型只需提交
projection 列表。TP 阶段耗时统一由 `TPHiddenStageProfiler` 记录异步 Event，并在
token 边界一次性汇总，禁止各模型再维护一套逐层同步探针。

延迟敏感的小投影可以使用公共 `ReplicatedSubgroupTensorParallel`，但语义必须是
所有子组同时计算同一逻辑算子，每个 rank 都直接获得完整输出；它不允许按层挑选
owner 子组。`compressed_state_update` 与 `head_rmsnorm_rope` 也按设备、ratio、
head/rope 维度注册，不按模型分支。前者只负责固定 ring state 写入，后者只负责
逐 head 归一化和 RoPE；模型适配层继续负责压缩池化与 Attention 数学拓扑。

单卡完整 decode graph 使用公共 `DecodeControl` 与 TP1 graph fast path。控制块只在
每个 token 发布一次 token/position，捕获算子从固定设备地址读取；`launch_tp1()`
不创建 Event，单 contribution reduction 不进入 all-rank kernel。模型可把多个完整
LayerGraph、final norm 和 lm_head 组合成一个 TokenGraph，但注册与选择仍依据设备、
dtype、packed 格式和数学能力，不能按模型名新增一套 graph 执行器。

## 公共分页稀疏 Attention

长上下文 decode 使用四个可独立替换的公共能力：

- `fused_compressor_cache_store`：一次写入 BF16 页、MODEL1 FP8 KV 和 FP8 Indexer；
- `paged_indexer_logits`：产生完整 FP32 Indexer logits；
- `persistent_topk_exact`：精确 Top-K，输出写入调用方固定缓冲；
- `sparse_paged_attention_splitkv`：直接消费 page index，不整理 selected KV。

注册键由 `dtype/cache_format/head_dim/top_k/page_layout/compression_ratio/architecture_features`
组成，模型名不得进入注册键。H20 当前后端为 FlashMLA SM90 SplitKV；非 SM90 后端
可以注册相同 operation 和自己的能力/页格式，无需复制模型系统。接口、MODEL1 字节
布局、回退规则、移植步骤和 CLI 见
[`公共分页稀疏Attention接口-v166.md`](../../docs/公共分页稀疏Attention接口-v166.md)。

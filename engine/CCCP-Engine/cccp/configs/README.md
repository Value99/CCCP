# 静态配置目录

这里保存可公开、可复用的架构默认值。运行时始终优先读取模型目录的 `cccp.json`，
不能根据模型文件夹名称猜参数。

新增配置必须写清适用架构，并由启动前检查验证层数、隐藏维度、专家数和量化档位。
机器相关的显存/RAM 数值不应写入架构配置。

DeepSeek-V4 有两个清单能力配置：`dsv4.json` 对应历史双投影兼容路径，
`dsv4_projection.json` 对应带逐层 Gate/Up/Down `projection_layouts`
或逐专家 `heterogeneous_expert_tiering` 的新格式。
启动器检查 `cccp.json` 的量化 schema 自动选择，目录名称不参与判断；两者仍调用
同一个 DSV4 模型和公共算子注册层。三投影配置组合 Attention Head-TP、共享
Dense/Router Row-TP、packed routed MoE 和 TPHidden；旧双投影配置只发布单卡。

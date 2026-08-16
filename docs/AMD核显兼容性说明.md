# AMD 显卡使用说明

启动器内置独立的 AMD ROCm/HIP 环境，不使用 DirectML，也不会替换 CPU 或 NVIDIA 环境。

使用条件：

- Windows 与 AMD 驱动支持当前 ROCm/HIP 版本。
- Ryzen APU 或 Radeon 设备位于 AMD 官方支持范围内。
- 启动器检测到 HIP 运行时、实际设备和 CCCP 融合算子均可用。

满足条件时，设备列表会开放“AMD ROCm/HIP”。第一次启动会针对当前显卡架构离线编译算子，可能需要等待几分钟；后续会复用缓存。编译期间终端持续显示活动进度条、已用时间、5 秒心跳及 HIP/Ninja 编译器输出；编译或加载失败时终端会显示原因，不会无提示卡住，也不会静默冒充 GPU 加速。

AMD 全显存 MoE Decode 的正常终端标识为 `hip.tp1-token-graph`，请求结束时 `decode_graph` 必须大于 0。若终端仍显示 `cuda.packed_moe_topk_fused`、`decode_graph=0`，或者解压目录中混有不同测试包的文件，请停止模型并重新解压完整发行版；不要只覆盖 `engine/CCCP-Engine/cccp` 中的部分文件。

Dense VQ 模型同样由 `cccp.json` 自动识别，但是否能启用 HIP 融合路径取决于模型声明的投影布局与当前 AMD 设备能力。启动器不会在融合算子不可用时静默宣称已经加速，具体原因会保留在终端。

随包提供的是用户态环境和编译工具，显卡驱动仍需由操作系统安装。若 AMD 选项置灰，请先查看设置页的设备探测信息和终端错误。

# CCCP Launcher 0.9.4 应用入口

- 版本：0.9.4
- 平台：Windows x64
- 首次安装：请前往 [v0.9.4 Release](https://github.com/Value99/CCCP/releases/tag/v0.9.4)，下载 Offline Setup、parts 清单和全部 4 个分卷。

0.9.4 修复 Linux 双路 NUMA 服务器上的 CPU Q4 内存页归属与自动线程规划。Qwen3.5 27B Dense VQ 的 64-token Decode 实测由约 7.05 提升到 9.77 token/s（约 +38.6%）；Windows 预编译 CPU 算子升级为 v195。0.9.3 已验证的 Qwen3.5 Dense VQ/MTP、DeepSeek-V4、Kimi K3、GLM-5.2 与桌面 GUI 路径保持不变。

发行包包含 CPU、NVIDIA CUDA、AMD ROCm/HIP 环境及离线编译工具，不包含模型权重、启动器/推理引擎源码或 CCCP 量化/训练框架。CPU 性能受处理器、内存通道和 NUMA 拓扑影响，本版不承诺 30 token/s。

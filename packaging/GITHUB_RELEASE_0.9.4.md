# CCCP Launcher 0.9.4

0.9.4 是 0.9.3 的 CPU NUMA 性能补丁。它不改变已验证的 NVIDIA、AMD、MoE、MTP、GUI 或配置格式，只修复 Linux 双路服务器上 Q4 执行映像的内存页归属和自动线程规划。

## 下载

第一次安装请把以下文件全部下载到同一目录，然后双击安装器：

- `CCCP-Launcher-0.9.4-Offline-Setup.exe`
- `CCCP-Launcher-v0.9.4-offline.parts.json`
- `CCCP-Launcher-v0.9.4-win-x64-offline.zip.001`
- `CCCP-Launcher-v0.9.4-win-x64-offline.zip.002`
- `CCCP-Launcher-v0.9.4-win-x64-offline.zip.003`
- `CCCP-Launcher-v0.9.4-win-x64-offline.zip.004`

安装器会校验每个分卷、合并、解压并启动。发行包包含 CPU、NVIDIA CUDA、AMD ROCm/HIP 环境和离线算子编译工具，不含模型权重。

## 主要变化

- Linux 双路 Q4 执行映像按行分片迁移到对应 NUMA 节点，避免复用分配器旧页后集中访问单路内存。
- 只在 Q4 NUMA 分片模式使用全部物理核，仍禁用 SMT；Windows P/E 核策略不变。
- Qwen3.5 27B Dense VQ 的 H20 CPU 64-token Decode 由约 7.05 提升到 9.77 token/s，输出有限。
- CPU 原生扩展升级到 v195，预编译 Windows 算子随包提供。
- 自动化回归：324 passed、11 skipped。

CPU 实测仍未达到 30 token/s，因此本版不作 CPU 目标吞吐承诺。Qwen3.5 的 NVIDIA FP8/MTP、DeepSeek-V4、Kimi K3、GLM-5.2 和桌面 GUI 路径沿用 0.9.3 已发布且通过的实现。

Windows 离线 CUDA 环境尚未内置 FlashMLA；DeepSeek-V4 在 2052 token 以上会明确提示依赖缺失，不会静默回退慢速路径。

# CCCP Launcher 0.9.2 应用入口

- 版本：0.9.2
- 平台：Windows x64
- 独立更新文件：`CCCP-Launcher.exe`
- 首次安装：请前往 [v0.9.2 Release](https://github.com/Value99/CCCP/releases/tag/v0.9.2)，下载 Offline Setup、parts 清单和全部 4 个分卷。

0.9.2 修复 Windows/WDDM 批量 DMA 延迟触发 CUDA illegal memory access，完善随包 CUDA/MSVC 工具链、按显卡架构编译缓存、显存→RAM→磁盘分级卸载、桌面进程退出与设置记忆。Windows CUDA 使用安全的编译层批量提交，Linux/TCC 保留原生批量 API；模型专家数量和原生 top-k 不变。

本版已完成 270 项自动化回归、最终离线目录 105,125 文件逐项 SHA-256 校验、CPU/CUDA/AMD 环境自检、冻结 GUI API 冒烟，并由 Windows NVIDIA 真机验证通过。

发行附件不含模型，也不含 CCCP 量化/训练框架。首次用户不能只下载这个单独 EXE；它主要用于覆盖更新已有完整离线目录。

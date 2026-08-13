# CCCP Launcher 0.9.0 应用入口

- 平台：Windows x64
- 版本：0.9.0
- 文件：`CCCP-Launcher.exe`
- SHA-256：`F09C7F8E6B249EE4AFF638BB252D70D1E30CBF656B71258DB0A4E319C0C52066`

此目录不包含模型权重。完整离线包还包含 CPU、NVIDIA CUDA、AMD ROCm/HIP 三套独立环境、CCCP Engine 与本地算子编译工具链；请按仓库首页 `latest.json` 指向的发布页获取完整离线包，再用本目录 EXE 覆盖同名文件。不要把 EXE 单独移出离线目录运行推理。

本次修订补齐 I‑1～I‑5 引擎接口契约、非阻塞双来源更新检查和安全回归；用户可见版本保持 0.9.0，因此没有修改仓库根目录的 `VERSION` 与 `latest.json`。

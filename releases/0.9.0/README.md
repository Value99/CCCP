# CCCP Launcher 0.9.0 应用入口

- 平台：Windows x64
- 版本：0.9.0
- 文件：`CCCP-Launcher.exe`
- SHA-256：`67BD11D3174F15EE81347AA58C89F19F98B8727A730B992F37A2B8CD47D10A73`

此目录提供启动器 EXE 更新文件。完整离线包包含 CPU、NVIDIA CUDA、AMD ROCm/HIP 三套独立环境、CCCP Engine 与本地算子编译工具链；请按仓库首页 `latest.json` 指向的发布页获取完整离线包，再用本目录 EXE 覆盖同名文件。推理时请让 EXE 保持在完整离线目录中。

本次修订补齐 I‑1～I‑5 引擎接口契约、非阻塞双来源更新检查、安全回归、CPU 多行 packed MoE 算子调度，以及 4096-token 路由扫描的分层进度和中间专家热力图持久化。用户可见版本保持 0.9.0，因此没有修改仓库根目录的 `VERSION` 与 `latest.json`。

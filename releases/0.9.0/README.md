# CCCP Launcher 0.9.0 应用入口

- 平台：Windows x64
- 版本：0.9.0
- 文件：`CCCP-Launcher.exe`
- SHA-256：`707DDF8E6B68B27F1EF599AFADC2D98D991A62682BDE66DAD26DAD31C7E661C1`

此目录提供启动器 EXE 更新文件。完整离线包包含 CPU、NVIDIA CUDA、AMD ROCm/HIP 三套独立环境、CCCP Engine 与本地算子编译工具链；请按仓库首页 `latest.json` 指向的发布页获取完整离线包，再用本目录 EXE 覆盖同名文件。推理时请让 EXE 保持在完整离线目录中。

本次修订补齐 I‑1～I‑5 引擎接口契约、非阻塞双来源更新检查、安全回归、通用 v192 CPU 分组批量 Prefill 算子，以及 4096-token 路由扫描的分层进度和中间专家热力图持久化。同一热力图现在可按不同覆盖率保存多份独立容量配置，快速拖动覆盖率也会严格以最后一次选择为准。用户可见版本保持 0.9.0，因此没有修改仓库根目录的 `VERSION` 与 `latest.json`。

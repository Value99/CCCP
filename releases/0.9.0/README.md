# CCCP Launcher 0.9.0 应用入口

- 平台：Windows x64
- 版本：0.9.0
- 文件：`CCCP-Launcher.exe`
- SHA-256：`12C38124D1E609336F4825DE2B79CADDF9F78E4AD6E83A4A1E46BBECB72AB0D6`

此目录提供启动器 EXE 更新文件。完整离线包包含 CPU、NVIDIA CUDA、AMD ROCm/HIP 三套独立环境、CCCP Engine 与本地算子编译工具链；请按仓库首页 `latest.json` 指向的发布页获取完整离线包，再用本目录 EXE 覆盖同名文件。推理时请让 EXE 保持在完整离线目录中。

本次修订完成真实 118 GiB 模型与 32G 配置的生成、停止、回退、KV 前缀复用/分支重建和模型能力思考档位回归；配置卡统一显示总驻留，训练任务自动进入终端并显示 4096-token 块与层进度。离线包只含推理运行时，不含模型或 CCCP 量化/训练框架；首次安装器会校验四个分卷并显示合并、解压进度。用户可见版本保持 0.9.0，因此没有修改仓库根目录的 `VERSION` 与 `latest.json`。

## CCCP Launcher 0.9.1 完整离线版

这是不含模型的 Windows x64 完整离线发行包，内含 CPU、NVIDIA CUDA、AMD ROCm/HIP 三套独立推理环境、Miniconda、仅用于加载和推理的 CCCP-Engine 运行时、预编译 CPU 算子和本地编译工具链。包内不含 CCCP 量化/训练框架及其开发工具。

### 下载与首次启动

请下载以下内容并放在同一个文件夹：

- `CCCP-Launcher-0.9.1-Offline-Setup.exe`
- `CCCP-Launcher-v0.9.1-offline.parts.json`
- 全部 `.zip.001`、`.zip.002`……分卷

双击离线安装器即可。它会显示校验、合并和解压进度，完成后自动启动主程序。无需安装 Python、Miniconda、CUDA/ROCm SDK 或编译器，运行时也不会自动下载这些依赖。以后直接双击解压目录中的 `CCCP-Launcher.exe`。

GPU 驱动仍需由操作系统提供。发行包不含模型权重；模型请单独放入 `models`。

GitHub 页面自动显示的 `Source code (zip/tar.gz)` 只是当前公开仓库的 README、版本文件、图片与独立启动器快照，不包含启动器源码、推理引擎源码或 CCCP 量化/训练框架。首次安装请下载离线安装器、parts 清单和全部分卷。

### 0.9.1 更新

- 完成 333.524 GiB GLM‑5.2 CCCP 模型的通用识别、4096-token 路由扫描、热力图配置保存、加载和真实 GUI 生成回归。
- 修复 Dense 层 0–2 被严格路由误判为少于 top-k=8 的问题。
- 修复组合 Gate+Up/Down Q4 专家单 token 解码的布局分派错误。
- 达到 `max_tokens` 后不再多执行一次无用的完整模型 decode。
- 流式错误会进入终端，聊天指标按请求精确关联，未收到完整结束标记时不再假装成功。
- 保持通用配置驱动，不按模型目录名增加专用分支；模型、专家编号和原生 top-k 均未削减。

可单独下载 `CCCP-Launcher.exe` 覆盖已有完整离线目录进行更新；首次安装必须下载完整分卷。

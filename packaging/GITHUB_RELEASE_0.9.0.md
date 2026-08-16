## CCCP Launcher 0.9.0 完整离线版

这是不含模型的 Windows x64 完整离线发行包，内含 CPU、NVIDIA CUDA、AMD ROCm/HIP 三套独立推理环境、Miniconda、仅用于加载和推理的 CCCP-Engine 运行时、预编译 CPU 算子和本地编译工具链。包内不含 CCCP 量化/训练框架及其开发工具。

### 下载与首次启动

请下载以下内容并放在同一个文件夹：

- `CCCP-Launcher-0.9.0-Offline-Setup.exe`
- `CCCP-Launcher-v0.9.0-offline.parts.json`
- 全部 `.zip.001`、`.zip.002`……分卷

双击离线安装器即可。它会显示校验、合并和解压进度，完成后自动启动主程序。无需安装 Python、Miniconda、CUDA/ROCm SDK 或编译器，运行时也不会自动下载这些依赖。以后直接双击解压目录中的 `CCCP-Launcher.exe`。

GPU 驱动仍需由操作系统提供。发行包不含模型权重；模型请单独放入 `models`。

GitHub 页面自动显示的 `Source code (zip/tar.gz)` 只是当前公开仓库的 README、版本文件和图片快照，不包含启动器源码、推理引擎源码或 CCCP 量化/训练框架。首次安装请下载离线安装器、parts 清单和全部 4 个分卷。

### 本次最终修订

- 通用 v192 CPU 分组批量 Prefill 算子同时用于训练扫描与正常聊天。
- 同一热力图可保存多份不同覆盖率/容量的独立专家配置。
- 配置卡统一显示 Dense、共享专家与动态专家相加后的配置总驻留；运行预检另计 KV 和工作区。
- 聊天支持停止、回退、KV 前缀复用/分支重建，并按模型声明自动显示可用思考档位。
- Token 扫描启动后自动进入终端，持续显示 4096-token 块、当前层和算子日志。
- CPU/GPU 首次算子编译显示活动进度、后端、已用时间、5 秒心跳和原始编译器输出，不再无提示等待。
- 版本保持 `0.9.0`；仓库 `VERSION` 与 `latest.json` 未修改。

可单独下载 `CCCP-Launcher.exe` 覆盖已有完整离线目录进行更新；首次安装必须下载完整分卷。

## CCCP Launcher 0.9.2 完整离线版

这是不含模型的 Windows x64 完整离线发行包，内含 CPU、NVIDIA CUDA、AMD ROCm/HIP 三套独立推理环境、Miniconda、仅用于加载和推理的 CCCP-Engine 运行时、预编译 CPU 算子和本地编译工具链。包内不含 CCCP 量化/训练框架及其开发工具。

### 下载与首次启动

请下载以下内容并放在同一个文件夹：

- `CCCP-Launcher-0.9.2-Offline-Setup.exe`
- `CCCP-Launcher-v0.9.2-offline.parts.json`
- 全部 `.zip.001`、`.zip.002`……分卷

双击离线安装器即可。它会显示校验、合并和解压进度，完成后自动启动原生桌面程序，不会自动打开外部浏览器。无需安装 Python、Miniconda、CUDA/ROCm SDK 或编译器，运行时也不会自动下载这些依赖。

GPU 驱动仍需由操作系统提供。发行包不含模型权重；模型请单独放入 `models`。

### 0.9.2 更新

- 移除启动器自动打开系统浏览器的所有启动/失败降级路径。
- GPU 预检严格按“显存 → 主机内存 → 磁盘”逐级降级，显存不足不会直接误报磁盘卸载。
- 修复离线 CUDA 首次编译找不到 cuBLAS DLL 或 MSVC librarian；封装内置并验证完整 CUDA/MSVC/Windows SDK/Ninja 工具链。
- CUDA Runtime 按当前隔离环境动态发现，支持随包 CUDA 13 的 `cudart64_13.dll`；终端正确解码 UTF-8 与 Windows 编译器混合输出。
- 自动适配并缓存 SM75（RTX 20 系）、SM86（RTX 30 系）、SM89（RTX 40 系）、SM90（H20/H100）、SM120（RTX 50 系）融合算子。
- 关闭桌面窗口会同步停止 Python 推理后端；重启清空旧终端；CPU/NVIDIA/AMD 上次选择会被记住。
- DSV4 受限显存路径改为配置内全部专家常驻 RAM、有界 VRAM 热缓存；页锁定资源不足时自动退回普通 RAM staging，不再因 `cudaHostRegisterMapped` 失败退出，也不减少配置专家或 top-k。
- Windows/WDDM 自动原地锁页配置内专家 RAM；每个路由层的全部缺失专家一次进入编译扩展，并在 C++ 内连续提交到同一 copy stream。Windows 使用稳定的 `cudaMemcpyAsync`，不再调用现场两次复现延迟非法访问的 `cudaMemcpyBatchAsync`；Linux/TCC 保留原生批量 API。只有实际未锁页来源或扩展拒绝批次时才回退 Python 异步 copy stream。
- 显存统一保留 1 GiB；Dense、KV、Prefill 最大层工作集、码本、模型原生 top-k 交换槽都先按真实体积计算，剩余空间用于语料热专家。运行期所有热槽执行严格 LRU 末位淘汰，不做永久保护，KV 增长时可自动收缩。终端会显示完整 CUDA 容量账、阶段切换、融合算子和 DMA 批次账。
- 保留 GLM 前置 Dense 层修复，严格路由只校验模型声明的专家层。

可单独下载 `CCCP-Launcher.exe` 覆盖已有完整离线目录进行更新；首次安装必须下载完整分卷。

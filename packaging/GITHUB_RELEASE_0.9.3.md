# CCCP Launcher 0.9.3

0.9.3 新增清单驱动的通用 Dense VQ 支持。Qwen3.5 27B Dense 可直接加载完整模型，不需要专家配置，也不会显示专家训练入口。

## 下载

第一次安装请把以下文件全部下载到同一目录，然后双击安装器：

- `CCCP-Launcher-0.9.3-Offline-Setup.exe`
- `CCCP-Launcher-v0.9.3-offline.parts.json`
- `CCCP-Launcher-v0.9.3-win-x64-offline.zip.001`
- `CCCP-Launcher-v0.9.3-win-x64-offline.zip.002`
- `CCCP-Launcher-v0.9.3-win-x64-offline.zip.003`
- `CCCP-Launcher-v0.9.3-win-x64-offline.zip.004`

安装器会校验每个分卷、合并、解压并启动。发行包包含 CPU、NVIDIA CUDA、AMD ROCm/HIP 环境和离线算子编译工具，不含模型权重。

## 主要变化

- Qwen3.5 Dense VQ 通用识别、GUI、CPU/CUDA 与 OpenAI API 链路。
- H20 单卡实模 Prefill 约 527 token/s、Decode 约 55.56 token/s。
- Kimi 10/15 行组合投影使用 packed grouped 融合执行器。
- DeepSeek-V4、Kimi K3、GLM-5.2 实模回归通过。
- AMD 全显存路径要求 `hip.tp1-token-graph` 和有效 Graph 提交，拒绝旧执行器伪加速。
- 自动化回归：313 passed、11 skipped。

CPU 功能和数值正确，但本版实测没有达到预设吞吐目标，因此不作 CPU 速度承诺。

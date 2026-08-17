# CCCP Launcher 0.9.6

0.9.6 是消费级 NVIDIA 显卡 Prefill 稳定性修正版。

- 修复 Prefill 编译投影核的 2 GiB 有符号偏移回绕：gate-up 平板超过 2^31 字节时专家分片会越界写入甚至静默损坏显存，现按核寻址上限硬性钳制分片大小。
- Prefill 工作区改为整块保留复用，不再逐层 synchronize/重分配；该循环在 WDDM 消费级显卡上曾以 `cudaErrorIllegalAddress` 形式暴露。
- Windows 批量 H2D 拷贝自动限制为每组 ≤8 份，避开曾复现越界错误的大批量提交窗口；可用 `CCCP_H2D_BATCH_MAX_COPIES` 调整。
- Kimi Prefill 与 MTP 的专家展开工作区改为统一块级收尾，不再泄漏进 Decode 阶段。
- 新增 `data/runtime/debug_env.txt` 诊断通道：无需终端即可给 serve 进程注入一次性环境变量（如 `CUDA_LAUNCH_BLOCKING=1`）。

离线包包含独立 Python、CPU/CUDA/AMD 运行环境及依赖，不包含模型权重，也不包含 CCCP 量化框架。

# CCCP Launcher 0.9.5

0.9.5 是离线 CUDA 启动、模型清单兼容性和动态上下文修正版。

- SM75 / SM86 / SM89 / SM90 / SM120 融合算子已预编译并随包提供；常见 NVIDIA 显卡不再依赖用户电脑现场编译。
- 修复部分 GLM 清单省略 `quant.vq` 时的 `KeyError`，从投影级 VQ 元数据严格推导并校验布局。
- 正常 GUI 启动不再写死上下文 token 上限；按模型声明上限动态扩展 KV。
- 修复 SM89 把模型声明的长上下文上限误判为当前上下文、从而在短提示启动时错误要求 FlashMLA 的问题；实际进入 sparse-only 区间时才执行能力门禁。
- RTX 4090/SM89 的 MoE Prefill 使用整批 VQ 展开后的 BF16 grouped GEMM，Decode 使用 packed 融合算子，避开 PyTorch 仅面向 SM90/SM100 的 grouped/rowwise FP8 接口限制，不回退逐 token GEMV。
- 配置库 Dense 卡片移除两个冗余标签。

离线包包含独立 Python、CPU/CUDA/AMD 运行环境及依赖，不包含模型权重，也不包含 CCCP 量化框架。

# 原生算子目录

`cpu_vq.cpp` 是 x86 公共 CPU 后端，`vq_gemv.cu` 与头文件是 CUDA 后端。

约定：

1. 算子直接消费量化索引和码本，不生成完整反量化权重矩阵。
2. p8–p16 的差异由公共后端处理，不复制模型专用 GEMV。
3. 精确路径默认开启；额外近似计算必须由显式环境变量启用。
4. Python 必须保留可读的参考实现；原生扩展不可用时自动回退。
5. 修改原生代码后要提升扩展缓存版本，并在 Linux 编译机运行数值对照测试。

当前三投影 CUDA 路径的能力键包含三个 projection 的 packed 格式、code dim、
codebook size 和 gated activation。SiTU 与带 clamp 的 SwiGLU 共享同一内核模板，
由调用参数选择数学公式；不得以 Kimi/DeepSeek 模型名注册两个内核。

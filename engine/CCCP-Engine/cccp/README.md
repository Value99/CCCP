# `cccp` 源码目录

这里是正式运行时，不存放模型权重、远端密码、临时跑分脚本或原始日志。

- `engine.py`、`launch.py`、`serve.py`：统一入口与资源规划。
- `model.py`、`dsv4model.py`、`kimi_model.py`：模型架构实现。
- `store.py`、`ramcache.py`：只读模型仓库与 RAM/VRAM 专家缓存。
- `grouped.py`、`cpuext.py`、`fusedext.py`：跨模型共用的算子调度。
- `csrc/`：原生 CPU/CUDA 实现；Python 层必须保留正确性回退。
- `chat_adapters/`：模型协议，不放数值计算。
- `configs/`：可发布的静态配置。

新增优化优先放进公共后端；仅当数学定义确实不同，才在模型文件中增加薄适配层。
所有新路径都要有无私有权重的单元测试，并保持 GLM、DeepSeek、Kimi 默认行为兼容。

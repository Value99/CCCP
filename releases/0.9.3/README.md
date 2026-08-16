# CCCP Launcher 0.9.3 应用入口

- 版本：0.9.3
- 平台：Windows x64
- 首次安装：请前往 [v0.9.3 Release](https://github.com/Value99/CCCP/releases/tag/v0.9.3)，下载 Offline Setup、parts 清单和全部 4 个分卷。

0.9.3 新增清单驱动的通用 Dense VQ 和架构配置驱动的 MTP。Qwen3.5 27B Dense 可直接完整加载，界面不会误要求专家配置或专家训练；DeepSeek-V4、Kimi K3 与 GLM-5.2 实模回归保持通过。发行包仍不包含模型权重或 CCCP 量化/训练框架。

已知范围：CPU 功能通过但不承诺达到性能目标；AMD 本轮未做真机性能复测；Windows CUDA 的 DeepSeek-V4 在 2052 token 以上需要当前离线包尚未内置的 FlashMLA，因此会明确报错而不会静默走慢速回退。

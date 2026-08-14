# CCCP Launcher 0.9.1 应用入口

- 版本：0.9.1
- 平台：Windows x64
- 独立更新文件：`CCCP-Launcher.exe`
- 首次安装：请前往 [v0.9.1 Release](https://github.com/Value99/CCCP/releases/tag/v0.9.1)，下载 Offline Setup、parts 清单和全部 4 个分卷。

0.9.1 修复严格路由把 Dense 层 0–2 误判为少于 top-k=8、组合 Gate+Up/Down Q4 专家单 token 解码布局分派错误，以及达到 `max_tokens` 后多算一次完整 decode。333.524 GiB GLM‑5.2 模型已完成 4096-token 路由扫描、热力图配置保存、811 专家加载与真实 GUI 流式生成回归。

发行附件不含模型，也不含 CCCP 量化/训练框架。首次用户不能只下载这个单独 EXE；它主要用于覆盖更新已有完整离线目录。

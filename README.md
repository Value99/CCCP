# WINUI-EXE — TPQ-Final 启动器

面向 **TPQ-Final** 的桌面启动器:深色原生窗口(pywebview/WebView2,打包为单个 EXE),
以"领域配置文件(Profile)"驱动 MoE 专家加载,提供聊天、配置、训练、API 服务四个选项卡。

> 本仓库**不修改 TPQ-Final 的任何文件**,只通过子进程 CLI / OpenAI HTTP / 约定文件三种外部方式集成。
> 集成契约与待 TPQ 侧开发的需求单见 [docs/INTERFACE.md](docs/INTERFACE.md)。

## 功能概览

| 选项卡 | 功能 |
|---|---|
| 聊天 Chat | 选择 profile 组合 → 发动 TPQ-Final → SSE 流式对话 |
| 配置 Profiles | 内置 角色扮演 / Python 代码 / 合同处理;显示专家数与占用体积;**多选组合按专家 key 去重实时重算**(重叠只计一次);内置 `drop` 占位专家自动路由到最相关专家;支持导入 YAML/JSON |
| 训练 Training | 语料经 CPU / disk 全量推理统计激活专家偏好;支持设定目标体积反向推荐专家子集;TPQ 未提供路由统计前为启发式估算(UI 显著标注) |
| API 服务 | OpenAI 兼容 `/v1/chat/completions` 代理 + profile / launch / training REST |
| 外壳 | pywebview 原生深色窗口(WebView2),失败时自动降级为浏览器;PyInstaller onefile `dist/TPQ-WinUI.exe` |

## 快速开始

```bash
# Python >= 3.10,Windows Git Bash
cd WINUI-EXE
python -m venv .venv && . .venv/Scripts/activate
pip install -r requirements.txt

# 启动(默认自动探测 ../TPQ-Final;原生窗口,失败降级浏览器)
python -m launcher.app --host 127.0.0.1 --port 8790
```

## 打包为桌面应用

```bash
bash scripts/build_app.sh    # 产出 dist/TPQ-WinUI.exe
```

## 验证

```bash
python -m pytest tests/ -q   # 单元测试(重叠算数/drop 路由/训练规划/导出 schema)
bash scripts/smoke.sh        # 端到端冒烟(不加载真模型)
```

## 目录结构

```
WINUI-EXE/
├── launcher/            # FastAPI 后端
│   ├── app.py           # 入口 / CLI
│   ├── profiles.py      # Profile 模型/注册表/重叠体积/drop 路由/导入
│   ├── tpq_adapter.py   # 与 TPQ-Final 的唯一集成层(不改动对方代码)
│   ├── chat.py          # OpenAI 兼容聊天代理(SSE 透传)
│   ├── training.py      # 语料扫描引擎 + 目标体积规划 + 导出
│   └── state.py         # data/ 轻量持久化
├── shell/               # pywebview 原生深色窗口外壳
├── webui/               # 深色 SPA(原生 HTML/JS/CSS,无构建链)
├── profiles/builtin/    # 内置领域配置文件(可被导入的同名 id 覆盖)
├── docs/INTERFACE.md    # ⇄ TPQ-Final 接口契约(I-0 落地;I-1~I-5 待 TPQ 开发)
├── tests/               # pytest 单元测试
├── scripts/             # build_app / smoke
└── packaging/           # PyInstaller spec
```

## Git 工作流

- 分支:`main`(可发布基线)、`feat/<scope>` 功能分支、`fix/<scope>` 修复分支。
- 提交:Conventional Commits(`feat:/fix:/docs:/chore:/refactor:/test:`),一个模块一个 commit。
- 合并:功能分支 → `main` 用 `--no-ff` 并写合并要点;里程碑打 tag(`v0.1.0` …)。

## 许可证

与 TPQ-Final 保持一致;本仓库代码 MIT。

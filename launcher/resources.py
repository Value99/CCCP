"""冻结(PyInstaller)与开发环境的资源路径兼容。

- 开发环境:以仓库根为基准(launcher/ 的上一级)。
- onefile 冻结:只读资源在 sys._MEIPASS;可写数据放 exe 同级 data/。
"""
from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def bundle_root() -> Path:
    """只读资源根(webui/ 前端、profiles/builtin 等被打包的内容)。"""
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def runtime_root() -> Path:
    """可写根:开发=仓库根;冻结=exe 所在目录。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return bundle_root()


def webui_static() -> Path:
    return bundle_root() / "webui"


def builtin_profile_dir() -> Path:
    return bundle_root() / "profiles" / "builtin"


def data_dir() -> Path:
    d = runtime_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_profile_dir() -> Path:
    """用户导入 profile 的落盘目录(可写;冻结时在 exe 同级)。"""
    d = runtime_root() / "profiles" / "user"
    d.mkdir(parents=True, exist_ok=True)
    return d


def detect_tpq_path() -> Path | None:
    """WINUI-EXE(或 exe)的兄弟目录 TPQ-Final;冻结时也允许上一级(开发布局)。"""
    for base in (runtime_root().parent, bundle_root().parent):
        cand = base / "TPQ-Final"
        if (cand / "tpq" / "__main__.py").exists():
            return cand
    return None

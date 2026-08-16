"""冻结(PyInstaller)与开发环境的资源路径兼容。

- 开发环境:以仓库根为基准(launcher/ 的上一级)。
- onefile 冻结:只读资源在 sys._MEIPASS;可写数据放 exe 同级 data/。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def bundle_root() -> Path:
    """只读资源根（冻结版中主要包含 webui/ 和中文文档）。"""
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


def data_dir() -> Path:
    d = runtime_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_profile_dir() -> Path:
    """用户导入 profile 的落盘目录(可写;冻结时在 exe 同级)。"""
    d = runtime_root() / "profiles" / "user"
    d.mkdir(parents=True, exist_ok=True)
    return d


def operator_cache_dir() -> Path:
    """跨发行目录复用的本机算子缓存。

    GPU 算子与机器、运行时和显卡架构绑定，不应跟随某个解压目录。Windows
    优先放入 LOCALAPPDATA，避免升级到新版本目录后再次编译；环境变量仅供
    自动化测试和高级部署覆盖。
    """
    configured = os.environ.get("CCCP_OPERATOR_CACHE_DIR", "").strip()
    if configured:
        directory = Path(configured).expanduser().resolve()
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA", "").strip():
        directory = (
            Path(os.environ["LOCALAPPDATA"]) / "CCCP-Launcher" / "operator-cache"
        ).resolve()
    else:
        directory = (data_dir() / "operator-cache").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def detect_engine_path() -> Path | None:
    """只发现发行目录内置的 CCCP Engine，不接受外部或旧布局。"""
    resolved = (runtime_root() / "engine" / "CCCP-Engine").resolve()
    if (resolved / "cccp" / "__main__.py").is_file():
        return resolved
    return None


def runtime_python_candidates(backend: str = "cpu") -> tuple[Path, ...]:
    """Return versioned runtime candidates for one isolated inference backend."""
    rr = runtime_root()
    if backend == "cpu":
        return (rr / "runtime" / "cpu" / "env" / "python.exe",)
    if backend == "cuda":
        return (rr / "runtime" / "cuda" / "env" / "python.exe",)
    if backend == "amd":
        return (rr / "runtime" / "amd" / "env" / "python.exe",)
    return ()


def detect_python_path(backend: str = "cpu") -> Path | None:
    """发现隔离的 CPU/CUDA/AMD Miniconda 环境，不依赖系统 Python。"""
    candidates = runtime_python_candidates(backend)
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def default_models_dir() -> Path:
    """发行目录固定的默认模型目录。"""
    directory = (runtime_root() / "models").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def detect_model_roots() -> list[Path]:
    return [default_models_dir()]

"""模型下载(ModelScope 默认 / HuggingFace 备用)与社区 profile 抓取。

- 内存任务表 + 后台线程;重启即清空(下载产物本身落盘在目标目录)。
- 下载 SDK 随离线 Miniconda 环境提供；冻结 EXE 委托该环境执行下载。
- 不触碰 CCCP-Engine 任何文件;能否直接被 CCCP 启动取决于产物是否为 CCCP 归档。
"""
from __future__ import annotations

import os
import importlib
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

from .resources import default_models_dir, detect_python_path, is_frozen
from .settings import Settings

Backend = Callable[["DownloadJob", Settings], str]


@dataclass
class DownloadJob:
    id: str
    repo: str
    source: str               # hf | modelscope
    target_dir: str
    revision: str = ""
    status: str = "running"   # running | done | failed
    message: str = ""
    error: str = ""
    result_path: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    def to_dict(self) -> dict:
        return {"id": self.id, "repo": self.repo, "source": self.source,
                "target_dir": self.target_dir, "revision": self.revision,
                "status": self.status, "message": self.message, "error": self.error,
                "result_path": self.result_path,
                "created_at": self.created_at, "finished_at": self.finished_at}


def default_target(settings: Settings, repo: str) -> Path:
    """空 target_dir 时固定落到发行目录的 models 文件夹。"""
    base = default_models_dir()
    tail = repo.rstrip("/").split("/")[-1] or "model"
    return base / tail


def safe_download_target(settings: Settings, repo: str, requested: str = "") -> Path:
    """Resolve a model target and keep it inside the bundled models folder."""
    base = default_models_dir().resolve()
    candidate = Path(requested).expanduser() if requested else default_target(settings, repo)
    if not candidate.is_absolute():
        candidate = base / candidate
    target = candidate.resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"下载目录必须位于内置 models 目录中: {base}") from exc
    if target == base:
        raise ValueError("下载目标必须是内置 models 目录下的独立模型文件夹")
    return target


def _hf_backend(job: DownloadJob, settings: Settings) -> str:
    if is_frozen():
        return _portable_download(job, settings)
    try:
        snapshot_download = importlib.import_module("huggingface_hub").snapshot_download
    except ImportError as e:
        raise RuntimeError("离线运行环境不完整：缺少内置 huggingface_hub，请修复发行包") from e
    ep = (settings.hf_endpoint or "").strip()
    if ep:
        os.environ["HF_ENDPOINT"] = ep  # 镜像端点,如 https://hf-mirror.com
    return snapshot_download(repo_id=job.repo, revision=job.revision or None,
                             local_dir=job.target_dir, max_workers=8)


def _modelscope_backend(job: DownloadJob, settings: Settings) -> str:
    if is_frozen():
        return _portable_download(job, settings)
    try:
        snapshot_download = importlib.import_module("modelscope").snapshot_download
    except ImportError as e:
        raise RuntimeError("离线运行环境不完整：缺少内置 modelscope，请修复发行包") from e
    kw = {"local_dir": job.target_dir}
    if job.revision:
        kw["revision"] = job.revision
    return snapshot_download(job.repo, **kw)


def _portable_download(job: DownloadJob, settings: Settings) -> str:
    """冻结 EXE 将大型可选下载 SDK 交给随包 Python，避免重复打入 EXE。"""
    python = detect_python_path()
    if python is None:
        raise RuntimeError("未找到随包 python/python.exe，无法使用在线下载")
    code = (
        "import importlib,json,os,sys;"
        "source,repo,target,revision,endpoint=sys.argv[1:6];"
        "os.environ.__setitem__('HF_ENDPOINT',endpoint) if endpoint else None;"
        "module=importlib.import_module('huggingface_hub' if source=='hf' else 'modelscope');"
        "kwargs={'local_dir':target};"
        "kwargs.update({'revision':revision} if revision else {});"
        "kwargs.update({'repo_id':repo} if source=='hf' else {});"
        "path=module.snapshot_download(repo if source!='hf' else None,**kwargs) "
        "if source!='hf' else module.snapshot_download(**kwargs);"
        "print(json.dumps(str(path),ensure_ascii=False))"
    )
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [
            str(python), "-c", code, job.source, job.repo, job.target_dir,
            job.revision, settings.hf_endpoint or "",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "下载子进程失败")
    try:
        return str(json.loads(proc.stdout.strip().splitlines()[-1]))
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(f"下载子进程未返回有效路径: {proc.stdout[-500:]}") from exc


class DownloadEngine:
    """线程式下载任务表(内存);backend 可注入以便离线测试。"""

    def __init__(self, settings: Settings, backend: Backend | None = None):
        self.settings = settings
        self._backend = backend
        self.jobs: dict[str, DownloadJob] = {}
        self._lock = threading.Lock()

    def _resolve(self, source: str) -> Backend:
        if self._backend:
            return self._backend
        return _hf_backend if source == "hf" else _modelscope_backend

    def submit(self, spec: dict) -> DownloadJob:
        repo = str(spec.get("repo") or "").strip()
        source = str(
            spec.get("source") or self.settings.default_download_source
        ).strip()
        if not repo:
            raise ValueError("repo 不能为空(如 deepseek-ai/DeepSeek-V4-Pro)")
        if source not in ("hf", "modelscope"):
            raise ValueError("source 必须是 hf 或 modelscope")
        target = safe_download_target(
            self.settings, repo, str(spec.get("target_dir") or "").strip()
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        job = DownloadJob(id=f"dl-{int(time.time())}-{len(self.jobs) + 1:03d}",
                          repo=repo, source=source, target_dir=str(target),
                          revision=str(spec.get("revision") or "").strip(),
                          message="排队下载中…")
        with self._lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job: DownloadJob) -> None:
        job.status = "running"
        job.message = f"从 {'HuggingFace' if job.source == 'hf' else 'ModelScope'} 下载 {job.repo} …"
        try:
            path = self._resolve(job.source)(job, self.settings)
            job.result_path = str(path)
            job.status = "done"
            job.message = "下载完成"
        except Exception as e:  # 库缺失/网络/鉴权 → 友好失败
            job.status = "failed"
            job.error = str(e)
            job.message = "下载失败"
        job.finished_at = time.time()

    def list(self) -> list[dict]:  # 新→旧
        return [j.to_dict() for j in sorted(self.jobs.values(),
                                            key=lambda x: x.created_at, reverse=True)]

    def get(self, jid: str) -> DownloadJob | None:
        return self.jobs.get(jid)

    def delete(self, jid: str) -> bool:
        j = self.jobs.get(jid)
        if j and j.status != "running":
            return self.jobs.pop(jid, None) is not None
        return False


# --------------------------------------------------------------------------
# 社区 profile(远程索引 + 文本下载;落盘导入交给 ProfileRegistry)
# --------------------------------------------------------------------------

async def fetch_index(url: str) -> list[dict]:
    """拉取社区索引;接受 {profiles:[…]} 或裸 […];每项至少需含 url。"""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise ValueError(f"拉取社区索引失败: {e}") from e
    entries = data.get("profiles", data) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError("社区索引格式错误(应为数组或 {profiles:[…]})")
    return [e for e in entries if isinstance(e, dict) and e.get("url")]


async def fetch_profile_text(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()
            return r.text
    except httpx.HTTPError as e:
        raise ValueError(f"下载 profile 失败: {e}") from e

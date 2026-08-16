"""CCCP 启动器入口:FastAPI 应用 + CLI。

职责:
- 装配 settings / ProfileRegistry / CCCPEngineAdapter / ChatProxy / TrainingEngine / AppState / DownloadEngine
- REST API + OpenAI 兼容 /v1/chat/completions
- 静态托管 webui/(浅色/跟随系统主题 SPA)
- 统一错误格式 {"error": {"code", "message"}}
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import ctypes
import hmac
import ipaddress
import shutil
import logging
import os
import secrets
import socket
import threading
import time
import webbrowser
import json
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .chat import ChatProxy
from .accelerators import detect_optional_accelerators, probe_backend
from .profiles import ProfileError, ProfileRegistry
from .resources import user_profile_dir, webui_static
from .settings import load_settings, Settings
from .state import AppState
from .cccp_adapter import (
    LaunchConfig, CCCPEngineAdapter, CCCPEngineError, _memory_status,
    discover_models, estimate_gpu_vram_plan, full_model_combination,
    inspect_model,
)
from .downloads import DownloadEngine, fetch_index, fetch_profile_text
from .logbuf import attach_ring_log, tail_lines
from .training import (
    CORPUS_DIR, TrainingEngine, delete_corpus, export_counts, export_profile,
    export_scores, list_corpus, save_corpus_file,
)
from .updates import UPDATE_DOWNLOAD_URLS, UpdateChecker, version_key

log = logging.getLogger("winui")
WEBUI_STATIC = webui_static()
USER_PROFILE_DIR = user_profile_dir()
_INSTANCE_MUTEX = None
MAX_PROFILE_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_CORPUS_UPLOAD_BYTES = 256 * 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024
UPLOAD_REQUESTS_PER_MINUTE = 12
_UPLOAD_LIMITS = {
    "/api/profiles/import": MAX_PROFILE_UPLOAD_BYTES,
    "/api/training/corpus": MAX_CORPUS_UPLOAD_BYTES,
}
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class _SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = max(1, int(limit))
        self.window_seconds = float(window_seconds)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True


def _trusted_request_host(host: str | None) -> bool:
    value = (host or "").strip().lower()
    if value in {"localhost", "testserver"}:
        return True
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return bool(address.is_loopback or address.is_private)


def _origin_identity(value: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return None
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError:
        return None
    return parsed.scheme.lower(), parsed.hostname.lower(), port


def _same_origin_request(request: Request, origin: str) -> bool:
    request_port = request.url.port or (443 if request.url.scheme == "https" else 80)
    expected = (request.url.scheme.lower(), (request.url.hostname or "").lower(), request_port)
    return _origin_identity(origin) == expected


async def _read_upload_limited(file: UploadFile, limit: int) -> bytes:
    declared_size = getattr(file, "size", None)
    if declared_size is not None and int(declared_size) > limit:
        raise HTTPException(status_code=413, detail=f"上传文件不能超过 {limit // 2**20} MiB")
    content = bytearray()
    while chunk := await file.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > limit:
            raise HTTPException(status_code=413, detail=f"上传文件不能超过 {limit // 2**20} MiB")
    return bytes(content)


def _acquire_desktop_instance() -> bool:
    """Windows 桌面版单实例保护；False 表示同一用户已有实例。"""
    global _INSTANCE_MUTEX
    if os.name != "nt":
        return True
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, "Local\\CCCP-Launcher-CPU-Launcher-v2")
    if not handle:
        return True  # 互斥量失败不阻断主程序，端口检测仍会保护后端。
    _INSTANCE_MUTEX = handle
    return int(kernel32.GetLastError()) != 183  # ERROR_ALREADY_EXISTS


def _available_ui_port(host: str, requested: int) -> int:
    """从 requested 开始选一个空闲 UI 端口。"""
    for port in range(requested, min(requested + 100, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"{requested} 起连续 100 个端口均被占用")


def downloads_default_hint(s: Settings) -> str:
    """模型下载固定落到发行目录内置 models 文件夹。"""
    from .resources import default_models_dir
    return str(default_models_dir())


def detect_hardware(settings: Settings | None = None) -> dict:
    """CUDA 可用性探测(nvidia-smi,零依赖)+ CPU/平台信息;任何失败降级。"""
    import os
    import platform
    import subprocess

    gpus: list[str] = []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            gpus = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.TimeoutExpired):
        pass
    total_gb = available_gb = 0.0
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_gb, available_gb = vm.total / 2**30, vm.available / 2**30
    except ImportError:
        pass
    disk = shutil.disk_usage(Path.cwd())
    optional = detect_optional_accelerators(
        settings, settings.cccp_engine_path if settings is not None else None
    )
    return {"cuda_available": bool(gpus), "gpus": gpus,
            "cpu_count": os.cpu_count() or 0,
            "platform": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "ram_total_gb": round(total_gb, 2),
            "ram_available_gb": round(available_gb, 2),
            "disk_free_gb": round(disk.free / 2**30, 2),
            **optional}


def _automatic_context(model_path: str, cache_gb: float, device: str,
                       settings: Settings) -> int:
    """按模型上限和当前设备剩余容量选择保守上下文，不接受前端覆盖。"""
    model = inspect_model(model_path) if model_path else None
    model_limit = max(64, (model.max_context if model else 0) or 32768)
    total_ram, available_ram = _memory_status()
    capacity = available_ram or total_ram
    if device in {"cuda", "amd"}:
        runtime = probe_backend(settings, device, settings.cccp_engine_path)
        capacity = float(
            runtime.get("device_available_memory_gb")
            or runtime.get("device_memory_gb")
            or capacity
        )
        if model is not None and capacity > 0:
            smallest = estimate_gpu_vram_plan(
                model,
                max_ctx=max(64, min(512, model_limit)),
                expert_cache_gb=cache_gb,
            )
            if (
                smallest.get("architecture") == "kimi_k3"
                and capacity < float(smallest["recommended_vram_gb"])
            ):
                # RAM-Dense Kimi is functional on consumer cards, but long
                # contexts amplify host projections and PCIe expert traffic.
                return max(64, min(512, model_limit))
            candidates = tuple(
                value for value in (4096, 2048, 1024, 512)
                if value <= model_limit
            ) or (max(64, model_limit),)
            for candidate in candidates:
                plan = estimate_gpu_vram_plan(
                    model,
                    max_ctx=candidate,
                    expert_cache_gb=cache_gb,
                )
                if capacity >= float(plan["minimum_vram_gb"]):
                    return candidate
            # 512 is the smallest normal GUI context.  Preflight will provide
            # the precise hard-working-set message if even this cannot fit.
            return max(64, min(512, model_limit))
    fixed = float(model.dense_gb if model else 0.0)
    # ``cache_gb`` is the compact expert RAM budget.  Subtracting it from
    # VRAM made every GPU launch choose 512 tokens whenever the selected
    # profile was larger than the card.  The engine independently sizes its
    # bounded VRAM hot cache after reserving Dense, context and full-batch
    # Prefill scratch, so GPU context admission must not count host RAM twice.
    expert_residency = 0.0 if device in {"cuda", "amd"} else float(cache_gb)
    device_reserve = 2.5 if device in {"cuda", "amd"} else 0.75
    remaining = capacity - fixed - expert_residency - device_reserve
    if remaining >= 6:
        selected = 4096
    elif remaining >= 3:
        selected = 2048
    elif remaining >= 1:
        selected = 1024
    else:
        selected = 512
    return max(64, min(selected, model_limit))


def _training_terminal_progress(job: dict) -> dict:
    """Map one route-scan job to the same progress contract as model loading."""
    status = str(job.get("status") or "pending")
    labels = {
        "pending": "正在准备 Token 扫描",
        "running": "正在扫描专家路由",
        "done": "Token 扫描已完成",
        "failed": "Token 扫描失败",
        "cancelled": "Token 扫描已停止",
    }
    states = {
        "pending": "loading", "running": "loading", "done": "ready",
        "failed": "error", "cancelled": "idle",
    }
    finished_at = float(job.get("finished_at") or 0.0)
    created_at = float(job.get("created_at") or time.time())
    elapsed = max(0.0, (finished_at or time.time()) - created_at)
    return {
        "state": states.get(status, "idle"),
        "percent": round(max(0.0, min(1.0, float(job.get("progress") or 0.0))) * 100),
        "phase": "token-route-scan",
        "label": labels.get(status, "Token 扫描"),
        "detail": str(job.get("message") or "等待扫描状态"),
        "elapsed_s": round(elapsed, 1),
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    registry = ProfileRegistry(settings.model_roots, USER_PROFILE_DIR)
    adapter = CCCPEngineAdapter(settings)
    chat = ChatProxy(adapter)
    training = TrainingEngine(
        registry,
        default_layers=settings.model_layers,
        default_experts_per_layer=settings.model_experts_per_layer,
        default_expert_size_mb=settings.default_expert_size_mb,
        engine_root=settings.cccp_engine_path,
        cpu_python=settings.cpu_python_path or settings.python_path,
    )
    downloads = DownloadEngine(settings)
    state = AppState()
    updates = UpdateChecker(__version__, lambda: settings.skipped_update_version)
    upload_limiter = _SlidingWindowLimiter(UPLOAD_REQUESTS_PER_MINUTE, 60.0)

    app = FastAPI(title="CCCP 启动器", version=__version__)
    app.state.settings = settings
    app.state.registry = registry
    app.state.adapter = adapter
    app.state.chat = chat
    app.state.training = training
    app.state.downloads = downloads
    app.state.app_state = state
    app.state.updates = updates
    app.state.terminal_training_job_id = None

    @app.middleware("http")
    async def _disable_webui_cache(request: Request, call_next):
        """Avoid an old frozen app.js continuing to send a model path.

        Desktop WebView2 and the browser fallback both keep a normal HTTP
        cache.  The launcher is replaced in-place during an upgrade, so stale
        UI assets must not outlive the backend that serves them.
        """
        path = request.url.path
        if request.method.upper() in _UNSAFE_METHODS and path.startswith(("/api/", "/v1/")):
            if not _trusted_request_host(request.url.hostname):
                return JSONResponse(
                    {"error": {"code": "untrusted-host", "message": "请求主机不受信任"}},
                    status_code=403,
                )
            origin = request.headers.get("origin")
            fetch_site = request.headers.get("sec-fetch-site", "").lower()
            if fetch_site == "cross-site" or (origin and not _same_origin_request(request, origin)):
                return JSONResponse(
                    {"error": {"code": "cross-site-request", "message": "已拒绝跨站写入请求"}},
                    status_code=403,
                )
        upload_limit = _UPLOAD_LIMITS.get(path) if request.method.upper() == "POST" else None
        if upload_limit is not None:
            client_host = request.client.host if request.client else "unknown"
            if not upload_limiter.allow(f"{client_host}:{path}"):
                return JSONResponse(
                    {"error": {"code": "upload-rate-limit", "message": "上传过于频繁，请稍后再试"}},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )
            declared_length = request.headers.get("content-length")
            if declared_length:
                try:
                    oversized = int(declared_length) > upload_limit + MULTIPART_OVERHEAD_BYTES
                except ValueError:
                    oversized = True
                if oversized:
                    return JSONResponse(
                        {"error": {"code": "upload-too-large", "message": f"上传内容不能超过 {upload_limit // 2**20} MiB"}},
                        status_code=413,
                    )
        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if (
            content_type.lower().startswith("application/json")
            and "charset=" not in content_type.lower()
        ):
            # PowerShell 5.1 interprets an unlabelled JSON response through
            # the active ANSI code page.  Model paths may contain Chinese
            # directory names, so make the RFC-mandated UTF-8 encoding
            # explicit for command-line/API clients as well as browsers.
            response.headers["Content-Type"] = "application/json; charset=utf-8"
        if request.url.path in {
            "/", "/index.html", "/app.js", "/style.css",
            "/images/icon.png", "/images/banner.jpg",
        }:
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        adapter.stop()

    # -- 统一错误 --
    @app.exception_handler(ProfileError)
    async def _profile_err(_: Request, exc: ProfileError):
        return JSONResponse({"error": {"code": "profile", "message": str(exc)}}, status_code=400)

    @app.exception_handler(CCCPEngineError)
    async def _cccp_err(_: Request, exc: CCCPEngineError):
        return JSONResponse({"error": {"code": "cccp", "message": str(exc)}}, status_code=409)

    @app.exception_handler(ValueError)
    async def _val_err(_: Request, exc: ValueError):
        return JSONResponse({"error": {"code": "bad-request", "message": str(exc)}}, status_code=400)

    # ------------------------------------------------------------------
    # 健康 / 设置 / 模型
    # ------------------------------------------------------------------
    @app.get("/api/health")
    async def api_health():
        cccp = await adapter.health()
        return {"winui": "ok", "version": __version__,
                "cccp_available": adapter.available(), "cccp": cccp,
                "theme_mode": settings.theme_mode}

    @app.get("/api/settings")
    async def api_get_settings():
        return settings.__dict__

    @app.post("/api/settings")
    async def api_set_settings(req: Request):
        body = await req.json()
        if (
            adapter.instance
            and "cccp_api_key" in body
            and str(body.get("cccp_api_key") or "").strip() != settings.cccp_api_key
        ):
            raise CCCPEngineError("模型运行期间不能更换 API Key；请先停止模型再保存")
        settings.update(body)
        settings.save()
        return settings.__dict__

    @app.post("/api/settings/api-key/generate")
    async def api_generate_api_key():
        return {"api_key": "cccp_" + secrets.token_urlsafe(32)}

    # ------------------------------------------------------------------
    # 更新检测：网络访问仅发生在守护线程中，不阻塞 UI 或其他 API。
    # ------------------------------------------------------------------
    @app.get("/api/update/status")
    async def api_update_status():
        return app.state.updates.snapshot()

    @app.post("/api/update/check")
    async def api_update_check(req: Request):
        body = await req.json() if req.headers.get("content-length") else {}
        return app.state.updates.start(force=bool(body.get("force", False)))

    @app.post("/api/update/ignore")
    async def api_update_ignore(req: Request):
        version = str((await req.json()).get("version") or "").strip()
        version_key(version)
        current = app.state.updates.snapshot()
        if version != current.get("latest_version"):
            raise ValueError("只能忽略本次检测到的更新版本")
        settings.skipped_update_version = version
        settings.save()
        return app.state.updates.refresh_ignored_state()

    @app.post("/api/update/open")
    async def api_update_open(req: Request):
        source = str((await req.json()).get("source") or "").strip()
        current = app.state.updates.snapshot()
        if source != current.get("source") or source not in UPDATE_DOWNLOAD_URLS:
            raise ValueError("更新来源无效，请重新检查更新")
        url = UPDATE_DOWNLOAD_URLS[source]
        threading.Thread(
            target=webbrowser.open, args=(url,), daemon=True, name="cccp-update-page"
        ).start()
        return {"ok": True, "url": url, "source": source}

    @app.get("/api/models")
    async def api_models():
        found = discover_models(settings.model_roots)
        return {
            "models": [m.__dict__ for m in found],
            "roots": settings.model_roots,
            "hint": "在设置里添加 model_roots(含 cccp.json 的上级目录)以发现模型",
        }

    @app.delete("/api/models")
    async def api_delete_model(req: Request):
        """只允许删除扫描结果中的一个完整模型目录，避免任意路径删除。"""
        body = await req.json()
        requested_text = str(body.get("path") or "").strip()
        if not requested_text:
            raise ValueError("缺少模型目录")
        requested = Path(requested_text).expanduser()
        try:
            target = requested.resolve(strict=True)
        except OSError as exc:
            raise ValueError("模型目录不存在") from exc
        discovered = {Path(model.path).resolve(): model for model in discover_models(settings.model_roots)}
        model = discovered.get(target)
        if model is None or not (target / "cccp.json").is_file():
            raise ValueError("只能删除模型库当前扫描到的 CCCP 模型目录")
        if adapter.instance and Path(adapter.instance.model).resolve() == target:
            raise CCCPEngineError("该模型正在运行，请先停止推理实例再删除")
        long_target = str(target)
        if os.name == "nt" and not long_target.startswith("\\\\?\\"):
            long_target = "\\\\?\\" + long_target
        shutil.rmtree(long_target)
        return {"ok": True, "deleted": str(target), "name": model.name}

    # ------------------------------------------------------------------
    # 系统 / 终端(v0.5):运行环境探测 + 日志查看
    # ------------------------------------------------------------------
    @app.get("/api/system")
    async def api_system():
        hw = detect_hardware(settings)
        return {**hw, "default_device": settings.default_device,
                "automatic_runtime_tuning": True}

    @app.get("/api/terminal/app")
    async def api_terminal_app():
        return {"lines": tail_lines(300)}

    @app.get("/api/terminal/cccp")
    async def api_terminal_cccp():
        return {"lines": adapter.tail_log(400).splitlines()}

    @app.get("/api/terminal")
    async def api_terminal_combined():
        """单一终端窗口：结构化状态 + CCCP 输出 + 启动器日志。"""
        health = await adapter.health()
        progress = adapter.loading_progress(health)
        instance = adapter.instance
        preflight = adapter.last_preflight
        focused_job = (
            training.get(app.state.terminal_training_job_id)
            if app.state.terminal_training_job_id else None
        )
        training_job = focused_job.to_dict() if focused_job else None
        if training_job is not None:
            progress = _training_terminal_progress(training_job)
        header = [
            "===== CCCP 动态专家启动器 · 实时状态 =====",
            f"状态: {progress['label']} ({progress['percent']}%)",
            f"阶段: {progress['phase']} · {progress['detail']} · 已用时 {progress['elapsed_s']:.1f}s",
        ]
        if training_job is not None:
            header.extend([
                f"训练任务: {training_job['id']} · 模式: {'强制硬盘' if training_job['mode'] == 'disk' else '自动高速/容量降级'}",
                f"模型: {training_job.get('model_name') or training_job.get('model_path') or '未记录'}",
                f"语料: {', '.join(training_job.get('corpus_files') or [])}",
                f"Token: {int(training_job.get('processed_tokens') or 0):,} / {int(training_job.get('token_budget') or 0):,} · {int(training_job.get('prefill_block_tokens') or 4096):,} token/块",
            ])
        if instance:
            header.extend([
                f"PID: {instance.pid} · API: {instance.base_url}",
                f"模型 ID: {instance.served_model_name}",
                f"权重目录: {instance.model}",
                f"配置: {'全量专家（无路由限制）' if instance.full_model else ', '.join(instance.profiles)}",
            ])
        if preflight and preflight.get("memory"):
            memory = preflight["memory"]
            header.append(
                "内存: 配置总驻留 "
                f"{memory['configuration_source_resident_gb']:.2f} GiB · "
                f"运行估算 {memory['total_estimate_gb']:.2f} GiB · "
                f"设备上限 {memory['device_capacity_gb']:.2f} GiB"
            )
        cccp_lines = [
            line for line in adapter.tail_log(400).splitlines()
            if '"GET /health ' not in line
        ]
        app_lines = tail_lines(180)
        lines = header
        if training_job is not None:
            lines.extend([
                "", "===== Token 路由扫描 =====",
                training_job.get("message") or "等待扫描输出",
                (
                    f"命中专家: {len(training_job.get('counts') or {}) if training_job.get('counts') else int(training_job.get('activated_experts') or 0):,} · "
                    f"路由观察: {int(training_job.get('route_observations') or 0):,}"
                ),
            ])
        if training_job is None:
            lines.extend(["", "===== CCCP 进程输出 ====="])
            lines.extend(cccp_lines or ["CCCP 未在运行 / 暂无输出"])
        lines.extend(["", "===== 启动器日志 ====="])
        lines.extend(app_lines or ["暂无启动器日志"])
        return {
            "lines": lines,
            "progress": progress,
            "health": health,
            "instance": instance.__dict__ if instance else None,
            "preflight": preflight,
            "training_job": training_job,
        }

    # ------------------------------------------------------------------
    # 社区(profile 下载 / Discord 链接)(v0.3)
    # ------------------------------------------------------------------
    @app.get("/api/community/config")
    async def api_community_config():
        return {"discord_url": settings.discord_url,
                "modelscope_profile_url": settings.modelscope_profile_url,
                "index_url": settings.community_index_url}

    @app.get("/api/community/profiles")
    async def api_community_profiles():
        url = (settings.community_index_url or "").strip()
        if not url:
            return {"profiles": [], "note": "当前版本未内置社区配置索引"}
        return {"profiles": await fetch_index(url)}

    @app.post("/api/community/profiles/install")
    async def api_community_install(req: Request):
        url = str((await req.json()).get("url") or "").strip()
        if not url:
            raise ProfileError("url 不能为空")
        text = await fetch_profile_text(url)
        p = registry.import_text(text, filename=url.rsplit("/", 1)[-1] or "profile.yaml")
        return {"ok": True, "profile": p.to_dict()}

    # ------------------------------------------------------------------
    # 模型下载(HuggingFace / ModelScope)(v0.3)
    # ------------------------------------------------------------------
    @app.post("/api/models/download")
    async def api_model_download(req: Request):
        job = downloads.submit(await req.json())
        return {"ok": True, "job": job.to_dict()}

    @app.get("/api/models/download/jobs")
    async def api_download_jobs():
        return {"jobs": downloads.list(),
                "default_dir": downloads_default_hint(settings)}

    @app.get("/api/models/download/jobs/{jid}")
    async def api_download_job(jid: str):
        j = downloads.get(jid)
        if not j:
            raise ValueError(f"下载任务不存在: {jid}")
        return j.to_dict()

    @app.delete("/api/models/download/jobs/{jid}")
    async def api_download_delete(jid: str):
        return {"deleted": downloads.delete(jid)}

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------
    @app.get("/api/profiles")
    async def api_profiles():
        registry.model_roots = [Path(root).expanduser() for root in settings.model_roots]
        registry.reload()
        models = discover_models(settings.model_roots)
        model_by_fingerprint = {model.manifest_sha256: model for model in models}
        profiles = []
        for profile in registry.list():
            item = profile.to_dict()
            matched = model_by_fingerprint.get(str(profile.meta.get("model_manifest_sha256") or ""))
            item["model_available"] = bool(matched and matched.complete)
            item["matched_model_path"] = matched.path if matched else ""
            item["model_status"] = (
                "可用" if matched and matched.complete
                else f"缺少对应模型：{profile.meta.get('model_name') or '未知模型'}"
            )
            profiles.append(item)
        return {"profiles": profiles,
                "selected": state.data.get("selected_profiles", [])}

    @app.get("/api/profiles/{pid}")
    async def api_profile_detail(pid: str):
        return registry.require(pid).to_dict(with_experts=True)

    @app.get("/api/profiles/{pid}/export")
    async def api_profile_export(pid: str):
        """下载完整、可再次导入和分享的领域专家配置。"""
        profile = registry.require(pid)
        payload = profile.to_dict(with_experts=True)
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        return StreamingResponse(
            iter([body]),
            media_type="application/json; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{profile.id}.json"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post("/api/profiles/combine")
    async def api_combine(req: Request):
        body = await req.json()
        ids = [str(x) for x in body.get("ids") or []]
        if not ids:
            raise ProfileError("ids 不能为空")
        combo = registry.combine(ids)
        out = combo.to_dict()
        out["per_profile"] = [
            {"id": p.id, "memory_gb": p.memory_gb, "expert_count": p.expert_count,
             "configuration_resident_gib": p.meta.get("configuration_resident_gib"),
             "model_name": p.meta.get("model_name"),
             "model_version": p.meta.get("model_version"),
             "model_total_bytes": p.meta.get("model_total_bytes"),
             "model_total_gib": p.meta.get("model_total_gib"),
             "model_manifest_sha256": p.meta.get("model_manifest_sha256"),
             "calibrated": p.calibrated}
            for p in (registry.get(i) for i in ids) if p
        ]
        return out

    @app.post("/api/profiles/select")
    async def api_select(req: Request):
        body = await req.json()
        ids = [str(x) for x in body.get("ids") or []]
        state.set_selected_profiles(ids)
        return {"ok": True, "selected": ids}

    @app.post("/api/profiles/import")
    async def api_import(req: Request, file: UploadFile | None = None):
        """两种导入:multipart 文件上传,或 JSON {yaml_text|json_text}。"""
        if file is not None:
            text = (await _read_upload_limited(file, MAX_PROFILE_UPLOAD_BYTES)).decode(
                "utf-8", errors="replace"
            )
            p = registry.import_text(text, filename=file.filename or "profile.yaml")
        else:
            body = await req.json()
            text = str(body.get("yaml_text") or body.get("json_text") or "")
            if not text:
                raise ProfileError("空内容")
            p = registry.import_text(text, filename="profile.yaml")
        return {"ok": True, "profile": p.to_dict()}

    @app.patch("/api/profiles/{pid}")
    async def api_update_profile(pid: str, req: Request):
        """只修改可分享元数据；专家编号、体积与模型指纹原样保留。"""
        profile = registry.require(pid)
        body = await req.json()
        name = str(body.get("name") or "").strip()
        if not name:
            raise ProfileError("配置名称不能为空")
        if len(name) > 80:
            raise ProfileError("配置名称不能超过 80 个字符")
        description = str(body.get("description") or "").strip()
        if len(description) > 1000:
            raise ProfileError("配置说明不能超过 1000 个字符")
        updated = registry.update_metadata(
            pid, name=name, description=description
        )
        return {"ok": True, "profile": updated.to_dict()}

    @app.delete("/api/profiles/{pid}")
    async def api_delete_profile(pid: str):
        registry.delete(pid)
        selected = [
            item for item in state.data.get("selected_profiles", [])
            if item != pid
        ]
        state.set_selected_profiles(selected)
        return {"ok": True}

    # ------------------------------------------------------------------
    # 启动 / 停止 / 状态
    # ------------------------------------------------------------------
    def _launch_cfg(body: dict, ids: list[str], combo) -> LaunchConfig:
        device = str(body.get("device") or settings.default_device)
        cpu_compile = "q4" if device == "cpu" else "auto"
        # 全部配置专家常驻且都可路由。Q4 只编译每层语料最热的 8 个执行页，
        # 其余保持紧凑 VQ 原生页；这是数据布局优化，不改变专家集合/top-k。
        if cpu_compile == "q4":
            by_layer: dict[int, list] = {}
            for expert in combo.union.values():
                by_layer.setdefault(expert.layer, []).append(expert)
            hot = [
                expert
                for experts in by_layer.values()
                for expert in sorted(
                    experts,
                    key=lambda item: (-item.route_count, -item.route_score, item.key),
                )[:8]
            ]
            hot_source_gib = sum(expert.size_mb for expert in hot) / 1024.0
            required_cache = combo.memory_gb + hot_source_gib * 1.75
        else:
            required_cache = combo.memory_gb
        automatic_cache = max(0.25, required_cache)
        model_path = str(body.get("model_path") or "")
        if (
            combo.profile_ids == ["__full_model__"]
            and str(body.get("profile_mode") or settings.default_profile_mode) == "mapped"
        ):
            model = inspect_model(model_path)
            total_ram, available_ram = _memory_status()
            capacities = [value for value in (total_ram, available_ram) if value > 0]
            usable = max(0.25, min(capacities) - model.dense_gb - 1.0) if capacities else 0.25
            automatic_cache = round(min(combo.memory_gb, usable), 3)
        return LaunchConfig(
            model_path=model_path,
            profiles=ids,
            combination=combo,
            port=int(body.get("port") or settings.api_port_alloc_start),
            host=str(body.get("host") or "127.0.0.1"),
            served_model_name=str(body.get("served_model_name") or "winui-model"),
            profile_mode=str(body.get("profile_mode") or settings.default_profile_mode),
            device=device,
            cache_gb=automatic_cache,
            vram_gb=body.get("vram_gb"),
            dense_residency=str(body.get("dense_residency") or "auto"),
            cpu_compile=cpu_compile,
            extreme=bool(body.get("extreme", False)),
            max_ctx=_automatic_context(model_path, automatic_cache, device, settings),
            cpu_threads=0,
            memory_limit_gb=0.0,
        )

    def _launch_selection(body: dict):
        ids = [str(x) for x in body.get("profile_ids") or []]
        if body.get("full_model"):
            model_path = str(body.get("model_path") or "").strip()
            if not model_path:
                raise CCCPEngineError("请先选择要全量加载的模型")
            return [], full_model_combination(model_path), True
        if not ids:
            raise ProfileError("请至少选择一个配置，或选择全量加载模型")
        return ids, registry.combine(ids), False

    @app.post("/api/launch")
    async def api_launch(req: Request):
        body = await req.json()
        ids, combo, full_model = _launch_selection(body)
        cfg = _launch_cfg(body, ids, combo)
        if not cfg.model_path:
            raise CCCPEngineError("缺少 model_path(先在「模型库」里下载/扫描并选择)")
        if body.get("dry_run_only"):
            return {"dry_run": adapter.dry_run(cfg)}
        # 设备选择属于用户偏好。即使本次模型稍后因容量或算子问题退出，
        # 下次打开也应保留用户实际点击启动时选择的后端。
        if settings.default_device != cfg.device:
            settings.default_device = cfg.device
            try:
                settings.save()
            except OSError as exc:
                log.warning("无法保存上次使用的推理设备 %s: %s", cfg.device, exc)
        inst = adapter.launch(cfg)
        app.state.terminal_training_job_id = None
        state.set_selected_profiles(ids)
        state.record_launch(
            cfg.model_path, ids, cfg.port, full_model=full_model
        )
        return {"ok": True, "instance": inst.__dict__, "combination": combo.to_dict(),
                "preflight": adapter.last_preflight,
                "tip": "轮询 /api/launch/status 直至 ready=true"}

    @app.post("/api/launch/dry-run")
    async def api_launch_dry(req: Request):
        body = await req.json()
        ids, combo, _ = _launch_selection(body)
        cfg = _launch_cfg(body, ids, combo)
        return adapter.dry_run(cfg)

    @app.post("/api/launch/preflight")
    async def api_launch_preflight(req: Request):
        body = await req.json()
        ids, combo, _ = _launch_selection(body)
        return adapter.preflight(_launch_cfg(body, ids, combo))

    @app.post("/api/launch/stop")
    async def api_launch_stop():
        return {"stopped": adapter.stop()}

    @app.get("/api/launch/status")
    async def api_launch_status():
        h = await adapter.health()
        return {"instance": adapter.instance.__dict__ if adapter.instance else None,
                "health": h, "progress": adapter.loading_progress(h),
                "preflight": adapter.last_preflight,
                "log_tail": adapter.tail_log(80),
                "last_launch": state.data.get("last_launch")}

    # ------------------------------------------------------------------
    # 聊天代理(含 OpenAI 兼容 /v1)
    # ------------------------------------------------------------------
    def _require_openai_key(req: Request) -> None:
        if not req.url.path.startswith("/v1") or not settings.cccp_api_key:
            return
        supplied = req.headers.get("Authorization", "")
        expected = f"Bearer {settings.cccp_api_key}"
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(
                status_code=401,
                detail="Incorrect API key provided.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/api/chat/models")
    @app.get("/v1/models")
    async def api_chat_models(req: Request):
        _require_openai_key(req)
        return await chat.models()

    @app.get("/v1/model-spec")
    async def api_model_spec(req: Request):
        _require_openai_key(req)
        return await chat.contract_get("model-spec")

    @app.get("/v1/expert-bytes")
    async def api_expert_bytes(req: Request):
        _require_openai_key(req)
        return await chat.contract_get("expert-bytes")

    @app.get("/v1/expert-stats")
    async def api_expert_stats(req: Request):
        _require_openai_key(req)
        return await chat.contract_get("expert-stats")

    @app.post("/v1/expert-stats/reset")
    async def api_expert_stats_reset(req: Request):
        _require_openai_key(req)
        return await chat.reset_expert_stats()

    @app.post("/api/chat/completions")
    @app.post("/v1/chat/completions")
    async def api_chat_completions(req: Request):
        _require_openai_key(req)
        payload = await req.json()
        payload.pop("profile_ids", None)  # profile_ids 仅作上下文标注
        if payload.get("stream", True):
            return StreamingResponse(
                chat.completions_stream(payload),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await chat.completions_once(payload)

    @app.get("/api/chat/metrics")
    async def api_chat_metrics(request_id: str | None = None):
        """CCCP 真实 KV/prefill/TTFT/tok/s；不含聊天正文。"""
        return {"metrics": adapter.latest_chat_metrics(request_id=request_id)}

    # -- 会话持久化 --
    @app.get("/api/chat/sessions")
    async def api_chat_sessions():
        return {"sessions": state.list_sessions()}

    @app.get("/api/chat/sessions/{sid}")
    async def api_chat_session_get(sid: str):
        s = state.get_session(sid)
        if not s:
            raise ValueError(f"会话不存在: {sid}")
        return s

    @app.post("/api/chat/sessions/{sid}")
    async def api_chat_session_save(sid: str, req: Request):
        body = await req.json()
        state.save_session(sid, str(body.get("title") or sid),
                            list(body.get("messages") or []))
        return {"ok": True}

    @app.delete("/api/chat/sessions/{sid}")
    async def api_chat_session_delete(sid: str):
        return {"deleted": state.delete_session(sid)}

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    @app.post("/api/training/corpus")
    async def api_corpus_upload(file: UploadFile):
        info = save_corpus_file(
            file.filename or "corpus.txt",
            await _read_upload_limited(file, MAX_CORPUS_UPLOAD_BYTES),
        )
        return {"ok": True, "file": info}

    @app.get("/api/training/corpus")
    async def api_corpus_list():
        return {"files": list_corpus()}

    @app.delete("/api/training/corpus/{name}")
    async def api_corpus_delete(name: str):
        return {"deleted": delete_corpus(name)}

    @app.post("/api/training/jobs")
    async def api_train_submit(req: Request):
        body = await req.json()
        model_path = str(body.get("model_path") or "")
        if not model_path:
            raise ValueError("请选择训练所对应的模型")
        model = inspect_model(model_path)
        if not model.complete:
            raise ValueError("模型不完整: " + "；".join(model.errors))
        if not getattr(model, "supports_route_training", True):
            raise ValueError(
                "该模型是 Dense VQ 架构，不包含动态专家；无需也不能进行"
                "语料路由扫描。请直接在配置库选择“完整 Dense 模型”启动。"
            )
        max_configuration_gib = round(model.dense_gb + model.expert_gb, 3)
        expert_count = max(1, model.expert_layer_count * model.experts_per_layer)
        body.update({
            "model_name": model.name,
            "model_version": model.model_version,
            "model_format": model.model_format,
            "model_manifest_sha256": model.manifest_sha256,
            "model_total_bytes": model.total_bytes,
            "model_total_gib": round(model.total_bytes / 2**30, 6),
            "model_top_k": model.top_k,
            "model_max_context": model.max_context,
            "layers": model.layers,
            "expert_layers": model.expert_layers,
            "experts_per_layer": model.experts_per_layer,
            "expert_size_mb": model.expert_gb * 1024.0 / expert_count,
            "dense_without_shared_gib": model.dense_without_shared_gb,
            "shared_expert_gib": model.shared_expert_gb,
            "fixed_model_gib": model.dense_gb,
            "model_max_configuration_gib": max_configuration_gib,
        })
        job = training.submit(body)
        job_payload = job.to_dict()
        app.state.terminal_training_job_id = str(job_payload.get("id") or getattr(job, "id", ""))
        return {"ok": True, "job": job_payload}

    @app.get("/api/training/jobs")
    async def api_train_list():
        return {"jobs": training.list()}

    @app.get("/api/training/jobs/{jid}")
    async def api_train_detail(jid: str):
        job = training.get(jid)
        if not job:
            raise ValueError(f"任务不存在: {jid}")
        return job.to_dict(with_counts=True)

    @app.delete("/api/training/jobs/{jid}")
    async def api_train_delete(jid: str):
        return {"deleted": training.delete(jid)}

    @app.post("/api/training/jobs/{jid}/cancel")
    async def api_train_cancel(jid: str):
        return {"cancelled": training.cancel(jid)}

    @app.post("/api/training/jobs/{jid}/plan")
    async def api_train_plan(jid: str, req: Request):
        body = await req.json()
        coverage_percent = float(body.get("coverage_percent") or 0.0)
        if not 1.0 <= coverage_percent <= 100.0:
            raise ValueError("coverage_percent 必须在 1 到 100 之间")
        job = training.replan(jid, coverage_percent / 100.0)
        return {"ok": True, "job": job.to_dict(with_counts=True)}

    @app.get("/api/training/jobs/{jid}/export")
    async def api_train_export(
        jid: str, kind: str = "scores", name: str = "", description: str = ""
    ):
        job = training.get(jid)
        if not job or job.status != "done":
            raise ValueError("任务未完成或不存在")
        if kind == "scores":
            return export_scores(job)
        if kind == "counts":
            return export_counts(job)
        if kind == "profile":
            if not str(name).strip() or not str(description).strip():
                raise ValueError("导出专家配置前请填写配置名称和介绍")
            return export_profile(job, name=name, description=description)
        raise ValueError("kind 必须是 scores|counts|profile")

    @app.post("/api/training/jobs/{jid}/register")
    async def api_train_register(jid: str, req: Request):
        """把训练产物注册到对应模型目录的 profiles/。"""
        job = training.get(jid)
        if not job or job.status != "done":
            raise ValueError("任务未完成或不存在")
        body = await req.json() if req.headers.get("content-length") else {}
        name = str(body.get("name") or "").strip()
        description = str(body.get("description") or "").strip()
        if not name:
            raise ValueError("请先填写配置名称")
        if not description:
            raise ValueError("请填写配置介绍，说明适用的任务或角色方向")
        if len(name) > 80:
            raise ValueError("配置名称不能超过 80 个字符")
        if len(description) > 1000:
            raise ValueError("配置描述不能超过 1000 个字符")
        data = export_profile(
            job,
            name=name,
            description=description,
        )
        p = registry.register_for_model(data, job.model_path)
        training.mark_registered(jid, p.id, name, description)
        return {"ok": True, "profile": p.to_dict()}

    # ------------------------------------------------------------------
    # API 信息服务(第四个选项卡)
    # ------------------------------------------------------------------
    @app.get("/api/service/info")
    async def api_service_info(req: Request):
        base = f"http://{req.base_url.hostname}:{req.base_url.port or 80}"
        inst = adapter.instance
        return {
            "base_url": base,
            "openai_endpoint": f"{base}/v1/chat/completions",
            "auth": bool(settings.cccp_api_key),
            "served_model": inst.served_model_name if inst else None,
            "engine_ready": (await adapter.health()).get("ready", False),
            "endpoints": {
                "openai": [
                    "GET /v1/models", "POST /v1/chat/completions",
                    "GET /v1/model-spec", "GET /v1/expert-bytes",
                    "GET /v1/expert-stats", "POST /v1/expert-stats/reset",
                ],
                "profiles": ["GET /api/profiles", "POST /api/profiles/combine",
                             "POST /api/profiles/import", "GET /api/profiles/{id}/export",
                             "DELETE /api/profiles/{id}"],
                "launch": ["POST /api/launch", "POST /api/launch/stop",
                           "GET /api/launch/status", "POST /api/launch/dry-run"],
                "training": ["POST /api/training/jobs", "GET /api/training/jobs",
                             "GET /api/training/jobs/{id}/export"],
                "community": ["GET /api/community/config", "GET /api/community/profiles",
                              "POST /api/community/profiles/install"],
                "downloads": ["POST /api/models/download", "GET /api/models/download/jobs",
                              "DELETE /api/models/download/jobs/{id}"],
                "system": ["GET /api/system", "GET /api/terminal"],
            },
            "curl_example": (
                f'curl -N {base}/v1/chat/completions \\\n'
                '  -H "Content-Type: application/json" \\\n'
                + ('  -H "Authorization: Bearer YOUR_CCCP_API_KEY" \\\n' if settings.cccp_api_key else '')
                + f'  -d \'{{"model": "{inst.served_model_name if inst else "winui-model"}", "stream": true,\n'
                '       "messages": [{"role": "user", "content": "hello"}]}\''
            ),
        }

    # ------------------------------------------------------------------
    # 静态前端
    # ------------------------------------------------------------------
    if WEBUI_STATIC.is_dir():
        app.mount("/", StaticFiles(directory=str(WEBUI_STATIC), html=True), name="webui")
    else:
        @app.get("/")
        async def _no_frontend():
            return {"winui": "ok", "note": "webui/ 前端目录缺失"}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="CCCP 启动器 (CCCP-Engine)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--model-root", action="append", default=[], help="模型扫描根目录(可多次)")
    ap.add_argument("--no-shell", action="store_true",
                    help="不开原生窗口，仅启动 HTTP/API 服务（维护用途）")
    args = ap.parse_args()

    if not args.no_shell and not _acquire_desktop_instance():
        log.info("CCCP 启动器已有桌面实例，忽略重复启动")
        return
    selected_port = _available_ui_port(args.host, args.port)
    if selected_port != args.port:
        log.warning("UI 端口 %d 已占用，自动改用 %d", args.port, selected_port)
        args.port = selected_port

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    attach_ring_log()
    settings = load_settings()
    for r in args.model_root:
        if r not in settings.model_roots:
            settings.model_roots.append(r)
    settings.save()

    app = create_app(settings)
    # 只在真正启动一个新的桌面/API 进程时清空终端；create_app 也被测试与
    # 嵌入方复用，不能因构造测试应用而擦除正在运行实例的日志。
    app.state.adapter.reset_terminal_session()
    log.info("CCCP 启动器启动: http://%s:%d (CCCP=%s)",
             args.host, args.port, settings.cccp_engine_path or "未探测到")

    if args.no_shell:
        uvicorn.run(
            app, host=args.host, port=args.port, log_level="info", use_colors=False
        )
    else:
        from .shell import run_with_shell
        run_with_shell(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

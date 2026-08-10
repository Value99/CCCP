"""WINUI-EXE 入口:FastAPI 应用 + CLI。

职责:
- 装配 settings / ProfileRegistry / TPQAdapter / ChatProxy / TrainingEngine / AppState
- REST API + OpenAI 兼容 /v1/chat/completions
- 静态托管 webui/(深色 SPA)
- 统一错误格式 {"error": {"code", "message"}}
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .chat import ChatProxy
from .profiles import ProfileError, ProfileRegistry
from .resources import builtin_profile_dir, user_profile_dir, webui_static
from .settings import load_settings, Settings
from .state import AppState
from .tpq_adapter import LaunchConfig, TPQAdapter, TPQError, discover_models
from .downloads import DownloadEngine, fetch_index, fetch_profile_text
from .training import (
    CORPUS_DIR, TrainingEngine, delete_corpus, export_counts, export_profile,
    export_scores, list_corpus, save_corpus_file,
)

log = logging.getLogger("winui")
WEBUI_STATIC = webui_static()
BUILTIN_PROFILE_DIR = builtin_profile_dir()
USER_PROFILE_DIR = user_profile_dir()


def downloads_default_hint(s: Settings) -> str:
    """模型下载默认落盘根目录(前端占位提示用)。"""
    if s.model_download_dir:
        return s.model_download_dir
    if s.model_roots:
        return s.model_roots[0]
    from .resources import runtime_root
    return str(runtime_root() / "data" / "models")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    registry = ProfileRegistry(BUILTIN_PROFILE_DIR, USER_PROFILE_DIR)
    adapter = TPQAdapter(settings)
    chat = ChatProxy(adapter)
    training = TrainingEngine(registry)
    downloads = DownloadEngine(settings)
    state = AppState()

    app = FastAPI(title="TPQ-Final WINUI-EXE Launcher", version=__version__)
    app.state.settings = settings
    app.state.registry = registry
    app.state.adapter = adapter
    app.state.chat = chat
    app.state.training = training
    app.state.app_state = state

    # -- 统一错误 --
    @app.exception_handler(ProfileError)
    async def _profile_err(_: Request, exc: ProfileError):
        return JSONResponse({"error": {"code": "profile", "message": str(exc)}}, status_code=400)

    @app.exception_handler(TPQError)
    async def _tpq_err(_: Request, exc: TPQError):
        return JSONResponse({"error": {"code": "tpq", "message": str(exc)}}, status_code=409)

    @app.exception_handler(ValueError)
    async def _val_err(_: Request, exc: ValueError):
        return JSONResponse({"error": {"code": "bad-request", "message": str(exc)}}, status_code=400)

    # ------------------------------------------------------------------
    # 健康 / 设置 / 模型
    # ------------------------------------------------------------------
    @app.get("/api/health")
    async def api_health():
        tpq = await adapter.health()
        return {"winui": "ok", "version": __version__,
                "tpq_available": adapter.available(), "tpq": tpq}

    @app.get("/api/settings")
    async def api_get_settings():
        return settings.__dict__

    @app.post("/api/settings")
    async def api_set_settings(req: Request):
        body = await req.json()
        for k, v in body.items():
            if hasattr(settings, k):
                setattr(settings, k, v)
        settings.save()
        return settings.__dict__

    @app.get("/api/models")
    async def api_models():
        found = discover_models(settings.model_roots)
        return {
            "models": [m.__dict__ for m in found],
            "roots": settings.model_roots,
            "hint": "在设置里添加 model_roots(含 cccp.json 的上级目录)以发现模型",
        }

    # ------------------------------------------------------------------
    # 社区(profile 下载 / Discord 链接)(v0.3)
    # ------------------------------------------------------------------
    @app.get("/api/community/config")
    async def api_community_config():
        return {"discord_url": settings.discord_url,
                "index_url": settings.community_index_url}

    @app.get("/api/community/profiles")
    async def api_community_profiles():
        url = (settings.community_index_url or "").strip()
        if not url:
            return {"profiles": [],
                    "note": "未配置 community_index_url(到「设置 · 社区与下载」填写索引地址)"}
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
        return {"profiles": [p.to_dict() for p in registry.list()],
                "selected": state.data.get("selected_profiles", [])}

    @app.get("/api/profiles/{pid}")
    async def api_profile_detail(pid: str):
        return registry.require(pid).to_dict(with_experts=True)

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
            text = (await file.read()).decode("utf-8", errors="replace")
            p = registry.import_text(text, filename=file.filename or "profile.yaml")
        else:
            body = await req.json()
            text = str(body.get("yaml_text") or body.get("json_text") or "")
            if not text:
                raise ProfileError("空内容")
            p = registry.import_text(text, filename="profile.yaml")
        return {"ok": True, "profile": p.to_dict()}

    @app.delete("/api/profiles/{pid}")
    async def api_delete_profile(pid: str):
        registry.delete(pid)
        return {"ok": True}

    # ------------------------------------------------------------------
    # 启动 / 停止 / 状态
    # ------------------------------------------------------------------
    def _launch_cfg(body: dict, ids: list[str], combo) -> LaunchConfig:
        return LaunchConfig(
            model_path=str(body.get("model_path") or ""),
            profiles=ids,
            combination=combo,
            port=int(body.get("port") or settings.api_port_alloc_start),
            host=str(body.get("host") or "127.0.0.1"),
            served_model_name=str(body.get("served_model_name") or "winui-model"),
            profile_mode=str(body.get("profile_mode") or "auto"),
            device=str(body.get("device") or "cuda"),
            cache_gb=body.get("cache_gb"),
            vram_gb=body.get("vram_gb"),
            dense_residency=str(body.get("dense_residency") or "auto"),
            cpu_compile=str(body.get("cpu_compile") or "auto"),
            extreme=bool(body.get("extreme", True)),
        )

    @app.post("/api/launch")
    async def api_launch(req: Request):
        body = await req.json()
        ids = [str(x) for x in body.get("profile_ids") or []]
        if not ids:
            raise ProfileError("至少选择一个 profile")
        combo = registry.combine(ids)
        cfg = _launch_cfg(body, ids, combo)
        if not cfg.model_path:
            raise TPQError("缺少 model_path(先在「模型」里扫描并选择)")
        if body.get("dry_run_only"):
            return {"dry_run": adapter.dry_run(cfg)}
        inst = adapter.launch(cfg)
        state.set_selected_profiles(ids)
        state.record_launch(cfg.model_path, ids, cfg.port)
        return {"ok": True, "instance": inst.__dict__, "combination": combo.to_dict(),
                "tip": "轮询 /api/launch/status 直至 ready=true"}

    @app.post("/api/launch/dry-run")
    async def api_launch_dry(req: Request):
        body = await req.json()
        ids = [str(x) for x in body.get("profile_ids") or []]
        combo = registry.combine(ids)
        cfg = _launch_cfg(body, ids, combo)
        return adapter.dry_run(cfg)

    @app.post("/api/launch/stop")
    async def api_launch_stop():
        return {"stopped": adapter.stop()}

    @app.get("/api/launch/status")
    async def api_launch_status():
        h = await adapter.health()
        return {"instance": adapter.instance.__dict__ if adapter.instance else None,
                "health": h, "log_tail": adapter.tail_log(40),
                "last_launch": state.data.get("last_launch")}

    # ------------------------------------------------------------------
    # 聊天代理(含 OpenAI 兼容 /v1)
    # ------------------------------------------------------------------
    @app.get("/api/chat/models")
    async def api_chat_models():
        return await chat.models()

    @app.post("/api/chat/completions")
    @app.post("/v1/chat/completions")
    async def api_chat_completions(req: Request):
        payload = await req.json()
        payload.pop("profile_ids", None)  # profile_ids 仅作上下文标注
        if payload.get("stream", True):
            return StreamingResponse(
                chat.completions_stream(payload),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await chat.completions_once(payload)

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
        name = save_corpus_file(file.filename or "corpus.txt", await file.read())
        return {"ok": True, "file": name}

    @app.get("/api/training/corpus")
    async def api_corpus_list():
        return {"files": list_corpus()}

    @app.delete("/api/training/corpus/{name}")
    async def api_corpus_delete(name: str):
        return {"deleted": delete_corpus(name)}

    @app.post("/api/training/jobs")
    async def api_train_submit(req: Request):
        job = training.submit(await req.json())
        return {"ok": True, "job": job.to_dict()}

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

    @app.get("/api/training/jobs/{jid}/export")
    async def api_train_export(jid: str, kind: str = "scores", name: str = ""):
        job = training.get(jid)
        if not job or job.status != "done":
            raise ValueError("任务未完成或不存在")
        if kind == "scores":
            return export_scores(job)
        if kind == "counts":
            return export_counts(job)
        if kind == "profile":
            return export_profile(job, name=name)
        raise ValueError("kind 必须是 scores|counts|profile")

    @app.post("/api/training/jobs/{jid}/register")
    async def api_train_register(jid: str, req: Request):
        """把训练产物注册为 trained profile(写入 profiles/user/)。"""
        job = training.get(jid)
        if not job or job.status != "done":
            raise ValueError("任务未完成或不存在")
        body = await req.json() if req.headers.get("content-length") else {}
        data = export_profile(job, name=str(body.get("name") or ""))
        import yaml
        p = registry.import_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            filename=f"{data['id']}.yaml",
        )
        p.source = "trained"
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
            "auth": bool(settings.tpq_api_key),
            "served_model": inst.model if inst else None,
            "tpq_ready": (await adapter.health()).get("ready", False),
            "endpoints": {
                "openai": ["POST /v1/chat/completions"],
                "profiles": ["GET /api/profiles", "POST /api/profiles/combine",
                             "POST /api/profiles/import", "DELETE /api/profiles/{id}"],
                "launch": ["POST /api/launch", "POST /api/launch/stop",
                           "GET /api/launch/status", "POST /api/launch/dry-run"],
                "training": ["POST /api/training/jobs", "GET /api/training/jobs",
                             "GET /api/training/jobs/{id}/export"],
                "community": ["GET /api/community/config", "GET /api/community/profiles",
                              "POST /api/community/profiles/install"],
                "downloads": ["POST /api/models/download", "GET /api/models/download/jobs",
                              "DELETE /api/models/download/jobs/{id}"],
                "chat": ["POST /api/chat/completions", "GET /api/chat/models"],
            },
            "curl_example": (
                f'curl -N {base}/v1/chat/completions \\\n'
                '  -H "Content-Type: application/json" \\\n'
                '  -d \'{"model": "winui-model", "stream": true,\n'
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
    ap = argparse.ArgumentParser(description="TPQ-Final WINUI-EXE Launcher")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--tpq-path", default="", help="TPQ-Final 根目录(默认自动探测 ../TPQ-Final)")
    ap.add_argument("--model-root", action="append", default=[], help="模型扫描根目录(可多次)")
    ap.add_argument("--no-shell", action="store_true",
                    help="不开原生窗口,仅启动 HTTP 服务(浏览器访问)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings = load_settings()
    if args.tpq_path:
        settings.tpq_path = args.tpq_path
    for r in args.model_root:
        if r not in settings.model_roots:
            settings.model_roots.append(r)
    settings.save()

    app = create_app(settings)
    log.info("WINUI-EXE 启动: http://%s:%d (TPQ=%s)",
             args.host, args.port, settings.tpq_path or "未探测到")

    if args.no_shell:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        from .shell import run_with_shell
        run_with_shell(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

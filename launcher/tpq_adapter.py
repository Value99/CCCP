"""与 TPQ-Final 的唯一集成层。

约束:不修改 TPQ-Final 的任何文件。集成方式三种(见 docs/INTERFACE.md):
1. 子进程:`python -m tpq launch serve --model <dir> ...`
2. 文件:生成的 profile.json(TPQ_PROFILE_JSON)与 extreme score file
   (--extreme-score-file,schema tpq-expert-residency-scores-v1)
3. HTTP:OpenAI 兼容 /health、/v1/models、/v1/chat/completions
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .profiles import Combination
from .settings import DATA_DIR, Settings

log = logging.getLogger("winui.tpq")

RUNTIME_DIR = DATA_DIR / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


class TPQError(RuntimeError):
    pass


@dataclass
class ModelInfo:
    path: str
    name: str
    architecture: str = "unknown"


@dataclass
class LaunchConfig:
    """一次启动请求:模型 + profile 组合 + 运行档位。"""

    model_path: str
    profiles: list[str]
    combination: Combination
    port: int
    host: str = "127.0.0.1"
    served_model_name: str = "winui-model"
    profile_mode: str = "auto"  # auto|ram|resident|mapped|parallel
    device: str = "cuda"  # cuda|cpu(cpu 供训练选项卡全量推理使用)
    cache_gb: float | None = None
    vram_gb: float | None = None
    dense_residency: str = "auto"
    extreme: bool = True
    cpu_compile: str = "auto"
    extra_args: list[str] = field(default_factory=list)


@dataclass
class TPQInstance:
    pid: int
    port: int
    model: str
    profiles: list[str]
    started_at: float
    log_file: str
    base_url: str


# --------------------------------------------------------------------------
# 模型扫描
# --------------------------------------------------------------------------

def discover_models(roots: list[str]) -> list[ModelInfo]:
    """扫描含 cccp.json 的目录(TPQ 模型仓标记)。"""
    out: list[ModelInfo] = []
    for root in roots:
        rp = Path(root)
        if not rp.is_dir():
            continue
        for child in sorted(rp.iterdir()):
            cccp = child / "cccp.json"
            if not cccp.is_file():
                continue
            arch = "unknown"
            try:
                meta = json.loads(cccp.read_text(encoding="utf-8"))
                cfg = meta.get("config") or {}
                arch = str(cfg.get("model_family") or cfg.get("arch") or "cccp")
            except (json.JSONDecodeError, OSError):
                pass
            out.append(ModelInfo(path=str(child), name=child.name, architecture=arch))
    return out


# --------------------------------------------------------------------------
# 集成层
# --------------------------------------------------------------------------

class TPQAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.instance: TPQInstance | None = None
        self._proc: subprocess.Popen | None = None

    # -- 基础 --
    @property
    def tpq_root(self) -> Path:
        if not self.settings.tpq_path:
            raise TPQError("未配置 TPQ-Final 路径(设置 tpq_path 或把 WINUI-EXE 放在其同级)")
        return Path(self.settings.tpq_path)

    def _python(self) -> str:
        venv_py = self.tpq_root / ".venv" / "Scripts" / "python.exe"
        return str(venv_py) if venv_py.exists() else sys.executable

    def available(self) -> bool:
        try:
            return (self.tpq_root / "tpq" / "__main__.py").exists()
        except TPQError:
            return False

    # -- 文件生成:profile.json / extreme score file --
    @staticmethod
    def _score_file(combo: Combination, job: str = "launch") -> Path:
        """生成 extreme placement 偏好文件(tpq-expert-residency-scores-v1)。

        组合内专家给 1.0;drop 解析结果仅作 meta 提示(不改变 TPQ,
        TPQ 的 drop masking 在路由前自动生效,见 docs/INTERFACE.md §1)。
        """
        payload = {
            "schema": "tpq-expert-residency-scores-v1",
            "scores": {k: 1.0 for k in combo.union},
            "meta": {
                "generator": "tpq-winui-launcher",
                "profiles": combo.profile_ids,
                "expert_count": len(combo.union),
                "drop_resolution": combo.drop_resolution,
            },
        }
        path = RUNTIME_DIR / f"scores-{job}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @staticmethod
    def _profile_counts_file(combo: Combination, job: str = "launch") -> Path:
        """生成 TPQ_PROFILE_JSON 计数档案(schema counts:{layer:{expert:count}})。"""
        counts: dict[str, dict[str, int]] = {}
        for key in combo.union:
            layer, eid = key.split(":", 1)
            counts.setdefault(layer, {})[eid] = 1
        path = RUNTIME_DIR / f"profile-{job}.json"
        path.write_text(json.dumps({"counts": counts}), encoding="utf-8")
        return path

    # -- 命令行构造 --
    def build_command(self, cfg: LaunchConfig, *, dry_run: bool = False) -> list[str]:
        cmd = [
            self._python(), "-m", "tpq", "launch", "serve",
            "--model", cfg.model_path,
            "--host", cfg.host,
            "--port", str(cfg.port),
            "--served-model-name", cfg.served_model_name,
            "--profile", cfg.profile_mode,
            "--device", cfg.device,
            "--dense-residency", cfg.dense_residency,
            "--cpu-compile", cfg.cpu_compile,
        ]
        if cfg.cache_gb:
            cmd += ["--cache-gb", str(cfg.cache_gb)]
        if cfg.vram_gb:
            cmd += ["--vram-gb", str(cfg.vram_gb)]
        if cfg.extreme:
            cmd += ["--extreme", "--extreme-placement", "auto",
                    "--extreme-score-file", str(self._score_file(cfg.combination))]
        if dry_run:
            cmd.append("--dry-run")
        cmd += cfg.extra_args
        return cmd

    def _env(self, cfg: LaunchConfig) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.tpq_root)
        env["TPQ_PROFILE_JSON"] = str(self._profile_counts_file(cfg.combination))
        if self.settings.tpq_api_key:
            env["TPQ_API_KEY"] = self.settings.tpq_api_key
        return env

    # -- 预检 --
    def dry_run(self, cfg: LaunchConfig) -> dict:
        cmd = self.build_command(cfg, dry_run=True)
        proc = subprocess.run(
            cmd, cwd=str(self.tpq_root), env=self._env(cfg),
            capture_output=True, text=True, timeout=120,
        )
        return {
            "ok": proc.returncode == 0,
            "cmd": cmd,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }

    # -- 生命周期 --
    def launch(self, cfg: LaunchConfig) -> TPQInstance:
        if self._proc and self._proc.poll() is None:
            raise TPQError("已有 TPQ 实例在运行,请先停止")
        log_path = RUNTIME_DIR / "tpq-serve.log"
        logf = open(log_path, "a", encoding="utf-8", buffering=1)
        logf.write(f"\n===== launch {time.strftime('%F %T')} profiles={cfg.profiles} =====\n")
        cmd = self.build_command(cfg)
        log.info("spawn: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd, cwd=str(self.tpq_root), env=self._env(cfg),
            stdout=logf, stderr=subprocess.STDOUT,
        )
        self.instance = TPQInstance(
            pid=self._proc.pid,
            port=cfg.port,
            model=cfg.model_path,
            profiles=list(cfg.profiles),
            started_at=time.time(),
            log_file=str(log_path),
            base_url=f"http://{cfg.host}:{cfg.port}",
        )
        return self.instance

    def stop(self) -> bool:
        if not self._proc:
            return False
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self.instance = None
        return True

    # -- 健康 / 状态(HTTP 集成面) --
    async def health(self) -> dict:
        if not self.instance:
            return {"ready": False, "running": False}
        if self._proc and self._proc.poll() is not None:
            return {"ready": False, "running": False, "exit": self._proc.returncode}
        try:
            async with httpx.AsyncClient(timeout=3) as cli:
                r = await cli.get(f"{self.instance.base_url}/health")
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                body["running"] = True
                return body
        except (httpx.HTTPError, OSError):
            return {"ready": False, "running": True, "note": "进程存活,/health 尚未就绪"}

    async def wait_ready(self, timeout_s: float = 600.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            h = await self.health()
            if h.get("ready"):
                return True
            if not h.get("running"):
                return False
            time.sleep(2.0)
        return False

    def tail_log(self, lines: int = 80) -> str:
        lf = RUNTIME_DIR / "tpq-serve.log"
        if not lf.exists():
            return ""
        data = lf.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(data[-lines:])

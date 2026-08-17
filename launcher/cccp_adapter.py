"""启动器与内置 CCCP-Engine 的稳定集成层。

集成方式三种（开发归档见 archive/docs-internal-20260813/INTERFACE.md）：
1. 子进程:`python -m cccp launch serve --model <dir> ...`
2. 文件:生成的 profile.json(CCCP_PROFILE_JSON)与 extreme score file
   (--extreme-score-file,schema cccp-expert-residency-scores-v1)
3. HTTP:OpenAI 兼容端点、模型规格、专家字节表、路由统计与加载回执
"""
from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import locale
import logging
import os
import re
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .accelerators import backend_python, probe_backend

_WINDOWS = os.name == "nt"
from .io_utils import atomic_write_text
from .profiles import Combination
from .settings import DATA_DIR, Settings
from .resources import operator_cache_dir

log = logging.getLogger("winui.cccp")

RUNTIME_DIR = DATA_DIR / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
CHAT_METRICS_FILE = RUNTIME_DIR / "chat-metrics.jsonl"


class CCCPEngineError(RuntimeError):
    pass


FULL_MODEL_PROFILE_ID = "__full_model__"


def _decode_mixed_process_log(payload: bytes) -> str:
    """Decode UTF-8 Python output mixed with the Windows compiler code page."""
    encodings = tuple(dict.fromkeys((
        "utf-8",
        locale.getpreferredencoding(False) or "",
        "gb18030",
    )))
    decoded: list[str] = []
    for raw_line in payload.splitlines():
        for encoding in encodings:
            if not encoding:
                continue
            try:
                decoded.append(raw_line.decode(encoding))
                break
            except (LookupError, UnicodeDecodeError):
                continue
        else:
            decoded.append(raw_line.decode("utf-8", errors="replace"))
    return "\n".join(decoded)


@dataclass
class ModelInfo:
    path: str
    name: str
    architecture: str = "unknown"
    model_format: str = "unknown"
    model_version: str = "unknown"
    manifest_sha256: str = ""
    total_bytes: int = 0
    total_gb: float = 0.0
    dense_gb: float = 0.0
    dense_without_shared_gb: float = 0.0
    shared_expert_gb: float = 0.0
    expert_gb: float = 0.0
    layers: int = 0
    expert_layers: list[int] = field(default_factory=list)
    expert_layer_count: int = 0
    experts_per_layer: int = 0
    top_k: int = 0
    max_context: int = 0
    complete: bool = False
    errors: list[str] = field(default_factory=list)
    profile_count: int = 0
    execution_kind: str = "dynamic_moe"
    has_dynamic_experts: bool = True
    supports_route_training: bool = True


@dataclass
class LaunchConfig:
    """一次启动请求:模型 + profile 组合 + 运行档位。"""

    model_path: str
    profiles: list[str]
    combination: Combination
    port: int
    host: str = "127.0.0.1"
    served_model_name: str = "winui-model"
    profile_mode: str = "auto"  # auto（逐级降级）|mapped（强制磁盘映射）
    device: str = "cpu"
    cache_gb: float | None = None
    vram_gb: float | None = None
    dense_residency: str = "auto"
    extreme: bool = False
    cpu_compile: str = "auto"
    max_ctx: int = 0  # 0=模型声明上限；KV 按实际请求动态扩展
    cpu_threads: int = 0
    memory_limit_gb: float = 32.0
    extra_args: list[str] = field(default_factory=list)


@dataclass
class CCCPEngineInstance:
    pid: int
    port: int
    model: str
    served_model_name: str
    profiles: list[str]
    started_at: float
    log_file: str
    base_url: str
    full_model: bool = False


# --------------------------------------------------------------------------
# 模型扫描
# --------------------------------------------------------------------------

_TRANSIENT_MODEL_SUFFIXES = (
    ".copying",
    ".partial",
    ".downloading",
    ".download",
    ".tmp",
    ".temp",
)


def _is_transient_model_directory(path: Path) -> bool:
    """复制或下载中的目录永远不进入用户可见模型库。"""
    name = path.name.casefold()
    return name.startswith(".") or name.endswith(_TRANSIENT_MODEL_SUFFIXES)


def _manifest_dense_files(meta: dict) -> list[str]:
    """返回 CCCP-1 两种清单布局声明的全部固定权重分片。"""
    values = meta.get("dense_files")
    if not values:
        nonexpert = meta.get("nonexpert") or {}
        if isinstance(nonexpert, dict):
            values = nonexpert.get("files")
    if not values:
        values = [meta.get("dense_file") or "dense.safetensors"]
    if isinstance(values, (str, os.PathLike)):
        values = [values]
    result = [str(value) for value in values if value]
    tensor_files = meta.get("tensor_files") or ()
    if isinstance(tensor_files, (str, os.PathLike)):
        tensor_files = [tensor_files]
    result.extend(str(value) for value in tensor_files if value)
    return list(dict.fromkeys(result))


def _manifest_expert_files(meta: dict) -> dict[str, str]:
    """统一读取扁平 expert_files 与 routed_experts.layer_files。"""
    values = meta.get("expert_files") or {}
    if isinstance(values, dict) and values:
        return {str(layer): str(name) for layer, name in values.items()}
    routed = (meta.get("routed_experts") or {}).get("layer_files") or {}
    if not isinstance(routed, dict):
        return {}
    result: dict[str, str] = {}
    for layer, item in routed.items():
        name = item.get("path") if isinstance(item, dict) else item
        if name:
            result[str(layer)] = str(name)
    return result


def _manifest_tokenizer_files(meta: dict) -> list[str]:
    values = meta.get("tokenizer_files")
    if values:
        if isinstance(values, (str, os.PathLike)):
            values = [values]
        return [str(value) for value in values if value]
    if str(meta.get("model_family") or "").strip().lower() == "kimi_k3":
        return ["tokenizer_config.json", "tiktoken.model", "tokenization_kimi.py"]
    return ["tokenizer.json"]


def _dense_audit_bytes(root: Path, meta: dict) -> tuple[int, int]:
    """从分片审计读取（固定权重字节，共享专家字节）。"""
    audit_name = meta.get("dense_audit_file")
    if not audit_name:
        return 0, 0
    try:
        audit = json.loads((root / str(audit_name)).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return 0, 0
    fixed = int(audit.get("fixed_bytes") or 0)
    entries = audit.get("entries") or {}
    shared = 0
    if isinstance(entries, dict):
        shared = sum(
            int(item.get("stored_bytes") or item.get("source_bytes") or 0)
            for name, item in entries.items()
            if isinstance(item, dict) and "shared_experts" in str(name)
        )
    return fixed, shared

def inspect_model(path: str | Path) -> ModelInfo:
    """只读检查一个 CCCP 模型，不读取巨型权重内容。"""
    root = Path(path).expanduser().resolve()
    info = ModelInfo(path=str(root), name=root.name)
    manifest_path = root / "cccp.json"
    if not manifest_path.is_file():
        info.errors.append("缺少 cccp.json")
        return info
    try:
        manifest_bytes = manifest_path.read_bytes()
        meta = json.loads(manifest_bytes.decode("utf-8"))
        info.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    except (json.JSONDecodeError, OSError) as exc:
        info.errors.append(f"cccp.json 无法解析: {exc}")
        return info
    cfg = meta.get("config") or {}
    info.model_format = str(meta.get("format") or "cccp")
    info.model_version = str(
        meta.get("version") or cfg.get("model_version") or root.name
    )
    info.architecture = str(
        meta.get("architecture") or meta.get("model_family")
        or cfg.get("model_family") or cfg.get("arch") or "cccp"
    )
    expert_files = _manifest_expert_files(meta)
    tensor_vq = meta.get("tensor_vq") or {}
    routed = meta.get("routed_experts") or {}
    is_dense_vq = bool(tensor_vq) and not bool(expert_files) and not bool(
        routed.get("layer_files") if isinstance(routed, dict) else None
    )
    if is_dense_vq:
        info.execution_kind = "dense_vq"
        info.has_dynamic_experts = False
        info.supports_route_training = False
    info.layers = int(cfg.get("n_layers") or len(expert_files))
    info.experts_per_layer = int(cfg.get("n_experts") or 0)
    info.top_k = int(cfg.get("top_k") or 0)
    info.max_context = int(cfg.get("max_position_embeddings") or 0)
    dense_files = _manifest_dense_files(meta)
    dense_paths = [root / name for name in dense_files]
    dense_bytes = 0
    for dense_name, dense_path in zip(dense_files, dense_paths):
        if not dense_path.is_file():
            info.errors.append(f"缺少 Dense 权重: {dense_name}")
        else:
            dense_bytes += dense_path.stat().st_size
    audit_fixed_bytes, audit_shared_bytes = _dense_audit_bytes(root, meta)
    if audit_fixed_bytes:
        dense_bytes = audit_fixed_bytes
    info.dense_gb = round(dense_bytes / 2**30, 3)
    if audit_shared_bytes:
        info.shared_expert_gb = round(audit_shared_bytes / 2**30, 3)
        info.dense_without_shared_gb = round(
            max(0, dense_bytes - audit_shared_bytes) / 2**30, 3
        )
    elif len(dense_paths) == 1 and dense_paths[0].is_file():
        dense_path = dense_paths[0]
        try:
            with dense_path.open("rb") as handle:
                prefix = handle.read(8)
                if len(prefix) != 8:
                    raise ValueError("safetensors header 不完整")
                header_size = struct.unpack("<Q", prefix)[0]
                if header_size > dense_path.stat().st_size - 8 or header_size > 256 * 2**20:
                    raise ValueError("safetensors header 长度非法")
                header = json.loads(handle.read(header_size))
            shared_bytes = sum(
                int(value["data_offsets"][1]) - int(value["data_offsets"][0])
                for key, value in header.items()
                if key != "__metadata__" and "shared_experts" in key
            )
            info.shared_expert_gb = round(shared_bytes / 2**30, 3)
            info.dense_without_shared_gb = round(
                (dense_path.stat().st_size - shared_bytes) / 2**30, 3
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError, struct.error):
            # Unknown dense container: it is still counted exactly as fixed
            # model memory, only the optional shared-vs-dense split is absent.
            info.dense_without_shared_gb = info.dense_gb
    elif dense_bytes:
        info.dense_without_shared_gb = info.dense_gb
    if is_dense_vq:
        if not isinstance(tensor_vq, dict):
            info.errors.append("cccp.json tensor_vq 格式无效")
        else:
            declared_files = set(dense_files)
            for tensor_name, item in tensor_vq.items():
                if not isinstance(item, dict):
                    info.errors.append(f"Dense VQ 张量定义无效: {tensor_name}")
                    continue
                filename = str(item.get("file") or "")
                if not filename or filename not in declared_files:
                    info.errors.append(
                        f"Dense VQ 张量 {tensor_name} 引用了未声明分片: {filename or '空'}"
                    )
        info.expert_layers = []
        info.expert_layer_count = 0
        info.experts_per_layer = 0
        info.top_k = 0
        info.expert_gb = 0.0
    elif not isinstance(expert_files, dict) or not expert_files:
        info.errors.append("cccp.json 没有动态专家层文件")
    else:
        try:
            info.expert_layers = sorted(int(layer) for layer in expert_files)
            info.expert_layer_count = len(info.expert_layers)
        except (TypeError, ValueError):
            info.errors.append("cccp.json expert_files 层号无效")
        expert_bytes = 0
        for layer, name in expert_files.items():
            shard = root / str(name)
            if not shard.is_file():
                info.errors.append(f"缺少专家分片 L{layer}: {name}")
            else:
                expert_bytes += shard.stat().st_size
        info.expert_gb = round(expert_bytes / 2**30, 3)
    for required in ("config.json", *_manifest_tokenizer_files(meta)):
        if not (root / required).is_file():
            info.errors.append(f"缺少 {required}")
    try:
        info.total_bytes = sum(
            item.stat().st_size for item in root.rglob("*") if item.is_file()
        )
        info.total_gb = round(info.total_bytes / 2**30, 3)
    except OSError as exc:
        info.errors.append(f"无法统计模型体积: {exc}")
    profiles_dir = root / "profiles"
    if profiles_dir.is_dir():
        info.profile_count = sum(
            1 for path in profiles_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}
        )
    info.complete = not info.errors
    return info


def estimate_gpu_vram_plan(
    model: ModelInfo,
    *,
    max_ctx: int,
    expert_cache_gb: float,
) -> dict[str, float | str]:
    """Estimate the two GPU admission thresholds without loading weights.

    ``minimum_vram_gb`` is the fixed CUDA working set: Dense, the active
    context/prefill workspace and the architecture safety margin.  Falling
    below it cannot be fixed by shrinking the expert hot cache.  The
    ``recommended_vram_gb`` threshold adds a useful (but optional) expert
    arena; cards between the two thresholds remain valid and keep the full
    expert set in host RAM while executing bounded chunks on the GPU.

    This deliberately does *not* count the complete selected expert profile
    as VRAM.  CCCP's dynamic-expert design keeps that compact set in RAM and
    uses only the remaining VRAM as a hot arena.
    """
    architecture = model.architecture.strip().lower()
    config: dict = {}
    manifest: dict = {}
    try:
        manifest = json.loads(
            (Path(model.path) / "cccp.json").read_text(encoding="utf-8")
        )
        config = manifest.get("config") or {}
    except (OSError, json.JSONDecodeError, TypeError):
        pass

    if (
        "hc_mult" in config
        or "compress_ratios" in config
        or architecture in {"dsv4", "deepseek_v4"}
    ):
        architecture = "dsv4"
        # Mirrors the engine's full-batch DSV4 admission reserve.  The 1.5
        # GiB addition covers CUDA/private allocator and transient Dense load
        # buffers which are not represented by the on-disk file size.
        block = min(4096, max(1, int(max_ctx)))
        context_workspace = 1.0 + max(0.25, 8.75 * block / 4096.0)
        dense_runtime = max(0.0, model.dense_gb) + 1.5
        driver_reserve = 2.0
        minimum_expert_arena = 1.0
    elif (
        str(manifest.get("model_family") or "").lower() == "kimi_k3"
        or ("kda_layers" in config and "routed_hidden" in config)
    ):
        architecture = "kimi_k3"
        # Consumer cards use Kimi's RAM-Dense/CUDA-packed-MoE path.  The hard
        # floor is its heterogeneous CUDA working set, while the full-speed
        # recommendation still includes native Dense GPU placement.
        context_workspace = 3.0
        dense_runtime = 0.0
        driver_reserve = 6.5
        minimum_expert_arena = 1.0
        full_gpu_recommended = (
            max(0.0, model.dense_gb) + 1.5
            + context_workspace + 2.0 + 4.0
        )
    elif getattr(model, "execution_kind", "dynamic_moe") == "dense_vq" or (
        bool(manifest.get("tensor_vq")) and not bool(_manifest_expert_files(manifest))
    ):
        architecture = "qwen3_5_dense"
        # All projection payloads are fixed Dense weights.  There is no
        # elastic expert arena or RAM expert offload tier; the GPU path keeps
        # the compact archive resident and allocates recurrent/KV state on
        # top.  The workspace grows with the initial active context only.
        initial_ctx = min(max(1, int(max_ctx)), 4096)
        dense_runtime = max(0.0, model.dense_gb) + 0.75
        context_workspace = 0.75 + 0.00055 * initial_ctx
        driver_reserve = 1.0
        minimum_expert_arena = 0.0
    else:
        architecture = "glm"
        vocab = int(config.get("vocab") or config.get("vocab_size") or 0)
        hidden = int(config.get("hidden") or config.get("hidden_size") or 0)
        head_gb = vocab * hidden * 4 / 2**30 if vocab and hidden else 0.0
        mtp_gb = 0.0
        attachment = manifest.get("mtp_file")
        if attachment:
            try:
                mtp_gb = (Path(model.path) / str(attachment)).stat().st_size / 2**30
            except OSError:
                pass
        initial_ctx = min(max(1, int(max_ctx)), 4096)
        latent_kv_gb = 2.3 + 0.09 * initial_ctx / 1024.0
        dense_runtime = (
            max(0.0, model.dense_gb)
            + head_gb
            + mtp_gb
            + 1.5
            + latent_kv_gb
        )
        context_workspace = 3.0
        driver_reserve = 3.0
        minimum_expert_arena = 0.5

    minimum = (
        dense_runtime
        + context_workspace
        + driver_reserve
        + minimum_expert_arena
    )
    preferred_expert_arena = (
        0.0 if architecture == "qwen3_5_dense"
        else min(4.0, max(0.5, float(expert_cache_gb)))
    )
    recommended = (
        max(minimum + preferred_expert_arena, full_gpu_recommended)
        if architecture == "kimi_k3"
        else minimum + preferred_expert_arena
    )
    return {
        "architecture": architecture,
        "dense_runtime_gb": round(dense_runtime, 3),
        "context_workspace_gb": round(context_workspace, 3),
        "driver_reserve_gb": round(driver_reserve, 3),
        "minimum_expert_arena_gb": round(minimum_expert_arena, 3),
        "preferred_expert_arena_gb": round(preferred_expert_arena, 3),
        "minimum_vram_gb": round(minimum, 3),
        "recommended_vram_gb": round(recommended, 3),
        "hybrid_dense_ram": architecture == "kimi_k3",
    }


def discover_models(roots: list[str]) -> list[ModelInfo]:
    """扫描根目录自身及其一级子目录中的 CCCP 模型。"""
    out: list[ModelInfo] = []
    seen: set[Path] = set()
    for root in roots:
        rp = Path(root).expanduser()
        if not rp.is_dir():
            continue
        candidates = (
            [rp]
            if (rp / "cccp.json").is_file()
            and not _is_transient_model_directory(rp)
            else []
        )
        candidates.extend(
            path for path in sorted(rp.iterdir())
            if path.is_dir() and not _is_transient_model_directory(path)
        )
        for child in candidates:
            try:
                resolved = child.resolve()
            except OSError:
                continue
            if resolved in seen or not (resolved / "cccp.json").is_file():
                continue
            seen.add(resolved)
            out.append(inspect_model(resolved))
    return out


def full_model_combination(model_path: str | Path) -> Combination:
    """构造覆盖模型全部动态专家的运行组合，不生成配置文件。"""
    from .profiles import ExpertRef
    from .training import load_expert_sizes

    model = inspect_model(model_path)
    if not model.complete:
        raise CCCPEngineError("模型不完整：" + "；".join(model.errors))
    expected = [
        f"{layer}:{expert}"
        for layer in model.expert_layers
        for expert in range(model.experts_per_layer)
    ]
    raw_sizes = load_expert_sizes(model.path)
    sizes = {key: float(raw_sizes[key]) for key in expected if key in raw_sizes}
    if sizes:
        fallback_mb = (model.expert_gb * 1024.0 - sum(sizes.values())) / max(
            1, len(expected) - len(sizes)
        )
    else:
        fallback_mb = model.expert_gb * 1024.0 / max(1, len(expected))
    fallback_mb = max(0.001, fallback_mb)
    union = {
        key: ExpertRef(key=key, size_mb=float(sizes.get(key, fallback_mb)))
        for key in expected
    }
    return Combination(
        profile_ids=[FULL_MODEL_PROFILE_ID],
        union=union,
        overlap_mb=0.0,
        model_manifest_sha256=model.manifest_sha256,
        model_name=model.name,
        model_version=model.model_version,
        model_total_bytes=model.total_bytes,
        fixed_model_gib=model.dense_gb,
        dense_without_shared_gib=model.dense_without_shared_gb,
        shared_expert_gib=model.shared_expert_gb,
    )


def _memory_status() -> tuple[float, float]:
    """返回 (总内存 GiB, 可用内存 GiB)，无需 psutil。"""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / 2**30, vm.available / 2**30
    except ImportError:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return 0.0, 0.0
        return status.ullTotalPhys / 2**30, status.ullAvailPhys / 2**30


def port_available(host: str, port: int) -> bool:
    if not 1 <= int(port) <= 65535:
        return False
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


# --------------------------------------------------------------------------
# 集成层
# --------------------------------------------------------------------------

class CCCPEngineAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.instance: CCCPEngineInstance | None = None
        self._proc: subprocess.Popen | None = None
        self._log_handle = None
        self.last_preflight: dict | None = None
        # Readiness belongs to one concrete engine PID.  A long fused HIP
        # kernel can temporarily keep Python from serving /health within the
        # launcher's 3-second timeout; once that same process has answered
        # ready, a transient timeout must not move the GUI back to "loading".
        self._ready_pid: int | None = None

    def reset_terminal_session(self) -> None:
        """Start each launcher process with an empty inference terminal."""
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        for path in (RUNTIME_DIR / "cccp-serve.log", CHAT_METRICS_FILE):
            try:
                path.write_bytes(b"")
            except OSError as exc:
                # A stale process from an older launcher may briefly retain a
                # handle.  Do not make the desktop UI unavailable solely
                # because its historical terminal could not be truncated.
                log.warning("无法清空旧会话文件 %s: %s", path, exc)

    # -- 基础 --
    @property
    def cccp_root(self) -> Path:
        if not self.settings.cccp_engine_path:
            raise CCCPEngineError("发行目录缺少内置 engine/CCCP-Engine")
        return Path(self.settings.cccp_engine_path)

    def _python(self, device: str = "cpu") -> str:
        configured = backend_python(self.settings, device)
        if configured and configured.is_file():
            return str(configured)
        raise CCCPEngineError(
            f"发行目录缺少内置 runtime/{device}/env/python.exe"
        )

    def available(self, device: str = "cpu") -> bool:
        try:
            return ((self.cccp_root / "cccp" / "__main__.py").is_file()
                    and Path(self._python(device)).is_file())
        except CCCPEngineError:
            return False

    def preflight(self, cfg: LaunchConfig, *, check_port: bool = True) -> dict:
        """启动前静态门禁；不加载权重。"""
        errors: list[str] = []
        warnings: list[str] = []
        if not self.available(cfg.device):
            errors.append(f"CCCP 推理引擎或 {cfg.device} 独立 Python 环境不可用")
        model = inspect_model(cfg.model_path)
        errors.extend(model.errors)
        if (
            cfg.combination.model_manifest_sha256
            and model.manifest_sha256
            and cfg.combination.model_manifest_sha256 != model.manifest_sha256
        ):
            errors.append(
                "配置记录的模型版本与当前所选模型不一致；"
                "请改选匹配模型或重新生成配置"
            )
        runtime: dict = {}
        if cfg.device not in {"cpu", "cuda", "amd"}:
            errors.append("device 必须是 cpu、cuda 或 amd")
        else:
            runtime = probe_backend(self.settings, cfg.device, self.cccp_root)
            if not runtime["ready"]:
                errors.append(f"{runtime['label']} 环境不可用：{runtime['reason']}")
        if cfg.cpu_compile not in {"auto", "off", "q4"}:
            errors.append("cpu_compile 必须是 auto、off 或 q4")
        if cfg.profile_mode not in {"auto", "mapped"}:
            errors.append("profile_mode 非法")
        if cfg.device == "cpu" and cfg.extreme:
            errors.append("CPU 模式不能启用 extreme；该模式用于 RAM+VRAM 极限常驻")
        if cfg.device == "cpu" and cfg.vram_gb:
            errors.append("CPU 模式不能设置 vram_gb")
        cache_gb = float(cfg.cache_gb if cfg.cache_gb is not None else self.settings.expert_cache_gb)
        has_dynamic_experts = bool(
            getattr(model, "has_dynamic_experts", True)
        )
        if has_dynamic_experts and cache_gb < 0.25:
            errors.append("专家缓存至少为 0.25 GiB")
        if cfg.max_ctx and not 64 <= int(cfg.max_ctx) <= max(
            64, model.max_context or 32768
        ):
            errors.append("max_ctx 超出模型或启动器允许范围")
        planning_context = (
            int(cfg.max_ctx)
            if cfg.max_ctx
            # Admission only needs the small startup working set.  This value
            # is deliberately not forwarded to the engine and therefore is
            # not a context ceiling: KV grows with the real request up to the
            # limit declared by the model manifest.
            else min(512, max(64, int(model.max_context or 512)))
        )
        total_ram, available_ram = _memory_status()
        # 配置预算与运行内存是两个不同概念。配置预算先包含 Dense+共享专家，
        # 余量才装入动态专家；运行估算则由固定权重、专家执行缓存、上下文和
        # 少量工作区组成。不要再对 Dense 重复乘 1.2 或再固定加 2 GiB，
        # 否则 24 GiB 配置会被错误膨胀为越过 32 GiB 的高风险任务。
        is_gpu = cfg.device in {"cuda", "amd"}
        gpu_plan = estimate_gpu_vram_plan(
            model,
            max_ctx=planning_context,
            expert_cache_gb=cache_gb,
        )
        dense_runtime = (
            float(gpu_plan["dense_runtime_gb"])
            if is_gpu else model.dense_gb
        )
        context_gb = (
            float(gpu_plan["context_workspace_gb"])
            if is_gpu else 0.5
        )
        overhead_gb = 0.75
        # This is combined host+device residency, not a claim that the whole
        # expert profile must fit VRAM.  GPU admission is governed by the two
        # explicit thresholds below.
        effective_cache_gb = cache_gb if has_dynamic_experts else 0.0
        estimated = dense_runtime + effective_cache_gb + context_gb + overhead_gb
        runtime_capacity = float(runtime.get("device_memory_gb") or 0.0)
        runtime_available = float(
            runtime.get("device_available_memory_gb") or 0.0
        )
        physical_capacity = (
            runtime_capacity if cfg.device in {"cuda", "amd"} and runtime_capacity
            else total_ram
        )
        current_available = (
            (runtime_available or runtime_capacity)
            if cfg.device in {"cuda", "amd"}
            else available_ram
        )
        capacity_kind = "vram" if is_gpu else "ram"
        capacity_label = "显存" if capacity_kind == "vram" else "内存"
        close_program_hint = (
            "建议先关闭占用显存较大的程序"
            if capacity_kind == "vram"
            else "建议先关闭占用内存较大的程序"
        )
        # 0 表示自动：CPU 取物理 RAM，GPU 取运行时探测到的 VRAM。
        limit = float(cfg.memory_limit_gb) if cfg.memory_limit_gb > 0 else physical_capacity
        device_capacity = min(limit, physical_capacity) if physical_capacity else limit
        risk_level = "safe"
        ram_offload_likely = False
        disk_offload_likely = False
        risk_reasons: list[dict[str, str | float]] = []
        offload_target = "none"
        gpu_execution_tier = "not_applicable"
        minimum_vram_gb = 0.0
        recommended_vram_gb = 0.0
        expert_vram_capacity_gb = 0.0
        if is_gpu:
            # GPU capacity has two independent gates.  Dense + CUDA workspace
            # is a hard requirement; expert VRAM is elastic and may shrink to
            # bounded chunks without dropping a single configured expert.
            available_vram = min(
                value for value in (device_capacity, current_available)
                if value > 0
            ) if (device_capacity > 0 or current_available > 0) else 0.0
            minimum_vram_gb = float(gpu_plan["minimum_vram_gb"])
            recommended_vram_gb = float(gpu_plan["recommended_vram_gb"])
            expert_vram_capacity_gb = max(
                0.0, available_vram - minimum_vram_gb
            )

            if available_vram <= 0:
                gpu_execution_tier = "unknown"
            elif available_vram < minimum_vram_gb:
                gpu_execution_tier = "below_minimum"
                risk_level = "danger"
                offload_target = "cpu"
                shortfall = minimum_vram_gb - available_vram
                hybrid_dense_ram = bool(gpu_plan.get("hybrid_dense_ram"))
                capacity_explanation = (
                    "RAM Dense 混合模式的 CUDA 基础工作区与最小专家块也无法容纳；"
                    if hybrid_dense_ram
                    else "缩小专家显存块也无法释放 Dense/工作区；"
                )
                message = (
                    f"当前可用显存约 {available_vram:.2f} GiB，低于该模型的 "
                    f"动态上下文初始 CUDA 工作集 "
                    f"{minimum_vram_gb:.2f} GiB（尚差约 {shortfall:.2f} GiB）。"
                    f"{capacity_explanation}请关闭占用显存的程序，"
                    "或改用 CPU 推理"
                )
                risk_reasons.append({
                    "code": "cuda_working_set_insufficient",
                    "message": message,
                })
                errors.append(message)
            elif available_vram < recommended_vram_gb:
                gpu_execution_tier = "reduced_expert_arena"
                ram_offload_likely = True
                offload_target = "ram"
                risk_level = "warning"
                hybrid_dense_ram = bool(gpu_plan.get("hybrid_dense_ram"))
                reason_code = (
                    "below_recommended_vram"
                    if device_capacity and recommended_vram_gb > device_capacity
                    else "vram_currently_busy"
                )
                risk_reasons.append({
                    "code": reason_code,
                    "message": (
                        f"当前可用显存约 {available_vram:.2f} GiB，低于全速建议值 "
                        f"{recommended_vram_gb:.2f} GiB，但高于 CUDA 最低工作集 "
                        f"{minimum_vram_gb:.2f} GiB；将自动缩小专家显存块并使用主机内存，"
                        + (
                            "Kimi Dense 也会驻留主机内存、速度会明显低于全显存模式，"
                            if hybrid_dense_ram else ""
                        )
                        + "不会减少专家"
                    ),
                })
                busy_hint = (
                    f"；{close_program_hint}可增大专家显存块"
                    if current_available and runtime_capacity
                    and current_available < runtime_capacity
                    else ""
                )
                warnings.append(
                    f"显存低于全速建议值：当前约 {available_vram:.2f} GiB / "
                    f"建议 {recommended_vram_gb:.2f} GiB；引擎会自动分块，"
                    f"完整专家配置保留在主机内存{busy_hint}"
                )
            else:
                gpu_execution_tier = "recommended"

            if (
                has_dynamic_experts
                and
                gpu_execution_tier not in {"below_minimum", "unknown"}
                and cache_gb > expert_vram_capacity_gb
            ):
                # The host copy is intentional dynamic-expert residency, not
                # an indication that the fixed CUDA working set failed.
                ram_offload_likely = True
                offload_target = "ram"

            hybrid_dense_ram = bool(
                gpu_plan.get("hybrid_dense_ram")
                and available_vram < recommended_vram_gb
            )
            host_expert_need = (
                effective_cache_gb + overhead_gb
                + (model.dense_gb if hybrid_dense_ram else 0.0)
            )
            if (
                gpu_execution_tier != "below_minimum"
                and available_ram < host_expert_need
                and ram_offload_likely
            ):
                disk_offload_likely = True
                offload_target = "disk"
                risk_level = "danger"
                ram_shortage = host_expert_need - max(0.0, available_ram)
                risk_reasons.append({
                    "code": "host_memory_insufficient",
                    "message": (
                        f"完整专家配置预计需要约 {host_expert_need:.2f} GiB 主机内存，"
                        f"当前仅可用 {available_ram:.2f} GiB；仍缺约 "
                        f"{ram_shortage:.2f} GiB，将继续降级到磁盘"
                    ),
                })
                warnings.append(
                    f"主机内存仍不足：建议关闭占用内存较大的程序；若继续，"
                    f"约 {ram_shortage:.2f} GiB 将由磁盘映射/系统虚拟内存兜底，"
                    "推理速度会明显变慢"
                )
        else:
            if estimated > device_capacity:
                risk_level = "danger"
                disk_offload_likely = True
                offload_target = "disk"
                risk_reasons.append({
                    "code": "device_capacity_exceeded",
                    "message": (
                        f"预计运行占用 {estimated:.2f} GiB 超过内存可用上限 "
                        f"{device_capacity:.2f} GiB；将自动使用磁盘映射或系统虚拟内存"
                    ),
                })
                warnings.append(
                    f"预计运行内存 {estimated:.2f} GiB 超过内存可用上限 "
                    f"{device_capacity:.2f} GiB；继续时将自动换页到磁盘，"
                    "推理速度会明显变慢"
                )
            if current_available and current_available < estimated:
                if risk_level != "danger":
                    risk_level = "warning"
                disk_offload_likely = True
                offload_target = "disk"
                shortage = estimated - current_available
                risk_reasons.append({
                    "code": "system_memory_busy",
                    "message": (
                        f"当前可用内存 {current_available:.2f} GiB，低于本次预计需要的 "
                        f"{estimated:.2f} GiB，尚差约 {shortage:.2f} GiB"
                    ),
                })
                warnings.append(
                    f"当前可用内存 {current_available:.2f} GiB 低于预计运行需要 "
                    f"{estimated:.2f} GiB；{close_program_hint}。"
                    "若仍继续，系统会自动换页到磁盘并明显降速"
                )
        full_model = cfg.combination.profile_ids == [FULL_MODEL_PROFILE_ID]
        if not full_model and cfg.combination.memory_gb > cache_gb:
            errors.append(
                f"配置专家 {cfg.combination.memory_gb:.2f} GiB 大于缓存 {cache_gb:.2f} GiB；"
                "严格路由要求完整保留所选专家；请使用更大内存/显存的设备，或改用强制磁盘映射"
            )
        if not full_model and model.expert_layers and model.top_k:
            layer_counts: dict[int, int] = {}
            for key in cfg.combination.union:
                layer = int(key.split(":", 1)[0])
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            missing = [
                layer for layer in model.expert_layers
                if layer_counts.get(layer, 0) < model.top_k
            ]
            if missing:
                preview = ",".join(str(layer) for layer in missing[:8])
                errors.append(
                    f"配置有 {len(missing)} 层不足 top_k={model.top_k} 个专家"
                    f"（例如 L{preview}），不能启用严格路由"
                )
        if check_port and not port_available(cfg.host, cfg.port):
            errors.append(f"端口 {cfg.host}:{cfg.port} 已被占用或不可绑定")
        native_dir = self.cccp_root / "cccp" / "native"
        native_ops = sorted(native_dir.glob("cccp_cpu_kernels_v*.pyd"))
        if cfg.device == "cpu" and not native_ops:
            warnings.append("未发现随包 CPU 原生算子；启动时将尝试本机自动编译")
        if cfg.device in {"cuda", "amd"}:
            warnings.append(
                f"{runtime.get('label', cfg.device)} 融合算子会在首次启动时针对当前设备自动编译；"
                "若编译失败将停止启动并在终端给出完整原因"
            )
        preflight_status = "blocked" if errors else risk_level
        return {
            "ok": not errors,
            "status": preflight_status,
            "errors": errors,
            "warnings": warnings,
            "inference_runtime": runtime,
            "model": model.__dict__,
            "memory": {
                "total_gb": round(total_ram, 2),
                "available_gb": round(current_available, 2),
                "host_total_gb": round(total_ram, 2),
                "host_available_gb": round(available_ram, 2),
                "device_total_gb": round(physical_capacity, 2),
                "device_available_gb": round(current_available, 2),
                "capacity_kind": capacity_kind,
                "capacity_label": capacity_label,
                "limit_gb": round(limit, 2),
                "device_capacity_gb": round(device_capacity, 2),
                "risk_level": risk_level,
                "risk_reasons": risk_reasons,
                "ram_offload_likely": ram_offload_likely,
                "disk_offload_likely": disk_offload_likely,
                "offload_target": offload_target,
                "automatic_offload_mode": {
                    "none": "none",
                    "ram": "gpu_to_host_ram",
                    "cpu": "gpu_to_cpu_fallback",
                    "disk": (
                        "gpu_to_host_ram_to_mapped_disk"
                        if is_gpu else "ram_to_mapped_disk"
                    ),
                }[offload_target],
                "gpu_execution_tier": gpu_execution_tier,
                "minimum_vram_gb": round(minimum_vram_gb, 2),
                "recommended_vram_gb": round(recommended_vram_gb, 2),
                "expert_vram_capacity_gb": round(expert_vram_capacity_gb, 2),
                "hybrid_dense_ram": bool(
                    gpu_plan.get("hybrid_dense_ram")
                    and is_gpu
                    and current_available < recommended_vram_gb
                ),
                "dense_runtime_estimate_gb": round(dense_runtime, 2),
                "dense_source_gb": model.dense_gb,
                "dense_without_shared_source_gb": model.dense_without_shared_gb,
                "shared_expert_source_gb": model.shared_expert_gb,
                "routed_expert_source_gb": cfg.combination.memory_gb,
                "configuration_source_resident_gb": round(
                    model.dense_gb + cfg.combination.memory_mb / 1024.0, 3
                ),
                "expert_cache_gb": round(cache_gb, 2),
                "context_estimate_gb": round(context_gb, 3),
                "runtime_workspace_gb": round(overhead_gb, 2),
                "total_estimate_gb": round(estimated, 2),
            },
            "native_cpu_operator": {
                "bundled": bool(native_ops),
                "files": [path.name for path in native_ops],
                "auto_build_fallback": True,
                "avx2": True,
            },
        }

    # -- 文件生成:profile.json / extreme score file --
    @staticmethod
    def _score_file(combo: Combination, job: str = "launch") -> Path:
        """生成 extreme placement 偏好文件(cccp-expert-residency-scores-v1)。

        组合内专家给 1.0;drop 解析结果仅作 meta 提示(不改变 CCCP,
        CCCP 的 drop masking 在路由前自动生效。
        """
        payload = {
            "schema": "cccp-expert-residency-scores-v1",
            "scores": {k: 1.0 for k in combo.union},
            "meta": {
                "generator": "cccp-winui-launcher",
                "profiles": combo.profile_ids,
                "expert_count": len(combo.union),
                "drop_resolution": combo.drop_resolution,
            },
        }
        path = RUNTIME_DIR / f"scores-{job}.json"
        atomic_write_text(path, json.dumps(payload))
        return path

    @staticmethod
    def _profile_counts_file(combo: Combination, job: str = "launch") -> Path:
        """生成 CCCP_PROFILE_JSON 计数档案(schema counts:{layer:{expert:count}})。"""
        counts: dict[str, dict[str, int]] = {}
        for key, expert in combo.union.items():
            layer, eid = key.split(":", 1)
            counts.setdefault(layer, {})[eid] = max(1, int(expert.route_count))
        payload = {
            "schema": "cccp-runtime-route-profile-v1",
            "counts": counts,
            "allowed_experts": {
                layer: sorted((int(expert) for expert in experts), key=int)
                for layer, experts in counts.items()
            },
            "strict_route": True,
            "load_all": True,
            "profiles": combo.profile_ids,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        path = RUNTIME_DIR / f"profile-{job}-{fingerprint}.json"
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False))
        return path

    # -- 命令行构造 --
    def build_command(self, cfg: LaunchConfig, *, dry_run: bool = False) -> list[str]:
        automatic_amd_capacity = (
            cfg.device == "amd" and cfg.profile_mode == "auto"
        )
        cmd = [
            self._python(cfg.device), "-m", "cccp", "launch", "serve",
            "--model", cfg.model_path,
            "--host", cfg.host,
            "--port", str(cfg.port),
            "--served-model-name", cfg.served_model_name,
            "--profile", cfg.profile_mode,
            "--device", "cuda" if cfg.device == "amd" else cfg.device,
            "--dense-residency", cfg.dense_residency,
            "--cpu-compile", cfg.cpu_compile,
            "--metrics-jsonl", str(CHAT_METRICS_FILE),
        ]
        if cfg.max_ctx:
            # Only an explicit diagnostic/reproduction override becomes a
            # command-line ceiling. Normal GUI launches stay dynamic.
            cmd += ["--max-ctx", str(cfg.max_ctx)]
        cache_gb = cfg.cache_gb if cfg.cache_gb is not None else self.settings.expert_cache_gb
        if (
            cache_gb is not None
            and not cfg.extreme
            and not automatic_amd_capacity
        ):
            cmd += ["--cache-gb", str(cache_gb)]
        if cfg.vram_gb:
            cmd += ["--vram-gb", str(cfg.vram_gb)]
        if cfg.extreme:
            cmd += ["--extreme", "--extreme-placement", "auto",
                    "--extreme-score-file", str(self._score_file(cfg.combination))]
        elif not automatic_amd_capacity:
            cmd.append("--no-extreme")
        if dry_run:
            cmd.append("--dry-run")
        cmd += cfg.extra_args
        return cmd

    def _env(self, cfg: LaunchConfig) -> dict[str, str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        import_paths = [str(self.cccp_root)]
        vendor = self.cccp_root / "_vendor"
        if vendor.is_dir():
            import_paths.append(str(vendor))
        if existing:
            import_paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(import_paths)
        full_model = cfg.combination.profile_ids == [FULL_MODEL_PROFILE_ID]
        if full_model:
            # 全量模式不制造虚假的受限 Profile。即使模型目录自带 profile.json，
            # CCCP_ROUTE_PROFILE=0 也只把它当热度提示，路由器仍可选择全部专家。
            env.pop("CCCP_PROFILE_JSON", None)
            env["CCCP_ROUTE_PROFILE"] = "0"
            env["CCCP_PROFILE_FULL_LOAD"] = "0"
        else:
            env["CCCP_PROFILE_JSON"] = str(self._profile_counts_file(cfg.combination))
            env["CCCP_ROUTE_PROFILE"] = "1"
            env["CCCP_PROFILE_FULL_LOAD"] = "1"
        env["CCCP_CPU_Q4_HOT_PER_LAYER"] = "8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # 推理引擎位于发行根/engine 下；显式传根目录，确保 CUDA/AMD 首次
        # 编译始终能发现随包 MSVC 与 Windows SDK，不依赖当前工作目录。
        env["CCCP_LAUNCHER_ROOT"] = str(self.cccp_root.parent.parent.resolve())
        env["CCCP_RUNTIME_BACKEND"] = cfg.device
        # GPU JIT 产物属于本机缓存，不属于某个版本的解压目录。固定路径使
        # 同一套 Torch/CUDA、源码与显卡架构在升级或移动启动器后仍可复用。
        cache_root = operator_cache_dir()
        env["CCCP_OPERATOR_CACHE_DIR"] = str(cache_root)
        env["TORCH_EXTENSIONS_DIR"] = str(cache_root / "torch")
        env["CCCP_OPERATOR_BUILD_PROGRESS"] = "1"
        env["CCCP_OPERATOR_BUILD_HEARTBEAT_S"] = "5"
        if cfg.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
            # 全量模式先尝试完整常驻；预检发现 RAM 不足后 launch() 会把模式
            # 切到 mapped，此时关闭全量常驻并由 LRU + 映射文件逐步降级。
            env["CCCP_FULL_RESIDENT"] = (
                "1" if full_model and cfg.profile_mode == "auto" else "0"
            )
            env["CCCP_CPU_COMPILE"] = cfg.cpu_compile
            env["CCCP_CPU_AUTOBUILD"] = "1"
            env["CCCP_CPU_HIGH_PRIORITY"] = "1"
            env["CCCP_CPU_PCORE_AFFINITY"] = "1"
            if cfg.cpu_threads > 0:
                env["CCCP_CPU_THREADS"] = str(cfg.cpu_threads)
        elif cfg.device == "amd":
            # ROCm PyTorch deliberately exposes HIP through torch.cuda APIs.
            # PyTorch cpp_extension HIPifies CCCP's .cu source and calls hipcc.
            #
            # The desktop launcher owns the automatic profile decision.  Do
            # not let a stale parent-shell diagnostic override the selected
            # profile inside the engine process.  In particular, inheriting
            # ``CCCP_SINGLE_GPU_LAYER_GRAPH=0`` while the capacity planner
            # selects the resident profile leaves every expert on the GPU but
            # silently executes all 43 layers through eager Python dispatch.
            # That exact split is visible as H2D=0 together with
            # decode_graph=0 and costs most of the AMD decode throughput.
            # Removing these inherited values is AMD-launcher isolation only;
            # the resolved engine profile immediately supplies its own values.
            for key in (
                "CCCP_PACKED_FULL_GPU",
                "CCCP_SINGLE_GPU_LAYER_GRAPH",
                "CCCP_DSV4_TOKEN_GRAPH",
                "CCCP_TP_LAYER_GRAPH",
                "CCCP_TP_HIDDEN",
                "CCCP_TP_NO_OWNER",
                "CCCP_STATIC_DECODE_GRAPHS",
                "CCCP_STATIC_FFN_GRAPH",
            ):
                env.pop(key, None)
            env["CCCP_REQUIRE_FUSED"] = "1"
            env["CCCP_FLASHINFER_MLA"] = "0"
            env["CCCP_PREFILL_ATTN_TRITON"] = "0"
        elif cfg.device == "cuda":
            # 用户系统/开发终端里的旧架构变量不能污染傻瓜式启动；融合算子
            # 在子进程内读取当前显卡，分别选择 SM86/89/90/120 等真实目标。
            env.pop("CCCP_CUDA_ARCH", None)
            env.pop("TORCH_CUDA_ARCH_LIST", None)
            env["CCCP_REQUIRE_FUSED"] = "1"
            if _WINDOWS:
                # Normal GUI launches use the engine's automatic layer-wide
                # native batch submission.  Remove inherited diagnostics so
                # one stale shell variable cannot silently restore per-expert
                # copy calls.
                env.pop("CCCP_H2D_BATCH", None)
                # Registered-source DMA is automatic as well. Page-locked
                # expert RAM is submitted directly; only genuinely pageable
                # sources use the CPU bounce ring.
                env.pop("CCCP_WDDM_DIRECT_PIN", None)
        if self.settings.cccp_api_key:
            env["CCCP_API_KEY"] = self.settings.cccp_api_key
        return env

    # -- 预检 --
    def dry_run(self, cfg: LaunchConfig) -> dict:
        preflight = self.preflight(cfg)
        if not preflight["ok"]:
            return {"ok": False, "preflight": preflight, "cmd": self.build_command(cfg, dry_run=True),
                    "stdout": "", "stderr": "；".join(preflight["errors"])}
        cmd = self.build_command(cfg, dry_run=True)
        try:
            proc = subprocess.run(
                cmd, cwd=str(self.cccp_root), env=self._env(cfg),
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "preflight": preflight, "cmd": cmd,
                    "stdout": "", "stderr": str(exc)}
        return {
            "ok": proc.returncode == 0,
            "preflight": preflight,
            "cmd": cmd,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }

    # -- 生命周期 --
    def launch(self, cfg: LaunchConfig) -> CCCPEngineInstance:
        if self._proc and self._proc.poll() is None:
            raise CCCPEngineError("已有 CCCP 实例在运行,请先停止")
        self._ready_pid = None
        preflight = self.preflight(cfg)
        self.last_preflight = preflight
        if not preflight["ok"]:
            raise CCCPEngineError("；".join(preflight["errors"]))
        memory = preflight.get("memory", {})
        if memory.get("ram_offload_likely") or memory.get("disk_offload_likely"):
            # Keep the complete expert set.  Mapped mode moves GPU cold data to
            # host memory; when physical RAM is also short Windows transparently
            # backs those pages with the system page file.  No expert is removed.
            cfg.profile_mode = "mapped"
            if (
                memory.get("disk_offload_likely")
                and cfg.combination.profile_ids == [FULL_MODEL_PROFILE_ID]
            ):
                # 预检按“全部专家常驻”给出真实风险；确认降级后，不能仍把
                # 69+ GiB 当作 RAM 缓存目标。保留全部专家可路由，只把实际
                # LRU 缓存收敛到当前可用物理内存，冷专家由磁盘映射补齐。
                model = inspect_model(cfg.model_path)
                total_ram, available_ram = _memory_status()
                usable = max(0.25, min(
                    value for value in (total_ram, available_ram) if value > 0
                ) - model.dense_gb - 1.0) if (total_ram > 0 or available_ram > 0) else 0.25
                cfg.cache_gb = round(min(cfg.combination.memory_gb, usable), 3)
        # Metrics belong to the current engine instance.  Keeping the file but
        # truncating it avoids showing a prior model's KV/prefill result.
        CHAT_METRICS_FILE.write_text("", encoding="utf-8")
        log_path = RUNTIME_DIR / "cccp-serve.log"
        self._log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
        self._log_handle.write(f"\n===== launch {time.strftime('%F %T')} profiles={cfg.profiles} =====\n")
        self._log_handle.write("[cccp-winui-progress] phase=spawn current=1 total=100\n")
        if memory.get("disk_offload_likely"):
            self._log_handle.write(
                "[cccp-winui-offload] target=disk 显存/内存逐级检查后仍不足："
                "已启用磁盘映射/系统虚拟内存兜底；"
                "全部配置专家保持不变，模型会继续加载，但推理速度会明显变慢。\n"
            )
        elif memory.get("ram_offload_likely"):
            if memory.get("gpu_execution_tier") == "reduced_expert_arena":
                detail = (
                    "显存低于全速建议值：已缩小 GPU 专家块；完整专家配置保留在"
                    "主机内存，专家数量不变"
                )
            else:
                detail = (
                    "已启用动态专家内存驻留：GPU 使用剩余显存作为热缓存，"
                    "完整专家配置保留在主机内存"
                )
            self._log_handle.write(
                f"[cccp-winui-offload] target=ram {detail}；"
                "当前内存充足，不会使用磁盘兜底。\n"
            )
        cmd = self.build_command(cfg)
        log.info("spawn: %s", " ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=str(self.cccp_root), env=self._env(cfg),
                stdout=self._log_handle, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            self._log_handle.close()
            self._log_handle = None
            raise CCCPEngineError(f"无法启动 CCCP: {exc}") from exc
        self.instance = CCCPEngineInstance(
            pid=self._proc.pid,
            port=cfg.port,
            model=cfg.model_path,
            served_model_name=cfg.served_model_name,
            profiles=list(cfg.profiles),
            started_at=time.time(),
            log_file=str(log_path),
            base_url=f"http://{cfg.host}:{cfg.port}",
            full_model=cfg.combination.profile_ids == [FULL_MODEL_PROFILE_ID],
        )
        time.sleep(0.3)
        if self._proc.poll() is not None:
            code = self._proc.returncode
            tail = self.tail_log(30)
            self.stop()
            raise CCCPEngineError(f"CCCP 启动后立即退出(code={code})\n{tail}")
        return self.instance

    def latest_chat_metrics(self, request_id: str | None = None) -> dict:
        """Return the newest matching content-free engine metric record."""
        try:
            lines = CHAT_METRICS_FILE.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                record = json.loads(line)
                if request_id is None or record.get("request_id") == request_id:
                    return record
            return {}
        except (OSError, json.JSONDecodeError):
            return {}

    def stop(self) -> bool:
        if not self._proc:
            self._ready_pid = None
            return False
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        self._proc = None
        self._ready_pid = None
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None
        self.instance = None
        return True

    # -- 健康 / 状态(HTTP 集成面) --
    async def health(self) -> dict:
        if not self.instance:
            self._ready_pid = None
            return {"ready": False, "running": False}
        if self._proc and self._proc.poll() is not None:
            self._ready_pid = None
            return {"ready": False, "running": False, "exit": self._proc.returncode}
        try:
            async with httpx.AsyncClient(timeout=3) as cli:
                r = await cli.get(f"{self.instance.base_url}/health")
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                body["running"] = True
                if body.get("ready"):
                    self._ready_pid = self.instance.pid
                return body
        except (httpx.HTTPError, OSError):
            if self._ready_pid == self.instance.pid:
                return {
                    "ready": True,
                    "running": True,
                    "busy": True,
                    "note": "模型已就绪；推理期间健康端点暂时繁忙",
                }
            return {"ready": False, "running": True, "note": "进程存活,/health 尚未就绪"}

    async def wait_ready(self, timeout_s: float = 600.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            h = await self.health()
            if h.get("ready"):
                return True
            if not h.get("running"):
                return False
            await asyncio.sleep(2.0)
        return False

    def loading_progress(self, health: dict | None = None) -> dict:
        """从当前启动段日志提取可供 UI 使用的结构化加载进度。"""
        if not self.instance:
            return {
                "state": "idle", "percent": 0, "phase": "idle",
                "label": "尚未启动", "detail": "请选择模型与配置后启动",
                "elapsed_s": 0.0,
            }
        elapsed = max(0.0, time.time() - self.instance.started_at)
        if health and health.get("ready"):
            return {
                "state": "ready", "percent": 100, "phase": "ready",
                "label": "模型已就绪", "detail": "OpenAI 兼容接口可以接收请求",
                "elapsed_s": round(elapsed, 1),
            }
        if health and not health.get("running", True):
            return {
                "state": "failed", "percent": 100, "phase": "failed",
                "label": "模型进程已退出",
                "detail": f"exit={health.get('exit', 'unknown')}，请查看下方日志",
                "elapsed_s": round(elapsed, 1),
            }

        text = self.tail_log(800)
        current_launch = text.rsplit("===== launch ", 1)[-1]
        percent = 3
        phase = "spawn"
        label = "正在创建 CCCP 进程"
        detail = "等待推理引擎开始读取模型"
        ram_offload = "[cccp-winui-offload] target=ram" in current_launch
        disk_offload = "[cccp-winui-offload] target=disk" in current_launch

        compile_matches = list(re.finditer(
            r"\[cccp-winui-progress\]\s+phase=operator-build\s+"
            r"event=([a-z]+)\s+backend=([A-Za-z0-9_-]+)\s+elapsed=(\d+)",
            current_launch,
        ))

        matches = list(re.finditer(
            r"\[cccp-winui-progress\]\s+phase=experts\s+current=(\d+)\s+total=(\d+)",
            current_launch,
        ))
        pin_matches = list(re.finditer(
            r"\[cccp-winui-progress\]\s+phase=expert-pin\s+"
            r"current=(\d+)\s+total=(\d+)",
            current_launch,
        ))
        upload_matches = list(re.finditer(
            r"\[cccp-winui-progress\]\s+phase=expert-upload\s+"
            r"current=(\d+)\s+total=(\d+)",
            current_launch,
        ))
        indeterminate = False
        if upload_matches:
            current = int(upload_matches[-1].group(1))
            total = max(1, int(upload_matches[-1].group(2)))
            percent = min(98, 90 + round(8 * current / total))
            phase = "expert-upload"
            label = "正在把配置专家预热到 GPU"
            detail = f"已上传 {current}/{total} 个专家；完成后首轮对话无需冷加载"
        elif pin_matches:
            current = int(pin_matches[-1].group(1))
            total = max(1, int(pin_matches[-1].group(2)))
            percent = min(94, 90 + round(4 * current / total))
            phase = "expert-pin"
            label = "正在启用 RAM 到 GPU 高速通道"
            detail = (
                f"已锁页 {current / 1024:.1f}/{total / 1024:.1f} GiB；"
                "无法锁页的部分会自动使用普通内存"
            )
        elif matches:
            current = int(matches[-1].group(1))
            total = max(1, int(matches[-1].group(2)))
            percent = min(90, 10 + round(80 * current / total))
            phase = "experts"
            label = "正在加载并编译配置专家"
            detail = f"已处理 {current}/{total} 个专家；全部专家编号保持不变"
        elif compile_matches:
            event, backend, compile_elapsed = compile_matches[-1].groups()
            backend_name = {
                "CPU": "CPU",
                "NVIDIA-CUDA": "NVIDIA CUDA",
                "AMD-HIP": "AMD HIP",
            }.get(backend, backend)
            if event in {"start", "running"}:
                percent, phase = 8, "operator-build"
                label = f"正在编译 {backend_name} 加速算子"
                detail = (
                    f"编译器仍在运行 · 已用 {compile_elapsed} 秒；"
                    "详细 Ninja/编译器输出见下方终端"
                )
                indeterminate = True
            elif event == "success":
                percent, phase = 10, "operators"
                label = f"{backend_name} 加速算子已编译并缓存"
                detail = "正在装入模型与配置中的全部专家"
            else:
                percent, phase = 10, "operator-build-failed"
                label = f"{backend_name} 加速算子编译失败"
                detail = "推理进程将停止；请查看下方编译器错误"
        elif "CPU Profile 热集预载" in current_launch:
            percent, phase = 92, "execution_graph"
            label = "专家执行映像已驻留"
            detail = "正在构建 CPU MoE 执行图"
        elif "CPU融合内核编译成功" in current_launch:
            percent, phase = 10, "operators"
            label = "CPU 加速算子已就绪"
            detail = "正在装入配置中的全部专家"
        elif "[cccp-launch]" in current_launch:
            percent, phase = 6, "manifest"
            label = "模型清单校验完成"
            detail = "正在初始化 CPU 推理运行时"

        if disk_offload and phase in {"spawn", "manifest", "operators", "experts", "expert-pin"}:
            detail += "；显存和内存均不足，正在使用磁盘卸载，速度较慢"
        elif ram_offload and phase in {"spawn", "manifest", "operators", "experts", "expert-pin"}:
            detail += "；显存不足，正在卸载到主机内存（未使用磁盘）"

        if "模型加载完成" in current_launch:
            percent, phase = 96, "server"
            label = "模型加载完成"
            detail = "正在启动本地 API 服务"
        if "Application startup complete" in current_launch:
            percent, phase = 99, "health"
            label = "API 服务已启动"
            detail = "正在等待健康检查通过"
        return {
            "state": "loading", "percent": percent, "phase": phase,
            "label": label, "detail": detail, "elapsed_s": round(elapsed, 1),
            "ram_offload": ram_offload,
            "disk_offload": disk_offload,
            "indeterminate": indeterminate,
        }

    def tail_log(self, lines: int = 80) -> str:
        lf = RUNTIME_DIR / "cccp-serve.log"
        if not lf.exists():
            return ""
        data = _decode_mixed_process_log(lf.read_bytes()).splitlines()
        return "\n".join(data[-lines:])

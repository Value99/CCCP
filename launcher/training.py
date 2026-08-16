"""Token 驱动的动态专家配置生成流程。

流程只有一条：持久化语料 -> 保留角色模板的纯 prefill 路由扫描 ->
专家热力图/覆盖率规划 -> 命名并保存模型专用配置。这里不生成模型权重。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Iterable

from .profiles import ProfileRegistry, SCHEMA
from .settings import DATA_DIR
from .io_utils import atomic_write_bytes, atomic_write_text

TRAIN_DIR = DATA_DIR / "training"
CORPUS_DIR = DATA_DIR / "corpus"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

JOBS_SCHEMA = "cccp-token-route-jobs-v2"
DEFAULT_LAYERS = 43
DEFAULT_EXPERTS_PER_LAYER = 256
DEFAULT_EXPERT_SIZE_MB = 6.4
DEFAULT_TOKEN_BUDGET = 500_000
MIN_TOKEN_BUDGET = 4096
MAX_TOKEN_BUDGET = 20_000_000
PREFILL_BLOCK_TOKENS = 4096
DEFAULT_ROUTE_COVERAGE = 0.95
SUPPORTED_CORPUS_SUFFIXES = {".jsonl", ".txt"}
SUPPORTED_ROLES = {"system", "developer", "user", "assistant"}


class TrainingCancelled(RuntimeError):
    """Raised after a user stops a route scan from the launcher UI."""


@dataclass
class TrainingJob:
    id: str
    corpus_files: list[str]
    model_path: str
    token_budget: int = DEFAULT_TOKEN_BUDGET
    prefill_block_tokens: int = PREFILL_BLOCK_TOKENS
    mode: str = "auto"
    sample_seed: int = 5090

    model_name: str = ""
    model_version: str = ""
    model_format: str = ""
    model_manifest_sha256: str = ""
    model_total_bytes: int = 0
    model_total_gib: float = 0.0
    model_top_k: int = 0
    model_max_context: int = 0
    layers: int = DEFAULT_LAYERS
    expert_layers: list[int] = field(default_factory=list)
    experts_per_layer: int = DEFAULT_EXPERTS_PER_LAYER
    expert_size_mb: float = DEFAULT_EXPERT_SIZE_MB
    dense_without_shared_gib: float = 0.0
    shared_expert_gib: float = 0.0
    fixed_model_gib: float = 0.0
    model_max_configuration_gib: float = 0.0

    status: str = "pending"
    data_source: str = "cccp-token-prefill-routes"
    calibrated: bool = False
    progress: float = 0.0
    message: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    processed_tokens: int = 0
    total_documents: int = 0
    scan_context_tokens: int = 0
    truncated_documents: int = 0
    tokenize_seconds: float = 0.0
    model_load_seconds: float = 0.0
    prefill_seconds: float = 0.0
    prefill_tokens_per_second: float = 0.0
    route_observations: int = 0
    cpu_operator_audit: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    coverage_target: float = DEFAULT_ROUTE_COVERAGE
    actual_coverage: float = 0.0
    layer_coverages: dict[str, float] = field(default_factory=dict)
    plan_keys: list[str] = field(default_factory=list)
    plan_bytes_mb: float = 0.0
    plan_sizes_mb: dict[str, float] = field(default_factory=dict)
    profile_name: str = ""
    profile_description: str = ""
    registered_profile_id: str = ""
    registered_profiles: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, with_counts: bool = False) -> dict[str, Any]:
        document = asdict(self)
        if not with_counts:
            document["counts"] = {}
            document["counts_truncated"] = bool(self.counts)
            document["plan_sizes_mb"] = {}
        document["dynamic_resident_gib"] = round(self.plan_bytes_mb / 1024.0, 3)
        document["configuration_resident_gib"] = round(
            self.fixed_model_gib + self.plan_bytes_mb / 1024.0, 3
        )
        return document


# ---------------------------------------------------------------------------
# 语料：上传后持久化，JSONL messages 保留角色与完整上下文
# ---------------------------------------------------------------------------

def _clean_message(raw: object) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    role = str(raw.get("role") or "").strip().lower()
    content = raw.get("content")
    if role not in SUPPORTED_ROLES or not isinstance(content, str) or not content.strip():
        return None
    return {"role": role, "content": content}


def _jsonl_record(line: str) -> dict[str, list[dict[str, str]]] | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    raw_messages = obj.get("messages")
    if isinstance(raw_messages, list):
        messages = [message for raw in raw_messages if (message := _clean_message(raw))]
        if messages:
            return {"messages": messages}
    text = obj.get("prompt") or obj.get("text") or obj.get("content")
    if isinstance(text, str) and text.strip():
        return {"messages": [{"role": "user", "content": text}]}
    return None


def _record_text(record: dict[str, list[dict[str, str]]]) -> str:
    return "\n".join(message["content"] for message in record["messages"])


def _jsonl_text(line: str) -> str | None:
    record = _jsonl_record(line)
    return _record_text(record).strip() if record else None


def iter_corpus_records(paths: Iterable[Path]) -> Iterable[dict[str, list[dict[str, str]]]]:
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_CORPUS_SUFFIXES:
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if path.suffix.lower() == ".jsonl":
                    record = _jsonl_record(line)
                    if record:
                        yield record
                else:
                    yield {"messages": [{"role": "user", "content": line}]}


def iter_corpus(paths: Iterable[Path]) -> Iterable[str]:
    """返回纯文本视图，仅用于语料检查；路由扫描使用结构化 records。"""
    for record in iter_corpus_records(paths):
        yield _record_text(record)


def prepare_scan_input(
    paths: Iterable[Path], destination: Path, token_budget: int, seed: int
) -> tuple[int, int]:
    """准备随机化候选集；实际停止条件由引擎 tokenizer 的累计 token 决定。"""
    rng = random.Random(seed)
    desired_characters = max(token_budget * 6, 2_000_000)
    reservoir: list[dict[str, list[dict[str, str]]]] = []
    characters = 0
    seen = 0
    for record in iter_corpus_records(paths):
        seen += 1
        record_characters = len(_record_text(record))
        if characters < desired_characters:
            reservoir.append(record)
            characters += record_characters
            continue
        slot = rng.randrange(seen)
        if slot < len(reservoir):
            characters -= len(_record_text(reservoir[slot]))
            reservoir[slot] = record
            characters += record_characters
    if not reservoir:
        raise ValueError("语料为空或没有符合格式的记录")
    rng.shuffle(reservoir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in reservoir:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return len(reservoir), characters


def _corpus_meta_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.meta.json")


def inspect_corpus(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    samples = invalid = nonempty = characters = messages = longest = 0
    roles: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            value = raw_line.strip()
            if not value or value.startswith("#"):
                continue
            nonempty += 1
            record = (
                _jsonl_record(value) if suffix == ".jsonl"
                else {"messages": [{"role": "user", "content": value}]}
            )
            if not record:
                invalid += 1
                continue
            samples += 1
            text_length = len(_record_text(record))
            characters += text_length
            longest = max(longest, text_length)
            messages += len(record["messages"])
            roles.update(message["role"] for message in record["messages"])
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    stat = path.stat()
    return {
        "meta_schema": 2,
        "name": path.name,
        "bytes": stat.st_size,
        "format": suffix.removeprefix(".").upper(),
        "samples": samples,
        "messages": messages,
        "characters": characters,
        "max_sample_characters": longest,
        "roles": sorted(roles),
        "invalid_lines": invalid,
        "nonempty_lines": nonempty,
        "sha256": digest.hexdigest(),
        "modified_at": stat.st_mtime,
        "stored_path": f"data/corpus/{path.name}",
        "persistent": True,
    }


def save_corpus_file(name: str, content: bytes) -> dict[str, Any]:
    safe = Path(name).name.replace("\\", "_").replace("/", "_")
    if not safe:
        raise ValueError("文件名不能为空")
    if Path(safe).suffix.lower() not in SUPPORTED_CORPUS_SUFFIXES:
        raise ValueError("语料仅支持 UTF-8 的 .jsonl 或 .txt 文件")
    if not content:
        raise ValueError("语料文件为空")
    path = CORPUS_DIR / safe
    atomic_write_bytes(path, content)
    meta = inspect_corpus(path)
    if meta["samples"] <= 0:
        path.unlink(missing_ok=True)
        raise ValueError("没有读到有效语料；JSONL messages 需包含 role 和 content")
    atomic_write_text(
        _corpus_meta_path(path), json.dumps(meta, ensure_ascii=False, indent=2)
    )
    return meta


def list_corpus() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(CORPUS_DIR.glob("*")):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in SUPPORTED_CORPUS_SUFFIXES:
            continue
        meta_path = _corpus_meta_path(path)
        meta: dict[str, Any] | None = None
        try:
            cached = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                cached.get("meta_schema") == 2
                and int(cached.get("bytes") or -1) == path.stat().st_size
            ):
                meta = cached
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
        if meta is None:
            meta = inspect_corpus(path)
            atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))
        result.append(meta)
    return result


def delete_corpus(name: str) -> bool:
    path = CORPUS_DIR / Path(name).name
    if not path.is_file():
        return False
    path.unlink()
    _corpus_meta_path(path).unlink(missing_ok=True)
    return True


# ---------------------------------------------------------------------------
# 真路由扫描与覆盖率规划
# ---------------------------------------------------------------------------

def load_expert_sizes(model_path: str | Path) -> dict[str, float]:
    model = Path(model_path)
    sizes: dict[str, float] = {}
    for audit_path in sorted(model.glob("experts.L*.audit.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        layer = int(audit["layer"])
        raw: dict[int, int] = {}
        for expert_id, detail in (audit.get("experts") or {}).items():
            projections = detail.get("projections")
            if isinstance(projections, dict):
                byte_count = sum(
                    int((projection or {}).get("packed_bytes") or 0)
                    for projection in projections.values()
                )
            else:
                byte_count = sum(
                    int((detail.get(projection) or {}).get("packed_bytes") or 0)
                    for projection in ("gate", "up", "down")
                )
            raw[int(expert_id)] = byte_count
        raw_total = sum(raw.values())
        file_bytes = int(audit.get("file_bytes") or audit.get("bytes") or raw_total)
        scale = file_bytes / raw_total if raw_total else 0.0
        equal_share = file_bytes / max(1, len(raw)) if not raw_total else 0.0
        for expert_id, byte_count in raw.items():
            calibrated_bytes = byte_count * scale if raw_total else equal_share
            sizes[f"{layer}:{expert_id}"] = round(calibrated_bytes / 2**20, 6)
    return sizes


def measured_activation(
    job: TrainingJob,
    scan_input: Path,
    *,
    engine_root: Path,
    python_path: Path,
    train_dir: Path = TRAIN_DIR,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    if not (engine_root / "cccp" / "__main__.py").is_file():
        raise RuntimeError("内置 CCCP 引擎缺少 route-scan 入口")
    if not python_path.is_file():
        raise RuntimeError("内置 CPU Python 环境不可用")
    route_path = train_dir / f"{job.id}.routes.json"
    report_path = train_dir / f"{job.id}.scan-report.json"
    command = [
        str(python_path), "-m", "cccp", "route-scan",
        "--model", job.model_path,
        "--input", str(scan_input),
        "--output", str(route_path),
        "--report", str(report_path),
        "--profile", "mapped" if job.mode == "disk" else "ram",
        "--token-budget", str(job.token_budget),
        "--prefill-block-tokens", str(PREFILL_BLOCK_TOKENS),
    ]
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join([str(engine_root), *([existing] if existing else [])])
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command,
        cwd=str(engine_root),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    tail: list[str] = []
    assert process.stdout is not None
    output_queue: Queue[str | None] = Queue()

    def read_output() -> None:
        try:
            for raw_line in process.stdout:
                output_queue.put(raw_line.rstrip())
        finally:
            output_queue.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    last_processed = 0
    last_stage = "加载模型"
    started = time.monotonic()
    last_heartbeat = started
    last_route_snapshot: tuple[int, int] | None = None

    def refresh_partial_counts() -> None:
        nonlocal last_route_snapshot
        try:
            stat = route_path.stat()
            signature = (int(stat.st_mtime_ns), int(stat.st_size))
            if signature == last_route_snapshot:
                return
            document = json.loads(route_path.read_text(encoding="utf-8"))
            if document.get("format") != "cccp-expert-residency-scores-v1":
                return
            job.counts = {
                str(key): int(float(value))
                for key, value in (document.get("scores") or {}).items()
                if float(value) > 0
            }
            job.route_observations = int(
                document.get("observations")
                or sum(job.counts.values())
            )
            last_route_snapshot = signature
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return

    while True:
        if cancel_event is not None and cancel_event.is_set():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise TrainingCancelled("用户已停止 token 路由扫描")
        try:
            line = output_queue.get(timeout=1.0)
        except Empty:
            line = ""
        if line is None:
            break
        if not line:
            now = time.monotonic()
            if progress_callback and now - last_heartbeat >= 10.0:
                elapsed = int(now - started)
                progress_callback(
                    last_processed,
                    job.token_budget,
                    f"{last_stage} · 本次扫描已运行 {elapsed:,} 秒",
                )
                last_heartbeat = now
            continue
        tail.append(line)
        del tail[:-40]
        marker = "CCCP_ROUTE_SCAN_PROGRESS "
        if line.startswith(marker):
            try:
                event = json.loads(line[len(marker):])
                last_processed = int(event.get("processed_tokens") or 0)
                last_stage = str(event.get("stage") or "prefill")
                refresh_partial_counts()
                if progress_callback:
                    progress_callback(
                        last_processed,
                        int(event.get("token_budget") or job.token_budget),
                        last_stage,
                    )
                    last_heartbeat = time.monotonic()
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    return_code = process.wait()
    if return_code != 0 or not route_path.is_file() or not report_path.is_file():
        raise RuntimeError("CCCP token 路由扫描失败：" + "\n".join(tail[-12:])[-1600:])
    document = json.loads(route_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if document.get("format") != "cccp-expert-residency-scores-v1":
        raise RuntimeError("CCCP 路由统计文件格式不受支持")
    counts = {
        str(key): int(float(value))
        for key, value in (document.get("scores") or {}).items()
        if float(value) > 0
    }
    if not counts:
        raise RuntimeError("CCCP 没有记录到任何专家路由")
    if int(report.get("processed_tokens") or 0) < job.token_budget:
        raise RuntimeError(
            f"语料 token 不足：需要 {job.token_budget:,}，实际只有 "
            f"{int(report.get('processed_tokens') or 0):,}；请增加语料后重试"
        )
    return counts, report


def plan_route_coverage(
    counts: dict[str, int],
    coverage: float,
    sizes_mb: dict[str, float] | None = None,
    default_size_mb: float = DEFAULT_EXPERT_SIZE_MB,
    *,
    top_k: int,
    layers: int,
    expert_layers: list[int] | None = None,
) -> tuple[list[str], float, float, dict[str, float]]:
    """逐层选择达到目标路由覆盖率的最小热专家前缀。"""
    if not 0.01 <= coverage <= 1.0:
        raise ValueError("专家覆盖率必须在 1% 到 100% 之间")
    if top_k <= 0 or layers <= 0:
        raise ValueError("模型 top-k 或层数无效")
    selected: list[str] = []
    layer_coverages: dict[str, float] = {}
    total_hits = selected_hits = 0
    layer_ids = (
        [int(layer) for layer in expert_layers]
        if expert_layers
        else list(range(layers))
    )
    if len(layer_ids) != len(set(layer_ids)):
        raise ValueError("专家层编号存在重复")
    for layer in layer_ids:
        prefix = f"{layer}:"
        ranked = sorted(
            ((key, int(value)) for key, value in counts.items() if key.startswith(prefix) and int(value) > 0),
            key=lambda item: (-item[1], item[0]),
        )
        if len(ranked) < top_k:
            raise ValueError(f"第 {layer} 层命中专家少于模型 top-k={top_k}")
        layer_total = sum(value for _, value in ranked)
        threshold = layer_total * coverage
        layer_selected = 0
        layer_picked = 0
        for key, value in ranked:
            if layer_picked >= top_k and layer_selected >= threshold:
                break
            selected.append(key)
            layer_selected += value
            layer_picked += 1
        total_hits += layer_total
        selected_hits += layer_selected
        layer_coverages[str(layer)] = round(layer_selected / layer_total, 6)
    size_of = lambda key: (sizes_mb or {}).get(key, default_size_mb)
    used_mb = round(sum(size_of(key) for key in selected), 1)
    actual = round(selected_hits / total_hits, 6) if total_hits else 0.0
    return selected, used_mb, actual, layer_coverages


# ---------------------------------------------------------------------------
# 导出：配置体积等于当前覆盖率选择出来的真实驻留体积
# ---------------------------------------------------------------------------

def export_scores(job: TrainingJob) -> dict[str, Any]:
    selected = set(job.plan_keys)
    max_hit = max(job.counts.values(), default=1)
    scores: dict[str, float] = {}
    layer_ids = job.expert_layers or list(range(job.layers))
    for layer in layer_ids:
        for expert in range(job.experts_per_layer):
            key = f"{layer}:{expert}"
            scores[key] = round(job.counts.get(key, 0) / max_hit, 6) if key in selected else 0.0
    return {
        "schema": "cccp-expert-residency-scores-v1",
        "scores": scores,
        "meta": {
            "generator": "cccp-winui-launcher/token-route-training",
            "job": job.id,
            "processed_tokens": job.processed_tokens,
            "prefill_block_tokens": job.prefill_block_tokens,
            "target_route_coverage": job.coverage_target,
            "actual_route_coverage": job.actual_coverage,
            "selected": len(selected),
        },
    }


def export_counts(job: TrainingJob) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    for key, value in job.counts.items():
        layer, expert = key.split(":", 1)
        counts.setdefault(layer, {})[expert] = value
    return {
        "counts": counts,
        "meta": {
            "job": job.id,
            "data_source": job.data_source,
            "processed_tokens": job.processed_tokens,
        },
    }


def export_profile(
    job: TrainingJob, name: str = "", description: str | None = None
) -> dict[str, Any]:
    if job.status != "done" or not job.plan_keys:
        raise ValueError("路由扫描和覆盖率规划尚未完成")
    max_hit = max(job.counts.values(), default=1)
    experts = [
        {
            "key": key,
            "size_mb": job.plan_sizes_mb.get(key, job.expert_size_mb),
            "tags": ["trained", "token-prefill"],
            "route_count": int(job.counts[key]),
            "route_score": round(job.counts[key] / max_hit, 6),
        }
        for key in job.plan_keys
    ]
    configuration_resident = round(job.fixed_model_gib + job.plan_bytes_mb / 1024.0, 3)
    final_name = str(name or job.profile_name).strip()
    final_description = (
        str(description).strip() if description is not None else job.profile_description.strip()
    )
    variant_material = json.dumps(
        {
            "name": final_name,
            "target": round(float(job.coverage_target), 6),
            "experts": sorted(job.plan_keys),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    variant_id = hashlib.sha256(variant_material.encode("utf-8")).hexdigest()[:10]
    meta: dict[str, Any] = {
        "source": "trained",
        "calibrated": job.calibrated,
        "training_job": job.id,
        "model_name": job.model_name,
        "model_version": job.model_version or job.model_format,
        "model_format": job.model_format,
        "model_total_bytes": job.model_total_bytes,
        "model_total_gib": job.model_total_gib,
        "model_manifest_sha256": job.model_manifest_sha256,
        "model_layers": job.layers,
        "model_expert_layers": job.expert_layers or list(range(job.layers)),
        "model_experts_per_layer": job.experts_per_layer,
        "model_top_k": job.model_top_k,
        "fixed_model_gib": job.fixed_model_gib,
        "dense_without_shared_gib": job.dense_without_shared_gib,
        "shared_expert_gib": job.shared_expert_gib,
        "routed_expert_budget_gib": round(job.plan_bytes_mb / 1024.0, 3),
        "configuration_budget_gib": configuration_resident,
        "configuration_resident_gib": configuration_resident,
        "selected_experts": len(experts),
        "strict_route": True,
        "route_token_budget": job.token_budget,
        "route_processed_tokens": job.processed_tokens,
        "prefill_block_tokens": job.prefill_block_tokens,
        "target_route_coverage": job.coverage_target,
        "actual_route_coverage": job.actual_coverage,
    }
    return {
        "schema": SCHEMA,
        "id": f"trained-{job.id[:8]}-{variant_id}",
        "name": final_name or f"训练配置 {job.id[:8]}",
        "description": final_description or (
            f"基于 {job.processed_tokens:,} token 的完整角色对话 prefill 路由统计；"
            f"专家命中覆盖率 {job.actual_coverage * 100:.2f}%，"
            f"总驻留约 {configuration_resident:.2f} GiB。"
        ),
        "tags": ["trained", "token-prefill"],
        "experts": experts,
        "drop": {"enabled": True, "hint_tags": ["trained"]},
        "meta": meta,
    }


class TrainingEngine:
    def __init__(
        self,
        registry: ProfileRegistry,
        *,
        default_layers: int = DEFAULT_LAYERS,
        default_experts_per_layer: int = DEFAULT_EXPERTS_PER_LAYER,
        default_expert_size_mb: float = DEFAULT_EXPERT_SIZE_MB,
        engine_root: str | Path = "",
        cpu_python: str | Path = "",
    ):
        self.registry = registry
        self.default_layers = default_layers
        self.default_experts_per_layer = default_experts_per_layer
        self.default_expert_size_mb = default_expert_size_mb
        self.engine_root = Path(engine_root).expanduser() if engine_root else Path()
        self.cpu_python = Path(cpu_python).expanduser() if cpu_python else Path()
        self._jobs: dict[str, TrainingJob] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._load()

    def _file(self) -> Path:
        return TRAIN_DIR / "jobs.json"

    def _load(self) -> None:
        path = self._file()
        if not path.exists():
            return
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or document.get("schema") != JOBS_SCHEMA:
                return
            for raw in document.get("jobs") or []:
                job = TrainingJob(**raw)
                if job.status in {"pending", "running"}:
                    job.status = "failed"
                    job.message = "启动器重启，token 路由扫描已中断"
                    job.finished_at = time.time()
                self._jobs[job.id] = job
        except (json.JSONDecodeError, TypeError, OSError, ValueError):
            self._jobs = {}

    def _save(self) -> None:
        target = self._file()
        atomic_write_text(
            target,
            json.dumps(
                {"schema": JOBS_SCHEMA, "jobs": [asdict(job) for job in self._jobs.values()]},
                ensure_ascii=False,
            ),
        )

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                job.to_dict()
                for job in sorted(self._jobs.values(), key=lambda item: -item.created_at)
            ]

    def get(self, job_id: str) -> TrainingJob | None:
        return self._jobs.get(job_id)

    def submit(self, spec: dict[str, Any]) -> TrainingJob:
        allowed = {
            "corpus_files", "model_path", "token_budget", "mode", "sample_seed",
            "model_name", "model_version", "model_format", "model_manifest_sha256",
            "model_total_bytes", "model_total_gib", "model_top_k", "model_max_context",
            "layers", "expert_layers", "experts_per_layer", "expert_size_mb", "dense_without_shared_gib",
            "shared_expert_gib", "fixed_model_gib", "model_max_configuration_gib",
        }
        unexpected = sorted(set(spec) - allowed)
        if unexpected:
            raise ValueError("不支持的训练字段：" + ", ".join(unexpected))
        job = TrainingJob(
            id=uuid.uuid4().hex[:12],
            corpus_files=[str(value) for value in spec.get("corpus_files") or []],
            model_path=str(spec.get("model_path") or ""),
            token_budget=int(spec.get("token_budget") or DEFAULT_TOKEN_BUDGET),
            mode=str(spec.get("mode") or "auto"),
            sample_seed=int(spec.get("sample_seed") or 5090),
            model_name=str(spec.get("model_name") or ""),
            model_version=str(spec.get("model_version") or ""),
            model_format=str(spec.get("model_format") or ""),
            model_manifest_sha256=str(spec.get("model_manifest_sha256") or ""),
            model_total_bytes=int(spec.get("model_total_bytes") or 0),
            model_total_gib=float(spec.get("model_total_gib") or 0.0),
            model_top_k=int(spec.get("model_top_k") or 0),
            model_max_context=int(spec.get("model_max_context") or 0),
            layers=int(spec.get("layers") or self.default_layers),
            expert_layers=[int(value) for value in spec.get("expert_layers") or []],
            experts_per_layer=int(spec.get("experts_per_layer") or self.default_experts_per_layer),
            expert_size_mb=float(spec.get("expert_size_mb") or self.default_expert_size_mb),
            dense_without_shared_gib=float(spec.get("dense_without_shared_gib") or 0.0),
            shared_expert_gib=float(spec.get("shared_expert_gib") or 0.0),
            fixed_model_gib=float(spec.get("fixed_model_gib") or 0.0),
            model_max_configuration_gib=float(spec.get("model_max_configuration_gib") or 0.0),
        )
        if job.mode not in {"auto", "disk"}:
            raise ValueError("mode 必须是 auto 或 disk")
        if not job.corpus_files:
            raise ValueError("至少选择一个已上传的语料文件")
        if not MIN_TOKEN_BUDGET <= job.token_budget <= MAX_TOKEN_BUDGET:
            raise ValueError(
                f"token_budget 必须在 {MIN_TOKEN_BUDGET:,} 到 {MAX_TOKEN_BUDGET:,} 之间"
            )
        if not job.model_path:
            raise ValueError("请选择扫描对应的模型")
        if not 1 <= job.layers <= 512 or not 1 <= job.experts_per_layer <= 65536:
            raise ValueError("模型层数或每层专家数非法")
        if job.expert_layers and (
            len(job.expert_layers) != len(set(job.expert_layers))
            or any(layer < 0 or layer >= job.layers for layer in job.expert_layers)
        ):
            raise ValueError("模型专家层编号非法")
        if not 1 <= job.model_top_k <= job.experts_per_layer:
            raise ValueError("模型 top-k 非法")
        missing = [
            name for name in job.corpus_files
            if not (CORPUS_DIR / Path(name).name).is_file()
        ]
        if missing:
            raise ValueError("语料文件不存在: " + ", ".join(missing))
        with self._lock:
            if any(
                item.status in {"pending", "running"}
                for item in self._jobs.values()
            ):
                raise ValueError("已有 token 扫描正在运行，请等待完成或先停止该任务")
            self._jobs[job.id] = job
            self._cancel_events[job.id] = threading.Event()
            self._save()
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if job.status not in {"pending", "running"}:
                raise ValueError("只有正在运行的扫描可以停止")
            event = self._cancel_events.get(job_id)
            if event is None:
                event = threading.Event()
                self._cancel_events[job_id] = event
            event.set()
            job.message = "正在停止 token 路由扫描…"
            self._save()
        return True

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in {"pending", "running"}:
                raise ValueError("扫描运行中，不能删除任务")
            removed = self._jobs.pop(job_id, None)
            if removed:
                self._save()
        if removed:
            for suffix in (".routes.json", ".scan-report.json", ".scan.jsonl"):
                (TRAIN_DIR / f"{job_id}{suffix}").unlink(missing_ok=True)
        return removed is not None

    def replan(self, job_id: str, coverage: float) -> TrainingJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != "done":
                raise ValueError("路由扫描尚未完成")
            sizes = load_expert_sizes(job.model_path)
            keys, used, actual, per_layer = plan_route_coverage(
                job.counts,
                coverage,
                sizes or None,
                job.expert_size_mb,
                top_k=job.model_top_k,
                layers=job.layers,
                expert_layers=job.expert_layers,
            )
            job.coverage_target = round(coverage, 6)
            job.actual_coverage = actual
            job.layer_coverages = per_layer
            job.plan_keys = keys
            job.plan_bytes_mb = used
            job.plan_sizes_mb = {
                key: round((sizes or {}).get(key, job.expert_size_mb), 6) for key in keys
            }
            total = job.fixed_model_gib + used / 1024.0
            job.message = (
                f"已按 {coverage * 100:.1f}% 目标覆盖率选择 {len(keys):,} 个专家；"
                f"实测覆盖 {actual * 100:.2f}%，总驻留约 {total:.2f} GiB"
            )
            self._save()
            return job

    def mark_registered(self, job_id: str, profile_id: str, name: str, description: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.registered_profile_id = profile_id
                job.profile_name = name
                job.profile_description = description
                record = {
                    "id": profile_id,
                    "name": name,
                    "description": description,
                    "target_route_coverage": round(job.coverage_target, 6),
                    "actual_route_coverage": round(job.actual_coverage, 6),
                    "selected_experts": len(job.plan_keys),
                    "configuration_resident_gib": round(
                        job.fixed_model_gib + job.plan_bytes_mb / 1024.0, 3
                    ),
                }
                job.registered_profiles = [
                    existing for existing in job.registered_profiles
                    if existing.get("id") != profile_id
                ]
                job.registered_profiles.append(record)
                self._save()

    def _update_scan_progress(self, job: TrainingJob, processed: int, budget: int, stage: str) -> None:
        job.processed_tokens = processed
        progress_tokens = float(processed)
        layer_first_match = re.search(
            r"层\s+(\d+)/(\d+)\s+·\s+块\s+(\d+)/(\d+)\s+·\s+token\s+(\d+)",
            stage,
        )
        layer_match = re.search(
            r"块\s+\d+/\d+\s+·\s+层\s+(\d+)/(\d+)",
            stage,
        )
        if layer_first_match and processed < budget:
            layer = int(layer_first_match.group(1))
            layer_count = max(1, int(layer_first_match.group(2)))
            block = int(layer_first_match.group(3))
            block_count = max(1, int(layer_first_match.group(4)))
            document_tokens = min(
                int(layer_first_match.group(5)),
                budget - processed,
            )
            completed_fraction = (
                (layer - 1) + min(1.0, block / block_count)
            ) / layer_count
            progress_tokens += document_tokens * completed_fraction
        elif layer_match and processed < budget:
            layer = int(layer_match.group(1))
            layer_count = max(1, int(layer_match.group(2)))
            current_block = min(PREFILL_BLOCK_TOKENS, budget - processed)
            progress_tokens += current_block * min(1.0, layer / layer_count)
        job.progress = min(
            0.92,
            0.08 + 0.82 * progress_tokens / max(1, budget),
        )
        job.message = (
            f"完整对话 prefill：{processed:,} / {budget:,} token · "
            f"{PREFILL_BLOCK_TOKENS} token/块 · {stage}"
        )
        self._save()

    def _run(self, job: TrainingJob) -> None:
        scan_input = TRAIN_DIR / f"{job.id}.scan.jsonl"
        try:
            job.status = "running"
            job.progress = 0.02
            job.message = "整理完整角色对话与 system 上下文"
            self._save()
            paths = [CORPUS_DIR / Path(name).name for name in job.corpus_files]
            candidate_documents, _characters = prepare_scan_input(
                paths, scan_input, job.token_budget, job.sample_seed
            )
            job.total_documents = candidate_documents
            job.progress = 0.06
            job.message = (
                f"正在加载模型；随后以 {PREFILL_BLOCK_TOKENS} token/块扫描 "
                f"{job.token_budget:,} token"
            )
            self._save()
            job.counts, report = measured_activation(
                job,
                scan_input,
                engine_root=self.engine_root,
                python_path=self.cpu_python,
                progress_callback=lambda processed, budget, stage: self._update_scan_progress(
                    job, processed, budget, stage
                ),
                cancel_event=self._cancel_events.get(job.id),
            )
            job.processed_tokens = int(report["processed_tokens"])
            job.total_documents = int(report.get("documents") or candidate_documents)
            job.scan_context_tokens = int(report.get("max_context_tokens") or 0)
            job.truncated_documents = int(report.get("truncated_documents") or 0)
            job.tokenize_seconds = float(report.get("tokenize_seconds") or 0.0)
            job.model_load_seconds = float(report.get("model_load_seconds") or 0.0)
            job.prefill_seconds = float(report.get("prefill_seconds") or 0.0)
            job.prefill_tokens_per_second = float(
                report.get("prefill_tokens_per_second") or 0.0
            )
            job.route_observations = int(report.get("route_observations") or 0)
            job.cpu_operator_audit = dict(report.get("cpu_operator_audit") or {})
            sizes = load_expert_sizes(job.model_path)
            job.calibrated = bool(sizes)
            job.progress = 0.94
            job.message = "生成专家热力图并计算默认覆盖率"
            self._save()
            keys, used, actual, per_layer = plan_route_coverage(
                job.counts,
                job.coverage_target,
                sizes or None,
                job.expert_size_mb,
                top_k=job.model_top_k,
                layers=job.layers,
                expert_layers=job.expert_layers,
            )
            job.plan_keys = keys
            job.plan_bytes_mb = used
            job.actual_coverage = actual
            job.layer_coverages = per_layer
            job.plan_sizes_mb = {
                key: round((sizes or {}).get(key, job.expert_size_mb), 6) for key in keys
            }
            job.status = "done"
            job.progress = 1.0
            job.finished_at = time.time()
            total = job.fixed_model_gib + used / 1024.0
            job.message = (
                f"扫描完成：{job.processed_tokens:,} token，命中 {len(job.counts):,} 个专家；"
                f"prefill {job.prefill_tokens_per_second:.3f} token/s；"
                f"默认覆盖 {actual * 100:.2f}%，总驻留约 {total:.2f} GiB"
            )
            self._save()
        except TrainingCancelled as exc:
            job.status = "cancelled"
            job.message = str(exc)
            job.finished_at = time.time()
            self._save()
        except Exception as exc:  # noqa: BLE001 - 后台任务必须把完整原因回显给 UI
            job.status = "failed"
            job.message = str(exc)
            job.finished_at = time.time()
            self._save()
        finally:
            scan_input.unlink(missing_ok=True)
            with self._lock:
                self._cancel_events.pop(job.id, None)

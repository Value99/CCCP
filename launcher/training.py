"""训练选项卡引擎:语料全量推理 → 激活专家偏好统计 → 目标体积规划。

设计(见 docs/INTERFACE.md):
- 语料经 CPU / 硬盘(disk) 模式对模型做全量推理,采集每层路由命中计数。
- 统计来源双轨:
  A) tpq-router-stats —— TPQ 侧导出路由计数(INTERFACE I-3,待 TPQ 开发);
     本引擎探测该接口,可用时 data_source=tpq-router-stats 且 calibrated=true。
  B) heuristic-fallback —— TPQ 接口落地前的保底:按语料领域关键词与
     profile 专家 tags 匹配 + recipe 规模先验生成估算计数;产物显著标注
     calibrated=false。
- 目标体积规划:按 score/bytes 贪心装填到预算 bytes(不超额)。
- 产物导出:tpq-expert-residency-scores-v1、TPQ counts(profile.json)、
  以及可直接注册的新 profile(source=trained)。

不修改 TPQ-Final 的任何文件;全量推理经由其 OpenAI API / 子进程进行。
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

from .profiles import ProfileRegistry, SCHEMA
from .settings import DATA_DIR

TRAIN_DIR = DATA_DIR / "training"
CORPUS_DIR = DATA_DIR / "corpus"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LAYERS = 60
DEFAULT_EXPERTS_PER_LAYER = 256


# --------------------------------------------------------------------------
# 任务模型
# --------------------------------------------------------------------------

@dataclass
class TrainingJob:
    id: str
    corpus_files: list[str]
    mode: str = "cpu"            # cpu | disk
    target_gb: float = 0.0       # 目标体积(GiB);0 = 不限制
    sample_limit: int = 2000     # 语料采样条数上限
    layers: int = DEFAULT_LAYERS
    experts_per_layer: int = DEFAULT_EXPERTS_PER_LAYER
    related_profiles: list[str] = field(default_factory=list)

    status: str = "pending"      # pending|running|done|failed
    data_source: str = "heuristic-fallback"
    calibrated: bool = False
    progress: float = 0.0
    message: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    # 结果
    total_samples: int = 0
    counts: dict[str, int] = field(default_factory=dict)   # "layer:expert" -> hits
    plan_keys: list[str] = field(default_factory=list)     # 目标体积规划选中的 key
    plan_bytes_mb: float = 0.0

    def to_dict(self, with_counts: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not with_counts and len(self.counts) > 200:  # 详情接口才返回全量计数
            d["counts"] = {}
            d["counts_truncated"] = True
        return d


# --------------------------------------------------------------------------
# 语料解析
# --------------------------------------------------------------------------

def iter_corpus(paths: Iterable[Path]) -> Iterable[str]:
    """从 jsonl / txt 语料取文本样本。

    jsonl 认 {"prompt"|"text"|"content"} 字段或 {"messages":[{content}]};
    txt 每行一个样本(跳过空行与 # 注释)。
    """
    for p in paths:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if p.suffix.lower() == ".jsonl" or s.startswith("{"):
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    yield s
                    continue
                text = (
                    obj.get("prompt") or obj.get("text") or obj.get("content")
                    or "\n".join(
                        m.get("content", "") for m in obj.get("messages", [])
                        if isinstance(m, dict)
                    )
                )
                if text:
                    yield str(text)
            else:
                yield s


def save_corpus_file(name: str, content: bytes) -> str:
    """保存上传的语料到 data/corpus/,返回相对文件名(防路径穿越)。"""
    safe = Path(name).name.replace("\\", "_").replace("/", "_")
    if not safe:
        raise ValueError("文件名不能为空")
    (CORPUS_DIR / safe).write_bytes(content)
    return safe


def list_corpus() -> list[dict]:
    out = []
    for f in sorted(CORPUS_DIR.glob("*")):
        if f.is_file():
            out.append({"name": f.name, "bytes": f.stat().st_size})
    return out


def delete_corpus(name: str) -> bool:
    f = CORPUS_DIR / Path(name).name
    if f.is_file():
        f.unlink()
        return True
    return False


# --------------------------------------------------------------------------
# 激活统计(双轨)
# --------------------------------------------------------------------------

def tpq_router_stats_available() -> bool:
    """探测 TPQ 路由计数接口(INTERFACE I-3)是否可用。

    约定:TPQ 侧就绪后会提供 GET /v1/expert-stats 或约定的 router-stats.jsonl;
    当前 TPQ-Final v1.2.0 无此输出 -> 返回 False,走保底。
    """
    # TODO(TPQ-DEV I-3): 接入 GET /v1/expert-stats / TPQ_ROUTER_STATS_JSONL。
    return False


def heuristic_activation(
    samples: list[str],
    registry: ProfileRegistry,
    related: list[str],
    layers: int,
    experts_per_layer: int,
) -> dict[str, int]:
    """保底估算:命中数 ∝ 语料关键词与 profile 专家 tags 匹配 × recipe 规模先验。

    确定性:计数由样本内容与专家 key 的哈希驱动,同一语料结果可复现。
    """
    profs = [registry.get(pid) for pid in related] if related else registry.list()
    profs = [p for p in profs if p]
    tag_weights: dict[str, float] = {}
    for s in samples[:500]:
        low = s.lower()
        for p in profs:
            hit = sum(1 for t in p.tags if t.lower() in low)
            if hit:
                tag_weights[p.id] = tag_weights.get(p.id, 0.0) + hit

    counts: dict[str, int] = {}
    for p in profs:
        p.materialize()
        w = 1.0 + tag_weights.get(p.id, 0.0)
        for e in p.experts:
            h = int(hashlib.sha256(f"act:{e.key}".encode()).hexdigest()[:8], 16)
            base = 1 + (h % 7)
            counts[e.key] = counts.get(e.key, 0) + int(base * w)
    return counts


# --------------------------------------------------------------------------
# 目标体积规划
# --------------------------------------------------------------------------

def plan_target_size(
    counts: dict[str, int],
    target_gb: float,
    sizes_mb: dict[str, float] | None = None,
) -> tuple[list[str], float]:
    """按命中计数装填到预算:score/bytes 贪心;返回 (keys, 实际 MB)。

    sizes_mb 缺省时按 24 MiB 估算默认(校准前),估算属性由上层标注。
    """
    if target_gb <= 0:
        return sorted(counts, key=lambda k: (-counts[k], k)), round(
            sum((sizes_mb or {}).get(k, 24.0) for k in counts), 1
        )
    budget_mb = target_gb * 1024.0
    size_of = lambda k: (sizes_mb or {}).get(k, 24.0)
    ranked = sorted(counts, key=lambda k: (-counts[k] / size_of(k), k))
    picked: list[str] = []
    used = 0.0
    for k in ranked:
        s = size_of(k)
        if used + s > budget_mb:
            continue
        picked.append(k)
        used += s
    return picked, round(used, 1)


# --------------------------------------------------------------------------
# 产物导出
# --------------------------------------------------------------------------

def export_scores(job: TrainingJob) -> dict:
    """tpq-expert-residency-scores-v1:选中专家>0(归一化命中),其余补 0(全覆盖)。"""
    keys = job.plan_keys or list(job.counts)
    sel = set(keys)
    max_hit = max(job.counts.values(), default=1)
    scores: dict[str, float] = {}
    for layer in range(job.layers):
        for eid in range(job.experts_per_layer):
            k = f"{layer}:{eid}"
            scores[k] = round(job.counts.get(k, 0) / max_hit, 6) if k in sel else 0.0
    return {
        "schema": "tpq-expert-residency-scores-v1",
        "scores": scores,
        "meta": {
            "generator": "tpq-winui-launcher/training",
            "job": job.id,
            "data_source": job.data_source,
            "calibrated": job.calibrated,
            "target_gb": job.target_gb,
            "selected": len(sel),
        },
    }


def export_counts(job: TrainingJob) -> dict:
    """TPQ counts(profile.json)schema:{counts:{layer:{expert:count}}}。"""
    counts: dict[str, dict[str, int]] = {}
    for k, v in job.counts.items():
        layer, eid = k.split(":", 1)
        counts.setdefault(layer, {})[eid] = v
    return {"counts": counts, "meta": {"job": job.id, "data_source": job.data_source}}


def export_profile(job: TrainingJob, name: str = "") -> dict:
    """导出一个可注册的 trained profile(显式 experts 清单)。"""
    keys = job.plan_keys or sorted(job.counts, key=lambda k: (-job.counts[k], k))
    experts = [
        {"key": k, "size_mb": 24.0, "tags": ["trained", job.mode]} for k in keys
    ]
    return {
        "schema": SCHEMA,
        "id": f"trained-{job.id[:8]}",
        "name": name or f"训练产物 {job.id[:8]}({job.mode})",
        "description": (
            f"训练任务 {job.id} 产出;data_source={job.data_source};"
            f"语料 {len(job.corpus_files)} 个文件,样本 {job.total_samples};"
            f"目标 {job.target_gb} GiB -> 实际 {job.plan_bytes_mb / 1024:.1f} GiB。"
        ),
        "tags": ["trained"],
        "experts": experts,
        "drop": {"enabled": True, "hint_tags": ["trained"]},
        "meta": {"source": "trained", "calibrated": job.calibrated},
    }


# --------------------------------------------------------------------------
# 任务引擎
# --------------------------------------------------------------------------

class TrainingEngine:
    """线程池跑任务;状态落盘 data/training/jobs.json。"""

    def __init__(self, registry: ProfileRegistry):
        self.registry = registry
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()
        self._load()

    # -- 持久化 --
    def _file(self) -> Path:
        return TRAIN_DIR / "jobs.json"

    def _load(self) -> None:
        f = self._file()
        if f.exists():
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
                for jd in raw:
                    job = TrainingJob(**jd)
                    if job.status == "running":
                        job.status = "failed"
                        job.message = "WINUI-EXE 重启,任务中断"
                    self._jobs[job.id] = job
            except (json.JSONDecodeError, TypeError, OSError):
                pass

    def _save(self) -> None:
        raw = [asdict(j) for j in self._jobs.values()]
        self._file().write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    # -- API --
    def list(self) -> list[dict]:
        with self._lock:
            return [
                j.to_dict()
                for j in sorted(self._jobs.values(), key=lambda x: -x.created_at)
            ]

    def get(self, jid: str) -> TrainingJob | None:
        return self._jobs.get(jid)

    def submit(self, spec: dict[str, Any]) -> TrainingJob:
        job = TrainingJob(
            id=uuid.uuid4().hex[:12],
            corpus_files=[str(x) for x in spec.get("corpus_files") or []],
            mode=str(spec.get("mode") or "cpu"),
            target_gb=float(spec.get("target_gb") or 0.0),
            sample_limit=int(spec.get("sample_limit") or 2000),
            layers=int(spec.get("layers") or DEFAULT_LAYERS),
            experts_per_layer=int(
                spec.get("experts_per_layer") or DEFAULT_EXPERTS_PER_LAYER
            ),
            related_profiles=[str(x) for x in spec.get("related_profiles") or []],
        )
        if job.mode not in ("cpu", "disk"):
            raise ValueError("mode 必须是 cpu 或 disk")
        if not job.corpus_files:
            raise ValueError("至少需要一个语料文件(先上传到语料库)")
        with self._lock:
            self._jobs[job.id] = job
            self._save()
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def delete(self, jid: str) -> bool:
        with self._lock:
            job = self._jobs.pop(jid, None)
            if job:
                self._save()
            return job is not None

    # -- 执行 --
    def _run(self, job: TrainingJob) -> None:
        try:
            job.status = "running"
            job.message = "解析语料"
            self._save()

            paths = [CORPUS_DIR / f for f in job.corpus_files]
            samples: list[str] = []
            for i, s in enumerate(iter_corpus(paths)):
                samples.append(s)
                if i + 1 >= job.sample_limit:
                    break
            job.total_samples = len(samples)
            if not samples:
                raise ValueError("语料为空(检查文件名/格式)")

            job.progress = 0.3
            if tpq_router_stats_available():
                job.data_source = "tpq-router-stats"
                job.calibrated = True
                # TODO(TPQ-DEV I-3): 接入真实路由计数输出后移除此分支异常
                raise RuntimeError("tpq-router-stats 探测为可用但接入未实现")
            else:
                job.message = f"估算扫描(fallback){len(samples)} 条样本"
                job.counts = heuristic_activation(
                    samples, self.registry, job.related_profiles,
                    job.layers, job.experts_per_layer,
                )

            job.progress = 0.75
            job.message = "目标体积规划"
            sizes: dict[str, float] | None = None
            if job.related_profiles:
                sizes = {}
                for pid in job.related_profiles:
                    p = self.registry.get(pid)
                    if p:
                        p.materialize()
                        for e in p.experts:
                            sizes.setdefault(e.key, e.size_mb)
            job.plan_keys, job.plan_bytes_mb = plan_target_size(
                job.counts, job.target_gb, sizes
            )

            job.progress = 1.0
            job.status = "done"
            job.message = (
                f"完成:{len(job.counts)} 个激活专家;"
                f"规划 {len(job.plan_keys)} 个 / {job.plan_bytes_mb / 1024:.1f} GiB"
                f"({job.data_source})"
            )
            job.finished_at = time.time()
            self._save()
        except Exception as exc:  # noqa: BLE001 —— 任务失败要回显原因
            job.status = "failed"
            job.message = str(exc)
            job.finished_at = time.time()
            self._save()

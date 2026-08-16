"""领域配置文件(Profile)模型与组合计算。

Profile 描述"为这个领域/任务载入哪些专家":
- 显式列举专家实例(key = "层号:专家号",携带打包后体积)
- 或通过确定性 recipe 抽样生成(仅供旧数据/测试使用,可校准)
- 多 Profile 组合时按 key 求并集:重叠专家只计一次体积
  (例:合同 100G + 代码 200G -> 并集约 250G)
- 每个 Profile 内置一个标记为 drop 的占位专家:
  不携带权重,启动时自动"路由"到当前组合中最相关的已加载专家
  (对应 CCCP-Engine route 前的 drop-expert masking 语义)。
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .io_utils import atomic_write_text

SCHEMA = "winui-expert-profile-v1"
_KEY_RE = re.compile(r"^\d+:\d+$")
_ID_RE = re.compile(r"[a-z0-9][a-z0-9\-_.]{1,63}")


@dataclass(frozen=True)
class ExpertRef:
    """一个被引用的路由专家实例。"""

    key: str  # "layer:expert_id",与 CCCP score-file 的 "layer:expert" 一致
    size_mb: float  # 打包 (VQ) 形态占用,MiB
    tags: tuple[str, ...] = ()
    route_count: int = 1  # 语料实测命中次数；未校准配置为 1
    route_score: float = 0.0

    @staticmethod
    def valid_key(key: str) -> bool:
        return bool(_KEY_RE.match(key))

    @property
    def layer(self) -> int:
        return int(self.key.split(":", 1)[0])

    @property
    def expert_id(self) -> int:
        return int(self.key.split(":", 1)[1])


@dataclass
class DropExpert:
    """drop 占位专家:无权重,自动路由到组合内最相关的已加载专家。

    hint_tags 用于相关性匹配;resolved 由 resolve_drop() 填充,
    作为 launch plan 的一部分输出(不改变 CCCP,仅生成 placement 提示)。
    """

    enabled: bool = True
    hint_tags: list[str] = field(default_factory=list)
    resolved: str | None = None  # 解析出的专家 key


@dataclass
class Recipe:
    """确定性抽样配方:用于轻量测试 Profile,避免巨型清单文件。

    训练选项卡产出真实数据后,可导出为显式 experts 覆盖 recipe。
    """

    seed: str
    layers: int
    experts_per_layer: int
    density: float  # 每层抽样比例 0..1
    layer_affinity: str = "uniform"  # uniform|deep|shallow
    mean_size_mb: float = 24.0
    size_jitter: float = 0.35


@dataclass
class Profile:
    id: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    experts: list[ExpertRef] = field(default_factory=list)
    recipe: Recipe | None = None
    drop: DropExpert = field(default_factory=DropExpert)
    source: str = "imported"  # model|imported
    calibrated: bool = False  # True = 体积来自 CCCP 精确字节表（INTERFACE I-1）
    meta: dict[str, Any] = field(default_factory=dict)
    _materialized: bool = field(default=False, repr=False)

    # ---- 物化(从 recipe 生成确定性专家集) ----
    def materialize(self) -> "Profile":
        if self._materialized or self.recipe is None:
            self._materialized = True
            return self
        r = self.recipe
        rnd = random.Random(_seed_int(r.seed, self.id))
        out: list[ExpertRef] = []
        for layer in range(r.layers):
            if r.layer_affinity == "deep":
                w = 0.4 + 1.4 * (layer / max(1, r.layers - 1))
            elif r.layer_affinity == "shallow":
                w = 1.8 - 1.4 * (layer / max(1, r.layers - 1))
            else:
                w = 1.0
            for eid in range(r.experts_per_layer):
                if rnd.random() < r.density * w:
                    jitter = 1.0 + rnd.uniform(-r.size_jitter, r.size_jitter)
                    out.append(
                        ExpertRef(
                            key=f"{layer}:{eid}",
                            size_mb=round(r.mean_size_mb * jitter, 3),
                            tags=tuple(self.tags),
                        )
                    )
        self.experts = out
        self._materialized = True
        return self

    # ---- 统计 ----
    @property
    def expert_count(self) -> int:
        self.materialize()
        return len(self.experts)

    @property
    def memory_mb(self) -> float:
        self.materialize()
        return round(sum(e.size_mb for e in self.experts), 1)

    @property
    def memory_gb(self) -> float:
        return round(self.memory_mb / 1024.0, 2)

    def to_dict(self, with_experts: bool = False) -> dict[str, Any]:
        self.materialize()
        d: dict[str, Any] = {
            "schema": SCHEMA,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "source": self.source,
            "calibrated": self.calibrated,
            "meta": self.meta,
            "expert_count": self.expert_count,
            "memory_mb": self.memory_mb,
            "memory_gb": self.memory_gb,
            "drop": {
                "enabled": self.drop.enabled,
                "hint_tags": self.drop.hint_tags,
                "resolved": self.drop.resolved,
            },
        }
        if self.recipe and not self.experts_explicit():
            d["recipe"] = {
                "seed": self.recipe.seed,
                "layers": self.recipe.layers,
                "experts_per_layer": self.recipe.experts_per_layer,
                "density": self.recipe.density,
                "layer_affinity": self.recipe.layer_affinity,
                "mean_size_mb": self.recipe.mean_size_mb,
            }
        if with_experts:
            d["experts"] = [
                {"key": e.key, "size_mb": e.size_mb, "tags": list(e.tags),
                 "route_count": e.route_count, "route_score": e.route_score}
                for e in self.experts
            ]
        return d

    def experts_explicit(self) -> bool:
        """该 profile 是否携带显式 experts 清单(而非 recipe 生成)。"""
        return self.recipe is None and bool(self.experts)


# --------------------------------------------------------------------------
# 组合计算:重叠并集
# --------------------------------------------------------------------------

@dataclass
class Combination:
    profile_ids: list[str]
    union: dict[str, ExpertRef]
    overlap_mb: float
    drop_resolution: dict[str, str] = field(default_factory=dict)
    model_manifest_sha256: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    model_total_bytes: int = 0
    fixed_model_gib: float = 0.0
    dense_without_shared_gib: float = 0.0
    shared_expert_gib: float = 0.0

    @property
    def expert_count(self) -> int:
        return len(self.union)

    @property
    def memory_mb(self) -> float:
        return round(sum(e.size_mb for e in self.union.values()), 1)

    @property
    def memory_gb(self) -> float:
        return round(self.memory_mb / 1024.0, 2)

    @property
    def configuration_resident_gib(self) -> float:
        """模型固定权重只计一次，再加去重后的动态专家。"""
        return round(self.fixed_model_gib + self.memory_mb / 1024.0, 3)

    @property
    def fixed_deduplicated_gib(self) -> float:
        """多配置合并时，被复用而不重复加载的 Dense+共享专家体积。"""
        return round(max(0, len(self.profile_ids) - 1) * self.fixed_model_gib, 3)

    @property
    def total_deduplicated_gib(self) -> float:
        """固定权重复用与重复动态专家共同节省的总配置体积。"""
        return round(self.fixed_deduplicated_gib + self.overlap_mb / 1024.0, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profiles": self.profile_ids,
            "expert_count": self.expert_count,
            "memory_mb": self.memory_mb,
            "memory_gb": self.memory_gb,
            "overlap_mb": round(self.overlap_mb, 1),
            "overlap_gb": round(self.overlap_mb / 1024.0, 2),
            "drop_resolution": self.drop_resolution,
            "model_manifest_sha256": self.model_manifest_sha256,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_total_bytes": self.model_total_bytes,
            "model_total_gib": round(self.model_total_bytes / 2**30, 6),
            "fixed_model_gib": round(self.fixed_model_gib, 3),
            "dense_without_shared_gib": round(self.dense_without_shared_gib, 3),
            "shared_expert_gib": round(self.shared_expert_gib, 3),
            "configuration_resident_gib": self.configuration_resident_gib,
            "fixed_deduplicated_gib": self.fixed_deduplicated_gib,
            "total_deduplicated_gib": self.total_deduplicated_gib,
        }


def combine(profiles: Iterable[Profile]) -> Combination:
    """同一模型的多 Profile 按专家编号求并集，固定权重和重复专家只计一次。"""
    profiles = list(profiles)
    fingerprints = {
        str(p.meta.get("model_manifest_sha256"))
        for p in profiles if p.meta.get("model_manifest_sha256")
    }
    if len(fingerprints) > 1:
        raise ProfileError("所选配置来自不同模型版本，不能合并")
    if fingerprints and any(not p.meta.get("model_manifest_sha256") for p in profiles):
        raise ProfileError("不能把缺少模型指纹的旧配置与模型专用配置合并")
    if fingerprints and profiles:
        # 指纹是兼容性的主键；这些字段用于发现被手工改坏的分享配置。
        identity_fields = (
            "model_name", "model_version", "model_total_bytes", "model_format",
            "fixed_model_gib", "dense_without_shared_gib", "shared_expert_gib",
        )
        expected = profiles[0].meta
        for profile in profiles[1:]:
            for field_name in identity_fields:
                if profile.meta.get(field_name) != expected.get(field_name):
                    raise ProfileError(
                        f"同一模型指纹的 meta.{field_name} 不一致，配置可能已损坏"
                    )
    union: dict[str, ExpertRef] = {}
    total_mb = 0.0
    ids: list[str] = []
    for p in profiles:
        p.materialize()
        ids.append(p.id)
        for e in p.experts:
            total_mb += e.size_mb
            existing = union.get(e.key)
            if existing is None:
                union[e.key] = e
            elif abs(existing.size_mb - e.size_mb) > 0.01:
                raise ProfileError(
                    f"同一专家 {e.key} 在配置中的体积不一致，配置可能已损坏"
                )
            else:
                # 多方向配置共享同一专家时权重只驻留一次，但路由热度应聚合，
                # 供运行时决定哪些已选专家优先采用高速算子布局。
                union[e.key] = ExpertRef(
                    key=e.key,
                    size_mb=existing.size_mb,
                    tags=tuple(sorted(set(existing.tags) | set(e.tags))),
                    route_count=existing.route_count + e.route_count,
                    route_score=max(existing.route_score, e.route_score),
                )
    union_mb = sum(e.size_mb for e in union.values())
    first = profiles[0] if profiles else None
    return Combination(
        profile_ids=ids,
        union=union,
        overlap_mb=round(total_mb - union_mb, 1),
        model_manifest_sha256=(next(iter(fingerprints)) if fingerprints else None),
        model_name=(str(first.meta.get("model_name")) if first and first.meta.get("model_name") else None),
        model_version=(str(first.meta.get("model_version")) if first and first.meta.get("model_version") else None),
        model_total_bytes=(int(first.meta.get("model_total_bytes") or 0) if first else 0),
        fixed_model_gib=(float(first.meta.get("fixed_model_gib") or 0.0) if first else 0.0),
        dense_without_shared_gib=(
            float(first.meta.get("dense_without_shared_gib") or 0.0) if first else 0.0
        ),
        shared_expert_gib=(
            float(first.meta.get("shared_expert_gib") or 0.0) if first else 0.0
        ),
    )


# --------------------------------------------------------------------------
# drop 占位专家路由
# --------------------------------------------------------------------------

def resolve_drop(profile: Profile, union: dict[str, ExpertRef]) -> str | None:
    """把 drop 占位专家路由到组合中最相关的已加载专家。

    相关性 = hint_tags/profile tags 与候选专家 tags 的交集数;
    平票时取 key 字典序最小者(确定性)。无 hint 时按 profile tags。
    """
    best: tuple[int, str] | None = None
    hints = set(profile.drop.hint_tags or profile.tags)
    for key, e in union.items():
        score = len(hints & set(e.tags)) if hints else 0
        if best is None or score > best[0] or (score == best[0] and key < best[1]):
            best = (score, key)
    profile.drop.resolved = best[1] if best else None
    return profile.drop.resolved


# --------------------------------------------------------------------------
# 加载 / 校验
# --------------------------------------------------------------------------

class ProfileError(ValueError):
    pass


def _seed_int(seed: str, salt: str) -> int:
    h = hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()
    return int(h[:16], 16)


def load_profile_dict(data: dict[str, Any], *, source: str = "memory") -> Profile:
    if data.get("schema") != SCHEMA:
        raise ProfileError(f"schema 必须是 {SCHEMA}")
    pid = str(data.get("id") or "").strip()
    if not pid:
        raise ProfileError("缺少 id")
    if not _ID_RE.fullmatch(pid):
        raise ProfileError(f"非法 id: {pid!r}")

    experts: list[ExpertRef] = []
    seen_keys: set[str] = set()
    for raw in data.get("experts") or []:
        if not isinstance(raw, dict):
            raise ProfileError("experts 的每一项都必须是对象")
        key = str(raw.get("key", ""))
        if not ExpertRef.valid_key(key):
            raise ProfileError(f"非法专家 key: {key!r}(应为 layer:expert)")
        size = float(raw.get("size_mb", 0))
        if size <= 0:
            raise ProfileError(f"专家 {key} 的 size_mb 必须 > 0")
        if size > 1024 * 1024:
            raise ProfileError(f"专家 {key} 的 size_mb 异常过大")
        if key in seen_keys:
            raise ProfileError(f"专家 key 重复: {key}")
        route_count = int(raw.get("route_count", raw.get("count", 1)))
        route_score = float(raw.get("route_score", raw.get("score", 0.0)))
        if route_count < 0 or not 0.0 <= route_score <= 1.0:
            raise ProfileError(f"专家 {key} 的路由统计非法")
        seen_keys.add(key)
        experts.append(
            ExpertRef(
                key=key,
                size_mb=size,
                tags=tuple(map(str, raw.get("tags") or ())),
                route_count=route_count,
                route_score=route_score,
            )
        )

    recipe = None
    if data.get("recipe"):
        rd = data["recipe"]
        if not isinstance(rd, dict):
            raise ProfileError("recipe 必须是对象")
        recipe = Recipe(
            seed=str(rd.get("seed", pid)),
            layers=int(rd["layers"]),
            experts_per_layer=int(rd["experts_per_layer"]),
            density=float(rd["density"]),
            layer_affinity=str(rd.get("layer_affinity", "uniform")),
            mean_size_mb=float(rd.get("mean_size_mb", 24.0)),
            size_jitter=float(rd.get("size_jitter", 0.35)),
        )
        if experts:
            raise ProfileError("experts 与 recipe 只能提供一种")
        if not 1 <= recipe.layers <= 512:
            raise ProfileError("recipe.layers 必须在 1 到 512 之间")
        if not 1 <= recipe.experts_per_layer <= 65536:
            raise ProfileError("recipe.experts_per_layer 必须在 1 到 65536 之间")
        if recipe.layers * recipe.experts_per_layer > 1_000_000:
            raise ProfileError("recipe 规模过大")
        if not 0 < recipe.density <= 1:
            raise ProfileError("recipe.density 必须在 0 到 1 之间")
        if recipe.layer_affinity not in {"uniform", "deep", "shallow"}:
            raise ProfileError("recipe.layer_affinity 必须是 uniform|deep|shallow")
        if not 0 < recipe.mean_size_mb <= 1024 * 1024:
            raise ProfileError("recipe.mean_size_mb 必须为合理正数")
        if not 0 <= recipe.size_jitter <= 1:
            raise ProfileError("recipe.size_jitter 必须在 0 到 1 之间")
    if not experts and recipe is None:
        raise ProfileError("profile 必须提供 experts 清单或 recipe")

    dd = data.get("drop") or data.get("drop_expert") or {}
    meta = data.get("meta") or {}
    if not isinstance(meta, dict):
        raise ProfileError("meta 必须是对象")
    fingerprint = meta.get("model_manifest_sha256")
    if fingerprint is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", str(fingerprint)):
        raise ProfileError("meta.model_manifest_sha256 必须是 64 位 SHA-256")
    if source in {"model", "imported", "trained"} and fingerprint is None:
        raise ProfileError(
            "模型配置必须记录 meta.model_name、model_version、model_format 和 model_manifest_sha256"
        )
    for field_name in (
        "fixed_model_gib", "dense_without_shared_gib", "shared_expert_gib",
        "routed_expert_budget_gib", "configuration_resident_gib",
        "configuration_budget_gib",
    ):
        if field_name in meta and float(meta[field_name]) < 0:
            raise ProfileError(f"meta.{field_name} 不能为负数")
    if fingerprint is not None:
        required_model_fields = (
            "model_name", "model_version", "model_format", "model_total_bytes",
            "model_total_gib", "model_layers", "model_experts_per_layer", "model_top_k",
            "fixed_model_gib", "dense_without_shared_gib", "shared_expert_gib",
            "configuration_budget_gib", "configuration_resident_gib",
        )
        missing = [name for name in required_model_fields if meta.get(name) is None]
        if missing:
            raise ProfileError("模型专用配置缺少 meta 字段: " + ", ".join(missing))
        if not str(meta["model_name"]).strip() or not str(meta["model_version"]).strip():
            raise ProfileError("meta.model_name/model_version 不能为空")
        if int(meta["model_total_bytes"]) <= 0:
            raise ProfileError("meta.model_total_bytes 必须 > 0")
        if float(meta["model_total_gib"]) <= 0:
            raise ProfileError("meta.model_total_gib 必须 > 0")
        if int(meta["model_layers"]) <= 0 or int(meta["model_experts_per_layer"]) <= 0:
            raise ProfileError("meta.model_layers/model_experts_per_layer 必须 > 0")
        if not 1 <= int(meta["model_top_k"]) <= int(meta["model_experts_per_layer"]):
            raise ProfileError("meta.model_top_k 超出模型专家范围")
    p = Profile(
        id=pid,
        name=str(data.get("name") or pid),
        description=str(data.get("description") or ""),
        tags=[str(t)[:64] for t in (data.get("tags") or [])][:64],
        experts=experts,
        recipe=recipe,
        drop=DropExpert(
            enabled=bool(dd.get("enabled", True)),
            hint_tags=[str(t) for t in (dd.get("hint_tags") or [])],
        ),
        source=source,
        calibrated=bool(data.get("calibrated", meta.get("calibrated", False))),
        meta={str(k): v for k, v in meta.items()},
    )
    p.materialize()
    if fingerprint is not None:
        fixed = float(meta["fixed_model_gib"])
        dense = float(meta["dense_without_shared_gib"])
        shared = float(meta["shared_expert_gib"])
        configured = float(meta["configuration_resident_gib"])
        budget = float(meta["configuration_budget_gib"])
        calculated = fixed + p.memory_mb / 1024.0
        if abs(fixed - (dense + shared)) > 0.002:
            raise ProfileError("固定模型体积不等于 Dense 与共享专家之和")
        if abs(configured - calculated) > 0.002:
            raise ProfileError("配置总驻留体积与专家清单计算结果不一致")
        if configured > budget + 0.002:
            raise ProfileError("配置总驻留体积超过配置预算")
        if abs(float(meta["model_total_gib"]) - int(meta["model_total_bytes"]) / 2**30) > 0.002:
            raise ProfileError("meta.model_total_bytes 与 model_total_gib 不一致")
        layers = int(meta["model_layers"])
        experts_per_layer = int(meta["model_experts_per_layer"])
        raw_expert_layers = meta.get("model_expert_layers")
        if raw_expert_layers is None:
            # 0.9.0 配置默认每层都是专家层；0.9.1 起训练配置会精确记录
            # 清单声明的专家层，支持前置 Dense 层及其他稀疏层布局。
            expert_layers = list(range(layers))
        elif not isinstance(raw_expert_layers, list) or not raw_expert_layers:
            raise ProfileError("meta.model_expert_layers 必须是非空层编号列表")
        else:
            try:
                expert_layers = [int(layer) for layer in raw_expert_layers]
            except (TypeError, ValueError) as exc:
                raise ProfileError("meta.model_expert_layers 包含非法层编号") from exc
            if (
                len(expert_layers) != len(set(expert_layers))
                or any(layer < 0 or layer >= layers for layer in expert_layers)
            ):
                raise ProfileError("meta.model_expert_layers 包含重复或越界层编号")
        expert_layer_set = set(expert_layers)
        per_layer: dict[int, int] = {}
        for expert in p.experts:
            if expert.layer >= layers or expert.expert_id >= experts_per_layer:
                raise ProfileError(f"专家 {expert.key} 超出声明的模型范围")
            if expert.layer not in expert_layer_set:
                raise ProfileError(f"专家 {expert.key} 位于非专家层")
            per_layer[expert.layer] = per_layer.get(expert.layer, 0) + 1
            if meta.get("strict_route") and expert.route_count <= 0:
                raise ProfileError(f"严格路由配置的专家 {expert.key} 缺少有效命中计数")
        if meta.get("strict_route"):
            top_k = int(meta["model_top_k"])
            invalid = [
                layer for layer in expert_layers
                if per_layer.get(layer, 0) < top_k
            ]
            if invalid:
                raise ProfileError(f"严格路由配置以下层少于模型 top-k={top_k}: {invalid}")
        selected_experts = meta.get("selected_experts")
        if selected_experts is not None and int(selected_experts) != p.expert_count:
            raise ProfileError("meta.selected_experts 与专家清单数量不一致")
    return p


def load_profile_file(path: Path, *, source: str = "memory") -> Profile:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    except (OSError, UnicodeError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ProfileError(f"{path.name}: 解析失败: {exc}") from exc
    if not isinstance(data, dict):
        raise ProfileError(f"{path.name}: 顶层必须是对象")
    return load_profile_dict(data, source=source)


# --------------------------------------------------------------------------
# 注册表
# --------------------------------------------------------------------------

class ProfileRegistry:
    """模型目录 ``profiles/`` + 启动器 ``profiles/user/`` 配置注册表。

    模型预设始终随模型移动；发行包本身不提供模型专用配置。用户导入的配置
    可单独保存在启动器目录，但仍必须携带完整模型身份并匹配模型库。
    """

    def __init__(self, model_roots: Iterable[str | Path], user_dir: Path):
        self.model_roots = [Path(root).expanduser() for root in model_roots]
        self.user_dir = user_dir
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, Profile] = {}
        self._model_profile_paths: dict[str, Path] = {}
        self.reload()

    # -- 加载 --
    def _load_dir(self, d: Path, source: str) -> list[Profile]:
        out: list[Profile] = []
        if not d.is_dir():
            return out
        for f in sorted(d.glob("*")):
            if f.suffix.lower() not in (".yaml", ".yml", ".json"):
                continue
            try:
                out.append(load_profile_file(f, source=source))
            except ProfileError:
                continue
        return out

    def _model_dirs(self) -> list[Path]:
        out: list[Path] = []
        seen: set[Path] = set()
        for root in self.model_roots:
            if not root.is_dir():
                continue
            candidates = [root] if (root / "cccp.json").is_file() else [
                item for item in sorted(root.iterdir()) if item.is_dir()
            ]
            for candidate in candidates:
                try:
                    resolved = candidate.resolve()
                except OSError:
                    continue
                if resolved not in seen and (resolved / "cccp.json").is_file():
                    seen.add(resolved)
                    out.append(resolved)
        return out

    def _load_model_profiles(self) -> None:
        for model_dir in self._model_dirs():
            try:
                model_fingerprint = hashlib.sha256(
                    (model_dir / "cccp.json").read_bytes()
                ).hexdigest()
            except OSError:
                continue
            for path in sorted((model_dir / "profiles").glob("*")):
                if path.suffix.lower() not in (".yaml", ".yml", ".json"):
                    continue
                try:
                    profile = load_profile_file(path, source="model")
                except ProfileError:
                    continue
                if profile.meta.get("model_name") != model_dir.name:
                    continue
                if profile.meta.get("model_manifest_sha256") != model_fingerprint:
                    continue
                self._profiles[profile.id] = profile
                self._model_profile_paths[profile.id] = path.resolve()

    def load_user(self) -> None:
        """用户导入配置覆盖同 id 模型预设；模型身份仍由 schema 强校验。"""
        for f in sorted(self.user_dir.glob("*")):
            if f.suffix.lower() not in (".yaml", ".yml", ".json"):
                continue
            try:
                p = load_profile_file(f, source="imported")
            except ProfileError:
                continue
            self._profiles[p.id] = p

    def reload(self) -> None:
        self._profiles.clear()
        self._model_profile_paths.clear()
        self._load_model_profiles()
        self.load_user()

    # -- 查询 --
    def list(self) -> list[Profile]:
        return [self._profiles[k] for k in sorted(self._profiles)]

    def get(self, pid: str) -> Profile | None:
        return self._profiles.get(pid)

    def require(self, pid: str) -> Profile:
        p = self.get(pid)
        if p is None:
            raise ProfileError(f"profile 不存在: {pid}")
        return p

    # -- 导入 / 删除 --
    def import_text(self, text: str, filename: str = "import.yaml") -> Profile:
        """校验模型身份完整的 YAML/JSON -> profiles/user/。"""
        suffix = Path(filename).suffix.lower()
        try:
            data = yaml.safe_load(text) if suffix in (".yaml", ".yml", "") else json.loads(text)
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise ProfileError(f"解析失败: {exc}") from exc
        if not isinstance(data, dict):
            raise ProfileError("顶层必须是对象")
        p = load_profile_dict(data, source="imported")
        if not p.meta.get("model_manifest_sha256"):
            raise ProfileError(
                "导入配置必须记录 model_name、model_version、model_format 和模型指纹"
            )
        out = self.user_dir / f"{p.id}.yaml"
        atomic_write_text(
            out, yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        )
        p.source = "imported"
        self._profiles[p.id] = p
        return p

    def import_file(self, path: Path) -> Profile:
        p = load_profile_file(path, source="imported")
        out = self.user_dir / f"{p.id}.yaml"
        data = {
            "schema": SCHEMA,
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "tags": p.tags,
            "experts": [
                {"key": e.key, "size_mb": e.size_mb, "tags": list(e.tags),
                 "route_count": e.route_count, "route_score": e.route_score}
                for e in p.experts
            ],
            "drop": {"enabled": p.drop.enabled, "hint_tags": p.drop.hint_tags},
            "meta": {
                **p.meta,
                "source": "imported",
                "calibrated": p.calibrated,
            },
        }
        if p.recipe:
            data["recipe"] = {
                "seed": p.recipe.seed,
                "layers": p.recipe.layers,
                "experts_per_layer": p.recipe.experts_per_layer,
                "density": p.recipe.density,
                "layer_affinity": p.recipe.layer_affinity,
                "mean_size_mb": p.recipe.mean_size_mb,
                "size_jitter": p.recipe.size_jitter,
            }
        atomic_write_text(
            out, yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        )
        p.source = "imported"
        self._profiles[p.id] = p
        return p

    def delete(self, pid: str) -> None:
        """删除用户配置或对应模型目录内的模型预设。"""
        profile = self.require(pid)
        if profile.source == "imported":
            for suffix in (".yaml", ".yml", ".json"):
                override = self.user_dir / f"{pid}{suffix}"
                if override.exists():
                    override.unlink()
        else:
            model_path = self._model_profile_paths.pop(pid, None)
            if model_path and model_path.is_file():
                model_path.unlink()
        self.reload()

    def update_metadata(self, pid: str, *, name: str, description: str) -> Profile:
        """修改名称/说明，同时保持配置继续归属于原模型或导入目录。"""
        profile = self.require(pid)
        data = profile.to_dict(with_experts=True)
        data["name"] = name
        data["description"] = description
        if profile.source == "model":
            output = self._model_profile_paths.get(pid)
            if output is None:
                raise ProfileError("找不到模型配置文件")
            if output.suffix.lower() in {".yaml", ".yml"}:
                atomic_write_text(
                    output, yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
                )
            else:
                atomic_write_text(
                    output, json.dumps(data, ensure_ascii=False, indent=2)
                )
            updated = load_profile_dict(data, source="model")
            self._profiles[pid] = updated
            return updated
        return self.import_text(
            json.dumps(data, ensure_ascii=False), filename=f"{pid}.json"
        )

    def register_for_model(self, data: dict[str, Any], model_path: str | Path) -> Profile:
        """把训练产物写入对应模型 ``profiles/``，并立即注册。"""
        model_dir = Path(model_path).expanduser().resolve()
        if not (model_dir / "cccp.json").is_file():
            raise ProfileError("训练配置对应的模型目录无效")
        profile = load_profile_dict(data, source="model")
        if profile.meta.get("model_name") != model_dir.name:
            raise ProfileError("训练配置的 model_name 与模型目录名称不一致")
        actual_fingerprint = hashlib.sha256((model_dir / "cccp.json").read_bytes()).hexdigest()
        if profile.meta.get("model_manifest_sha256") != actual_fingerprint:
            raise ProfileError("训练配置的模型指纹与所选模型不一致")
        output_dir = model_dir / "profiles"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{profile.id}.json"
        atomic_write_text(
            output, json.dumps(data, ensure_ascii=False, indent=2)
        )
        self._profiles[profile.id] = profile
        self._model_profile_paths[profile.id] = output
        return profile

    # -- 组合 + drop 解析 --
    def combine(self, ids: list[str]) -> Combination:
        profiles = [self.require(i) for i in ids]
        combo = combine(profiles)
        for p in profiles:
            if p.drop.enabled:
                key = resolve_drop(p, combo.union)
                if key:
                    combo.drop_resolution[p.id] = key
        return combo

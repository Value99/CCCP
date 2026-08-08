"""领域配置文件(Profile)模型与组合计算。

Profile 描述"为这个领域/任务载入哪些专家":
- 显式列举专家实例(key = "层号:专家号",携带打包后体积)
- 或通过确定性 recipe 抽样生成(内置配置使用,可校准)
- 多 Profile 组合时按 key 求并集:重叠专家只计一次体积
  (例:合同 100G + 代码 200G -> 并集约 250G)
- 每个 Profile 内置一个标记为 drop 的占位专家:
  不携带权重,启动时自动"路由"到当前组合中最相关的已加载专家
  (对应 TPQ-Final route 前的 drop-expert masking 语义)。
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

SCHEMA = "winui-expert-profile-v1"
_KEY_RE = re.compile(r"^\d+:\d+$")
_ID_RE = re.compile(r"[a-z0-9][a-z0-9\-_.]{1,63}")


@dataclass(frozen=True)
class ExpertRef:
    """一个被引用的路由专家实例。"""

    key: str  # "layer:expert_id",与 TPQ score-file 的 "layer:expert" 一致
    size_mb: float  # 打包 (VQ) 形态占用,MiB
    tags: tuple[str, ...] = ()

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
    作为 launch plan 的一部分输出(不改变 TPQ,仅生成 placement 提示)。
    """

    enabled: bool = True
    hint_tags: list[str] = field(default_factory=list)
    resolved: str | None = None  # 解析出的专家 key


@dataclass
class Recipe:
    """确定性抽样配方:用于内置 Profile,避免巨型清单文件。

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
    builtin: bool = False
    source: str = "builtin"  # builtin|imported|trained
    calibrated: bool = False  # True = 体积来自 TPQ 精确字节表(INTERFACE I-1)
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
            "builtin": self.builtin,
            "calibrated": self.calibrated,
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
                {"key": e.key, "size_mb": e.size_mb, "tags": list(e.tags)}
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

    @property
    def expert_count(self) -> int:
        return len(self.union)

    @property
    def memory_mb(self) -> float:
        return round(sum(e.size_mb for e in self.union.values()), 1)

    @property
    def memory_gb(self) -> float:
        return round(self.memory_mb / 1024.0, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profiles": self.profile_ids,
            "expert_count": self.expert_count,
            "memory_mb": self.memory_mb,
            "memory_gb": self.memory_gb,
            "overlap_mb": round(self.overlap_mb, 1),
            "overlap_gb": round(self.overlap_mb / 1024.0, 2),
            "drop_resolution": self.drop_resolution,
        }


def combine(profiles: Iterable[Profile]) -> Combination:
    """多 Profile 组合:按专家 key 求并集,重叠部分体积只计一次。"""
    union: dict[str, ExpertRef] = {}
    total_mb = 0.0
    ids: list[str] = []
    for p in profiles:
        p.materialize()
        ids.append(p.id)
        for e in p.experts:
            total_mb += e.size_mb
            if e.key not in union:
                union[e.key] = e
    union_mb = sum(e.size_mb for e in union.values())
    return Combination(
        profile_ids=ids,
        union=union,
        overlap_mb=round(total_mb - union_mb, 1),
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


def load_profile_dict(data: dict[str, Any], *, source: str = "imported") -> Profile:
    if data.get("schema") != SCHEMA:
        raise ProfileError(f"schema 必须是 {SCHEMA}")
    pid = str(data.get("id") or "").strip()
    if not pid:
        raise ProfileError("缺少 id")
    if not _ID_RE.fullmatch(pid):
        raise ProfileError(f"非法 id: {pid!r}")

    experts: list[ExpertRef] = []
    for raw in data.get("experts") or []:
        key = str(raw.get("key", ""))
        if not ExpertRef.valid_key(key):
            raise ProfileError(f"非法专家 key: {key!r}(应为 layer:expert)")
        size = float(raw.get("size_mb", 0))
        if size <= 0:
            raise ProfileError(f"专家 {key} 的 size_mb 必须 > 0")
        experts.append(
            ExpertRef(key=key, size_mb=size, tags=tuple(map(str, raw.get("tags") or ())))
        )

    recipe = None
    if data.get("recipe"):
        rd = data["recipe"]
        recipe = Recipe(
            seed=str(rd.get("seed", pid)),
            layers=int(rd["layers"]),
            experts_per_layer=int(rd["experts_per_layer"]),
            density=float(rd["density"]),
            layer_affinity=str(rd.get("layer_affinity", "uniform")),
            mean_size_mb=float(rd.get("mean_size_mb", 24.0)),
            size_jitter=float(rd.get("size_jitter", 0.35)),
        )
    if not experts and recipe is None:
        raise ProfileError("profile 必须提供 experts 清单或 recipe")

    dd = data.get("drop") or data.get("drop_expert") or {}
    meta = data.get("meta") or {}
    p = Profile(
        id=pid,
        name=str(data.get("name") or pid),
        description=str(data.get("description") or ""),
        tags=[str(t) for t in (data.get("tags") or [])],
        experts=experts,
        recipe=recipe,
        drop=DropExpert(
            enabled=bool(dd.get("enabled", True)),
            hint_tags=[str(t) for t in (dd.get("hint_tags") or [])],
        ),
        builtin=False,
        source=source,
        calibrated=bool(meta.get("calibrated", False)),
    )
    p.materialize()
    return p


def load_profile_file(path: Path, *, source: str = "imported") -> Profile:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ProfileError(f"{path.name}: 顶层必须是对象")
    return load_profile_dict(data, source=source)


# --------------------------------------------------------------------------
# 注册表
# --------------------------------------------------------------------------

class ProfileRegistry:
    """内置(profiles/builtin/*.yaml)+ 用户导入(profiles/user/*.yaml)注册表。

    导入的 profile 与内置同 id 时覆盖内置;删除覆盖即可还原内置。
    """

    def __init__(self, builtin_dir: Path, user_dir: Path):
        self.builtin_dir = builtin_dir
        self.user_dir = user_dir
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, Profile] = {}
        self._builtin_ids: set[str] = set()
        self._load_builtin()
        self.load_user()

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

    def _load_builtin(self) -> None:
        for p in self._load_dir(self.builtin_dir, "builtin"):
            p.builtin = True
            p.source = "builtin"
            self._profiles[p.id] = p
            self._builtin_ids.add(p.id)

    def load_user(self) -> None:
        """重载 user 目录:导入的同 id 覆盖内置。"""
        for f in sorted(self.user_dir.glob("*")):
            if f.suffix.lower() not in (".yaml", ".yml", ".json"):
                continue
            try:
                p = load_profile_file(f, source="imported")
            except ProfileError:
                continue
            if p.id in self._builtin_ids:
                p.source = "imported"  # 覆盖内置,但保留不可删除标记
            self._profiles[p.id] = p

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
        """校验 YAML/JSON 文本 -> 落盘 profiles/user/ -> 注册(可覆盖内置同 id)。"""
        suffix = Path(filename).suffix.lower()
        try:
            data = yaml.safe_load(text) if suffix in (".yaml", ".yml", "") else json.loads(text)
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise ProfileError(f"解析失败: {exc}") from exc
        if not isinstance(data, dict):
            raise ProfileError("顶层必须是对象")
        p = load_profile_dict(data, source="imported")
        out = self.user_dir / f"{p.id}.yaml"
        out.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
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
                {"key": e.key, "size_mb": e.size_mb, "tags": list(e.tags)}
                for e in p.experts
            ],
            "drop": {"enabled": p.drop.enabled, "hint_tags": p.drop.hint_tags},
            "meta": {"source": "imported", "calibrated": p.calibrated},
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
        out.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        p.source = "imported"
        self._profiles[p.id] = p
        return p

    def delete(self, pid: str) -> None:
        """删除导入 profile;若为内置的覆盖则还原内置定义。"""
        p = self.require(pid)
        override = self.user_dir / f"{pid}.yaml"
        if pid in self._builtin_ids and not override.exists():
            raise ProfileError("内置 profile 不可删除")
        if override.exists():
            override.unlink()
        if pid in self._builtin_ids:
            # 还原内置
            for bp in self._load_dir(self.builtin_dir, "builtin"):
                if bp.id == pid:
                    bp.builtin = True
                    bp.source = "builtin"
                    self._profiles[pid] = bp
                    break
        else:
            self._profiles.pop(pid, None)

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

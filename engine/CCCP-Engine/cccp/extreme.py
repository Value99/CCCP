"""单卡 RAM+VRAM 极限常驻模式的公共配置与容量规划。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Hashable, Mapping


GIB = 2**30
EXTREME_GPU_LOAD_WORKSPACE_GIB = 1.5
EXTREME_RAM_LOAD_WORKSPACE_GIB = 0.5
EXTREME_RAM_RESERVE_GIB = 2.0


def effective_available_memory_bytes(
    *,
    system_available_bytes: int | None = None,
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    cgroup_file: str | os.PathLike[str] = "/proc/self/cgroup",
) -> int:
    """返回物理机和 cgroup 两种约束下更小的可用内存。

    原生小内存机器由 psutil 正确报告；容器、systemd service 和受限桌面
    环境还必须扣除 cgroup v2 的 ``memory.current``，否则极限模式会按宿主机
    总内存规划，直到被 OOM killer 终止。
    """

    if system_available_bytes is None:
        import psutil

        system_available_bytes = int(psutil.virtual_memory().available)
    available = max(0, int(system_available_bytes))
    try:
        cgroup_path = next(
            line.split("::", 1)[1].strip()
            for line in Path(cgroup_file).read_text(
                encoding="utf-8",
            ).splitlines()
            if "::" in line
        )
        root = Path(cgroup_root) / cgroup_path.lstrip("/")
        maximum_text = (root / "memory.max").read_text(
            encoding="ascii",
        ).strip()
        if maximum_text != "max":
            maximum = int(maximum_text)
            current = int(
                (root / "memory.current").read_text(
                    encoding="ascii",
                ).strip()
            )
            # cgroup memory.current includes filesystem page cache.  Model
            # loading creates a large inactive file cache that the kernel can
            # reclaim before charging packed expert tensors.  Treating it as
            # anonymous residency makes an otherwise identical capacity plan
            # depend on whether the file was read recently.
            inactive_file = 0
            try:
                memory_stat = dict(
                    line.split(maxsplit=1)
                    for line in (root / "memory.stat").read_text(
                        encoding="ascii",
                    ).splitlines()
                    if " " in line
                )
                inactive_file = max(
                    0,
                    int(memory_stat.get("inactive_file", "0")),
                )
            except (OSError, ValueError):
                pass
            cgroup_available = min(
                maximum,
                max(0, maximum - current) + inactive_file,
            )
            available = min(available, cgroup_available)
    except (OSError, StopIteration, ValueError):
        pass
    return available


@dataclass(frozen=True)
class ExtremeLayerPlacement:
    """连续层放置结果；RAM 前缀之后的层全部进入 VRAM。"""

    ram_layers: tuple[int, ...]
    gpu_layers: tuple[int, ...]
    ram_bytes: int
    gpu_bytes: int
    ram_capacity: int
    gpu_capacity: int


@dataclass(frozen=True)
class ExtremeExpertPlacement:
    """Capacity-safe expert placement ranked by a model-provided precision signal."""

    ram_keys: tuple[Hashable, ...]
    gpu_keys: tuple[Hashable, ...]
    ram_bytes: int
    gpu_bytes: int
    ram_capacity: int
    gpu_capacity: int


@dataclass(frozen=True)
class CompactArchiveCapacity:
    """模型无关的紧凑三投影归档容量与公共算子签名。"""

    expert_bytes: int
    layers: tuple[int, ...]
    packed_formats: tuple[str, ...]
    code_dims: tuple[int, ...]
    codebook_sizes: tuple[int, ...]


def _expert_audit_payload_bytes(detail: object) -> int:
    """Return the compact index payload recorded for one routed expert."""

    if not isinstance(detail, dict):
        return 0
    projections = detail.get("projections")
    if isinstance(projections, dict):
        return sum(
            int((projection or {}).get("packed_bytes") or 0)
            for projection in projections.values()
            if isinstance(projection, dict)
        )
    if "gu_bytes" in detail or "down_bytes" in detail:
        return int(detail.get("gu_bytes") or 0) + int(
            detail.get("down_bytes") or 0
        )
    return sum(
        int((detail.get(projection) or {}).get("packed_bytes") or 0)
        for projection in ("gate", "up", "down", "gu", "dn")
        if isinstance(detail.get(projection), dict)
    )


def _strict_profile_allowlist(
    model_root: Path,
) -> dict[int, set[int]] | None:
    """Read the same strict route selection that CCCPStore will enforce."""

    if os.environ.get("CCCP_ROUTE_PROFILE", "0") == "0":
        return None
    profile_path = Path(
        os.environ.get("CCCP_PROFILE_JSON")
        or model_root / "profile.json"
    )
    if not profile_path.is_file():
        return None
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    raw_allowed = profile.get("allowed_experts")
    if raw_allowed is None and profile.get("strict_route"):
        raw_allowed = {
            layer: list(counts)
            for layer, counts in (profile.get("counts") or {}).items()
        }
    if not isinstance(raw_allowed, dict):
        raise ValueError("路由 Profile 缺少 allowed_experts 对象")
    return {
        int(layer): {int(expert) for expert in experts}
        for layer, experts in raw_allowed.items()
    }


@dataclass(frozen=True)
class AutoExtremeDecision:
    """配置驱动的单卡自动放置结论。"""

    activate: bool
    mode: str
    reason: str
    expert_bytes: int = 0
    available_ram_bytes: int = 0
    normal_ram_capacity: int = 0
    extreme_ram_capacity: int = 0
    gpu_expert_capacity: int = 0
    spill_bytes: int = 0


def plan_auto_extreme(
    *,
    compact_expert_bytes: int,
    available_ram_bytes: int,
    free_gpu_bytes: int,
    fixed_gpu_bytes: int,
    normal_ram_reserve_bytes: int = 2 * GIB,
    extreme_ram_reserve_bytes: int = 2 * GIB,
    load_workspace_bytes: int = int(EXTREME_RAM_LOAD_WORKSPACE_GIB * GIB),
    gpu_reserve_bytes: int = 512 * 2**20,
) -> AutoExtremeDecision:
    """在全显存、普通 RAM、RAM+VRAM 极限放置之间自动选择。

    输入全部来自 Manifest、文件实际字节和当前机器容量，因此不识别模型名。
    紧凑专家与固定权重能完整进入单卡时优先选择 resident；否则仅在普通
    RAM 安全容量不足、但紧凑专家溢出部分能进入固定显存余量时激活极限
    模式；两侧都不足时保留普通路径，让上层给出明确容量诊断。
    """

    expert_bytes = max(0, int(compact_expert_bytes))
    available_ram = max(0, int(available_ram_bytes))
    normal_ram_capacity = max(
        0,
        available_ram - max(0, int(normal_ram_reserve_bytes)),
    )
    extreme_ram_capacity = max(
        0,
        available_ram
        - max(0, int(extreme_ram_reserve_bytes))
        - max(0, int(load_workspace_bytes)),
    )
    gpu_expert_capacity = max(
        0,
        int(free_gpu_bytes)
        - max(0, int(fixed_gpu_bytes))
        - max(0, int(gpu_reserve_bytes)),
    )
    spill_bytes = max(0, expert_bytes - extreme_ram_capacity)
    common = dict(
        expert_bytes=expert_bytes,
        available_ram_bytes=available_ram,
        normal_ram_capacity=normal_ram_capacity,
        extreme_ram_capacity=extreme_ram_capacity,
        gpu_expert_capacity=gpu_expert_capacity,
        spill_bytes=spill_bytes,
    )
    if expert_bytes <= gpu_expert_capacity:
        return AutoExtremeDecision(
            activate=False,
            mode="resident",
            reason="固定权重与紧凑专家可完整进入单卡显存",
            **common,
        )
    if expert_bytes <= normal_ram_capacity:
        return AutoExtremeDecision(
            activate=False,
            mode="ram",
            reason="普通 RAM 安全容量足够",
            **common,
        )
    if spill_bytes <= gpu_expert_capacity:
        return AutoExtremeDecision(
            activate=True,
            mode="extreme",
            reason="RAM 单侧不足，但 RAM+VRAM 可容纳紧凑归档",
            **common,
        )
    return AutoExtremeDecision(
        activate=False,
        mode="insufficient",
        reason="RAM+VRAM 扣除固定权重和工作区后仍不足",
        **common,
    )


def detect_auto_extreme(
    model_dir: str | os.PathLike[str],
    *,
    max_ctx: int,
    device: str,
    tp_size: int,
    normal_ram_reserve_gib: float = 2.0,
    environment: Mapping[str, str] | None = None,
) -> AutoExtremeDecision:
    """读取公共 Manifest 和当前硬件，给出自动极限模式结论。"""

    if str(device) != "cuda" or int(tp_size) != 1:
        return AutoExtremeDecision(
            False,
            "disabled",
            "自动极限模式只适用于单卡 CUDA",
        )
    try:
        archive = inspect_compact_projection_archive(model_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        return AutoExtremeDecision(
            False,
            "unsupported",
            str(exc),
        )

    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return AutoExtremeDecision(False, "disabled", "CUDA 不可用")
        free_gpu, _total_gpu = torch.cuda.mem_get_info(0)
    except (ImportError, RuntimeError) as exc:
        return AutoExtremeDecision(False, "disabled", f"显存探测失败：{exc}")

    from .check import fixed_vram_gib
    from .presets import detect_architecture, load_manifest

    root, manifest = load_manifest(model_dir)
    architecture = detect_architecture(manifest)
    effective_environment = dict(environment or {})
    fixed_gpu = int(
        max(
            0.0,
            fixed_vram_gib(
                root,
                manifest,
                architecture,
                int(max_ctx),
                effective_environment,
            ) - EXTREME_GPU_LOAD_WORKSPACE_GIB,
        )
        * GIB
    )
    cap_gib = max(
        0.0,
        float(os.environ.get("CCCP_EXTREME_VRAM_CAP_GB", "0") or 0),
    )
    if cap_gib:
        free_gpu = min(int(free_gpu), int(cap_gib * GIB))
    loader_workspace = int(
        max(
            EXTREME_RAM_LOAD_WORKSPACE_GIB,
            float(
                effective_environment.get(
                    "CCCP_EXTREME_LOAD_WORKSPACE_GB",
                    EXTREME_RAM_LOAD_WORKSPACE_GIB,
                )
            ),
        )
        * GIB
    )
    return plan_auto_extreme(
        compact_expert_bytes=archive.expert_bytes,
        available_ram_bytes=effective_available_memory_bytes(),
        free_gpu_bytes=int(free_gpu),
        fixed_gpu_bytes=fixed_gpu,
        normal_ram_reserve_bytes=int(normal_ram_reserve_gib * GIB),
        load_workspace_bytes=loader_workspace,
    )


def inspect_compact_projection_archive(
    model_dir: str | os.PathLike[str],
) -> CompactArchiveCapacity:
    """通过公共 Manifest 审计可现场解包的 projection-VQ 归档。

    这里不识别模型名，也不要求旧版 ``projection_layouts`` 字段。逐层布局和
    逐专家异构档位都会先由 :class:`Manifest` 规范化，再解析为公共算子能力键。
    专家容量按实际归档文件计算，包含码本和 safetensors 元数据，因此用于显存
    fast-path 判定时比只统计索引 payload 更保守。
    """

    from .store import Manifest

    root = Path(model_dir)
    manifest = Manifest(str(root))
    if not manifest.packed_expert_vq:
        raise RuntimeError(
            "极限模式只接受可由公共 packed 算子直接计算的 CCCP "
            "归档；当前模型会展开专家索引。"
        )
    formats: set[str] = set()
    dimensions: set[int] = set()
    codebooks: set[int] = set()
    layers = tuple(sorted(int(layer) for layer in manifest.expert_files))
    for layer in layers:
        capability = manifest.projection_operator_capability(layer)
        formats.update(
            str(value) for value in capability.get("packed_formats", ())
        )
        dimensions.update(
            int(value) for value in capability.get("code_dims", ())
        )
        codebooks.update(
            int(value) for value in capability.get("codebook_sizes", ())
        )
    unsupported = sorted(
        value
        for value in formats
        if not value.startswith("p")
        or not value.removeprefix("p").isdigit()
        or not 8 <= int(value.removeprefix("p")) <= 16
    )
    if unsupported:
        raise RuntimeError(
            "极限模式不支持以下专家索引格式：" + ", ".join(unsupported)
        )
    files = {
        int(layer): root / str(name)
        for layer, name in manifest.expert_files.items()
    }
    missing = sorted(str(path) for path in files.values() if not path.is_file())
    if missing:
        raise RuntimeError("极限模式缺少专家文件：" + ", ".join(missing))
    allowlist = _strict_profile_allowlist(root)
    expert_bytes = 0
    for layer, path in files.items():
        file_bytes = int(path.stat().st_size)
        if allowlist is None:
            expert_bytes += file_bytes
            continue
        audit_name = manifest.expert_audit_files.get(int(layer))
        if not audit_name:
            # Without a per-expert audit there is no safe way to subtract the
            # excluded payload. Keep the physical shard size conservative.
            expert_bytes += file_bytes
            continue
        audit_path = root / str(audit_name)
        if not audit_path.is_file():
            raise RuntimeError(
                f"严格路由容量审计缺少 L{layer} 文件：{audit_path}"
            )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        payloads = {
            int(str(expert_id).lstrip("e")):
                _expert_audit_payload_bytes(detail)
            for expert_id, detail in (audit.get("experts") or {}).items()
        }
        complete_payload = sum(payloads.values())
        selected_payload = sum(
            payloads.get(expert_id, 0)
            for expert_id in allowlist.get(int(layer), set())
        )
        # The shard also contains the experts' private codebooks and
        # safetensors bookkeeping. The training/configuration UI calibrates
        # these bytes by the audited packed payload ratio; use the same rule
        # here so excluded experts do not bring their private codebooks back
        # into a supposedly strict-resident capacity estimate.
        if complete_payload > 0:
            expert_bytes += (
                file_bytes * selected_payload + complete_payload - 1
            ) // complete_payload
        else:
            expert_bytes += file_bytes
    return CompactArchiveCapacity(
        expert_bytes=expert_bytes,
        layers=layers,
        packed_formats=tuple(sorted(formats)),
        code_dims=tuple(sorted(dimensions)),
        codebook_sizes=tuple(sorted(codebooks)),
    )


def choose_extreme_strategy(
    *,
    compact_expert_bytes: int,
    fixed_gpu_bytes: int,
    gpu_limit_bytes: int,
) -> str:
    """显存能容纳完整紧凑模型时直接复用 resident 公共 fast-path。"""

    required = int(compact_expert_bytes) + int(fixed_gpu_bytes)
    return "full-gpu" if required <= int(gpu_limit_bytes) else "layered"


def plan_extreme_layer_placement(
    layer_bytes: Mapping[int, int],
    *,
    available_ram_bytes: int,
    gpu_expert_bytes: int,
    ram_reserve_bytes: int = 2 * GIB,
    fixed_ram_bytes: int = 0,
    fixed_gpu_bytes: int = 0,
) -> ExtremeLayerPlacement:
    """把最大连续层前缀放入 RAM，其余完整层放入 VRAM。

    规划只使用紧凑专家 payload 字节。共享码本和加载 workspace 必须由调用方
    通过 ``fixed_ram_bytes`` 先行扣除；Dense、KV、GPU workspace 和 staging
    通过 ``fixed_gpu_bytes`` 扣除。不允许把半层专家拆到两种设备，也不允许
    回退到运行期磁盘读取。
    """

    ordered = tuple(sorted((int(k), int(v)) for k, v in layer_bytes.items()))
    if any(size < 0 for _layer, size in ordered):
        raise ValueError("极限模式层字节数不能为负")
    ram_capacity = max(
        0,
        int(available_ram_bytes)
        - int(ram_reserve_bytes)
        - int(fixed_ram_bytes),
    )
    gpu_capacity = max(0, int(gpu_expert_bytes) - int(fixed_gpu_bytes))
    ram_layers: list[int] = []
    gpu_layers: list[int] = []
    ram_used = 0
    overflow = False
    for layer, size in ordered:
        if not overflow and ram_used + size <= ram_capacity:
            ram_layers.append(layer)
            ram_used += size
        else:
            overflow = True
            gpu_layers.append(layer)
    gpu_used = sum(dict(ordered)[layer] for layer in gpu_layers)
    if gpu_used > gpu_capacity:
        raise RuntimeError(
            "极限模式容量不足：RAM 保留 "
            f"{ram_reserve_bytes / GIB:.2f} GiB 后只能放 "
            f"{len(ram_layers)}/{len(ordered)} 层，剩余 "
            f"{gpu_used / GIB:.2f} GiB 专家需要 VRAM，但可用仅 "
            f"{gpu_capacity / GIB:.2f} GiB。请降低上下文、关闭其他进程，"
            "或换用更小模型。"
        )
    return ExtremeLayerPlacement(
        ram_layers=tuple(ram_layers),
        gpu_layers=tuple(gpu_layers),
        ram_bytes=ram_used,
        gpu_bytes=gpu_used,
        ram_capacity=ram_capacity,
        gpu_capacity=gpu_capacity,
    )


def plan_extreme_expert_placement(
    expert_bytes: Mapping[Hashable, int],
    precision_scores: Mapping[Hashable, float],
    *,
    placement_groups: Mapping[Hashable, Hashable] | None = None,
    available_ram_bytes: int,
    gpu_expert_bytes: int,
    ram_reserve_bytes: int = 2 * GIB,
    fixed_ram_bytes: int = 0,
    fixed_gpu_bytes: int = 0,
) -> ExtremeExpertPlacement:
    """Keep precision-budgeted experts on GPU while satisfying RAM capacity.

    CCCP quantizers may assign more packed bits to frequently routed or more
    sensitive experts.  The score is deliberately supplied by the manifest
    adapter: this common planner neither recognizes model names nor assumes a
    particular set of bit widths.
    """

    sizes = {key: int(value) for key, value in expert_bytes.items()}
    if any(value < 0 for value in sizes.values()):
        raise ValueError("extreme expert bytes cannot be negative")
    ram_capacity = max(
        0,
        int(available_ram_bytes)
        - int(ram_reserve_bytes)
        - int(fixed_ram_bytes),
    )
    gpu_capacity = max(0, int(gpu_expert_bytes) - int(fixed_gpu_bytes))
    ram_used = sum(sizes.values())
    if placement_groups is None:
        ranked = sorted(
            sizes,
            key=lambda key: (
                float(precision_scores.get(key, 0.0)),
                sizes[key],
                str(key),
            ),
            reverse=True,
        )
    else:
        missing = sizes.keys() - placement_groups.keys()
        if missing:
            raise ValueError(
                "extreme placement groups do not cover every expert"
            )
        grouped: dict[Hashable, list[Hashable]] = {}
        for key in sizes:
            grouped.setdefault(placement_groups[key], []).append(key)
        ranked_rows: list[tuple[int, float, int, str, Hashable]] = []
        for keys in grouped.values():
            keys.sort(
                key=lambda key: (
                    float(precision_scores.get(key, 0.0)),
                    sizes[key],
                    str(key),
                ),
                reverse=True,
            )
            maximum = max(
                (float(precision_scores.get(key, 0.0)) for key in keys),
                default=0.0,
            )
            for rank, key in enumerate(keys):
                relative_score = (
                    float(precision_scores.get(key, 0.0)) / maximum
                    if maximum > 0
                    else 0.0
                )
                # Rank is primary: each group contributes its hottest expert
                # before any group contributes its second hottest.  CCCP's
                # fixed-per-layer budgets are only comparable within a layer,
                # while every routed layer runs once per token.
                ranked_rows.append(
                    (-rank, relative_score, sizes[key], str(key), key)
                )
        ranked_rows.sort(reverse=True)
        ranked = [row[-1] for row in ranked_rows]
    gpu_keys: list[Hashable] = []
    for key in ranked:
        if ram_used <= ram_capacity:
            break
        gpu_keys.append(key)
        ram_used -= sizes[key]
    if ram_used > ram_capacity:
        raise RuntimeError("extreme mode cannot satisfy RAM expert capacity")
    gpu_set = set(gpu_keys)
    gpu_used = sum(sizes[key] for key in gpu_keys)
    if gpu_used > gpu_capacity:
        raise RuntimeError(
            "extreme mode precision-weighted GPU experts need "
            f"{gpu_used / GIB:.2f} GiB, but only "
            f"{gpu_capacity / GIB:.2f} GiB is available"
        )
    return ExtremeExpertPlacement(
        ram_keys=tuple(key for key in sizes if key not in gpu_set),
        gpu_keys=tuple(gpu_keys),
        ram_bytes=ram_used,
        gpu_bytes=gpu_used,
        ram_capacity=ram_capacity,
        gpu_capacity=gpu_capacity,
    )


def load_expert_residency_scores(
    path: str | os.PathLike[str],
) -> dict[tuple[int, int], float]:
    """Load portable expert hotness scores without recognizing a model.

    Quantizers can emit either the compact CCCP score schema or CCCP's existing
    expert-preference audit.  Both describe layer/expert coordinates and a
    non-negative score; runtime placement remains independent of architecture.
    """

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    format_name = str(payload.get("format", ""))
    output: dict[tuple[int, int], float] = {}
    if format_name == "cccp-expert-residency-scores-v1":
        for coordinate, value in (payload.get("scores") or {}).items():
            layer_text, separator, expert_text = str(coordinate).partition(":")
            if not separator:
                raise ValueError(
                    "expert residency score keys must use layer:expert"
                )
            output[(int(layer_text), int(expert_text))] = float(value)
    elif format_name == "cccp-expert-projection-preference-map-v1":
        for layer_text, layer_data in (payload.get("layers") or {}).items():
            for expert in layer_data.get("experts", ()):
                output[(int(layer_text), int(expert["expert"]))] = float(
                    expert["route_mass"]
                )
    else:
        raise ValueError(
            "unsupported expert residency score format: " + format_name
        )
    if not output:
        raise ValueError("expert residency score file contains no experts")
    if any(not math.isfinite(value) or value < 0 for value in output.values()):
        raise ValueError("expert residency scores must be finite and non-negative")
    return output


def configure_extreme_environment() -> None:
    """应用统一极限模式；启动器随后仍可用显式 CLI 覆盖 VRAM 预留。"""

    os.environ["CCCP_EXTREME_MODE"] = "1"
    os.environ["CCCP_FULL_RESIDENT"] = "1"
    os.environ["CCCP_RAM_MIRROR"] = "0"
    os.environ["CCCP_RAM_RESERVE_GB"] = str(EXTREME_RAM_RESERVE_GIB)
    os.environ["CCCP_RESIDENT_RESERVE_GB"] = str(EXTREME_RAM_RESERVE_GIB)
    # 极限模式已经把 RAM 压到 1 GiB 安全线。此时再 mlock 数十 GiB 会迫使
    # 内核把 Python/文件页换出，既不增加容量还会显著拖慢启动和 decode；
    # 保持普通小型 pinned staging，用户有额外 RAM 时仍可显式覆盖。
    os.environ.setdefault("CCCP_HOST_PIN_GB", "0")
    # 同一时刻只保留少量专家物化临时态，避免 12 个大专家并发把 1 GiB
    # 系统余量顶入 swap。SSD 顺序读吞吐基本不受这个保守并发数影响。
    os.environ.setdefault("CCCP_LOAD_WORKERS", "2")
    os.environ.setdefault("CCCP_VRAM_RESERVE_GB", "0.25")
    # Dense、KV 与公共算子 workspace 都在 packed arena 之前完成实际分配；
    # 此处只需给 decode 热路径保留小块临时余量。若仍按普通模式再留 1 GiB，
    # 16 GiB 卡会少约 128 个专家槽，恰好容不下 40 层 Top-8 的一轮路由，
    # 进而触发跨 token 的环形 LRU 踩踏。
    os.environ.setdefault("CCCP_VRAM_RESERVE_GB", "0.25")
    os.environ["CCCP_VRAM_WATCH"] = "0"


def extreme_enabled() -> bool:
    return os.environ.get("CCCP_EXTREME_MODE", "0") != "0"


__all__ = [
    "AutoExtremeDecision",
    "CompactArchiveCapacity",
    "EXTREME_GPU_LOAD_WORKSPACE_GIB",
    "EXTREME_RAM_LOAD_WORKSPACE_GIB",
    "EXTREME_RAM_RESERVE_GIB",
    "ExtremeLayerPlacement",
    "choose_extreme_strategy",
    "configure_extreme_environment",
    "detect_auto_extreme",
    "effective_available_memory_bytes",
    "extreme_enabled",
    "inspect_compact_projection_archive",
    "plan_auto_extreme",
    "plan_extreme_layer_placement",
]

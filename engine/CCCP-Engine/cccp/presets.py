"""模型识别与发布预设加载。

发布入口只依赖模型目录中的 ``cccp.json``，不依赖模型文件名：

* 含 ``hc_mult`` 或 ``compress_ratios`` 的模型识别为 DeepSeek-V4；
* 含 ``tensor_vq``、不含动态专家，并声明 Qwen3.5 文本架构的模型识别为
  通用 Dense VQ Qwen3.5；
* 其他当前 CCCP MoE 模型识别为 GLM。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).with_name("configs")


@dataclass(frozen=True)
class ResolvedPreset:
    model_dir: Path
    manifest: dict[str, Any]
    architecture: str
    display_name: str
    profile: str
    config_profile: str
    tp: int
    ep_layout: str | None
    defaults: dict[str, Any]
    environment: dict[str, str]
    supports_parallel: bool


def load_manifest(model_dir: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    root = Path(model_dir).expanduser().resolve()
    manifest_path = root / "cccp.json"
    if not root.is_dir():
        raise ValueError(f"模型目录不存在：{root}")
    if not manifest_path.is_file():
        raise ValueError(f"模型目录缺少 cccp.json：{root}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "cccp-1":
        raise ValueError(
            f"不支持的模型格式：{manifest.get('format')!r}，需要 'cccp-1'"
        )
    if not isinstance(manifest.get("config"), dict):
        raise ValueError("cccp.json 缺少 config 对象")
    return root, manifest


def detect_architecture(manifest: dict[str, Any]) -> str:
    config = manifest["config"]
    if str(manifest.get("architecture") or "").lower() == "glm5_next":
        return "glm5_next"
    if (
        str(manifest.get("model_family", "")).lower() == "kimi_k3"
        or ("kda_layers" in config and "routed_hidden" in config)
    ):
        return "kimi_k3"
    if "hc_mult" in config or "compress_ratios" in config:
        return "dsv4"
    routed = manifest.get("routed_experts") or {}
    if (
        isinstance(manifest.get("tensor_vq"), dict)
        and bool(manifest["tensor_vq"])
        and not bool(manifest.get("expert_files"))
        and not bool(routed.get("layer_files"))
        and str(
            config.get("text_model_type")
            or config.get("outer_model_type")
            or manifest.get("architecture")
            or ""
        ).lower().startswith("qwen3_5")
    ):
        return "qwen3_5_dense"
    return "glm"


def model_context_limit(manifest: dict[str, Any]) -> int:
    """Return the model-declared logical context ceiling.

    Runtime KV pages still start small and grow on demand; this value is only
    the tokenizer/model contract, never an eager cache allocation request.
    """
    config = manifest.get("config") or {}
    for name in (
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
        "model_max_length",
    ):
        value = int(config.get(name) or 0)
        if value > 0:
            return value
    # A malformed early fixture may omit the contract. Keep parsing possible,
    # but released models are expected to declare one of the fields above.
    return 32768


def load_arch_config(architecture: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{architecture}.json"
    if not path.is_file():
        raise ValueError(f"没有架构配置：{path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema") != 1 or data.get("architecture") != architecture:
        raise ValueError(f"架构配置格式错误：{path}")
    return data


def choose_ep_layout(manifest: dict[str, Any], tp: int) -> str:
    """选择多卡专家布局；不能整除时自动改用 expert-ID 分片。"""
    if tp < 2:
        raise ValueError("并行布局要求 tp >= 2")
    if detect_architecture(manifest) == "kimi_k3":
        # This is an operator layout selected by configuration, not a
        # separate Kimi execution system.
        return "tensor"
    config = manifest["config"]
    intermediate = int(config["moe_inter"])
    quant = manifest.get("quant", {})
    dims = {
        int(value[0])
        for value in quant.get("vq", {}).values()
    }
    if quant.get("method") == "projection-vq":
        projection_layouts = quant.get("projection_layouts", {})
        used_layouts = set()
        for projections in projection_layouts.values():
            if isinstance(projections, dict):
                used_layouts.update(str(layout) for layout in projections.values())
            else:
                used_layouts.add(str(projections))
        heterogeneous = quant.get(
            "heterogeneous_expert_tiering"
        ) or {}
        used_layouts.update(
            str(layout)
            for projections in heterogeneous.get(
                "precision_levels", {}
            ).values()
            for layout in projections.values()
        )
        declared_layouts = quant.get("layouts", {})
        for layout in used_layouts:
            item = declared_layouts.get(layout)
            if isinstance(item, dict) and "dim" in item:
                dims.add(int(item["dim"]))
                continue
            prefix = str(layout).split("-", 1)[0].lower()
            if prefix.startswith("d") and prefix[1:].isdigit():
                dims.add(int(prefix[1:]))
                continue
            raise ValueError(f"无法解析专家布局维度：{layout}")
    tensor_ok = intermediate % tp == 0
    if tensor_ok:
        local = intermediate // tp
        tensor_ok = all(local % dim == 0 for dim in dims)
    return "tensor" if tensor_ok else "expert"


def _environment_value(
    environment: dict[str, str],
    key: str,
    default: str,
) -> str:
    """Return the value which ``apply_preset_environment`` will expose."""
    return str(os.environ.get(key, environment.get(key, default)))


def _environment_enabled(
    environment: dict[str, str],
    key: str,
    default: str = "0",
) -> bool:
    return _environment_value(environment, key, default) != "0"


def _kimi_small_tp_width(
    tp: int,
    environment: dict[str, str],
) -> int:
    """Mirror Kimi's public small-op subgroup selection without CUDA."""
    no_owner = (
        _environment_enabled(environment, "CCCP_TP_HIDDEN_STATE")
        and _environment_enabled(environment, "CCCP_TP_NO_OWNER", "1")
    )
    if no_owner:
        return int(tp)
    width = min(
        int(tp),
        max(
            1,
            int(_environment_value(
                environment,
                "CCCP_SMALL_OP_TP",
                "4",
            )),
        ),
    )
    while width > 1 and int(tp) % width:
        width -= 1
    return width


def _packing_bits(value: Any) -> int:
    if isinstance(value, dict):
        value = value.get("bits")
    if isinstance(value, int):
        return int(value)
    matched = re.search(r"(?:packed-)?u(\d+)$", str(value))
    if matched is None:
        raise ValueError(f"无法识别 packed 索引格式 {value!r}")
    return int(matched.group(1))


def _kimi_used_down_layouts(manifest: dict[str, Any]) -> set[str]:
    """Collect only the projection layouts actually referenced by experts."""
    used: set[str] = set()
    routed = manifest.get("routed_experts", {})
    for item in routed.get("layer_files", {}).values():
        projection = item.get("projection_layout", {})
        if projection.get("down") is not None:
            used.add(str(projection["down"]))
    quant = manifest.get("quant", {})
    heterogeneous = quant.get("heterogeneous_expert_tiering") or {}
    for projection in heterogeneous.get("precision_levels", {}).values():
        if projection.get("down") is not None:
            used.add(str(projection["down"]))
    if not used:
        definitions = quant.get("layouts") or quant.get(
            "projection_layouts", {}
        )
        if all(
            isinstance(item, dict) and "dim" in item
            for item in definitions.values()
        ):
            used.update(str(name) for name in definitions)
    return used


def validate_parallel_shapes(
    manifest: dict[str, Any],
    tp: int,
    environment: dict[str, str],
    ep_layout: str,
) -> None:
    """Reject unsupported tensor partitions before CUDA allocation.

    The checks describe public operator capabilities, not model-directory
    names.  They intentionally mirror the divisibility contracts enforced by
    packed expert, Attention, gated-MLP, Router/Down and row-linear kernels.
    """
    if detect_architecture(manifest) != "kimi_k3" or ep_layout != "tensor":
        return
    config = manifest["config"]
    required = {
        "hidden",
        "routed_hidden",
        "n_experts",
        "moe_inter",
        "n_shared",
        "inter_dense",
        "first_dense_layers",
        "n_heads",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(
            "Kimi cccp.json config 缺少真TP所需字段："
            + ",".join(missing)
        )
    tp = int(tp)
    small_tp = _kimi_small_tp_width(tp, environment)
    failures: list[str] = []

    def require(label: str, value: int, divisor: int) -> None:
        if int(value) % int(divisor):
            failures.append(
                f"{label}={int(value)} 不能整除 TP{int(divisor)}"
            )

    # Every routed packed expert is sharded across the complete TP group.
    require("moe_inter", int(config["moe_inter"]), tp)

    attention_tp = _environment_enabled(
        environment,
        "CCCP_ATTENTION_TP",
        "1",
    )
    if attention_tp:
        require("n_heads", int(config["n_heads"]), small_tp)

    first_dense_tp = _environment_enabled(
        environment,
        "CCCP_FIRST_DENSE_TP",
        _environment_value(
            environment,
            "CCCP_DENSE_TP",
            "1",
        ),
    )
    if first_dense_tp and int(config.get("first_dense_layers", 0)):
        require("inter_dense", int(config["inter_dense"]), small_tp)

    shared_tp = _environment_enabled(
        environment,
        "CCCP_SHARED_MLP_TP",
        _environment_value(
            environment,
            "CCCP_DENSE_TP",
            "1",
        ),
    )
    if shared_tp and int(config.get("n_shared", 0)):
        require(
            "shared_intermediate",
            int(config["n_shared"]) * int(config["moe_inter"]),
            small_tp,
        )

    hidden_state = _environment_enabled(
        environment,
        "CCCP_TP_HIDDEN_STATE",
    )
    route_down = _environment_enabled(
        environment,
        "CCCP_MOE_ROUTE_DOWN_TP",
        "1" if hidden_state else "0",
    )
    moe_prelude = _environment_enabled(
        environment,
        "CCCP_MOE_PRELUDE_TP",
    )
    if route_down or moe_prelude:
        require("hidden", int(config["hidden"]), small_tp)
        require("n_experts", int(config["n_experts"]), small_tp)
    routed_projection = (
        hidden_state
        or _environment_enabled(environment, "CCCP_ROUTED_PROJECTION_TP")
    )
    if routed_projection:
        require(
            "routed_hidden",
            int(config["routed_hidden"]),
            small_tp,
        )

    # Down is Row-TP in packed block space.  A valid mathematical split must
    # also land on a whole byte for p9/p10/p12/p14 payloads.
    quant = manifest.get("quant", {})
    layouts = quant.get("layouts") or quant.get("projection_layouts", {})
    packing = quant.get("index_packing", {})
    routed_hidden = int(config["routed_hidden"])
    for name in sorted(_kimi_used_down_layouts(manifest)):
        layout = layouts.get(name)
        if not isinstance(layout, dict) or name not in packing:
            failures.append(f"down layout {name!r} 缺少布局或 packing 定义")
            continue
        dim = int(layout["dim"])
        bits = _packing_bits(packing[name])
        if routed_hidden % dim:
            failures.append(
                f"down[{name}] routed_hidden={routed_hidden} 不能整除 dim={dim}"
            )
            continue
        blocks = routed_hidden // dim
        if blocks % tp:
            failures.append(
                f"down[{name}] blocks={blocks} 不能整除 TP{tp}"
            )
            continue
        if (blocks // tp) * bits % 8:
            failures.append(
                f"down[{name}] TP{tp} 分片边界不是整字节"
            )

    if failures:
        raise ValueError(
            f"Kimi 公共真TP算子不支持当前 TP={tp}："
            + "；".join(dict.fromkeys(failures))
        )


def resolve_preset(
    model_dir: str | os.PathLike[str],
    *,
    profile: str = "auto",
    tp: int | None = None,
) -> ResolvedPreset:
    root, manifest = load_manifest(model_dir)
    architecture = detect_architecture(manifest)
    config_architecture = architecture
    if (
        architecture == "dsv4"
        and manifest.get("quant", {}).get("method")
        == "projection-vq"
    ):
        config_architecture = "dsv4_projection"
    config = load_arch_config(config_architecture)
    supports_parallel = bool(config.get("supports_parallel", False))

    if profile not in {"auto", "ram", "resident", "mapped", "parallel"}:
        raise ValueError(f"未知 profile：{profile}")
    if profile == "auto":
        profile = (
            "parallel"
            if supports_parallel and tp is not None and tp > 1
            else "ram"
        )
    if profile == "parallel" and not supports_parallel:
        raise ValueError(
            f"{config['display_name']} 当前没有多卡执行路径；请使用 --profile ram --tp 1"
        )
    profiles = config.get("profiles", {})
    if profile == "mapped" and profile not in profiles and "ram" in profiles:
        # ``mapped`` 是容量策略而不是模型家族。某些架构没有单独发布
        # GPU UVA 映射算子；此时退回公共 RAM 路径，CPU 仍由 mmap/系统
        # 虚拟内存完成磁盘兜底，GPU 则使用该架构已发布的主机驻留路径。
        # 这里按配置能力判断，不依赖模型名称或目录名称。
        profile = "ram"
    if profile not in profiles:
        raise ValueError(f"{architecture} 没有 profile={profile!r}")

    selected = profiles[profile]
    resolved_tp = int(selected.get("tp", 1) if tp is None else tp)
    if resolved_tp <= 0:
        raise ValueError("tp 必须为正整数")
    if profile == "ram" and resolved_tp != 1:
        raise ValueError("RAM profile 固定使用 tp=1；多卡请选择 --profile parallel")
    if profile == "resident" and resolved_tp != 1:
        raise ValueError("resident profile 固定使用 tp=1")
    if profile == "mapped" and resolved_tp != 1:
        raise ValueError("mapped profile requires tp=1")
    if profile == "parallel" and resolved_tp < 2:
        raise ValueError("parallel profile 要求 tp >= 2")
    tested_tp_values = config.get("tested_tp_values")
    if (
        profile == "parallel"
        and tested_tp_values is not None
        and resolved_tp not in {
            int(value) for value in tested_tp_values
        }
    ):
        allowed = ",".join(str(value) for value in tested_tp_values)
        raise ValueError(
            f"{architecture} 当前只发布 TP={allowed}，"
            f"收到 tp={resolved_tp}"
        )

    config_profile = profile
    if profile == "parallel":
        tp_profile = f"parallel_tp{resolved_tp}"
        if tp_profile in profiles:
            selected = profiles[tp_profile]
            configured_tp = int(selected.get("tp", resolved_tp))
            if configured_tp != resolved_tp:
                raise ValueError(
                    f"{architecture} profile={tp_profile!r} has tp="
                    f"{configured_tp}, requested tp={resolved_tp}"
                )
            config_profile = tp_profile

    environment = {
        str(key): str(value)
        for key, value in config.get("environment", {}).items()
    }
    environment.update(
        {
            str(key): str(value)
            for key, value in selected.get("environment", {}).items()
        }
    )
    quant_method = str(manifest.get("quant", {}).get("method", "")).lower()
    routed_vq = bool(manifest.get("expert_files")) and "vq" in quant_method
    if profile == "resident" and routed_vq:
        # ``resident`` is a capacity contract shared by every routed-codebook
        # model: packed expert bytes live in the public all-GPU backend and
        # runtime H2D/LRU is absent.  Keeping this switch in architecture JSON
        # made two identical projection-VQ manifests select different pools.
        environment.setdefault("CCCP_PACKED_FULL_GPU", "1")
    ep_layout = (
        choose_ep_layout(manifest, resolved_tp)
        if profile == "parallel"
        else None
    )
    if ep_layout is not None:
        validate_parallel_shapes(
            manifest,
            resolved_tp,
            environment,
            ep_layout,
        )

    return ResolvedPreset(
        model_dir=root,
        manifest=manifest,
        architecture=architecture,
        display_name=str(config["display_name"]),
        profile=profile,
        config_profile=config_profile,
        tp=resolved_tp,
        ep_layout=ep_layout,
        defaults=dict(config.get("defaults", {})),
        environment=environment,
        supports_parallel=supports_parallel,
    )


def resolve_capacity_profile(
    preset: ResolvedPreset,
    capacity_mode: str,
) -> ResolvedPreset:
    """Select the fastest published single-GPU profile for a capacity mode.

    Capacity detection deliberately remains model agnostic.  An architecture
    opts in by publishing ``resident`` or ``mapped`` in its configuration;
    unsupported profiles fall back to the already resolved safe preset.
    """

    if preset.tp != 1:
        return preset
    preferred = {
        "resident": "resident",
        "ram": "mapped",
    }.get(str(capacity_mode))
    if preferred is None or preset.profile == preferred:
        return preset
    try:
        return resolve_preset(
            preset.model_dir,
            profile=preferred,
            tp=1,
        )
    except ValueError:
        return preset


def apply_preset_environment(
    preset: ResolvedPreset,
) -> dict[str, str]:
    """Apply a resolved profile without overriding explicit user choices."""
    effective: dict[str, str] = {}
    for key, value in preset.environment.items():
        effective[key] = os.environ.setdefault(key, value)

    if preset.ep_layout is not None:
        configured_layout = os.environ.get("CCCP_EP_LAYOUT")
        if (
            configured_layout == "tensor" and
            preset.ep_layout != "tensor"
        ):
            raise ValueError(
                f"tp={preset.tp} 不能整除该模型专家中间维；"
                "请取消 CCCP_EP_LAYOUT=tensor 或改用 expert"
            )
        effective["CCCP_EP_LAYOUT"] = os.environ.setdefault(
            "CCCP_EP_LAYOUT",
            preset.ep_layout,
        )
    return effective


__all__ = [
    "ResolvedPreset",
    "apply_preset_environment",
    "choose_ep_layout",
    "detect_architecture",
    "load_arch_config",
    "load_manifest",
    "model_context_limit",
    "resolve_capacity_profile",
    "resolve_preset",
    "validate_parallel_shapes",
]

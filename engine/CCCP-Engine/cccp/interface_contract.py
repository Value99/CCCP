"""Stable launcher-facing CCCP model metadata contracts.

The helpers in this module are deliberately read-only and do not import
``torch``.  They are shared by ``cccp check`` and the OpenAI HTTP server so a
launcher receives exactly the same model specification and expert byte table
before and after model loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_manifest(model_root: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(model_root).expanduser().resolve()
    with (root / "cccp.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
        raise ValueError("cccp.json 缺少 config 对象")
    return root, manifest


def _audit_path(
    root: Path,
    manifest: dict[str, Any],
    layer: int,
) -> Path | None:
    declared = manifest.get("expert_audit_files") or {}
    name = declared.get(str(layer), declared.get(layer))
    routed_item = (
        (manifest.get("routed_experts") or {}).get("layer_files") or {}
    ).get(str(layer))
    if routed_item is None:
        routed_item = (
            (manifest.get("routed_experts") or {}).get("layer_files") or {}
        ).get(layer)
    if not name and isinstance(routed_item, dict):
        name = routed_item.get("audit_path")
    candidates = []
    if name:
        candidates.append(root / str(name))
    candidates.extend(
        (
            root / f"experts.L{layer:02d}.audit.json",
            root / f"experts.L{layer:03d}.audit.json",
            root / f"experts.L{layer}.audit.json",
        )
    )
    return next((path for path in candidates if path.is_file()), None)


def _expert_files(manifest: dict[str, Any]) -> dict[int, str]:
    flat = manifest.get("expert_files") or {}
    if isinstance(flat, dict) and flat:
        return {int(layer): str(name) for layer, name in flat.items()}
    routed = (manifest.get("routed_experts") or {}).get("layer_files") or {}
    if not isinstance(routed, dict):
        return {}
    result: dict[int, str] = {}
    for layer, item in routed.items():
        name = item.get("path") if isinstance(item, dict) else item
        if name:
            result[int(layer)] = str(name)
    return result


def _packed_bytes(detail: object) -> int:
    if not isinstance(detail, dict):
        return 0
    if "gu_bytes" in detail or "down_bytes" in detail:
        return int(detail.get("gu_bytes") or 0) + int(detail.get("down_bytes") or 0)
    return sum(
        int(value.get("packed_bytes") or 0)
        for value in detail.values()
        if isinstance(value, dict)
    )


def _scaled_layer_bytes(
    raw: dict[int, int],
    file_bytes: int,
) -> dict[int, int]:
    """Assign shard metadata/codebook bytes while preserving its exact size."""
    raw_total = sum(raw.values())
    if not raw or raw_total <= 0:
        return {}
    scaled = {
        expert: (value * file_bytes) // raw_total
        for expert, value in raw.items()
    }
    remainder = file_bytes - sum(scaled.values())
    # Deterministic largest-remainder distribution.  The result sums to the
    # actual shard size, unlike a rounded GiB estimate.
    order = sorted(
        raw,
        key=lambda expert: (
            -((raw[expert] * file_bytes) % raw_total),
            expert,
        ),
    )
    for expert in order[:remainder]:
        scaled[expert] += 1
    return scaled


def expert_bytes_payload(model_root: str | Path) -> dict[str, Any]:
    """Return exact per-expert stored bytes from CCCP layer audits.

    ``calibrated`` is false when any routed layer lacks an audit.  In that
    case available layers are still returned, but callers must not label the
    partial table as an exact whole-model size.
    """
    root, manifest = _load_manifest(model_root)
    expert_files = _expert_files(manifest)
    if not expert_files:
        raise ValueError("cccp.json 缺少动态专家层文件")

    result: dict[str, int] = {}
    missing_layers: list[int] = []
    layer_expert_counts: dict[str, int] = {}
    for raw_layer, shard_name in sorted(
        expert_files.items(), key=lambda item: int(item[0])
    ):
        layer = int(raw_layer)
        audit_path = _audit_path(root, manifest, layer)
        if audit_path is None:
            missing_layers.append(layer)
            continue
        with audit_path.open("r", encoding="utf-8") as handle:
            audit = json.load(handle)
        experts = audit.get("experts") or {}
        raw = {
            int(expert): _packed_bytes(detail)
            for expert, detail in experts.items()
        }
        raw = {expert: value for expert, value in raw.items() if value > 0}
        shard = root / str(shard_name)
        file_bytes = int(audit.get("file_bytes") or 0)
        if shard.is_file():
            file_bytes = shard.stat().st_size
        if file_bytes <= 0:
            file_bytes = sum(raw.values())
        scaled = _scaled_layer_bytes(raw, file_bytes)
        layer_expert_counts[str(layer)] = len(scaled)
        result.update(
            {f"{layer}:{expert}": value for expert, value in scaled.items()}
        )

    return {
        "schema": "cccp-expert-bytes-v1",
        "model": root.name,
        "layers": len(expert_files),
        "experts_per_layer": int(manifest["config"].get("n_experts") or 0),
        "layer_expert_counts": layer_expert_counts,
        "calibrated": not missing_layers,
        "missing_audit_layers": missing_layers,
        "total_bytes": sum(result.values()),
        "bytes": result,
    }


def model_spec_payload(model_root: str | Path) -> dict[str, Any]:
    """Return the machine-readable heterogeneous MoE model specification."""
    root, manifest = _load_manifest(model_root)
    config = manifest["config"]
    expert_files = _expert_files(manifest)
    byte_table = expert_bytes_payload(root)
    layer_counts = dict(byte_table["layer_expert_counts"])
    # A valid homogeneous archive may omit audit files.  Its declared expert
    # count remains useful as a specification, while calibrated stays false.
    declared = int(config.get("n_experts") or 0)
    for layer in expert_files:
        layer_counts.setdefault(str(int(layer)), declared)
    return {
        "schema": "cccp-model-spec-v1",
        "model": root.name,
        "model_format": str(manifest.get("format") or "cccp"),
        "model_version": str(
            manifest.get("version")
            or config.get("model_version")
            or root.name
        ),
        "architecture": str(
            manifest.get("architecture")
            or manifest.get("model_family")
            or config.get("model_family")
            or config.get("arch")
            or "cccp"
        ),
        "layers": int(config.get("n_layers") or len(expert_files)),
        "routed_layers": sorted(int(layer) for layer in expert_files),
        "experts_per_layer": declared,
        "layer_expert_counts": layer_counts,
        "top_k": int(config.get("top_k") or 0),
        "max_context": int(config.get("max_position_embeddings") or 0),
        "expert_bytes_calibrated": bool(byte_table["calibrated"]),
        "expert_bytes_total": int(byte_table["total_bytes"]),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically persist one contract document."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


__all__ = ["expert_bytes_payload", "model_spec_payload", "write_json"]

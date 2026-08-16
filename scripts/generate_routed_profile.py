"""把 CCCP 实测路由计数转换成可分享、可全量常驻的动态专家配置。

不生成、复制或修改任何模型权重。配置只包含专家坐标、精确体积和聚合路由计数。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expert_sizes(model: Path) -> tuple[dict[str, float], dict[str, int | float]]:
    """从模型 audit 读取每个专家的精确打包体积。

    这个函数保留在真实路由生成器内，避免依赖已弃用的启发式配置生成脚本。
    audit 中三投影的 packed_bytes 会按所在专家分片的实际文件体积等比例校正，
    因而同一模型/同一专家在任意配置中得到完全相同的 size_mb。
    """
    sizes: dict[str, float] = {}
    layers = 0
    exact_bytes = 0
    for audit_path in sorted(model.glob("experts.L*.audit.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        layer = int(audit["layer"])
        experts = audit.get("experts") or {}
        raw: dict[int, int] = {}
        for expert_id, detail in experts.items():
            raw[int(expert_id)] = sum(
                int((detail.get(projection) or {}).get("packed_bytes") or 0)
                for projection in ("gate", "up", "down")
            )
        raw_total = sum(raw.values())
        file_bytes = int(audit.get("file_bytes") or raw_total)
        scale = file_bytes / raw_total if raw_total else 1.0
        for expert_id, byte_count in raw.items():
            sizes[f"{layer}:{expert_id}"] = round(byte_count * scale / 2**20, 6)
        exact_bytes += file_bytes
        layers = max(layers, layer + 1)
    if not sizes:
        raise RuntimeError("模型目录没有可用的 experts.L*.audit.json")
    experts_per_layer = max(int(key.split(":")[1]) for key in sizes) + 1
    return sizes, {
        "layers": layers,
        "experts_per_layer": experts_per_layer,
        "expert_file_bytes": exact_bytes,
        "expert_file_gib": round(exact_bytes / 2**30, 6),
    }


def select_with_budget(
    counts: dict[str, int], sizes: dict[str, float], *, max_gb: float, top_k: int,
) -> list[str]:
    nonzero = {key: count for key, count in counts.items() if count > 0 and key in sizes}
    by_layer: dict[int, list[str]] = defaultdict(list)
    for key in nonzero:
        by_layer[int(key.split(":", 1)[0])].append(key)
    expected_layers = sorted({int(key.split(":", 1)[0]) for key in sizes})
    missing = [layer for layer in expected_layers if len(by_layer[layer]) < top_k]
    if missing:
        raise RuntimeError(f"实测路由中以下层少于 top_k={top_k}: {missing}")

    budget_mb = max_gb * 1024
    selected: set[str] = set()
    # 每层最常用的 top-k 是保持模型路由定义有效的硬下限。
    for layer in expected_layers:
        ranked = sorted(by_layer[layer], key=lambda k: (-nonzero[k], sizes[k], k))
        selected.update(ranked[:top_k])
    used = sum(sizes[key] for key in selected)
    if used > budget_mb:
        raise RuntimeError(f"仅逐层 top-k 已需 {used / 1024:.3f} GiB，超过预算")

    remaining = sorted(
        (key for key in nonzero if key not in selected),
        key=lambda key: (-nonzero[key] / sizes[key], -nonzero[key], key),
    )
    for key in remaining:
        if used + sizes[key] <= budget_mb:
            selected.add(key)
            used += sizes[key]
    return sorted(selected, key=lambda key: tuple(map(int, key.split(":"))))


def dense_budget_breakdown(model: Path) -> dict[str, float | int | str]:
    """从模型本身读取 Dense/共享专家字节，避免任何架构或领域硬编码。"""
    manifest = json.loads((model / "cccp.json").read_text(encoding="utf-8"))
    dense_name = str(manifest.get("dense_file") or "dense.safetensors")
    dense_path = model / dense_name
    with dense_path.open("rb") as handle:
        header_bytes = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_bytes))
    shared_bytes = sum(
        int(value["data_offsets"][1]) - int(value["data_offsets"][0])
        for key, value in header.items()
        if key != "__metadata__" and "shared_experts" in key
    )
    dense_file_bytes = dense_path.stat().st_size
    return {
        "dense_file": dense_name,
        "fixed_model_bytes": dense_file_bytes,
        "fixed_model_gib": dense_file_bytes / 2**30,
        "shared_expert_bytes": shared_bytes,
        "shared_expert_gib": shared_bytes / 2**30,
        "dense_without_shared_bytes": dense_file_bytes - shared_bytes,
        "dense_without_shared_gib": (dense_file_bytes - shared_bytes) / 2**30,
        "model_format": str(manifest.get("format") or "unknown"),
        "model_architecture": str(manifest.get("architecture") or "unknown"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routes", type=Path, nargs="+", required=True,
        help="一个或多个同领域 CCCP 实测路由文件；命中次数按专家累加",
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--calibration-audit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id", default="roleplay-romance")
    parser.add_argument("--name", default="爱情类角色扮演")
    parser.add_argument("--description", default="基于本地角色扮演语料的真实 CCCP 路由命中生成，侧重爱情、关系与情绪互动。")
    parser.add_argument(
        "--tags", default="roleplay,romance,love,relationship,emotion,cn",
        help="逗号分隔的配置/专家标签",
    )
    parser.add_argument("--max-gb", type=float, default=12.0)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    if not 0.25 <= args.max_gb <= 24.0:
        raise SystemExit("max-gb 必须在 0.25..24.0")

    counts: dict[str, int] = {}
    for route_path in args.routes:
        route_doc = json.loads(route_path.read_text(encoding="utf-8"))
        if route_doc.get("format") != "cccp-expert-residency-scores-v1":
            raise RuntimeError(f"路由文件格式不受支持: {route_path}")
        for key, value in route_doc["scores"].items():
            counts[str(key)] = counts.get(str(key), 0) + int(float(value))
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    audit = json.loads(args.calibration_audit.read_text(encoding="utf-8"))
    category = audit["categories"][args.id]
    sizes, model_stats = load_expert_sizes(args.model)
    fixed = dense_budget_breakdown(args.model)
    routed_budget_gib = args.max_gb - float(fixed["fixed_model_gib"])
    if routed_budget_gib <= 0:
        raise RuntimeError(
            f"配置总预算 {args.max_gb:.3f} GiB 小于固定 Dense+共享专家 "
            f"{float(fixed['fixed_model_gib']):.3f} GiB"
        )
    selected = select_with_budget(
        counts, sizes, max_gb=routed_budget_gib, top_k=args.top_k,
    )
    maximum = max(counts[key] for key in selected)
    selected_mb = sum(sizes[key] for key in selected)
    selected_observations = sum(counts[key] for key in selected)
    all_observations = sum(max(0, count) for count in counts.values())
    layers = defaultdict(int)
    for key in selected:
        layers[int(key.split(":", 1)[0])] += 1

    tags = [item.strip() for item in args.tags.split(",") if item.strip()]
    if not tags:
        raise RuntimeError("至少需要一个标签")
    manifest = args.model / "cccp.json"
    model_total_bytes = sum(
        path.stat().st_size for path in args.model.iterdir() if path.is_file()
    )
    profile = {
        "schema": "winui-expert-profile-v1",
        "id": args.id,
        "name": args.name,
        "description": args.description + " 仅生成配置，不生成或改动模型权重。",
        "tags": tags,
        "experts": [
            {
                "key": key,
                "size_mb": sizes[key],
                "tags": tags,
                "route_count": counts[key],
                "route_score": round(counts[key] / maximum, 8),
            }
            for key in selected
        ],
        "drop": {"enabled": True, "hint_tags": tags},
        "meta": {
            "source": "trained",
            "calibrated": True,
            "configuration_only": True,
            "strict_route": True,
            "load_all_selected_experts": True,
            "routing_source": "cccp-cpu-measured-route-counts",
            "routing_input_files": [path.name for path in args.routes],
            "routing_input_count": len(args.routes),
            "size_source": "experts.L*.audit.json",
            "corpus_name": audit["corpus_name"],
            "corpus_sha256": audit["corpus_sha256"],
            "sample_seed": audit["seed"],
            "sample_characters": category["characters"],
            "prompt_sha256": category["prompt_sha256"],
            "prompt_redacted": True,
            "model_name": args.model.name,
            # 可读版本用于界面/分享识别；SHA-256 指纹用于严格兼容性判断。
            "model_version": args.model.name,
            "model_format": fixed["model_format"],
            "model_format_version": fixed["model_format"],
            "model_architecture": fixed["model_architecture"],
            "model_manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
            "model_total_bytes": model_total_bytes,
            "model_total_gib": round(model_total_bytes / 2**30, 6),
            "model_layers": model_stats["layers"],
            "model_experts_per_layer": model_stats["experts_per_layer"],
            "model_top_k": args.top_k,
            "route_observations": all_observations,
            "selected_route_observations": selected_observations,
            "route_coverage": round(selected_observations / max(1, all_observations), 8),
            "selected_experts": len(selected),
            "resident_expert_gib": round(selected_mb / 1024, 6),
            "configuration_budget_gib": args.max_gb,
            "fixed_model_gib": round(float(fixed["fixed_model_gib"]), 6),
            "dense_without_shared_gib": round(float(fixed["dense_without_shared_gib"]), 6),
            "shared_expert_gib": round(float(fixed["shared_expert_gib"]), 6),
            "routed_expert_budget_gib": round(routed_budget_gib, 6),
            "configuration_resident_gib": round(
                float(fixed["fixed_model_gib"]) + selected_mb / 1024, 6
            ),
            "min_experts_per_layer": min(layers.values()),
            "max_experts_per_layer": max(layers.values()),
            "calibration_decode_tok_s": benchmark.get("throughput_tok_s_median"),
            "calibration_cpu_compile": benchmark.get("environment", {}).get("CCCP_CPU_COMPILE"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "experts": len(selected),
        "resident_expert_gib": round(selected_mb / 1024, 6),
        "configuration_budget_gib": args.max_gb,
        "fixed_model_gib": round(float(fixed["fixed_model_gib"]), 6),
        "configuration_resident_gib": profile["meta"]["configuration_resident_gib"],
        "route_coverage": profile["meta"]["route_coverage"],
        "layers": len(layers),
        "min_experts_per_layer": min(layers.values()),
        "max_experts_per_layer": max(layers.values()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

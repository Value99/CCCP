"""从 cccp.json 生成统一执行配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOperatorConfig:
    model_family: str
    attention_kind: str
    expert_activation: str
    top_k: int
    scoring_func: str
    normalize_route: bool
    routed_scaling: float

    @classmethod
    def from_manifest(cls, manifest: dict) -> "ModelOperatorConfig":
        values = manifest["config"]
        family = str(manifest.get("model_family", "glm")).lower()
        if "kda_layers" in values:
            attention_kind = "hybrid_kda_mla"
        elif "hc_mult" in values or "compress_ratios" in values:
            attention_kind = "hybrid_window_compressed"
        else:
            attention_kind = "mla"
        return cls(
            model_family=family,
            attention_kind=attention_kind,
            expert_activation=str(
                values.get("activation", "swiglu")
            ).lower(),
            top_k=int(values["top_k"]),
            scoring_func=str(
                values.get("scoring_func", "sigmoid")
            ).lower(),
            normalize_route=bool(
                values.get("norm_topk_prob", True)
            ),
            routed_scaling=float(
                values.get("routed_scaling", 1.0)
            ),
        )


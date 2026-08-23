"""Architecture-neutral speculative draft acceptance policy."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import torch


@dataclass(frozen=True)
class DraftAcceptancePolicy:
    """Bounded fast policy shared by every CCCP MTP implementation.

    A draft is accepted only while it remains in the main model's Top-N set.
    Acceptance always stops at the first miss so the committed sequence and
    every architecture's KV/recurrent state have one unambiguous prefix.
    """

    top_n: int = 3
    max_draft: int = 5

    def __post_init__(self) -> None:
        if not 1 <= int(self.top_n) <= 3:
            raise ValueError("MTP acceptance top_n must be between 1 and 3")
        if not 1 <= int(self.max_draft) <= 5:
            raise ValueError("MTP max_draft must be between 1 and 5")

    @property
    def mode(self) -> str:
        return "strict-top1" if self.top_n == 1 else f"fast-top{self.top_n}"

    def draft_count(self, requested: int, room: int | None = None) -> int:
        count = min(max(0, int(requested)), int(self.max_draft))
        if room is not None:
            count = min(count, max(0, int(room)))
        return count

    def accepts(self, logits: torch.Tensor, draft: int) -> bool:
        row = logits.reshape(-1)
        if row.numel() == 0 or not torch.isfinite(row).any():
            return False
        width = min(int(self.top_n), int(row.numel()))
        candidates = torch.topk(row, k=width, dim=-1).indices
        return bool((candidates == int(draft)).any().item())

    def accepted_prefix(
        self,
        logits: torch.Tensor,
        drafts: list[int],
    ) -> int:
        if logits.ndim != 2 or logits.shape[0] < len(drafts):
            raise ValueError("main verification logits do not cover all drafts")
        accepted = 0
        for index, draft in enumerate(drafts):
            if not self.accepts(logits[index], draft):
                break
            accepted += 1
        return accepted

    def accepted_prefix_batched(
        self,
        logits: torch.Tensor,
        drafts: list[int],
    ) -> int:
        """qwen3.5 MTP 专用批量判定:一次 topk/一次 isfinite/一次同步。

        判定语义与 accepted_prefix() 逐位一致(行内 top-n 含 draft 且
        行有限,首个未中即停);GLM/DSV4/Kimi 路径继续使用原版
        accepted_prefix——通用策略的行为对它们完全恢复(第三十二轮)。
        """
        total = len(drafts)
        if total == 0:
            return 0
        if logits.ndim != 2 or logits.shape[0] < total:
            raise ValueError("main verification logits do not cover all drafts")
        rows = logits[:total]
        width = min(int(self.top_n), int(rows.shape[-1]))
        top = torch.topk(rows, k=width, dim=-1).indices
        draft_tensor = torch.as_tensor(
            drafts, device=rows.device, dtype=torch.long
        ).unsqueeze(1)
        hits = (top == draft_tensor).any(dim=-1)
        finite_rows = torch.isfinite(rows).all(dim=-1)
        hits = hits & finite_rows
        leading = hits.to(torch.int32).cumprod(dim=0).sum()
        return int(leading.item())


FAST_MTP_POLICY = DraftAcceptancePolicy(top_n=3, max_draft=5)
STRICT_MTP_POLICY = DraftAcceptancePolicy(top_n=1, max_draft=5)


@dataclass(frozen=True)
class SpeculativeProviderSpec:
    provider: str
    policy: DraftAcceptancePolicy
    attachment_kind: str
    attachment_field: str | None = None
    residency: tuple[tuple[str, str | bool], ...] = ()
    execution: tuple[tuple[str, str], ...] = ()

    def residency_value(self, key: str, default=None):
        return dict(self.residency).get(str(key), default)

    def execution_value(self, key: str, default=None):
        return dict(self.execution).get(str(key), default)


_KNOWN_PROVIDERS = frozenset({
    "qwen35_mtp",
    "dsv4_dspark",
    "glm_mtp",
    "kimi_prompt_lookup",
})


@lru_cache(maxsize=None)
def provider_for_architecture(architecture: str) -> SpeculativeProviderSpec:
    """Load the public draft provider from the architecture config."""
    from .presets import load_arch_config

    raw = load_arch_config(str(architecture)).get("speculative")
    if not isinstance(raw, dict):
        raise ValueError(
            f"architecture {architecture!r} has no speculative provider"
        )
    provider = str(raw.get("provider") or "")
    if provider not in _KNOWN_PROVIDERS:
        raise ValueError(
            f"architecture {architecture!r} declares unknown speculative "
            f"provider {provider!r}"
        )
    attachment = raw.get("attachment") or {}
    residency = raw.get("residency") or {}
    execution = raw.get("execution") or {}
    kind = str(attachment.get("kind") or "builtin")
    field = attachment.get("field")
    if kind not in {"builtin", "manifest-file", "config-positive"}:
        raise ValueError(
            f"unsupported speculative attachment kind {kind!r}"
        )
    if kind != "builtin" and not field:
        raise ValueError("speculative attachment field is required")
    if not isinstance(residency, dict) or not isinstance(execution, dict):
        raise ValueError("speculative residency/execution must be mappings")
    return SpeculativeProviderSpec(
        provider=provider,
        policy=DraftAcceptancePolicy(
            top_n=int(raw.get("accept_top_n") or 3),
            max_draft=int(raw.get("max_draft") or 5),
        ),
        attachment_kind=kind,
        attachment_field=str(field) if field else None,
        residency=tuple(
            (str(key), value)
            for key, value in sorted(residency.items())
            if isinstance(value, (str, bool))
        ),
        execution=tuple(
            (str(key), str(value))
            for key, value in sorted(execution.items())
        ),
    )


def _model_manifest(model) -> dict:
    archive = getattr(model, "archive", None)
    manifest = getattr(archive, "manifest", None)
    if isinstance(manifest, dict):
        return manifest
    root = getattr(getattr(model, "store", None), "root", None)
    if root is None:
        return {}
    try:
        return json.loads(
            (Path(root) / "cccp.json").read_text(encoding="utf-8")
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def provider_attachment_available(
    spec: SpeculativeProviderSpec,
    model,
) -> bool:
    """Validate an attachment from the selected model's own manifest."""
    manifest = _model_manifest(model)
    root = getattr(getattr(model, "store", None), "root", None)
    if root is None:
        root = getattr(getattr(model, "archive", None), "root", None)
    return provider_attachment_available_in_manifest(spec, manifest, root)


def provider_attachment_available_in_manifest(
    spec: SpeculativeProviderSpec,
    manifest: dict,
    model_root: str | Path | None,
) -> bool:
    """Resolve automatic drafting before the heavyweight model is loaded.

    Launch defaults must never enable an architecture's MTP path merely
    because that architecture can support one.  The selected model directory
    must contain the attachment declared by its public architecture config.
    """
    if spec.attachment_kind == "builtin":
        return True
    field = str(spec.attachment_field)
    if spec.attachment_kind == "config-positive":
        return int((manifest.get("config") or {}).get(field) or 0) > 0
    filename = manifest.get(field)
    root = Path(model_root) if model_root is not None else None
    return bool(filename and root and (Path(root) / str(filename)).is_file())


__all__ = [
    "DraftAcceptancePolicy",
    "FAST_MTP_POLICY",
    "SpeculativeProviderSpec",
    "STRICT_MTP_POLICY",
    "provider_attachment_available",
    "provider_attachment_available_in_manifest",
    "provider_for_architecture",
]

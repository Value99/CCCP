"""Lazy FlashMLA SplitKV adapter used by the public operator backend."""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import torch


def _pad_prefill_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pad sparse indices to the SM90 kernel's 128-entry tile width."""

    width = int(indices.shape[-1])
    padded_width = ((width + 127) // 128) * 128
    if padded_width == width:
        return indices
    padding = torch.full(
        (*indices.shape[:-1], padded_width - width),
        -1,
        dtype=indices.dtype,
        device=indices.device,
    )
    return torch.cat((indices, padding), dim=-1).contiguous()


def _import_flash_mla():
    root = os.environ.get("CCCP_FLASHMLA_ROOT", "").strip()
    if root and root not in sys.path:
        sys.path.insert(0, root)
    import flash_mla  # type: ignore

    return flash_mla


def available(device=None) -> tuple[bool, str | None]:
    if not torch.cuda.is_available():
        return False, "CUDA unavailable"
    target = torch.device("cuda" if device is None else device)
    major, minor = torch.cuda.get_device_capability(target)
    if (major, minor) not in ((9, 0), (10, 0)):
        return False, f"FlashMLA sparse SplitKV unsupported on sm{major}{minor}"
    try:
        _import_flash_mla()
    except Exception as exc:  # optional accelerator
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


@dataclass
class FlashMLASparseRunner:
    """One fixed-shape scheduler metadata object per LayerGraph bucket."""

    metadata: object

    @classmethod
    def create(cls) -> "FlashMLASparseRunner":
        flash_mla = _import_flash_mla()
        metadata, _ = flash_mla.get_mla_metadata()
        return cls(metadata)

    def decode(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        indices: torch.Tensor,
        sink: torch.Tensor | None,
        scale: float,
        extra_key_cache: torch.Tensor | None,
        extra_indices: torch.Tensor | None,
        topk_length: torch.Tensor | None,
        extra_topk_length: torch.Tensor | None,
    ) -> torch.Tensor:
        flash_mla = _import_flash_mla()
        output, _ = flash_mla.flash_mla_with_kvcache(
            query,
            key_cache,
            None,
            None,
            512,
            self.metadata,
            None,
            float(scale),
            False,
            True,
            indices,
            sink,
            extra_key_cache,
            extra_indices,
            topk_length,
            extra_topk_length,
        )
        return output

    def prefill(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        indices: torch.Tensor,
        sink: torch.Tensor | None,
        scale: float,
    ) -> torch.Tensor:
        """Run the native SM90/SM100 sparse-Prefill kernel."""

        flash_mla = _import_flash_mla()
        indices = _pad_prefill_indices(indices)
        output, _maximum, _lse = flash_mla.flash_mla_sparse_fwd(
            query,
            key_cache,
            indices,
            float(scale),
            512,
            sink,
            None,
        )
        return output


__all__ = ["FlashMLASparseRunner", "available"]

"""Platform policy for the public CUDA latent-attention backend."""

from __future__ import annotations

import sys


def select_cuda_mla_backend(
    *,
    flashinfer_ready: bool,
    platform_name: str | None = None,
) -> str:
    """Choose one tested CUDA MLA implementation without a BF16 fallback.

    Linux prefers FlashInfer when it is usable.  Native Windows always uses
    the CCCP-bundled paged latent-attention operator, so a missing optional
    FlashInfer wheel cannot silently select the portable PyTorch reference.
    """
    platform = str(platform_name or sys.platform).lower()
    if platform.startswith("win"):
        return "cccp-paged"
    return "flashinfer" if flashinfer_ready else "cccp-paged"


__all__ = ["select_cuda_mla_backend"]

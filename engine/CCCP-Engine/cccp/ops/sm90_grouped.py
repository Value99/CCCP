"""Packaged SM90 grouped FP8 execution helpers.

DeepGEMM is used only on Hopper when its wheel is present in CCCP's bundled
environment.  Other supported architectures retain PyTorch's native scaled
grouped primitive; importing CCCP therefore never makes DeepGEMM a global
dependency for CPU, AMD or consumer NVIDIA builds.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import torch


def _deepgemm_available() -> bool:
    try:
        return importlib.util.find_spec("deep_gemm") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def select_grouped_fp8_backend(
    capability: tuple[int, int],
    *,
    deepgemm_available: bool | None = None,
) -> str | None:
    """Select the fastest packaged exact grouped FP8 executor for an SM."""

    major, minor = int(capability[0]), int(capability[1])
    has_deepgemm = (
        _deepgemm_available()
        if deepgemm_available is None
        else bool(deepgemm_available)
    )
    if (major, minor) == (9, 0) and has_deepgemm:
        return "deepgemm-sm90"
    if major in (9, 10):
        return "torch-scaled-grouped-mm"
    return None


def row_block_scales(
    source: torch.Tensor,
    *,
    k: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Replicate CCCP's exact per-row FP8 scale into DeepGEMM K blocks."""

    if source.ndim != 2 or int(source.shape[1]) != 1:
        raise ValueError("row scales must have shape [rows, 1]")
    if int(k) <= 0 or int(k) % 128:
        raise ValueError("DeepGEMM SM90 K must be a positive multiple of 128")
    expected = (int(source.shape[0]), int(k) // 128)
    if tuple(output.shape) != expected:
        raise ValueError(
            f"row block-scale output must have shape {expected}, got "
            f"{tuple(output.shape)}"
        )
    output.copy_(source.expand(expected))
    return output


def projection_block_scales(
    source: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
    gate_up_output: torch.Tensor,
    down_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build DeepGEMM 128x128 scale fields from VQ projection scales."""

    if source.ndim != 2 or int(source.shape[1]) != 3:
        raise ValueError("projection scales must have shape [experts, 3]")
    if int(hidden) % 128 or int(intermediate) % 128:
        raise ValueError("DeepGEMM SM90 dimensions must be multiples of 128")
    experts = int(source.shape[0])
    hidden_blocks = int(hidden) // 128
    intermediate_blocks = int(intermediate) // 128
    gate_up_shape = (experts, 2 * intermediate_blocks, hidden_blocks)
    down_shape = (experts, hidden_blocks, intermediate_blocks)
    if tuple(gate_up_output.shape) != gate_up_shape:
        raise ValueError(
            f"gate/up block scales must have shape {gate_up_shape}, got "
            f"{tuple(gate_up_output.shape)}"
        )
    if tuple(down_output.shape) != down_shape:
        raise ValueError(
            f"down block scales must have shape {down_shape}, got "
            f"{tuple(down_output.shape)}"
        )
    gate_up_output[:, :intermediate_blocks].copy_(
        source[:, 0, None, None].expand(
            experts, intermediate_blocks, hidden_blocks
        )
    )
    gate_up_output[:, intermediate_blocks:].copy_(
        source[:, 1, None, None].expand(
            experts, intermediate_blocks, hidden_blocks
        )
    )
    down_output.copy_(
        source[:, 2, None, None].expand(
            experts, hidden_blocks, intermediate_blocks
        )
    )
    return gate_up_output, down_output


def execute_deepgemm_grouped_fp8(
    value: torch.Tensor,
    weights: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    group_ids: torch.Tensor,
    output: torch.Tensor,
    module: ModuleType | Any | None = None,
) -> torch.Tensor:
    """Execute one preallocated m-grouped E4M3 GEMM through DeepGEMM."""

    if module is None:
        import deep_gemm as module  # type: ignore[no-redef]

    layout = (
        group_ids
        if group_ids.dtype == torch.int32 and group_ids.is_contiguous()
        else group_ids.to(dtype=torch.int32).contiguous()
    )
    module.m_grouped_fp8_gemm_nt_contiguous(
        (value, scale_a),
        (weights, scale_b),
        output,
        layout,
        disable_ue8m0_cast=True,
    )
    return output

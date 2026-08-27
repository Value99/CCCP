"""Packaged SM90 grouped FP8 execution helpers.

DeepGEMM is used only on Hopper when its wheel is present in CCCP's bundled
environment.  Other supported architectures retain PyTorch's native scaled
grouped primitive; importing CCCP therefore never makes DeepGEMM a global
dependency for CPU, AMD or consumer NVIDIA builds.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch


@dataclass(frozen=True)
class DeepGEMMGroupedLayout:
    """Padded row directory required by DeepGEMM M-grouped kernels."""

    row_positions: torch.Tensor
    group_ids: torch.Tensor
    padded_rows: int


@dataclass(frozen=True)
class DeepGEMMGroupedWorkspace:
    """Reusable padded activation, scale and result buffers."""

    value: torch.Tensor
    scale_a: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True)
class ActiveGroupedRoutes:
    """Model-independent remapping from global experts to active groups."""

    order: torch.Tensor
    sorted_group_ids: torch.Tensor
    unique_group_ids: torch.Tensor
    active_group_count: int
    execution_group_count: int


def deepgemm_grouped_alignment(
    module: ModuleType | Any | None = None,
) -> int:
    """Read the packaged backend's required M-group alignment."""

    if module is None:
        import deep_gemm as module  # type: ignore[no-redef]
    alignment = int(module.get_mk_alignment_for_contiguous_layout())
    if alignment <= 0:
        raise RuntimeError("DeepGEMM returned an invalid grouped-row alignment")
    return alignment


def deepgemm_grouped_padded_rows(
    rows: int,
    group_count: int,
    *,
    alignment: int,
) -> int:
    """Return DeepGEMM's synchronization-free padded row upper bound."""

    row_count = int(rows)
    groups = int(group_count)
    aligned_to = int(alignment)
    if row_count <= 0:
        raise ValueError("DeepGEMM grouped execution requires routed rows")
    if groups <= 0:
        raise ValueError("DeepGEMM group count must be positive")
    if aligned_to <= 0:
        raise ValueError("DeepGEMM row alignment must be positive")
    return row_count + min(row_count, groups) * (aligned_to - 1)


def build_deepgemm_grouped_layout(
    sorted_group_ids: torch.Tensor,
    *,
    group_count: int,
    alignment: int,
) -> DeepGEMMGroupedLayout:
    """Align each routed expert's row range and mark padding with ``-1``."""

    if sorted_group_ids.ndim != 1:
        raise ValueError("DeepGEMM group ids must be one-dimensional")
    rows = int(sorted_group_ids.numel())
    groups = int(group_count)
    aligned_to = int(alignment)
    if rows <= 0:
        raise ValueError("DeepGEMM grouped execution requires routed rows")
    if groups <= 0:
        raise ValueError("DeepGEMM group count must be positive")
    if aligned_to <= 0:
        raise ValueError("DeepGEMM row alignment must be positive")

    index = sorted_group_ids.to(dtype=torch.long)
    counts = torch.bincount(index, minlength=groups)[:groups]
    aligned = ((counts + aligned_to - 1) // aligned_to) * aligned_to
    cumulative_padding = torch.nn.functional.pad(
        (aligned - counts).cumsum(0),
        (1, 0),
    )
    row_positions = torch.arange(rows, device=index.device) + (
        cumulative_padding[index]
    )
    padded_rows = deepgemm_grouped_padded_rows(
        rows,
        groups,
        alignment=aligned_to,
    )
    layout = torch.full(
        (padded_rows,),
        -1,
        dtype=torch.int32,
        device=index.device,
    )
    layout[row_positions] = index.to(torch.int32)
    return DeepGEMMGroupedLayout(
        row_positions=row_positions,
        group_ids=layout,
        padded_rows=padded_rows,
    )


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


def grouped_jit_bucket(active_groups: int, *, capacity: int) -> int:
    """Bound dynamic routed-expert shapes to a small reusable JIT set.

    DeepGEMM specializes on the weight-group dimension.  Compiling the exact
    number of experts hit by every layer creates dozens of NVCC invocations
    during the first long Prefill.  Power-of-two buckets (with the physical
    capacity as the final bucket) keep that dimension stable.  Callers pad
    only metadata/weights for groups that have no routed rows, so routing and
    numerics are unchanged.
    """

    active = int(active_groups)
    limit = int(capacity)
    if active <= 0:
        raise ValueError("active groups must be positive")
    if limit <= 0 or active > limit:
        raise ValueError("group capacity must cover active groups")
    if active == limit:
        return limit
    bucket = min(32, limit)
    while bucket < active and bucket < limit:
        bucket = min(limit, bucket * 2)
    return bucket


def build_active_grouped_routes(
    flat_group_ids: torch.Tensor,
    *,
    group_capacity: int,
    bucketed: bool,
) -> ActiveGroupedRoutes:
    """Sort routes and compact unused experts without model-specific logic."""

    if flat_group_ids.ndim != 1 or int(flat_group_ids.numel()) <= 0:
        raise ValueError("active grouped routes require a non-empty vector")
    capacity = int(group_capacity)
    if capacity <= 0:
        raise ValueError("group capacity must be positive")
    order = torch.argsort(flat_group_ids)
    sorted_global = flat_group_ids.index_select(0, order).to(torch.long)
    unique = torch.unique_consecutive(sorted_global)
    active = int(unique.numel())
    local = torch.searchsorted(unique, sorted_global).to(torch.long)
    execution = (
        grouped_jit_bucket(active, capacity=capacity)
        if bucketed
        else active
    )
    return ActiveGroupedRoutes(
        order=order,
        sorted_group_ids=local,
        unique_group_ids=unique,
        active_group_count=active,
        execution_group_count=execution,
    )


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
    layout: DeepGEMMGroupedLayout,
    output: torch.Tensor,
    workspace: DeepGEMMGroupedWorkspace,
    module: ModuleType | Any | None = None,
) -> torch.Tensor:
    """Execute aligned M-grouped E4M3 GEMM and restore routed row order."""

    if module is None:
        import deep_gemm as module  # type: ignore[no-redef]

    padded_rows = int(layout.padded_rows)
    padded_value = workspace.value[:padded_rows]
    padded_scale_a = workspace.scale_a[:padded_rows]
    padded_output = workspace.output[:padded_rows]
    if tuple(padded_value.shape[1:]) != tuple(value.shape[1:]):
        raise ValueError("DeepGEMM padded value workspace has the wrong shape")
    if tuple(padded_scale_a.shape[1:]) != tuple(scale_a.shape[1:]):
        raise ValueError("DeepGEMM padded scale workspace has the wrong shape")
    if tuple(padded_output.shape[1:]) != tuple(output.shape[1:]):
        raise ValueError("DeepGEMM padded output workspace has the wrong shape")
    positions = layout.row_positions
    # Copy the byte-exact FP8 payload.  The uint8 view also keeps the public
    # helper testable on CPU builds where index_copy_ has no Float8 kernel.
    padded_value.view(torch.uint8).index_copy_(
        0,
        positions,
        value.view(torch.uint8),
    )
    padded_scale_a.index_copy_(0, positions, scale_a)
    module.m_grouped_fp8_gemm_nt_contiguous(
        (padded_value, padded_scale_a),
        (weights, scale_b),
        padded_output,
        layout.group_ids,
        disable_ue8m0_cast=True,
    )
    torch.index_select(padded_output, 0, positions, out=output)
    return output


def execute_grouped_fp8(
    value: torch.Tensor,
    weights: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offsets: torch.Tensor | None,
    backend: str,
    deepgemm_layout: DeepGEMMGroupedLayout | None = None,
    deepgemm_workspace: DeepGEMMGroupedWorkspace | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute grouped FP8 by capability, independent of model family."""

    if backend == "deepgemm-sm90":
        if deepgemm_layout is None or deepgemm_workspace is None or output is None:
            raise RuntimeError(
                "DeepGEMM grouped FP8 requires aligned layout and padded workspace"
            )
        return execute_deepgemm_grouped_fp8(
            value,
            weights,
            scale_a=scale_a,
            scale_b=scale_b,
            layout=deepgemm_layout,
            output=output,
            workspace=deepgemm_workspace,
        )
    if backend == "torch-scaled-grouped-mm":
        if offsets is None:
            raise RuntimeError(
                "PyTorch grouped FP8 requires cumulative offsets"
            )
        return torch._scaled_grouped_mm(
            value,
            weights.transpose(1, 2),
            scale_a=scale_a.view(-1),
            scale_b=scale_b,
            offs=offsets,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
    raise RuntimeError(f"unsupported grouped FP8 backend: {backend}")

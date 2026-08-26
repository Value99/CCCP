"""Public CPU-shared/GPU-routed MoE overlap primitives.

The model layer supplies only three gated-MLP weights.  This module owns the
fixed pinned staging buffers and the exact CPU execution path; it does not
dispatch on an architecture or model directory name.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..kernels import BlockFP8Weight, ProjectionGroup
from .api import linear


def cpu_gated_mlp(
    value: torch.Tensor,
    gate_weight,
    up_weight,
    down_weight,
    *,
    limit: float,
) -> torch.Tensor:
    """Execute one exact gated MLP from CPU-resident compact weights."""

    if isinstance(gate_weight, BlockFP8Weight) and isinstance(
        up_weight, BlockFP8Weight
    ):
        gate_up = linear(
            value,
            ProjectionGroup((gate_weight, up_weight)),
        )
        intermediate = int(gate_weight.shape[0])
        gate = gate_up[:, :intermediate]
        up = gate_up[:, intermediate:]
    else:
        gate = F.linear(value.to(gate_weight.dtype), gate_weight)
        up = F.linear(value.to(up_weight.dtype), up_weight)
    if limit > 0:
        up = up.clamp(min=-limit, max=limit)
        gate = gate.clamp(max=limit)
    activated = F.silu(gate) * up
    return linear(activated, down_weight)


class HeterogeneousSharedExpertExecutor:
    """Run shared experts on CPU while routed experts occupy the GPU."""

    host_input: torch.Tensor
    host_output: torch.Tensor
    device_output: torch.Tensor

    def __init__(
        self,
        *,
        device: torch.device,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("heterogeneous shared executor requires CUDA")
        self.device = device
        self.hidden_size = int(hidden_size)
        self.dtype = dtype
        self.layers: dict[int, tuple[object, object, object]] = {}
        self.host_input = torch.empty(
            (1, self.hidden_size), dtype=dtype, pin_memory=True
        )
        self.host_output = torch.empty_like(
            self.host_input, pin_memory=True
        )
        self.device_output = torch.empty(
            (1, self.hidden_size), dtype=dtype, device=device
        )
        self.input_ready = torch.cuda.Event()

    def add_layer(
        self,
        layer: int,
        gate_weight,
        up_weight,
        down_weight,
    ) -> None:
        self.layers[int(layer)] = (
            gate_weight,
            up_weight,
            down_weight,
        )

    def run(self, layer: int, value: torch.Tensor, *, limit: float) -> torch.Tensor:
        if value.shape != self.host_input.shape:
            raise ValueError(
                "heterogeneous shared executor accepts exactly one token"
            )
        stream = torch.cuda.current_stream(self.device)
        self.host_input.copy_(value, non_blocking=True)
        self.input_ready.record(stream)
        # The CPU cannot consume the pinned input until the tiny D2H completes.
        # Routed GPU work was queued by the caller before entering here and
        # continues independently during this synchronization and CPU GEMV.
        self.input_ready.synchronize()
        gate, up, down = self.layers[int(layer)]
        output = cpu_gated_mlp(
            self.host_input,
            gate,
            up,
            down,
            limit=float(limit),
        )
        self.host_output.copy_(output)
        self.device_output.copy_(self.host_output, non_blocking=True)
        return self.device_output


__all__ = ["HeterogeneousSharedExpertExecutor", "cpu_gated_mlp"]

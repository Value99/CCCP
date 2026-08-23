"""Model-independent fixed-address MoE scheduling primitives."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch

from ..kernels import BlockFP8Weight, ProjectionGroup


def _linear_weight_device(weight) -> torch.device:
    if isinstance(weight, ProjectionGroup):
        devices = {_linear_weight_device(item) for item in weight.weights}
        if len(devices) != 1:
            raise ValueError("projection group weights must share a device")
        return devices.pop()
    return weight.device


def _is_bf16_linear(weight) -> bool:
    return (
        isinstance(weight, (BlockFP8Weight, ProjectionGroup))
        or (
            isinstance(weight, torch.Tensor)
            and weight.dtype == torch.bfloat16
        )
    )


@dataclass(frozen=True)
class FixedMoEPreludeSpec:
    """Mathematics required by a fused owner-local Router+Down graph."""

    hidden_size: int
    routed_hidden_size: int
    expert_count: int
    top_k: int
    scoring_func: str = "sigmoid"
    normalize: bool = True
    scaling: float = 1.0
    n_group: int = 1
    topk_group: int = 1


@dataclass
class _FixedMoEPreludeLayer:
    source: torch.Tensor
    gate_weight: torch.Tensor
    correction: torch.Tensor
    available: torch.Tensor
    down_weight: object
    down_workspace: torch.Tensor | None
    route_buffers: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    latent: torch.Tensor
    stream: torch.cuda.Stream
    done: torch.cuda.Event
    graph: torch.cuda.CUDAGraph | None = None


class FixedMoEPrelude:
    """Capture owner-local Router and routed-Down at fixed addresses.

    The executor is keyed only by tensor shapes and routing mathematics.  A
    model runtime supplies its fixed input/weights and the input-ready event;
    no model-family name participates in dispatch.
    """

    def __init__(self, spec: FixedMoEPreludeSpec) -> None:
        if (
            spec.hidden_size <= 0
            or spec.routed_hidden_size <= 0
            or spec.expert_count <= 0
            or not 0 < spec.top_k <= 16
        ):
            raise ValueError("invalid fixed MoE prelude specification")
        self.spec = spec
        self.layers: dict[int, _FixedMoEPreludeLayer] = {}

    def add_layer(
        self,
        layer: int,
        source: torch.Tensor,
        gate_weight: torch.Tensor,
        correction: torch.Tensor,
        available: torch.Tensor,
        down_weight,
        route_buffers: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        latent: torch.Tensor,
    ) -> None:
        spec = self.spec
        if layer in self.layers:
            raise ValueError(f"fixed MoE prelude layer {layer} exists")
        device = source.device
        logits, weights, indices = route_buffers
        if (
            not source.is_cuda
            or source.shape != (1, spec.hidden_size)
            or source.dtype != torch.bfloat16
            or gate_weight.device != device
            or gate_weight.dtype != torch.float32
            or gate_weight.shape
            != (spec.expert_count, spec.hidden_size)
            or correction.device != device
            or correction.dtype != torch.float32
            or correction.numel() != spec.expert_count
            or available.device != device
            or available.dtype != torch.bool
            or available.numel() != spec.expert_count
            or _linear_weight_device(down_weight) != device
            or not _is_bf16_linear(down_weight)
            or down_weight.shape
            != (spec.routed_hidden_size, spec.hidden_size)
            or logits.device != device
            or logits.dtype != torch.float32
            or logits.shape != (1, spec.expert_count)
            or weights.device != device
            or weights.dtype != torch.float32
            or weights.shape != (1, spec.top_k)
            or indices.device != device
            or indices.dtype != torch.long
            or indices.shape != (1, spec.top_k)
            or latent.device != device
            or latent.dtype != torch.bfloat16
            or latent.shape != (1, spec.routed_hidden_size)
        ):
            raise ValueError("fixed MoE prelude tensor layout mismatch")
        with torch.cuda.device(device):
            down_workspace = (
                torch.empty(
                    1,
                    spec.routed_hidden_size,
                    dtype=torch.float32,
                    device=device,
                )
                if isinstance(
                    down_weight,
                    (BlockFP8Weight, ProjectionGroup),
                )
                else None
            )
            self.layers[int(layer)] = _FixedMoEPreludeLayer(
                source=source,
                gate_weight=gate_weight,
                correction=correction,
                available=available,
                down_weight=down_weight,
                down_workspace=down_workspace,
                route_buffers=route_buffers,
                latent=latent,
                stream=torch.cuda.Stream(device=device),
                done=torch.cuda.Event(),
            )

    def _execute(self, state: _FixedMoEPreludeLayer) -> None:
        from .api import linear, linear_route_topk

        route = linear_route_topk(
            state.source,
            state.gate_weight,
            state.correction,
            state.available,
            scoring_func=self.spec.scoring_func,
            top_k=self.spec.top_k,
            normalize=self.spec.normalize,
            scaling=self.spec.scaling,
            n_group=self.spec.n_group,
            topk_group=self.spec.topk_group,
            output_buffers=state.route_buffers,
        )
        if route is None:
            raise RuntimeError(
                "fixed MoE prelude requires a registered fused router"
            )
        if state.down_workspace is None:
            torch.mm(
                state.source,
                state.down_weight.t(),
                out=state.latent,
            )
        else:
            linear(
                state.source,
                state.down_weight,
                output=state.down_workspace,
            )
            state.latent.copy_(state.down_workspace)

    def capture(self) -> None:
        for state in self.layers.values():
            device = state.source.device
            with (
                torch.cuda.device(device),
                torch.cuda.stream(state.stream),
            ):
                state.source.zero_()
                self._execute(state)
                state.stream.synchronize()
                retained = (
                    os.environ.get("CCCP_TP_LAYER_GRAPH", "0") != "0"
                )
                graph = torch.cuda.CUDAGraph(keep_graph=retained)
                with torch.cuda.graph(graph, stream=state.stream):
                    self._execute(state)
                if retained:
                    graph.instantiate()
                state.done.record(state.stream)
                state.stream.synchronize()
                state.graph = graph

    def run(
        self,
        layer: int,
        source: torch.Tensor,
        ready_event: torch.cuda.Event,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        state = self.layers[int(layer)]
        if state.graph is None:
            raise RuntimeError("fixed MoE prelude graph is not captured")
        if source.data_ptr() != state.source.data_ptr():
            raise ValueError(
                "fixed MoE prelude input address changed after capture"
            )
        device = state.source.device
        with (
            torch.cuda.device(device),
            torch.cuda.stream(state.stream),
        ):
            state.stream.wait_event(ready_event)
            state.graph.replay()
            state.done.record(state.stream)
        with torch.cuda.device(device):
            torch.cuda.current_stream(device).wait_event(state.done)
        return (
            (state.route_buffers[1], state.route_buffers[2]),
            state.latent,
        )

    def retained_graph(self, layer: int) -> torch.cuda.CUDAGraph:
        state = self.layers[int(layer)]
        if (
            state.graph is None
            or os.environ.get("CCCP_TP_LAYER_GRAPH", "0") == "0"
        ):
            raise RuntimeError("fixed MoE prelude graph is not retained")
        return state.graph

    def result(
        self,
        layer: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return outputs after a parent layer Graph has replayed the child."""
        state = self.layers[int(layer)]
        return (
            (state.route_buffers[1], state.route_buffers[2]),
            state.latent,
        )


__all__ = ["FixedMoEPrelude", "FixedMoEPreludeSpec"]

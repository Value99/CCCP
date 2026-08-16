"""Fixed-address CUDA Graph wrapper for public packed Top-K MoE operators."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch


@dataclass(frozen=True)
class PackedMoEGraphSpec:
    """Capability and mathematics captured by one packed MoE graph."""

    activation: str
    activation_beta: float
    activation_linear_beta: float
    limit: float
    top_k: int
    grouped_prefix: int
    packed_formats: tuple[str, ...]
    code_dims: tuple[int, ...]
    codebook_sizes: tuple[int, ...]


class FixedPackedMoEGraph:
    """Replay a packed Top-K MoE chain without Python launch fan-out.

    Tensor addresses and shapes are fixed, while route weights and the
    pointer metadata *contents* may change between replays.  This makes the
    wrapper suitable for RAM+VRAM caches: a caller publishes newly leased
    packed slot addresses, orders H2D before :meth:`run`, then launches the
    exact same registered kernels as the eager path.
    """

    def __init__(
        self,
        value: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        metadata: torch.Tensor,
        hidden_workspace: torch.Tensor,
        output_workspace: torch.Tensor,
        result: torch.Tensor,
        spec: PackedMoEGraphSpec,
    ) -> None:
        tensors = (
            value,
            route_ids,
            route_weights,
            metadata,
            hidden_workspace,
            output_workspace,
            result,
        )
        if not all(tensor.is_cuda for tensor in tensors):
            raise ValueError("fixed packed MoE graph requires CUDA tensors")
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            raise ValueError("fixed packed MoE graph tensors must share device")
        self.value = value
        self.route_ids = route_ids
        self.route_weights = route_weights
        self.metadata = metadata
        self.hidden_workspace = hidden_workspace
        self.output_workspace = output_workspace
        self.result = result
        self.spec = spec
        self.device = value.device
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
        self.graph: torch.cuda.CUDAGraph | None = None

    @property
    def addresses(self) -> tuple[int, ...]:
        return tuple(
            tensor.data_ptr()
            for tensor in (
                self.value,
                self.route_ids,
                self.route_weights,
                self.metadata,
                self.hidden_workspace,
                self.output_workspace,
                self.result,
            )
        )

    def matches(
        self,
        value: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        metadata: torch.Tensor,
        hidden_workspace: torch.Tensor,
        output_workspace: torch.Tensor,
        result: torch.Tensor,
        spec: PackedMoEGraphSpec,
    ) -> bool:
        return self.spec == spec and self.addresses == tuple(
            tensor.data_ptr()
            for tensor in (
                value,
                route_ids,
                route_weights,
                metadata,
                hidden_workspace,
                output_workspace,
                result,
            )
        )

    def _execute(self) -> torch.Tensor:
        from .api import packed_moe_topk

        return packed_moe_topk(
            self.value,
            self.route_ids,
            self.route_weights,
            self.metadata,
            activation=self.spec.activation,
            activation_beta=self.spec.activation_beta,
            activation_linear_beta=self.spec.activation_linear_beta,
            hidden_workspace=self.hidden_workspace,
            output_workspace=self.output_workspace,
            result=self.result,
            grouped_prefix=self.spec.grouped_prefix,
            packed_formats=self.spec.packed_formats,
            code_dims=self.spec.code_dims,
            codebook_sizes=self.spec.codebook_sizes,
            limit=self.spec.limit,
        )

    def capture(self) -> None:
        if self.graph is not None:
            return
        current = torch.cuda.current_stream(self.device)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.stream.wait_stream(current)
            self._execute()
            self.stream.synchronize()
            retained = os.environ.get("CCCP_TP_LAYER_GRAPH", "0") != "0"
            graph = torch.cuda.CUDAGraph(keep_graph=retained)
            with torch.cuda.graph(graph, stream=self.stream):
                self._execute()
            if retained:
                graph.instantiate()
            self.stream.synchronize()
            self.graph = graph

    def run(self) -> torch.Tensor:
        if self.graph is None:
            self.capture()
        if self.graph is None:
            raise RuntimeError("fixed packed MoE graph capture failed")
        # ``CUDAGraph.replay`` is submitted to PyTorch's current stream.  The
        # caller has already ordered packed H2D, metadata publication and the
        # fixed input copies on that stream, so an extra private-stream hop and
        # one CUDA Event per MoE layer are both redundant.  Kimi K3 executes
        # 92 routed layers per token; those seemingly small cross-stream
        # boundaries otherwise dominate the batch=1 decode launch time.
        with torch.cuda.device(self.device):
            self.graph.replay()
        return self.result


__all__ = ["FixedPackedMoEGraph", "PackedMoEGraphSpec"]

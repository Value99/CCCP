"""Model-independent tensor-parallel decode operators.

The executors in this module are keyed by tensor shapes and mathematical
capabilities.  Model runtimes only provide weights plus configuration values;
no model family name participates in dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import torch
import torch.nn.functional as F

from ..kernels import BlockFP8Weight, Int4Weight, ProjectionGroup


def _weight_shape(weight) -> torch.Size:
    return torch.Size(weight.shape)


def _weight_is_bf16_linear(weight) -> bool:
    return (
        isinstance(weight, (BlockFP8Weight, Int4Weight, ProjectionGroup))
        or (
            isinstance(weight, torch.Tensor)
            and weight.dtype == torch.bfloat16
        )
    )


def _linear(
    value: torch.Tensor,
    weight,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Public linear dispatch for BF16 and compact audited block-FP8."""
    from .api import linear

    return linear(value, weight, output_dtype=output_dtype)


def _linear_batch(
    value: torch.Tensor,
    weight,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Public shape-based batched projection dispatcher."""
    from .api import linear_batch

    return linear_batch(value, weight, output_dtype=output_dtype)


def _int4_tensor_core_half(
    weight: Int4Weight,
    device: torch.device,
) -> bool:
    """Select compact Int4 expansion dtype from the target device."""
    mode = os.environ.get("CCCP_INT4_HALF", "auto").strip().lower()
    if mode in {"1", "true", "on", "yes"}:
        return True
    if mode in {"0", "false", "off", "no"}:
        return False
    if mode not in {"", "auto"}:
        raise ValueError("CCCP_INT4_HALF must be auto, 0, or 1")
    return bool(weight.half or device.type == "cuda")


def _projection_parts(weight, rows: tuple[int, ...]) -> tuple:
    if isinstance(weight, ProjectionGroup):
        values = weight.weights
        if tuple(int(value.shape[0]) for value in values) != rows:
            raise ValueError("compact projection group row layout mismatch")
        return values
    if int(weight.shape[0]) != sum(rows):
        raise ValueError("projection row layout mismatch")
    return tuple(weight.split(rows, dim=0))


def _row_slice(weight, start: int, stop: int, device: torch.device):
    # TP1 is a view of the complete operator, not a shard.  Reusing an
    # already-resident compact weight is both mathematically exact and avoids
    # retaining a second full Dense copy merely because the common TP path is
    # enabled with width one.
    if (
        start == 0
        and stop == int(weight.shape[0])
        and getattr(weight, "device", None) == device
    ):
        return weight
    if isinstance(weight, BlockFP8Weight):
        if start % weight.block:
            return weight.dequant_rows(
                start,
                stop,
                torch.bfloat16,
            ).to(device).contiguous()
        return weight.row_slice(start, stop).to(device)
    if isinstance(weight, Int4Weight):
        return Int4Weight(
            weight.q[start:stop].to(device).contiguous(),
            weight.s[start:stop].to(device).contiguous(),
            weight.cols,
            weight.gs,
            half=_int4_tensor_core_half(weight, device),
        )
    shard = weight[start:stop]
    if shard.device == device:
        return shard.clone(memory_format=torch.contiguous_format)
    return shard.to(device).contiguous()


def _column_slice(weight, start: int, stop: int, device: torch.device):
    if (
        start == 0
        and stop == int(weight.shape[1])
        and getattr(weight, "device", None) == device
    ):
        return weight
    if isinstance(weight, BlockFP8Weight):
        if start % weight.block:
            # This is reserved for small irregular projections. All major
            # hidden/intermediate TP widths are block aligned.
            return weight.dequant_rows(
                0,
                weight.shape[0],
                torch.bfloat16,
            )[:, start:stop].to(device).contiguous()
        return weight.column_slice(start, stop).to(device)
    if isinstance(weight, Int4Weight):
        if start % weight.gs or stop % weight.gs:
            return weight.dequant_rows(0, weight.shape[0])[
                :, start:stop
            ].to(device).contiguous()
        return Int4Weight(
            weight.q[:, start // 2:stop // 2].to(device).contiguous(),
            weight.s[:, start // weight.gs:stop // weight.gs]
            .to(device)
            .contiguous(),
            stop - start,
            weight.gs,
            half=_int4_tensor_core_half(weight, device),
        )
    return weight[:, start:stop].to(device).contiguous()


def _combine_projection_parts(values) -> object:
    parts = tuple(values)
    if all(isinstance(value, torch.Tensor) for value in parts):
        return torch.cat(parts, dim=0).contiguous()
    return ProjectionGroup(parts)


def _head_slice(
    value: torch.Tensor,
    rank: int,
    local_heads: int,
) -> torch.Tensor:
    start = int(rank) * int(local_heads)
    stop = start + int(local_heads)
    if value.ndim == 0 or int(value.shape[0]) < stop:
        raise ValueError("head parameter does not cover requested TP rank")
    return value[start:stop]


def shard_linear_output(weight, rank: int, ranks: int, device):
    """Return one compact Column-TP output-row shard."""
    rows = int(weight.shape[0])
    if ranks <= 0 or rows % ranks:
        raise ValueError("linear output rows must divide the TP width")
    local = rows // int(ranks)
    return _row_slice(
        weight,
        int(rank) * local,
        (int(rank) + 1) * local,
        torch.device(device),
    )


def shard_linear_input(weight, rank: int, ranks: int, device):
    """Return one compact Row-TP input-column shard."""
    columns = int(weight.shape[1])
    if ranks <= 0 or columns % ranks:
        raise ValueError("linear input columns must divide the TP width")
    local = columns // int(ranks)
    return _column_slice(
        weight,
        int(rank) * local,
        (int(rank) + 1) * local,
        torch.device(device),
    )


class TensorParallelVocab:
    """Generic vocabulary-row TP for embedding and output projection."""

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        embedding_shards: tuple[torch.Tensor, ...],
        output_shards: tuple[torch.Tensor, ...],
        offsets: tuple[int, ...],
    ) -> None:
        ranks = len(devices)
        if (
            ranks <= 0
            or len(embedding_shards) != ranks
            or len(output_shards) != ranks
            or len(offsets) != ranks + 1
        ):
            raise ValueError("vocabulary TP shard count mismatch")
        hidden = int(embedding_shards[0].shape[1])
        if any(
            shard.device != devices[rank]
            or shard.dtype != torch.bfloat16
            or shard.ndim != 2
            or int(shard.shape[1]) != hidden
            or output_shards[rank].device != devices[rank]
            or output_shards[rank].dtype != torch.bfloat16
            or output_shards[rank].shape != shard.shape
            or int(shard.shape[0]) != offsets[rank + 1] - offsets[rank]
            for rank, shard in enumerate(embedding_shards)
        ):
            raise ValueError("vocabulary TP shard layout mismatch")
        self.devices = devices
        self.embedding_shards = embedding_shards
        self.output_shards = output_shards
        self.offsets = offsets
        self.hidden_size = hidden

    @staticmethod
    def offsets_for(vocab_size: int, ranks: int) -> tuple[int, ...]:
        return tuple(
            (vocab_size * rank) // ranks
            for rank in range(ranks + 1)
        )

    def embed(self, ids: list[int] | torch.Tensor) -> torch.Tensor:
        values = (
            ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        )
        if not values:
            raise ValueError("vocabulary TP embedding requires token ids")
        ranks = {
            max(
                0,
                min(
                    len(self.devices) - 1,
                    next(
                        rank
                        for rank in range(len(self.devices))
                        if token < self.offsets[rank + 1]
                    ),
                ),
            )
            for token in values
        }
        if len(ranks) != 1:
            # 多 token 预填跨词表分片:按 rank 分组嵌入、汇到 rank0、按
            # 原序回填(调用方从返回值继承 device/dtype)。原实现为硬
            # 错误——TP 预填由此不可用;decode 单 token 不经此分支。
            base = self.devices[0]
            by_rank: dict[int, list[int]] = {}
            for index, token in enumerate(values):
                for rank in range(len(self.devices)):
                    if token < self.offsets[rank + 1]:
                        by_rank.setdefault(rank, []).append(index)
                        break
            rows = [None] * len(values)
            for rank, indexes in by_rank.items():
                local = torch.as_tensor(
                    [values[i] - self.offsets[rank] for i in indexes],
                    dtype=torch.long,
                    device=self.devices[rank],
                )
                embedded = F.embedding(
                    local, self.embedding_shards[rank]
                ).to(base)
                for offset, i in enumerate(indexes):
                    rows[i] = embedded[offset]
            return torch.stack(rows)
        rank = ranks.pop()
        local_ids = torch.as_tensor(
            [token - self.offsets[rank] for token in values],
            dtype=torch.long,
            device=self.devices[rank],
        )
        return F.embedding(local_ids, self.embedding_shards[rank])

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError("vocabulary TP hidden width mismatch")
        target = hidden.device
        partials = []
        for rank, device in enumerate(self.devices):
            local_hidden = hidden if target == device else hidden.to(device)
            partials.append(
                torch.mm(
                    local_hidden.to(torch.bfloat16),
                    self.output_shards[rank].t(),
                    out_dtype=torch.float32,
                )
            )
        return torch.cat(
            [
                partial if partial.device == target else partial.to(target)
                for partial in partials
            ],
            dim=-1,
        )


def _new_cuda_graph() -> torch.cuda.CUDAGraph:
    """Retain graph topology only for fixed-address layer composition."""
    return torch.cuda.CUDAGraph(
        keep_graph=os.environ.get("CCCP_TP_LAYER_GRAPH", "0") != "0"
    )


def _instantiate_retained_graph(graph: torch.cuda.CUDAGraph) -> None:
    if os.environ.get("CCCP_TP_LAYER_GRAPH", "0") != "0":
        graph.instantiate()


def _no_owner_rank_order(executor, state) -> tuple[int, ...]:
    """Return canonical rank order for formal TP, legacy order otherwise.

    ``state.owner`` is permitted to describe where an unsplit source weight
    was staged during construction.  It must not choose the launch or
    reduction order once the all-rank TPHidden data flow is enabled.
    """
    ranks = len(executor.devices)
    if (
        getattr(executor, "hidden_mode", False)
        and os.environ.get("CCCP_TP_NO_OWNER", "1") != "0"
    ):
        return tuple(range(ranks))
    owner = int(state.owner)
    return (
        owner,
        *(rank for rank in range(ranks) if rank != owner),
    )


def _compose_normalize_prelude(
    executor,
    layer: int,
    source,
    residual,
    active_rows: int,
    projections,
    norm_weights,
    post_norm_weights,
    workspaces,
    eps: float,
) -> None:
    """Prefix a retained Attention graph with fixed rank-local normalization."""
    if os.environ.get("CCCP_TP_LAYER_GRAPH", "0") == "0":
        raise RuntimeError("TP layer Graph composition is disabled")
    from ..fusedext import make_tp_graph_sequence_batch

    state = executor.layers[layer]
    if (
        state.graphs is None
        or state.events is None
        or state.source_event is None
    ):
        raise RuntimeError("TP rank graphs are not captured")
    local_source = (
        source
        if tuple(source.devices) == executor.devices
        else source.subset(executor.devices)
    )
    target = executor.input_hidden(layer)
    preludes = local_source.capture_normalize_graphs(
        target,
        executor.streams,
        post_norm_weights,
        float(eps),
        residual=residual,
        active_rows=int(active_rows),
        projections=projections,
        norm_weights=norm_weights,
        workspaces=workspaces,
    )
    rank_order = _no_owner_rank_order(executor, state)
    state.graph_batch = make_tp_graph_sequence_batch(
        [
            int(executor.devices[rank].index)
            for rank in rank_order
        ],
        [
            [preludes[rank], state.graphs[ordered_rank]]
            for ordered_rank, rank in enumerate(rank_order)
        ],
        [executor.streams[rank] for rank in rank_order],
        list(state.events),
        state.source_event,
    )
    state.composed_input_addresses = local_source.fixed_addresses


def _compose_mlp_prelude(
    executor,
    layer: int,
    source,
    attention,
    prefix_output,
    residual,
    active_rows: int,
    projections,
    norm_weights,
    post_norm_weights,
    workspaces,
    eps: float,
    boundary: bool,
) -> None:
    """Prefix a retained gated-MLP graph with fixed residual preparation."""
    if os.environ.get("CCCP_TP_LAYER_GRAPH", "0") == "0":
        raise RuntimeError("TP layer Graph composition is disabled")
    from ..fusedext import make_tp_graph_sequence_batch

    state = executor.layers[layer]
    if (
        state.graphs is None
        or state.events is None
        or state.source_event is None
    ):
        raise RuntimeError("TP gated MLP rank graphs are not captured")
    local_source = (
        source
        if tuple(source.devices) == executor.devices
        else source.subset(executor.devices)
    )
    local_attention = (
        attention
        if tuple(attention.devices) == executor.devices
        else attention.subset(executor.devices)
    )
    local_prefix = (
        prefix_output
        if tuple(prefix_output.devices) == executor.devices
        else prefix_output.subset(executor.devices)
    )
    target = executor.input_hidden(layer)
    preludes = local_source.capture_mlp_prelude_graphs(
        local_attention,
        local_prefix,
        target,
        executor.streams,
        residual,
        int(active_rows),
        projections,
        norm_weights,
        post_norm_weights,
        workspaces,
        float(eps),
        boundary=bool(boundary),
    )
    rank_order = _no_owner_rank_order(executor, state)
    state.graph_batch = make_tp_graph_sequence_batch(
        [
            int(executor.devices[rank].index)
            for rank in rank_order
        ],
        [
            [preludes[rank], state.graphs[ordered_rank]]
            for ordered_rank, rank in enumerate(rank_order)
        ],
        [executor.streams[rank] for rank in rank_order],
        list(state.events),
        state.source_event,
    )
    state.composed_input_addresses = local_attention.fixed_addresses
    state.composed_prefix_graphs = preludes


class OwnerGroupedTensorParallel:
    """Dispatch layers to the TP subgroup containing their owner rank.

    Large packed operators may still use the complete device tuple, while
    latency-bound projections use smaller contiguous subgroups.  The wrapper
    preserves one executor interface and keeps group selection out of model
    names and operator registry keys.
    """

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        group_size: int,
        factory: Callable[
            [tuple[torch.device, ...]],
            object,
        ],
    ) -> None:
        group_size = int(group_size)
        if (
            group_size <= 1
            or group_size > len(devices)
            or len(devices) % group_size
        ):
            raise ValueError(
                "TP subgroup size must divide the visible device count"
            )
        self.devices = devices
        self.group_size = group_size
        self.groups = tuple(
            devices[start:start + group_size]
            for start in range(0, len(devices), group_size)
        )
        self.executors = tuple(factory(group) for group in self.groups)
        self.layer_groups: dict[int, int] = {}
        self._global_outputs: dict[int, object] = {}

    def add_layer(
        self,
        layer: int,
        owner: int,
        *args,
        **kwargs,
    ) -> None:
        if layer in self.layer_groups:
            raise ValueError(
                f"TP subgroup layer {layer} is already registered"
            )
        group_index = int(owner) // self.group_size
        local_owner = int(owner) % self.group_size
        self.executors[group_index].add_layer(
            layer,
            local_owner,
            *args,
            **kwargs,
        )
        self.layer_groups[layer] = group_index

    def capture(self) -> None:
        for executor in self.executors:
            executor.capture()

    def _executor(self, layer: int):
        return self.executors[self.layer_groups[layer]]

    def run(self, layer: int, *args, **kwargs):
        return self._executor(layer).run(layer, *args, **kwargs)

    def input_buffer(self, layer: int) -> torch.Tensor:
        return self._executor(layer).input_buffer(layer)

    def input_hidden(self, layer: int):
        return self._executor(layer).input_hidden(layer)

    def output_hidden(self, layer: int):
        local = self._executor(layer).output_hidden(layer)
        if self.group_size == len(self.devices):
            return local
        output = self._global_outputs.get(layer)
        if output is None:
            from .hidden import TPHidden

            output = TPHidden.empty(
                self.devices,
                tuple(local.shape),
                dtype=local.dtype,
            )
            self._global_outputs[layer] = output
        return output

    def input_sharded(self, layer: int):
        return self._executor(layer).input_sharded(layer)

    def composed_input_sharded(self, layer: int, *args, **kwargs):
        return self._executor(layer).composed_input_sharded(
            layer,
            *args,
            **kwargs,
        )

    def run_prepared(self, layer: int, *args, **kwargs):
        return self._executor(layer).run_prepared(
            layer,
            *args,
            **kwargs,
        )

    def run_hidden(self, layer: int, *args, **kwargs):
        executor = self._executor(layer)
        if (
            args
            and hasattr(args[0], "subset")
            and tuple(args[0].devices) != executor.devices
        ):
            args = (args[0].subset(executor.devices), *args[1:])
        kwargs.setdefault("output", self.output_hidden(layer))
        return executor.run_hidden(
            layer,
            *args,
            **kwargs,
        )

    def compose_normalize_prelude(
        self,
        layer: int,
        *args,
        **kwargs,
    ) -> None:
        return self._executor(layer).compose_normalize_prelude(
            layer,
            *args,
            **kwargs,
        )

    def compose_mlp_prelude(
        self,
        layer: int,
        *args,
        **kwargs,
    ) -> None:
        return self._executor(layer).compose_mlp_prelude(
            layer,
            *args,
            **kwargs,
        )

    def compose_owner_branch(
        self,
        layer: int,
        owner_graph: torch.cuda.CUDAGraph,
    ) -> None:
        return self._executor(layer).compose_owner_branch(
            layer,
            owner_graph,
        )

    def run_sharded(self, layer: int, *args, **kwargs):
        kwargs.setdefault("output", self.output_hidden(layer))
        return self._executor(layer).run_sharded(
            layer,
            *args,
            **kwargs,
        )

    def launch_partials(self, layer: int, *args, **kwargs):
        executor = self._executor(layer)
        if (
            args
            and hasattr(args[0], "subset")
            and tuple(args[0].devices) != executor.devices
        ):
            args = (args[0].subset(executor.devices), *args[1:])
        return executor.launch_partials(
            layer,
            *args,
            **kwargs,
        )

    def last_partials(self, layer: int):
        return self._executor(layer).last_partials(layer)

    def finalize_moe(self, layer: int, *args, **kwargs):
        kwargs.setdefault("output", self.output_hidden(layer))
        return self._executor(layer).finalize_moe(
            layer,
            *args,
            **kwargs,
        )

    def finalize_moe_full(self, layer: int, *args, **kwargs):
        kwargs.setdefault("output", self.output_hidden(layer))
        return self._executor(layer).finalize_moe_full(
            layer,
            *args,
            **kwargs,
        )

    def start(self, layer: int, *args, **kwargs):
        return self._executor(layer).start(layer, *args, **kwargs)

    def start_prepared(self, layer: int, *args, **kwargs):
        return self._executor(layer).start_prepared(
            layer,
            *args,
            **kwargs,
        )

    def finish(self, layer: int, *args, **kwargs):
        return self._executor(layer).finish(layer, *args, **kwargs)

    def reset(self) -> None:
        for executor in self.executors:
            reset = getattr(executor, "reset", None)
            if reset is not None:
                reset()


@dataclass(frozen=True)
class GatedMLPSpec:
    hidden_size: int
    intermediate_size: int
    activation: str
    activation_beta: float
    activation_linear_beta: float | None
    activation_limit: float = 0.0


@dataclass
class _GatedMLPLayer:
    owner: int
    source: torch.Tensor
    local_inputs: list[torch.Tensor]
    gate_up: list[torch.Tensor]
    down: list[torch.Tensor]
    contributions: list[torch.Tensor]
    zero: torch.Tensor
    graphs: list[torch.cuda.CUDAGraph] | None = None
    events: list[torch.cuda.Event] | None = None
    source_event: torch.cuda.Event | None = None
    input_events: list[torch.cuda.Event] | None = None
    output_replicas: list[torch.Tensor] | None = None
    output_events: list[torch.cuda.Event] | None = None
    graph_batch: object | None = None
    composed_input_addresses: tuple[int, ...] | None = None
    composed_prefix_graphs: tuple[torch.cuda.CUDAGraph, ...] | None = None
    launch_stream: torch.cuda.Stream | None = None
    ready_event: torch.cuda.Event | None = None
    pending_output: torch.Tensor | None = None


class TensorParallelGatedMLP:
    """Row-TP gated MLP with one persistent CUDA graph per rank.

    Gate/Up output rows and matching Down input columns are sharded.  Every
    rank reads one fixed input, computes its local intermediate slice, and
    returns a FP32 partial output.  Formal TPHidden execution reduces directly
    to every rank, so the next layer consumes its local replica without an
    owner broadcast.  Weights remain in their original BF16 representation.
    """

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        spec: GatedMLPSpec,
    ) -> None:
        if not devices:
            raise ValueError("gated MLP graph requires at least one rank")
        if spec.intermediate_size % len(devices):
            raise ValueError(
                "intermediate size must divide the tensor-parallel size"
            )
        self.devices = devices
        self.spec = spec
        self.hidden_mode = (
            os.environ.get("CCCP_TP_HIDDEN", "0") != "0"
        )
        self.streams = [
            torch.cuda.Stream(device=device) for device in devices
        ]
        self.layers: dict[int, _GatedMLPLayer] = {}

    def add_layer(
        self,
        layer: int,
        owner: int,
        combined_gate_up,
        down,
    ) -> None:
        if layer in self.layers:
            raise ValueError(f"TP MLP layer {layer} is already registered")
        if (
            not _weight_is_bf16_linear(combined_gate_up)
            or not _weight_is_bf16_linear(down)
            or _weight_shape(combined_gate_up)
            != (
                2 * self.spec.intermediate_size,
                self.spec.hidden_size,
            )
            or _weight_shape(down)
            != (
                self.spec.hidden_size,
                self.spec.intermediate_size,
            )
        ):
            raise ValueError(
                f"TP MLP layer {layer} weight shape/dtype mismatch"
            )
        gate, up = _projection_parts(
            combined_gate_up,
            (
                self.spec.intermediate_size,
                self.spec.intermediate_size,
            ),
        )
        local_intermediate = (
            self.spec.intermediate_size // len(self.devices)
        )
        owner_device = self.devices[owner]
        gate_up_shards: list[object] = []
        down_shards: list[object] = []
        local_inputs: list[torch.Tensor] = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                gate_up_shards.append(
                    _combine_projection_parts(
                        (
                            _row_slice(
                                gate,
                                rank * local_intermediate,
                                (rank + 1) * local_intermediate,
                                device,
                            ),
                            _row_slice(
                                up,
                                rank * local_intermediate,
                                (rank + 1) * local_intermediate,
                                device,
                            ),
                        )
                    )
                )
                down_shards.append(
                    _column_slice(
                        down,
                        rank * local_intermediate,
                        (rank + 1) * local_intermediate,
                        device,
                    )
                )
                local_inputs.append(
                    torch.empty(
                        1,
                        self.spec.hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
        with torch.cuda.device(owner_device):
            source = torch.empty(
                1,
                self.spec.hidden_size,
                dtype=torch.bfloat16,
                device=owner_device,
            )
            zero = torch.zeros(
                1,
                self.spec.hidden_size,
                dtype=torch.float32,
                device=owner_device,
            )
            launch_stream = torch.cuda.Stream(device=owner_device)
            ready_event = torch.cuda.Event()
        self.layers[layer] = _GatedMLPLayer(
            owner=owner,
            source=source,
            local_inputs=local_inputs,
            gate_up=gate_up_shards,
            down=down_shards,
            contributions=[],
            zero=zero,
            launch_stream=launch_stream,
            ready_event=ready_event,
        )

    def capture(self) -> None:
        from ..fusedext import (
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )
        from .api import gated_activation

        for device in self.devices:
            torch.cuda.synchronize(device)
        for layer, state in self.layers.items():
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                state.source.zero_()
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    state.local_inputs[rank].zero_()
            rank_order = _no_owner_rank_order(self, state)
            graphs: list[torch.cuda.CUDAGraph] = []
            contributions: list[torch.Tensor] = []
            events: list[torch.cuda.Event] = []
            ordered_streams: list[torch.cuda.Stream] = []
            source_event = torch.cuda.Event()
            with torch.cuda.device(owner_device):
                source_event.record(torch.cuda.current_stream(owner_device))
                torch.cuda.synchronize(owner_device)
            for rank in rank_order:
                device = self.devices[rank]
                stream = self.streams[rank]

                def execute_rank() -> torch.Tensor:
                    if (
                        not self.hidden_mode
                        and not tp_peer_copy_fused(
                            state.source,
                            state.local_inputs[rank],
                        )
                    ):
                        raise RuntimeError(
                            "TP gated MLP input dispatch was rejected"
                        )
                    projected = _linear(
                        state.local_inputs[rank],
                        state.gate_up[rank],
                        torch.bfloat16,
                    )
                    gate, up = projected.chunk(2, dim=-1)
                    activated = gated_activation(
                        gate,
                        up,
                        activation=self.spec.activation,
                        beta=self.spec.activation_beta,
                        linear_beta=self.spec.activation_linear_beta,
                        limit=self.spec.activation_limit,
                        output=gate,
                    )
                    if activated is None:
                        raise RuntimeError(
                            "TP gated MLP activation was rejected"
                        )
                    return _linear(
                        activated,
                        state.down[rank],
                    ).float()

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    execute_rank()
                    stream.synchronize()
                    event = torch.cuda.Event()
                    graph = _new_cuda_graph()
                    with torch.cuda.graph(graph, stream=stream):
                        contribution = execute_rank()
                    _instantiate_retained_graph(graph)
                    event.record(stream)
                    stream.synchronize()
                graphs.append(graph)
                contributions.append(contribution)
                events.append(event)
                ordered_streams.append(stream)
            state.contributions = contributions
            state.graphs = graphs
            state.events = events
            state.source_event = source_event
            state.input_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.input_events.append(event)
            state.output_replicas = []
            state.output_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    state.output_replicas.append(
                        torch.empty(
                            1,
                            self.spec.hidden_size,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                    )
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.output_events.append(event)
            state.graph_batch = make_tp_graph_launch_batch(
                [
                    int(self.devices[rank].index)
                    for rank in rank_order
                ],
                graphs,
                ordered_streams,
                events,
                source_event,
            )

    def run(self, layer: int, value: torch.Tensor) -> torch.Tensor:
        if self.hidden_mode:
            hidden = self.input_hidden(layer)
            state = self.layers[layer]
            hidden.copy_from_owner(value, state.owner)
            output = self.run_hidden(layer, hidden)
            owner_output = output.local(state.owner)
            with torch.cuda.device(self.devices[state.owner]):
                torch.cuda.current_stream().wait_event(
                    output.ready_events[state.owner]
                )
            return owner_output
        output = self.start(layer, value)
        return self.finish(layer, output)

    def input_buffer(self, layer: int) -> torch.Tensor:
        return self.layers[layer].source

    def input_hidden(self, layer: int):
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.input_events is None:
            raise RuntimeError("TP gated MLP graphs are not captured")
        return TPHidden(
            self.devices,
            tuple(state.local_inputs),
            tuple(state.input_events),
        )

    def compose_mlp_prelude(
        self,
        layer: int,
        source,
        attention,
        prefix_output,
        residual,
        active_rows: int,
        projections,
        norm_weights,
        post_norm_weights,
        workspaces,
        eps: float,
        *,
        boundary: bool,
    ) -> None:
        _compose_mlp_prelude(
            self,
            layer,
            source,
            attention,
            prefix_output,
            residual,
            active_rows,
            projections,
            norm_weights,
            post_norm_weights,
            workspaces,
            eps,
            boundary,
        )

    def compose_owner_branch(
        self,
        layer: int,
        owner_graph: torch.cuda.CUDAGraph,
    ) -> None:
        """Run shared MLP and one owner-local child in parallel."""
        from ..fusedext import make_tp_graph_dag_batch

        state = self.layers[layer]
        if (
            state.graphs is None
            or state.events is None
            or state.source_event is None
            or state.composed_prefix_graphs is None
        ):
            raise RuntimeError(
                "owner branch requires a composed gated-MLP prelude"
            )
        rank_order = _no_owner_rank_order(self, state)
        graph_stages = []
        for ordered_rank, rank in enumerate(rank_order):
            parallel = [state.graphs[ordered_rank]]
            if rank == state.owner:
                parallel.append(owner_graph)
            graph_stages.append(
                [
                    [state.composed_prefix_graphs[rank]],
                    parallel,
                ]
            )
        state.graph_batch = make_tp_graph_dag_batch(
            [
                int(self.devices[rank].index)
                for rank in rank_order
            ],
            graph_stages,
            [self.streams[rank] for rank in rank_order],
            list(state.events),
            state.source_event,
        )

    def compose_rank_parallel_branch(
        self,
        layer: int,
        branch_graphs: tuple[torch.cuda.CUDAGraph, ...],
    ) -> None:
        """Run one additional fixed-address graph beside each rank MLP.

        The normalization/residual prelude remains the sole first stage.
        Afterwards both branches consume the same rank-local hidden and run
        concurrently.  This is a graph capability, not a model-specific MoE
        path.
        """
        from ..fusedext import make_tp_graph_dag_batch

        state = self.layers[layer]
        if (
            state.graphs is None
            or state.events is None
            or state.source_event is None
            or state.composed_prefix_graphs is None
            or len(branch_graphs) != len(self.devices)
        ):
            raise RuntimeError(
                "rank-parallel branch requires one retained graph per rank"
            )
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch = make_tp_graph_dag_batch(
            [
                int(self.devices[rank].index)
                for rank in rank_order
            ],
            [
                [
                    [state.composed_prefix_graphs[rank]],
                    [
                        state.graphs[ordered_rank],
                        branch_graphs[rank],
                    ],
                ]
                for ordered_rank, rank in enumerate(rank_order)
            ],
            [self.streams[rank] for rank in rank_order],
            list(state.events),
            state.source_event,
        )

    def output_hidden(self, layer: int):
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.output_replicas is None or state.output_events is None:
            raise RuntimeError("TP gated MLP outputs are unavailable")
        return TPHidden(
            self.devices,
            tuple(state.output_replicas),
            tuple(state.output_events),
        )

    def run_hidden(self, layer: int, hidden, output=None):
        state = self.layers[layer]
        if state.graph_batch is None or not self.hidden_mode:
            raise RuntimeError(
                "TP gated MLP TPHidden graph is not captured"
            )
        if output is None:
            output = self.output_hidden(layer)
        if (
            tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.hidden_size))
            or output.shape != hidden.shape
            or hidden.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
            or hidden.ready_events is None
            or output.ready_events is None
        ):
            raise ValueError("TP gated MLP TPHidden layout mismatch")
        expected_addresses = (
            state.composed_input_addresses
            if state.composed_input_addresses is not None
            else tuple(
                item.data_ptr() for item in state.local_inputs
            )
        )
        if hidden.fixed_addresses != expected_addresses:
            raise ValueError(
                "TP gated MLP input must use captured fixed addresses: "
                f"layer={layer}, expected={expected_addresses}, "
                f"actual={hidden.fixed_addresses}"
            )
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_all_rank_from_events(
            [
                hidden.ready_events[rank].cuda_event
                for rank in rank_order
            ],
            state.contributions,
            list(output.replicas),
            [
                event.cuda_event
                for event in output.ready_events
            ],
        )
        return output

    def launch_partials(self, layer: int, hidden):
        """Launch a shared/Dense branch without a hidden collective."""
        from .hidden import TPPartials

        state = self.layers[layer]
        if state.graph_batch is None or not self.hidden_mode:
            raise RuntimeError(
                "TP gated MLP partial graph is not captured"
            )
        if (
            tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.hidden_size))
            or hidden.dtype != torch.bfloat16
            or hidden.ready_events is None
            or state.events is None
        ):
            raise ValueError("TP gated MLP partial input mismatch")
        expected_addresses = (
            state.composed_input_addresses
            if state.composed_input_addresses is not None
            else tuple(
                item.data_ptr() for item in state.local_inputs
            )
        )
        if hidden.fixed_addresses != expected_addresses:
            raise ValueError(
                "TP gated MLP partial input must use fixed addresses"
            )
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_from_events(
            [
                hidden.ready_events[rank].cuda_event
                for rank in rank_order
            ]
        )
        return TPPartials(
            tuple(self.devices[rank] for rank in rank_order),
            tuple(state.contributions),
            tuple(state.events),
        )

    def launch_partials_tp1(self, layer: int, hidden) -> torch.Tensor:
        """Run a width-one fixed graph on the caller stream without events."""
        state = self.layers[layer]
        if len(self.devices) != 1 or state.graph_batch is None:
            raise RuntimeError("TP1 gated MLP requires one captured rank")
        expected_addresses = (
            state.composed_input_addresses
            if state.composed_input_addresses is not None
            else tuple(item.data_ptr() for item in state.local_inputs)
        )
        if hidden.fixed_addresses != expected_addresses:
            raise ValueError("TP1 gated MLP input address mismatch")
        state.graph_batch.launch_tp1()
        return state.contributions[0]

    def run_prepared(self, layer: int) -> torch.Tensor:
        if self.hidden_mode:
            state = self.layers[layer]
            return self.run(layer, state.source)
        output = self.start_prepared(layer)
        return self.finish(layer, output)

    def start(self, layer: int, value: torch.Tensor) -> torch.Tensor:
        """Launch one MLP branch on its persistent auxiliary stream.

        The caller may execute an independent branch on the owner default
        stream before calling :meth:`finish`.  Inputs, graphs and result
        buffers remain fixed-size; this only changes scheduling.
        """
        state = self.layers[layer]
        if self.hidden_mode:
            if state.pending_output is not None:
                raise RuntimeError(
                    "TP gated MLP layer already has pending work"
                )
            hidden = self.input_hidden(layer)
            hidden.copy_from_owner(value, state.owner)
            output = self.run_hidden(layer, hidden)
            state.pending_output = output.local(state.owner)
            return state.pending_output
        if state.graph_batch is None:
            raise RuntimeError("TP gated MLP graphs are not captured")
        if (
            state.launch_stream is None
            or state.ready_event is None
        ):
            raise RuntimeError("TP gated MLP async state is not initialized")
        if state.pending_output is not None:
            raise RuntimeError("TP gated MLP layer already has pending work")
        owner_device = self.devices[state.owner]
        if value.device != owner_device:
            raise ValueError("TP gated MLP input is not on its owner rank")
        with torch.cuda.device(owner_device):
            state.source.copy_(value)
        return self.start_prepared(layer)

    def start_prepared(self, layer: int) -> torch.Tensor:
        """Launch using data already written into the fixed source buffer."""
        state = self.layers[layer]
        if self.hidden_mode:
            return self.start(layer, state.source)
        if state.graph_batch is None:
            raise RuntimeError("TP gated MLP graphs are not captured")
        if (
            state.launch_stream is None
            or state.ready_event is None
        ):
            raise RuntimeError("TP gated MLP async state is not initialized")
        if state.pending_output is not None:
            raise RuntimeError("TP gated MLP layer already has pending work")
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            current = torch.cuda.current_stream(owner_device)
            state.launch_stream.wait_stream(current)
            with torch.cuda.stream(state.launch_stream):
                output = state.graph_batch.launch_reduce(
                    state.contributions,
                    state.zero,
                )
                state.ready_event.record(state.launch_stream)
        state.pending_output = output
        return output

    def finish(
        self,
        layer: int,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Join an MLP branch previously launched by :meth:`start`."""
        state = self.layers[layer]
        if self.hidden_mode:
            if (
                state.pending_output is None
                or state.pending_output.data_ptr() != output.data_ptr()
                or state.output_events is None
            ):
                raise RuntimeError(
                    "TP gated MLP layer has no matching hidden work"
                )
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                torch.cuda.current_stream(owner_device).wait_event(
                    state.output_events[state.owner]
                )
            state.pending_output = None
            return output
        if (
            state.ready_event is None
            or state.pending_output is None
            or state.pending_output.data_ptr() != output.data_ptr()
        ):
            raise RuntimeError("TP gated MLP layer has no matching work")
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            torch.cuda.current_stream(owner_device).wait_event(
                state.ready_event
            )
        state.pending_output = None
        return output


@dataclass(frozen=True)
class ReplicatedLinearSpec:
    """Capability description for a small rank-local replicated Linear."""

    in_features: int
    out_features: int
    input_dtype: torch.dtype = torch.bfloat16
    output_dtype: torch.dtype = torch.float32


@dataclass
class _ReplicatedLinearLayer:
    weights: list[object]
    local_inputs: list[torch.Tensor]
    outputs: list[torch.Tensor]
    streams: list[torch.cuda.Stream]
    events: list[torch.cuda.Event]
    source_event: torch.cuda.Event
    bound_input_hidden: object | None = None
    graph_batch: object | None = None
    graphs: list[torch.cuda.CUDAGraph] | None = None


class TensorParallelReplicatedLinear:
    """Run one compact Linear independently on every TP rank.

    This is for small projections whose output is consumed locally before the
    next large collective.  Replicating a compact Router avoids splitting a
    single-token GEMV into tiny shards and removes the logits collective.
    """

    def __init__(self, devices, spec: ReplicatedLinearSpec) -> None:
        if not devices:
            raise ValueError("replicated linear requires at least one rank")
        self.devices = tuple(torch.device(device) for device in devices)
        self.spec = spec
        self.layers: dict[int, _ReplicatedLinearLayer] = {}

    def add_layer(self, layer: int, weight) -> None:
        if layer in self.layers:
            raise ValueError(
                f"replicated linear layer {layer} is already registered"
            )
        if _weight_shape(weight) != (
            self.spec.out_features,
            self.spec.in_features,
        ):
            raise ValueError(
                f"replicated linear layer {layer} weight shape mismatch"
            )
        weights: list[object] = []
        inputs: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        streams: list[torch.cuda.Stream] = []
        events: list[torch.cuda.Event] = []
        for device in self.devices:
            with torch.cuda.device(device):
                weights.append(
                    _row_slice(
                        weight, 0, self.spec.out_features, device
                    )
                )
                inputs.append(torch.empty(
                    1, self.spec.in_features,
                    dtype=self.spec.input_dtype, device=device,
                ))
                outputs.append(torch.empty(
                    1, self.spec.out_features,
                    dtype=self.spec.output_dtype, device=device,
                ))
                streams.append(torch.cuda.Stream(device=device))
                events.append(torch.cuda.Event())
        with torch.cuda.device(self.devices[0]):
            source_event = torch.cuda.Event()
            source_event.record(torch.cuda.current_stream(self.devices[0]))
        self.layers[layer] = _ReplicatedLinearLayer(
            weights, inputs, outputs, streams, events, source_event,
        )

    def bind_input_hidden(self, layer: int, hidden) -> None:
        state = self.layers[layer]
        if state.graph_batch is not None:
            raise RuntimeError(
                "replicated linear input must be bound before capture"
            )
        if (
            tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.in_features))
            or hidden.dtype != self.spec.input_dtype
            or hidden.ready_events is None
        ):
            raise ValueError("replicated linear TPHidden layout mismatch")
        state.local_inputs = list(hidden.replicas)
        state.bound_input_hidden = hidden

    def capture(self) -> None:
        from ..fusedext import make_tp_graph_launch_batch
        from .api import linear

        for device in self.devices:
            torch.cuda.synchronize(device)
        for state in self.layers.values():
            if state.bound_input_hidden is None:
                raise RuntimeError(
                    "replicated linear requires a fixed TPHidden binding"
                )
            graphs: list[torch.cuda.CUDAGraph] = []
            for rank, device in enumerate(self.devices):
                stream = state.streams[rank]

                def execute_rank(rank_index: int = rank) -> None:
                    linear(
                        state.local_inputs[rank_index],
                        state.weights[rank_index],
                        output=state.outputs[rank_index],
                    )

                with torch.cuda.device(device), torch.cuda.stream(stream):
                    execute_rank()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph, stream=stream):
                        execute_rank()
                    graph.instantiate()
                    stream.synchronize()
                    graphs.append(graph)
                    state.events[rank].record(stream)
            with torch.cuda.device(self.devices[0]):
                state.graph_batch = make_tp_graph_launch_batch(
                    [int(device.index) for device in self.devices],
                    graphs,
                    state.streams,
                    state.events,
                    state.source_event,
                )
                state.graphs = graphs

    def output_hidden(self, layer: int):
        from .hidden import TPHidden

        state = self.layers[layer]
        return TPHidden(
            self.devices, tuple(state.outputs), tuple(state.events)
        )

    def run_hidden(self, layer: int, hidden):
        state = self.layers[layer]
        if state.graph_batch is None or state.bound_input_hidden is None:
            raise RuntimeError("replicated linear graph is not captured")
        if (
            hidden.fixed_addresses
            != state.bound_input_hidden.fixed_addresses
            or hidden.ready_events is None
        ):
            raise ValueError(
                "replicated linear input must use captured fixed addresses"
            )
        state.graph_batch.launch_from_events(
            [event.cuda_event for event in hidden.ready_events]
        )
        return self.output_hidden(layer)

    def run_tp1(self, layer: int, hidden) -> torch.Tensor:
        """Run one replicated Linear graph without an event boundary."""
        state = self.layers[layer]
        if len(self.devices) != 1 or state.graph_batch is None:
            raise RuntimeError("TP1 replicated Linear requires one rank")
        if hidden.fixed_addresses != (
            state.local_inputs[0].data_ptr(),
        ):
            raise ValueError("TP1 replicated Linear input address mismatch")
        state.graph_batch.launch_tp1()
        return state.outputs[0]


@dataclass(frozen=True)
class PackedMoEFinalizerSpec:
    hidden_size: int
    dtype: torch.dtype = torch.bfloat16


@dataclass(frozen=True)
class RoutePackedPlanSpec:
    """Model-independent routing math for a packed TP expert graph."""

    scoring_func: str
    top_k: int
    normalize: bool
    scaling: float
    n_group: int = 1
    topk_group: int = 1

    def normalized(self) -> "RoutePackedPlanSpec":
        return RoutePackedPlanSpec(
            scoring_func=self.scoring_func.strip().lower(),
            top_k=int(self.top_k),
            normalize=bool(self.normalize),
            scaling=float(self.scaling),
            n_group=int(self.n_group),
            topk_group=int(self.topk_group),
        )


class TensorParallelRoutePackedPlan:
    """Compose registered Top-K routing with all-rank packed experts.

    This is a public scheduling component keyed only by routing mathematics
    and TP capabilities.  Model runtimes provide fixed logits, correction,
    mask and output buffers; model family names never participate in the
    selection or execution path.
    """

    def __init__(
        self,
        devices,
        spec: RoutePackedPlanSpec,
        expert_executor,
        logits_by_layer,
        corrections_by_layer,
        masks_by_layer,
        route_buffers_by_layer,
        *,
        layers=None,
    ) -> None:
        if not devices:
            raise ValueError("route/packed plan requires TP ranks")
        self.devices = tuple(torch.device(device) for device in devices)
        self.spec = spec.normalized()
        if self.spec.top_k <= 0:
            raise ValueError("route/packed plan top_k must be positive")
        if self.spec.n_group <= 0 or self.spec.topk_group <= 0:
            raise ValueError("route/packed plan groups must be positive")
        if not hasattr(expert_executor, "compose_route_topk"):
            raise TypeError(
                "packed expert executor lacks route composition capability"
            )
        executor_devices = tuple(
            torch.device(device)
            for device in getattr(expert_executor, "devices", self.devices)
        )
        if executor_devices != self.devices:
            raise ValueError("route/packed executor TP layout mismatch")
        selected = (
            tuple(sorted(int(layer) for layer in logits_by_layer))
            if layers is None
            else tuple(sorted({int(layer) for layer in layers}))
        )
        if not selected:
            raise ValueError("route/packed plan requires at least one layer")
        required = set(selected)
        mappings = (
            logits_by_layer,
            corrections_by_layer,
            masks_by_layer,
            route_buffers_by_layer,
        )
        if any(required - set(mapping) for mapping in mappings):
            raise ValueError("route/packed plan has incomplete layer bindings")
        self.layers = frozenset(selected)
        self.expert_executor = expert_executor
        expert_executor.compose_route_topk(
            logits_by_layer,
            corrections_by_layer,
            masks_by_layer,
            route_buffers_by_layer,
            scoring_func=self.spec.scoring_func,
            top_k=self.spec.top_k,
            normalize=self.spec.normalize,
            scaling=self.spec.scaling,
            n_group=self.spec.n_group,
            topk_group=self.spec.topk_group,
            layers=selected,
        )

    @property
    def component_key(self) -> tuple:
        """Stable public capability key used by configs and diagnostics."""
        return (
            "tensor_parallel_route_packed",
            self.spec.scoring_func,
            self.spec.top_k,
            self.spec.normalize,
            self.spec.n_group,
            self.spec.topk_group,
            len(self.devices),
        )

    def handles(self, layer: int) -> bool:
        return int(layer) in self.layers


@dataclass
class _PackedMoEFinalizerLayer:
    graph_batch: object
    expert_contributions: tuple[torch.Tensor, ...]
    zero_residual: object
    routed_workspaces: list[torch.Tensor]
    shared_workspaces: list[torch.Tensor]
    output: object


class TensorParallelAllRankCollective:
    """Publish fixed FP32 Row-TP partials to every rank in one host call."""

    def __init__(self, devices) -> None:
        if not devices:
            raise ValueError("all-rank collective requires TP ranks")
        self.devices = tuple(torch.device(device) for device in devices)

    def reduce_from_events(self, launch_batch, partials, output):
        if (
            tuple(partials.devices) != self.devices
            or tuple(output.devices) != self.devices
            or partials.shape != output.shape
            or partials.ready_events is None
            or output.ready_events is None
        ):
            raise ValueError("all-rank collective TP layout mismatch")
        if len(self.devices) == 1:
            from ..fusedext import tp_all_rank_reduce_fused

            result = tp_all_rank_reduce_fused(
                [partials.contributions[0]],
                [output.replicas[0]],
            )
            if result is None:
                raise RuntimeError("TP1 publication was rejected")
            with torch.cuda.device(self.devices[0]):
                output.ready_events[0].record(torch.cuda.current_stream())
            return output
        launch_batch.reduce_all_rank_many_from_events(
            [event.cuda_event for event in partials.ready_events],
            [list(partials.contributions)],
            [list(output.replicas)],
            [event.cuda_event for event in output.ready_events],
        )
        return output


class ReplicatedSubgroupTensorParallel:
    """Run one logical Row-TP operator in every contiguous subgroup.

    Every visible rank participates: each subgroup owns the same logical
    projection shards and independently publishes the same complete hidden.
    This differs from owner-group scheduling, where only one subgroup runs a
    layer.  Large packed operators can therefore retain full-width TP while
    latency-bound batch-1 projections use a smaller useful shard width.
    """

    def __init__(self, devices, group_size: int) -> None:
        self.devices = tuple(torch.device(device) for device in devices)
        self.group_size = int(group_size)
        if (
            not self.devices
            or self.group_size <= 0
            or self.group_size > len(self.devices)
            or len(self.devices) % self.group_size
        ):
            raise ValueError(
                "replicated TP subgroup size must divide rank count"
            )
        self.rank_groups = tuple(
            tuple(range(start, start + self.group_size))
            for start in range(0, len(self.devices), self.group_size)
        )

    def group_index(self, rank: int) -> int:
        rank = int(rank)
        if not 0 <= rank < len(self.devices):
            raise IndexError("TP rank is outside replicated subgroups")
        return rank // self.group_size

    def local_rank(self, rank: int) -> int:
        return int(rank) % self.group_size

    @property
    def component_key(self) -> tuple:
        return (
            "replicated_subgroup_tensor_parallel",
            len(self.devices),
            self.group_size,
        )

    def reduce_from_events(self, partials, output):
        """Reduce each subgroup independently into all of its local ranks."""
        from .hidden import TPPartials, TPHidden

        if not isinstance(partials, TPPartials) or not isinstance(
            output, TPHidden
        ):
            raise TypeError("replicated subgroup reduction needs TP buffers")
        if (
            tuple(partials.devices) != self.devices
            or tuple(output.devices) != self.devices
            or partials.shape != output.shape
            or output.ready_events is None
        ):
            raise ValueError("replicated subgroup TP layout mismatch")
        from ..fusedext import tp_all_rank_reduce_from_events_fused

        for ranks in self.rank_groups:
            result = tp_all_rank_reduce_from_events_fused(
                [partials.contributions[rank] for rank in ranks],
                [partials.ready_events[rank] for rank in ranks],
                [output.replicas[rank] for rank in ranks],
                [output.ready_events[rank] for rank in ranks],
            )
            if result is None:
                raise RuntimeError(
                    "replicated subgroup event reduction was rejected"
                )
        return output


class TensorParallelPackedMoEFinalizer:
    """Publish packed routed + shared Row-TP partials once per layer."""

    def __init__(self, devices, spec, expert_executor) -> None:
        if not devices:
            raise ValueError("packed MoE finalizer requires TP ranks")
        self.devices = tuple(torch.device(device) for device in devices)
        self.spec = spec
        self.expert_executor = expert_executor
        self.layers: dict[int, _PackedMoEFinalizerLayer] = {}

    def add_layer(self, layer: int) -> None:
        from .hidden import TPHidden

        if layer in self.layers:
            raise ValueError(
                f"packed MoE finalizer layer {layer} is already registered"
            )
        graph_batch, contributions, _ = (
            self.expert_executor.fixed_layer_plan(layer)
        )
        shape = (1, self.spec.hidden_size)
        if any(
            item.numel() != self.spec.hidden_size
            for item in contributions
        ):
            raise ValueError(
                "packed expert partial width does not match finalizer"
            )
        zero = TPHidden.empty(
            self.devices, shape, dtype=self.spec.dtype
        )
        for rank, replica in enumerate(zero.replicas):
            with torch.cuda.device(self.devices[rank]):
                replica.zero_()
                zero.ready_events[rank].record(
                    torch.cuda.current_stream(self.devices[rank])
                )
        output = TPHidden.empty(
            self.devices, shape, dtype=self.spec.dtype
        )
        routed_workspaces = []
        shared_workspaces = []
        for device in self.devices:
            with torch.cuda.device(device):
                routed_workspaces.append(torch.empty(
                    shape, dtype=self.spec.dtype, device=device,
                ))
                shared_workspaces.append(torch.empty(
                    shape, dtype=self.spec.dtype, device=device,
                ))
        self.layers[layer] = _PackedMoEFinalizerLayer(
            graph_batch,
            tuple(contributions),
            zero,
            routed_workspaces,
            shared_workspaces,
            output,
        )

    def output_hidden(self, layer: int):
        return self.layers[layer].output

    def launch_batch(self, layer: int):
        """Expose the common all-rank scheduler, not packed internals."""
        return self.layers[layer].graph_batch

    def run_from_events(self, layer, input_events, shared_partials):
        state = self.layers[layer]
        if (
            len(input_events) != len(self.devices)
            or tuple(shared_partials.devices) != self.devices
            or shared_partials.shape
            != torch.Size((1, self.spec.hidden_size))
            or state.output.ready_events is None
            or state.zero_residual.ready_events is None
        ):
            raise ValueError("packed MoE finalizer TP layout mismatch")
        state.graph_batch.launch_moe_all_rank_from_events(
            [event.cuda_event for event in input_events],
            list(state.expert_contributions),
            list(shared_partials.contributions),
            [event.cuda_event for event in shared_partials.ready_events],
            list(state.zero_residual.replicas),
            [
                event.cuda_event
                for event in state.zero_residual.ready_events
            ],
            state.routed_workspaces,
            state.shared_workspaces,
            list(state.output.replicas),
            [event.cuda_event for event in state.output.ready_events],
        )
        return state.output

    def run_tp1(
        self,
        layer: int,
        shared_contribution: torch.Tensor,
    ) -> torch.Tensor:
        """Route/compute/finalize TP1 on one stream with no collectives."""
        from ..fusedext import tp1_moe_finalize_fused

        state = self.layers[int(layer)]
        if len(self.devices) != 1:
            raise RuntimeError("TP1 finalizer requires exactly one rank")
        state.graph_batch.launch_tp1()
        output = tp1_moe_finalize_fused(
            state.expert_contributions[0],
            shared_contribution,
            state.zero_residual.replicas[0],
            state.routed_workspaces[0],
            state.shared_workspaces[0],
            state.output.replicas[0],
        )
        if output is None:
            raise RuntimeError("TP1 fused MoE finalizer was rejected")
        return output


@dataclass(frozen=True)
class RowParallelLinearSpec:
    in_features: int
    out_features: int
    input_dtype: torch.dtype = torch.bfloat16
    weight_dtype: torch.dtype = torch.bfloat16
    output_dtype: torch.dtype | None = None
    capture_owner_dispatch: bool = False


@dataclass
class _RowParallelLinearLayer:
    owner: int
    source: torch.Tensor
    source_parts: list[torch.Tensor]
    local_inputs: list[torch.Tensor]
    weights: list[torch.Tensor]
    contributions: list[torch.Tensor]
    zero: torch.Tensor
    graphs: list[torch.cuda.CUDAGraph] | None = None
    events: list[torch.cuda.Event] | None = None
    source_event: torch.cuda.Event | None = None
    input_events: list[torch.cuda.Event] | None = None
    output_replicas: list[torch.Tensor] | None = None
    output_events: list[torch.cuda.Event] | None = None
    routed_workspaces: list[torch.Tensor] | None = None
    shared_workspaces: list[torch.Tensor] | None = None
    global_workspaces: dict[
        tuple[int, ...],
        tuple[list[torch.Tensor], list[torch.Tensor]],
    ] | None = None
    graph_batch: object | None = None
    bound_input_addresses: tuple[int, ...] | None = None
    bound_input_hidden: object | None = None
    composed_input_addresses: tuple[int, ...] | None = None


class TensorParallelRowLinear:
    """Model-independent row-parallel BF16 linear projection.

    Input columns and matching weight columns are sharded across ranks.  Each
    rank receives only its input slice and produces one FP32 partial output.
    Formal TPHidden execution publishes the sole mathematical reduction to
    every rank without selecting a hidden owner.  No rank retains a complete
    copy of the weight.
    """

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        spec: RowParallelLinearSpec,
    ) -> None:
        if not devices:
            raise ValueError("row-linear graph requires at least one rank")
        if spec.in_features % len(devices):
            raise ValueError(
                "linear input width must divide the tensor-parallel size"
            )
        self.devices = devices
        self.spec = spec
        self.output_dtype = spec.output_dtype or spec.input_dtype
        self.local_width = spec.in_features // len(devices)
        self.hidden_mode = (
            os.environ.get("CCCP_TP_HIDDEN", "0") != "0"
        )
        self.streams = [
            torch.cuda.Stream(device=device) for device in devices
        ]
        self.layers: dict[int, _RowParallelLinearLayer] = {}

    def add_layer(
        self,
        layer: int,
        owner: int,
        weight,
    ) -> None:
        if layer in self.layers:
            raise ValueError(
                f"row-parallel linear layer {layer} is already registered"
            )
        if (
            (
                self.spec.weight_dtype == torch.bfloat16
                and not _weight_is_bf16_linear(weight)
            )
            or _weight_shape(weight)
            != (self.spec.out_features, self.spec.in_features)
        ):
            raise ValueError(
                f"row-parallel linear layer {layer} weight shape/dtype "
                "mismatch"
            )
        owner_device = self.devices[owner]
        with torch.cuda.device(owner_device):
            source = torch.empty(
                1,
                self.spec.in_features,
                dtype=self.spec.input_dtype,
                device=owner_device,
            )
            source_parts = list(
                source.split(self.local_width, dim=-1)
            )
            zero = torch.zeros(
                1,
                self.spec.out_features,
                dtype=torch.float32,
                device=owner_device,
            )
        local_inputs = []
        weights = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                local_inputs.append(
                    torch.empty(
                        1,
                        self.local_width,
                        dtype=self.spec.input_dtype,
                        device=device,
                    )
                )
                weights.append(
                    _column_slice(
                        weight,
                        rank * self.local_width,
                        (rank + 1) * self.local_width,
                        device,
                    )
                )
        self.layers[layer] = _RowParallelLinearLayer(
            owner=owner,
            source=source,
            source_parts=source_parts,
            local_inputs=local_inputs,
            weights=weights,
            contributions=[],
            zero=zero,
        )

    def bind_input_hidden(self, layer: int, hidden) -> None:
        """Bind local slices of a fixed all-rank producer before capture.

        The producer remains replicated because the previous Row-TP
        collective publishes onto every rank.  This operator reads only the
        rank-local column slice directly from that replica, so there is no
        owner dispatch and no intermediate shard copy.
        """
        state = self.layers[layer]
        if state.graph_batch is not None:
            raise RuntimeError(
                "row-parallel input must be bound before graph capture"
            )
        if (
            tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.in_features))
            or hidden.dtype != self.spec.input_dtype
            or hidden.ready_events is None
        ):
            raise ValueError("row-parallel bound TPHidden layout mismatch")
        state.local_inputs = [
            hidden.replicas[rank][
                :,
                rank * self.local_width:
                (rank + 1) * self.local_width,
            ]
            for rank in range(len(self.devices))
        ]
        state.bound_input_addresses = hidden.fixed_addresses
        state.bound_input_hidden = hidden

    def input_hidden(self, layer: int):
        """Return the fixed full replicas backing local Row-TP slices."""
        hidden = self.layers[layer].bound_input_hidden
        if hidden is None:
            raise RuntimeError(
                "row-parallel linear has no bound TPHidden input"
            )
        return hidden

    def compose_normalize_prelude(
        self,
        layer: int,
        source,
        post_norm_weights,
        eps: float,
    ) -> None:
        """Fuse rank-local RMSNorm ahead of the retained Row-TP graphs."""
        _compose_normalize_prelude(
            self,
            layer,
            source,
            None,
            0,
            (),
            (),
            post_norm_weights,
            (),
            float(eps),
        )

    def capture(self) -> None:
        from ..fusedext import (
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )

        for device in self.devices:
            torch.cuda.synchronize(device)
        for state in self.layers.values():
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                state.source.zero_()
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    state.local_inputs[rank].zero_()
            rank_order = _no_owner_rank_order(self, state)
            graphs = []
            events = []
            contributions = []
            ordered_streams = []
            source_event = torch.cuda.Event()
            with torch.cuda.device(owner_device):
                source_event.record(torch.cuda.current_stream(owner_device))
                torch.cuda.synchronize(owner_device)
            for rank in rank_order:
                device = self.devices[rank]
                stream = self.streams[rank]

                def execute_rank() -> torch.Tensor:
                    if (
                        (
                            not self.hidden_mode
                            or self.spec.capture_owner_dispatch
                        )
                        and not tp_peer_copy_fused(
                            state.source_parts[rank],
                            state.local_inputs[rank],
                        )
                    ):
                        raise RuntimeError(
                            "row-parallel linear input dispatch was rejected"
                        )
                    return _linear(
                        state.local_inputs[rank],
                        state.weights[rank],
                    ).float()

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    execute_rank()
                    stream.synchronize()
                    event = torch.cuda.Event()
                    graph = _new_cuda_graph()
                    with torch.cuda.graph(graph, stream=stream):
                        contribution = execute_rank()
                    _instantiate_retained_graph(graph)
                    event.record(stream)
                    stream.synchronize()
                graphs.append(graph)
                events.append(event)
                contributions.append(contribution)
                ordered_streams.append(stream)
            state.graphs = graphs
            state.events = events
            state.contributions = contributions
            state.source_event = source_event
            state.input_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.input_events.append(event)
            state.output_replicas = []
            state.output_events = []
            state.routed_workspaces = []
            state.shared_workspaces = []
            state.global_workspaces = {}
            for device in self.devices:
                with torch.cuda.device(device):
                    state.output_replicas.append(
                        torch.empty(
                            1,
                            self.spec.out_features,
                            dtype=self.output_dtype,
                            device=device,
                        )
                    )
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.output_events.append(event)
                    state.routed_workspaces.append(
                        torch.empty(
                            1,
                            self.spec.out_features,
                            dtype=self.output_dtype,
                            device=device,
                        )
                    )
                    state.shared_workspaces.append(
                        torch.empty(
                            1,
                            self.spec.out_features,
                            dtype=self.output_dtype,
                            device=device,
                        )
                    )
            state.graph_batch = make_tp_graph_launch_batch(
                [
                    int(self.devices[rank].index)
                    for rank in rank_order
                ],
                graphs,
                ordered_streams,
                events,
                source_event,
            )

    def run(self, layer: int, value: torch.Tensor) -> torch.Tensor:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError(
                "row-parallel linear graphs are not captured"
            )
        owner_device = self.devices[state.owner]
        if (
            value.device != owner_device
            or value.dtype != self.spec.input_dtype
            or value.shape != (1, self.spec.in_features)
        ):
            raise ValueError(
                "row-parallel linear input shape/dtype/device mismatch"
            )
        with torch.cuda.device(owner_device):
            if self.hidden_mode:
                sharded = self.input_sharded(layer)
                sharded.copy_from_full(value)
                output = self.run_sharded(layer, sharded)
                owner_output = output.local(state.owner)
                torch.cuda.current_stream(owner_device).wait_event(
                    output.ready_events[state.owner]
                )
                return owner_output
            state.source.copy_(value)
            return state.graph_batch.launch_reduce(
                state.contributions,
                state.zero,
            )

    def input_buffer(self, layer: int) -> torch.Tensor:
        return self.layers[layer].source

    def input_sharded(self, layer: int):
        from .hidden import TPSharded

        state = self.layers[layer]
        if state.input_events is None:
            raise RuntimeError(
                "row-parallel linear graphs are not captured"
            )
        return TPSharded(
            self.devices,
            tuple(state.local_inputs),
            self.spec.in_features,
            tuple(state.input_events),
        )

    def bound_input_sharded(self, layer: int, hidden):
        """Expose bound input views with the producer's current events."""
        from .hidden import TPSharded

        state = self.layers[layer]
        if (
            state.bound_input_addresses is None
            or hidden.fixed_addresses != state.bound_input_addresses
            or tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.in_features))
            or hidden.dtype != self.spec.input_dtype
            or hidden.ready_events is None
        ):
            raise ValueError(
                "row-parallel input does not match its bound producer"
            )
        return TPSharded(
            self.devices,
            tuple(state.local_inputs),
            self.spec.in_features,
            tuple(hidden.ready_events),
        )

    def composed_input_sharded(self, layer: int, source):
        """Use producer events to launch a normalize→Row-TP parent graph."""
        from .hidden import TPSharded

        state = self.layers[layer]
        if (
            state.composed_input_addresses is None
            or source.fixed_addresses
            != state.composed_input_addresses
            or tuple(source.devices) != self.devices
            or source.shape
            != torch.Size((1, self.spec.in_features))
            or source.dtype != self.spec.input_dtype
            or source.ready_events is None
        ):
            raise ValueError(
                "row-parallel composed source layout mismatch"
            )
        return TPSharded(
            self.devices,
            tuple(state.local_inputs),
            self.spec.in_features,
            tuple(source.ready_events),
        )

    def output_hidden(self, layer: int):
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.output_replicas is None or state.output_events is None:
            raise RuntimeError(
                "row-parallel linear outputs are unavailable"
            )
        return TPHidden(
            self.devices,
            tuple(state.output_replicas),
            tuple(state.output_events),
        )

    def run_sharded(self, layer: int, sharded, output=None):
        state = self.layers[layer]
        if state.graph_batch is None or not self.hidden_mode:
            raise RuntimeError(
                "row-parallel sharded-input graph is not captured"
            )
        if output is None:
            output = self.output_hidden(layer)
        if (
            tuple(sharded.devices) != self.devices
            or sharded.shape
            != torch.Size((1, self.spec.in_features))
            or sharded.dtype != self.spec.input_dtype
            or output.shape
            != torch.Size((1, self.spec.out_features))
            or output.dtype != self.output_dtype
            or sharded.ready_events is None
            or output.ready_events is None
        ):
            raise ValueError("row-parallel hidden layout mismatch")
        if any(
            sharded.shards[rank].data_ptr()
            != state.local_inputs[rank].data_ptr()
            for rank in range(len(self.devices))
        ):
            raise ValueError(
                "row-parallel input must use captured fixed addresses"
            )
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_all_rank_from_events(
            [
                sharded.ready_events[rank].cuda_event
                for rank in rank_order
            ],
            state.contributions,
            list(output.replicas),
            [
                event.cuda_event
                for event in output.ready_events
            ],
        )
        return output

    def finalize_moe_full(
        self,
        layer: int,
        value: torch.Tensor,
        shared_partials,
        residual,
        output=None,
    ):
        """Dispatch a fixed owner row through the captured Row-TP graph."""
        from .hidden import TPSharded

        state = self.layers[layer]
        if (
            not self.spec.capture_owner_dispatch
            or state.source_event is None
            or value.device != self.devices[state.owner]
            or value.shape != state.source.shape
            or value.dtype != state.source.dtype
        ):
            raise ValueError(
                "full-owner MoE finalizer layout/capability mismatch"
            )
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            state.source.copy_(value)
            state.source_event.record(
                torch.cuda.current_stream(owner_device)
            )
        dispatched = TPSharded(
            self.devices,
            tuple(state.local_inputs),
            self.spec.in_features,
            tuple(state.source_event for _ in self.devices),
        )
        return self.finalize_moe(
            layer,
            dispatched,
            shared_partials,
            residual,
            output=output,
        )

    def launch_partials(self, layer: int, sharded):
        """Launch Row-TP and expose FP32 partials without reducing them."""
        from .hidden import TPPartials

        state = self.layers[layer]
        if (
            state.graph_batch is None
            or not self.hidden_mode
            or state.events is None
        ):
            raise RuntimeError(
                "row-parallel partial graph is not captured"
            )
        self._validate_sharded_input(state, sharded)
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_from_events(
            [
                sharded.ready_events[rank].cuda_event
                for rank in rank_order
            ]
        )
        return TPPartials(
            tuple(self.devices[rank] for rank in rank_order),
            tuple(state.contributions),
            tuple(state.events),
        )

    def last_partials(self, layer: int):
        """Expose the captured fixed-address partials for diagnostics."""
        from .hidden import TPPartials

        state = self.layers[layer]
        if state.events is None:
            raise RuntimeError(
                "row-parallel partial graph is not captured"
            )
        rank_order = _no_owner_rank_order(self, state)
        return TPPartials(
            tuple(self.devices[rank] for rank in rank_order),
            tuple(state.contributions),
            tuple(state.events),
        )

    def finalize_moe(
        self,
        layer: int,
        sharded,
        shared_partials,
        residual,
        output=None,
    ):
        """Launch routed Row-TP then perform the sole MoE hidden collective."""
        state = self.layers[layer]
        if state.graph_batch is None or not self.hidden_mode:
            raise RuntimeError("TP MoE finalizer graph is not captured")
        self._validate_sharded_input(state, sharded)
        if output is None:
            output = self.output_hidden(layer)
        if (
            shared_partials.shape
            != torch.Size((1, self.spec.out_features))
            or tuple(shared_partials.devices)
            != tuple(
                self.devices[rank]
                for rank in _no_owner_rank_order(self, state)
            )
            or tuple(residual.devices) != tuple(output.devices)
            or residual.shape
            != torch.Size((1, self.spec.out_features))
            or output.shape != residual.shape
            or residual.dtype != self.spec.input_dtype
            or output.dtype != self.spec.input_dtype
            or residual.ready_events is None
            or output.ready_events is None
            or state.global_workspaces is None
        ):
            raise ValueError("TP MoE finalizer hidden layout mismatch")
        output_key = tuple(
            int(device.index) for device in output.devices
        )
        if tuple(output.devices) == self.devices:
            if (
                state.routed_workspaces is None
                or state.shared_workspaces is None
            ):
                raise RuntimeError(
                    "TP MoE local workspaces are unavailable"
                )
            routed_workspaces = state.routed_workspaces
            shared_workspaces = state.shared_workspaces
        else:
            workspace_pair = state.global_workspaces.get(output_key)
            if workspace_pair is None:
                routed_workspaces = []
                shared_workspaces = []
                for device in output.devices:
                    with torch.cuda.device(device):
                        routed_workspaces.append(
                            torch.empty_like(output.on_device(device))
                        )
                        shared_workspaces.append(
                            torch.empty_like(output.on_device(device))
                        )
                workspace_pair = (
                    routed_workspaces,
                    shared_workspaces,
                )
                state.global_workspaces[output_key] = workspace_pair
            routed_workspaces, shared_workspaces = workspace_pair
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_moe_all_rank_from_events(
            [
                sharded.ready_events[rank].cuda_event
                for rank in rank_order
            ],
            state.contributions,
            list(shared_partials.contributions),
            [
                event.cuda_event
                for event in shared_partials.ready_events
            ],
            list(residual.replicas),
            [
                event.cuda_event
                for event in residual.ready_events
            ],
            routed_workspaces,
            shared_workspaces,
            list(output.replicas),
            [
                event.cuda_event
                for event in output.ready_events
            ],
        )
        return output

    def _validate_sharded_input(self, state, sharded) -> None:
        if (
            tuple(sharded.devices) != self.devices
            or sharded.shape
            != torch.Size((1, self.spec.in_features))
            or sharded.dtype != self.spec.input_dtype
            or sharded.ready_events is None
        ):
            raise ValueError("row-parallel sharded input mismatch")
        if any(
            sharded.shards[rank].data_ptr()
            != state.local_inputs[rank].data_ptr()
            for rank in range(len(self.devices))
        ):
            raise ValueError(
                "row-parallel input must use captured fixed addresses"
            )

    def run_prepared(self, layer: int) -> torch.Tensor:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError(
                "row-parallel linear graphs are not captured"
            )
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            if self.hidden_mode:
                return self.run(layer, state.source)
            return state.graph_batch.launch_reduce(
                state.contributions,
                state.zero,
            )


@dataclass(frozen=True)
class RouteDownSpec:
    hidden_size: int
    routed_hidden_size: int
    expert_count: int


@dataclass
class _RouteDownLayer:
    owner: int
    source: torch.Tensor
    source_parts: list[torch.Tensor]
    router_inputs: list[torch.Tensor]
    down_inputs: list[torch.Tensor]
    router: list[torch.Tensor]
    routed_down: list[torch.Tensor]
    router_contributions: list[torch.Tensor]
    latent_contributions: list[torch.Tensor]
    zeros: list[torch.Tensor]
    graphs: list[torch.cuda.CUDAGraph] | None = None
    events: list[torch.cuda.Event] | None = None
    source_event: torch.cuda.Event | None = None
    input_events: list[torch.cuda.Event] | None = None
    router_output_replicas: list[torch.Tensor] | None = None
    latent_output_replicas: list[torch.Tensor] | None = None
    output_events: list[torch.cuda.Event] | None = None
    graph_batch: object | None = None
    bound_input_addresses: tuple[int, ...] | None = None


class TensorParallelRouteDown:
    """Fused Column-TP Router and Row-TP routed Down projection.

    Both projections consume the same normalized hidden state.  The input is
    kept locally replicated for the Router, whose expert-output rows are
    sharded.  Routed Down consumes a hidden-width shard and produces one FP32
    partial per rank.  Router logits are assembled by one small all-rank
    reduction while routed Down is summed in the same collective launch.  In
    TPHidden mode both results are published directly onto every rank; no
    hidden owner exists in the steady-state data flow.  The implementation is
    keyed by tensor shapes rather than a model family.
    """

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        spec: RouteDownSpec,
    ) -> None:
        ranks = len(devices)
        if ranks <= 0 or spec.hidden_size % ranks:
            raise ValueError(
                "route/down hidden width must divide the TP size"
            )
        if spec.expert_count % ranks:
            raise ValueError(
                "Router expert rows must divide the TP size"
            )
        self.devices = devices
        self.spec = spec
        self.local_hidden = spec.hidden_size // ranks
        self.local_experts = spec.expert_count // ranks
        self.hidden_mode = (
            os.environ.get("CCCP_TP_HIDDEN", "0") != "0"
        )
        self.streams = [
            torch.cuda.Stream(device=device) for device in devices
        ]
        self.layers: dict[int, _RouteDownLayer] = {}

    def add_layer(
        self,
        layer: int,
        owner: int,
        router: torch.Tensor,
        routed_down,
    ) -> None:
        spec = self.spec
        if layer in self.layers:
            raise ValueError(f"route/down layer {layer} already exists")
        if (
            router.dtype != torch.float32
            or router.shape != (spec.expert_count, spec.hidden_size)
            or not _weight_is_bf16_linear(routed_down)
            or _weight_shape(routed_down)
            != (spec.routed_hidden_size, spec.hidden_size)
        ):
            raise ValueError(
                f"route/down layer {layer} weight shape/dtype mismatch"
            )
        owner_device = self.devices[owner]
        with torch.cuda.device(owner_device):
            source = torch.empty(
                1,
                spec.hidden_size,
                dtype=torch.bfloat16,
                device=owner_device,
            )
            source_parts = list(
                source.split(self.local_hidden, dim=-1)
            )
            zeros = [
                torch.zeros(
                    1,
                    width,
                    dtype=torch.float32,
                    device=owner_device,
                )
                for width in (
                    spec.expert_count,
                    spec.routed_hidden_size,
                )
            ]
        router_inputs = []
        down_inputs = []
        router_weights = []
        routed_down_weights = []
        for rank, device in enumerate(self.devices):
            hidden_start = rank * self.local_hidden
            hidden_end = hidden_start + self.local_hidden
            expert_start = rank * self.local_experts
            expert_end = expert_start + self.local_experts
            with torch.cuda.device(device):
                router_inputs.append(
                    torch.empty(
                        1,
                        spec.hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                down_inputs.append(
                    torch.empty(
                        1,
                        self.local_hidden,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                router_weights.append(
                    router[expert_start:expert_end]
                    .to(device)
                    .contiguous()
                )
                routed_down_weights.append(
                    _column_slice(
                        routed_down,
                        hidden_start,
                        hidden_end,
                        device,
                    )
                )
        self.layers[layer] = _RouteDownLayer(
            owner=owner,
            source=source,
            source_parts=source_parts,
            router_inputs=router_inputs,
            down_inputs=down_inputs,
            router=router_weights,
            routed_down=routed_down_weights,
            router_contributions=[],
            latent_contributions=[],
            zeros=zeros,
        )

    def bind_input_hidden(self, layer: int, hidden) -> None:
        """Bind one fixed all-rank producer directly before graph capture."""
        state = self.layers[layer]
        if (
            tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.hidden_size))
            or hidden.dtype != torch.bfloat16
            or hidden.ready_events is None
        ):
            raise ValueError("route/down bound TPHidden layout mismatch")
        state.router_inputs = list(hidden.replicas)
        state.down_inputs = [
            hidden.replicas[rank][
                :,
                rank * self.local_hidden:
                (rank + 1) * self.local_hidden,
            ]
            for rank in range(len(self.devices))
        ]
        state.bound_input_addresses = hidden.fixed_addresses

    def capture(self) -> None:
        from ..fusedext import (
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )

        for device in self.devices:
            torch.cuda.synchronize(device)
        for state in self.layers.values():
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                state.source.zero_()
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    state.router_inputs[rank].zero_()
                    state.down_inputs[rank].zero_()
            rank_order = (
                tuple(range(len(self.devices)))
                if self.hidden_mode
                else (
                    state.owner,
                    *(
                        rank
                        for rank in range(len(self.devices))
                        if rank != state.owner
                    ),
                )
            )
            graphs = []
            events = []
            ordered_streams = []
            router_contributions = []
            latent_contributions = []
            source_event = torch.cuda.Event()
            with torch.cuda.device(owner_device):
                source_event.record(torch.cuda.current_stream(owner_device))
                torch.cuda.synchronize(owner_device)
            for rank in rank_order:
                device = self.devices[rank]
                stream = self.streams[rank]
                expert_start = rank * self.local_experts
                expert_end = expert_start + self.local_experts
                with torch.cuda.device(device):
                    router_contribution = torch.zeros(
                        1,
                        self.spec.expert_count,
                        dtype=torch.float32,
                        device=device,
                    )

                def execute_rank():
                    if (
                        not self.hidden_mode
                        and (
                            not tp_peer_copy_fused(
                                state.source,
                                state.router_inputs[rank],
                            )
                            or not tp_peer_copy_fused(
                                state.source_parts[rank],
                                state.down_inputs[rank],
                            )
                        )
                    ):
                        raise RuntimeError(
                            "route/down input dispatch was rejected"
                        )
                    torch.mm(
                        state.router_inputs[rank].float(),
                        state.router[rank].t(),
                        out=router_contribution[
                            :, expert_start:expert_end
                        ],
                    )
                    latent_contribution = _linear(
                        state.down_inputs[rank],
                        state.routed_down[rank],
                    ).float()
                    return router_contribution, latent_contribution

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    execute_rank()
                    stream.synchronize()
                    event = torch.cuda.Event()
                    graph = _new_cuda_graph()
                    with torch.cuda.graph(graph, stream=stream):
                        outputs = execute_rank()
                    _instantiate_retained_graph(graph)
                    event.record(stream)
                    stream.synchronize()
                graphs.append(graph)
                events.append(event)
                ordered_streams.append(stream)
                router_contributions.append(outputs[0])
                latent_contributions.append(outputs[1])
            state.graphs = graphs
            state.events = events
            state.source_event = source_event
            state.router_contributions = router_contributions
            state.latent_contributions = latent_contributions
            if self.hidden_mode:
                state.input_events = []
                state.router_output_replicas = []
                state.latent_output_replicas = []
                state.output_events = []
                for device in self.devices:
                    with torch.cuda.device(device):
                        input_event = torch.cuda.Event()
                        input_event.record(
                            torch.cuda.current_stream(device)
                        )
                        state.input_events.append(input_event)
                        state.router_output_replicas.append(
                            torch.empty(
                                1,
                                self.spec.expert_count,
                                dtype=torch.float32,
                                device=device,
                            )
                        )
                        state.latent_output_replicas.append(
                            torch.empty(
                                1,
                                self.spec.routed_hidden_size,
                                dtype=torch.bfloat16,
                                device=device,
                            )
                        )
                        output_event = torch.cuda.Event()
                        output_event.record(
                            torch.cuda.current_stream(device)
                        )
                        state.output_events.append(output_event)
            state.graph_batch = make_tp_graph_launch_batch(
                [
                    int(self.devices[rank].index)
                    for rank in rank_order
                ],
                graphs,
                ordered_streams,
                events,
                source_event,
            )

    def run(
        self,
        layer: int,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError("route/down graphs are not captured")
        owner_device = self.devices[state.owner]
        if (
            value.device != owner_device
            or value.dtype != torch.bfloat16
            or value.shape != (1, self.spec.hidden_size)
        ):
            raise ValueError(
                "route/down input shape/dtype/device mismatch"
            )
        with torch.cuda.device(owner_device):
            state.source.copy_(value)
            outputs = state.graph_batch.launch_reduce_many(
                [
                    state.router_contributions,
                    state.latent_contributions,
                ],
                state.zeros,
            )
        return outputs[0], outputs[1]

    def input_sharded(self, layer: int):
        """Return the fixed per-rank hidden slices consumed by this graph."""
        from .hidden import TPSharded

        state = self.layers[layer]
        if state.input_events is None:
            raise RuntimeError(
                "route/down TPHidden input buffers are unavailable"
            )
        return TPSharded(
            self.devices,
            tuple(state.down_inputs),
            self.spec.hidden_size,
            tuple(state.input_events),
        )

    def output_hidden(self, layer: int):
        """Return replicated Router logits and routed latent on every rank."""
        from .hidden import TPHidden

        state = self.layers[layer]
        if (
            state.router_output_replicas is None
            or state.latent_output_replicas is None
            or state.output_events is None
        ):
            raise RuntimeError(
                "route/down TPHidden outputs are unavailable"
            )
        events = tuple(state.output_events)
        return (
            TPHidden(
                self.devices,
                tuple(state.router_output_replicas),
                events,
            ),
            TPHidden(
                self.devices,
                tuple(state.latent_output_replicas),
                events,
            ),
        )

    def retained_rank_graphs(
        self,
        layer: int,
    ) -> tuple[torch.cuda.CUDAGraph, ...]:
        """Return the fixed graph mapped to canonical rank order."""
        state = self.layers[layer]
        if state.graphs is None:
            raise RuntimeError("route/down retained graphs are unavailable")
        rank_order = _no_owner_rank_order(self, state)
        by_rank: list[torch.cuda.CUDAGraph | None] = [
            None
            for _ in self.devices
        ]
        for ordered_rank, rank in enumerate(rank_order):
            by_rank[rank] = state.graphs[ordered_rank]
        if any(graph is None for graph in by_rank):
            raise RuntimeError("route/down retained graph mapping is invalid")
        return tuple(by_rank)  # type: ignore[arg-type]

    def reduce_hidden_from_events(self, layer: int, ready_events):
        """Publish already-computed rank partials without relaunching graphs."""
        state = self.layers[layer]
        if (
            not self.hidden_mode
            or state.graph_batch is None
            or state.output_events is None
            or len(ready_events) != len(self.devices)
        ):
            raise RuntimeError(
                "route/down collective-only path is unavailable"
            )
        router_output, latent_output = self.output_hidden(layer)
        state.graph_batch.reduce_all_rank_many_from_events(
            [event.cuda_event for event in ready_events],
            [
                state.router_contributions,
                state.latent_contributions,
            ],
            [
                list(router_output.replicas),
                list(latent_output.replicas),
            ],
            [event.cuda_event for event in state.output_events],
        )
        return router_output, latent_output

    def run_hidden(self, layer: int, hidden):
        """Run sharded Router/Down and publish both reductions to all ranks."""
        state = self.layers[layer]
        if (
            not self.hidden_mode
            or state.graph_batch is None
            or state.input_events is None
            or state.output_events is None
        ):
            raise RuntimeError(
                "route/down all-rank graph is not captured"
            )
        sharded = self.input_sharded(layer)
        if (
            state.bound_input_addresses is not None
            and hidden.fixed_addresses == state.bound_input_addresses
        ):
            input_events = hidden.ready_events
        else:
            sharded.copy_from_replicated(hidden)
            for rank, device in enumerate(self.devices):
                hidden_rank = hidden.devices.index(device)
                with torch.cuda.device(device):
                    torch.cuda.current_stream(device).wait_event(
                        hidden.ready_events[hidden_rank]
                    )
                    state.router_inputs[rank].copy_(
                        hidden.replicas[hidden_rank]
                    )
                    state.input_events[rank].record(
                        torch.cuda.current_stream(device)
                    )
            input_events = tuple(state.input_events)
        router_output, latent_output = self.output_hidden(layer)
        state.graph_batch.launch_all_rank_many_from_events(
            [event.cuda_event for event in input_events],
            [
                state.router_contributions,
                state.latent_contributions,
            ],
            [
                list(router_output.replicas),
                list(latent_output.replicas),
            ],
            [event.cuda_event for event in state.output_events],
        )
        return router_output, latent_output


@dataclass(frozen=True)
class MoEPreludeSpec:
    hidden_size: int
    routed_hidden_size: int
    shared_intermediate_size: int
    expert_count: int
    activation: str
    activation_beta: float
    activation_linear_beta: float | None


@dataclass
class _MoEPreludeLayer:
    owner: int
    source: torch.Tensor
    local_inputs: list[torch.Tensor]
    router: list[torch.Tensor]
    routed_down: list[torch.Tensor]
    shared_gate_up: list[torch.Tensor]
    shared_down: list[torch.Tensor]
    router_contributions: list[torch.Tensor]
    latent_contributions: list[torch.Tensor]
    shared_contributions: list[torch.Tensor]
    zeros: list[torch.Tensor]
    graphs: list[torch.cuda.CUDAGraph] | None = None
    events: list[torch.cuda.Event] | None = None
    source_event: torch.cuda.Event | None = None
    graph_batch: object | None = None


class TensorParallelMoEPrelude:
    """One-broadcast TP prelude for Router, routed Down and shared MLP.

    All three branches consume the same normalized hidden state.  Capturing
    them in one rank-local Graph removes duplicate peer broadcasts and host
    launches.  The caller receives reduced FP32 router logits, routed latent
    and shared contribution; packed expert execution remains an independent
    capability and keeps its compact indices.
    """

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        spec: MoEPreludeSpec,
    ) -> None:
        ranks = len(devices)
        if ranks <= 0:
            raise ValueError("MoE prelude graph requires at least one rank")
        if (
            spec.hidden_size % ranks
            or spec.shared_intermediate_size % ranks
        ):
            raise ValueError(
                "MoE prelude hidden/intermediate widths must divide TP"
            )
        self.devices = devices
        self.spec = spec
        self.local_hidden = spec.hidden_size // ranks
        self.streams = [
            torch.cuda.Stream(device=device) for device in devices
        ]
        self.layers: dict[int, _MoEPreludeLayer] = {}

    def add_layer(
        self,
        layer: int,
        owner: int,
        router: torch.Tensor,
        routed_down,
        shared_gate_up,
        shared_down,
    ) -> None:
        spec = self.spec
        if layer in self.layers:
            raise ValueError(f"MoE prelude layer {layer} already exists")
        expected = (
            router.dtype == torch.float32
            and router.shape
            == (spec.expert_count, spec.hidden_size)
            and _weight_is_bf16_linear(routed_down)
            and _weight_shape(routed_down)
            == (spec.routed_hidden_size, spec.hidden_size)
            and _weight_is_bf16_linear(shared_gate_up)
            and _weight_shape(shared_gate_up)
            == (2 * spec.shared_intermediate_size, spec.hidden_size)
            and _weight_is_bf16_linear(shared_down)
            and _weight_shape(shared_down)
            == (spec.hidden_size, spec.shared_intermediate_size)
        )
        if not expected:
            raise ValueError(
                f"MoE prelude layer {layer} weight shape/dtype mismatch"
            )
        gate, up = _projection_parts(
            shared_gate_up,
            (
                spec.shared_intermediate_size,
                spec.shared_intermediate_size,
            ),
        )
        local_intermediate = (
            spec.shared_intermediate_size // len(self.devices)
        )
        router_parts = router.split(self.local_hidden, dim=1)
        local_inputs = []
        router_weights = []
        routed_down_weights = []
        shared_gate_up_weights = []
        shared_down_weights = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                local_inputs.append(
                    torch.empty(
                        1,
                        spec.hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                router_weights.append(
                    router_parts[rank].to(device).contiguous()
                )
                routed_down_weights.append(
                    _column_slice(
                        routed_down,
                        rank * self.local_hidden,
                        (rank + 1) * self.local_hidden,
                        device,
                    )
                )
                shared_gate_up_weights.append(
                    _combine_projection_parts(
                        (
                            _row_slice(
                                gate,
                                rank * local_intermediate,
                                (rank + 1) * local_intermediate,
                                device,
                            ),
                            _row_slice(
                                up,
                                rank * local_intermediate,
                                (rank + 1) * local_intermediate,
                                device,
                            ),
                        )
                    )
                )
                shared_down_weights.append(
                    _column_slice(
                        shared_down,
                        rank * local_intermediate,
                        (rank + 1) * local_intermediate,
                        device,
                    )
                )
        owner_device = self.devices[owner]
        with torch.cuda.device(owner_device):
            source = torch.empty(
                1,
                spec.hidden_size,
                dtype=torch.bfloat16,
                device=owner_device,
            )
            zeros = [
                torch.zeros(
                    1,
                    width,
                    dtype=torch.float32,
                    device=owner_device,
                )
                for width in (
                    spec.expert_count,
                    spec.routed_hidden_size,
                    spec.hidden_size,
                )
            ]
        self.layers[layer] = _MoEPreludeLayer(
            owner=owner,
            source=source,
            local_inputs=local_inputs,
            router=router_weights,
            routed_down=routed_down_weights,
            shared_gate_up=shared_gate_up_weights,
            shared_down=shared_down_weights,
            router_contributions=[],
            latent_contributions=[],
            shared_contributions=[],
            zeros=zeros,
        )

    def capture(self) -> None:
        from ..fusedext import (
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )
        from .api import gated_activation

        for device in self.devices:
            torch.cuda.synchronize(device)
        spec = self.spec
        for state in self.layers.values():
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                state.source.zero_()
            rank_order = _no_owner_rank_order(self, state)
            graphs = []
            events = []
            ordered_streams = []
            router_contributions = []
            latent_contributions = []
            shared_contributions = []
            source_event = torch.cuda.Event()
            with torch.cuda.device(owner_device):
                source_event.record(torch.cuda.current_stream(owner_device))
                torch.cuda.synchronize(owner_device)
            for rank in rank_order:
                device = self.devices[rank]
                stream = self.streams[rank]
                start = rank * self.local_hidden
                end = start + self.local_hidden

                def execute_rank():
                    if not tp_peer_copy_fused(
                        state.source,
                        state.local_inputs[rank],
                    ):
                        raise RuntimeError(
                            "MoE prelude input dispatch was rejected"
                        )
                    local = state.local_inputs[rank]
                    local_slice = local[:, start:end]
                    router_partial = F.linear(
                        local_slice.float(),
                        state.router[rank],
                    )
                    latent_partial = _linear(
                        local_slice,
                        state.routed_down[rank],
                    ).float()
                    projected = _linear(
                        local,
                        state.shared_gate_up[rank],
                        torch.bfloat16,
                    )
                    gate, up = projected.chunk(2, dim=-1)
                    activated = gated_activation(
                        gate,
                        up,
                        activation=spec.activation,
                        beta=spec.activation_beta,
                        linear_beta=spec.activation_linear_beta,
                        output=gate,
                    )
                    if activated is None:
                        raise RuntimeError(
                            "MoE prelude gated activation was rejected"
                        )
                    shared_partial = _linear(
                        activated,
                        state.shared_down[rank],
                    ).float()
                    return (
                        router_partial,
                        latent_partial,
                        shared_partial,
                    )

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    execute_rank()
                    stream.synchronize()
                    event = torch.cuda.Event()
                    graph = _new_cuda_graph()
                    with torch.cuda.graph(graph, stream=stream):
                        outputs = execute_rank()
                    _instantiate_retained_graph(graph)
                    event.record(stream)
                    stream.synchronize()
                graphs.append(graph)
                events.append(event)
                ordered_streams.append(stream)
                router_contributions.append(outputs[0])
                latent_contributions.append(outputs[1])
                shared_contributions.append(outputs[2])
            state.graphs = graphs
            state.events = events
            state.source_event = source_event
            state.router_contributions = router_contributions
            state.latent_contributions = latent_contributions
            state.shared_contributions = shared_contributions
            state.graph_batch = make_tp_graph_launch_batch(
                [
                    int(self.devices[rank].index)
                    for rank in rank_order
                ],
                graphs,
                ordered_streams,
                events,
                source_event,
            )

    def run(
        self,
        layer: int,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError("MoE prelude graphs are not captured")
        owner_device = self.devices[state.owner]
        if (
            value.device != owner_device
            or value.dtype != torch.bfloat16
            or value.shape != (1, self.spec.hidden_size)
        ):
            raise ValueError("MoE prelude input shape/dtype/device mismatch")
        with torch.cuda.device(owner_device):
            state.source.copy_(value)
            outputs = state.graph_batch.launch_reduce_many(
                [
                    state.router_contributions,
                    state.latent_contributions,
                    state.shared_contributions,
                ],
                state.zeros,
            )
        return outputs[0], outputs[1], outputs[2]


@dataclass(frozen=True)
class KDASpec:
    hidden_size: int
    heads: int
    head_dim: int
    gate_rank: int
    rms_eps: float
    gate_lower_bound: float
    conv_history: int


@dataclass
class _KDALayer:
    owner: int
    source: torch.Tensor
    local_inputs: list[torch.Tensor]
    input_projection: list[torch.Tensor]
    gate_projection: list[torch.Tensor]
    conv_weights: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]
    a_log: list[torch.Tensor]
    dt_bias: list[torch.Tensor]
    norm_weight: list[torch.Tensor]
    output_projection: list[torch.Tensor]
    recurrent_state: list[torch.Tensor]
    conv_state: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]
    workspaces: list[torch.Tensor]
    recurrent_outputs: list[torch.Tensor]
    contributions: list[torch.Tensor]
    zero: torch.Tensor
    graphs: list[torch.cuda.CUDAGraph] | None = None
    events: list[torch.cuda.Event] | None = None
    source_event: torch.cuda.Event | None = None
    input_events: list[torch.cuda.Event] | None = None
    output_replicas: list[torch.Tensor] | None = None
    output_events: list[torch.cuda.Event] | None = None
    graph_batch: object | None = None
    composed_input_addresses: tuple[int, ...] | None = None


class TensorParallelKDA:
    """Head-parallel recurrent attention selected by a KDA config."""

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        spec: KDASpec,
    ) -> None:
        if not devices or spec.heads % len(devices):
            raise ValueError("KDA heads must divide the TP size")
        self.devices = devices
        self.spec = spec
        self.local_heads = spec.heads // len(devices)
        self.local_width = self.local_heads * spec.head_dim
        self.hidden_mode = (
            os.environ.get("CCCP_TP_HIDDEN", "0") != "0"
        )
        self.streams = [
            torch.cuda.Stream(device=device) for device in devices
        ]
        self.layers: dict[int, _KDALayer] = {}

    def add_layer(
        self,
        layer: int,
        owner: int,
        combined_input,
        gate_projection,
        conv_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        output_projection,
    ) -> None:
        spec = self.spec
        total_width = spec.heads * spec.head_dim
        q, k, v, g, gate_a, beta = _projection_parts(
            combined_input,
            (
                total_width,
                total_width,
                total_width,
                total_width,
                spec.gate_rank,
                spec.heads,
            ),
        )
        conv_parts = [
            weight.chunk(len(self.devices), dim=0)
            for weight in conv_weights
        ]
        dt_parts = dt_bias.chunk(len(self.devices), dim=0)
        local_inputs: list[torch.Tensor] = []
        input_weights: list[torch.Tensor] = []
        gate_weights: list[torch.Tensor] = []
        local_conv_weights = []
        local_a_log = []
        local_dt_bias = []
        local_norm_weight = []
        local_output_projection = []
        recurrent_state = []
        conv_state = []
        workspaces = []
        recurrent_outputs = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                local_inputs.append(
                    torch.empty(
                        1,
                        spec.hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                input_weights.append(
                    _combine_projection_parts(
                        (
                            *(
                                _row_slice(
                                    value,
                                    rank * self.local_width,
                                    (rank + 1) * self.local_width,
                                    device,
                                )
                                for value in (q, k, v, g)
                            ),
                            (
                                gate_a.to(device)
                                if isinstance(gate_a, BlockFP8Weight)
                                else gate_a.to(device).contiguous()
                            ),
                            _row_slice(
                                beta,
                                rank * self.local_heads,
                                (rank + 1) * self.local_heads,
                                device,
                            ),
                        )
                    )
                )
                gate_weights.append(
                    _row_slice(
                        gate_projection,
                        rank * self.local_width,
                        (rank + 1) * self.local_width,
                        device,
                    )
                )
                local_conv_weights.append(
                    tuple(
                        parts[rank].to(device).contiguous()
                        for parts in conv_parts
                    )
                )
                local_a_log.append(
                    _head_slice(
                        a_log,
                        rank,
                        self.local_heads,
                    )
                    .to(device)
                    .contiguous()
                )
                local_dt_bias.append(
                    dt_parts[rank].to(device).contiguous()
                )
                local_norm_weight.append(
                    norm_weight.to(device).contiguous()
                )
                local_output_projection.append(
                    _column_slice(
                        output_projection,
                        rank * self.local_width,
                        (rank + 1) * self.local_width,
                        device,
                    )
                )
                recurrent_state.append(
                    torch.zeros(
                        self.local_heads,
                        spec.head_dim,
                        spec.head_dim,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                conv_state.append(
                    tuple(
                        torch.zeros(
                            self.local_width,
                            spec.conv_history,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        for _ in range(3)
                    )
                )
                workspaces.append(
                    torch.empty(
                        3 * self.local_width,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                recurrent_outputs.append(
                    torch.empty(
                        self.local_heads,
                        spec.head_dim,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
        owner_device = self.devices[owner]
        with torch.cuda.device(owner_device):
            source = torch.empty(
                1,
                spec.hidden_size,
                dtype=torch.bfloat16,
                device=owner_device,
            )
            zero = torch.zeros(
                1,
                spec.hidden_size,
                dtype=torch.float32,
                device=owner_device,
            )
        self.layers[layer] = _KDALayer(
            owner=owner,
            source=source,
            local_inputs=local_inputs,
            input_projection=input_weights,
            gate_projection=gate_weights,
            conv_weights=local_conv_weights,
            a_log=local_a_log,
            dt_bias=local_dt_bias,
            norm_weight=local_norm_weight,
            output_projection=local_output_projection,
            recurrent_state=recurrent_state,
            conv_state=conv_state,
            workspaces=workspaces,
            recurrent_outputs=recurrent_outputs,
            contributions=[],
            zero=zero,
        )

    def capture(self) -> None:
        from ..fusedext import (
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )
        from .api import attention_step

        for device in self.devices:
            torch.cuda.synchronize(device)
        spec = self.spec
        split = (
            self.local_width,
            self.local_width,
            self.local_width,
            self.local_width,
            spec.gate_rank,
            self.local_heads,
        )
        for state in self.layers.values():
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                state.source.zero_()
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    state.local_inputs[rank].zero_()
            rank_order = _no_owner_rank_order(self, state)
            graphs = []
            events = []
            contributions = []
            ordered_streams = []
            source_event = torch.cuda.Event()
            with torch.cuda.device(owner_device):
                source_event.record(torch.cuda.current_stream(owner_device))
                torch.cuda.synchronize(owner_device)
            for rank in rank_order:
                device = self.devices[rank]
                stream = self.streams[rank]

                def execute_rank() -> torch.Tensor:
                    if (
                        not self.hidden_mode
                        and not tp_peer_copy_fused(
                            state.source,
                            state.local_inputs[rank],
                        )
                    ):
                        raise RuntimeError(
                            "TP KDA input dispatch was rejected"
                        )
                    projected = _linear(
                        state.local_inputs[rank],
                        state.input_projection[rank],
                        torch.bfloat16,
                    ).split(split, dim=-1)
                    query, key, value, output_gate = (
                        item.reshape(
                            self.local_heads,
                            spec.head_dim,
                        )
                        for item in projected[:4]
                    )
                    if not attention_step(
                        "short_conv3",
                        "cuda",
                        query=query.reshape(-1),
                        key=key.reshape(-1),
                        value=value.reshape(-1),
                        states=state.conv_state[rank],
                        weights=state.conv_weights[rank],
                    ):
                        raise RuntimeError(
                            "TP KDA short convolution was rejected"
                        )
                    recurrent_gate = _linear(
                        projected[4],
                        state.gate_projection[rank],
                        torch.bfloat16,
                    ).view(self.local_heads, spec.head_dim)
                    recurrent = attention_step(
                        "kda_recurrent",
                        "cuda",
                        query=query,
                        key=key,
                        value=value,
                        gate=recurrent_gate,
                        beta=projected[5].reshape(
                            self.local_heads
                        ).float(),
                        a_log=state.a_log[rank],
                        dt_bias=state.dt_bias[rank],
                        state=state.recurrent_state[rank],
                        workspace=state.workspaces[rank],
                        output=state.recurrent_outputs[rank],
                        lower_bound=spec.gate_lower_bound,
                    )
                    normalized = attention_step(
                        "gated_rmsnorm",
                        "cuda",
                        value=recurrent,
                        gate=output_gate.reshape(
                            self.local_heads,
                            spec.head_dim,
                        ),
                        weight=state.norm_weight[rank],
                        output=recurrent,
                        eps=spec.rms_eps,
                    )
                    if normalized is None:
                        raise RuntimeError(
                            "TP KDA gated RMSNorm was rejected"
                        )
                    return _linear(
                        normalized.reshape(1, -1),
                        state.output_projection[rank],
                    ).float()

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    execute_rank()
                    stream.synchronize()
                    event = torch.cuda.Event()
                    graph = _new_cuda_graph()
                    with torch.cuda.graph(graph, stream=stream):
                        contribution = execute_rank()
                    _instantiate_retained_graph(graph)
                    event.record(stream)
                    stream.synchronize()
                graphs.append(graph)
                events.append(event)
                contributions.append(contribution)
                ordered_streams.append(stream)
            state.graphs = graphs
            state.events = events
            state.source_event = source_event
            state.input_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.input_events.append(event)
            # Kimi performs one final KDA recapture after the retained MLP
            # and MoE plans have been assembled.  Those parent plans read
            # these exact buffers and wait on these exact events, so a
            # recapture must only replace the KDA graph handles.  Replacing
            # either list here leaves the already-captured parents pointing
            # at stale storage/events.
            if state.output_replicas is None:
                state.output_replicas = []
                for device in self.devices:
                    with torch.cuda.device(device):
                        state.output_replicas.append(
                            torch.empty(
                                1,
                                spec.hidden_size,
                                dtype=torch.bfloat16,
                                device=device,
                            )
                        )
            if state.output_events is None:
                state.output_events = []
                for device in self.devices:
                    with torch.cuda.device(device):
                        event = torch.cuda.Event()
                        event.record(torch.cuda.current_stream(device))
                        state.output_events.append(event)
            state.contributions = contributions
            state.graph_batch = make_tp_graph_launch_batch(
                [
                    int(self.devices[rank].index)
                    for rank in rank_order
                ],
                graphs,
                ordered_streams,
                events,
                source_event,
            )

    def run(self, layer: int, value: torch.Tensor) -> torch.Tensor:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError("TP KDA graphs are not captured")
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            state.source.copy_(value)
            return state.graph_batch.launch_reduce(
                state.contributions,
                state.zero,
            )

    def input_buffer(self, layer: int) -> torch.Tensor:
        return self.layers[layer].source

    def input_hidden(self, layer: int):
        """Return the fixed per-rank buffers captured by this executor."""
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.input_events is None:
            raise RuntimeError("TP KDA graphs are not captured")
        return TPHidden(
            self.devices,
            tuple(state.local_inputs),
            tuple(state.input_events),
        )

    def compose_normalize_prelude(
        self,
        layer: int,
        source,
        residual,
        active_rows: int,
        projections,
        norm_weights,
        post_norm_weights,
        workspaces,
        eps: float,
    ) -> None:
        _compose_normalize_prelude(
            self,
            layer,
            source,
            residual,
            active_rows,
            projections,
            norm_weights,
            post_norm_weights,
            workspaces,
            eps,
        )

    def run_hidden(
        self,
        layer: int,
        hidden,
        output=None,
    ):
        """Run Column→Row attention and publish the result on every rank."""
        state = self.layers[layer]
        if state.graph_batch is None or not self.hidden_mode:
            raise RuntimeError("TP KDA TPHidden graph is not captured")
        if output is None:
            output = self.output_hidden(layer)
        input_events = self.prepare_hidden_events(
            layer,
            hidden,
            output=output,
        )
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_all_rank_from_events(
            [
                input_events[rank].cuda_event
                for rank in rank_order
            ],
            state.contributions,
            list(output.replicas),
            [
                event.cuda_event
                for event in output.ready_events
            ],
        )
        return output

    def prepare_hidden_events(
        self,
        layer: int,
        hidden,
        position: int | None = None,
        *,
        output=None,
    ):
        """Validate a fixed KDA input for a larger all-rank layer plan."""
        del position
        state = self.layers[layer]
        if output is None:
            output = self.output_hidden(layer)
        self._validate_hidden_pair(state, hidden, output)
        if hidden.ready_events is None or output.ready_events is None:
            raise ValueError("CUDA TPHidden requires ready events")
        return hidden.ready_events

    def output_hidden(self, layer: int):
        """Return this layer's stable all-rank Row-TP output buffers."""
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.output_replicas is None or state.output_events is None:
            raise RuntimeError("TP KDA output buffers are unavailable")
        return TPHidden(
            self.devices,
            tuple(state.output_replicas),
            tuple(state.output_events),
        )

    def _validate_hidden_pair(self, state, hidden, output) -> None:
        if (
            tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.hidden_size))
            or output.shape != hidden.shape
            or hidden.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
        ):
            raise ValueError("TP KDA TPHidden layout mismatch")
        expected_addresses = (
            state.composed_input_addresses
            if state.composed_input_addresses is not None
            else tuple(
                item.data_ptr() for item in state.local_inputs
            )
        )
        if hidden.fixed_addresses != expected_addresses:
            raise ValueError(
                "TP KDA input must use its captured fixed addresses"
            )

    def run_prepared(self, layer: int) -> torch.Tensor:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError("TP KDA graphs are not captured")
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            return state.graph_batch.launch_reduce(
                state.contributions,
                state.zero,
            )

    def reset(self) -> None:
        for state in self.layers.values():
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    state.recurrent_state[rank].zero_()
                    for conv in state.conv_state[rank]:
                        conv.zero_()


@dataclass(frozen=True)
class MLASpec:
    hidden_size: int
    heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    max_ctx: int
    rms_eps: float
    output_gate: bool = True


@dataclass
class _MLALayer:
    owner: int
    source: torch.Tensor
    source_position: torch.Tensor
    local_inputs: list[torch.Tensor]
    local_positions: list[torch.Tensor]
    input_projection: list[torch.Tensor]
    query_norm: list[torch.Tensor]
    query_projection: list[torch.Tensor]
    kv_norm: list[torch.Tensor]
    key_absorb: list[torch.Tensor]
    value_absorb: list[torch.Tensor]
    output_projection: list[torch.Tensor]
    latent_cache: list[torch.Tensor]
    rope_cache: list[torch.Tensor]
    score_workspace: list[torch.Tensor]
    attention_output: list[torch.Tensor]
    contributions: list[torch.Tensor]
    zero: torch.Tensor
    graphs: list[torch.cuda.CUDAGraph] | None = None
    events: list[torch.cuda.Event] | None = None
    source_event: torch.cuda.Event | None = None
    input_events: list[torch.cuda.Event] | None = None
    output_replicas: list[torch.Tensor] | None = None
    output_events: list[torch.cuda.Event] | None = None
    graph_batch: object | None = None
    composed_input_addresses: tuple[int, ...] | None = None


class TensorParallelMLA:
    """Head-parallel latent attention with dynamic device-side length.

    Head-dependent Q-B/G/O and absorbed KV factors are sharded.  The small
    MQA low-rank projections are replicated so every rank can keep its local
    KV state without returning to a layer-owner bottleneck.  Each rank is one
    fixed CUDA Graph; only the source hidden state and device position change.
    """

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        spec: MLASpec,
    ) -> None:
        if not devices or spec.heads % len(devices):
            raise ValueError("MLA heads must divide the TP size")
        if spec.max_ctx <= 0:
            raise ValueError("MLA max_ctx must be positive")
        self.devices = devices
        self.spec = spec
        self.local_heads = spec.heads // len(devices)
        self.hidden_mode = (
            os.environ.get("CCCP_TP_HIDDEN", "0") != "0"
        )
        self.streams = [
            torch.cuda.Stream(device=device) for device in devices
        ]
        self.layers: dict[int, _MLALayer] = {}
        self._paged_runners: list[object] | None = None
        self._paged_position: int | tuple[str, int, int] | None = None
        self._paged_ready_events: list[torch.cuda.Event] | None = None
        self._paged_init_error: str | None = None
        self._cache_capacity = spec.max_ctx
        self._prefill_input_events: list[torch.cuda.Event] | None = None
        self._prefill_output_events: list[torch.cuda.Event] | None = None
        self._prepare_paged_runners()

    @property
    def attention_backend(self) -> str:
        from .mla_backend import select_cuda_mla_backend

        return select_cuda_mla_backend(
            flashinfer_ready=self._paged_runners is not None,
        )

    def _prepare_paged_runners(self) -> None:
        """Create one shared split-KV runner per rank, not per layer."""
        from .mla_backend import select_cuda_mla_backend

        if (
            select_cuda_mla_backend(flashinfer_ready=True) != "flashinfer"
            or
            os.environ.get("CCCP_FLASHINFER_MLA", "1") == "0"
            or self.spec.kv_lora_rank != 512
            or self.spec.qk_rope_head_dim != 64
        ):
            return
        from .api import attention_step

        runners: list[object] = []
        try:
            for device in self.devices:
                with torch.cuda.device(device):
                    runner = attention_step(
                        "paged_latent_create",
                        "cuda",
                        device=device,
                        max_ctx=self.spec.max_ctx,
                        heads=self.local_heads,
                        ckv_dim=self.spec.kv_lora_rank,
                        kpe_dim=self.spec.qk_rope_head_dim,
                        dtype=torch.bfloat16,
                        qk_head_dim=(
                            self.spec.qk_nope_head_dim
                            + self.spec.qk_rope_head_dim
                        ),
                    )
                    if runner is None or not attention_step(
                        "paged_latent_prepare",
                        "cuda",
                        runner=runner,
                        length=1,
                    ):
                        raise RuntimeError(
                            "FlashInfer MLA runner initialization failed"
                        )
                    runners.append(runner)
            capacities = {
                int(runner.max_blocks * runner.page_size)
                for runner in runners
            }
            if len(capacities) != 1:
                raise RuntimeError("FlashInfer MLA rank capacities differ")
            self._cache_capacity = capacities.pop()
            self._paged_runners = runners
            self._paged_position = 0
        except Exception as exc:
            self._paged_init_error = f"{type(exc).__name__}: {exc}"
            self._paged_runners = None
            self._paged_position = None
            self._cache_capacity = self.spec.max_ctx

    def _prepare_paged_position(
        self,
        position: int,
    ) -> tuple[torch.cuda.Event, ...] | None:
        if self._paged_runners is None:
            return None
        if (
            self._paged_position == int(position)
            and self._paged_ready_events is not None
        ):
            return tuple(self._paged_ready_events)
        from .api import attention_step

        events: list[torch.cuda.Event] = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                if not attention_step(
                    "paged_latent_prepare",
                    "cuda",
                    runner=self._paged_runners[rank],
                    length=int(position) + 1,
                ):
                    from ..flashinfer_mla import last_error

                    cause = last_error()
                    detail = (
                        "unknown planner error"
                        if cause is None
                        else f"{type(cause).__name__}: {cause}"
                    )
                    raise RuntimeError(
                        "FlashInfer MLA dynamic split-KV plan failed: "
                        f"{detail}"
                    )
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                events.append(event)
        self._paged_position = int(position)
        self._paged_ready_events = events
        return tuple(events)

    def prepare_paged_prefill(
        self,
        position: int,
        rows: int,
    ) -> bool:
        """Plan one causal query block per rank for every MLA layer."""
        if self._paged_runners is None or position < 0 or rows <= 0:
            return False
        signature = ("prefill", int(position), int(rows))
        if self._paged_position == signature:
            return True
        from .api import attention_step

        length = int(position) + int(rows)
        events: list[torch.cuda.Event] = []
        try:
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    if not attention_step(
                        "paged_latent_prepare_prefill",
                        "cuda",
                        runner=self._paged_runners[rank],
                        query_length=int(rows),
                        length=length,
                    ):
                        return False
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    events.append(event)
        except (ImportError, LookupError, RuntimeError):
            return False
        self._paged_position = signature
        self._paged_ready_events = events
        return True

    def add_layer(
        self,
        layer: int,
        owner: int,
        combined_input,
        query_norm: torch.Tensor,
        query_projection,
        kv_norm: torch.Tensor,
        key_absorb: torch.Tensor,
        value_absorb: torch.Tensor,
        output_projection,
    ) -> None:
        spec = self.spec
        q_width = spec.qk_nope_head_dim + spec.qk_rope_head_dim
        expected_input_rows = (
            spec.q_lora_rank
            + spec.kv_lora_rank
            + spec.qk_rope_head_dim
            + (
                spec.heads * spec.v_head_dim
                if spec.output_gate
                else 0
            )
        )
        if (
            not _weight_is_bf16_linear(combined_input)
            or _weight_shape(combined_input)
            != (expected_input_rows, spec.hidden_size)
            or _weight_shape(query_projection)
            != (spec.heads * q_width, spec.q_lora_rank)
            or key_absorb.shape
            != (
                spec.heads,
                spec.qk_nope_head_dim,
                spec.kv_lora_rank,
            )
            or value_absorb.shape
            != (
                spec.heads,
                spec.v_head_dim,
                spec.kv_lora_rank,
            )
            or _weight_shape(output_projection)
            != (
                spec.hidden_size,
                spec.heads * spec.v_head_dim,
            )
        ):
            raise ValueError(
                f"TP MLA layer {layer} weight shape/dtype mismatch"
            )
        input_rows = (
            (
                spec.q_lora_rank,
                spec.kv_lora_rank + spec.qk_rope_head_dim,
                spec.heads * spec.v_head_dim,
            )
            if spec.output_gate
            else (
                spec.q_lora_rank,
                spec.kv_lora_rank + spec.qk_rope_head_dim,
            )
        )
        input_parts = _projection_parts(combined_input, input_rows)
        query_a, kv_a = input_parts[:2]
        gate = input_parts[2] if spec.output_gate else None
        key_parts = key_absorb.chunk(len(self.devices), dim=0)
        value_parts = value_absorb.chunk(len(self.devices), dim=0)
        local_gate_width = self.local_heads * spec.v_head_dim
        local_query_width = self.local_heads * q_width
        local_inputs = []
        local_positions = []
        input_weights = []
        query_norms = []
        query_weights = []
        kv_norms = []
        key_weights = []
        value_weights = []
        output_weights = []
        latent_cache = []
        rope_cache = []
        score_workspace = []
        attention_output = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                local_inputs.append(
                    torch.empty(
                        1,
                        spec.hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                local_positions.append(
                    torch.zeros(1, dtype=torch.long, device=device)
                )
                replicated = (
                    _row_slice(query_a, 0, int(query_a.shape[0]), device),
                    _row_slice(kv_a, 0, int(kv_a.shape[0]), device),
                )
                input_weights.append(
                    _combine_projection_parts(
                        replicated
                        + (
                            _row_slice(
                                gate,
                                rank * local_gate_width,
                                (rank + 1) * local_gate_width,
                                device,
                            ),
                        )
                        if gate is not None
                        else replicated
                    )
                )
                query_norms.append(query_norm.to(device).contiguous())
                query_weights.append(
                    _row_slice(
                        query_projection,
                        rank * local_query_width,
                        (rank + 1) * local_query_width,
                        device,
                    )
                )
                kv_norms.append(kv_norm.to(device).contiguous())
                key_weights.append(
                    key_parts[rank].to(device).contiguous()
                )
                value_weights.append(
                    value_parts[rank].to(device).contiguous()
                )
                output_weights.append(
                    _column_slice(
                        output_projection,
                        rank * local_gate_width,
                        (rank + 1) * local_gate_width,
                        device,
                    )
                )
                latent_cache.append(
                    torch.zeros(
                        self._cache_capacity,
                        spec.kv_lora_rank,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                rope_cache.append(
                    torch.zeros(
                        self._cache_capacity,
                        spec.qk_rope_head_dim,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                score_workspace.append(
                    torch.empty(
                        self.local_heads,
                        self._cache_capacity,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                attention_output.append(
                    torch.empty(
                        self.local_heads,
                        1,
                        spec.kv_lora_rank,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
        owner_device = self.devices[owner]
        with torch.cuda.device(owner_device):
            source = torch.empty(
                1,
                spec.hidden_size,
                dtype=torch.bfloat16,
                device=owner_device,
            )
            source_position = torch.zeros(
                1,
                dtype=torch.long,
                device=owner_device,
            )
            zero = torch.zeros(
                1,
                spec.hidden_size,
                dtype=torch.float32,
                device=owner_device,
            )
        self.layers[layer] = _MLALayer(
            owner=owner,
            source=source,
            source_position=source_position,
            local_inputs=local_inputs,
            local_positions=local_positions,
            input_projection=input_weights,
            query_norm=query_norms,
            query_projection=query_weights,
            kv_norm=kv_norms,
            key_absorb=key_weights,
            value_absorb=value_weights,
            output_projection=output_weights,
            latent_cache=latent_cache,
            rope_cache=rope_cache,
            score_workspace=score_workspace,
            attention_output=attention_output,
            contributions=[],
            zero=zero,
        )

    def capture(self) -> None:
        from ..fusedext import (
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )
        from .api import attention_step, rmsnorm

        for device in self.devices:
            torch.cuda.synchronize(device)
        spec = self.spec
        q_width = spec.qk_nope_head_dim + spec.qk_rope_head_dim
        split = (
            spec.q_lora_rank,
            spec.kv_lora_rank + spec.qk_rope_head_dim,
            *(
                (self.local_heads * spec.v_head_dim,)
                if spec.output_gate
                else ()
            ),
        )
        scale_denominator = float(q_width**0.5)
        for state in self.layers.values():
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                state.source.zero_()
                state.source_position.zero_()
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    state.local_inputs[rank].zero_()
                    state.local_positions[rank].zero_()
            rank_order = _no_owner_rank_order(self, state)
            graphs = []
            events = []
            contributions = []
            ordered_streams = []
            source_event = torch.cuda.Event()
            with torch.cuda.device(owner_device):
                source_event.record(torch.cuda.current_stream(owner_device))
                torch.cuda.synchronize(owner_device)
            for rank in rank_order:
                device = self.devices[rank]
                stream = self.streams[rank]

                def execute_rank() -> torch.Tensor:
                    if (
                        not self.hidden_mode
                        and (
                            not tp_peer_copy_fused(
                                state.source,
                                state.local_inputs[rank],
                            )
                            or not tp_peer_copy_fused(
                                state.source_position,
                                state.local_positions[rank],
                            )
                        )
                    ):
                        raise RuntimeError(
                            "TP MLA input dispatch was rejected"
                        )
                    projected = _linear(
                        state.local_inputs[rank],
                        state.input_projection[rank],
                        torch.bfloat16,
                    ).split(split, dim=-1)
                    query_source, compressed = projected[:2]
                    output_gate = (
                        projected[2] if spec.output_gate else None
                    )
                    query_source = rmsnorm(
                        query_source,
                        state.query_norm[rank],
                        1e-6,
                    )
                    if query_source is None:
                        raise RuntimeError("TP MLA query RMSNorm unavailable")
                    query = _linear(
                        query_source,
                        state.query_projection[rank],
                        torch.bfloat16,
                    ).view(self.local_heads, q_width)
                    query_nope, query_rope = query.split(
                        (
                            spec.qk_nope_head_dim,
                            spec.qk_rope_head_dim,
                        ),
                        dim=-1,
                    )
                    latent, key_rope = compressed.split(
                        (
                            spec.kv_lora_rank,
                            spec.qk_rope_head_dim,
                        ),
                        dim=-1,
                    )
                    latent = rmsnorm(
                        latent,
                        state.kv_norm[rank],
                        1e-6,
                    )
                    if latent is None:
                        raise RuntimeError("TP MLA KV RMSNorm unavailable")
                    state.latent_cache[rank].index_copy_(
                        0,
                        state.local_positions[rank],
                        latent,
                    )
                    state.rope_cache[rank].index_copy_(
                        0,
                        state.local_positions[rank],
                        key_rope,
                    )
                    absorbed_query = torch.bmm(
                        query_nope[:, None, :],
                        state.key_absorb[rank],
                    )
                    if self._paged_runners is not None:
                        page_size = int(
                            self._paged_runners[rank].page_size
                        )
                        context = attention_step(
                            "paged_latent_decode",
                            "cuda",
                            runner=self._paged_runners[rank],
                            query_nope=absorbed_query.transpose(0, 1),
                            query_rope=query_rope[None, :, :],
                            latent_cache=state.latent_cache[rank].view(
                                self._cache_capacity // page_size,
                                page_size,
                                spec.kv_lora_rank,
                            ),
                            rope_cache=state.rope_cache[rank].view(
                                self._cache_capacity // page_size,
                                page_size,
                                spec.qk_rope_head_dim,
                            ),
                        )
                        if context is not None:
                            context = context.transpose(0, 1)
                    else:
                        context = attention_step(
                            "compressed_kv_decode",
                            "cuda",
                            query_nope=absorbed_query,
                            query_rope=query_rope[:, None, :],
                            latent_cache=state.latent_cache[rank],
                            rope_cache=state.rope_cache[rank],
                            position=state.local_positions[rank],
                            scale_denominator=scale_denominator,
                            score_workspace=state.score_workspace[rank],
                            output=state.attention_output[rank],
                        )
                    if context is None:
                        raise RuntimeError("TP MLA decode core unavailable")
                    output = torch.bmm(
                        context,
                        state.value_absorb[rank].transpose(1, 2),
                    ).reshape(1, -1)
                    if output_gate is not None:
                        output.mul_(output_gate.sigmoid())
                    return _linear(
                        output,
                        state.output_projection[rank],
                    ).float()

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    execute_rank()
                    stream.synchronize()
                    event = torch.cuda.Event()
                    graph = _new_cuda_graph()
                    with torch.cuda.graph(graph, stream=stream):
                        contribution = execute_rank()
                    _instantiate_retained_graph(graph)
                    event.record(stream)
                    stream.synchronize()
                graphs.append(graph)
                events.append(event)
                contributions.append(contribution)
                ordered_streams.append(stream)
            state.graphs = graphs
            state.events = events
            state.source_event = source_event
            state.input_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.input_events.append(event)
            state.output_replicas = []
            state.output_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    state.output_replicas.append(
                        torch.empty(
                            1,
                            spec.hidden_size,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                    )
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.output_events.append(event)
            state.contributions = contributions
            state.graph_batch = make_tp_graph_launch_batch(
                [
                    int(self.devices[rank].index)
                    for rank in rank_order
                ],
                graphs,
                ordered_streams,
                events,
                source_event,
            )

    def prefill_rank(
        self,
        layer: int,
        rank: int,
        value: torch.Tensor,
        position: int,
        paged_prefill: bool,
        *,
        projected_input: torch.Tensor | None = None,
        rotary: Callable[
            [torch.Tensor, torch.Tensor, int],
            tuple[torch.Tensor, torch.Tensor],
        ] | None = None,
        stat_add: Callable[[str, int], None] | None = None,
    ) -> torch.Tensor:
        """Run the shared batched latent-attention implementation."""
        from .api import attention_step, rmsnorm

        if rank < 0 or rank >= len(self.devices):
            raise ValueError("TP MLA rank is out of range")
        state = self.layers[layer]
        spec = self.spec
        device = self.devices[rank]
        rows = int(value.shape[0])
        if projected_input is None and value.device != device:
            raise ValueError("TP MLA prefill input is on the wrong rank")
        if rows <= 0 or position < 0 or position + rows > self._cache_capacity:
            raise ValueError("TP MLA prefill range exceeds cache capacity")

        local_heads = self.local_heads
        q_width = spec.qk_nope_head_dim + spec.qk_rope_head_dim
        projected = projected_input
        if projected is None:
            projected = _linear_batch(
                value,
                state.input_projection[rank],
                torch.bfloat16,
            ).to(torch.bfloat16)
        elif spec.output_gate:
            raise ValueError(
                "shared MLA input projection requires an ungated topology"
            )
        split = (
            spec.q_lora_rank,
            spec.kv_lora_rank + spec.qk_rope_head_dim,
            *((local_heads * spec.v_head_dim,) if spec.output_gate else ()),
        )
        parts = projected.split(split, dim=-1)
        query_source, compressed = parts[:2]
        output_gate = parts[2] if spec.output_gate else None
        query_source = rmsnorm(
            query_source.contiguous(),
            state.query_norm[rank],
            spec.rms_eps,
        )
        if query_source is None:
            raise RuntimeError("TP MLA query RMSNorm unavailable")
        query = _linear_batch(
            query_source,
            state.query_projection[rank],
            torch.bfloat16,
        ).to(torch.bfloat16).view(rows, local_heads, q_width)
        query_nope, query_rope = query.split(
            (spec.qk_nope_head_dim, spec.qk_rope_head_dim),
            dim=-1,
        )
        latent, key_rope = compressed.split(
            (spec.kv_lora_rank, spec.qk_rope_head_dim),
            dim=-1,
        )
        latent = rmsnorm(
            latent.contiguous(),
            state.kv_norm[rank],
            spec.rms_eps,
        )
        if latent is None:
            raise RuntimeError("TP MLA KV RMSNorm unavailable")
        if rotary is not None:
            query_rope, key_rope = rotary(
                query_rope,
                key_rope,
                int(position),
            )
            query_rope = query_rope.to(latent.dtype)
            key_rope = key_rope.to(state.rope_cache[rank].dtype)

        positions = torch.arange(
            position,
            position + rows,
            dtype=torch.long,
            device=device,
        )
        state.latent_cache[rank].index_copy_(0, positions, latent)
        state.rope_cache[rank].index_copy_(
            0, positions, key_rope.contiguous()
        )
        # Heads are the BMM batch dimension; never materialize token×head
        # broadcast expansions for long context.
        absorbed = torch.bmm(
            query_nope.transpose(0, 1).contiguous(),
            state.key_absorb[rank],
        ).transpose(0, 1)
        contexts = torch.empty(
            rows,
            local_heads,
            spec.kv_lora_rank,
            dtype=torch.bfloat16,
            device=device,
        )
        requested_backend = os.environ.get(
            "CCCP_PREFILL_MLA_BACKEND", "auto"
        ).strip().lower()
        if requested_backend not in {"auto", "flashinfer", "cccp-paged"}:
            raise ValueError(
                "CCCP_PREFILL_MLA_BACKEND must be auto, flashinfer or "
                f"cccp-paged, got {requested_backend!r}"
            )
        runners = self._paged_runners
        from .mla_backend import select_cuda_mla_backend

        if requested_backend == "auto":
            backend = select_cuda_mla_backend(
                flashinfer_ready=(
                    paged_prefill and runners is not None
                ),
            )
        else:
            backend = requested_backend
        context = None
        if (
            backend == "flashinfer"
            and paged_prefill
            and runners is not None
        ):
            runner = runners[rank]
            page_size = int(runner.page_size)
            context = attention_step(
                "paged_latent_prefill",
                "cuda",
                runner=runner,
                query_nope=absorbed,
                query_rope=query_rope,
                latent_cache=state.latent_cache[rank].view(
                    self._cache_capacity // page_size,
                    page_size,
                    spec.kv_lora_rank,
                ),
                rope_cache=state.rope_cache[rank].view(
                    self._cache_capacity // page_size,
                    page_size,
                    spec.qk_rope_head_dim,
                ),
                output=contexts,
            )
            if context is not None and stat_add is not None:
                stat_add("mla_paged_calls", 1)
                stat_add("mla_paged_tokens", rows)
        if backend == "flashinfer" and context is None:
            raise RuntimeError(
                "TP MLA FlashInfer prefill was selected but is unavailable"
            )
        if backend == "cccp-paged":
            context = attention_step(
                "cccp_paged_latent_prefill",
                "cuda",
                query_nope=absorbed,
                query_rope=query_rope,
                latent_cache=state.latent_cache[rank],
                rope_cache=state.rope_cache[rank],
                query_start=position,
                scale_denominator=float(q_width**0.5),
                output=contexts,
            )
            if context is not None and stat_add is not None:
                stat_add("mla_cccp_paged_calls", 1)
                stat_add("mla_cccp_paged_tokens", rows)
        if context is None:
            raise RuntimeError(
                f"TP MLA optimized CUDA prefill backend {backend!r} "
                "is unavailable; ordinary BF16 attention is forbidden"
            )

        output = torch.bmm(
            context.transpose(0, 1).contiguous(),
            state.value_absorb[rank].transpose(1, 2),
        ).transpose(0, 1).reshape(rows, -1)
        if self.spec.output_gate:
            output.mul_(output_gate.sigmoid())
        return _linear_batch(
            output,
            state.output_projection[rank],
        ).float()

    def prefill_primary(
        self,
        layer: int,
        value: torch.Tensor,
        position: int,
        *,
        rotary: Callable[
            [torch.Tensor, torch.Tensor, int, int],
            tuple[torch.Tensor, torch.Tensor],
        ] | None = None,
        stat_add: Callable[[str, int], None] | None = None,
    ) -> torch.Tensor:
        """Shard one batched MLA block and return the owner-rank result."""
        from ..fusedext import tp_all_rank_reduce_from_events_fused

        state = self.layers[layer]
        owner = int(state.owner)
        owner_device = self.devices[owner]
        if value.device != owner_device:
            raise ValueError("TP MLA primary input is not on its owner rank")
        rows = int(value.shape[0])
        paged = self.prepare_paged_prefill(position, rows)
        projected_primary = (
            _linear_batch(
                value,
                state.input_projection[owner],
                torch.bfloat16,
            ).to(torch.bfloat16)
            if not self.spec.output_gate
            else None
        )
        partials: list[torch.Tensor] = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                local = (
                    value
                    if rank == owner or projected_primary is not None
                    else value.to(device, non_blocking=True)
                )
                projected_input = (
                    None
                    if projected_primary is None
                    else projected_primary
                    if rank == owner
                    else projected_primary.to(device, non_blocking=True)
                )
                rank_rotary = (
                    None
                    if rotary is None
                    else lambda query, key, offset, rank=rank: rotary(
                        query, key, offset, rank
                    )
                )
                partials.append(
                    self.prefill_rank(
                        layer,
                        rank,
                        local,
                        position,
                        paged,
                        projected_input=projected_input,
                        rotary=rank_rotary,
                        stat_add=stat_add,
                    )
                )
        if self._prefill_input_events is None:
            self._prefill_input_events = []
            self._prefill_output_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    self._prefill_input_events.append(torch.cuda.Event())
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    self._prefill_output_events.append(event)
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                self._prefill_input_events[rank].record(
                    torch.cuda.current_stream(device)
                )
        with torch.cuda.device(owner_device):
            output = torch.empty(
                rows,
                self.spec.hidden_size,
                dtype=torch.bfloat16,
                device=owner_device,
            )
        reduced = tp_all_rank_reduce_from_events_fused(
            partials,
            self._prefill_input_events,
            [output],
            [self._prefill_output_events[owner]],
        )
        if reduced is None:
            for device in self.devices:
                torch.cuda.synchronize(device)
            total = partials[0]
            for partial in partials[1:]:
                total = total + partial.to(owner_device)
            output.copy_(total.to(torch.bfloat16))
            return output
        with torch.cuda.device(owner_device):
            torch.cuda.current_stream(owner_device).wait_event(
                self._prefill_output_events[owner]
            )
        return output

    def run(
        self,
        layer: int,
        value: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError("TP MLA graphs are not captured")
        if not 0 <= int(position) < self.spec.max_ctx:
            raise ValueError("TP MLA position exceeds max_ctx")
        owner_device = self.devices[state.owner]
        if value.device != owner_device:
            raise ValueError("TP MLA input is not on its owner rank")
        paged_events = self._prepare_paged_position(position)
        if paged_events is not None:
            for device in self.devices:
                torch.cuda.synchronize(device)
        with torch.cuda.device(owner_device):
            state.source.copy_(value)
            state.source_position.fill_(int(position))
            return state.graph_batch.launch_reduce(
                state.contributions,
                state.zero,
            )

    def input_buffer(self, layer: int) -> torch.Tensor:
        return self.layers[layer].source

    def input_hidden(self, layer: int):
        """Return the fixed per-rank buffers captured by this executor."""
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.input_events is None:
            raise RuntimeError("TP MLA graphs are not captured")
        return TPHidden(
            self.devices,
            tuple(state.local_inputs),
            tuple(state.input_events),
        )

    def run_hidden(
        self,
        layer: int,
        hidden,
        position: int,
        output=None,
    ):
        """Run Column→Row MLA and publish the result on every rank."""
        state = self.layers[layer]
        if state.graph_batch is None or not self.hidden_mode:
            raise RuntimeError("TP MLA TPHidden graph is not captured")
        if not 0 <= int(position) < self.spec.max_ctx:
            raise ValueError("TP MLA position exceeds max_ctx")
        if output is None:
            output = self.output_hidden(layer)
        input_events = self.prepare_hidden_events(
            layer,
            hidden,
            position,
            output=output,
        )
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_all_rank_from_events(
            [
                input_events[rank].cuda_event
                for rank in rank_order
            ],
            state.contributions,
            list(output.replicas),
            [
                event.cuda_event
                for event in output.ready_events
            ],
        )
        return output

    def prepare_hidden_events(
        self,
        layer: int,
        hidden,
        position: int | None = None,
        *,
        output=None,
    ):
        """Prepare fixed MLA position events without launching its Graph."""
        if position is None:
            raise ValueError("TP MLA position is required")
        state = self.layers[layer]
        if not 0 <= int(position) < self.spec.max_ctx:
            raise ValueError("TP MLA position exceeds max_ctx")
        if output is None:
            output = self.output_hidden(layer)
        self._validate_hidden_pair(state, hidden, output)
        if hidden.ready_events is None or output.ready_events is None:
            raise ValueError("CUDA TPHidden requires ready events")
        if state.input_events is None:
            raise RuntimeError("TP MLA input events are unavailable")
        paged_events = self._prepare_paged_position(int(position))
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                stream = torch.cuda.current_stream(device)
                stream.wait_event(hidden.ready_events[rank])
                if paged_events is not None:
                    stream.wait_event(paged_events[rank])
                state.local_positions[rank].fill_(int(position))
                state.input_events[rank].record(stream)
        return tuple(state.input_events)

    def compose_normalize_prelude(
        self,
        layer: int,
        source,
        residual,
        active_rows: int,
        projections,
        norm_weights,
        post_norm_weights,
        workspaces,
        eps: float,
    ) -> None:
        _compose_normalize_prelude(
            self,
            layer,
            source,
            residual,
            active_rows,
            projections,
            norm_weights,
            post_norm_weights,
            workspaces,
            eps,
        )

    def output_hidden(self, layer: int):
        """Return this layer's stable all-rank Row-TP output buffers."""
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.output_replicas is None or state.output_events is None:
            raise RuntimeError("TP MLA output buffers are unavailable")
        return TPHidden(
            self.devices,
            tuple(state.output_replicas),
            tuple(state.output_events),
        )

    def _validate_hidden_pair(self, state, hidden, output) -> None:
        if (
            tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.hidden_size))
            or output.shape != hidden.shape
            or hidden.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
        ):
            raise ValueError("TP MLA TPHidden layout mismatch")
        expected_addresses = (
            state.composed_input_addresses
            if state.composed_input_addresses is not None
            else tuple(
                item.data_ptr() for item in state.local_inputs
            )
        )
        if hidden.fixed_addresses != expected_addresses:
            raise ValueError(
                "TP MLA input must use its captured fixed addresses"
            )

    def run_prepared(
        self,
        layer: int,
        position: int,
    ) -> torch.Tensor:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError("TP MLA graphs are not captured")
        if not 0 <= int(position) < self.spec.max_ctx:
            raise ValueError("TP MLA position exceeds max_ctx")
        paged_events = self._prepare_paged_position(position)
        if paged_events is not None:
            for device in self.devices:
                torch.cuda.synchronize(device)
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            state.source_position.fill_(int(position))
            return state.graph_batch.launch_reduce(
                state.contributions,
                state.zero,
            )

    def reset(self) -> None:
        self._paged_position = None
        self._paged_ready_events = None
        for state in self.layers.values():
            with torch.cuda.device(self.devices[state.owner]):
                state.source_position.zero_()


class TensorParallelMoELayerPlan:
    """Submit one fixed-address no-owner MoE layer in one host call.

    This is a scheduling primitive, not a model-specific mathematical
    operator.  All four phases retain their existing all-rank collectives and
    compact packed-expert kernels; the plan only removes repeated
    Python→C++ transitions between them.
    """

    def __init__(
        self,
        layer: int,
        input_hidden,
        residual,
        shared_executor,
        route_executor,
        expert_executor,
        final_executor,
    ) -> None:
        from ..fusedext import make_tp_no_owner_moe_layer_plan

        layer = int(layer)
        devices = tuple(input_hidden.devices)
        shared_state = shared_executor.layers[layer]
        route_state = route_executor.layers[layer]
        final_state = final_executor.layers[layer]
        if (
            devices != tuple(shared_executor.devices)
            or devices != tuple(route_executor.devices)
            or devices != tuple(final_executor.devices)
            or tuple(residual.devices) != devices
            or input_hidden.ready_events is None
            or residual.ready_events is None
            or shared_state.graph_batch is None
            or shared_state.events is None
            or route_state.graph_batch is None
            or route_state.output_events is None
            or final_state.graph_batch is None
            or final_state.routed_workspaces is None
            or final_state.shared_workspaces is None
        ):
            raise RuntimeError(
                "fixed no-owner MoE plan requires complete all-rank state"
            )
        expected_input_addresses = (
            shared_state.composed_input_addresses
            if shared_state.composed_input_addresses is not None
            else tuple(
                item.data_ptr() for item in shared_state.local_inputs
            )
        )
        if input_hidden.fixed_addresses != expected_input_addresses:
            raise ValueError(
                "fixed no-owner MoE plan input addresses do not match"
            )
        router_output, latent_output = route_executor.output_hidden(layer)
        (
            expert_batch,
            expert_contributions,
            packed_output,
        ) = expert_executor.fixed_layer_plan(layer)
        output = final_executor.output_hidden(layer)
        plan = make_tp_no_owner_moe_layer_plan(
            shared_state.graph_batch,
            route_state.graph_batch,
            expert_batch,
            final_state.graph_batch,
            input_hidden.ready_events,
            (
                tuple(route_state.router_contributions),
                tuple(route_state.latent_contributions),
            ),
            (
                tuple(router_output.replicas),
                tuple(latent_output.replicas),
            ),
            route_state.output_events,
            tuple(expert_contributions),
            tuple(packed_output.replicas),
            packed_output.ready_events,
            tuple(final_state.contributions),
            tuple(shared_state.contributions),
            tuple(shared_state.events),
            tuple(residual.replicas),
            # The MLP parent graph produces both the normalized input and
            # prefix residual.  Its all-rank done events therefore guard both.
            tuple(shared_state.events),
            tuple(final_state.routed_workspaces),
            tuple(final_state.shared_workspaces),
            tuple(output.replicas),
            output.ready_events,
        )
        if plan is None:
            raise RuntimeError(
                "fixed no-owner MoE extension plan is unavailable"
            )
        self.layer = layer
        self.devices = devices
        self.output = output
        self._plan = plan
        # Retain Python owners for CUDA Graph and event handles cached by the
        # extension plan.  Tensors are additionally retained in C++.
        self._dependencies = (
            input_hidden,
            residual,
            shared_executor,
            route_executor,
            expert_executor,
            final_executor,
            router_output,
            latent_output,
            packed_output,
        )

    def launch(self, input_events=None):
        if input_events is None:
            self._plan.launch()
        else:
            if len(input_events) != len(self.devices):
                raise ValueError(
                    "profiled no-owner MoE events must match TP ranks"
                )
            self._plan.launch_from_events(
                [event.cuda_event for event in input_events]
            )
        return self.output


class TensorParallelDecodeLayerPlan:
    """Submit Attention→routed MoE for one layer in one host call.

    This plan only composes already-captured generic all-rank operators.
    Attention still performs Column/Head-TP→Row-TP and the packed expert
    remains tensor-sharded across every rank.  The fixed attention output
    events directly trigger MoE, so no owner, hidden broadcast, or new
    collective is introduced.
    """

    def __init__(
        self,
        layer: int,
        attention_executor,
        moe_plan: TensorParallelMoELayerPlan,
    ) -> None:
        from ..fusedext import make_tp_no_owner_decode_layer_plan

        layer = int(layer)
        state = attention_executor.layers[layer]
        attention_output = attention_executor.output_hidden(layer)
        if (
            state.graph_batch is None
            or not state.contributions
            or attention_output.ready_events is None
            or tuple(attention_output.devices) != tuple(moe_plan.devices)
        ):
            raise RuntimeError(
                "decode layer plan requires captured all-rank attention"
            )
        plan = make_tp_no_owner_decode_layer_plan(
            state.graph_batch,
            moe_plan._plan,
            list(state.contributions),
            list(attention_output.replicas),
            list(attention_output.ready_events),
        )
        if plan is None:
            raise RuntimeError(
                "fixed no-owner decode layer plan is unavailable"
            )
        self.layer = layer
        self.devices = tuple(attention_output.devices)
        self.attention_executor = attention_executor
        self.attention_output = attention_output
        self.output = moe_plan.output
        self._plan = plan
        self._dependencies = (
            attention_executor,
            moe_plan,
            attention_output,
        )

    def launch(self, hidden, position: int | None = None):
        input_events = self.attention_executor.prepare_hidden_events(
            self.layer,
            hidden,
            position,
            output=self.attention_output,
        )
        self._plan.launch_from_events(
            [event.cuda_event for event in input_events]
        )
        return self.output


class TensorParallelHyperConnectionDecodeLayerPlan:
    """Submit one complete no-owner Hyper-Connection TP layer.

    The plan is keyed by the mathematical capability, not a model family.
    It preserves the existing Attention, shared MLP, Router and compact
    packed-expert implementations while collapsing their fixed scheduling
    metadata into one Python→C++ transition.  The final routed/shared
    reduction writes the four-channel HC state directly, so no intermediate
    owner hidden or extra publication is introduced.
    """

    def __init__(
        self,
        layer: int,
        attention_batch,
        attention_partials,
        attention_output,
        attention_aux,
        prefix_output,
        ffn_input,
        ffn_parameters,
        ffn_aux_buffers,
        shared_executor,
        route_executor,
        expert_executor,
        output,
        *,
        sinkhorn_iters: int,
        eps: float,
    ) -> None:
        from ..fusedext import make_tp_no_owner_hc_decode_layer_plan

        layer = int(layer)
        devices = tuple(attention_output.devices)
        ranks = len(devices)
        shared_state = shared_executor.layers[layer]
        route_state = route_executor.layers[layer]
        expert_batch, expert_contributions, _ = (
            expert_executor.fixed_layer_plan(layer)
        )
        if (
            ranks <= 1
            or tuple(attention_partials.devices) != devices
            or tuple(prefix_output.devices) != devices
            or tuple(ffn_input.devices) != devices
            or tuple(output.devices) != devices
            or attention_output.ready_events is None
            or ffn_input.ready_events is None
            or output.ready_events is None
            or shared_state.graph_batch is None
            or shared_state.events is None
            or route_state.graph_batch is None
            or route_state.events is None
            or len(attention_aux) != ranks
            or len(ffn_parameters) != ranks
            or len(ffn_aux_buffers) != ranks
        ):
            raise RuntimeError(
                "fixed HC decode plan requires complete all-rank state"
            )
        expected_addresses = tuple(
            shared_state.composed_input_addresses
            if shared_state.composed_input_addresses is not None
            else tuple(item.data_ptr() for item in shared_state.local_inputs)
        )
        if ffn_input.fixed_addresses != expected_addresses:
            raise ValueError(
                "fixed HC decode plan shared input addresses do not match"
            )
        if tuple(
            item.data_ptr() for item in route_state.local_inputs
        ) != ffn_input.fixed_addresses:
            raise ValueError(
                "fixed HC decode plan Router input addresses do not match"
            )
        attention_residuals = [item[0] for item in attention_aux]
        attention_posts = [item[2] for item in attention_aux]
        attention_combs = [item[3] for item in attention_aux]
        ffn_functions = [item[0] for item in ffn_parameters]
        ffn_scales = [item[1] for item in ffn_parameters]
        ffn_bases = [item[2] for item in ffn_parameters]
        ffn_norms = [item[3] for item in ffn_parameters]
        ffn_posts = [item[0] for item in ffn_aux_buffers]
        ffn_combs = [item[1] for item in ffn_aux_buffers]
        plan = make_tp_no_owner_hc_decode_layer_plan(
            attention_batch,
            shared_state.graph_batch,
            route_state.graph_batch,
            expert_batch,
            list(attention_partials.contributions),
            list(attention_output.replicas),
            list(attention_output.ready_events),
            attention_residuals,
            attention_posts,
            attention_combs,
            list(prefix_output.replicas),
            ffn_functions,
            ffn_scales,
            ffn_bases,
            ffn_norms,
            list(ffn_input.replicas),
            ffn_posts,
            ffn_combs,
            list(ffn_input.ready_events),
            list(route_state.events),
            list(expert_contributions),
            list(shared_state.contributions),
            list(shared_state.events),
            list(output.replicas),
            list(output.ready_events),
            int(sinkhorn_iters),
            float(eps),
        )
        if plan is None:
            raise RuntimeError(
                "fixed no-owner HC decode extension plan is unavailable"
            )
        self.layer = layer
        self.devices = devices
        self.output = output
        self._plan = plan
        self._dependencies = (
            attention_batch,
            attention_partials,
            attention_output,
            attention_aux,
            prefix_output,
            ffn_input,
            ffn_parameters,
            ffn_aux_buffers,
            shared_executor,
            route_executor,
            expert_executor,
            expert_batch,
            output,
        )

    def launch(self, input_events):
        if len(input_events) != len(self.devices):
            raise ValueError("HC decode input events must match TP ranks")
        self._plan.launch_from_events(
            [event.cuda_event for event in input_events]
        )
        return self.output


__all__ = [
    "GatedMLPSpec",
    "KDASpec",
    "MLASpec",
    "MoEPreludeSpec",
    "OwnerGroupedTensorParallel",
    "ReplicatedSubgroupTensorParallel",
    "PackedMoEFinalizerSpec",
    "RoutePackedPlanSpec",
    "ReplicatedLinearSpec",
    "RouteDownSpec",
    "RowParallelLinearSpec",
    "TensorParallelGatedMLP",
    "TensorParallelKDA",
    "TensorParallelMLA",
    "TensorParallelAllRankCollective",
    "TensorParallelMoELayerPlan",
    "TensorParallelDecodeLayerPlan",
    "TensorParallelHyperConnectionDecodeLayerPlan",
    "TensorParallelMoEPrelude",
    "TensorParallelPackedMoEFinalizer",
    "TensorParallelRoutePackedPlan",
    "TensorParallelReplicatedLinear",
    "TensorParallelRouteDown",
    "TensorParallelRowLinear",
    "TensorParallelVocab",
    "shard_linear_input",
    "shard_linear_output",
]

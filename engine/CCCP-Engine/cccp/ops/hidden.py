"""Fixed-address tensor-parallel hidden-state containers.

``TPHidden`` describes where a decode hidden state lives; it does not encode
model-family behavior.  The first supported layout is a complete replica on
every participating rank.  Column-parallel operators consume the local
replica, Row-parallel operators publish their reduced result back into all
replicas, so the following layer does not need an owner-to-rank broadcast.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import torch


@dataclass(frozen=True)
class TPHidden:
    """One fixed-address hidden-state replica per TP rank."""

    devices: tuple[torch.device, ...]
    replicas: tuple[torch.Tensor, ...]
    ready_events: tuple[torch.cuda.Event, ...] | None = None

    def __post_init__(self) -> None:
        if not self.devices or len(self.devices) != len(self.replicas):
            raise ValueError(
                "TPHidden requires one replica per non-empty device tuple"
            )
        shape = self.replicas[0].shape
        dtype = self.replicas[0].dtype
        for rank, (device, replica) in enumerate(
            zip(self.devices, self.replicas)
        ):
            if (
                replica.device != device
                or replica.shape != shape
                or replica.dtype != dtype
                or not replica.is_contiguous()
            ):
                raise ValueError(
                    "TPHidden replica "
                    f"{rank} does not match device/shape/dtype/layout"
                )
        if (
            self.ready_events is not None
            and len(self.ready_events) != len(self.devices)
        ):
            raise ValueError(
                "TPHidden ready events must match the rank count"
            )

    @classmethod
    def empty(
        cls,
        devices: Iterable[torch.device],
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
    ) -> "TPHidden":
        normalized = tuple(torch.device(device) for device in devices)
        replicas = []
        events = []
        for device in normalized:
            with (
                torch.cuda.device(device)
                if device.type == "cuda"
                else torch.no_grad()
            ):
                replicas.append(
                    torch.empty(
                        shape,
                        dtype=dtype,
                        device=device,
                    )
                )
                if device.type == "cuda":
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    events.append(event)
        return cls(
            normalized,
            tuple(replicas),
            tuple(events) if events else None,
        )

    @property
    def shape(self) -> torch.Size:
        return self.replicas[0].shape

    @property
    def dtype(self) -> torch.dtype:
        return self.replicas[0].dtype

    @property
    def layout(self) -> str:
        return "replicated"

    @property
    def fixed_addresses(self) -> tuple[int, ...]:
        return tuple(replica.data_ptr() for replica in self.replicas)

    def local(self, rank: int) -> torch.Tensor:
        return self.replicas[int(rank)]

    def subset(
        self,
        devices: Iterable[torch.device],
    ) -> "TPHidden":
        """Create a zero-copy view for a TP subgroup."""
        selected = tuple(torch.device(device) for device in devices)
        ranks = tuple(self.devices.index(device) for device in selected)
        return TPHidden(
            selected,
            tuple(self.replicas[rank] for rank in ranks),
            (
                tuple(self.ready_events[rank] for rank in ranks)
                if self.ready_events is not None
                else None
            ),
        )

    def on_device(self, device: torch.device | str) -> torch.Tensor:
        target = torch.device(device)
        try:
            rank = self.devices.index(target)
        except ValueError as exc:
            raise ValueError(
                f"TPHidden has no replica on {target}"
            ) from exc
        return self.replicas[rank]

    def wait_on(self, device: torch.device | str) -> torch.Tensor:
        """Wait for one local replica and return it."""
        target = torch.device(device)
        rank = self.devices.index(target)
        if self.ready_events is not None and len(self.devices) != 1:
            with torch.cuda.device(target):
                torch.cuda.current_stream(target).wait_event(
                    self.ready_events[rank]
                )
        return self.replicas[rank]

    def copy_from_owner(
        self,
        value: torch.Tensor,
        owner: int,
    ) -> "TPHidden":
        """Publish an initialization value; steady decode uses collective."""
        owner = int(owner)
        if value.shape != self.shape or value.dtype != self.dtype:
            raise ValueError("TPHidden publication shape/dtype mismatch")
        for rank, replica in enumerate(self.replicas):
            with (
                torch.cuda.device(replica.device)
                if replica.device.type == "cuda"
                else torch.no_grad()
            ):
                if not (
                    rank == owner
                    and replica.data_ptr() == value.data_ptr()
                ):
                    replica.copy_(value)
                if self.ready_events is not None and len(self.devices) != 1:
                    self.ready_events[rank].record(
                        torch.cuda.current_stream(replica.device)
                    )
        return self

    def reduce_from(
        self,
        contributions: Iterable[torch.Tensor],
    ) -> "TPHidden":
        """Reduce canonical FP32 Row-TP partials into every replica."""
        if self.devices[0].type != "cuda":
            raise RuntimeError(
                "TPHidden all-rank reduction currently requires CUDA"
            )
        from ..fusedext import tp_all_rank_reduce_fused

        result = tp_all_rank_reduce_fused(
            list(contributions),
            list(self.replicas),
        )
        if result is None:
            raise RuntimeError("TPHidden all-rank reduction was rejected")
        if self.ready_events is not None:
            for device, event in zip(self.devices, self.ready_events):
                with torch.cuda.device(device):
                    event.record(torch.cuda.current_stream(device))
        return self

    def copy_to(self, output: "TPHidden") -> "TPHidden":
        """Copy replicas locally without an owner-to-rank broadcast."""
        self._validate_output(output)
        for output_rank, (device, target) in enumerate(
            zip(output.devices, output.replicas)
        ):
            source_rank = self.devices.index(device)
            with (
                torch.cuda.device(device)
                if device.type == "cuda"
                else torch.no_grad()
            ):
                if self.ready_events is not None:
                    torch.cuda.current_stream(device).wait_event(
                        self.ready_events[source_rank]
                    )
                target.copy_(self.replicas[source_rank])
                if output.ready_events is not None:
                    output.ready_events[output_rank].record(
                        torch.cuda.current_stream(device)
                    )
        return output

    def add_to(
        self,
        other: "TPHidden",
        output: "TPHidden",
    ) -> "TPHidden":
        """Add matching local replicas and publish fixed-address results."""
        self._validate_output(output)
        if self.shape != other.shape or self.dtype != other.dtype:
            raise ValueError("TPHidden add operands must match")
        if (
            output.devices[0].type == "cuda"
            and self.ready_events is not None
            and other.ready_events is not None
            and output.ready_events is not None
            and self.dtype == torch.bfloat16
        ):
            from ..fusedext import tp_hidden_add_batch_fused

            left_ranks = tuple(
                self.devices.index(device)
                for device in output.devices
            )
            right_ranks = tuple(
                other.devices.index(device)
                for device in output.devices
            )
            result = tp_hidden_add_batch_fused(
                [self.replicas[rank] for rank in left_ranks],
                [self.ready_events[rank] for rank in left_ranks],
                [other.replicas[rank] for rank in right_ranks],
                [other.ready_events[rank] for rank in right_ranks],
                list(output.replicas),
                list(output.ready_events),
            )
            if result is not None:
                return output
        for output_rank, (device, target) in enumerate(
            zip(output.devices, output.replicas)
        ):
            left_rank = self.devices.index(device)
            right_rank = other.devices.index(device)
            with (
                torch.cuda.device(device)
                if device.type == "cuda"
                else torch.no_grad()
            ):
                stream = (
                    torch.cuda.current_stream(device)
                    if device.type == "cuda"
                    else None
                )
                if self.ready_events is not None:
                    stream.wait_event(self.ready_events[left_rank])
                if other.ready_events is not None:
                    stream.wait_event(other.ready_events[right_rank])
                torch.add(
                    self.replicas[left_rank],
                    other.replicas[right_rank],
                    out=target,
                )
                if output.ready_events is not None:
                    output.ready_events[output_rank].record(stream)
        return output

    def rmsnorm_to(
        self,
        weights: Iterable[torch.Tensor],
        eps: float,
        output: "TPHidden",
    ) -> "TPHidden":
        """Apply the same RMSNorm definition independently on each rank."""
        self._validate_output(output)
        local_weights = tuple(weights)
        if len(local_weights) != len(output.devices):
            raise ValueError("TPHidden RMSNorm weights must match outputs")
        if (
            output.devices[0].type == "cuda"
            and self.ready_events is not None
            and output.ready_events is not None
            and self.dtype == torch.bfloat16
        ):
            from ..fusedext import tp_hidden_rmsnorm_batch_fused

            source_ranks = tuple(
                self.devices.index(device)
                for device in output.devices
            )
            result = tp_hidden_rmsnorm_batch_fused(
                [self.replicas[rank] for rank in source_ranks],
                [self.ready_events[rank] for rank in source_ranks],
                list(local_weights),
                float(eps),
                list(output.replicas),
                list(output.ready_events),
            )
            if result is not None:
                return output
        from .api import rmsnorm

        for output_rank, (device, target, weight) in enumerate(
            zip(output.devices, output.replicas, local_weights)
        ):
            source_rank = self.devices.index(device)
            if weight.device != device:
                raise ValueError("TPHidden RMSNorm weight device mismatch")
            with (
                torch.cuda.device(device)
                if device.type == "cuda"
                else torch.no_grad()
            ):
                stream = (
                    torch.cuda.current_stream(device)
                    if device.type == "cuda"
                    else None
                )
                if self.ready_events is not None:
                    stream.wait_event(self.ready_events[source_rank])
                result = rmsnorm(
                    self.replicas[source_rank],
                    weight,
                    float(eps),
                    output=target,
                )
                if result is None:
                    source = self.replicas[source_rank]
                    work = source.float()
                    normalized = work * torch.rsqrt(
                        work.square().mean(dim=-1, keepdim=True)
                        + float(eps)
                    )
                    target.copy_(
                        weight.to(source.dtype)
                        * normalized.to(source.dtype)
                    )
                if output.ready_events is not None:
                    output.ready_events[output_rank].record(stream)
        return output

    def residual_mix_to(
        self,
        residual: "TPResidualBuffer",
        projections: Iterable[torch.Tensor],
        norm_weights: Iterable[torch.Tensor],
        eps: float,
        output: "TPHidden",
        *,
        post_norm_weights: Iterable[torch.Tensor] | None = None,
        workspaces: Iterable[torch.Tensor] | None = None,
    ) -> "TPHidden":
        """Apply replicated residual mixing without any hidden collective."""
        self._validate_output(output)
        if residual.active_rows <= 0:
            raise ValueError("TP residual mix requires at least one row")
        local_projections = tuple(projections)
        local_norms = tuple(norm_weights)
        local_posts = (
            tuple(post_norm_weights)
            if post_norm_weights is not None
            else (None,) * len(output.devices)
        )
        local_workspaces = (
            tuple(workspaces)
            if workspaces is not None
            else (None,) * len(output.devices)
        )
        use_inverse = (
            os.environ.get("CCCP_RESIDUAL_INVERSE_CACHE", "1") != "0"
        )
        count = len(output.devices)
        if not (
            len(local_projections)
            == len(local_norms)
            == len(local_posts)
            == len(local_workspaces)
            == count
        ):
            raise ValueError("TP residual mix rank metadata mismatch")
        if (
            output.devices[0].type == "cuda"
            and self.ready_events is not None
            and residual.ready_events is not None
            and output.ready_events is not None
            and all(item is not None for item in local_posts)
            and all(item is not None for item in local_workspaces)
            and use_inverse
        ):
            from ..fusedext import tp_hidden_residual_mix_batch_fused

            prefix_ranks = tuple(
                self.devices.index(device)
                for device in output.devices
            )
            residual_ranks = tuple(
                residual.devices.index(device)
                for device in output.devices
            )
            result = tp_hidden_residual_mix_batch_fused(
                [self.replicas[rank] for rank in prefix_ranks],
                [self.ready_events[rank] for rank in prefix_ranks],
                [
                    residual.replicas[rank][
                        :, :residual.active_rows
                    ]
                    for rank in residual_ranks
                ],
                [
                    residual.ready_events[rank]
                    for rank in residual_ranks
                ],
                list(local_projections),
                list(local_norms),
                list(local_posts),
                list(local_workspaces),
                [
                    residual.inverses[rank][
                        :residual.active_rows
                    ]
                    for rank in residual_ranks
                ],
                float(eps),
                list(output.replicas),
                list(output.ready_events),
            )
            if result is not None:
                return output
        from .api import residual_mix

        for output_rank, device in enumerate(output.devices):
            prefix_rank = self.devices.index(device)
            residual_rank = residual.devices.index(device)
            projection = local_projections[output_rank]
            norm_weight = local_norms[output_rank]
            post_norm = local_posts[output_rank]
            workspace = local_workspaces[output_rank]
            if (
                projection.device != device
                or norm_weight.device != device
                or (
                    post_norm is not None
                    and post_norm.device != device
                )
                or (
                    workspace is not None
                    and workspace.device != device
                )
            ):
                raise ValueError("TP residual mix tensor device mismatch")
            with torch.cuda.device(device):
                stream = torch.cuda.current_stream(device)
                if self.ready_events is not None:
                    stream.wait_event(self.ready_events[prefix_rank])
                if residual.ready_events is not None:
                    stream.wait_event(
                        residual.ready_events[residual_rank]
                    )
                result = residual_mix(
                    "attention",
                    self.replicas[prefix_rank],
                    residual.replicas[residual_rank][
                        :, :residual.active_rows
                    ],
                    projection,
                    norm_weight,
                    float(eps),
                    output=output.replicas[output_rank],
                    post_norm_weight=post_norm,
                    workspace=workspace,
                    residual_inverse=residual.inverses[residual_rank][
                        :residual.active_rows
                    ] if use_inverse else None,
                )
                if result is None:
                    raise RuntimeError(
                        "registered TP residual mix rejected CUDA inputs"
                    )
                if output.ready_events is not None:
                    output.ready_events[output_rank].record(stream)
        return output

    def capture_normalize_graphs(
        self,
        output: "TPHidden",
        streams: Iterable[torch.cuda.Stream],
        post_norm_weights: Iterable[torch.Tensor],
        eps: float,
        *,
        residual: "TPResidualBuffer | None" = None,
        active_rows: int = 0,
        projections: Iterable[torch.Tensor] | None = None,
        norm_weights: Iterable[torch.Tensor] | None = None,
        workspaces: Iterable[torch.Tensor] | None = None,
    ) -> tuple[torch.cuda.CUDAGraph, ...]:
        """Capture fixed-address rank-local normalization as child graphs.

        The caller composes these children ahead of an Attention/MLP rank
        graph.  Runtime readiness is supplied to the parent graph launcher;
        no owner broadcast or additional collective is introduced here.
        """
        self._validate_output(output)
        local_streams = tuple(streams)
        local_posts = tuple(post_norm_weights)
        count = len(output.devices)
        if (
            output.devices[0].type != "cuda"
            or len(local_streams) != count
            or len(local_posts) != count
        ):
            raise ValueError(
                "fixed normalize Graph requires one CUDA stream/weight "
                "per output rank"
            )
        use_residual = residual is not None and int(active_rows) > 0
        if use_residual:
            rows = int(active_rows)
            if not 0 < rows <= residual.max_rows:
                raise ValueError("fixed residual row count is invalid")
            local_projections = tuple(projections or ())
            local_norms = tuple(norm_weights or ())
            local_workspaces = tuple(workspaces or ())
            if not (
                len(local_projections)
                == len(local_norms)
                == len(local_workspaces)
                == count
            ):
                raise ValueError(
                    "fixed residual Graph metadata must match rank count"
                )
        else:
            rows = 0
            local_projections = ()
            local_norms = ()
            local_workspaces = ()

        from .api import residual_mix, rmsnorm

        graphs = []
        for output_rank, (device, target, stream, post_norm) in enumerate(
            zip(
                output.devices,
                output.replicas,
                local_streams,
                local_posts,
            )
        ):
            source_rank = self.devices.index(device)

            def execute_rank(
                rank: int = output_rank,
                source_index: int = source_rank,
            ) -> None:
                if rows:
                    residual_rank = residual.devices.index(device)
                    result = residual_mix(
                        "attention",
                        self.replicas[source_index],
                        residual.replicas[residual_rank][:, :rows],
                        local_projections[rank],
                        local_norms[rank],
                        float(eps),
                        output=target,
                        post_norm_weight=post_norm,
                        workspace=local_workspaces[rank],
                        residual_inverse=(
                            residual.inverses[residual_rank][:rows]
                        ),
                    )
                else:
                    result = rmsnorm(
                        self.replicas[source_index],
                        post_norm,
                        float(eps),
                        output=target,
                    )
                if result is None:
                    raise RuntimeError(
                        "registered fixed normalize Graph operation "
                        "was rejected"
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
        return tuple(graphs)

    def capture_mlp_prelude_graphs(
        self,
        attention: "TPHidden",
        prefix_output: "TPHidden",
        normalized_output: "TPHidden",
        streams: Iterable[torch.cuda.Stream],
        residual: "TPResidualBuffer",
        active_rows: int,
        projections: Iterable[torch.Tensor],
        norm_weights: Iterable[torch.Tensor],
        post_norm_weights: Iterable[torch.Tensor],
        workspaces: Iterable[torch.Tensor],
        eps: float,
        *,
        boundary: bool,
    ) -> tuple[torch.cuda.CUDAGraph, ...]:
        """Capture prefix add/copy plus MLP-input normalization per rank."""
        self._validate_output(prefix_output)
        attention._validate_output(prefix_output)
        prefix_output._validate_output(normalized_output)
        local_streams = tuple(streams)
        local_projections = tuple(projections)
        local_norms = tuple(norm_weights)
        local_posts = tuple(post_norm_weights)
        local_workspaces = tuple(workspaces)
        count = len(normalized_output.devices)
        rows = int(active_rows)
        if (
            normalized_output.devices[0].type != "cuda"
            or not 0 < rows <= residual.max_rows
            or not (
                len(local_streams)
                == len(local_projections)
                == len(local_norms)
                == len(local_posts)
                == len(local_workspaces)
                == count
            )
        ):
            raise ValueError("fixed MLP prelude metadata is invalid")

        from .api import residual_mix

        graphs = []
        for output_rank, (device, stream) in enumerate(
            zip(normalized_output.devices, local_streams)
        ):
            hidden_rank = self.devices.index(device)
            attention_rank = attention.devices.index(device)
            prefix_rank = prefix_output.devices.index(device)
            normalized_rank = normalized_output.devices.index(device)
            residual_rank = residual.devices.index(device)

            def execute_rank(
                rank: int = output_rank,
                source_index: int = hidden_rank,
                attention_index: int = attention_rank,
                prefix_index: int = prefix_rank,
                normalized_index: int = normalized_rank,
                residual_index: int = residual_rank,
            ) -> None:
                prefix = prefix_output.replicas[prefix_index]
                if boundary:
                    prefix.copy_(attention.replicas[attention_index])
                else:
                    torch.add(
                        self.replicas[source_index],
                        attention.replicas[attention_index],
                        out=prefix,
                    )
                result = residual_mix(
                    "attention",
                    prefix,
                    residual.replicas[residual_index][:, :rows],
                    local_projections[rank],
                    local_norms[rank],
                    float(eps),
                    output=normalized_output.replicas[
                        normalized_index
                    ],
                    post_norm_weight=local_posts[rank],
                    workspace=local_workspaces[rank],
                    residual_inverse=residual.inverses[
                        residual_index
                    ][:rows],
                )
                if result is None:
                    raise RuntimeError(
                        "registered fixed MLP prelude was rejected"
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
        return tuple(graphs)

    def _validate_output(self, output: "TPHidden") -> None:
        if (
            output.shape != self.shape
            or output.dtype != self.dtype
            or any(device not in self.devices for device in output.devices)
        ):
            raise ValueError("TPHidden output layout mismatch")


@dataclass(frozen=True)
class TPSharded:
    """Fixed-address equal-width shards of one hidden vector."""

    devices: tuple[torch.device, ...]
    shards: tuple[torch.Tensor, ...]
    global_width: int
    ready_events: tuple[torch.cuda.Event, ...] | None = None

    def __post_init__(self) -> None:
        if (
            not self.devices
            or len(self.devices) != len(self.shards)
            or self.global_width <= 0
            or self.global_width % len(self.devices)
        ):
            raise ValueError("TPSharded requires equal non-empty shards")
        local_width = self.global_width // len(self.devices)
        prefix = self.shards[0].shape[:-1]
        dtype = self.shards[0].dtype
        for device, shard in zip(self.devices, self.shards):
            if (
                shard.device != device
                or shard.shape != (*prefix, local_width)
                or shard.dtype != dtype
                or not shard.is_contiguous()
            ):
                raise ValueError(
                    "TPSharded shard device/shape/dtype/layout mismatch"
                )
        if (
            self.ready_events is not None
            and len(self.ready_events) != len(self.devices)
        ):
            raise ValueError("TPSharded events must match rank count")

    @property
    def dtype(self) -> torch.dtype:
        return self.shards[0].dtype

    @property
    def shape(self) -> torch.Size:
        return torch.Size(
            (*self.shards[0].shape[:-1], self.global_width)
        )

    @property
    def layout(self) -> str:
        return "sharded"

    @property
    def fixed_addresses(self) -> tuple[int, ...]:
        return tuple(shard.data_ptr() for shard in self.shards)

    def copy_from_full(self, value: torch.Tensor) -> "TPSharded":
        if value.shape != self.shape or value.dtype != self.dtype:
            raise ValueError("TPSharded source shape/dtype mismatch")
        parts = value.split(self.global_width // len(self.devices), dim=-1)
        for rank, (device, shard, part) in enumerate(
            zip(self.devices, self.shards, parts)
        ):
            with (
                torch.cuda.device(device)
                if device.type == "cuda"
                else torch.no_grad()
            ):
                shard.copy_(part)
                if self.ready_events is not None:
                    self.ready_events[rank].record(
                        torch.cuda.current_stream(device)
                    )
        return self

    def copy_from_replicated(self, hidden: TPHidden) -> "TPSharded":
        """Slice each shard from its local full replica, without peer DMA."""
        if hidden.shape != self.shape or hidden.dtype != self.dtype:
            raise ValueError("TPSharded replicated source mismatch")
        local_width = self.global_width // len(self.devices)
        for rank, (device, shard) in enumerate(
            zip(self.devices, self.shards)
        ):
            hidden_rank = hidden.devices.index(device)
            with (
                torch.cuda.device(device)
                if device.type == "cuda"
                else torch.no_grad()
            ):
                stream = (
                    torch.cuda.current_stream(device)
                    if device.type == "cuda"
                    else None
                )
                if hidden.ready_events is not None:
                    stream.wait_event(hidden.ready_events[hidden_rank])
                start = rank * local_width
                shard.copy_(
                    hidden.replicas[hidden_rank][
                        ..., start:start + local_width
                    ]
                )
                if self.ready_events is not None:
                    self.ready_events[rank].record(stream)
        return self


@dataclass(frozen=True)
class TPPartials:
    """Canonical FP32 Row-TP partials plus their completion events."""

    devices: tuple[torch.device, ...]
    contributions: tuple[torch.Tensor, ...]
    ready_events: tuple[torch.cuda.Event, ...]

    def __post_init__(self) -> None:
        if (
            not self.devices
            or len(self.devices) != len(self.contributions)
            or len(self.devices) != len(self.ready_events)
        ):
            raise ValueError("TPPartials ranks/events must be size-equal")
        shape = self.contributions[0].shape
        for device, contribution in zip(
            self.devices,
            self.contributions,
        ):
            if (
                contribution.device != device
                or contribution.dtype != torch.float32
                or contribution.shape != shape
                or not contribution.is_contiguous()
            ):
                raise ValueError(
                    "TPPartials require matching contiguous FP32 tensors"
                )

    @classmethod
    def empty(
        cls,
        devices: Iterable[torch.device],
        shape: tuple[int, ...],
    ) -> "TPPartials":
        """Allocate one reusable FP32 Row-TP workspace per rank.

        Decode operators publish into these fixed addresses and only record
        their retained completion events.  Keeping this allocation outside
        the token loop is required for stable CUDA Graph composition and
        avoids creating one tensor and one event per rank, layer and token.
        """
        normalized = tuple(torch.device(device) for device in devices)
        contributions = []
        events = []
        for device in normalized:
            if device.type != "cuda":
                raise ValueError("TPPartials workspaces require CUDA ranks")
            with torch.cuda.device(device):
                contributions.append(
                    torch.empty(shape, dtype=torch.float32, device=device)
                )
                # ``torch.cuda.Event`` is associated with a device lazily.
                # Materialize it while that rank is current; otherwise a
                # later all-rank batch may first query every raw handle under
                # the last CUDA device and create invalid cross-device
                # resources.
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                events.append(event)
        return cls(
            normalized,
            tuple(contributions),
            tuple(events),
        )

    @property
    def shape(self) -> torch.Size:
        return self.contributions[0].shape

    @property
    def fixed_addresses(self) -> tuple[int, ...]:
        return tuple(item.data_ptr() for item in self.contributions)


@dataclass
class TPResidualBuffer:
    """Fixed-capacity replicated rows used by residual-mixing operators."""

    devices: tuple[torch.device, ...]
    replicas: tuple[torch.Tensor, ...]
    inverses: tuple[torch.Tensor, ...]
    ready_events: tuple[torch.cuda.Event, ...] | None
    active_rows: int = 0

    def __post_init__(self) -> None:
        if (
            not self.devices
            or len(self.devices) != len(self.replicas)
            or len(self.devices) != len(self.inverses)
        ):
            raise ValueError(
                "TPResidualBuffer requires one replica per rank"
            )
        shape = self.replicas[0].shape
        if len(shape) != 3 or shape[0] != 1 or shape[1] <= 0:
            raise ValueError(
                "TPResidualBuffer shape must be [1,max_rows,width]"
            )
        for device, replica, inverse in zip(
            self.devices,
            self.replicas,
            self.inverses,
        ):
            if (
                replica.device != device
                or replica.shape != shape
                or replica.dtype != self.replicas[0].dtype
                or not replica.is_contiguous()
                or inverse.device != device
                or inverse.dtype != torch.float32
                or inverse.shape != (shape[1],)
                or not inverse.is_contiguous()
            ):
                raise ValueError(
                    "TPResidualBuffer replica/inverse layout mismatch"
                )
        if (
            self.ready_events is not None
            and len(self.ready_events) != len(self.devices)
        ):
            raise ValueError(
                "TPResidualBuffer events must match the rank count"
            )
        if not 0 <= self.active_rows <= shape[1]:
            raise ValueError("TPResidualBuffer active row count is invalid")

    @classmethod
    def empty(
        cls,
        devices: Iterable[torch.device],
        max_rows: int,
        width: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "TPResidualBuffer":
        normalized = tuple(torch.device(device) for device in devices)
        replicas = []
        inverses = []
        events = []
        for device in normalized:
            with torch.cuda.device(device):
                replicas.append(
                    torch.empty(
                        1,
                        int(max_rows),
                        int(width),
                        dtype=dtype,
                        device=device,
                    )
                )
                inverses.append(
                    torch.zeros(
                        int(max_rows),
                        dtype=torch.float32,
                        device=device,
                    )
                )
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                events.append(event)
        return cls(
            normalized,
            tuple(replicas),
            tuple(inverses),
            tuple(events),
        )

    @property
    def max_rows(self) -> int:
        return int(self.replicas[0].shape[1])

    @property
    def width(self) -> int:
        return int(self.replicas[0].shape[2])

    def reset(self) -> None:
        self.active_rows = 0
        for device, inverse in zip(self.devices, self.inverses):
            with torch.cuda.device(device):
                inverse.zero_()
                if self.ready_events is not None:
                    self.ready_events[
                        self.devices.index(device)
                    ].record(torch.cuda.current_stream(device))

    def append(self, hidden: TPHidden) -> int:
        if self.active_rows >= self.max_rows:
            raise RuntimeError("TP residual buffer capacity exceeded")
        if (
            hidden.shape != torch.Size((1, self.width))
            or hidden.dtype != self.replicas[0].dtype
            or any(device not in hidden.devices for device in self.devices)
        ):
            raise ValueError("TP residual append hidden layout mismatch")
        row = self.active_rows
        for rank, (device, replica, inverse) in enumerate(
            zip(self.devices, self.replicas, self.inverses)
        ):
            hidden_rank = hidden.devices.index(device)
            with torch.cuda.device(device):
                stream = torch.cuda.current_stream(device)
                if hidden.ready_events is not None:
                    stream.wait_event(hidden.ready_events[hidden_rank])
                replica[:, row].copy_(hidden.replicas[hidden_rank])
                inverse[row].zero_()
                if self.ready_events is not None:
                    self.ready_events[rank].record(stream)
        self.active_rows += 1
        return row

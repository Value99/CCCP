"""Tensor-parallel packed routed experts with a bounded per-rank GPU cache.

This pool is the multi-GPU counterpart of :mod:`cccp.packed_hybrid`.  The full
CCCP expert payload remains packed in host RAM.  On a cache miss each GPU gets
only its row/column tensor-parallel shard; neither RAM nor VRAM contains a
dequantized expert matrix.

The dynamic expert pointers intentionally sit outside the fixed-address
decode-layer graph.  Dense, Attention, KDA and the shared expert retain their
normal TP graphs.  Routed Top-K is evaluated first, the selected packed shards
are installed into fixed arenas, and the usual registered packed-MoE kernel is
then launched on every rank followed by one all-rank Row-TP reduction.
"""

from __future__ import annotations

import gc
import os
import threading
import time
from collections import Counter, OrderedDict

import torch

from .kimi_experts import PackedExpertPool
from .packed_hybrid import (
    DeviceExpert,
    HostPackedWeight,
    PackedHybridPool,
    PackedExpert,
    PackedExpertSignature,
    PackedWeightSignature,
    _PackedArenas,
    allocate_packed_slots,
)
from .store import PinnedStage


def tensor_shard_signature(
    expert: PackedExpert,
    ranks: int,
) -> PackedExpertSignature:
    """Describe one packed TP shard without allocating or unpacking indices."""
    ranks = int(ranks)
    if ranks <= 0:
        raise ValueError("packed TP rank count must be positive")
    if len(expert) not in (2, 3):
        raise ValueError(
            "packed TP expert must contain GU+Down or Gate+Up+Down"
        )
    output = []
    for projection, weight in enumerate(expert):
        is_down = projection == len(expert) - 1
        if (
            weight.raw.numel() % ranks
            or (
                weight.blocks % ranks
                if is_down
                else weight.rows % ranks
            )
            or (weight.cols % ranks if is_down else 0)
        ):
            raise ValueError("packed expert shape is not TP divisible")
        output.append(
            PackedWeightSignature(
                raw_bytes=weight.raw.numel() // ranks,
                cb_shape=tuple(weight.cb.shape),
                rows=weight.rows if is_down else weight.rows // ranks,
                cols=weight.cols // ranks if is_down else weight.cols,
                blocks=(
                    weight.blocks // ranks
                    if is_down
                    else weight.blocks
                ),
                dim=weight.dim,
                bits=weight.bits,
            )
        )
    return PackedExpertSignature(tuple(output))


def tensor_shard_host_expert(
    expert: PackedExpert,
    *,
    rank: int,
    ranks: int,
    intermediate: int,
) -> PackedExpert:
    """Create one temporary packed host shard, preserving p8/p12/p14 bits."""
    if len(expert) not in (2, 3):
        raise ValueError(
            "packed TP expert must contain GU+Down or Gate+Up+Down"
        )
    output = []
    for projection, weight in enumerate(expert):
        is_down = projection == len(expert) - 1
        if len(expert) == 3:
            raw, blocks = PackedExpertPool._tensor_shard_projection_raw(
                weight,
                projection=projection,
                rank=int(rank),
                ranks=int(ranks),
                intermediate=int(intermediate),
            )
        else:
            raw, blocks = PackedExpertPool._tensor_shard_raw(
                weight,
                projection=5 if is_down else 0,
                rank=int(rank),
                ranks=int(ranks),
                intermediate=int(intermediate),
            )
        output.append(
            HostPackedWeight(
                raw,
                weight.cb,
                weight.rows if is_down else weight.rows // int(ranks),
                weight.cols // int(ranks) if is_down else weight.cols,
                blocks,
                weight.dim,
                weight.bits,
            )
        )
    return tuple(output)


class PackedTensorHybridPool(PackedHybridPool):
    """All-rank packed TP with full host RAM and bounded per-rank VRAM."""

    device_routed = True
    full_resident = False
    hidden_mode = True
    parallelism = "tensor"
    prefetch_default = False
    manages_per_rank_budget = True
    supports_vram_watch = False
    retains_store_ram_blobs = True

    def __init__(
        self,
        store,
        devices: tuple[torch.device, ...],
        plan,
        budget_gb: float,
        *,
        ram_gb: float = 0.0,
    ):
        if len(devices) < 2:
            raise ValueError("packed tensor hybrid requires at least two GPUs")
        normalized = tuple(torch.device(device) for device in devices)
        if any(device.type != "cuda" for device in normalized):
            raise ValueError("packed tensor hybrid requires CUDA devices")
        super().__init__(
            store,
            budget_gb,
            device=normalized[0],
            ram_gb=ram_gb,
        )
        self.devices = normalized
        self.plan = plan
        self.tensor_group_size = len(normalized)
        self.requested_budget_per_rank = self.budget
        self.budget_per_rank = 0
        self.budget = self.requested_budget_per_rank * len(normalized)
        self.cache: OrderedDict[
            tuple[int, int],
            tuple[DeviceExpert, ...],
        ] = OrderedDict()
        self._rank_arenas: list[_PackedArenas] = []
        self._rank_device_codebooks: list[
            dict[int, torch.Tensor]
        ] = []
        self._rank_stages = [
            PinnedStage(device)
            for device in self.devices
        ]
        self._rank_metadata: list[torch.Tensor] = []
        self._rank_route_ids: list[torch.Tensor] = []
        self._rank_order: list[torch.Tensor] = []
        self._rank_weights: list[torch.Tensor] = []
        self._rank_workspaces: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        self._contribution_events: list[torch.cuda.Event] = []
        self._output_by_layer: dict[int, object] = {}
        self._last_ids: dict[int, list[int]] = {}
        self._transfer_lock = threading.RLock()
        self.active = False
        self.arena_slots: dict[str, int] = {}
        self.shard_seconds = 0.0

    @property
    def gpu_arena_bytes_by_rank(self) -> tuple[int, ...]:
        return tuple(arena.nbytes for arena in self._rank_arenas)

    @property
    def gpu_arena_bytes(self) -> int:
        return sum(self.gpu_arena_bytes_by_rank)

    @property
    def gpu_storage_bytes_by_rank(self) -> tuple[int, ...]:
        output = []
        for rank in range(len(self.devices)):
            workspace = 0
            if rank < len(self._rank_workspaces):
                workspace += sum(
                    tensor.nbytes
                    for tensor in self._rank_workspaces[rank]
                )
                workspace += self._rank_metadata[rank].nbytes
                workspace += self._rank_route_ids[rank].nbytes
                workspace += self._rank_order[rank].nbytes
                workspace += self._rank_weights[rank].nbytes
            arena = (
                self._rank_arenas[rank].nbytes
                if rank < len(self._rank_arenas)
                else 0
            )
            if rank < len(self._rank_device_codebooks):
                workspace += sum(
                    tensor.nbytes
                    for tensor in self._rank_device_codebooks[rank].values()
                )
            output.append(arena + workspace)
        return tuple(output)

    @property
    def gpu_storage_bytes(self) -> int:
        return sum(self.gpu_storage_bytes_by_rank)

    def allocate(self) -> None:
        """Dense is allocated first; the live expert budget is sealed later."""
        return

    def bind_hidden_inputs(
        self,
        layer: int,
        value,
        weights: tuple[torch.Tensor, ...],
        indices: tuple[torch.Tensor, ...],
    ) -> None:
        del layer
        if (
            tuple(value.devices) != self.devices
            or len(weights) != len(self.devices)
            or len(indices) != len(self.devices)
        ):
            raise ValueError("packed TP RAM route layout mismatch")

    def _safe_budget_per_rank(self) -> int:
        reserve = int(
            float(os.environ.get(
                "CCCP_VRAM_HEADROOM_GB",
                os.environ.get("CCCP_VRAM_RESERVE_GB", "1"),
            ))
            * 2**30
        )
        safe = []
        for device in self.devices:
            with torch.cuda.device(device):
                free, total = torch.cuda.mem_get_info(device)
                allocated = torch.cuda.memory_allocated(device)
                index = (
                    torch.cuda.current_device()
                    if device.index is None
                    else int(device.index)
                )
                try:
                    fraction = torch.cuda.get_per_process_memory_fraction(
                        index
                    )
                except (AttributeError, RuntimeError):
                    fraction = 1.0
            process_room = max(
                0,
                int(total * fraction) - allocated - reserve,
            )
            device_room = max(0, free - reserve)
            safe.append(
                min(
                    self.requested_budget_per_rank,
                    process_room,
                    device_room,
                )
            )
        return min(safe)

    def build_gpu_arenas(self) -> float:
        if self._rank_arenas:
            return self.gpu_arena_bytes / 2**30
        if not self.pinned:
            raise RuntimeError("packed TP RAM experts are not loaded")
        safe_budget = self._safe_budget_per_rank()
        if safe_budget <= 0:
            raise RuntimeError("packed TP RAM path has no safe GPU cache room")

        counts = Counter(
            tensor_shard_signature(expert, len(self.devices))
            for expert in self.pinned.values()
        )
        host_codebooks = {
            weight.cb.data_ptr(): weight.cb
            for expert in self.pinned.values()
            for weight in expert
        }
        resident_codebook_bytes = (
            sum(codebook.nbytes for codebook in host_codebooks.values())
            if self._resident_codebooks
            else 0
        )
        arena_budget = safe_budget - resident_codebook_bytes
        if arena_budget <= 0:
            raise RuntimeError(
                "packed TP RAM cache cannot fit resident codebooks"
            )
        top_k = int(self.store.cfg["top_k"])
        weights = None
        if self.slot_mix is not None:
            tier_totals = Counter(
                self._signature_tier(signature)
                for signature, count in counts.items()
                for _ in range(count)
            )
            weights = {
                signature: (
                    self.slot_mix.get(
                        self._signature_tier(signature),
                        0.0,
                    )
                    * count
                    / max(
                        1,
                        tier_totals[self._signature_tier(signature)],
                    )
                )
                for signature, count in counts.items()
            }
        specs = allocate_packed_slots(
            counts,
            arena_budget,
            top_k,
            weights=weights,
            resident_codebooks=self._resident_codebooks,
        )

        try:
            self._rank_arenas = [
                _PackedArenas(
                    specs,
                    device,
                    resident_codebooks=self._resident_codebooks,
                )
                for device in self.devices
            ]
            if self._resident_codebooks:
                self._rank_device_codebooks = [
                    {
                        pointer: codebook.to(
                            device=device,
                            dtype=torch.bfloat16,
                            non_blocking=False,
                        )
                        for pointer, codebook in host_codebooks.items()
                    }
                    for device in self.devices
                ]
            else:
                self._rank_device_codebooks = [
                    {} for _ in self.devices
                ]
        except Exception:
            self._rank_arenas.clear()
            self._rank_device_codebooks.clear()
            gc.collect()
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            raise

        hidden = int(self.store.cfg["routed_hidden"])
        intermediate = int(self.store.cfg["moe_inter"])
        local_intermediate = intermediate // len(self.devices)
        for device in self.devices:
            with torch.cuda.device(device):
                self._rank_metadata.append(
                    torch.empty(
                        15 if self.store.man.projection_vq else 10,
                        top_k,
                        dtype=torch.long,
                        device=device,
                    )
                )
                self._rank_route_ids.append(
                    torch.arange(
                        top_k,
                        dtype=torch.long,
                        device=device,
                    )
                )
                self._rank_order.append(
                    torch.arange(
                        top_k,
                        dtype=torch.long,
                        device=device,
                    )
                )
                self._rank_weights.append(
                    torch.empty(
                        top_k,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self._rank_workspaces.append(
                    (
                        torch.empty(
                            top_k,
                            2 * local_intermediate,
                            dtype=torch.bfloat16,
                            device=device,
                        ),
                        torch.empty(
                            top_k,
                            hidden,
                            dtype=torch.bfloat16,
                            device=device,
                        ),
                        torch.empty(
                            hidden,
                            dtype=torch.float32,
                            device=device,
                        ),
                    )
                )
                self._contribution_events.append(torch.cuda.Event())

        self.budget_per_rank = safe_budget
        self.budget = safe_budget * len(self.devices)
        self.arena_slots = {}
        for signature, count in specs.items():
            tier = self._signature_tier(signature)
            self.arena_slots[tier] = (
                self.arena_slots.get(tier, 0) + count
            )
        detail = ", ".join(
            f"{tier}={count}"
            for tier, count in sorted(self.arena_slots.items())
        )
        print(
            "[cccp-kimi] packed TP RAM 缓存已分配："
            f"{safe_budget / 2**30:.2f}GiB/卡，"
            f"{sum(specs.values())} 槽/卡（{detail}）；"
            "RAM/VRAM 均保持 p8/p12/p14，运行时不反量化整矩阵",
            flush=True,
        )
        return self.gpu_arena_bytes / 2**30

    def preload(self) -> None:
        started = time.perf_counter()
        if not self.preload_all():
            raise RuntimeError(
                "packed TP RAM offload requires all routed experts in RAM"
            )
        self.pin_host_resident()
        self.build_gpu_arenas()
        self.active = True
        print(
            "[cccp-kimi] packed TP RAM offload 就绪："
            f"主机专家 {self.host_expert_bytes / 2**30:.2f}GiB，"
            f"显存缓存 {self.gpu_arena_bytes / 2**30:.2f}GiB/"
            f"{len(self.devices)}卡，{time.perf_counter() - started:.1f}s",
            flush=True,
        )

    def _shard_expert(
        self,
        expert: PackedExpert,
        rank: int,
    ) -> PackedExpert:
        return tensor_shard_host_expert(
            expert,
            rank=rank,
            ranks=len(self.devices),
            intermediate=int(self.store.cfg["moe_inter"]),
        )

    def _copy_pairs(
        self,
        host: PackedExpert,
        device: DeviceExpert,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if len(host) != len(device):
            raise ValueError("packed host/device projection count mismatch")
        pairs = [
            (source.raw, target.raw)
            for source, target in zip(host, device)
        ]
        if not self._resident_codebooks:
            pairs.extend(
                (source.cb, target.cb)
                for source, target in zip(host, device)
            )
        return pairs

    def _ensure(
        self,
        keys: list[tuple[int, int]],
    ) -> dict[tuple[int, int], tuple[DeviceExpert, ...]]:
        selected = {}
        missing = []
        for key in dict.fromkeys(keys):
            cached = self.cache.get(key)
            if cached is None:
                missing.append(key)
                continue
            self.cache.move_to_end(key)
            for arena in self._rank_arenas:
                arena.touch(key)
            selected[key] = cached
            self.hits += 1
        if not missing:
            self.last_transfer_seconds = 0.0
            return selected

        started = time.perf_counter()
        shard_started = time.perf_counter()
        pairs_by_rank: list[
            list[tuple[torch.Tensor, torch.Tensor]]
        ] = [[] for _ in self.devices]
        staged: list[
            tuple[tuple[int, int], tuple[DeviceExpert, ...]]
        ] = []
        for key in missing:
            host = self.pinned.get(key)
            if host is None:
                raise KeyError(f"Kimi packed RAM expert missing: {key}")
            rank_values = []
            replaced_keys = set()
            for rank in range(len(self.devices)):
                shard = self._shard_expert(host, rank)
                replaced, value = self._rank_arenas[rank].lease(
                    key,
                    shard,
                    self._rank_device_codebooks[rank],
                )
                if replaced is not None:
                    replaced_keys.add(replaced)
                pairs_by_rank[rank].extend(
                    self._copy_pairs(shard, value)
                )
                rank_values.append(value)
            for replaced in replaced_keys:
                self.cache.pop(replaced, None)
                selected.pop(replaced, None)
            staged.append((key, tuple(rank_values)))
        self.shard_seconds += time.perf_counter() - shard_started

        for stage, pairs in zip(self._rank_stages, pairs_by_rank):
            stage.upload_batch(pairs)
        for stage in self._rank_stages:
            stage.last.synchronize()

        uploaded = sum(
            source.nbytes
            for pairs in pairs_by_rank
            for source, _target in pairs
        )
        for key, values in staged:
            self.cache[key] = values
            selected[key] = values
            self.miss += 1
        self.uploaded_bytes += uploaded
        elapsed = time.perf_counter() - started
        self.last_transfer_seconds = elapsed
        self.transfer_seconds += elapsed
        return selected

    def prefetch(self, _keys) -> None:
        # Routes are GPU-resident and dynamic.  A future predictor can call
        # _ensure on a dedicated ordered queue; the correctness path does not
        # speculate or evict useful slots behind the current token.
        return

    def last_expert_ids(self, layer: int) -> list[int]:
        return self._last_ids[int(layer)]

    def output_hidden(self, layer: int):
        from .ops import TPHidden

        layer = int(layer)
        output = self._output_by_layer.get(layer)
        if output is None:
            output = TPHidden.empty(
                self.devices,
                (1, int(self.store.cfg["routed_hidden"])),
                dtype=torch.bfloat16,
            )
            self._output_by_layer[layer] = output
        return output

    def run_hidden(
        self,
        layer: int,
        value,
        routes: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
    ):
        if (
            not self.active
            or tuple(value.devices) != self.devices
            or value.ready_events is None
            or len(routes) != len(self.devices)
        ):
            raise RuntimeError("packed TP RAM all-rank state is unavailable")

        expert_ids = [
            int(item)
            for item in routes[0][1].reshape(-1).tolist()
        ]
        self._last_ids[int(layer)] = expert_ids
        keys = [(int(layer), expert_id) for expert_id in expert_ids]
        with self._transfer_lock:
            selected = self._ensure(keys)
            rank_experts = [
                [selected[key][rank] for key in keys]
                for rank in range(len(self.devices))
            ]

            p12_positions = [
                position
                for position, expert in enumerate(rank_experts[0])
                if (
                    len(expert) == 2
                    and
                    expert[0].bits == 12
                    and expert[0].dim in (4, 8)
                    and expert[1].bits == 12
                    and expert[1].dim in (4, 8)
                )
            ]
            p12_set = set(p12_positions)
            order_cpu = p12_positions + [
                position
                for position in range(len(expert_ids))
                if position not in p12_set
            ]
            grouped_prefix = len(p12_positions)

            contributions = []
            from .ops import packed_moe_topk

            for rank, device in enumerate(self.devices):
                weights, _indices = routes[rank]
                ordered_experts = [
                    rank_experts[rank][position]
                    for position in order_cpu
                ]
                metadata_cpu = torch.tensor(
                    self._metadata_rows(ordered_experts),
                    dtype=torch.long,
                )
                with torch.cuda.device(device):
                    stream = torch.cuda.current_stream(device)
                    stream.wait_event(value.ready_events[rank])
                    self._rank_metadata[rank][
                        :, : len(ordered_experts)
                    ].copy_(metadata_cpu)
                    self._rank_order[rank][
                        : len(order_cpu)
                    ].copy_(
                        torch.tensor(order_cpu, dtype=torch.long)
                    )
                    torch.index_select(
                        weights.reshape(-1).float().contiguous(),
                        0,
                        self._rank_order[rank][: len(order_cpu)],
                        out=self._rank_weights[rank][: len(order_cpu)],
                    )
                    hidden, output, result = self._rank_workspaces[rank]
                    computed = packed_moe_topk(
                        value.replicas[rank].to(torch.bfloat16),
                        self._rank_route_ids[rank][: len(order_cpu)],
                        self._rank_weights[rank][: len(order_cpu)],
                        self._rank_metadata[rank][
                            :, : len(ordered_experts)
                        ],
                        activation=activation,
                        activation_beta=float(activation_beta),
                        activation_linear_beta=(
                            0.0
                            if activation_linear_beta is None
                            else float(activation_linear_beta)
                        ),
                        hidden_workspace=hidden[: len(order_cpu)],
                        output_workspace=output[: len(order_cpu)],
                        result=result,
                        grouped_prefix=grouped_prefix,
                        **self.store.man.projection_operator_capability(
                            layer
                        ),
                    )
                    if computed is None:
                        raise RuntimeError(
                            "registered packed TP RAM MoE was rejected"
                        )
                    self._contribution_events[rank].record(stream)
                    contributions.append(computed.reshape(-1))

            for device in self.devices:
                with torch.cuda.device(device):
                    stream = torch.cuda.current_stream(device)
                    for event in self._contribution_events:
                        stream.wait_event(event)
            return self.output_hidden(layer).reduce_from(contributions)


__all__ = [
    "PackedTensorHybridPool",
    "tensor_shard_host_expert",
    "tensor_shard_signature",
]

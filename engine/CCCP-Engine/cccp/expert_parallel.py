"""GPU-resident routed-expert parallelism for GLM decode.

Dense weights, attention, router, shared expert, residual state and KV remain
on the primary device. Routed experts use either intermediate-dimension tensor
shards (default) or contiguous expert-ID ownership across ``tp_size`` CUDA
devices and live in permanent signature-specific arenas. Runtime expert lookup
never falls back to RAM or H2D.
"""

from __future__ import annotations

import gc
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .expert_slots import ExpertSignature, GpuExpertArenas
from .kernels import VQWeight, cb_compute

try:
    from . import fusedext as _fusedext

    _routed_slots_fused = (
        _fusedext.moe_mlp_routed_slots_fused
        if _fusedext.available()
        else None
    )
    _routed_codegemm_fused = (
        _fusedext.moe_mlp_routed_codegemm_fused
        if _fusedext.available()
        else None
    )
    _routed_vv_fused = (
        _fusedext.moe_mlp_routed_vv_fused
        if _fusedext.available()
        else None
    )
    _pack_codegemm_shard = (
        _fusedext.pack_vq_tensor_shard_codegemm
        if _fusedext.available()
        else None
    )
    _unpack_codegemm = (
        _fusedext.unpack_vq_codegemm
        if _fusedext.available()
        else None
    )
    _expert_dispatch_pack_fused = (
        _fusedext.expert_dispatch_pack_fused
        if _fusedext.available()
        else None
    )
    _ep_reduce_residual_fused = (
        _fusedext.glm_ep_reduce_residual_fused
        if _fusedext.available()
        else None
    )
except Exception:
    _routed_slots_fused = None
    _routed_codegemm_fused = None
    _routed_vv_fused = None
    _pack_codegemm_shard = None
    _unpack_codegemm = None
    _expert_dispatch_pack_fused = None
    _ep_reduce_residual_fused = None


def expert_owner(expert_id: int, n_experts: int, tp_size: int) -> int:
    """Return the contiguous expert-parallel owner rank."""
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if not 0 <= expert_id < n_experts:
        raise ValueError(f"expert_id {expert_id} outside [0, {n_experts})")
    return min(tp_size - 1, expert_id * tp_size // n_experts)


def reduce_rank_partials(
    partials: list[torch.Tensor],
) -> torch.Tensor:
    """Reduce one already weighted hidden vector per expert rank."""
    if not partials:
        raise ValueError("at least one rank partial is required")
    dtype = partials[0].dtype
    result = partials[0].float()
    for partial in partials[1:]:
        result.add_(partial.float())
    return result.to(dtype)


def _expert_signature(store, layer: int, expert_id: int) -> ExpertSignature:
    kind = store.expert_kind(layer, expert_id)
    if kind == "drop":
        raise KeyError((layer, expert_id))
    base = kind.rstrip("z")
    dim, codebook_size = store.man.vq_dims[base]
    dtype = torch.uint16 if codebook_size > 256 else torch.uint8
    hidden = int(store.cfg["hidden"])
    intermediate = int(store.cfg["moe_inter"])
    return ExpertSignature(
        gu_shape=(2 * intermediate, hidden // dim),
        gu_dtype=dtype,
        dn_shape=(hidden, intermediate // dim),
        dn_dtype=dtype,
    )


def _tensor_shard_signature(
    store,
    layer: int,
    expert_id: int,
    tp_size: int,
) -> ExpertSignature:
    signature = _expert_signature(store, layer, expert_id)
    intermediate = int(store.cfg["moe_inter"])
    if intermediate % tp_size:
        raise ValueError(
            f"moe_inter={intermediate} cannot be split across tp={tp_size}"
        )
    local_intermediate = intermediate // tp_size
    kind = store.expert_kind(layer, expert_id).rstrip("z")
    dim, _codebook_size = store.man.vq_dims[kind]
    if local_intermediate % dim:
        raise ValueError(
            f"local intermediate={local_intermediate} is not divisible "
            f"by VQ dim={dim}"
        )
    return ExpertSignature(
        gu_shape=(2 * local_intermediate, signature.gu_shape[1]),
        gu_dtype=signature.gu_dtype,
        dn_shape=(signature.dn_shape[0], local_intermediate // dim),
        dn_dtype=signature.dn_dtype,
    )


def _runtime_expert_layers(store) -> list[int]:
    """Return only expert layers executed by the configured base model."""
    available = {
        int(layer) for layer in store.man.expert_files
    }
    configured = store.cfg.get("moe_layers")
    if configured is not None:
        return sorted(
            int(layer)
            for layer in configured
            if int(layer) in available
        )
    n_layers = int(store.cfg.get("n_layers", 0))
    return sorted(
        layer
        for layer in available
        if n_layers <= 0 or layer < n_layers
    )


def _resident_codebook_profile(
    store,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """Describe resident index dtypes and VQ shapes without model names."""
    used: set[str] = set()
    n_experts = int(store.cfg["n_experts"])
    for layer in _runtime_expert_layers(store):
        for expert_id in range(n_experts):
            kind = store.expert_kind(int(layer), expert_id)
            if kind != "drop":
                used.add(kind.rstrip("z"))
    if not used:
        return (), (), ()
    shapes = {store.man.vq_dims[kind] for kind in used}
    formats = tuple(
        sorted(
            {
                "u16" if codebook_size > 256 else "u8"
                for _dim, codebook_size in shapes
            }
        )
    )
    return (
        formats,
        tuple(sorted({int(dim) for dim, _size in shapes})),
        tuple(sorted({int(size) for _dim, size in shapes})),
    )


def _codegemm_codebook_sizes(store) -> tuple[int, ...]:
    """Return the resident D4 codebook profile, or empty when unsupported."""
    _formats, dims, sizes = _resident_codebook_profile(store)
    if dims != (4,) or not set(sizes).issubset({256, 4096}):
        return ()
    return sizes


def _supports_codegemm(store) -> bool:
    """Whether resident experts use supported D4 u8/u16 codebooks."""
    return bool(_codegemm_codebook_sizes(store))


@dataclass(frozen=True)
class ExpertParallelPlan:
    tp_size: int
    keys_by_rank: tuple[tuple[tuple[int, int], ...], ...]
    specs_by_rank: tuple[tuple[tuple[ExpertSignature, int], ...], ...]
    bytes_by_rank: tuple[int, ...]

    @property
    def total_bytes(self) -> int:
        return sum(self.bytes_by_rank)


def build_expert_parallel_plan(store, tp_size: int) -> ExpertParallelPlan:
    """Build a deterministic contiguous-ID ownership and exact byte plan."""
    n_experts = int(store.cfg["n_experts"])
    keys: list[list[tuple[int, int]]] = [[] for _ in range(tp_size)]
    counts: list[Counter] = [Counter() for _ in range(tp_size)]
    for layer in _runtime_expert_layers(store):
        for expert_id in range(n_experts):
            if store.expert_kind(layer, expert_id) == "drop":
                continue
            rank = expert_owner(expert_id, n_experts, tp_size)
            key = (layer, expert_id)
            signature = _expert_signature(store, layer, expert_id)
            keys[rank].append(key)
            counts[rank][signature] += 1
    specs = [
        tuple(sorted(counter.items(), key=lambda item: repr(item[0])))
        for counter in counts
    ]
    sizes = [
        sum(signature.slot_bytes * count for signature, count in spec)
        for spec in specs
    ]
    return ExpertParallelPlan(
        tp_size=tp_size,
        keys_by_rank=tuple(tuple(items) for items in keys),
        specs_by_rank=tuple(specs),
        bytes_by_rank=tuple(sizes),
    )


def build_expert_tensor_parallel_plan(
    store,
    tp_size: int,
) -> ExpertParallelPlan:
    """Plan one intermediate-dimension shard of every expert per rank."""
    keys: list[tuple[int, int]] = []
    counts: Counter = Counter()
    n_experts = int(store.cfg["n_experts"])
    for layer in _runtime_expert_layers(store):
        for expert_id in range(n_experts):
            if store.expert_kind(layer, expert_id) == "drop":
                continue
            key = (layer, expert_id)
            keys.append(key)
            counts[
                _tensor_shard_signature(
                    store,
                    layer,
                    expert_id,
                    tp_size,
                )
            ] += 1
    specs = tuple(
        sorted(counts.items(), key=lambda item: repr(item[0]))
    )
    size = sum(
        signature.slot_bytes * count for signature, count in specs
    )
    return ExpertParallelPlan(
        tp_size=tp_size,
        keys_by_rank=tuple(tuple(keys) for _ in range(tp_size)),
        specs_by_rank=tuple(specs for _ in range(tp_size)),
        bytes_by_rank=tuple(size for _ in range(tp_size)),
    )


class _ResidentShard:
    def __init__(
        self,
        rank: int,
        device: torch.device,
        specs: tuple[tuple[ExpertSignature, int], ...],
        codegemm_enabled: bool = False,
    ):
        self.rank = rank
        self.device = device
        self.arenas = GpuExpertArenas(specs, device)
        self.codegemm_enabled = bool(codegemm_enabled)
        self.experts: dict[tuple[int, int], tuple[VQWeight, VQWeight]] = {}
        self._codebooks: dict[int, torch.Tensor] = {}
        self._runtime_codebooks: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.routed_metadata: torch.Tensor | None = None
        self.routed_hidden: torch.Tensor | None = None
        self.routed_output: torch.Tensor | None = None
        self.routed_result: torch.Tensor | None = None
        self.routed_x: torch.Tensor | None = None
        self.routed_ids: torch.Tensor | None = None
        self.routed_weights: torch.Tensor | None = None
        self.codegemm_gu_sum: torch.Tensor | None = None
        self.codegemm_activation: torch.Tensor | None = None
        self.codegemm_dn_sum: torch.Tensor | None = None
        self.codegemm_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.codegemm_dispatch_graphs: dict[
            int,
            torch.cuda.CUDAGraph,
        ] = {}
        self.codegemm_graph_outputs: dict[int, torch.Tensor] = {}
        self.stream = torch.cuda.Stream(device=device)

    @property
    def nbytes(self) -> int:
        return self.arenas.nbytes

    def _codebook(self, source: torch.Tensor) -> torch.Tensor:
        key = source.data_ptr()
        result = self._codebooks.get(key)
        if result is None:
            result = source.to(self.device)
            self._codebooks[key] = result
        return result

    def add(
        self,
        key: tuple[int, int],
        expert: tuple[VQWeight, VQWeight],
    ) -> None:
        _lease, gu_target, dn_target = self.arenas.lease(key, expert)
        gu, dn = expert
        gu_target.copy_(gu.idx)
        dn_target.copy_(dn.idx)
        self.experts[key] = (
            VQWeight(gu_target, self._codebook(gu.cb), gu.cols),
            VQWeight(dn_target, self._codebook(dn.cb), dn.cols),
        )

    def add_tensor_shard(
        self,
        key: tuple[int, int],
        expert: tuple[VQWeight, VQWeight],
        shard_start: int,
        shard_end: int,
    ) -> None:
        """Copy one expert's intermediate slice into this permanent arena."""
        gu, dn = expert
        intermediate = dn.cols
        local_intermediate = shard_end - shard_start
        if not (
            0 <= shard_start < shard_end <= intermediate
            and gu.idx.shape[0] == 2 * intermediate
        ):
            raise ValueError(
                f"invalid expert tensor shard [{shard_start}:{shard_end}] "
                f"for intermediate={intermediate}"
            )
        if shard_start % dn.dim or shard_end % dn.dim:
            raise ValueError(
                f"expert tensor shard boundaries must align to VQ dim={dn.dim}"
            )
        # Gate 和 Up 在原权重中按 [I, I] 连续排列，切片后仍保持
        # [local gate, local up]，使现有 SwiGLU kernel 无需改数值顺序。
        block_start = shard_start // dn.dim
        block_end = shard_end // dn.dim
        local_dn_idx = dn.idx[:, block_start:block_end]
        # 这里只用视图形状选择 arena；实际写入分两段完成，避免为每个专家
        # 在 CPU 上执行一次多线程 torch.cat 和分配一个临时拼接副本。
        signature_expert = (
            VQWeight(
                gu.idx[:2 * local_intermediate],
                gu.cb,
                gu.cols,
            ),
            VQWeight(local_dn_idx, dn.cb, local_intermediate),
        )
        _lease, gu_target, dn_target = self.arenas.lease(
            key,
            signature_expert,
        )
        codegemm_weight = (
            self.codegemm_enabled
            and gu.idx.dtype == torch.uint8
            and dn.idx.dtype == torch.uint8
            and gu.cb.shape == (256, 4)
            and dn.cb.shape == (256, 4)
        )
        if codegemm_weight:
            if (
                _pack_codegemm_shard is None
                or not _pack_codegemm_shard(
                    gu.idx,
                    dn.idx,
                    gu_target,
                    dn_target,
                    intermediate,
                    shard_start,
                    local_intermediate,
                )
            ):
                raise RuntimeError(
                    "CodeGEMM requires uint8 v256/D4 expert indices"
                )
        else:
            gu_target[:local_intermediate].copy_(
                gu.idx[shard_start:shard_end]
            )
            gu_target[local_intermediate:].copy_(
                gu.idx[
                    intermediate + shard_start:
                    intermediate + shard_end
                ]
            )
            dn_target.copy_(local_dn_idx)
        self.experts[key] = (
            VQWeight(gu_target, self._codebook(gu.cb), gu.cols),
            VQWeight(
                dn_target,
                self._codebook(dn.cb),
                local_intermediate,
            ),
        )

    def prefill_expert(
        self,
        key: tuple[int, int],
    ) -> tuple[VQWeight, VQWeight]:
        """Return row-major weights, unpacking only for full-GPU prefill."""
        gu, dn = self.experts[key]
        if (
            not self.codegemm_enabled
            or gu.idx.dtype != torch.uint8
        ):
            return gu, dn
        if _unpack_codegemm is None:
            raise RuntimeError("CodeGEMM unpack kernel is unavailable")
        gu_indices = _unpack_codegemm(
            gu.idx,
            gu.idx.shape[0],
            gu.idx.shape[1],
        )
        dn_indices = _unpack_codegemm(
            dn.idx,
            dn.idx.shape[0],
            dn.idx.shape[1],
        )
        if gu_indices is None or dn_indices is None:
            raise RuntimeError("CodeGEMM prefill unpack failed")
        return (
            VQWeight(gu_indices, gu.cb, gu.cols),
            VQWeight(dn_indices, dn.cb, dn.cols),
        )

    def prepare_device_routing(
        self,
        n_layers: int,
        n_experts: int,
        top_k: int,
        hidden: int,
        intermediate: int,
    ) -> None:
        """Build fixed GPU pointer tables for host-sync-free EP decode."""
        metadata = torch.zeros(
            n_layers,
            10,
            n_experts,
            dtype=torch.long,
        )
        for (layer, expert_id), (gu, dn) in self.experts.items():
            if gu.idx.dtype not in (torch.uint8, torch.uint16):
                raise TypeError(f"unsupported GU index dtype {gu.idx.dtype}")
            if dn.idx.dtype not in (torch.uint8, torch.uint16):
                raise TypeError(f"unsupported DN index dtype {dn.idx.dtype}")
            gu_codebook = cb_compute(
                gu.cb, torch.bfloat16
            ).contiguous()
            dn_codebook = cb_compute(
                dn.cb, torch.bfloat16
            ).contiguous()
            self._runtime_codebooks[(layer, expert_id)] = (
                gu_codebook,
                dn_codebook,
            )
            valid_codegemm = (
                gu.idx.dtype == torch.uint8
                and dn.idx.dtype == torch.uint8
                and gu_codebook.shape == (256, 4)
                and dn_codebook.shape == (256, 4)
            )
            valid_vv = (
                gu.idx.dtype == torch.uint16
                and dn.idx.dtype == torch.uint16
                and gu_codebook.shape == (4096, 4)
                and dn_codebook.shape == (4096, 4)
            )
            if (
                self.codegemm_enabled
                and not (valid_codegemm or valid_vv)
            ):
                raise RuntimeError(
                    "resident mixed CodeGEMM requires D4 "
                    "u8/K256 or u16/K4096 experts"
                )
            metadata[layer, 0, expert_id] = gu.idx.data_ptr()
            metadata[layer, 1, expert_id] = gu_codebook.data_ptr()
            metadata[layer, 2, expert_id] = gu.idx.shape[1]
            metadata[layer, 3, expert_id] = gu_codebook.shape[1]
            metadata[layer, 4, expert_id] = (
                0 if gu.idx.dtype == torch.uint8 else 1
            )
            metadata[layer, 5, expert_id] = dn.idx.data_ptr()
            metadata[layer, 6, expert_id] = dn_codebook.data_ptr()
            metadata[layer, 7, expert_id] = dn.idx.shape[1]
            metadata[layer, 8, expert_id] = dn_codebook.shape[1]
            metadata[layer, 9, expert_id] = (
                0 if dn.idx.dtype == torch.uint8 else 1
            )
        self.routed_metadata = metadata.to(self.device)
        self.routed_hidden = torch.empty(
            top_k,
            2 * intermediate,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.routed_output = torch.empty(
            top_k,
            hidden,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.routed_result = torch.empty(
            hidden,
            dtype=torch.float32,
            device=self.device,
        )
        self.routed_x = torch.empty(
            1,
            hidden,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.routed_ids = torch.empty(
            top_k,
            dtype=torch.long,
            device=self.device,
        )
        self.routed_weights = torch.empty(
            top_k,
            dtype=torch.float32,
            device=self.device,
        )
        if self.codegemm_enabled:
            self.codegemm_gu_sum = torch.empty(
                top_k,
                2 * intermediate,
                dtype=torch.float32,
                device=self.device,
            )
            self.codegemm_activation = torch.empty(
                top_k,
                intermediate,
                dtype=torch.bfloat16,
                device=self.device,
            )
            self.codegemm_dn_sum = torch.empty(
                top_k,
                hidden,
                dtype=torch.float32,
                device=self.device,
            )


class GpuResidentExpertParallel:
    """All routed experts resident across a group of CUDA devices."""

    full_resident = True

    def __init__(
        self,
        store,
        tp_size: int,
        primary_device: torch.device | str,
        layout: str | None = None,
    ):
        if tp_size < 2:
            raise ValueError("expert parallel requires tp_size >= 2")
        if not torch.cuda.is_available():
            raise RuntimeError("expert parallel requires CUDA")
        self.store = store
        self.tp_size = int(tp_size)
        primary = torch.device(primary_device)
        primary_index = (
            torch.cuda.current_device()
            if primary.index is None
            else primary.index
        )
        self.devices = tuple(
            torch.device("cuda", primary_index + rank)
            for rank in range(self.tp_size)
        )
        if self.devices[-1].index >= torch.cuda.device_count():
            raise RuntimeError(
                f"tp={tp_size} requires CUDA devices "
                f"{primary_index}..{self.devices[-1].index}, "
                f"but only {torch.cuda.device_count()} visible"
            )
        self.primary_device = self.devices[0]
        requested_layout = (
            layout
            if layout is not None
            else os.environ.get("CCCP_EP_LAYOUT", "tensor")
        ).strip().lower()
        if requested_layout not in ("expert", "tensor"):
            raise ValueError(
                "CCCP_EP_LAYOUT must be 'expert' or 'tensor'"
            )
        self.layout = requested_layout
        self.tensor_sharded = self.layout == "tensor"
        requested_codegemm = (
            os.environ.get("CCCP_CODEGEMM_VQ", "0") == "1"
        )
        (
            self.resident_packed_formats,
            self.resident_code_dims,
            self.resident_codebook_sizes,
        ) = _resident_codebook_profile(store)
        self.codegemm_codebook_sizes = (
            self.resident_codebook_sizes
            if (
                self.resident_code_dims == (4,)
                and set(self.resident_codebook_sizes).issubset(
                    {256, 4096}
                )
            )
            else ()
        )
        self.codegemm_enabled = bool(
            requested_codegemm
            and self.tensor_sharded
            and _routed_codegemm_fused is not None
            and (
                4096 not in self.codegemm_codebook_sizes
                or _routed_vv_fused is not None
            )
            and _pack_codegemm_shard is not None
            and _unpack_codegemm is not None
            and self.codegemm_codebook_sizes
        )
        if requested_codegemm and not self.codegemm_enabled:
            print(
                "[cccp] CodeGEMM 专家路径不兼容，保留原 v256/D4 算子",
                flush=True,
            )
        self.global_intermediate = int(store.cfg["moe_inter"])
        self.local_intermediate = (
            self.global_intermediate // self.tp_size
            if self.tensor_sharded
            else self.global_intermediate
        )
        self.plan = (
            build_expert_tensor_parallel_plan(store, self.tp_size)
            if self.tensor_sharded
            else build_expert_parallel_plan(store, self.tp_size)
        )
        self.shards: list[_ResidentShard] = []
        self.active = False
        self.failure_reason: str | None = None
        self.peer_access = True
        self.hits = 0
        self.miss = 0
        self.budget = self.plan.total_bytes
        self._profile_enabled = False
        self._profile_cuda_events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = {}
        self._profile_cpu: dict[str, tuple[int, float]] = {}
        self._profile_route_ids: list[torch.Tensor] = []
        self._source_events: list[torch.cuda.Event] = []
        self._done_events: list[list[torch.cuda.Event]] = []
        self._return_buffers: list[torch.Tensor | None] = []
        self._dispatch_source_x: torch.Tensor | None = None
        self._dispatch_source_ids: torch.Tensor | None = None
        self._dispatch_source_weights: torch.Tensor | None = None

    @property
    def gpu_storage_bytes(self) -> int:
        return sum(shard.nbytes for shard in self.shards)

    @property
    def gpu_arena_bytes(self) -> int:
        return self.gpu_storage_bytes

    @property
    def host_expert_bytes(self) -> int:
        return 0

    @property
    def profile_enabled(self) -> bool:
        return self._profile_enabled

    def start_profile(self) -> None:
        """Enable opt-in decode probes without changing the compute path."""
        self._profile_cuda_events.clear()
        self._profile_cpu.clear()
        self._profile_route_ids.clear()
        self._profile_enabled = True

    def profile_event(
        self,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.cuda.Event | None:
        if not self._profile_enabled:
            return None
        event = torch.cuda.Event(enable_timing=True)
        if stream is None:
            stream = torch.cuda.current_stream(self.primary_device)
        event.record(stream)
        return event

    def profile_cuda(
        self,
        name: str,
        start: torch.cuda.Event | None,
        end: torch.cuda.Event | None,
    ) -> None:
        if not self._profile_enabled or start is None or end is None:
            return
        self._profile_cuda_events.setdefault(name, []).append((start, end))

    def profile_cpu(self, name: str, seconds: float) -> None:
        if not self._profile_enabled:
            return
        calls, total = self._profile_cpu.get(name, (0, 0.0))
        self._profile_cpu[name] = (calls + 1, total + seconds)

    def finish_profile(self) -> dict:
        """Synchronize all TP devices and resolve the current probe window."""
        if not self._profile_enabled:
            return {}
        for device in self.devices:
            torch.cuda.synchronize(device)
        self._profile_enabled = False
        cuda = {}
        for name, pairs in self._profile_cuda_events.items():
            total_ms = sum(start.elapsed_time(end) for start, end in pairs)
            cuda[name] = {
                "calls": len(pairs),
                "total_ms": total_ms,
                "ms_call": total_ms / len(pairs) if pairs else 0.0,
            }
        cpu = {}
        for name, (calls, total_s) in self._profile_cpu.items():
            total_ms = total_s * 1000.0
            cpu[name] = {
                "calls": calls,
                "total_ms": total_ms,
                "ms_call": total_ms / calls if calls else 0.0,
            }
        route_balance = None
        if self._profile_route_ids:
            counts_by_rank = [0] * self.tp_size
            max_owned_total = 0
            total_slots = 0
            histogram: Counter = Counter()
            for route_ids in self._profile_route_ids:
                if self.tensor_sharded:
                    # 每个 rank 都处理全部 Top-K，但只计算 1/tp 中间维。
                    owned = [route_ids.numel()] * self.tp_size
                else:
                    owned = [0] * self.tp_size
                    for expert_id in route_ids.tolist():
                        owned[self.owner(int(expert_id))] += 1
                counts_by_rank = [
                    total + count
                    for total, count in zip(counts_by_rank, owned)
                ]
                max_owned_total += max(owned)
                total_slots += sum(owned)
                histogram[tuple(owned)] += 1
            calls = len(self._profile_route_ids)
            ideal = total_slots / max(1, calls * self.tp_size)
            average_max = max_owned_total / calls
            route_balance = {
                "calls": calls,
                "counts_by_rank": counts_by_rank,
                "average_max_owned": average_max,
                "ideal_owned": ideal,
                "imbalance_factor": (
                    average_max / ideal if ideal else 1.0
                ),
                "histogram": {
                    ",".join(str(item) for item in owned): count
                    for owned, count in sorted(histogram.items())
                },
            }
        self._profile_route_ids.clear()
        return {
            "tp_size": self.tp_size,
            "cuda": cuda,
            "cpu": cpu,
            "route_balance": route_balance,
        }

    def _check_p2p(self) -> bool:
        primary = self.primary_device.index
        assert primary is not None
        for device in self.devices[1:]:
            peer = device.index
            assert peer is not None
            if not (
                torch.cuda.can_device_access_peer(primary, peer)
                and torch.cuda.can_device_access_peer(peer, primary)
            ):
                self.peer_access = False
        if not self.peer_access:
            print(
                "[cccp] Expert Parallel 未检测到双向 CUDA P2P；"
                "激活数据将走 CUDA staged peer copy，专家权重仍保持全显存",
                flush=True,
            )
        return True

    def _check_capacity(self) -> bool:
        reserve = float(os.environ.get(
            "CCCP_VRAM_HEADROOM_GB",
            os.environ.get("CCCP_VRAM_RESERVE_GB", "1"),
        ))
        reserve_bytes = int(reserve * 2**30)
        details = []
        enough = True
        for rank, (device, required) in enumerate(
            zip(self.devices, self.plan.bytes_by_rank)
        ):
            with torch.cuda.device(device):
                free, total = torch.cuda.mem_get_info(device)
            available = max(0, free - reserve_bytes)
            details.append(
                f"cuda:{device.index} 专家需 {required / 2**30:.2f}GB / "
                f"可用 {available / 2**30:.2f}GB"
            )
            if required > available:
                enough = False
        if not enough:
            self.failure_reason = "；".join(details)
        return enough

    def preload_if_fits(self) -> bool:
        """Activate only when every shard fits; otherwise leave GPU untouched."""
        if not self._check_p2p() or not self._check_capacity():
            print(
                f"[cccp] TP={self.tp_size} 全显存专家未激活："
                f"{self.failure_reason}",
                flush=True,
            )
            return False

        for rank, (device, specs) in enumerate(
            zip(self.devices, self.plan.specs_by_rank)
        ):
            free, total = torch.cuda.mem_get_info(device)
            allocated = torch.cuda.memory_allocated(device)
            reserve = float(os.environ.get(
                "CCCP_VRAM_HEADROOM_GB",
                os.environ.get("CCCP_VRAM_RESERVE_GB", "1"),
            ))
            fraction = max(
                0.10,
                min(
                    0.99,
                    (
                        allocated / 2**30
                        + free / 2**30
                        - reserve
                    )
                    / (total / 2**30),
                ),
            )
            torch.cuda.set_per_process_memory_fraction(
                fraction, device=device.index
            )
            self.shards.append(
                _ResidentShard(
                    rank,
                    device,
                    specs,
                    codegemm_enabled=self.codegemm_enabled,
                )
            )

        total_shards = sum(
            len(items) for items in self.plan.keys_by_rank
        )
        unique_experts = len(
            {
                key
                for items in self.plan.keys_by_rank
                for key in items
            }
        )
        print(
            f"[cccp] TP={self.tp_size} 专家显存 arena 已分配："
            f"{self.plan.total_bytes / 2**30:.2f}GB，"
            + "，".join(
                f"cuda:{device.index}={size / 2**30:.2f}GB"
                for device, size in zip(self.devices, self.plan.bytes_by_rank)
            )
            + (
                f"；intermediate 张量分片，读取 {unique_experts} 个专家"
                f"并写入 {total_shards} 个半宽分片"
                if self.tensor_sharded
                else f"；开始填充 {unique_experts} 个专家"
            ),
            flush=True,
        )
        if self.codegemm_enabled:
            print(
                "[cccp] 全显存专家采用 CodeGEMM Psumbook 索引布局"
                "（同字节原位重排；RAM 专家布局不变）",
                flush=True,
            )
        started = time.time()
        loaded = 0
        if self.tensor_sharded:
            # 每个专家只从模型文件读取一次，再把不重叠的 intermediate
            # 切片直接写入各卡；不建立 RAM 镜像，也不重复读盘。
            #
            # Down-Proj 的列切片在 CPU 上不是连续视图，直接 copy_ 会为
            # 19K 个专家反复启动多线程打包。先把每个完整索引矩阵连续上传
            # 到主卡复用 staging，再由各卡 stream 在 GPU 上提取切片。
            staging_by_signature: dict[
                ExpertSignature,
                tuple[torch.Tensor, torch.Tensor],
            ] = {}
            primary_stream = torch.cuda.current_stream(
                self.primary_device
            )
            stage_ready = torch.cuda.Event()
            stage_consumed = [
                torch.cuda.Event() for _ in self.shards
            ]
            for key in self.plan.keys_by_rank[0]:
                expert = self.store.load_expert(*key)
                gu, dn = expert
                signature = ExpertSignature.of(expert)
                staging = staging_by_signature.get(signature)
                if staging is None:
                    with torch.cuda.device(self.primary_device):
                        staging = (
                            torch.empty_like(
                                gu.idx,
                                device=self.primary_device,
                            ),
                            torch.empty_like(
                                dn.idx,
                                device=self.primary_device,
                            ),
                        )
                    staging_by_signature[signature] = staging
                staging_gu, staging_dn = staging
                with (
                    torch.cuda.device(self.primary_device),
                    torch.cuda.stream(primary_stream),
                ):
                    staging_gu.copy_(gu.idx)
                    staging_dn.copy_(dn.idx)
                    stage_ready.record(primary_stream)
                staged_expert = (
                    VQWeight(staging_gu, gu.cb, gu.cols),
                    VQWeight(staging_dn, dn.cb, dn.cols),
                )
                for rank, shard in enumerate(self.shards):
                    shard_start = rank * self.local_intermediate
                    shard_end = shard_start + self.local_intermediate
                    with (
                        torch.cuda.device(shard.device),
                        torch.cuda.stream(shard.stream),
                    ):
                        shard.stream.wait_event(stage_ready)
                        shard.add_tensor_shard(
                            key,
                            staged_expert,
                            shard_start,
                            shard_end,
                        )
                        stage_consumed[rank].record(shard.stream)
                # 下一专家复用 staging 前，主卡上传 stream 等待所有分片
                # copy 完成；只建立设备依赖，不做逐专家 CPU synchronize。
                with (
                    torch.cuda.device(self.primary_device),
                    torch.cuda.stream(primary_stream),
                ):
                    for done in stage_consumed:
                        primary_stream.wait_event(done)
                del staged_expert
                del expert
                loaded += 1
                if loaded % 1000 == 0:
                    print(
                        f"[cccp] 专家权重读取/分片写入显存 "
                        f"{loaded}/{unique_experts}",
                        flush=True,
                    )
            for shard in self.shards:
                torch.cuda.synchronize(shard.device)
            staging_by_signature.clear()
        else:
            for rank, keys in enumerate(self.plan.keys_by_rank):
                shard = self.shards[rank]
                with torch.cuda.device(shard.device):
                    for key in keys:
                        expert = self.store.load_expert(*key)
                        shard.add(key, expert)
                        del expert
                        loaded += 1
                        if loaded % 1000 == 0:
                            print(
                                f"[cccp] 专家权重写入显存 "
                                f"{loaded}/{unique_experts}",
                                flush=True,
                            )
                    torch.cuda.synchronize(shard.device)

        # CPU codebooks were only loader inputs; all runtime VQWeight objects
        # now reference device copies.
        self.store._cb_cache.clear()
        gc.collect()
        self.prepare_device_routing()
        self.prepare_codegemm_graphs()
        self.active = True
        print(
            f"[cccp] TP={self.tp_size} 专家权重写入完成："
            f"{loaded} 个唯一专家，{time.time() - started:.1f}s，"
            f"布局={self.layout}，运行时专家 H2D=0",
            flush=True,
        )
        return True

    def owner(self, expert_id: int) -> int:
        return expert_owner(
            expert_id, int(self.store.cfg["n_experts"]), self.tp_size
        )

    def _expert(self, layer: int, expert_id: int):
        if self.tensor_sharded:
            raise RuntimeError(
                "tensor-sharded experts do not have a single owning rank"
            )
        rank = self.owner(expert_id)
        return self.shards[rank].experts[(layer, expert_id)]

    def prefetch(self, _keys) -> None:
        return

    def prepare_device_routing(self) -> bool:
        """Prepare CUDA-resident route descriptors for every expert shard."""
        if (
            self.codegemm_enabled
            and _routed_codegemm_fused is None
        ) or (
            not self.codegemm_enabled
            and _routed_slots_fused is None
        ):
            return False
        cfg = self.store.cfg
        n_layers = max(
            int(cfg.get("n_layers", 0)),
            max(int(layer) for layer in self.store.man.expert_files) + 1,
        )
        n_experts = int(cfg["n_experts"])
        top_k = int(cfg["top_k"])
        hidden = int(cfg["hidden"])
        intermediate = int(cfg["moe_inter"])
        for shard in self.shards:
            shard.prepare_device_routing(
                n_layers,
                n_experts,
                top_k,
                hidden,
                (
                    self.local_intermediate
                    if self.tensor_sharded
                    else intermediate
                ),
            )
        self._source_events = [
            torch.cuda.Event() for _ in range(n_layers)
        ]
        self._done_events = [
            [torch.cuda.Event() for _ in range(n_layers)]
            for _ in self.shards
        ]
        self._return_buffers = [
            None
            if rank == 0
            else torch.empty(
                n_layers,
                hidden,
                dtype=torch.float32,
                device=self.primary_device,
            )
            for rank in range(self.tp_size)
        ]
        self._dispatch_source_x = torch.empty(
            1,
            hidden,
            dtype=torch.float32,
            device=self.primary_device,
        )
        self._dispatch_source_ids = torch.empty(
            top_k,
            dtype=torch.long,
            device=self.primary_device,
        )
        self._dispatch_source_weights = torch.empty(
            top_k,
            dtype=torch.float32,
            device=self.primary_device,
        )
        return True

    def decode_norm_output(self) -> torch.Tensor | None:
        """Fixed FP32 normalized row consumed by dispatch CUDA Graphs."""
        if (
            not self.codegemm_enabled
            or os.environ.get(
                "CCCP_CODEGEMM_DISPATCH_GRAPH",
                "1",
            ) == "0"
        ):
            return None
        return self._dispatch_source_x

    def decode_route_outputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Fixed route outputs consumed by dispatch CUDA Graphs."""
        if (
            self.decode_norm_output() is None
            or self._dispatch_source_weights is None
            or self._dispatch_source_ids is None
        ):
            return None
        return (
            self._dispatch_source_weights.view(1, -1),
            self._dispatch_source_ids.view(1, -1),
        )

    def _run_registered_resident_moe(
        self,
        shard: _ResidentShard,
        value: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        metadata: torch.Tensor,
        result: torch.Tensor,
    ) -> torch.Tensor | None:
        """Dispatch the resident mixed-codebook kernel by capability."""
        from .ops import resident_moe_topk

        assert shard.routed_hidden is not None
        assert shard.routed_output is not None
        if self.codegemm_enabled:
            assert shard.codegemm_gu_sum is not None
            assert shard.codegemm_activation is not None
            assert shard.codegemm_dn_sum is not None
            packed_formats = (
                ("psumbook_u8", "u16")
                if 4096 in self.codegemm_codebook_sizes
                else ("psumbook_u8",)
            )
        else:
            packed_formats = self.resident_packed_formats
        return resident_moe_topk(
            value,
            route_ids,
            route_weights,
            metadata,
            activation="swiglu",
            limit=0.0,
            codegemm_gu_workspace=shard.codegemm_gu_sum,
            codegemm_activation_workspace=(
                shard.codegemm_activation
            ),
            codegemm_down_workspace=shard.codegemm_dn_sum,
            hidden_workspace=shard.routed_hidden,
            output_workspace=shard.routed_output,
            result=result,
            packed_formats=packed_formats,
            code_dims=self.resident_code_dims,
            codebook_sizes=(
                self.codegemm_codebook_sizes
                if self.codegemm_enabled
                else self.resident_codebook_sizes
            ),
        )

    def prepare_codegemm_graphs(self) -> bool:
        """Capture the fixed-buffer routed kernel chain once per layer/rank."""
        if (
            not self.codegemm_enabled
            or _routed_codegemm_fused is None
            or _expert_dispatch_pack_fused is None
            or os.environ.get("CCCP_CODEGEMM_GRAPH", "1") == "0"
            or os.environ.get("CCCP_EP_DIRECT_RETURN", "1") == "0"
        ):
            return False
        layers = _runtime_expert_layers(self.store)
        assert self._dispatch_source_x is not None
        assert self._dispatch_source_ids is not None
        assert self._dispatch_source_weights is not None
        self._dispatch_source_x.zero_()
        self._dispatch_source_weights.fill_(
            1.0 / self._dispatch_source_weights.numel()
        )
        started = time.time()
        for rank, shard in enumerate(self.shards):
            assert shard.routed_x is not None
            assert shard.routed_ids is not None
            assert shard.routed_weights is not None
            assert shard.routed_metadata is not None
            assert shard.codegemm_gu_sum is not None
            assert shard.codegemm_activation is not None
            assert shard.codegemm_dn_sum is not None
            assert shard.routed_result is not None
            first_expert_by_layer: dict[int, int] = {}
            for layer, expert_id in shard.experts:
                first_expert_by_layer.setdefault(layer, expert_id)
            with torch.cuda.device(shard.device):
                shard.routed_x.zero_()
                shard.routed_weights.fill_(
                    1.0 / shard.routed_weights.numel()
                )
                for layer in layers:
                    expert_id = first_expert_by_layer[layer]
                    shard.routed_ids.fill_(expert_id)
                    result_buffer = (
                        shard.routed_result
                        if rank == 0
                        else self._return_buffers[rank][layer]
                    )
                    assert result_buffer is not None
                    with torch.cuda.stream(shard.stream):
                        self._run_registered_resident_moe(
                            shard,
                            shard.routed_x,
                            shard.routed_ids,
                            shard.routed_weights,
                            shard.routed_metadata[layer],
                            result_buffer,
                        )
                    shard.stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(
                        graph,
                        stream=shard.stream,
                    ):
                        self._run_registered_resident_moe(
                            shard,
                            shard.routed_x,
                            shard.routed_ids,
                            shard.routed_weights,
                            shard.routed_metadata[layer],
                            result_buffer,
                        )
                    shard.codegemm_graphs[layer] = graph
                    shard.codegemm_graph_outputs[layer] = (
                        result_buffer
                    )
                    self._dispatch_source_ids.fill_(expert_id)
                    torch.cuda.synchronize(self.primary_device)
                    with torch.cuda.stream(shard.stream):
                        dispatched = _expert_dispatch_pack_fused(
                            self._dispatch_source_x,
                            self._dispatch_source_ids,
                            self._dispatch_source_weights,
                            shard.routed_x,
                            shard.routed_ids,
                            shard.routed_weights,
                        )
                        if not dispatched:
                            raise RuntimeError(
                                "CodeGEMM dispatch graph warmup failed"
                            )
                        self._run_registered_resident_moe(
                            shard,
                            shard.routed_x,
                            shard.routed_ids,
                            shard.routed_weights,
                            shard.routed_metadata[layer],
                            result_buffer,
                        )
                    shard.stream.synchronize()
                    dispatch_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(
                        dispatch_graph,
                        stream=shard.stream,
                    ):
                        _expert_dispatch_pack_fused(
                            self._dispatch_source_x,
                            self._dispatch_source_ids,
                            self._dispatch_source_weights,
                            shard.routed_x,
                            shard.routed_ids,
                            shard.routed_weights,
                        )
                        self._run_registered_resident_moe(
                            shard,
                            shard.routed_x,
                            shard.routed_ids,
                            shard.routed_weights,
                            shard.routed_metadata[layer],
                            result_buffer,
                        )
                    shard.codegemm_dispatch_graphs[layer] = (
                        dispatch_graph
                    )
        for device in self.devices:
            torch.cuda.synchronize(device)
        print(
            f"[cccp] CodeGEMM CUDA Graph 捕获完成："
            f"{len(layers)} 层 × {self.tp_size} 卡，"
            f"{time.time() - started:.1f}s",
            flush=True,
        )
        return True

    def _device_routing_ready(self, layer: int) -> bool:
        return (
            (
                _routed_codegemm_fused is not None
                if self.codegemm_enabled
                else _routed_slots_fused is not None
            )
            and bool(self.shards)
            and all(
                shard.routed_metadata is not None
                and layer < shard.routed_metadata.shape[0]
                and shard.routed_x is not None
                and shard.routed_ids is not None
                and shard.routed_weights is not None
                for shard in self.shards
            )
            and layer < len(self._source_events)
        )

    def _compute_decode_device_routed(
        self,
        x: torch.Tensor,
        layer: int,
        route_ids: torch.Tensor,
        weights: torch.Tensor,
        shared: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        shared_fn: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run resident Top-K work on every rank without a CUDA-to-host sync."""
        if self.codegemm_enabled:
            assert _routed_codegemm_fused is not None
        else:
            assert _routed_slots_fused is not None
        routed_start = self.profile_event()
        source_ready = self._source_events[layer]
        source_ready.record(torch.cuda.current_stream(self.primary_device))
        overlap_shared = (
            shared is None
            and shared_fn is not None
            and residual is not None
            and self.tp_size > 1
            and os.environ.get("CCCP_EP_OVERLAP_SHARED", "1") != "0"
        )
        # Start both routed ranks first, secondary rank first, then enqueue the
        # shared expert on GPU 0's primary stream.  CUDA may overlap the
        # independent rank-0 graph and shared-expert kernels when resources
        # permit, while rank 1 always proceeds independently.
        rank_order = (
            list(range(1, self.tp_size)) + [0]
            if overlap_shared
            else list(range(self.tp_size))
        )
        pending: dict[int, tuple[torch.Tensor, torch.cuda.Event]] = {}
        for rank in rank_order:
            shard = self.shards[rank]
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.stream):
                shard.stream.wait_event(source_ready)
                dispatch_start = self.profile_event(shard.stream)
                assert shard.routed_x is not None
                assert shard.routed_ids is not None
                assert shard.routed_weights is not None
                use_graph = (
                    self.codegemm_enabled
                    and os.environ.get(
                        "CCCP_CODEGEMM_GRAPH",
                        "1",
                    ) != "0"
                    and layer in shard.codegemm_graphs
                )
                use_dispatch_graph = (
                    use_graph
                    and os.environ.get(
                        "CCCP_CODEGEMM_DISPATCH_GRAPH",
                        "1",
                    ) != "0"
                    and layer in shard.codegemm_dispatch_graphs
                    and self._dispatch_source_x is not None
                    and self._dispatch_source_ids is not None
                    and self._dispatch_source_weights is not None
                    and x.data_ptr()
                    == self._dispatch_source_x.data_ptr()
                    and route_ids.data_ptr()
                    == self._dispatch_source_ids.data_ptr()
                    and weights.data_ptr()
                    == self._dispatch_source_weights.data_ptr()
                )
                if use_dispatch_graph:
                    local_x = shard.routed_x
                    local_ids = shard.routed_ids
                    local_weights = shard.routed_weights
                elif rank == 0 and not use_graph:
                    # 模型残差是 FP32，专家 kernel 的 BF16 边界仍需保留；
                    # 但 Top-K ID/权重已经是目标 dtype 和设备，不必再复制。
                    shard.routed_x.copy_(
                        x,
                        non_blocking=True,
                    )
                    local_x = shard.routed_x
                    local_ids = route_ids
                    local_weights = weights
                else:
                    fused_dispatch = (
                        self.peer_access
                        and _expert_dispatch_pack_fused is not None
                        and _expert_dispatch_pack_fused(
                            x,
                            route_ids,
                            weights,
                            shard.routed_x,
                            shard.routed_ids,
                            shard.routed_weights,
                        )
                    )
                    if not fused_dispatch:
                        shard.routed_x.copy_(
                            x,
                            non_blocking=True,
                        )
                        shard.routed_ids.copy_(
                            route_ids,
                            non_blocking=True,
                        )
                        shard.routed_weights.copy_(
                            weights,
                            non_blocking=True,
                        )
                    local_x = shard.routed_x
                    local_ids = shard.routed_ids
                    local_weights = shard.routed_weights
                dispatch_end = self.profile_event(shard.stream)
                assert shard.routed_metadata is not None
                assert shard.routed_result is not None
                direct_peer_return = (
                    rank != 0
                    and self.peer_access
                    and os.environ.get(
                        "CCCP_EP_DIRECT_RETURN",
                        "1",
                    ) != "0"
                )
                result_buffer = (
                    self._return_buffers[rank][layer]
                    if direct_peer_return
                    else shard.routed_result
                )
                if self.codegemm_enabled:
                    assert shard.codegemm_gu_sum is not None
                    assert shard.codegemm_activation is not None
                    assert shard.codegemm_dn_sum is not None
                    if use_dispatch_graph:
                        captured_output = (
                            shard.codegemm_graph_outputs[layer]
                        )
                        if (
                            captured_output.data_ptr() !=
                            result_buffer.data_ptr()
                        ):
                            raise RuntimeError(
                                "CodeGEMM dispatch graph output changed"
                            )
                        shard.codegemm_dispatch_graphs[layer].replay()
                        local_partial = captured_output
                    elif use_graph:
                        captured_output = (
                            shard.codegemm_graph_outputs[layer]
                        )
                        if (
                            captured_output.data_ptr() !=
                            result_buffer.data_ptr()
                        ):
                            raise RuntimeError(
                                "CodeGEMM graph output buffer changed"
                            )
                        shard.codegemm_graphs[layer].replay()
                        local_partial = captured_output
                    else:
                        local_partial = (
                            self._run_registered_resident_moe(
                                shard,
                                local_x,
                                local_ids,
                                local_weights,
                                shard.routed_metadata[layer],
                                result_buffer,
                            )
                        )
                else:
                    assert shard.routed_hidden is not None
                    assert shard.routed_output is not None
                    local_partial = self._run_registered_resident_moe(
                        shard,
                        local_x,
                        local_ids,
                        local_weights,
                        shard.routed_metadata[layer],
                        result_buffer,
                    )
                if local_partial is None:
                    raise RuntimeError(
                        "full-resident device-routed expert kernel rejected "
                        "a validated decode input"
                    )
                compute_end = self.profile_event(shard.stream)
                primary_partial = (
                    local_partial
                    if rank == 0 or direct_peer_return
                    else self._return_buffers[rank][layer]
                )
                if rank != 0 and not direct_peer_return:
                    primary_partial.copy_(
                        local_partial,
                        non_blocking=True,
                    )
                return_end = self.profile_event(shard.stream)
                done = self._done_events[rank][layer]
                done.record(shard.stream)
                self.profile_cuda(
                    f"rank{rank}_dispatch",
                    dispatch_start,
                    dispatch_end,
                )
                self.profile_cuda(
                    f"rank{rank}_expert_compute",
                    dispatch_end,
                    compute_end,
                )
                self.profile_cuda(
                    f"rank{rank}_return",
                    compute_end,
                    return_end,
                )
            pending[rank] = (primary_partial, done)

        primary_stream = torch.cuda.current_stream(self.primary_device)
        if overlap_shared:
            shared_start = self.profile_event(primary_stream)
            assert shared_fn is not None
            shared = shared_fn()
            shared_end = self.profile_event(primary_stream)
            self.profile_cuda(
                "shared_expert",
                shared_start,
                shared_end,
            )
        wait_start = self.profile_event(primary_stream)
        partials = []
        for rank in range(self.tp_size):
            partial, done = pending[rank]
            primary_stream.wait_event(done)
            partials.append(partial)
        wait_end = self.profile_event(primary_stream)
        self.profile_cuda("primary_wait_shards", wait_start, wait_end)

        reduce_start = self.profile_event(primary_stream)
        result = (
            _ep_reduce_residual_fused(
                [*partials, shared],
                residual,
            )
            if (
                shared is not None
                and residual is not None
                and len(partials) == 2
                and _ep_reduce_residual_fused is not None
            )
            else None
        )
        if result is None:
            result = reduce_rank_partials(partials)
            if shared is not None and residual is not None:
                result = residual + (result + shared)
        reduce_end = self.profile_event(primary_stream)
        self.profile_cuda("partial_reduce", reduce_start, reduce_end)
        routed_end = self.profile_event(primary_stream)
        self.profile_cuda("routed_total", routed_start, routed_end)
        self.hits += route_ids.numel()
        if self._profile_enabled:
            self._profile_route_ids.append(route_ids.detach())
        return result

    def compute_final(
        self,
        x: torch.Tensor,
        layer: int,
        indices: torch.Tensor,
        weights: torch.Tensor,
        shared: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor | None:
        """Decode routed TP reduction and final residual in one kernel."""
        if (
            os.environ.get("CCCP_GLM_EP_FINAL_FUSED", "1") == "0"
            or
            x.shape[0] != 1
            or not self._device_routing_ready(layer)
        ):
            return None
        return self._compute_decode_device_routed(
            x,
            layer,
            indices[0],
            weights[0],
            shared,
            residual,
        )

    def compute_final_overlap(
        self,
        x: torch.Tensor,
        layer: int,
        indices: torch.Tensor,
        weights: torch.Tensor,
        shared_fn: Callable[[], torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor | None:
        """Overlap secondary-rank routed experts with GPU-0 shared expert."""
        if (
            os.environ.get("CCCP_EP_OVERLAP_SHARED", "1") == "0"
            or os.environ.get("CCCP_GLM_EP_FINAL_FUSED", "1") == "0"
            or x.shape[0] != 1
            or self.tp_size <= 1
            or not self._device_routing_ready(layer)
        ):
            return None
        return self._compute_decode_device_routed(
            x,
            layer,
            indices[0],
            weights[0],
            residual=residual,
            shared_fn=shared_fn,
        )

    def _compute_decode(
        self,
        x: torch.Tensor,
        layer: int,
        expert_ids: list[int],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        from .grouped import moe_mlp_grouped_partial

        routed_start = self.profile_event()
        if self.tensor_sharded:
            # 每卡持有每个专家的不重叠 intermediate 分片，因此两卡都处理
            # 完整 Top-K；最终 hidden 向量是各分片 Down-Proj 的 FP32 和。
            positions_by_rank = [
                list(range(len(expert_ids)))
                for _ in range(self.tp_size)
            ]
        else:
            positions_by_rank = [
                [] for _ in range(self.tp_size)
            ]
            for position, expert_id in enumerate(expert_ids):
                positions_by_rank[self.owner(expert_id)].append(position)

        weights_by_rank = [
            weights[positions].contiguous() if positions else None
            for positions in positions_by_rank
        ]
        source_ready = torch.cuda.Event()
        source_ready.record(torch.cuda.current_stream(self.primary_device))
        pending = []
        for rank, positions in enumerate(positions_by_rank):
            if not positions:
                continue
            shard = self.shards[rank]
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.stream):
                local_experts = [
                    shard.prefill_expert(
                        (layer, expert_ids[position])
                    )
                    for position in positions
                ]
                shard.stream.wait_event(source_ready)
                dispatch_start = self.profile_event(shard.stream)
                local_x = (
                    x
                    if rank == 0
                    else x.to(shard.device, non_blocking=True)
                )
                local_weights = weights_by_rank[rank]
                assert local_weights is not None
                if rank != 0:
                    local_weights = local_weights.to(
                        shard.device,
                        non_blocking=True,
                    )
                dispatch_end = self.profile_event(shard.stream)
                local_partial = moe_mlp_grouped_partial(
                    local_x,
                    local_experts,
                    local_weights,
                    limit=0.0,
                ).clone()
                compute_end = self.profile_event(shard.stream)
                primary_partial = (
                    local_partial
                    if rank == 0
                    else local_partial.to(
                        self.primary_device, non_blocking=True
                    )
                )
                return_end = self.profile_event(shard.stream)
                done = torch.cuda.Event()
                done.record(shard.stream)
                self.profile_cuda(
                    f"rank{rank}_dispatch",
                    dispatch_start,
                    dispatch_end,
                )
                self.profile_cuda(
                    f"rank{rank}_expert_compute",
                    dispatch_end,
                    compute_end,
                )
                self.profile_cuda(
                    f"rank{rank}_return",
                    compute_end,
                    return_end,
                )
            pending.append((primary_partial, done))

        primary_stream = torch.cuda.current_stream(self.primary_device)
        wait_start = self.profile_event(primary_stream)
        partials = []
        for partial, done in pending:
            primary_stream.wait_event(done)
            partials.append(partial)
        wait_end = self.profile_event(primary_stream)
        self.profile_cuda("primary_wait_shards", wait_start, wait_end)

        reduce_start = self.profile_event(primary_stream)
        result = reduce_rank_partials(partials)
        reduce_end = self.profile_event(primary_stream)
        self.profile_cuda("partial_reduce", reduce_start, reduce_end)
        routed_end = self.profile_event(primary_stream)
        self.profile_cuda("routed_total", routed_start, routed_end)
        self.hits += len(expert_ids)
        return result

    def compute(
        self,
        x: torch.Tensor,
        layer: int,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return routed output [T,D], preserving Top-K merge order."""
        if x.shape[0] > 1:
            return self._compute_prefill(x, layer, indices, weights)
        if self._device_routing_ready(layer):
            return self._compute_decode_device_routed(
                x,
                layer,
                indices[0],
                weights[0],
            ).unsqueeze(0)
        rows = []
        for token in range(x.shape[0]):
            host_start = time.perf_counter()
            expert_ids = indices[token].tolist()
            self.profile_cpu(
                "topk_to_host_inner",
                time.perf_counter() - host_start,
            )
            rows.append(
                self._compute_decode(
                    x[token:token + 1],
                    layer,
                    expert_ids,
                    weights[token],
                )
            )
        return torch.stack(rows)

    def _compute_prefill(
        self,
        x: torch.Tensor,
        layer: int,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Mirror model.py's original multi-token expert/index_add order."""
        result = torch.zeros_like(x)
        expert_ids = indices.unique().tolist()
        for expert_id in expert_ids:
            tokens, slots = (indices == expert_id).nonzero(as_tuple=True)
            selected_x = x[tokens]
            ranks = (
                range(self.tp_size)
                if self.tensor_sharded
                else (self.owner(expert_id),)
            )
            primary_partials = []
            for rank in ranks:
                shard = self.shards[rank]
                with torch.cuda.device(shard.device):
                    gu, dn = shard.prefill_expert(
                        (layer, expert_id)
                    )
                    local_intermediate = dn.cols
                    local_x = (
                        selected_x
                        if rank == 0
                        else selected_x.to(shard.device)
                    )
                    hidden = gu.matmul_T(local_x)
                    activated = (
                        F.silu(hidden[:, :local_intermediate])
                        * hidden[:, local_intermediate:]
                    )
                    local_output = dn.matmul_T(activated)
                    primary_partials.append(
                        local_output
                        if rank == 0
                        else local_output.to(self.primary_device)
                    )
            primary_output = primary_partials[0]
            for partial in primary_partials[1:]:
                primary_output.add_(partial)
            result.index_add_(
                0,
                tokens,
                primary_output
                * weights[tokens, slots].unsqueeze(1),
            )
        self.hits += indices.numel()
        return result

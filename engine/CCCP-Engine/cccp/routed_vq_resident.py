"""Model-independent resident routed-codebook storage and execution.

Projection archives store mixed indices at their real 9/10/12/14-bit width.
Expanding them to uint16 would defeat compact residency. This module keeps
the byte-exact payload in one stable arena per
pipeline rank and publishes CUDA pointer metadata for direct packed GEMV.
All model families use this implementation through the public routed-VQ
runtime.  Architecture adapters provide topology only; this module owns the
common resident layout, codebook expansion, grouped Prefill, and Decode path.
"""

from __future__ import annotations

import ctypes
import gc
import glob
import json
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass

import torch

from .ops.codebook import (
    compile_shared_codebook_image,
    rewrite_packed_codebook_metadata,
    run_compact_q8_codebook_decode,
)
from .prefill import prefill_moe_batch_size


def _dequant_scratch_bytes(
    expert_count: int,
    local_inter: int,
    hidden: int,
) -> int:
    """Return BF16 bytes for gate/up and down dense scratch."""
    return (
        int(expert_count)
        * int(local_inter)
        * int(hidden)
        * 3
        * 2
    )


_CUDA_HOST_REGISTER_MAPPED = 2
_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDART_LIBRARY = None
_CUDART_DLL_DIRECTORY_HANDLES: list[object] = []


def _cudart_candidates() -> tuple[str, ...]:
    """Find the CUDA runtime shipped with the active isolated environment."""
    candidates: list[str] = []
    if os.name == "nt":
        site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
        patterns = (
            os.path.join(site_packages, "torch", "lib", "cudart64_*.dll"),
            os.path.join(
                site_packages, "nvidia", "*", "bin", "**", "cudart64_*.dll"
            ),
            os.path.join(
                os.environ.get("CUDA_HOME", ""), "bin", "cudart64_*.dll"
            ),
        )
        for pattern in patterns:
            if not pattern or pattern.startswith(os.sep + "bin"):
                continue
            candidates.extend(sorted(glob.glob(pattern, recursive=True), reverse=True))
        cuda_major = str(torch.version.cuda or "").split(".", 1)[0]
        if cuda_major.isdigit():
            candidates.append(f"cudart64_{cuda_major}.dll")
        candidates.extend(("cudart64_13.dll", "cudart64_12.dll"))
    else:
        site_packages = os.path.join(
            sys.prefix,
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
        )
        candidates.extend(sorted(glob.glob(
            os.path.join(site_packages, "nvidia", "cuda_runtime", "lib", "libcudart.so*")
        ), reverse=True))
        candidates.extend(("libcudart.so", "libcudart.so.13", "libcudart.so.12"))
    return tuple(dict.fromkeys(candidates))


def _cudart_library():
    """Return libcudart for APIs not exposed by ``torch.cuda.cudart``."""

    global _CUDART_LIBRARY
    if _CUDART_LIBRARY is not None:
        return _CUDART_LIBRARY
    errors = []
    for name in _cudart_candidates():
        try:
            if (
                os.name == "nt"
                and os.path.isabs(name)
                and hasattr(os, "add_dll_directory")
            ):
                directory = os.path.dirname(name)
                try:
                    _CUDART_DLL_DIRECTORY_HANDLES.append(
                        os.add_dll_directory(directory)
                    )
                except OSError:
                    pass
            _CUDART_LIBRARY = ctypes.CDLL(name)
            return _CUDART_LIBRARY
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        "无法加载当前离线环境的 CUDA Runtime：" + "; ".join(errors)
    )


def _cuda_host_device_pointer(pointer: int) -> int:
    """Resolve a mapped host allocation to the UVA pointer used by kernels."""

    function = _cudart_library().cudaHostGetDevicePointer
    function.argtypes = (
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    output = ctypes.c_void_p()
    error = int(
        function(
            ctypes.byref(output),
            ctypes.c_void_p(int(pointer)),
            0,
        )
    )
    if error or not output.value:
        raise RuntimeError(
            f"cudaHostGetDevicePointer failed with CUDA error {error}"
        )
    return int(output.value)


def _cuda_memcpy_h2d_async(
    destination: int,
    source: int,
    nbytes: int,
    stream: int,
) -> None:
    """Submit one registered-host to device copy on an existing stream."""

    function = _cudart_library().cudaMemcpyAsync
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    function.restype = ctypes.c_int
    error = int(
        function(
            ctypes.c_void_p(int(destination)),
            ctypes.c_void_p(int(source)),
            ctypes.c_size_t(int(nbytes)),
            _CUDA_MEMCPY_HOST_TO_DEVICE,
            ctypes.c_void_p(int(stream)),
        )
    )
    if error:
        raise RuntimeError(f"cudaMemcpyAsync failed with CUDA error {error}")


_ROOT = "language_model"


@dataclass(frozen=True)
class RoutedVQLayoutPlan:
    ranges: tuple[tuple[int, int], ...]
    owner_by_layer: tuple[int, ...]
    bytes_by_rank: tuple[int, ...]
    dense_bytes_by_rank: tuple[int, ...]
    expert_bytes_by_rank: tuple[int, ...]
    expert_payload_by_layer: tuple[int, ...]
    expert_payload_by_expert: tuple[tuple[int, ...], ...]
    expert_aux_by_layer: tuple[int, ...]

    @property
    def tp_size(self) -> int:
        return len(self.ranges)


@dataclass
class _PackedPrefillWorkspace:
    """Bounded rank-local scratch for exact row-batched packed MoE."""

    capacity: int
    top_k: int
    inputs: torch.Tensor
    route_ids: torch.Tensor
    route_weights: torch.Tensor
    hidden: torch.Tensor
    output: torch.Tensor
    result: torch.Tensor


@dataclass
class _GroupedPrefillWorkspace:
    token_ids: torch.Tensor
    weights: torch.Tensor
    group_experts: torch.Tensor
    group_offsets: torch.Tensor
    hidden: torch.Tensor
    result: torch.Tensor


def _contiguous_minimax(
    layer_bytes: list[int],
    ranks: int,
    first_extra: int,
    last_extra: int,
) -> list[tuple[int, int]]:
    """Exact contiguous minimax partition with endpoint-only tensors."""
    count = len(layer_bytes)
    if not 1 <= ranks <= count:
        raise ValueError(f"cannot split {count} layers over {ranks} ranks")
    prefix = [0]
    for value in layer_bytes:
        prefix.append(prefix[-1] + int(value))
    infinity = 1 << 100
    cost = [[infinity] * (count + 1) for _ in range(ranks + 1)]
    previous = [[-1] * (count + 1) for _ in range(ranks + 1)]
    cost[0][0] = 0
    for rank in range(1, ranks + 1):
        for end in range(rank, count + 1):
            endpoint = first_extra if rank == 1 else 0
            if rank == ranks and end == count:
                endpoint += last_extra
            for start in range(rank - 1, end):
                if cost[rank - 1][start] == infinity:
                    continue
                current = prefix[end] - prefix[start] + endpoint
                candidate = max(cost[rank - 1][start], current)
                if candidate < cost[rank][end]:
                    cost[rank][end] = candidate
                    previous[rank][end] = start
    ranges: list[tuple[int, int]] = []
    end = count
    for rank in range(ranks, 0, -1):
        start = previous[rank][end]
        if start < 0:
            raise RuntimeError("failed to construct Kimi layer partition")
        ranges.append((start, end))
        end = start
    ranges.reverse()
    return ranges


def build_routed_vq_layer_plan(store, tp_size: int) -> RoutedVQLayoutPlan:
    """Balance packed expert and dense payload while preserving layer order."""
    from .extreme import _expert_audit_payload_bytes

    n_layers = int(store.cfg["n_layers"])
    dense_by_layer = [0] * n_layers
    first_extra = 0
    last_extra = 0
    marker = ".model.layers."
    for name in store.dense_names():
        if not name.startswith(f"{_ROOT}."):
            continue
        resident_size = getattr(
            store, "dense_resident_nbytes", store.dense_nbytes
        )
        size = resident_size(name)
        if marker in name:
            layer = int(name.split(marker, 1)[1].split(".", 1)[0])
            dense_by_layer[layer] += size
        elif ".model.embed_tokens." in name:
            first_extra += size
        else:
            last_extra += size

    expert_file_by_layer = [0] * n_layers
    expert_payload_by_layer = [0] * n_layers
    expert_payload_by_expert: list[tuple[int, ...]] = [
        tuple() for _ in range(n_layers)
    ]
    expert_aux_by_layer = [0] * n_layers
    for layer, filename in store.man.expert_files.items():
        if not 0 <= int(layer) < n_layers:
            continue
        expert_file_by_layer[layer] = os.path.getsize(
            os.path.join(store.root, filename)
        )
        audit_name = store.man.expert_audit_files.get(layer)
        if audit_name is None:
            raise ValueError(
                f"packed residency requires expert audit for layer {layer}"
            )
        with open(
            os.path.join(store.root, audit_name),
            "r",
            encoding="utf-8",
        ) as handle:
            audit = json.load(handle)
        experts = audit.get("experts", {})
        maximum = max(
            (
                int(str(expert_id).lstrip("e"))
                for expert_id in experts
            ),
            default=-1,
        )
        n_experts = int(store.cfg.get("n_experts", maximum + 1))
        payloads = [0] * n_experts
        for expert_id, item in experts.items():
            index = int(str(expert_id).lstrip("e"))
            payloads[index] = _expert_audit_payload_bytes(item)
        complete_payload_bytes = sum(payloads)
        allowlist = getattr(store, "route_allowlist", None)
        if allowlist is not None:
            allowed = allowlist.get(int(layer), set())
            payloads = [
                payload if expert_id in allowed else 0
                for expert_id, payload in enumerate(payloads)
            ]
        expert_payload_by_expert[layer] = tuple(payloads)
        expert_payload_by_layer[layer] = sum(payloads)
        if allowlist is not None and complete_payload_bytes > 0:
            selected_file_bytes = (
                expert_file_by_layer[layer]
                * expert_payload_by_layer[layer]
                + complete_payload_bytes
                - 1
            ) // complete_payload_bytes
        else:
            selected_file_bytes = expert_file_by_layer[layer]
        expert_aux_by_layer[layer] = max(
            0,
            selected_file_bytes - expert_payload_by_layer[layer],
        )
        # A strict route profile changes runtime residency, not the physical
        # archive.  Budget only selected expert payloads plus the layer's
        # shared codebooks/metadata; charging the complete shard prevents a
        # valid profile from entering the full-GPU path.
        expert_file_by_layer[layer] = selected_file_bytes

    layer_bytes = [
        dense + expert
        for dense, expert in zip(dense_by_layer, expert_file_by_layer)
    ]
    ranges = _contiguous_minimax(
        layer_bytes,
        int(tp_size),
        first_extra,
        last_extra,
    )
    owner = [0] * n_layers
    bytes_by_rank = []
    dense_bytes_by_rank = []
    expert_bytes_by_rank = []
    for rank, (start, end) in enumerate(ranges):
        for layer in range(start, end):
            owner[layer] = rank
        dense = sum(dense_by_layer[start:end])
        expert = sum(expert_file_by_layer[start:end])
        total = dense + expert
        if rank == 0:
            dense += first_extra
            total += first_extra
        if rank == len(ranges) - 1:
            dense += last_extra
            total += last_extra
        bytes_by_rank.append(total)
        dense_bytes_by_rank.append(dense)
        expert_bytes_by_rank.append(expert)
    return RoutedVQLayoutPlan(
        ranges=tuple(ranges),
        owner_by_layer=tuple(owner),
        bytes_by_rank=tuple(bytes_by_rank),
        dense_bytes_by_rank=tuple(dense_bytes_by_rank),
        expert_bytes_by_rank=tuple(expert_bytes_by_rank),
        expert_payload_by_layer=tuple(expert_payload_by_layer),
        expert_payload_by_expert=tuple(expert_payload_by_expert),
        expert_aux_by_layer=tuple(expert_aux_by_layer),
    )


def build_primary_dense_packed_plan(store, tp_size: int) -> RoutedVQLayoutPlan:
    """Build a tensor-sharded expert plan for a primary-device dense graph.

    This is the transition plan used by architectures whose Attention/Dense
    graph is not yet represented as ``TPHidden``.  Every routed expert is
    still column/row sharded across all ranks and no rank stores a complete
    expert.  Only the comparatively small non-expert graph remains on rank 0.
    """
    base = build_routed_vq_layer_plan(store, int(tp_size))
    n_layers = int(store.cfg["n_layers"])
    dense_total = sum(
        int(
            getattr(
                store,
                "dense_resident_nbytes",
                store.dense_nbytes,
            )(name)
        )
        for name in store.dense_names()
    )
    ranges = ((0, n_layers),) + tuple(
        (n_layers, n_layers) for _ in range(int(tp_size) - 1)
    )
    dense_by_rank = (dense_total,) + (0,) * (int(tp_size) - 1)
    return RoutedVQLayoutPlan(
        ranges=ranges,
        owner_by_layer=(0,) * n_layers,
        bytes_by_rank=tuple(
            dense_by_rank[rank] + base.expert_bytes_by_rank[rank]
            for rank in range(int(tp_size))
        ),
        dense_bytes_by_rank=dense_by_rank,
        expert_bytes_by_rank=base.expert_bytes_by_rank,
        expert_payload_by_layer=base.expert_payload_by_layer,
        expert_payload_by_expert=base.expert_payload_by_expert,
        expert_aux_by_layer=base.expert_aux_by_layer,
    )


def _packed_startup_required_bytes(
    plan: RoutedVQLayoutPlan,
    rank: int,
    rank_payload_bytes: int,
    *,
    host_mapped: bool,
    parallelism: str,
    dense_resident: bool,
) -> int:
    """Return only the VRAM which still has to be allocated at this stage."""
    if host_mapped:
        required = (
            plan.dense_bytes_by_rank[rank]
            + sum(plan.expert_aux_by_layer)
        )
    elif parallelism == "pipeline":
        required = plan.bytes_by_rank[rank]
    else:
        required = (
            plan.dense_bytes_by_rank[rank]
            + int(rank_payload_bytes)
            + sum(plan.expert_aux_by_layer)
        )
    if dense_resident:
        required = max(0, required - plan.dense_bytes_by_rank[rank])
    return required + 512 * 2**20


def _packed_grouped_prefill_supported(manifest, layer: int) -> bool:
    """Validate grouped Prefill from per-expert projection layouts.

    A heterogeneous layer can use more than three distinct layout names in
    total even though every expert still has exactly Gate/Up/Down.  Testing
    the deduplicated layer-wide capability length therefore rejected valid
    resident models after they had loaded successfully.  The compiled grouped
    kernel consumes one 15-row metadata directory and supports mixed packed
    widths on CUDA and HIP, so the correct contract is three projections for
    every concrete expert capability.
    """
    if not manifest.projection_vq:
        return False
    capabilities = manifest.projection_operator_capabilities(int(layer))
    return bool(capabilities) and all(
        len(capability.get("packed_formats", ())) == 3
        and len(capability.get("code_dims", ())) == 3
        and len(capability.get("codebook_sizes", ())) == 3
        for capability in capabilities
    )


def _select_packed_grouped_prefill(
    manifest,
    layer: int,
    *,
    metadata_rows: int,
    dequant_prefill: bool,
    hip_runtime: bool,
) -> bool:
    """Choose the packed grouped executor without changing CUDA fast paths."""
    capable = bool(
        int(metadata_rows) == 10
        or (
            int(metadata_rows) == 15
            and _packed_grouped_prefill_supported(manifest, layer)
        )
    )
    return bool(capable and (bool(hip_runtime) or not dequant_prefill))


class ResidentRoutedVQPool:
    """配置驱动的全显存 packed 专家执行器。"""

    full_resident = True

    def __init__(
        self,
        store,
        devices: tuple[torch.device, ...],
        plan: RoutedVQLayoutPlan,
        *,
        parallelism: str = "pipeline",
        tensor_group_size: int | None = None,
    ):
        self.store = store
        self.devices = devices
        self.plan = plan
        if parallelism not in {
            "pipeline",
            "expert",
            "tensor",
            "hybrid",
        }:
            raise ValueError(
                f"unsupported packed parallelism {parallelism!r}"
            )
        self.parallelism = parallelism
        self.host_mapped = bool(
            len(devices) == 1
            and os.environ.get("CCCP_PACKED_HOST_MAPPED", "0") == "1"
        )
        self._mapped_stage_enabled = bool(
            self.host_mapped
            and os.environ.get("CCCP_MAPPED_STAGE", "0") == "1"
        )
        self._mapped_elastic_enabled = bool(
            self.host_mapped
            and os.environ.get("CCCP_MAPPED_ELASTIC_CACHE", "0") == "1"
        )
        self.hidden_mode = (
            parallelism == "tensor"
            and os.environ.get("CCCP_TP_HIDDEN", "0") != "0"
            and os.environ.get("CCCP_TP_NO_OWNER", "1") != "0"
        )
        if parallelism == "tensor":
            tensor_group_size = len(devices)
        elif parallelism == "hybrid":
            tensor_group_size = (
                2 if tensor_group_size is None else int(tensor_group_size)
            )
        else:
            tensor_group_size = 1
        if (
            tensor_group_size <= 0
            or len(devices) % tensor_group_size
            or (
                parallelism == "hybrid"
                and tensor_group_size == len(devices)
            )
        ):
            raise ValueError(
                "packed MoE tensor group must be a proper divisor "
                "of the device count"
            )
        self.tensor_group_size = tensor_group_size
        self.expert_group_count = len(devices) // tensor_group_size
        self.budget = sum(plan.expert_bytes_by_rank)
        self.hits = 0
        self.miss = 0
        self.prefill_batch_rows = 0
        self.prefill_batch_submissions = 0
        self.prefill_batch_max = 0
        self.prefill_executor = "cuda.grouped-packed"
        self.decode_fused_submissions = 0
        self.decode_graph_submissions = 0
        self.decode_reference_submissions = 0
        # Every rank owns a complete metadata table for its resident expert
        # shard.  Row-batched prefill therefore follows the same public
        # packed operator + Row-TP reduction as decode; model adapters query
        # this explicit capability instead of inferring it from a method name.
        self.prefill_rows_supported = True
        self.active = False
        self._allocated = False
        self._payload_loaded = False
        self._payload_loaded_count = 0
        self._payload_load_seconds = 0.0
        self._arenas: list[torch.Tensor] = []
        # Full-resident experts are split by layer. Windows ROCm rejects very
        # large single allocations with hipErrorInvalidValue even when total
        # free VRAM is sufficient; layer slabs preserve stable pointers while
        # avoiding a 20--70 GiB monolithic allocation.
        self._resident_layer_arenas: dict[
            tuple[int, int], torch.Tensor
        ] = {}
        self._mapped_experts: list[tuple] = []
        self._mapped_payloads: list[torch.Tensor] = []
        self._mapped_device_pointers: dict[int, int] = {}
        self._mapped_host_bytes = 0
        self._mapped_entries: dict[tuple[int, int], tuple] = {}
        self._mapped_host_metadata: dict[int, torch.Tensor] = {}
        self._mapped_uva_metadata: dict[int, tuple[torch.Tensor, ...]] = {}
        self._mapped_projection_max: dict[int, int] = {}
        self._mapped_stage_projection_strides: tuple[int, int, int] = ()
        self._mapped_stage_workspace: torch.Tensor | None = None
        self._mapped_stage_metadata: torch.Tensor | None = None
        self._mapped_stage_route_ids: torch.Tensor | None = None
        self._mapped_size_layout: dict[int, tuple[int, int]] = {}
        self._mapped_slot_lru: dict[
            int, OrderedDict[tuple[int, int], int]
        ] = {}
        self._mapped_slot_owner: dict[
            int, list[tuple[int, int] | None]
        ] = {}
        self._mapped_global_lru: OrderedDict[
            tuple[int, int], tuple[int, int]
        ] = OrderedDict()
        self._mapped_free_ranges: list[tuple[int, int]] = []
        self._mapped_slots_per_layer = 0
        self._mapped_total_slots = 0
        self._mapped_peak_slots = 0
        self.mapped_cache_evictions = 0
        self.mapped_cache_hits = 0
        self.mapped_cache_misses = 0
        self.mapped_cache_hits_by_class: dict[int, int] = {}
        self.mapped_cache_misses_by_class: dict[int, int] = {}
        self.mapped_cache_uploaded_bytes = 0
        self.mapped_cache_refresh_seconds = 0.0
        self._metadata: dict[int, tuple[torch.Tensor, ...]] = {}
        self._codebooks: dict[
            tuple[int, str, str, int], torch.Tensor
        ] = {}
        self._workspaces: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        # Prefill reuses one bounded packed workspace per rank.  Expert
        # indices remain packed in the resident arena; this is only the
        # activation/output scratch required by the fused CUDA kernels.
        self._prefill_workspaces: dict[
            int,
            _PackedPrefillWorkspace,
        ] = {}
        self._prefill_grouped_workspaces: dict[
            int,
            _GroupedPrefillWorkspace,
        ] = {}
        # TP grouped prefill 的本地专家掩码缓存：(layer, rank) -> [E] bool。
        self._grouped_local_masks: dict[
            tuple[int, int],
            torch.Tensor,
        ] = {}
        # dequant+grouped GEMM prefill 的稠密权重 scratch（每 rank 一份，
        # 跨层/跨微批复用）与能力缓存。
        self._dequant_buffers: dict[
            int,
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._dequant_supported_cache: dict[int, bool] = {}
        self._dequant_id_ranges: dict[int, torch.Tensor] = {}
        self._native8_prefill_enabled = False
        self._native8_grouped_backend: str | None = None
        self._native8_codebooks: dict[int, torch.Tensor] = {}
        self._native8_metadata: dict[
            int, tuple[torch.Tensor, ...]
        ] = {}
        self._native8_scales: dict[
            int, tuple[torch.Tensor, ...]
        ] = {}
        self._native8_workspaces: dict[int, dict[str, object]] = {}
        self._compact_decode_enabled = False
        self._compact_decode_codebooks: dict[int, torch.Tensor] = {}
        self._compact_decode_metadata: dict[
            int, tuple[torch.Tensor, ...]
        ] = {}
        self._compact_decode_scales: dict[
            int, tuple[torch.Tensor, ...]
        ] = {}
        self._compact_decode_workspaces: dict[int, dict[str, torch.Tensor]] = {}
        self._streams: list[torch.cuda.Stream] = []
        self._source_events: list[torch.cuda.Event] = []
        self._done_events: list[list[torch.cuda.Event]] = []
        self._output_events: list[list[torch.cuda.Event]] = []
        self._routed_inputs: list[torch.Tensor] = []
        self._routed_ids: list[torch.Tensor] = []
        self._routed_weights: list[torch.Tensor] = []
        self._source_inputs: list[torch.Tensor] = []
        self._source_ids: list[torch.Tensor] = []
        self._source_weights: list[torch.Tensor] = []
        self._return_buffers: list[torch.Tensor] = []
        self._reduce_buffers: list[torch.Tensor] = []
        self._zero_buffers: list[torch.Tensor] = []
        self._graphs: dict[int, list[torch.cuda.CUDAGraph]] = {}
        self._graph_batches: dict[int, object] = {}
        self._graph_rank_order: dict[int, tuple[int, ...]] = {}
        self._output_replicas: dict[int, list[torch.Tensor]] = {}
        self._route_graphs: dict[
            int, tuple[torch.cuda.CUDAGraph, ...]
        ] = {}
        self._bound_hidden_inputs: dict[int, tuple] = {}
        if self.parallelism == "pipeline":
            self._rank_payload_bytes = tuple(
                sum(
                    plan.expert_payload_by_layer[layer]
                    for layer in range(start, end)
                )
                for start, end in plan.ranges
            )
        elif self.parallelism == "expert":
            rank_bytes = [0] * len(devices)
            for layer, payloads in enumerate(
                plan.expert_payload_by_expert
            ):
                for expert_id, payload in enumerate(payloads):
                    rank_bytes[self.expert_owner(layer, expert_id)] += payload
            self._rank_payload_bytes = tuple(rank_bytes)
        elif self.parallelism == "hybrid":
            rank_bytes = [0] * len(devices)
            for layer, payloads in enumerate(
                plan.expert_payload_by_expert
            ):
                for expert_id, payload in enumerate(payloads):
                    if payload % self.tensor_group_size:
                        raise ValueError(
                            "packed expert payload is not group-TP divisible"
                        )
                    for rank in self.expert_ranks(layer, expert_id):
                        rank_bytes[rank] += (
                            payload // self.tensor_group_size
                        )
            self._rank_payload_bytes = tuple(rank_bytes)
        else:
            rank_bytes = [0] * len(devices)
            for payloads in plan.expert_payload_by_expert:
                for payload in payloads:
                    if payload % len(devices):
                        raise ValueError(
                            "packed expert payload is not TP divisible"
                        )
                    for rank in range(len(devices)):
                        rank_bytes[rank] += payload // len(devices)
            self._rank_payload_bytes = tuple(rank_bytes)

    def expert_owner(self, layer: int, expert_id: int) -> int:
        if self.parallelism == "pipeline":
            return self.plan.owner_by_layer[layer]
        if self.parallelism in {"tensor", "hybrid"}:
            raise RuntimeError("tensor-sharded experts have no single owner")
        return (int(layer) + int(expert_id)) % len(self.devices)

    def expert_ranks(
        self,
        layer: int,
        expert_id: int,
    ) -> range:
        """Return the contiguous TP group assigned to one routed expert."""
        if self.parallelism == "tensor":
            return range(len(self.devices))
        if self.parallelism != "hybrid":
            owner = self.expert_owner(layer, expert_id)
            return range(owner, owner + 1)
        group = (
            int(layer) + int(expert_id)
        ) % self.expert_group_count
        start = group * self.tensor_group_size
        return range(start, start + self.tensor_group_size)

    @property
    def gpu_storage_bytes(self) -> int:
        return (
            sum(tensor.nbytes for tensor in self._arenas)
            + sum(
                tensor.nbytes
                for tensor in self._resident_layer_arenas.values()
            )
            + sum(tensor.nbytes for tensor in self._codebooks.values())
            + sum(
                tensor.nbytes for tensor in self._native8_codebooks.values()
            )
            + sum(
                tensor.nbytes
                for tensor in self._compact_decode_codebooks.values()
            )
        )

    @property
    def gpu_arena_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self._arenas) + sum(
            tensor.nbytes for tensor in self._resident_layer_arenas.values()
        )

    def _resident_layer_payload_bytes(self, rank: int, layer: int) -> int:
        """Return one rank's packed payload for a single logical layer."""

        payloads = self.plan.expert_payload_by_expert[int(layer)]
        if self.parallelism == "pipeline":
            return (
                sum(payloads)
                if self.plan.owner_by_layer[int(layer)] == int(rank)
                else 0
            )
        if self.parallelism == "expert":
            return sum(
                payload
                for expert_id, payload in enumerate(payloads)
                if self.expert_owner(int(layer), expert_id) == int(rank)
            )
        if self.parallelism == "tensor":
            return sum(payload // len(self.devices) for payload in payloads)
        return sum(
            payload // self.tensor_group_size
            for expert_id, payload in enumerate(payloads)
            if int(rank) in self.expert_ranks(int(layer), expert_id)
        )

    @property
    def host_expert_bytes(self) -> int:
        return self._mapped_host_bytes

    def _register_mapped_payload(self, payload: torch.Tensor) -> int:
        """Page-lock one compact CPU blob and expose its stable UVA address."""

        if (
            payload.device.type != "cpu"
            or payload.dtype != torch.uint8
            or payload.ndim != 1
            or not payload.is_contiguous()
        ):
            raise ValueError("mapped packed payload must be contiguous CPU uint8")
        pointer = int(payload.data_ptr())
        cached = self._mapped_device_pointers.get(pointer)
        if cached is not None:
            return cached
        cudart = torch.cuda.cudart()
        error = cudart.cudaHostRegister(
            pointer,
            int(payload.nbytes),
            _CUDA_HOST_REGISTER_MAPPED,
        )
        error_code = getattr(error, "value", None)
        if error_code is None:
            error_code = int(error)
        if int(error_code) != 0:
            try:
                message = cudart.cudaGetErrorString(error)
            except (AttributeError, RuntimeError, TypeError):
                message = error
            try:
                cudart.cudaGetLastError()
            except (AttributeError, RuntimeError, TypeError):
                pass
            raise RuntimeError(
                "cudaHostRegisterMapped failed for packed expert "
                f"({payload.nbytes} bytes): {message}"
            )
        device_pointer = _cuda_host_device_pointer(pointer)
        self._mapped_device_pointers[pointer] = device_pointer
        self._mapped_payloads.append(payload)
        self._mapped_host_bytes += int(payload.nbytes)
        return device_pointer

    @staticmethod
    def _mapped_elastic_size(size: int) -> int:
        """Round one variable payload to a compact allocator granule."""

        alignment = 64 * 1024
        return (int(size) + alignment - 1) // alignment * alignment

    def _mapped_elastic_take(self, size: int) -> int | None:
        """Take the smallest fitting range from the shared mapped arena."""

        requested = self._mapped_elastic_size(size)
        best_index = None
        best_size = None
        for index, (_offset, available) in enumerate(
            self._mapped_free_ranges
        ):
            if available < requested:
                continue
            if best_size is None or available < best_size:
                best_index = index
                best_size = available
        if best_index is None:
            return None
        offset, available = self._mapped_free_ranges.pop(best_index)
        remainder = available - requested
        if remainder:
            self._mapped_free_ranges.insert(
                best_index, (offset + requested, remainder)
            )
        return int(offset)

    def _mapped_elastic_release(self, offset: int, size: int) -> None:
        """Return and coalesce one range in the shared mapped arena."""

        released = self._mapped_elastic_size(size)
        ranges = self._mapped_free_ranges
        ranges.append((int(offset), released))
        ranges.sort(key=lambda item: item[0])
        coalesced: list[tuple[int, int]] = []
        for current_offset, current_size in ranges:
            if (
                coalesced
                and coalesced[-1][0] + coalesced[-1][1]
                == current_offset
            ):
                previous_offset, previous_size = coalesced[-1]
                coalesced[-1] = (
                    previous_offset,
                    previous_size + current_size,
                )
            else:
                coalesced.append((current_offset, current_size))
        self._mapped_free_ranges = coalesced

    def _mapped_restore_host_pointer(
        self,
        key: tuple[int, int],
    ) -> int:
        """Restore one evicted expert's metadata to its exact UVA source."""

        layer, expert_id = key
        payload, projection_offsets = self._mapped_entries[key]
        source = self._mapped_device_pointers[int(payload.data_ptr())]
        metadata = self._mapped_host_metadata[layer]
        for base, projection_offset in projection_offsets:
            metadata[base, expert_id] = source + projection_offset
        return int(layer)

    def refresh_mapped_cache(self) -> None:
        """Promote the previous token's routed experts into fixed VRAM slots.

        Every non-resident expert keeps an exact mapped-host pointer, so a
        prediction miss changes performance only.  The captured TokenGraph
        and its metadata address remain stable while pointer values switch
        between the VRAM mirror and the UVA cold path.
        """

        if (
            not self.host_mapped
            or not self.active
            or self._mapped_slots_per_layer <= 0
            or not self._bound_hidden_inputs
        ):
            return
        started = time.perf_counter()
        device = self.devices[0]
        arena = self._arenas[0]
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            stream_handle = int(stream.cuda_stream)
            requests_by_layer = {}
            protected_by_size: dict[int, set[tuple[int, int]]] = {}
            protected_keys: set[tuple[int, int]] = set()
            for layer in sorted(self._bound_hidden_inputs):
                route_ids = self._bound_hidden_inputs[layer][2][0]
                requested = tuple(dict.fromkeys(
                    int(value)
                    for value in route_ids.detach().reshape(-1).cpu().tolist()
                ))
                if not requested:
                    continue
                requests_by_layer[layer] = requested
                for expert_id in requested:
                    payload, _offsets = self._mapped_entries[
                        (layer, expert_id)
                    ]
                    stride = int(payload.nbytes)
                    protected_by_size.setdefault(
                        stride, set()
                    ).add((layer, expert_id))
                    protected_keys.add((layer, expert_id))
            changed_layers = set()
            for layer, requested in requests_by_layer.items():
                for expert_id in requested:
                    key = (layer, expert_id)
                    payload, projection_offsets = self._mapped_entries[key]
                    size = int(payload.nbytes)
                    stride = size
                    if self._mapped_elastic_enabled:
                        cached = self._mapped_global_lru.get(key)
                        if cached is not None:
                            self._mapped_global_lru.move_to_end(key)
                            self.mapped_cache_hits += 1
                            self.mapped_cache_hits_by_class[stride] = (
                                self.mapped_cache_hits_by_class.get(
                                    stride, 0
                                )
                                + 1
                            )
                            continue
                        self.mapped_cache_misses += 1
                        self.mapped_cache_misses_by_class[stride] = (
                            self.mapped_cache_misses_by_class.get(
                                stride, 0
                            )
                            + 1
                        )
                        reserved = self._mapped_elastic_size(size)
                        offset = self._mapped_elastic_take(reserved)
                        while offset is None:
                            victim = next(
                                (
                                    candidate
                                    for candidate in self._mapped_global_lru
                                    if candidate not in protected_keys
                                ),
                                None,
                            )
                            if victim is None:
                                break
                            victim_offset, victim_size = (
                                self._mapped_global_lru.pop(victim)
                            )
                            changed_layers.add(
                                self._mapped_restore_host_pointer(victim)
                            )
                            self._mapped_elastic_release(
                                victim_offset, victim_size
                            )
                            self.mapped_cache_evictions += 1
                            offset = self._mapped_elastic_take(reserved)
                        if offset is None:
                            # Every resident range is needed by this token;
                            # preserve exact UVA execution for this expert.
                            continue
                        destination = int(arena.data_ptr()) + offset
                        _cuda_memcpy_h2d_async(
                            destination,
                            int(payload.data_ptr()),
                            size,
                            stream_handle,
                        )
                        host_metadata = self._mapped_host_metadata[layer]
                        for base, projection_offset in projection_offsets:
                            host_metadata[base, expert_id] = (
                                destination + projection_offset
                            )
                        self._mapped_global_lru[key] = (offset, reserved)
                        self._mapped_total_slots = len(
                            self._mapped_global_lru
                        )
                        self._mapped_peak_slots = max(
                            self._mapped_peak_slots,
                            self._mapped_total_slots,
                        )
                        self.mapped_cache_uploaded_bytes += size
                        changed_layers.add(layer)
                        continue
                    lru = self._mapped_slot_lru[stride]
                    owners = self._mapped_slot_owner[stride]
                    slot = lru.get(key)
                    if slot is not None:
                        lru.move_to_end(key)
                        self.mapped_cache_hits += 1
                        self.mapped_cache_hits_by_class[stride] = (
                            self.mapped_cache_hits_by_class.get(stride, 0) + 1
                        )
                        continue
                    self.mapped_cache_misses += 1
                    self.mapped_cache_misses_by_class[stride] = (
                        self.mapped_cache_misses_by_class.get(stride, 0) + 1
                    )
                    try:
                        slot = owners.index(None)
                    except ValueError:
                        victim = next(
                            (
                                candidate
                                for candidate in lru
                                if candidate
                                not in protected_by_size.get(stride, set())
                            ),
                            None,
                        )
                        if victim is None:
                            # This is possible only when a user configures
                            # fewer slots than Top-K. The exact UVA path stays
                            # valid, so skip promotion instead of corrupting a
                            # route that is needed by the same token.
                            continue
                        slot = lru.pop(victim)
                        owners[slot] = None
                        changed_layers.add(
                            self._mapped_restore_host_pointer(victim)
                        )
                    size_offset, _slot_count = self._mapped_size_layout[stride]
                    destination = (
                        int(arena.data_ptr())
                        + size_offset
                        + slot * stride
                    )
                    _cuda_memcpy_h2d_async(
                        destination,
                        int(payload.data_ptr()),
                        size,
                        stream_handle,
                    )
                    host_metadata = self._mapped_host_metadata[layer]
                    for base, projection_offset in projection_offsets:
                        host_metadata[base, expert_id] = (
                            destination + projection_offset
                        )
                    owners[slot] = key
                    lru[key] = slot
                    self.mapped_cache_uploaded_bytes += size
                    changed_layers.add(layer)
            for layer in sorted(changed_layers):
                if layer in self._metadata:
                    self._metadata[layer][0].copy_(
                        self._mapped_host_metadata[layer],
                        non_blocking=True,
                    )
        self.mapped_cache_refresh_seconds += time.perf_counter() - started

    def output_hidden(self, layer: int):
        """Expose fixed all-rank packed outputs for parent-graph composition."""
        from .ops import TPHidden

        outputs = self._output_replicas.get(int(layer))
        if outputs is None:
            raise RuntimeError(
                f"packed MoE layer {layer} outputs are unavailable"
            )
        return TPHidden(
            self.devices,
            tuple(outputs),
            tuple(
                self._output_events[rank][int(layer)]
                for rank in range(len(self.devices))
            ),
        )

    def fixed_layer_plan(self, layer: int):
        """Expose immutable packed-TP scheduling metadata to a common plan."""
        layer = int(layer)
        graph_batch = self._graph_batches.get(layer)
        if graph_batch is None:
            raise RuntimeError(
                f"packed MoE layer {layer} graph is unavailable"
            )
        return (
            graph_batch,
            tuple(
                self._workspaces[rank][2]
                for rank in range(len(self.devices))
            ),
            self.output_hidden(layer),
        )

    def fixed_layer_child_graphs(self, layer: int):
        """Return retained rank-local child graphs for parent composition.

        The result is capability metadata, not a model hook: each rank gets
        its optional route graph followed by the packed expert graph.
        """
        layer = int(layer)
        rank_order = self._graph_rank_order.get(layer)
        graphs = self._graphs.get(layer)
        if rank_order is None or graphs is None:
            raise RuntimeError(
                f"packed MoE layer {layer} retained graphs are unavailable"
            )
        expert_by_rank = {
            rank: graphs[ordered]
            for ordered, rank in enumerate(rank_order)
        }
        routes = self._route_graphs.get(layer)
        return tuple(
            (
                (routes[rank], expert_by_rank[rank])
                if routes is not None
                else (expert_by_rank[rank],)
            )
            for rank in range(len(self.devices))
        )

    def prefetch(self, _keys) -> None:
        return

    def bind_hidden_inputs(
        self,
        layer: int,
        value,
        weights: tuple[torch.Tensor, ...],
        indices: tuple[torch.Tensor, ...],
    ) -> None:
        """Bind fixed all-rank Router/Down outputs to packed expert graphs."""
        if (
            not self.hidden_mode
            or tuple(value.devices) != self.devices
            or value.ready_events is None
            or len(weights) != len(self.devices)
            or len(indices) != len(self.devices)
        ):
            raise ValueError("packed MoE fixed input layout mismatch")
        self._bound_hidden_inputs[int(layer)] = (
            tuple(value.replicas),
            tuple(item.reshape(-1) for item in weights),
            tuple(item.reshape(-1) for item in indices),
        )

    def allocate(self, *, dense_resident: bool = False) -> None:
        """Reserve packed arenas before fragmented dense allocations begin."""
        if self._allocated:
            return
        reserve = float(os.environ.get(
            "CCCP_VRAM_HEADROOM_GB",
            os.environ.get("CCCP_VRAM_RESERVE_GB", "1"),
        ))
        reserve_bytes = int(reserve * 2**30)
        mapped_arena_bytes = 0
        details = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                free, _total = torch.cuda.mem_get_info(device)
            # ``bytes_by_rank`` includes dense tensors and FP32 codebooks.  Add
            # a conservative 512 MiB for KDA state, router FP32 promotion and
            # decode workspaces on each rank.
            # DSV4 builds its fixed Dense/Attention graph before the packed
            # expert pool.  ``mem_get_info`` already excludes those resident
            # allocations, so counting Dense again rejects valid mapped RAM
            # offload (for example 8.7 GiB Dense on a 20 GiB card).  Kimi TP
            # calls allocate before Dense and keeps the default False.
            required = _packed_startup_required_bytes(
                self.plan,
                rank,
                self._rank_payload_bytes[rank],
                host_mapped=self.host_mapped,
                parallelism=self.parallelism,
                dense_resident=dense_resident,
            )
            available = max(0, free - reserve_bytes)
            details.append(
                f"cuda:{device.index} 需{required / 2**30:.2f}GiB/"
                f"可用{available / 2**30:.2f}GiB"
            )
            if required > available:
                placement = (
                    "packed RAM 映射热缓存"
                    if self.host_mapped
                    else "packed TP 全显存"
                )
                raise RuntimeError(
                    f"{placement}容量不足：" + "，".join(details)
                )
        if self.host_mapped:
            # The process fraction is the real 32 GiB hard cap; mem_get_info
            # alone reports the physical H20 capacity and would over-allocate.
            fraction = float(
                torch.cuda.get_per_process_memory_fraction(device)
            )
            allocator_cap = int(_total * fraction)
            allocator_used = int(torch.cuda.memory_reserved(device))
            configured_future_gib = float(
                os.environ.get("CCCP_MAPPED_FUTURE_FIXED_GB", "0")
            )
            future_fixed = (
                int(configured_future_gib * 2**30)
                if configured_future_gib > 0
                else (
                    sum(self.plan.expert_aux_by_layer)
                    + 512 * 2**20
                )
            )
            hot_budget = min(
                max(
                    0,
                    allocator_cap
                    - allocator_used
                    - reserve_bytes
                    - future_fixed,
                ),
                max(0, free - reserve_bytes - future_fixed),
            )
            configured_gib = float(
                os.environ.get("CCCP_MAPPED_CACHE_GB", "0")
            )
            if configured_gib > 0:
                hot_budget = min(
                    hot_budget,
                    int(configured_gib * 2**30),
                )
            if self._mapped_elastic_enabled:
                alignment = 64 * 1024
                mapped_arena_bytes = hot_budget // alignment * alignment
                self._mapped_free_ranges = [
                    (0, mapped_arena_bytes)
                ] if mapped_arena_bytes else []
                # This field is also the refresh enable flag. Elastic mode
                # has no per-layer quota, so one means "enabled" only.
                self._mapped_slots_per_layer = (
                    1 if mapped_arena_bytes else 0
                )
            else:
                size_counts: dict[int, int] = {}
                for layer in sorted(self.store.man.expert_files):
                    for size in self.plan.expert_payload_by_expert[
                        int(layer)
                    ]:
                        size = int(size)
                        size_counts[size] = size_counts.get(size, 0) + 1
                total_payload = sum(
                    size * count for size, count in size_counts.items()
                )
                slot_counts = {
                    size: min(
                        count,
                        int(hot_budget * count // max(1, total_payload)),
                    )
                    for size, count in size_counts.items()
                }
                used = sum(
                    size * slot_counts[size] for size in slot_counts
                )
                # Consume rounding slack with the smallest fitting class.
                while True:
                    candidate = next(
                        (
                            size
                            for size in sorted(size_counts)
                            if slot_counts[size] < size_counts[size]
                            and used + size <= hot_budget
                        ),
                        None,
                    )
                    if candidate is None:
                        break
                    slot_counts[candidate] += 1
                    used += candidate
                offset = 0
                for size in sorted(slot_counts):
                    count = slot_counts[size]
                    self._mapped_size_layout[size] = (offset, count)
                    offset += size * count
                    self._mapped_slot_lru[size] = OrderedDict()
                    self._mapped_slot_owner[size] = [None] * count
                mapped_arena_bytes = offset
                self._mapped_total_slots = sum(slot_counts.values())
                self._mapped_slots_per_layer = (
                    self._mapped_total_slots
                    // max(1, len(self.store.man.expert_files))
                )
        try:
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    if self.host_mapped:
                        self._arenas.append(torch.empty(
                            mapped_arena_bytes,
                            dtype=torch.uint8,
                            device=device,
                        ))
                    else:
                        for layer in sorted(self.store.man.expert_files):
                            layer_bytes = self._resident_layer_payload_bytes(
                                rank, int(layer)
                            )
                            if layer_bytes <= 0:
                                continue
                            self._resident_layer_arenas[(rank, int(layer))] = (
                                torch.empty(
                                    layer_bytes,
                                    dtype=torch.uint8,
                                    device=device,
                                )
                            )
        except Exception:
            self._arenas.clear()
            self._resident_layer_arenas.clear()
            gc.collect()
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            raise
        print(
            "[cccp-packed] packed 专家 arena 已分配："
            + "，".join(
                f"cuda:{device.index}="
                f"{(mapped_arena_bytes if self.host_mapped else size) / 2**30:.2f}GiB"
                for device, size in zip(
                    self.devices,
                    self._rank_payload_bytes,
                )
            ),
            flush=True,
        )
        if self.host_mapped:
            print(
                "[cccp-packed] packed 专家固定 UVA 映射已启用："
                "热镜像与 UVA 冷路径共享固定地址，索引主体保持紧凑",
                flush=True,
            )
        if self.host_mapped:
            layout = (
                "全局弹性变长 arena"
                if self._mapped_elastic_enabled
                else (
                    f"按 payload 尺寸分级固定槽="
                    f"{self._mapped_total_slots} 个"
                    f"（折合每层 {self._mapped_slots_per_layer}）"
                )
            )
            print(
                "[cccp-packed] mapped 热镜像规划："
                f"{mapped_arena_bytes / 2**30:.2f}GiB，"
                f"{layout}；"
                "未命中仍由精确 UVA 冷路径执行",
                flush=True,
            )
        self._allocated = True

    def _device_codebook(
        self,
        rank: int,
        layer: int,
        tier: str,
        variant: str,
        projection: int,
        cb: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        key = (rank, tier, variant, projection)
        cached = self._codebooks.get(key)
        if cached is None:
            cached = cb.to(
                device=device,
                dtype=torch.bfloat16,
            ).contiguous()
            self._codebooks[key] = cached
        return cached

    @staticmethod
    def _tensor_shard_raw(
        weight,
        *,
        projection: int,
        rank: int,
        ranks: int,
        intermediate: int,
    ) -> tuple[torch.Tensor, int]:
        """Slice packed rows/columns without expanding their bit width."""
        row_bits = weight.blocks * weight.bits
        if row_bits % 8:
            raise ValueError("packed expert row is not byte aligned")
        row_bytes = row_bits // 8
        rows = weight.raw.view(weight.rows, row_bytes)
        local_intermediate = intermediate // ranks
        if projection == 0:
            start = rank * local_intermediate
            end = start + local_intermediate
            shard = torch.cat(
                (
                    rows[start:end],
                    rows[
                        intermediate + start:
                        intermediate + end
                    ],
                ),
                dim=0,
            ).contiguous()
            return shard.reshape(-1), weight.blocks
        if weight.blocks % ranks:
            raise ValueError("packed Down blocks are not TP divisible")
        local_blocks = weight.blocks // ranks
        start_bits = rank * local_blocks * weight.bits
        shard_bits = local_blocks * weight.bits
        if start_bits % 8 or shard_bits % 8:
            raise ValueError(
                "packed Down shard boundary is not byte aligned"
            )
        start_byte = start_bits // 8
        end_byte = start_byte + shard_bits // 8
        return (
            rows[:, start_byte:end_byte].contiguous().reshape(-1),
            local_blocks,
        )

    @staticmethod
    def _tensor_shard_projection_raw(
        weight,
        *,
        projection: int,
        rank: int,
        ranks: int,
        intermediate: int,
    ) -> tuple[torch.Tensor, int]:
        """按三投影数学维切分紧凑索引，不展开 p14。

        gate/up 是 Column-TP，沿输出行切；down 是 Row-TP，沿输入块切。
        """
        row_bits = weight.blocks * weight.bits
        if row_bits % 8:
            raise ValueError("packed expert row is not byte aligned")
        row_bytes = row_bits // 8
        rows = weight.raw.view(weight.rows, row_bytes)
        local_intermediate = intermediate // ranks
        if projection in (0, 1):
            start = rank * local_intermediate
            end = start + local_intermediate
            return (
                rows[start:end].contiguous().reshape(-1),
                weight.blocks,
            )
        if projection != 2:
            raise ValueError(f"invalid expert projection {projection}")
        if weight.blocks % ranks:
            raise ValueError("packed Down blocks are not TP divisible")
        local_blocks = weight.blocks // ranks
        start_bits = rank * local_blocks * weight.bits
        shard_bits = local_blocks * weight.bits
        if start_bits % 8 or shard_bits % 8:
            raise ValueError(
                "packed Down shard boundary is not byte aligned"
            )
        start_byte = start_bits // 8
        end_byte = start_byte + shard_bits // 8
        return (
            rows[:, start_byte:end_byte].contiguous().reshape(-1),
            local_blocks,
        )

    def preload(
        self,
        *,
        dense_resident: bool = False,
        capture_graphs: bool = True,
    ) -> None:
        """Read each packed expert once and write it directly into its arena."""
        if self._payload_loaded:
            if capture_graphs and not self.active:
                if os.environ.get("CCCP_TP_GRAPH", "1") != "0":
                    self._prepare_expert_graphs()
                self.active = True
                print(
                    "[cccp-packed] packed 专家固定数据先于 Dense/Attention "
                    "Graph 完成；运行图已在最终固定地址上捕获",
                    flush=True,
                )
            return
        self.allocate(dense_resident=dense_resident)
        started = time.time()
        n_experts = int(self.store.cfg["n_experts"])
        top_k = int(self.store.cfg["top_k"])
        intermediate = int(self.store.cfg["moe_inter"])
        routed_hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        offsets = [0] * len(self.devices)
        layer_offsets: dict[tuple[int, int], int] = {
            key: 0 for key in self._resident_layer_arenas
        }
        loaded = 0
        for layer in sorted(self.store.man.expert_files):
            projection_vq = bool(self.store.man.projection_vq)
            metadata_by_rank = [
                torch.zeros(
                    15 if projection_vq else 10,
                    n_experts,
                    dtype=torch.long,
                )
                for _ in self.devices
            ]
            for expert_id in range(n_experts):
                if (
                    self.store.route_allowlist is not None
                    and expert_id not in self.store.route_allowlist.get(
                        int(layer), set()
                    )
                ):
                    continue
                tier = self.store.expert_kind(layer, expert_id)
                if tier == "drop":
                    continue
                base_tier = tier.rstrip("z")
                if projection_vq:
                    codebook_variants = (
                        self.store.projection_codebook_variants(
                            layer,
                            expert_id,
                        )
                    )
                    weights = self.store.load_expert_packed(
                        layer,
                        expert_id,
                    )
                    if self.host_mapped:
                        from .packed_hybrid import (
                            HostPackedWeight,
                            _coalesce_host_expert,
                        )

                        weights = _coalesce_host_expert(tuple(
                            HostPackedWeight.from_store(weight, weight.cb)
                            for weight in weights
                        ))
                    projection_weights = tuple(
                        zip((0, 5, 10), weights)
                    )
                else:
                    codebook_variants = self.store.codebook_variants(
                        layer,
                        base_tier,
                        expert_id,
                    )
                    gu, down = self.store.load_expert_packed(
                        layer,
                        expert_id,
                    )
                    if self.host_mapped:
                        from .packed_hybrid import (
                            HostPackedWeight,
                            _coalesce_host_expert,
                        )

                        gu, down = _coalesce_host_expert((
                            HostPackedWeight.from_store(gu, gu.cb),
                            HostPackedWeight.from_store(down, down.cb),
                        ))
                    projection_weights = ((0, gu), (5, down))
                if self.host_mapped:
                    from .packed_hybrid import _contiguous_expert_raw

                    mapped_expert = tuple(
                        weight for _base, weight in projection_weights
                    )
                    payload = _contiguous_expert_raw(mapped_expert)
                    if payload is None:
                        raise RuntimeError(
                            "mapped packed expert projections are not contiguous"
                        )
                    mapped_base = self._register_mapped_payload(payload)
                    host_base = int(payload.data_ptr())
                    projection_offsets = []
                    for base_and_weight in projection_weights:
                        base, weight = base_and_weight
                        projection_offset = (
                            int(weight.raw.data_ptr()) - host_base
                        )
                        self._mapped_device_pointers[
                            int(weight.raw.data_ptr())
                        ] = mapped_base + projection_offset
                        projection_offsets.append(
                            (int(base), int(projection_offset))
                        )
                    for projection_index, (base, projection_offset) in enumerate(
                        projection_offsets
                    ):
                        projection_end = (
                            projection_offsets[projection_index + 1][1]
                            if projection_index + 1 < len(projection_offsets)
                            else int(payload.nbytes)
                        )
                        self._mapped_projection_max[base] = max(
                            self._mapped_projection_max.get(base, 0),
                            projection_end - projection_offset,
                        )
                    self._mapped_experts.append(mapped_expert)
                    self._mapped_entries[(layer, expert_id)] = (
                        payload,
                        tuple(projection_offsets),
                    )
                target_ranks = self.expert_ranks(layer, expert_id)
                for rank in target_ranks:
                    device = self.devices[rank]
                    arena = (
                        self._arenas[rank]
                        if self.host_mapped
                        else self._resident_layer_arenas[(rank, int(layer))]
                    )
                    metadata = metadata_by_rank[rank]
                    with torch.cuda.device(device):
                        for base, weight in projection_weights:
                            if (
                                not self.host_mapped
                                and self.parallelism in {"tensor", "hybrid"}
                            ):
                                group_rank = (
                                    rank
                                    if self.parallelism == "tensor"
                                    else rank % self.tensor_group_size
                                )
                                if projection_vq:
                                    raw, blocks = (
                                        self._tensor_shard_projection_raw(
                                            weight,
                                            projection=base // 5,
                                            rank=group_rank,
                                            ranks=self.tensor_group_size,
                                            intermediate=intermediate,
                                        )
                                    )
                                else:
                                    raw, blocks = self._tensor_shard_raw(
                                        weight,
                                        projection=base,
                                        rank=group_rank,
                                        ranks=self.tensor_group_size,
                                        intermediate=intermediate,
                                    )
                            else:
                                raw = weight.raw
                                blocks = weight.blocks
                            if self.host_mapped:
                                target_pointer = self._mapped_device_pointers[
                                    int(raw.data_ptr())
                                ]
                                end = offsets[rank]
                            else:
                                offset_key = (rank, int(layer))
                                start = layer_offsets[offset_key]
                                end = start + raw.numel()
                                if end > arena.numel():
                                    raise RuntimeError(
                                        "packed arena overflow on "
                                        f"rank {rank}"
                                    )
                                target = arena[start:end]
                                target.copy_(raw)
                                target_pointer = int(target.data_ptr())
                            codebook = self._device_codebook(
                                rank,
                                layer,
                                base_tier,
                                codebook_variants[
                                    (
                                        base // 5
                                        if projection_vq
                                        else (0 if base == 0 else 1)
                                    )
                                ],
                                base,
                                weight.cb,
                                device,
                            )
                            metadata[base + 0, expert_id] = target_pointer
                            metadata[base + 1, expert_id] = (
                                codebook.data_ptr()
                            )
                            metadata[base + 2, expert_id] = blocks
                            metadata[base + 3, expert_id] = weight.dim
                            metadata[base + 4, expert_id] = (
                                weight.dtype_tag
                            )
                            if not self.host_mapped:
                                layer_offsets[offset_key] = end
                loaded += 1
                if loaded % 2000 == 0:
                    print(
                        f"[cccp-packed] packed 专家写入 "
                        f"{loaded}",
                        flush=True,
                    )
            if self.host_mapped:
                metadata = metadata_by_rank[0]
                if not metadata.is_pinned():
                    metadata = metadata.pin_memory()
                metadata_by_rank[0] = metadata
                self._mapped_host_metadata[layer] = metadata
            device_metadata = tuple(
                metadata.to(device, non_blocking=self.host_mapped)
                for metadata, device in zip(
                    metadata_by_rank,
                    self.devices,
                )
            )
            self._metadata[layer] = device_metadata
            if self._mapped_stage_enabled:
                self._mapped_uva_metadata[layer] = tuple(
                    metadata.clone() for metadata in device_metadata
                )
        if not self.host_mapped:
            for key, arena in self._resident_layer_arenas.items():
                actual = layer_offsets[key]
                expected = int(arena.numel())
                if actual != expected:
                    raise RuntimeError(
                        f"rank {key[0]} L{key[1]} packed bytes mismatch: "
                        f"{actual} != {expected}"
                    )
        if self._mapped_stage_enabled:
            if set(self._mapped_projection_max) != {0, 5, 10}:
                raise RuntimeError(
                    "mapped staging requires three packed projections"
                )
            alignment = 256
            self._mapped_stage_projection_strides = tuple(
                (
                    (self._mapped_projection_max[base] + alignment - 1)
                    // alignment
                    * alignment
                )
                for base in (0, 5, 10)
            )
            stage_stride = sum(self._mapped_stage_projection_strides)
            device = self.devices[0]
            with torch.cuda.device(device):
                self._mapped_stage_workspace = torch.empty(
                    top_k * stage_stride,
                    dtype=torch.uint8,
                    device=device,
                )
                self._mapped_stage_metadata = torch.empty(
                    15,
                    top_k,
                    dtype=torch.long,
                    device=device,
                )
                self._mapped_stage_route_ids = torch.arange(
                    top_k,
                    dtype=torch.long,
                    device=device,
                )
        workspace_intermediate = (
            intermediate // self.tensor_group_size
            if self.parallelism in {"tensor", "hybrid"}
            else intermediate
        )
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                self._workspaces[rank] = (
                    torch.empty(
                        top_k,
                        2 * workspace_intermediate,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    torch.empty(
                        top_k,
                        routed_hidden,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    torch.empty(
                        routed_hidden,
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                torch.cuda.synchronize(device)
        self._prepare_compact_decode()
        if self.parallelism in {"expert", "tensor", "hybrid"}:
            self._streams = [
                torch.cuda.Stream(device=device)
                for device in self.devices
            ]
            self._routed_inputs = [
                torch.empty(
                    1,
                    routed_hidden,
                    dtype=torch.bfloat16,
                    device=device,
                )
                for device in self.devices
            ]
            self._routed_ids = [
                torch.empty(
                    top_k,
                    dtype=torch.long,
                    device=device,
                )
                for device in self.devices
            ]
            self._routed_weights = [
                torch.empty(
                    top_k,
                    dtype=torch.float32,
                    device=device,
                )
                for device in self.devices
            ]
            self._source_inputs = [
                torch.empty(
                    1,
                    routed_hidden,
                    dtype=torch.bfloat16,
                    device=device,
                )
                for device in self.devices
            ]
            self._source_ids = [
                torch.empty(
                    top_k,
                    dtype=torch.long,
                    device=device,
                )
                for device in self.devices
            ]
            self._source_weights = [
                torch.empty(
                    top_k,
                    dtype=torch.float32,
                    device=device,
                )
                for device in self.devices
            ]
            self._source_events = [
                torch.cuda.Event()
                for _ in range(int(self.store.cfg["n_layers"]))
            ]
            self._done_events = [
                [
                    torch.cuda.Event()
                    for _ in range(int(self.store.cfg["n_layers"]))
                ]
                for _ in self.devices
            ]
            self._output_events = [
                [
                    torch.cuda.Event()
                    for _ in range(int(self.store.cfg["n_layers"]))
                ]
                for _ in self.devices
            ]
            for layer, event in enumerate(self._source_events):
                owner = self.plan.owner_by_layer[layer]
                with torch.cuda.device(self.devices[owner]):
                    event.cuda_event
            for rank, events in enumerate(self._done_events):
                with torch.cuda.device(self.devices[rank]):
                    for event in events:
                        event.cuda_event
                    for event in self._output_events[rank]:
                        event.record(
                            torch.cuda.current_stream(self.devices[rank])
                        )
            for owner, (start, end) in enumerate(self.plan.ranges):
                device = self.devices[owner]
                with torch.cuda.device(device):
                    self._return_buffers.append(
                        torch.empty(
                            len(self.devices),
                            end - start,
                            routed_hidden,
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    self._reduce_buffers.append(
                        torch.empty(
                            end - start,
                            routed_hidden,
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    self._zero_buffers.append(
                        torch.zeros(
                            routed_hidden,
                            dtype=torch.float32,
                            device=device,
                        )
                    )
            if (
                capture_graphs
                and os.environ.get("CCCP_TP_GRAPH", "1") != "0"
                and not self._compact_decode_enabled
            ):
                self._prepare_expert_graphs()
        self._prepare_native8_prefill()
        self.store._cb_cache.clear()
        gc.collect()
        self._payload_loaded = True
        self._payload_loaded_count = loaded
        self._payload_load_seconds = time.time() - started
        self.active = bool(capture_graphs)
        if self.host_mapped:
            print(
                f"[cccp-packed] packed 专家 UVA 映射完成：{loaded} 个，"
                f"RAM={self.host_expert_bytes / 2**30:.2f}GiB，"
                f"VRAM码本/工作区={self.gpu_storage_bytes / 2**30:.2f}GiB，"
                f"{self._payload_load_seconds:.1f}s；运行期无逐层 H2D",
                flush=True,
            )
        else:
            print(
                f"[cccp-packed] packed 专家全显存完成：{loaded} 个，"
                f"{self.gpu_storage_bytes / 2**30:.2f}GiB，"
                f"{self._payload_load_seconds:.1f}s，运行期专家 H2D=0",
                flush=True,
            )

    def _prepare_expert_graphs(self) -> None:
        """Capture one fixed-buffer packed MoE graph per layer and rank."""
        from .fusedext import (
            expert_dispatch_pack_fused,
            make_tp_graph_launch_batch,
            packed_stage_topk_three_projection_fused,
            tp_peer_copy_fused,
        )
        from .ops import packed_moe_topk

        if not self._streams:
            return
        started = time.time()
        top_k = int(self.store.cfg["top_k"])
        activation = str(
            self.store.cfg.get("activation", "situ")
        )
        activation_beta = float(
            self.store.cfg.get("situ_beta", 4.0)
        )
        linear_value = self.store.cfg.get("situ_linear_beta")
        activation_linear_beta = (
            0.0 if linear_value is None else float(linear_value)
        )
        activation_limit = float(
            self.store.cfg.get("swiglu_limit", 0.0)
        )
        for layer in sorted(self.store.man.expert_files):
            owner = self.plan.owner_by_layer[layer]
            owner_device = self.devices[owner]
            local_layer = layer - self.plan.ranges[owner][0]
            available = (
                self.store.available_mask(layer)
                .nonzero()
                .reshape(-1)[:top_k]
            )
            if available.numel() != top_k:
                raise RuntimeError(
                    f"layer {layer} has fewer than Top-K experts"
                )
            with torch.cuda.device(owner_device):
                self._source_inputs[owner].zero_()
                self._source_ids[owner].copy_(available)
                self._source_weights[owner].fill_(1.0 / top_k)
                torch.cuda.synchronize(owner_device)
            if self.hidden_mode:
                bound = self._bound_hidden_inputs.get(layer)
                if bound is None:
                    raise RuntimeError(
                        f"packed MoE layer {layer} has no fixed all-rank input"
                    )
                rank_inputs, rank_weights, rank_ids = bound
                for rank, device in enumerate(self.devices):
                    with torch.cuda.device(device):
                        rank_inputs[rank].zero_()
                        rank_ids[rank].copy_(
                            available.to(device)
                        )
                        rank_weights[rank].fill_(1.0 / top_k)
                        torch.cuda.synchronize(device)
            else:
                rank_inputs = tuple(self._routed_inputs)
                rank_weights = tuple(self._routed_weights)
                rank_ids = tuple(self._routed_ids)
            graphs: list[torch.cuda.CUDAGraph] = []
            rank_order = (
                tuple(range(len(self.devices)))
                if self.hidden_mode
                else (
                    owner,
                    *(
                        rank
                        for rank in range(len(self.devices))
                        if rank != owner
                    ),
                )
            )
            for rank in rank_order:
                device = self.devices[rank]
                stream = self._streams[rank]
                hidden, output, local_result = self._workspaces[rank]
                destination = (
                    None
                    if self.hidden_mode
                    else self._return_buffers[owner][rank, local_layer]
                )
                result = (
                    local_result
                    if self.hidden_mode or rank != owner
                    else destination
                )

                def launch_rank() -> None:
                    if (
                        not self.hidden_mode
                        and not expert_dispatch_pack_fused(
                            self._source_inputs[owner],
                            self._source_ids[owner],
                            self._source_weights[owner],
                            self._routed_inputs[rank],
                            self._routed_ids[rank],
                            self._routed_weights[rank],
                        )
                    ):
                        raise RuntimeError(
                            "packed MoE graph dispatch rejected fixed buffers"
                        )
                    packed_route_ids = rank_ids[rank].reshape(-1)
                    packed_metadata = self._metadata[layer][rank]
                    if self._mapped_stage_enabled:
                        staged = packed_stage_topk_three_projection_fused(
                            packed_route_ids,
                            packed_metadata,
                            self._mapped_uva_metadata[layer][rank],
                            self._mapped_stage_workspace,
                            self._mapped_stage_metadata,
                            self._mapped_stage_route_ids,
                            hidden=int(
                                self.store.cfg.get(
                                    "routed_hidden",
                                    self.store.cfg["hidden"],
                                )
                            ),
                            intermediate=int(self.store.cfg["moe_inter"]),
                            projection_strides=(
                                self._mapped_stage_projection_strides
                            ),
                        )
                        if staged is None:
                            raise RuntimeError(
                                "mapped packed staging rejected fixed buffers"
                            )
                        packed_route_ids = self._mapped_stage_route_ids
                        packed_metadata = staged
                    if self._compact_decode_enabled:
                        self._run_compact_decode_rank(
                            layer,
                            rank,
                            rank_inputs[rank],
                            packed_route_ids,
                            rank_weights[rank],
                            activation=activation,
                            activation_beta=activation_beta,
                            activation_linear_beta=activation_linear_beta,
                            limit=activation_limit,
                            result=result,
                        )
                    else:
                        packed_moe_topk(
                            rank_inputs[rank],
                            packed_route_ids,
                            rank_weights[rank].reshape(-1),
                            packed_metadata,
                            activation=activation,
                            activation_beta=activation_beta,
                            activation_linear_beta=(
                                activation_linear_beta
                            ),
                            limit=activation_limit,
                            hidden_workspace=hidden,
                            output_workspace=output,
                            result=result,
                            grouped_prefix=-1,
                            **self.store.man.projection_operator_capability(
                                layer
                            ),
                        )
                    if (
                        not self.hidden_mode
                        and destination is not None
                        and
                        rank != owner
                        and not tp_peer_copy_fused(
                            local_result,
                            destination,
                        )
                    ):
                        raise RuntimeError(
                            "packed MoE local reduction dispatch "
                            "was rejected"
                        )

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    launch_rank()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph(
                        keep_graph=self.hidden_mode,
                    )
                    with torch.cuda.graph(graph, stream=stream):
                        launch_rank()
                    if self.hidden_mode:
                        graph.instantiate()
                    graphs.append(graph)
            for device in self.devices:
                torch.cuda.synchronize(device)
            ordered_streams = [
                self._streams[rank] for rank in rank_order
            ]
            ordered_events = [
                self._done_events[rank][layer]
                for rank in rank_order
            ]
            for rank in rank_order:
                device = self.devices[rank]
                stream = self._streams[rank]
                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    self._done_events[rank][layer].record(stream)
                    stream.synchronize()
            with torch.cuda.device(owner_device):
                self._source_events[layer].record(
                    torch.cuda.current_stream(owner_device)
                )
                torch.cuda.synchronize(owner_device)
                self._graphs[layer] = graphs
                self._graph_rank_order[layer] = rank_order
                self._graph_batches[layer] = make_tp_graph_launch_batch(
                    [int(self.devices[rank].index) for rank in rank_order],
                    graphs,
                    ordered_streams,
                    ordered_events,
                    self._source_events[layer],
                )
            if self.hidden_mode:
                self._output_replicas[layer] = []
                for rank, device in enumerate(self.devices):
                    with torch.cuda.device(device):
                        self._output_replicas[layer].append(
                            torch.empty(
                                1,
                                int(
                                    self.store.cfg.get(
                                        "routed_hidden",
                                        self.store.cfg["hidden"],
                                    )
                                ),
                                dtype=torch.bfloat16,
                                device=device,
                            )
                        )
        print(
            f"[cccp-packed] 通用 packed MoE TP Graph 完成："
            f"{len(self._graphs)} 层×{len(self.devices)} 卡，"
            f"{time.time() - started:.1f}s",
            flush=True,
        )

    def compose_route_topk(
        self,
        logits_by_layer: dict[int, object],
        corrections_by_layer: dict[int, tuple[torch.Tensor, ...]],
        masks_by_layer: dict[int, tuple[torch.Tensor, ...]],
        route_buffers_by_layer: dict[int, tuple],
        *,
        scoring_func: str,
        top_k: int,
        normalize: bool,
        scaling: float,
        n_group: int,
        topk_group: int,
        layers=None,
    ) -> None:
        """Compose registered Top-K and packed-expert graphs per TP rank.

        The Router/Down collective publishes fixed logits and latent replicas.
        Each rank then performs the same registered Top-K and computes its
        shard of every selected packed expert.  Only graph scheduling changes;
        packed indices and all-rank expert ownership are unchanged.
        """
        if not self.hidden_mode or not self._graphs:
            raise RuntimeError(
                "route/packed composition requires all-rank packed graphs"
            )
        from .fusedext import make_tp_graph_sequence_batch
        from .ops import route_topk

        selected_layers = (
            sorted(self._graphs)
            if layers is None
            else sorted({int(layer) for layer in layers})
        )
        unknown = set(selected_layers) - set(self._graphs)
        if unknown:
            raise ValueError(
                "route/packed composition references unknown layers: "
                + ",".join(str(layer) for layer in sorted(unknown))
            )
        for layer in selected_layers:
            logits = logits_by_layer[layer]
            corrections = corrections_by_layer[layer]
            masks = masks_by_layer[layer]
            weight_buffers, index_buffers = route_buffers_by_layer[layer]
            if (
                tuple(logits.devices) != self.devices
                or logits.ready_events is None
                or len(corrections) != len(self.devices)
                or len(masks) != len(self.devices)
            ):
                raise ValueError(
                    "route/packed fixed all-rank layout mismatch"
                )
            rank_order = self._graph_rank_order[layer]
            if tuple(rank_order) != tuple(range(len(self.devices))):
                raise RuntimeError(
                    "route/packed composition forbids owner-ordered graphs"
                )
            expert_by_rank = {
                rank: self._graphs[layer][ordered_rank]
                for ordered_rank, rank in enumerate(rank_order)
            }
            route_graphs = []
            for rank, device in enumerate(self.devices):
                stream = self._streams[rank]

                def launch_route(rank_index: int = rank) -> None:
                    route = route_topk(
                        logits.replicas[rank_index],
                        corrections[rank_index],
                        masks[rank_index],
                        scoring_func=scoring_func,
                        top_k=int(top_k),
                        normalize=bool(normalize),
                        scaling=float(scaling),
                        n_group=int(n_group),
                        topk_group=int(topk_group),
                        output_buffers=(
                            weight_buffers[rank_index],
                            index_buffers[rank_index],
                        ),
                    )
                    if route is None:
                        raise RuntimeError(
                            "registered route Top-K rejected graph inputs"
                        )

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    launch_route()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph, stream=stream):
                        launch_route()
                    graph.instantiate()
                    stream.synchronize()
                route_graphs.append(graph)
            self._route_graphs[layer] = tuple(route_graphs)
            self._graph_batches[layer] = make_tp_graph_sequence_batch(
                [int(device.index) for device in self.devices],
                [
                    [
                        route_graphs[rank],
                        expert_by_rank[rank],
                    ]
                    for rank in range(len(self.devices))
                ],
                list(self._streams),
                [
                    self._done_events[rank][layer]
                    for rank in range(len(self.devices))
                ],
                self._source_events[layer],
            )
        print(
            "[cccp-packed] 通用 Route TopK→packed MoE 全rank父图完成："
            f"{len(selected_layers)} 层×{len(self.devices)} rank",
            flush=True,
        )

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
        """Run one tensor-sharded expert set and publish it on every rank.

        ``value`` and every route pair are rank-local replicas.  Each rank
        executes the same selected experts using its packed Column/Row shard;
        the only packed-MoE collective is the final all-rank Row reduction.
        """
        del activation, activation_beta, activation_linear_beta
        from .ops import TPHidden

        if (
            not self.active
            or not self.hidden_mode
            or len(routes) != len(self.devices)
            or value.ready_events is None
        ):
            raise RuntimeError(
                "packed MoE all-rank state is unavailable"
            )
        graph_batch = self._graph_batches.get(layer)
        outputs = self._output_replicas.get(layer)
        if graph_batch is None or outputs is None:
            raise RuntimeError(
                f"packed MoE layer {layer} graph is unavailable"
            )
        bound = self._bound_hidden_inputs.get(layer)
        if bound is None:
            raise RuntimeError(
                f"packed MoE layer {layer} fixed inputs are unavailable"
            )
        bound_inputs, bound_weights, bound_ids = bound
        for rank, device in enumerate(self.devices):
            weights, indices = routes[rank]
            weights = weights.reshape(-1)
            indices = indices.reshape(-1)
            if (
                weights.device != device
                or indices.device != device
                or weights.shape != bound_weights[rank].shape
                or indices.shape != bound_ids[rank].shape
                or weights.data_ptr() != bound_weights[rank].data_ptr()
                or indices.data_ptr() != bound_ids[rank].data_ptr()
                or value.replicas[rank].data_ptr()
                != bound_inputs[rank].data_ptr()
            ):
                raise ValueError(
                    "packed MoE route replica layout mismatch"
                )
        with torch.cuda.device(self.devices[0]):
            graph_batch.launch_all_rank_from_events(
                [
                    value.ready_events[rank].cuda_event
                    for rank in range(len(self.devices))
                ],
                [
                    self._workspaces[rank][2]
                    for rank in range(len(self.devices))
                ],
                outputs,
                [
                    self._output_events[rank][layer].cuda_event
                    for rank in range(len(self.devices))
                ],
            )
        self.hits += int(routes[0][1].numel())
        return TPHidden(
            self.devices,
            tuple(outputs),
            tuple(
                self._output_events[rank][layer]
                for rank in range(len(self.devices))
            ),
        )

    def prefill_rows_available(self, layer: int) -> bool:
        """行批量 prefill MoE 是否可用（三投影 projection-VQ 元数据）。"""
        return bool(
            self.full_resident
            and self.store.man.projection_operator_capability(layer)
        )

    def _grouped_local_mask(
        self,
        layer: int,
        rank: int,
    ) -> torch.Tensor:
        """Return resident experts whose complete routed MLP is present.

        Projection-VQ stores Gate/Up/Down in three five-row directories.
        The original CCCP layout stores the fused Gate+Up matrix and Down in
        two five-row directories.  Both layouts are accepted by the public
        projection dequantizer, so the residency test must follow the actual
        metadata height instead of assuming the newer 15-row form.
        """
        key = (int(layer), int(rank))
        mask = self._grouped_local_masks.get(key)
        if mask is None:
            metadata = self._metadata[layer][rank]
            if int(metadata.shape[0]) == 15:
                down = 10
            elif int(metadata.shape[0]) == 10:
                down = 5
            else:
                raise RuntimeError(
                    "packed expert metadata must have 10 or 15 rows for "
                    "grouped Prefill"
                )
            mask = (
                (metadata[0] != 0)
                & (metadata[1] != 0)
                & (metadata[2] > 0)
                & (metadata[down] != 0)
                & (metadata[down + 1] != 0)
                & (metadata[down + 2] > 0)
            )
            self._grouped_local_masks[key] = mask
        return mask

    def _prepare_native8_prefill(self) -> None:
        """Compile shared codebooks and metadata for transient FP8 Prefill.

        This experimental gate is capability based and intentionally lives in
        the common resident runtime.  It will be removed after the H20 A/B:
        the winning executor becomes the sole default, while a losing path is
        deleted instead of retained as another fallback.
        """
        if (
            os.environ.get("CCCP_RESIDENT_NATIVE8_PREFILL", "0") != "1"
            or torch.version.hip is not None
            or not self.store.man.projection_vq
            or not self.devices
        ):
            return
        from .ops.sm90_grouped import select_grouped_fp8_backend

        capability = torch.cuda.get_device_capability(self.devices[0])
        backend = select_grouped_fp8_backend(
            (int(capability[0]), int(capability[1]))
        )
        if backend is None:
            return
        hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        intermediate = int(self.store.cfg["moe_inter"])
        local_intermediate = (
            intermediate // self.tensor_group_size
            if self.parallelism in {"tensor", "hybrid"}
            else intermediate
        )
        if backend == "deepgemm-sm90" and (
            hidden % 128 or local_intermediate % 128
        ):
            return

        replacement_by_rank: list[dict[int, tuple[int, float]]] = []
        for rank in range(len(self.devices)):
            image = compile_shared_codebook_image(
                [
                    codebook
                    for (owner, _tier, _variant, _projection), codebook
                    in self._codebooks.items()
                    if int(owner) == int(rank)
                ],
                mode="e4m3",
            )
            self._native8_codebooks.update(image.tensors)
            replacement_by_rank.append(image.replacements)

        metadata_by_layer: dict[int, tuple[torch.Tensor, ...]] = {}
        scales_by_layer: dict[int, tuple[torch.Tensor, ...]] = {}
        for layer, rank_metadata in self._metadata.items():
            native_metadata = []
            native_scales = []
            for rank, (device, metadata) in enumerate(
                zip(self.devices, rank_metadata)
            ):
                rewritten, scales = rewrite_packed_codebook_metadata(
                    metadata,
                    replacement_by_rank[rank],
                )
                native_metadata.append(rewritten.to(device=device))
                native_scales.append(scales.to(device=device))
            metadata_by_layer[int(layer)] = tuple(native_metadata)
            scales_by_layer[int(layer)] = tuple(native_scales)
        self._native8_metadata = metadata_by_layer
        self._native8_scales = scales_by_layer
        self._native8_grouped_backend = backend
        self._native8_prefill_enabled = bool(metadata_by_layer)
        if self._native8_prefill_enabled:
            print(
                "[cccp-native8] resident Prefill A/B enabled; "
                "algorithm=packed-index→E4M3-layer-image→grouped-TensorCore; "
                f"backend={backend}; model-branch=none",
                flush=True,
            )

    def _native8_layer_supported(self, layer: int) -> bool:
        if not self._native8_prefill_enabled:
            return False
        return all(
            bool(self._grouped_local_mask(int(layer), rank).all())
            for rank in range(len(self.devices))
        )

    def _prepare_compact_decode(self) -> None:
        """Compile the public Q8/DP4A Decode image for resident codebooks.

        Selection is based only on packed metadata and device capability; no
        architecture adapter is allowed to select or replace this path.
        """

        if (
            torch.version.hip is not None
            or not self.devices
        ):
            return
        major, _minor = torch.cuda.get_device_capability(self.devices[0])
        if int(major) < 7:
            return
        if any(
            int(metadata.shape[0]) != 15
            for rank_metadata in self._metadata.values()
            for metadata in rank_metadata
        ):
            return

        replacement_by_rank: list[dict[int, tuple[int, float]]] = []
        for rank in range(len(self.devices)):
            image = compile_shared_codebook_image(
                [
                    codebook
                    for (owner, _tier, _variant, _projection), codebook
                    in self._codebooks.items()
                    if int(owner) == int(rank)
                ],
                mode="q8",
            )
            self._compact_decode_codebooks.update(image.tensors)
            replacement_by_rank.append(image.replacements)

        metadata_by_layer: dict[int, tuple[torch.Tensor, ...]] = {}
        scales_by_layer: dict[int, tuple[torch.Tensor, ...]] = {}
        for layer, rank_metadata in self._metadata.items():
            compiled_metadata = []
            compiled_scales = []
            for rank, (device, metadata) in enumerate(
                zip(self.devices, rank_metadata)
            ):
                rewritten, scales = rewrite_packed_codebook_metadata(
                    metadata,
                    replacement_by_rank[rank],
                )
                compiled_metadata.append(rewritten.to(device=device))
                compiled_scales.append(scales.to(device=device))
            metadata_by_layer[int(layer)] = tuple(compiled_metadata)
            scales_by_layer[int(layer)] = tuple(compiled_scales)
        self._compact_decode_metadata = metadata_by_layer
        self._compact_decode_scales = scales_by_layer

        top_k = int(self.store.cfg["top_k"])
        hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        intermediate = int(self.store.cfg["moe_inter"])
        local_intermediate = (
            intermediate // self.tensor_group_size
            if self.parallelism in {"tensor", "hybrid"}
            else intermediate
        )
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                self._compact_decode_workspaces[rank] = {
                    "metadata": torch.empty(
                        15, top_k, dtype=torch.long, device=device
                    ),
                    "scales": torch.empty(
                        top_k, 3, dtype=torch.float32, device=device
                    ),
                    "route_ids": torch.arange(
                        top_k, dtype=torch.long, device=device
                    ),
                    "gate_quant": torch.empty(
                        top_k,
                        4 * ((hidden + 15) & ~15),
                        dtype=torch.uint8,
                        device=device,
                    ),
                    "down_quant": torch.empty(
                        top_k,
                        2 * ((local_intermediate + 15) & ~15),
                        dtype=torch.uint8,
                        device=device,
                    ),
                }
        self._compact_decode_enabled = bool(metadata_by_layer)
        if self._compact_decode_enabled:
            print(
                "[cccp-codebook] resident Decode="
                "packed-index+Q8-codebook+DP4A; model-branch=none; "
                f"codebook_bytes={sum(t.nbytes for t in self._compact_decode_codebooks.values()) / 2**20:.2f}MiB",
                flush=True,
            )

    def _run_compact_decode_rank(
        self,
        layer: int,
        rank: int,
        value: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float,
        result: torch.Tensor,
    ) -> torch.Tensor:
        """Select Top-K metadata, then invoke the common compact executor."""

        ids = route_ids.reshape(-1)
        count = int(ids.numel())
        workspace = self._compact_decode_workspaces[int(rank)]
        metadata = workspace["metadata"][:, :count]
        scales = workspace["scales"][:count]
        torch.index_select(
            self._compact_decode_metadata[int(layer)][int(rank)],
            1,
            ids,
            out=metadata,
        )
        torch.index_select(
            self._compact_decode_scales[int(layer)][int(rank)],
            0,
            ids,
            out=scales,
        )
        hidden_workspace, output_workspace, _ = self._workspaces[int(rank)]
        return run_compact_q8_codebook_decode(
            value=value,
            route_ids=workspace["route_ids"][:count],
            route_weights=route_weights.reshape(-1).float().contiguous(),
            metadata=metadata,
            scales=scales,
            activation=activation,
            activation_beta=float(activation_beta),
            activation_linear_beta=activation_linear_beta,
            limit=float(limit),
            hidden_workspace=hidden_workspace[:count],
            output_workspace=output_workspace[:count],
            result=result,
            gate_quant_workspace=workspace["gate_quant"][:count],
            down_quant_workspace=workspace["down_quant"][:count],
        )

    def _native8_workspace(
        self,
        rank: int,
        *,
        expert_count: int,
        routed_rows: int,
        local_intermediate: int,
        hidden: int,
    ) -> dict[str, object]:
        cached = self._native8_workspaces.get(int(rank))
        if (
            cached is not None
            and int(cached["expert_count"]) >= int(expert_count)
            and int(cached["routed_rows"]) >= int(routed_rows)
        ):
            return cached
        device = self.devices[int(rank)]
        options = {"dtype": torch.float8_e4m3fn, "device": device}
        cached = {
            "expert_count": int(expert_count),
            "routed_rows": int(routed_rows),
            "gu": torch.empty(
                expert_count, 2 * local_intermediate, hidden, **options
            ),
            "down": torch.empty(
                expert_count, hidden, local_intermediate, **options
            ),
            "input": torch.empty(routed_rows, hidden, **options),
            "input_scales": torch.empty(
                routed_rows, 1, dtype=torch.float32, device=device
            ),
            "activated": torch.empty(
                routed_rows, local_intermediate, **options
            ),
            "activated_scales": torch.empty(
                routed_rows, 1, dtype=torch.float32, device=device
            ),
            "inverse": torch.empty(
                routed_rows, dtype=torch.long, device=device
            ),
            # Metadata and projection scales are tiny but their second
            # dimension must exactly match the active DeepGEMM bucket.  Keep
            # one buffer per public bucket while reusing the large weight and
            # activation workspaces across every routed-VQ model.
            "metadata_buffers": {},
            "projection_scale_buffers": {},
        }
        if self._native8_grouped_backend == "deepgemm-sm90":
            from .ops.sm90_grouped import (
                deepgemm_grouped_alignment,
                deepgemm_grouped_padded_rows,
                DeepGEMMGroupedWorkspace,
            )

            padded_rows = deepgemm_grouped_padded_rows(
                routed_rows,
                expert_count,
                alignment=deepgemm_grouped_alignment(),
            )
            cached.update({
                "input_block_scales": torch.empty(
                    routed_rows,
                    hidden // 128,
                    dtype=torch.float32,
                    device=device,
                ),
                "activated_block_scales": torch.empty(
                    routed_rows,
                    local_intermediate // 128,
                    dtype=torch.float32,
                    device=device,
                ),
                "gu_block_scales": torch.empty(
                    expert_count,
                    (2 * local_intermediate) // 128,
                    hidden // 128,
                    dtype=torch.float32,
                    device=device,
                ),
                "down_block_scales": torch.empty(
                    expert_count,
                    hidden // 128,
                    local_intermediate // 128,
                    dtype=torch.float32,
                    device=device,
                ),
                "gate_up_output": torch.empty(
                    routed_rows,
                    2 * local_intermediate,
                    dtype=torch.bfloat16,
                    device=device,
                ),
                "down_output": torch.empty(
                    routed_rows,
                    hidden,
                    dtype=torch.bfloat16,
                    device=device,
                ),
                "gate_up_deepgemm": DeepGEMMGroupedWorkspace(
                    value=torch.empty(
                        padded_rows, hidden, **options
                    ),
                    scale_a=torch.empty(
                        padded_rows,
                        hidden // 128,
                        dtype=torch.float32,
                        device=device,
                    ),
                    output=torch.empty(
                        padded_rows,
                        2 * local_intermediate,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                ),
                "down_deepgemm": DeepGEMMGroupedWorkspace(
                    value=torch.empty(
                        padded_rows, local_intermediate, **options
                    ),
                    scale_a=torch.empty(
                        padded_rows,
                        local_intermediate // 128,
                        dtype=torch.float32,
                        device=device,
                    ),
                    output=torch.empty(
                        padded_rows,
                        hidden,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                ),
            })
        else:
            cached.update({
                "gu_scales": torch.empty(
                    expert_count,
                    2 * local_intermediate,
                    dtype=torch.float32,
                    device=device,
                ),
                "down_scales": torch.empty(
                    expert_count,
                    hidden,
                    dtype=torch.float32,
                    device=device,
                ),
            })
        self._native8_workspaces[int(rank)] = cached
        return cached

    def _run_rows_native8_rank(
        self,
        layer: int,
        rank: int,
        inputs: torch.Tensor,
        route_ids_mb: torch.Tensor,
        route_weights_mb: torch.Tensor,
        result: torch.Tensor,
        count: int,
        top_k: int,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float,
    ) -> torch.Tensor:
        """Execute one resident layer via common E4M3 grouped Tensor Cores."""
        from .fusedext import (
            dense_fp8_quantize_rows_fused,
            gated_activation_fp8_quantize_rows_fused,
            routed_weighted_reduce_fused,
        )
        from .ops import projection_expand_native8
        from .ops.sm90_grouped import (
            build_active_grouped_routes,
            build_deepgemm_grouped_layout,
            deepgemm_grouped_alignment,
            execute_grouped_fp8,
            projection_block_scales,
            row_block_scales,
        )

        device = self.devices[int(rank)]
        metadata = self._native8_metadata[int(layer)][int(rank)]
        projection_scales = self._native8_scales[int(layer)][int(rank)]
        expert_capacity = int(metadata.shape[1])
        hidden = int(inputs.shape[1])
        intermediate = int(self.store.cfg["moe_inter"])
        local_intermediate = (
            intermediate // self.tensor_group_size
            if self.parallelism in {"tensor", "hybrid"}
            else intermediate
        )
        routed_rows = int(count) * int(top_k)
        route_groups = build_active_grouped_routes(
            route_ids_mb.reshape(-1),
            group_capacity=expert_capacity,
            bucketed=(
                self._native8_grouped_backend == "deepgemm-sm90"
            ),
        )
        active_groups = int(route_groups.active_group_count)
        execution_groups = int(route_groups.execution_group_count)
        workspace = self._native8_workspace(
            rank,
            expert_count=expert_capacity,
            routed_rows=routed_rows,
            local_intermediate=local_intermediate,
            hidden=hidden,
        )
        metadata_key = (int(metadata.shape[0]), execution_groups)
        metadata_buffers = workspace["metadata_buffers"]
        selected_metadata = metadata_buffers.get(metadata_key)
        if selected_metadata is None:
            selected_metadata = torch.empty(
                int(metadata.shape[0]),
                execution_groups,
                dtype=torch.long,
                device=device,
            )
            metadata_buffers[metadata_key] = selected_metadata
        scale_buffers = workspace["projection_scale_buffers"]
        selected_scales = scale_buffers.get(execution_groups)
        if selected_scales is None:
            selected_scales = torch.empty(
                execution_groups,
                3,
                dtype=torch.float32,
                device=device,
            )
            scale_buffers[execution_groups] = selected_scales
        torch.index_select(
            metadata,
            1,
            route_groups.unique_group_ids,
            out=selected_metadata[:, :active_groups],
        )
        torch.index_select(
            projection_scales,
            0,
            route_groups.unique_group_ids,
            out=selected_scales[:active_groups],
        )
        if execution_groups > active_groups:
            selected_metadata[:, active_groups:].copy_(
                selected_metadata[:, active_groups - 1 : active_groups]
                .expand(-1, execution_groups - active_groups)
            )
            selected_scales[active_groups:].copy_(
                selected_scales[active_groups - 1 : active_groups]
                .expand(execution_groups - active_groups, -1)
            )
        gu = workspace["gu"][:execution_groups]
        down_weights = workspace["down"][:execution_groups]
        projection_expand_native8(selected_metadata, gu, down_weights)

        order = route_groups.order
        sorted_experts = route_groups.sorted_group_ids
        token_ids = (
            torch.arange(count, dtype=torch.long, device=device)
            .view(-1, 1)
            .expand(count, top_k)
            .reshape(-1)
        )
        sorted_tokens = token_ids[order].contiguous()
        grouped_input = inputs.index_select(0, sorted_tokens).contiguous()
        native_input = workspace["input"][:routed_rows]
        input_scales = workspace["input_scales"][:routed_rows]
        if dense_fp8_quantize_rows_fused(
            grouped_input, native_input, input_scales
        ) is None:
            raise RuntimeError("resident native8 input quantizer rejected rows")

        backend = str(self._native8_grouped_backend)
        offsets = None
        deepgemm_layout = None
        gate_up_output = None
        if backend == "deepgemm-sm90":
            gu_scales = workspace["gu_block_scales"][:execution_groups]
            down_scales = workspace["down_block_scales"][:execution_groups]
            projection_block_scales(
                selected_scales,
                hidden=hidden,
                intermediate=local_intermediate,
                gate_up_output=gu_scales,
                down_output=down_scales,
            )
            input_gemm_scales = row_block_scales(
                input_scales,
                k=hidden,
                output=workspace["input_block_scales"][:routed_rows],
            )
            deepgemm_layout = build_deepgemm_grouped_layout(
                sorted_experts,
                group_count=execution_groups,
                alignment=deepgemm_grouped_alignment(),
            )
            gate_up_output = workspace["gate_up_output"][:routed_rows]
        else:
            gu_scales = workspace["gu_scales"][:execution_groups]
            down_scales = workspace["down_scales"][:execution_groups]
            gu_scales[:, :local_intermediate].copy_(
                selected_scales[:, 0:1]
            )
            gu_scales[:, local_intermediate:].copy_(
                selected_scales[:, 1:2]
            )
            down_scales.copy_(selected_scales[:, 2:3])
            expert_ids = self._dequant_id_ranges.get(rank)
            if expert_ids is None or expert_ids.numel() != execution_groups:
                expert_ids = torch.arange(
                    execution_groups, dtype=torch.long, device=device
                )
                self._dequant_id_ranges[rank] = expert_ids
            offsets = torch.searchsorted(
                sorted_experts, expert_ids, right=True
            ).to(torch.int32)
            input_gemm_scales = input_scales.view(-1)
        gate_up = execute_grouped_fp8(
            native_input,
            gu,
            scale_a=input_gemm_scales,
            scale_b=gu_scales,
            offsets=offsets,
            backend=backend,
            deepgemm_layout=deepgemm_layout,
            deepgemm_workspace=(
                workspace["gate_up_deepgemm"]
                if backend == "deepgemm-sm90"
                else None
            ),
            output=gate_up_output,
        )

        native_activated = workspace["activated"][:routed_rows]
        activated_scales = workspace["activated_scales"][:routed_rows]
        if gated_activation_fp8_quantize_rows_fused(
            gate_up,
            native_activated,
            activated_scales,
            activation=activation,
            beta=float(activation_beta),
            linear_beta=activation_linear_beta,
            limit=float(limit),
        ) is None:
            raise RuntimeError(
                "resident native8 gated activation quantizer rejected rows"
            )
        down_output = None
        if backend == "deepgemm-sm90":
            activated_gemm_scales = row_block_scales(
                activated_scales,
                k=local_intermediate,
                output=workspace["activated_block_scales"][:routed_rows],
            )
            down_output = workspace["down_output"][:routed_rows]
        else:
            activated_gemm_scales = activated_scales.view(-1)
        routed = execute_grouped_fp8(
            native_activated,
            down_weights,
            scale_a=activated_gemm_scales,
            scale_b=down_scales,
            offsets=offsets,
            backend=backend,
            deepgemm_layout=deepgemm_layout,
            deepgemm_workspace=(
                workspace["down_deepgemm"]
                if backend == "deepgemm-sm90"
                else None
            ),
            output=down_output,
        )
        reduced = routed_weighted_reduce_fused(
            routed,
            order,
            route_weights_mb.reshape(-1).float(),
            workspace["inverse"][:routed_rows],
            result,
            top_k=top_k,
        )
        if reduced is None:
            raise RuntimeError("resident native8 route reduction was rejected")
        return reduced

    def _dequant_supported(self, layer: int) -> bool:
        """dequant+grouped GEMM 路径是否适用于该层（全部专家本地有效）。

        b 矩阵按全局专家 id 行对齐，因此要求 rank 上每个专家的三投影
        元数据都有效（张量并行按 intermediate 分片时成立）。同时按
        空闲显存门控 scratch 分配，避免超大模型挤占 KV/运行时预算。
        """
        cached = self._dequant_supported_cache.get(layer)
        if cached is not None:
            return cached
        # Windows ROCm's private torch._grouped_mm currently terminates the
        # process inside torch_hip.dll on gfx1150. AMD uses CCCP's public
        # packed grouped operator and never probes this native crash path.
        ok = torch.version.hip is None and hasattr(torch, "_grouped_mm")
        if ok:
            for rank, device in enumerate(self.devices):
                mask = self._grouped_local_mask(layer, rank)
                if not bool(mask.all()):
                    ok = False
                    break
                expert_count = int(mask.numel())
                intermediate = int(self.store.cfg["moe_inter"])
                hidden = int(
                    self.store.cfg.get(
                        "routed_hidden", self.store.cfg["hidden"]
                    )
                )
                local_inter = (
                    intermediate // self.tensor_group_size
                    if self.parallelism in {"tensor", "hybrid"}
                    else intermediate
                )
                needed = _dequant_scratch_bytes(
                    expert_count,
                    local_inter,
                    hidden,
                )
                try:
                    free, _total = torch.cuda.mem_get_info(device)
                except Exception:
                    free = 0
                if free < needed * 2:
                    ok = False
                    break
        self._dequant_supported_cache[layer] = ok
        return ok

    def _dequant_scratch(
        self,
        rank: int,
        expert_count: int,
        local_inter: int,
        hidden: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self._dequant_buffers.get(rank)
        gu_shape = (expert_count, 2 * local_inter, hidden)
        down_shape = (expert_count, hidden, local_inter)
        if (
            entry is None
            or tuple(entry[0].shape) != gu_shape
            or tuple(entry[1].shape) != down_shape
        ):
            with torch.cuda.device(device):
                entry = (
                    torch.empty(
                        gu_shape, dtype=torch.bfloat16, device=device
                    ),
                    torch.empty(
                        down_shape, dtype=torch.bfloat16, device=device
                    ),
                )
            self._dequant_buffers[rank] = entry
        return entry

    def _prefill_dequant_launch(
        self,
        layer: int,
        intermediate: int,
        hidden: int,
    ) -> None:
        """本层全部 rank 的稠密权重展开（每次 run_rows 调用一次）。"""
        from .ops import projection_dequant

        local_inter = (
            intermediate // self.tensor_group_size
            if self.parallelism in {"tensor", "hybrid"}
            else intermediate
        )
        for rank, device in enumerate(self.devices):
            metadata = self._metadata[layer][rank]
            stream = (
                self._streams[rank]
                if self._streams
                else torch.cuda.current_stream(device)
            )
            with torch.cuda.device(device), torch.cuda.stream(stream):
                gu, down = self._dequant_scratch(
                    rank,
                    int(metadata.shape[1]),
                    local_inter,
                    hidden,
                    device,
                )
                projection_dequant(metadata, gu, down)

    def _run_rows_dequant_rank(
        self,
        layer: int,
        rank: int,
        inputs: torch.Tensor,
        route_ids_mb: torch.Tensor,
        route_weights_mb: torch.Tensor,
        result: torch.Tensor,
        count: int,
        top_k: int,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float,
    ) -> torch.Tensor:
        """单 rank dequant + 公共 grouped GEMM 的 Top-K MoE 部分和。

        稠密权重已由 ``_prefill_dequant_launch`` 展开到本 rank scratch；
        这里只做路由排序与两次 ``torch._grouped_mm``（cuBLAS 分组
        GEMM），权重读取从每 (row,k) 一次降为每层一次。
        """
        from .grouped import activate_gate_up

        device = self.devices[rank]
        gu_buf, down_buf = self._dequant_buffers[rank]
        expert_ids = self._dequant_id_ranges.get(rank)
        if expert_ids is None or expert_ids.numel() != gu_buf.shape[0]:
            expert_ids = torch.arange(
                int(gu_buf.shape[0]), dtype=torch.long, device=device
            )
            self._dequant_id_ranges[rank] = expert_ids
        flat_ids = route_ids_mb.reshape(-1)
        order = torch.argsort(flat_ids)
        sorted_experts = flat_ids[order].contiguous()
        token_ids = (
            torch.arange(count, dtype=torch.long, device=device)
            .view(-1, 1)
            .expand(count, top_k)
            .reshape(-1)
        )
        sorted_tokens = token_ids[order].contiguous()
        sorted_weights = (
            route_weights_mb.reshape(-1)[order].float().contiguous()
        )
        offs = torch.searchsorted(
            sorted_experts, expert_ids, right=True
        ).to(torch.int32)
        x_sorted = inputs.index_select(0, sorted_tokens)
        gu = torch._grouped_mm(
            x_sorted, gu_buf.transpose(1, 2), offs=offs
        )
        gate, up = gu.chunk(2, dim=-1)
        if float(limit) > 0.0:
            # 与 csrc projection_gate_up_activation 的 swiglu 分支一致：
            # gate 截上限、up 对称截断后再走激活（DSV4 swiglu_limit=10）。
            gate = gate.clamp(max=float(limit))
            up = up.clamp(min=-float(limit), max=float(limit))
        activated = activate_gate_up(
            gate,
            up,
            activation=activation,
            situ_beta=float(activation_beta),
            situ_linear_beta=(
                None
                if activation_linear_beta is None
                else float(activation_linear_beta)
            ),
        )
        down = torch._grouped_mm(
            activated.contiguous(), down_buf.transpose(1, 2), offs=offs
        )
        result.zero_()
        result.index_add_(
            0,
            sorted_tokens,
            down.float() * sorted_weights.unsqueeze(1),
        )
        return result

    def _run_rows_packed_grouped_rank(
        self,
        layer: int,
        rank: int,
        inputs: torch.Tensor,
        route_ids_mb: torch.Tensor,
        route_weights_mb: torch.Tensor,
        hidden_workspace: torch.Tensor,
        result: torch.Tensor,
        count: int,
        top_k: int,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float,
    ) -> torch.Tensor:
        """Run a complete AMD Prefill micro-batch from resident packed data."""

        from .ops import packed_moe_topk_grouped

        device = self.devices[rank]
        flat_ids = route_ids_mb.reshape(-1)
        token_ids = (
            torch.arange(count, dtype=torch.long, device=device)
            .view(-1, 1)
            .expand(count, top_k)
            .reshape(-1)
        )
        flat_weights = route_weights_mb.reshape(-1).float()
        local_mask = self._grouped_local_mask(layer, rank)
        local_positions = local_mask.index_select(0, flat_ids).nonzero(
            as_tuple=False
        ).reshape(-1)
        result.zero_()
        if local_positions.numel() == 0:
            return result
        local_experts = flat_ids.index_select(0, local_positions)
        order = torch.argsort(local_experts)
        sorted_positions = local_positions.index_select(0, order)
        sorted_experts = local_experts.index_select(0, order).contiguous()
        group_experts, group_counts = torch.unique_consecutive(
            sorted_experts,
            return_counts=True,
        )
        group_offsets = torch.empty(
            int(group_experts.numel()) + 1,
            dtype=torch.int32,
            device=device,
        )
        group_offsets[0] = 0
        group_offsets[1:].copy_(torch.cumsum(group_counts, dim=0).to(
            torch.int32
        ))
        sorted_tokens = token_ids.index_select(
            0, sorted_positions
        ).contiguous()
        sorted_weights = flat_weights.index_select(
            0, sorted_positions
        ).contiguous()
        return packed_moe_topk_grouped(
            inputs,
            sorted_tokens,
            group_experts.contiguous(),
            group_offsets,
            sorted_weights,
            self._metadata[layer][rank],
            activation=activation,
            activation_beta=float(activation_beta),
            activation_linear_beta=(
                0.0
                if activation_linear_beta is None
                else float(activation_linear_beta)
            ),
            limit=float(limit),
            hidden_workspace=hidden_workspace[: int(sorted_tokens.numel())],
            result=result,
        )

    def run_rows(
        self,
        layer: int,
        value: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float = 0.0,
        prefill_default: int = 4096,
    ) -> torch.Tensor:
        """Run exact packed Top-K MoE for a prefill row batch.

        The public CUDA operator consumes ``[N,K]`` routes directly.  Every
        tensor/expert rank executes its resident packed shard with those exact
        routes, then one public Row-TP collective reduces the FP32 partials.
        A bounded micro-batch prevents activation scratch from growing with
        context length and reduces Python/CUDA submissions from N per layer
        to ceil(N/micro_batch).  No packed index or full expert is expanded.
        """
        if not self.active:
            raise RuntimeError("packed experts are not ready")
        if value.ndim != 2 or route_ids.ndim != 2:
            raise ValueError("packed prefill expects value [N,D], routes [N,K]")
        if route_weights.shape != route_ids.shape:
            raise ValueError("packed prefill route weights must match routes")
        rows = int(value.shape[0])
        top_k = int(route_ids.shape[1])
        if rows <= 0 or top_k <= 0:
            raise ValueError("packed prefill requires non-empty rows and routes")
        # The full-resident TP path uses a 4096-row scratch by default;
        # callers may override CCCP_PREFILL_MOE_BATCH after measuring a larger
        # workspace on the target GPU.  8192 raises the MLA prefill score
        # tensor past the per-GPU headroom on H20 TP8 (OOM at ~13k tokens);
        # 2048 is stable but caps throughput ~65% lower than 4096.
        micro_batch = min(
            prefill_moe_batch_size(default=int(prefill_default)),
            rows,
        )
        owner = self.plan.owner_by_layer[layer]
        owner_device = self.devices[owner]
        intermediate = int(self.store.cfg["moe_inter"])
        hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        local_intermediate = (
            intermediate // self.tensor_group_size
            if self.parallelism in {"tensor", "hybrid"}
            else intermediate
        )

        def workspace(rank: int) -> _PackedPrefillWorkspace:
            cached = self._prefill_workspaces.get(rank)
            if (
                cached is not None
                and cached.capacity >= micro_batch
                and cached.top_k == top_k
                and cached.hidden.shape[1] == 2 * local_intermediate
                and cached.result.shape[1] == hidden
            ):
                return cached
            device = self.devices[rank]
            with torch.cuda.device(device):
                cached = _PackedPrefillWorkspace(
                    capacity=micro_batch,
                    top_k=top_k,
                    inputs=torch.empty(
                        micro_batch,
                        hidden,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    route_ids=torch.empty(
                        micro_batch,
                        top_k,
                        dtype=torch.long,
                        device=device,
                    ),
                    route_weights=torch.empty(
                        micro_batch,
                        top_k,
                        dtype=torch.float32,
                        device=device,
                    ),
                    hidden=torch.empty(
                        micro_batch * top_k,
                        2 * local_intermediate,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    output=torch.empty(
                        micro_batch * top_k,
                        hidden,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    result=torch.empty(
                        micro_batch,
                        hidden,
                        dtype=torch.float32,
                        device=device,
                    ),
                )
            self._prefill_workspaces[rank] = cached
            return cached

        result = torch.empty(
            rows,
            hidden,
            dtype=torch.float32,
            device=owner_device,
        )

        # Packed multi-token execution is never allowed to fall back to the
        # width-one decode/GEMV operator. NVIDIA keeps the measured exact
        # dequant + cuBLAS grouped-GEMM path. Windows ROCm cannot safely call
        # PyTorch's private grouped-mm and uses CCCP's compiled packed grouped
        # kernel for the same complete micro-batch instead.
        native8_prefill = self._native8_layer_supported(layer)
        dequant_prefill = bool(
            not native8_prefill and self._dequant_supported(layer)
        )
        metadata_rows = int(self._metadata[layer][owner].shape[0])
        # NVIDIA retains its faster measured dequant+cuBLAS path whenever the
        # complete scratch image fits.  When it does not (for example Kimi
        # TP4 with 896 experts/rank), the same public packed grouped kernel is
        # the exact no-expansion fallback. HIP always uses the public packed
        # operator because torch._grouped_mm is unsafe on Windows ROCm.
        packed_grouped_prefill = _select_packed_grouped_prefill(
            self.store.man,
            layer,
            metadata_rows=metadata_rows,
            dequant_prefill=bool(dequant_prefill or native8_prefill),
            hip_runtime=torch.version.hip is not None,
        )
        if native8_prefill:
            self.prefill_executor = (
                "cuda.vq-to-e4m3-scaled-grouped-gemm."
                f"{self._native8_grouped_backend}"
            )
        elif packed_grouped_prefill:
            self.prefill_executor = "cuda.packed-moe-grouped-fused"
            print(
                "[cccp-prefill] "
                f"executor={self.prefill_executor}; "
                f"layer={layer}; token batch={rows}; "
                "resident_experts=GPU-only; H2D=0",
                flush=True,
            )
        elif dequant_prefill:
            self.prefill_executor = "cuda.dequant-grouped-gemm"
            self._prefill_dequant_launch(layer, intermediate, hidden)

        # The width-one path deliberately remains free of collectives.  It is
        # the exact fast path used by the measured 8192-token TP1 prefill.
        if len(self.devices) == 1:
            if native8_prefill or dequant_prefill or packed_grouped_prefill:
                for start in range(0, rows, micro_batch):
                    stop = min(rows, start + micro_batch)
                    count = stop - start
                    with torch.cuda.device(owner_device):
                        inputs = value[start:stop].to(
                            torch.bfloat16
                        ).contiguous()
                        ids = route_ids[start:stop].contiguous()
                        weights = route_weights[start:stop].float().contiguous()
                        if native8_prefill:
                            self._run_rows_native8_rank(
                                layer, owner, inputs, ids, weights,
                                result[start:stop], count, top_k, activation,
                                activation_beta, activation_linear_beta, limit,
                            )
                        elif packed_grouped_prefill:
                            cached = workspace(owner)
                            self._run_rows_packed_grouped_rank(
                                layer, owner, inputs, ids, weights,
                                cached.hidden, result[start:stop], count,
                                top_k, activation, activation_beta,
                                activation_linear_beta, limit,
                            )
                        else:
                            self._run_rows_dequant_rank(
                                layer, owner, inputs, ids, weights,
                                result[start:stop], count, top_k, activation,
                                activation_beta, activation_linear_beta, limit,
                            )
                    self.prefill_batch_submissions += 1
                    self.prefill_batch_max = max(
                        self.prefill_batch_max,
                        count,
                    )
                self.prefill_batch_rows += rows
                self.hits += int(route_ids.numel())
                return result
            raise RuntimeError(
                "packed GPU Prefill requires a registered grouped executor; "
                "decode GEMV and compact grouped fallbacks are forbidden"
            )

        if self.parallelism == "pipeline":
            raise RuntimeError(
                "multi-rank row batching requires sharded packed experts"
            )
        if not self._streams or not self._source_events:
            raise RuntimeError("packed prefill TP streams are unavailable")

        from .fusedext import tp_all_rank_reduce_from_events_fused

        rank_order = tuple(range(len(self.devices)))
        rank_workspaces = tuple(workspace(rank) for rank in rank_order)
        source_ready = self._source_events[layer]
        done_events = [
            self._done_events[rank][layer] for rank in rank_order
        ]
        output_event = self._output_events[owner][layer]
        with torch.cuda.device(owner_device):
            owner_stream = torch.cuda.current_stream(owner_device)
            for start in range(0, rows, micro_batch):
                stop = min(rows, start + micro_batch)
                count = stop - start
                source_values = value[start:stop].to(
                    torch.bfloat16
                ).contiguous()
                source_ids = route_ids[start:stop].contiguous()
                source_weights = route_weights[start:stop].float().contiguous()
                source_ready.record(owner_stream)
                contributions = []
                for rank, cached in zip(rank_order, rank_workspaces):
                    device = self.devices[rank]
                    stream = self._streams[rank]
                    with (
                        torch.cuda.device(device),
                        torch.cuda.stream(stream),
                    ):
                        stream.wait_event(source_ready)
                        cached.inputs[:count].copy_(
                            source_values,
                            non_blocking=True,
                        )
                        cached.route_ids[:count].copy_(
                            source_ids,
                            non_blocking=True,
                        )
                        cached.route_weights[:count].copy_(
                            source_weights,
                            non_blocking=True,
                        )
                        if native8_prefill:
                            partial = self._run_rows_native8_rank(
                                layer,
                                rank,
                                cached.inputs[:count],
                                cached.route_ids[:count],
                                cached.route_weights[:count],
                                cached.result[:count],
                                count,
                                top_k,
                                activation,
                                activation_beta,
                                activation_linear_beta,
                                limit,
                            )
                        elif packed_grouped_prefill:
                            partial = self._run_rows_packed_grouped_rank(
                                layer,
                                rank,
                                cached.inputs[:count],
                                cached.route_ids[:count],
                                cached.route_weights[:count],
                                cached.hidden,
                                cached.result[:count],
                                count,
                                top_k,
                                activation,
                                activation_beta,
                                activation_linear_beta,
                                limit,
                            )
                        elif dequant_prefill:
                            partial = self._run_rows_dequant_rank(
                                layer,
                                rank,
                                cached.inputs[:count],
                                cached.route_ids[:count],
                                cached.route_weights[:count],
                                cached.result[:count],
                                count,
                                top_k,
                                activation,
                                activation_beta,
                                activation_linear_beta,
                                limit,
                            )
                        else:
                            raise RuntimeError(
                                "packed Row-TP Prefill requires a registered "
                                "grouped executor; decode GEMV is forbidden"
                            )
                        contributions.append(partial)
                        done_events[rank].record(stream)
                reduced = tp_all_rank_reduce_from_events_fused(
                    contributions,
                    done_events,
                    [result[start:stop]],
                    [output_event],
                )
                if reduced is None:
                    raise RuntimeError(
                        "packed prefill Row-TP reduction was rejected"
                    )
                self.prefill_batch_submissions += 1
                self.prefill_batch_max = max(
                    self.prefill_batch_max,
                    count,
                )
        self.prefill_batch_rows += rows
        self.hits += int(route_ids.numel())
        return result

    def run_rows_replicated(
        self,
        layer: int,
        values: tuple[torch.Tensor, ...] | list[torch.Tensor],
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float = 0.0,
        prefill_default: int = 4096,
    ) -> tuple[torch.Tensor, ...]:
        """Run a row-batched packed MoE from rank-local latent replicas.

        ``run_rows`` is retained for callers that own one complete latent
        matrix.  Long-context TP prefill already has the Row-TP partial on
        every rank, however, so sending that matrix back to one owner and
        broadcasting it again is wasted bandwidth.  This public executor
        consumes one replicated ``values[rank]`` tensor per rank, copies only
        the small ``[rows, top_k]`` route tensors, executes the resident
        packed shard locally, and publishes one reduced result per rank.

        The method is model-independent: callers provide packed metadata,
        projection activation and rank-local tensors; no model name or
        owner-specific policy enters dispatch.  ``pipeline`` parallelism is
        intentionally rejected because it does not have a tensor-sharded
        expert on every rank.
        """
        if not self.active:
            raise RuntimeError("packed experts are not ready")
        if self.parallelism == "pipeline":
            if len(self.devices) == 1:
                # TP1/DSV4 has no row collective.  Keep the public API useful
                # for that layout by delegating to the proven single-rank
                # implementation instead of forcing callers to special-case
                # the model.
                return (
                    self.run_rows(
                        layer,
                        values[0],
                        route_ids,
                        route_weights,
                        activation=activation,
                        activation_beta=activation_beta,
                        activation_linear_beta=activation_linear_beta,
                        limit=limit,
                        prefill_default=prefill_default,
                    ),
                )
            raise RuntimeError(
                "replicated packed prefill requires sharded packed experts"
            )
        if len(values) != len(self.devices):
            raise ValueError("replicated packed prefill values must match TP")
        if route_ids.ndim != 2 or route_weights.shape != route_ids.shape:
            raise ValueError(
                "replicated packed prefill routes must be [N,K] and match"
            )
        rows = int(route_ids.shape[0])
        top_k = int(route_ids.shape[1])
        if rows <= 0 or top_k <= 0:
            raise ValueError("replicated packed prefill requires non-empty rows")
        reference = values[0]
        if (
            reference.ndim != 2
            or reference.shape[0] != rows
            or reference.dtype != torch.bfloat16
            or not reference.is_contiguous()
        ):
            raise ValueError(
                "replicated packed prefill values must be contiguous BF16 [N,D]"
            )
        hidden = int(reference.shape[1])
        for rank, (device, value) in enumerate(zip(self.devices, values)):
            if (
                value.device != device
                or value.shape != reference.shape
                or value.dtype != torch.bfloat16
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"replicated packed prefill value {rank} does not match TP"
                )
        owner = self.plan.owner_by_layer[layer]
        route_device = self.devices[owner]
        if (
            route_ids.device != route_device
            or route_weights.device != route_device
        ):
            raise ValueError(
                "replicated packed prefill routes must be on the layer owner"
            )

        micro_batch = min(
            prefill_moe_batch_size(default=int(prefill_default)),
            rows,
        )
        intermediate = int(self.store.cfg["moe_inter"])
        local_intermediate = (
            intermediate // self.tensor_group_size
            if self.parallelism in {"tensor", "hybrid"}
            else intermediate
        )

        def workspace(rank: int) -> _PackedPrefillWorkspace:
            cached = self._prefill_workspaces.get(rank)
            if (
                cached is not None
                and cached.capacity >= micro_batch
                and cached.top_k == top_k
                and cached.hidden.shape[1] == 2 * local_intermediate
                and cached.result.shape[1] == hidden
            ):
                return cached
            device = self.devices[rank]
            with torch.cuda.device(device):
                cached = _PackedPrefillWorkspace(
                    capacity=micro_batch,
                    top_k=top_k,
                    inputs=torch.empty(
                        micro_batch,
                        hidden,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    route_ids=torch.empty(
                        micro_batch,
                        top_k,
                        dtype=torch.long,
                        device=device,
                    ),
                    route_weights=torch.empty(
                        micro_batch,
                        top_k,
                        dtype=torch.float32,
                        device=device,
                    ),
                    hidden=torch.empty(
                        micro_batch * top_k,
                        2 * local_intermediate,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    output=torch.empty(
                        micro_batch * top_k,
                        hidden,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    result=torch.empty(
                        micro_batch,
                        hidden,
                        dtype=torch.float32,
                        device=device,
                    ),
                )
            self._prefill_workspaces[rank] = cached
            return cached

        from .fusedext import tp_all_rank_reduce_from_events_fused

        capability = self.store.man.projection_operator_capability(layer)
        dequant_prefill = bool(
            len(capability.get("packed_formats", ())) == 3
            and self.store.man.projection_vq
            and self._dequant_supported(layer)
        )
        if not dequant_prefill:
            raise RuntimeError(
                "replicated packed CUDA Prefill requires exact dequant + "
                "grouped GEMM; decode GEMV fallback is forbidden"
            )
        self.prefill_executor = "cuda.dequant-grouped-gemm"
        self._prefill_dequant_launch(layer, intermediate, hidden)
        outputs = [
            torch.empty(
                rows,
                hidden,
                dtype=torch.float32,
                device=device,
            )
            for device in self.devices
        ]
        rank_workspaces = tuple(
            workspace(rank) for rank in range(len(self.devices))
        )
        done_events = [
            self._done_events[rank][layer]
            for rank in range(len(self.devices))
        ]
        output_events = [
            self._output_events[rank][layer]
            for rank in range(len(self.devices))
        ]
        # Each source replica may have been produced on a different current
        # stream by the preceding Row-TP reduction.  Record one local event
        # and make the packed streams wait before reading it.
        source_events = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                source_events.append(event)
        owner_device = self.devices[owner]
        with torch.cuda.device(owner_device):
            route_ready = torch.cuda.Event()
            route_ready.record(torch.cuda.current_stream(owner_device))

        for start in range(0, rows, micro_batch):
            stop = min(rows, start + micro_batch)
            count = stop - start
            contributions = []
            for rank, cached in enumerate(rank_workspaces):
                device = self.devices[rank]
                stream = self._streams[rank]
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    # The previous reduction must no longer read this rank's
                    # scratch before its next result/kernel is overwritten.
                    stream.wait_event(output_events[rank])
                    stream.wait_event(source_events[rank])
                    stream.wait_event(route_ready)
                    cached.inputs[:count].copy_(
                        values[rank][start:stop],
                        non_blocking=True,
                    )
                    cached.route_ids[:count].copy_(
                        route_ids[start:stop],
                        non_blocking=True,
                    )
                    cached.route_weights[:count].copy_(
                        route_weights[start:stop],
                        non_blocking=True,
                    )
                    partial = self._run_rows_dequant_rank(
                        layer,
                        rank,
                        cached.inputs[:count],
                        cached.route_ids[:count],
                        cached.route_weights[:count],
                        cached.result[:count],
                        count,
                        top_k,
                        activation,
                        activation_beta,
                        activation_linear_beta,
                        limit,
                    )
                    contributions.append(partial)
                    done_events[rank].record(stream)
            reduced = tp_all_rank_reduce_from_events_fused(
                contributions,
                done_events,
                [output[start:stop] for output in outputs],
                output_events,
            )
            if reduced is None:
                raise RuntimeError(
                    "replicated packed prefill Row-TP reduction was rejected"
                )
            self.prefill_batch_submissions += 1
            self.prefill_batch_max = max(self.prefill_batch_max, count)

        # The fused reducer enqueues one kernel per destination device.  Make
        # the caller's ordinary stream observe that completion before it
        # slices the returned replicas for the next TP projection.
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                torch.cuda.current_stream(device).wait_event(
                    output_events[rank]
                )
        self.prefill_batch_rows += rows
        self.hits += int(route_ids.numel())
        return tuple(outputs)

    def run(
        self,
        layer: int,
        value: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float = 0.0,
    ) -> torch.Tensor:
        if not self.active:
            raise RuntimeError("packed experts are not ready")
        if value.ndim != 2 or int(value.shape[0]) != 1:
            raise RuntimeError(
                "packed decode GEMV accepts exactly one token; use run_rows "
                "for multi-token Prefill"
            )
        from .ops import packed_moe_topk

        if self.parallelism == "pipeline":
            rank = self.plan.owner_by_layer[layer]
            device = self.devices[rank]
            hidden, output, result = self._workspaces[rank]
            with torch.cuda.device(device):
                if self._compact_decode_enabled:
                    return self._run_compact_decode_rank(
                        layer,
                        rank,
                        value,
                        route_ids,
                        route_weights,
                        activation=activation,
                        activation_beta=activation_beta,
                        activation_linear_beta=activation_linear_beta,
                        limit=limit,
                        result=result,
                    )
                return packed_moe_topk(
                    value.to(torch.bfloat16),
                    route_ids.reshape(-1),
                    route_weights.reshape(-1),
                    self._metadata[layer][rank],
                    activation=activation,
                    activation_beta=float(activation_beta),
                    activation_linear_beta=(
                        0.0
                        if activation_linear_beta is None
                        else float(activation_linear_beta)
                    ),
                    hidden_workspace=hidden,
                    output_workspace=output,
                    result=result,
                    grouped_prefix=-1,
                    **self.store.man.projection_operator_capability(
                        layer
                    ),
                    limit=float(limit),
                )

        owner = self.plan.owner_by_layer[layer]
        owner_device = self.devices[owner]
        graph_batch = self._graph_batches.get(layer)
        if graph_batch is not None:
            from .fusedext import expert_dispatch_pack_fused

            local_layer = layer - self.plan.ranges[owner][0]
            with torch.cuda.device(owner_device):
                dispatched = expert_dispatch_pack_fused(
                    value,
                    route_ids.reshape(-1),
                    route_weights.reshape(-1),
                    self._source_inputs[owner],
                    self._source_ids[owner],
                    self._source_weights[owner],
                )
                if not dispatched:
                    raise RuntimeError(
                        "packed MoE source publication was rejected"
                    )
                rank_order = self._graph_rank_order[layer]
                contributions = [
                    self._return_buffers[owner][rank, local_layer]
                    for rank in rank_order
                ]
                reduced = graph_batch.launch_reduce(
                    contributions,
                    self._zero_buffers[owner],
                )
            self.hits += route_ids.numel()
            return reduced

        source_ready = self._source_events[layer]
        source_ready.record(torch.cuda.current_stream(owner_device))
        local_layer = layer - self.plan.ranges[owner][0]
        for rank, device in enumerate(self.devices):
            stream = self._streams[rank]
            hidden, output, result = self._workspaces[rank]
            with (
                torch.cuda.device(device),
                torch.cuda.stream(stream),
            ):
                stream.wait_event(source_ready)
                self._routed_inputs[rank].copy_(
                    value,
                    non_blocking=True,
                )
                self._routed_ids[rank].copy_(
                    route_ids.reshape(-1),
                    non_blocking=True,
                )
                self._routed_weights[rank].copy_(
                    route_weights.reshape(-1),
                    non_blocking=True,
                )
                if self._compact_decode_enabled:
                    partial = self._run_compact_decode_rank(
                        layer,
                        rank,
                        self._routed_inputs[rank],
                        self._routed_ids[rank],
                        self._routed_weights[rank],
                        activation=activation,
                        activation_beta=activation_beta,
                        activation_linear_beta=activation_linear_beta,
                        limit=limit,
                        result=result,
                    )
                else:
                    partial = packed_moe_topk(
                        self._routed_inputs[rank],
                        self._routed_ids[rank],
                        self._routed_weights[rank],
                        self._metadata[layer][rank],
                        activation=activation,
                        activation_beta=float(activation_beta),
                        activation_linear_beta=(
                            0.0
                            if activation_linear_beta is None
                            else float(activation_linear_beta)
                        ),
                        hidden_workspace=hidden,
                        output_workspace=output,
                        result=result,
                        grouped_prefix=-1,
                        **self.store.man.projection_operator_capability(
                            layer
                        ),
                        limit=float(limit),
                    )
                self._return_buffers[owner][
                    rank, local_layer
                ].copy_(
                    partial,
                    non_blocking=True,
                )
                self._done_events[rank][layer].record(stream)
        self.hits += route_ids.numel()
        with torch.cuda.device(owner_device):
            owner_stream = torch.cuda.current_stream(owner_device)
            for rank in range(len(self.devices)):
                owner_stream.wait_event(
                    self._done_events[rank][layer]
                )
            contributions = self._return_buffers[owner][
                :, local_layer
            ]
            reduced = self._reduce_buffers[owner][local_layer]
            reduced.copy_(contributions[0])
            for rank in range(1, len(self.devices)):
                reduced.add_(contributions[rank])
        return reduced


__all__ = [
    "ResidentRoutedVQPool",
    "RoutedVQLayoutPlan",
    "build_primary_dense_packed_plan",
    "build_routed_vq_layer_plan",
]

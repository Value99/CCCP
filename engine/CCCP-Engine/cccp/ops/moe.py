"""Model-independent fixed-address MoE scheduling primitives."""

from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
from dataclasses import dataclass
import os

import torch

from ..kernels import BlockFP8Weight, ProjectionGroup


@dataclass(frozen=True)
class RoutedVQPoolPlan:
    """Model-independent concrete storage plan for routed VQ experts."""

    kind: str
    packed_device_pool: bool = False
    packed_full_gpu: bool = False
    parallelism: str | None = None


@dataclass
class RoutedVQRuntime:
    """Concrete public routed-codebook runtime selected from capabilities."""

    executor: RoutedVQExecutor
    plan: RoutedVQPoolPlan
    topology_plan: object | None = None


@dataclass(frozen=True)
class RoutedVQResidencyState:
    """Result of model-independent codebook residency initialization."""

    resident_all: bool
    native_layers: int = 0
    gpu_arena_gb: float = 0.0


@dataclass(frozen=True)
class RoutedVQCapabilities:
    """Typed, model-independent capabilities of one routed VQ backend."""

    parallelism: str | None = None
    hidden_mode: bool = False
    device_routed: bool = False
    prefetch_default: bool = True
    compact_full_resident: bool = False
    full_resident: bool = False
    fixed_token_graph_candidate: bool = False
    fixed_token_graph_capable: bool = False
    profile_hot_cache_enabled: bool = False
    speculative_prefetch: bool = True
    layer_prefetch_only: bool = False
    host_mapped: bool = False
    supports_vram_watch: bool = True


@dataclass(frozen=True)
class RoutedVQRuntimeStats:
    """Stable diagnostics shared by every routed-codebook model adapter."""

    hits: int = 0
    misses: int = 0
    bytes: int = 0
    gpu_storage_bytes: int = 0
    gpu_arena_bytes: int = 0
    host_expert_bytes: int = 0
    host_pinned_bytes: int = 0
    uploaded_bytes: int = 0
    transfer_seconds: float = 0.0
    prefetch_hits: int = 0
    route_plan_hits: int = 0
    route_plan_misses: int = 0
    device_route_lookups: int = 0
    device_route_full_hits: int = 0
    device_route_fallbacks: int = 0
    h2d_batch_submissions: int = 0
    h2d_batch_copies: int = 0
    h2d_batch_fallbacks: int = 0
    native_packed_hits: int = 0
    native_packed_fallbacks: int = 0
    cpu_compile_mode: str = "off"
    compiled_source_bytes: int = 0
    compiled_index_bytes: int = 0
    expanded_index_bytes: int = 0


def plan_routed_vq_pool(
    manifest,
    *,
    device: torch.device,
    tp_size: int,
    full_gpu_requested: bool,
    tensor_hybrid_requested: bool,
    layer_graph_requested: bool,
    parallelism_requested: str | None = None,
) -> RoutedVQPoolPlan:
    """Select routed-codebook storage only from format and capabilities.

    Model adapters are intentionally absent from the inputs. They provide
    topology metadata later, but cannot choose CPU/GPU cache algorithms or
    concrete pool classes.
    """
    resolved = torch.device(device)
    tp_size = int(tp_size)
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    requested_parallelism = (
        str(parallelism_requested).strip().lower()
        if parallelism_requested is not None
        else ""
    )
    if requested_parallelism not in {"", "pipeline", "expert", "tensor"}:
        raise ValueError(
            "routed VQ parallelism must be pipeline, expert, or tensor"
        )
    projection_vq = bool(getattr(manifest, "projection_vq", False))
    packed_cpu = bool(
        getattr(manifest, "packed_expert_vq", projection_vq)
    )
    expert_codebook_vq = bool(
        getattr(manifest, "expert_codebook_vq", projection_vq or packed_cpu)
    )
    if resolved.type == "cpu":
        return RoutedVQPoolPlan(
            (
                "packed_cpu"
                if packed_cpu
                else "codebook_combined" if expert_codebook_vq else "legacy"
            )
        )
    if not projection_vq:
        return RoutedVQPoolPlan(
            (
                "codebook_combined_parallel"
                if expert_codebook_vq and tp_size > 1
                else "codebook_combined"
                if expert_codebook_vq
                else "legacy_parallel"
                if tp_size > 1
                else "legacy"
            )
        )
    if tp_size > 1 and tensor_hybrid_requested:
        return RoutedVQPoolPlan(
            "packed_tensor_hybrid",
            packed_device_pool=True,
        )
    if tp_size > 1 or full_gpu_requested:
        return RoutedVQPoolPlan(
            "packed_full",
            packed_device_pool=True,
            packed_full_gpu=True,
            parallelism=(
                requested_parallelism
                or (
                    "tensor"
                    if layer_graph_requested or tp_size > 1
                    else "pipeline"
                )
            ),
        )
    return RoutedVQPoolPlan(
        "packed_hybrid",
        packed_device_pool=True,
    )


def _runtime_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def build_routed_vq_topology_plan(
    store,
    tp_size: int,
    *,
    primary_dense: bool = False,
):
    """Build manifest-derived routed-VQ layout through the public API.

    ``primary_dense`` describes a tensor topology, not a model family.  The
    concrete layout implementation remains private to the shared runtime so
    architecture adapters cannot depend on resident/cache backend classes.
    """
    from ..routed_vq_backend import (
        build_primary_dense_packed_plan,
        build_routed_vq_layer_plan,
    )

    builder = (
        build_primary_dense_packed_plan
        if primary_dense
        else build_routed_vq_layer_plan
    )
    return builder(store, int(tp_size))


def create_routed_vq_runtime(
    store,
    *,
    device: str | torch.device,
    cache_gb: float,
    vram_cache_gb: float,
    tp_size: int = 1,
    devices: tuple[torch.device, ...] | None = None,
    startup_gpu_reserve_bytes: int = 0,
    pin_gb: float = 0.0,
    topology_plan_factory=None,
    full_gpu_requested: bool | None = None,
    tensor_hybrid_requested: bool | None = None,
    layer_graph_requested: bool | None = None,
) -> RoutedVQRuntime:
    """Create the sole routed-codebook pool/executor pair.

    The caller supplies only model topology through ``topology_plan_factory``.
    Storage format, residency, CPU/CUDA backend and cache implementation are
    selected here from the manifest and generic runtime capabilities.
    """
    resolved = torch.device(device)
    tp_size = int(tp_size)
    normalized_devices = tuple(
        torch.device(item) for item in (
            devices
            if devices is not None
            else (resolved,)
        )
    )
    if len(normalized_devices) != tp_size:
        raise ValueError("routed VQ device count must match tp_size")
    full_gpu = (
        _runtime_flag("CCCP_PACKED_FULL_GPU")
        if full_gpu_requested is None
        else bool(full_gpu_requested)
    )
    tensor_hybrid = (
        _runtime_flag("CCCP_TP_PACKED_HYBRID")
        if tensor_hybrid_requested is None
        else bool(tensor_hybrid_requested)
    )
    layer_graph = (
        _runtime_flag("CCCP_SINGLE_GPU_LAYER_GRAPH")
        if layer_graph_requested is None
        else bool(layer_graph_requested)
    )
    plan = plan_routed_vq_pool(
        store.man,
        device=resolved,
        tp_size=tp_size,
        full_gpu_requested=full_gpu,
        tensor_hybrid_requested=tensor_hybrid,
        layer_graph_requested=layer_graph,
        parallelism_requested=os.environ.get("CCCP_MOE_PARALLELISM"),
    )
    topology_plan = None
    if plan.kind in {"packed_full", "packed_tensor_hybrid"} or layer_graph:
        if topology_plan_factory is None:
            if plan.kind in {"packed_full", "packed_tensor_hybrid"}:
                from ..routed_vq_backend import (
                    build_primary_dense_packed_plan,
                )

                topology_plan = build_primary_dense_packed_plan(
                    store,
                    tp_size,
                )
        else:
            topology_plan = topology_plan_factory(store, tp_size)

    if plan.kind == "packed_cpu":
        from ..store import PackedCpuExpertPool

        pool = PackedCpuExpertPool(store, budget_gb=cache_gb)
    elif plan.kind == "packed_hybrid":
        from ..packed_hybrid import PackedHybridPool

        pool = PackedHybridPool(
            store,
            vram_cache_gb,
            device=resolved,
            ram_gb=cache_gb,
            startup_gpu_reserve_bytes=startup_gpu_reserve_bytes,
        )
    elif plan.kind == "packed_full":
        from ..routed_vq_backend import ResidentRoutedVQPool

        pool = ResidentRoutedVQPool(
            store,
            normalized_devices,
            topology_plan,
            parallelism=plan.parallelism or "pipeline",
        )
    elif plan.kind == "packed_tensor_hybrid":
        from ..routed_vq_tp_hybrid import TensorHybridRoutedVQPool

        pool = TensorHybridRoutedVQPool(
            store,
            normalized_devices,
            topology_plan,
            vram_cache_gb,
            ram_gb=cache_gb,
        )
    elif plan.kind in {"legacy_parallel", "codebook_combined_parallel"}:
        from ..expert_parallel import GpuResidentExpertParallel

        pool = GpuResidentExpertParallel(store, tp_size, resolved)
    else:
        from ..store import ExpertPool

        pool = ExpertPool(
            store,
            vram_cache_gb if resolved.type != "cpu" else cache_gb,
            device=str(resolved),
            ram_gb=(
                max(0.0, float(cache_gb) - float(pin_gb))
                if resolved.type != "cpu"
                else 0.0
            ),
            pin_gb=float(pin_gb),
        )
    return RoutedVQRuntime(
        executor=RoutedVQExecutor(pool),
        plan=plan,
        topology_plan=topology_plan,
    )


class RoutedVQExecutor:
    """Single public entry point for routed codebook execution.

    Architecture adapters provide topology-derived routes and activation
    parameters only.  This executor owns Decode/Prefill dispatch and the
    packed-pool phase lifecycle, so model files cannot grow private VQ
    dequantisation, GEMV/GEMM or cache-selection branches.
    """

    def __init__(self, pool) -> None:
        self._pool = pool

    @property
    def capabilities(self) -> RoutedVQCapabilities:
        """Return immutable backend capabilities without exposing its pool."""
        pool = self._pool
        return RoutedVQCapabilities(
            parallelism=getattr(pool, "parallelism", None),
            hidden_mode=bool(getattr(pool, "hidden_mode", False)),
            device_routed=bool(getattr(pool, "device_routed", False)),
            prefetch_default=bool(getattr(pool, "prefetch_default", True)),
            compact_full_resident=bool(
                getattr(pool, "compact_full_resident", False)
            ),
            full_resident=bool(getattr(pool, "full_resident", False)),
            fixed_token_graph_candidate=bool(
                getattr(pool, "fixed_token_graph_candidate", False)
            ),
            fixed_token_graph_capable=bool(
                getattr(pool, "fixed_token_graph_capable", False)
            ),
            profile_hot_cache_enabled=bool(
                getattr(pool, "profile_hot_cache_enabled", False)
            ),
            speculative_prefetch=bool(
                getattr(pool, "speculative_prefetch", True)
            ),
            layer_prefetch_only=bool(
                getattr(pool, "layer_prefetch_only", False)
            ),
            host_mapped=bool(getattr(pool, "host_mapped", False)),
            supports_vram_watch=bool(
                getattr(pool, "supports_vram_watch", True)
            ),
        )

    def stats(self) -> RoutedVQRuntimeStats:
        """Read common monotonic storage/cache diagnostics."""
        pool = self._pool
        stage = getattr(pool, "_stage", None)
        return RoutedVQRuntimeStats(
            hits=int(getattr(pool, "hits", 0)),
            misses=int(getattr(pool, "misses", getattr(pool, "miss", 0))),
            bytes=int(getattr(pool, "bytes", 0)),
            gpu_storage_bytes=int(getattr(pool, "gpu_storage_bytes", 0)),
            gpu_arena_bytes=int(getattr(pool, "gpu_arena_bytes", 0)),
            host_expert_bytes=int(getattr(pool, "host_expert_bytes", 0)),
            host_pinned_bytes=int(getattr(pool, "_host_pinned_bytes", 0)),
            uploaded_bytes=int(getattr(pool, "uploaded_bytes", 0)),
            transfer_seconds=float(getattr(pool, "transfer_seconds", 0.0)),
            prefetch_hits=int(getattr(pool, "prefetch_hits", 0)),
            route_plan_hits=int(getattr(pool, "route_plan_hits", 0)),
            route_plan_misses=int(getattr(pool, "route_plan_misses", 0)),
            device_route_lookups=int(
                getattr(pool, "device_route_lookups", 0)
            ),
            device_route_full_hits=int(
                getattr(pool, "device_route_full_hits", 0)
            ),
            device_route_fallbacks=int(
                getattr(pool, "device_route_fallbacks", 0)
            ),
            h2d_batch_submissions=int(
                getattr(stage, "batch_submissions", 0)
            ),
            h2d_batch_copies=int(getattr(stage, "batch_copies", 0)),
            h2d_batch_fallbacks=int(
                getattr(stage, "batch_fallbacks", 0)
            ),
            native_packed_hits=int(getattr(pool, "native_hits", 0)),
            native_packed_fallbacks=int(
                getattr(pool, "native_fallbacks", 0)
            ),
            cpu_compile_mode=str(getattr(pool, "cpu_compile_mode", "off")),
            compiled_source_bytes=int(
                getattr(pool, "compiled_source_bytes", 0)
            ),
            compiled_index_bytes=int(
                getattr(pool, "compiled_index_bytes", 0)
            ),
            expanded_index_bytes=int(
                getattr(pool, "expanded_index_bytes", 0)
            ),
        )

    def profile_counters(self) -> dict[str, float | int]:
        """Return the common monotonic counters used by stage profilers."""
        stats = self.stats()
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "prefetch_hits": stats.prefetch_hits,
            "uploaded_bytes": stats.uploaded_bytes,
            "transfer_seconds": stats.transfer_seconds,
            "route_plan_hits": stats.route_plan_hits,
            "route_plan_misses": stats.route_plan_misses,
            "device_route_lookups": stats.device_route_lookups,
            "device_route_full_hits": stats.device_route_full_hits,
            "device_route_fallbacks": stats.device_route_fallbacks,
            "h2d_batch_submissions": stats.h2d_batch_submissions,
            "h2d_batch_copies": stats.h2d_batch_copies,
            "h2d_batch_fallbacks": stats.h2d_batch_fallbacks,
            "native_packed_hits": stats.native_packed_hits,
            "native_packed_fallbacks": stats.native_packed_fallbacks,
        }

    @property
    def devices(self) -> tuple[torch.device, ...]:
        devices = getattr(self._pool, "devices", None)
        if devices is None:
            device = getattr(self._pool, "device", torch.device("cpu"))
            return (torch.device(device),)
        return tuple(torch.device(device) for device in devices)

    @property
    def fixed_graph_generation(self) -> int:
        return int(getattr(self._pool, "fixed_graph_generation", -1))

    def resident_entry(self, layer: int, expert: int):
        entries = getattr(self._pool, "pinned", None)
        return entries.get((int(layer), int(expert))) if entries is not None else None

    def resident_entries(self, layer: int, expert_count: int) -> tuple:
        return tuple(
            self.resident_entry(layer, expert)
            for expert in range(int(expert_count))
        )

    def native_layer(self, layer: int):
        operation = getattr(self._pool, "native_layer", None)
        return operation(int(layer)) if callable(operation) else None

    def bind_hidden_inputs(self, *args, **kwargs) -> None:
        operation = getattr(self._pool, "bind_hidden_inputs", None)
        if not callable(operation):
            raise RuntimeError("public routed VQ backend cannot bind hidden inputs")
        operation(*args, **kwargs)

    def output_hidden(self, layer: int):
        operation = getattr(self._pool, "output_hidden", None)
        if not callable(operation):
            raise RuntimeError("public routed VQ backend has no hidden output")
        return operation(int(layer))

    def fixed_layer_plan(self, layer: int):
        operation = getattr(self._pool, "fixed_layer_plan", None)
        if not callable(operation):
            raise RuntimeError("public routed VQ backend has no fixed layer plan")
        return operation(int(layer))

    def fixed_layer_child_graphs(self, layer: int):
        operation = getattr(self._pool, "fixed_layer_child_graphs", None)
        if not callable(operation):
            raise RuntimeError("public routed VQ backend has no fixed child graphs")
        return operation(int(layer))

    def owner_for_layer(self, layer: int) -> int:
        plan = getattr(self._pool, "plan", None)
        owners = getattr(plan, "owner_by_layer", None)
        if owners is None:
            raise RuntimeError("public routed VQ backend has no owner topology")
        return int(owners[int(layer)])

    def last_expert_ids(self, layer: int):
        operation = getattr(self._pool, "last_expert_ids", None)
        return operation(int(layer)) if callable(operation) else None

    def begin_prefill(self) -> bool:
        operation = getattr(self._pool, "activate_prefill_arena", None)
        if callable(operation):
            operation()
            return True
        return False

    def end_prefill(self, *, restore_decode: bool = True) -> None:
        release = getattr(self._pool, "release_host_rows_workspace", None)
        if callable(release):
            release()
        if restore_decode:
            self.activate_decode()

    def activate_decode(self) -> bool:
        operation = getattr(self._pool, "activate_decode_arena", None)
        if callable(operation):
            operation()
            return True
        return False

    def prepare_fixed_token_graphs(self, **kwargs) -> bool:
        operation = getattr(self._pool, "prepare_fixed_token_graphs", None)
        return bool(operation(**kwargs)) if callable(operation) else False

    def refresh_mapped_cache(self) -> None:
        operation = getattr(self._pool, "refresh_mapped_cache", None)
        if callable(operation):
            operation()

    def compose_route_topk(self, *args, **kwargs):
        operation = getattr(self._pool, "compose_route_topk", None)
        if not callable(operation):
            raise RuntimeError(
                "public routed VQ backend cannot compose fixed routing"
            )
        return operation(*args, **kwargs)

    def prepare_prefill_layer(self, *args, **kwargs):
        operation = getattr(self._pool, "prepare_prefill_layer", None)
        return operation(*args, **kwargs) if callable(operation) else None

    def prefill_rows_available(self, layer: int) -> bool:
        operation = getattr(self._pool, "prefill_rows_available", None)
        return bool(operation(int(layer))) if callable(operation) else False

    def collect_transfer_timing(self, *, synchronize: bool = True) -> None:
        operation = getattr(self._pool, "collect_transfer_timing", None)
        if callable(operation):
            operation(synchronize=bool(synchronize))

    def start_vram_watch(
        self,
        *,
        low_gb: float,
        high_gb: float,
        quiet: bool,
    ):
        """Start the generic cache watcher without exposing mutable storage."""
        if not self.capabilities.supports_vram_watch:
            return None
        budget = getattr(self._pool, "budget", None)
        if budget is None:
            return None
        from ..vramwatch import VramWatch

        watch = VramWatch(
            self._pool,
            max_budget=budget,
            low_gb=float(low_gb),
            high_gb=float(high_gb),
            quiet=bool(quiet),
        )
        watch.start()
        return watch

    @property
    def cache_budget(self) -> int | None:
        value = getattr(self._pool, "budget", None)
        return None if value is None else int(value)

    @property
    def manages_per_rank_budget(self) -> bool:
        return bool(getattr(self._pool, "manages_per_rank_budget", False))

    @property
    def retains_store_ram_blobs(self) -> bool:
        return bool(getattr(self._pool, "retains_store_ram_blobs", False))

    def resize_gpu_arenas(self, budget: int) -> tuple[int, int] | None:
        operation = getattr(self._pool, "resize_gpu_arenas", None)
        return operation(int(budget)) if callable(operation) else None

    def trim_to(self, budget: int) -> None:
        operation = getattr(self._pool, "trim_to", None)
        if callable(operation):
            operation(int(budget))

    def record_cache_hits(
        self,
        count: int,
        *,
        fused_submissions: int = 0,
        graph_submissions: int = 0,
    ) -> None:
        count = int(count)
        self._pool.hits = int(getattr(self._pool, "hits", 0)) + count
        if hasattr(self._pool, "decode_fused_submissions"):
            self._pool.decode_fused_submissions = int(
                getattr(self._pool, "decode_fused_submissions", 0)
            ) + int(fused_submissions)
        if hasattr(self._pool, "decode_graph_submissions"):
            self._pool.decode_graph_submissions = int(
                getattr(self._pool, "decode_graph_submissions", 0)
            ) + int(graph_submissions)

    @property
    def parallelism(self) -> str | None:
        return self.capabilities.parallelism

    @property
    def hidden_mode(self) -> bool:
        return self.capabilities.hidden_mode

    @property
    def device_routed(self) -> bool:
        return self.capabilities.device_routed

    @property
    def prefetch_default(self) -> bool:
        return self.capabilities.prefetch_default

    @property
    def compact_full_resident(self) -> bool:
        return self.capabilities.compact_full_resident

    @property
    def fixed_token_graph_candidate(self) -> bool:
        return self.capabilities.fixed_token_graph_candidate

    @property
    def fixed_token_graph_capable(self) -> bool:
        return self.capabilities.fixed_token_graph_capable

    @property
    def profile_hot_cache_enabled(self) -> bool:
        return self.capabilities.profile_hot_cache_enabled

    @property
    def speculative_prefetch(self) -> bool:
        return self.capabilities.speculative_prefetch

    @property
    def layer_prefetch_only(self) -> bool:
        return self.capabilities.layer_prefetch_only

    @property
    def host_mapped(self) -> bool:
        return self.capabilities.host_mapped

    @property
    def startup_gpu_reserve_bytes(self) -> int:
        return int(getattr(self._pool, "startup_gpu_reserve_bytes", 0))

    @property
    def fixed_extreme_residency(self) -> bool:
        return bool(getattr(self._pool, "fixed_extreme_residency", False))

    @property
    def extreme_ram_layers(self) -> tuple:
        return tuple(getattr(self._pool, "extreme_ram_layers", ()))

    @property
    def extreme_gpu_layers(self) -> tuple:
        return tuple(getattr(self._pool, "extreme_gpu_layers", ()))

    @property
    def extreme_route_history_resident(self) -> bool:
        return bool(
            getattr(self._pool, "extreme_route_history_resident", False)
        )

    @property
    def extreme_storage_ratio(self) -> float:
        return float(getattr(self._pool, "extreme_storage_ratio", 0.0))

    @property
    def prefill_batch_rows(self) -> int:
        return int(getattr(self._pool, "prefill_batch_rows", 0))

    @property
    def prefill_batch_submissions(self) -> int:
        return int(getattr(self._pool, "prefill_batch_submissions", 0))

    @property
    def prefill_batch_max(self) -> int:
        return int(getattr(self._pool, "prefill_batch_max", 0))

    @property
    def short_reset_decode_tokens(self) -> int:
        return int(getattr(self._pool, "short_reset_decode_tokens", 0))

    @property
    def last_transfer_seconds(self) -> float:
        return float(getattr(self._pool, "last_transfer_seconds", 0.0))

    @property
    def decode_fused_submissions(self) -> int:
        return int(getattr(self._pool, "decode_fused_submissions", 0))

    @property
    def decode_graph_submissions(self) -> int:
        return int(getattr(self._pool, "decode_graph_submissions", 0))

    @property
    def prefill_rows_supported(self) -> bool:
        """Expose the common row-batched capability without leaking pools."""
        return bool(getattr(self._pool, "prefill_rows_supported", False))

    @property
    def host_overlap_supported(self) -> bool:
        """Return whether the common pool exposes host-overlap execution."""
        return bool(
            callable(getattr(self._pool, "prepare_host_run", None))
            and callable(getattr(self._pool, "finish_host_run", None))
        )

    @property
    def overlap_supported(self) -> bool:
        """Return whether the common pool exposes device-overlap execution."""
        return bool(
            callable(getattr(self._pool, "prepare_run", None))
            and callable(getattr(self._pool, "finish_run", None))
        )

    @property
    def resident_parallel_supported(self) -> bool:
        """Return whether a full-resident TP backend is active.

        Architecture adapters may use this capability to select topology
        workspaces, but never receive or call the concrete parallel pool.
        """
        return callable(getattr(self._pool, "compute", None))

    @property
    def full_resident(self) -> bool:
        """Expose residency as a public runtime capability."""
        return bool(getattr(self._pool, "full_resident", False))

    def preload_resident_if_fits(self) -> bool:
        """Try the public all-resident TP image without leaking its pool."""
        operation = getattr(self._pool, "preload_if_fits", None)
        return bool(operation()) if callable(operation) else False

    @staticmethod
    def _normalize_residency_device(device_type: str) -> str:
        normalized = str(device_type).strip().lower()
        if normalized not in {"cpu", "cuda"}:
            raise ValueError(
                f"unsupported routed VQ device type {device_type!r}"
            )
        return normalized

    def _prepare_residency(
        self,
        *,
        device_type: str,
    ) -> RoutedVQResidencyState:
        normalized = self._normalize_residency_device(device_type)
        preload_all = getattr(self._pool, "preload_all", None)
        if callable(preload_all):
            resident_all = bool(preload_all())
        else:
            preload_full = getattr(self._pool, "preload", None)
            resident_all = callable(preload_full)
            if resident_all:
                preload_full()
        if not resident_all:
            preload_pinned = getattr(self._pool, "preload_pinned", None)
            if callable(preload_pinned):
                preload_pinned()

        native_layers = 0
        if (
            normalized == "cpu"
            and bool(getattr(self._pool, "compact_full_resident", False))
        ):
            prepare_native = getattr(self._pool, "prepare_native_layers", None)
            if callable(prepare_native):
                native_layers = int(prepare_native() or 0)
        return RoutedVQResidencyState(
            resident_all=resident_all,
            native_layers=native_layers,
        )

    def prepare_required_residency(
        self,
        *,
        device_type: str,
    ) -> RoutedVQResidencyState:
        """Prepare an expert-first residency plan required by extreme mode."""
        state = self._prepare_residency(device_type=device_type)
        if not state.resident_all:
            raise RuntimeError(
                "fixed extreme residency requires all compact experts"
            )
        release = getattr(self._pool, "release_startup_gpu_reservation", None)
        if callable(release):
            release()
        return state

    def finalize_residency(
        self,
        state: RoutedVQResidencyState,
        *,
        device_type: str,
    ) -> RoutedVQResidencyState:
        """Finish device storage after topology weights have been loaded."""
        normalized = self._normalize_residency_device(device_type)
        if normalized == "cpu":
            return state

        # Allocate the device arena before pinning the complete host image.
        # On Windows/WDDM and large Linux BAR mappings, doing this in the
        # opposite order can consume the address space needed by the arena.
        gpu_arena_gb = 0.0
        build_arenas = getattr(self._pool, "build_gpu_arenas", None)
        if callable(build_arenas):
            gpu_arena_gb = float(build_arenas() or 0.0)
        if state.resident_all:
            pin_host = getattr(self._pool, "pin_host_resident", None)
            if callable(pin_host):
                pin_host()
            preload_profile = getattr(self._pool, "preload_profile_gpu", None)
            if callable(preload_profile):
                preload_profile()
        return RoutedVQResidencyState(
            resident_all=state.resident_all,
            native_layers=state.native_layers,
            gpu_arena_gb=gpu_arena_gb,
        )

    def verify_required_residency(self) -> None:
        """Verify a previously prepared startup reservation."""
        operation = getattr(self._pool, "verify_startup_gpu_reservation", None)
        if callable(operation):
            operation()

    def materialize_full_device(
        self,
        *,
        allocate: bool = False,
        load_payload: bool = True,
        capture_graphs: bool | None = None,
        dense_resident: bool | None = None,
    ):
        """Materialize a full-device packed image through one public path."""
        if allocate:
            allocator = getattr(self._pool, "allocate", None)
            if callable(allocator):
                allocator()
        if not load_payload:
            return None
        operation = getattr(self._pool, "preload", None)
        if not callable(operation):
            return None
        options = {}
        if capture_graphs is not None:
            options["capture_graphs"] = bool(capture_graphs)
        if dense_resident is not None:
            options["dense_resident"] = bool(dense_resident)
        return operation(**options)

    def initialize_residency(
        self,
        *,
        device_type: str,
    ) -> RoutedVQResidencyState:
        """Initialize the generic CPU/CUDA routed-codebook residency plan.

        Model adapters must not sequence concrete pool methods themselves.
        The public executor probes generic storage capabilities and owns the
        preload, pinning, native-image and GPU-arena lifecycle.
        """
        prepared = self._prepare_residency(device_type=device_type)
        return self.finalize_residency(
            prepared,
            device_type=device_type,
        )

    def decode_norm_output(self):
        """Return an optional fixed normalized-row output workspace."""
        operation = getattr(self._pool, "decode_norm_output", None)
        return operation() if callable(operation) else None

    def decode_route_outputs(self):
        """Return optional fixed route output workspaces."""
        operation = getattr(self._pool, "decode_route_outputs", None)
        return operation() if callable(operation) else None

    def begin_route_profile(self):
        """Start an optional backend-owned route timing interval."""
        if not bool(getattr(self._pool, "profile_enabled", False)):
            return None
        operation = getattr(self._pool, "profile_event", None)
        return operation() if callable(operation) else None

    def end_route_profile(self, started) -> None:
        """Finish an optional backend-owned route timing interval."""
        if started is None:
            return
        event = getattr(self._pool, "profile_event", None)
        record = getattr(self._pool, "profile_cuda", None)
        if callable(event) and callable(record):
            record("route", started, event())

    def execute_resident_parallel(
        self,
        layer: int,
        value: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        shared_fn,
        residual: torch.Tensor | None,
        merge_fn,
    ) -> torch.Tensor | None:
        """Own resident TP compute, shared overlap and final reduction.

        The callbacks contain only architecture topology.  Pool profiling,
        shard dispatch, reduction and fixed-workspace policy stay inside the
        common routed-codebook runtime.
        """
        if not self.resident_parallel_supported:
            return None
        profile_event = getattr(self._pool, "profile_event", None)
        profile_cuda = getattr(self._pool, "profile_cuda", None)

        overlap = getattr(self._pool, "compute_final_overlap", None)
        if residual is not None and callable(overlap):
            result = overlap(
                value,
                int(layer),
                route_ids,
                route_weights,
                shared_fn,
                residual,
            )
            if result is not None:
                return result

        shared_started = (
            profile_event() if callable(profile_event) else None
        )
        shared = shared_fn()
        shared_finished = (
            profile_event() if callable(profile_event) else None
        )
        if callable(profile_cuda):
            profile_cuda(
                "shared_expert",
                shared_started,
                shared_finished,
            )

        final = getattr(self._pool, "compute_final", None)
        result = (
            final(
                value,
                int(layer),
                route_ids,
                route_weights,
                shared,
                residual,
            )
            if residual is not None and callable(final)
            else None
        )
        if result is not None:
            return result

        routed = self._pool.compute(
            value,
            int(layer),
            route_ids,
            route_weights,
        )
        merge_started = (
            profile_event() if callable(profile_event) else None
        )
        result = merge_fn(routed, shared)
        merge_finished = (
            profile_event() if callable(profile_event) else None
        )
        if callable(profile_cuda):
            profile_cuda("final_add", merge_started, merge_finished)
        return result

    def prefetch_routes(
        self,
        layer: int,
        expert_ids: torch.Tensor | list[int] | tuple[int, ...],
    ) -> None:
        """Submit one deduplicated route hint through the common cache API."""
        operation = getattr(self._pool, "prefetch", None)
        if not callable(operation):
            return
        if isinstance(expert_ids, torch.Tensor):
            values = expert_ids.detach().reshape(-1).to("cpu").tolist()
        else:
            values = list(expert_ids)
        keys = list(
            dict.fromkeys((int(layer), int(expert_id)) for expert_id in values)
        )
        if keys:
            operation(keys)

    def release_scan_layer(self, layer: int) -> bool:
        """Release a completed one-way scan layer through the common API."""
        operation = getattr(self._pool, "release_scan_layer", None)
        return bool(operation(int(layer))) if callable(operation) else False

    def record_routes(
        self,
        layer: int,
        indices: torch.Tensor,
    ) -> None:
        """Store calibration counts behind the public codebook boundary."""
        counts = getattr(self._pool, "route_counts", None)
        if counts is None:
            counts = Counter()
            self._pool.route_counts = counts
        counts.update(
            (int(layer), int(expert))
            for expert in indices.detach().reshape(-1).to("cpu").tolist()
        )

    def prepare_host(
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
    ):
        operation = getattr(self._pool, "prepare_host_run", None)
        if not callable(operation):
            return None
        return operation(
            int(layer),
            value,
            route_ids,
            route_weights,
            activation=activation,
            activation_beta=float(activation_beta),
            activation_linear_beta=activation_linear_beta,
            limit=float(limit),
        )

    def finish_host(self, pending):
        if pending is None:
            raise RuntimeError("public routed VQ host execution was not prepared")
        operation = getattr(self._pool, "finish_host_run", None)
        if not callable(operation):
            raise RuntimeError("public routed VQ executor cannot finish a host run")
        return operation(pending)

    @contextmanager
    def phase(self, *, rows: int):
        from ..prefill import begin_prefill_block, end_prefill_block

        if int(rows) > 1:
            begin_prefill_block(self._pool)
            try:
                yield
            finally:
                end_prefill_block(self._pool)
            return
        end_prefill_block(self._pool)
        yield

    def execute(
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
        rows = int(value.numel() // value.shape[-1])
        if rows == 1 and route_ids.ndim > 1:
            if int(route_ids.shape[0]) != 1:
                raise ValueError("Decode routes must describe exactly one row")
            route_ids = route_ids.reshape(-1)
            route_weights = route_weights.reshape(-1)
        prepare_run = getattr(self._pool, "prepare_run", None)
        finish_run = getattr(self._pool, "finish_run", None)
        if rows == 1 and callable(prepare_run) and callable(finish_run):
            pending = None
            try:
                pending = self.prepare(
                    int(layer),
                    value,
                    route_ids,
                    route_weights,
                    activation=activation,
                    activation_beta=float(activation_beta),
                    activation_linear_beta=activation_linear_beta,
                    limit=float(limit),
                )
                return self.finish(pending)
            except BaseException:
                self.cancel(pending)
                raise
        operation = (
            getattr(self._pool, "run_rows", None)
            if rows > 1
            else getattr(self._pool, "run", None)
        )
        if rows == 1 and not callable(operation):
            operation = getattr(self._pool, "run_native", None)
        if not callable(operation) and rows > 1:
            mode = "Prefill" if rows > 1 else "Decode"
            raise RuntimeError(
                f"public routed VQ executor has no {mode} implementation"
            )
        result = (
            operation(
                int(layer),
                value,
                route_ids,
                route_weights,
                activation=activation,
                activation_beta=float(activation_beta),
                activation_linear_beta=activation_linear_beta,
                limit=float(limit),
            )
            if callable(operation)
            else None
        )
        if result is not None:
            return result
        if rows != 1:
            raise RuntimeError("public routed VQ Prefill returned no result")
        get_many = getattr(self._pool, "get_many", None)
        if not callable(get_many):
            raise RuntimeError("public routed VQ Decode returned no result")
        ids = [int(item) for item in route_ids.reshape(-1).tolist()]
        keys = [(int(layer), expert_id) for expert_id in ids]
        selected = get_many(keys)
        experts = [selected[key] for key in keys]
        from .api import packed_moe_selected_topk

        fallback = packed_moe_selected_topk(
            value.float(),
            experts,
            route_weights.reshape(-1).float(),
            activation=activation,
            activation_beta=float(activation_beta),
            activation_linear_beta=activation_linear_beta,
            limit=float(limit),
        )
        if fallback is None:
            from ..grouped import moe_mlp_grouped_mixed

            fallback = moe_mlp_grouped_mixed(
                value.float(),
                experts,
                route_weights.reshape(-1).float(),
                limit=float(limit),
                activation=activation,
                situ_beta=float(activation_beta),
                situ_linear_beta=activation_linear_beta,
            )
        return fallback

    def prepare(
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
    ):
        operation = getattr(self._pool, "prepare_run", None)
        if not callable(operation):
            return None
        return operation(
            int(layer),
            value,
            route_ids,
            route_weights,
            activation=activation,
            activation_beta=float(activation_beta),
            activation_linear_beta=activation_linear_beta,
            limit=float(limit),
        )

    def finish(self, pending):
        if pending is None:
            raise RuntimeError("public routed VQ execution was not prepared")
        operation = getattr(self._pool, "finish_run", None)
        if not callable(operation):
            raise RuntimeError("public routed VQ executor cannot finish a run")
        return operation(pending)

    def cancel(self, pending) -> None:
        if pending is None:
            return
        operation = getattr(self._pool, "cancel_run", None)
        if callable(operation):
            operation(pending)

    def execute_hidden(
        self,
        layer: int,
        value,
        routes: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float = 0.0,
    ):
        """Execute a fixed-address tensor-parallel routed projection.

        ``limit`` is accepted as part of the common routed-VQ contract even
        though the fixed-address backend has already captured its activation
        policy.  Architecture adapters therefore never need to branch on the
        concrete packed pool implementation.
        """
        del limit
        operation = getattr(self._pool, "run_hidden", None)
        if not callable(operation):
            raise RuntimeError(
                "public routed VQ executor has no tensor-parallel Decode "
                "implementation"
            )
        return operation(
            int(layer),
            value,
            routes,
            activation=activation,
            activation_beta=float(activation_beta),
            activation_linear_beta=activation_linear_beta,
        )

    def execute_replicated(
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
        """Execute row-batched routed VQ from rank-local input replicas."""
        operation = getattr(self._pool, "run_rows_replicated", None)
        if not callable(operation):
            raise RuntimeError(
                "public routed VQ executor has no replicated Prefill "
                "implementation"
            )
        return operation(
            int(layer),
            values,
            route_ids,
            route_weights,
            activation=activation,
            activation_beta=float(activation_beta),
            activation_linear_beta=activation_linear_beta,
            limit=float(limit),
            prefill_default=int(prefill_default),
        )


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


__all__ = [
    "build_routed_vq_topology_plan",
    "create_routed_vq_runtime",
    "FixedMoEPrelude",
    "FixedMoEPreludeSpec",
    "RoutedVQPoolPlan",
    "RoutedVQCapabilities",
    "RoutedVQResidencyState",
    "RoutedVQRuntimeStats",
    "RoutedVQRuntime",
    "RoutedVQExecutor",
    "plan_routed_vq_pool",
]

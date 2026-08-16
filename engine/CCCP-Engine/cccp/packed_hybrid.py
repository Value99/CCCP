"""配置驱动的公共单卡紧凑专家 RAM+GPU 执行池。

与通用 ``ExpertPool`` 的主要区别：

* 9..15-bit 索引在 RAM 中保持 CCCP 原始打包格式，不展开为 uint16；
* 按专家签名预分配稳定 GPU 槽，换专家只覆盖槽内容；
* 上一个 token 的路由在后台预取，需求路径按层等待；
* Gate/Up、gated activation、Down、路由加权直接使用公共融合 CUDA 核。

非专家 dense、注意力、共享专家和 KV 路径完全不变。实现按 projection-VQ
清单和算子能力分派，同时服务 Kimi 与 DeepSeek-V4。
"""

from __future__ import annotations

import gc
import os
import threading
import time
import ctypes
from collections.abc import Mapping
from collections import Counter, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import torch

from .expert_slots import SlotBook
from .extreme import EXTREME_RAM_LOAD_WORKSPACE_GIB
from .store import CCCPStore, PackedVQWeight, PinnedStage
from .ops.packed_view import (
    build_runtime_metadata_rows,
    runtime_metadata_row_count,
)
from .prefill import prefill_moe_batch_size


def _release_host_allocator() -> None:
    """Return dead loader arenas to the OS before enforcing the RAM floor."""

    gc.collect()
    if os.name == "nt":
        # ``ctypes.CDLL(None)`` is the POSIX process namespace.  CPython on
        # Windows passes the value through its DLL-name path handling and
        # raises TypeError before a library can be opened.  The Windows CRT
        # also does not expose glibc's malloc_trim, so collection above is the
        # complete safe operation on this platform.
        return
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (AttributeError, OSError, TypeError):
        # Windows and non-glibc allocators do not expose malloc_trim. The
        # capacity check still uses their real MemAvailable after GC.
        pass


def automatic_host_pin_budget(
    *,
    payload_bytes: int,
    available_ram_bytes: int,
    device_bytes: int,
    driver_multiplier: float,
) -> tuple[int, int]:
    """Return (pin budget, host safety floor) without allocating a copy.

    cudaHostRegister locks existing packed pages; it does not create another
    payload-sized RAM allocation. A fixed 64 GiB free-RAM gate therefore
    disabled the fast DMA path on otherwise healthy 64/128 GiB workstations.
    Keep 2 GiB pageable for the OS and runtime.  Automatic mode does not cap
    registrations by VRAM size: registered host pages consume RAM and an IOMMU
    mapping, not an equally sized device allocation.  ``driver_multiplier`` is
    retained only as an explicit diagnostic override.
    """
    payload = max(0, int(payload_bytes))
    available = max(0, int(available_ram_bytes))
    device = max(0, int(device_bytes))
    gib = 2**30
    safety_floor = 2 * gib
    pageable_room = max(0, available - safety_floor)
    driver_budget = (
        payload
        if driver_multiplier <= 0
        else max(0, int(device * driver_multiplier))
    )
    return min(payload, pageable_room, driver_budget), safety_floor


@dataclass(frozen=True)
class HostPackedWeight:
    raw: torch.Tensor
    cb: torch.Tensor
    rows: int
    cols: int
    blocks: int
    dim: int
    bits: int

    @classmethod
    def from_store(
        cls,
        weight: PackedVQWeight,
        codebook: torch.Tensor,
    ) -> "HostPackedWeight":
        return cls(
            raw=weight.raw,
            cb=codebook,
            rows=weight.rows,
            cols=weight.cols,
            blocks=weight.blocks,
            dim=weight.dim,
            bits=weight.bits,
        )

    @property
    def dtype_tag(self) -> int:
        return {
            8: 0, 16: 1, 12: 2, 14: 3, 10: 4, 9: 5,
            11: 6, 13: 7, 15: 8,
        }[
            self.bits
        ]

    @property
    def nbytes(self) -> int:
        return self.raw.nbytes + self.cb.nbytes


@dataclass(frozen=True)
class DevicePackedWeight:
    raw: torch.Tensor
    cb: torch.Tensor
    rows: int
    cols: int
    blocks: int
    dim: int
    bits: int

    @property
    def dtype_tag(self) -> int:
        return {
            8: 0, 16: 1, 12: 2, 14: 3, 10: 4, 9: 5,
            11: 6, 13: 7, 15: 8,
        }[
            self.bits
        ]

    @property
    def nbytes(self) -> int:
        return self.raw.nbytes + self.cb.nbytes


PackedExpert = tuple[HostPackedWeight, ...]
DeviceExpert = tuple[DevicePackedWeight, ...]


def _contiguous_expert_raw(expert) -> torch.Tensor | None:
    """Return one byte view when all projection payloads are adjacent."""

    if not expert:
        return None
    raws = tuple(weight.raw for weight in expert)
    if any(
        raw.dtype != torch.uint8 or raw.ndim != 1 or not raw.is_contiguous()
        for raw in raws
    ):
        return None
    storage = raws[0].untyped_storage()
    storage_pointer = storage.data_ptr()
    offset = raws[0].storage_offset()
    expected = offset
    for raw in raws:
        if (
            raw.untyped_storage().data_ptr() != storage_pointer
            or raw.storage_offset() != expected
        ):
            return None
        expected += raw.numel()
    return raws[0].as_strided(
        (expected - offset,),
        (1,),
        storage_offset=offset,
    )


def _coalesce_host_expert(expert: PackedExpert) -> PackedExpert:
    """Store one expert's packed projections in one unchanged byte blob."""

    if _contiguous_expert_raw(expert) is not None:
        return expert
    total = sum(weight.raw.nbytes for weight in expert)
    blob = torch.empty(total, dtype=torch.uint8)
    output = []
    offset = 0
    for weight in expert:
        count = weight.raw.nbytes
        raw = blob[offset : offset + count]
        raw.copy_(weight.raw.view(torch.uint8).reshape(-1))
        output.append(
            HostPackedWeight(
                raw,
                weight.cb,
                weight.rows,
                weight.cols,
                weight.blocks,
                weight.dim,
                weight.bits,
            )
        )
        offset += count
    return tuple(output)


@dataclass
class PendingPackedRun:
    """A staged packed-MoE call whose arena slots remain exclusively leased."""

    layer: int
    value: torch.Tensor
    expert_count: int
    grouped_prefix: int
    activation: str
    activation_beta: float
    activation_linear_beta: float | None
    limit: float
    wait_for_stage: bool
    route_order: torch.Tensor | None = None
    ordered_weights: torch.Tensor | None = None
    metadata: torch.Tensor | None = None
    native8_scales: torch.Tensor | None = None
    device_route_probe: bool = False
    active: bool = True


@dataclass(frozen=True)
class PackedRoutePlan:
    """Device-resident metadata for an immediately reusable Top-K route."""

    expert_ids: tuple[int, ...]
    keys: tuple[tuple[int, int], ...]
    experts: tuple[DeviceExpert, ...]
    order: torch.Tensor
    metadata: torch.Tensor
    grouped_prefix: int
    identity_order: bool


@dataclass(frozen=True)
class PackedWeightSignature:
    raw_bytes: int
    cb_shape: tuple[int, int]
    rows: int
    cols: int
    blocks: int
    dim: int
    bits: int

    @classmethod
    def of(cls, weight: HostPackedWeight) -> "PackedWeightSignature":
        return cls(
            raw_bytes=weight.raw.numel(),
            cb_shape=tuple(weight.cb.shape),
            rows=weight.rows,
            cols=weight.cols,
            blocks=weight.blocks,
            dim=weight.dim,
            bits=weight.bits,
        )


@dataclass(frozen=True)
class PackedExpertSignature:
    """Shape-only cache key for either GU+Down or Gate+Up+Down experts."""

    weights: tuple[PackedWeightSignature, ...]

    @classmethod
    def of(cls, expert: PackedExpert) -> "PackedExpertSignature":
        if len(expert) not in (2, 3):
            raise ValueError(
                "packed expert must contain GU+Down or Gate+Up+Down"
            )
        return cls(tuple(PackedWeightSignature.of(weight) for weight in expert))

    @property
    def projection_count(self) -> int:
        return len(self.weights)

    @property
    def gu(self) -> PackedWeightSignature:
        return self.weights[0]

    @property
    def up(self) -> PackedWeightSignature | None:
        return self.weights[1] if len(self.weights) == 3 else None

    @property
    def down(self) -> PackedWeightSignature:
        return self.weights[-1]

    # Compatibility properties retain the public diagnostics/tests used by the
    # original two-projection archive while the backing key is projection-count
    # agnostic.
    gu_raw_bytes = property(lambda self: self.gu.raw_bytes)
    gu_cb_shape = property(lambda self: self.gu.cb_shape)
    gu_rows = property(lambda self: self.gu.rows)
    gu_cols = property(lambda self: self.gu.cols)
    gu_blocks = property(lambda self: self.gu.blocks)
    gu_dim = property(lambda self: self.gu.dim)
    gu_bits = property(lambda self: self.gu.bits)
    down_raw_bytes = property(lambda self: self.down.raw_bytes)
    down_cb_shape = property(lambda self: self.down.cb_shape)
    down_rows = property(lambda self: self.down.rows)
    down_cols = property(lambda self: self.down.cols)
    down_blocks = property(lambda self: self.down.blocks)
    down_dim = property(lambda self: self.down.dim)
    down_bits = property(lambda self: self.down.bits)

    @property
    def raw_slot_bytes(self) -> int:
        return sum(weight.raw_bytes for weight in self.weights)

    @property
    def codebook_slot_bytes(self) -> int:
        return sum(
            weight.cb_shape[0] * weight.cb_shape[1]
            for weight in self.weights
        ) * torch.bfloat16.itemsize

    @property
    def slot_bytes(self) -> int:
        return self.raw_slot_bytes + self.codebook_slot_bytes

    def storage_bytes(self, resident_codebooks: bool) -> int:
        return (
            self.raw_slot_bytes
            if resident_codebooks
            else self.slot_bytes
        )


def allocate_packed_slots(
    counts: dict[PackedExpertSignature, int],
    budget: int,
    minimum: int | Mapping[PackedExpertSignature, int],
    weights: dict[PackedExpertSignature, float] | None = None,
    *,
    resident_codebooks: bool = False,
) -> dict[PackedExpertSignature, int]:
    """分配槽位，同时保证任一签名容得下完整 Top-K。

    ``weights`` 表示运行时路由流量，而不是模型中各档专家的静态数量。
    混合精度模型的高精度专家数量可能很少、调用却很频繁；若仍按静态数量
    分槽，该档位会在每个 token 内循环淘汰，显著放大 PCIe 传输。
    """
    if budget <= 0 or not counts:
        return {}
    if isinstance(minimum, Mapping):
        minimums = {
            signature: min(
                count,
                max(1, int(minimum.get(signature, 1))),
            )
            for signature, count in counts.items()
        }
    else:
        minimums = {
            signature: min(count, max(1, int(minimum)))
            for signature, count in counts.items()
        }
    minimum_bytes = sum(
        signature.storage_bytes(resident_codebooks) * count
        for signature, count in minimums.items()
    )
    if minimum_bytes > budget:
        raise RuntimeError(
            "packed GPU cache is too small for one complete Top-K of "
            f"every tier: need {minimum_bytes / 2**30:.2f} GiB, "
            f"have {budget / 2**30:.2f} GiB"
        )

    usable_weights = None
    if weights is not None:
        usable_weights = {
            signature: max(0.0, float(weights.get(signature, 0.0)))
            for signature in counts
        }
        if not any(usable_weights.values()):
            usable_weights = None

    if usable_weights is None:
        total_bytes = sum(
            signature.storage_bytes(resident_codebooks) * count
            for signature, count in counts.items()
        )
        scale = min(1.0, budget / max(1, total_bytes))
        allocated = {
            signature: min(
                count,
                max(minimums[signature], int(count * scale)),
            )
            for signature, count in counts.items()
        }
    else:
        allocated = dict(minimums)

    def used() -> int:
        return sum(
            signature.storage_bytes(resident_codebooks) * count
            for signature, count in allocated.items()
        )

    while used() > budget:
        candidates = [
            signature
            for signature in counts
            if allocated[signature] > minimums[signature]
        ]
        if not candidates:
            raise RuntimeError("cannot fit minimum packed GPU slots")
        signature = max(
            candidates,
            key=lambda item: (
                item.storage_bytes(resident_codebooks)
                * allocated[item]
            ),
        )
        allocated[signature] -= 1

    while True:
        candidates = [
            signature
            for signature, total in counts.items()
            if allocated[signature] < total
            and (
                used() + signature.storage_bytes(resident_codebooks)
                <= budget
            )
        ]
        if not candidates:
            break
        target = usable_weights or counts
        signature = min(
            candidates,
            key=lambda item: (
                allocated[item] / max(float(target[item]), 1e-12),
                item.storage_bytes(resident_codebooks),
            ),
        )
        allocated[signature] += 1
    return allocated


class _PackedArena:
    def __init__(
        self,
        count: int,
        signature: PackedExpertSignature,
        device: torch.device,
        *,
        resident_codebooks: bool,
        raw_storage: tuple[torch.Tensor, ...] | None = None,
        codebook_storage: tuple[torch.Tensor, ...] | None = None,
    ):
        self.signature = signature
        self.resident_codebooks = resident_codebooks
        self.book = SlotBook(count)
        self.raw = raw_storage or tuple(
            torch.empty(
                count,
                weight.raw_bytes,
                dtype=torch.uint8,
                device=device,
            )
            for weight in signature.weights
        )
        self.codebooks: tuple[torch.Tensor, ...] | None = None
        if not resident_codebooks:
            self.codebooks = codebook_storage or tuple(
                torch.empty(
                    count,
                    *weight.cb_shape,
                    dtype=torch.bfloat16,
                    device=device,
                )
                for weight in signature.weights
            )

    @property
    def nbytes(self) -> int:
        output = sum(tensor.nbytes for tensor in self.raw)
        if self.codebooks is not None:
            output += sum(tensor.nbytes for tensor in self.codebooks)
        return output

    def lease(
        self,
        key: tuple[int, int],
        codebooks: tuple[torch.Tensor, ...],
    ) -> tuple[object, DeviceExpert]:
        lease = self.book.acquire(key)
        slot = lease.slot
        signature = self.signature
        if not self.resident_codebooks:
            if self.codebooks is None:
                raise RuntimeError("packed slot codebook storage is missing")
            codebooks = tuple(
                storage[slot]
                for storage in self.codebooks
            )
        if len(codebooks) != signature.projection_count:
            raise ValueError("packed expert codebook count mismatch")
        return lease, tuple(
            DevicePackedWeight(
                self.raw[index][slot],
                codebooks[index],
                weight.rows,
                weight.cols,
                weight.blocks,
                weight.dim,
                weight.bits,
            )
            for index, weight in enumerate(signature.weights)
        )


class _PackedArenas:
    def __init__(
        self,
        specs: dict[PackedExpertSignature, int],
        device: torch.device,
        *,
        resident_codebooks: bool,
    ):
        entries = tuple(
            (signature, count)
            for signature, count in specs.items()
            if count > 0
        )
        # Hundreds of heterogeneous signatures used to create three CUDA
        # allocations each.  Their allocator blocks and fragmentation were
        # not represented by logical tensor.nbytes and could consume hundreds
        # of MiB beyond the capacity plan.  One public byte slab (plus one
        # optional BF16 codebook slab) keeps all layouts compact while every
        # signature still exposes the same contiguous tensor views.
        raw_elements = sum(
            count * weight.raw_bytes
            for signature, count in entries
            for weight in signature.weights
        )
        self._raw_storage = torch.empty(
            raw_elements,
            dtype=torch.uint8,
            device=device,
        )
        codebook_elements = (
            0
            if resident_codebooks
            else sum(
                count * weight.cb_shape[0] * weight.cb_shape[1]
                for signature, count in entries
                for weight in signature.weights
            )
        )
        self._codebook_storage = (
            None
            if resident_codebooks
            else torch.empty(
                codebook_elements,
                dtype=torch.bfloat16,
                device=device,
            )
        )
        raw_offset = 0
        codebook_offset = 0
        arenas: dict[PackedExpertSignature, _PackedArena] = {}
        for signature, count in entries:
            expert_raw_bytes = signature.raw_slot_bytes
            signature_raw_count = count * expert_raw_bytes
            expert_raw = self._raw_storage[
                raw_offset : raw_offset + signature_raw_count
            ].view(count, expert_raw_bytes)
            raw_offset += signature_raw_count
            raw_views: list[torch.Tensor] = []
            codebook_views: list[torch.Tensor] = []
            projection_offset = 0
            for weight in signature.weights:
                raw_views.append(
                    expert_raw[
                        :,
                        projection_offset:
                        projection_offset + weight.raw_bytes,
                    ]
                )
                projection_offset += weight.raw_bytes
                if self._codebook_storage is not None:
                    codebook_count = (
                        count
                        * weight.cb_shape[0]
                        * weight.cb_shape[1]
                    )
                    codebook_views.append(
                        self._codebook_storage[
                            codebook_offset:
                            codebook_offset + codebook_count
                        ].view(count, *weight.cb_shape)
                    )
                    codebook_offset += codebook_count
            arenas[signature] = _PackedArena(
                count,
                signature,
                device,
                resident_codebooks=resident_codebooks,
                raw_storage=tuple(raw_views),
                codebook_storage=(
                    tuple(codebook_views)
                    if codebook_views
                    else None
                ),
            )
        self.arenas = arenas
        self.leases: dict[
            tuple[int, int],
            tuple[PackedExpertSignature, object],
        ] = {}

    def repartition(
        self,
        specs: Mapping[PackedExpertSignature, int],
    ) -> None:
        """Re-slice the fixed byte slabs without allocating GPU memory.

        Layer-first Prefill needs the complete precision mix of one layer,
        whereas decode benefits from a global statistical mix.  Both layouts
        reuse the same stable slab allocation; only Python tensor views and
        slot books are rebuilt at a synchronized layer boundary.
        """

        entries = tuple(
            (signature, int(count))
            for signature, count in specs.items()
            if int(count) > 0
        )
        raw_required = sum(
            count * signature.raw_slot_bytes
            for signature, count in entries
        )
        if raw_required > self._raw_storage.numel():
            raise RuntimeError(
                "packed layer partition exceeds the fixed GPU byte slab: "
                f"{raw_required / 2**30:.2f} > "
                f"{self._raw_storage.numel() / 2**30:.2f} GiB"
            )
        codebook_required = sum(
            count * signature.codebook_slot_bytes
            for signature, count in entries
        )
        if (
            self._codebook_storage is not None
            and codebook_required > self._codebook_storage.nbytes
        ):
            raise RuntimeError(
                "packed layer codebooks exceed the fixed GPU codebook slab"
            )

        raw_offset = 0
        codebook_offset = 0
        arenas: dict[PackedExpertSignature, _PackedArena] = {}
        for signature, count in entries:
            signature_raw_count = count * signature.raw_slot_bytes
            expert_raw = self._raw_storage[
                raw_offset : raw_offset + signature_raw_count
            ].view(count, signature.raw_slot_bytes)
            raw_offset += signature_raw_count
            raw_views: list[torch.Tensor] = []
            codebook_views: list[torch.Tensor] = []
            projection_offset = 0
            for weight in signature.weights:
                raw_views.append(
                    expert_raw[
                        :,
                        projection_offset:
                        projection_offset + weight.raw_bytes,
                    ]
                )
                projection_offset += weight.raw_bytes
                if self._codebook_storage is not None:
                    item_count = (
                        count * weight.cb_shape[0] * weight.cb_shape[1]
                    )
                    codebook_views.append(
                        self._codebook_storage[
                            codebook_offset : codebook_offset + item_count
                        ].view(count, *weight.cb_shape)
                    )
                    codebook_offset += item_count
            arenas[signature] = _PackedArena(
                count,
                signature,
                self._raw_storage.device,
                resident_codebooks=self._codebook_storage is None,
                raw_storage=tuple(raw_views),
                codebook_storage=(
                    tuple(codebook_views) if codebook_views else None
                ),
            )
        self.arenas = arenas
        self.leases.clear()

    @property
    def nbytes(self) -> int:
        return self._raw_storage.nbytes + (
            0
            if self._codebook_storage is None
            else self._codebook_storage.nbytes
        )

    def touch(self, key: tuple[int, int]) -> None:
        item = self.leases.get(key)
        if item is not None:
            signature, _lease = item
            self.arenas[signature].book.touch(key)

    def protect(self, key: tuple[int, int]) -> bool:
        item = self.leases.get(key)
        if item is None:
            return False
        signature, _lease = item
        return self.arenas[signature].book.protect(key)

    def unprotect(self, key: tuple[int, int]) -> None:
        item = self.leases.get(key)
        if item is not None:
            signature, _lease = item
            self.arenas[signature].book.unprotect(key)

    def mark_inflight(self, key: tuple[int, int]) -> None:
        """Prevent a selected slot from being recycled inside one route batch."""

        signature, lease = self.leases[key]
        self.arenas[signature].book.mark_inflight(lease.slot)

    def clear_inflight(self, key: tuple[int, int]) -> None:
        item = self.leases.get(key)
        if item is not None:
            signature, lease = item
            self.arenas[signature].book.clear_inflight(lease.slot)

    @property
    def protected_count(self) -> int:
        return sum(
            arena.book.protected_count
            for arena in self.arenas.values()
        )

    def lease(
        self,
        key: tuple[int, int],
        expert: PackedExpert,
        device_codebooks: dict[int, torch.Tensor],
    ) -> tuple[tuple[int, int] | None, DeviceExpert]:
        signature = PackedExpertSignature.of(expert)
        arena = self.arenas[signature]
        codebooks = tuple(weight.cb for weight in expert)
        if arena.resident_codebooks:
            codebooks = tuple(
                device_codebooks[weight.cb.data_ptr()]
                for weight in expert
            )
        lease, device_expert = arena.lease(
            key,
            codebooks,
        )
        replaced = lease.replaced
        if replaced is not None:
            self.leases.pop(replaced, None)
        self.leases[key] = (signature, lease)
        return replaced, device_expert


class PackedHybridPool:
    """全量紧凑 RAM + 有界稳定 VRAM 的配置驱动 Top-K 专家池。"""

    device_routed = True
    full_resident = False
    prefetch_default = False
    # p8-p16 索引在磁盘、RAM、VRAM 中都保持原始 packed 布局。
    expanded_index_bytes = 0
    # Arena addresses, routing buffers and compute workspaces are stable within
    # each execution phase.  DSV4 may rebuild the same single arena at the
    # Prefill/Decode boundary so batch expansion and decode heat do not compete
    # for VRAM; addresses never change while a kernel or request phase uses it.
    fixed_arena_addresses = True
    fixed_route_buffers = True
    fixed_expert_assignment = False
    # The common routed-row executor accepts both Prefill batches and one-row
    # Decode.  Packed indices remain the authoritative resident image; only
    # the current expert chunk is expanded into native Tensor-Core bytes.
    prefill_rows_supported = True
    native8_rows_supported = False

    def __init__(
        self,
        store: CCCPStore,
        budget_gb: float,
        *,
        device: str | torch.device,
        ram_gb: float = 0.0,
        startup_gpu_reserve_bytes: int = 0,
    ):
        self.store = store
        self.device = torch.device(device)
        # This is a correctness/performance invariant, not a tuning switch.
        # Multi-token CUDA Prefill may not fall back to the width-one decode
        # executor, so the production hybrid pool always exposes run_rows.
        self.prefill_rows_supported = True
        self.budget = int(float(budget_gb) * 2**30)
        self._configured_budget = int(self.budget)
        self.ram_budget = int(float(ram_gb) * 2**30)
        self.startup_gpu_reserve_bytes = max(
            0,
            int(startup_gpu_reserve_bytes),
        )
        self._startup_gpu_reservation: torch.Tensor | None = None
        self._startup_gpu_after_experts = 0
        self.pinned: dict[tuple[int, int], PackedExpert] = {}
        self.cache: OrderedDict[tuple[int, int], DeviceExpert] = OrderedDict()
        self.bytes = 0
        self.ram_bytes = 0
        self.hits = 0
        self.miss = 0
        self.prefetch_hits = 0
        self.uploaded_bytes = 0
        self.transfer_seconds = 0.0
        self.last_transfer_seconds = 0.0
        self._host_codebooks: dict[
            tuple[str, str, str],
            torch.Tensor,
        ] = {}
        self._device_codebooks: dict[int, torch.Tensor] = {}
        self._native8_codebooks: dict[int, torch.Tensor] = {}
        self._native8_codebook_scales: dict[int, float] = {}
        self._native8_prefill_enabled = False
        self.native8_rows_supported = False
        self._host_pinned_bytes = 0
        self._host_registrations: dict[int, int] = {}
        self._resident_codebooks = (
            os.environ.get("CCCP_KIMI_RESIDENT_CODEBOOKS", "1") != "0"
        )
        self.initial_free_slots = 0
        self._arenas: _PackedArenas | None = None
        self._default_arena_specs: dict[PackedExpertSignature, int] = {}
        self._prefill_partition_layer: int | None = None
        self._decode_arena_target_budget = 0
        self._prefill_arena_target_budget = 0
        self._arena_phase = (
            "prefill"
            if (
                "hc_mult" in self.store.cfg
                or "compress_ratios" in self.store.cfg
            )
            else "decode"
        )
        self._adaptive_decode_arena = False
        self.adaptive_decode_repartitions = 0
        self._stage = PinnedStage(self.device, measure=True)
        self._lock = threading.RLock()
        self._transfer_lock = threading.RLock()
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="cccp-packed-prefetch",
        )
        self._prefetch_futures: set[Future] = set()
        self._last_ids: dict[int, list[int]] = {}
        self.profile_hot_keys: tuple[tuple[int, int], ...] = ()
        self.profile_hot_cache_enabled = False
        self.profile_hot_slots = 0
        self._profile_hot_ready = False
        self.speculative_prefetch = False
        self._route_plans: dict[int, PackedRoutePlan] = {}
        self.route_plan_hits = 0
        self.route_plan_misses = 0
        self._route_ids: torch.Tensor | None = None
        self._route_order_identity = False
        self._ordered_weights: torch.Tensor | None = None
        self._metadata: torch.Tensor | None = None
        self._slot_directory: torch.Tensor | None = None
        self._slot_update_host: torch.Tensor | None = None
        self._slot_scale_directory: torch.Tensor | None = None
        self._slot_scale_update_host: torch.Tensor | None = None
        self._native8_route_scales: torch.Tensor | None = None
        self._native8_route_scales_host: torch.Tensor | None = None
        self._compact_profile_all_resident = False
        self._route_hit_mask: torch.Tensor | None = None
        self._route_all_hit: torch.Tensor | None = None
        self._route_all_hit_host: torch.Tensor | None = None
        self._route_host_ids: torch.Tensor | None = None
        self._route_copy_done: torch.cuda.Event | None = None
        self.device_route_lookups = 0
        self.device_route_full_hits = 0
        self.device_route_fallbacks = 0
        self.route_counts: Counter[tuple[int, int]] = Counter()
        self.prefill_batch_submissions = 0
        self.prefill_batch_rows = 0
        self.prefill_batch_max = 0
        self.prefill_executor = (
            "cuda.packed-moe-grouped-fused"
            if torch.version.hip is not None
            else "cuda.chunked-dequant-grouped-gemm"
        )
        self._prefill_executor_announced = False
        self.prefill_batch_fallbacks = 0
        self.decode_fused_submissions = 0
        self.decode_graph_submissions = 0
        self.decode_reference_submissions = 0
        self.max_expert_slot_bytes = 0
        self.topk_staging_bytes = 0
        self.fixed_gpu_bytes_before_arena = 0
        self.vram_safety_reserve_bytes = 0
        self.vram_runtime_headroom_bytes = 0
        self.prefill_expert_chunk_capacity = 0
        self.prefill_expert_chunk_submissions = 0
        self.prefill_layer_unique_max = 0
        self._prefill_workspace: dict[str, object] | None = None
        self._prefill_dequant_workspace: tuple[
            torch.Tensor,
            torch.Tensor,
        ] | None = None
        self._prefill_native8_workspace: dict[str, torch.Tensor | int] | None = (
            None
        )
        self._native8_decode_workspace: dict[str, torch.Tensor | int] | None = (
            None
        )
        self._workspaces: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ] | None = None
        self._packed_moe_graphs: dict[object, object] = {}
        self._packed_graph_input: torch.Tensor | None = None
        self._packed_graph_weights: torch.Tensor | None = None
        self.slot_mix = self._read_slot_mix()
        self.arena_slots: dict[str, int] = {}
        self.fixed_extreme_residency = False
        # Fixed slots are physically rebuilt when Attention/KV grows.  Growth
        # stays disabled for the process lifetime: changing only a numeric
        # budget without reallocating the slab was the historical source of
        # false capacity and shared-memory spill.
        self.supports_vram_watch = True
        self.supports_vram_growth = False
        self.extreme_ram_layers: tuple[int, ...] = ()
        self.extreme_gpu_layers: tuple[int, ...] = ()
        self.extreme_mixed_layers: tuple[int, ...] = ()
        self.extreme_placement_mode = "layer"
        self.extreme_score_source = "none"
        self.extreme_gpu_expert_count = 0
        self.extreme_storage_ratio = 0.0
        self._extreme_specs: dict[PackedExpertSignature, int] | None = None
        self._extreme_gpu_keys: set[tuple[int, int]] = set()
        self.extreme_stage_slots = 0
        self.extreme_route_working_set = 0
        self.extreme_route_history_resident = False

    def _reserve_startup_gpu_capacity(self) -> None:
        """Physically reserve Dense/runtime capacity before expert placement.

        A bookkeeping-only subtraction can still overcommit because CUDA
        fragmentation and the per-process limit are only known at allocation
        time. This byte tensor proves the reservation is physically available.
        It is released into PyTorch's reusable cache immediately before Dense
        is streamed to the device.
        """

        if (
            self.startup_gpu_reserve_bytes <= 0
            or self._startup_gpu_reservation is not None
        ):
            return
        self._startup_gpu_reservation = torch.empty(
            self.startup_gpu_reserve_bytes,
            dtype=torch.uint8,
            device=self.device,
        )
        print(
            "[cccp-extreme] Dense/上下文显存物理预留完成："
            f"{self.startup_gpu_reserve_bytes / 2**30:.2f}GiB；"
            "随后先放置紧凑专家",
            flush=True,
        )

    def release_startup_gpu_reservation(self, *, dense_next: bool = True) -> int:
        """Release the placeholder while keeping its CUDA block reusable."""

        if self._startup_gpu_reservation is None:
            return 0
        released = self._startup_gpu_reservation.nbytes
        self._startup_gpu_reservation = None
        gc.collect()
        self._startup_gpu_after_experts = torch.cuda.memory_allocated(
            self.device
        )
        if dense_next:
            message = (
                "[cccp-extreme] 专家放置完成，释放显存占位并开始流式加载 "
                f"Dense：{released / 2**30:.2f}GiB（分配器块直接复用）"
            )
        else:
            message = (
                "[cccp-extreme] 专家规划/加载失败，已释放 Dense 显存占位："
                f"{released / 2**30:.2f}GiB"
            )
        print(message, flush=True)
        return released

    def verify_startup_gpu_reservation(self) -> None:
        """Reject an underestimated fixed allocation before inference."""

        if self.startup_gpu_reserve_bytes <= 0:
            return
        actual = max(
            0,
            torch.cuda.memory_allocated(self.device)
            - self._startup_gpu_after_experts,
        )
        if actual > self.startup_gpu_reserve_bytes:
            raise RuntimeError(
                "极限模式固定显存估算不足：Dense/上下文实际增加 "
                f"{actual / 2**30:.2f}GiB > 预留 "
                f"{self.startup_gpu_reserve_bytes / 2**30:.2f}GiB；"
                "拒绝带着不可靠容量规划继续推理。"
            )
        print(
            "[cccp-extreme] Dense 显存替换校验通过："
            f"实际={actual / 2**30:.2f}GiB / "
            f"预留={self.startup_gpu_reserve_bytes / 2**30:.2f}GiB；"
            "无完整模型副本",
            flush=True,
        )

    @staticmethod
    def _read_slot_mix() -> dict[str, float] | None:
        value = os.environ.get("CCCP_KIMI_SLOT_MIX", "").strip()
        if not value or value.lower() in ("model", "static", "off", "0"):
            return None
        output: dict[str, float] = {}
        for item in value.split(","):
            name, separator, weight = item.partition("=")
            if not separator:
                raise ValueError(
                    "CCCP_KIMI_SLOT_MIX must use tier=weight entries"
                )
            name = name.strip().lower()
            if name not in {"x", "w", "v", "vv"}:
                raise ValueError(f"unknown packed slot tier: {name}")
            output[name] = max(0.0, float(weight))
        if not any(output.values()):
            raise ValueError("CCCP_KIMI_SLOT_MIX contains no positive weight")
        return output

    @staticmethod
    def _signature_tier(signature: PackedExpertSignature) -> str:
        if signature.gu_bits == 8 and signature.gu_dim == 4:
            return "v"
        if signature.gu_bits == 14 and signature.gu_dim == 8:
            return "w"
        if signature.gu_bits == 12 and signature.gu_dim == 4:
            return "vv"
        if signature.gu_bits == 12 and signature.gu_dim == 8:
            return "x"
        return f"p{signature.gu_bits}d{signature.gu_dim}"

    @property
    def host_expert_bytes(self) -> int:
        raw = sum(
            weight.raw.nbytes
            for expert in self.pinned.values()
            for weight in expert
        )
        codebooks = sum(cb.nbytes for cb in self._host_codebooks.values())
        return raw + codebooks

    @property
    def gpu_arena_bytes(self) -> int:
        return 0 if self._arenas is None else self._arenas.nbytes

    @property
    def gpu_storage_bytes(self) -> int:
        workspace = 0
        if self._workspaces is not None:
            workspace += sum(tensor.nbytes for tensor in self._workspaces)
        if self._metadata is not None:
            workspace += self._metadata.nbytes
        if self._slot_directory is not None:
            workspace += self._slot_directory.nbytes
        if self._slot_scale_directory is not None:
            workspace += self._slot_scale_directory.nbytes
        if self._route_hit_mask is not None:
            workspace += self._route_hit_mask.nbytes
        if self._route_all_hit is not None:
            workspace += self._route_all_hit.nbytes
        if self._route_ids is not None:
            workspace += self._route_ids.nbytes
        if self._ordered_weights is not None:
            workspace += self._ordered_weights.nbytes
        workspace += sum(
            plan.order.nbytes + plan.metadata.nbytes
            for plan in self._route_plans.values()
        )
        workspace += sum(
            codebook.nbytes
            for codebook in self._device_codebooks.values()
        )
        workspace += sum(
            codebook.nbytes
            for codebook in self._native8_codebooks.values()
        )
        return self.gpu_arena_bytes + workspace

    @property
    def protected_experts(self) -> int:
        return (
            0
            if self._arenas is None
            else self._arenas.protected_count
        )

    def _host_codebook(
        self,
        key: tuple[str, str, str],
        cb: torch.Tensor,
    ) -> torch.Tensor:
        """Return one stable BF16 codebook for a semantic archive key.

        The store's FP32 codebook cache is populated concurrently during expert
        preload.  A losing duplicate tensor can be freed immediately and its
        ``data_ptr`` reused by another layer.  Pointer-only host keys therefore
        caused rare cross-layer codebook aliasing and nondeterministic logits.
        """
        with self._lock:
            value = self._host_codebooks.get(key)
            if value is None:
                value = cb.to(dtype=torch.bfloat16).contiguous()
                self._host_codebooks[key] = value
        return value

    def _load_one(self, layer: int, expert_id: int) -> PackedExpert:
        packed = self.store.load_expert_packed(layer, expert_id)
        if self.store.man.projection_vq:
            variants = self.store.projection_codebook_variants(
                layer,
                expert_id,
            )
            names = ("gate", "up", "down")
            if len(packed) != 3 or len(variants) != 3:
                raise ValueError(
                    f"L{layer} projection-VQ expert must have three weights"
                )
            return _coalesce_host_expert(tuple(
                HostPackedWeight.from_store(
                    weight,
                    self._host_codebook(
                        ("projection-vq", variant, projection),
                        weight.cb,
                    ),
                )
                for projection, variant, weight in zip(
                    names,
                    variants,
                    packed,
                )
            ))

        gu, down = packed
        tier = self.store.expert_kind(layer, expert_id).rstrip("z")
        codebook_variants = self.store.codebook_variants(
            layer,
            tier,
            expert_id,
        )
        return _coalesce_host_expert((
            HostPackedWeight(
                gu.raw,
                self._host_codebook(
                    (tier, codebook_variants[0], "gu"),
                    gu.cb,
                ),
                gu.rows,
                gu.cols,
                gu.blocks,
                gu.dim,
                gu.bits,
            ),
            HostPackedWeight(
                down.raw,
                self._host_codebook(
                    (tier, codebook_variants[1], "down"),
                    down.cb,
                ),
                down.rows,
                down.cols,
                down.blocks,
                down.dim,
                down.bits,
            ),
        ))

    @staticmethod
    def _packed_width(format_name: str) -> int:
        value = str(format_name).lower()
        if not value.startswith("p") or not value[1:].isdigit():
            raise ValueError(f"极限模式不支持 packed 格式 {format_name!r}")
        bits = int(value[1:])
        if not 8 <= bits <= 16:
            raise ValueError(f"极限模式 packed 位宽非法: {bits}")
        return bits

    def _extreme_signature(
        self,
        layer: int,
        expert_id: int,
    ) -> PackedExpertSignature:
        """只读取共享码本与清单元数据，不读取专家索引主体。"""

        if not self.store.man.projection_vq:
            raise RuntimeError(
                "极限模式要求三投影 packed VQ；旧专家格式可能展开索引"
            )
        # Capacity and arena signatures are per expert.  Heterogeneous layers
        # return a layout union when expert_id is omitted, which is correct for
        # registry discovery but must never be used as one expert's Gate/Up/
        # Down tuple.
        capability = self.store.man.projection_operator_capability(
            layer,
            expert_id,
        )
        formats = capability["packed_formats"]
        dims = capability["code_dims"]
        codebooks = self.store.projection_codebooks(layer, expert_id)
        variants = self.store.projection_codebook_variants(
            layer,
            expert_id,
        )
        hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        intermediate = int(self.store.cfg["moe_inter"])
        shapes = (
            (intermediate, hidden),
            (intermediate, hidden),
            (hidden, intermediate),
        )
        weights = []
        for projection, format_name, dim, codebook, variant, shape in zip(
            ("gate", "up", "down"),
            formats,
            dims,
            codebooks,
            variants,
            shapes,
        ):
            bits = self._packed_width(format_name)
            rows, cols = shape
            blocks = cols // int(dim)
            payload_bits = rows * blocks * bits
            if payload_bits % 8:
                raise RuntimeError(
                    f"L{layer} e{expert_id} {projection} 不是整字节 packed payload"
                )
            host_codebook = self._host_codebook(
                ("projection-vq", variant, projection),
                codebook,
            )
            weights.append(
                PackedWeightSignature(
                    raw_bytes=payload_bits // 8,
                    cb_shape=tuple(host_codebook.shape),
                    rows=rows,
                    cols=cols,
                    blocks=blocks,
                    dim=int(dim),
                    bits=bits,
                )
            )
        return PackedExpertSignature(tuple(weights))

    def _preload_extreme(self, reserve_gb: float) -> bool:
        """把完整层分到 RAM/VRAM，运行时不再读取专家文件。"""

        if self.device.type != "cuda":
            raise RuntimeError("极限模式要求单卡 CUDA packed 专家池")
        if not self.store.man.projection_vq:
            raise RuntimeError(
                "极限模式拒绝旧专家格式：其索引加载后可能大于磁盘 payload"
            )
        import psutil

        layers = tuple(sorted(int(x) for x in self.store.man.expert_files))
        n_experts = int(self.store.cfg["n_experts"])
        signatures_by_layer: dict[int, tuple[PackedExpertSignature, ...]] = {}
        layer_bytes: dict[int, int] = {}
        keys_by_layer: dict[int, tuple[tuple[int, int], ...]] = {}
        print(
            "[cccp-extreme] 读取码本与 packing 元数据，规划整层 RAM/VRAM 放置…",
            flush=True,
        )
        for layer in layers:
            keys = tuple(
                (layer, expert_id)
                for expert_id in range(n_experts)
                if self.store.expert_kind(layer, expert_id) != "drop"
            )
            signatures = tuple(
                self._extreme_signature(*key)
                for key in keys
            )
            keys_by_layer[layer] = keys
            signatures_by_layer[layer] = signatures
            layer_bytes[layer] = sum(
                signature.raw_slot_bytes for signature in signatures
            )
        # projection_codebooks() 使用 FP32 作为读取中间态；运行时只保留公共
        # HostPackedWeight 所引用的 BF16 码本，避免双份码本常驻。
        self.store._cb_cache.clear()
        gc.collect()

        from .extreme import (
            GIB,
            effective_available_memory_bytes,
            load_expert_residency_scores,
            plan_extreme_expert_placement,
            plan_extreme_layer_placement,
        )

        available = effective_available_memory_bytes()
        ram_available = min(available, self.ram_budget + int(reserve_gb * GIB))
        safe_gpu = self._safe_budget()
        codebook_bytes = sum(
            value.nbytes for value in self._host_codebooks.values()
        )
        # mmap-backed packed rows and the two loader futures briefly coexist
        # while one expert is materialized. Keep one bounded 0.5 GiB loader
        # workspace in addition to the user's 1 GiB system reserve. A 5%
        # model-sized default silently held back several GiB on a 64 GiB host
        # and contradicted extreme mode's documented "fill until 1 GiB" rule;
        # 256 MiB was insufficient before releasing glibc's dead loader arenas
        # on a real 64 GiB host; the final physical 1 GiB check remains hard.
        configured_loader_workspace = os.environ.get(
            "CCCP_EXTREME_LOAD_WORKSPACE_GB",
            "",
        ).strip()
        loader_workspace = (
            int(float(configured_loader_workspace) * GIB)
            if configured_loader_workspace
            else int(EXTREME_RAM_LOAD_WORKSPACE_GIB * GIB)
        )
        if loader_workspace < 256 * 2**20:
            raise RuntimeError(
                "极限模式加载工作区至少需要 0.25 GiB"
            )
        print(
            "[cccp-extreme] RAM规划："
            f"payload={sum(layer_bytes.values()) / GIB:.2f}GiB；"
            f"码本={codebook_bytes / GIB:.2f}GiB；"
            f"加载工作区={loader_workspace / GIB:.2f}GiB；"
            f"系统预留={reserve_gb:.2f}GiB",
            flush=True,
        )
        signature_by_key = {
            key: signature
            for layer in layers
            for key, signature in zip(
                keys_by_layer[layer],
                signatures_by_layer[layer],
            )
        }
        size_by_key = {
            key: signature.raw_slot_bytes
            for key, signature in signature_by_key.items()
        }
        top_k = int(self.store.cfg["top_k"])
        # Capacity order is fixed: Dense/runtime reservation (already held),
        # shared codebooks, one executable Top-K for every packed signature,
        # then GPU-only experts. The old order filled GPU experts first and
        # discovered too late that the model could not execute a RAM layer.
        minimum_stage_by_signature: dict[PackedExpertSignature, int] = {}
        for layer in layers:
            layer_counts = Counter(signatures_by_layer[layer])
            for signature, count in layer_counts.items():
                minimum_stage_by_signature[signature] = max(
                    minimum_stage_by_signature.get(signature, 0),
                    min(top_k, count),
                )
        minimum_stage_bytes = sum(
            signature.storage_bytes(True) * count
            for signature, count in minimum_stage_by_signature.items()
        )
        gpu_expert_budget = max(
            0,
            safe_gpu - codebook_bytes - minimum_stage_bytes,
        )
        print(
            "[cccp-extreme] 显存规划顺序："
            f"固定Dense占位→码本 {codebook_bytes / GIB:.2f}GiB→"
            f"最小Top-K {minimum_stage_bytes / GIB:.2f}GiB→"
            f"GPU专家上限 {gpu_expert_budget / GIB:.2f}GiB",
            flush=True,
        )
        requested_placement = os.environ.get(
            "CCCP_EXTREME_PLACEMENT",
            "auto",
        ).strip().lower()
        if requested_placement not in {"auto", "layer", "precision"}:
            raise RuntimeError(
                "CCCP_EXTREME_PLACEMENT 只接受 auto/layer/precision"
            )
        precision_placement = (
            requested_placement == "precision"
            or (
                requested_placement == "auto"
                and self.store.man.heterogeneous_projection_vq
            )
        )
        if precision_placement:
            # The packed bytes assigned by the quantizer are a self-contained
            # precision/importance signal: all routed experts have identical
            # logical matrix shapes, so a larger compact payload represents a
            # larger bit budget.  This remains manifest-driven and introduces
            # no model-name or tier-name branch.
            score_file = os.environ.get(
                "CCCP_EXTREME_SCORE_FILE",
                "",
            ).strip()
            if score_file:
                precision_scores = load_expert_residency_scores(score_file)
                missing_scores = size_by_key.keys() - precision_scores.keys()
                extra_scores = precision_scores.keys() - size_by_key.keys()
                if missing_scores or extra_scores:
                    raise RuntimeError(
                        "专家常驻分数必须与归档一一覆盖："
                        f"缺少={len(missing_scores)}，多余={len(extra_scores)}"
                    )
                placement_groups = None
                self.extreme_score_source = "route-mass"
            else:
                precision_scores = {
                    key: float(size)
                    for key, size in size_by_key.items()
                }
                placement_groups = {
                    key: key[0] for key in size_by_key
                }
                self.extreme_score_source = "packed-bit-budget"
            try:
                placement = plan_extreme_expert_placement(
                    size_by_key,
                    precision_scores,
                    placement_groups=placement_groups,
                    available_ram_bytes=ram_available,
                    ram_reserve_bytes=int(reserve_gb * GIB),
                    # Host codebooks are already materialized above and are
                    # therefore already reflected in ``available``. Only the
                    # not-yet-allocated loader workspace is subtracted here.
                    fixed_ram_bytes=loader_workspace,
                    gpu_expert_bytes=gpu_expert_budget,
                )
            except RuntimeError:
                if requested_placement != "auto":
                    raise
                try:
                    # Per-layer fairness is a performance preference, not a
                    # capacity invariant. On a genuinely tight machine keep
                    # the largest/highest-bit experts in GPU globally; every
                    # layer remains executable through the reserved staging.
                    placement = plan_extreme_expert_placement(
                        size_by_key,
                        precision_scores,
                        placement_groups=None,
                        available_ram_bytes=ram_available,
                        ram_reserve_bytes=int(reserve_gb * GIB),
                        fixed_ram_bytes=loader_workspace,
                        gpu_expert_bytes=gpu_expert_budget,
                    )
                    self.extreme_score_source = (
                        "packed-bit-budget-capacity"
                    )
                    print(
                        "[cccp-extreme] 分层均衡精度放置超出容量，改用全局 "
                        "bit-budget 放置；最小 Top-K 仍完整保留",
                        flush=True,
                    )
                except RuntimeError as capacity_error:
                    precision_placement = False
                    print(
                        "[cccp-extreme] 全局精度放置仍无法满足容量："
                        f"{capacity_error}；最后尝试整层放置",
                        flush=True,
                    )
        if precision_placement:
            # Selection order follows score rank.  Loading order follows the
            # archive so codebook metadata is reused and startup does not
            # thrash one layer's temporary handles per expert.
            gpu_keys = tuple(sorted(placement.gpu_keys))
            ram_keys = tuple(placement.ram_keys)
            gpu_key_set = set(gpu_keys)
            ram_key_set = set(ram_keys)
            self.extreme_gpu_layers = tuple(
                layer
                for layer in layers
                if all(key in gpu_key_set for key in keys_by_layer[layer])
            )
            self.extreme_ram_layers = tuple(
                layer
                for layer in layers
                if any(key in ram_key_set for key in keys_by_layer[layer])
            )
            self.extreme_mixed_layers = tuple(
                layer
                for layer in layers
                if any(key in gpu_key_set for key in keys_by_layer[layer])
                and any(key in ram_key_set for key in keys_by_layer[layer])
            )
            self.extreme_placement_mode = "precision"
        if not precision_placement:
            placement = plan_extreme_layer_placement(
                layer_bytes,
                available_ram_bytes=ram_available,
                # Host codebooks are already reflected in current available
                # RAM. Keep one bounded *additional* loader workspace so
                # cgroup MemoryMax is not reached by materialization futures.
                ram_reserve_bytes=int(reserve_gb * GIB),
                fixed_ram_bytes=loader_workspace,
                gpu_expert_bytes=gpu_expert_budget,
            )
            self.extreme_ram_layers = placement.ram_layers
            self.extreme_gpu_layers = placement.gpu_layers
            self.extreme_mixed_layers = ()
            gpu_keys = tuple(
                key
                for layer in self.extreme_gpu_layers
                for key in keys_by_layer[layer]
            )
            ram_keys = tuple(
                key
                for layer in self.extreme_ram_layers
                for key in keys_by_layer[layer]
            )
            gpu_key_set = set(gpu_keys)
            ram_key_set = set(ram_keys)
            self.extreme_placement_mode = "layer"
            self.extreme_score_source = "none"
        self.extreme_gpu_expert_count = len(gpu_keys)

        resident_counts: Counter[PackedExpertSignature] = Counter(
            signature_by_key[key] for key in gpu_keys
        )
        ram_counts: Counter[PackedExpertSignature] = Counter(
            signature_by_key[key] for key in ram_keys
        )
        compact_baseline = sum(layer_bytes.values()) + codebook_bytes
        max_ratio = float(os.environ.get("CCCP_EXTREME_MAX_OVERHEAD", "1.10"))
        resident_bytes = sum(
            signature.storage_bytes(True) * count
            for signature, count in resident_counts.items()
        )
        # 受保护 GPU 层永不参与 LRU；其余安全显存用于跨 token 热专家槽。
        # 重复 staging 仍受常驻倍率约束，不能靠把整模再复制进 VRAM 换速度。
        duplicate_limit = max(
            0,
            int((max_ratio - 1.0) * compact_baseline)
            - codebook_bytes
            - 64 * 2**20,
        )
        stage_budget = min(
            max(0, safe_gpu - codebook_bytes - resident_bytes),
            duplicate_limit,
        )
        # A heterogeneous archive can contain hundreds of signatures while a
        # single layer has only one or two experts of most signatures.  The
        # arena only needs to hold the maximum number of same-signature
        # experts that one routed Top-K can request in one layer, not Top-K
        # slots for every signature in the whole model.
        minimum_by_signature: dict[PackedExpertSignature, int] = {}
        for layer in self.extreme_ram_layers:
            layer_counts = Counter(
                signature_by_key[key]
                for key in keys_by_layer[layer]
                if key in ram_key_set
            )
            for signature, count in layer_counts.items():
                minimum_by_signature[signature] = max(
                    minimum_by_signature.get(signature, 1),
                    min(top_k, count),
                )
        stage_counts = (
            allocate_packed_slots(
                dict(ram_counts),
                stage_budget,
                minimum_by_signature,
                resident_codebooks=True,
            )
            if ram_counts
            else {}
        )
        self.extreme_stage_slots = sum(stage_counts.values())
        self.extreme_route_working_set = sum(
            min(
                top_k,
                sum(key in ram_key_set for key in keys_by_layer[layer]),
            )
            for layer in self.extreme_ram_layers
        )
        # 单一 packed 签名时，每个槽对任意 RAM 层都可复用。只要能保留
        # 一整轮 Top-K，就保护各层上一 token 的实际路由，杜绝全局 LRU
        # 从浅层开始把深层即将使用的槽逐层赶走。混合签名模型仍使用
        # 公共预取路径，避免按错误的档位比例作出过度承诺。
        self.extreme_route_history_resident = (
            len(ram_counts) == 1
            and self.extreme_stage_slots >= self.extreme_route_working_set
        )
        specs = dict(resident_counts)
        for signature, count in stage_counts.items():
            specs[signature] = specs.get(signature, 0) + count
        arena_bytes = sum(
            signature.storage_bytes(True) * count
            for signature, count in specs.items()
        )
        if arena_bytes + codebook_bytes > safe_gpu:
            raise RuntimeError(
                "极限模式显存不足：GPU 专家整层 + RAM Top-K staging + "
                f"共享码本需要 {(arena_bytes + codebook_bytes) / GIB:.2f} GiB，"
                f"安全可用仅 {safe_gpu / GIB:.2f} GiB。请降低 --max-ctx、"
                "关闭其他显存进程或换用更小模型。"
            )
        self._extreme_specs = specs
        self.build_gpu_arenas()

        gpu_total = len(gpu_keys)
        loaded_gpu = 0
        previous_layer = None
        for key in gpu_keys:
            if previous_layer is not None and key[0] != previous_layer:
                self.store._cb_cache.clear()
                gc.collect()
            host = self._load_one(*key)
            self.pinned[key] = host
            self._ensure_locked([key], prefetch=True)
            if self._arenas is None or not self._arenas.protect(key):
                raise RuntimeError(
                    f"极限模式无法保护 GPU-only 专家 L{key[0]}/e{key[1]}"
                )
            self._extreme_gpu_keys.add(key)
            self.pinned.pop(key, None)
            loaded_gpu += 1
            previous_layer = key[0]
            if loaded_gpu % 256 == 0:
                print(
                    f"[cccp-extreme] GPU-only 专家 {loaded_gpu}/{gpu_total}",
                    flush=True,
                )
        self.store._cb_cache.clear()
        gc.collect()

        workers = max(1, int(os.environ.get("CCCP_LOAD_WORKERS", "12")))
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="cccp-extreme-ram",
        ) as executor:
            futures = {
                executor.submit(self._load_one, *key): key
                for key in ram_keys
            }
            for index, future in enumerate(as_completed(futures), 1):
                self.pinned[futures[future]] = future.result()
                if index % 2000 == 0:
                    print(
                        f"[cccp-extreme] RAM 专家 {index}/{len(ram_keys)}",
                        flush=True,
                    )
        self.store._cb_cache.clear()
        # GPU-only 层的主机码本不再需要；device 码本已经独立持有。
        retained_codebook_ptrs = {
            weight.cb.data_ptr()
            for expert in self.pinned.values()
            for weight in expert
        }
        self._host_codebooks = {
            key: value
            for key, value in self._host_codebooks.items()
            if value.data_ptr() in retained_codebook_ptrs
        }
        _release_host_allocator()
        self.ram_bytes = self.host_expert_bytes
        remaining_ram = effective_available_memory_bytes()
        required_reserve = int(reserve_gb * GIB)
        if remaining_ram < required_reserve:
            raise RuntimeError(
                "极限模式 RAM 预留未满足：加载后可用 "
                f"{remaining_ram / GIB:.2f} GiB < 要求 {reserve_gb:.2f} GiB。"
                "请关闭其他进程或换用更小模型。"
            )
        actual = self.host_expert_bytes + self.gpu_storage_bytes
        self.extreme_storage_ratio = actual / max(1, compact_baseline)
        if self.extreme_storage_ratio > max_ratio:
            raise RuntimeError(
                "极限模式常驻放大超过限制："
                f"{self.extreme_storage_ratio:.3f}x > {max_ratio:.3f}x；"
                "拒绝用隐式副本换取表面容量。"
            )
        self.fixed_extreme_residency = True
        self.supports_vram_watch = False
        self.speculative_prefetch = False
        self.bytes = self.gpu_storage_bytes
        print(
            "[cccp-extreme] 紧凑常驻完成："
            f"策略={self.extreme_placement_mode}；"
            f"热度来源={self.extreme_score_source}；"
            f"RAM参与层={list(self.extreme_ram_layers)} "
            f"{self.host_expert_bytes / GIB:.2f}GiB；"
            f"GPU整层={list(self.extreme_gpu_layers)}；"
            f"混合层={list(self.extreme_mixed_layers)}；"
            f"GPU-only专家={self.extreme_gpu_expert_count} / "
            f"专家与staging {self.gpu_storage_bytes / GIB:.2f}GiB；"
            f"RAM热槽={self.extreme_stage_slots}/"
            f"一轮路由={self.extreme_route_working_set}；"
            f"常驻/紧凑基准={self.extreme_storage_ratio:.3f}x；"
            "expanded_index_bytes=0，运行期专家磁盘读取=0",
            flush=True,
        )
        return True

    def preload_all(self, reserve_gb: float | None = None) -> bool:
        if self.pinned:
            return True
        if os.environ.get("CCCP_FULL_RESIDENT", "1") == "0":
            return False
        import psutil

        if reserve_gb is None:
            reserve_gb = float(
                os.environ.get("CCCP_RESIDENT_RESERVE_GB", "2.0")
            )
        if os.environ.get("CCCP_EXTREME_MODE", "0") != "0":
            self._reserve_startup_gpu_capacity()
            try:
                return self._preload_extreme(float(reserve_gb))
            except BaseException:
                # A failed startup must not strand a multi-GiB placeholder in
                # a long-lived API worker or a retrying Engine process.
                self.release_startup_gpu_reservation(dense_next=False)
                raise
        n_experts = int(self.store.cfg["n_experts"])
        physical_keys = [
            (layer, expert_id)
            for layer in sorted(self.store.man.expert_files)
            for expert_id in range(n_experts)
            if self.store.expert_kind(layer, expert_id) != "drop"
        ]
        # Validate the physical archive before applying a runtime profile.  A
        # strict route profile intentionally contains fewer experts and must
        # not be mistaken for quantization-time expert loss.
        if self.store.man.no_expert_drop:
            declared_layers = self.store.man.routed_layers
            if declared_layers and declared_layers != len(
                self.store.man.expert_files
            ):
                raise RuntimeError(
                    "projection-VQ CCCP 专家清单未收敛："
                    f"声明 {declared_layers} 层，"
                    f"实际只有 {len(self.store.man.expert_files)} 层"
                )
            declared_experts = (
                self.store.man.routed_experts_per_layer or n_experts
            )
            expected = len(self.store.man.expert_files) * declared_experts
            if declared_experts != n_experts or len(physical_keys) != expected:
                present = set(physical_keys)
                missing = [
                    f"L{layer}/e{expert_id}"
                    for layer in sorted(self.store.man.expert_files)
                    for expert_id in range(n_experts)
                    if (layer, expert_id) not in present
                ][:8]
                raise RuntimeError(
                    "projection-VQ no_expert_drop 清单与专家文件不一致："
                    f"期望 {expected}，实际 {len(physical_keys)}"
                    + (f"，缺失示例 {', '.join(missing)}" if missing else "")
                )
        allowlist = self.store.route_allowlist
        keys = (
            [
                key for key in physical_keys
                if key[1] in allowlist.get(key[0], set())
            ]
            if allowlist is not None
            else physical_keys
        )
        if self.store.man.projection_vq:
            # Expert layer files also contain experts excluded by the selected
            # profile.  Charging the entire files here can reject a valid
            # 24/32/44-GiB configuration before loading.  Projection metadata
            # gives the exact packed bytes for the experts that remain routed.
            stored_bytes = sum(
                self._extreme_signature(*key).raw_slot_bytes
                for key in keys
            )
        else:
            expert_files = [
                os.path.join(self.store.root, filename)
                for filename in self.store.man.expert_files.values()
            ]
            stored_bytes = sum(
                os.path.getsize(path)
                for path in expert_files
                if os.path.exists(path)
            )
        virtual = psutil.virtual_memory()
        available = virtual.available
        reserve_bytes = int(reserve_gb * 2**30)
        allow_pagefile = (
            os.environ.get("CCCP_ALLOW_PAGEFILE_RESIDENT", "0") == "1"
        )
        if stored_bytes + reserve_bytes > available:
            swap_free = psutil.swap_memory().free if allow_pagefile else 0
            if stored_bytes + reserve_bytes > available + swap_free:
                print(
                    "[cccp-packed] 紧凑专家无法完整载入 RAM/系统虚拟内存："
                    f"配置专家 {stored_bytes / 2**30:.1f}GiB + "
                    f"预留 {reserve_gb:.1f}GiB > "
                    f"可用物理内存与交换空间 "
                    f"{(available + swap_free) / 2**30:.1f}GiB",
                    flush=True,
                )
                return False
            print(
                "[cccp-packed] 物理内存不足，配置内全部专家仍会载入；"
                f"约 {(stored_bytes + reserve_bytes - available) / 2**30:.1f}GiB "
                "可能由系统虚拟内存承载，推理速度会明显降低",
                flush=True,
            )
        workers = max(1, int(os.environ.get("CCCP_LOAD_WORKERS", "12")))
        started = time.perf_counter()
        print(
            f"[cccp-packed] 紧凑专家常驻 RAM：{len(keys)} 个，"
            f"文件约 {stored_bytes / 2**30:.1f}GiB，workers={workers}",
            flush=True,
        )
        print(
            "[cccp-winui-progress] phase=experts "
            f"current=0 total={len(keys)}",
            flush=True,
        )
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="cccp-packed-load",
        ) as executor:
            # Emit roughly one hundred UI updates for both small profiles and
            # full archives.  The old fixed 2,000-expert interval meant a
            # 1,856-expert profile stayed at 0% until it was completely loaded.
            progress_step = max(1, len(keys) // 100)
            log_step = max(1, len(keys) // 10)
            futures = {
                executor.submit(self._load_one, *key): key
                for key in keys
            }
            for index, future in enumerate(as_completed(futures), 1):
                self.pinned[futures[future]] = future.result()
                if index % log_step == 0:
                    print(
                        f"[cccp-packed] 紧凑专家常驻 "
                        f"{index}/{len(keys)}",
                        flush=True,
                    )
                if index % progress_step == 0:
                    print(
                        "[cccp-winui-progress] phase=experts "
                        f"current={index} total={len(keys)}",
                        flush=True,
                    )
        # 所有运行时专家都只引用 BF16 码本；释放 store 的 FP32 中间副本。
        self.store._cb_cache.clear()
        gc.collect()
        self.ram_bytes = self.host_expert_bytes
        print(
            f"[cccp-packed] 紧凑专家 RAM 常驻完成："
            f"{len(self.pinned)} 个 / {self.ram_bytes / 2**30:.1f}GiB，"
            f"{time.perf_counter() - started:.1f}s；运行期零磁盘读",
            flush=True,
        )
        print(
            "[cccp-winui-progress] phase=experts "
            f"current={len(keys)} total={len(keys)}",
            flush=True,
        )
        return True

    def preload_pinned(self) -> None:
        if not self.preload_all():
            raise RuntimeError(
                "packed hybrid currently requires all experts in RAM"
            )

    def pin_host_resident(self, budget_gb: float | None = None) -> float:
        """Register compact resident payloads in place for direct DMA.

        ``pin_memory()`` would allocate a second copy of a several-hundred-GiB
        expert archive. ``cudaHostRegister`` page-locks the existing bytearray
        storage instead: disk/RAM/VRAM all retain the original packed width and
        ``PinnedStage`` can bypass its small rotating bounce buffers.
        """
        if not self.pinned:
            return 0.0
        total = sum(
            weight.raw.nbytes
            for expert in self.pinned.values()
            for weight in expert
        )
        if torch.version.hip is not None:
            # Windows ROCm can accept cudaHostRegister through its CUDA
            # compatibility surface, but torch HIP later rejects copy_ from
            # those externally registered tensor addresses with
            # hipErrorInvalidValue.  Do not leave the pool in that poisoned
            # half-registered state.  Capacity-fit AMD launches use the
            # per-layer full-resident pool; the bounded hybrid fallback uses
            # only PyTorch-owned pinned staging buffers.
            self.host_dma_mode = "hip-pinned-stage"
            print(
                "[cccp-packed] AMD/HIP 跳过外部 RAM 注册；"
                "混合回退使用 PyTorch 原生锁页暂存",
                flush=True,
            )
            print(
                "[cccp-dma] mode=hip-pinned-stage "
                "external_host_register=disabled；"
                "容量足够时由全显存常驻路径消除运行期 H2D",
                flush=True,
            )
            print(
                "[cccp-winui-progress] phase=expert-pin current=1 total=1",
                flush=True,
            )
            return 0.0
        direct_mode = os.environ.get(
            "CCCP_WDDM_DIRECT_PIN", "auto"
        ).strip().lower()
        if direct_mode in {"0", "false", "off", "no"}:
            print(
                "[cccp-packed] RAM 专家原地锁页已显式关闭；"
                "传输将回退到连续 pinned ring",
                flush=True,
            )
            print(
                "[cccp-winui-progress] phase=expert-pin current=1 total=1",
                flush=True,
            )
            return 0.0
        if budget_gb is None:
            raw = os.environ.get("CCCP_HOST_PIN_GB", "auto").strip().lower()
            if raw in ("", "auto"):
                # Register existing expert pages up to the real RAM budget.
                # This is not a device allocation, so normal automatic mode is
                # intentionally not derived from VRAM capacity.  A non-zero
                # multiplier remains available only for driver diagnostics.
                device_bytes = torch.cuda.get_device_properties(
                    self.device
                ).total_memory
                configured_caps = tuple(
                    cap
                    for cap in (
                        max(
                            0.0,
                            float(os.environ.get("CCCP_VRAM_LIMIT_GB", "0")),
                        ),
                        max(
                            0.0,
                            float(
                                os.environ.get(
                                    "CCCP_EXTREME_VRAM_CAP_GB",
                                    "0",
                                )
                            ),
                        ),
                    )
                    if cap > 0
                )
                if configured_caps:
                    device_bytes = min(
                        device_bytes,
                        int(min(configured_caps) * 2**30),
                    )
                multiplier = max(
                    0.0,
                    float(
                        os.environ.get(
                            "CCCP_HOST_PIN_VRAM_MULTIPLIER",
                            "0",
                        )
                    ),
                )
                import psutil

                available = psutil.virtual_memory().available
                budget, host_floor = automatic_host_pin_budget(
                    payload_bytes=total,
                    available_ram_bytes=available,
                    device_bytes=device_bytes,
                    driver_multiplier=multiplier,
                )
                print(
                    "[cccp-packed] 紧凑专家原地锁页自动预算："
                    f"{budget / 2**30:.1f}GiB / "
                    f"总量 {total / 2**30:.1f}GiB（映射上限="
                    f"{'RAM安全余量' if multiplier <= 0 else f'VRAM×{multiplier:g}'}；"
                    f"保留可分页RAM {host_floor / 2**30:.1f}GiB）",
                    flush=True,
                )
            else:
                budget = max(0, int(float(raw) * 2**30))
        else:
            budget = max(0, int(float(budget_gb) * 2**30))
        if budget <= self._host_pinned_bytes:
            return self._host_pinned_bytes / 2**30
        if budget <= 0:
            print(
                "[cccp-packed] 可用 RAM 未达到原地锁页安全余量；"
                "传输将回退到连续 pinned ring",
                flush=True,
            )
            print(
                "[cccp-winui-progress] phase=expert-pin current=1 total=1",
                flush=True,
            )
            return 0.0

        started = time.perf_counter()
        pinned_experts = 0
        registered_tensors = 0
        stop = False
        mib = 2**20
        progress_total = max(1, (budget + mib - 1) // mib)
        progress_step = max(1, progress_total // 100)
        next_progress = progress_step
        print(
            "[cccp-winui-progress] phase=expert-pin "
            f"current=0 total={progress_total}",
            flush=True,
        )
        cudart = torch.cuda.cudart()
        # Cycle through every layer for the same expert id before moving on.
        # A partial budget therefore accelerates all layers instead of pinning
        # only a shallow contiguous prefix of the network.
        ordered_keys = sorted(
            self.pinned,
            key=lambda key: (key[1], key[0]),
        )
        for key in ordered_keys:
            expert = self.pinned[key]
            combined = _contiguous_expert_raw(expert)
            sources = (
                (combined,)
                if combined is not None
                else tuple(weight.raw for weight in expert)
            )
            for source in sources:
                pointer = source.data_ptr()
                if pointer in self._host_registrations or source.is_pinned():
                    continue
                if self._host_pinned_bytes + source.nbytes > budget:
                    stop = True
                    break
                error = cudart.cudaHostRegister(
                    pointer,
                    source.nbytes,
                    0,
                )
                error_code = getattr(error, "value", None)
                if error_code is None:
                    error_code = int(error)
                if error_code != 0:
                    try:
                        error_name = cudart.cudaGetErrorString(error)
                    except (AttributeError, RuntimeError, TypeError):
                        error_name = error
                    # cudaHostRegister reports an ordinary return code through
                    # cudart, so the loop can degrade to pageable RAM.  CUDA also
                    # records the same failure as the thread's last error; if it
                    # is not consumed, the next unrelated tensor operation sees
                    # a misleading asynchronous OOM.  Clear only that runtime
                    # status and preserve every successful registration.
                    try:
                        cudart.cudaGetLastError()
                    except (AttributeError, RuntimeError, TypeError):
                        pass
                    print(
                        "[cccp-packed] 紧凑专家原地锁页停止："
                        f"{self._host_pinned_bytes / 2**30:.1f}GiB，"
                        f"cudaHostRegister={error_name}",
                        flush=True,
                    )
                    stop = True
                    break
                self._host_registrations[pointer] = source.nbytes
                self._host_pinned_bytes += source.nbytes
                registered_tensors += 1
                current_progress = min(
                    progress_total,
                    self._host_pinned_bytes // mib,
                )
                if current_progress >= next_progress:
                    print(
                        "[cccp-winui-progress] phase=expert-pin "
                        f"current={current_progress} total={progress_total}",
                        flush=True,
                    )
                    next_progress = current_progress + progress_step
            pinned_experts += 1
            if pinned_experts % 2000 == 0:
                print(
                    "[cccp-packed] 紧凑专家原地锁页 "
                    f"{pinned_experts}/{len(self.pinned)} "
                    f"({self._host_pinned_bytes / 2**30:.1f}GiB)",
                    flush=True,
                )
            if stop:
                break
        print(
            "[cccp-packed] 紧凑专家原地锁页完成："
            f"{registered_tensors} 个 packed 张量 / "
            f"{self._host_pinned_bytes / 2**30:.1f}GiB / "
            f"{time.perf_counter() - started:.1f}s；直接异步 DMA",
            flush=True,
        )
        direct_complete = self._host_pinned_bytes >= total
        if direct_complete:
            self.host_dma_mode = "direct-registered"
            print(
                "[cccp-dma] mode=direct-registered cpu_bridge=disabled "
                f"locked={self._host_pinned_bytes / 2**30:.2f}GiB；"
                "GPU 直接从锁页专家 RAM 发起异步 DMA，不经过 CPU 中转缓冲",
                flush=True,
            )
        else:
            self.host_dma_mode = "mixed"
            print(
                "[cccp-dma] mode=mixed cpu_bridge=fallback-for-unlocked "
                f"locked={self._host_pinned_bytes / 2**30:.2f}GiB "
                f"pageable={(total - self._host_pinned_bytes) / 2**30:.2f}GiB；"
                "已锁页部分直接 DMA，仅未锁页部分使用 CPU 中转缓冲",
                flush=True,
            )
        final_progress = (
            progress_total
            if direct_complete
            else min(progress_total, self._host_pinned_bytes // mib)
        )
        print(
            "[cccp-winui-progress] phase=expert-pin "
            f"current={final_progress} "
            f"total={progress_total}",
            flush=True,
        )
        return self._host_pinned_bytes / 2**30

    def _safe_budget(self) -> int:
        allocated = torch.cuda.memory_allocated(self.device)
        self.fixed_gpu_bytes_before_arena = int(allocated)
        free, total = torch.cuda.mem_get_info(self.device)
        safety_reserve = float(os.environ.get("CCCP_VRAM_RESERVE_GB", "1"))
        runtime_headroom = float(os.environ.get(
            "CCCP_VRAM_HEADROOM_GB",
            str(safety_reserve),
        ))
        reserve = int(max(safety_reserve, runtime_headroom) * 2**30)
        self.vram_safety_reserve_bytes = int(safety_reserve * 2**30)
        self.vram_runtime_headroom_bytes = reserve
        index = self.device.index
        if index is None:
            index = torch.cuda.current_device()
        try:
            fraction = torch.cuda.get_per_process_memory_fraction(index)
        except (AttributeError, RuntimeError):
            fraction = 1.0
        process_limit = int(total * fraction)
        # On Windows/WDDM a device can expose allocatable shared system memory
        # after physical VRAM is exhausted.  ``mem_get_info`` alone is
        # therefore not a physical-residency ceiling.  Honour the launcher's
        # explicit whole-process cap independently of allocator reporting so
        # the fixed expert arena cannot grow into shared GPU memory.
        configured_limits = []
        for name in ("CCCP_VRAM_LIMIT_GB", "CCCP_EXTREME_VRAM_CAP_GB"):
            try:
                configured = float(os.environ.get(name, "0"))
            except (TypeError, ValueError):
                configured = 0.0
            if configured > 0:
                configured_limits.append(int(configured * 2**30))
        if configured_limits:
            process_limit = min(process_limit, min(configured_limits))
        process_room = max(0, process_limit - allocated - reserve)
        device_room = max(0, free - reserve)
        selected = max(0, min(self.budget, process_room, device_room))
        hip_single_arena_cap = 0
        if os.name == "nt" and torch.version.hip is not None:
            # This is the largest monolithic hybrid arena already exercised
            # successfully on gfx1150.  Larger 7.98/26-GiB allocations fail
            # despite ample aggregate VRAM because Windows ROCm applies a
            # much smaller single-allocation ceiling.  This is not tunable:
            # capacity-fit configurations use the per-layer full-resident
            # pool, while hybrid remains a bounded compatibility fallback.
            hip_single_arena_cap = int(1.5 * 2**30)
            selected = min(selected, hip_single_arena_cap)
        print(
            "[cccp-vram-plan] phase=packed-arena "
            f"requested={self.budget / 2**30:.2f}GiB "
            f"allocated_before={allocated / 2**30:.2f}GiB "
            f"driver_free={free / 2**30:.2f}GiB "
            f"physical_total={total / 2**30:.2f}GiB "
            f"process_limit={process_limit / 2**30:.2f}GiB "
            f"safety_reserve={safety_reserve:.2f}GiB "
            f"runtime_headroom={reserve / 2**30:.2f}GiB "
            f"process_room={process_room / 2**30:.2f}GiB "
            f"device_room={device_room / 2**30:.2f}GiB "
            f"selected={selected / 2**30:.2f}GiB"
            + (
                f" hip_single_arena_cap="
                f"{hip_single_arena_cap / 2**30:.2f}GiB"
                if hip_single_arena_cap
                else ""
            ),
            flush=True,
        )
        return selected

    def build_gpu_arenas(self) -> float:
        if self._arenas is not None:
            return self._arenas.nbytes / 2**30
        if not self.pinned and self._extreme_specs is None:
            return 0.0
        safe_budget = self._safe_budget()
        if safe_budget <= 0:
            raise RuntimeError("packed hybrid has no safe GPU cache room")
        counts = (
            Counter(self._extreme_specs)
            if self._extreme_specs is not None
            else Counter(
                PackedExpertSignature.of(expert)
                for expert in self.pinned.values()
            )
        )
        host_codebooks = {
            codebook.data_ptr(): codebook
            for codebook in self._host_codebooks.values()
        }
        codebook_bytes = sum(
            codebook.nbytes
            for codebook in host_codebooks.values()
        ) if self._resident_codebooks else 0
        by_layer: dict[int, Counter[PackedExpertSignature]] = {}
        if self.pinned:
            for (layer, _expert_id), expert in self.pinned.items():
                signature = PackedExpertSignature.of(expert)
                by_layer.setdefault(int(layer), Counter())[signature] += 1
        layer_envelope = {
            signature: max(
                layer_counts.get(signature, 0)
                for layer_counts in by_layer.values()
            )
            for signature in counts
        } if by_layer else {}
        envelope_bytes = sum(
            signature.storage_bytes(self._resident_codebooks) * count
            for signature, count in layer_envelope.items()
        )
        if (
            self._arena_phase == "prefill"
            and self._extreme_specs is None
            and layer_envelope
        ):
            self._decode_arena_target_budget = max(
                self._decode_arena_target_budget,
                safe_budget,
            )
            prefill_budget = min(
                safe_budget,
                max(codebook_bytes + envelope_bytes, 512 * 2**20),
            )
            self._prefill_arena_target_budget = prefill_budget
            if prefill_budget < safe_budget:
                print(
                    "[cccp-vram-plan] phase=prefill-arena "
                    f"compact_layer={envelope_bytes / 2**30:.2f}GiB "
                    f"codebooks={codebook_bytes / 2**30:.2f}GiB "
                    f"selected={prefill_budget / 2**30:.2f}GiB "
                    f"decode_target={safe_budget / 2**30:.2f}GiB",
                    flush=True,
                )
                safe_budget = prefill_budget
        arena_budget = safe_budget - codebook_bytes
        if arena_budget <= 0:
            raise RuntimeError(
                "packed GPU cache cannot fit resident codebooks"
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
        topk_minimum = {
            signature: min(int(count), top_k)
            for signature, count in counts.items()
        }
        topk_minimum_bytes = sum(
            signature.storage_bytes(self._resident_codebooks) * count
            for signature, count in topk_minimum.items()
        )
        self.max_expert_slot_bytes = max(
            signature.storage_bytes(self._resident_codebooks)
            for signature in counts
        )
        # Public minimum: one complete model-native Top-K route measured with
        # the largest expert layout. Heterogeneous archives may reserve a few
        # more physical slots per signature for correctness, but never less
        # than this architecture-independent byte floor.
        self.topk_staging_bytes = top_k * self.max_expert_slot_bytes
        self._adaptive_decode_arena = topk_minimum_bytes > arena_budget
        minimum_slots: int | Mapping[PackedExpertSignature, int] = (
            {
                # Small-VRAM cards cannot afford speculative decode slots for
                # every precision signature.  Keep only one bootstrap view per
                # signature; the exact current route is fitted atomically at
                # the layer boundary by _ensure_decode_route_capacity_locked.
                signature: min(int(count), 1)
                for signature, count in counts.items()
            }
            if self._adaptive_decode_arena
            else topk_minimum
        )
        if self._adaptive_decode_arena:
            print(
                "[cccp-packed] 显存优先留给整批 Prefill；"
                "decode 专家槽按当前层路由动态重分配 "
                f"({topk_minimum_bytes / 2**30:.2f}GiB Top-K 全精度预留 > "
                f"{arena_budget / 2**30:.2f}GiB 可用)",
                flush=True,
            )
        if self.pinned:
            # Prefill is layer-first.  Reserve enough slots per exact packed
            # signature for every expert in the largest single layer, rather
            # than merely one Top-K route.  When this layer envelope fits the
            # existing arena budget, a 4096-token outer block can use bounded
            # row micro-batches without repeatedly halving on rare precision
            # classes.  No expert is expanded and the total VRAM budget is
            # unchanged.
            if envelope_bytes <= arena_budget:
                minimum_slots = layer_envelope
        specs = (
            dict(self._extreme_specs)
            if self._extreme_specs is not None
            else allocate_packed_slots(
                counts,
                arena_budget,
                minimum_slots,
                weights=weights,
                resident_codebooks=self._resident_codebooks,
            )
        )
        required = sum(
            signature.storage_bytes(self._resident_codebooks) * count
            for signature, count in specs.items()
        )
        if required > arena_budget:
            raise RuntimeError(
                "极限模式固定专家槽超过安全显存预算："
                f"{required / 2**30:.2f} > {arena_budget / 2**30:.2f} GiB"
            )
        # Keep one empty model-native Top-K window only during startup warmup.
        # Once requests begin, every slot joins the same strict LRU and the
        # current route is protected only by its short in-flight lease.
        warmup_free_minimum = {
            signature: min(
                int(specs.get(signature, 0)),
                int(topk_minimum.get(signature, 0)),
            )
            for signature in specs
        }
        self.initial_free_slots = sum(warmup_free_minimum.values())
        self.profile_hot_keys = self._plan_profile_hot_keys(
            specs,
            warmup_free_minimum,
        )
        self.profile_hot_slots = len(self.profile_hot_keys)
        self.profile_hot_cache_enabled = bool(self.profile_hot_keys)
        self.budget = max(safe_budget, self._decode_arena_target_budget)
        self._default_arena_specs = dict(specs)
        self._default_arena_covers_all_pinned = None
        self._prefill_partition_layer = None
        self._arenas = _PackedArenas(
            specs,
            self.device,
            resident_codebooks=self._resident_codebooks,
        )
        if self._resident_codebooks:
            self._device_codebooks = {
                pointer: codebook.to(
                    device=self.device,
                    dtype=torch.bfloat16,
                    non_blocking=False,
                )
                for pointer, codebook in host_codebooks.items()
            }
        self._prepare_native8_codebooks()
        self.arena_slots = {}
        for signature, count in specs.items():
            tier = self._signature_tier(signature)
            self.arena_slots[tier] = (
                self.arena_slots.get(tier, 0) + count
            )

        # Projection-VQ manifests normalize Kimi and DeepSeek dimensions here.
        # Kimi has a separate routed latent width while DeepSeek routes the
        # model hidden state directly, so model-specific fields must not leak
        # into the shared packed arena.
        intermediate = int(self.store.cfg["moe_inter"])
        hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        self._route_ids = torch.arange(
            top_k,
            dtype=torch.long,
            device=self.device,
        )
        self._route_order_identity = True
        self._ordered_weights = torch.empty(
            top_k,
            dtype=torch.float32,
            device=self.device,
        )
        projection_count = next(iter(specs)).projection_count
        static_tile_view = (
            projection_count == 3
            and os.environ.get("CCCP_PROJECTION_TILE_VIEW", "0") == "1"
        )
        self._metadata = torch.empty(
            (
                runtime_metadata_row_count(projection_count)
                if static_tile_view
                else projection_count * 5
            ),
            top_k,
            dtype=torch.long,
            device=self.device,
        )
        metadata_rows = int(self._metadata.shape[0])
        self._slot_directory = torch.zeros(
            int(self.store.cfg["n_layers"]),
            int(self.store.cfg["n_experts"]),
            metadata_rows,
            dtype=torch.long,
            device=self.device,
        )
        self._slot_update_host = torch.empty(
            metadata_rows,
            dtype=torch.long,
        )
        self._slot_scale_directory = torch.zeros(
            int(self.store.cfg["n_layers"]),
            int(self.store.cfg["n_experts"]),
            3,
            dtype=torch.float32,
            device=self.device,
        )
        self._slot_scale_update_host = torch.empty(3, dtype=torch.float32)
        self._native8_route_scales = torch.empty(
            top_k,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        self._native8_route_scales_host = torch.empty(
            top_k,
            3,
            dtype=torch.float32,
        )
        self._route_hit_mask = torch.empty(
            top_k,
            dtype=torch.bool,
            device=self.device,
        )
        self._route_all_hit = torch.empty(
            (),
            dtype=torch.bool,
            device=self.device,
        )
        self._route_all_hit_host = torch.empty(
            (),
            dtype=torch.bool,
            pin_memory=True,
        )
        self._route_host_ids = torch.empty(
            top_k,
            dtype=torch.long,
            pin_memory=True,
        )
        self._route_copy_done = torch.cuda.Event()
        self._metadata_host = torch.empty(
            self._metadata.shape,
            dtype=torch.long,
        )
        self._workspaces = (
            torch.empty(
                top_k,
                2 * intermediate,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.empty(
                top_k,
                hidden,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.empty(hidden, dtype=torch.float32, device=self.device),
        )
        self.bytes = self.gpu_storage_bytes
        detail = ", ".join(
            f"{tier}={count}"
            for tier, count in sorted(self.arena_slots.items())
        )
        print(
            f"[cccp-packed] 紧凑专家固定显存槽："
            f"{sum(specs.values())} 个 / "
            f"{self.gpu_arena_bytes / 2**30:.2f}GiB（{detail}；"
            f"码本={'全局常驻' if self._resident_codebooks else '随槽复制'}）",
            flush=True,
        )
        print(
            "[cccp-cache-plan] "
            f"arena_slots={sum(specs.values())} "
            f"warmup_hot_slots={self.profile_hot_slots} "
            f"initial_free_slots={self.initial_free_slots} "
            "policy=strict-lru "
            "permanent_protection=0 prefetch=off "
            f"status={'按语料热度预热，运行期全槽末位淘汰' if self.profile_hot_cache_enabled else '运行期全槽末位淘汰'}",
            flush=True,
        )
        planned_hot_bytes = sum(
            weight.raw.nbytes
            for key in self.profile_hot_keys
            for weight in self.pinned.get(key, ())
        )
        print(
            "[cccp-residency-plan] "
            f"dense_and_fixed={self.fixed_gpu_bytes_before_arena / 2**30:.2f}GiB "
            f"safety={self.vram_safety_reserve_bytes / 2**30:.2f}GiB "
            f"runtime_headroom={self.vram_runtime_headroom_bytes / 2**30:.2f}GiB "
            f"prefill_layer_max={envelope_bytes / 2**30:.2f}GiB "
            f"codebooks={codebook_bytes / 2**30:.2f}GiB "
            f"largest_expert={self.max_expert_slot_bytes / 2**20:.2f}MiB "
            f"model_topk={top_k} "
            f"topk_largest_floor={self.topk_staging_bytes / 2**30:.2f}GiB "
            f"topk_signature_slots={sum(topk_minimum.values())} "
            f"topk_signature_bytes={topk_minimum_bytes / 2**30:.2f}GiB "
            f"decode_arena={self.gpu_arena_bytes / 2**30:.2f}GiB "
            f"hot_experts={self.profile_hot_slots} "
            f"hot_start_bytes={planned_hot_bytes / 2**30:.2f}GiB",
            flush=True,
        )
        return self.gpu_arena_bytes / 2**30

    def _plan_profile_hot_keys(
        self,
        specs: Mapping[PackedExpertSignature, int],
        dynamic_reserve_minimum: Mapping[PackedExpertSignature, int],
    ) -> tuple[tuple[int, int], ...]:
        """Use every slot outside the native Top-K staging set for hot experts.

        ``counts`` in a trained profile are the measured route observations,
        not merely a per-layer ordering hint.  Selecting round-robin by layer
        gave a cold expert in a flat layer the same residency priority as a
        globally dominant expert.  Sort the complete profile by measured hit
        count instead, while respecting each immutable packed-signature arena.
        The startup reserve is calculated before this method so warmup leaves
        room for one model-native Top-K route.  After warmup all slots join the
        same strict LRU; no hot entry is permanently protected.
        """

        if self._adaptive_decode_arena or self._extreme_specs is not None:
            return ()
        ranks = self.store.heat_ranks or {}
        if not ranks:
            return ()
        totals = Counter(
            PackedExpertSignature.of(expert) for expert in self.pinned.values()
        )
        all_pinned_fit = bool(
            getattr(self.store, "route_allowlist", None) is not None
            and all(
                int(specs.get(signature, 0)) >= int(count)
                for signature, count in totals.items()
            )
        )
        remaining = {
            signature: max(
                0,
                int(count)
                - (
                    0
                    if all_pinned_fit
                    else min(
                        int(count),
                        int(dynamic_reserve_minimum.get(signature, 0)),
                    )
                ),
            )
            for signature, count in specs.items()
        }
        heat_counts = getattr(self.store, "heat_counts", None) or {}
        candidates: list[
            tuple[float, int, int, tuple[int, int], PackedExpertSignature]
        ] = []
        for layer in sorted(int(value) for value in ranks):
            layer_counts = heat_counts.get(layer, {})
            for rank, expert_id in enumerate(ranks.get(layer, ())):
                key = (layer, int(expert_id))
                expert = self.pinned.get(key)
                if expert is None:
                    continue
                # Profiles written before the raw count was attached are not
                # a supported file format, but rank fallback keeps synthetic
                # stores used by diagnostics deterministic.
                score = float(layer_counts.get(int(expert_id), -rank))
                candidates.append((
                    -score,
                    layer,
                    int(expert_id),
                    key,
                    PackedExpertSignature.of(expert),
                ))
        candidates.sort()
        output: list[tuple[int, int]] = []
        for _negative_score, _layer, _expert_id, key, signature in candidates:
            if remaining.get(signature, 0) <= 0:
                continue
            output.append(key)
            remaining[signature] -= 1
            if not any(value > 0 for value in remaining.values()):
                break
        return tuple(output)

    def _warm_profile_hot_locked(self) -> None:
        """Seed strict LRU from corpus heat without permanent protection."""

        if (
            not self.profile_hot_cache_enabled
            or self._profile_hot_ready
            or self._arenas is None
        ):
            return
        started = time.perf_counter()
        # The planner stores hottest first. Load in reverse so the hottest
        # entries finish at the MRU end and the first miss evicts the coldest
        # warmup entry, not the most valuable one.
        selected = self._ensure_locked(
            list(reversed(self.profile_hot_keys)),
            prefetch=True,
        )
        selected_keys = frozenset(selected)
        warmed_bytes = sum(
            weight.raw.nbytes
            for key in selected_keys
            for weight in self.pinned[key]
        )
        del selected
        self._profile_hot_ready = True
        self._compact_profile_all_resident = (
            len(self.cache) == len(self.pinned)
        )
        print(
            "[cccp-cache] lru-hot-start="
            f"{len(selected_keys)} experts / "
            f"{warmed_bytes / 2**30:.2f}GiB "
            f"elapsed={time.perf_counter() - started:.2f}s；"
            "permanent_protection=0；所有槽按命中顺序末位淘汰；"
            f"RAM专家保持完整={self.host_expert_bytes / 2**30:.2f}GiB",
            flush=True,
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
        host_raw = _contiguous_expert_raw(host)
        device_raw = _contiguous_expert_raw(device)
        if host_raw is not None and device_raw is not None:
            pairs = [(host_raw, device_raw)]
        if not self._resident_codebooks:
            pairs.extend(
                (source.cb, target.cb)
                for source, target in zip(host, device)
            )
        return pairs

    def _ensure_locked(
        self,
        keys: list[tuple[int, int]],
        *,
        prefetch: bool,
        defer_wait: bool = False,
    ) -> dict[tuple[int, int], DeviceExpert]:
        if self._arenas is None:
            raise RuntimeError("packed GPU arenas are not initialized")
        self.transfer_seconds = self._stage.collect_timing()
        keys = list(dict.fromkeys(keys))
        output: dict[tuple[int, int], DeviceExpert] = {}
        missing: list[tuple[int, int]] = []
        inflight: list[tuple[int, int]] = []
        with self._lock:
            for key in keys:
                value = self.cache.get(key)
                if value is None:
                    missing.append(key)
                    continue
                self.cache.move_to_end(key)
                self._arenas.touch(key)
                self._arenas.mark_inflight(key)
                inflight.append(key)
                output[key] = value
                if prefetch:
                    self.prefetch_hits += 1
                else:
                    self.hits += 1
        if not missing:
            self.last_transfer_seconds = 0.0
            with self._lock:
                for key in inflight:
                    self._arenas.clear_inflight(key)
            return output

        started = time.perf_counter()
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        staged: list[tuple[tuple[int, int], DeviceExpert]] = []
        with self._lock:
            for key in missing:
                # 可能已被前一个等待者装入。
                value = self.cache.get(key)
                if value is not None:
                    self.cache.move_to_end(key)
                    self._arenas.touch(key)
                    self._arenas.mark_inflight(key)
                    inflight.append(key)
                    output[key] = value
                    continue
                host = self.pinned.get(key)
                if host is None:
                    raise KeyError(f"packed RAM expert missing: {key}")
                replaced, value = self._arenas.lease(
                    key,
                    host,
                    self._device_codebooks,
                )
                self._arenas.mark_inflight(key)
                inflight.append(key)
                if replaced is not None:
                    self.cache.pop(replaced, None)
                    self._set_slot_directory(replaced, None)
                pairs.extend(self._copy_pairs(host, value))
                staged.append((key, value))
        if pairs:
            self._stage.upload_batch(pairs)
            if not defer_wait:
                self._stage.last.synchronize()
            uploaded = sum(source.nbytes for source, _target in pairs)
            with self._lock:
                for key, value in staged:
                    self.cache[key] = value
                    output[key] = value
                    self._set_slot_directory(key, value)
                    if not prefetch:
                        self.miss += 1
                self.uploaded_bytes += uploaded
        elapsed = time.perf_counter() - started
        self.last_transfer_seconds = (
            0.0 if defer_wait and pairs else elapsed
        )
        self.transfer_seconds = self._stage.collect_timing()
        with self._lock:
            for key in inflight:
                self._arenas.clear_inflight(key)
        return output

    def collect_transfer_timing(self, *, synchronize: bool = False) -> float:
        self.transfer_seconds = self._stage.collect_timing(
            synchronize=synchronize,
        )
        return self.transfer_seconds

    def _ensure(
        self,
        keys: list[tuple[int, int]],
        *,
        prefetch: bool,
    ) -> dict[tuple[int, int], DeviceExpert]:
        with self._transfer_lock:
            return self._ensure_locked(keys, prefetch=prefetch)

    def prefetch(self, keys: list[tuple[int, int]]) -> None:
        if not keys or os.environ.get("CCCP_PREFETCH_STAGE", "1") == "0":
            return
        with self._lock:
            self._prefetch_futures = {
                future
                for future in self._prefetch_futures
                if not future.done()
            }
            # 模型在 token 开始时按层提交约 92 个请求；单线程执行保证
            # 槽位顺序，队列本身必须能容纳一整轮，否则只会预取浅层。
            if len(self._prefetch_futures) >= 128:
                return
            if all(key in self.cache for key in keys):
                return
            future = self._prefetch_executor.submit(
                self._ensure,
                list(keys),
                prefetch=True,
            )
            self._prefetch_futures.add(future)

    def get_many(
        self,
        keys: list[tuple[int, int]],
    ) -> dict[tuple[int, int], DeviceExpert]:
        return self._ensure(keys, prefetch=False)

    def last_expert_ids(self, layer: int) -> list[int]:
        return self._last_ids[layer]

    def route_tier_profile(self) -> dict[str, object]:
        """Summarize measured routes by packed capability tier.

        This is computed only when a CLI asks for final metrics; the decode
        path keeps recording the existing ``(layer, expert)`` counter and
        pays no additional per-route signature work.
        """

        observations: Counter[str] = Counter()
        unique: dict[str, set[tuple[int, int]]] = {}
        for key, count in self.route_counts.items():
            expert = self.pinned.get(key)
            if expert is None:
                continue
            tier = self._signature_tier(PackedExpertSignature.of(expert))
            observations[tier] += int(count)
            unique.setdefault(tier, set()).add(key)
        return {
            "observations": dict(sorted(observations.items())),
            "unique_experts": {
                tier: len(keys) for tier, keys in sorted(unique.items())
            },
            "arena_slots": dict(sorted(self.arena_slots.items())),
        }

    @staticmethod
    def _metadata_rows(experts: list[DeviceExpert]) -> list[list[int]]:
        return build_runtime_metadata_rows(experts)

    def _prepare_native8_codebooks(self) -> None:
        """Quantize each small shared codebook once for Tensor Core execution."""

        self._native8_codebooks = {}
        self._native8_codebook_scales = {}
        self._native8_prefill_enabled = False
        self.native8_rows_supported = False
        if (
            self.device.type != "cuda"
            or torch.version.hip is not None
            or not self._resident_codebooks
            or not hasattr(torch, "_scaled_grouped_mm")
            or os.environ.get("CCCP_PROJECTION_TILE_VIEW", "0") == "1"
        ):
            return
        major, minor = torch.cuda.get_device_capability(self.device)
        if (int(major), int(minor)) < (8, 9):
            # SM75/80/86 use the same native8 cache layout, but their INT8
            # grouped short-batch executor is supplied by cuBLASLt/CUTLASS.
            return
        for codebook in self._device_codebooks.values():
            pointer = int(codebook.data_ptr())
            scale = max(float(codebook.abs().amax().item()) / 448.0, 1.0e-12)
            quantized = (
                codebook.float()
                .div(scale)
                .clamp(-448.0, 448.0)
                .to(torch.float8_e4m3fn)
                .contiguous()
            )
            self._native8_codebooks[pointer] = quantized
            self._native8_codebook_scales[pointer] = scale
        self._native8_prefill_enabled = bool(self._native8_codebooks)
        self.native8_rows_supported = self._native8_prefill_enabled
        if self._native8_prefill_enabled:
            self.prefill_executor = "cuda.vq-to-e4m3-scaled-grouped-gemm"
            print(
                "[cccp-native8] shared codebooks=E4M3; "
                f"count={len(self._native8_codebooks)}; "
                "runtime reconstruction=index-unpack+aligned-copy; "
                "compute=Tensor-Core scaled-grouped-GEMM",
                flush=True,
            )

    def _native8_metadata_rows(
        self,
        experts: list[DeviceExpert],
    ) -> tuple[list[list[int]], list[tuple[float, ...]]]:
        """Replace BF16 codebook pointers with native8 pointers and scales."""

        rows = self._metadata_rows(experts)[:15]
        scales: list[tuple[float, ...]] = []
        for expert in experts:
            expert_scales: list[float] = []
            for projection, weight in enumerate(expert):
                pointer = int(weight.cb.data_ptr())
                quantized = self._native8_codebooks.get(pointer)
                scale = self._native8_codebook_scales.get(pointer)
                if quantized is None or scale is None:
                    raise RuntimeError(
                        "native8 shared codebook cache is incomplete"
                    )
                rows[projection * 5 + 1][len(scales)] = int(
                    quantized.data_ptr()
                )
                expert_scales.append(float(scale))
            scales.append(tuple(expert_scales))
        return rows, scales

    def _copy_metadata(self, experts: list[DeviceExpert]) -> None:
        """Publish tiny route metadata before the packed kernel consumes it.

        This buffer is reused by every layer.  A pinned, non-blocking H2D copy
        allowed Python to overwrite it for layer N+1 while DMA for layer N was
        still reading it, which paired correct packed bytes with the wrong
        pointers/shapes and produced fluent-looking garbage tokens.  The copy
        is under one KiB for normal Top-K routes, so a blocking pageable copy is
        the correct fixed-address boundary.
        """
        count = len(experts)
        rows = self._metadata_rows(experts)[: self._metadata.shape[0]]
        host = getattr(self, "_metadata_host", None)
        if host is None or host.shape[0] != len(rows) or host.shape[1] < count:
            host = torch.empty(len(rows), count, dtype=torch.long)
        host[:, :count].copy_(torch.tensor(rows, dtype=torch.long))
        self._metadata[:, :count].copy_(
            host[:, :count],
            non_blocking=False,
        )

    def _copy_native8_metadata(
        self,
        experts: list[DeviceExpert],
    ) -> torch.Tensor:
        """Publish Top-K packed pointers with prequantized codebook scales."""

        count = len(experts)
        rows, scales = self._native8_metadata_rows(experts)
        rows = rows[: self._metadata.shape[0]]
        host = self._metadata_host
        if host is None or host.shape[0] != len(rows) or host.shape[1] < count:
            host = torch.empty(len(rows), count, dtype=torch.long)
            self._metadata_host = host
        host[:, :count].copy_(torch.tensor(rows, dtype=torch.long))
        self._metadata[:, :count].copy_(host[:, :count], non_blocking=False)

        scale_host = getattr(self, "_native8_route_scales_host", None)
        scale_device = getattr(self, "_native8_route_scales", None)
        if scale_host is None or int(scale_host.shape[0]) < count:
            scale_host = torch.empty(count, 3, dtype=torch.float32)
            self._native8_route_scales_host = scale_host
        if scale_device is None or int(scale_device.shape[0]) < count:
            scale_device = torch.empty(
                count,
                3,
                dtype=torch.float32,
                device=self.device,
            )
            self._native8_route_scales = scale_device
        scale_host[:count].copy_(torch.tensor(scales, dtype=torch.float32))
        scale_device[:count].copy_(scale_host[:count], non_blocking=False)
        return scale_device[:count]

    def _set_slot_directory(
        self,
        key: tuple[int, int],
        expert: DeviceExpert | None,
    ) -> None:
        """Publish or invalidate one stable slot in the CUDA directory."""
        if self._slot_directory is None:
            return
        layer, expert_id = key
        target = self._slot_directory[layer, expert_id]
        scale_target = (
            None
            if self._slot_scale_directory is None
            else self._slot_scale_directory[layer, expert_id]
        )
        if expert is None:
            target.zero_()
            if scale_target is not None:
                scale_target.zero_()
            return
        scale_values = None
        if self.native8_rows_supported:
            rows, native_scales = self._native8_metadata_rows([expert])
            rows = rows[: target.numel()]
            scale_values = native_scales[0]
        else:
            rows = self._metadata_rows([expert])[: target.numel()]
        values = [row[0] for row in rows]
        host = self._slot_update_host
        if host is None or host.numel() != len(values):
            host = torch.empty(len(values), dtype=torch.long)
        host.copy_(torch.tensor(values, dtype=torch.long))
        # ``_slot_update_host`` is one shared scratch row.  Keep publication
        # synchronous so the next cache replacement cannot rewrite host data
        # before this directory update has consumed it.
        target.copy_(host, non_blocking=False)
        if scale_target is not None and scale_values is not None:
            scale_host = self._slot_scale_update_host
            if scale_host is None:
                scale_host = torch.empty(3, dtype=torch.float32)
                self._slot_scale_update_host = scale_host
            scale_host.copy_(torch.tensor(scale_values, dtype=torch.float32))
            scale_target.copy_(scale_host, non_blocking=False)

    def _prepare_resident_native8_run_locked(
        self,
        layer: int,
        value: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float,
    ) -> PendingPackedRun:
        """Publish an all-resident native8 route without a host decision."""

        if (
            self._slot_scale_directory is None
            or self._native8_route_scales is None
        ):
            raise RuntimeError("native8 resident route directory is unavailable")
        flat_ids = route_ids.reshape(-1)
        count = int(flat_ids.numel())
        from .ops import packed_route_slots

        if not packed_route_slots(
            flat_ids,
            self._slot_directory[int(layer)],
            output=self._metadata[:, :count],
            hit_mask=self._route_hit_mask[:count],
        ):
            raise RuntimeError("native8 resident route lookup was rejected")
        torch.index_select(
            self._slot_scale_directory[int(layer)],
            0,
            flat_ids,
            out=self._native8_route_scales[:count],
        )
        self.device_route_lookups += 1
        self.device_route_full_hits += 1
        self.hits += count
        return PendingPackedRun(
            layer=layer,
            value=value,
            expert_count=count,
            grouped_prefix=-1,
            activation=activation,
            activation_beta=float(activation_beta),
            activation_linear_beta=activation_linear_beta,
            limit=float(limit),
            wait_for_stage=False,
            route_order=self._route_ids[:count],
            ordered_weights=route_weights.reshape(-1).float().contiguous(),
            metadata=self._metadata[:, :count],
            native8_scales=self._native8_route_scales[:count],
        )

    def _begin_device_route_metadata(
        self,
        layer: int,
        route_ids: torch.Tensor,
    ) -> bool:
        """Launch the fixed-slot route probe without synchronizing the host.

        The fixed metadata directory is the source of truth for resident-slot
        hits.  The tiny all-hit flag and Top-K ID vector are copied to pinned
        memory now, but consumed later by :meth:`finish_run`.  DSV4/Kimi can
        therefore enqueue their independent shared branch before the only
        unavoidable host decision.  This mirrors llama.cpp's delayed graph
        update boundary while retaining exact cache-miss fallback semantics.
        """
        if os.environ.get("CCCP_DEVICE_ROUTE_METADATA", "1") == "0":
            return False
        if (
            getattr(self, "_slot_directory", None) is None
            or getattr(self, "_metadata", None) is None
            or getattr(self, "_route_hit_mask", None) is None
            or getattr(self, "_route_all_hit", None) is None
            or getattr(self, "_route_all_hit_host", None) is None
            or getattr(self, "_route_host_ids", None) is None
            or getattr(self, "_route_copy_done", None) is None
        ):
            return False
        flat_ids = route_ids.reshape(-1)
        count = int(flat_ids.numel())
        from .ops import packed_route_slots

        if not packed_route_slots(
            flat_ids,
            self._slot_directory[int(layer)],
            output=self._metadata[:, :count],
            hit_mask=self._route_hit_mask[:count],
        ):
            return False
        torch.all(
            self._route_hit_mask[:count],
            out=self._route_all_hit,
        )
        self._route_all_hit_host.copy_(
            self._route_all_hit,
            non_blocking=True,
        )
        # Copy the tiny Top-K vector in the same synchronization window.  It
        # is consumed only if CUDA reports at least one non-resident expert.
        self._route_host_ids[:count].copy_(flat_ids, non_blocking=True)
        self._route_copy_done.record(torch.cuda.current_stream(self.device))
        self.device_route_lookups += 1
        return True

    def _consume_device_route_metadata(
        self,
        layer: int,
        count: int,
    ) -> tuple[bool, list[int] | None]:
        """Consume a previously launched route probe at the latest boundary."""

        if self._route_copy_done is None or self._route_host_ids is None:
            raise RuntimeError("device route probe is not initialized")
        self._route_copy_done.synchronize()
        host_ids = [
            int(self._route_host_ids[index])
            for index in range(count)
        ]
        self._record_route_ids(layer, host_ids)
        if bool(self._route_all_hit_host):
            self.device_route_full_hits += 1
            self.hits += count
            self.last_transfer_seconds = 0.0
            return True, None
        self.device_route_fallbacks += 1
        return False, host_ids

    def _device_route_metadata(
        self,
        layer: int,
        route_ids: torch.Tensor,
    ) -> tuple[bool, list[int] | None]:
        """Synchronous compatibility wrapper for callers without split work."""

        if not self._begin_device_route_metadata(layer, route_ids):
            return False, None
        return self._consume_device_route_metadata(
            layer,
            int(route_ids.numel()),
        )

    def _host_route_ids(self, route_ids: torch.Tensor) -> list[int]:
        """Compatibility fallback without ``Tensor.tolist()``."""
        flat_ids = route_ids.reshape(-1)
        count = int(flat_ids.numel())
        if (
            flat_ids.is_cuda
            and getattr(self, "_route_host_ids", None) is not None
            and getattr(self, "_route_copy_done", None) is not None
        ):
            self._route_host_ids[:count].copy_(flat_ids, non_blocking=True)
            self._route_copy_done.record(
                torch.cuda.current_stream(self.device)
            )
            self._route_copy_done.synchronize()
            source = self._route_host_ids
        else:
            source = flat_ids.detach().cpu()
        return [int(source[index]) for index in range(count)]

    def _record_route_ids(self, layer: int, expert_ids: list[int]) -> None:
        counts = getattr(self, "route_counts", None)
        if counts is None:
            counts = self.route_counts = Counter()
        counts.update((int(layer), int(expert_id)) for expert_id in expert_ids)

    def _set_route_order(
        self,
        p12_positions: list[int],
        generic_positions: list[int],
    ) -> bool:
        """Update packed dispatch order only when it is not already identity.

        Projection-VQ archives use three weights per expert and therefore do
        not enter the legacy two-weight p12 prefix.  Their order is the fixed
        ``arange(top_k)`` allocated with the arena; re-uploading that same
        128-byte tensor once per layer introduced a needless CUDA host sync.
        """
        order_values = p12_positions + generic_positions
        identity = all(
            position == value
            for position, value in enumerate(order_values)
        )
        if not identity or not getattr(
            self,
            "_route_order_identity",
            False,
        ):
            order = torch.tensor(order_values, dtype=torch.long)
            self._route_ids[: len(order_values)].copy_(
                order,
                non_blocking=False,
            )
        self._route_order_identity = identity
        return identity

    def prepare_run(
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
    ) -> PendingPackedRun:
        """Start expert DMA while retaining exclusive ownership of its slots.

        Independent default-stream work may be enqueued before ``finish_run``.
        The packed kernel itself waits on the copy-stream event, so the arena is
        never read early and a prefetch thread cannot replace a leased slot.
        """
        if not self._decode_runtime_ready():
            # A failed/interrupted Prefill may leave the compact arena alive
            # after Decode-only buffers were cleared.  Repair that state once
            # instead of carrying an opaque failure into the first MoE layer.
            self.activate_decode_arena()
        if not self._decode_runtime_ready():
            raise RuntimeError(
                "packed hybrid Decode pool is not ready after arena restore: "
                f"arenas={self._arenas is not None},"
                f"workspaces={self._workspaces is not None},"
                f"metadata={self._metadata is not None},"
                f"route_ids={self._route_ids is not None},"
                f"route_weights={self._ordered_weights is not None},"
                f"phase={self._arena_phase},"
                f"pinned_experts={len(self.pinned)}"
            )
        if value.ndim != 2 or int(value.shape[0]) != 1:
            raise RuntimeError(
                "packed decode GEMV accepts exactly one token; use run_rows "
                "for multi-token Prefill"
            )
        self._transfer_lock.acquire()
        try:
            # ``run_rows`` temporarily gives the fixed packed slab to one
            # whole Prefill layer.  DSV decode enters through
            # prepare_run/finish_run (rather than ``run``), so restore the
            # normal signature partition here before probing or leasing any
            # decode slot.  This is a directory re-partition of the same GPU
            # byte slab; it does not allocate another cache.
            self._restore_decode_arena_locked()
            self._warm_profile_hot_locked()
            if self.native8_rows_supported:
                if self._compact_profile_all_resident:
                    return self._prepare_resident_native8_run_locked(
                        layer,
                        value,
                        route_ids,
                        route_weights,
                        activation=activation,
                        activation_beta=activation_beta,
                        activation_linear_beta=activation_linear_beta,
                        limit=limit,
                    )
                expert_ids = self._host_route_ids(route_ids)
                self._record_route_ids(layer, expert_ids)
                return self._prepare_selected_run_locked(
                    layer,
                    value,
                    expert_ids,
                    route_weights,
                    activation=activation,
                    activation_beta=activation_beta,
                    activation_linear_beta=activation_linear_beta,
                    limit=limit,
                )
            # Launch only the tiny route-directory probe here.  The previous
            # speculative executor also launched packed MoE before the all-hit
            # flag was known.  A partial miss therefore fed zero/stale slot
            # pointers to CUDA and corrupted the shared workspaces before the
            # exact fallback ran.  Delaying the packed kernel until finish_run
            # retains overlap with the independent shared branch and is also
            # faster than executing a throw-away miss path.
            if self._begin_device_route_metadata(layer, route_ids):
                count = int(route_ids.numel())
                return PendingPackedRun(
                    layer=layer,
                    value=value,
                    expert_count=count,
                    grouped_prefix=-1,
                    activation=activation,
                    activation_beta=float(activation_beta),
                    activation_linear_beta=activation_linear_beta,
                    limit=float(limit),
                    wait_for_stage=False,
                    route_order=self._route_ids[:count],
                    ordered_weights=(
                        route_weights.reshape(-1).float().contiguous()
                    ),
                    metadata=self._metadata[:, :count],
                    device_route_probe=True,
                )
            device_hit, expert_ids = self._device_route_metadata(
                layer,
                route_ids,
            )
            if device_hit:
                count = int(route_ids.numel())
                return PendingPackedRun(
                    layer=layer,
                    value=value,
                    expert_count=count,
                    grouped_prefix=-1,
                    activation=activation,
                    activation_beta=float(activation_beta),
                    activation_linear_beta=activation_linear_beta,
                    limit=float(limit),
                    wait_for_stage=False,
                    route_order=self._route_ids[:count],
                    ordered_weights=(
                        route_weights.reshape(-1).float().contiguous()
                    ),
                    metadata=self._metadata[:, :count],
                )
            if expert_ids is not None:
                return self._prepare_selected_run_locked(
                    layer,
                    value,
                    expert_ids,
                    route_weights,
                    activation=activation,
                    activation_beta=activation_beta,
                    activation_linear_beta=activation_linear_beta,
                    limit=limit,
                )
            expert_ids = self._host_route_ids(route_ids)
            self._record_route_ids(layer, expert_ids)
            return self._prepare_selected_run_locked(
                layer,
                value,
                expert_ids,
                route_weights,
                activation=activation,
                activation_beta=activation_beta,
                activation_linear_beta=activation_linear_beta,
                limit=limit,
            )
        except BaseException:
            self._transfer_lock.release()
            raise

    def _prepare_selected_run_locked(
        self,
        layer: int,
        value: torch.Tensor,
        expert_ids: list[int],
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float,
    ) -> PendingPackedRun:
        """Resolve a host-known Top-K route without a CUDA round trip.

        The caller owns ``_transfer_lock``.  This helper is shared by CUDA
        Router fallback and RAM-Dense/CPU Router execution, so the cache,
        metadata and numerical kernel remain identical.  Only route ownership
        changes: already-host-resident IDs never travel CPU→GPU→CPU merely to
        select the packed RAM source.
        """

        plan = (
            None
            if self.native8_rows_supported
            else self._find_route_plan(layer, expert_ids)
        )
        if plan is not None:
            if plan.identity_order:
                ordered_weights = (
                    route_weights.reshape(-1).float().contiguous()
                )
            else:
                torch.index_select(
                    route_weights.reshape(-1).float().contiguous(),
                    0,
                    plan.order,
                    out=self._ordered_weights[: len(expert_ids)],
                )
                ordered_weights = self._ordered_weights[: len(expert_ids)]
            return PendingPackedRun(
                layer=layer,
                value=value,
                expert_count=len(expert_ids),
                grouped_prefix=plan.grouped_prefix,
                activation=activation,
                activation_beta=float(activation_beta),
                activation_linear_beta=activation_linear_beta,
                limit=float(limit),
                wait_for_stage=False,
                route_order=plan.order,
                ordered_weights=ordered_weights,
                metadata=plan.metadata,
            )
        self._last_ids[layer] = expert_ids
        keys = [(layer, expert_id) for expert_id in expert_ids]
        self._ensure_decode_route_capacity_locked(keys)
        uploaded_before = self.uploaded_bytes
        selected = self._ensure_locked(
            keys,
            prefetch=False,
            defer_wait=True,
        )
        experts = [selected[key] for key in keys]
        native8_scales = None
        if self.native8_rows_supported:
            native8_scales = self._copy_native8_metadata(experts)
        else:
            self._copy_metadata(experts)
        p12_positions = [
            position
            for position, expert in enumerate(experts)
            if (
                len(expert) == 2
                and expert[0].bits == 12
                and expert[0].dim in (4, 8)
                and expert[1].bits == 12
                and expert[1].dim in (4, 8)
            )
        ]
        generic_positions = [
            position
            for position in range(len(experts))
            if position not in p12_positions
        ]
        identity_order = self._set_route_order(
            p12_positions,
            generic_positions,
        )
        if identity_order:
            ordered_weights = route_weights.reshape(-1).float().contiguous()
        else:
            torch.index_select(
                route_weights.reshape(-1).float().contiguous(),
                0,
                self._route_ids[: len(experts)],
                out=self._ordered_weights[: len(experts)],
            )
            ordered_weights = self._ordered_weights[: len(experts)]
        self._save_route_plan(
            layer,
            expert_ids,
            keys,
            experts,
            len(p12_positions),
            identity_order,
        )
        return PendingPackedRun(
            layer=layer,
            value=value,
            expert_count=len(experts),
            grouped_prefix=len(p12_positions),
            activation=activation,
            activation_beta=float(activation_beta),
            activation_linear_beta=activation_linear_beta,
            limit=float(limit),
            wait_for_stage=self.uploaded_bytes > uploaded_before,
            route_order=self._route_ids[: len(experts)],
            ordered_weights=ordered_weights,
            metadata=self._metadata[:, : len(experts)],
            native8_scales=native8_scales,
        )

    def _launch_native8_decode(
        self,
        resolved: PendingPackedRun,
        metadata: torch.Tensor,
        ordered_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Expand one Top-K route and execute it with native FP8 Tensor Cores."""

        scales = resolved.native8_scales
        if scales is None:
            raise RuntimeError("native8 Decode is missing codebook scales")
        count = int(resolved.expert_count)
        workspace = self._native8_decode_workspace_for(count)
        gu = workspace["gu"][:count]
        down_tc = workspace["down_tc"][:, : count * int(self.store.cfg["moe_inter"])]

        from .fusedext import dense_fp8_quantize_rows_fused
        from .grouped import activate_gate_up
        from .ops import projection_expand_native8

        projection_expand_native8(metadata.contiguous(), gu, down_tc)
        intermediate = int(self.store.cfg["moe_inter"])
        gu_scales = workspace["gu_scales"][:, : count * 2 * intermediate]
        gu_scale_view = gu_scales.view(count, 2 * intermediate)
        gu_scale_view[:, :intermediate].copy_(scales[:, 0:1])
        gu_scale_view[:, intermediate:].copy_(scales[:, 1:2])

        native_input = workspace["input"]
        input_scales = workspace["input_scales"]
        if dense_fp8_quantize_rows_fused(
            resolved.value.to(torch.bfloat16),
            native_input,
            input_scales,
        ) is None:
            raise RuntimeError("native E4M3 Decode input quantizer rejected rows")
        gate_up = torch._scaled_mm(
            native_input,
            gu.reshape(count * 2 * intermediate, -1).t(),
            scale_a=input_scales,
            scale_b=gu_scales,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        ).view(count, 2 * intermediate)
        gate, up = gate_up.chunk(2, dim=-1)
        if float(resolved.limit) > 0.0:
            gate = gate.clamp(max=float(resolved.limit))
            up = up.clamp(
                min=-float(resolved.limit),
                max=float(resolved.limit),
            )
        activated = activate_gate_up(
            gate,
            up,
            activation=resolved.activation,
            situ_beta=float(resolved.activation_beta),
            situ_linear_beta=resolved.activation_linear_beta,
        ).contiguous()
        # Fold route probability and each expert's Down codebook scale into A.
        # B can then be one contiguous [TopK*I,H] E4M3 matrix with unit scale,
        # so the expert reduction is performed by the same native GEMM.
        weighted_activated = (
            activated.float()
            * ordered_weights[:count].view(-1, 1)
            * scales[:, 2:3]
        ).reshape(1, count * intermediate)
        native_activated = workspace["activated"][:, : count * intermediate]
        activated_scales = workspace["activated_scales"]
        if dense_fp8_quantize_rows_fused(
            weighted_activated,
            native_activated,
            activated_scales,
        ) is None:
            raise RuntimeError("native E4M3 Decode hidden quantizer rejected rows")
        down = torch._scaled_mm(
            native_activated,
            down_tc.t(),
            scale_a=activated_scales,
            scale_b=workspace["unit_scale"],
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        _hidden, _output, result = self._workspaces
        result.copy_(down.reshape(-1).float())
        return result

    def _launch_packed_run(
        self,
        resolved: PendingPackedRun,
    ) -> torch.Tensor:
        """Submit one fixed-workspace packed MoE execution."""

        if resolved.wait_for_stage:
            self._stage.wait()
        hidden, output, result = self._workspaces
        from .ops import packed_moe_topk

        route_order = (
            self._route_ids[: resolved.expert_count]
            if resolved.route_order is None
            else resolved.route_order
        )
        ordered_weights = (
            self._ordered_weights[: resolved.expert_count]
            if resolved.ordered_weights is None
            else resolved.ordered_weights
        )
        metadata = (
            self._metadata[:, : resolved.expert_count]
            if resolved.metadata is None
            else resolved.metadata
        )
        if resolved.native8_scales is not None:
            self.decode_fused_submissions += 1
            return self._launch_native8_decode(
                resolved,
                metadata,
                ordered_weights,
            )
        capability = self.store.man.projection_operator_capability(
            resolved.layer
        )
        # Numerical isolation switch used by the public CUDA audit.  It stages
        # the exact routed experts into dedicated allocations and therefore
        # bypasses the reusable arena, while retaining the same public packed
        # operator and route order.  Normal launches never enter this path.
        if os.environ.get("CCCP_HYBRID_FRESH_REFERENCE", "0") != "0":
            self.decode_reference_submissions += 1
            expert_ids = self._last_ids.get(int(resolved.layer), ())
            if len(expert_ids) != resolved.expert_count:
                raise RuntimeError(
                    "fresh packed reference has no exact host route"
                )
            fresh_experts = []
            for expert_id in expert_ids:
                # Reload the archive view as well, rather than reusing the
                # hybrid pool's normalized host/codebook cache.  This lets the
                # audit distinguish a bad semantic codebook key from a bad
                # reusable device slot.
                host = self.store.load_expert_packed(
                    int(resolved.layer),
                    int(expert_id),
                )
                fresh_experts.append(tuple(
                    DevicePackedWeight(
                        raw=weight.raw.to(self.device),
                        cb=weight.cb.to(
                            device=self.device,
                            dtype=torch.bfloat16,
                        ),
                        rows=weight.rows,
                        cols=weight.cols,
                        blocks=weight.blocks,
                        dim=weight.dim,
                        bits=weight.bits,
                    )
                    for weight in host
                ))
            fresh_rows = torch.tensor(
                build_runtime_metadata_rows(fresh_experts)[
                    : metadata.shape[0]
                ],
                dtype=torch.long,
                device=self.device,
            )
            # Mirror the known-good full-resident ABI exactly: metadata is
            # indexed by the model's global expert ID rather than a compact
            # temporary 0..TopK route.  Only the selected columns need live
            # pointers for this audit.
            fresh_metadata = torch.zeros(
                metadata.shape[0],
                int(self.store.cfg["n_experts"]),
                dtype=torch.long,
                device=self.device,
            )
            fresh_route_ids = torch.tensor(
                expert_ids,
                dtype=torch.long,
                device=self.device,
            )
            fresh_metadata[:, fresh_route_ids] = fresh_rows
            fresh_hidden = torch.empty_like(hidden[: resolved.expert_count])
            fresh_output = torch.empty_like(output[: resolved.expert_count])
            fresh_result = torch.empty_like(result)
            fresh = packed_moe_topk(
                resolved.value.to(torch.bfloat16),
                fresh_route_ids,
                ordered_weights,
                fresh_metadata,
                activation=resolved.activation,
                activation_beta=resolved.activation_beta,
                activation_linear_beta=(
                    0.0
                    if resolved.activation_linear_beta is None
                    else float(resolved.activation_linear_beta)
                ),
                limit=resolved.limit,
                hidden_workspace=fresh_hidden,
                output_workspace=fresh_output,
                result=fresh_result,
                grouped_prefix=-1,
                **capability,
            )
            # The dedicated tensors are intentionally short-lived; complete
            # their diagnostic use before returning them to the allocator.
            torch.cuda.current_stream(self.device).synchronize()
            return fresh
        graph_enabled = (
            resolved.value.is_cuda
            and metadata.shape[0] in (15, 27)
            and os.environ.get("CCCP_PACKED_MOE_GRAPH", "1") != "0"
        )
        if graph_enabled:
            from .ops.packed_graph import (
                FixedPackedMoEGraph,
                PackedMoEGraphSpec,
            )

            count = resolved.expert_count
            fixed_metadata = self._metadata[:, :count]
            if metadata.data_ptr() != fixed_metadata.data_ptr():
                fixed_metadata.copy_(metadata)
            fixed_order = self._route_ids[:count]
            if route_order.data_ptr() != fixed_order.data_ptr():
                fixed_order.copy_(route_order)
            value_bf16 = resolved.value.to(torch.bfloat16)
            if (
                self._packed_graph_input is None
                or self._packed_graph_input.shape != value_bf16.shape
            ):
                self._packed_graph_input = torch.empty_like(value_bf16)
            if (
                self._packed_graph_weights is None
                or self._packed_graph_weights.numel() < count
            ):
                self._packed_graph_weights = torch.empty(
                    count,
                    dtype=torch.float32,
                    device=self.device,
                )
            graph_input = self._packed_graph_input
            graph_weights = self._packed_graph_weights[:count]
            graph_input.copy_(value_bf16)
            graph_weights.copy_(ordered_weights)
            spec = PackedMoEGraphSpec(
                activation=resolved.activation,
                activation_beta=float(resolved.activation_beta),
                activation_linear_beta=(
                    0.0
                    if resolved.activation_linear_beta is None
                    else float(resolved.activation_linear_beta)
                ),
                limit=float(resolved.limit),
                top_k=count,
                grouped_prefix=-1,
                packed_formats=tuple(capability["packed_formats"]),
                code_dims=tuple(capability["code_dims"]),
                codebook_sizes=tuple(capability["codebook_sizes"]),
            )
            graph = self._packed_moe_graphs.get(spec)
            graph_arguments = (
                graph_input,
                fixed_order,
                graph_weights,
                fixed_metadata,
                hidden[:count],
                output[:count],
                result,
                spec,
            )
            if graph is None or not graph.matches(*graph_arguments):
                graph = FixedPackedMoEGraph(*graph_arguments)
                graph.capture()
                self._packed_moe_graphs[spec] = graph
            self.decode_graph_submissions += 1
            self.decode_fused_submissions += 1
            return graph.run()

        self.decode_fused_submissions += 1
        return packed_moe_topk(
            resolved.value.to(torch.bfloat16),
            route_order,
            ordered_weights,
            metadata,
            activation=resolved.activation,
            activation_beta=resolved.activation_beta,
            activation_linear_beta=(
                0.0
                if resolved.activation_linear_beta is None
                else float(resolved.activation_linear_beta)
            ),
            limit=resolved.limit,
            hidden_workspace=hidden[: resolved.expert_count],
            output_workspace=output[: resolved.expert_count],
            result=result,
            grouped_prefix=resolved.grouped_prefix,
            **capability,
        )

    def finish_run(self, pending: PendingPackedRun) -> torch.Tensor:
        if not pending.active:
            raise RuntimeError("packed MoE pending call is no longer active")
        resolved = pending
        try:
            if pending.device_route_probe:
                device_hit, expert_ids = self._consume_device_route_metadata(
                    pending.layer,
                    pending.expert_count,
                )
                if not device_hit:
                    resolved = self._prepare_selected_run_locked(
                        pending.layer,
                        pending.value,
                        expert_ids,
                        pending.ordered_weights,
                        activation=pending.activation,
                        activation_beta=pending.activation_beta,
                        activation_linear_beta=(
                            pending.activation_linear_beta
                        ),
                        limit=pending.limit,
                    )
            return self._launch_packed_run(resolved)
        finally:
            pending.active = False
            resolved.active = False
            self._transfer_lock.release()

    def cancel_run(self, pending: PendingPackedRun) -> None:
        if pending.active:
            pending.active = False
            self._transfer_lock.release()

    def _find_route_plan(
        self,
        layer: int,
        expert_ids: list[int],
    ) -> PackedRoutePlan | None:
        """Reuse device pointer metadata while its arena leases stay valid."""
        if os.environ.get("CCCP_KIMI_ROUTE_PLAN", "1") == "0":
            return None
        plan = self._route_plans.get(int(layer))
        if plan is None or tuple(expert_ids) != plan.expert_ids:
            self.route_plan_misses += 1
            return None
        with self._lock:
            valid = all(
                self.cache.get(key) is expert
                for key, expert in zip(plan.keys, plan.experts)
            )
            if valid:
                for key in plan.keys:
                    self.cache.move_to_end(key)
                    self._arenas.touch(key)
                    self.hits += 1
        if not valid:
            self.route_plan_misses += 1
            return None
        self._last_ids[int(layer)] = list(plan.expert_ids)
        self.last_transfer_seconds = 0.0
        self.route_plan_hits += 1
        return plan

    def _reuse_route_plan(
        self,
        layer: int,
        value: torch.Tensor,
        expert_ids: list[int],
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float = 0.0,
    ) -> torch.Tensor | None:
        plan = self._find_route_plan(layer, expert_ids)
        if plan is None:
            return None
        if plan.identity_order:
            ordered_weights = route_weights.reshape(-1).float().contiguous()
        else:
            torch.index_select(
                route_weights.reshape(-1).float().contiguous(),
                0,
                plan.order,
                out=self._ordered_weights[: len(plan.expert_ids)],
            )
            ordered_weights = self._ordered_weights[: len(plan.expert_ids)]
        hidden, output, result = self._workspaces
        from .ops import packed_moe_topk

        return packed_moe_topk(
            value.to(torch.bfloat16),
            plan.order,
            ordered_weights,
            plan.metadata,
            activation=activation,
            activation_beta=float(activation_beta),
            activation_linear_beta=(
                0.0
                if activation_linear_beta is None
                else float(activation_linear_beta)
            ),
            limit=float(limit),
            hidden_workspace=hidden[: len(plan.expert_ids)],
            output_workspace=output[: len(plan.expert_ids)],
            result=result,
            grouped_prefix=plan.grouped_prefix,
            **self.store.man.projection_operator_capability(layer),
        )

    def _save_route_plan(
        self,
        layer: int,
        expert_ids: list[int],
        keys: list[tuple[int, int]],
        experts: list[DeviceExpert],
        grouped_prefix: int,
        identity_order: bool,
    ) -> None:
        if os.environ.get("CCCP_KIMI_ROUTE_PLAN", "1") == "0":
            return
        count = len(expert_ids)
        previous = self._route_plans.get(int(layer))
        if previous is None or previous.order.numel() != count:
            saved_order = torch.empty(
                count,
                dtype=torch.long,
                device=self.device,
            )
            saved_metadata = torch.empty(
                self._metadata.shape[0],
                count,
                dtype=torch.long,
                device=self.device,
            )
        else:
            saved_order = previous.order
            saved_metadata = previous.metadata
        saved_order.copy_(
            self._route_ids[:count],
            non_blocking=False,
        )
        saved_metadata.copy_(
            self._metadata[:, :count],
            non_blocking=False,
        )
        self._route_plans[int(layer)] = PackedRoutePlan(
            expert_ids=tuple(expert_ids),
            keys=tuple(keys),
            experts=tuple(experts),
            order=saved_order,
            metadata=saved_metadata,
            grouped_prefix=int(grouped_prefix),
            identity_order=bool(identity_order),
        )

    def _prepare_host_bridge(
        self,
        value: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        copy_route_ids: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Copy only token-sized operands into fixed CUDA buffers."""

        count = int(route_ids.numel())
        input_buffer = getattr(self, "_host_bridge_input", None)
        if (
            input_buffer is None
            or input_buffer.shape != value.shape
            or input_buffer.dtype != torch.bfloat16
        ):
            input_buffer = torch.empty(
                value.shape,
                dtype=torch.bfloat16,
                device=self.device,
            )
            self._host_bridge_input = input_buffer
        ids_buffer = getattr(self, "_host_bridge_ids", None)
        if ids_buffer is None or ids_buffer.numel() < count:
            ids_buffer = torch.empty(
                count,
                dtype=torch.long,
                device=self.device,
            )
            self._host_bridge_ids = ids_buffer
        weights_buffer = getattr(self, "_host_bridge_weights", None)
        if weights_buffer is None or weights_buffer.numel() < count:
            weights_buffer = torch.empty(
                count,
                dtype=torch.float32,
                device=self.device,
            )
            self._host_bridge_weights = weights_buffer
        input_buffer.copy_(value.to(torch.bfloat16))
        if copy_route_ids:
            ids_buffer[:count].copy_(route_ids.reshape(-1))
        weights_buffer[:count].copy_(route_weights.reshape(-1).float())
        return input_buffer, ids_buffer[:count], weights_buffer[:count]

    def _finish_host_bridge(self, device_result: torch.Tensor) -> torch.Tensor:
        output_size = int(self.store.cfg["routed_hidden"])
        host_result = getattr(self, "_host_bridge_result", None)
        if host_result is None or host_result.numel() != output_size:
            host_result = torch.empty(
                output_size,
                dtype=torch.bfloat16,
                pin_memory=True,
            )
            self._host_bridge_result = host_result
        host_result.copy_(device_result, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        return host_result

    def _prepare_host_rows_bridge(
        self,
        value: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage one complete host Prefill block once per routed layer.

        RAM-Dense/CUDA-MoE execution keeps the large dense projections on the
        host, but Prefill must still be layer-major.  Replaying ``run`` for
        every token uploads the same compact experts repeatedly and turns a
        short prompt into hundreds of GiB of PCIe traffic.  These persistent
        buffers move the latent rows and routes once, then let ``run_rows``
        use its ordinary expert-grouped CUDA executor.
        """

        rows = int(value.shape[0])
        top_k = int(route_ids.shape[1])
        input_shape = (rows, int(value.shape[1]))
        input_buffer = getattr(self, "_host_rows_input", None)
        if (
            input_buffer is None
            or tuple(input_buffer.shape) != input_shape
            or input_buffer.dtype != torch.bfloat16
        ):
            input_buffer = torch.empty(
                input_shape,
                dtype=torch.bfloat16,
                device=self.device,
            )
            self._host_rows_input = input_buffer
        route_shape = (rows, top_k)
        ids_buffer = getattr(self, "_host_rows_ids", None)
        if ids_buffer is None or tuple(ids_buffer.shape) != route_shape:
            ids_buffer = torch.empty(
                route_shape,
                dtype=torch.long,
                device=self.device,
            )
            self._host_rows_ids = ids_buffer
        weights_buffer = getattr(self, "_host_rows_weights", None)
        if (
            weights_buffer is None
            or tuple(weights_buffer.shape) != route_shape
        ):
            weights_buffer = torch.empty(
                route_shape,
                dtype=torch.float32,
                device=self.device,
            )
            self._host_rows_weights = weights_buffer
        input_buffer.copy_(value.to(torch.bfloat16))
        ids_buffer.copy_(route_ids.to(torch.long))
        weights_buffer.copy_(route_weights.float())
        return input_buffer, ids_buffer, weights_buffer

    def _finish_host_rows_bridge(
        self,
        device_result: torch.Tensor,
    ) -> torch.Tensor:
        shape = tuple(device_result.shape)
        host_result = getattr(self, "_host_rows_result", None)
        if (
            host_result is None
            or tuple(host_result.shape) != shape
            or host_result.dtype != device_result.dtype
        ):
            host_result = torch.empty(
                shape,
                dtype=device_result.dtype,
                pin_memory=True,
            )
            self._host_rows_result = host_result
        host_result.copy_(device_result, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        return host_result

    def prepare_host_run(
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
    ) -> PendingPackedRun:
        """Start packed DMA while the host computes an independent branch."""

        if route_ids.is_cuda:
            raise ValueError("host-routed packed execution requires CPU route IDs")
        expert_ids = [int(value) for value in route_ids.reshape(-1)]
        self._record_route_ids(layer, expert_ids)
        device_value, _device_ids, device_weights = self._prepare_host_bridge(
            value,
            route_ids,
            route_weights,
            copy_route_ids=False,
        )
        self._transfer_lock.acquire()
        try:
            # ``run_rows`` may temporarily repartition a small fixed arena by
            # the current Prefill layer's signatures.  Autoregressive decode
            # must restore the default multi-signature slab before leasing its
            # first Top-K route; otherwise a valid signature absent from the
            # final Prefill layer raises KeyError after Prefill has completed.
            self._restore_decode_arena_locked()
            return self._prepare_selected_run_locked(
                layer,
                device_value,
                expert_ids,
                device_weights,
                activation=activation,
                activation_beta=activation_beta,
                activation_linear_beta=activation_linear_beta,
                limit=limit,
            )
        except BaseException:
            self._transfer_lock.release()
            raise

    def finish_host_run(self, pending: PendingPackedRun) -> torch.Tensor:
        return self._finish_host_bridge(self.finish_run(pending))

    def _prefill_unique_fits(
        self,
        layer: int,
        expert_ids: list[int],
    ) -> bool:
        """Return whether every unique expert can coexist in fixed arenas."""

        if self._arenas is None:
            return False
        needed: Counter[PackedExpertSignature] = Counter()
        for expert_id in expert_ids:
            expert = self.pinned.get((int(layer), int(expert_id)))
            if expert is None:
                return False
            needed[PackedExpertSignature.of(expert)] += 1
        return all(
            signature in self._arenas.arenas
            and count <= self._arenas.arenas[signature].book.count
            for signature, count in needed.items()
        )

    def _reset_arena_directory_locked(self) -> None:
        self.cache.clear()
        self._route_plans.clear()
        self._profile_hot_ready = False
        self._compact_profile_all_resident = False
        if self._slot_directory is not None:
            self._slot_directory.zero_()
        if self._slot_scale_directory is not None:
            self._slot_scale_directory.zero_()

    def _ensure_decode_route_capacity_locked(
        self,
        keys: list[tuple[int, int]],
    ) -> None:
        """Re-slice the fixed slab when one decode route cannot coexist.

        Small cards must not reserve a complete Top-K for every precision tier
        simultaneously.  The slab size and addresses stay fixed; only its
        signature views are rebuilt at a layer boundary.  Every selected
        expert remains present and the packed bytes are uploaded normally.
        """

        if (
            not self._adaptive_decode_arena
            or self._arenas is None
            or self._extreme_specs is not None
            or not keys
        ):
            return
        needed: Counter[PackedExpertSignature] = Counter()
        for key in keys:
            expert = self.pinned.get(key)
            if expert is None:
                raise KeyError(f"packed RAM expert missing: {key}")
            needed[PackedExpertSignature.of(expert)] += 1
        if all(
            signature in self._arenas.arenas
            and count <= self._arenas.arenas[signature].book.count
            for signature, count in needed.items()
        ):
            return

        counts = Counter(
            PackedExpertSignature.of(expert)
            for expert in self.pinned.values()
        )
        minimum = {
            # Do not strand VRAM in precision signatures absent from the
            # current route.  One slot preserves each arena view; all experts
            # required by this decode layer are raised to their exact count
            # below without changing the fixed slab size.
            signature: min(int(count), 1)
            for signature, count in counts.items()
        }
        for signature, count in needed.items():
            minimum[signature] = max(minimum.get(signature, 1), int(count))
        specs = allocate_packed_slots(
            counts,
            self._arenas.nbytes,
            minimum,
            resident_codebooks=self._resident_codebooks,
        )
        # Stop prior compute/copies before the same storage receives new views.
        torch.cuda.synchronize(self.device)
        self._reset_arena_directory_locked()
        self._arenas.repartition(specs)
        self._default_arena_specs = dict(specs)
        self._default_arena_covers_all_pinned = None
        self.arena_slots = {}
        for signature, count in specs.items():
            tier = self._signature_tier(signature)
            self.arena_slots[tier] = self.arena_slots.get(tier, 0) + count
        self.adaptive_decode_repartitions += 1
        print(
            "[cccp-packed] decode 当前层专家槽重分配完成："
            f"{sum(needed.values())} 个路由专家；"
            f"固定显存 {self._arenas.nbytes / 2**30:.2f}GiB",
            flush=True,
        )

    def _partition_prefill_layer_locked(self, layer: int) -> None:
        """Use a whole-layer view when it fits the fixed expert slab.

        On small cards the slab can be narrower than one physical layer.  In
        that case keep the global signature views and let ``run_rows`` split
        only the unique experts into fitting chunks.  The token batch remains
        intact and never falls back to the decode GEMV.
        """

        if (
            self._arenas is None
            or self._prefill_partition_layer == int(layer)
            or self._extreme_specs is not None
        ):
            return
        # A strict profile can fit every selected compact expert in the
        # default arena at once.  In that case repartitioning the slab for
        # every prefill layer only invalidates already uploaded experts: the
        # next decode then re-uploads tens of GiB and loses the whole benefit
        # of fixed residency.  Keep the global directory stable when its
        # per-signature slot counts cover the complete pinned profile.
        covers_all = getattr(
            self, "_default_arena_covers_all_pinned", None
        )
        if covers_all is None:
            global_counts = Counter(
                PackedExpertSignature.of(expert)
                for expert in self.pinned.values()
            )
            covers_all = all(
                signature in self._arenas.arenas
                and count
                <= self._arenas.arenas[signature].book.count
                for signature, count in global_counts.items()
            )
            self._default_arena_covers_all_pinned = bool(covers_all)
        if covers_all:
            return
        specs: Counter[PackedExpertSignature] = Counter()
        for (expert_layer, _expert_id), expert in self.pinned.items():
            if int(expert_layer) == int(layer):
                specs[PackedExpertSignature.of(expert)] += 1
        if not specs:
            raise RuntimeError(f"packed Prefill layer {layer} has no experts")
        required = sum(
            signature.storage_bytes(self._resident_codebooks) * count
            for signature, count in specs.items()
        )
        if required > self._arenas.nbytes:
            # A preceding layer may have fit completely and therefore left
            # the slab in a layer-specific signature layout.  Reusing that
            # layout for a wider next layer can leave one of its signatures
            # with zero slots, so even a one-expert chunk is reported as not
            # fitting.  Restore the global layout before the ordinary
            # chunked path plans this layer.  This changes only arena views;
            # the complete configured expert set remains resident in RAM.
            if self._prefill_partition_layer is not None:
                torch.cuda.synchronize(self.device)
                self._stage.wait()
                self._reset_arena_directory_locked()
                self._arenas.repartition(self._default_arena_specs)
                self._prefill_partition_layer = None
            return
        # The previous layer's dequant/GEMM must finish before its packed slab
        # bytes are assigned new signature views and overwritten by H2D.
        torch.cuda.synchronize(self.device)
        self._stage.wait()
        self._reset_arena_directory_locked()
        self._arenas.repartition(specs)
        self._prefill_partition_layer = int(layer)

    def _restore_decode_arena_locked(self) -> None:
        if self._arenas is None or self._prefill_partition_layer is None:
            return
        torch.cuda.synchronize(self.device)
        self._stage.wait()
        self._reset_arena_directory_locked()
        self._arenas.repartition(self._default_arena_specs)
        self._prefill_partition_layer = None

    def _prefill_workspace_for(
        self,
        rows: int,
        micro_batch: int,
        top_k: int,
    ) -> dict[str, object]:
        intermediate = int(self.store.cfg["moe_inter"])
        hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        expert_count = int(self.store.cfg["n_experts"])
        cached = self._prefill_workspace
        if (
            cached is not None
            and int(cached["rows"]) >= rows
            and int(cached["micro_batch"]) >= micro_batch
            and int(cached["top_k"]) == top_k
            and int(cached["intermediate"]) == intermediate
            and int(cached["hidden_size"]) == hidden
        ):
            return cached
        cached = {
            "rows": rows,
            "micro_batch": micro_batch,
            "top_k": top_k,
            "intermediate": intermediate,
            "hidden_size": hidden,
            "result": torch.empty(
                rows,
                hidden,
                dtype=torch.float32,
                device=self.device,
            ),
            "metadata": torch.empty(
                15,
                expert_count,
                dtype=torch.long,
                device=self.device,
            ),
            "metadata_host": torch.empty(
                15,
                expert_count,
                dtype=torch.long,
            ),
        }
        if torch.version.hip is not None:
            # Windows ROCm's internal torch._grouped_mm currently terminates
            # the process on gfx1150.  The CCCP grouped packed operator keeps
            # the same complete token batch but reads compact experts directly
            # and owns its activation/result workspaces explicitly.
            cached["packed_hidden"] = torch.empty(
                micro_batch * top_k,
                2 * intermediate,
                dtype=torch.bfloat16,
                device=self.device,
            )
            cached["packed_result"] = torch.empty(
                micro_batch,
                hidden,
                dtype=torch.float32,
                device=self.device,
            )
        self._prefill_workspace = cached
        return cached

    def _prefill_dequant_chunk_capacity(self, expert_count: int) -> int:
        """Choose a dense expert scratch that stays inside the VRAM cap."""

        cached = getattr(self, "_prefill_dequant_workspace", None)
        if cached is not None:
            return max(1, min(int(expert_count), int(cached[0].shape[0])))
        # The previous layer deliberately releases its expanded-BF16 expert
        # scratch so Attention and MoE can share the same limited VRAM.  Under
        # a per-process memory fraction PyTorch may retain that differently
        # sized block as reserved-but-unallocated; a following layer then
        # cannot grow a new scratch even though live allocations are safely
        # below the cap.  Return only that stale cache to the driver before
        # planning the next automatic expert chunk.  Compact expert arenas
        # and live attention state remain allocated and are never evicted.
        reserved = int(torch.cuda.memory_reserved(self.device))
        allocated = int(torch.cuda.memory_allocated(self.device))
        if reserved - allocated >= 256 * 2**20:
            # Expert uploads use a dedicated stage stream.  Synchronize all
            # device work before asking the allocator to return event-guarded
            # blocks; otherwise ``empty_cache`` can leave the complete prior
            # layer workspace reserved and the next differently sized chunk
            # still fails under a hard process fraction.
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        intermediate = int(self.store.cfg["moe_inter"])
        # Native8 stores gate + up + down in one byte per value. BF16 remains
        # the exact fallback on architectures without native FP8 Tensor Cores.
        execution_bytes = (
            1 if getattr(self, "_native8_prefill_enabled", False) else 2
        )
        bytes_per_expert = 3 * hidden * intermediate * execution_bytes
        try:
            explicit_limit = float(os.environ.get("CCCP_VRAM_LIMIT_GB", "0"))
        except (TypeError, ValueError):
            explicit_limit = 0.0
        physical_free, _physical_total = torch.cuda.mem_get_info(
            self.device
        )
        physical_free = int(physical_free)
        if explicit_limit > 0:
            limit_bytes = int(explicit_limit * 2**30)
            # Reserved-but-unallocated blocks are immediately reusable by the
            # CUDA caching allocator.  Counting them as live made an expert
            # workspace released after the previous layer permanently shrink
            # every following layer's automatic chunk.
            used_bytes = int(torch.cuda.memory_allocated(self.device))
            # The allocator cap cannot see memory consumed by another process.
            # Respect both ceilings or a shared GPU can plan a 3+ GiB expert
            # chunk while the driver has only 2 GiB physically free.
            available = min(
                physical_free,
                max(0, limit_bytes - used_bytes),
            )
        else:
            available = physical_free
        # Keep the launcher's physical safety line available to the dequant
        # operator itself.  Planning only 15% slack chose 55--59 expanded
        # experts under a 20-GiB cap, filled the allocator to 19.2 GiB and
        # made ``projection_dequant`` reject the otherwise valid batch.  The
        # token batch remains intact; only the expert list is split further.
        try:
            configured_headroom = max(
                float(os.environ.get("CCCP_VRAM_RESERVE_GB", "1")),
                float(os.environ.get("CCCP_VRAM_HEADROOM_GB", "1")),
            )
        except (TypeError, ValueError):
            configured_headroom = 1.0
        safety = max(
            512 * 2**20,
            int(available * 0.15),
            int(configured_headroom * 2**30),
        )
        automatic = max(1, (available - safety) // bytes_per_expert)
        try:
            requested = int(os.environ.get(
                "CCCP_PREFILL_DEQUANT_EXPERTS", "0"
            ))
        except (TypeError, ValueError):
            requested = 0
        if requested > 0:
            automatic = min(automatic, requested)
        return max(1, min(int(expert_count), int(automatic)))

    def _prefill_dequant_workspace_for(
        self,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        intermediate = int(self.store.cfg["moe_inter"])
        cached = getattr(self, "_prefill_dequant_workspace", None)
        if cached is not None and int(cached[0].shape[0]) >= int(capacity):
            return cached
        cached = (
            torch.empty(
                int(capacity),
                2 * intermediate,
                hidden,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.empty(
                int(capacity),
                hidden,
                intermediate,
                dtype=torch.bfloat16,
                device=self.device,
            ),
        )
        self._prefill_dequant_workspace = cached
        return cached

    def _prefill_native8_workspace_for(
        self,
        capacity: int,
        routed_rows: int,
    ) -> dict[str, torch.Tensor | int]:
        """Return fixed E4M3 projection and activation buffers."""

        hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        intermediate = int(self.store.cfg["moe_inter"])
        cached = self._prefill_native8_workspace
        if (
            cached is not None
            and int(cached["capacity"]) >= int(capacity)
            and int(cached["routed_rows"]) >= int(routed_rows)
        ):
            return cached
        cached = {
            "capacity": int(capacity),
            "routed_rows": int(routed_rows),
            "gu": torch.empty(
                int(capacity),
                2 * intermediate,
                hidden,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            ),
            "down": torch.empty(
                int(capacity),
                hidden,
                intermediate,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            ),
            "input": torch.empty(
                int(routed_rows),
                hidden,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            ),
            "input_scales": torch.empty(
                int(routed_rows), 1, dtype=torch.float32, device=self.device
            ),
            "activated": torch.empty(
                int(routed_rows),
                intermediate,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            ),
            "activated_scales": torch.empty(
                int(routed_rows), 1, dtype=torch.float32, device=self.device
            ),
            "gu_scales": torch.empty(
                int(capacity),
                2 * intermediate,
                dtype=torch.float32,
                device=self.device,
            ),
            "down_scales": torch.empty(
                int(capacity),
                hidden,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        self._prefill_native8_workspace = cached
        return cached

    def _native8_decode_workspace_for(
        self,
        capacity: int,
    ) -> dict[str, torch.Tensor | int]:
        """Return fixed buffers for the two-GEMM Top-K Decode transform."""

        hidden = int(
            self.store.cfg.get("routed_hidden", self.store.cfg["hidden"])
        )
        intermediate = int(self.store.cfg["moe_inter"])
        cached = self._native8_decode_workspace
        if cached is not None and int(cached["capacity"]) >= int(capacity):
            return cached
        options = {
            "dtype": torch.float8_e4m3fn,
            "device": self.device,
        }
        cached = {
            "capacity": int(capacity),
            "gu": torch.empty(
                int(capacity), 2 * intermediate, hidden, **options
            ),
            "down_tc": torch.empty(
                hidden, int(capacity) * intermediate, **options
            ),
            "input": torch.empty(1, hidden, **options),
            "input_scales": torch.empty(
                1, 1, dtype=torch.float32, device=self.device
            ),
            "activated": torch.empty(
                1, int(capacity) * intermediate, **options
            ),
            "activated_scales": torch.empty(
                1, 1, dtype=torch.float32, device=self.device
            ),
            "gu_scales": torch.empty(
                1,
                int(capacity) * 2 * intermediate,
                dtype=torch.float32,
                device=self.device,
            ),
            "unit_scale": torch.ones(
                1, 1, dtype=torch.float32, device=self.device
            ),
        }
        self._native8_decode_workspace = cached
        return cached

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
        """Execute routed rows through native grouped Tensor-Core GEMMs.

        The input may be a long Prefill block or a one-row Decode step.  Only
        the layer's unique experts are partitioned according to remaining VRAM
        and compact-arena capacity.  Every expert chunk is expanded once to
        E4M3/INT8 bytes and fed to native grouped GEMMs; the old packed GEMV is
        not part of this executor.
        """

        if self._arenas is None or self._metadata is None:
            raise RuntimeError("packed hybrid pool is not ready")
        if not value.is_cuda:
            # One host Prefill block calls this entry once per model layer.
            # Retain the first layer's automatically sized dequant workspace
            # so following layers reuse the exact allocation instead of
            # fragmenting a small-card allocator with 92 slightly different
            # BF16 expert slabs.  The model releases it after the block.
            self._retain_prefill_workspace = True
            device_value, device_ids, device_weights = (
                self._prepare_host_rows_bridge(
                    value,
                    route_ids,
                    route_weights,
                )
            )
            device_result = self.run_rows(
                layer,
                device_value,
                device_ids,
                device_weights,
                activation=activation,
                activation_beta=activation_beta,
                activation_linear_beta=activation_linear_beta,
                limit=limit,
                prefill_default=prefill_default,
            )
            return self._finish_host_rows_bridge(device_result)
        if not value.is_cuda or value.device != self.device:
            raise ValueError("hybrid packed Prefill requires CUDA row tensors")
        if value.ndim != 2 or route_ids.ndim != 2:
            raise ValueError("packed Prefill expects value [N,D], routes [N,K]")
        if route_weights.shape != route_ids.shape:
            raise ValueError("packed Prefill route weights must match routes")
        if int(self._metadata.shape[0]) != 15:
            raise RuntimeError("grouped packed Prefill requires three projections")
        rows = int(value.shape[0])
        top_k = int(route_ids.shape[1])
        if rows <= 0 or top_k <= 0:
            raise ValueError("packed Prefill requires non-empty rows and routes")
        hip_packed_prefill = torch.version.hip is not None
        if not self._prefill_executor_announced:
            print(
                "[cccp-native8] "
                f"executor={self.prefill_executor}; "
                "decode GEMV fallback=forbidden; rows=Prefill+Decode",
                flush=True,
            )
            self._prefill_executor_announced = True

        target_batch = min(
            prefill_moe_batch_size(default=int(prefill_default), maximum=4096),
            rows,
        )
        workspace = self._prefill_workspace_for(
            rows,
            target_batch,
            top_k,
        )
        result = workspace["result"][:rows]
        metadata = workspace["metadata"]
        metadata_host = workspace["metadata_host"]
        from .grouped import activate_gate_up
        from .ops import (
            packed_moe_topk_grouped,
            projection_dequant,
            projection_expand_native8,
        )
        from .fusedext import dense_fp8_quantize_rows_fused

        with self._transfer_lock:
            self._partition_prefill_layer_locked(layer)
            start = 0
            while start < rows:
                count = min(target_batch, rows - start)
                flat_ids = route_ids[start : start + count].reshape(-1)
                unique_global, unique_counts = torch.unique(
                    flat_ids,
                    sorted=True,
                    return_counts=True,
                )
                unique_ids = [
                    int(item)
                    for item in unique_global.detach().cpu().tolist()
                ]
                stop = start + count
                self._last_ids[int(layer)] = list(unique_ids)
                unique_count = len(unique_ids)

                # Routes remain global indices into the complete sorted list.
                # Each expert chunk below stages only its compact experts, then
                # expands them once for the complete token batch.
                local_route_ids = torch.searchsorted(
                    unique_global,
                    route_ids[start:stop],
                ).contiguous()
                flat_local_ids = local_route_ids.reshape(-1)
                token_ids = (
                    torch.arange(
                        count,
                        dtype=torch.long,
                        device=self.device,
                    )
                    .view(-1, 1)
                    .expand(count, top_k)
                    .reshape(-1)
                )
                self._stage.wait()
                input_rows = value[start:stop].to(
                    torch.bfloat16
                ).contiguous()
                flat_weights = route_weights[start:stop].reshape(-1).float()
                batch_result = result[start:stop]
                batch_result.zero_()
                # Plan against the model's per-layer expert ceiling rather
                # than the first layer's observed unique count.  Otherwise a
                # cool first layer permanently undersized the reused scratch
                # and caused unnecessary extra grouped-GEMM submissions.
                planned_capacity = (
                    unique_count
                    if hip_packed_prefill
                    else self._prefill_dequant_chunk_capacity(
                        int(self.store.cfg["n_experts"])
                    )
                )
                chunk_capacity = min(unique_count, planned_capacity)
                expert_chunks: list[tuple[int, int]] = []
                chunk_start = 0
                while chunk_start < unique_count:
                    chunk_stop = min(
                        unique_count, chunk_start + chunk_capacity
                    )
                    while not self._prefill_unique_fits(
                        layer, unique_ids[chunk_start:chunk_stop]
                    ):
                        width = chunk_stop - chunk_start
                        if width <= 1:
                            raise RuntimeError(
                                "packed GPU arena cannot hold one Prefill expert"
                            )
                        chunk_stop = chunk_start + max(1, width // 2)
                        self.prefill_batch_fallbacks += 1
                    expert_chunks.append((chunk_start, chunk_stop))
                    chunk_start = chunk_stop
                effective_capacity = max(
                    stop_index - start_index
                    for start_index, stop_index in expert_chunks
                )
                gu_buffer = down_buffer = None
                native8_workspace = None
                if not hip_packed_prefill:
                    if self._native8_prefill_enabled:
                        native8_workspace = self._prefill_native8_workspace_for(
                            effective_capacity,
                            count * top_k,
                        )
                        gu_buffer = native8_workspace["gu"]
                        down_buffer = native8_workspace["down"]
                    else:
                        gu_buffer, down_buffer = (
                            self._prefill_dequant_workspace_for(
                                effective_capacity
                            )
                        )
                self.prefill_expert_chunk_capacity = max(
                    self.prefill_expert_chunk_capacity,
                    int(effective_capacity),
                )
                self.prefill_layer_unique_max = max(
                    self.prefill_layer_unique_max, unique_count
                )
                print(
                    f"[cccp-native8] layer={layer}; token batch={count}; "
                    f"unique experts={unique_count}; "
                    f"expert chunk={effective_capacity}; "
                    f"groups={len(expert_chunks)}; "
                    "capacity=automatic free-VRAM",
                    flush=True,
                )
                for chunk_start, chunk_stop in expert_chunks:
                    keys = [
                        (int(layer), expert_id)
                        for expert_id in unique_ids[chunk_start:chunk_stop]
                    ]
                    selected = self._ensure_locked(
                        keys,
                        prefetch=False,
                        defer_wait=True,
                    )
                    # ``defer_wait`` keeps the host free while the packed
                    # bytes travel on PinnedStage's copy stream.  The dense
                    # expansion below consumes those bytes on the default
                    # stream, so it must wait on *this batch's* tail event.
                    # The wait previously sat only before the upload and
                    # therefore covered the preceding batch.  Windows/WDDM
                    # could then launch projection_dequant against an arena
                    # slot whose H2D copy was still in flight, eventually
                    # surfacing as cudaErrorIllegalAddress at an unrelated
                    # torch.tensor/copy_ call.  This is a GPU event dependency,
                    # not a CPU synchronize, and preserves batched overlap.
                    self._stage.wait()
                    experts = [selected[key] for key in keys]
                    native8_scales = None
                    if native8_workspace is not None:
                        rows_host, native8_scales = (
                            self._native8_metadata_rows(experts)
                        )
                    else:
                        rows_host = self._metadata_rows(experts)
                    if len(rows_host) < 15:
                        raise RuntimeError(
                            "grouped packed Prefill received incomplete metadata"
                        )
                    chunk_count = chunk_stop - chunk_start
                    metadata_host[:, :chunk_count].copy_(
                        torch.tensor(rows_host[:15], dtype=torch.long)
                    )
                    metadata[:, :chunk_count].copy_(
                        metadata_host[:, :chunk_count],
                        non_blocking=False,
                    )
                    selected_positions = (
                        (flat_local_ids >= chunk_start)
                        & (flat_local_ids < chunk_stop)
                    ).nonzero(as_tuple=False).reshape(-1)
                    chunk_ids = (
                        flat_local_ids.index_select(0, selected_positions)
                        - chunk_start
                    )
                    order = torch.argsort(chunk_ids)
                    sorted_ids = chunk_ids[order].contiguous()
                    sorted_positions = selected_positions[order].contiguous()
                    sorted_tokens = token_ids.index_select(
                        0, sorted_positions
                    ).contiguous()
                    sorted_weights = flat_weights.index_select(
                        0, sorted_positions
                    ).contiguous()
                    group_ids = torch.arange(
                        chunk_count,
                        dtype=torch.long,
                        device=self.device,
                    )
                    offsets = torch.searchsorted(
                        sorted_ids, group_ids, right=True
                    ).to(torch.int32)
                    if hip_packed_prefill:
                        group_offsets = torch.empty(
                            chunk_count + 1,
                            dtype=torch.int32,
                            device=self.device,
                        )
                        group_offsets[0] = 0
                        group_offsets[1:].copy_(offsets)
                        packed_hidden = workspace["packed_hidden"][
                            : int(sorted_tokens.numel())
                        ]
                        packed_result = workspace["packed_result"][:count]
                        packed_moe_topk_grouped(
                            input_rows,
                            sorted_tokens,
                            group_ids,
                            group_offsets,
                            sorted_weights,
                            metadata[:, :chunk_count].contiguous(),
                            activation=activation,
                            activation_beta=float(activation_beta),
                            activation_linear_beta=(
                                0.0
                                if activation_linear_beta is None
                                else float(activation_linear_beta)
                            ),
                            limit=float(limit),
                            hidden_workspace=packed_hidden,
                            result=packed_result,
                        )
                        batch_result.add_(packed_result)
                    else:
                        grouped_input = input_rows.index_select(
                            0, sorted_tokens
                        ).contiguous()
                        if native8_workspace is not None:
                            if native8_scales is None:
                                raise RuntimeError(
                                    "native8 projection scales are missing"
                                )
                            projection_expand_native8(
                                metadata[:, :chunk_count].contiguous(),
                                gu_buffer[:chunk_count],
                                down_buffer[:chunk_count],
                            )
                            scale_values = torch.tensor(
                                native8_scales,
                                dtype=torch.float32,
                                device=self.device,
                            )
                            gu_scales = native8_workspace["gu_scales"][
                                :chunk_count
                            ]
                            down_scales = native8_workspace["down_scales"][
                                :chunk_count
                            ]
                            intermediate = int(self.store.cfg["moe_inter"])
                            gu_scales[:, :intermediate].copy_(
                                scale_values[:, 0:1]
                            )
                            gu_scales[:, intermediate:].copy_(
                                scale_values[:, 1:2]
                            )
                            down_scales.copy_(scale_values[:, 2:3])
                            routed_count = int(grouped_input.shape[0])
                            native_input = native8_workspace["input"][
                                :routed_count
                            ]
                            input_scales = native8_workspace["input_scales"][
                                :routed_count
                            ]
                            if dense_fp8_quantize_rows_fused(
                                grouped_input, native_input, input_scales
                            ) is None:
                                raise RuntimeError(
                                    "native E4M3 activation quantizer rejected Prefill"
                                )
                            gate_up = torch._scaled_grouped_mm(
                                native_input,
                                gu_buffer[:chunk_count].transpose(1, 2),
                                scale_a=input_scales.view(-1),
                                scale_b=gu_scales,
                                offs=offsets,
                                out_dtype=torch.bfloat16,
                                use_fast_accum=True,
                            )
                        else:
                            projection_dequant(
                                metadata[:, :chunk_count].contiguous(),
                                gu_buffer[:chunk_count],
                                down_buffer[:chunk_count],
                            )
                            gate_up = torch._grouped_mm(
                                grouped_input,
                                gu_buffer[:chunk_count].transpose(1, 2),
                                offs=offsets,
                            )
                        gate, up = gate_up.chunk(2, dim=-1)
                        if float(limit) > 0.0:
                            gate = gate.clamp(max=float(limit))
                            up = up.clamp(
                                min=-float(limit), max=float(limit)
                            )
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
                        activated = activated.contiguous()
                        if native8_workspace is not None:
                            routed_count = int(activated.shape[0])
                            native_activated = native8_workspace["activated"][
                                :routed_count
                            ]
                            activated_scales = native8_workspace[
                                "activated_scales"
                            ][:routed_count]
                            if dense_fp8_quantize_rows_fused(
                                activated,
                                native_activated,
                                activated_scales,
                            ) is None:
                                raise RuntimeError(
                                    "native E4M3 activation quantizer rejected MoE hidden rows"
                                )
                            down = torch._scaled_grouped_mm(
                                native_activated,
                                down_buffer[:chunk_count].transpose(1, 2),
                                scale_a=activated_scales.view(-1),
                                scale_b=down_scales,
                                offs=offsets,
                                out_dtype=torch.bfloat16,
                                use_fast_accum=True,
                            )
                        else:
                            down = torch._grouped_mm(
                                activated,
                                down_buffer[:chunk_count].transpose(1, 2),
                                offs=offsets,
                            )
                        batch_result.index_add_(
                            0,
                            sorted_tokens,
                            down.float() * sorted_weights.unsqueeze(1),
                        )
                    self.prefill_expert_chunk_submissions += 1
                counts_host = unique_counts.detach().cpu().tolist()
                self.route_counts.update(
                    {
                        (int(layer), expert_id): int(hit_count)
                        for expert_id, hit_count in zip(
                            unique_ids,
                            counts_host,
                        )
                    }
                )
                self.prefill_batch_submissions += 1
                self.prefill_batch_rows += count
                self.prefill_batch_max = max(self.prefill_batch_max, count)
                start = stop
            # Expert BF16 expansion is a per-layer scratch, not residency.
            # Release it before the next layer's full-batch Attention/Indexer
            # so both phases reuse the same CUDA allocator blocks.  Stream-
            # ordered allocator semantics keep queued grouped GEMMs safe.
            if not getattr(self, "_retain_prefill_workspace", False):
                self._prefill_dequant_workspace = None
        return result

    def release_host_rows_workspace(self) -> None:
        """Release the block-scoped hybrid Prefill expansion workspace."""

        had_workspace = (
            self._prefill_dequant_workspace is not None
            or self._prefill_native8_workspace is not None
        )
        self._retain_prefill_workspace = False
        if had_workspace and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._prefill_dequant_workspace = None
        self._prefill_native8_workspace = None
        if had_workspace and self.device.type == "cuda":
            torch.cuda.empty_cache()

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
        # Public RAM-Dense/CUDA-MoE bridge.  Only the token-sized latent,
        # Top-K route and token-sized result cross PCIe; compact expert bytes
        # continue through the ordinary arena miss path.  The fixed buffers
        # avoid per-layer tensor allocation and never materialize an expert.
        if not value.is_cuda:
            input_buffer, ids_buffer, weights_buffer = (
                self._prepare_host_bridge(
                    value,
                    route_ids,
                    route_weights,
                )
            )
            device_result = self.run(
                layer,
                input_buffer,
                ids_buffer,
                weights_buffer,
                activation=activation,
                activation_beta=activation_beta,
                activation_linear_beta=activation_linear_beta,
                limit=limit,
            )
            return self._finish_host_bridge(device_result)
        if (
            self._workspaces is None
            or self._metadata is None
            or self._route_ids is None
            or self._ordered_weights is None
        ):
            raise RuntimeError("packed hybrid pool is not ready")
        if value.ndim != 2 or int(value.shape[0]) != 1:
            raise RuntimeError(
                "packed decode GEMV accepts exactly one token; use run_rows "
                "for multi-token Prefill"
            )
        # 这是 RAM 地址选择所需的唯一 GPU→CPU 路由同步；权重计算仍留在 GPU。
        # 在融合核排入默认流之前不允许后台预取复用槽位。释放锁后，
        # PinnedStage 的 wait_stream 会把后续覆盖排在本核之后。
        with self._transfer_lock:
            self._restore_decode_arena_locked()
            self._warm_profile_hot_locked()
            device_hit, expert_ids = self._device_route_metadata(
                layer,
                route_ids,
            )
            if device_hit:
                count = int(route_ids.numel())
                hidden, output, result = self._workspaces
                from .ops import packed_moe_topk

                return packed_moe_topk(
                    value.to(torch.bfloat16),
                    self._route_ids[:count],
                    route_weights.reshape(-1).float().contiguous(),
                    self._metadata[:, :count],
                    activation=activation,
                    activation_beta=float(activation_beta),
                    activation_linear_beta=(
                        0.0
                        if activation_linear_beta is None
                        else float(activation_linear_beta)
                    ),
                    limit=float(limit),
                    hidden_workspace=hidden[:count],
                    output_workspace=output[:count],
                    result=result,
                    grouped_prefix=-1,
                    **self.store.man.projection_operator_capability(layer),
                )
            if expert_ids is None:
                expert_ids = self._host_route_ids(route_ids)
                self._record_route_ids(layer, expert_ids)
            reused = self._reuse_route_plan(
                layer,
                value,
                expert_ids,
                route_weights,
                activation=activation,
                activation_beta=activation_beta,
                activation_linear_beta=activation_linear_beta,
                limit=float(limit),
            )
            if reused is not None:
                return reused
            self._last_ids[layer] = expert_ids
            keys = [(layer, expert_id) for expert_id in expert_ids]
            self._ensure_decode_route_capacity_locked(keys)
            async_stage = (
                os.environ.get("CCCP_KIMI_ASYNC_STAGE", "1") != "0"
                and os.environ.get("CCCP_KIMI_LAYER_TIMING", "0") == "0"
            )
            selected = self._ensure_locked(
                keys,
                prefetch=False,
                defer_wait=async_stage,
            )
            experts = [selected[key] for key in keys]
            self._copy_metadata(experts)
            p12_positions = [
                position
                for position, expert in enumerate(experts)
                if (
                    len(expert) == 2
                    and
                    expert[0].bits == 12
                    and expert[0].dim in (4, 8)
                    and expert[1].bits == 12
                    and expert[1].dim in (4, 8)
                )
            ]
            generic_positions = [
                position
                for position in range(len(experts))
                if position not in p12_positions
            ]
            identity_order = self._set_route_order(
                p12_positions,
                generic_positions,
            )
            if identity_order:
                ordered_weights = (
                    route_weights.reshape(-1).float().contiguous()
                )
            else:
                torch.index_select(
                    route_weights.reshape(-1).float().contiguous(),
                    0,
                    self._route_ids[: len(experts)],
                    out=self._ordered_weights[: len(experts)],
                )
                ordered_weights = self._ordered_weights[: len(experts)]
            self._save_route_plan(
                layer,
                expert_ids,
                keys,
                experts,
                len(p12_positions),
                identity_order,
            )
            # 大块 expert DMA 在 copy stream 继续进行时，CPU 已完成指针元数据
            # 和路由顺序构造；仅在融合核真正读取槽位前建立 GPU 事件依赖。
            # 这消除每层 cudaEventSynchronize 后的主机唤醒空洞，不改变槽位
            # 生命周期，也不把 packed 索引展开成中间矩阵。
            if async_stage:
                self._stage.wait()
            hidden, output, result = self._workspaces
            from .ops import packed_moe_topk

            computed = packed_moe_topk(
                value.to(torch.bfloat16),
                self._route_ids[: len(experts)],
                ordered_weights,
                self._metadata[:, : len(experts)],
                activation=activation,
                activation_beta=float(activation_beta),
                activation_linear_beta=(
                    0.0
                    if activation_linear_beta is None
                    else float(activation_linear_beta)
                ),
                limit=float(limit),
                hidden_workspace=hidden[: len(experts)],
                output_workspace=output[: len(experts)],
                result=result,
                grouped_prefix=len(p12_positions),
                **self.store.man.projection_operator_capability(
                    layer
                ),
            )
            return computed

    def resize_gpu_arenas(
        self,
        budget: int,
        *,
        staging_timeout_s: float = 30.0,
        force_rebuild: bool = False,
    ) -> tuple[int, int]:
        del staging_timeout_s
        if getattr(self, "fixed_extreme_residency", False):
            raise RuntimeError(
                "极限模式 GPU-only 专家不可驱逐；请降低上下文后重新启动"
            )
        budget = max(0, int(budget))
        old = self.gpu_arena_bytes
        if budget == old and not force_rebuild:
            self.budget = budget
            return old, old
        warmed_before_resize = self._profile_hot_ready
        if self._arena_phase == "decode":
            self._decode_arena_target_budget = min(
                self._decode_arena_target_budget or budget,
                budget,
            )
        with self._transfer_lock:
            torch.cuda.synchronize(self.device)
            self._stage.wait()
            self.cache.clear()
            self._arenas = None
            self._device_codebooks = {}
            self._native8_codebooks = {}
            self._native8_codebook_scales = {}
            self._native8_prefill_enabled = False
            self._last_ids.clear()
            self._route_plans.clear()
            self.profile_hot_keys = ()
            self.profile_hot_cache_enabled = False
            self.profile_hot_slots = 0
            self._profile_hot_ready = False
            self._workspaces = None
            self._prefill_workspace = None
            self._prefill_dequant_workspace = None
            self._prefill_native8_workspace = None
            self._native8_decode_workspace = None
            self._native8_route_scales_host = None
            self._native8_route_scales = None
            self._native8_decode_offsets = None
            self._metadata = None
            self._metadata_host = None
            self._slot_directory = None
            self._slot_update_host = None
            self._slot_scale_directory = None
            self._slot_scale_update_host = None
            self._route_hit_mask = None
            self._route_all_hit = None
            self._route_all_hit_host = None
            self._route_host_ids = None
            self._route_copy_done = None
            self._route_ids = None
            self._ordered_weights = None
            self._compact_profile_all_resident = False
            self._packed_moe_graphs.clear()
            self._packed_graph_input = None
            self._packed_graph_weights = None
            self.bytes = 0
            self.budget = budget
            gc.collect()
            torch.cuda.empty_cache()
            self.build_gpu_arenas()
            if warmed_before_resize and self._arena_phase == "decode":
                # Runtime KV pressure intentionally starts the smaller arena
                # empty and lets demand refill it. Re-uploading the complete
                # corpus warm set during a live request would stall Decode and
                # immediately consume the headroom that triggered shrinkage.
                self._profile_hot_ready = True
        return old, self.gpu_arena_bytes

    def activate_prefill_arena(self) -> tuple[int, int] | None:
        """Give a long batched Prefill enough expert-expansion workspace."""

        target = int(self._prefill_arena_target_budget)
        if target <= 0 or self._extreme_specs is not None:
            return None
        current = self.gpu_arena_bytes
        if current <= target:
            self._arena_phase = "prefill"
            return current, current
        self._arena_phase = "prefill"
        changed = self.resize_gpu_arenas(target)
        print(
            "[cccp-vram-plan] phase=prefill-switch "
            f"arena={changed[0] / 2**30:.2f}->"
            f"{changed[1] / 2**30:.2f}GiB",
            flush=True,
        )
        return changed

    def activate_decode_arena(self) -> tuple[int, int] | None:
        """Restore the largest safe hot arena after batch Prefill releases scratch."""

        target = int(self._decode_arena_target_budget)
        if target <= 0 or self._extreme_specs is not None:
            return None
        current = self.gpu_arena_bytes
        ready = self._decode_runtime_ready()
        if current >= target and ready:
            self._arena_phase = "decode"
            return current, current
        self._arena_phase = "decode"
        changed = self.resize_gpu_arenas(
            max(current, target),
            force_rebuild=not ready,
        )
        if not self._decode_runtime_ready():
            raise RuntimeError(
                "packed hybrid Decode arena rebuild did not create runtime "
                f"buffers (pinned_experts={len(self.pinned)}, "
                f"arena={self.gpu_arena_bytes / 2**30:.2f}GiB)"
            )
        print(
            "[cccp-vram-plan] phase=decode-switch "
            f"arena={changed[0] / 2**30:.2f}->"
            f"{changed[1] / 2**30:.2f}GiB",
            flush=True,
        )
        return changed

    def _decode_runtime_ready(self) -> bool:
        """Return whether the fixed Decode arena and all launch buffers exist."""

        return (
            self._arenas is not None
            and self._workspaces is not None
            and self._metadata is not None
            and self._route_ids is not None
            and self._ordered_weights is not None
        )

    def trim_to(self, budget: int) -> None:
        if (
            getattr(self, "fixed_extreme_residency", False)
            and int(budget) < self.gpu_storage_bytes
        ):
            raise RuntimeError(
                "极限模式没有可收缩专家副本；请降低上下文后重新启动"
            )
        budget = max(0, int(budget))
        if self._decode_arena_target_budget > budget:
            self._decode_arena_target_budget = budget
        if self.gpu_arena_bytes > budget:
            self.resize_gpu_arenas(budget)
        else:
            self.budget = budget


__all__ = [
    "HostPackedWeight",
    "PackedHybridPool",
    "PackedExpertSignature",
    "PackedWeightSignature",
    "allocate_packed_slots",
]

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
from .ops.sm90_grouped import (
    build_deepgemm_grouped_layout,
    deepgemm_grouped_alignment,
    deepgemm_grouped_padded_rows,
    DeepGEMMGroupedWorkspace,
    execute_grouped_fp8,
    grouped_jit_bucket,
    projection_block_scales,
    row_block_scales,
    select_grouped_fp8_backend,
)
from .ops.codebook import compile_shared_codebook_image
from .prefill import (
    prefill_moe_batch_size,
    should_retain_prefill_workspace,
)


def _packed_executor_log_kind(
    *, rows: int, batch_start: int, announced: bool
) -> str | None:
    """Return the bounded log event for one public packed execution call."""

    if int(batch_start) != 0:
        return None
    if int(rows) > 1:
        return "prefill"
    if int(rows) == 1 and not announced:
        return "decode"
    return None


def _process_memory_room(
    *,
    process_limit: int,
    allocated: int,
    allocator_reserved: int,
    safety_reserve: int,
) -> int:
    """Return allocatable process room without double-using cached blocks."""

    committed = max(int(allocated), int(allocator_reserved))
    return max(0, int(process_limit) - committed - int(safety_reserve))


def _projection_kernel_chunk_limit(
    *,
    hidden: int,
    intermediate: int,
    execution_bytes: int,
    native8: bool,
) -> int | None:
    """Return the legacy projection offset ceiling when it still applies.

    Native8 expansion uses 64-bit projection offsets end-to-end, so its only
    ceiling is live VRAM.  The BF16 compatibility kernel still exposes the
    older signed-int32 gate/up offset and must stay below 2 GiB.
    """

    if native8:
        return None
    gate_up_bytes = (
        2 * int(intermediate) * int(hidden) * int(execution_bytes)
    )
    return max(1, (2**31 - 1) // max(1, gate_up_bytes))


def _prefill_workspace_matches(
    cached: dict[str, object] | None,
    *,
    rows: int,
    micro_batch: int,
    top_k: int,
    intermediate: int,
    hidden: int,
    compact: bool,
) -> bool:
    """Return whether one public MoE workspace supports the next call.

    Long native8 Prefill and short compact Prefill share result/metadata
    storage, but only the compact executor owns packed activation buffers.
    Treating a preceding long workspace as universally reusable made the
    first short request after a long context fail with ``KeyError``.
    """

    if cached is None:
        return False
    common = (
        int(cached["rows"]) >= int(rows)
        and int(cached["micro_batch"]) >= int(micro_batch)
        and int(cached["top_k"]) == int(top_k)
        and int(cached["intermediate"]) == int(intermediate)
        and int(cached["hidden_size"]) == int(hidden)
    )
    if not common:
        return False
    return not compact or (
        "packed_hidden" in cached and "packed_result" in cached
    )


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
    codebook_scales: torch.Tensor | None = None
    device_route_probe: bool = False
    wait_for_device_cache: bool = False
    prelaunched_result: torch.Tensor | None = None
    wait_for_routed: bool = False
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

    def device_slot_layout(
        self,
    ) -> tuple[
        tuple[PackedExpertSignature, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, ...], ...],
    ]:
        """Return immutable signature segments and stable slot addresses."""

        signatures: list[PackedExpertSignature] = []
        offsets = [0]
        pointers: list[int] = []
        projection_offsets: list[tuple[int, ...]] = []
        for signature, arena in self.arenas.items():
            signatures.append(signature)
            running = 0
            current_offsets = []
            for weight in signature.weights:
                current_offsets.append(running)
                running += int(weight.raw_bytes)
            projection_offsets.append(tuple(current_offsets))
            for slot in range(arena.book.count):
                pointers.append(int(arena.raw[0][slot].data_ptr()))
            offsets.append(len(pointers))
        return (
            tuple(signatures),
            tuple(offsets),
            tuple(pointers),
            tuple(projection_offsets),
        )

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


def _widest_prefill_layer_specs(
    by_layer: Mapping[int, Counter[PackedExpertSignature]],
    *,
    resident_codebooks: bool,
) -> tuple[dict[PackedExpertSignature, int], int]:
    """Return the physical widest complete layer, not a mixed envelope."""

    if not by_layer:
        return {}, 0
    widest = max(
        by_layer.values(),
        key=lambda counts: sum(
            signature.storage_bytes(resident_codebooks) * int(count)
            for signature, count in counts.items()
        ),
    )
    specs = {signature: int(count) for signature, count in widest.items()}
    total = sum(
        signature.storage_bytes(resident_codebooks) * count
        for signature, count in specs.items()
    )
    return specs, total


class PackedHybridPool:
    """全量紧凑 RAM + 有界稳定 VRAM 的配置驱动 Top-K 专家池。"""

    device_routed = True
    full_resident = False
    # Short prompts are cheaper and substantially less disruptive through
    # the exact Decode executor: the long-Prefill arena would otherwise evict
    # the hot expert slab, rebuild it at the phase boundary, and erase the
    # cache benefit before generation begins.  The engine reads this public
    # capability rather than branching on a GPU or model name.
    short_reset_decode_tokens = 64
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
    # Capability advertised before RAM pages and the Decode arena exist.  The
    # runtime property below becomes true only after Linux/NVIDIA UVA, the
    # device-side segmented LRU and compact Q8 math are all ready.  Model code
    # may use this candidate bit to allocate fixed TP1 Router/Dense buffers,
    # but it must not capture or launch a graph until the runtime bit is true.
    fixed_token_graph_candidate = bool(
        os.name != "nt" and torch.version.hip is None
    )

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
        self._compact_q8_codebooks: dict[int, torch.Tensor] = {}
        self._compact_q8_codebook_scales: dict[int, float] = {}
        self._compact_q8_decode_enabled = False
        self.native8_rows_supported = False
        self._native8_grouped_backend: str | None = None
        self._prefill_planned_chunk_capacity = 0
        self._host_pinned_bytes = 0
        self.host_dma_mode = "pageable"
        self._host_registrations: dict[int, int] = {}
        self._resident_codebooks = (
            os.environ.get("CCCP_RESIDENT_CODEBOOKS", "1") != "0"
        )
        self.initial_free_slots = 0
        self._arenas: _PackedArenas | None = None
        self._default_arena_specs: dict[PackedExpertSignature, int] = {}
        self._prefill_partition_layer: int | None = None
        self._prefill_layer_specs: dict[
            int, dict[PackedExpertSignature, int]
        ] = {}
        self._prefill_layer_local_maps: dict[int, torch.Tensor] = {}
        self._prefill_spare_arenas: _PackedArenas | None = None
        self._prefill_hot_arenas: _PackedArenas | None = None
        self._prefill_hot_selected: dict[
            tuple[int, int], DeviceExpert
        ] = {}
        self._prefill_prepared: dict[
            int,
            tuple[
                _PackedArenas,
                dict[tuple[int, int], DeviceExpert],
                torch.cuda.Event,
            ],
        ] = {}
        self._decode_arena_target_budget = 0
        self._prefill_arena_target_budget = 0
        # This pool exposes one public multi-token packed executor regardless
        # of the model adapter.  Start every hybrid layout in Prefill planning
        # mode so the initial VRAM split reserves expansion/GEMM workspace and
        # layer ping-pong slabs.  Detecting this from DSV4-only config keys left
        # GLM/Kimi with a decode-sized hot arena and reduced their grouped
        # Prefill to one expert per submission under the same hard VRAM cap.
        self._arena_phase = "prefill"
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
        self._prefill_directory_dirty = False
        self._decode_codebook_scales: torch.Tensor | None = None
        self._decode_codebook_scales_host: torch.Tensor | None = None
        self._compact_profile_all_resident = False
        self._route_hit_mask: torch.Tensor | None = None
        self._route_all_hit: torch.Tensor | None = None
        self._route_all_hit_host: torch.Tensor | None = None
        self._route_host_ids: torch.Tensor | None = None
        self._route_copy_done: torch.cuda.Event | None = None
        self._device_cache_enabled = False
        self._cache_signature_of_id: torch.Tensor | None = None
        self._cache_segment_offsets: torch.Tensor | None = None
        self._cache_slot_for_id: torch.Tensor | None = None
        self._cache_id_of_slot: torch.Tensor | None = None
        self._cache_last_used: torch.Tensor | None = None
        self._cache_step: torch.Tensor | None = None
        self._cache_route_slots: torch.Tensor | None = None
        self._cache_input_route_ids: torch.Tensor | None = None
        self._cache_input_logical_ids: torch.Tensor | None = None
        self._cache_source_ids: torch.Tensor | None = None
        self._cache_destination_slots: torch.Tensor | None = None
        self._cache_counts: torch.Tensor | None = None
        self._cache_profile_totals: torch.Tensor | None = None
        self._cache_source_ptrs: torch.Tensor | None = None
        self._cache_destination_ptrs: torch.Tensor | None = None
        self._cache_signature_bytes: torch.Tensor | None = None
        self._cache_projection_offsets: torch.Tensor | None = None
        self._cache_metadata_of_id: torch.Tensor | None = None
        self._cache_native_scales_of_id: torch.Tensor | None = None
        self._device_cache_stream: torch.cuda.Stream | None = None
        self._device_cache_ready: torch.cuda.Event | None = None
        self._device_routed_stream: torch.cuda.Stream | None = None
        self._device_routed_ready: torch.cuda.Event | None = None
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
        self._decode_executor_announced = False
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
        self._prefill_native8_workspace: dict[str, object] | None = (
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
        self._compact_q8_activation_workspaces: tuple[
            torch.Tensor,
            torch.Tensor,
        ] | None = None
        self._packed_moe_graphs: dict[object, object] = {}
        self._packed_graph_input: torch.Tensor | None = None
        self._packed_graph_weights: torch.Tensor | None = None
        # The same fixed-address graph contract is implemented by the
        # full-resident packed pool.  Here the expert graph also contains the
        # device LRU plan, registered-host UVA gather and metadata publication,
        # so changing route IDs never changes a captured pointer.
        self.devices = (self.device,)
        self._bound_hidden_inputs: dict[int, tuple] = {}
        self._fixed_expert_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._fixed_route_graphs: dict[int, tuple[torch.cuda.CUDAGraph, ...]] = {}
        self._fixed_graph_batches: dict[int, object] = {}
        self._fixed_graph_stream: torch.cuda.Stream | None = None
        self._fixed_source_events: dict[int, torch.cuda.Event] = {}
        self._fixed_done_events: dict[int, torch.cuda.Event] = {}
        self._fixed_output_events: dict[int, torch.cuda.Event] = {}
        self._fixed_graph_dependencies: list[object] = []
        self.fixed_graph_generation = 0
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
        value = os.environ.get("CCCP_VQ_SLOT_MIX", "").strip()
        if not value or value.lower() in ("model", "static", "off", "0"):
            return None
        output: dict[str, float] = {}
        for item in value.split(","):
            name, separator, weight = item.partition("=")
            if not separator:
                raise ValueError(
                    "CCCP_VQ_SLOT_MIX must use tier=weight entries"
                )
            name = name.strip().lower()
            if name not in {"x", "w", "v", "vv"}:
                raise ValueError(f"unknown packed slot tier: {name}")
            output[name] = max(0.0, float(weight))
        if not any(output.values()):
            raise ValueError("CCCP_VQ_SLOT_MIX contains no positive weight")
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
        primary = 0 if self._arenas is None else self._arenas.nbytes
        spare = (
            0
            if self._prefill_spare_arenas is None
            else self._prefill_spare_arenas.nbytes
        )
        hot = (
            0
            if self._prefill_hot_arenas is None
            else self._prefill_hot_arenas.nbytes
        )
        return primary + spare + hot

    @property
    def gpu_storage_bytes(self) -> int:
        workspace = 0
        if self._workspaces is not None:
            workspace += sum(tensor.nbytes for tensor in self._workspaces)
        if self._compact_q8_activation_workspaces is not None:
            workspace += sum(
                tensor.nbytes
                for tensor in self._compact_q8_activation_workspaces
            )
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
        workspace += sum(
            codebook.nbytes
            for codebook in self._compact_q8_codebooks.values()
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
                    # Linux/TCC Decode consumes the compact payload through a
                    # graph-captured UVA gather.  Portable+mapped keeps those
                    # pages GPU-addressable; Windows retains its established
                    # copy-engine path until WDDM alias handling is enabled.
                    3 if os.name != "nt" else 0,
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
        if direct_complete and os.name != "nt":
            # Seed the corpus-hot slots through the mature copy engine once,
            # then hand their exact ownership to the graph-resident cache.
            # Runtime Decode never returns to the Python LRU afterwards.
            if self._prefill_hot_arenas is not None:
                self._warm_prefill_hot_locked()
            else:
                self._warm_profile_hot_locked()
            self._build_device_cache_runtime()
        return self._host_pinned_bytes / 2**30

    def _build_device_cache_runtime(self) -> None:
        """Publish the exact packed-cache state and descriptors on CUDA."""

        self._device_cache_enabled = False
        if (
            self.device.type != "cuda"
            or torch.version.hip is not None
            or self._arenas is None
            or self._prefill_spare_arenas is not None
            or not self.pinned
            or self.host_dma_mode != "direct-registered"
        ):
            return
        (
            signatures,
            segment_offsets,
            destination_ptrs,
            projection_offsets,
        ) = self._arenas.device_slot_layout()
        if not signatures or not destination_ptrs:
            return
        projection_count = signatures[0].projection_count
        if any(
            signature.projection_count != projection_count
            for signature in signatures
        ):
            raise RuntimeError(
                "graph-resident packed cache requires one projection count"
            )
        signature_index = {
            signature: index for index, signature in enumerate(signatures)
        }
        n_layers = int(self.store.cfg["n_layers"])
        n_experts = int(self.store.cfg["n_experts"])
        logical_count = n_layers * n_experts
        metadata_rows = int(self._metadata.shape[0])
        signature_of_id = torch.full(
            (logical_count,), -1, dtype=torch.int32
        )
        source_ptrs = torch.zeros(logical_count, dtype=torch.int64)
        metadata_of_id = torch.zeros(
            logical_count, metadata_rows, dtype=torch.int64
        )
        native_scales_of_id = torch.zeros(
            logical_count, 3, dtype=torch.float32
        )
        for (layer, expert_id), expert in self.pinned.items():
            logical_id = int(layer) * n_experts + int(expert_id)
            signature = PackedExpertSignature.of(expert)
            signature_of_id[logical_id] = signature_index[signature]
            raw = _contiguous_expert_raw(expert)
            if raw is None:
                raise RuntimeError(
                    "graph-resident packed cache requires contiguous experts"
                )
            if raw.data_ptr() not in self._host_registrations:
                raise RuntimeError(
                    "graph-resident packed cache found an unmapped host expert"
                )
            # On Linux/TCC with UVA identity the registered host VA is the
            # device-visible alias.  WDDM is intentionally not enabled above.
            source_ptrs[logical_id] = int(raw.data_ptr())
            device_view = tuple(
                DevicePackedWeight(
                    raw=weight.raw,
                    cb=self._device_codebooks[weight.cb.data_ptr()],
                    rows=weight.rows,
                    cols=weight.cols,
                    blocks=weight.blocks,
                    dim=weight.dim,
                    bits=weight.bits,
                )
                for weight in expert
            )
            if self._compact_q8_decode_enabled:
                rows, native_scales = self._decode_codebook_metadata_rows(
                    [device_view]
                )
                native_scales_of_id[logical_id].copy_(
                    torch.tensor(native_scales[0], dtype=torch.float32)
                )
            else:
                rows = build_runtime_metadata_rows([device_view])
            rows = rows[:metadata_rows]
            metadata_of_id[logical_id].copy_(
                torch.tensor([row[0] for row in rows], dtype=torch.int64)
            )

        slot_for_id = torch.full(
            (logical_count,), -1, dtype=torch.int32
        )
        id_of_slot = torch.full(
            (len(destination_ptrs),), -1, dtype=torch.int32
        )
        last_used = torch.zeros(len(destination_ptrs), dtype=torch.int64)
        offsets_by_signature = {
            signature: int(segment_offsets[index])
            for index, signature in enumerate(signatures)
        }
        recency = {
            key: timestamp
            for timestamp, key in enumerate(self.cache.keys(), start=1)
        }
        for key, (signature, lease) in self._arenas.leases.items():
            logical_id = int(key[0]) * n_experts + int(key[1])
            slot = offsets_by_signature[signature] + int(lease.slot)
            slot_for_id[logical_id] = slot
            id_of_slot[slot] = logical_id
            last_used[slot] = int(recency.get(key, 0))

        device = self.device
        top_k = int(self.store.cfg["top_k"])
        self._cache_signature_of_id = signature_of_id.to(device)
        self._cache_segment_offsets = torch.tensor(
            segment_offsets, dtype=torch.int32, device=device
        )
        self._cache_slot_for_id = slot_for_id.to(device)
        self._cache_id_of_slot = id_of_slot.to(device)
        self._cache_last_used = last_used.to(device)
        self._cache_step = torch.tensor(
            max(recency.values(), default=0), dtype=torch.int64, device=device
        )
        self._cache_route_slots = torch.full(
            (top_k,), -1, dtype=torch.int32, device=device
        )
        self._cache_input_route_ids = torch.full(
            (top_k,), -1, dtype=torch.long, device=device
        )
        self._cache_input_logical_ids = torch.full(
            (top_k,), -1, dtype=torch.long, device=device
        )
        self._cache_source_ids = torch.full(
            (top_k,), -1, dtype=torch.int32, device=device
        )
        self._cache_destination_slots = torch.full(
            (top_k,), -1, dtype=torch.int32, device=device
        )
        self._cache_counts = torch.zeros(4, dtype=torch.int32, device=device)
        self._cache_profile_totals = (
            torch.zeros(4, dtype=torch.int64, device=device)
            if os.environ.get("CCCP_CACHE_TELEMETRY", "0") != "0"
            else None
        )
        self._cache_source_ptrs = source_ptrs.to(device)
        self._cache_destination_ptrs = torch.tensor(
            destination_ptrs, dtype=torch.int64, device=device
        )
        self._cache_signature_bytes = torch.tensor(
            [signature.raw_slot_bytes for signature in signatures],
            dtype=torch.int64,
            device=device,
        )
        self._cache_projection_offsets = torch.tensor(
            projection_offsets, dtype=torch.int64, device=device
        )
        self._cache_metadata_of_id = metadata_of_id.to(device)
        self._cache_native_scales_of_id = (
            native_scales_of_id.to(device)
            if self._compact_q8_decode_enabled
            else None
        )
        if self._device_cache_stream is None:
            self._device_cache_stream = torch.cuda.Stream(
                device=device,
                priority=-1,
            )
        if self._device_cache_ready is None:
            self._device_cache_ready = torch.cuda.Event()
        if self._device_routed_stream is None:
            self._device_routed_stream = torch.cuda.Stream(
                device=device,
                priority=-1,
            )
        if self._device_routed_ready is None:
            self._device_routed_ready = torch.cuda.Event()
        self._device_cache_enabled = True
        print(
            "[cccp-cache] controller=device-segmented-lru "
            "route_d2h=0 python_lru=0 transfer=graph-uva-multibank "
            f"slots={len(destination_ptrs)} signatures={len(signatures)} "
            "blocks_per_expert=64",
            flush=True,
        )

    def _drop_device_cache_runtime(self) -> None:
        """Release every descriptor that belongs to the current arena slab."""

        self._drop_fixed_token_graphs()
        self._device_cache_enabled = False
        for name in (
            "_cache_signature_of_id",
            "_cache_segment_offsets",
            "_cache_slot_for_id",
            "_cache_id_of_slot",
            "_cache_last_used",
            "_cache_step",
            "_cache_route_slots",
            "_cache_input_route_ids",
            "_cache_input_logical_ids",
            "_cache_source_ids",
            "_cache_destination_slots",
            "_cache_counts",
            "_cache_profile_totals",
            "_cache_source_ptrs",
            "_cache_destination_ptrs",
            "_cache_signature_bytes",
            "_cache_projection_offsets",
            "_cache_metadata_of_id",
            "_cache_native_scales_of_id",
        ):
            setattr(self, name, None)
    def device_cache_telemetry(self) -> dict[str, int | float]:
        """Return opt-in device-LRU counters after the caller's sync point."""

        totals = self._cache_profile_totals
        if totals is None:
            return {}
        routes, unique, hits, fetches = (
            int(value) for value in totals.detach().cpu().tolist()
        )
        return {
            "routes": routes,
            "unique": unique,
            "hits": hits,
            "fetches": fetches,
            "hit_rate": (float(hits) / unique if unique > 0 else 0.0),
        }

    @property
    def decode_executor_name(self) -> str:
        """Describe the actual compact-weight Decode math path."""

        if self._compact_q8_decode_enabled:
            return "cuda.compact-vq-q8-dp4a"
        if self.native8_rows_supported:
            return "cuda.compact-vq-e4m3-direct-dot"
        return "cuda.compact-vq-bf16-direct-dot"

    @property
    def fixed_token_graph_capable(self) -> bool:
        """Return whether dynamic experts can safely enter a parent graph.

        Stable arena addresses alone are insufficient: a cache miss must also
        be resolved entirely on-device.  Linux registered-host UVA provides
        stable source pointers, the segmented LRU owns stable destination
        pointers, and the common native-8 codebook kernels consume resident
        codebooks without an expanded weight image.  The graph contract is
        identical for compact E4M3 and Q8; model adapters never select it.
        """

        return bool(
            self.fixed_token_graph_candidate
            and getattr(self, "_device_cache_enabled", False)
            and (
                getattr(self, "_compact_q8_decode_enabled", False)
                or getattr(self, "native8_rows_supported", False)
            )
            and getattr(self, "_cache_native_scales_of_id", None) is not None
        )

    def bind_hidden_inputs(
        self,
        layer: int,
        value,
        weights: tuple[torch.Tensor, ...],
        indices: tuple[torch.Tensor, ...],
    ) -> None:
        """Bind the public TP1 Router buffers used by a fixed parent graph."""

        if (
            tuple(value.devices) != self.devices
            or value.ready_events is None
            or len(weights) != 1
            or len(indices) != 1
        ):
            raise ValueError("hybrid packed fixed input layout mismatch")
        self._bound_hidden_inputs[int(layer)] = (
            tuple(value.replicas),
            tuple(item.reshape(-1) for item in weights),
            tuple(item.reshape(-1) for item in indices),
        )

    def _drop_fixed_token_graphs(self) -> None:
        """Invalidate graphs whose arena/cache descriptor addresses expired."""

        had_graphs = bool(
            getattr(self, "_fixed_expert_graphs", None)
            or getattr(self, "_fixed_route_graphs", None)
            or getattr(self, "_fixed_graph_batches", None)
        )
        self._fixed_expert_graphs.clear()
        self._fixed_route_graphs.clear()
        self._fixed_graph_batches.clear()
        self._fixed_source_events.clear()
        self._fixed_done_events.clear()
        self._fixed_output_events.clear()
        self._fixed_graph_dependencies.clear()
        if had_graphs:
            self.fixed_graph_generation += 1

    def prepare_fixed_token_graphs(
        self,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float = 0.0,
    ) -> bool:
        """Capture one route-dependent compact expert graph per layer.

        Every replay performs route copy -> segmented LRU -> registered-host
        UVA gather -> metadata publish -> compact native-8 MoE.  All pointers
        stay fixed while route IDs and cache ownership are device data, so
        cold and hot routes share exactly the same graph executable.
        """

        if not self.fixed_token_graph_capable:
            return False
        selected_layers = tuple(sorted(self._bound_hidden_inputs))
        if (
            selected_layers
            and set(self._fixed_expert_graphs) == set(selected_layers)
        ):
            return True
        if self._workspaces is None or self._metadata is None:
            raise RuntimeError("hybrid fixed graph requires Decode workspaces")

        from .fusedext import make_tp_graph_launch_batch

        self._drop_fixed_token_graphs()
        device = self.device
        top_k = int(self.store.cfg["top_k"])
        stream = torch.cuda.Stream(device=device)
        self._fixed_graph_stream = stream
        graph_pool = torch.cuda.graph_pool_handle()
        started = time.perf_counter()
        for layer in selected_layers:
            replicas, route_weights, route_ids = self._bound_hidden_inputs[layer]
            value = replicas[0].reshape(1, -1)
            weights = route_weights[0][:top_k]
            indices = route_ids[0][:top_k]
            if value.dtype != torch.bfloat16 or weights.dtype != torch.float32:
                raise RuntimeError(
                    "hybrid fixed graph requires BF16 hidden and FP32 routes"
                )
            available = (
                self.store.available_mask(layer)
                .nonzero()
                .reshape(-1)[:top_k]
                .to(device=device, dtype=torch.long)
            )
            if int(available.numel()) != top_k:
                raise RuntimeError(
                    f"layer {layer} has fewer than Top-K graph experts"
                )
            value.zero_()
            indices.copy_(available)
            weights.fill_(1.0 / top_k)

            def launch_expert(
                layer_index: int = layer,
                hidden_value: torch.Tensor = value,
                route_weight_buffer: torch.Tensor = weights,
                route_id_buffer: torch.Tensor = indices,
            ) -> torch.Tensor:
                count = int(route_id_buffer.numel())
                self._enqueue_device_cache_update(
                    layer_index,
                    route_id_buffer,
                )
                pending = PendingPackedRun(
                    layer=layer_index,
                    value=hidden_value,
                    expert_count=count,
                    grouped_prefix=-1,
                    activation=activation,
                    activation_beta=float(activation_beta),
                    activation_linear_beta=activation_linear_beta,
                    limit=float(limit),
                    wait_for_stage=False,
                    route_order=self._route_ids[:count],
                    ordered_weights=route_weight_buffer,
                    metadata=self._metadata[:, :count],
                    codebook_scales=self._decode_codebook_scales[:count],
                    wait_for_device_cache=False,
                )
                return self._launch_compact_q8_decode(
                    pending,
                    pending.metadata,
                    route_weight_buffer,
                )

            current = torch.cuda.current_stream(device)
            stream.wait_stream(current)
            with torch.cuda.device(device), torch.cuda.stream(stream):
                result = launch_expert()
            current.wait_stream(stream)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.device(device), torch.cuda.graph(
                graph,
                stream=stream,
                pool=graph_pool,
            ):
                result = launch_expert()
            graph.instantiate()
            torch.cuda.synchronize(device)

            source_event = torch.cuda.Event()
            done_event = torch.cuda.Event()
            output_event = torch.cuda.Event()
            source_event.record(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                done_event.record(stream)
                output_event.record(stream)
            stream.synchronize()
            batch = make_tp_graph_launch_batch(
                [int(device.index)],
                [graph],
                [stream],
                [done_event],
                source_event,
            )
            if batch is None:
                raise RuntimeError("hybrid fixed expert graph batch unavailable")
            self._fixed_expert_graphs[layer] = graph
            self._fixed_graph_batches[layer] = batch
            self._fixed_source_events[layer] = source_event
            self._fixed_done_events[layer] = done_event
            self._fixed_output_events[layer] = output_event
            self._fixed_graph_dependencies.extend((
                graph,
                stream,
                source_event,
                done_event,
                output_event,
                result,
            ))
        self.fixed_graph_generation += 1
        print(
            "[cccp-cache] dynamic packed expert graphs ready: "
            f"layers={len(selected_layers)} controller=device-segmented-lru "
            "codebooks=VRAM-resident "
            "l2_prefetch=route-local-overlapped expanded_weights=0 "
            f"capture={(time.perf_counter() - started) * 1000:.1f}ms",
            flush=True,
        )
        return True

    def output_hidden(self, layer: int):
        """Expose the fixed compact-MoE contribution to the common finalizer."""

        from .ops import TPHidden

        layer = int(layer)
        if layer not in self._fixed_expert_graphs or self._workspaces is None:
            raise RuntimeError(
                f"hybrid packed MoE layer {layer} graph is unavailable"
            )
        return TPHidden(
            self.devices,
            (self._workspaces[2].view(1, -1),),
            (self._fixed_output_events[layer],),
        )

    def fixed_layer_plan(self, layer: int):
        layer = int(layer)
        batch = self._fixed_graph_batches.get(layer)
        if batch is None or self._workspaces is None:
            raise RuntimeError(
                f"hybrid packed MoE layer {layer} graph is unavailable"
            )
        return batch, (self._workspaces[2],), self.output_hidden(layer)

    def fixed_layer_child_graphs(self, layer: int):
        layer = int(layer)
        expert = self._fixed_expert_graphs.get(layer)
        if expert is None:
            raise RuntimeError(
                f"hybrid packed MoE layer {layer} graph is unavailable"
            )
        routes = self._fixed_route_graphs.get(layer)
        return ((routes[0], expert),) if routes is not None else ((expert,),)

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
        """Compose registered Top-K with dynamic-cache expert child graphs."""

        if not self._fixed_expert_graphs or self._fixed_graph_stream is None:
            raise RuntimeError("hybrid route composition requires expert graphs")
        from .fusedext import make_tp_graph_sequence_batch
        from .ops import route_topk

        selected_layers = (
            sorted(self._fixed_expert_graphs)
            if layers is None
            else sorted({int(layer) for layer in layers})
        )
        for layer in selected_layers:
            logits = logits_by_layer[layer]
            corrections = corrections_by_layer[layer]
            masks = masks_by_layer[layer]
            weight_buffers, index_buffers = route_buffers_by_layer[layer]
            if (
                tuple(logits.devices) != self.devices
                or logits.ready_events is None
                or len(corrections) != 1
                or len(masks) != 1
            ):
                raise ValueError("hybrid route fixed TP1 layout mismatch")
            stream = self._fixed_graph_stream

            def launch_route() -> None:
                route = route_topk(
                    logits.replicas[0],
                    corrections[0],
                    masks[0],
                    scoring_func=scoring_func,
                    top_k=int(top_k),
                    normalize=bool(normalize),
                    scaling=float(scaling),
                    n_group=int(n_group),
                    topk_group=int(topk_group),
                    output_buffers=(weight_buffers[0], index_buffers[0]),
                )
                if route is None:
                    raise RuntimeError("registered route Top-K rejected graph inputs")

            with torch.cuda.device(self.device), torch.cuda.stream(stream):
                launch_route()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.device(self.device), torch.cuda.graph(
                graph,
                stream=stream,
            ):
                launch_route()
            graph.instantiate()
            stream.synchronize()
            self._fixed_route_graphs[layer] = (graph,)
            batch = make_tp_graph_sequence_batch(
                [int(self.device.index)],
                [[graph, self._fixed_expert_graphs[layer]]],
                [stream],
                [self._fixed_done_events[layer]],
                self._fixed_source_events[layer],
            )
            if batch is None:
                raise RuntimeError("hybrid route/expert graph batch unavailable")
            self._fixed_graph_batches[layer] = batch
            self._fixed_graph_dependencies.extend((graph, batch))
        print(
            "[cccp-cache] Route TopK -> device LRU/UVA -> compact Q8 graphs "
            f"ready: layers={len(selected_layers)}",
            flush=True,
        )

    def _safe_budget(self) -> int:
        allocated = torch.cuda.memory_allocated(self.device)
        allocator_reserved = torch.cuda.memory_reserved(self.device)
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
        process_room = _process_memory_room(
            process_limit=process_limit,
            allocated=allocated,
            allocator_reserved=allocator_reserved,
            safety_reserve=reserve,
        )
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
            f"allocator_reserved={allocator_reserved / 2**30:.2f}GiB "
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
            return self.gpu_arena_bytes / 2**30
        if not self.pinned and self._extreme_specs is None:
            return 0.0
        host_codebooks = {
            codebook.data_ptr(): codebook
            for codebook in self._host_codebooks.values()
        }
        codebook_bytes = self._ensure_process_resident_codebooks(
            host_codebooks
        )
        # Resident BF16, FP8 and Q8 codebooks are allocated before capacity is
        # measured.  _safe_budget therefore excludes their complete physical
        # footprint exactly once, and the remaining budget belongs entirely
        # to compact expert slabs.  Keeping their addresses stable also lets
        # Prefill/Decode arena switches reuse captured metadata safely.
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
        by_layer: dict[int, Counter[PackedExpertSignature]] = {}
        if self.pinned:
            for (layer, _expert_id), expert in self.pinned.items():
                signature = PackedExpertSignature.of(expert)
                by_layer.setdefault(int(layer), Counter())[signature] += 1
        self._prefill_layer_specs = {
            int(layer): {
                signature: int(count)
                for signature, count in layer_counts.items()
            }
            for layer, layer_counts in by_layer.items()
        }
        widest_layer_specs, widest_layer_bytes = _widest_prefill_layer_specs(
            by_layer,
            resident_codebooks=self._resident_codebooks,
        )
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
            and widest_layer_specs
        ):
            self._decode_arena_target_budget = max(
                self._decode_arena_target_budget,
                safe_budget,
            )
            minimum_prefill_budget = max(
                2 * widest_layer_bytes,
                512 * 2**20,
            )
            if minimum_prefill_budget > safe_budget:
                raise RuntimeError(
                    "packed Prefill cannot fit two complete compact layers: "
                    f"{minimum_prefill_budget / 2**30:.2f} > "
                    f"{safe_budget / 2**30:.2f} GiB"
                )
            # Use every safe byte.  Two physical layer slabs provide the
            # ping-pong pipeline; the remainder becomes an immutable corpus-
            # hot arena and removes those experts from Prefill H2D entirely.
            prefill_budget = safe_budget
            self._prefill_arena_target_budget = prefill_budget
            print(
                "[cccp-vram-plan] phase=prefill-arena "
                f"compact_layer={widest_layer_bytes / 2**30:.2f}GiB "
                "buffers=2 "
                f"process_resident_codebooks={codebook_bytes / 2**30:.2f}GiB "
                f"selected={prefill_budget / 2**30:.2f}GiB "
                f"decode_target={safe_budget / 2**30:.2f}GiB",
                flush=True,
            )
        arena_budget = safe_budget
        top_k = int(self.store.cfg["top_k"])
        weights = self._profile_signature_heat_weights()
        if weights is None and self.slot_mix is not None:
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
        double_buffer_prefill = bool(
            self._arena_phase == "prefill"
            and self._extreme_specs is None
            and widest_layer_specs
        )
        if double_buffer_prefill:
            # Each slab is repartitioned to one *real* physical layer.  The
            # second slab receives L+1 while the default stream computes L.
            # Charging a cross-layer signature envelope here both wastes VRAM
            # and can still miss the actual widest layer.
            specs = dict(widest_layer_specs)
            hidden = int(
                self.store.cfg.get(
                    "routed_hidden",
                    self.store.cfg["hidden"],
                )
            )
            intermediate = int(self.store.cfg["moe_inter"])
            major, minor = torch.cuda.get_device_capability(self.device)
            native8_planned = bool(
                torch.version.hip is None
                and hasattr(torch, "_scaled_grouped_mm")
                and select_grouped_fp8_backend(
                    (int(major), int(minor))
                ) is not None
            )
            execution_bytes = 1 if native8_planned else 2
            prefill_compute_reserve = (
                int(self.store.cfg["n_experts"])
                * 3
                * hidden
                * intermediate
                * execution_bytes
                + int(3.5 * 2**30)
            )
            bytes_per_expert = (
                3 * hidden * intermediate * execution_bytes
            )
            # Retained expanded weights overlap the following layer's full
            # 4096-token Indexer/Attention.  Reserve 3.5 GiB for those live
            # tensors and fixed accumulation buffers; only the remainder may
            # become the persistent expert expansion slab.
            next_layer_aux_reserve = int(3.5 * 2**30)
            self._prefill_planned_chunk_capacity = max(
                1,
                min(
                    int(self.store.cfg["n_experts"]),
                    (
                        prefill_compute_reserve - next_layer_aux_reserve
                    ) // max(1, bytes_per_expert),
                ),
            )
            hot_budget = max(
                0,
                arena_budget
                - 2 * widest_layer_bytes
                - prefill_compute_reserve,
            )
            hot_specs = allocate_packed_slots(
                counts,
                hot_budget,
                1,
                weights=weights,
                resident_codebooks=self._resident_codebooks,
            )
        else:
            hot_specs = {}
            prefill_compute_reserve = 0
            self._prefill_planned_chunk_capacity = 0
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
        hot_required = sum(
            signature.storage_bytes(self._resident_codebooks) * count
            for signature, count in hot_specs.items()
        )
        required_total = (
            required * (2 if double_buffer_prefill else 1)
            + hot_required
        )
        if required_total > arena_budget:
            raise RuntimeError(
                "极限模式固定专家槽超过安全显存预算："
                f"{required_total / 2**30:.2f} > "
                f"{arena_budget / 2**30:.2f} GiB"
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
        self.initial_free_slots = (
            0
            if double_buffer_prefill
            else sum(warmup_free_minimum.values())
        )
        self.profile_hot_keys = (
            self._plan_prefill_hot_keys(
                hot_specs,
            )
            if double_buffer_prefill
            else self._plan_profile_hot_keys(
                specs,
                warmup_free_minimum,
            )
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
        self._prefill_spare_arenas = (
            _PackedArenas(
                specs,
                self.device,
                resident_codebooks=self._resident_codebooks,
            )
            if double_buffer_prefill
            else None
        )
        self._prefill_hot_arenas = (
            _PackedArenas(
                hot_specs,
                self.device,
                resident_codebooks=self._resident_codebooks,
            )
            if double_buffer_prefill and hot_specs
            else None
        )
        self._prefill_hot_selected.clear()
        self._prefill_prepared.clear()
        self._prefill_layer_local_maps = {}
        model_expert_count = int(self.store.cfg["n_experts"])
        for layer in self._prefill_layer_specs:
            expert_ids = sorted(
                int(expert_id)
                for expert_layer, expert_id in self.pinned
                if int(expert_layer) == int(layer)
            )
            host_map = torch.full(
                (model_expert_count,), -1, dtype=torch.long
            )
            if expert_ids:
                host_map[torch.tensor(expert_ids, dtype=torch.long)] = (
                    torch.arange(len(expert_ids), dtype=torch.long)
                )
            self._prefill_layer_local_maps[int(layer)] = host_map.to(
                device=self.device,
                non_blocking=False,
            )
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
        self._decode_codebook_scales = torch.empty(
            top_k,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        self._decode_codebook_scales_host = torch.empty(
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
        self._compact_q8_activation_workspaces = None
        if self._compact_q8_decode_enabled:
            hidden_span = (hidden + 15) & ~15
            intermediate_span = (intermediate + 15) & ~15
            self._compact_q8_activation_workspaces = (
                torch.empty(
                    top_k,
                    4 * hidden_span,
                    dtype=torch.uint8,
                    device=self.device,
                ),
                torch.empty(
                    top_k,
                    2 * intermediate_span,
                    dtype=torch.uint8,
                    device=self.device,
                ),
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
            f"prefill_layer_max={widest_layer_bytes / 2**30:.2f}GiB "
            f"prefill_compute_reserve={prefill_compute_reserve / 2**30:.2f}GiB "
            f"prefill_expert_chunk_cap={self._prefill_planned_chunk_capacity} "
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
        if self.host_dma_mode == "direct-registered" and os.name != "nt":
            self._build_device_cache_runtime()
        return self.gpu_arena_bytes / 2**30

    def _plan_prefill_hot_keys(
        self,
        specs: Mapping[PackedExpertSignature, int],
    ) -> tuple[tuple[int, int], ...]:
        """Choose immutable hot residents, even without a corpus heat map."""

        ranked = self._plan_profile_hot_keys(
            specs,
            {signature: 0 for signature in specs},
        )
        if ranked or not specs:
            return ranked
        remaining = {
            signature: int(count) for signature, count in specs.items()
        }
        # Expert-ID-major order distributes an unranked residency budget over
        # every physical layer instead of exhausting it on the first layers.
        candidates = sorted(
            self.pinned,
            key=lambda key: (int(key[1]), int(key[0])),
        )
        output: list[tuple[int, int]] = []
        for key in candidates:
            signature = PackedExpertSignature.of(self.pinned[key])
            if remaining.get(signature, 0) <= 0:
                continue
            output.append(key)
            remaining[signature] -= 1
            if not any(value > 0 for value in remaining.values()):
                break
        return tuple(output)

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
        totals = Counter(
            PackedExpertSignature.of(expert) for expert in self.pinned.values()
        )
        all_pinned_fit = bool(self.pinned) and all(
            int(specs.get(signature, 0)) >= int(count)
            for signature, count in totals.items()
        )
        if all_pinned_fit:
            # A full-resident Decode arena needs no empty Top-K staging set:
            # every possible route already has a slot.  Seed its complete
            # directory even when the model has no corpus heat profile.  The
            # former heat-only branch rebuilt the GLM Flash arena after
            # Prefill, left every slot unregistered, and re-uploaded the whole
            # 70-GiB expert set repeatedly during Decode.
            return tuple(sorted(
                self.pinned,
                key=lambda key: (int(key[1]), int(key[0])),
            ))
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
        ranks = self.store.heat_ranks or {}
        if not ranks:
            # A model can be launched before the user has trained a heat
            # profile.  Starting a nearly full Decode arena completely empty
            # wastes the available VRAM and turns the first request into a
            # model-sized transfer storm.  Expert-ID-major ordering spreads
            # this neutral warm start over every layer; trained profiles still
            # use their exact measured global hit counts below.
            output: list[tuple[int, int]] = []
            for key in sorted(
                self.pinned,
                key=lambda item: (int(item[1]), int(item[0])),
            ):
                signature = PackedExpertSignature.of(self.pinned[key])
                if remaining.get(signature, 0) <= 0:
                    continue
                output.append(key)
                remaining[signature] -= 1
                if not any(value > 0 for value in remaining.values()):
                    break
            return tuple(output)
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

    def _profile_signature_heat_weights(
        self,
    ) -> dict[PackedExpertSignature, float] | None:
        """Aggregate measured route traffic for cache-capacity allocation."""

        heat_counts = getattr(self.store, "heat_counts", None) or {}
        if not heat_counts:
            return None
        weights: Counter[PackedExpertSignature] = Counter()
        for (layer, expert_id), expert in self.pinned.items():
            score = float(
                heat_counts.get(int(layer), {}).get(int(expert_id), 0.0)
            )
            if score > 0.0:
                weights[PackedExpertSignature.of(expert)] += score
        return dict(weights) if any(weights.values()) else None

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
        if self._compact_profile_all_resident:
            self._publish_full_resident_directory_locked()
        print(
            "[cccp-cache] lru-hot-start="
            f"{len(selected_keys)} experts / "
            f"{warmed_bytes / 2**30:.2f}GiB "
            f"elapsed={time.perf_counter() - started:.2f}s；"
            "permanent_protection=0；所有槽按命中顺序末位淘汰；"
            f"RAM专家保持完整={self.host_expert_bytes / 2**30:.2f}GiB",
            flush=True,
        )

    def _publish_full_resident_directory_locked(self) -> None:
        """Publish the complete resident expert directory with one H2D copy.

        A full-model arena has no cache misses to plan.  Publishing one row at
        a time both obscured that invariant and allowed the graph-resident LRU
        descriptors to become the effective source of truth after a Prefill
        arena rebuild.  Build the exact layer/expert directory from the live
        leases and publish it atomically instead.
        """

        if self._slot_directory is None:
            raise RuntimeError("full resident route directory is unavailable")
        host_directory = torch.zeros(
            tuple(self._slot_directory.shape),
            dtype=self._slot_directory.dtype,
        )
        host_scales = (
            torch.zeros(
                tuple(self._slot_scale_directory.shape),
                dtype=self._slot_scale_directory.dtype,
            )
            if self._slot_scale_directory is not None
            else None
        )
        for (layer, expert_id), expert in self.cache.items():
            if self._compact_q8_decode_enabled:
                rows, scales = self._decode_codebook_metadata_rows([expert])
                if host_scales is not None:
                    host_scales[int(layer), int(expert_id)].copy_(
                        torch.tensor(scales[0], dtype=host_scales.dtype)
                    )
            else:
                rows = self._metadata_rows([expert])
            values = [row[0] for row in rows[: self._slot_directory.shape[2]]]
            host_directory[int(layer), int(expert_id), : len(values)].copy_(
                torch.tensor(values, dtype=host_directory.dtype)
            )
        self._slot_directory.copy_(host_directory, non_blocking=False)
        if self._slot_scale_directory is not None and host_scales is not None:
            self._slot_scale_directory.copy_(host_scales, non_blocking=False)

    def _warm_prefill_hot_locked(self) -> None:
        """Populate the immutable Prefill residency arena once at startup."""

        if (
            self._prefill_hot_arenas is None
            or self._prefill_hot_selected
            or not self.profile_hot_keys
        ):
            return
        started = time.perf_counter()
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        selected: dict[tuple[int, int], DeviceExpert] = {}
        for key in self.profile_hot_keys:
            host = self.pinned.get(key)
            if host is None:
                continue
            _replaced, device_expert = self._prefill_hot_arenas.lease(
                key,
                host,
                self._device_codebooks,
            )
            selected[key] = device_expert
            pairs.extend(self._copy_pairs(host, device_expert))
        if pairs:
            self._stage.upload_batch(pairs)
            self._stage.last.synchronize()
        self._prefill_hot_selected = selected
        uploaded = sum(int(source.nbytes) for source, _target in pairs)
        self.uploaded_bytes += uploaded
        self._profile_hot_ready = True
        print(
            "[cccp-prefill] immutable-hot-residency="
            f"{len(selected)} experts / {uploaded / 2**30:.2f}GiB "
            f"elapsed={time.perf_counter() - started:.2f}s；"
            "remaining experts use double-buffer layer staging",
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
        publish_directory = not getattr(
            self, "_retain_prefill_workspace", False
        )
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
                    if publish_directory:
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
                    if publish_directory:
                        self._set_slot_directory(key, value)
                    if not prefetch:
                        self.miss += 1
                if staged and not publish_directory:
                    # Prefill consumes the returned fixed DeviceExpert views
                    # directly. Publishing each of thousands of tiny
                    # descriptors performed a blocking H2D copy per expert
                    # and serialized the complete layer. Decode rebuilds one
                    # coherent directory after the block instead.
                    self._prefill_directory_dirty = True
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

    def _ensure_process_resident_codebooks(
        self,
        host_codebooks: Mapping[int, torch.Tensor],
    ) -> int:
        """Materialize shared execution codebooks once per loaded process."""

        if not self._resident_codebooks:
            return 0
        expected = set(int(pointer) for pointer in host_codebooks)
        if self._device_codebooks:
            if set(self._device_codebooks) != expected:
                raise RuntimeError(
                    "resident codebook identity changed during arena resize"
                )
        else:
            self._device_codebooks = {
                int(pointer): codebook.to(
                    device=self.device,
                    dtype=torch.bfloat16,
                    non_blocking=False,
                )
                for pointer, codebook in host_codebooks.items()
            }
            self._prepare_native8_codebooks()
            self._prepare_compact_q8_codebooks()
        return sum(
            tensor.nbytes
            for collection in (
                self._device_codebooks,
                self._native8_codebooks,
                self._compact_q8_codebooks,
            )
            for tensor in collection.values()
        )

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
        self._native8_grouped_backend = select_grouped_fp8_backend(
            (int(major), int(minor))
        )
        if self._native8_grouped_backend is None:
            return
        image = compile_shared_codebook_image(
            list(self._device_codebooks.values()),
            mode="e4m3",
        )
        self._native8_codebooks = image.tensors
        self._native8_codebook_scales = image.scales
        self._native8_prefill_enabled = bool(self._native8_codebooks)
        self.native8_rows_supported = self._native8_prefill_enabled
        if self._native8_prefill_enabled:
            self.prefill_executor = "cuda.vq-to-e4m3-scaled-grouped-gemm"
            print(
                "[cccp-native8] shared codebooks=E4M3; "
                f"count={len(self._native8_codebooks)}; "
                "runtime reconstruction=index-unpack+aligned-copy; "
                "compute=Tensor-Core FP8 grouped-GEMM; "
                f"backend={self._native8_grouped_backend}",
                flush=True,
            )

    def _prepare_compact_q8_codebooks(self) -> None:
        """Compile only the shared VQ codebooks to Q8 for direct DP4A.

        Expert payloads remain in their original p8--p16 representation.  In
        particular this method must never allocate a decoded expert row or a
        full INT8/FP8 weight image.
        """

        self._compact_q8_codebooks = {}
        self._compact_q8_codebook_scales = {}
        self._compact_q8_decode_enabled = False
        if (
            self.device.type != "cuda"
            or torch.version.hip is not None
            or not self._resident_codebooks
            or os.environ.get("CCCP_PROJECTION_TILE_VIEW", "0") == "1"
        ):
            return
        major, _minor = torch.cuda.get_device_capability(self.device)
        if int(major) < 7:
            return
        image = compile_shared_codebook_image(
            list(self._device_codebooks.values()),
            mode="q8",
        )
        self._compact_q8_codebooks = image.tensors
        self._compact_q8_codebook_scales = image.scales
        self._compact_q8_decode_enabled = bool(self._compact_q8_codebooks)
        if self._compact_q8_decode_enabled:
            compact_bytes = sum(
                tensor.nbytes for tensor in self._compact_q8_codebooks.values()
            )
            print(
                "[cccp-codebook] Decode=packed-index+Q8-codebook+DP4A；"
                f"shared_codebooks={len(self._compact_q8_codebooks)}；"
                f"codebook_bytes={compact_bytes / 2**20:.2f}MiB；"
                "expanded_expert_weight_bytes=0",
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

    def _decode_codebook_metadata_rows(
        self,
        experts: list[DeviceExpert],
    ) -> tuple[list[list[int]], list[tuple[float, ...]]]:
        """Publish compact Decode codebooks without changing expert bytes."""

        if not self._compact_q8_decode_enabled:
            return self._native8_metadata_rows(experts)
        rows = self._metadata_rows(experts)[:15]
        scales: list[tuple[float, ...]] = []
        for expert in experts:
            expert_scales: list[float] = []
            for projection, weight in enumerate(expert):
                pointer = int(weight.cb.data_ptr())
                quantized = self._compact_q8_codebooks.get(pointer)
                scale = self._compact_q8_codebook_scales.get(pointer)
                if quantized is None or scale is None:
                    raise RuntimeError(
                        "compact Q8 shared codebook cache is incomplete"
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

    def _copy_decode_codebook_metadata(
        self,
        experts: list[DeviceExpert],
    ) -> torch.Tensor:
        """Publish Top-K packed pointers with prequantized codebook scales."""

        count = len(experts)
        rows, scales = self._decode_codebook_metadata_rows(experts)
        rows = rows[: self._metadata.shape[0]]
        host = self._metadata_host
        if host is None or host.shape[0] != len(rows) or host.shape[1] < count:
            host = torch.empty(len(rows), count, dtype=torch.long)
            self._metadata_host = host
        host[:, :count].copy_(torch.tensor(rows, dtype=torch.long))
        self._metadata[:, :count].copy_(host[:, :count], non_blocking=False)

        scale_host = getattr(self, "_decode_codebook_scales_host", None)
        scale_device = getattr(self, "_decode_codebook_scales", None)
        if scale_host is None or int(scale_host.shape[0]) < count:
            scale_host = torch.empty(count, 3, dtype=torch.float32)
            self._decode_codebook_scales_host = scale_host
        if scale_device is None or int(scale_device.shape[0]) < count:
            scale_device = torch.empty(
                count,
                3,
                dtype=torch.float32,
                device=self.device,
            )
            self._decode_codebook_scales = scale_device
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
        if self._compact_q8_decode_enabled:
            rows, native_scales = self._decode_codebook_metadata_rows([expert])
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

    def _prepare_resident_compact_run_locked(
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
            or self._decode_codebook_scales is None
        ):
            raise RuntimeError("compact Q8 resident route directory is unavailable")
        flat_ids = route_ids.reshape(-1)
        count = int(flat_ids.numel())
        from .ops import packed_route_slots

        if not packed_route_slots(
            flat_ids,
            self._slot_directory[int(layer)],
            output=self._metadata[:, :count],
            hit_mask=self._route_hit_mask[:count],
        ):
            raise RuntimeError("compact Q8 resident route lookup was rejected")
        torch.index_select(
            self._slot_scale_directory[int(layer)],
            0,
            flat_ids,
            out=self._decode_codebook_scales[:count],
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
            codebook_scales=self._decode_codebook_scales[:count],
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

    def _enqueue_device_cache_update(
        self,
        layer: int,
        route_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Publish one route using only graph-capturable device operations."""

        from . import fusedext

        required = (
            self._cache_signature_of_id,
            self._cache_segment_offsets,
            self._cache_slot_for_id,
            self._cache_id_of_slot,
            self._cache_last_used,
            self._cache_step,
            self._cache_route_slots,
            self._cache_source_ids,
            self._cache_destination_slots,
            self._cache_counts,
            self._cache_source_ptrs,
            self._cache_destination_ptrs,
            self._cache_signature_bytes,
            self._cache_projection_offsets,
            self._cache_metadata_of_id,
        )
        if any(tensor is None for tensor in required):
            raise RuntimeError(
                "graph-resident packed cache descriptor is incomplete"
            )
        count = int(route_ids.numel())
        n_experts = int(self.store.cfg["n_experts"])
        if (
            self._cache_input_route_ids is None
            or self._cache_input_route_ids.numel() < count
        ):
            raise RuntimeError("device packed-cache route buffer is undersized")
        flat_ids = self._cache_input_route_ids[:count]
        flat_ids.copy_(route_ids.reshape(-1), non_blocking=True)
        if not fusedext.packed_cache_plan_fused(
            flat_ids,
            int(layer),
            n_experts,
            self._cache_signature_of_id,
            self._cache_segment_offsets,
            self._cache_slot_for_id,
            self._cache_id_of_slot,
            self._cache_last_used,
            self._cache_step,
            self._cache_route_slots,
            self._cache_source_ids,
            self._cache_destination_slots,
            self._cache_counts,
        ):
            raise RuntimeError("device packed-cache planner rejected its buffers")
        if self._cache_profile_totals is not None:
            self._cache_profile_totals.add_(
                self._cache_counts.to(dtype=torch.int64)
            )
        if not fusedext.packed_cache_uva_copy_fused(
            self._cache_source_ptrs,
            self._cache_destination_ptrs,
            self._cache_signature_of_id,
            self._cache_signature_bytes,
            self._cache_source_ids,
            self._cache_destination_slots,
            self._cache_counts,
            int(layer),
            n_experts,
            64,
        ):
            raise RuntimeError("device packed-cache UVA gather was rejected")
        if not fusedext.packed_cache_metadata_fused(
            flat_ids,
            self._cache_route_slots,
            self._cache_signature_of_id,
            self._cache_destination_ptrs,
            self._cache_projection_offsets,
            self._cache_metadata_of_id,
            self._metadata[:, :count],
            int(layer),
            n_experts,
        ):
            raise RuntimeError(
                "device packed-cache metadata publish was rejected"
            )
        if (
            self._compact_q8_decode_enabled
            and not fusedext.compact_q8_codebook_l2_prefetch_fused(
                self._metadata[:, :count]
            )
        ):
            raise RuntimeError("compact Q8 codebook L2 prefetch was rejected")
        if self._compact_q8_decode_enabled:
            if (
                self._cache_input_logical_ids is None
                or self._cache_native_scales_of_id is None
                or self._decode_codebook_scales is None
            ):
                raise RuntimeError(
                    "device native8 route-scale directory is incomplete"
                )
            torch.add(
                flat_ids,
                int(layer) * n_experts,
                out=self._cache_input_logical_ids[:count],
            )
            torch.index_select(
                self._cache_native_scales_of_id,
                0,
                self._cache_input_logical_ids[:count],
                out=self._decode_codebook_scales[:count],
            )
        return flat_ids

    def _prepare_device_cache_run_locked(
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
        """Plan, gather and publish one routed layer without host decisions."""

        count = int(route_ids.numel())
        if self._device_cache_stream is None or self._device_cache_ready is None:
            raise RuntimeError("device packed-cache stream is unavailable")
        default_stream = torch.cuda.current_stream(self.device)
        # Keep the asynchronous planner input at a stable address.  A
        # reshape/contiguous temporary can otherwise return to PyTorch's
        # allocator as soon as this function exits while the cache stream is
        # still consuming it.
        # The current route was produced on the default stream, and the prior
        # layer may still be reading slots there.  This one dependency protects
        # both boundaries while leaving the current shared branch free to run
        # concurrently after prepare_run returns.
        self._device_cache_stream.wait_stream(default_stream)
        with torch.cuda.stream(self._device_cache_stream):
            self._enqueue_device_cache_update(layer, route_ids)
            self._device_cache_ready.record(self._device_cache_stream)
        pending = PendingPackedRun(
            layer=int(layer),
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
            codebook_scales=(
                self._decode_codebook_scales[:count]
                if self._cache_native_scales_of_id is not None
                else None
            ),
            wait_for_device_cache=True,
        )
        # H20 measurements show that running routed MoE on a second GPU stream
        # contends for the same SMs as the shared expert and is slower than the
        # ordered default-stream DAG.  Keep the experimental overlap explicit
        # until a heterogeneous CPU worker can own the shared branch without a
        # Python rendezvous.
        if os.environ.get("CCCP_DEVICE_ROUTED_PRELAUNCH", "0") != "0":
            return self._prelaunch_device_routed_locked(pending)
        return pending

    def _prelaunch_device_routed_locked(
        self,
        pending: PendingPackedRun,
    ) -> PendingPackedRun:
        """Queue routed MoE before the caller submits the shared branch.

        The cache stream performs route planning and UVA gathers.  This
        second stream waits for that event, runs the packed routed graph, and
        publishes one completion event.  The model's default stream is then
        free to execute the shared expert concurrently and joins only in
        :meth:`finish_run`.
        """

        if (
            self._device_cache_ready is None
            or self._device_routed_stream is None
            or self._device_routed_ready is None
        ):
            raise RuntimeError("device packed routed stream is unavailable")
        self._device_routed_stream.wait_event(self._device_cache_ready)
        with torch.cuda.stream(self._device_routed_stream):
            pending.prelaunched_result = self._launch_packed_run(pending)
            self._device_routed_ready.record(self._device_routed_stream)
        pending.wait_for_routed = True
        return pending

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
            if (
                self._compact_q8_decode_enabled
                and self._compact_profile_all_resident
            ):
                return self._prepare_resident_compact_run_locked(
                    layer,
                    value,
                    route_ids,
                    route_weights,
                    activation=activation,
                    activation_beta=activation_beta,
                    activation_linear_beta=activation_linear_beta,
                    limit=limit,
                )
            if self._device_cache_enabled:
                return self._prepare_device_cache_run_locked(
                    layer,
                    value,
                    route_ids,
                    route_weights,
                    activation=activation,
                    activation_beta=activation_beta,
                    activation_linear_beta=activation_linear_beta,
                    limit=limit,
                )
            if self._compact_q8_decode_enabled:
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
            if self._compact_q8_decode_enabled
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
        codebook_scales = None
        if self._compact_q8_decode_enabled:
            codebook_scales = self._copy_decode_codebook_metadata(experts)
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
            codebook_scales=codebook_scales,
        )

    def _launch_compact_q8_decode(
        self,
        resolved: PendingPackedRun,
        metadata: torch.Tensor,
        ordered_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Execute compact VQ codebook math with no weight expansion."""

        scales = resolved.codebook_scales
        if scales is None:
            raise RuntimeError("compact Q8 Decode is missing codebook scales")
        count = int(resolved.expert_count)
        route_order = (
            self._route_ids[:count]
            if resolved.route_order is None
            else resolved.route_order
        )
        hidden, output, result = self._workspaces
        from .ops.codebook import run_compact_q8_codebook_decode

        operator_kwargs = {}
        if self._compact_q8_decode_enabled:
            if self._compact_q8_activation_workspaces is None:
                raise RuntimeError("compact Q8 activation workspace is missing")
            gate_quant, down_quant = self._compact_q8_activation_workspaces
            operator_kwargs = {
                "gate_quant_workspace": gate_quant[:count],
                "down_quant_workspace": down_quant[:count],
            }
        if not self._compact_q8_decode_enabled:
            raise RuntimeError("public compact Decode requires Q8/DP4A")
        return run_compact_q8_codebook_decode(
            value=resolved.value,
            route_ids=route_order,
            route_weights=ordered_weights[:count],
            metadata=metadata,
            scales=scales,
            activation=resolved.activation,
            activation_beta=float(resolved.activation_beta),
            activation_linear_beta=resolved.activation_linear_beta,
            limit=float(resolved.limit),
            hidden_workspace=hidden[:count],
            output_workspace=output[:count],
            result=result,
            **operator_kwargs,
        )

    def _launch_packed_run(
        self,
        resolved: PendingPackedRun,
    ) -> torch.Tensor:
        """Submit one fixed-workspace packed MoE execution."""

        if resolved.wait_for_stage:
            self._stage.wait()
        if resolved.wait_for_device_cache:
            if self._device_cache_ready is None:
                raise RuntimeError("device packed-cache completion event is missing")
            torch.cuda.current_stream(self.device).wait_event(
                self._device_cache_ready
            )
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
        if resolved.codebook_scales is not None:
            self.decode_fused_submissions += 1
            return self._launch_compact_q8_decode(
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
            if pending.wait_for_routed:
                if (
                    self._device_routed_ready is None
                    or pending.prelaunched_result is None
                ):
                    raise RuntimeError(
                        "prelaunched packed routed result is incomplete"
                    )
                torch.cuda.current_stream(self.device).wait_event(
                    self._device_routed_ready
                )
                return pending.prelaunched_result
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
            if pending.wait_for_routed and self._device_routed_ready is not None:
                # Cancellation is already an exceptional path.  Complete the
                # in-flight routed graph before releasing its exclusive arena
                # lease so a following request cannot overwrite live slots.
                self._device_routed_ready.synchronize()
            pending.active = False
            self._transfer_lock.release()

    def _find_route_plan(
        self,
        layer: int,
        expert_ids: list[int],
    ) -> PackedRoutePlan | None:
        """Reuse device pointer metadata while its arena leases stay valid."""
        if os.environ.get("CCCP_VQ_ROUTE_PLAN", "1") == "0":
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
        if os.environ.get("CCCP_VQ_ROUTE_PLAN", "1") == "0":
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

    def prepare_prefill_layer(self, layer: int) -> bool:
        """Stage one complete compact expert layer into the ping-pong slab.

        This is deliberately invoked by the model *before* it queues the
        layer's Attention work.  The copy stream waits for the previous layer,
        then uploads L while the default stream executes Attention(L).  Route
        selection remains exact: ``run_rows`` later consumes only the experts
        selected by the router, although every configured compact expert for
        this physical layer is already available.
        """

        layer = int(layer)
        if (
            self.device.type != "cuda"
            or torch.version.hip is not None
            or self._arena_phase != "prefill"
            or self._extreme_specs is not None
            or self._arenas is None
            or self._prefill_spare_arenas is None
        ):
            return False
        specs = self._prefill_layer_specs.get(layer)
        if not specs:
            return False
        arena = (
            self._arenas
            if layer % 2 == 0
            else self._prefill_spare_arenas
        )
        with self._transfer_lock:
            # The copy stream's wait_stream below protects the parity slab's
            # preceding reader.  Dropping stale Python views is sufficient;
            # the allocator itself is fixed for the whole Prefill block.
            for prepared_layer, prepared in tuple(
                self._prefill_prepared.items()
            ):
                if prepared[0] is arena:
                    self._prefill_prepared.pop(prepared_layer, None)
            arena.repartition(specs)
            keys = sorted(
                key for key in self.pinned if int(key[0]) == layer
            )
            if not keys:
                raise RuntimeError(
                    f"packed Prefill layer {layer} has no configured experts"
                )
            selected: dict[tuple[int, int], DeviceExpert] = {}
            pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
            for key in keys:
                resident = self._prefill_hot_selected.get(key)
                if resident is not None:
                    selected[key] = resident
                    continue
                host = self.pinned[key]
                _replaced, device_expert = arena.lease(
                    key,
                    host,
                    self._device_codebooks,
                )
                selected[key] = device_expert
                pairs.extend(self._copy_pairs(host, device_expert))
            self._stage.upload_batch(pairs)
            ready = torch.cuda.Event()
            ready.record(self._stage.stream)
            self._prefill_prepared[layer] = (arena, selected, ready)
            self.uploaded_bytes += sum(
                int(source.nbytes) for source, _target in pairs
            )
            self.miss += len(keys)
            self._prefill_directory_dirty = True
            if not getattr(self, "_prefill_layer_stage_announced", False):
                print(
                    "[cccp-prefill] expert-staging="
                    "double-buffer-layer-ahead; compact=all-configured; "
                    "routing=exact; attention-overlap=enabled; "
                    f"immutable-hot={len(self._prefill_hot_selected)}",
                    flush=True,
                )
                self._prefill_layer_stage_announced = True
        return True

    def _restore_decode_arena_locked(self) -> None:
        if self._arenas is None or self._prefill_partition_layer is None:
            return
        torch.cuda.synchronize(self.device)
        self._stage.wait()
        self._reset_arena_directory_locked()
        self._arenas.repartition(self._default_arena_specs)
        self._prefill_partition_layer = None
        self._build_device_cache_runtime()

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
        compact = bool(
            torch.version.hip is not None
            or rows <= int(self.short_reset_decode_tokens)
        )
        if _prefill_workspace_matches(
            cached,
            rows=rows,
            micro_batch=micro_batch,
            top_k=top_k,
            intermediate=intermediate,
            hidden=hidden,
            compact=compact,
        ):
            return cached
        cached: dict[str, torch.Tensor | int] = {
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
            # Reused FP32 accumulation rows avoid the expression
            # ``down.float() * weights`` allocating two 384-MiB temporaries at
            # a 4096xTopK batch boundary under the hard VRAM fraction.
            "weighted_down": torch.empty(
                micro_batch * top_k,
                hidden,
                dtype=torch.float32,
                device=self.device,
            ),
            "inverse_route_order": torch.empty(
                micro_batch * top_k,
                dtype=torch.long,
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
                pin_memory=self.device.type == "cuda",
            ),
        }
        if compact:
            # HIP cannot use the private grouped-mm implementation. Short
            # NVIDIA batches are also bandwidth-bound: expanding every routed
            # expert to FP8 writes and rereads far more bytes than the compact
            # grouped operator. Both cases therefore own the same bounded
            # activation/result workspaces and read VQ directly.
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

    def _prefill_dequant_chunk_capacity(
        self,
        expert_count: int,
        *,
        routed_rows: int = 0,
    ) -> int:
        """Choose a dense expert scratch that stays inside the VRAM cap."""

        cached = getattr(self, "_prefill_dequant_workspace", None)
        if cached is not None:
            return max(1, min(int(expert_count), int(cached[0].shape[0])))
        native8_cached = getattr(self, "_prefill_native8_workspace", None)
        if (
            getattr(self, "_native8_prefill_enabled", False)
            and native8_cached is not None
            and int(native8_cached["routed_rows"]) >= int(routed_rows)
        ):
            # ``mem_get_info`` excludes this live workspace from free VRAM.
            # Replanning from that reduced number would progressively shrink
            # every following layer even though no new allocation is needed.
            return max(
                1,
                min(int(expert_count), int(native8_cached["capacity"])),
            )
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
        if os.name == "nt" and reserved - allocated >= 256 * 2**20:
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
        kernel_chunk_limit = _projection_kernel_chunk_limit(
            hidden=hidden,
            intermediate=intermediate,
            execution_bytes=execution_bytes,
            native8=bool(getattr(self, "_native8_prefill_enabled", False)),
        )
        if kernel_chunk_limit is not None:
            automatic = min(automatic, kernel_chunk_limit)
        try:
            requested = int(os.environ.get(
                "CCCP_PREFILL_DEQUANT_EXPERTS", "0"
            ))
        except (TypeError, ValueError):
            requested = 0
        if requested > 0:
            automatic = min(automatic, requested)
        planned_capacity = int(
            getattr(self, "_prefill_planned_chunk_capacity", 0)
        )
        if planned_capacity > 0:
            automatic = min(automatic, planned_capacity)
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
    ) -> dict[str, object]:
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
        if cached is not None:
            # A longer following prompt needs larger row workspaces.  Drop
            # the old CUDA tensors before allocating their replacements;
            # otherwise the caching allocator must temporarily hold both
            # complete FP8/DeepGEMM workspaces and can OOM even though either
            # workspace independently fits the planned Prefill reserve.
            self._prefill_native8_workspace = None
            del cached
            torch.cuda.synchronize(self.device)
            gc.collect()
            torch.cuda.empty_cache()
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
            "projection_scales_host": torch.empty(
                int(capacity),
                3,
                dtype=torch.float32,
                pin_memory=True,
            ),
            "projection_scales": torch.empty(
                int(capacity),
                3,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        if self._native8_grouped_backend == "deepgemm-sm90":
            padded_rows = deepgemm_grouped_padded_rows(
                routed_rows,
                capacity,
                alignment=deepgemm_grouped_alignment(),
            )
            cached.update({
                "input_block_scales": torch.empty(
                    int(routed_rows),
                    hidden // 128,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "activated_block_scales": torch.empty(
                    int(routed_rows),
                    intermediate // 128,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "gu_block_scales": torch.empty(
                    int(capacity),
                    (2 * intermediate) // 128,
                    hidden // 128,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "down_block_scales": torch.empty(
                    int(capacity),
                    hidden // 128,
                    intermediate // 128,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "gate_up_output": torch.empty(
                    int(routed_rows),
                    2 * intermediate,
                    dtype=torch.bfloat16,
                    device=self.device,
                ),
                "down_output": torch.empty(
                    int(routed_rows),
                    hidden,
                    dtype=torch.bfloat16,
                    device=self.device,
                ),
                "gate_up_deepgemm": DeepGEMMGroupedWorkspace(
                    value=torch.empty(
                        int(padded_rows),
                        hidden,
                        dtype=torch.float8_e4m3fn,
                        device=self.device,
                    ),
                    scale_a=torch.empty(
                        int(padded_rows),
                        hidden // 128,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    output=torch.empty(
                        int(padded_rows),
                        2 * intermediate,
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                ),
                "down_deepgemm": DeepGEMMGroupedWorkspace(
                    value=torch.empty(
                        int(padded_rows),
                        intermediate,
                        dtype=torch.float8_e4m3fn,
                        device=self.device,
                    ),
                    scale_a=torch.empty(
                        int(padded_rows),
                        intermediate // 128,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    output=torch.empty(
                        int(padded_rows),
                        hidden,
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                ),
            })
        else:
            cached.update({
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
            })
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
        # One Prefill block calls this entry once per model layer.  When the
        # common capacity planner reserved both this scratch and the next
        # layer's Attention/KV allowance, retain one stable allocation for the
        # whole block.  This policy is shared by every routed-VQ architecture
        # and depends on measured capacity, never a model name or host OS.
        self._retain_prefill_workspace = should_retain_prefill_workspace(
            device_type=self.device.type,
            planned_chunk_capacity=int(
                getattr(self, "_prefill_planned_chunk_capacity", 0)
            ),
        )
        if not value.is_cuda:
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
        compact_grouped_prefill = bool(
            torch.version.hip is not None
            or rows <= int(self.short_reset_decode_tokens)
        )
        active_prefill_executor = (
            "cuda.packed-vq-grouped-direct"
            if compact_grouped_prefill
            else self.prefill_executor
        )
        if not self._prefill_executor_announced:
            print(
                "[cccp-prefill] "
                "short_executor=cuda.packed-vq-grouped-direct; "
                f"long_executor={self.prefill_executor}; "
                f"short_threshold={self.short_reset_decode_tokens}; "
                "decode GEMV fallback=forbidden",
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
        from .fusedext import (
            dense_fp8_quantize_rows_fused,
            gated_activation_fp8_quantize_rows_fused,
            routed_weighted_reduce_fused,
        )

        with self._transfer_lock:
            prepared = self._prefill_prepared.pop(int(layer), None)
            prepared_selected: dict[
                tuple[int, int], DeviceExpert
            ] | None = None
            prepared_ready: torch.cuda.Event | None = None
            if prepared is None:
                self._partition_prefill_layer_locked(layer)
            else:
                _prepared_arena, prepared_selected, prepared_ready = prepared
            start = 0
            while start < rows:
                count = min(target_batch, rows - start)
                flat_ids = route_ids[start : start + count].reshape(-1)
                stop = start + count
                # B-residency stages the complete configured layer before
                # Attention.  Consume that static expert table directly:
                # pulling ``torch.unique`` back to the CPU inserted one hard
                # synchronization per layer and duplicated the model's gated
                # route counter.  Empty grouped-GEMM segments are valid, so
                # experts with zero hits cost no routed rows and need no
                # special host-side pruning.
                if prepared_selected is not None:
                    unique_ids = [
                        int(expert_id)
                        for expert_layer, expert_id in sorted(prepared_selected)
                        if int(expert_layer) == int(layer)
                    ]
                    layer_map = self._prefill_layer_local_maps.get(int(layer))
                    if layer_map is None:
                        raise RuntimeError(
                            f"packed Prefill layer {layer} has no route map"
                        )
                    local_route_ids = layer_map.index_select(
                        0, flat_ids
                    ).view_as(route_ids[start:stop]).contiguous()
                    torch._assert_async(
                        (local_route_ids >= 0).all(),
                        "router selected an expert outside the configured layer",
                    )
                else:
                    unique_global = torch.unique(flat_ids, sorted=True)
                    unique_ids = [
                        int(item)
                        for item in unique_global.detach().cpu().tolist()
                    ]
                    local_route_ids = torch.searchsorted(
                        unique_global,
                        route_ids[start:stop],
                    ).contiguous()
                self._last_ids[int(layer)] = list(unique_ids)
                unique_count = len(unique_ids)

                # Routes remain indices into the complete configured list.
                # Each expert chunk expands its compact weights once for the
                # complete token batch; zero-hit groups have repeated offsets.
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
                if prepared_ready is not None:
                    torch.cuda.current_stream(self.device).wait_event(
                        prepared_ready
                    )
                else:
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
                    if compact_grouped_prefill
                    else self._prefill_dequant_chunk_capacity(
                        int(self.store.cfg["n_experts"]),
                        routed_rows=count * top_k,
                    )
                )
                chunk_capacity = min(unique_count, planned_capacity)
                expert_chunks: list[tuple[int, int]] = []
                chunk_start = 0
                while chunk_start < unique_count:
                    chunk_stop = min(
                        unique_count, chunk_start + chunk_capacity
                    )
                    while (
                        prepared_selected is None
                        and not self._prefill_unique_fits(
                            layer, unique_ids[chunk_start:chunk_stop]
                        )
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
                kernel_group_capacity = int(effective_capacity)
                if (
                    not compact_grouped_prefill
                    and self._native8_grouped_backend == "deepgemm-sm90"
                ):
                    kernel_group_capacity = grouped_jit_bucket(
                        effective_capacity,
                        capacity=max(effective_capacity, planned_capacity),
                    )
                gu_buffer = down_buffer = None
                native8_workspace = None
                if not compact_grouped_prefill:
                    if self._native8_prefill_enabled:
                        native8_workspace = self._prefill_native8_workspace_for(
                            kernel_group_capacity,
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
                log_kind = _packed_executor_log_kind(
                    rows=rows,
                    batch_start=start,
                    announced=self._decode_executor_announced,
                )
                if log_kind == "prefill":
                    print(
                        f"[cccp-prefill] executor={active_prefill_executor}; "
                        f"layer={layer}; token batch={count}; "
                        f"unique experts={unique_count}; "
                        f"expert chunk={effective_capacity}; "
                        f"kernel bucket={kernel_group_capacity}; "
                        f"groups={len(expert_chunks)}; "
                        "capacity=automatic free-VRAM",
                        flush=True,
                    )
                elif log_kind == "decode":
                    print(
                        "[cccp-decode] "
                        f"executor={active_prefill_executor}; "
                        "routing=exact; detail=request-summary",
                        flush=True,
                    )
                    self._decode_executor_announced = True
                for chunk_start, chunk_stop in expert_chunks:
                    keys = [
                        (int(layer), expert_id)
                        for expert_id in unique_ids[chunk_start:chunk_stop]
                    ]
                    selected = (
                        {
                            key: prepared_selected[key]
                            for key in keys
                        }
                        if prepared_selected is not None
                        else self._ensure_locked(
                            keys,
                            prefetch=False,
                            defer_wait=True,
                        )
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
                    if prepared_selected is None:
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
                    execution_group_count = int(chunk_count)
                    if (
                        native8_workspace is not None
                        and self._native8_grouped_backend == "deepgemm-sm90"
                    ):
                        execution_group_count = grouped_jit_bucket(
                            chunk_count,
                            capacity=kernel_group_capacity,
                        )
                    if execution_group_count > chunk_count:
                        padding = execution_group_count - chunk_count
                        rows_host = [
                            list(row) + [row[-1]] * padding
                            for row in rows_host[:15]
                        ]
                        if native8_scales is not None:
                            native8_scales = list(native8_scales) + [
                                native8_scales[-1]
                            ] * padding
                    metadata_host[:, :execution_group_count].copy_(
                        torch.tensor(rows_host[:15], dtype=torch.long)
                    )
                    metadata[:, :execution_group_count].copy_(
                        metadata_host[:, :execution_group_count],
                        non_blocking=True,
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
                    deepgemm_layout = None
                    if (
                        native8_workspace is not None
                        and self._native8_grouped_backend == "deepgemm-sm90"
                    ):
                        # DeepGEMM consumes the already sorted per-row group
                        # id directly.  Avoid the extra arange/searchsorted
                        # kernel pair paid by torch._scaled_grouped_mm.
                        group_ids = None
                        offsets = None
                        deepgemm_layout = build_deepgemm_grouped_layout(
                            sorted_ids,
                            group_count=execution_group_count,
                            alignment=deepgemm_grouped_alignment(),
                        )
                    else:
                        group_ids = torch.arange(
                            chunk_count,
                            dtype=torch.long,
                            device=self.device,
                        )
                        offsets = torch.searchsorted(
                            sorted_ids, group_ids, right=True
                        ).to(torch.int32)
                    if compact_grouped_prefill:
                        if group_ids is None or offsets is None:
                            raise RuntimeError(
                                "HIP grouped Prefill is missing group offsets"
                            )
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
                                metadata[:, :execution_group_count].contiguous(),
                                gu_buffer[:execution_group_count],
                                down_buffer[:execution_group_count],
                            )
                            scale_host = native8_workspace[
                                "projection_scales_host"
                            ][:execution_group_count]
                            scale_values = native8_workspace[
                                "projection_scales"
                            ][:execution_group_count]
                            scale_host.copy_(torch.tensor(
                                native8_scales,
                                dtype=torch.float32,
                            ))
                            scale_values.copy_(
                                scale_host,
                                non_blocking=True,
                            )
                            intermediate = int(self.store.cfg["moe_inter"])
                            hidden = int(self.store.cfg.get(
                                "routed_hidden", self.store.cfg["hidden"]
                            ))
                            if self._native8_grouped_backend == "deepgemm-sm90":
                                gu_scales = native8_workspace[
                                    "gu_block_scales"
                                ][:execution_group_count]
                                down_scales = native8_workspace[
                                    "down_block_scales"
                                ][:execution_group_count]
                                projection_block_scales(
                                    scale_values,
                                    hidden=hidden,
                                    intermediate=intermediate,
                                    gate_up_output=gu_scales,
                                    down_output=down_scales,
                                )
                            else:
                                gu_scales = native8_workspace["gu_scales"][
                                    :execution_group_count
                                ]
                                down_scales = native8_workspace[
                                    "down_scales"
                                ][:execution_group_count]
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
                            input_gemm_scales = input_scales.view(-1)
                            gate_up_output = None
                            if self._native8_grouped_backend == "deepgemm-sm90":
                                input_gemm_scales = row_block_scales(
                                    input_scales,
                                    k=hidden,
                                    output=native8_workspace[
                                        "input_block_scales"
                                    ][:routed_count],
                                )
                                gate_up_output = native8_workspace[
                                    "gate_up_output"
                                ][:routed_count]
                            gate_up = execute_grouped_fp8(
                                native_input,
                                gu_buffer[:execution_group_count],
                                scale_a=input_gemm_scales,
                                scale_b=gu_scales,
                                offsets=offsets,
                                backend=self._native8_grouped_backend,
                                deepgemm_layout=deepgemm_layout,
                                deepgemm_workspace=(
                                    native8_workspace["gate_up_deepgemm"]
                                    if self._native8_grouped_backend
                                    == "deepgemm-sm90"
                                    else None
                                ),
                                output=gate_up_output,
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
                        native_activation_ready = False
                        if native8_workspace is not None:
                            routed_count = int(gate_up.shape[0])
                            native_activated = native8_workspace["activated"][
                                :routed_count
                            ]
                            activated_scales = native8_workspace[
                                "activated_scales"
                            ][:routed_count]
                            native_activation_ready = (
                                gated_activation_fp8_quantize_rows_fused(
                                    gate_up,
                                    native_activated,
                                    activated_scales,
                                    activation=activation,
                                    beta=float(activation_beta),
                                    linear_beta=activation_linear_beta,
                                    limit=float(limit),
                                )
                                is not None
                            )
                        if not native_activation_ready:
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
                            ).contiguous()
                        if native8_workspace is not None:
                            if (
                                not native_activation_ready
                                and dense_fp8_quantize_rows_fused(
                                    activated,
                                    native_activated,
                                    activated_scales,
                                )
                                is None
                            ):
                                raise RuntimeError(
                                    "native E4M3 activation quantizer rejected MoE hidden rows"
                                )
                            activated_gemm_scales = activated_scales.view(-1)
                            down_output = None
                            if self._native8_grouped_backend == "deepgemm-sm90":
                                activated_gemm_scales = row_block_scales(
                                    activated_scales,
                                    k=intermediate,
                                    output=native8_workspace[
                                        "activated_block_scales"
                                    ][:routed_count],
                                )
                                down_output = native8_workspace["down_output"][
                                    :routed_count
                                ]
                            down = execute_grouped_fp8(
                                native_activated,
                                down_buffer[:execution_group_count],
                                scale_a=activated_gemm_scales,
                                scale_b=down_scales,
                                offsets=offsets,
                                backend=self._native8_grouped_backend,
                                deepgemm_layout=deepgemm_layout,
                                deepgemm_workspace=(
                                    native8_workspace["down_deepgemm"]
                                    if self._native8_grouped_backend
                                    == "deepgemm-sm90"
                                    else None
                                ),
                                output=down_output,
                            )
                        else:
                            down = torch._grouped_mm(
                                activated,
                                down_buffer[:chunk_count].transpose(1, 2),
                                offs=offsets,
                            )
                        fused_reduce = None
                        if len(expert_chunks) == 1:
                            fused_reduce = routed_weighted_reduce_fused(
                                down,
                                sorted_positions,
                                flat_weights,
                                workspace["inverse_route_order"][
                                    : int(flat_weights.numel())
                                ],
                                batch_result,
                                top_k=top_k,
                            )
                        if fused_reduce is None:
                            routed_count = int(down.shape[0])
                            weighted_down = workspace["weighted_down"][
                                :routed_count
                            ]
                            weighted_down.copy_(down)
                            weighted_down.mul_(sorted_weights.unsqueeze(1))
                            batch_result.index_add_(
                                0,
                                sorted_tokens,
                                weighted_down,
                            )
                    self.prefill_expert_chunk_submissions += 1
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
                self._prefill_native8_workspace = None
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
        self._prefill_prepared.clear()
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
        """Compatibility alias for the single public Decode lifecycle.

        All GPU routed-VQ Decode calls use prepare_run plus finish_run.
        Keeping a second direct cache/metadata/kernel implementation here made
        model adapters select subtly different codebook formats and allowed the
        native8 directory to reach the BF16 packed kernel.  The alias preserves
        the external pool API without retaining that duplicate execution path.
        """

        pending = None
        try:
            pending = self.prepare_run(
                layer,
                value,
                route_ids,
                route_weights,
                activation=activation,
                activation_beta=activation_beta,
                activation_linear_beta=activation_linear_beta,
                limit=limit,
            )
            return self.finish_run(pending)
        except BaseException:
            self.cancel_run(pending)
            raise
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
            self._drop_device_cache_runtime()
            self.cache.clear()
            self._arenas = None
            self._prefill_spare_arenas = None
            self._prefill_hot_arenas = None
            self._prefill_hot_selected.clear()
            self._prefill_prepared.clear()
            self._prefill_planned_chunk_capacity = 0
            self._last_ids.clear()
            self._route_plans.clear()
            self.profile_hot_keys = ()
            self.profile_hot_cache_enabled = False
            self.profile_hot_slots = 0
            self._profile_hot_ready = False
            self._workspaces = None
            self._compact_q8_activation_workspaces = None
            self._prefill_workspace = None
            self._prefill_dequant_workspace = None
            self._prefill_native8_workspace = None
            self._native8_decode_workspace = None
            self._decode_codebook_scales_host = None
            self._decode_codebook_scales = None
            self._native8_decode_offsets = None
            self._metadata = None
            self._metadata_host = None
            self._slot_directory = None
            self._slot_update_host = None
            self._slot_scale_directory = None
            self._slot_scale_update_host = None
            self._prefill_directory_dirty = False
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
            # CUDA Graph teardown and non-default-stream allocator events can
            # enqueue their final releases during the first collection.  A
            # second synchronized trim prevents the old Prefill slab from
            # remaining reserved while the larger Decode slab is requested
            # under a hard per-process memory fraction.
            torch.cuda.synchronize(self.device)
            gc.collect()
            torch.cuda.empty_cache()
            self.build_gpu_arenas()
            if (
                warmed_before_resize
                and self._arena_phase == "decode"
                and not force_rebuild
            ):
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
        rebuild = self._arena_phase != "prefill"
        if current <= target and not rebuild:
            self._arena_phase = "prefill"
            return current, current
        self._arena_phase = "prefill"
        changed = self.resize_gpu_arenas(
            target,
            force_rebuild=rebuild,
        )
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
        ready = (
            self._decode_runtime_ready()
            and not self._prefill_directory_dirty
        )
        rebuild = self._arena_phase != "decode" or not ready
        # ``target`` is a byte budget, while ``current`` is the sum of whole
        # signature slots that fit inside it. The latter is intentionally a
        # little smaller because of per-signature rounding. Once a valid
        # Decode directory exists, comparing those unlike values rebuilt the
        # same 21.04-GiB slab for every request against a 21.38-GiB budget and
        # erased the complete LRU. Runtime growth/shrink uses ``trim_to``;
        # this phase transition only needs to initialize or repair Decode.
        if not rebuild:
            self._arena_phase = "decode"
            return current, current
        self._arena_phase = "decode"
        changed = self.resize_gpu_arenas(
            max(current, target),
            force_rebuild=rebuild,
        )
        if (
            rebuild
            and getattr(self, "host_dma_mode", "") == "direct-registered"
            and os.name != "nt"
        ):
            # The Prefill hot slab and Decode slab have different physical
            # addresses.  A phase rebuild must seed the newly allocated
            # Decode slab from measured corpus heat *before* publishing the
            # segmented device directory.  Runtime KV-pressure trims use the
            # non-forced resize path above and intentionally skip this upload.
            with self._transfer_lock:
                self._drop_device_cache_runtime()
                self._warm_profile_hot_locked()
                self._build_device_cache_runtime()
        if not self._decode_runtime_ready():
            raise RuntimeError(
                "packed hybrid Decode arena rebuild did not create runtime "
                f"buffers (pinned_experts={len(self.pinned)}, "
                f"arena={self.gpu_arena_bytes / 2**30:.2f}GiB)"
            )
        self._prefill_directory_dirty = False
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

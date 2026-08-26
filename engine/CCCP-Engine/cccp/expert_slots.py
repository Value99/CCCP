"""Stable GPU storage for the routed-expert VQ index tensors.

The expert cache changes ownership frequently, but its device addresses do not
need to change.  These small bookkeeping classes keep a fixed tensor arena and
lease integer slots to cache keys.  Replacing an owner therefore overwrites an
existing view instead of asking the CUDA allocator for another tensor.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence

import torch


ExpertKey = Hashable


@dataclass(frozen=True)
class SlotLease:
    slot: int
    key: ExpertKey
    generation: int
    replaced: ExpertKey | None = None


class SlotBook:
    """LRU ownership table for a fixed number of reusable slots."""

    def __init__(self, count: int):
        if count <= 0:
            raise ValueError("slot count must be positive")
        self.count = int(count)
        self._owners: list[ExpertKey | None] = [None] * self.count
        self._generations = [0] * self.count
        self._by_key: dict[ExpertKey, int] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._free = deque(range(self.count))
        self._inflight: set[int] = set()
        self._protected: set[int] = set()

    def acquire(self, key: ExpertKey) -> SlotLease:
        existing = self._by_key.get(key)
        if existing is not None:
            self._lru.move_to_end(existing)
            return SlotLease(
                existing,
                key,
                self._generations[existing],
            )

        replaced = None
        if self._free:
            slot = self._free.popleft()
        else:
            slot = -1
            for candidate in self._lru:
                if (
                    candidate not in self._inflight
                    and candidate not in self._protected
                ):
                    slot = candidate
                    break
            # Protection is a cache hint, not a correctness condition.  A
            # skewed route tier may temporarily protect every slot; fall back
            # to the oldest non-inflight owner instead of deadlocking.
            if slot < 0:
                for candidate in self._lru:
                    if candidate not in self._inflight:
                        slot = candidate
                        break
            if slot < 0:
                raise RuntimeError("all expert slots are in-flight")
            self._lru.pop(slot)
            self._protected.discard(slot)
            replaced = self._owners[slot]
            if replaced is not None:
                self._by_key.pop(replaced, None)

        self._generations[slot] += 1
        self._owners[slot] = key
        self._by_key[key] = slot
        self._lru[slot] = None
        return SlotLease(
            slot,
            key,
            self._generations[slot],
            replaced=replaced,
        )

    def release(self, lease: SlotLease) -> bool:
        slot = lease.slot
        if (
            slot < 0
            or slot >= self.count
            or self._owners[slot] != lease.key
            or self._generations[slot] != lease.generation
        ):
            return False
        if slot in self._inflight:
            raise RuntimeError("cannot release an in-flight expert slot")
        self._owners[slot] = None
        self._protected.discard(slot)
        self._by_key.pop(lease.key, None)
        self._lru.pop(slot, None)
        self._free.appendleft(slot)
        return True

    def touch(self, key: ExpertKey) -> None:
        slot = self._by_key.get(key)
        if slot is not None:
            self._lru.move_to_end(slot)

    def protect(self, key: ExpertKey) -> bool:
        slot = self._by_key.get(key)
        if slot is None:
            return False
        self._protected.add(slot)
        return True

    def unprotect(self, key: ExpertKey) -> None:
        slot = self._by_key.get(key)
        if slot is not None:
            self._protected.discard(slot)

    @property
    def protected_count(self) -> int:
        return len(self._protected)

    def mark_inflight(self, slot: int) -> None:
        if self._owners[slot] is None:
            raise RuntimeError("cannot mark an unowned slot in-flight")
        self._inflight.add(slot)

    def clear_inflight(self, slot: int) -> None:
        self._inflight.discard(slot)

    def owner(self, slot: int) -> ExpertKey | None:
        return self._owners[slot]


@dataclass(frozen=True)
class SegmentedCachePlan:
    """Fixed-shape result produced by one device-cache planning step.

    The production CUDA kernel writes equivalent tensors in place.  Tuples are
    used here so the CPU reference is deterministic and easy to compare in
    tests without becoming another runtime cache implementation.
    """

    num_routes: int
    num_unique: int
    num_hits: int
    num_fetch: int
    route_slots: tuple[int, ...]
    src_ids: tuple[int, ...]
    evict_slots: tuple[int, ...]


class SegmentedCacheReference:
    """Exact CPU oracle for the graph-resident heterogeneous expert LRU.

    Recency is global across layers while physical victims stay inside the
    compatible packed-signature segment.  This class is deliberately not used
    by production inference; the CUDA controller is validated against it.
    """

    def __init__(
        self,
        *,
        n_layers: int,
        n_experts: int,
        signature_of_id: Sequence[int],
        slots_per_signature: Mapping[int, int],
        max_routes: int,
    ):
        self.n_layers = int(n_layers)
        self.n_experts = int(n_experts)
        self.max_routes = int(max_routes)
        if self.n_layers <= 0 or self.n_experts <= 0:
            raise ValueError("cache dimensions must be positive")
        if self.max_routes <= 0:
            raise ValueError("max_routes must be positive")
        expected = self.n_layers * self.n_experts
        self.signature_of_id = tuple(int(value) for value in signature_of_id)
        if len(self.signature_of_id) != expected:
            raise ValueError(
                "signature_of_id must contain one entry per logical expert"
            )

        offset = 0
        self._segments: dict[int, range] = {}
        self._signature_for_slot: list[int] = []
        for signature, raw_count in sorted(
            (int(key), int(value))
            for key, value in slots_per_signature.items()
        ):
            if raw_count <= 0:
                raise ValueError("each signature segment must have slots")
            segment = range(offset, offset + raw_count)
            self._segments[signature] = segment
            self._signature_for_slot.extend([signature] * raw_count)
            offset += raw_count
        missing = set(self.signature_of_id) - self._segments.keys()
        if missing:
            raise ValueError(
                "missing slot segment for signatures: "
                + ", ".join(str(value) for value in sorted(missing))
            )

        self._slot_for_id = [-1] * expected
        self._id_of_slot = [-1] * offset
        self._last_used = [0] * offset
        self._step = 0

    def _logical_id(self, layer: int, expert: int) -> int:
        layer = int(layer)
        expert = int(expert)
        if layer < 0 or layer >= self.n_layers:
            raise IndexError(f"layer out of range: {layer}")
        if expert < 0 or expert >= self.n_experts:
            raise IndexError(f"expert out of range: {expert}")
        return layer * self.n_experts + expert

    def slot_for(self, layer: int, expert: int) -> int:
        return self._slot_for_id[self._logical_id(layer, expert)]

    def signature_for_slot(self, slot: int) -> int:
        return self._signature_for_slot[int(slot)]

    def _touch(self, slot: int) -> None:
        self._step += 1
        self._last_used[slot] = self._step

    def plan(
        self,
        layer: int,
        expert_ids: Sequence[int],
    ) -> SegmentedCachePlan:
        route = tuple(int(expert) for expert in expert_ids)
        if len(route) > self.max_routes:
            raise ValueError(
                f"route count {len(route)} exceeds max_routes={self.max_routes}"
            )
        logical_route = tuple(
            self._logical_id(layer, expert) for expert in route
        )
        unique_logical = tuple(dict.fromkeys(logical_route))

        required: dict[int, int] = {}
        for logical in unique_logical:
            signature = self.signature_of_id[logical]
            required[signature] = required.get(signature, 0) + 1
        for signature, count in required.items():
            capacity = len(self._segments[signature])
            if count > capacity:
                raise RuntimeError(
                    "simultaneous routes exceed compatible signature segment: "
                    f"signature={signature}, need={count}, have={capacity}"
                )

        assigned: dict[int, int] = {}
        reserved_slots: set[int] = set()
        misses: list[tuple[int, int]] = []
        hits = 0
        for logical in unique_logical:
            existing = self._slot_for_id[logical]
            if existing >= 0:
                hits += 1
                assigned[logical] = existing
                reserved_slots.add(existing)
                self._touch(existing)
                continue

            signature = self.signature_of_id[logical]
            segment = self._segments[signature]
            free = next(
                (
                    slot for slot in segment
                    if self._id_of_slot[slot] < 0
                    and slot not in reserved_slots
                ),
                None,
            )
            if free is None:
                candidates = [
                    slot for slot in segment if slot not in reserved_slots
                ]
                if not candidates:
                    raise RuntimeError(
                        "simultaneous routes cannot share one cache slot"
                    )
                free = min(
                    candidates,
                    key=lambda slot: (self._last_used[slot], slot),
                )
            previous = self._id_of_slot[free]
            if previous >= 0:
                self._slot_for_id[previous] = -1
            self._id_of_slot[free] = logical
            self._slot_for_id[logical] = free
            assigned[logical] = free
            reserved_slots.add(free)
            self._touch(free)
            misses.append((logical, free))

        route_slots = [assigned[logical] for logical in logical_route]
        route_slots.extend([-1] * (self.max_routes - len(route_slots)))
        src_ids = [logical % self.n_experts for logical, _slot in misses]
        evict_slots = [slot for _logical, slot in misses]
        padding = self.max_routes - len(misses)
        src_ids.extend([-1] * padding)
        evict_slots.extend([-1] * padding)
        return SegmentedCachePlan(
            num_routes=len(route),
            num_unique=len(unique_logical),
            num_hits=hits,
            num_fetch=len(misses),
            route_slots=tuple(route_slots),
            src_ids=tuple(src_ids),
            evict_slots=tuple(evict_slots),
        )


@dataclass(frozen=True)
class ExpertSignature:
    gu_shape: tuple[int, ...]
    gu_dtype: torch.dtype
    dn_shape: tuple[int, ...]
    dn_dtype: torch.dtype

    @classmethod
    def of(cls, expert) -> "ExpertSignature":
        gu, dn = expert
        return cls(
            tuple(gu.idx.shape),
            gu.idx.dtype,
            tuple(dn.idx.shape),
            dn.idx.dtype,
        )

    @property
    def slot_bytes(self) -> int:
        gu_items = 1
        for item in self.gu_shape:
            gu_items *= item
        dn_items = 1
        for item in self.dn_shape:
            dn_items *= item
        return (
            gu_items * torch.empty((), dtype=self.gu_dtype).element_size()
            + dn_items * torch.empty((), dtype=self.dn_dtype).element_size()
        )


class GpuExpertArena:
    """One fixed pair of contiguous index tensors for one expert signature."""

    def __init__(
        self,
        count: int,
        signature: ExpertSignature,
        device: torch.device | str,
    ):
        self.signature = signature
        self.book = SlotBook(count)
        self.gu = torch.empty(
            (count, *signature.gu_shape),
            dtype=signature.gu_dtype,
            device=device,
        )
        self.dn = torch.empty(
            (count, *signature.dn_shape),
            dtype=signature.dn_dtype,
            device=device,
        )

    @property
    def nbytes(self) -> int:
        return (
            self.gu.numel() * self.gu.element_size()
            + self.dn.numel() * self.dn.element_size()
        )

    def lease(self, key: ExpertKey):
        lease = self.book.acquire(key)
        return lease, self.gu[lease.slot], self.dn[lease.slot]


class GpuExpertArenas:
    """Collection of signature-specific arenas with key-to-lease lookup."""

    def __init__(
        self,
        specs: Iterable[tuple[ExpertSignature, int]],
        device: torch.device | str,
    ):
        self.arenas = {
            signature: GpuExpertArena(count, signature, device)
            for signature, count in specs
            if count > 0
        }
        if not self.arenas:
            raise ValueError("at least one expert arena is required")
        self._leases: dict[ExpertKey, tuple[ExpertSignature, SlotLease]] = {}

    @property
    def nbytes(self) -> int:
        return sum(arena.nbytes for arena in self.arenas.values())

    def owns(self, key: ExpertKey) -> bool:
        return key in self._leases

    def supports(self, expert) -> bool:
        return ExpertSignature.of(expert) in self.arenas

    def capacity(self, expert) -> int:
        arena = self.arenas.get(ExpertSignature.of(expert))
        return 0 if arena is None else arena.book.count

    def lease(self, key: ExpertKey, expert):
        signature = ExpertSignature.of(expert)
        arena = self.arenas[signature]
        lease, gu, dn = arena.lease(key)
        if lease.replaced is not None:
            self._leases.pop(lease.replaced, None)
        self._leases[key] = (signature, lease)
        return lease, gu, dn

    def release(self, key: ExpertKey) -> bool:
        item = self._leases.pop(key, None)
        if item is None:
            return False
        signature, lease = item
        return self.arenas[signature].book.release(lease)

    def touch(self, key: ExpertKey) -> None:
        item = self._leases.get(key)
        if item is not None:
            signature, _ = item
            self.arenas[signature].book.touch(key)

    def mark_inflight(self, key: ExpertKey) -> None:
        signature, lease = self._leases[key]
        self.arenas[signature].book.mark_inflight(lease.slot)

    def clear_inflight(self, key: ExpertKey) -> None:
        item = self._leases.get(key)
        if item is not None:
            signature, lease = item
            self.arenas[signature].book.clear_inflight(lease.slot)

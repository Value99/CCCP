"""Stable GPU storage for the routed-expert VQ index tensors.

The expert cache changes ownership frequently, but its device addresses do not
need to change.  These small bookkeeping classes keep a fixed tensor arena and
lease integer slots to cache keys.  Replacing an owner therefore overwrites an
existing view instead of asking the CUDA allocator for another tensor.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Hashable, Iterable

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

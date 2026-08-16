"""Paged state storage used by long-context DSV4 attention."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from .fusedext import paged_gather_bf16_fused as _paged_gather_bf16_fused


class ContextCapacityError(RuntimeError):
    """A cache page could not be reserved before mutating model state."""

    def __init__(self, position: int, cause: BaseException):
        super().__init__(
            f"无法为 position={position} 扩展 DSV4 KV cache: {cause}"
        )
        self.position = position
        self.cause = cause
        self.committed = 0


class PagedKV:
    """Stable-address pages for compressed KV or Indexer state."""

    def __init__(
        self,
        *,
        batch: int,
        page_items: int,
        dim: int,
        device,
        dtype=torch.bfloat16,
        max_items: int | None = None,
        page_allocator: Callable[..., torch.Tensor] | None = None,
    ):
        if batch <= 0 or page_items <= 0 or dim <= 0:
            raise ValueError("batch, page_items and dim must be positive")
        self.batch = batch
        self.page_items = page_items
        self.dim = dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_items = max_items
        self.pages: list[torch.Tensor] = []
        self.length = 0
        self._device_ptrs: torch.Tensor | None = None
        self._page_allocator = page_allocator or torch.empty

    def ensure_page(self, page_index: int) -> torch.Tensor:
        if page_index < 0:
            raise IndexError("page_index must be non-negative")
        page_start = page_index * self.page_items
        if self.max_items is not None and page_start >= self.max_items:
            raise RuntimeError(f"KV cache reached max_items={self.max_items}")
        while len(self.pages) <= page_index:
            # Paged KV is persistent mutable model state. Decode and CUDA/HIP
            # graph preparation can call reserve() under inference_mode; a
            # tensor created there may not be updated later by the canonical
            # batched-prefill/no_grad path. Allocate the backing page outside
            # inference_mode so every scheduler can safely commit into the
            # same fixed-address storage. This is an ownership invariant, not
            # a clone/fallback on the write path.
            with torch.inference_mode(False):
                page = self._page_allocator(
                    (self.batch, self.page_items, self.dim),
                    device=self.device,
                    dtype=self.dtype,
                )
            if page.shape != (self.batch, self.page_items, self.dim):
                raise RuntimeError(
                    "page allocator returned shape "
                    f"{tuple(page.shape)}, expected "
                    f"{(self.batch, self.page_items, self.dim)}"
                )
            self.pages.append(page)
            self._device_ptrs = None
        return self.pages[page_index]

    def reserve(self, item: int) -> None:
        self._validate_item(item)
        self.ensure_page(item // self.page_items)

    def write(self, item: int, value: torch.Tensor) -> None:
        self._validate_item(item)
        page_index, offset = divmod(item, self.page_items)
        page = self.ensure_page(page_index)
        converted = value.to(device=self.device, dtype=self.dtype)
        if converted.shape == (self.dim,) and self.batch == 1:
            converted = converted.unsqueeze(0)
        if converted.shape != (self.batch, self.dim):
            raise ValueError(
                f"expected value shape {(self.batch, self.dim)}, "
                f"got {tuple(converted.shape)}"
            )
        page[:, offset].copy_(converted)
        self.length = max(self.length, item + 1)

    def write_many(self, start: int, values: torch.Tensor) -> None:
        converted = values.to(device=self.device, dtype=self.dtype)
        if converted.dim() != 3 or converted.shape[0] != self.batch:
            raise ValueError(
                f"expected [batch, items, dim], got {tuple(converted.shape)}"
            )
        if converted.shape[2] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {converted.shape[2]}")
        count = converted.shape[1]
        if count == 0:
            return
        self._validate_item(start)
        self._validate_item(start + count - 1)
        first_page = start // self.page_items
        last_page = (start + count - 1) // self.page_items
        for page_index in range(first_page, last_page + 1):
            self.ensure_page(page_index)
        copied = 0
        while copied < count:
            item = start + copied
            page_index, offset = divmod(item, self.page_items)
            take = min(count - copied, self.page_items - offset)
            self.pages[page_index][:, offset:offset + take].copy_(
                converted[:, copied:copied + take]
            )
            copied += take
        self.length = max(self.length, start + count)

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        fused = self._gather_cuda(indices.reshape(-1))
        if fused is not None:
            return fused.view(self.batch, -1, self.dim)
        flat = indices.to("cpu", dtype=torch.long).reshape(-1).tolist()
        if not flat:
            return torch.empty(
                self.batch, 0, self.dim, device=self.device, dtype=self.dtype
            )
        values = []
        for item in flat:
            if item < 0:
                raise IndexError("KV index must be non-negative")
            page_index, offset = divmod(item, self.page_items)
            if page_index >= len(self.pages):
                raise IndexError(f"KV page for item={item} is not allocated")
            values.append(self.pages[page_index][:, offset])
        return torch.stack(values, dim=1)

    def gather_batched(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather per-batch indices into ``[batch, ..., dim]``."""
        if indices.ndim < 1 or indices.shape[0] != self.batch:
            raise ValueError(
                f"expected indices leading batch={self.batch}, "
                f"got {tuple(indices.shape)}"
            )
        fused = self._gather_cuda(indices)
        if fused is not None:
            return fused.view(*indices.shape, self.dim)
        flat = indices.to("cpu", dtype=torch.long).reshape(self.batch, -1)
        rows = []
        for batch_index in range(self.batch):
            values = []
            for item in flat[batch_index].tolist():
                if item < 0:
                    raise IndexError("KV index must be non-negative")
                page_index, offset = divmod(item, self.page_items)
                if page_index >= len(self.pages):
                    raise IndexError(f"KV page for item={item} is not allocated")
                values.append(self.pages[page_index][batch_index, offset])
            if values:
                rows.append(torch.stack(values))
            else:
                rows.append(torch.empty(
                    0, self.dim, device=self.device, dtype=self.dtype
                ))
        return torch.stack(rows).view(*indices.shape, self.dim)

    def _gather_cuda(self, indices: torch.Tensor) -> torch.Tensor | None:
        if (
            self.device.type != "cuda"
            or self.batch != 1
            or self.dtype != torch.bfloat16
            or os.environ.get("CCCP_PAGED_KV_STRICT", "0") == "1"
        ):
            return None
        # Production decode indices come from CUDA arange/topk. Do not inspect
        # their values on the host or launch a validation reduction here.
        converted = indices.to(device=self.device, dtype=torch.long)
        return _paged_gather_bf16_fused(
            self.device_page_ptrs(),
            converted,
            self.page_items,
            self.dim,
        )

    def contiguous_prefix(self, length: int | None = None) -> torch.Tensor:
        """Expose the legacy full-attention view while it fits in page zero."""
        length = self.length if length is None else length
        if length < 0 or length > self.length:
            raise IndexError(
                f"requested prefix length={length}, logical length={self.length}"
            )
        if length == 0:
            return torch.empty(
                self.batch, 0, self.dim, device=self.device, dtype=self.dtype
            )
        if length > self.page_items:
            raise RuntimeError(
                "compressed KV spans multiple pages; sparse gather is required"
            )
        return self.pages[0][:, :length]

    def device_page_ptrs(self) -> torch.Tensor:
        if self._device_ptrs is None:
            self._device_ptrs = torch.tensor(
                [page.data_ptr() for page in self.pages],
                device=self.device,
                dtype=torch.int64,
            )
        return self._device_ptrs

    def truncate(self, length: int) -> None:
        if length < 0 or length > self.length:
            raise ValueError(
                f"cannot truncate logical length {self.length} to {length}"
            )
        self.length = length

    def reset(self) -> None:
        self.length = 0

    def _validate_item(self, item: int) -> None:
        if item < 0:
            raise IndexError("KV item must be non-negative")
        if self.max_items is not None and item >= self.max_items:
            raise RuntimeError(f"KV cache reached max_items={self.max_items}")

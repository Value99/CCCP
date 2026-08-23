"""Model-independent fixed-address caches for sparse paged attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch


MODEL1_HEAD_DIM = 512
MODEL1_NOPE_DIM = 448
MODEL1_ROPE_DIM = 64
MODEL1_SCALE_COUNT = 7
MODEL1_PAYLOAD_BYTES = 576
MODEL1_SCALE_STRIDE = 8
MODEL1_STORAGE_BYTES = MODEL1_PAYLOAD_BYTES + MODEL1_SCALE_STRIDE


@dataclass
class Model1FP8PagedCache:
    """FlashMLA Model1 cache with stable page addresses.

    The final dimension is a byte carrier.  Within each page all 576-byte
    token payloads precede the eight-byte E8M0 scale records; callers must
    never interpret ``storage[page, slot]`` as a conventional row tensor.
    """

    storage: torch.Tensor
    page_items: int
    max_items: int

    @classmethod
    def allocate(
        cls,
        *,
        max_items: int,
        page_items: int,
        device,
    ) -> "Model1FP8PagedCache":
        if max_items <= 0 or page_items <= 0:
            raise ValueError("max_items and page_items must be positive")
        pages = (int(max_items) + int(page_items) - 1) // int(page_items)
        storage = torch.zeros(
            pages,
            int(page_items),
            1,
            MODEL1_STORAGE_BYTES,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        return cls(storage, int(page_items), int(max_items))

    @property
    def page_count(self) -> int:
        return int(self.storage.shape[0])

    def physical_indices(self, count: int) -> torch.Tensor:
        if count < 0 or count > self.max_items:
            raise ValueError(f"count={count} exceeds max_items={self.max_items}")
        return torch.arange(count, dtype=torch.int32, device=self.storage.device)

    @torch.no_grad()
    def load_bf16(self, values: torch.Tensor) -> None:
        """Build the runtime view once from an existing BF16/FP32 prefix."""
        rows = values.reshape(-1, MODEL1_HEAD_DIM).to(
            device=self.storage.device,
            dtype=torch.bfloat16,
        )
        if rows.shape[0] > self.max_items:
            raise ValueError("source prefix exceeds Model1 cache capacity")
        for page in range(self.page_count):
            start = page * self.page_items
            count = min(self.page_items, max(0, rows.shape[0] - start))
            if count == 0:
                break
            source = rows[start:start + count]
            raw = self.storage[page].view(torch.uint8).reshape(-1)
            payload = raw[:self.page_items * MODEL1_PAYLOAD_BYTES].view(
                self.page_items, MODEL1_PAYLOAD_BYTES
            )
            scale_bytes = raw[
                self.page_items * MODEL1_PAYLOAD_BYTES:
            ].view(self.page_items, MODEL1_SCALE_STRIDE)
            tiles = source[:, :MODEL1_NOPE_DIM].float().view(
                count, MODEL1_SCALE_COUNT, 64
            )
            scales = (tiles.abs().amax(dim=-1) / 448.0).clamp_min(1.0e-4)
            scales = scales.log2().ceil().exp2()
            quantized = (tiles / scales.unsqueeze(-1)).to(
                torch.float8_e4m3fn
            ).reshape(count, MODEL1_NOPE_DIM)
            payload[:count, :MODEL1_NOPE_DIM].copy_(quantized.view(torch.uint8))
            payload[:count, MODEL1_NOPE_DIM:].view(torch.bfloat16).copy_(
                source[:, MODEL1_NOPE_DIM:]
            )
            scale_bytes[:count, :MODEL1_SCALE_COUNT].copy_(
                scales.to(torch.float8_e8m0fnu).view(torch.uint8)
            )
            scale_bytes[:count, MODEL1_SCALE_COUNT].zero_()


@dataclass
class IndexerFP8PagedCache:
    """Contiguous FP8 Indexer keys plus per-row FP32 dequant scales."""

    values: torch.Tensor
    scales: torch.Tensor
    page_items: int
    max_items: int

    @classmethod
    def allocate(
        cls,
        *,
        max_items: int,
        head_dim: int,
        page_items: int,
        device,
    ) -> "IndexerFP8PagedCache":
        if max_items <= 0 or head_dim <= 0 or page_items <= 0:
            raise ValueError("cache dimensions must be positive")
        padded = (
            (int(max_items) + int(page_items) - 1) // int(page_items)
            * int(page_items)
        )
        return cls(
            torch.zeros(
                padded,
                int(head_dim),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            torch.ones(padded, dtype=torch.float32, device=device),
            int(page_items),
            int(max_items),
        )

    @torch.no_grad()
    def load_bf16(self, values: torch.Tensor) -> None:
        rows = values.reshape(-1, self.values.shape[1]).to(
            device=self.values.device,
            dtype=torch.bfloat16,
        )
        if rows.shape[0] > self.max_items:
            raise ValueError("source prefix exceeds Indexer cache capacity")
        scales = (rows.float().abs().amax(dim=-1) / 448.0).clamp_min(1.0e-8)
        self.scales[:rows.shape[0]].copy_(scales)
        self.values[:rows.shape[0]].copy_(
            (rows.float() / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
        )


def cuda_architecture_features(device) -> tuple[str, ...]:
    """Return stable registry capability names, never a model identifier."""
    target = torch.device(device)
    if target.type != "cuda":
        return ()
    major, minor = torch.cuda.get_device_capability(target)
    features = [f"sm{major}{minor}"]
    if major >= 9:
        features.extend(("tensorcore", "tensorcore_fp8", "splitkv"))
    elif major == 8 and minor >= 9:
        # Ada has native FP8 Tensor Cores, but a SplitKV implementation is a
        # separately registered backend capability rather than an assumption.
        features.extend(("tensorcore", "tensorcore_fp8"))
    elif major >= 8:
        features.append("tensorcore")
    return tuple(features)


__all__ = [
    "IndexerFP8PagedCache",
    "Model1FP8PagedCache",
    "cuda_architecture_features",
]

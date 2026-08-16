"""不加载 torch 的模型常驻容量计算。"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
from dataclasses import dataclass
from typing import Mapping, Any


@dataclass(frozen=True)
class DSV4ContextMemory:
    """Exact batch-1 DSV4 state bytes at a declared context ceiling."""

    max_ctx: int
    rope_bytes: int
    fixed_state_bytes: int
    paged_state_bytes: int
    total_bytes: int
    asymptotic_bytes_per_token: int


def _page_capacity(items: int, page_items: int) -> int:
    return 0 if items <= 0 else math.ceil(items / page_items) * page_items


def dsv4_context_runtime_bytes(
    config: Mapping[str, Any],
    max_ctx: int,
    *,
    batch: int = 1,
) -> DSV4ContextMemory:
    """Return full DSV4 RoPE, sliding-window and paged-state residency.

    This follows ``DSV4CCCPModel._allocate_states`` exactly. Compressed KV and
    Indexer pages grow lazily during inference, while both RoPE tables are
    allocated up front. The result therefore describes memory at the end of
    the declared context rather than claiming that all pages exist at startup.
    """

    context = max(0, int(max_ctx))
    batch_size = max(1, int(batch))
    layers = int(config["n_layers"])
    head_dim = int(config["head_dim"])
    rope_dim = int(config["qk_rope_head_dim"])
    window = int(config.get("sliding_window", 0))
    index_dim = int(config.get("index_head_dim", 128))
    ratios = (
        list(config.get("compress_ratios") or []) + [0] * layers
    )[:layers]

    # Two RoPE caches, each with cos+sin FP32 [max_ctx+8, rope_dim/2].
    rope_bytes = (context + 8) * rope_dim * 8
    # Every layer retains a raw FP32 MQA window plus int64 positions.
    fixed = layers * batch_size * window * (head_dim * 4 + 8)
    paged = 0
    per_token = rope_dim * 8
    for raw_ratio in ratios:
        ratio = int(raw_ratio)
        if ratio <= 0:
            continue
        page_items = max(1, 4096 // ratio)
        items = math.ceil(context / ratio)
        capacity = _page_capacity(items, page_items)
        # Main compressed MQA values use BF16; page pointers are int64.
        pages = 0 if capacity == 0 else capacity // page_items
        paged += batch_size * capacity * head_dim * 2 + pages * 8
        per_token += math.ceil(batch_size * head_dim * 2 / ratio)

        coff = 2 if ratio == 4 else 1
        main_scratch = batch_size * (coff * ratio) * (coff * head_dim)
        fixed += main_scratch * (2 + 4)
        if ratio == 4:
            # Lightning Indexer keeps a second BF16 paged key stream and its
            # own compressor scratch (BF16 values + FP32 scores).
            paged += batch_size * capacity * index_dim * 2 + pages * 8
            per_token += math.ceil(batch_size * index_dim * 2 / ratio)
            index_scratch = (
                batch_size * (coff * ratio) * (coff * index_dim)
            )
            fixed += index_scratch * (2 + 4)

    total = rope_bytes + fixed + paged
    return DSV4ContextMemory(
        max_ctx=context,
        rope_bytes=rope_bytes,
        fixed_state_bytes=fixed,
        paged_state_bytes=paged,
        total_bytes=total,
        asymptotic_bytes_per_token=per_token,
    )


def dsv4_max_context_for_budget(
    config: Mapping[str, Any],
    budget_bytes: int,
    *,
    batch: int = 1,
) -> int:
    """Largest configured context whose complete paged state fits a budget."""

    limit = max(0, int(config.get("max_position_embeddings", 1_048_576)))
    budget = max(0, int(budget_bytes))
    low, high = 0, limit
    while low < high:
        middle = (low + high + 1) // 2
        need = dsv4_context_runtime_bytes(
            config,
            middle,
            batch=batch,
        ).total_bytes
        if need <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def _dense_bf16_all(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "all"}


def _dsv4_bf16_eligible(name: str) -> bool:
    if name in {"head.weight", "embed.weight", "norm.weight"}:
        return True
    if ".ffn.shared_experts." in name or name.endswith("_fn"):
        return True
    if name.endswith(".attn_norm.weight") or name.endswith(".ffn_norm.weight"):
        return True
    if name.endswith(".q_norm.weight") or name.endswith(".kv_norm.weight"):
        return False
    if name.endswith(".norm.weight") or name.endswith(".attn.attn_sink"):
        return False
    return (
        ".attn.indexer." in name
        or ".attn.compressor." in name
        or ".attn." in name
    )


def dsv4_dense_runtime_bytes(
    path: str | Path,
    dense_bf16: str | None,
) -> int:
    """Return DSV4 Dense's exact steady GPU bytes from its tensor header.

    Int4 张量在 safetensors 中的最后一维为逻辑宽度的一半；展开成 BF16 后，
    字节数是打包元素数的四倍。未进入 BF16 常驻组的量化张量保持 q+s，普通
    张量保持源 dtype。``head.weight`` 若本来就是 BF16 也只占源文件中的
    BF16 字节，不能再按 vocab×hidden×FP32 重复追加；这正是小显存极限模式
    过去产生约 2 GiB 假 OOM 的原因。
    """
    expand_bf16 = _dense_bf16_all(dense_bf16)
    source = Path(path)
    with source.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size).decode("utf-8"))
    tensors = {
        name: info
        for name, info in header.items()
        if name != "__metadata__"
    }
    total = 0
    for name, info in tensors.items():
        if name.endswith(".qs"):
            continue
        elements = math.prod(int(value) for value in info["shape"])
        scale = tensors.get(name + ".qs")
        quantized = scale is not None
        if expand_bf16 and _dsv4_bf16_eligible(name):
            total += elements * (4 if quantized else 2)
        elif quantized:
            if name == "head.weight" or name.endswith("_fn"):
                # Packed Int4 has half as many stored elements as its logical
                # matrix. These consumers explicitly materialize FP32.
                total += elements * 8
            else:
                total += (
                    int(info["data_offsets"][1])
                    - int(info["data_offsets"][0])
                    + int(scale["data_offsets"][1])
                    - int(scale["data_offsets"][0])
                )
        else:
            total += (
                int(info["data_offsets"][1])
                - int(info["data_offsets"][0])
            )
    return total


__all__ = [
    "DSV4ContextMemory",
    "dsv4_context_runtime_bytes",
    "dsv4_dense_runtime_bytes",
    "dsv4_max_context_for_budget",
]

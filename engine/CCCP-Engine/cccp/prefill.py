"""Model-independent scheduling helpers for long-context block prefill."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import TypeVar


T = TypeVar("T")


def prefill_ranges(tokens: int, chunk_size: int = 4096):
    """Yield ``(start, end)`` ranges covering ``tokens`` exactly."""
    tokens = int(tokens)
    chunk_size = int(chunk_size)
    if tokens < 0 or chunk_size <= 0:
        raise ValueError("tokens must be non-negative and chunk_size positive")
    for start in range(0, tokens, chunk_size):
        yield start, min(tokens, start + chunk_size)


def batched_packed_prefill_available(pool, full_gpu: bool) -> bool:
    """Require an explicit row-batched packed-MoE capability."""
    return bool(
        getattr(pool, "prefill_rows_supported", False)
        and callable(getattr(pool, "run_rows", None))
    )


def begin_prefill_block(pool) -> bool:
    """Switch a packed pool into its batched-Prefill arena when supported.

    Returns whether the pool exposes the phase API, so the caller can pair
    the call with :func:`end_prefill_block` in a ``finally`` regardless of
    the pool implementation behind it.
    """
    activate = getattr(pool, "activate_prefill_arena", None)
    if callable(activate):
        activate()
        return True
    return False


def end_prefill_block(pool, *, restore_decode: bool = True) -> None:
    """Release the block-scoped Prefill workspace and restore the Decode arena.

    ``run_rows`` retains its expert-expansion scratch for the whole block;
    every block driver (engine schedulers, Kimi/MTP prefill) must close the
    block through this helper so the scratch cannot leak into Decode.  Pools
    without the phase API are a no-op.
    """
    release = getattr(pool, "release_host_rows_workspace", None)
    if callable(release):
        release()
    if restore_decode:
        activate = getattr(pool, "activate_decode_arena", None)
        if callable(activate):
            activate()


def prefill_block_size(
    *,
    env_name: str = "CCCP_PREFILL_BLOCK_TOKENS",
    default: int = 4096,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Resolve the shared outer-block size, preserving explicit overrides."""
    try:
        value = int(os.environ.get(env_name, str(default)))
    except (TypeError, ValueError):
        value = int(default)
    value = max(int(minimum), value)
    return value if maximum is None else min(int(maximum), value)


def prefill_block_size_with_legacy(
    legacy_env_name: str,
    *,
    default: int = 4096,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Use a model-specific legacy override, otherwise the shared setting."""
    env_name = (
        legacy_env_name
        if legacy_env_name in os.environ
        else "CCCP_PREFILL_BLOCK_TOKENS"
    )
    return prefill_block_size(
        env_name=env_name,
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def run_prefill_blocks(
    values: Iterable[T],
    evaluate_block: Callable[[list[T]], T],
    *,
    block_size: int,
) -> list[T]:
    """Evaluate a sequence in bounded blocks using one common scheduler."""
    items = list(values)
    return [
        evaluate_block(items[start:end])
        for start, end in prefill_ranges(len(items), block_size)
    ]


def prefill_moe_batch_size(
    default: int = 256,
    maximum: int = 8192,
) -> int:
    """Resolve a bounded packed-MoE micro-batch for one runtime.

    The conservative shared default is deliberately 256: packed expert
    workspaces are allocated by the common executor and DSV4 historically
    used that bound.  A model with a measured larger-safe workspace may pass
    an explicit ``default`` (Kimi's full-resident TP path uses 8192), while an
    operator/CLI override remains authoritative for both runtimes.
    """
    try:
        value = int(os.environ.get("CCCP_PREFILL_MOE_BATCH", str(default)))
    except (TypeError, ValueError):
        value = int(default)
    return max(1, min(int(maximum), value))


__all__ = [
    "batched_packed_prefill_available",
    "begin_prefill_block",
    "end_prefill_block",
    "prefill_block_size",
    "prefill_block_size_with_legacy",
    "prefill_ranges",
    "run_prefill_blocks",
    "prefill_moe_batch_size",
]

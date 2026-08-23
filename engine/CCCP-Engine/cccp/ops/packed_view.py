"""Static runtime views for compact packed-VQ projection weights.

The on-disk/RAM/VRAM payload is never rewritten or expanded.  A view only
adds a few immutable integers per projection so CUDA kernels can address a
row as fixed groups of eight indices, in the same spirit as Marlin's static
tile descriptors.  The descriptor is deliberately model agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence


LEGACY_ROWS_PER_PROJECTION = 5
TILE_ROWS_PER_PROJECTION = 4
PACKED_TILE_GROUP = 8
PACKED_TILE_BLOCKS = 32


def packed_tile_descriptor(
    *,
    bits: int,
    blocks: int,
) -> tuple[int, int, int, int]:
    """Return ``(bits, row_bytes, group, tile_blocks)`` for one weight.

    A group of eight indices occupies exactly ``bits`` bytes.  Odd 9..15 bit
    formats therefore need no row padding when their block count is a
    multiple of eight.  Other formats retain a valid descriptor but use the
    legacy decoder until a specialized tile reader is registered.
    """

    bits = int(bits)
    blocks = int(blocks)
    if not 8 <= bits <= 16:
        raise ValueError(f"packed VQ bit width must be in [8,16], got {bits}")
    if blocks <= 0:
        raise ValueError("packed VQ block count must be positive")
    row_bytes = (blocks * bits + 7) // 8
    group = (
        PACKED_TILE_GROUP
        if 9 <= bits <= 15 and blocks % PACKED_TILE_GROUP == 0
        else 1
    )
    return bits, row_bytes, group, PACKED_TILE_BLOCKS


def runtime_metadata_row_count(projection_count: int) -> int:
    """Metadata rows used by the static runtime view.

    Two-projection legacy Kimi archives stay byte-for-byte/API compatible.
    Three-projection projection-VQ archives receive the additional static
    tile rows consumed by the common CUDA backend.
    """

    projection_count = int(projection_count)
    if projection_count not in (2, 3):
        raise ValueError("packed experts must have two or three projections")
    legacy = projection_count * LEGACY_ROWS_PER_PROJECTION
    if projection_count == 2:
        return legacy
    return legacy + projection_count * TILE_ROWS_PER_PROJECTION


def build_runtime_metadata_rows(experts: Sequence[Sequence]) -> list[list[int]]:
    """Build pointer metadata plus immutable tile descriptors.

    The returned rows are tiny scheduling metadata.  They contain no index
    values, no expanded ``uint16`` matrix and no dequantized weights.
    """

    if not experts:
        return []
    projection_count = len(experts[0])
    if projection_count not in (2, 3) or any(
        len(expert) != projection_count for expert in experts
    ):
        raise ValueError("inconsistent packed expert projection count")

    rows = [
        values
        for projection in range(projection_count)
        for values in (
            [expert[projection].raw.data_ptr() for expert in experts],
            [expert[projection].cb.data_ptr() for expert in experts],
            [int(expert[projection].blocks) for expert in experts],
            [int(expert[projection].dim) for expert in experts],
            [int(expert[projection].dtype_tag) for expert in experts],
        )
    ]
    if projection_count == 3:
        descriptors = [
            [
                packed_tile_descriptor(
                    bits=int(expert[projection].bits),
                    blocks=int(expert[projection].blocks),
                )[field]
                for expert in experts
            ]
            for projection in range(projection_count)
            for field in range(TILE_ROWS_PER_PROJECTION)
        ]
        rows.extend(descriptors)
    return rows


def has_static_tile_view(metadata_rows: int) -> bool:
    return int(metadata_rows) == runtime_metadata_row_count(3)

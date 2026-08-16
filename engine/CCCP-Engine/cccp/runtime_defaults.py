"""Device-capability defaults shared by every CLI entry point."""

from __future__ import annotations

import os
import struct
from pathlib import Path


CPU_OPERATOR_DEFAULTS = {
    "CCCP_CPU_FUSED": "1",
    "CCCP_CPU_PACKED_SINGLE_TEAM": "1",
    "CCCP_CPU_PACKED_DIRECT_ROWS8": "1",
    "CCCP_CPU_PACKED_FUSED_GATE_UP": "1",
    "CCCP_CPU_PACKED_FUSED_DOWN_REDUCE": "1",
    # llama.cpp-style runtime repack: keep the exact p8..p16 byte count but
    # arrange eight output rows as one CPU traversal tile. This is an
    # in-memory execution view and never writes a derived model.
    "CCCP_CPU_PACKED_LAYOUT": "tile8",
    "CCCP_CPU_BLOCK_FP8_BF16": "1",
    "CCCP_CPU_BLOCK_FP8_ROWS8": "0",
    "CCCP_FULL_RESIDENT": "1",
    "CCCP_PREFETCH": "0",
    "OMP_PROC_BIND": "true",
    "OMP_PLACES": "cores",
}


def _parse_cache_size(value: str) -> int:
    text = value.strip().upper()
    multiplier = 1
    if text.endswith("K"):
        multiplier = 1024
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1024**2
        text = text[:-1]
    return int(text) * multiplier


def _detect_windows_cache_bytes(level: int) -> int | None:
    """Read one Windows data/unified cache instance from the native topology.

    ``Win32_Processor.L2CacheSize`` reports the sum on hybrid processors, which
    is not useful to the per-team packed scheduler.  The kernel API exposes one
    ``CACHE_RELATIONSHIP`` record per real cache.  Use the largest instance at
    the requested level so a P-core cache or shared LLC is not confused with a
    machine-wide sum.
    """

    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        query = kernel32.GetLogicalProcessorInformationEx
        query.argtypes = [
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.DWORD),
        ]
        query.restype = wintypes.BOOL
        required = wintypes.DWORD(0)
        # RelationCache = 2. The first call intentionally obtains the size.
        query(2, None, ctypes.byref(required))
        if required.value <= 0:
            return None
        buffer = ctypes.create_string_buffer(required.value)
        if not query(2, buffer, ctypes.byref(required)):
            return None
        raw = memoryview(buffer.raw)[: required.value]
        offset = 0
        sizes: list[int] = []
        while offset + 20 <= len(raw):
            relationship, record_size = struct.unpack_from("<II", raw, offset)
            if record_size < 20 or offset + record_size > len(raw):
                break
            if relationship == 2:
                cache_level = int(raw[offset + 8])
                cache_size = struct.unpack_from("<I", raw, offset + 12)[0]
                cache_type = struct.unpack_from("<I", raw, offset + 16)[0]
                # CacheUnified=0, CacheData=1. Instruction/trace caches do not
                # hold VQ codebooks and therefore must not influence tiling.
                if cache_level == int(level) and cache_type in {0, 1}:
                    sizes.append(int(cache_size))
            offset += int(record_size)
        return max(sizes) if sizes else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def detect_cpu_cache_bytes(level: int, fallback: int) -> int:
    """Return one CPU cache-instance size without assuming a processor name."""

    windows_value = _detect_windows_cache_bytes(level)
    if windows_value is not None:
        return windows_value

    root = Path("/sys/devices/system/cpu/cpu0/cache")
    try:
        for entry in root.glob("index*"):
            if int((entry / "level").read_text().strip()) != level:
                continue
            cache_type = (entry / "type").read_text().strip().lower()
            if cache_type not in {"unified", "data"}:
                continue
            return _parse_cache_size((entry / "size").read_text())
    except (OSError, ValueError):
        pass
    return int(fallback)


def configure_cpu_operator_defaults(
    *,
    cpu_compile: str | None = None,
) -> None:
    """Enable the public CPU operator stack without overriding user choices."""

    for key, value in CPU_OPERATOR_DEFAULTS.items():
        os.environ.setdefault(key, value)
    if (
        os.name == "nt"
        and os.environ.get("CCCP_CPU_PCORE_AFFINITY", "1").strip().lower()
        not in {"0", "false", "off", "none"}
    ):
        # Windows vcomp resolves OMP_PLACES before the later process-affinity
        # call and aborts when a hybrid CPU's E cores are subsequently outside
        # that mask. Process affinity is the single placement authority here.
        os.environ["OMP_PROC_BIND"] = "false"
        os.environ.pop("OMP_PLACES", None)
    # The native packed scheduler publishes these values for cache-aware
    # tiling and benchmark audit; it never assumes a specific processor name.
    # Users can override them when a VM or container exposes incomplete
    # topology.  Defaults match a conservative modern server core/socket.
    l2_bytes = detect_cpu_cache_bytes(2, 2 * 1024**2)
    llc_bytes = detect_cpu_cache_bytes(3, 32 * 1024**2)
    os.environ.setdefault("CCCP_CPU_L2_BYTES", str(l2_bytes))
    os.environ.setdefault("CCCP_CPU_LLC_BYTES", str(llc_bytes))
    try:
        import psutil

        physical_cores = psutil.cpu_count(logical=False) or (os.cpu_count() or 1)
    except ImportError:
        physical_cores = os.cpu_count() or 1
    # Four adjacent row tiles won on the validated 96-core server by keeping
    # one expert's codebooks hot across consecutive work. On the bundled
    # 8-thread client path, one tile was faster and leaves enough independent
    # tasks for every worker. Select between those measured schedules from
    # topology while preserving an explicit deployment override.
    l2_task_tiles = 4 if physical_cores >= 32 and l2_bytes >= 1024**2 else 1
    os.environ.setdefault("CCCP_CPU_L2_TASK_TILES", str(l2_task_tiles))
    os.environ.setdefault("CCCP_CPU_COMPILE", cpu_compile or "auto")
    if "CCCP_PREFILL_MOE_BATCH" not in os.environ:
        try:
            import psutil

            available_gib = psutil.virtual_memory().available / 2**30
        except ImportError:
            available_gib = 8.0
        # Larger batches reuse the same exact routed experts across more rows
        # and sharply reduce repeated mapped-file loads.  Keep the choice
        # automatic: low-memory machines receive a smaller bounded workspace.
        if available_gib >= 12.0:
            moe_batch = 256
        elif available_gib >= 8.0:
            moe_batch = 128
        elif available_gib >= 5.0:
            moe_batch = 64
        elif available_gib >= 3.0:
            moe_batch = 32
        else:
            moe_batch = 8
        os.environ["CCCP_PREFILL_MOE_BATCH"] = str(moe_batch)


__all__ = [
    "CPU_OPERATOR_DEFAULTS",
    "configure_cpu_operator_defaults",
    "detect_cpu_cache_bytes",
]

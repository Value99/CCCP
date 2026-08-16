"""Optional x86 CPU kernels.

The module is deliberately lazy: CUDA inference never compiles or loads the
CPU extension.  CPU inference attempts one cached JIT build and falls back to
the existing PyTorch implementation if the compiler toolchain is unavailable.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import shutil
import sys
import threading

import torch

from .build_progress import operator_build_progress

_EXT = None
_TRIED = False
_ERR: str | None = None
_SOURCE = "unavailable"
_EXTENSION_NAME = "cccp_cpu_kernels_v193"
_PACKED_MOE_WORKSPACE: tuple[torch.Tensor, torch.Tensor] | None = None
_PACKED_THREE_WORKSPACE: tuple[torch.Tensor, ...] | None = None
_PACKED_MOE_LOCK = threading.Lock()


def configure_windows_performance() -> bool:
    """关闭 Windows EcoQoS，并使用“高于正常”优先级调度到 P 核。"""
    if (
        os.name != "nt"
        or os.environ.get("CCCP_CPU_HIGH_PRIORITY", "1").strip().lower()
        in {"0", "false", "off", "none"}
    ):
        return False
    try:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # ctypes defaults both arguments and return values to 32-bit ``int``.
        # A Windows HANDLE is 64-bit in this build; without explicit
        # prototypes the pseudo-handle becomes a truncated -1 and every
        # performance call fails with ERROR_INVALID_HANDLE.
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        kernel32.SetProcessAffinityMask.argtypes = [
            wintypes.HANDLE,
            ctypes.c_size_t,
        ]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        # ABOVE_NORMAL_PRIORITY_CLASS；避免 HIGH/REALTIME 对桌面响应造成影响。
        priority_ok = bool(kernel32.SetPriorityClass(process, 0x00008000))

        class PowerThrottlingState(ctypes.Structure):
            _fields_ = [
                ("Version", ctypes.c_ulong),
                ("ControlMask", ctypes.c_ulong),
                ("StateMask", ctypes.c_ulong),
            ]

        kernel32.SetProcessInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        memory_ok = False
        working_set_ok = False
        if os.environ.get("CCCP_CPU_MEMORY_PRIORITY", "1") != "0":
            memory_priority = wintypes.DWORD(5)  # MEMORY_PRIORITY_NORMAL
            memory_ok = bool(
                kernel32.SetProcessInformation(
                    process,
                    0,  # ProcessMemoryPriority
                    ctypes.byref(memory_priority),
                    ctypes.sizeof(memory_priority),
                )
            )
            try:
                import psutil

                total = int(psutil.virtual_memory().total)
                requested = os.environ.get(
                    "CCCP_CPU_WORKING_SET_GB", "auto"
                ).strip().lower()
                if requested not in {"0", "false", "off", "none"}:
                    target = (
                        int(float(requested) * 2**30)
                        if requested not in {"", "auto"}
                        else int(total * 0.80)
                    )
                    reserve = min(4 * 2**30, max(1 * 2**30, total // 5))
                    minimum = max(128 * 2**20, min(target, total - reserve))
                    maximum = max(minimum, total - min(2 * 2**30, reserve))
                    kernel32.SetProcessWorkingSetSizeEx.argtypes = [
                        wintypes.HANDLE,
                        ctypes.c_size_t,
                        ctypes.c_size_t,
                        wintypes.DWORD,
                    ]
                    kernel32.SetProcessWorkingSetSizeEx.restype = wintypes.BOOL
                    # This is a scheduler residency preference, not locked
                    # memory. Windows can still reclaim pages under pressure.
                    working_set_ok = bool(
                        kernel32.SetProcessWorkingSetSizeEx(
                            process,
                            minimum,
                            maximum,
                            0x1 | 0x4,  # enable hard min/max working set
                        )
                    )
            except (ImportError, AttributeError, OSError, ValueError):
                working_set_ok = False
        state = PowerThrottlingState(1, 1, 0)
        power_ok = bool(
            kernel32.SetProcessInformation(
                process, 4, ctypes.byref(state), ctypes.sizeof(state)
            )
        )
        affinity_ok = False
        if os.environ.get("CCCP_CPU_PCORE_AFFINITY", "1") != "0":
            try:
                import psutil

                physical = psutil.cpu_count(logical=False) or 0
                logical = psutil.cpu_count(logical=True) or 0
                p_cores = logical - physical
                if (
                    "intel64" in os.environ.get("PROCESSOR_IDENTIFIER", "").lower()
                    and 0 < p_cores < physical
                    and p_cores * 2 < 64
                ):
                    # Windows 在当前 Intel 混合架构上先编号 P 核的两个 SMT
                    # 逻辑处理器，再编号 E 核；限制到 P 核避免 vcomp 同步拖尾。
                    mask = (1 << (p_cores * 2)) - 1
                    affinity_ok = bool(kernel32.SetProcessAffinityMask(process, mask))
                    if affinity_ok:
                        # vcomp expands OMP_PLACES against the machine-wide
                        # topology, including E cores excluded by this mask,
                        # then aborts with OpenMP error 135. The process mask
                        # already supplies placement, so do not bind again.
                        os.environ["OMP_PROC_BIND"] = "false"
                        os.environ.pop("OMP_PLACES", None)
            except (ImportError, AttributeError, OSError, ValueError):
                affinity_ok = False
        return (
            priority_ok
            or power_ok
            or memory_ok
            or working_set_ok
            or affinity_ok
        )
    except (AttributeError, OSError, ValueError):
        return False


def _load_prebuilt():
    """优先加载发行包内与当前 Python ABI 匹配的离线算子。"""
    native_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native")
    if not os.path.isdir(native_dir):
        return None
    candidates = sorted(
        os.path.join(native_dir, name)
        for name in os.listdir(native_dir)
        if name.startswith(_EXTENSION_NAME) and name.endswith((".pyd", ".so"))
    )
    for path in candidates:
        try:
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(native_dir)
            spec = importlib.util.spec_from_file_location(_EXTENSION_NAME, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[_EXTENSION_NAME] = module
            spec.loader.exec_module(module)
            return module
        except Exception:
            sys.modules.pop(_EXTENSION_NAME, None)
    return None


def configure_cpu_threads() -> int:
    """为大核数双路 CPU 选择物理核与 SMT 之间的低延迟甜点。"""
    configure_windows_performance()
    raw = os.environ.get("CCCP_CPU_THREADS", "auto").strip().lower()
    if raw in ("0", "false", "off", "none"):
        return torch.get_num_threads()
    logical = os.cpu_count() or torch.get_num_threads()
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or logical
    except ImportError:
        physical = logical
    if raw == "physical":
        target = physical
    elif raw not in ("", "auto"):
        target = max(1, int(raw))
    else:
        if physical >= 32 and logical >= 2 * physical:
            # Large homogeneous SMT/NUMA systems are commonly limited by
            # memory bandwidth and cross-node scheduling before every core is
            # useful.  Keep the measured low-latency candidate; do not infer
            # that a larger worker count is universally faster.
            target = max(1, physical * 3 // 4)
        elif (
            os.name == "nt"
            and "intel64" in os.environ.get("PROCESSOR_IDENTIFIER", "").lower()
            and physical >= 8
            and physical < logical < 2 * physical
        ):
            # Intel 混合架构：logical-physical 等于带 SMT 的 P 核数。
            # 使用全部 P 核并增加两个 SMT worker；13900H 的干净整模扫描
            # 测得 8 线程 1.734 token/s，而 6/12 线程及混入 E 核都更慢。
            # 两个 P 核不安排第二 worker，仍给系统与 UI 留出响应空间。
            p_cores = logical - physical
            target = min(p_cores * 2, p_cores + 2)
        else:
            target = min(logical, physical)
    torch.set_num_threads(target)
    return torch.get_num_threads()


def configure_numa_interleave() -> bool:
    """让双路 Linux CPU 的后续大块分配均匀落在所有 NUMA 节点。"""
    mode = os.environ.get("CCCP_CPU_NUMA", "auto").strip().lower()
    compile_mode = os.environ.get("CCCP_CPU_COMPILE", "auto").strip().lower()
    if (
        sys.platform != "linux"
        or mode in ("0", "false", "off", "none")
        or mode in ("local", "shard", "sharded")
        or (mode == "auto" and compile_mode == "q4")
    ):
        return False
    try:
        library = ctypes.CDLL("libnuma.so.1", use_errno=True)
        library.numa_available.restype = ctypes.c_int
        library.numa_num_configured_nodes.restype = ctypes.c_int
        library.numa_set_interleave_mask.argtypes = [ctypes.c_void_p]
        if (
            library.numa_available() < 0
            or library.numa_num_configured_nodes() < 2
        ):
            return False
        all_nodes = ctypes.c_void_p.in_dll(
            library, "numa_all_nodes_ptr"
        )
        ctypes.set_errno(0)
        library.numa_set_interleave_mask(all_nodes)
        return ctypes.get_errno() == 0
    except (OSError, ValueError):
        return False


def _ensure_ninja_on_path() -> None:
    if shutil.which("ninja") is not None:
        return
    try:
        import ninja
    except ImportError:
        return
    bin_dir = getattr(ninja, "BIN_DIR", None)
    if not bin_dir:
        return
    executable = "ninja.exe" if os.name == "nt" else "ninja"
    if os.path.isfile(os.path.join(bin_dir, executable)):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _configure_bundled_windows_toolchain() -> bool:
    """Expose the portable MSVC/Windows SDK bundled with the launcher.

    The release normally loads its ABI-matched prebuilt ``.pyd``.  This is the
    offline recovery path for a missing/incompatible operator: torch's JIT
    builder can compile from ``csrc`` without Visual Studio being installed.
    Existing developer-shell settings are deliberately left untouched.
    """
    if os.name != "nt":
        return False
    if all(shutil.which(tool) is not None for tool in ("cl.exe", "lib.exe", "link.exe")):
        return False

    repo_root = os.path.realpath(
        os.environ.get("CCCP_LAUNCHER_ROOT")
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    msvc_tools_root = os.path.join(
        repo_root, "toolchain", "portable", "Contents", "VC", "Tools", "MSVC"
    )
    sdk_root = os.path.join(
        repo_root, "toolchain", "winsdk-portable", "Windows Kits", "10"
    )

    try:
        msvc_versions = sorted(
            name
            for name in os.listdir(msvc_tools_root)
            if os.path.isdir(os.path.join(msvc_tools_root, name))
        )
        sdk_versions = sorted(
            name
            for name in os.listdir(os.path.join(sdk_root, "Include"))
            if os.path.isdir(os.path.join(sdk_root, "Include", name))
            and name[:1].isdigit()
        )
    except OSError:
        return False
    if not msvc_versions or not sdk_versions:
        return False

    msvc_root = os.path.join(msvc_tools_root, msvc_versions[-1])
    sdk_version = sdk_versions[-1]
    compiler_bin = os.path.join(msvc_root, "bin", "Hostx64", "x64")
    sdk_bin = os.path.join(sdk_root, "bin", sdk_version, "x64")
    if not os.path.isfile(os.path.join(compiler_bin, "cl.exe")):
        return False

    def prepend_env(name: str, paths: list[str]) -> None:
        valid = [path for path in paths if os.path.isdir(path)]
        previous = os.environ.get(name, "")
        if previous:
            valid.append(previous)
        os.environ[name] = os.pathsep.join(valid)

    prepend_env("PATH", [compiler_bin, sdk_bin])
    prepend_env(
        "INCLUDE",
        [
            os.path.join(msvc_root, "include"),
            os.path.join(sdk_root, "Include", sdk_version, "ucrt"),
            os.path.join(sdk_root, "Include", sdk_version, "shared"),
            os.path.join(sdk_root, "Include", sdk_version, "um"),
            os.path.join(sdk_root, "Include", sdk_version, "winrt"),
            os.path.join(sdk_root, "Include", sdk_version, "cppwinrt"),
        ],
    )
    prepend_env(
        "LIB",
        [
            os.path.join(msvc_root, "lib", "x64"),
            os.path.join(sdk_root, "Lib", sdk_version, "ucrt", "x64"),
            os.path.join(sdk_root, "Lib", sdk_version, "um", "x64"),
        ],
    )
    vc_root = os.path.realpath(os.path.join(msvc_tools_root, "..", ".."))
    os.environ.setdefault("DISTUTILS_USE_SDK", "1")
    os.environ.setdefault("MSSdk", "1")
    os.environ.setdefault("VisualStudioVersion", "17.0")
    os.environ.setdefault("VCINSTALLDIR", vc_root + os.sep)
    os.environ.setdefault("VCToolsInstallDir", msvc_root + os.sep)
    os.environ.setdefault("WindowsSdkDir", sdk_root + os.sep)
    os.environ.setdefault("WindowsSDKVersion", sdk_version + os.sep)
    return all(shutil.which(tool) is not None for tool in ("cl.exe", "lib.exe", "link.exe"))


def _build(verbose: bool = False):
    global _EXT, _TRIED, _ERR, _SOURCE
    if _EXT is not None or _TRIED:
        return _EXT
    _TRIED = True
    if os.environ.get("CCCP_CPU_FUSED", "1") == "0":
        _ERR = "CCCP_CPU_FUSED=0"
        return None
    try:
        _EXT = _load_prebuilt()
        if _EXT is not None:
            _ERR = None
            _SOURCE = "bundled-prebuilt"
            return _EXT
        if os.environ.get("CCCP_CPU_AUTOBUILD", "1").strip().lower() in {
            "0", "false", "off", "none"
        }:
            _ERR = "未找到兼容的预编译算子，且 CCCP_CPU_AUTOBUILD=0"
            return None
        _configure_bundled_windows_toolchain()
        _ensure_ninja_on_path()
        from torch.utils.cpp_extension import load

        source = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "csrc", "cpu_vq.cpp"
        )
        compile_flags = (
            [
                "/O2", "/Ot", "/Oi", "/GL", "/Gw", "/Gy", "/fp:fast",
                "/favor:INTEL64", "/openmp", "/arch:AVX2", "/DNDEBUG",
            ]
            if os.name == "nt"
            else ["-O3", "-march=native", "-fopenmp"]
        )
        link_flags = (
            ["/LTCG", "/OPT:REF", "/OPT:ICF"]
            if os.name == "nt"
            else ["-fopenmp"]
        )
        with operator_build_progress("CPU") as build_progress:
            _EXT = load(
                name=_EXTENSION_NAME,
                sources=[source],
                extra_cflags=compile_flags,
                extra_ldflags=link_flags,
                # First-run users should see the real compiler/Ninja output in
                # the launcher's terminal as well as the periodic heartbeat.
                verbose=verbose or build_progress.enabled,
            )
        _ERR = None
        _SOURCE = "local-jit"
    except Exception as exc:  # a missing compiler must not break inference
        _EXT = None
        _ERR = f"{type(exc).__name__}: {exc}"
        _SOURCE = "fallback"
    return _EXT


def vq_gemv_cpu(
    x_rows: torch.Tensor,
    indices: torch.Tensor,
    codebooks: torch.Tensor,
) -> torch.Tensor | None:
    if (
        x_rows.is_cuda
        or indices.is_cuda
        or codebooks.is_cuda
        or indices.dtype not in (torch.uint8, torch.uint16)
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.vq_gemv(
        x_rows,
        indices,
        codebooks,
    )


def vq_repack_block_major_cpu(
    payload: torch.Tensor,
    rows: int,
    blocks: int,
    bits: int,
) -> torch.Tensor | None:
    """Reorder packed VQ indices while preserving their exact bit width."""
    if payload.is_cuda or payload.dtype != torch.uint8:
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.vq_repack_block_major(
        payload.contiguous().reshape(-1),
        int(rows),
        int(blocks),
        int(bits),
    )


def vq_repack_row_tile_cpu(
    payload: torch.Tensor,
    rows: int,
    blocks: int,
    bits: int,
    tile_rows: int = 8,
) -> torch.Tensor | None:
    """Build a compact row-tile traversal without expanding indices."""
    if payload.is_cuda or payload.dtype != torch.uint8:
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.vq_repack_row_tile(
        payload.contiguous().reshape(-1),
        int(rows),
        int(blocks),
        int(bits),
        int(tile_rows),
    )


def vq_compile_u16_row_tile_cpu(
    payload: torch.Tensor,
    rows: int,
    blocks: int,
    bits: int,
    tile_rows: int = 8,
) -> torch.Tensor | None:
    """Compile packed indices directly into their final CPU execution arena."""
    if payload.is_cuda or payload.dtype != torch.uint8:
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.vq_compile_u16_row_tile(
        payload.contiguous().reshape(-1),
        int(rows),
        int(blocks),
        int(bits),
        int(tile_rows),
    )


def vq_gemv_list_cpu(
    x_rows: torch.Tensor,
    indices: list[torch.Tensor],
    codebook: torch.Tensor,
) -> torch.Tensor | None:
    if (
        x_rows.is_cuda
        or codebook.is_cuda
        or not indices
        or any(
            index.is_cuda
            or index.dtype not in (torch.uint8, torch.uint16)
            or index.dtype != indices[0].dtype
            for index in indices
        )
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.vq_gemv_list(
        x_rows,
        indices,
        codebook,
    )


def block_fp8_gemv_cpu(
    value: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    cols: int,
    block_size: int,
    output: torch.Tensor | None = None,
    *,
    rows: int | None = None,
) -> torch.Tensor | None:
    """Direct compact E4M3FN block-scaled GEMV for one CPU token."""
    if (
        value.is_cuda
        or value.ndim != 2
        or value.shape != (1, int(cols))
        or value.dtype not in (torch.float32, torch.bfloat16)
        or weights.is_cuda
        or weights.dtype != torch.uint8
        or weights.ndim not in (2, 5)
        or scales.is_cuda
        or scales.dtype != torch.float32
        or int(block_size) != 128
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    if rows is None:
        rows = (
            int(weights.shape[0])
            if weights.ndim == 2
            else int(weights.shape[0]) * int(block_size)
        )
    if output is None:
        output = torch.empty(int(rows), dtype=value.dtype)
    return extension.block_fp8_gemv(
        value,
        weights,
        scales,
        int(rows),
        int(cols),
        int(block_size),
        output,
    )


def block_fp8_gemm_cpu(
    value: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    cols: int,
    block_size: int,
    output: torch.Tensor | None = None,
    *,
    rows: int | None = None,
) -> torch.Tensor | None:
    """Scan compact block-FP8 once for 2..16 candidate tokens."""
    if (
        value.is_cuda
        or value.ndim != 2
        or not 2 <= value.shape[0] <= 16
        or value.shape[1] != int(cols)
        or value.dtype not in (torch.float32, torch.bfloat16)
        or weights.is_cuda
        or weights.dtype != torch.uint8
        or weights.ndim not in (2, 5)
        or scales.is_cuda
        or scales.dtype != torch.float32
        or int(block_size) != 128
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    if rows is None:
        rows = (
            int(weights.shape[0])
            if weights.ndim == 2
            else int(weights.shape[0]) * int(block_size)
        )
    if output is None:
        output = torch.empty(
            int(value.shape[0]), int(rows), dtype=value.dtype
        )
    return extension.block_fp8_gemm(
        value,
        weights,
        scales,
        int(rows),
        int(cols),
        int(block_size),
        output,
    )


def block_fp8_to_block_major_cpu(
    weights: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor | None:
    """Pack compact FP8 bytes into the public 32x128 CPU tile layout."""
    if (
        weights.is_cuda
        or weights.dtype != torch.uint8
        or weights.ndim != 2
        or not weights.is_contiguous()
        or int(block_size) != 128
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.block_fp8_to_block_major(weights, int(block_size))


def block_fp8_compile_q4_0_cpu(
    weights: torch.Tensor,
    scales: torch.Tensor,
    rows: int,
    cols: int,
    block_size: int = 128,
) -> torch.Tensor | None:
    """Compile native block-FP8 into an in-memory Q4 block-dot image."""
    if (
        weights.is_cuda
        or weights.dtype != torch.uint8
        or weights.ndim != 2
        or not weights.is_contiguous()
        or scales.is_cuda
        or scales.dtype != torch.float32
        or scales.ndim != 2
        or int(cols) % 32
        or int(block_size) != 128
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.block_fp8_compile_q4_0(
        weights, scales, int(rows), int(cols), int(block_size)
    )


def vq_compile_q4_0_cpu(
    payload: torch.Tensor,
    codebook: torch.Tensor,
    rows: int,
    blocks: int,
    bits: int,
) -> torch.Tensor | None:
    """Compile one compact VQ matrix into a linear Q4 block-dot image."""
    if (
        payload.is_cuda
        or payload.dtype != torch.uint8
        or codebook.is_cuda
        or codebook.ndim != 2
        or int(blocks) * int(codebook.shape[1]) % 32
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.vq_compile_q4_0(
        payload.contiguous().reshape(-1),
        codebook.float().contiguous(),
        int(rows),
        int(blocks),
        int(bits),
    )


def q4_0_gemv_cpu(
    value: torch.Tensor,
    weights: torch.Tensor,
    rows: int,
    cols: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Apply a load-time Q4 image using one shared Q8 activation row."""
    if (
        value.is_cuda
        or value.shape != (1, int(cols))
        or value.dtype != torch.float32
        or weights.is_cuda
        or weights.dtype != torch.uint8
        or weights.ndim != 1
        or int(cols) % 32
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    if output is None:
        output = torch.empty(1, int(rows), dtype=torch.float32)
    return extension.q4_0_gemv(
        value.contiguous(), weights.contiguous(), int(rows), int(cols), output
    )


def q4_0_gemm_cpu(
    value: torch.Tensor,
    weights: torch.Tensor,
    rows: int,
    cols: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Apply a Q4 image to 2..64 rows while scanning each weight tile once."""
    if (
        value.is_cuda
        or value.ndim != 2
        or not 2 <= value.shape[0] <= 64
        or value.shape[1] != int(cols)
        or value.dtype != torch.float32
        or weights.is_cuda
        or weights.dtype != torch.uint8
        or weights.ndim != 1
        or int(cols) % 32
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    if output is None:
        output = torch.empty(
            int(value.shape[0]), int(rows), dtype=torch.float32
        )
    return extension.q4_0_gemm(
        value.contiguous(), weights.contiguous(), int(rows), int(cols), output
    )


def bf16_grouped_gemv_cpu(
    value: torch.Tensor,
    weight_ptrs: torch.Tensor,
    row_offsets: torch.Tensor,
    total_rows: int,
    cols: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Apply several BF16 matrices to one CPU token in one native call."""
    if (
        value.is_cuda
        or value.ndim != 2
        or value.shape != (1, int(cols))
        or value.dtype not in (torch.float32, torch.bfloat16)
        or weight_ptrs.is_cuda
        or weight_ptrs.dtype != torch.int64
        or row_offsets.is_cuda
        or row_offsets.dtype != torch.int32
        or int(total_rows) <= 0
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    if output is None:
        output = torch.empty(
            1, int(total_rows), dtype=torch.bfloat16
        )
    return extension.bf16_grouped_gemv(
        value.contiguous(),
        weight_ptrs.contiguous(),
        row_offsets.contiguous(),
        int(total_rows),
        int(cols),
        output,
    )


def block_fp8_grouped_gemv_cpu(
    value: torch.Tensor,
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    row_offsets: torch.Tensor,
    total_rows: int,
    cols: int,
    block_size: int,
    output: torch.Tensor | None = None,
    *,
    block_major: bool = False,
) -> torch.Tensor | None:
    """Evaluate a logical row-concatenation of compact CPU FP8 weights."""
    if (
        value.is_cuda
        or value.ndim != 2
        or value.shape != (1, int(cols))
        or value.dtype not in (torch.float32, torch.bfloat16)
        or weight_ptrs.is_cuda
        or weight_ptrs.dtype != torch.int64
        or scale_ptrs.is_cuda
        or scale_ptrs.dtype != torch.int64
        or row_offsets.is_cuda
        or row_offsets.dtype != torch.int32
        or int(block_size) != 128
        or int(total_rows) <= 0
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    if output is None:
        output = torch.empty(int(total_rows), dtype=value.dtype)
    return extension.block_fp8_grouped_gemv(
        value,
        weight_ptrs.contiguous(),
        scale_ptrs.contiguous(),
        row_offsets.contiguous(),
        int(total_rows),
        int(cols),
        int(block_size),
        bool(block_major),
        output,
    )


def block_fp8_grouped_rows_gemv_cpu(
    value: torch.Tensor,
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    row_offsets: torch.Tensor,
    total_rows: int,
    cols: int,
    block_size: int,
    output: torch.Tensor | None = None,
    *,
    block_major: bool = False,
) -> torch.Tensor | None:
    """One compact FP8 projection per matching input row."""
    groups = int(weight_ptrs.numel())
    if (
        value.is_cuda
        or value.ndim != 2
        or value.shape != (groups, int(cols))
        or value.dtype not in (torch.float32, torch.bfloat16)
        or weight_ptrs.is_cuda
        or weight_ptrs.dtype != torch.int64
        or scale_ptrs.is_cuda
        or scale_ptrs.dtype != torch.int64
        or row_offsets.is_cuda
        or row_offsets.dtype != torch.int32
        or int(block_size) != 128
        or int(total_rows) <= 0
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    if output is None:
        output = torch.empty(int(total_rows), dtype=value.dtype)
    return extension.block_fp8_grouped_rows_gemv(
        value.contiguous(),
        weight_ptrs.contiguous(),
        scale_ptrs.contiguous(),
        row_offsets.contiguous(),
        int(total_rows),
        int(cols),
        int(block_size),
        bool(block_major),
        output,
    )


def block_fp8_grouped_gemm_cpu(
    value: torch.Tensor,
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    row_offsets: torch.Tensor,
    total_rows: int,
    cols: int,
    block_size: int,
    output: torch.Tensor | None = None,
    *,
    block_major: bool = False,
) -> torch.Tensor | None:
    """Batch a logical compact FP8 row-concatenation in one CPU call."""
    if (
        value.is_cuda
        or value.ndim != 2
        or not 2 <= value.shape[0] <= 16
        or value.shape[1] != int(cols)
        or value.dtype not in (torch.float32, torch.bfloat16)
        or any(item.is_cuda for item in (
            weight_ptrs, scale_ptrs, row_offsets
        ))
        or weight_ptrs.dtype != torch.int64
        or scale_ptrs.dtype != torch.int64
        or row_offsets.dtype != torch.int32
        or int(block_size) != 128
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    if output is None:
        output = torch.empty(
            int(value.shape[0]), int(total_rows), dtype=value.dtype
        )
    return extension.block_fp8_grouped_gemm(
        value,
        weight_ptrs,
        scale_ptrs,
        row_offsets,
        int(total_rows),
        int(cols),
        int(block_size),
        bool(block_major),
        output,
    )


def vq_gemv_packed_list_cpu(
    x_rows: torch.Tensor,
    payloads: list[torch.Tensor],
    codebook: torch.Tensor,
    rows: int,
    blocks: int,
    bits: int,
    *,
    allow_direct: bool = False,
) -> torch.Tensor | None:
    """Directly evaluate byte-packed VQ indices without a uint16 copy."""
    if (
        x_rows.is_cuda
        or codebook.is_cuda
        or not payloads
        or not 8 <= bits <= 16
        or any(
            payload.is_cuda or payload.dtype != torch.uint8
            for payload in payloads
        )
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.vq_gemv_packed_list(
        x_rows,
        payloads,
        codebook,
        int(rows),
        int(blocks),
        int(bits),
        bool(allow_direct),
    )


def vq_dequant_packed_cpu(
    payload: torch.Tensor,
    codebook: torch.Tensor,
    rows: int,
    blocks: int,
    bits: int,
    layout: str = "row-major",
) -> torch.Tensor | None:
    """Expand one row-major packed VQ matrix for a grouped CPU GEMM.

    Route-scan prefill groups all tokens that selected the same expert.  The
    expert is expanded once into a short-lived float32 matrix and immediately
    consumed by a multi-row GEMM; no expanded expert survives the group.
    """
    if (
        payload.is_cuda
        or payload.dtype != torch.uint8
        or codebook.is_cuda
        or codebook.dtype != torch.float32
        or codebook.ndim != 2
        or not 8 <= int(bits) <= 16
    ):
        return None
    extension = _build()
    if extension is None or not hasattr(extension, "vq_dequant_packed"):
        return None
    layout_id = {
        "row-major": 0,
        "block-major": 1,
        "row-tile-8": 2,
        "u16-row-tile-8": 2,
    }.get(str(layout))
    if layout_id is None:
        return None
    return extension.vq_dequant_packed(
        payload.contiguous().reshape(-1),
        codebook.contiguous(),
        int(rows),
        int(blocks),
        int(bits),
        int(layout_id),
    )


def q4_0_dequant_cpu(
    payload: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor | None:
    """Expand one runtime-only Q4 image for multi-token GEMM."""
    if (
        payload.is_cuda
        or payload.dtype != torch.uint8
        or payload.ndim != 1
        or int(cols) % 32
    ):
        return None
    extension = _build()
    if extension is None or not hasattr(extension, "q4_0_dequant"):
        return None
    return extension.q4_0_dequant(
        payload.contiguous(), int(rows), int(cols)
    )


def _shared_projection_spec(weights: list[object]):
    """Return common layer metadata or ``None`` for an invalid mixed list."""
    first = weights[0]
    if any(
        int(weight.rows) != int(first.rows)
        or int(weight.blocks) != int(first.blocks)
        or int(weight.bits) != int(first.bits)
        or int(weight.dim) != int(first.dim)
        # ``bits`` describes the archive index width and deliberately stays
        # unchanged when a hot expert is compiled to a Q4 execution image.
        # A partial resident cache may therefore place q4_0 and row-tile VQ
        # payloads in the same projection.  They must be dispatched as
        # separate native groups; otherwise the first payload's byte layout
        # is incorrectly applied to every expert in the list.
        or str(weight.layout) != str(first.layout)
        or int(weight.raw.numel()) != int(first.raw.numel())
        or tuple(weight.cb.shape) != tuple(first.cb.shape)
        or weight.cb.data_ptr() != first.cb.data_ptr()
        for weight in weights[1:]
    ):
        return None
    return first


def _shared_projection(
    weights: list[object],
    x_rows: torch.Tensor,
    *,
    allow_direct: bool = False,
) -> torch.Tensor | None:
    """Run one projection whose selected experts share layer metadata."""
    first = _shared_projection_spec(weights)
    if first is None:
        return None
    return vq_gemv_packed_list_cpu(
        x_rows,
        [weight.raw for weight in weights],
        first.cb.float().contiguous(),
        int(first.rows),
        int(first.blocks),
        int(first.bits),
        allow_direct=allow_direct,
    )


def _grouped_projection(
    weights: list[object],
    x_rows: torch.Tensor,
    output: torch.Tensor,
    *,
    allow_direct: bool = False,
) -> torch.Tensor | None:
    """Evaluate one projection while preserving per-group codebooks.

    Most projection archives share one layer codebook and take the fused
    single-call path.  Multi-codebook layouts group selected experts by the
    exact codebook pointer, invoke the same native packed GEMV for each group,
    and scatter into one persistent Top-K workspace.  Indices stay packed.
    """
    if not weights or output.shape[0] < len(weights):
        return None
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, weight in enumerate(weights):
        key = (
            int(weight.cb.data_ptr()),
            int(weight.rows),
            int(weight.blocks),
            int(weight.bits),
            int(weight.dim),
            int(weight.cb.shape[0]),
            int(weight.cb.shape[1]),
            str(weight.layout),
            int(weight.raw.numel()),
        )
        groups.setdefault(key, []).append(index)
    for positions in groups.values():
        group_weights = [weights[index] for index in positions]
        spec = _shared_projection_spec(group_weights)
        if spec is None:
            return None
        selection = torch.tensor(positions, dtype=torch.long)
        if x_rows.shape[0] == 1:
            inputs = x_rows
        else:
            inputs = x_rows.index_select(0, selection)
        values = vq_gemv_packed_list_cpu(
            inputs,
            [weight.raw for weight in group_weights],
            spec.cb.float().contiguous(),
            int(spec.rows),
            int(spec.blocks),
            int(spec.bits),
            allow_direct=allow_direct,
        )
        if values is None:
            return None
        # A Python list triggers advanced indexing and returns a temporary;
        # copying into it would leave the persistent workspace uninitialized.
        output[:, : int(spec.rows)].index_copy_(0, selection, values)
    return output[: len(weights), : int(weights[0].rows)]


def moe_packed_rows_cpu(
    x_rows: torch.Tensor,
    experts: list[list[tuple[object, ...]]],
    route_weights: torch.Tensor,
    limit: float,
    *,
    activation: str,
    activation_beta: float,
    activation_linear_beta: float | None,
) -> torch.Tensor | None:
    """Evaluate several independently routed CPU rows in packed form.

    ``experts[row][slot]`` is the exact expert selected for that token.  The
    implementation flattens only the temporary route dimension and reuses the
    existing native packed-list projection, so expert IDs, codebooks and
    packed index widths remain unchanged.  Callers bound the row count to keep
    score workspaces inside the configured expert cache budget.
    """
    if (
        x_rows.is_cuda
        or x_rows.ndim != 2
        or x_rows.shape[0] < 2
        or route_weights.is_cuda
        or route_weights.ndim != 2
        or route_weights.shape[0] != x_rows.shape[0]
        or len(experts) != x_rows.shape[0]
    ):
        return None
    rows = int(x_rows.shape[0])
    top_k = int(route_weights.shape[1])
    if top_k <= 0 or top_k > 16 or any(len(row) != top_k for row in experts):
        return None
    flat = [bundle for row in experts for bundle in row]
    if (
        not flat
        or any(
            len(bundle) not in (2, 3)
            or any(not hasattr(weight, "raw") for weight in bundle)
            for bundle in flat
        )
        or any(len(bundle) != len(flat[0]) for bundle in flat)
    ):
        return None
    projection_count = len(flat[0])
    hidden = int(x_rows.shape[1])
    if projection_count == 3:
        gate = [bundle[0] for bundle in flat]
        up = [bundle[1] for bundle in flat]
        down = [bundle[2] for bundle in flat]
        intermediate = int(gate[0].rows)
        invalid_shapes = (
            any(int(weight.rows) != intermediate for weight in gate + up)
            or any(int(weight.rows) != hidden for weight in down)
            or any(int(weight.cols) != hidden for weight in gate + up)
            or any(int(weight.cols) != intermediate for weight in down)
        )
        all_weights = gate + up + down
    else:
        gu = [bundle[0] for bundle in flat]
        down = [bundle[1] for bundle in flat]
        intermediate = int(down[0].cols)
        invalid_shapes = (
            any(int(weight.rows) != 2 * intermediate for weight in gu)
            or any(int(weight.rows) != hidden for weight in down)
            or any(int(weight.cols) != hidden for weight in gu)
            or any(int(weight.cols) != intermediate for weight in down)
        )
        all_weights = gu + down
    if (
        invalid_shapes
        or any(
            str(getattr(weight, "layout", "row-major"))
            not in {
                "row-major", "block-major", "row-tile-8",
                "u16-row-tile-8", "q4_0",
            }
            for weight in all_weights
        )
    ):
        return None

    # Long CPU prefill must be grouped by expert.  Merely flattening
    # rows*TopK into one native call still rereads an expert once per routed
    # token and is effectively batched GEMV.  Here every expert is decoded
    # exactly once, all of its routed tokens run through GEMM together, and
    # the temporary dense matrix is released before the next expert.
    grouped_threshold = max(
        2, int(os.environ.get("CCCP_CPU_GROUPED_DEQUANT_MIN_ROWS", "16"))
    )
    has_q4 = any(
        str(getattr(weight, "layout", "row-major")) == "q4_0"
        for bundle in flat
        for weight in bundle
    )
    if has_q4:
        grouped_threshold = 2
    if projection_count == 2:
        grouped_threshold = 2
    unique_route_experts = len({id(bundle) for bundle in flat})
    route_count = rows * top_k
    # Three-projection VQ dequant+GEMM pays off when prompt rows reuse an
    # expert.  With nearly disjoint routes it expands one dense matrix per
    # token/expert and can be slower than the native packed row kernel.  Keep
    # the grouped path for Q4 and legacy combined-GU archives (where it is
    # already faster), and require at least 2x route reuse for ordinary
    # three-projection VQ.  Long prompts naturally exceed this ratio by a
    # large margin once their routes saturate the model's finite expert set.
    grouped_reuse_ok = (
        has_q4
        or projection_count == 2
        or route_count >= 2 * unique_route_experts
    )
    if rows >= grouped_threshold and grouped_reuse_ok:
        grouped: dict[int, tuple[list[tuple[object, ...]], list[int], list[int]]] = {}
        for row_index, row_experts in enumerate(experts):
            for slot, bundle in enumerate(row_experts):
                key = id(bundle)
                item = grouped.get(key)
                if item is None:
                    item = ([bundle], [], [])
                    grouped[key] = item
                item[1].append(row_index)
                item[2].append(slot)

        output = torch.zeros((rows, hidden), dtype=torch.float32)

        def dense(weight) -> torch.Tensor | None:
            layout = str(getattr(weight, "layout", "row-major"))
            if layout == "q4_0":
                return q4_0_dequant_cpu(
                    weight.raw,
                    int(weight.rows),
                    int(weight.cols),
                )
            expanded = vq_dequant_packed_cpu(
                weight.raw,
                weight.cb.float().contiguous(),
                int(weight.rows),
                int(weight.blocks),
                int(weight.bits),
                layout,
            )
            if expanded is not None:
                return expanded
            try:
                indices = weight.unpack().long()
            except (AttributeError, RuntimeError, ValueError):
                return None
            return weight.cb.index_select(0, indices.reshape(-1)).reshape(
                int(weight.rows), int(weight.cols)
            )

        with torch.inference_mode():
            for bundles, token_positions, slots in grouped.values():
                bundle = bundles[0]
                token_index = torch.tensor(token_positions, dtype=torch.long)
                slot_index = torch.tensor(slots, dtype=torch.long)
                inputs = x_rows.index_select(0, token_index).float().contiguous()

                if projection_count == 3:
                    gate_matrix = dense(bundle[0])
                    if gate_matrix is None:
                        return None
                    gate_values = torch.mm(inputs, gate_matrix.t())
                    del gate_matrix

                    up_matrix = dense(bundle[1])
                    if up_matrix is None:
                        return None
                    up_values = torch.mm(inputs, up_matrix.t())
                    del up_matrix
                    down_weight = bundle[2]
                else:
                    gu_matrix = dense(bundle[0])
                    if gu_matrix is None:
                        return None
                    gu_values = torch.mm(inputs, gu_matrix.t())
                    del gu_matrix
                    gate_values = gu_values[:, :intermediate]
                    up_values = gu_values[:, intermediate:]
                    down_weight = bundle[1]

                if limit != 0.0:
                    gate_values.clamp_max_(float(limit))
                    up_values.clamp_(-float(limit), float(limit))
                normalized = str(activation).strip().lower()
                if normalized == "situ":
                    beta = float(activation_beta)
                    gate_sigmoid = torch.sigmoid(gate_values)
                    gate_values.div_(beta).tanh_().mul_(beta)
                    gate_values.mul_(gate_sigmoid)
                    if (
                        activation_linear_beta is not None
                        and float(activation_linear_beta) > 0.0
                    ):
                        linear_beta = float(activation_linear_beta)
                        up_values.div_(linear_beta).tanh_().mul_(linear_beta)
                    gate_values.mul_(up_values)
                elif normalized in {"silu", "swiglu"}:
                    gate_values.mul_(torch.sigmoid(gate_values)).mul_(up_values)
                else:
                    return None
                del up_values

                down_matrix = dense(down_weight)
                if down_matrix is None:
                    return None
                routed = torch.mm(gate_values, down_matrix.t())
                del down_matrix, gate_values
                selected_weights = route_weights[token_index, slot_index].float()
                routed.mul_(selected_weights.unsqueeze(1))
                output.index_add_(0, token_index, routed)
        return output

    route_inputs = x_rows.float().contiguous().repeat_interleave(top_k, dim=0)
    gate_workspace = torch.empty(route_count, intermediate, dtype=torch.float32)
    up_workspace = torch.empty_like(gate_workspace)
    down_workspace = torch.empty(route_count, hidden, dtype=torch.float32)
    gate_values = _grouped_projection(
        gate, route_inputs, gate_workspace, allow_direct=True
    )
    up_values = _grouped_projection(
        up, route_inputs, up_workspace, allow_direct=True
    )
    if gate_values is None or up_values is None:
        return None
    if limit != 0.0:
        gate_values.clamp_max_(float(limit))
        up_values.clamp_(-float(limit), float(limit))
    normalized = str(activation).strip().lower()
    if normalized == "situ":
        beta = float(activation_beta)
        gate_sigmoid = torch.sigmoid(gate_values)
        gate_values.div_(beta).tanh_().mul_(beta)
        gate_values.mul_(gate_sigmoid)
        if activation_linear_beta is not None and float(activation_linear_beta) > 0.0:
            linear_beta = float(activation_linear_beta)
            up_values.div_(linear_beta).tanh_().mul_(linear_beta)
        gate_values.mul_(up_values)
    elif normalized in {"silu", "swiglu"}:
        gate_values.mul_(torch.sigmoid(gate_values)).mul_(up_values)
    else:
        return None
    down_values = _grouped_projection(
        down, gate_values, down_workspace, allow_direct=True
    )
    if down_values is None:
        return None
    route = route_weights.float().contiguous().view(rows, 1, top_k)
    return torch.bmm(route, down_values.view(rows, top_k, hidden)).squeeze(1)


def moe_packed_topk_cpu(
    x_row: torch.Tensor,
    experts: list[tuple[object, ...]],
    route_weights: torch.Tensor,
    limit: float,
    *,
    activation: str,
    activation_beta: float,
    activation_linear_beta: float | None,
) -> torch.Tensor | None:
    """Run mixed packed Top-K MoE through one native registered invocation.

    The payloads remain p8..p16. Legacy combined Gate+Up experts
    use the native fused entry below.  Three-projection archives are scheduled
    through the same registered call as Gate VQ -> Up VQ -> activation -> Down
    VQ, preserving every projection's own code dimension and codebook.
    """
    global _PACKED_MOE_WORKSPACE, _PACKED_THREE_WORKSPACE
    if (
        x_row.is_cuda
        or x_row.ndim != 2
        or x_row.shape[0] != 1
        or not 0 < len(experts) <= 16
        or route_weights.is_cuda
        or route_weights.numel() != len(experts)
        or len(experts[0]) not in (2, 3)
        or any(
            len(bundle) != len(experts[0])
            or any(not hasattr(weight, "raw") for weight in bundle)
            for bundle in experts
        )
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    x_float = x_row.float().contiguous()
    weights = route_weights.float().contiguous()
    hidden = int(x_float.shape[1])

    if len(experts[0]) == 3:
        gate = [bundle[0] for bundle in experts]
        up = [bundle[1] for bundle in experts]
        dn = [bundle[2] for bundle in experts]
        intermediate = int(gate[0].rows)
        if (
            any(int(weight.rows) != intermediate for weight in up)
            or any(int(weight.rows) != hidden for weight in dn)
            or any(int(weight.cols) != hidden for weight in gate + up)
            or any(int(weight.cols) != intermediate for weight in dn)
        ):
            return None
        if any(
            str(getattr(weight, "layout", "row-major")) != "row-major"
            for weight in gate + up + dn
        ):
            # The compact list operator predates runtime execution layouts
            # and accepts row-major VQ bytes only.  Partial residency can mix
            # compiled q4_0 hot experts with row-tile cold experts in one
            # Top-K.  Reuse the layout-aware mixed native executor for that
            # uncommon fallback instead of reinterpreting either payload.
            mixed = make_packed_three_layer_cpu(
                tuple(experts), force_mixed=True
            )
            if mixed is None:
                return None
            return mixed.forward(
                x_float,
                torch.arange(len(experts), dtype=torch.int64),
                weights,
                float(limit),
                str(activation).strip().lower(),
                float(activation_beta),
                (
                    -1.0
                    if activation_linear_beta is None
                    else float(activation_linear_beta)
                ),
            )
        gate_spec = _shared_projection_spec(gate)
        up_spec = _shared_projection_spec(up)
        down_spec = _shared_projection_spec(dn)
        with _PACKED_MOE_LOCK:
            if (
                gate_spec is not None
                and up_spec is not None
                and down_spec is not None
            ):
                required = (
                    max(
                        int(gate_spec.blocks * gate_spec.cb.shape[0]),
                        int(up_spec.blocks * up_spec.cb.shape[0]),
                    )
                    + len(experts) * intermediate * 2
                    + len(experts)
                    * int(down_spec.blocks * down_spec.cb.shape[0])
                    + len(experts) * hidden
                )
                if (
                    _PACKED_MOE_WORKSPACE is None
                    or _PACKED_MOE_WORKSPACE[0].numel() < required
                    or _PACKED_MOE_WORKSPACE[1].numel() < hidden
                ):
                    _PACKED_MOE_WORKSPACE = (
                        torch.empty(required, dtype=torch.float32),
                        torch.empty(hidden, dtype=torch.float32),
                    )
                workspace, result = _PACKED_MOE_WORKSPACE
                return extension.moe_packed_three_projection(
                    x_float,
                    [weight.raw for weight in gate],
                    gate_spec.cb.float().contiguous(),
                    int(gate_spec.rows),
                    int(gate_spec.blocks),
                    int(gate_spec.bits),
                    [weight.raw for weight in up],
                    up_spec.cb.float().contiguous(),
                    int(up_spec.rows),
                    int(up_spec.blocks),
                    int(up_spec.bits),
                    [weight.raw for weight in dn],
                    down_spec.cb.float().contiguous(),
                    int(down_spec.rows),
                    int(down_spec.blocks),
                    int(down_spec.bits),
                    weights,
                    float(limit),
                    str(activation).strip().lower(),
                    float(activation_beta),
                    (
                        -1.0
                        if activation_linear_beta is None
                        else float(activation_linear_beta)
                    ),
                    workspace,
                    result,
                )

            if (
                _PACKED_MOE_WORKSPACE is None
                or _PACKED_MOE_WORKSPACE[1].numel() < hidden
            ):
                _PACKED_MOE_WORKSPACE = (
                    torch.empty(1, dtype=torch.float32),
                    torch.empty(hidden, dtype=torch.float32),
                )
            result = _PACKED_MOE_WORKSPACE[1]

            # Grouped projection codebooks (for example one codebook per
            # contiguous expert band) cannot use the one-codebook native fast
            # path above.  Retain one public operator call and one persistent
            # workspace while dispatching only the affected projection by
            # exact codebook group.
            required_shape = (len(experts), intermediate)
            down_shape = (len(experts), hidden)
            if (
                _PACKED_THREE_WORKSPACE is None
                or _PACKED_THREE_WORKSPACE[0].shape[0]
                < required_shape[0]
                or _PACKED_THREE_WORKSPACE[0].shape[1]
                < required_shape[1]
                or _PACKED_THREE_WORKSPACE[3].shape[0]
                < down_shape[0]
                or _PACKED_THREE_WORKSPACE[3].shape[1]
                < down_shape[1]
            ):
                _PACKED_THREE_WORKSPACE = (
                    torch.empty(required_shape, dtype=torch.float32),
                    torch.empty(required_shape, dtype=torch.float32),
                    torch.empty(required_shape, dtype=torch.float32),
                    torch.empty(down_shape, dtype=torch.float32),
                )
            gate_workspace, up_workspace, activated_workspace, down_workspace = (
                _PACKED_THREE_WORKSPACE
            )
            gate_values = _grouped_projection(
                gate,
                x_float,
                gate_workspace,
                allow_direct=True,
            )
            up_values = _grouped_projection(
                up,
                x_float,
                up_workspace,
                allow_direct=True,
            )
            if gate_values is None or up_values is None:
                return None
            activated = activated_workspace[
                : len(experts), :intermediate
            ]
            if limit != 0.0:
                gate_values.clamp_max_(float(limit))
                up_values.clamp_(-float(limit), float(limit))
            normalized_activation = str(activation).strip().lower()
            if normalized_activation == "situ":
                activated.copy_(gate_values)
                activated.div_(float(activation_beta)).tanh_()
                activated.mul_(float(activation_beta))
                activated.mul_(gate_values.sigmoid())
                if (
                    activation_linear_beta is not None
                    and float(activation_linear_beta) > 0.0
                ):
                    up_values.div_(float(activation_linear_beta)).tanh_()
                    up_values.mul_(float(activation_linear_beta))
                activated.mul_(up_values)
            elif normalized_activation in {"silu", "swiglu"}:
                activated.copy_(gate_values)
                activated.mul_(gate_values.sigmoid()).mul_(up_values)
            else:
                return None
            down_result = _grouped_projection(
                dn,
                activated,
                down_workspace,
                allow_direct=True,
            )
            if down_result is None:
                return None
            torch.mv(
                down_result.transpose(0, 1),
                weights,
                out=result[:hidden],
            )
            return result[:hidden]

    if any(
        str(getattr(weight, "layout", "row-major")) != "row-major"
        for bundle in experts
        for weight in bundle
    ):
        # A runtime Q4/u16/tiled image is no longer a packed VQ bitstream.
        # Reuse the generic resident mixed executor so combined GU archives
        # retain their execution layout instead of being reinterpreted by the
        # legacy row-major Top-K entry point.
        mixed = make_packed_two_layer_cpu(
            tuple(experts), force_mixed=True
        )
        if mixed is None:
            return None
        return mixed.forward(
            x_float,
            torch.arange(len(experts), dtype=torch.int64),
            weights,
            float(limit),
            str(activation).strip().lower(),
            float(activation_beta),
            (
                -1.0
                if activation_linear_beta is None
                else float(activation_linear_beta)
            ),
        )

    gu = [pair[0] for pair in experts]
    dn = [pair[1] for pair in experts]
    unique_gu: dict[tuple[int, int, int, int], object] = {}
    gu_score_count = 0
    for weight in gu:
        key = (
            weight.cb.data_ptr(),
            int(weight.blocks),
            int(weight.cb.shape[0]),
            int(weight.cb.shape[1]),
        )
        if key not in unique_gu:
            unique_gu[key] = weight
            gu_score_count += int(weight.blocks) * int(
                weight.cb.shape[0]
    )
    intermediate = int(dn[0].cols)
    dn_score_count = sum(
        int(weight.blocks) * int(weight.cb.shape[0])
        for weight in dn
        if int(weight.rows) * int(weight.dim)
        >= int(weight.cb.shape[0]) * int(weight.dim)
        + int(weight.rows)
    )
    required = (
        gu_score_count
        + 4 * len(experts) * intermediate
        + dn_score_count
        + len(experts) * hidden
    )

    with _PACKED_MOE_LOCK:
        if (
            _PACKED_MOE_WORKSPACE is None
            or _PACKED_MOE_WORKSPACE[0].numel() < required
            or _PACKED_MOE_WORKSPACE[1].numel() < hidden
        ):
            _PACKED_MOE_WORKSPACE = (
                torch.empty(required, dtype=torch.float32),
                torch.empty(hidden, dtype=torch.float32),
            )
        workspace, result = _PACKED_MOE_WORKSPACE
        return extension.moe_packed_topk(
            x_float,
            [weight.raw for weight in gu],
            [weight.cb.float().contiguous() for weight in gu],
            [int(weight.rows) for weight in gu],
            [int(weight.blocks) for weight in gu],
            [int(weight.bits) for weight in gu],
            [weight.raw for weight in dn],
            [weight.cb.float().contiguous() for weight in dn],
            [int(weight.rows) for weight in dn],
            [int(weight.blocks) for weight in dn],
            [int(weight.bits) for weight in dn],
            weights,
            float(limit),
            str(activation).strip().lower(),
            float(activation_beta),
            (
                -1.0
                if activation_linear_beta is None
                else float(activation_linear_beta)
            ),
            workspace,
            result,
        )


def reset_packed_moe_phase_profile() -> None:
    extension = _build()
    if extension is not None:
        extension.reset_packed_moe_phase_profile()


def make_resident_projection_cpu(weights: tuple[object, ...]):
    """Build one fixed-address mixed BF16/block-FP8 decode projection.

    The returned native executor owns token-sized workspaces only. Source
    weights remain in their original format and a single OpenMP team covers
    every logical output row.
    """
    from .kernels import BlockFP8Weight

    if not weights:
        return None
    cols = int(weights[0].shape[1])
    payloads: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    rows: list[int] = []
    kinds: list[int] = []
    empty_scale = torch.empty(0, dtype=torch.float32)
    for weight in weights:
        if int(weight.shape[1]) != cols:
            return None
        if (
            isinstance(weight, torch.Tensor)
            and not weight.is_cuda
            and weight.dtype == torch.bfloat16
            and weight.ndim == 2
            and weight.is_contiguous()
        ):
            payloads.append(weight)
            scales.append(empty_scale)
            rows.append(int(weight.shape[0]))
            kinds.append(0)
        elif (
            isinstance(weight, BlockFP8Weight)
            and not weight.q.is_cuda
            and weight.block == 128
            and weight.q.is_contiguous()
            and weight.s.is_contiguous()
        ):
            payloads.append(weight.q)
            scales.append(weight.s)
            rows.append(int(weight.rows))
            kinds.append(2 if weight.layout == "q4_0" else 1)
        else:
            return None
    extension = _build()
    if extension is None:
        return None
    return extension.CpuResidentProjectionLayer(
        payloads,
        scales,
        rows,
        kinds,
        cols,
        128,
    )


def reset_resident_projection_profile() -> None:
    extension = _build()
    if extension is not None:
        extension.reset_resident_projection_profile()


def resident_projection_profile() -> dict[str, float | int]:
    extension = _build()
    if extension is None:
        return {"calls": 0, "seconds": 0.0}
    values = extension.resident_projection_profile()
    return {
        "calls": int(values[0]),
        "seconds": float(values[1]),
    }


def make_packed_three_layer_cpu(
    experts: tuple[tuple[object, object, object], ...],
    *,
    force_mixed: bool = False,
):
    """Build one generic resident packed Gate/Up/Down layer executor.

    The executor retains compact p8--p16 payload tensors and their codebooks;
    it never materializes indices or dequantized expert matrices.  Uniform
    layers use the C++ resident directory.  Heterogeneous layers retain the
    same public resident interface and select compact bundles directly before
    entering the mixed-codebook fused operator, avoiding store/LRU rebuilds.
    """
    if not experts or any(len(bundle) != 3 for bundle in experts):
        return None
    gate = tuple(bundle[0] for bundle in experts)
    up = tuple(bundle[1] for bundle in experts)
    down = tuple(bundle[2] for bundle in experts)

    def common(weights):
        first = weights[0]
        signature = (
            int(first.rows),
            int(first.blocks),
            int(first.bits),
            str(getattr(first, "layout", "row-major")),
        )
        if any(
            (
                int(weight.rows),
                int(weight.blocks),
                int(weight.bits),
                str(getattr(weight, "layout", "row-major")),
            )
            != signature
            for weight in weights[1:]
        ):
            return None
        return signature

    def shares_one_codebook(weights) -> bool:
        """Return whether one projection can use the uniform executor.

        Shape and packing equality is not sufficient: expert-assigned and
        grouped-codebook archives deliberately keep multiple codebooks under
        the same p8..p16 layout. Sending such a layer to the legacy uniform
        directory makes its first routed combination fall back to Python.
        Compare the retained tensor address so every multi-codebook layer
        enters the generic mixed executor prepared below.
        """
        first = weights[0].cb
        return all(
            weight.cb.data_ptr() == first.data_ptr()
            for weight in weights[1:]
        )

    gate_spec = common(gate)
    up_spec = common(up)
    down_spec = common(down)
    extension = _build()
    if extension is None:
        return None
    uses_non_row_major = any(
        getattr(weight, "layout", "row-major") != "row-major"
        for weights in (gate, up, down)
        for weight in weights
    )
    uses_multiple_codebooks = any(
        not shares_one_codebook(weights)
        for weights in (gate, up, down)
    )
    if (
        force_mixed
        or uses_non_row_major
        or uses_multiple_codebooks
        or gate_spec is None
        or up_spec is None
        or down_spec is None
    ):
        def metadata(weights):
            return (
                [weight.raw for weight in weights],
                [weight.cb.float().contiguous() for weight in weights],
                [int(weight.rows) for weight in weights],
                [int(weight.blocks) for weight in weights],
                [int(weight.bits) for weight in weights],
                [
                    {
                        "row-major": 0,
                        "block-major": 1,
                        "row-tile-8": 2,
                        "u16-row-tile-8": 2,
                        "q4_0": 3,
                    }[getattr(weight, "layout", "row-major")]
                    for weight in weights
                ],
            )

        return extension.CpuPackedThreeMixedLayer(
            *metadata(gate),
            *metadata(up),
            *metadata(down),
        )
    return extension.CpuPackedThreeLayer(
        [weight.raw for weight in gate],
        [weight.cb.float().contiguous() for weight in gate],
        *gate_spec[:3],
        [weight.raw for weight in up],
        [weight.cb.float().contiguous() for weight in up],
        *up_spec[:3],
        [weight.raw for weight in down],
        [weight.cb.float().contiguous() for weight in down],
        *down_spec[:3],
    )


class _PackedProjectionView:
    """A zero-copy logical row slice of one packed execution image."""

    __slots__ = (
        "raw",
        "cb",
        "rows",
        "cols",
        "blocks",
        "dim",
        "bits",
        "source_bits",
        "layout",
    )

    def __init__(self, source, raw: torch.Tensor, rows: int) -> None:
        self.raw = raw.contiguous()
        self.cb = source.cb
        self.rows = int(rows)
        self.cols = int(source.cols)
        self.blocks = int(source.blocks)
        self.dim = int(source.dim)
        self.bits = int(source.bits)
        self.source_bits = int(getattr(source, "source_bits", source.bits))
        self.layout = str(getattr(source, "layout", "row-major"))


def _split_combined_gu_projection(weight):
    """Split a manifest-declared combined GU image without copying bytes."""
    rows = int(weight.rows)
    if rows <= 0 or rows % 2:
        return None
    half = rows // 2
    layout = str(getattr(weight, "layout", "row-major"))
    raw = weight.raw.contiguous().view(torch.uint8).reshape(-1)
    if layout == "q4_0":
        if half % 8 or int(weight.cols) % 32:
            return None
        row_bytes = (int(weight.cols) // 32) * 18
        expected = rows * row_bytes
        split = half * row_bytes
    elif layout in {"row-major", "row-tile-8", "u16-row-tile-8"}:
        if layout != "row-major" and half % 8:
            return None
        half_bits = half * int(weight.blocks) * int(weight.bits)
        if half_bits % 8:
            return None
        expected = rows * int(weight.blocks) * int(weight.bits) // 8
        split = half_bits // 8
    else:
        # Whole-layer block-major payloads interleave the two logical halves.
        return None
    if raw.numel() != expected or not 0 < split < raw.numel():
        return None
    return (
        _PackedProjectionView(weight, raw[:split], half),
        _PackedProjectionView(weight, raw[split:], half),
    )


def make_packed_two_layer_cpu(
    experts: tuple[tuple[object, object], ...],
    *,
    force_mixed: bool = False,
):
    """Build a resident Gate/Up/Down plan from combined GU + Down storage.

    Capability is derived solely from the two-projection archive layout.  The
    combined source tensor remains shared; Gate and Up are zero-copy row views
    and the existing format-driven mixed executor handles Q4 and packed tiers.
    """
    if not experts or any(len(bundle) != 2 for bundle in experts):
        return None
    expanded = []
    for gu, down in experts:
        split = _split_combined_gu_projection(gu)
        if split is None:
            return None
        gate, up = split
        expanded.append((gate, up, down))
    return make_packed_three_layer_cpu(
        tuple(expanded), force_mixed=force_mixed
    )


def configure_packed_latent_moe_cpu(
    executor,
    input_weights: tuple[object, object, object, object],
    output_weights: tuple[object, object],
    route_correction: torch.Tensor,
    route_mask: torch.Tensor,
    routed_norm: torch.Tensor,
    *,
    top_k: int,
    normalize_route: bool,
    routed_scaling: float,
    rms_eps: float,
    limit: float,
    scoring: str,
    activation: str,
    beta: float,
    linear_beta: float | None,
):
    """Attach a full latent-MoE decode graph to a resident packed layer."""
    from .kernels import BlockFP8Weight

    if executor is None or not hasattr(executor, "configure_latent_moe"):
        return None

    def metadata(weights):
        payloads = []
        scales = []
        rows = []
        cols = []
        kinds = []
        for weight in weights:
            if isinstance(weight, BlockFP8Weight):
                if weight.q.is_cuda or weight.block != 128:
                    return None
                payloads.append(weight.q)
                scales.append(weight.s)
                rows.append(int(weight.rows))
                cols.append(int(weight.cols))
                kinds.append(1)
            elif (
                isinstance(weight, torch.Tensor)
                and not weight.is_cuda
                and weight.ndim == 2
                and weight.dtype in (torch.bfloat16, torch.float32)
            ):
                payloads.append(weight.contiguous())
                scales.append(torch.empty(0, dtype=torch.float32))
                rows.append(int(weight.shape[0]))
                cols.append(int(weight.shape[1]))
                kinds.append(0 if weight.dtype == torch.bfloat16 else 2)
            else:
                return None
        return payloads, scales, rows, cols, kinds

    input_meta = metadata(input_weights)
    output_meta = metadata(output_weights)
    if input_meta is None or output_meta is None:
        return None
    if len(set(input_meta[3])) != 1:
        return None
    executor.configure_latent_moe(
        input_meta[0],
        input_meta[1],
        input_meta[2],
        input_meta[4],
        input_meta[3][0],
        output_meta[0],
        output_meta[1],
        output_meta[2],
        output_meta[3],
        output_meta[4],
        route_correction.float().contiguous(),
        route_mask.bool().contiguous(),
        routed_norm.to(torch.bfloat16).contiguous(),
        128,
        int(top_k),
        bool(normalize_route),
        float(routed_scaling),
        float(rms_eps),
        float(limit),
        str(scoring),
        str(activation),
        float(beta),
        -1.0 if linear_beta is None else float(linear_beta),
    )
    return executor


def reset_latent_moe_phase_profile() -> None:
    extension = _build()
    if extension is not None:
        extension.reset_latent_moe_phase_profile()


def latent_moe_phase_profile() -> dict[str, float | int]:
    extension = _build()
    if extension is None:
        return {}
    values = extension.latent_moe_phase_profile()
    names = (
        "calls",
        "prelude_route_seconds",
        "packed_experts_seconds",
        "norm_output_seconds",
        "total_seconds",
    )
    return {
        name: int(value) if name == "calls" else float(value)
        for name, value in zip(names, values)
    }


def configure_packed_resident_moe_cpu(
    executor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    router_mask: torch.Tensor,
    shared_weights: tuple[object, object, object],
    *,
    top_k: int,
    normalize_route: bool,
    routed_scaling: float,
):
    """Attach dense Router/shared projections to a compact resident layer.

    The native executor keeps each source in its own compact format: routed
    experts remain packed p8--p16 and shared projections remain block-FP8.
    No logical weight matrix is materialized by this configuration step.
    """
    from .kernels import BlockFP8Weight

    if (
        executor is None
        or not hasattr(executor, "configure_fused_moe")
        or router_weight.is_cuda
        or router_weight.ndim != 2
        or router_weight.dtype not in (torch.float32, torch.bfloat16)
        or len(shared_weights) != 3
        or not all(
            isinstance(weight, BlockFP8Weight)
            and not weight.q.is_cuda
            and weight.block == 128
            for weight in shared_weights
        )
    ):
        return None
    executor.configure_fused_moe(
        router_weight.contiguous(),
        router_bias.float().contiguous(),
        router_mask.bool().contiguous(),
        [weight.q for weight in shared_weights],
        [weight.s for weight in shared_weights],
        [int(weight.rows) for weight in shared_weights],
        [int(weight.cols) for weight in shared_weights],
        128,
        int(top_k),
        bool(normalize_route),
        float(routed_scaling),
    )
    return executor


def packed_moe_phase_profile() -> dict[str, float | int]:
    extension = _build()
    if extension is None:
        return {}
    values = extension.packed_moe_phase_profile()
    names = (
        "calls",
        "gu_score_seconds",
        "gu_lookup_seconds",
        "activation_seconds",
        "down_score_seconds",
        "down_compute_seconds",
        "reduce_seconds",
    )
    return {
        name: int(value) if name == "calls" else float(value)
        for name, value in zip(names, values)
    }


def reset_three_projection_phase_profile() -> None:
    extension = _build()
    if extension is not None:
        extension.reset_three_projection_phase_profile()


def three_projection_phase_profile() -> dict[str, float | int]:
    extension = _build()
    if extension is None:
        return {}
    values = extension.three_projection_phase_profile()
    names = (
        "calls",
        "gate_seconds",
        "up_seconds",
        "activation_seconds",
        "down_seconds",
        "reduce_seconds",
    )
    return {
        name: int(value) if name == "calls" else float(value)
        for name, value in zip(names, values)
    }


def reset_resident_moe_phase_profile() -> None:
    extension = _build()
    if extension is not None:
        extension.reset_resident_moe_phase_profile()


def resident_moe_phase_profile() -> dict[str, float | int]:
    extension = _build()
    if extension is None:
        return {}
    values = extension.resident_moe_phase_profile()
    names = (
        "calls",
        "router_shared_gu_seconds",
        "routed_gu_seconds",
        "shared_routed_down_seconds",
        "selected_experts",
        "q4_selected_experts",
    )
    return {
        name: int(value)
        if name in {"calls", "selected_experts", "q4_selected_experts"}
        else float(value)
        for name, value in zip(names, values)
    }


def route_topk_sigmoid_cpu(
    logits: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    top_k: int,
    normalize: bool,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run stable sigmoid routing without materializing sort auxiliaries."""
    if logits.is_cuda or bias.is_cuda or mask.is_cuda:
        return None
    extension = _build()
    if extension is None:
        return None
    weights, indices = extension.route_topk_sigmoid(
        logits,
        bias,
        mask,
        int(top_k),
        bool(normalize),
        float(scaling),
    )
    return weights, indices


def reset_block_fp8_gemv_profile() -> None:
    extension = _build()
    if extension is not None:
        extension.reset_block_fp8_gemv_profile()


def block_fp8_gemv_profile() -> dict[str, float | int]:
    extension = _build()
    if extension is None:
        return {}
    values = extension.block_fp8_gemv_profile()
    return {
        "calls": int(values[0]),
        "seconds": float(values[1]),
        "weight_elements": int(values[2]),
        "block_major_pack_calls": int(values[3]),
        "block_major_packed_bytes": int(values[4]),
        "numa_bound_tasks": int(values[5]),
        "block_major_rows8_tasks": (
            int(values[6]) if len(values) > 6 else 0
        ),
        "block_gemm_calls": int(values[7]) if len(values) > 7 else 0,
        "block_gemm_tokens": int(values[8]) if len(values) > 8 else 0,
    }


def kda_recurrent_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    workspace: torch.Tensor,
    output: torch.Tensor,
    lower_bound: float,
    output_gate: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> torch.Tensor | None:
    if (
        query.is_cuda
        or query.ndim != 2
        or query.dtype not in (torch.float32, torch.bfloat16)
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    if output_gate is None:
        output_gate = torch.empty(0, dtype=query.dtype)
    if norm_weight is None:
        norm_weight = torch.empty(0, dtype=query.dtype)
    return extension.kda_recurrent(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        gate.contiguous(),
        beta.contiguous(),
        a_log.contiguous(),
        dt_bias.contiguous(),
        state,
        workspace,
        output,
        float(lower_bound),
        output_gate.contiguous(),
        norm_weight.contiguous(),
        float(norm_eps),
    )


def qwen35_delta_recurrent_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run one cached Qwen3.5 gated-delta step without ATen dispatch churn."""
    if (
        query.is_cuda
        or query.ndim != 2
        or query.dtype not in (torch.float32, torch.bfloat16)
        or key.shape != query.shape
        or value.ndim != 2
        or state.dtype != torch.float32
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    output = torch.empty_like(value)
    result = extension.qwen35_delta_recurrent(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        gate.contiguous(),
        beta.contiguous(),
        state,
        output,
    )
    return result, state


def qwen35_conv1d_update_cpu(
    value: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor | None:
    """Update Qwen3.5's cached depthwise convolution in one native pass."""
    if (
        value.is_cuda
        or value.ndim != 3
        or value.shape[-1] != 1
        or value.dtype not in (torch.float32, torch.bfloat16)
        or state.dtype != value.dtype
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    output = torch.empty_like(value)
    return extension.qwen35_conv1d_update(
        value.contiguous(),
        state,
        weight.contiguous(),
        output,
    )


def short_conv3_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> bool:
    if (
        query.is_cuda
        or query.ndim != 1
        or query.dtype not in (torch.float32, torch.bfloat16)
    ):
        return False
    extension = _build()
    if extension is None:
        return False
    return bool(
        extension.short_conv3(
            query,
            key,
            value,
            list(states),
            list(weights),
        )
    )


def gated_rmsnorm_cpu(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    if (
        value.is_cuda
        or value.ndim != 2
        or value.dtype not in (torch.float32, torch.bfloat16)
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.gated_rmsnorm(
        value,
        gate,
        weight,
        output,
        float(eps),
    )


def moe_mixed_cpu(
    x_row: torch.Tensor,
    gu_indices: list[torch.Tensor],
    gu_codebooks: list[torch.Tensor],
    dn_indices: list[torch.Tensor],
    dn_codebooks: list[torch.Tensor],
    route_weights: torch.Tensor,
    shared_w1_q: torch.Tensor,
    shared_w1_s: torch.Tensor,
    shared_w3_q: torch.Tensor,
    shared_w3_s: torch.Tensor,
    shared_w2_q: torch.Tensor,
    shared_w2_s: torch.Tensor,
    group_size: int,
    limit: float,
) -> torch.Tensor | None:
    if (
        x_row.is_cuda
        or x_row.ndim != 2
        or x_row.shape[0] != 1
        or not gu_indices
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.moe_mixed(
        x_row,
        gu_indices,
        gu_codebooks,
        dn_indices,
        dn_codebooks,
        route_weights,
        shared_w1_q,
        shared_w1_s,
        shared_w3_q,
        shared_w3_s,
        shared_w2_q,
        shared_w2_s,
        int(group_size),
        float(limit),
        False,
    )


def make_moe_layer_cpu(
    experts: tuple[tuple[object, object] | None, ...],
    shared_w1_q: torch.Tensor,
    shared_w1_s: torch.Tensor,
    shared_w3_q: torch.Tensor,
    shared_w3_s: torch.Tensor,
    shared_w2_q: torch.Tensor,
    shared_w2_s: torch.Tensor,
    gate_q: torch.Tensor,
    gate_s: torch.Tensor,
    gate_bias: torch.Tensor,
    gate_mask: torch.Tensor,
    group_size: int,
    limit: float,
    top_k: int,
    normalize_route: bool,
    routed_scaling: float,
):
    present = [expert for expert in experts if expert is not None]
    if not present:
        return None
    fallback_gu, fallback_dn = present[0]
    gu = [
        expert[0] if expert is not None else fallback_gu
        for expert in experts
    ]
    dn = [
        expert[1] if expert is not None else fallback_dn
        for expert in experts
    ]
    extension = _build()
    if extension is None:
        return None
    return extension.CpuMoeLayer(
        [weight.idx for weight in gu],
        [weight.cb for weight in gu],
        [weight.idx for weight in dn],
        [weight.cb for weight in dn],
        torch.tensor(
            [expert is not None for expert in experts],
            dtype=torch.bool,
        ),
        shared_w1_q,
        shared_w1_s,
        shared_w3_q,
        shared_w3_s,
        shared_w2_q,
        shared_w2_s,
        gate_q,
        gate_s,
        gate_bias,
        gate_mask,
        int(group_size),
        float(limit),
        int(top_k),
        bool(normalize_route),
        float(routed_scaling),
    )


def reset_moe_phase_profile_cpu() -> None:
    extension = _build()
    if extension is not None:
        extension.reset_moe_phase_profile()


def moe_phase_profile_cpu() -> torch.Tensor | None:
    extension = _build()
    if extension is None:
        return None
    return extension.moe_phase_profile()


def int4_gemv_cpu(
    x_row: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    cols: int,
    group_size: int,
) -> torch.Tensor | None:
    if (
        x_row.is_cuda
        or packed.is_cuda
        or scales.is_cuda
        or x_row.ndim != 2
        or x_row.shape[0] != 1
        or packed.dtype != torch.uint8
        or scales.dtype != torch.float16
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.int4_gemv(
        x_row,
        packed,
        scales,
        int(cols),
        int(group_size),
    )


def int4_gemv_many_cpu(
    x_row: torch.Tensor,
    packed: list[torch.Tensor],
    scales: list[torch.Tensor],
    group_size: int,
) -> list[torch.Tensor] | None:
    if (
        x_row.is_cuda
        or x_row.ndim != 2
        or x_row.shape[0] != 1
        or not packed
        or len(packed) != len(scales)
        or any(weight.is_cuda for weight in packed + scales)
        or any(weight.dtype != torch.uint8 for weight in packed)
        or any(scale.dtype != torch.float16 for scale in scales)
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.int4_gemv_many(
        x_row,
        packed,
        scales,
        int(group_size),
    )


def int4_grouped_gemv_cpu(
    x_groups: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    cols: int,
    group_size: int,
    rows_per_input: int,
) -> torch.Tensor | None:
    if (
        x_groups.is_cuda
        or x_groups.ndim != 2
        or packed.dtype != torch.uint8
        or scales.dtype != torch.float16
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.int4_grouped_gemv(
        x_groups,
        packed,
        scales,
        int(cols),
        int(group_size),
        int(rows_per_input),
    )


def o_proj_int4_cpu(
    x_groups: torch.Tensor,
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    a_cols: int,
    a_group_size: int,
    rows_per_input: int,
    b_packed: torch.Tensor,
    b_scales: torch.Tensor,
    b_cols: int,
    b_group_size: int,
) -> torch.Tensor | None:
    if (
        x_groups.is_cuda
        or x_groups.dtype != torch.float32
        or x_groups.ndim != 2
        or a_packed.dtype != torch.uint8
        or a_scales.dtype != torch.float16
        or b_packed.dtype != torch.uint8
        or b_scales.dtype != torch.float16
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.o_proj_int4(
        x_groups,
        a_packed,
        a_scales,
        int(a_cols),
        int(a_group_size),
        int(rows_per_input),
        b_packed,
        b_scales,
        int(b_cols),
        int(b_group_size),
    )


def hc_pre_norm_cpu(
    x: torch.Tensor,
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm: torch.Tensor,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
    output_buffers: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        x.is_cuda
        or x.dtype != torch.float32
        or mixes.dtype != torch.float32
        or x.ndim != 4
        or x.shape[0] * x.shape[1] != 1
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    hc = int(x.shape[2])
    hidden = int(x.shape[3])
    if output_buffers is None:
        options = {"dtype": torch.float32, "device": "cpu"}
        output_buffers = (
            torch.empty((1, 1, hidden), **options),
            torch.empty((1, 1, hc), **options),
            torch.empty((1, 1, hc, hc), **options),
        )
    return extension.hc_pre_norm(
        x,
        mixes,
        scale,
        base,
        norm,
        int(sinkhorn_iters),
        float(rms_eps),
        float(hc_eps),
        *output_buffers,
    )


def hc_post_cpu(
    out: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if (
        out.is_cuda
        or out.dtype != torch.float32
        or residual.dtype != torch.float32
        or residual.ndim != 4
        or residual.shape[0] * residual.shape[1] != 1
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    if output is None:
        output = torch.empty_like(residual)
    return extension.hc_post(
        out,
        residual,
        post,
        comb,
        output,
    )


def qkv_pre_cpu(
    q_rank_raw: torch.Tensor,
    kv_raw: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    rms_eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        q_rank_raw.is_cuda
        or kv_raw.is_cuda
        or q_rank_raw.dtype != torch.float32
        or kv_raw.dtype != torch.float32
        or q_rank_raw.ndim != 2
        or q_rank_raw.shape[0] != 1
        or kv_raw.ndim != 2
        or kv_raw.shape[0] != 1
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.qkv_pre(
        q_rank_raw,
        kv_raw,
        q_norm,
        kv_norm,
        rope_cos,
        rope_sin,
        float(rms_eps),
    )


def q_post_cpu(
    query: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    rms_eps: float,
) -> torch.Tensor | None:
    if (
        query.is_cuda
        or query.dtype != torch.float32
        or query.ndim != 4
        or query.shape[0] * query.shape[1] != 1
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.q_post(
        query,
        rope_cos,
        rope_sin,
        float(rms_eps),
    )


def q_int4_post_cpu(
    q_rank: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    cols: int,
    group_size: int,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    heads: int,
    head_dim: int,
    rms_eps: float,
) -> torch.Tensor | None:
    if (
        q_rank.is_cuda
        or q_rank.dtype != torch.float32
        or packed.dtype != torch.uint8
        or scales.dtype != torch.float16
        or q_rank.ndim != 2
        or q_rank.shape[0] != 1
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.q_int4_post(
        q_rank,
        packed,
        scales,
        int(cols),
        int(group_size),
        rope_cos,
        rope_sin,
        int(heads),
        int(head_dim),
        float(rms_eps),
    )


def attention_decode_cpu(
    query: torch.Tensor,
    raw_values: torch.Tensor,
    raw_positions: torch.Tensor,
    selected_values: torch.Tensor,
    sink: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    scale: float,
) -> torch.Tensor | None:
    if (
        query.is_cuda
        or query.dtype != torch.float32
        or raw_values.dtype != torch.float32
        or selected_values.dtype != torch.float32
        or raw_positions.dtype != torch.long
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.attention_decode(
        query,
        raw_values,
        raw_positions,
        selected_values,
        sink,
        rope_cos,
        rope_sin,
        float(scale),
    )


def prebuild() -> bool:
    ok = _build(verbose=True) is not None
    print(
        "[cccp] CPU融合内核"
        + ("编译成功" if ok else f"不可用（{_ERR}），使用PyTorch回退")
    )
    return ok


def last_error() -> str | None:
    return _ERR


def extension_status() -> dict[str, object]:
    """返回离线原生算子的可审计状态，并触发一次懒加载。"""
    extension = _build()
    return {
        "available": extension is not None,
        "name": _EXTENSION_NAME,
        "source": _SOURCE,
        "error": _ERR,
        "threads": configure_cpu_threads(),
        "windows_performance_mode": configure_windows_performance(),
    }

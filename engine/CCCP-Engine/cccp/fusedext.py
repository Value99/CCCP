"""CCCP 可选 CUDA 融合 kernel 的加载器。

包含六类融合 kernel（csrc/vq_gemv.cu）：
  - vq_gemv：VQ 分组 GEMV（码本查表+点积单 kernel，u8/u16 索引、码本/索引广播）；
  - hc_sinkhorn：Hyper-Connections 4×4 双随机归一化（softmax+20 轮一次 launch）；
  - rmsnorm：f32 行归一化；rope1：decode 单相位交错 RoPE。
  - dsv4_attn_decode：单 token 的 score/sink-softmax/value/RoPE。
  - dsv4_hc_pre：HC 的 RMS/GEMV/Sinkhorn/通道归约整段融合。

行为：
  - 导入时尝试用 torch.utils.cpp_extension.load 编译/复用缓存
    （已编译过走缓存，约 1-2s；未编译且工具链缺失时静默记为不可用）；
  - 可用时 available() 为 True，grouped.py / dsv4model.py 的钩子优先走融合路径；
  - 不可用（无 CUDA / 无 nvcc+MSVC+ninja / 编译失败）时自动回退
    torch 批量路径 —— 推理功能完全不依赖本模块。

手动预编译（推荐随安装执行一次）：
  python -c "from cccp import fusedext; fusedext.prebuild()"
环境变量：
  CCCP_FUSED=0  强制禁用（调试用）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import torch

from .build_progress import operator_build_progress

_EXT = None
_ERR: str | None = None
_DLL_DIRECTORY_HANDLES: list[object] = []
_EXTENSION_ABI = "v15"
_EXTENSION_NAME = "cccp_vq_gemv_hc_rms_gpu_" + _EXTENSION_ABI


def _operator_cache_root() -> Path:
    """Return a writable, version-neutral cache root for local GPU binaries."""
    configured = os.environ.get("CCCP_OPERATOR_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if os.name == "nt" and local_app_data:
        return (Path(local_app_data) / "CCCP-Launcher" / "operator-cache").resolve()
    return (_launcher_root() / "data" / "operator-cache").resolve()


def _safe_cache_component(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _operator_cache_identity(
    source: str | Path,
    cuda_capability: tuple[int, int] | None,
    rocm_arch: str = "",
) -> tuple[str, Path, str, str]:
    """Build a deterministic module name/directory for one compatible binary.

    Absolute source and launcher paths are intentionally excluded. Moving or
    upgrading the portable launcher must not rebuild an unchanged operator.
    """
    source_path = Path(source)
    if torch.version.hip is not None:
        backend = "hip"
        arch = _safe_cache_component(rocm_arch or "auto") or "auto"
        runtime = str(torch.version.hip or "unknown")
    else:
        backend = "cuda"
        arch = (
            f"sm{cuda_capability[0]}{cuda_capability[1]}"
            if cuda_capability is not None
            else "sm_auto"
        )
        runtime = str(torch.version.cuda or "unknown")
    identity = {
        "abi": _EXTENSION_ABI,
        "backend": backend,
        "arch": arch,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "runtime": runtime,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "torch": str(torch.__version__).split("+")[0],
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=True, sort_keys=True).encode("ascii")
    ).hexdigest()[:16]
    module_name = f"cccp_vq_gpu_{_EXTENSION_ABI}_{backend}_{arch}_{digest}"
    build_directory = _operator_cache_root() / backend / arch / digest
    return module_name, build_directory, backend, arch


def _load_extension_binary(module_name: str, binary: Path):
    """Load one exact Python extension binary without invoking a compiler."""
    spec = importlib.util.spec_from_file_location(module_name, binary)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法读取算子缓存：{binary}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_cached_extension(module_name: str, build_directory: Path, suffix: str):
    """Load a completed local extension without invoking Ninja or a compiler."""
    binary = build_directory / f"{module_name}{suffix}"
    if not binary.is_file():
        return None
    return _load_extension_binary(module_name, binary)


def _packaged_gpu_operator_root() -> Path:
    """Return the inference package's architecture-specific GPU binaries."""
    return Path(__file__).resolve().parent / "native" / "gpu"


def _packaged_extension_path(
    module_name: str,
    backend: str,
    arch: str,
    suffix: str,
) -> Path:
    return (
        _packaged_gpu_operator_root()
        / _safe_cache_component(backend)
        / _safe_cache_component(arch)
        / f"{module_name}{suffix}"
    )


def _load_packaged_extension(
    module_name: str,
    backend: str,
    arch: str,
    suffix: str,
):
    """Load an ABI/source/architecture-exact binary shipped in the package."""
    binary = _packaged_extension_path(
        module_name, backend, arch, suffix
    )
    if not binary.is_file():
        return None
    return _load_extension_binary(module_name, binary)


def _validate_cuda_toolchain_arch(capability: tuple[int, int]) -> None:
    """Fail clearly when a toolkit cannot emit code for the active GPU.

    The common kernels do not select an implementation by model or GPU name.
    Hopper and Blackwell therefore share the same source; JIT compilation
    targets the active compute capability.  Blackwell SM120 requires a CUDA
    12.8-or-newer PyTorch toolchain instead of silently compiling a Hopper
    binary that cannot execute on an RTX 5090.
    """
    # ROCm intentionally exposes HIP through torch.cuda. cpp_extension will
    # HIPify the .cu source and use hipcc; NVIDIA SM/CUDA-version rules do not
    # apply to a gfx architecture.
    if torch.version.hip is not None:
        return
    if capability < (7, 5):
        raise RuntimeError(
            f"当前离线 CUDA 13 环境最低支持 SM75（RTX 20 系）；"
            f"检测到 SM{capability[0]}{capability[1]}"
        )
    if capability[0] < 12:
        return
    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise RuntimeError("SM120 requires a CUDA-enabled PyTorch build")
    major, minor = (int(part) for part in cuda_version.split(".")[:2])
    if (major, minor) < (12, 8):
        raise RuntimeError(
            "SM120 requires CUDA 12.8 or newer; "
            f"current PyTorch CUDA is {cuda_version}"
        )


def _apply_cuda_hardware_defaults() -> None:
    """Apply measured kernel defaults without overriding explicit tuning.

    This lives in the common CUDA loader because the routed packed operator is
    shared by every architecture.  Setting the environment value before the
    extension is loaded also avoids stale extension-cache binaries silently
    retaining an older C++ default.
    """
    if "CCCP_ROUTED_WARPS" in os.environ or not torch.cuda.is_available():
        return
    try:
        capability = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0).upper()
    except Exception:
        return
    if capability == (9, 0) and "H20" in device_name:
        os.environ["CCCP_ROUTED_WARPS"] = "16"


def _select_cuda_architecture() -> tuple[int, int] | None:
    """选择当前 NVIDIA 卡的唯一目标架构并写入 JIT 编译环境。"""
    forced = os.environ.get("CCCP_CUDA_ARCH", "").strip()
    if forced:
        parts = forced.split(".", 1)
        capability = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    else:
        try:
            capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
        except Exception:
            return None
    _validate_cuda_toolchain_arch(capability)
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST", f"{capability[0]}.{capability[1]}"
    )
    return capability


def _ensure_ninja_on_path() -> None:
    """让 PyTorch JIT 构建能找到 pip 安装的 Ninja 可执行文件。

    非交互 SSH 会话不一定继承 ``~/.local/bin``，即使 Python 已经可以
    导入 ninja 包。PyTorch 在构建前使用 ``shutil.which`` 查找可执行文件，
    因此在需要时补入该包声明的二进制目录。
    """
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
    if not os.path.isfile(os.path.join(bin_dir, executable)):
        return
    current = os.environ.get("PATH", "")
    paths = current.split(os.pathsep) if current else []
    normalized = {os.path.normcase(os.path.abspath(path)) for path in paths}
    if os.path.normcase(os.path.abspath(bin_dir)) not in normalized:
        os.environ["PATH"] = bin_dir + (os.pathsep + current if current else "")


def _configure_packaged_gpu_toolchain() -> None:
    """Expose pip-bundled CUDA/ROCm compilers before cpp_extension imports."""
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    if torch.version.hip is not None:
        rocm_root = os.path.join(site_packages, "_rocm_sdk_core")
        if os.path.isfile(os.path.join(rocm_root, "bin", "hipcc.exe")):
            # 隔离运行时必须优先使用随包 SDK；不能继承用户机器上另一个
            # ROCm/CUDA_HOME，否则编译器、头文件和 DLL 版本会被混用。
            os.environ["ROCM_HOME"] = rocm_root
            os.environ["ROCM_PATH"] = rocm_root
            # PyTorch's Windows HIP extension helper still consults
            # CUDA_HOME while writing the Ninja link rule.
            os.environ["CUDA_HOME"] = rocm_root
            rocm_library_bin = os.path.join(
                site_packages, "_rocm_sdk_libraries_custom", "bin"
            )
            rocm_bins = [os.path.join(rocm_root, "bin"), rocm_library_bin]
            os.environ["PATH"] = os.pathsep.join(rocm_bins) + os.pathsep + os.environ.get("PATH", "")
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                for directory in rocm_bins:
                    if os.path.isdir(directory):
                        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory))
        return
    cuda_root = os.path.join(site_packages, "nvidia", "cu13")
    if os.path.isfile(os.path.join(cuda_root, "bin", "nvcc.exe")):
        os.environ["CUDA_HOME"] = cuda_root
        os.environ["CUDA_PATH"] = cuda_root
        cuda_bins = [
            os.path.join(cuda_root, "bin"),
            os.path.join(cuda_root, "bin", "x86_64"),
        ]
        os.environ["PATH"] = os.pathsep.join(cuda_bins) + os.pathsep + os.environ.get("PATH", "")
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for directory in cuda_bins:
                if os.path.isdir(directory):
                    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory))


def _launcher_root() -> Path:
    configured = os.environ.get("CCCP_LAUNCHER_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    # .../engine/CCCP-Engine/cccp/fusedext.py -> 发行根
    return Path(__file__).resolve().parents[3]


def _bundled_windows_tool(name: str) -> str | None:
    """Prefer the release's MSVC tool and consult the system only as fallback."""
    vc_tools = (
        _launcher_root() / "toolchain" / "portable" / "Contents" /
        "VC" / "Tools" / "MSVC"
    )
    try:
        versions = sorted(
            (path for path in vc_tools.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
    except OSError:
        versions = []
    for version in versions:
        candidate = version / "bin" / "Hostx64" / "x64" / name
        if candidate.is_file():
            return str(candidate)
    located = shutil.which(name)
    if located and os.path.isfile(located):
        return located
    return None


def _find_windows_cublas_dll(cuda_root: str) -> Path | None:
    root = Path(cuda_root)
    candidates: list[Path] = []
    for directory in (root / "bin" / "x86_64", root / "bin"):
        try:
            candidates.extend(directory.glob("cublas64_*.dll"))
        except OSError:
            pass
    return sorted(candidates, key=lambda path: path.name, reverse=True)[0] if candidates else None


def _ensure_windows_cublas_import_library() -> str:
    """Create the tiny cuBLAS import library omitted by NVIDIA's pip wheel.

    The runtime wheel ships the DLL and headers, but not ``cublas.lib`` on
    Windows.  This extension calls only two public cuBLAS entry points, so the
    bundled MSVC librarian can create the corresponding import library fully
    offline.  The result is cached inside the private CUDA environment.
    """
    cuda_root = os.environ.get("CUDA_HOME", "")
    lib_dir = os.path.join(cuda_root, "lib", "x64")
    dll_path = _find_windows_cublas_dll(cuda_root)
    librarian = _bundled_windows_tool("lib.exe")
    if dll_path is None:
        raise RuntimeError(
            f"离线 CUDA 环境缺少 cuBLAS DLL（已检查 {cuda_root}\\bin 与 "
            f"{cuda_root}\\bin\\x86_64）；请重新安装完整离线包"
        )
    if not librarian:
        raise RuntimeError(
            "离线 CUDA 环境缺少随包 MSVC librarian（toolchain/portable）；"
            "请重新安装完整离线包"
        )
    import_lib = os.path.join(lib_dir, f"cccp_{dll_path.stem}_import.lib")
    if os.path.isfile(import_lib):
        return import_lib
    os.makedirs(lib_dir, exist_ok=True)
    def_path = os.path.join(lib_dir, "cccp_cublas_import.def")
    with open(def_path, "w", encoding="ascii", newline="\r\n") as handle:
        handle.write(
            f"LIBRARY {dll_path.name}\n"
            "EXPORTS\n"
            "    cublasSetStream_v2\n"
            "    cublasGemmStridedBatchedEx\n"
        )
    completed = subprocess.run(
        [librarian, "/nologo", "/machine:x64", f"/def:{def_path}", f"/out:{import_lib}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode or not os.path.isfile(import_lib):
        detail = (completed.stderr or completed.stdout or "未知错误").strip()
        raise RuntimeError(f"生成 cuBLAS 导入库失败：{detail}")
    return import_lib


def _ensure_windows_hipblas_import_library() -> str:
    """Create the HIPBLAS import library omitted by the ROCm SDK wheel."""
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    rocm_root = os.environ.get("ROCM_HOME", "")
    lib_dir = os.path.join(rocm_root, "lib")
    import_lib = os.path.join(lib_dir, "cccp_hipblas_import.lib")
    if os.path.isfile(import_lib):
        return import_lib
    dll_path = os.path.join(
        site_packages, "_rocm_sdk_libraries_custom", "bin", "hipblas.dll"
    )
    librarian = _bundled_windows_tool("lib.exe")
    if not os.path.isfile(dll_path) or not librarian:
        raise RuntimeError("离线 ROCm 环境缺少 HIPBLAS DLL 或 MSVC librarian")
    os.makedirs(lib_dir, exist_ok=True)
    def_path = os.path.join(lib_dir, "cccp_hipblas_import.def")
    with open(def_path, "w", encoding="ascii", newline="\r\n") as handle:
        handle.write(
            "LIBRARY hipblas.dll\n"
            "EXPORTS\n"
            "    hipblasSetStream\n"
            "    hipblasGemmStridedBatchedEx\n"
        )
    completed = subprocess.run(
        [librarian, "/nologo", "/machine:x64", f"/def:{def_path}", f"/out:{import_lib}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode or not os.path.isfile(import_lib):
        detail = (completed.stderr or completed.stdout or "未知错误").strip()
        raise RuntimeError(f"生成 HIPBLAS 导入库失败：{detail}")
    return import_lib


def _windows_path_drive(path: str | Path) -> str:
    """Return a normalized Windows drive for HIPify cross-volume checks."""
    return os.path.splitdrive(os.path.abspath(os.fspath(path)))[0].casefold()


def _prepare_windows_hipify_extra_files(
    extra_files, output_directory: str | Path
) -> tuple[list[str], list[tuple[str, str]]]:
    """Stage HIP sources beside the build cache when Windows drives differ.

    PyTorch HIPify calls ``os.path.relpath(source, output_directory)``.  On
    Windows that raises ``ValueError`` when a portable launcher is on (for
    example) D: while the persistent operator cache is under LOCALAPPDATA on
    C:.  Only the HIP wrapper calls this helper.  CUDA and CPU builds retain
    their original source paths.
    """
    output_abs = os.path.abspath(os.fspath(output_directory))
    output_drive = _windows_path_drive(output_abs)
    prepared: list[str] = []
    aliases: list[tuple[str, str]] = []
    for source in extra_files:
        source_abs = os.path.abspath(os.fspath(source))
        source_drive = _windows_path_drive(source_abs)
        if not source_drive or not output_drive or source_drive == output_drive:
            prepared.append(source_abs)
            continue

        source_path = Path(source_abs)
        source_key = hashlib.sha256(
            os.path.normcase(source_abs).encode("utf-8")
        ).hexdigest()[:12]
        stage_dir = Path(output_abs) / "_cccp_hipify_inputs" / source_key
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged_source = stage_dir / source_path.name
        shutil.copy2(source_path, staged_source)

        # Quoted includes are resolved relative to the generated HIP source.
        # Copy local headers with it so a cross-volume stage remains complete.
        for sibling in source_path.parent.iterdir():
            if sibling.is_file() and sibling.suffix.casefold() in {
                ".cuh", ".h", ".hpp", ".inc"
            }:
                shutil.copy2(sibling, stage_dir / sibling.name)

        staged_abs = os.path.abspath(os.fspath(staged_source))
        prepared.append(staged_abs)
        aliases.append((source_abs, staged_abs))
    return prepared, aliases


def _patch_windows_rocm_extension_linker(cpp_extension) -> None:
    """Correct PyTorch 2.9's CUDA-named Windows link rule for HIP wheels."""
    if os.name != "nt" or torch.version.hip is None:
        return
    torch_lib = cpp_extension.TORCH_LIB_PATH
    rocm_root = os.environ.get("ROCM_HOME", "")

    def prepare_ldflags(extra_ldflags, with_cuda, verbose, is_standalone):
        if not with_cuda:
            return cpp_extension._cccp_original_prepare_ldflags(
                extra_ldflags, with_cuda, verbose, is_standalone
            )
        extra_ldflags.extend([
            "c10.lib",
            "c10_hip.lib",
            "torch_cpu.lib",
            "torch_hip.lib",
            "torch.lib",
            f"/LIBPATH:{torch_lib}",
        ])
        if not is_standalone:
            extra_ldflags.extend([
                "torch_python.lib",
                f"/LIBPATH:{os.path.join(sys.base_exec_prefix, 'libs')}",
            ])
        extra_ldflags.extend([
            f"/LIBPATH:{os.path.join(rocm_root, 'lib')}",
            "amdhip64.lib",
            _ensure_windows_hipblas_import_library(),
            # hipcc emits a static-MSVC-runtime object on Windows but omits
            # the usual CRT/Win32 default-library directives.  Supply the
            # portable SDK libraries explicitly for a fully offline link.
            "libcmt.lib",
            "libvcruntime.lib",
            "libucrt.lib",
            "kernel32.lib",
            "advapi32.lib",
            "user32.lib",
            "shell32.lib",
            "ole32.lib",
            "oleaut32.lib",
            "uuid.lib",
            "ws2_32.lib",
        ])
        return extra_ldflags

    if not hasattr(cpp_extension, "_cccp_original_prepare_ldflags"):
        cpp_extension._cccp_original_prepare_ldflags = cpp_extension._prepare_ldflags
        cpp_extension._prepare_ldflags = prepare_ldflags

    # PyTorch 2.9.1's Windows helper has a truncated _get_hipcc_path(): the
    # branch computes hipcc_exe but forgets to return it.  Keep this local to
    # the CCCP process so the vendor environment stays byte-for-byte intact.
    if cpp_extension._get_hipcc_path() is None:
        cpp_extension._get_hipcc_path = lambda: os.path.join(
            os.environ["ROCM_HOME"], "bin", "hipcc.exe"
        )
    # Clang/hipcc rejects MSVC's /std spelling, but this wheel unconditionally
    # inserts it before the HIP-specific flags on Windows.
    original_quote = cpp_extension._nt_quote_args

    def quote_without_msvc_std(args):
        filtered = [arg for arg in (args or []) if arg != "/std:c++17"]
        return original_quote(filtered)

    cpp_extension._nt_quote_args = quote_without_msvc_std

    # The same wheel mixes slash-normalized source lists with native absolute
    # paths in hipify, causing the only extra .cu file to be marked ignored and
    # replaced with None. Normalize the lookup result at the call boundary.
    from torch.utils.hipify import hipify_python
    if not hasattr(hipify_python, "_cccp_original_hipify"):
        hipify_python._cccp_original_hipify = hipify_python.hipify

        def hipify_windows_paths(*args, **kwargs):
            original_extra_files = tuple(kwargs.get("extra_files", ()))
            call_kwargs = dict(kwargs)
            staged_files, aliases = _prepare_windows_hipify_extra_files(
                original_extra_files,
                kwargs.get("output_directory", kwargs.get("project_directory", "")),
            )
            call_kwargs["extra_files"] = staged_files
            result = hipify_python._cccp_original_hipify(*args, **call_kwargs)

            # cpp_extension indexes HIPify's result with the original source
            # path.  Point that key at the staged result before returning.
            for original_abs, staged_abs in aliases:
                item = result.get(staged_abs)
                if item is None:
                    item = result.get(staged_abs.replace(os.sep, "/"))
                if item is not None:
                    result[original_abs] = item

            for source in original_extra_files:
                source_abs = os.path.abspath(source)
                item = result.get(source_abs)
                if item is None:
                    item = result.get(source_abs.replace(os.sep, "/"))
                if item is not None and item.hipified_path is None:
                    # HIPIFY generated no file because of its Windows path
                    # comparison bug. Re-run one-file preprocessing with a
                    # slash-normalized membership list.
                    output_directory = kwargs.get("output_directory", "")
                    clean_ctx = kwargs.get("clean_ctx")
                    stats = {"unsupported_calls": [], "kernel_launches": []}
                    normalized = source_abs.replace(os.sep, "/")
                    hipify_python.preprocess_file_and_save_result(
                        output_directory,
                        source_abs,
                        [normalized],
                        kwargs.get("header_include_dirs", ()),
                        stats,
                        kwargs.get("hip_clang_launch", False),
                        kwargs.get("is_pytorch_extension", False),
                        clean_ctx,
                        kwargs.get("show_progress", False),
                    )
            return result

        hipify_python.hipify = hipify_windows_paths


def _build(verbose: bool = False):
    """编译（或命中缓存）并返回扩展模块；失败返回 None 并记录 last_error。"""
    global _EXT, _ERR
    if _EXT is not None:
        return _EXT
    if os.environ.get("CCCP_FUSED", "1") == "0":
        _ERR = "CCCP_FUSED=0 禁用"
        return None
    force_build = os.environ.get("CCCP_FORCE_GPU_BUILD", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    force_rebuild = os.environ.get("CCCP_REBUILD_GPU_OP", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not torch.cuda.is_available() and not force_build:
        _ERR = "无 CUDA"
        return None
    try:
        _apply_cuda_hardware_defaults()
        _configure_packaged_gpu_toolchain()
        # Reuse the launcher's portable MSVC/Windows SDK so end users do not
        # need Visual Studio for first-run CUDA/HIP operator compilation.
        if os.name == "nt":
            try:
                from .cpuext import _configure_bundled_windows_toolchain
                configured = _configure_bundled_windows_toolchain(force=True)
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "离线包内置 MSVC/Windows SDK 初始化失败"
                ) from exc
            if not configured:
                raise RuntimeError(
                    "离线包内置 MSVC/Windows SDK 不完整；拒绝使用本机 Visual Studio"
                )
            host_compiler = _bundled_windows_tool("cl.exe")
            toolchain_root = (
                _launcher_root() / "toolchain" / "portable"
            ).resolve()
            if (
                not host_compiler
                or not Path(host_compiler).resolve().is_relative_to(toolchain_root)
            ):
                raise RuntimeError(
                    "CUDA Host 编译器未指向离线包 toolchain；拒绝使用本机 MSVC"
                )
            os.environ["CC"] = host_compiler
            os.environ["CXX"] = host_compiler
            os.environ["CUDAHOSTCXX"] = host_compiler
        _ensure_ninja_on_path()
        # Windows 下新版 setuptools 的 distutils shim 不自动挂
        # _msvccompiler 子模块，而 torch._run_ninja_build 以属性方式访问它；
        # Linux 不得导入该 Windows 专用模块，否则会在编译 CUDA 扩展前失败。
        if os.name == "nt":
            import distutils._msvccompiler  # noqa: F401
        # 锁定当前卡的 arch（否则 torch 警告"all archs"且按全架构编译，很慢）。
        _maj = _min = None
        selected_capability = None
        if torch.version.hip is None:
            selected_capability = _select_cuda_architecture()
            if selected_capability is not None:
                _maj, _min = selected_capability
        if torch.version.hip is not None and "PYTORCH_ROCM_ARCH" not in os.environ:
            forced_rocm_arch = os.environ.get("CCCP_ROCM_ARCH", "").strip()
            if forced_rocm_arch:
                os.environ["PYTORCH_ROCM_ARCH"] = forced_rocm_arch
            try:
                if "PYTORCH_ROCM_ARCH" not in os.environ:
                    props = torch.cuda.get_device_properties(0)
                    arch = str(getattr(props, "gcnArchName", "")).split(":", 1)[0]
                    if arch:
                        os.environ["PYTORCH_ROCM_ARCH"] = arch
            except Exception:
                pass
        from torch.utils import cpp_extension
        _patch_windows_rocm_extension_linker(cpp_extension)
        load = cpp_extension.load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "csrc", "vq_gemv.cu")
        rocm_arch = os.environ.get("PYTORCH_ROCM_ARCH", "").strip()
        module_name, build_directory, cache_backend, cache_arch = (
            _operator_cache_identity(src, selected_capability, rocm_arch)
        )
        build_directory.mkdir(parents=True, exist_ok=True)
        if not force_rebuild:
            try:
                _EXT = _load_packaged_extension(
                    module_name,
                    cache_backend,
                    cache_arch,
                    cpp_extension.LIB_EXT,
                )
            except Exception as packaged_error:
                _EXT = None
                print(
                    f"[cccp-op-cache] invalid-bundled backend={cache_backend} "
                    f"arch={cache_arch} key={build_directory.name} "
                    f"error={type(packaged_error).__name__}: {packaged_error}",
                    flush=True,
                )
            if _EXT is not None:
                print(
                    f"[cccp-op-cache] bundled backend={cache_backend} "
                    f"arch={cache_arch} key={build_directory.name}",
                    flush=True,
                )
                _ERR = None
                return _EXT
            try:
                _EXT = _load_cached_extension(
                    module_name, build_directory, cpp_extension.LIB_EXT
                )
            except Exception as cache_error:  # stale/corrupt cache: rebuild it
                _EXT = None
                print(
                    f"[cccp-op-cache] invalid backend={cache_backend} "
                    f"arch={cache_arch} key={build_directory.name} "
                    f"error={type(cache_error).__name__}: {cache_error}",
                    flush=True,
                )
                cached_binary = build_directory / (
                    module_name + cpp_extension.LIB_EXT
                )
                try:
                    cached_binary.unlink(missing_ok=True)
                except OSError:
                    pass
            if _EXT is not None:
                print(
                    f"[cccp-op-cache] hit backend={cache_backend} "
                    f"arch={cache_arch} key={build_directory.name}",
                    flush=True,
                )
                _ERR = None
                return _EXT
        print(
            f"[cccp-op-cache] miss backend={cache_backend} arch={cache_arch} "
            f"key={build_directory.name} first-build",
            flush=True,
        )
        extra_cuda_cflags = ["-O3"]
        if torch.version.hip is not None:
            rocm_root = os.environ.get("ROCM_HOME", "")
            device_libs = os.path.join(
                rocm_root, "lib", "llvm", "amdgcn", "bitcode"
            )
            extra_cuda_cflags.extend([
                "--rocm-path=" + rocm_root,
                "--rocm-device-lib-path=" + device_libs,
            ])
        # CUDA 13's documented escape hatch is needed for newer VS 2022
        # toolsets that have not yet been added to nvcc's version table.
        # It is not used for HIP/ROCm.
        if os.name == "nt" and torch.version.hip is None:
            host_compiler = _bundled_windows_tool("cl.exe")
            if not host_compiler:
                raise RuntimeError("离线包缺少 CUDA Host 编译器 cl.exe")
            extra_cuda_cflags.extend([
                "-ccbin", str(Path(host_compiler).parent),
                "--allow-unsupported-compiler",
                # CUDA 13 parses C++20's ``module`` token noisily inside
                # PyTorch headers and reports a few intentionally unused
                # template locals. These diagnostics are harmless and used
                # to look like launch failures in the end-user terminal.
                "-Xcudafe", "--diag_suppress=3189",
                "-Xcudafe", "--diag_suppress=177",
                "-Xcudafe", "--diag_suppress=221",
            ])
        # vq_gemv.cu directly calls cuBLAS for the MLA batched GEMM path.
        # torch's extension helper does not add cuBLAS automatically, so the
        # DLL otherwise compiles but fails at the final Windows link step.
        extra_ldflags: list[str] = []
        if torch.version.hip is None:
            extra_ldflags.append(
                _ensure_windows_cublas_import_library()
                if os.name == "nt" else "-lcublas"
            )
        extra_cflags = ["-std=c++20"] if torch.version.hip is not None else None
        extra_include_paths = None
        if torch.version.hip is not None:
            # Header-only rocThrust/hipCUB/rocPRIM are extracted during the
            # AMD environment build to a compact, relocation-safe directory.
            compact_headers = os.path.abspath(
                os.path.join(sys.prefix, "..", "devinclude")
            )
            if os.path.isdir(compact_headers):
                extra_include_paths = [compact_headers]
        backend_label = "AMD-HIP" if torch.version.hip is not None else "NVIDIA-CUDA"
        with operator_build_progress(backend_label) as build_progress:
            _EXT = load(name=module_name, sources=[src],
                        extra_cflags=extra_cflags,
                        extra_cuda_cflags=extra_cuda_cflags,
                        extra_ldflags=extra_ldflags,
                        extra_include_paths=extra_include_paths,
                        build_directory=str(build_directory),
                        # Keep Ninja/compiler output in the unified terminal;
                        # heartbeats cover long compiler phases with no output.
                        verbose=verbose or build_progress.enabled)
        _ERR = None
    except Exception as e:  # noqa: BLE001 —— 任何编译/加载失败都回退
        _EXT = None
        _ERR = f"{type(e).__name__}: {e}"
        if force_build:
            _ERR += "\n" + traceback.format_exc()
    return _EXT


def available() -> bool:
    """融合 kernel 是否可用。"""
    return _EXT is not None


def last_error() -> str | None:
    """最近一次构建失败的原因（诊断用；可用时返回 None）。"""
    return _ERR


def prebuild() -> bool:
    """显式预编译入口（scripts/prebuild_gpu_ops.ps1 调用），返回是否成功。"""
    ok = _build(verbose=True) is not None
    print("[fusedext] 融合 kernel " + ("编译成功并已缓存" if ok else
          f"不可用（{_ERR}），将使用 torch 批量路径"))
    return ok


def install_prebuilt() -> Path:
    """Copy the active exact operator into the distributable native tree."""
    extension = _build(verbose=True)
    if extension is None:
        raise RuntimeError(f"GPU operator is unavailable: {_ERR}")
    src = Path(__file__).resolve().parent / "csrc" / "vq_gemv.cu"
    capability = None
    rocm_arch = os.environ.get("PYTORCH_ROCM_ARCH", "").strip()
    if torch.version.hip is None:
        capability = _select_cuda_architecture()
    module_name, _build_directory, backend, arch = _operator_cache_identity(
        src, capability, rocm_arch
    )
    from torch.utils import cpp_extension

    destination = _packaged_extension_path(
        module_name, backend, arch, cpp_extension.LIB_EXT
    )
    source_binary = Path(extension.__file__).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_binary != destination.resolve():
        shutil.copy2(source_binary, destination)
    return destination


_build()

if _EXT is None and os.environ.get("CCCP_REQUIRE_FUSED", "0").strip().lower() in {
    "1", "true", "yes", "on"
}:
    raise RuntimeError(f"CCCP GPU 融合算子编译/加载失败：{_ERR}")

if _EXT is not None:

    def vq_gemv_fused(x_rows: torch.Tensor, idx: torch.Tensor,
                      cb: torch.Tensor) -> torch.Tensor:
        """融合 VQ 分组 GEMV：x_rows [N|1, C] f32，idx u8/u16 [N,R,B]，
        cb f32 [N|1,K,D]（1 = 同层共享码本广播）→ [N,R] f32。"""
        return _EXT.vq_gemv(x_rows.contiguous(), idx.contiguous(), cb.contiguous())

    def dense_vq_gemv_packed_fused(
        x_rows: torch.Tensor,
        payload: torch.Tensor,
        codebook: torch.Tensor,
        rows: int,
        blocks: int,
        bits: int,
    ) -> torch.Tensor:
        """Dense Linear GEMV that keeps row-major p8--p16 indices compact."""
        return _EXT.dense_vq_gemv_packed(
            x_rows.float().contiguous(),
            payload.contiguous().reshape(-1),
            codebook.float().contiguous(),
            int(rows),
            int(blocks),
            int(bits),
        )

    def dense_vq_gemv_grouped_fp8_codebook_fused(
        x_rows: torch.Tensor,
        metadata: torch.Tensor,
        total_rows: int,
    ) -> torch.Tensor:
        """One GGUF-style direct-dot launch for shared-input VQ Decode."""
        return _EXT.dense_vq_gemv_grouped_fp8_codebook(
            x_rows.to(torch.bfloat16).contiguous(),
            metadata.contiguous(),
            int(total_rows),
        )

    def dense_vq_dequant_packed_fused(
        payload: torch.Tensor,
        codebook: torch.Tensor,
        rows: int,
        blocks: int,
        bits: int,
        row_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expand all or selected Dense VQ rows to transient CUDA BF16."""
        if row_ids is None:
            row_ids = torch.empty(
                0, dtype=torch.int64, device=payload.device
            )
        return _EXT.dense_vq_dequant_packed(
            payload.contiguous().reshape(-1),
            codebook.float().contiguous(),
            int(rows),
            int(blocks),
            int(bits),
            row_ids.contiguous(),
        )

    def dense_vq_expand_native8_fused(
        payload: torch.Tensor,
        quantized_codebook: torch.Tensor,
        output: torch.Tensor,
        rows: int,
        blocks: int,
        bits: int,
        row_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expand VQ into a reusable native E4M3/INT8 execution buffer.

        The small codebook is quantized once by the caller. Conversion then
        performs only index unpacking and aligned vector copies, so its cost is
        bounded primarily by writing the one-byte execution image.
        """
        if row_ids is None:
            row_ids = torch.empty(
                0, dtype=torch.int64, device=payload.device
            )
        if quantized_codebook.dtype not in (torch.float8_e4m3fn, torch.int8):
            raise ValueError("native8 codebook must be E4M3 or INT8")
        if output.dtype != quantized_codebook.dtype:
            raise ValueError("native8 output dtype must match its codebook")
        return _EXT.dense_vq_expand_native8(
            payload.contiguous().reshape(-1),
            quantized_codebook.contiguous(),
            output,
            int(rows),
            int(blocks),
            int(bits),
            row_ids.contiguous(),
        )

    def dense_fp8_quantize_rows_fused(
        value: torch.Tensor,
        output: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor | None:
        """Quantize CUDA activation rows into preallocated native E4M3.

        Preallocated buffers make this entry safe inside a whole-token CUDA
        graph.  A scalar scale selects the CUDA-12.8-compatible tensor-scaled
        GEMM path; ``[rows,1]`` remains available to callers on runtimes with
        native row-scaled GEMM support.
        """
        if (
            not value.is_cuda
            or value.ndim != 2
            or value.dtype not in (torch.bfloat16, torch.float32)
            or not value.is_contiguous()
            or not output.is_cuda
            or output.dtype != torch.float8_e4m3fn
            or output.shape != value.shape
            or not output.is_contiguous()
            or not scales.is_cuda
            or scales.dtype != torch.float32
            or scales.shape not in {
                (1, 1),
                (value.shape[0], 1),
            }
            or not scales.is_contiguous()
        ):
            return None
        return _EXT.dense_fp8_quantize_rows(value, output, scales)

    def short_conv3_fused(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> bool:
        """Update three BF16 depthwise short-convolution states in one launch."""
        if (
            not query.is_cuda
            or query.dtype != torch.bfloat16
            or key.dtype != torch.bfloat16
            or value.dtype != torch.bfloat16
            or any(item.dtype != torch.bfloat16 for item in states)
            or weights[0].dtype not in (
                torch.bfloat16,
                torch.float32,
            )
            or any(item.dtype != weights[0].dtype for item in weights)
        ):
            return False
        return bool(_EXT.kimi_short_conv3(
            query,
            key,
            value,
            states[0],
            states[1],
            states[2],
            weights[0],
            weights[1],
            weights[2],
        ))

    def qwen35_conv1d_update_fused(
        value: torch.Tensor,
        state: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor | None:
        """Fuse one cached Qwen3.5 depthwise-convolution step."""
        if (
            not value.is_cuda
            or value.dtype != torch.bfloat16
            or value.ndim != 3
            or value.shape[0] != 1
            or value.shape[-1] != 1
            or state.dtype != torch.bfloat16
            or state.ndim != 3
            or weight.dtype not in (torch.bfloat16, torch.float32)
            or output.shape != value.shape
            or output.dtype != torch.bfloat16
        ):
            return None
        return _EXT.qwen35_conv1d_update(
            value.contiguous(), state, weight.contiguous(), output
        )

    def qwen35_delta_recurrent_fused(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor | None:
        """Fuse one cached Qwen3.5 gated-delta recurrent step."""
        if (
            not query.is_cuda
            or query.dtype != torch.bfloat16
            or query.ndim != 2
            or key.shape != query.shape
            or key.dtype != torch.bfloat16
            or value.ndim != 2
            or value.dtype != torch.bfloat16
            or value.shape[0] != query.shape[0]
            or state.dtype != torch.float32
            or state.shape != (
                query.shape[0], query.shape[1], value.shape[1]
            )
            or gate.numel() != query.shape[0]
            or beta.numel() != query.shape[0]
            or output.shape != value.shape
            or output.dtype != torch.bfloat16
        ):
            return None
        return _EXT.qwen35_delta_recurrent(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            gate.contiguous(),
            beta.contiguous(),
            state,
            output,
        )

    def qwen35_delta_recurrent_batch_fused(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor | None:
        """Fuse an ordered Qwen3.5 gated-delta token block."""
        if (
            not query.is_cuda
            or query.dtype != torch.bfloat16
            or query.ndim != 3
            or key.shape != query.shape
            or key.dtype != torch.bfloat16
            or value.ndim != 3
            or value.dtype != torch.bfloat16
            or value.shape[:2] != query.shape[:2]
            or state.dtype != torch.float32
            or state.shape != (
                query.shape[1], query.shape[2], value.shape[2]
            )
            or gate.numel() != query.shape[0] * query.shape[1]
            or beta.numel() != query.shape[0] * query.shape[1]
            or output.shape != value.shape
            or output.dtype != torch.bfloat16
        ):
            return None
        return _EXT.qwen35_delta_recurrent_batch(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            gate.contiguous(),
            beta.contiguous(),
            state,
            output,
        )

    def qwen35_delta_recurrent_batch_checkpoint_fused(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
        output: torch.Tensor,
        checkpoints: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run an ordered block and retain the state after every token."""
        expected = (
            query.shape[0],
            query.shape[1],
            query.shape[2],
            value.shape[2],
        )
        if (
            not query.is_cuda
            or query.dtype != torch.bfloat16
            or query.ndim != 3
            or key.shape != query.shape
            or key.dtype != torch.bfloat16
            or value.ndim != 3
            or value.dtype != torch.bfloat16
            or value.shape[:2] != query.shape[:2]
            or state.dtype != torch.float32
            or state.shape != expected[1:]
            or gate.numel() != query.shape[0] * query.shape[1]
            or beta.numel() != query.shape[0] * query.shape[1]
            or output.shape != value.shape
            or output.dtype != torch.bfloat16
            or checkpoints.dtype != torch.float32
            or checkpoints.shape != expected
        ):
            return None
        return _EXT.qwen35_delta_recurrent_batch_checkpoint(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            gate.contiguous(),
            beta.contiguous(),
            state,
            output,
            checkpoints,
        )

    def kda_recurrent_fused(
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
        lower_bound: float = -5.0,
    ) -> torch.Tensor:
        """KDA decode update with persistent FP32 V-first state."""
        return _EXT.kimi_kda_recurrent(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            gate.contiguous(),
            beta.float().contiguous(),
            a_log.float().contiguous(),
            dt_bias.float().contiguous(),
            state,
            workspace,
            output,
            float(lower_bound),
        )

    def kda_recurrent_batch_fused(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        output: torch.Tensor,
        lower_bound: float = -5.0,
    ) -> torch.Tensor | None:
        """Run ordered KDA recurrence for a token block in one CUDA launch."""
        if (
            not query.is_cuda
            or query.dtype != torch.bfloat16
            or query.ndim != 3
            or key.shape != query.shape
            or gate.shape != query.shape
            or value.ndim != 3
            or value.shape[:2] != query.shape[:2]
            or beta.shape != query.shape[:2]
            or output.shape != value.shape
        ):
            return None
        return _EXT.kimi_kda_recurrent_batch(
            query.contiguous(), key.contiguous(), value.contiguous(),
            gate.contiguous(), beta.float().contiguous(),
            a_log.float().contiguous(), dt_bias.float().contiguous(),
            state, output, float(lower_bound),
        )

    def gated_rmsnorm_fused(
        value: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        eps: float,
    ) -> torch.Tensor | None:
        """Fuse decode RMSNorm, sigmoid gate and multiply."""
        if (
            not value.is_cuda
            or value.dtype != torch.bfloat16
            or gate.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
        ):
            return None
        return _EXT.kimi_gated_rmsnorm(
            value,
            gate,
            weight,
            output,
            float(eps),
        )

    def packed_moe_topk_fused(
        value: torch.Tensor,
        route_ids: torch.Tensor,
        weights: torch.Tensor,
        metadata: torch.Tensor,
        activation: str,
        beta: float,
        linear_beta: float,
        limit: float,
        hidden_workspace: torch.Tensor,
        out_workspace: torch.Tensor,
        result: torch.Tensor,
        p12_count: int = 0,
        projection_layout_tag: int = 0,
    ) -> torch.Tensor | None:
        """Run Top-K gated MLP directly from packed indices.

        ``p12_count`` is the grouped p12 prefix length.  ``-1`` scans an
        ungrouped route vector on device, which avoids a GPU→CPU sync in the
        full-resident multi-GPU path.
        """
        if (
            not value.is_cuda
            or value.dtype != torch.bfloat16
            or value.ndim != 2
            or value.shape[0] != 1
            or route_ids.dtype != torch.long
            or route_ids.ndim != 1
            or not 0 < route_ids.shape[-1] <= 16
            or weights.dtype != torch.float32
            or weights.shape != route_ids.shape
            or metadata.dtype != torch.long
            or metadata.ndim != 2
            or metadata.shape[0] not in (10, 15, 27)
            or not -1 <= p12_count <= route_ids.shape[-1]
            or projection_layout_tag not in (0, 1, 2)
        ):
            return None
        activation_name = str(activation).strip().lower()
        activation_kind = {
            "situ": 0,
            "silu": 1,
            "swiglu": 1,
        }.get(activation_name)
        if activation_kind is None:
            return None
        return _EXT.packed_moe_topk(
            value.contiguous(),
            route_ids.contiguous(),
            weights.contiguous(),
            metadata.contiguous(),
            int(activation_kind),
            float(beta),
            float(linear_beta),
            float(limit),
            hidden_workspace,
            out_workspace,
            result,
            int(p12_count),
            int(projection_layout_tag),
        )

    def packed_stage_topk_three_projection_fused(
        route_ids: torch.Tensor,
        metadata: torch.Tensor,
        uva_metadata: torch.Tensor,
        stage_workspace: torch.Tensor,
        staged_metadata: torch.Tensor,
        staged_route_ids: torch.Tensor,
        *,
        hidden: int,
        intermediate: int,
        projection_strides: tuple[int, int, int],
    ) -> torch.Tensor | None:
        """Coalesce mapped-host cold experts before the packed MoE kernels."""

        if (
            not route_ids.is_cuda
            or route_ids.dtype != torch.long
            or route_ids.ndim != 1
            or not 0 < route_ids.numel() <= 16
            or metadata.dtype != torch.long
            or metadata.ndim != 2
            or metadata.shape[0] != 15
            or uva_metadata.shape != metadata.shape
            or uva_metadata.dtype != torch.long
            or stage_workspace.dtype != torch.uint8
            or staged_metadata.shape != (15, route_ids.numel())
            or staged_metadata.dtype != torch.long
            or staged_route_ids.shape != route_ids.shape
            or staged_route_ids.dtype != torch.long
            or len(projection_strides) != 3
        ):
            return None
        return _EXT.packed_stage_topk_three_projection(
            route_ids.contiguous(),
            metadata.contiguous(),
            uva_metadata.contiguous(),
            stage_workspace,
            staged_metadata,
            staged_route_ids,
            int(hidden),
            int(intermediate),
            *(int(value) for value in projection_strides),
        )

    def packed_route_slots_fused(
        route_ids: torch.Tensor,
        directory: torch.Tensor,
        selected: torch.Tensor,
        hit_mask: torch.Tensor,
    ) -> bool:
        """Gather dynamic Top-K expert slot metadata entirely on CUDA."""
        if (
            not route_ids.is_cuda
            or route_ids.dtype != torch.long
            or route_ids.ndim != 1
            or not 0 < route_ids.numel() <= 16
            or directory.dtype != torch.long
            or directory.ndim != 2
            or selected.dtype != torch.long
            or selected.shape
            != (directory.shape[1], route_ids.numel())
            or hit_mask.dtype != torch.bool
            or hit_mask.shape != route_ids.shape
        ):
            return False
        return bool(
            _EXT.packed_route_slots_out(
                route_ids.contiguous(),
                directory.contiguous(),
                selected,
                hit_mask,
            )
        )

    def packed_h2d_batch_fused(
        sources: list[torch.Tensor],
        destinations: list[torch.Tensor],
    ) -> bool:
        """Submit one routed layer through the compiled H2D backend.

        Linux/TCC uses CUDA batch DMA.  Windows/WDDM uses a compiled async
        copy loop because the native batch API is not reliable there.
        """
        if (
            not sources
            or len(sources) != len(destinations)
            or len(sources) > 128
            or any(source.device.type != "cpu" for source in sources)
            or any(not source.is_pinned() for source in sources)
            or any(not target.is_cuda for target in destinations)
            or any(not source.is_contiguous() for source in sources)
            or any(not target.is_contiguous() for target in destinations)
            or any(
                source.nbytes != target.nbytes
                for source, target in zip(sources, destinations)
            )
        ):
            return False
        device = destinations[0].device
        if any(target.device != device for target in destinations):
            return False
        return bool(_EXT.packed_h2d_batch(sources, destinations))

    def moe_mlp_slots_fused(
        x_rows: torch.Tensor,
        gu_indices: list[torch.Tensor],
        gu_codebooks: list[torch.Tensor],
        dn_indices: list[torch.Tensor],
        dn_codebooks: list[torch.Tensor],
        weights: torch.Tensor,
        limit: float,
        hidden_workspace: torch.Tensor,
        out_workspace: torch.Tensor,
        result: torch.Tensor,
    ) -> torch.Tensor:
        """稳定专家槽 BF16 MLP；四个 kernel 完成 GU/SwiGLU/DN/加权。"""
        return _EXT.moe_mlp_slots(
            x_rows,
            gu_indices,
            gu_codebooks,
            dn_indices,
            dn_codebooks,
            weights,
            float(limit),
            hidden_workspace,
            out_workspace,
            result,
        )

    def moe_mlp_routed_slots_fused(
        x_rows: torch.Tensor,
        route_ids: torch.Tensor,
        weights: torch.Tensor,
        metadata: torch.Tensor,
        limit: float,
        hidden_workspace: torch.Tensor,
        out_workspace: torch.Tensor,
        result: torch.Tensor,
        accumulate: bool = False,
    ) -> torch.Tensor | None:
        """Full-resident EP MLP whose Top-K selection remains on CUDA."""
        if (
            os.environ.get("CCCP_EP_DEVICE_ROUTE", "1") == "0"
            or not x_rows.is_cuda
            or x_rows.dtype != torch.bfloat16
            or x_rows.shape[0] != 1
            or route_ids.dtype != torch.long
            or route_ids.ndim != 1
            or route_ids.numel() == 0
            or route_ids.numel() > 8
            or weights.dtype != torch.float32
            or weights.shape != route_ids.shape
            or metadata.dtype != torch.long
            or metadata.ndim != 2
            or metadata.shape[0] != 10
        ):
            return None
        return _EXT.moe_mlp_routed_slots(
            x_rows.contiguous(),
            route_ids.contiguous(),
            weights.contiguous(),
            metadata.contiguous(),
            float(limit),
            hidden_workspace,
            out_workspace,
            result,
            os.environ.get("CCCP_VQ_D4_SPECIALIZED", "1") != "0",
            bool(accumulate),
        )

    def moe_mlp_routed_vv_fused(
        x_rows: torch.Tensor,
        route_ids: torch.Tensor,
        weights: torch.Tensor,
        metadata: torch.Tensor,
        limit: float,
        hidden_workspace: torch.Tensor,
        out_workspace: torch.Tensor,
        result: torch.Tensor,
        accumulate: bool = False,
    ) -> torch.Tensor | None:
        """Run independent D4/K4096 experts with a shared-codebook kernel."""
        if (
            os.environ.get("CCCP_EP_DEVICE_ROUTE", "1") == "0"
            or not x_rows.is_cuda
            or x_rows.dtype != torch.bfloat16
            or x_rows.shape[0] != 1
            or route_ids.dtype != torch.long
            or route_ids.ndim != 1
            or route_ids.numel() == 0
            or route_ids.numel() > 8
            or weights.dtype != torch.float32
            or weights.shape != route_ids.shape
            or metadata.dtype != torch.long
            or metadata.ndim != 2
            or metadata.shape[0] != 10
        ):
            return None
        return _EXT.moe_mlp_routed_vv(
            x_rows.contiguous(),
            route_ids.contiguous(),
            weights.contiguous(),
            metadata.contiguous(),
            float(limit),
            hidden_workspace,
            out_workspace,
            result,
            bool(accumulate),
        )

    def moe_mlp_routed_codegemm_fused(
        x_rows: torch.Tensor,
        route_ids: torch.Tensor,
        weights: torch.Tensor,
        metadata: torch.Tensor,
        gu_sum: torch.Tensor,
        activation: torch.Tensor,
        dn_sum: torch.Tensor,
        result: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run the full-resident v256/D4 Psumbook expert kernel."""
        if (
            os.environ.get("CCCP_EP_DEVICE_ROUTE", "1") == "0"
            or not x_rows.is_cuda
            or x_rows.dtype != torch.bfloat16
            or x_rows.shape[0] != 1
            or route_ids.dtype != torch.long
            or route_ids.ndim != 1
            or route_ids.numel() == 0
            or route_ids.numel() > 8
            or weights.dtype != torch.float32
            or weights.shape != route_ids.shape
            or metadata.dtype != torch.long
            or metadata.ndim != 2
            or metadata.shape[0] != 10
        ):
            return None
        return _EXT.moe_mlp_routed_codegemm(
            x_rows.contiguous(),
            route_ids.contiguous(),
            weights.contiguous(),
            metadata.contiguous(),
            gu_sum,
            activation,
            dn_sum,
            result,
        )

    def pack_vq_tensor_shard_codegemm(
        source_gu: torch.Tensor,
        source_dn: torch.Tensor,
        target_gu: torch.Tensor,
        target_dn: torch.Tensor,
        global_intermediate: int,
        shard_start: int,
        local_intermediate: int,
    ) -> bool:
        """Pack one full-GPU tensor shard without changing its byte size."""
        if (
            source_gu.dtype != torch.uint8
            or source_dn.dtype != torch.uint8
            or target_gu.dtype != torch.uint8
            or target_dn.dtype != torch.uint8
        ):
            return False
        _EXT.pack_vq_tensor_shard_codegemm(
            source_gu,
            source_dn,
            target_gu,
            target_dn,
            int(global_intermediate),
            int(shard_start),
            int(local_intermediate),
        )
        return True

    def unpack_vq_codegemm(
        storage: torch.Tensor,
        rows: int,
        blocks: int,
    ) -> torch.Tensor:
        """Restore a temporary row-major index matrix for prefill."""
        return _EXT.unpack_vq_codegemm(
            storage,
            int(rows),
            int(blocks),
        )

    def expert_dispatch_pack_fused(
        x: torch.Tensor,
        route_ids: torch.Tensor,
        weights: torch.Tensor,
        x_out: torch.Tensor,
        route_ids_out: torch.Tensor,
        weights_out: torch.Tensor,
    ) -> bool:
        """一次 peer kernel 完成远端专家输入、ID 和权重分发。"""
        if (
            os.environ.get("CCCP_EP_FUSED_DISPATCH", "1") == "0"
            or not x.is_cuda
            or x.dtype not in (torch.float32, torch.bfloat16)
            or x.ndim != 2
            or x.shape[0] != 1
            or route_ids.dtype != torch.long
            or route_ids.ndim != 1
            or weights.dtype != torch.float32
            or weights.shape != route_ids.shape
            or x_out.dtype != torch.bfloat16
            or x_out.shape != x.shape
            or route_ids_out.dtype != torch.long
            or route_ids_out.shape != route_ids.shape
            or weights_out.dtype != torch.float32
            or weights_out.shape != weights.shape
        ):
            return False
        _EXT.expert_dispatch_pack(
            x.contiguous(),
            route_ids.contiguous(),
            weights.contiguous(),
            x_out,
            route_ids_out,
            weights_out,
        )
        return True

    def tp_peer_copy_fused(
        source: torch.Tensor,
        destination: torch.Tensor,
    ) -> bool:
        """Graph-safe rank dispatch without CUDA memcpy capture edges."""
        if (
            not source.is_cuda
            or not destination.is_cuda
            or source.dtype not in (
                torch.float32,
                torch.bfloat16,
                torch.long,
            )
            or source.dtype != destination.dtype
            or source.shape != destination.shape
        ):
            return False
        _EXT.tp_peer_copy(
            source.contiguous(),
            destination,
        )
        return True

    def tp_attention_peer_dispatch_fused(
        source_q: torch.Tensor,
        source_c: torch.Tensor,
        source_k: torch.Tensor,
        source_position: torch.Tensor,
        destination_q: torch.Tensor,
        destination_c: torch.Tensor,
        destination_k: torch.Tensor,
        destination_position: torch.Tensor,
    ) -> bool:
        """Copy all fixed Attention TP inputs with one graph kernel."""
        float_pairs = (
            (source_q, destination_q),
            (source_c, destination_c),
            (source_k, destination_k),
        )
        if any(
            not source.is_cuda
            or not destination.is_cuda
            or source.dtype != torch.float32
            or destination.dtype != torch.float32
            or source.shape != destination.shape
            for source, destination in float_pairs
        ):
            return False
        if (
            not source_position.is_cuda
            or not destination_position.is_cuda
            or source_position.dtype != torch.long
            or destination_position.dtype != torch.long
            or source_position.numel() != 1
            or destination_position.numel() != 1
        ):
            return False
        _EXT.tp_attention_peer_dispatch(
            source_q.contiguous(),
            source_c.contiguous(),
            source_k.contiguous(),
            source_position.contiguous(),
            destination_q,
            destination_c,
            destination_k,
            destination_position,
        )
        return True

    def tp_attention_source_pack_fused(
        source_q: torch.Tensor,
        source_c: torch.Tensor,
        source_k: torch.Tensor,
        destination_q: torch.Tensor,
        destination_c: torch.Tensor,
        destination_k: torch.Tensor,
        destination_position: torch.Tensor,
        position: int,
    ) -> bool:
        """Pack changing primary-rank Attention inputs with one kernel."""
        float_pairs = (
            (source_q, destination_q),
            (source_c, destination_c),
            (source_k, destination_k),
        )
        if any(
            not source.is_cuda
            or not destination.is_cuda
            or source.dtype != torch.float32
            or destination.dtype != torch.float32
            or source.shape != destination.shape
            or source.device != destination.device
            for source, destination in float_pairs
        ):
            return False
        if (
            not destination_position.is_cuda
            or destination_position.dtype != torch.long
            or destination_position.numel() != 1
            or destination_position.device != source_q.device
        ):
            return False
        _EXT.tp_attention_source_pack(
            source_q.contiguous(),
            source_c.contiguous(),
            source_k.contiguous(),
            destination_q,
            destination_c,
            destination_k,
            destination_position,
            int(position),
        )
        return True

    def hc_split_fused(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
                       hc: int, iters: int, eps: float):
        """融合 HC sinkhorn：mixes [..., 24] f32 CUDA + hc==4 时返回
        (pre, post, comb)（单次 kernel 完成 softmax + 全部归一化迭代）；
        不满足条件返回 None（调用方回退 CCCP/dsv4.hc_split 的 torch 循环）。
        数值与 torch 版同序 fp32 计算，差异在 1e-7 量级。"""
        if (hc != 4 or not mixes.is_cuda or mixes.dtype != torch.float32
                or scale.dtype != torch.float32):
            return None
        out = _EXT.hc_sinkhorn(mixes, scale, base, iters, float(eps))
        pre = out[..., :4]
        post = out[..., 4:8]
        comb = out[..., 8:].unflatten(-1, (4, 4))
        return pre, post, comb

    def rmsnorm_fused(
        x: torch.Tensor,
        w: torch.Tensor,
        eps: float,
        output: torch.Tensor | None = None,
    ):
        """融合 RMSNorm（f32 CUDA）：不满足条件返回 None（回退 torch 路径）。"""
        if not x.is_cuda or x.dtype != torch.float32 or w.dtype != torch.float32:
            return None
        return _EXT.rmsnorm(x, w, float(eps), output)

    def rmsnorm_bf16_fused(
        x: torch.Tensor,
        w: torch.Tensor,
        eps: float,
        output: torch.Tensor | None = None,
    ):
        """One-launch BF16 RMSNorm with source BF16 or FP32 weights."""
        if (
            not x.is_cuda
            or x.dtype != torch.bfloat16
            or w.dtype not in (torch.bfloat16, torch.float32)
            or w.ndim != 1
            or w.numel() != x.shape[-1]
        ):
            return None
        return _EXT.rmsnorm_bf16(
            x,
            w,
            float(eps),
            output,
        )

    def attention_residual_bf16_fused(
        prefix: torch.Tensor,
        residual: torch.Tensor,
        projection: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        output: torch.Tensor | None = None,
        post_norm_weight: torch.Tensor | None = None,
        score_workspace: torch.Tensor | None = None,
        residual_inverse: torch.Tensor | None = None,
    ):
        if (
            not prefix.is_cuda
            or prefix.dtype != torch.bfloat16
            or residual.dtype != torch.bfloat16
            or projection.dtype != torch.bfloat16
            or norm_weight.dtype != torch.bfloat16
            or (
                post_norm_weight is not None
                and post_norm_weight.dtype != torch.bfloat16
            )
            or prefix.ndim != 2
            or prefix.shape[0] <= 0
            or residual.ndim != 3
            or residual.shape[0] != prefix.shape[0]
            or not 0 < residual.shape[1] <= 31
            or (
                residual.shape[1] + 1
                > int(
                    os.environ.get(
                        "CCCP_RESIDUAL_SINGLE_MAX_ROWS",
                        "2",
                    )
                )
                and (
                    score_workspace is None
                    or not score_workspace.is_cuda
                    or score_workspace.dtype != torch.float32
                    or score_workspace.numel() < prefix.shape[0] * 32
                    or score_workspace.device != prefix.device
                )
            )
            or (
                residual_inverse is not None
                and (
                    not residual_inverse.is_cuda
                    or residual_inverse.dtype != torch.float32
                    or residual_inverse.numel() < prefix.shape[0] * residual.shape[1]
                    or residual_inverse.device != prefix.device
                )
            )
        ):
            return None
        return _EXT.attention_residual_bf16(
            prefix,
            residual,
            projection,
            norm_weight,
            post_norm_weight,
            float(eps),
            output,
            score_workspace,
            int(
                os.environ.get(
                    "CCCP_RESIDUAL_SINGLE_MAX_ROWS",
                    "2",
                )
            ),
            residual_inverse,
        )

    def gated_activation_bf16_fused(
        gate: torch.Tensor,
        up: torch.Tensor,
        activation: str,
        beta: float,
        linear_beta: float | None,
        limit: float = 0.0,
        output: torch.Tensor | None = None,
    ):
        normalized = activation.strip().lower()
        if (
            not gate.is_cuda
            or gate.dtype != torch.bfloat16
            or up.dtype != torch.bfloat16
            or gate.shape != up.shape
            or normalized not in {"silu", "swiglu", "situ"}
        ):
            return None
        return _EXT.gated_activation_bf16(
            gate.contiguous(),
            up.contiguous(),
            1 if normalized == "situ" else 0,
            float(beta),
            -1.0 if linear_beta is None else float(linear_beta),
            float(limit),
            output,
        )

    def glm_mla_bmm_decode_fused(
        input: torch.Tensor,
        weight: torch.Tensor,
        transpose_weight: bool,
        output: torch.Tensor | None = None,
    ):
        """Decode-only BF16 MLA batch GEMM through direct cuBLAS."""
        if (
            not input.is_cuda
            or input.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or input.ndim != 3
            or weight.ndim != 3
            or input.shape[1] != 1
        ):
            return None
        return _EXT.glm_mla_bmm_decode(
            input,
            weight,
            bool(transpose_weight),
            output,
        )

    def rope1_fused(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                    inverse: bool = False):
        """融合 RoPE（交错对）：仅 decode 单相位场景——x [..., rd] f32 CUDA 且
        cos/sin 各 rd/2 个元素（全部行同相位）时生效，否则 None（回退 torch）。
        数值与 CCCP.dsv4.rope_apply 逐项一致。"""
        if (not x.is_cuda or x.dtype != torch.float32
                or cos.numel() * 2 != x.shape[-1] or sin.numel() * 2 != x.shape[-1]):
            return None
        return _EXT.rope1(x, cos.reshape(-1), sin.reshape(-1), bool(inverse))

    def glm_rope_qk_fused(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        """Fuse GLM MLA Q/K RoPE while preserving the reference cat layout."""
        if (
            os.environ.get("CCCP_GLM_ROPE_FUSED", "1") == "0"
            or not q.is_cuda
            or q.dtype != torch.float32
            or k.dtype != torch.float32
            or q.ndim != 3
            or k.ndim != 3
            or cos.ndim != 2
            or sin.ndim != 2
            or q.shape[1:] != k.shape[1:]
            or k.shape[0] != 1
            or cos.shape != sin.shape
            or cos.shape != (q.shape[1], q.shape[2] // 2)
        ):
            return None
        return _EXT.glm_rope_qk(q, k, cos, sin)

    def glm_latent_kv_decode_prepare_fused(
        c_raw: torch.Tensor,
        c_weight: torch.Tensor,
        q_rot: torch.Tensor,
        k_rot: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        ckv_buffer: torch.Tensor,
        krot_buffer: torch.Tensor,
        position: torch.Tensor,
        eps: float,
        output: torch.Tensor | None = None,
    ):
        """Fuse decode C RMSNorm, Q/K RoPE and BF16 latent-cache writes."""
        if (
            os.environ.get("CCCP_GLM_LATENT_PREP_FUSED", "1") == "0"
            or not c_raw.is_cuda
            or c_raw.dtype != torch.float32
            or c_weight.dtype != torch.float32
            or q_rot.dtype != torch.float32
            or k_rot.dtype != torch.float32
            or cos_cache.dtype != torch.float32
            or sin_cache.dtype != torch.float32
            or ckv_buffer.dtype != torch.bfloat16
            or krot_buffer.dtype != torch.bfloat16
            or not position.is_cuda
            or position.dtype != torch.long
            or position.numel() != 1
            or c_raw.shape[0] != 1
            or q_rot.ndim != 3
            or q_rot.shape[1] != 1
            or k_rot.shape != (1, 1, q_rot.shape[2])
        ):
            return None
        return _EXT.glm_latent_kv_decode_prepare(
            c_raw,
            c_weight,
            q_rot,
            k_rot,
            cos_cache,
            sin_cache,
            ckv_buffer,
            krot_buffer,
            position.contiguous(),
            float(eps),
            output,
        )

    def flashinfer_mla_batch1_plan_fused(
        int_workspace: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        length: int,
        page_size: int,
        heads: int,
        plan_info,
    ) -> bool:
        """Build the supported batch-1 MLA schedule directly on the GPU.

        FlashInfer may change the concrete planner layout with the head count.
        The public CCCP kernel implements FlashInfer's one- and two-CTA
        cluster layouts.  A future layout is not an error: callers must fall
        back to FlashInfer's own planner instead of aborting model execution.
        """
        normalized_plan = [int(value) for value in plan_info]
        if (
            os.environ.get(
                "CCCP_FLASHINFER_GPU_PLAN",
                "1",
            )
            == "0"
            or not int_workspace.is_cuda
            or int_workspace.dtype != torch.uint8
            or kv_indptr.dtype != torch.int32
            or kv_indices.dtype != torch.int32
            or kv_len_arr.dtype != torch.int32
            or length <= 0
            or page_size <= 0
            or heads <= 0
            or len(normalized_plan) != 18
            or normalized_plan[0] not in (1, 2)
        ):
            return False
        return bool(
            _EXT.flashinfer_mla_batch1_plan(
                int_workspace,
                kv_indptr,
                kv_indices,
                kv_len_arr,
                int(length),
                int(page_size),
                int(heads),
                normalized_plan,
            )
        )

    def latent_mla_attention_decode_fused(
        query_nope: torch.Tensor,
        query_rope: torch.Tensor,
        latent_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        position: torch.Tensor,
        scale_denominator: float,
        score_workspace: torch.Tensor,
        output: torch.Tensor | None = None,
    ):
        """Dynamic-length latent MLA selected by tensor capabilities."""
        if (
            not query_nope.is_cuda
            or query_nope.dtype != torch.bfloat16
            or query_rope.dtype != torch.bfloat16
            or latent_cache.dtype != torch.bfloat16
            or rope_cache.dtype != torch.bfloat16
            or position.dtype != torch.long
            or score_workspace.dtype != torch.float32
            or query_nope.ndim != 3
            or query_nope.shape[1] != 1
            or query_rope.ndim != 3
            or query_rope.shape[:2] != query_nope.shape[:2]
            or latent_cache.ndim != 2
            or rope_cache.ndim != 2
            or latent_cache.shape[0] != rope_cache.shape[0]
            or latent_cache.shape[1] != query_nope.shape[2]
            or rope_cache.shape[1] != query_rope.shape[2]
            or score_workspace.shape
            != (query_nope.shape[0], latent_cache.shape[0])
            or position.numel() != 1
            or scale_denominator <= 0.0
        ):
            return None
        return _EXT.latent_mla_attention_decode(
            query_nope.contiguous(),
            query_rope.contiguous(),
            latent_cache.contiguous(),
            rope_cache.contiguous(),
            position.contiguous(),
            float(scale_denominator),
            score_workspace,
            output,
        )

    def glm_merge_scores_fused(
        a: torch.Tensor,
        b: torch.Tensor,
        scale: float,
    ):
        """Fuse latent MLA score casts, scaling and addition."""
        if (
            os.environ.get("CCCP_GLM_SCORE_FUSED", "1") == "0"
            or not a.is_cuda
            or a.dtype != torch.bfloat16
            or b.dtype != torch.bfloat16
            or a.shape != b.shape
        ):
            return None
        return _EXT.glm_merge_scores(a, b, float(scale))

    def dsv4_attn_decode_fused(q: torch.Tensor, win_kv: torch.Tensor,
                               win_pos: torch.Tensor, comp_kv: torch.Tensor,
                               sink: torch.Tensor, cos: torch.Tensor,
                               sin: torch.Tensor, scale: float):
        """DSV4 B=1,T=1 attention 核；过长或 dtype/shape 不满足时返回 None。"""
        seq = win_kv.shape[1] + comp_kv.shape[1]
        if (not q.is_cuda or q.dtype != torch.float32 or q.shape[0] != 1
                or win_kv.dtype != torch.float32 or comp_kv.dtype != torch.float32
                or win_pos.dtype != torch.long or seq > 4096):
            return None
        return _EXT.dsv4_attn_decode(
            q.contiguous(), win_kv, win_pos, comp_kv, sink,
            cos.reshape(-1), sin.reshape(-1), float(scale)
        )

    def dsv4_hc_pre_fused(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor,
                          base: torch.Tensor, iters: int, eps: float):
        """融合 HC pre；返回形状与 dsv4.hc_pre 一致，不满足条件时返回 None。"""
        if (not x.is_cuda or x.dtype != torch.float32 or x.shape[-2] != 4
                or fn.dtype not in (torch.float32, torch.bfloat16)
                or scale.dtype != torch.float32 or base.dtype != torch.float32):
            return None
        y, post, comb = _EXT.dsv4_hc_pre(
            x, fn, scale, base, int(iters), float(eps)
        )
        lead = x.shape[:-2]
        return (
            y.view(*lead, x.shape[-1]),
            post.view(*lead, 4),
            comb.view(*lead, 4, 4),
        )

    def dsv4_hc_pre_norm_fused(
        x: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm: torch.Tensor,
        iters: int,
        eps: float,
        output_buffers: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ] | None = None,
    ):
        """BF16 HC pre + RMSNorm；可复用调用方固定输出缓冲。"""
        if (
            not x.is_cuda
            or x.dtype != torch.bfloat16
            or x.shape[-2] != 4
            or not all(
                isinstance(t, torch.Tensor)
                for t in (fn, scale, base, norm)
            )
            or fn.dtype not in (torch.float32, torch.bfloat16)
            or norm.dtype not in (torch.float32, torch.bfloat16)
            or scale.dtype != torch.float32
            or base.dtype != torch.float32
        ):
            return None
        if output_buffers is None:
            y, post, comb = _EXT.dsv4_hc_pre_norm(
                x, fn, scale, base, norm, int(iters), float(eps)
            )
        else:
            y, post, comb = _EXT.dsv4_hc_pre_norm_out(
                x,
                fn,
                scale,
                base,
                norm,
                *output_buffers,
                int(iters),
                float(eps),
            )
        lead = x.shape[:-2]
        return (
            y.view(*lead, x.shape[-1]),
            post.view(*lead, 4),
            comb.view(*lead, 4, 4),
        )

    def dsv4_hc_post_fused(
        out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        output: torch.Tensor | None = None,
    ):
        """BF16 HC post residual mix; accumulation stays FP32."""
        if (
            not out.is_cuda
            or out.dtype not in (torch.float32, torch.bfloat16)
            or residual.dtype != torch.bfloat16
            or post.dtype != torch.bfloat16
            or comb.dtype != torch.bfloat16
            or residual.shape[-2] != 4
        ):
            return None
        arguments = (
            out.contiguous(),
            residual.contiguous(),
            post.contiguous(),
            comb.contiguous(),
        )
        if output is None:
            return _EXT.dsv4_hc_post(*arguments)
        return _EXT.dsv4_hc_post_out(*arguments, output)

    def dsv4_hc_post_moe_fused(
        routed: torch.Tensor,
        shared: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        output: torch.Tensor | None = None,
    ):
        """Fuse BF16 routed/shared merge with Hyper-Connection post."""
        if (
            not routed.is_cuda
            or routed.dtype != torch.float32
            or shared.dtype not in (torch.bfloat16, torch.float32)
            or residual.dtype != torch.bfloat16
            or post.dtype != torch.bfloat16
            or comb.dtype != torch.bfloat16
            or residual.shape[-2] != 4
            or routed.numel() * 4 != residual.numel()
            or shared.numel() * 4 != residual.numel()
        ):
            return None
        arguments = (
            routed.contiguous(),
            shared.contiguous(),
            residual.contiguous(),
            post.contiguous(),
            comb.contiguous(),
        )
        if output is None:
            return _EXT.dsv4_hc_post_moe(*arguments)
        return _EXT.dsv4_hc_post_moe_out(*arguments, output)

    def dsv4_route_post_fused(
        scores: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        top_k: int,
    ):
        """Fuse learned-router masked top-k selection and score gather."""
        if (
            not scores.is_cuda
            or scores.dtype != torch.float32
            or scores.dim() != 2
            or scores.shape[0] != 1
            or bias.dtype != torch.float32
            or mask.dtype != torch.bool
            or bias.numel() != scores.shape[1]
            or mask.numel() != scores.shape[1]
            or top_k <= 0
            or top_k > 16
        ):
            return None
        weights, indices = _EXT.dsv4_route_post(
            scores.contiguous(),
            bias.contiguous(),
            mask.contiguous(),
            int(top_k),
        )
        return weights, indices

    def route_topk_sigmoid_fused(
        logits: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        top_k: int,
        routed_scaling: float,
        output_buffers: tuple[
            torch.Tensor,
            torch.Tensor,
        ] | None = None,
    ):
        """Fuse sigmoid, corrected Top-K, gather and normalization."""
        if (
            os.environ.get(
                "CCCP_ROUTE_FUSED",
                os.environ.get("CCCP_GLM_ROUTE_FUSED", "1"),
            ) == "0"
            or not logits.is_cuda
            or logits.dtype != torch.float32
            or logits.dim() != 2
            or logits.shape[0] != 1
            or bias.dtype != torch.float32
            or mask.dtype != torch.bool
            or bias.numel() != logits.shape[1]
            or mask.numel() != logits.shape[1]
            or top_k <= 0
            or top_k > 16
        ):
            return None
        if output_buffers is None:
            weights, indices = _EXT.sigmoid_route(
                logits.contiguous(),
                bias.contiguous(),
                mask.contiguous(),
                int(top_k),
                float(routed_scaling),
            )
        else:
            weights, indices = output_buffers
            if (
                weights.dtype != torch.float32
                or indices.dtype != torch.long
                or weights.shape != (1, top_k)
                or indices.shape != (1, top_k)
                or weights.device != logits.device
                or indices.device != logits.device
            ):
                return None
            weights, indices = _EXT.sigmoid_route_out(
                logits.contiguous(),
                bias.contiguous(),
                mask.contiguous(),
                int(top_k),
                float(routed_scaling),
                weights,
                indices,
            )
        return weights, indices

    def linear_route_topk_sigmoid_fused(
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        top_k: int,
        routed_scaling: float,
        output_buffers: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ):
        """Fuse a source-native FP32 router projection with sigmoid Top-K."""
        if (
            os.environ.get("CCCP_LINEAR_ROUTE_FUSED", "1") == "0"
            or not value.is_cuda
            or value.dtype not in (torch.bfloat16, torch.float32)
            or weight.dtype != torch.float32
            or weight.ndim != 2
            or value.shape != (1, weight.shape[1])
            or bias.dtype != torch.float32
            or mask.dtype != torch.bool
            or bias.numel() != weight.shape[0]
            or mask.numel() != weight.shape[0]
            or top_k <= 0
            or top_k > 16
        ):
            return None
        logits, weights, indices = output_buffers
        if (
            logits.dtype != torch.float32
            or logits.shape != (1, weight.shape[0])
            or weights.dtype != torch.float32
            or weights.shape != (1, top_k)
            or indices.dtype != torch.long
            or indices.shape != (1, top_k)
            or any(
                tensor.device != value.device
                for tensor in (
                    weight,
                    bias,
                    mask,
                    logits,
                    weights,
                    indices,
                )
            )
        ):
            return None
        weights, indices = _EXT.linear_sigmoid_route_out(
            value.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            mask.contiguous(),
            int(top_k),
            float(routed_scaling),
            logits,
            weights,
            indices,
        )
        return weights, indices

    # 旧公开名仅保留给外部脚本过渡；模型运行时统一经过 ops.route_topk。
    glm_route_fused = route_topk_sigmoid_fused

    def paged_gather_bf16_fused(
        page_ptrs: torch.Tensor,
        indices: torch.Tensor,
        page_items: int,
        dim: int,
    ):
        """Copy batch-1 BF16 paged entries without host synchronization."""
        if (
            os.environ.get("CCCP_PAGED_KV_FUSED", "1") == "0"
            or not page_ptrs.is_cuda
            or page_ptrs.dtype != torch.long
            or page_ptrs.ndim != 1
            or page_ptrs.numel() == 0
            or not indices.is_cuda
            or indices.dtype != torch.long
            or page_items <= 0
            or dim <= 0
        ):
            return None
        shape = (*indices.shape, dim)
        return _EXT.paged_gather_bf16(
            page_ptrs,
            indices.contiguous(),
            int(page_items),
            int(dim),
        ).view(shape)

    def hadamard_bf16_fused(x: torch.Tensor):
        """One-block FP32 Walsh-Hadamard transform with BF16 boundaries."""
        width = x.shape[-1] if x.ndim else 0
        if (
            os.environ.get("CCCP_INDEXER_HADAMARD_FUSED", "1") == "0"
            or not x.is_cuda
            or x.dtype != torch.bfloat16
            or width <= 0
            or width > 256
            or width & (width - 1)
        ):
            return None
        return _EXT.hadamard_bf16(x)

    def int4_gemv_fused(
        x: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        cols: int,
        group_size: int,
        output: torch.Tensor | None = None,
        *,
        group_vector: bool | None = None,
    ):
        """Direct packed INT4 decode GEMV; set CCCP_INT4_GEMV_FUSED=0 to fall back."""
        if (
            os.environ.get("CCCP_INT4_GEMV_FUSED", "1") == "0"
            or not x.is_cuda
            or x.dtype not in (torch.float32, torch.bfloat16)
            or x.ndim != 2
            or x.shape[0] != 1
            or packed.dtype != torch.uint8
            or scales.dtype != torch.float16
            or group_size != 64
            or cols <= 0
            or cols % 64
        ):
            return None
        output_rows = int(packed.shape[0])
        if group_vector is None:
            group_vector = (
                os.environ.get("CCCP_INT4_GROUP_VECTOR", "0") != "0"
                or (
                    os.environ.get(
                        "CCCP_INT4_LM_HEAD_VECTOR",
                        "1",
                    ) != "0"
                    and output_rows == 154880
                    and cols == 6144
                )
            )
        return _EXT.int4_gemv_packed_f32(
            x.contiguous(),
            packed.contiguous(),
            scales.contiguous(),
            int(cols),
            int(group_size),
            bool(group_vector),
            output,
        )

    def block_fp8_gemv_fused(
        x: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        cols: int,
        block_size: int,
        output: torch.Tensor | None = None,
    ):
        """Decode directly from native E4M3 bytes and 128x128 scales."""
        if (
            os.environ.get("CCCP_FP8_GEMV_FUSED", "1") == "0"
            or not x.is_cuda
            or x.dtype not in (torch.float32, torch.bfloat16)
            or x.ndim != 2
            or x.shape != (1, cols)
            or weights.dtype != torch.uint8
            or weights.ndim != 2
            or weights.shape[1] != cols
            or scales.dtype != torch.float32
            or scales.ndim != 2
            or block_size != 128
        ):
            return None
        return _EXT.block_fp8_gemv_f32(
            x.contiguous(),
            weights.contiguous(),
            scales.contiguous(),
            int(cols),
            int(block_size),
            output,
        )

    def block_fp8_grouped_gemv_fused(
        x: torch.Tensor,
        weight_ptrs: torch.Tensor,
        scale_ptrs: torch.Tensor,
        row_offsets: torch.Tensor,
        total_rows: int,
        cols: int,
        block_size: int,
        output: torch.Tensor | None = None,
    ):
        """Run several compact block-FP8 projections in one CUDA launch."""
        if (
            os.environ.get("CCCP_FP8_GROUPED_GEMV", "1") == "0"
            or not x.is_cuda
            or x.dtype not in (torch.float32, torch.bfloat16)
            or x.ndim != 2
            or x.shape[1] != cols
            or weight_ptrs.dtype != torch.int64
            or scale_ptrs.dtype != torch.int64
            or row_offsets.dtype != torch.int32
            or not weight_ptrs.is_cuda
            or not scale_ptrs.is_cuda
            or not row_offsets.is_cuda
            or weight_ptrs.ndim != 1
            or scale_ptrs.shape != weight_ptrs.shape
            or row_offsets.shape != (weight_ptrs.numel() + 1,)
            or x.shape[0] not in (1, weight_ptrs.numel())
            or total_rows <= 0
            or block_size != 128
        ):
            return None
        return _EXT.block_fp8_grouped_gemv_f32(
            x.contiguous(),
            weight_ptrs.contiguous(),
            scale_ptrs.contiguous(),
            row_offsets.contiguous(),
            int(total_rows),
            int(cols),
            int(block_size),
            output,
        )

    def int4_glm_qb_split_fused(
        x: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        cols: int,
        group_size: int,
        heads: int,
        nope_width: int,
        rope_width: int,
        nope_output: torch.Tensor | None = None,
        rope_output: torch.Tensor | None = None,
    ):
        """Decode Q-B directly into BF16 no-PE and FP32 RoPE rows."""
        if (
            os.environ.get("CCCP_GLM_QB_SPLIT", "1") == "0"
            or not x.is_cuda
            or x.dtype != torch.float32
            or x.shape != (1, cols)
            or packed.dtype != torch.uint8
            or scales.dtype != torch.float16
            or group_size != 64
        ):
            return None
        return _EXT.int4_glm_qb_split(
            x.contiguous(),
            packed.contiguous(),
            scales.contiguous(),
            int(cols),
            int(group_size),
            (
                os.environ.get(
                    "CCCP_GLM_QB_GROUP_VECTOR",
                    "1",
                )
                != "0"
                and (cols // group_size) % 4 == 0
            ),
            int(heads),
            int(nope_width),
            int(rope_width),
            nope_output,
            rope_output,
        )

    def int4_embedding_fused(
        packed: torch.Tensor,
        scales: torch.Tensor,
        row: int,
        cols: int,
        group_size: int,
        output: torch.Tensor | None = None,
    ):
        """Decode one packed INT4 embedding row directly into FP32."""
        if (
            os.environ.get("CCCP_INT4_EMBEDDING_FUSED", "1") == "0"
            or not packed.is_cuda
            or packed.dtype != torch.uint8
            or scales.dtype != torch.float16
            or packed.ndim != 2
            or scales.ndim != 2
            or group_size != 64
            or cols <= 0
            or cols % 64
            or row < 0
            or row >= packed.shape[0]
        ):
            return None
        return _EXT.int4_embedding_lookup(
            packed.contiguous(),
            scales.contiguous(),
            int(row),
            int(cols),
            int(group_size),
            output,
        )

    def int4_embedding_device_fused(
        packed: torch.Tensor,
        scales: torch.Tensor,
        row: torch.Tensor,
        cols: int,
        group_size: int,
        output: torch.Tensor | None = None,
    ):
        """Decode one embedding row selected by a CUDA int64 scalar."""
        if (
            os.environ.get("CCCP_INT4_EMBEDDING_FUSED", "1") == "0"
            or not packed.is_cuda
            or not row.is_cuda
            or packed.dtype != torch.uint8
            or scales.dtype != torch.float16
            or row.dtype != torch.long
            or row.numel() != 1
            or group_size != 64
        ):
            return None
        return _EXT.int4_embedding_lookup_device_row(
            packed.contiguous(),
            scales.contiguous(),
            row.contiguous(),
            int(cols),
            int(group_size),
            output,
        )

    def glm_norm_qkv_int4_fused(
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        q_packed: torch.Tensor,
        q_scales: torch.Tensor,
        kv_packed: torch.Tensor,
        kv_scales: torch.Tensor,
        cols: int,
        group_size: int,
        eps: float,
        output_buffers: tuple[
            torch.Tensor,
            torch.Tensor,
        ] | None = None,
    ):
        """Fuse GLM decode input RMSNorm with Q-A and KV-A INT4 GEMVs."""
        if (
            os.environ.get("CCCP_GLM_NORM_QKV_FUSED", "1") == "0"
            or not x.is_cuda
            or x.dtype != torch.float32
            or x.ndim != 2
            or x.shape != (1, cols)
            or norm_weight.dtype != torch.float32
            or norm_weight.shape != (cols,)
            or q_packed.dtype != torch.uint8
            or kv_packed.dtype != torch.uint8
            or q_scales.dtype != torch.float16
            or kv_scales.dtype != torch.float16
            or q_packed.ndim != 2
            or kv_packed.ndim != 2
            or q_packed.shape[1] * 2 != cols
            or kv_packed.shape[1] * 2 != cols
            or group_size != 64
            or cols <= 0
            or cols % 64
            or q_scales.shape != (
                q_packed.shape[0],
                cols // group_size,
            )
            or kv_scales.shape != (
                kv_packed.shape[0],
                cols // group_size,
            )
        ):
            return None
        return _EXT.glm_norm_qkv_int4(
            x.contiguous(),
            norm_weight.contiguous(),
            q_packed.contiguous(),
            q_scales.contiguous(),
            kv_packed.contiguous(),
            kv_scales.contiguous(),
            int(cols),
            int(group_size),
            float(eps),
            None,
            (
                output_buffers[0]
                if output_buffers is not None
                else None
            ),
            (
                output_buffers[1]
                if output_buffers is not None
                else None
            ),
        )

    def glm_residual_norm_qkv_int4_fused(
        residual: torch.Tensor,
        update: torch.Tensor,
        norm_weight: torch.Tensor,
        q_packed: torch.Tensor,
        q_scales: torch.Tensor,
        kv_packed: torch.Tensor,
        kv_scales: torch.Tensor,
        cols: int,
        group_size: int,
        eps: float,
    ):
        """Fuse a residual add into input RMSNorm plus Q-A/KV-A."""
        if (
            os.environ.get(
                "CCCP_GLM_RESIDUAL_NORM_QKV",
                "1",
            ) == "0"
            or not residual.is_cuda
            or residual.dtype != torch.float32
            or residual.ndim != 2
            or residual.shape != (1, cols)
            or update.dtype != torch.float32
            or update.shape != residual.shape
            or norm_weight.dtype != torch.float32
            or norm_weight.shape != (cols,)
            or q_packed.dtype != torch.uint8
            or kv_packed.dtype != torch.uint8
            or q_scales.dtype != torch.float16
            or kv_scales.dtype != torch.float16
            or q_packed.ndim != 2
            or kv_packed.ndim != 2
            or q_packed.shape[1] * 2 != cols
            or kv_packed.shape[1] * 2 != cols
            or group_size != 64
            or cols <= 0
            or cols % 64
            or q_scales.shape != (
                q_packed.shape[0],
                cols // group_size,
            )
            or kv_scales.shape != (
                kv_packed.shape[0],
                cols // group_size,
            )
        ):
            return None
        return _EXT.glm_norm_qkv_int4(
            residual.contiguous(),
            norm_weight.contiguous(),
            q_packed.contiguous(),
            q_scales.contiguous(),
            kv_packed.contiguous(),
            kv_scales.contiguous(),
            int(cols),
            int(group_size),
            float(eps),
            update.contiguous(),
            None,
            None,
        )

    def glm_residual_norm_router_fused(
        residual: torch.Tensor,
        update: torch.Tensor,
        norm_weight: torch.Tensor,
        router_weight: torch.Tensor,
        eps: float,
        norm_output: torch.Tensor | None = None,
        output_buffers: tuple[
            torch.Tensor,
            torch.Tensor,
        ] | None = None,
    ):
        """Fuse GLM decode residual add, RMSNorm and router GEMV."""
        if (
            os.environ.get(
                "CCCP_GLM_RESIDUAL_NORM_ROUTER",
                "1",
            ) == "0"
            or not residual.is_cuda
            or residual.dtype != torch.float32
            or residual.ndim != 2
            or residual.shape[0] != 1
            or update.dtype != torch.float32
            or update.shape != residual.shape
            or norm_weight.dtype != torch.float32
            or norm_weight.shape != (residual.shape[1],)
            or router_weight.dtype != torch.float32
            or router_weight.ndim != 2
            or router_weight.shape[1] != residual.shape[1]
        ):
            return None
        if norm_output is None:
            return _EXT.glm_residual_norm_router(
                residual.contiguous(),
                update.contiguous(),
                norm_weight.contiguous(),
                router_weight.contiguous(),
                float(eps),
            )
        if (
            norm_output.dtype != torch.float32
            or norm_output.shape != residual.shape
            or norm_output.device != residual.device
            or not norm_output.is_contiguous()
        ):
            return None
        return _EXT.glm_residual_norm_router_norm_out(
            residual.contiguous(),
            update.contiguous(),
            norm_weight.contiguous(),
            router_weight.contiguous(),
            float(eps),
            norm_output,
            (
                output_buffers[0]
                if output_buffers is not None
                else None
            ),
            (
                output_buffers[1]
                if output_buffers is not None
                else None
            ),
        )

    def residual_add3_fused(
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
    ):
        """Fuse ``residual + (routed + shared)`` with source dtype rounding."""
        if (
            not residual.is_cuda
            or residual.dtype not in {torch.float32, torch.bfloat16}
            or not residual.is_contiguous()
            or routed.dtype != residual.dtype
            or shared.dtype != residual.dtype
            or routed.shape != residual.shape
            or shared.shape != residual.shape
        ):
            return None
        return _EXT.residual_add3(
            residual,
            routed.contiguous(),
            shared.contiguous(),
        )

    def glm_moe_residual_add_fused(
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
    ):
        """Compatibility entry for the generic three-way residual operator."""
        if (
            os.environ.get("CCCP_GLM_MOE_RESIDUAL_ADD", "1") == "0"
            or residual.dtype != torch.float32
        ):
            return None
        return residual_add3_fused(residual, routed, shared)

    def glm_ep_reduce_residual_fused(
        contributions: list[torch.Tensor],
        residual: torch.Tensor,
    ):
        """Fuse up to 16 TP routed/shared contributions with the residual."""
        if (
            os.environ.get("CCCP_GLM_EP_FINAL_FUSED", "1") == "0"
            or not 1 <= len(contributions) <= 16
            or not contributions[0].is_cuda
            or any(
                item.dtype != torch.float32
                for item in contributions
            )
            or residual.dtype != torch.float32
            or any(
                item.numel() != contributions[0].numel()
                for item in contributions[1:]
            )
            or contributions[0].numel() != residual.numel()
        ):
            return None
        return _EXT.glm_ep_reduce_residual(
            [item.contiguous() for item in contributions],
            residual.contiguous(),
        )

    def tp_all_rank_reduce_fused(
        contributions: list[torch.Tensor],
        outputs: list[torch.Tensor],
    ):
        """Reduce canonical FP32 partials into fixed buffers on all ranks."""
        if (
            not 1 <= len(contributions) <= 16
            or not outputs
            or any(
                not item.is_cuda
                or item.dtype != torch.float32
                or not item.is_contiguous()
                for item in contributions
            )
            or any(
                not item.is_cuda
                or item.dtype not in {torch.float32, torch.bfloat16}
                or not item.is_contiguous()
                for item in outputs
            )
            or any(
                item.numel() != contributions[0].numel()
                for item in (*contributions[1:], *outputs)
            )
        ):
            return None
        return _EXT.tp_all_rank_reduce(contributions, outputs)

    def tp1_moe_finalize_fused(
        routed_contribution: torch.Tensor,
        shared_contribution: torch.Tensor,
        residual: torch.Tensor,
        routed_workspace: torch.Tensor,
        shared_workspace: torch.Tensor,
        output: torch.Tensor,
    ):
        """Width-one finalizer used inside a retained layer graph."""
        tensors = (
            routed_contribution,
            shared_contribution,
            residual,
            routed_workspace,
            shared_workspace,
            output,
        )
        if any(not item.is_cuda for item in tensors):
            return None
        return _EXT.tp1_moe_finalize(*tensors)

    def dsv4_kv_commit_controlled_fused(
        kv: torch.Tensor,
        window: torch.Tensor,
        window_positions: torch.Tensor,
        control: torch.Tensor,
        fp8_cache: torch.Tensor | None = None,
    ) -> bool:
        if (
            not kv.is_cuda
            or kv.dtype != torch.float32
            or window.dtype != torch.float32
            or window_positions.dtype != torch.long
            or control.dtype != torch.long
        ):
            return False
        _EXT.dsv4_kv_commit_controlled(
            kv.contiguous(), window, window_positions, control, fp8_cache
        )
        return True

    def dsv4_compressor_step_controlled_fused(
        projected: torch.Tensor,
        ape: torch.Tensor,
        ckv: torch.Tensor,
        cscore: torch.Tensor,
        norm: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        page_ptrs: torch.Tensor,
        control: torch.Tensor,
        model1_cache: torch.Tensor | None = None,
        indexer_cache: torch.Tensor | None = None,
        indexer_scales: torch.Tensor | None = None,
        *,
        ratio: int,
        kv_rows: int,
        width: int,
        rope_width: int,
        page_items: int,
        overlap: bool,
        hadamard: bool,
        eps: float,
    ) -> bool:
        if (
            not projected.is_cuda
            or projected.dtype != torch.bfloat16
            or ckv.dtype != torch.bfloat16
            or cscore.dtype != torch.float32
            or norm.dtype != torch.bfloat16
            or page_ptrs.dtype != torch.long
            or control.dtype != torch.long
        ):
            return False
        _EXT.dsv4_compressor_step_controlled(
            projected.contiguous(), ape.contiguous(), ckv, cscore,
            norm.contiguous(), rope_cos.reshape(-1), rope_sin.reshape(-1),
            page_ptrs, control, model1_cache, indexer_cache, indexer_scales,
            int(ratio), int(kv_rows), int(width),
            int(rope_width), int(page_items), bool(overlap), bool(hadamard),
            float(eps),
        )
        return True

    def paged_indexer_query_fp8_fused(
        query: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        output: torch.Tensor,
        scales: torch.Tensor,
    ) -> bool:
        if (
            not query.is_cuda
            or query.dtype != torch.bfloat16
            or cos.dtype != torch.float32
            or sin.dtype != torch.float32
            or output.dtype != torch.float8_e4m3fn
            or scales.dtype != torch.float32
        ):
            return False
        _EXT.paged_indexer_query_fp8(
            query.contiguous(), cos.reshape(-1), sin.reshape(-1),
            output, scales,
        )
        return True

    def paged_indexer_reduce_logits_fused(
        head_logits: torch.Tensor,
        head_weights: torch.Tensor,
        control: torch.Tensor,
        output: torch.Tensor,
        compression_ratio: int,
    ) -> bool:
        if (
            not head_logits.is_cuda
            or head_logits.dtype != torch.bfloat16
            or head_weights.dtype != torch.float32
            or control.dtype != torch.long
            or output.dtype != torch.float32
        ):
            return False
        _EXT.paged_indexer_reduce_logits(
            head_logits, head_weights.contiguous(), control, output,
            int(compression_ratio),
        )
        return True

    def sparse_attention_inverse_rope_fused(
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            not value.is_cuda
            or value.dtype != torch.bfloat16
            or cos.dtype != torch.float32
            or sin.dtype != torch.float32
        ):
            return None
        return _EXT.sparse_attention_inverse_rope(
            value.contiguous(), cos.reshape(-1), sin.reshape(-1)
        )

    def dsv4_attn_decode_controlled_fused(
        q: torch.Tensor,
        win_kv: torch.Tensor,
        win_pos: torch.Tensor,
        comp_kv: torch.Tensor,
        sink: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        control: torch.Tensor,
        scale: float,
        *,
        ratio: int,
        selected_topk: bool,
    ):
        if (
            not q.is_cuda
            or q.dtype != torch.float32
            or win_kv.dtype != torch.float32
            or win_pos.dtype != torch.long
            or comp_kv.dtype != torch.bfloat16
            or control.dtype != torch.long
            or win_kv.shape[1] + comp_kv.shape[1] > 4096
        ):
            return None
        return _EXT.dsv4_attn_decode_controlled(
            q.contiguous(), win_kv, win_pos, comp_kv, sink,
            cos.reshape(-1), sin.reshape(-1), control, float(scale),
            int(ratio), bool(selected_topk),
        )

    def compressed_state_update_fused(
        projected: torch.Tensor,
        ape: torch.Tensor,
        ckv: torch.Tensor,
        cscore: torch.Tensor,
        ratio: int,
        position: int,
        kv_rows: int,
    ):
        if (
            not projected.is_cuda
            or projected.dtype != torch.bfloat16
            or ape.dtype != torch.bfloat16
            or ckv.dtype != torch.float32
            or cscore.dtype != torch.float32
        ):
            return None
        return _EXT.compressed_state_update(
            projected,
            ape,
            ckv,
            cscore,
            int(ratio),
            int(position),
            int(kv_rows),
        )

    def head_rmsnorm_rope_fused(
        rows: torch.Tensor,
        weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_width: int,
        eps: float,
    ):
        if (
            not rows.is_cuda
            or rows.dtype != torch.float32
            or weight.dtype != torch.bfloat16
            or cos.dtype != torch.float32
            or sin.dtype != torch.float32
        ):
            return None
        return _EXT.head_rmsnorm_rope(
            rows,
            weight,
            cos,
            sin,
            int(rope_width),
            float(eps),
        )

    def tp_all_rank_reduce_from_events_fused(
        contributions: list[torch.Tensor],
        input_events: list[torch.cuda.Event],
        outputs: list[torch.Tensor],
        output_events: list[torch.cuda.Event],
    ):
        """Wait producer events and publish one TP reduction per output.

        This is the public collective used by replicated no-owner subgroups.
        The C++ entry point performs all device switches, waits, reductions
        and completion records in one host call.
        """
        if (
            not 1 <= len(contributions) <= 16
            or len(input_events) != len(contributions)
            or not 1 <= len(outputs) <= 16
            or len(output_events) != len(outputs)
            or any(
                not item.is_cuda
                or item.dtype != torch.float32
                or not item.is_contiguous()
                for item in contributions
            )
            or any(
                not item.is_cuda
                or item.dtype not in {torch.float32, torch.bfloat16}
                or not item.is_contiguous()
                for item in outputs
            )
            or any(
                item.numel() != contributions[0].numel()
                for item in (*contributions[1:], *outputs)
            )
        ):
            return None
        return _EXT.tp_all_rank_reduce_from_events(
            contributions,
            [event.cuda_event for event in input_events],
            outputs,
            [event.cuda_event for event in output_events],
        )

    def packed_moe_topk_grouped_fused(
        value: torch.Tensor,
        token_ids: torch.Tensor,
        group_experts: torch.Tensor,
        group_offsets: torch.Tensor,
        weights: torch.Tensor,
        metadata: torch.Tensor,
        *,
        activation: str,
        beta: float,
        linear_beta: float,
        limit: float,
        hidden_workspace: torch.Tensor,
        result: torch.Tensor,
        max_group_tiles: int = 1,
    ) -> torch.Tensor | None:
        """Group prefill routes by expert for the public packed operator."""
        if (
            not value.is_cuda
            or value.dtype != torch.bfloat16
            or value.ndim != 2
            or token_ids.dtype != torch.long
            or token_ids.ndim != 1
            or group_experts.dtype != torch.long
            or group_experts.ndim != 1
            or group_offsets.dtype != torch.int32
            or group_offsets.ndim != 1
            or weights.dtype != torch.float32
            or weights.ndim != 1
            or weights.numel() != token_ids.numel()
            or metadata.dtype != torch.long
            or metadata.ndim != 2
            or metadata.shape[0] not in (10, 15)
        ):
            return None
        activation_kind = {"situ": 0, "silu": 1, "swiglu": 1}.get(
            str(activation).strip().lower()
        )
        if activation_kind is None:
            return None
        with torch.cuda.device(value.device):
            return _EXT.packed_moe_topk_grouped(
                value.contiguous(), token_ids.contiguous(),
                group_experts.contiguous(), group_offsets.contiguous(),
                weights.contiguous(), metadata.contiguous(),
                int(activation_kind), float(beta), float(linear_beta),
                float(limit), hidden_workspace, result,
                0,
                max(1, int(max_group_tiles)),
            )

    def projection_dequant_fused(
        metadata: torch.Tensor,
        output_gu: torch.Tensor,
        output_down: torch.Tensor,
    ) -> torch.Tensor | None:
        """Expand packed two/three-projection experts into dense BF16 buffers."""
        if (
            not metadata.is_cuda
            or metadata.dtype != torch.long
            or metadata.ndim != 2
            or metadata.shape[0] not in (10, 15)
            or not output_gu.is_cuda
            or output_gu.dtype != torch.bfloat16
            or output_gu.ndim != 3
            or not output_down.is_cuda
            or output_down.dtype != torch.bfloat16
            or output_down.ndim != 3
        ):
            return None
        with torch.cuda.device(metadata.device):
            return _EXT.vq_projection_dequant(
                metadata.contiguous(),
                output_gu,
                output_down,
            )

    def projection_expand_native8_fused(
        metadata: torch.Tensor,
        output_gu: torch.Tensor,
        output_down: torch.Tensor,
    ) -> torch.Tensor | None:
        """Expand packed projections into reusable E4M3/INT8 buffers."""
        if (
            not metadata.is_cuda
            or metadata.dtype != torch.long
            or metadata.ndim != 2
            or metadata.shape[0] not in (10, 15)
            or not output_gu.is_cuda
            or output_gu.dtype not in (torch.float8_e4m3fn, torch.int8)
            or output_gu.ndim != 3
            or not output_down.is_cuda
            or output_down.dtype != output_gu.dtype
            or output_down.ndim not in (2, 3)
        ):
            return None
        with torch.cuda.device(metadata.device):
            return _EXT.vq_projection_expand_native8(
                metadata.contiguous(),
                output_gu,
                output_down,
            )

    def tp_moe_finalize_from_events_fused(
        routed_contributions: list[torch.Tensor],
        shared_contributions: list[torch.Tensor],
        input_events: list[torch.cuda.Event],
        residuals: list[torch.Tensor],
        outputs: list[torch.Tensor],
        output_events: list[torch.cuda.Event],
    ):
        """Finalize batched TP MoE with the decode rounding contract."""
        if (
            not 1 <= len(routed_contributions) <= 16
            or len(shared_contributions) != len(routed_contributions)
            or len(input_events) != len(routed_contributions)
            or not 1 <= len(outputs) <= 16
            or len(residuals) != len(outputs)
            or len(output_events) != len(outputs)
            or any(
                not item.is_cuda
                or item.dtype != torch.float32
                or not item.is_contiguous()
                for item in (*routed_contributions, *shared_contributions)
            )
            or any(
                not item.is_cuda
                or item.dtype != torch.bfloat16
                or not item.is_contiguous()
                for item in (*residuals, *outputs)
            )
            or any(
                item.numel() != routed_contributions[0].numel()
                for item in (
                    *routed_contributions[1:],
                    *shared_contributions,
                    *residuals,
                    *outputs,
                )
            )
        ):
            return None
        return _EXT.tp_moe_finalize_from_events(
            routed_contributions,
            shared_contributions,
            [event.cuda_event for event in input_events],
            residuals,
            outputs,
            [event.cuda_event for event in output_events],
        )

    def tp_hidden_add_batch_fused(
        left: list[torch.Tensor],
        left_events: list[torch.cuda.Event],
        right: list[torch.Tensor],
        right_events: list[torch.cuda.Event],
        outputs: list[torch.Tensor],
        output_events: list[torch.cuda.Event],
    ):
        if not (
            left
            and len(left)
            == len(left_events)
            == len(right)
            == len(right_events)
            == len(outputs)
            == len(output_events)
        ):
            return None
        return _EXT.tp_hidden_add_batch(
            left,
            [event.cuda_event for event in left_events],
            right,
            [event.cuda_event for event in right_events],
            outputs,
            [event.cuda_event for event in output_events],
        )

    def tp_hidden_rmsnorm_batch_fused(
        inputs: list[torch.Tensor],
        input_events: list[torch.cuda.Event],
        weights: list[torch.Tensor],
        eps: float,
        outputs: list[torch.Tensor],
        output_events: list[torch.cuda.Event],
    ):
        if not (
            inputs
            and len(inputs)
            == len(input_events)
            == len(weights)
            == len(outputs)
            == len(output_events)
        ):
            return None
        return _EXT.tp_hidden_rmsnorm_batch(
            inputs,
            [event.cuda_event for event in input_events],
            weights,
            float(eps),
            outputs,
            [event.cuda_event for event in output_events],
        )

    def tp_hidden_residual_mix_batch_fused(
        prefixes: list[torch.Tensor],
        prefix_events: list[torch.cuda.Event],
        residuals: list[torch.Tensor],
        residual_events: list[torch.cuda.Event],
        projections: list[torch.Tensor],
        norm_weights: list[torch.Tensor],
        post_norm_weights: list[torch.Tensor],
        workspaces: list[torch.Tensor],
        residual_inverses: list[torch.Tensor],
        eps: float,
        outputs: list[torch.Tensor],
        output_events: list[torch.cuda.Event],
    ):
        count = len(outputs)
        if not (
            count
            and len(prefixes)
            == len(prefix_events)
            == len(residuals)
            == len(residual_events)
            == len(projections)
            == len(norm_weights)
            == len(post_norm_weights)
            == len(workspaces)
            == len(residual_inverses)
            == len(output_events)
            == count
        ):
            return None
        return _EXT.tp_hidden_residual_mix_batch(
            prefixes,
            [event.cuda_event for event in prefix_events],
            residuals,
            [event.cuda_event for event in residual_events],
            projections,
            norm_weights,
            post_norm_weights,
            workspaces,
            residual_inverses,
            float(eps),
            int(os.environ.get("CCCP_RESIDUAL_SINGLE_MAX_ROWS", "2")),
            outputs,
            [event.cuda_event for event in output_events],
        )

    def launch_cuda_graphs_fused(
        devices: list[int],
        graphs: list[torch.cuda.CUDAGraph],
        streams: list[torch.cuda.Stream],
        done_events: list[torch.cuda.Event],
        source_event: torch.cuda.Event,
    ) -> None:
        """Launch all TP-rank graphs without per-rank Python context switches."""
        _EXT.launch_cuda_graphs(
            devices,
            [graph.raw_cuda_graph_exec() for graph in graphs],
            [stream.cuda_stream for stream in streams],
            [event.cuda_event for event in done_events],
            source_event.cuda_event,
        )

    def launch_cuda_graphs_reduce_fused(
        devices: list[int],
        graphs: list[torch.cuda.CUDAGraph],
        streams: list[torch.cuda.Stream],
        done_events: list[torch.cuda.Event],
        source_event: torch.cuda.Event,
        contributions: list[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Launch every rank and complete the Row-TP reduction in one call."""
        return _EXT.launch_cuda_graphs_reduce(
            devices,
            [graph.raw_cuda_graph_exec() for graph in graphs],
            [stream.cuda_stream for stream in streams],
            [event.cuda_event for event in done_events],
            source_event.cuda_event,
            contributions,
            residual,
        )

    def make_tp_graph_launch_batch(
        devices: list[int],
        graphs: list[torch.cuda.CUDAGraph],
        streams: list[torch.cuda.Stream],
        done_events: list[torch.cuda.Event],
        source_event: torch.cuda.Event,
    ):
        """Cache immutable CUDA handles once instead of per decode layer."""
        return _EXT.TPGraphLaunchBatch(
            devices,
            [graph.raw_cuda_graph_exec() for graph in graphs],
            [stream.cuda_stream for stream in streams],
            [event.cuda_event for event in done_events],
            source_event.cuda_event,
        )

    def make_tp_graph_sequence_batch(
        devices: list[int],
        graph_sequences: list[list[torch.cuda.CUDAGraph]],
        streams: list[torch.cuda.Stream],
        done_events: list[torch.cuda.Event],
        source_event: torch.cuda.Event,
    ):
        """Join fixed-address child graphs into one parent graph per rank."""
        if (
            not devices
            or len(devices) != len(graph_sequences)
            or len(devices) != len(streams)
            or len(devices) != len(done_events)
        ):
            raise ValueError(
                "TP graph sequences, devices and streams must be size-equal"
            )
        return _EXT.TPGraphLaunchBatch(
            devices,
            [
                [graph.raw_cuda_graph() for graph in sequence]
                for sequence in graph_sequences
            ],
            [stream.cuda_stream for stream in streams],
            [event.cuda_event for event in done_events],
            source_event.cuda_event,
        )

    def make_tp_graph_dag_batch(
        devices: list[int],
        graph_stages: list[list[list[torch.cuda.CUDAGraph]]],
        streams: list[torch.cuda.Stream],
        done_events: list[torch.cuda.Event],
        source_event: torch.cuda.Event,
    ):
        """Compose sequential stages while allowing children in one stage to overlap."""
        if (
            not devices
            or len(devices) != len(graph_stages)
            or len(devices) != len(streams)
            or len(devices) != len(done_events)
        ):
            raise ValueError(
                "TP graph DAGs, devices and streams must be size-equal"
            )
        return _EXT.TPGraphLaunchBatch(
            devices,
            [
                [
                    [graph.raw_cuda_graph() for graph in stage]
                    for stage in rank_stages
                ]
                for rank_stages in graph_stages
            ],
            [stream.cuda_stream for stream in streams],
            [event.cuda_event for event in done_events],
            source_event.cuda_event,
        )

    def make_tp_raw_graph_dag_batch(
        devices: list[int],
        graph_stages: list[list[list[int]]],
        streams: list[torch.cuda.Stream],
        done_events: list[torch.cuda.Event],
        source_event: torch.cuda.Event,
    ):
        """Compose retained CUDA child handles, including C++ parent graphs.

        This is the lower-level counterpart of ``make_tp_graph_dag_batch``.
        It is used when a stage contains an already-composed public operator
        batch rather than a Python-owned ``torch.cuda.CUDAGraph``.
        """
        if (
            not devices
            or len(devices) != len(graph_stages)
            or len(devices) != len(streams)
            or len(devices) != len(done_events)
        ):
            raise ValueError(
                "raw TP graph DAGs, devices and streams must be size-equal"
            )
        return _EXT.TPGraphLaunchBatch(
            devices,
            graph_stages,
            [stream.cuda_stream for stream in streams],
            [event.cuda_event for event in done_events],
            source_event.cuda_event,
        )

    def make_tp_no_owner_moe_layer_plan(
        shared_batch,
        route_batch,
        expert_batch,
        final_batch,
        input_events,
        route_contribution_groups,
        route_output_groups,
        route_output_events,
        expert_contributions,
        packed_outputs,
        packed_output_events,
        routed_contributions,
        shared_contributions,
        shared_events,
        residuals,
        residual_events,
        routed_workspaces,
        shared_workspaces,
        outputs,
        output_events,
    ):
        """Cache a complete fixed-address all-rank MoE submission plan.

        The plan only combines host scheduling.  Router/latent, packed expert
        and hidden collectives remain explicit all-rank event boundaries.
        """
        return _EXT.TPNoOwnerMoELayerPlan(
            shared_batch,
            route_batch,
            expert_batch,
            final_batch,
            [event.cuda_event for event in input_events],
            route_contribution_groups,
            route_output_groups,
            [event.cuda_event for event in route_output_events],
            expert_contributions,
            packed_outputs,
            [event.cuda_event for event in packed_output_events],
            routed_contributions,
            shared_contributions,
            [event.cuda_event for event in shared_events],
            residuals,
            [event.cuda_event for event in residual_events],
            routed_workspaces,
            shared_workspaces,
            outputs,
            [event.cuda_event for event in output_events],
        )

    def make_tp_no_owner_decode_layer_plan(
        attention_batch,
        moe_plan,
        attention_contributions,
        attention_outputs,
        attention_output_events,
    ):
        """Cache Attention→MoE as one fixed all-rank host submission."""
        return _EXT.TPNoOwnerDecodeLayerPlan(
            attention_batch,
            moe_plan,
            attention_contributions,
            attention_outputs,
            [
                event.cuda_event
                for event in attention_output_events
            ],
        )

    def make_tp_no_owner_hc_decode_layer_plan(
        attention_batch,
        shared_batch,
        route_batch,
        expert_batch,
        attention_contributions,
        attention_outputs,
        attention_output_events,
        attention_residuals,
        attention_posts,
        attention_combs,
        prefixes,
        ffn_functions,
        ffn_scales,
        ffn_bases,
        ffn_norms,
        ffn_inputs,
        ffn_posts,
        ffn_combs,
        ffn_events,
        route_output_events,
        expert_contributions,
        shared_contributions,
        shared_events,
        outputs,
        output_events,
        sinkhorn_iters,
        eps,
    ):
        """Cache one complete no-owner Hyper-Connection TP layer plan."""
        return _EXT.TPNoOwnerHCDecodeLayerPlan(
            attention_batch,
            shared_batch,
            route_batch,
            expert_batch,
            attention_contributions,
            attention_outputs,
            [event.cuda_event for event in attention_output_events],
            attention_residuals,
            attention_posts,
            attention_combs,
            prefixes,
            ffn_functions,
            ffn_scales,
            ffn_bases,
            ffn_norms,
            ffn_inputs,
            ffn_posts,
            ffn_combs,
            [event.cuda_event for event in ffn_events],
            [event.cuda_event for event in route_output_events],
            expert_contributions,
            shared_contributions,
            [event.cuda_event for event in shared_events],
            outputs,
            [event.cuda_event for event in output_events],
            int(sinkhorn_iters),
            float(eps),
        )

    def bf16_gemv_fused(
        value: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run registered-shape BF16 GEMV into a fixed BF16/FP32 buffer."""
        if (
            not value.is_cuda
            or not weight.is_cuda
            or not output.is_cuda
            or value.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or output.dtype not in (torch.bfloat16, torch.float32)
            or value.ndim != 2
            or value.shape[0] != 1
            or weight.ndim != 2
            or weight.shape[1] != value.shape[1]
            or output.shape != (1, weight.shape[0])
            or value.device != weight.device
            or value.device != output.device
        ):
            return None
        return _EXT.bf16_gemv_out(value, weight, output)

    def int4_swiglu_fused(
        x: torch.Tensor,
        gate_packed: torch.Tensor,
        gate_scales: torch.Tensor,
        up_packed: torch.Tensor,
        up_scales: torch.Tensor,
        cols: int,
        group_size: int,
        output: torch.Tensor | None = None,
    ):
        """Fuse two packed INT4 decode GEMVs with their FP32 SwiGLU."""
        if (
            os.environ.get("CCCP_INT4_SWIGLU_FUSED", "1") == "0"
            or not x.is_cuda
            or x.dtype not in (torch.float32, torch.bfloat16)
            or x.ndim != 2
            or x.shape[0] != 1
            or gate_packed.dtype != torch.uint8
            or up_packed.dtype != torch.uint8
            or gate_scales.dtype != torch.float16
            or up_scales.dtype != torch.float16
            or gate_packed.shape != up_packed.shape
            or gate_scales.shape != up_scales.shape
            or group_size != 64
            or cols <= 0
            or cols % 64
        ):
            return None
        return _EXT.int4_swiglu_packed_f32(
            x.contiguous(),
            gate_packed.contiguous(),
            gate_scales.contiguous(),
            up_packed.contiguous(),
            up_scales.contiguous(),
            int(cols),
            int(group_size),
            os.environ.get(
                "CCCP_INT4_SWIGLU_GROUP_VECTOR",
                "0",
            ) != "0",
            output,
        )

else:

    def vq_gemv_fused(x_rows: torch.Tensor, idx: torch.Tensor,
                      cb: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(f"{_EXTENSION_NAME} 扩展不可用：{_ERR}")

    def dense_vq_gemv_packed_fused(*args, **kwargs):
        raise RuntimeError(f"{_EXTENSION_NAME} 扩展不可用：{_ERR}")

    def dense_vq_gemv_grouped_fp8_codebook_fused(*args, **kwargs):
        raise RuntimeError(f"{_EXTENSION_NAME} 扩展不可用：{_ERR}")

    def dense_vq_dequant_packed_fused(*args, **kwargs):
        raise RuntimeError(f"{_EXTENSION_NAME} 扩展不可用：{_ERR}")

    def dense_fp8_quantize_rows_fused(*args, **kwargs):
        return None

    def kda_recurrent_fused(*args, **kwargs):
        return None

    def kda_recurrent_batch_fused(*args, **kwargs):
        return None

    def short_conv3_fused(*args, **kwargs):
        return False

    def qwen35_conv1d_update_fused(*args, **kwargs):
        return None

    def qwen35_delta_recurrent_fused(*args, **kwargs):
        return None

    def qwen35_delta_recurrent_batch_fused(*args, **kwargs):
        return None

    def qwen35_delta_recurrent_batch_checkpoint_fused(*args, **kwargs):
        return None

    def gated_rmsnorm_fused(*args, **kwargs):
        return None

    def packed_moe_topk_fused(*args, **kwargs):
        return None

    def projection_dequant_fused(*args, **kwargs):
        return None

    def projection_expand_native8_fused(*args, **kwargs):
        return None

    def packed_stage_topk_three_projection_fused(*args, **kwargs):
        return None

    def packed_route_slots_fused(*args, **kwargs):
        return False

    def packed_h2d_batch_fused(*args, **kwargs):
        return False

    def moe_mlp_slots_fused(
        x_rows,
        gu_indices,
        gu_codebooks,
        dn_indices,
        dn_codebooks,
        weights,
        limit,
        hidden_workspace,
        out_workspace,
        result,
    ):
        return None

    def moe_mlp_routed_slots_fused(
        x_rows,
        route_ids,
        weights,
        metadata,
        limit,
        hidden_workspace,
        out_workspace,
        result,
        accumulate=False,
    ):
        return None

    def moe_mlp_routed_vv_fused(*args, **kwargs):
        return None

    def moe_mlp_routed_codegemm_fused(*args, **kwargs):
        return None

    def pack_vq_tensor_shard_codegemm(*args, **kwargs):
        return False

    def unpack_vq_codegemm(*args, **kwargs):
        return None

    def expert_dispatch_pack_fused(*args, **kwargs):
        return False

    def tp_peer_copy_fused(*args, **kwargs):
        return False

    def tp_attention_peer_dispatch_fused(*args, **kwargs):
        return False

    def tp_attention_source_pack_fused(*args, **kwargs):
        return False

    def hc_split_fused(mixes, scale, base, hc, iters, eps):
        return None

    def rmsnorm_fused(x, w, eps, output=None):
        return None

    def rmsnorm_bf16_fused(x, w, eps, output=None):
        return None

    def attention_residual_bf16_fused(*args, **kwargs):
        return None

    def gated_activation_bf16_fused(*args, **kwargs):
        return None

    def rope1_fused(x, cos, sin, inverse=False):
        return None

    def glm_rope_qk_fused(q, k, cos, sin):
        return None

    def glm_latent_kv_decode_prepare_fused(*args, **kwargs):
        return None

    def latent_mla_attention_decode_fused(*args, **kwargs):
        return None

    def flashinfer_mla_batch1_plan_fused(*args, **kwargs):
        return False

    def glm_mla_bmm_decode_fused(*args, **kwargs):
        return None

    def glm_merge_scores_fused(a, b, scale):
        return None

    def dsv4_attn_decode_fused(q, win_kv, win_pos, comp_kv, sink, cos, sin, scale):
        return None

    def dsv4_kv_commit_controlled_fused(*args, **kwargs):
        return False

    def dsv4_compressor_step_controlled_fused(*args, **kwargs):
        return False

    def dsv4_attn_decode_controlled_fused(*args, **kwargs):
        return None

    def paged_indexer_query_fp8_fused(*args, **kwargs):
        return False

    def paged_indexer_reduce_logits_fused(*args, **kwargs):
        return False

    def sparse_attention_inverse_rope_fused(*args, **kwargs):
        return None

    def dsv4_hc_pre_fused(x, fn, scale, base, iters, eps):
        return None

    def dsv4_hc_pre_norm_fused(
        x,
        fn,
        scale,
        base,
        norm,
        iters,
        eps,
        output_buffers=None,
    ):
        return None

    def dsv4_hc_post_fused(
        out,
        residual,
        post,
        comb,
        output=None,
    ):
        return None

    def dsv4_hc_post_moe_fused(
        routed,
        shared,
        residual,
        post,
        comb,
        output=None,
    ):
        return None

    def dsv4_route_post_fused(
        scores, bias, mask, top_k
    ):
        return None

    def route_topk_sigmoid_fused(
        logits,
        bias,
        mask,
        top_k,
        routed_scaling,
        output_buffers=None,
    ):
        return None

    def linear_route_topk_sigmoid_fused(*args, **kwargs):
        return None

    glm_route_fused = route_topk_sigmoid_fused

    def paged_gather_bf16_fused(
        page_ptrs, indices, page_items, dim
    ):
        return None

    def hadamard_bf16_fused(x):
        return None

    def int4_gemv_fused(
        x, packed, scales, cols, group_size, output=None, *,
        group_vector=None,
    ):
        return None

    def block_fp8_gemv_fused(*args, **kwargs):
        return None

    def block_fp8_grouped_gemv_fused(*args, **kwargs):
        return None

    def int4_glm_qb_split_fused(*args, **kwargs):
        return None

    def int4_embedding_fused(
        packed,
        scales,
        row,
        cols,
        group_size,
        output=None,
    ):
        return None

    def int4_embedding_device_fused(*args, **kwargs):
        return None

    def glm_norm_qkv_int4_fused(*args, **kwargs):
        return None

    def glm_residual_norm_qkv_int4_fused(*args, **kwargs):
        return None

    def glm_residual_norm_router_fused(*args, **kwargs):
        return None

    def residual_add3_fused(*args, **kwargs):
        return None

    def glm_moe_residual_add_fused(*args, **kwargs):
        return None

    def glm_ep_reduce_residual_fused(*args, **kwargs):
        return None

    def tp_all_rank_reduce_fused(*args, **kwargs):
        return None

    def tp1_moe_finalize_fused(*args, **kwargs):
        return None

    def compressed_state_update_fused(*args, **kwargs):
        return None

    def head_rmsnorm_rope_fused(*args, **kwargs):
        return None

    def tp_all_rank_reduce_from_events_fused(*args, **kwargs):
        return None

    def tp_moe_finalize_from_events_fused(*args, **kwargs):
        return None

    def tp_hidden_add_batch_fused(*args, **kwargs):
        return None

    def tp_hidden_rmsnorm_batch_fused(*args, **kwargs):
        return None

    def tp_hidden_residual_mix_batch_fused(*args, **kwargs):
        return None

    def launch_cuda_graphs_fused(*args, **kwargs):
        return None

    def launch_cuda_graphs_reduce_fused(*args, **kwargs):
        return None

    def make_tp_graph_launch_batch(*args, **kwargs):
        return None

    def make_tp_graph_sequence_batch(*args, **kwargs):
        return None

    def make_tp_graph_dag_batch(*args, **kwargs):
        return None

    def make_tp_raw_graph_dag_batch(*args, **kwargs):
        return None

    def make_tp_no_owner_moe_layer_plan(*args, **kwargs):
        return None

    def make_tp_no_owner_decode_layer_plan(*args, **kwargs):
        return None

    def make_tp_no_owner_hc_decode_layer_plan(*args, **kwargs):
        return None

    def bf16_gemv_fused(*args, **kwargs):
        return None

    def int4_swiglu_fused(
        x,
        gate_packed,
        gate_scales,
        up_packed,
        up_scales,
        cols,
        group_size,
        output=None,
    ):
        return None


# 旧公开名只作为外部脚本的兼容别名；注册层只引用通用名称。
kimi_short_conv3_fused = short_conv3_fused
kimi_kda_recurrent_fused = kda_recurrent_fused
kimi_gated_rmsnorm_fused = gated_rmsnorm_fused
kimi_moe_packed_fused = packed_moe_topk_fused

"""离线 GPU 首次编译必须只依赖发行目录内的 SDK 与 MSVC。"""
from __future__ import annotations

import os
import sys
from pathlib import Path


ENGINE_ROOT = Path(__file__).resolve().parents[1] / "engine" / "CCCP-Engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from cccp import fusedext  # noqa: E402
from cccp import kimi_experts  # noqa: E402


def test_windows_rocm_patch_does_not_touch_cuda_or_cpu(monkeypatch):
    class ForbiddenCppExtensionAccess:
        def __getattr__(self, name):
            raise AssertionError(f"non-HIP path accessed cpp_extension.{name}")

    monkeypatch.setattr(fusedext.torch.version, "hip", None)

    fusedext._patch_windows_rocm_extension_linker(ForbiddenCppExtensionAccess())


def test_windows_hipify_stages_cross_drive_source_and_local_headers(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "portable-launcher" / "csrc"
    source_dir.mkdir(parents=True)
    source = source_dir / "vq_gemv.cu"
    header = source_dir / "codegemm_vq.cuh"
    source.write_text('#include "codegemm_vq.cuh"\n', encoding="utf-8")
    header.write_text("// local HIP header\n", encoding="utf-8")
    output = tmp_path / "local-app-data" / "operator-cache"
    output.mkdir(parents=True)
    monkeypatch.setattr(
        fusedext,
        "_windows_path_drive",
        lambda path: "d:" if Path(path).name == source.name else "c:",
    )

    prepared, aliases = fusedext._prepare_windows_hipify_extra_files(
        [str(source)], output
    )

    staged = Path(prepared[0])
    assert staged.is_relative_to(output)
    assert staged.read_bytes() == source.read_bytes()
    assert (staged.parent / header.name).read_bytes() == header.read_bytes()
    assert aliases == [(str(source.resolve()), str(staged.resolve()))]


def test_windows_hipify_keeps_same_drive_source_in_place(tmp_path, monkeypatch):
    source = tmp_path / "vq_gemv.cu"
    source.write_bytes(b"same-drive source")
    output = tmp_path / "operator-cache"
    output.mkdir()
    monkeypatch.setattr(fusedext, "_windows_path_drive", lambda _path: "c:")

    prepared, aliases = fusedext._prepare_windows_hipify_extra_files(
        [str(source)], output
    )

    assert prepared == [str(source.resolve())]
    assert aliases == []
    assert not (output / "_cccp_hipify_inputs").exists()


def test_gpu_operator_cache_identity_is_stable_across_source_relocation(
    tmp_path, monkeypatch
):
    first = tmp_path / "release-a" / "vq_gemv.cu"
    second = tmp_path / "release-b" / "vq_gemv.cu"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"same operator source")
    second.write_bytes(first.read_bytes())
    cache_root = tmp_path / "persistent-cache"
    monkeypatch.setenv("CCCP_OPERATOR_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(fusedext.torch.version, "hip", None)
    monkeypatch.setattr(fusedext.torch.version, "cuda", "13.0")

    first_identity = fusedext._operator_cache_identity(first, (8, 6))
    second_identity = fusedext._operator_cache_identity(second, (8, 6))

    assert first_identity == second_identity
    assert first_identity[1].is_relative_to(cache_root)
    assert "sm86" in first_identity[0]


def test_gpu_operator_cache_identity_changes_for_arch_or_source(tmp_path, monkeypatch):
    source = tmp_path / "vq_gemv.cu"
    source.write_bytes(b"operator source v1")
    monkeypatch.setenv("CCCP_OPERATOR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(fusedext.torch.version, "hip", None)
    monkeypatch.setattr(fusedext.torch.version, "cuda", "13.0")

    sm86 = fusedext._operator_cache_identity(source, (8, 6))
    sm89 = fusedext._operator_cache_identity(source, (8, 9))
    source.write_bytes(b"operator source v2")
    changed_source = fusedext._operator_cache_identity(source, (8, 6))

    assert sm86[:2] != sm89[:2]
    assert sm86[:2] != changed_source[:2]


def test_bundled_windows_librarian_is_found_without_system_visual_studio(tmp_path, monkeypatch):
    librarian = (
        tmp_path / "toolchain" / "portable" / "Contents" / "VC" / "Tools" /
        "MSVC" / "14.44.35207" / "bin" / "Hostx64" / "x64" / "lib.exe"
    )
    librarian.parent.mkdir(parents=True)
    librarian.write_bytes(b"MZ")
    monkeypatch.setenv("CCCP_LAUNCHER_ROOT", str(tmp_path))
    monkeypatch.setattr(fusedext.shutil, "which", lambda _name: None)

    assert fusedext._bundled_windows_tool("lib.exe") == str(librarian)


def test_cublas_dll_discovery_accepts_pip_wheel_layout(tmp_path):
    dll = tmp_path / "bin" / "x86_64" / "cublas64_13.dll"
    dll.parent.mkdir(parents=True)
    dll.write_bytes(b"MZ")

    assert fusedext._find_windows_cublas_dll(str(tmp_path)) == dll


def test_cudart_discovery_uses_active_cuda13_environment(tmp_path, monkeypatch):
    dll = (
        tmp_path / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" /
        "x86_64" / "cudart64_13.dll"
    )
    dll.parent.mkdir(parents=True)
    dll.write_bytes(b"MZ")
    monkeypatch.setattr(kimi_experts.sys, "prefix", str(tmp_path))

    assert str(dll) in kimi_experts._cudart_candidates()


def test_cudart_loader_uses_discovered_absolute_dll(tmp_path, monkeypatch):
    dll = tmp_path / "cudart64_13.dll"
    dll.write_bytes(b"MZ")
    loaded = object()
    seen = []
    monkeypatch.setattr(kimi_experts, "_CUDART_LIBRARY", None)
    monkeypatch.setattr(kimi_experts, "_cudart_candidates", lambda: (str(dll),))
    monkeypatch.setattr(
        kimi_experts.ctypes,
        "CDLL",
        lambda path: seen.append(path) or loaded,
    )
    monkeypatch.setattr(kimi_experts.os, "add_dll_directory", lambda _path: object())

    assert kimi_experts._cudart_library() is loaded
    assert seen == [str(dll)]


def test_packaged_cuda_root_overrides_inherited_system_cuda_home(tmp_path, monkeypatch):
    cuda_root = tmp_path / "Lib" / "site-packages" / "nvidia" / "cu13"
    nvcc = cuda_root / "bin" / "nvcc.exe"
    (cuda_root / "bin" / "x86_64").mkdir(parents=True)
    nvcc.parent.mkdir(parents=True, exist_ok=True)
    nvcc.write_bytes(b"MZ")
    monkeypatch.setattr(fusedext.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(fusedext.torch.version, "hip", None)
    monkeypatch.setenv("CUDA_HOME", r"C:\unrelated\system\cuda")
    monkeypatch.setenv("CUDA_PATH", r"C:\unrelated\system\cuda")
    # add_dll_directory retains real OS handles and is irrelevant to this path test.
    monkeypatch.setattr(fusedext.os, "add_dll_directory", lambda _path: object())

    fusedext._configure_packaged_gpu_toolchain()

    assert Path(os.environ["CUDA_HOME"]) == cuda_root
    assert Path(os.environ["CUDA_PATH"]) == cuda_root


def test_cuda_architecture_is_selected_from_active_gpu(monkeypatch):
    monkeypatch.delenv("CCCP_CUDA_ARCH", raising=False)
    for capability, expected in (
        ((7, 5), "7.5"),   # RTX 2060/2070/2080
        ((8, 6), "8.6"),   # RTX 3090
        ((8, 9), "8.9"),   # RTX 4090
        ((9, 0), "9.0"),   # H20/H100
        ((12, 0), "12.0"), # RTX 5090
    ):
        monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
        monkeypatch.setattr(
            fusedext.torch.cuda, "get_device_capability", lambda _index, value=capability: value
        )
        monkeypatch.setattr(fusedext.torch.version, "cuda", "13.0")
        monkeypatch.setattr(fusedext.torch.version, "hip", None)

        assert fusedext._select_cuda_architecture() == capability
        assert os.environ["TORCH_CUDA_ARCH_LIST"] == expected


def test_cuda_13_boundary_is_rtx_20_series(monkeypatch):
    monkeypatch.setattr(fusedext.torch.version, "cuda", "13.0")
    monkeypatch.setattr(fusedext.torch.version, "hip", None)
    fusedext._validate_cuda_toolchain_arch((7, 5))

    import pytest
    with pytest.raises(RuntimeError, match="最低支持 SM75"):
        fusedext._validate_cuda_toolchain_arch((6, 1))

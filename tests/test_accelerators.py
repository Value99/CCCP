"""CPU/NVIDIA/AMD 独立环境探测必须以运行时自检为准。"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import launcher.accelerators as accelerators


def test_display_adapter_detection_is_separate_from_runtime(monkeypatch):
    monkeypatch.setattr(accelerators.platform, "system", lambda: "Windows")
    monkeypatch.setattr(accelerators.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='["AMD Radeon 780M Graphics","Basic Display"]', stderr="",
    ))
    report = accelerators.detect_optional_accelerators()
    assert report["amd_display_adapters"] == ["AMD Radeon 780M Graphics"]
    assert report["amd_display_detected"] is True
    assert report["inference_runtimes"] == {}


def test_amd_requires_hip_and_available_device(monkeypatch):
    payload = {
        "ok": True, "torch_version": "2.9.1+rocm7.2.1", "cuda": "",
        "hip": "7.2.1", "available": True, "name": "AMD Radeon 890M",
        "memory": 8 * 2**30, "cccp": True, "error": "",
    }
    monkeypatch.setattr(
        accelerators, "backend_python", lambda settings, backend: Path(sys.executable)
    )
    monkeypatch.setattr(
        accelerators.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""),
    )
    report = accelerators.probe_backend(SimpleNamespace(), "amd", None)
    assert report["ready"] is True
    assert report["compute_runtime"] == "ROCm/HIP 7.2.1"
    assert report["device_memory_gb"] == 8.0


def test_cuda_rejects_rocm_torch_even_when_cuda_api_is_available(monkeypatch):
    payload = {
        "ok": True, "torch_version": "2.9.1+rocm7.2.1", "cuda": "",
        "hip": "7.2.1", "available": True, "name": "AMD Radeon",
        "memory": 0, "cccp": True, "error": "",
    }
    monkeypatch.setattr(
        accelerators, "backend_python", lambda settings, backend: Path(sys.executable)
    )
    monkeypatch.setattr(
        accelerators.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""),
    )
    assert accelerators.probe_backend(SimpleNamespace(), "cuda", None)["ready"] is False

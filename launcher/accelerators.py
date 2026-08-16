"""Isolated CPU/NVIDIA CUDA/AMD ROCm runtime discovery and verification."""
from __future__ import annotations

import json
import os
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .resources import detect_python_path


BACKENDS = ("cpu", "cuda", "amd")


def _display_adapters_windows() -> list[str]:
    if platform.system() != "Windows":
        return []
    command = (
        "$ErrorActionPreference='Stop';"
        "@(Get-PnpDevice -Class Display -PresentOnly | "
        "Where-Object Status -eq 'OK' | ForEach-Object FriendlyName) | "
        "ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            return []
        value = json.loads(result.stdout.strip() or "[]")
        values = [value] if isinstance(value, str) else value
        return [str(item).strip() for item in values if str(item).strip()]
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, TypeError):
        return []


def backend_python(settings, backend: str) -> Path | None:
    """Resolve one configured backend without falling through to another vendor."""
    if backend not in BACKENDS:
        return None
    names = {
        "cpu": ("cpu_python_path", "python_path"),
        "cuda": ("cuda_python_path",),
        "amd": ("amd_python_path",),
    }[backend]
    for name in names:
        value = str(getattr(settings, name, "") or "").strip()
        if value and Path(value).is_file():
            return Path(value).resolve()
    return detect_python_path(backend)


def probe_backend(settings, backend: str, cccp_root: str | Path | None) -> dict:
    python = backend_python(settings, backend)
    base = {
        "backend": backend,
        "label": {"cpu": "CPU", "cuda": "NVIDIA CUDA", "amd": "AMD ROCm/HIP"}[backend],
        "python_path": str(python) if python else "",
        "installed": bool(python),
        "device_available": backend == "cpu",
        "cccp_importable": False,
        "ready": False,
        "torch_version": "",
        "compute_runtime": "",
        "device_name": "",
        "device_memory_gb": 0.0,
        "device_available_memory_gb": 0.0,
        "reason": "",
    }
    if python is None:
        base["reason"] = f"未找到 runtime/{backend}/env/python.exe"
        return base
    code = r'''
import json,sys
result={"ok":False,"torch_version":"","cuda":"","hip":"","available":False,"name":"","memory":0,"free_memory":0,"cccp":False,"error":""}
try:
 import torch
 result["torch_version"]=str(torch.__version__)
 result["cuda"]=str(torch.version.cuda or "")
 result["hip"]=str(torch.version.hip or "")
 result["available"]=bool(torch.cuda.is_available())
 if result["available"]:
  result["name"]=str(torch.cuda.get_device_name(0))
  result["memory"]=int(torch.cuda.get_device_properties(0).total_memory)
  result["free_memory"]=int(torch.cuda.mem_get_info(0)[0])
 import cccp.engine
 result["cccp"]=True
 result["ok"]=True
except Exception as exc:
 result["error"]=f"{type(exc).__name__}: {exc}"
print(json.dumps(result,ensure_ascii=False))
'''
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONNOUSERSITE"] = "1"
    env["CCCP_FUSED"] = "0"  # 探测不触发首次算子编译；启动时按后端自动编译
    if cccp_root:
        paths = [str(Path(cccp_root))]
        vendor = Path(cccp_root) / "_vendor"
        if vendor.is_dir():
            paths.append(str(vendor))
        existing = env.get("PYTHONPATH", "")
        if existing:
            paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(paths)
    try:
        result = subprocess.run(
            [str(python), "-c", code], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, IndexError) as exc:
        base["reason"] = f"运行时自检失败：{exc}"
        return base
    base.update({
        "torch_version": payload.get("torch_version", ""),
        "device_available": bool(payload.get("available")) if backend != "cpu" else True,
        "cccp_importable": bool(payload.get("cccp")),
        "device_name": payload.get("name", ""),
        "device_memory_gb": round(float(payload.get("memory", 0)) / 2**30, 2),
        "device_available_memory_gb": round(
            float(payload.get("free_memory", 0)) / 2**30, 2
        ),
    })
    if not payload.get("ok"):
        base["reason"] = payload.get("error") or "PyTorch/CCCP 导入失败"
        return base
    cuda_version = str(payload.get("cuda") or "")
    hip_version = str(payload.get("hip") or "")
    if backend == "cpu":
        base["compute_runtime"] = "CPU"
        base["ready"] = True
        base["reason"] = "CPU 推理环境可用"
    elif backend == "cuda":
        base["compute_runtime"] = f"CUDA {cuda_version}" if cuda_version else ""
        base["ready"] = bool(payload.get("available") and cuda_version and not hip_version)
        base["reason"] = (
            "NVIDIA CUDA 推理环境可用" if base["ready"]
            else "CUDA 环境已安装，但未检测到可用 NVIDIA CUDA 设备"
        )
    else:
        base["compute_runtime"] = f"ROCm/HIP {hip_version}" if hip_version else ""
        base["ready"] = bool(payload.get("available") and hip_version)
        base["reason"] = (
            "AMD ROCm/HIP 推理环境可用；CCCP 内部映射为 cuda API" if base["ready"]
            else "AMD 环境已安装，但未检测到受支持的 ROCm/HIP 设备"
        )
    return base


def detect_optional_accelerators(settings=None, cccp_root: str | Path | None = None) -> dict:
    adapters = _display_adapters_windows()
    amd = [name for name in adapters if "amd" in name.lower() or "radeon" in name.lower()]
    nvidia = [name for name in adapters if "nvidia" in name.lower()]
    runtimes: dict[str, dict] = {}
    if settings is not None:
        with ThreadPoolExecutor(max_workers=len(BACKENDS)) as executor:
            futures = {
                backend: executor.submit(probe_backend, settings, backend, cccp_root)
                for backend in BACKENDS
            }
            runtimes = {backend: futures[backend].result() for backend in BACKENDS}
    return {
        "display_adapters": adapters,
        "amd_display_adapters": amd,
        "nvidia_display_adapters": nvidia,
        "amd_display_detected": bool(amd),
        "nvidia_display_detected": bool(nvidia),
        "inference_runtimes": runtimes,
        "available_devices": [
            backend for backend, report in runtimes.items() if report["ready"]
        ],
    }

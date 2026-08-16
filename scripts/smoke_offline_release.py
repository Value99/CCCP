"""Start the packaged EXE from its own directory and audit core endpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from urllib.request import urlopen

import psutil


ROOT = Path(__file__).resolve().parents[1]
CURRENT_VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def free_port() -> int:
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def get_json(url: str) -> dict:
    with urlopen(url, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {url}")
        return json.loads(response.read().decode("utf-8"))


def stop_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    children = parent.children(recursive=True)
    for process in reversed(children):
        try:
            process.terminate()
        except psutil.Error:
            pass
    try:
        parent.terminate()
    except psutil.Error:
        pass
    _, alive = psutil.wait_procs(children + [parent], timeout=8)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass


def reset_mutable_data(root: Path) -> None:
    """Return a clean release to its pre-first-run state after the smoke test."""
    data = root / "data"
    if data.exists():
        for child in data.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        data.mkdir(parents=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_directory", type=Path)
    parser.add_argument("--version", default=CURRENT_VERSION)
    args = parser.parse_args()
    root = args.release_directory.resolve()
    executable = root / "CCCP-Launcher.exe"
    required = [
        executable,
        root / "runtime/cpu/env/python.exe",
        root / "runtime/cuda/env/python.exe",
        root / "runtime/amd/env/python.exe",
        root / "engine/CCCP-Engine/cccp/native/cccp_cpu_kernels_v194.pyd",
        root / "toolchain",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing release files:\n" + "\n".join(missing))
    if any((root / name).exists() for name in ("launcher", "webui", "tests", "scripts", "packaging", "build", "dist")):
        raise SystemExit("release contains launcher engineering directories")
    if any((root / "models").rglob("*")):
        raise SystemExit("release models directory is not empty")
    if any((root / "profiles/user").rglob("*")):
        raise SystemExit("release user profile directory is not empty")
    if any(path.is_file() for path in (root / "data").rglob("*")):
        raise SystemExit("release data directory is not pristine before smoke")

    port = free_port()
    child_env = os.environ.copy()
    # The smoke test must not mutate the finished release after its SHA-256
    # manifest has been generated.  The frozen launcher probes the bundled
    # Python environments in child processes, so propagate this guard to the
    # entire process tree rather than relying on the caller's shell.
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["PYTHONNOUSERSITE"] = "1"
    process = subprocess.Popen(
        [str(executable), "--host", "127.0.0.1", "--port", str(port), "--no-shell"],
        cwd=root,
        env=child_env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        deadline = time.monotonic() + 60
        health = None
        while time.monotonic() < deadline:
            try:
                health = get_json(f"http://127.0.0.1:{port}/api/health")
                break
            except Exception:
                if process.poll() is not None:
                    raise RuntimeError(f"packaged EXE exited with {process.returncode}")
                time.sleep(0.5)
        if not health:
            raise RuntimeError("packaged EXE health timeout")
        endpoints = (
            "/api/health", "/api/system", "/api/models", "/api/profiles",
            "/api/training/jobs", "/api/training/corpus", "/api/service/info",
        )
        results = {path: get_json(f"http://127.0.0.1:{port}{path}") for path in endpoints}
        if health.get("version") != args.version or not health.get("cccp_available"):
            raise RuntimeError(f"unexpected health: {health}")
        print(json.dumps({
            "ok": True,
            "version": health["version"],
            "cccp_available": health["cccp_available"],
            "endpoints": list(results),
            "port": port,
        }, ensure_ascii=False, indent=2))
    finally:
        stop_tree(process.pid)
        reset_mutable_data(root)


if __name__ == "__main__":
    main()

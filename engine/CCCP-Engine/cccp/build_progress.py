"""Progress heartbeat for offline CPU/CUDA/HIP operator builds.

Compiler and linker processes can legitimately stay quiet for several minutes.
The launcher consumes these stable marker lines while preserving the compiler's
ordinary stdout/stderr, so a first-run build never looks like a frozen process.
"""
from __future__ import annotations

import os
import threading
import time


_FALSE_VALUES = {"0", "false", "off", "none", "no"}


class OperatorBuildProgress:
    """Emit start/heartbeat/final markers around one blocking JIT build."""

    def __init__(self, backend: str) -> None:
        self.backend = "".join(
            char if char.isalnum() or char in "_-" else "-"
            for char in str(backend).strip()
        ) or "unknown"
        self.enabled = (
            os.environ.get("CCCP_OPERATOR_BUILD_PROGRESS", "1").strip().lower()
            not in _FALSE_VALUES
        )
        try:
            requested = float(os.environ.get("CCCP_OPERATOR_BUILD_HEARTBEAT_S", "5"))
        except ValueError:
            requested = 5.0
        self.interval = min(30.0, max(1.0, requested))
        self.started = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _emit(self, event: str) -> None:
        elapsed = max(0, round(time.monotonic() - self.started))
        try:
            print(
                "[cccp-winui-progress] phase=operator-build "
                f"event={event} backend={self.backend} elapsed={elapsed}",
                flush=True,
            )
        except (BrokenPipeError, OSError):
            pass

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval):
            self._emit("running")

    def __enter__(self) -> "OperatorBuildProgress":
        if not self.enabled:
            return self
        self.started = time.monotonic()
        self._emit("start")
        self._thread = threading.Thread(
            target=self._heartbeat,
            daemon=True,
            name=f"cccp-{self.backend.lower()}-build-progress",
        )
        self._thread.start()
        return self

    def __exit__(self, error_type, _error, _traceback) -> bool:
        if self.enabled:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=0.2)
            self._emit("failed" if error_type is not None else "success")
        return False


def operator_build_progress(backend: str) -> OperatorBuildProgress:
    return OperatorBuildProgress(backend)

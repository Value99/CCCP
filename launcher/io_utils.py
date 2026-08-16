"""Crash-safe helpers for durable launcher files.

The temporary file is created beside the destination so ``os.replace`` stays
on the same filesystem and remains atomic on Windows.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(
    path: str | Path, content: str, *, encoding: str = "utf-8"
) -> None:
    atomic_write_bytes(path, content.encode(encoding))

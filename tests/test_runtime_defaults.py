from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "CCCP-Engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from cccp.runtime_defaults import (  # noqa: E402
    configure_cpu_operator_defaults,
    detect_cpu_cache_bytes,
)


def test_cpu_cache_detection_returns_positive_instance_sizes() -> None:
    l2 = detect_cpu_cache_bytes(2, 2 * 1024**2)
    llc = detect_cpu_cache_bytes(3, 32 * 1024**2)
    assert l2 >= 256 * 1024
    assert llc >= l2
    if os.name == "nt":
        # The target i9-13900H exposes a 24 MiB LLC. This also proves Windows
        # did not silently retain the old 32 MiB fallback.
        assert llc != 32 * 1024**2


def test_cpu_cache_schedule_is_automatic_and_bounded(monkeypatch) -> None:
    monkeypatch.delenv("CCCP_CPU_L2_TASK_TILES", raising=False)
    monkeypatch.delenv("CCCP_PREFILL_MOE_BATCH", raising=False)
    configure_cpu_operator_defaults()
    assert os.environ["CCCP_CPU_L2_TASK_TILES"] in {"1", "4"}
    assert os.environ["CCCP_PREFILL_MOE_BATCH"] in {"8", "32", "64", "128", "256"}

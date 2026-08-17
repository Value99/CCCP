"""CCCP — one CCCP inference runtime for GLM, DeepSeek-V4 and Kimi K3.

The model directory's ``cccp.json`` selects an architecture configuration;
CPU/CUDA VQ, MoE, Attention and tensor-parallel kernels are selected through
the shared ``cccp.ops`` capability registry.  Packed expert indices stay
compact in storage, RAM and VRAM.

Use ``python -m cccp launch chat|serve --model <directory>``.  Per-model chat
entry points were removed after the unified launcher became the only public
runtime, so model differences remain configuration rather than duplicate CLI
systems.

Debug environment
-----------------
The GUI launcher cannot inject shell environment variables into the serve
process.  On startup this package loads ``data/runtime/debug_env.txt`` next to
the installation root (one ``KEY=VALUE`` per line, ``#`` comments) into
``os.environ`` if it exists, so one-off diagnostics such as
``CUDA_LAUNCH_BLOCKING=1`` or ``CCCP_STAGE_VERIFY=1`` can be enabled without a
terminal.  Keep the file empty or delete it for normal operation; every entry
here applies process-wide and can slow inference or change driver behaviour.
"""

import os
from pathlib import Path

__version__ = "1.2.0"


def _load_debug_env() -> None:
    try:
        path = (
            Path(__file__).resolve().parents[3]
            / "data"
            / "runtime"
            / "debug_env.txt"
        )
        text = path.read_text(encoding="utf-8")
    except (OSError, IndexError):
        return
    for line in text.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        os.environ[key.strip()] = value.strip()


_load_debug_env()

__all__ = ["__version__"]

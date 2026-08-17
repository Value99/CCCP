"""Verify that a release can start common NVIDIA GPUs without local JIT."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


CUDA_ARCHITECTURES = ("7.5", "8.6", "8.9", "9.0", "12.0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_directory", type=Path)
    args = parser.parse_args()
    root = args.release_directory.resolve()
    engine = root / "engine" / "CCCP-Engine"
    sys.path.insert(0, str(engine))

    # Identity calculation is GPU-independent.  It binds every binary to the
    # packaged Python, Torch/CUDA ABI and the exact operator source digest.
    # Import the identity helpers without triggering module-level build or
    # device probing on the packaging machine.
    os.environ["CCCP_FUSED"] = "0"
    os.environ["CCCP_FORCE_GPU_BUILD"] = "1"
    from cccp import fusedext

    verified: list[str] = []
    source = Path(fusedext.__file__).with_name("csrc") / "vq_gemv.cu"
    for architecture in CUDA_ARCHITECTURES:
        os.environ["CCCP_CUDA_ARCH"] = architecture
        capability = fusedext._select_cuda_architecture()
        module, _key, backend, arch = fusedext._operator_cache_identity(
            source, capability
        )
        binary = fusedext._packaged_extension_path(
            module, backend, arch, ".pyd"
        )
        if not binary.is_file():
            raise SystemExit(
                f"packaged CUDA operator missing for {architecture}: {binary}"
            )
        verified.append(f"{arch}:{binary.name}")

    os.environ.pop("CCCP_CUDA_ARCH", None)
    print("packaged CUDA operators ok: " + ", ".join(verified))


if __name__ == "__main__":
    main()

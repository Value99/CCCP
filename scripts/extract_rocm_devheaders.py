"""Extract the compact ROCm header set required by CCCP's HIP extension.

The official Windows development wheel stores its SDK in one tar archive.
Expanding the whole archive is both redundant and likely to exceed legacy
Windows path limits, so the offline release keeps only the header families
used by PyTorch and CCCP's fused operator.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath


HEADER_FAMILIES = (
    "thrust/",
    "hipcub/",
    "rocprim/",
    "hipblas/",
    "hipblas-common/",
    "hipsolver/",
    "hipsparse/",
    "hipblaslt/",
    "half/",
)

CUDA_FP8_COMPAT = r"""#pragma once
#include <hip/hip_fp8.h>

using __nv_fp8_e4m3 = __hip_fp8_e4m3;
using __nv_fp8x4_e4m3 = __hip_fp8x4_e4m3;

__host__ __device__ __forceinline__ __hip_bfloat16
__float2bfloat16_rn(float value) {
    return __float2bfloat16(value);
}

__host__ __device__ __forceinline__ __hip_bfloat162
__floats2bfloat162_rn(float low, float high) {
    return __halves2bfloat162(
        __float2bfloat16(low),
        __float2bfloat16(high));
}

__host__ __device__ __forceinline__ __hip_bfloat162
__float2bfloat162_rn(float value) {
    return __floats2bfloat162_rn(value, value);
}

__device__ __forceinline__ __hip_bfloat16
__ldg(const __hip_bfloat16* pointer) {
    return *pointer;
}

__device__ __forceinline__ __hip_bfloat162
__ldg(const __hip_bfloat162* pointer) {
    return *pointer;
}
"""

CUB_COMPAT = r"""#pragma once
#include <hipcub/block/block_radix_sort.hpp>
namespace cub { using namespace hipcub; }
"""


def extract_headers(python_prefix: Path, output: Path) -> int:
    archive = (
        python_prefix
        / "Lib"
        / "site-packages"
        / "rocm_sdk_devel"
        / "_devel.tar"
    )
    if not archive.is_file():
        raise FileNotFoundError(f"ROCm development archive not found: {archive}")
    output.mkdir(parents=True, exist_ok=True)
    prefix = PurePosixPath("_rocm_sdk_devel/include")
    copied = 0
    with tarfile.open(archive, "r") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            archive_path = PurePosixPath(member.name)
            try:
                relative = archive_path.relative_to(prefix)
            except ValueError:
                continue
            normalized = relative.as_posix()
            if not normalized.startswith(HEADER_FAMILIES):
                continue
            target = output.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            copied += 1
    (output / "cuda_fp8.h").write_text(CUDA_FP8_COMPAT, encoding="ascii")
    cub = output / "cub" / "block" / "block_radix_sort.cuh"
    cub.parent.mkdir(parents=True, exist_ok=True)
    cub.write_text(CUB_COMPAT, encoding="ascii")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, default=Path(sys.prefix))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = extract_headers(args.prefix.resolve(), args.output.resolve())
    print(f"Extracted {count} ROCm headers to {args.output.resolve()}")
    return 0 if count else 2


if __name__ == "__main__":
    raise SystemExit(main())

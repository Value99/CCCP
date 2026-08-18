"""v21 k-slice A/B: cold-stream GB/s per shape via payload rotation.

Env: CCCP_INT4_V21_KSLICE=512|2048|unset(auto). Static env cache means
one process = one mode; run twice.
"""
import os

import torch

from cccp import fusedext

torch.manual_seed(0)
dev = "cuda"
ROT = 6  # rotate payloads to defeat L2 residency

print("KSLICE env:", os.environ.get("CCCP_INT4_V21_KSLICE", "<unset:auto>"))

for rows, cols in [
    (17408, 5120),
    (15360, 5120),
    (5120, 5120),
    (5120, 17408),
    (154880, 6144),
]:
    groups = cols // 64
    packs = []
    repacks = []
    scale_list = []
    for r in range(ROT):
        packed = torch.randint(
            0, 256, (rows, cols // 2), dtype=torch.uint8, device=dev
        )
        scales = (torch.randn(rows, groups, device=dev) * 0.02).half()
        packs.append(packed)
        scale_list.append(scales)
        repacks.append(fusedext.int4_repack_v21_fused(packed, cols))
    x = torch.randn(1, cols, device=dev) * 0.1
    wb = rows * cols / 2 + rows * groups * 2

    def one_pass(i):
        return fusedext.int4_gemv_v21_fused(
            x, repacks[i % ROT], scale_list[i % ROT], cols, 64,
            group_vector=True,
        )

    for _ in range(10):
        one_pass(0)
    torch.cuda.synchronize()
    iters = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        one_pass(i)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    print(
        f"rows={rows} cols={cols}: {ms:.4f} ms/pass "
        f"-> {wb / ms / 1e6:.0f} GB/s (rot{ROT})"
    )
print("done")

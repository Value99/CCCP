"""VQWeight int4 快档 A/B:码本 LUT vs int4_g64 映像(批 1/6)+门控行为。

env CCCP_VQ_INT4_IMAGE=on|off|auto(默认 off=纯码本基线)。
"""
import os

import torch

from cccp.kernels import VQWeight

torch.manual_seed(0)
dev = "cuda"
mode = os.environ.get("CCCP_VQ_INT4_IMAGE", "off")
print("CCCP_VQ_INT4_IMAGE =", mode)

for rows, dim, K in [(5120, 8, 256), (15360, 4, 256)]:
    cols = 8192
    B = cols // dim
    idx = torch.randint(0, K, (rows, B), dtype=torch.uint8, device=dev)
    cb = (torch.randn(K, dim, device=dev) * 0.05).float()
    w = VQWeight(idx, cb, cols)
    x1 = torch.randn(1, cols, device=dev) * 0.1
    x6 = torch.randn(6, cols, device=dev) * 0.1

    wref = VQWeight(idx, cb, cols)
    wref._int4 = False  # 强制走码本 LUT 作参照
    ref6 = wref.matmul_T(x6.clone())

    def bench(fn, iters=50):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    t1 = bench(lambda: w.matmul_T(x1))
    t6 = bench(lambda: w.matmul_T(x6))
    if mode == "off":
        print(
            f"rows={rows} dim={dim} cols={cols}: "
            f"LUT T1={t1:.3f} ms T6={t6:.3f} ms"
        )
    else:
        d = (ref6 - w.matmul_T(x6)).abs().max().item()
        rel = d / ref6.abs().max().item()
        print(
            f"rows={rows} dim={dim} cols={cols}: "
            f"int4 T1={t1:.3f} ms T6={t6:.3f} ms rel={rel:.4f}"
        )
print("done")

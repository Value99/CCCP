"""v30 数值+速度:torch 反量化参照对照 + 与 v21b/fp8 同形对照。"""
import torch

from cccp import fusedext

torch.manual_seed(0)
dev = "cuda"
for rows, cols in [
    (5120, 5120),
    (17408, 5120),
    (5120, 17408),
    (2048, 2048),
]:
    groups = cols // 64
    packed = torch.randint(
        0, 256, (rows, cols // 2), dtype=torch.uint8, device=dev
    )
    scales = (torch.randn(rows, groups, device=dev) * 0.02).half()
    x6 = torch.randn(6, cols, device=dev) * 0.1

    q = packed.to(torch.int32)
    lo = (q & 15) - 8
    hi = (q >> 4) - 8
    deq = torch.empty(rows, cols, device=dev)
    deq[:, 0::2] = lo.float() * scales.repeat_interleave(32, dim=1)
    deq[:, 1::2] = hi.float() * scales.repeat_interleave(32, dim=1)
    tref = x6 @ deq.T

    got = fusedext.int4_gemv_v30_fused(
        x6, packed, scales, cols, 64, group_vector=True
    ).float()

    def timeit(fn, iters=100):
        for _ in range(20):
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

    ms = timeit(lambda: fusedext.int4_gemv_v30_fused(
        x6, packed, scales, cols, 64, group_vector=True))
    wb = rows * cols / 2 + rows * groups * 2
    d = (tref - got).abs().max().item()
    rel = d / tref.abs().max().item()
    print(
        f"rows={rows} cols={cols}: max_abs={d:.3e} rel={rel:.4f} "
        f"| v30 {ms:.3f} ms -> {wb / ms / 1e6:.0f} GB/s"
    )
print("done")

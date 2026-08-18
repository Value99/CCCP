"""Timing: 5x per-token vector4 loop vs 1x v1b B=5 pass, per shape."""
import torch

from cccp import fusedext

torch.manual_seed(0)
dev = "cuda"


def time_fn(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


for rows, cols in [
    (5120, 5120),
    (17408, 5120),
    (5120, 17408),
    (15360, 5120),
    (2048, 2048),
]:
    groups = cols // 64
    packed = torch.randint(
        0, 256, (rows, cols // 2), dtype=torch.uint8, device=dev
    )
    scales = (torch.randn(rows, groups, device=dev) * 0.02).half()
    x5 = torch.randn(5, cols, device=dev) * 0.1
    wb = rows * cols / 2 + rows * groups * 2  # weight bytes per pass

    def loop5():
        for b in range(5):
            fusedext.int4_gemv_fused(
                x5[b : b + 1].contiguous(),
                packed,
                scales,
                cols,
                64,
                group_vector=True,
            )

    def batch5():
        fusedext.int4_gemv_v1b_fused(
            x5, packed, scales, cols, 64, group_vector=True
        )

    t_loop = time_fn(loop5)
    t_b = time_fn(batch5)
    print(
        f"rows={rows} cols={cols}: 5xv1={t_loop:.3f}ms "
        f"({wb * 5 / t_loop / 1e6:.0f}GB/s eff) | "
        f"v1b={t_b:.3f}ms ({wb / t_b / 1e6:.0f}GB/s) | "
        f"speedup={t_loop / t_b:.2f}x"
    )
print("done")

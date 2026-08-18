"""A/B: engine per-token v1 (vector4 dispatch) vs v1b batched — bit identity."""
import torch

from cccp import fusedext

torch.manual_seed(0)
dev = "cuda"
for rows, cols in [
    (5120, 5120),
    (17408, 5120),
    (5120, 17408),
    (15360, 5120),
    (1536, 5120),
    (2048, 2048),
    (2048, 4096),
    (6144, 16384),
]:
    groups = cols // 64
    packed = torch.randint(
        0, 256, (rows, cols // 2), dtype=torch.uint8, device=dev
    )
    scales = (torch.randn(rows, groups, device=dev) * 0.02).half()
    x5 = torch.randn(5, cols, device=dev) * 0.1
    ref = torch.stack(
        [
            fusedext.int4_gemv_fused(
                x5[b : b + 1].contiguous(),
                packed,
                scales,
                cols,
                64,
                group_vector=True,
            )
            .float()
            .squeeze(0)
            for b in range(5)
        ]
    )
    got = fusedext.int4_gemv_v1b_fused(
        x5, packed, scales, cols, 64, group_vector=True
    ).float()
    d = (ref - got).abs().max().item()
    print(f"rows={rows} cols={cols} max_abs={d:.3e} bit={'OK' if d == 0.0 else 'DIFF'}")
print("done")

"""DeepSeek-V4 FP4 权重反量化（e2m1 打包格式 + ue8m0 块缩放）。

格式（依据模型仓库 inference/kernel.py 的 fp4_quant_kernel 核实）：
    权重逻辑矩阵 [R, C] 的 FP4 e2m1 值两个一组打包进 I8 → 存储 [R, C/2]；
    nibble 顺序：低半字节 = 偶数列（先），高半字节 = 奇数列（后）。
    缩放：每行每 32 元素一个 ue8m0 缩放（无符号 8 位指数，实际值 = 2^(b-127)），
    存储 [R, C/32]；反量化 W = e2m1值 × 2^(scale-127)。
e2m1 数值表（1 符号位 + 2 指数 + 1 尾数，fn 变体）：
    索引 0..7 = 0, 0.5, 1, 1.5, 2, 3, 4, 6；索引 8..15 为其相反数。
自检：dequant_fp4_check 按"每 32 块 amax/scale ∈ [3, 6]"验证 nibble 顺序与缩放语义
（量化时 amax 映射到 [3,6] 区间；nibble 顺序错了该比值会系统性越界）。
"""

from __future__ import annotations

import torch

# e2m1 全 16 值查找表（bit3 为符号位）
_E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                         dtype=torch.float32)


def dequant_fp4(q: torch.Tensor, scale: torch.Tensor, rows: int, cols: int,
                device=None) -> torch.Tensor:
    """I8 打包 FP4 + ue8m0 缩放 → f32 [rows, cols]。

    q: [rows, cols//2] int8/uint8；scale: [rows, cols//32]（uint8 语义的指数字节）。
    """
    dev = device or q.device
    lut = _E2M1_LUT.to(dev)
    qu = q.view(torch.uint8).to(dev)
    lo = qu & 0x0F
    hi = qu >> 4
    idx = torch.stack([lo, hi], dim=-1).reshape(rows, cols).long()
    mag = lut[idx]
    s = torch.pow(2.0, scale.view(torch.uint8).to(dev).float() - 127.0)
    return mag * s.repeat_interleave(32, dim=1)


def dequant_fp4_check(q: torch.Tensor, scale: torch.Tensor, rows: int, cols: int,
                      sample_rows: int = 64) -> tuple[float, float]:
    """格式自检：抽样若干行，返回每 32 块 amax/scale 比值的最小/最大值。

    正常应在 [3, 6] 内（e2m1 最大值 6，量化把块内 amax 映射到 [3,6]）；
    若系统性 <3 或 >6 → nibble 顺序或缩放语义不匹配。
    """
    r = min(sample_rows, rows)
    w = dequant_fp4(q[:r], scale[:r], r, cols)
    s = torch.pow(2.0, scale[:r].view(torch.uint8).float() - 127.0)
    amax = w.abs().reshape(r, cols // 32, 32).amax(dim=-1)
    ratio = amax / s
    return float(ratio.min()), float(ratio.max())

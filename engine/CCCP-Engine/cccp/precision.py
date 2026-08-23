"""CCCP 精度策略层：全框架计算路径统一从这里取计算 dtype（中文说明）。

设计动机：VQ/int4 量化本身的误差（相对 ~0.3-0.4）比半精度 GEMM 的舍入误差
（fp16 ~1e-3、bf16 ~8e-3）低两个数量级以上，因此 GEMM 内积可走半精度张量核，
对输出分布的影响可忽略（验收门：dspark_check 贪心/投机逐字一致 + KL 差 <0.01）。

auto 策略（按硬件能力自适应，换卡零改动）：
  - sm_8.x+（Ampere 及更新：3090/4090/A100/H100）→ bf16（动态范围好，无溢出问题）；
  - sm_7.x（Turing：2080/T4，无 bf16 硬件）→ fp16（张量核，~2× fp32 GEMM）；
  - CPU / 其他 → fp32（CPU 半精度无加速且部分算子不支持）。
环境变量 CCCP_COMPUTE_DTYPE=fp32|fp16|bf16 可强制覆盖（调试/对照用）。
"""

from __future__ import annotations

import os

import torch

_AUTO_CACHE: dict[str, torch.dtype] = {}


def compute_dtype(device=None) -> torch.dtype:
    """返回当前设备应使用的 GEMM 计算 dtype（见模块 docstring 的策略表）。"""
    ov = os.environ.get("CCCP_COMPUTE_DTYPE", "auto").strip().lower()
    if ov in ("fp32", "float32", "f32"):
        return torch.float32
    if ov in ("fp16", "float16", "f16", "half"):
        return torch.float16
    if ov in ("bf16", "bfloat16"):
        return torch.bfloat16
    dev = torch.device(device) if device is not None else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        return torch.float32
    key = str(dev)
    dt = _AUTO_CACHE.get(key)
    if dt is None:
        try:
            major, _minor = torch.cuda.get_device_capability(dev)
        except Exception:
            major = 0
        dt = torch.bfloat16 if major >= 8 else \
            (torch.float16 if major == 7 else torch.float32)
        _AUTO_CACHE[key] = dt
    return dt

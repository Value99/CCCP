"""VQWeight int4 快档(CCCP_VQ_INT4_IMAGE)CPU 回归:开关语义+数值。"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "engine" / "CCCP-Engine")
)

from cccp.kernels import VQWeight  # noqa: E402


def _make(rows=256, cols=512, dim=8, seed=0):
    torch.manual_seed(seed)
    idx = torch.randint(0, 256, (rows, cols // dim), dtype=torch.uint8)
    cb = (torch.randn(256, dim) * 0.05).float()
    return VQWeight(idx, cb, cols)


def test_default_off_keeps_lut(monkeypatch):
    monkeypatch.delenv("CCCP_VQ_INT4_IMAGE", raising=False)
    w = _make()
    x = torch.randn(2, 512) * 0.1
    assert w._int4_image(x) is None  # 默认零增长,走码本 LUT


def test_cpu_on_routes_int4(monkeypatch):
    monkeypatch.setenv("CCCP_VQ_INT4_IMAGE", "on")
    w = _make()
    wref = _make()
    wref._int4 = False
    x1 = torch.randn(1, 512) * 0.1
    x6 = torch.randn(6, 512) * 0.1
    got1 = w.matmul_T(x1.clone())
    got6 = w.matmul_T(x6.clone())
    ref1 = wref.matmul_T(x1.clone())
    ref6 = wref.matmul_T(x6.clone())
    # int4 对码本值的二次量化:max 偏差与组幅度同量级,相对值有界。
    rel6 = (got6 - ref6).abs().max().item() / ref6.abs().max().item()
    rel1 = (got1 - ref1).abs().max().item() / ref1.abs().max().item()
    assert rel6 < 0.2
    assert rel1 < 0.2


def test_cpu_auto_stays_off(monkeypatch):
    monkeypatch.setenv("CCCP_VQ_INT4_IMAGE", "auto")
    w = _make()
    x = torch.randn(2, 512)
    assert w._int4_image(x) is None  # CPU auto 不自动开(显式 on 才编译)

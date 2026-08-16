from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "CCCP-Engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from cccp.cpuext import attention_decode_cpu, block_fp8_gemv_cpu  # noqa: E402


def _e4m3_finite_bytes(shape: tuple[int, ...]) -> torch.Tensor:
    raw = torch.randint(0, 256, shape, dtype=torch.uint8)
    # E4M3FN reserves magnitude 0x7f for NaN.  The production weights are
    # finite, so keep the reference fixture in the same domain.
    magnitude = raw & 0x7F
    raw[magnitude == 0x7F] = 0
    return raw


def test_avx2_block_fp8_gemv_matches_compact_reference() -> None:
    torch.manual_seed(5090)
    rows, cols, block = 7, 256, 128
    weights = _e4m3_finite_bytes((rows, cols))
    scales = torch.rand((1, 2), dtype=torch.float32) * 0.05 + 0.01
    value = torch.randn((1, cols), dtype=torch.float32)

    decoded = weights.view(torch.float8_e4m3fn).float()
    logical = torch.cat(
        (
            decoded[:, :block] * scales[0, 0],
            decoded[:, block:] * scales[0, 1],
        ),
        dim=1,
    )
    expected = logical @ value[0]
    actual = block_fp8_gemv_cpu(value, weights, scales, cols, block)

    assert actual is not None
    torch.testing.assert_close(
        actual.float().reshape(-1), expected, rtol=2e-5, atol=2e-4
    )


def test_avx2_attention_decode_matches_reference() -> None:
    torch.manual_seed(5090)
    batch, heads, dim = 1, 3, 16
    raw_count, selected_count, rope_pairs = 5, 2, 4
    query = torch.randn((batch, heads, dim), dtype=torch.float32)
    raw = torch.randn((batch, raw_count, dim), dtype=torch.float32)
    positions = torch.tensor([[0, 1, -1, 3, 4]], dtype=torch.long)
    selected = torch.randn(
        (batch, selected_count, dim), dtype=torch.float32
    )
    sink = torch.randn((heads,), dtype=torch.float32)
    cos = torch.randn((1, 1, 1, rope_pairs), dtype=torch.float32)
    sin = torch.randn((1, 1, 1, rope_pairs), dtype=torch.float32)
    scale = dim**-0.5

    actual = attention_decode_cpu(
        query, raw, positions, selected, sink, cos, sin, scale
    )
    assert actual is not None

    expected = torch.zeros_like(query)
    sources = torch.cat((raw, selected), dim=1)
    valid = torch.cat(
        (
            positions >= 0,
            torch.ones((batch, selected_count), dtype=torch.bool),
        ),
        dim=1,
    )
    for head in range(heads):
        scores = (sources[0] @ query[0, head]) * scale
        scores = scores.masked_fill(~valid[0], float("-inf"))
        maximum = torch.maximum(scores.max(), sink[head])
        probabilities = torch.exp(scores - maximum)
        probabilities = torch.where(valid[0], probabilities, 0.0)
        denominator = probabilities.sum() + torch.exp(sink[head] - maximum)
        row = (probabilities[:, None] * sources[0]).sum(0) / denominator
        rope_start = dim - rope_pairs * 2
        first = row[rope_start::2].clone()
        second = row[rope_start + 1::2].clone()
        row[rope_start::2] = first * cos.flatten() + second * sin.flatten()
        row[rope_start + 1::2] = -first * sin.flatten() + second * cos.flatten()
        expected[0, head] = row

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

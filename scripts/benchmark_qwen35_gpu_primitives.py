"""Correctness and latency probe for Qwen3.5 CUDA recurrent primitives."""

from __future__ import annotations

import json
import math
import time
from types import SimpleNamespace

import torch

from cccp.dense_vq import DenseVQLinear, DenseVQLinearGroup
from cccp.fusedext import (
    qwen35_conv1d_update_fused,
    qwen35_delta_recurrent_batch_fused,
    qwen35_delta_recurrent_fused,
)


def _reference_delta(query, key, value, gate, beta, state):
    query_norm = torch.nn.functional.normalize(query.float(), dim=-1)
    key_norm = torch.nn.functional.normalize(key.float(), dim=-1)
    state.mul_(gate.float().exp()[:, None, None])
    prediction = torch.einsum("hkv,hk->hv", state, key_norm)
    delta = (value.float() - prediction) * beta.float()[:, None]
    state.add_(key_norm[:, :, None] * delta[:, None, :])
    output = torch.einsum("hkv,hk->hv", state, query_norm)
    return (output / math.sqrt(query.shape[-1])).to(torch.bfloat16)


def main() -> int:
    torch.manual_seed(5090)
    device = torch.device("cuda")
    heads = key_dim = value_dim = 128
    heads = 48
    query = torch.randn(heads, key_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn(heads, value_dim, device=device, dtype=torch.bfloat16)
    gate = -torch.rand(heads, device=device, dtype=torch.float32) * 0.2
    beta = torch.rand(heads, device=device, dtype=torch.bfloat16)
    state = torch.randn(
        heads, key_dim, value_dim, device=device, dtype=torch.float32
    ) * 0.01
    expected_state = state.clone()
    expected = _reference_delta(
        query, key, value, gate, beta, expected_state
    )
    actual_state = state.clone()
    actual = torch.empty_like(value)
    result = qwen35_delta_recurrent_fused(
        query, key, value, gate, beta, actual_state, actual
    )
    if result is None:
        raise RuntimeError("Qwen delta kernel rejected valid tensors")
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        actual_state, expected_state, rtol=2e-5, atol=2e-5
    )

    tokens = 8
    batch_query = torch.randn(
        tokens, heads, key_dim, device=device, dtype=torch.bfloat16
    )
    batch_key = torch.randn_like(batch_query)
    batch_value = torch.randn(
        tokens, heads, value_dim, device=device, dtype=torch.bfloat16
    )
    batch_gate = (
        -torch.rand(tokens, heads, device=device, dtype=torch.float32) * 0.2
    )
    batch_beta = torch.rand(
        tokens, heads, device=device, dtype=torch.bfloat16
    )
    batch_state = torch.randn(
        heads, key_dim, value_dim, device=device, dtype=torch.float32
    ) * 0.01
    expected_batch_state = batch_state.clone()
    expected_batch = torch.empty_like(batch_value)
    for token in range(tokens):
        expected_batch[token] = _reference_delta(
            batch_query[token],
            batch_key[token],
            batch_value[token],
            batch_gate[token],
            batch_beta[token],
            expected_batch_state,
        )
    actual_batch_state = batch_state.clone()
    actual_batch = torch.empty_like(batch_value)
    result = qwen35_delta_recurrent_batch_fused(
        batch_query,
        batch_key,
        batch_value,
        batch_gate,
        batch_beta,
        actual_batch_state,
        actual_batch,
    )
    if result is None:
        raise RuntimeError("Qwen batched delta kernel rejected valid tensors")
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual_batch.float(), expected_batch.float(), rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        actual_batch_state, expected_batch_state, rtol=2e-5, atol=2e-5
    )

    batch_iterations = 100
    for _ in range(10):
        qwen35_delta_recurrent_batch_fused(
            batch_query,
            batch_key,
            batch_value,
            batch_gate,
            batch_beta,
            actual_batch_state,
            actual_batch,
        )
    torch.cuda.synchronize()
    batch_started = time.perf_counter()
    for _ in range(batch_iterations):
        qwen35_delta_recurrent_batch_fused(
            batch_query,
            batch_key,
            batch_value,
            batch_gate,
            batch_beta,
            actual_batch_state,
            actual_batch,
        )
    torch.cuda.synchronize()
    batch_ms = (
        time.perf_counter() - batch_started
    ) * 1000.0 / batch_iterations

    channels, width = 10240, 4
    conv_input = torch.randn(
        1, channels, 1, device=device, dtype=torch.bfloat16
    )
    conv_state = torch.randn(
        1, channels, width, device=device, dtype=torch.bfloat16
    )
    conv_weight = torch.randn(
        channels, width, device=device, dtype=torch.bfloat16
    )
    expected_conv_state = torch.cat(
        [conv_state[:, :, 1:], conv_input], dim=-1
    )
    expected_conv = torch.nn.functional.silu(
        (expected_conv_state.float() * conv_weight.float()).sum(-1, keepdim=True)
    ).to(torch.bfloat16)
    actual_conv_state = conv_state.clone()
    actual_conv = torch.empty_like(conv_input)
    result = qwen35_conv1d_update_fused(
        conv_input, actual_conv_state, conv_weight, actual_conv
    )
    if result is None:
        raise RuntimeError("Qwen convolution kernel rejected valid tensors")
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual_conv.float(), expected_conv.float(), rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(actual_conv_state, expected_conv_state)

    # Generic Dense VQ projection coalescing: two architecture-selected
    # Linears sharing the exact same input must execute as one combined
    # projection while returning independent row views.
    group_input = torch.randn(
        3, 64, device=device, dtype=torch.bfloat16
    )
    group_weights = (
        torch.randn(48, 64, device=device, dtype=torch.bfloat16),
        torch.randn(80, 64, device=device, dtype=torch.bfloat16),
    )
    group_linears = tuple(
        DenseVQLinear(
            SimpleNamespace(
                rows=weight.shape[0],
                cols=weight.shape[1],
                blocks=weight.shape[1] // 4,
                bits=8,
                layout="bf16",
                source_bits=8,
                raw=weight,
                cb=torch.empty(0, device=device),
            ),
            name=f"probe.{index}",
        )
        for index, weight in enumerate(group_weights)
    )
    projection_group = DenseVQLinearGroup(group_linears)
    group_views = tuple(
        projection_group.view(index) for index in range(len(group_linears))
    )
    group_actual = tuple(view(group_input) for view in group_views)
    for actual_part, weight in zip(group_actual, group_weights):
        torch.testing.assert_close(
            actual_part,
            torch.nn.functional.linear(group_input, weight),
        )

    for _ in range(10):
        qwen35_delta_recurrent_fused(
            query, key, value, gate, beta, actual_state, actual
        )
        qwen35_conv1d_update_fused(
            conv_input, actual_conv_state, conv_weight, actual_conv
        )
    torch.cuda.synchronize()
    iterations = 200
    started = time.perf_counter()
    for _ in range(iterations):
        qwen35_delta_recurrent_fused(
            query, key, value, gate, beta, actual_state, actual
        )
        qwen35_conv1d_update_fused(
            conv_input, actual_conv_state, conv_weight, actual_conv
        )
    torch.cuda.synchronize()
    pair_ms = (time.perf_counter() - started) * 1000.0 / iterations
    print("[cccp-qwen35-gpu-primitives] " + json.dumps({
        "delta_and_conv_ms": pair_ms,
        "projected_48_layers_ms": pair_ms * 48,
        "ordered_batch_tokens": tokens,
        "ordered_batch_ms": batch_ms,
        "ordered_batch_tokens_per_second_per_layer": tokens / (batch_ms / 1000.0),
        "projection_group_rows": projection_group.rows,
        "projection_group_correct": True,
        "correct": True,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

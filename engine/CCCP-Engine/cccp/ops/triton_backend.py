"""Optional Triton implementations for model-independent operators."""

from __future__ import annotations

import importlib.util
import os

import torch

from .registry import OperatorRegistry
from .spec import OperatorCapability


def available() -> bool:
    """Return whether the validated FLA Triton backend can be imported."""
    return (
        importlib.util.find_spec("triton") is not None
        and importlib.util.find_spec("fla") is not None
    )


def _triton_prefill_enabled() -> bool:
    """Return whether optional Triton prefill kernels are opt-in enabled.

    The kernels are useful performance experiments, but their accumulation
    order is not bit-for-bit identical to the portable CUDA/reference paths.
    Keep numerical behavior stable by default and require an explicit
    ``CCCP_PREFILL_ATTN_TRITON=1`` (or another truthy value) before selecting
    either Triton prefill implementation.
    """
    value = os.environ.get("CCCP_PREFILL_ATTN_TRITON", "0")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _ordered_kda_chunk_prefill(**kwargs):
    """Run a safe-gated KDA scan in parallel 64-token chunks."""
    # Triton KDA is an optional numerical/performance experiment.  Returning
    # ``None`` lets the public dispatcher select the CUDA or reference backend
    # without importing or compiling Triton in the default configuration.
    if not _triton_prefill_enabled():
        return None
    query = kwargs["query"]
    key = kwargs["key"]
    value = kwargs["value"]
    gate = kwargs["gate"]
    beta = kwargs["beta"]
    state = kwargs["state"]
    output = kwargs["output"]
    if (
        not query.is_cuda
        or query.dtype != torch.bfloat16
        or query.ndim != 3
        or key.shape != query.shape
        or key.dtype != query.dtype
        or gate.shape != query.shape
        or gate.dtype != query.dtype
        or value.ndim != 3
        or value.shape[:2] != query.shape[:2]
        or value.dtype != query.dtype
        or beta.shape != query.shape[:2]
        or state.shape
        != (query.shape[1], value.shape[2], query.shape[2])
        or state.dtype != torch.float32
        or output.shape != value.shape
        or output.dtype != query.dtype
        or query.shape[2] > 256
    ):
        return None

    from fla.ops.kda import chunk_kda

    with torch.cuda.device(query.device):
        result, final_state = chunk_kda(
            query.contiguous().unsqueeze(0),
            key.contiguous().unsqueeze(0),
            value.contiguous().unsqueeze(0),
            gate.contiguous().unsqueeze(0),
            beta.contiguous().unsqueeze(0),
            initial_state=state.unsqueeze(0),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=float(kwargs.get("lower_bound", -5.0)),
            A_log=kwargs["a_log"].float().contiguous(),
            dt_bias=kwargs["dt_bias"].float().contiguous(),
            state_v_first=True,
            chunk_size=64,
        )
        output.copy_(result.squeeze(0))
        state.copy_(final_state.squeeze(0))
    return output


def _causal_latent_prefill(**kwargs):
    """Run causal latent attention with a blockwise online softmax.

    The fallback implementation in :mod:`attention_prefill` materialises a
    score tile for every query micro-batch.  That is useful as a portable
    reference, but for a long prefill it creates a large amount of temporary
    memory and launches several GEMMs per tile.  This kernel assigns one
    program to each ``(query row, local head)`` pair and walks the visible key
    cache in blocks while carrying ``(max, sum, weighted-value)`` state.  It
    therefore never allocates a ``[heads, rows, history]`` score tensor.

    This is deliberately registered by tensor capability only; there is no
    model-specific dispatch here.  Unsupported layouts return ``None`` so the
    normal CUDA/PyTorch implementation remains the safe fallback.
    """
    # Keep an unconditional model-independent fallback.  Registry selection
    # happens before tensor capability details are available, so a registered
    # Triton implementation must decline gracefully for unusual dimensions or
    # a transient compilation failure.
    from .attention_prefill import causal_latent_prefill as _reference

    def fallback():
        return _reference(**kwargs)

    # Keep a model-independent escape hatch for numerical A/B validation.
    # The Triton kernel intentionally accumulates dot products and online
    # softmax state in FP32, while some CUDA reference paths use BF16 GEMM
    # accumulation.  Callers can therefore compare the exact reference
    # contract without changing registry selection or model code.
    if not _triton_prefill_enabled():
        return fallback()

    query_nope = kwargs["query_nope"]
    query_rope = kwargs["query_rope"]
    latent_cache = kwargs["latent_cache"]
    rope_cache = kwargs["rope_cache"]
    query_start = int(kwargs["query_start"])
    scale_denominator = float(kwargs["scale_denominator"])
    output = kwargs.get("output")
    if (
        not query_nope.is_cuda
        or query_nope.dtype != torch.bfloat16
        or query_rope.dtype != torch.bfloat16
        or latent_cache.dtype != torch.bfloat16
        or rope_cache.dtype != torch.bfloat16
        or query_nope.ndim != 3
        or query_rope.ndim != 3
        or query_nope.shape[:2] != query_rope.shape[:2]
        or latent_cache.ndim != 2
        or rope_cache.ndim != 2
        or query_nope.shape[2] != latent_cache.shape[1]
        or query_rope.shape[2] != rope_cache.shape[1]
        or latent_cache.shape[0] != rope_cache.shape[0]
        or query_start < 0
        or scale_denominator <= 0.0
        or query_nope.device != query_rope.device
        or query_nope.device != latent_cache.device
        or query_nope.device != rope_cache.device
    ):
        return fallback()

    rows, heads, latent_dim = map(int, query_nope.shape)
    rope_dim = int(query_rope.shape[2])
    capacity = int(latent_cache.shape[0])
    if query_start + rows > capacity or rows <= 0:
        return fallback()
    # Keep compilation bounded.  These limits cover the latent sizes used by
    # current MLA models while allowing the reference path for unusual ones.
    if latent_dim > 1024 or rope_dim > 256:
        return fallback()
    if output is None:
        output = torch.empty_like(query_nope)
    if (
        output.shape != query_nope.shape
        or output.dtype != query_nope.dtype
        or output.device != query_nope.device
        or not output.is_contiguous()
    ):
        return fallback()

    try:
        import triton
        import triton.language as tl
    except (ImportError, ModuleNotFoundError):
        return fallback()

    # Defining the kernel lazily keeps importing CCCP possible on CPU-only
    # machines.  Triton caches the generated variants by constexpr shape.
    @triton.jit
    def _kernel(
        qn_ptr,
        qr_ptr,
        kv_ptr,
        kr_ptr,
        out_ptr,
        rows,
        heads,
        latent_dim,
        rope_dim,
        capacity,
        query_start,
        scale,
        qn_row_stride,
        qn_head_stride,
        qr_row_stride,
        qr_head_stride,
        kv_row_stride,
        kr_row_stride,
        out_row_stride,
        out_head_stride,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid // heads
        head = pid % heads
        if row >= rows:
            return

        d = tl.arange(0, BLOCK_D)
        r = tl.arange(0, BLOCK_R)
        d_mask = d < latent_dim
        r_mask = r < rope_dim
        qn = tl.load(
            qn_ptr + row * qn_row_stride + head * qn_head_stride + d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        qr = tl.load(
            qr_ptr + row * qr_row_stride + head * qr_head_stride + r,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)

        visible = query_start + row + 1
        running_max = tl.full((), float("-inf"), tl.float32)
        running_sum = tl.zeros((), dtype=tl.float32)
        running_value = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for key_start in tl.range(0, visible, BLOCK_K):
            key = key_start + tl.arange(0, BLOCK_K)
            key_mask = key < visible
            key_mask = key_mask & (key < capacity)

            k_offsets = key[:, None] * kv_row_stride + d[None, :]
            k_block = tl.load(
                kv_ptr + k_offsets,
                mask=key_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            kr_offsets = key[:, None] * kr_row_stride + r[None, :]
            kr_block = tl.load(
                kr_ptr + kr_offsets,
                mask=key_mask[:, None] & r_mask[None, :],
                other=0.0,
            )
            # Elementwise reductions are accepted across Triton versions and
            # avoid materialising a singleton ``[BLOCK_K, 1]`` dot result.
            score_nope = tl.sum(
                k_block.to(tl.float32) * qn[None, :], axis=1
            )
            score_rope = tl.sum(
                kr_block.to(tl.float32) * qr[None, :], axis=1
            )
            scores = (score_nope + score_rope) * scale
            scores = tl.where(key_mask, scores, float("-inf"))
            block_max = tl.max(scores, axis=0)
            next_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - next_max)
            weights = tl.exp(scores - next_max)
            weights = tl.where(key_mask, weights, 0.0)

            v_block = tl.load(
                kv_ptr + k_offsets,
                mask=key_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            weighted_value = tl.sum(
                weights[:, None] * v_block, axis=0
            )
            running_value = running_value * old_scale + weighted_value
            running_sum = running_sum * old_scale + tl.sum(weights, axis=0)
            running_max = next_max

        result = running_value / running_sum
        tl.store(
            out_ptr + row * out_row_stride + head * out_head_stride + d,
            result,
            mask=d_mask,
        )

    def _env_power_of_two(name: str, default: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            value = default
        value = max(16, min(128, value))
        return 1 << (value.bit_length() - 1)

    block_k = _env_power_of_two("CCCP_PREFILL_ATTN_TRITON_BLOCK", 64)
    block_d = triton.next_power_of_2(latent_dim)
    block_r = triton.next_power_of_2(rope_dim)
    try:
        with torch.cuda.device(query_nope.device):
            _kernel[(rows * heads,)](
                query_nope,
                query_rope,
                latent_cache,
                rope_cache,
                output,
                rows,
                heads,
                latent_dim,
                rope_dim,
                capacity,
                query_start,
                1.0 / scale_denominator,
                query_nope.stride(0),
                query_nope.stride(1),
                query_rope.stride(0),
                query_rope.stride(1),
                latent_cache.stride(0),
                rope_cache.stride(0),
                output.stride(0),
                output.stride(1),
                BLOCK_K=block_k,
                BLOCK_D=block_d,
                BLOCK_R=block_r,
                num_warps=8 if latent_dim >= 512 else 4,
                num_stages=2,
            )
    except (RuntimeError, ValueError):
        # A backend compiler may reject a new GPU/dimension combination.  Do
        # not make model startup depend on that optional specialization.
        return fallback()
    return output


def register(registry: OperatorRegistry) -> None:
    # Register the capability whenever Triton is installed.  The
    # implementations themselves perform the runtime opt-in check, so a
    # diagnostic can set CCCP_PREFILL_ATTN_TRITON after CCCP import/registry
    # construction while the default path still declines safely.
    if not available():
        return
    registry.register(
        "cuda.attention.ordered_kda_scan.triton_chunk64",
        OperatorCapability(
            operation="attention_step:kda_chunk_prefill",
            device_types=("cuda",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(1, 8193)),
            dtypes=("bfloat16",),
            head_dims=(32, 64, 128, 256),
            architecture_features=("triton",),
        ),
        _ordered_kda_chunk_prefill,
        priority=200,
    )
    registry.register(
        "cuda.attention.causal_latent.prefill.triton_online",
        OperatorCapability(
            operation="attention_step:causal_latent_prefill",
            device_types=("cuda",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(1, 8193)),
            dtypes=("bfloat16",),
            architecture_features=("triton",),
        ),
        _causal_latent_prefill,
        priority=200,
    )


__all__ = ["available", "register"]

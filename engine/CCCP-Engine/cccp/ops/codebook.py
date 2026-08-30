"""Public model-independent codebook storage and Linear executors.

Architecture adapters import only this public surface.  The storage objects
live in :mod:`cccp.codebook_storage`; no model adapter owns a private
dequantisation, codebook compilation, or matrix-multiplication policy.
"""

from dataclasses import dataclass
from typing import Literal

import torch

from ..codebook_storage import (
    DenseBF16Linear,
    DenseBF16LinearGroup,
    DenseBF16SwiGLU,
    DenseVQArchive,
    DenseVQEmbedding,
    DenseVQLinear,
    DenseVQLinearGroup,
    DenseVQPoolStats,
    DenseVQSwiGLU,
    DenseVQTensorSpec,
)


_GIB = 1 << 30


@dataclass(frozen=True)
class CompiledCodebookImage:
    """One model-independent execution image for shared VQ codebooks.

    Keys are the source device pointers stored in packed expert metadata.
    The packed expert indices are never copied or expanded by this object.
    """

    mode: Literal["e4m3", "q8"]
    tensors: dict[int, torch.Tensor]
    scales: dict[int, float]

    @property
    def replacements(self) -> dict[int, tuple[int, float]]:
        return {
            pointer: (int(self.tensors[pointer].data_ptr()), self.scales[pointer])
            for pointer in self.tensors
        }

    @property
    def storage_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors.values())


def quantize_e4m3_codebook(
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Compile one shared VQ codebook to E4M3 plus its exact tensor scale.

    The compact expert indices are unchanged.  All routed and dense VQ
    backends use this one conversion so resident, LRU and tensor-parallel
    layouts cannot drift into model-specific codebook numerics.
    """

    if codebook.numel() <= 0:
        raise ValueError("codebook must be non-empty")
    scale = max(float(codebook.abs().amax().item()) / 448.0, 1.0e-12)
    quantized = (
        codebook.float()
        .div(scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return quantized, scale


def quantize_q8_codebook(
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Compile one shared VQ codebook to symmetric Q8 for direct DP4A."""

    if codebook.numel() <= 0:
        raise ValueError("codebook must be non-empty")
    scale = max(float(codebook.abs().amax().item()) / 127.0, 1.0e-12)
    quantized = (
        codebook.float()
        .div(scale)
        .clamp(-127.0, 127.0)
        .round()
        .to(torch.int8)
        .contiguous()
    )
    return quantized, scale


def compile_shared_codebook_image(
    codebooks: tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    mode: Literal["e4m3", "q8"],
) -> CompiledCodebookImage:
    """Compile unique shared codebooks for every routed/dense VQ adapter."""

    quantizer = (
        quantize_e4m3_codebook if mode == "e4m3" else quantize_q8_codebook
    )
    tensors: dict[int, torch.Tensor] = {}
    scales: dict[int, float] = {}
    for codebook in codebooks:
        pointer = int(codebook.data_ptr())
        if pointer in tensors:
            continue
        quantized, scale = quantizer(codebook)
        tensors[pointer] = quantized
        scales[pointer] = float(scale)
    return CompiledCodebookImage(mode=mode, tensors=tensors, scales=scales)


def rewrite_packed_codebook_metadata(
    metadata: torch.Tensor,
    replacements: dict[int, tuple[int, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rewrite the public packed projection ABI to a compiled codebook image.

    Both supported metadata layouts are format contracts rather than model
    contracts: 15 rows describe separate gate/up/down projections, while 10
    rows describe fused gate-up plus down.  Resident and LRU pools call this
    same transformation for E4M3 and Q8 execution.
    """

    if metadata.ndim != 2 or int(metadata.shape[0]) not in (10, 15):
        raise ValueError("compiled codebook metadata requires [10,E] or [15,E]")
    rewritten = metadata.detach().to(device="cpu", dtype=torch.long).clone()
    expert_count = int(rewritten.shape[1])
    scales = torch.zeros(expert_count, 3, dtype=torch.float32)
    layout = (
        ((0, (0, 1)), (5, (2,)))
        if int(rewritten.shape[0]) == 10
        else ((0, (0,)), (5, (1,)), (10, (2,)))
    )
    for base, scale_columns in layout:
        for expert in range(expert_count):
            pointer = int(rewritten[base + 1, expert].item())
            if pointer == 0:
                continue
            replacement = replacements.get(pointer)
            if replacement is None:
                raise RuntimeError(
                    "compiled codebook image is missing pointer "
                    f"{pointer} at expert {expert}"
                )
            compiled_pointer, scale = replacement
            rewritten[base + 1, expert] = int(compiled_pointer)
            for column in scale_columns:
                scales[expert, column] = float(scale)
    return rewritten, scales


def run_compact_q8_codebook_decode(
    *,
    value: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    metadata: torch.Tensor,
    scales: torch.Tensor,
    activation: str,
    activation_beta: float,
    activation_linear_beta: float | None,
    limit: float,
    hidden_workspace: torch.Tensor,
    output_workspace: torch.Tensor,
    result: torch.Tensor,
    gate_quant_workspace: torch.Tensor | None = None,
    down_quant_workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute the sole CUDA compact routed-codebook Decode algorithm.

    Residency policy is deliberately absent from this interface.  Resident,
    LRU and tensor-parallel pools publish the same packed metadata and invoke
    this function; model adapters never choose an operator.
    """

    from ..fusedext import packed_moe_topk_compact_q8_codebook_fused

    common = {
        "value": value.to(torch.bfloat16),
        "route_ids": route_ids,
        "weights": route_weights,
        "metadata": metadata,
        "scales": scales,
        "activation": activation,
        "beta": float(activation_beta),
        "linear_beta": (
            0.0
            if activation_linear_beta is None
            else float(activation_linear_beta)
        ),
        "limit": float(limit),
        "hidden_workspace": hidden_workspace,
        "out_workspace": output_workspace,
        "result": result,
    }
    if gate_quant_workspace is None or down_quant_workspace is None:
        raise RuntimeError("compact Q8 Decode requires activation workspaces")
    launched = packed_moe_topk_compact_q8_codebook_fused(
        **common,
        gate_quant_workspace=gate_quant_workspace,
        down_quant_workspace=down_quant_workspace,
    )
    if launched is None:
        raise RuntimeError("compact Q8 codebook MoE rejected route")
    return launched


def native_fp8_gemm_available(device: torch.device) -> bool:
    """Return whether the public runtime can execute resident FP8 GEMM."""
    if (
        device.type != "cuda"
        or torch.version.hip is not None
        or not hasattr(torch, "_scaled_mm")
        or not hasattr(torch, "float8_e4m3fn")
    ):
        return False
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except (RuntimeError, TypeError, ValueError):
        return False
    return (int(major), int(minor)) >= (8, 9)


def plan_dense_vq_gpu_image(
    *,
    free_bytes: int,
    linear_bf16_bytes: int,
    packed_embedding_bytes: int,
    fixed_file_bytes: int,
    max_ctx: int,
    config,
    linear_fp8_bytes: int = 0,
    linear_compact_bytes: int = 0,
    embedding_bf16_bytes: int = 0,
    fp8_supported: bool = False,
    initial_ctx: int | None = None,
) -> tuple[str, dict[str, int]]:
    """Choose a Dense-VQ execution image from shapes and device capacity.

    The planner is deliberately model-name agnostic.  Architecture adapters
    supply only topology-derived sizes; image selection and memory policy stay
    in the common codebook runtime.
    """
    layer_types = list(getattr(config, "layer_types", ()) or ())
    full_attention_layers = sum(
        1 for item in layer_types if "full_attention" in str(item)
    )
    if not full_attention_layers:
        full_attention_layers = int(
            getattr(config, "num_hidden_layers", 0) or 0
        )
    kv_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
    head_dim = int(getattr(config, "head_dim", 0) or 0)
    kv_context = min(
        int(max_ctx),
        max(1, int(initial_ctx if initial_ctx is not None else max_ctx)),
    )
    kv_bytes = (
        kv_context
        * full_attention_layers
        * kv_heads
        * head_dim
        * 2
        * 2
    )
    runtime_bytes = max(4 * _GIB, kv_bytes + 3 * _GIB)
    embedding_bytes = int(embedding_bf16_bytes or packed_embedding_bytes)
    bf16_planned_bytes = (
        int(linear_bf16_bytes)
        + embedding_bytes
        + int(fixed_file_bytes)
        + int(runtime_bytes)
    )
    fp8_planned_bytes = (
        int(linear_fp8_bytes)
        + embedding_bytes
        + int(fixed_file_bytes)
        + int(runtime_bytes)
    )
    compact_planned_bytes = (
        int(linear_compact_bytes)
        + int(packed_embedding_bytes)
        + int(fixed_file_bytes)
        + int(runtime_bytes)
    )
    if (
        fp8_supported
        and int(linear_fp8_bytes) > 0
        and int(free_bytes) >= fp8_planned_bytes
    ):
        image = "fp8"
        planned_bytes = fp8_planned_bytes
    elif int(free_bytes) >= bf16_planned_bytes:
        image = "bf16"
        planned_bytes = bf16_planned_bytes
    else:
        image = "compact"
        planned_bytes = compact_planned_bytes
    details = {
        "free": int(free_bytes),
        "linear_bf16": int(linear_bf16_bytes),
        "packed_embedding": int(packed_embedding_bytes),
        "embedding_bf16": embedding_bytes,
        "linear_fp8": int(linear_fp8_bytes),
        "linear_compact": int(linear_compact_bytes),
        "fixed": int(fixed_file_bytes),
        "kv_context": int(kv_context),
        "kv": int(kv_bytes),
        "runtime": int(runtime_bytes),
        "planned": int(planned_bytes),
        "bf16_planned": int(bf16_planned_bytes),
        "fp8_planned": int(fp8_planned_bytes),
        "compact_planned": int(compact_planned_bytes),
    }
    return image, details


__all__ = [
    "CompiledCodebookImage",
    "DenseBF16Linear",
    "DenseBF16LinearGroup",
    "DenseBF16SwiGLU",
    "DenseVQArchive",
    "DenseVQEmbedding",
    "DenseVQLinear",
    "DenseVQLinearGroup",
    "DenseVQPoolStats",
    "DenseVQSwiGLU",
    "DenseVQTensorSpec",
    "compile_shared_codebook_image",
    "native_fp8_gemm_available",
    "plan_dense_vq_gpu_image",
    "quantize_e4m3_codebook",
    "quantize_q8_codebook",
    "rewrite_packed_codebook_metadata",
    "run_compact_q8_codebook_decode",
]

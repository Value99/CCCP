"""模型运行时使用的公共算子入口。"""

from __future__ import annotations

import torch

from .registry import REGISTRY
from .spec import OperatorRequest


_BUILTINS_READY = False
_ATTENTION_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_NORMALIZATION_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_RESIDUAL_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_RESIDUAL_ADD_IMPLEMENTATIONS: dict[str, object] = {}
_HYPER_CONNECTION_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_ACTIVATION_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_ROUTE_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_LINEAR_ROUTE_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_BLOCK_SCALED_GEMV_IMPLEMENTATIONS: dict[
    tuple[str, str, int], object
] = {}
_BLOCK_SCALED_GEMM_IMPLEMENTATIONS: dict[
    tuple[str, str, int], object
] = {}
_BLOCK_SCALED_GROUPED_GEMV_IMPLEMENTATIONS: dict[
    tuple[str, str, int], object
] = {}
_BLOCK_SCALED_GROUPED_ROWS_GEMV_IMPLEMENTATIONS: dict[
    tuple[str, str, int], object
] = {}
_BLOCK_SCALED_GROUPED_GEMM_IMPLEMENTATIONS: dict[
    tuple[str, str, int], object
] = {}
_DENSE_GROUPED_GEMV_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_DENSE_GEMV_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_RESIDENT_MOE_IMPLEMENTATIONS: dict[
    tuple[
        str,
        tuple[str, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
    object,
] = {}
_PACKED_ROUTE_SLOT_IMPLEMENTATIONS: dict[str, object] = {}
_PACKED_H2D_BATCH_IMPLEMENTATIONS: dict[str, object] = {}
_COMPRESSED_STATE_IMPLEMENTATIONS: dict[tuple[str, int], object] = {}
_HEAD_NORM_ROPE_IMPLEMENTATIONS: dict[tuple[str, int, int], object] = {}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.").lower()


def paged_indexer_logits(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    key_scales: torch.Tensor | None,
    head_weights: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    control: torch.Tensor,
    *,
    compression_ratio: int,
    page_layout: str,
    cache_format: str,
    query_fp8: torch.Tensor | None = None,
    query_scales: torch.Tensor | None = None,
    mm_workspace: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    architecture_features: tuple[str, ...] = (),
) -> torch.Tensor | None:
    """Compute exact full Indexer logits from a paged cache.

    Backends may quantize Q and use FP8 tensor cores internally, but the
    public result is always the complete FP32 logit vector.  Selection is a
    separate registered operation so non-SM90 backends can replace either
    stage independently.
    """
    _ensure_builtins()
    request = OperatorRequest(
        operation="paged_indexer_logits",
        device_type=query.device.type,
        activation="relu_weighted_sum",
        top_k=1,
        batch_size=int(query.shape[0]),
        dtype=_dtype_name(query.dtype),
        cache_format=cache_format,
        head_dim=int(query.shape[-1]),
        page_layout=page_layout,
        compression_ratio=int(compression_ratio),
        architecture_features=architecture_features,
    )
    try:
        return REGISTRY.call(
            request,
            query=query,
            key_cache=key_cache,
            key_scales=key_scales,
            head_weights=head_weights,
            cos=cos,
            sin=sin,
            control=control,
            compression_ratio=int(compression_ratio),
            query_fp8=query_fp8,
            query_scales=query_scales,
            mm_workspace=mm_workspace,
            output=output,
        )
    except LookupError:
        return None


def persistent_topk_exact(
    scores: torch.Tensor,
    top_k: int,
    *,
    values: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    page_layout: str = "flat-page-index",
    architecture_features: tuple[str, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Exact fixed-shape Top-K with caller-owned Graph-stable outputs."""
    _ensure_builtins()
    request = OperatorRequest(
        operation="persistent_topk_exact",
        device_type=scores.device.type,
        activation="none",
        top_k=int(top_k),
        batch_size=int(scores.shape[0]),
        dtype=_dtype_name(scores.dtype),
        head_dim=1,
        page_layout=page_layout,
        architecture_features=architecture_features,
    )
    try:
        return REGISTRY.call(
            request,
            scores=scores,
            top_k=int(top_k),
            values=values,
            indices=indices,
        )
    except LookupError:
        return None


def sparse_paged_attention_splitkv(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    sink: torch.Tensor | None,
    scale: float,
    cache_format: str,
    page_layout: str,
    compression_ratio: int,
    extra_key_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    runner=None,
    architecture_features: tuple[str, ...] = (),
) -> torch.Tensor | None:
    """Sparse paged decode without materializing selected K/V rows."""
    _ensure_builtins()
    request = OperatorRequest(
        operation="sparse_paged_attention_splitkv",
        device_type=query.device.type,
        activation="online_softmax",
        top_k=int(indices.shape[-1]),
        batch_size=int(query.shape[0]),
        dtype=_dtype_name(query.dtype),
        cache_format=cache_format,
        head_dim=int(query.shape[-1]),
        page_layout=page_layout,
        compression_ratio=int(compression_ratio),
        architecture_features=architecture_features,
    )
    try:
        return REGISTRY.call(
            request,
            query=query,
            key_cache=key_cache,
            indices=indices,
            sink=sink,
            scale=float(scale),
            extra_key_cache=extra_key_cache,
            extra_indices=extra_indices,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
            runner=runner,
        )
    except LookupError:
        return None


def fused_compressor_cache_store(
    **kwargs,
) -> bool:
    """Run a compressor step and atomically publish BF16/FP8 cache views."""
    projected = kwargs["projected"]
    _ensure_builtins()
    request = OperatorRequest(
        operation="fused_compressor_cache_store",
        device_type=projected.device.type,
        activation=("hadamard" if kwargs.get("hadamard") else "none"),
        top_k=1,
        batch_size=int(projected.shape[0]),
        dtype=_dtype_name(projected.dtype),
        cache_format=str(kwargs.get("cache_format", "bf16")),
        head_dim=int(kwargs["width"]),
        page_layout=str(kwargs.get("page_layout", "pointer-pages")),
        compression_ratio=int(kwargs["ratio"]),
        architecture_features=tuple(kwargs.pop("architecture_features", ())),
    )
    try:
        result = REGISTRY.call(request, **kwargs)
    except LookupError:
        return False
    return bool(result)


def linear(
    value: torch.Tensor,
    weight,
    *,
    output_dtype: torch.dtype | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """通用单 token Linear，直接读取 BF16 或紧凑 block-FP8 权重。

    ``ProjectionGroup`` 表示逻辑行拼接；每个成员保持自己的 128 行 scale
    原点，只拼接 token 大小的输出。这里不按模型名分派，也不生成完整反量化
    矩阵。调用方需要保持旧 ``F.linear`` dtype 时显式传 ``output_dtype``。
    """
    from ..kernels import BlockFP8Weight, ProjectionGroup

    if isinstance(weight, BlockFP8Weight):
        result = weight.matmul_T_decode_fused(value, output=output)
    elif isinstance(weight, ProjectionGroup):
        result = weight.matmul_T_decode_fused(
            value,
            output=output,
            output_dtype=output_dtype,
        )
    else:
        result = None
        if (
            isinstance(weight, torch.Tensor)
            and value.is_cuda
            and weight.is_cuda
            and value.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and value.ndim == 2
            and value.shape[0] == 1
            and weight.ndim == 2
            and weight.shape[1] == value.shape[1]
            and value.is_contiguous()
            and weight.is_contiguous()
            and value.device == weight.device
        ):
            target_dtype = output_dtype or torch.bfloat16
            target = output
            if target is None:
                target = torch.empty(
                    (1, int(weight.shape[0])),
                    dtype=target_dtype,
                    device=value.device,
                )
            result = dense_gemv(value, weight, output=target)
        if result is None:
            result = torch.nn.functional.linear(
                value.to(weight.dtype), weight
            )
            if output is not None:
                output.copy_(result)
                result = output
    return (
        result
        if output_dtype is None or result.dtype == output_dtype
        else result.to(output_dtype)
    )


def linear_batch(
    value: torch.Tensor,
    weight,
    *,
    output_dtype: torch.dtype | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Model-independent batched projection entry point.

    Compact weights already expose a shape-aware ``matmul_T`` implementation;
    dense tensors use the native GEMM.  This wrapper intentionally does not
    impose a model-specific batch limit and is the common hook for prefill.
    """
    from ..kernels import (
        BlockFP8Weight,
        Int4Weight,
        ProjectionGroup,
        VQWeight,
    )

    if value.ndim < 2:
        raise ValueError("batched projection expects at least a 2D input")
    shape = value.shape
    rows = value.reshape(-1, shape[-1])
    # Compact weights expose the same 2-D ``x @ W.T`` contract.  Flattening
    # here keeps the dispatcher independent of a model's [B,T,...] layout;
    # the result is restored before the optional caller-owned output copy.
    if isinstance(weight, torch.Tensor):
        result = torch.nn.functional.linear(rows.to(weight.dtype), weight)
    elif isinstance(weight, (BlockFP8Weight, ProjectionGroup)):
        result = weight.matmul_T_decode_fused(rows)
    elif isinstance(weight, (Int4Weight, VQWeight)):
        result = weight.matmul_T(rows)
    else:
        raise TypeError(
            f"unsupported batched projection weight {type(weight)!r}"
        )
    result = result.reshape(*shape[:-1], result.shape[-1])
    if output is not None:
        if output.shape != result.shape:
            raise ValueError(
                "batched projection output shape does not match result"
            )
        output.copy_(result)
        result = output
    return (
        result
        if output_dtype is None or result.dtype == output_dtype
        else result.to(output_dtype)
    )


def dense_gemv(
    value: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: torch.Tensor,
) -> torch.Tensor | None:
    """Registered one-token BF16 GEMV into fixed caller-owned storage."""
    _ensure_builtins()
    key = (value.device.type, "bf16")
    try:
        implementation = _DENSE_GEMV_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="dense_gemv",
                device_type=value.device.type,
                packed_formats=("bf16",),
                activation="none",
                batch_size=int(value.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _DENSE_GEMV_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            weight=weight,
            output=output,
        )
    except LookupError:
        return None


def linear_grouped_rows(
    value: torch.Tensor,
    weight,
    *,
    output_dtype: torch.dtype | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Apply compact projection ``i`` to input row ``i`` in one call."""
    from ..kernels import ProjectionGroup

    if not isinstance(weight, ProjectionGroup):
        return None
    result = weight.matmul_T_grouped_rows_fused(value, output=output)
    if result is None:
        return None
    return result if output_dtype is None else result.to(output_dtype)


def _projection_layout_tag(
    packed_formats: tuple[str, ...],
    code_dims: tuple[int, ...],
    codebook_sizes: tuple[int, ...],
) -> int:
    """Select a CUDA fast path from the exact public capability tuple."""
    if not packed_formats or not code_dims or not codebook_sizes:
        return 0
    # Tag 2 enables the paired p10 shared-codebook specialization.  All other
    # three-projection layouts use tag 1 and therefore avoid reserving its
    # 40 KiB scratch area when either Gate or Up is not p10.  A heterogeneous
    # layer reports the union of every layout used by its experts, rather than
    # exactly three positional entries.  Metadata is still [15,E] and carries
    # each selected expert's Gate/Up/Down format, so the common dynamic kernel
    # remains valid and must not be disabled merely because this capability
    # tuple contains more than three values.
    exact_projection_tuple = (
        len(packed_formats) == 3
        and len(code_dims) == 3
        and len(codebook_sizes) == 3
    )
    return (
        2
        if exact_projection_tuple
        and packed_formats[:2] == ("p10", "p10")
        else 1
    )


def _ensure_builtins() -> None:
    global _BUILTINS_READY
    if _BUILTINS_READY:
        return
    from . import cpu_backend, cuda_backend, triton_backend

    cpu_backend.register(REGISTRY)
    cuda_backend.register(REGISTRY)
    triton_backend.register(REGISTRY)
    _BUILTINS_READY = True


def vq_gemv(
    x_rows: torch.Tensor,
    indices: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor | None:
    """按能力分派 VQ GEMV；不在此接口中展开或反量化权重。"""
    _ensure_builtins()
    if indices.dtype == torch.uint8:
        packed_format = "u8"
    elif indices.dtype == torch.uint16:
        packed_format = "u16"
    else:
        return None
    request = OperatorRequest(
        operation="vq_gemv",
        device_type=x_rows.device.type,
        packed_formats=(packed_format,),
        code_dims=(int(codebook.shape[-1]),),
        codebook_sizes=(int(codebook.shape[-2]),),
        activation="none",
        top_k=1,
        batch_size=max(
            int(x_rows.shape[0]),
            int(indices.shape[0]),
        ),
    )
    try:
        return REGISTRY.call(
            request,
            x_rows=x_rows,
            indices=indices,
            codebook=codebook,
        )
    except LookupError:
        return None


def vq_gemv_packed_list(
    x_rows: torch.Tensor,
    payloads: list[torch.Tensor],
    codebook: torch.Tensor,
    rows: int,
    blocks: int,
    bits: int,
    *,
    allow_direct: bool = False,
) -> torch.Tensor | None:
    """Dispatch compact list-backed VQ without expanding packed indices."""
    _ensure_builtins()
    if not 8 <= bits <= 16 or not payloads:
        return None
    request = OperatorRequest(
        operation="vq_gemv:list",
        device_type=x_rows.device.type,
        packed_formats=(f"p{bits}",),
        code_dims=(int(codebook.shape[-1]),),
        codebook_sizes=(int(codebook.shape[-2]),),
        activation="none",
        top_k=len(payloads),
        batch_size=len(payloads),
    )
    try:
        return REGISTRY.call(
            request,
            x_rows=x_rows,
            payloads=payloads,
            codebook=codebook,
            rows=int(rows),
            blocks=int(blocks),
            bits=int(bits),
            allow_direct=bool(allow_direct),
        )
    except LookupError:
        return None


def block_scaled_gemv(
    value: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int,
    rows: int | None = None,
    cols: int | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """直接读取块缩放紧凑权重执行单 token GEMV。"""
    _ensure_builtins()
    if weights.dtype != torch.uint8 or scales.dtype != torch.float32:
        return None
    block_major = weights.ndim == 5
    format_name = (
        "e4m3fn-block-major32" if block_major else "e4m3fn"
    )
    logical_rows = int(weights.shape[0]) if rows is None else int(rows)
    logical_cols = int(weights.shape[1]) if cols is None else int(cols)
    key = (value.device.type, format_name, int(block_size))
    try:
        implementation = _BLOCK_SCALED_GEMV_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="block_scaled_gemv",
                device_type=value.device.type,
                packed_formats=(format_name,),
                code_dims=(int(block_size),),
                activation="none",
                top_k=1,
                batch_size=int(value.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _BLOCK_SCALED_GEMV_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            weights=weights,
            scales=scales,
            rows=logical_rows,
            cols=logical_cols,
            block_size=int(block_size),
            block_major=block_major,
            output=output,
        )
    except LookupError:
        return None


def block_scaled_gemm(
    value: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int,
    rows: int | None = None,
    cols: int | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Verify 2..16 tokens while scanning compact block-FP8 once."""
    _ensure_builtins()
    if (
        weights.dtype != torch.uint8
        or scales.dtype != torch.float32
        or value.ndim != 2
        or not 2 <= int(value.shape[0]) <= 16
    ):
        return None
    block_major = weights.ndim == 5
    format_name = (
        "e4m3fn-block-major32" if block_major else "e4m3fn"
    )
    logical_rows = int(weights.shape[0]) if rows is None else int(rows)
    logical_cols = int(weights.shape[1]) if cols is None else int(cols)
    key = (value.device.type, format_name, int(block_size))
    try:
        implementation = _BLOCK_SCALED_GEMM_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="block_scaled_gemm",
                device_type=value.device.type,
                packed_formats=(format_name,),
                code_dims=(int(block_size),),
                activation="none",
                top_k=1,
                batch_size=int(value.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _BLOCK_SCALED_GEMM_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            weights=weights,
            scales=scales,
            rows=logical_rows,
            cols=logical_cols,
            block_size=int(block_size),
            block_major=block_major,
            output=output,
        )
    except LookupError:
        return None


def block_scaled_grouped_gemv(
    value: torch.Tensor,
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    row_offsets: torch.Tensor,
    *,
    total_rows: int,
    cols: int,
    block_size: int = 128,
    block_major: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """One-token logical row concatenation over compact block-FP8 weights.

    Pointer metadata is a fixed-address device plan.  The underlying compact
    payload remains owned by its original weights, so this public operation
    neither concatenates nor dequantizes a complete matrix.
    """
    _ensure_builtins()
    format_name = (
        "e4m3fn-block-major32" if block_major else "e4m3fn"
    )
    key = (value.device.type, format_name, int(block_size))
    try:
        implementation = _BLOCK_SCALED_GROUPED_GEMV_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="block_scaled_grouped_gemv",
                device_type=value.device.type,
                packed_formats=(format_name,),
                code_dims=(int(block_size),),
                activation="none",
                batch_size=int(value.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _BLOCK_SCALED_GROUPED_GEMV_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            weight_ptrs=weight_ptrs,
            scale_ptrs=scale_ptrs,
            row_offsets=row_offsets,
            total_rows=int(total_rows),
            cols=int(cols),
            block_size=int(block_size),
            block_major=bool(block_major),
            output=output,
        )
    except LookupError:
        return None


def block_scaled_grouped_gemm(
    value: torch.Tensor,
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    row_offsets: torch.Tensor,
    *,
    total_rows: int,
    cols: int,
    block_size: int = 128,
    block_major: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Multi-token logical row concatenation over compact FP8 weights."""
    _ensure_builtins()
    format_name = (
        "e4m3fn-block-major32" if block_major else "e4m3fn"
    )
    key = (value.device.type, format_name, int(block_size))
    try:
        implementation = _BLOCK_SCALED_GROUPED_GEMM_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="block_scaled_grouped_gemm",
                device_type=value.device.type,
                packed_formats=(format_name,),
                code_dims=(int(block_size),),
                activation="none",
                batch_size=int(value.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _BLOCK_SCALED_GROUPED_GEMM_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            weight_ptrs=weight_ptrs,
            scale_ptrs=scale_ptrs,
            row_offsets=row_offsets,
            total_rows=int(total_rows),
            cols=int(cols),
            block_size=int(block_size),
            block_major=bool(block_major),
            output=output,
        )
    except LookupError:
        return None


def block_scaled_grouped_rows_gemv(
    value: torch.Tensor,
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    row_offsets: torch.Tensor,
    *,
    total_rows: int,
    cols: int,
    block_size: int = 128,
    block_major: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """One compact FP8 projection for each matching input row."""
    _ensure_builtins()
    format_name = (
        "e4m3fn-block-major32" if block_major else "e4m3fn"
    )
    key = (value.device.type, format_name, int(block_size))
    try:
        implementation = (
            _BLOCK_SCALED_GROUPED_ROWS_GEMV_IMPLEMENTATIONS.get(key)
        )
        if implementation is None:
            request = OperatorRequest(
                operation="block_scaled_grouped_rows_gemv",
                device_type=value.device.type,
                packed_formats=(format_name,),
                code_dims=(int(block_size),),
                activation="none",
                batch_size=int(value.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _BLOCK_SCALED_GROUPED_ROWS_GEMV_IMPLEMENTATIONS[key] = (
                implementation
            )
        return implementation(
            value=value,
            weight_ptrs=weight_ptrs,
            scale_ptrs=scale_ptrs,
            row_offsets=row_offsets,
            total_rows=int(total_rows),
            cols=int(cols),
            block_size=int(block_size),
            block_major=bool(block_major),
            output=output,
        )
    except LookupError:
        return None


def vq_relayout_block_major(
    payload: torch.Tensor,
    *,
    rows: int,
    blocks: int,
    bits: int,
    code_dim: int,
    codebook_size: int,
) -> torch.Tensor | None:
    """Convert compact VQ index traversal order without expanding indices."""
    _ensure_builtins()
    if (
        payload.device.type != "cpu"
        or payload.dtype != torch.uint8
        or not 8 <= int(bits) <= 16
    ):
        return None
    request = OperatorRequest(
        operation="vq_relayout:block_major",
        device_type="cpu",
        packed_formats=(f"p{int(bits)}",),
        code_dims=(int(code_dim),),
        codebook_sizes=(int(codebook_size),),
        activation="none",
        top_k=1,
        batch_size=1,
    )
    try:
        return REGISTRY.call(
            request,
            payload=payload,
            rows=int(rows),
            blocks=int(blocks),
            bits=int(bits),
        )
    except LookupError:
        return None


def vq_relayout_row_tile(
    payload: torch.Tensor,
    *,
    rows: int,
    blocks: int,
    bits: int,
    code_dim: int,
    codebook_size: int,
    tile_rows: int = 8,
) -> torch.Tensor | None:
    """Convert compact indices to a CPU row-tile traversal."""
    _ensure_builtins()
    if (
        payload.device.type != "cpu"
        or payload.dtype != torch.uint8
        or not 8 <= int(bits) <= 16
        or int(tile_rows) <= 0
        or int(tile_rows) % 8
    ):
        return None
    request = OperatorRequest(
        operation="vq_relayout:row_tile",
        device_type="cpu",
        packed_formats=(f"p{int(bits)}",),
        code_dims=(int(code_dim),),
        codebook_sizes=(int(codebook_size),),
        activation="none",
        top_k=1,
        batch_size=1,
    )
    try:
        return REGISTRY.call(
            request,
            payload=payload,
            rows=int(rows),
            blocks=int(blocks),
            bits=int(bits),
            tile_rows=int(tile_rows),
        )
    except LookupError:
        return None


def vq_compile_u16_row_tile(
    payload: torch.Tensor,
    *,
    rows: int,
    blocks: int,
    bits: int,
    code_dim: int,
    codebook_size: int,
    tile_rows: int = 8,
) -> torch.Tensor | None:
    """Compile compact VQ indices into an exact in-RAM CPU execution image.

    Dispatch remains format driven.  The returned uint16 tensor is transient
    runtime state and is never written into, or substituted for, the model
    archive on disk.
    """
    _ensure_builtins()
    request = OperatorRequest(
        operation="vq_compile:u16_row_tile",
        device_type=payload.device.type,
        packed_formats=(f"p{int(bits)}",),
        code_dims=(int(code_dim),),
        codebook_sizes=(int(codebook_size),),
        activation="none",
        top_k=1,
        batch_size=1,
    )
    try:
        return REGISTRY.call(
            request,
            payload=payload,
            rows=int(rows),
            blocks=int(blocks),
            bits=int(bits),
            tile_rows=int(tile_rows),
        )
    except LookupError:
        return None


def vq_compile_q4_0(
    payload: torch.Tensor,
    codebook: torch.Tensor,
    *,
    rows: int,
    blocks: int,
    bits: int,
    code_dim: int,
    codebook_size: int,
) -> torch.Tensor | None:
    """Compile compact VQ into the public linear Q4 CPU execution format."""
    _ensure_builtins()
    request = OperatorRequest(
        operation="vq_compile:q4_0",
        device_type=payload.device.type,
        packed_formats=(f"p{int(bits)}",),
        code_dims=(int(code_dim),),
        codebook_sizes=(int(codebook_size),),
        activation="none",
        top_k=1,
        batch_size=1,
    )
    try:
        return REGISTRY.call(
            request,
            payload=payload,
            codebook=codebook,
            rows=int(rows),
            blocks=int(blocks),
            bits=int(bits),
        )
    except LookupError:
        return None


def dense_grouped_gemv(
    value: torch.Tensor,
    weight_ptrs: torch.Tensor,
    row_offsets: torch.Tensor,
    *,
    total_rows: int,
    cols: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """One-token logical row concatenation over BF16 CPU matrices."""
    _ensure_builtins()
    key = (value.device.type, "bf16")
    try:
        implementation = _DENSE_GROUPED_GEMV_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="dense_grouped_gemv",
                device_type=value.device.type,
                packed_formats=("bf16",),
                activation="none",
                batch_size=int(value.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _DENSE_GROUPED_GEMV_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            weight_ptrs=weight_ptrs,
            row_offsets=row_offsets,
            total_rows=int(total_rows),
            cols=int(cols),
            output=output,
        )
    except LookupError:
        return None


def create_resident_projection_layer(weights: tuple[object, ...]):
    """Create a fixed-address single-token projection executor.

    Selection is based on source formats, not model family.  Mixed BF16 and
    block-FP8 projections sharing one input can therefore enter one persistent
    CPU team without copying or dequantizing their model weights.
    """
    from ..kernels import BlockFP8Weight

    _ensure_builtins()
    if not weights:
        return None
    formats: set[str] = set()
    for weight in weights:
        if (
            isinstance(weight, torch.Tensor)
            and not weight.is_cuda
            and weight.dtype == torch.bfloat16
        ):
            formats.add("bf16")
        elif isinstance(weight, BlockFP8Weight) and not weight.q.is_cuda:
            formats.add(
                "q4_0-linear-block32"
                if weight.layout == "q4_0"
                else "e4m3fn-block-major32"
                if weight.layout == "block-major32"
                else "e4m3fn"
            )
        else:
            return None
    request = OperatorRequest(
        operation="resident_projection_layer",
        device_type="cpu",
        packed_formats=tuple(formats),
        code_dims=tuple(
            dim
            for dim, present in (
                (32, "q4_0-linear-block32" in formats),
                (128, any("e4m3fn" in item for item in formats)),
            )
            if present
        ),
        activation="none",
        top_k=1,
        batch_size=1,
    )
    try:
        return REGISTRY.call(request, weights=weights)
    except LookupError:
        return None


def compressed_state_update(
    projected: torch.Tensor,
    ape: torch.Tensor,
    ckv: torch.Tensor,
    cscore: torch.Tensor,
    *,
    ratio: int,
    position: int,
    kv_rows: int,
) -> bool:
    """Write a one-token KV/score projection into compact ring state.

    The operation is keyed only by device, ratio and decode batch shape.  It
    is reusable by any compressed-attention configuration and does not own
    projection weights or model-specific pooling rules.
    """
    _ensure_builtins()
    key = (projected.device.type, int(ratio))
    try:
        implementation = _COMPRESSED_STATE_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="compressed_state_update",
                device_type=projected.device.type,
                code_dims=(int(ratio),),
                activation="none",
                batch_size=int(projected.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _COMPRESSED_STATE_IMPLEMENTATIONS[key] = implementation
        return bool(
            implementation(
                projected=projected,
                ape=ape,
                ckv=ckv,
                cscore=cscore,
                ratio=int(ratio),
                position=int(position),
                kv_rows=int(kv_rows),
            )
        )
    except LookupError:
        return False


def head_rmsnorm_rope(
    rows: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rope_width: int,
    eps: float,
) -> bool:
    """In-place per-head RMSNorm plus interleaved tail RoPE."""
    _ensure_builtins()
    width = int(rows.shape[-1])
    key = (rows.device.type, width, int(rope_width))
    try:
        implementation = _HEAD_NORM_ROPE_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="head_rmsnorm_rope",
                device_type=rows.device.type,
                code_dims=(width, int(rope_width)),
                activation="none",
                batch_size=1,
            )
            implementation = REGISTRY.resolve(request).implementation
            _HEAD_NORM_ROPE_IMPLEMENTATIONS[key] = implementation
        return bool(
            implementation(
                rows=rows,
                weight=weight,
                cos=cos,
                sin=sin,
                rope_width=int(rope_width),
                eps=float(eps),
            )
        )
    except LookupError:
        return False


def route_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    *,
    scoring_func: str,
    top_k: int,
    normalize: bool,
    scaling: float,
    n_group: int = 1,
    topk_group: int = 1,
    output_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """配置驱动的 Top-K 路由；注册键只描述数学与设备能力。"""
    _ensure_builtins()
    if int(n_group) != 1 or int(topk_group) != 1:
        return None
    normalized_scoring = scoring_func.strip().lower()
    key = (logits.device.type, normalized_scoring)
    try:
        implementation = _ROUTE_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="route_topk",
                device_type=logits.device.type,
                activation=normalized_scoring,
                top_k=int(top_k),
                batch_size=int(logits.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _ROUTE_IMPLEMENTATIONS[key] = implementation
        return implementation(
            logits=logits,
            bias=bias,
            mask=mask,
            top_k=int(top_k),
            normalize=bool(normalize),
            scaling=float(scaling),
            output_buffers=output_buffers,
        )
    except LookupError:
        return None


def linear_route_topk(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    *,
    scoring_func: str,
    top_k: int,
    normalize: bool,
    scaling: float,
    n_group: int = 1,
    topk_group: int = 1,
    output_buffers: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """通用线性投影与 Top-K 路由接口，保持路由权重的源生精度。"""
    _ensure_builtins()
    if (
        int(n_group) != 1
        or int(topk_group) != 1
        or not normalize
    ):
        return None
    normalized_scoring = scoring_func.strip().lower()
    key = (value.device.type, normalized_scoring)
    try:
        implementation = _LINEAR_ROUTE_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="linear_route_topk",
                device_type=value.device.type,
                activation=normalized_scoring,
                top_k=int(top_k),
                batch_size=int(value.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _LINEAR_ROUTE_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            weight=weight,
            bias=bias,
            mask=mask,
            top_k=int(top_k),
            normalize=bool(normalize),
            scaling=float(scaling),
            output_buffers=output_buffers,
        )
    except LookupError:
        return None


def attention_step(kind: str, device_type: str, **kwargs):
    """配置驱动的 Attention 注册入口。

    各注意力数学实现按 ``kind`` 注册；公共运行时不按模型名称分派。
    """
    _ensure_builtins()
    normalized_kind = kind.strip().lower()
    normalized_device = device_type.strip().lower()
    key = (normalized_kind, normalized_device)
    implementation = _ATTENTION_IMPLEMENTATIONS.get(key)
    if implementation is None:
        query = kwargs.get("query")
        if not isinstance(query, torch.Tensor):
            query = kwargs.get("query_nope")
        request = OperatorRequest(
            operation=f"attention_step:{normalized_kind}",
            device_type=normalized_device,
            activation="none",
            batch_size=(
                int(query.shape[0])
                if isinstance(query, torch.Tensor)
                and query.ndim >= 3
                else 1
            ),
        )
        implementation = REGISTRY.resolve(request).implementation
        _ATTENTION_IMPLEMENTATIONS[key] = implementation
    return implementation(**kwargs)


def ordered_recurrent_scan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    backend: str = "auto",
    return_backend: bool = False,
):
    """Run an ordered KDA recurrence through the best public backend.

    ``auto`` prefers the parallel Triton chunk scan, then the single-launch
    ordered CUDA scan, and finally the registered token reference.  The model
    adapter supplies mathematical tensors only; backend selection and fallback
    never depend on a model name.
    """
    if (
        query.ndim != 3
        or key.shape != query.shape
        or gate.shape != query.shape
        or value.ndim != 3
        or value.shape[:2] != query.shape[:2]
        or beta.shape != query.shape[:2]
    ):
        raise ValueError("ordered recurrent scan tensor shapes do not match")
    if output is None:
        output = torch.empty_like(value)
    elif output.shape != value.shape:
        raise ValueError("ordered recurrent scan output shape does not match")
    selected = backend.strip().lower()
    if selected not in {"auto", "triton", "cuda", "reference"}:
        raise ValueError(f"unsupported recurrent scan backend {backend!r}")

    common = dict(
        query=query,
        key=key,
        value=value,
        gate=gate,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        state=state,
        output=output,
        lower_bound=float(lower_bound),
    )
    if selected in {"auto", "triton"}:
        try:
            result = attention_step(
                "kda_chunk_prefill", query.device.type, **common
            )
        except (ImportError, LookupError):
            result = None
        if result is not None:
            return (result, "triton") if return_backend else result
        if selected == "triton":
            raise RuntimeError("Triton KDA chunk prefill is unavailable")

    if selected in {"auto", "cuda"}:
        try:
            result = attention_step(
                "kda_recurrent_batch", query.device.type, **common
            )
        except LookupError:
            result = None
        if result is not None:
            return (result, "cuda") if return_backend else result
        if selected == "cuda":
            raise RuntimeError("ordered CUDA KDA prefill is unavailable")

    if workspace is None:
        workspace = torch.empty(
            3 * int(query.shape[1]) * int(query.shape[2]),
            dtype=torch.float32,
            device=query.device,
        )
    for token in range(int(query.shape[0])):
        result = attention_step(
            "kda_recurrent",
            query.device.type,
            query=query[token],
            key=key[token],
            value=value[token],
            gate=gate[token],
            beta=beta[token],
            a_log=a_log,
            dt_bias=dt_bias,
            state=state,
            workspace=workspace,
            output=output[token],
            lower_bound=float(lower_bound),
        )
        if result is None:
            raise RuntimeError("registered KDA token reference is unavailable")
    return (output, "reference") if return_backend else output


def rmsnorm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """设备无关的 RMSNorm 注册入口。"""
    _ensure_builtins()
    try:
        batch_size = max(1, value.numel() // value.shape[-1])
        key = (value.device.type, str(batch_size))
        implementation = _NORMALIZATION_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="normalization:rmsnorm",
                device_type=value.device.type,
                activation="none",
                batch_size=batch_size,
            )
            implementation = REGISTRY.resolve(request).implementation
            _NORMALIZATION_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            weight=weight,
            eps=float(eps),
            output=output,
        )
    except LookupError:
        return None


def residual_mix(
    kind: str,
    prefix: torch.Tensor,
    residual: torch.Tensor,
    projection: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    output: torch.Tensor | None = None,
    post_norm_weight: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    residual_inverse: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """配置驱动的残差合并入口。"""
    _ensure_builtins()
    normalized_kind = kind.strip().lower()
    key = (normalized_kind, prefix.device.type)
    try:
        implementation = _RESIDUAL_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation=f"residual_mix:{normalized_kind}",
                device_type=prefix.device.type,
                activation="none",
                batch_size=int(prefix.shape[0]) * (int(residual.shape[-2]) + 1),
            )
            implementation = REGISTRY.resolve(request).implementation
            _RESIDUAL_IMPLEMENTATIONS[key] = implementation
        return implementation(
            prefix=prefix,
            residual=residual,
            projection=projection,
            norm_weight=norm_weight,
            eps=float(eps),
            output=output,
            post_norm_weight=post_norm_weight,
            score_workspace=workspace,
            residual_inverse=residual_inverse,
        )
    except LookupError:
        return None


def residual_add3(
    residual: torch.Tensor,
    routed: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor | None:
    """按源 dtype 顺序计算 ``residual + (routed + shared)``。"""
    _ensure_builtins()
    device_type = residual.device.type
    try:
        implementation = _RESIDUAL_ADD_IMPLEMENTATIONS.get(device_type)
        if implementation is None:
            request = OperatorRequest(
                operation="residual_add:three_way",
                device_type=device_type,
                activation="none",
                batch_size=max(
                    1,
                    residual.numel() // residual.shape[-1],
                ),
            )
            implementation = REGISTRY.resolve(request).implementation
            _RESIDUAL_ADD_IMPLEMENTATIONS[device_type] = implementation
        return implementation(
            residual=residual,
            routed=routed,
            shared=shared,
        )
    except LookupError:
        return None


def _hyper_connection_implementation(
    operation: str,
    value: torch.Tensor,
    *,
    activation: str,
    batch_size: int,
):
    """Resolve one model-agnostic Hyper-Connection decode capability."""
    _ensure_builtins()
    key = (operation, value.device.type)
    implementation = _HYPER_CONNECTION_IMPLEMENTATIONS.get(key)
    if implementation is None:
        request = OperatorRequest(
            operation=f"hyper_connection:{operation}",
            device_type=value.device.type,
            activation=activation,
            batch_size=max(1, int(batch_size)),
        )
        implementation = REGISTRY.resolve(request).implementation
        _HYPER_CONNECTION_IMPLEMENTATIONS[key] = implementation
    return implementation


def hyper_connection_pre_norm(
    value: torch.Tensor,
    projection: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    sinkhorn_iters: int,
    eps: float,
    *,
    output_buffers: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None = None,
):
    """Fuse H/C input projection, Sinkhorn reduction and RMSNorm.

    The key describes mathematical capability rather than a model family.
    Caller-owned buffers keep decode addresses stable for every compatible
    configuration without forcing a model-specific implementation.
    """
    try:
        implementation = _hyper_connection_implementation(
            "pre_norm",
            value,
            activation="rmsnorm",
            batch_size=value.numel() // (4 * value.shape[-1]),
        )
        return implementation(
            value=value,
            projection=projection,
            scale=scale,
            base=base,
            norm_weight=norm_weight,
            sinkhorn_iters=int(sinkhorn_iters),
            eps=float(eps),
            output_buffers=output_buffers,
        )
    except LookupError:
        return None


def hyper_connection_post(
    value: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    combine: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
):
    """Apply one H/C post mix, optionally into a stable output buffer."""
    try:
        implementation = _hyper_connection_implementation(
            "post",
            residual,
            activation="none",
            batch_size=residual.numel() // (4 * residual.shape[-1]),
        )
        return implementation(
            value=value,
            residual=residual,
            post=post,
            combine=combine,
            output=output,
        )
    except LookupError:
        return None


def hyper_connection_post_moe(
    routed: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    combine: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
):
    """Fuse routed/shared BF16 merge and H/C post without temporaries."""
    try:
        implementation = _hyper_connection_implementation(
            "post_moe",
            residual,
            activation="none",
            batch_size=residual.numel() // (4 * residual.shape[-1]),
        )
        return implementation(
            routed=routed,
            shared=shared,
            residual=residual,
            post=post,
            combine=combine,
            output=output,
        )
    except LookupError:
        return None


def gated_activation(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    activation: str,
    beta: float,
    linear_beta: float | None,
    limit: float = 0.0,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """按激活能力选择融合 Gate×Up 算子。"""
    _ensure_builtins()
    normalized = activation.strip().lower()
    key = (gate.device.type, normalized)
    try:
        implementation = _ACTIVATION_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="gated_activation",
                device_type=gate.device.type,
                activation=normalized,
                batch_size=max(1, gate.numel() // gate.shape[-1]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _ACTIVATION_IMPLEMENTATIONS[key] = implementation
        return implementation(
            gate=gate,
            up=up,
            activation=normalized,
            beta=float(beta),
            linear_beta=linear_beta,
            limit=float(limit),
            output=output,
        )
    except LookupError:
        return None


def packed_moe_topk(
    value: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    metadata: torch.Tensor,
    *,
    activation: str,
    activation_beta: float,
    activation_linear_beta: float,
    hidden_workspace: torch.Tensor,
    output_workspace: torch.Tensor,
    result: torch.Tensor,
    grouped_prefix: int,
    packed_formats: tuple[str, ...] | None = None,
    code_dims: tuple[int, ...] | None = None,
    codebook_sizes: tuple[int, ...] | None = None,
    limit: float = 0.0,
) -> torch.Tensor:
    """执行严格单-token packed Top-K decode GEMV。"""
    if value.ndim != 2 or int(value.shape[0]) != 1:
        raise RuntimeError(
            "packed_moe_topk is decode-only and accepts exactly one token; "
            "multi-token Prefill must use grouped GEMM"
        )
    _ensure_builtins()
    projection_vq = (
        metadata.ndim == 2 and metadata.shape[0] in (15, 27)
    )
    if packed_formats is None:
        packed_formats = (
            tuple(f"p{bits}" for bits in range(8, 17))
            if projection_vq
            else ("p8", "p12", "p14")
        )
    if code_dims is None:
        code_dims = (4, 8, 16) if projection_vq else (4, 8)
    if codebook_sizes is None:
        codebook_sizes = (
            (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
            if projection_vq
            else (256, 4096, 16384)
        )
    projection_layout_tag = (
        _projection_layout_tag(
            packed_formats,
            code_dims,
            codebook_sizes,
        )
        if projection_vq
        else 0
    )
    route_width = (
        int(route_ids.shape[-1])
        if route_ids.ndim == 2
        else int(route_ids.numel())
    )
    request = OperatorRequest(
        operation="moe_topk",
        device_type=value.device.type,
        packed_formats=packed_formats,
        code_dims=code_dims,
        codebook_sizes=codebook_sizes,
        activation=activation,
        top_k=route_width,
        batch_size=value.shape[0],
    )
    output = REGISTRY.call(
        request,
        value=value,
        route_ids=route_ids,
        weights=route_weights,
        metadata=metadata,
        activation=activation,
        beta=activation_beta,
        linear_beta=activation_linear_beta,
        limit=float(limit),
        hidden_workspace=hidden_workspace,
        out_workspace=output_workspace,
        result=result,
        p12_count=grouped_prefix,
        projection_layout_tag=projection_layout_tag,
    )
    if output is None:
        raise RuntimeError(
            f"算子 {REGISTRY.resolve(request).name} 拒绝了兼容输入"
        )
    return output


def packed_moe_topk_grouped(
    value: torch.Tensor,
    token_ids: torch.Tensor,
    group_experts: torch.Tensor,
    group_offsets: torch.Tensor,
    route_weights: torch.Tensor,
    metadata: torch.Tensor,
    *,
    activation: str,
    activation_beta: float,
    activation_linear_beta: float,
    hidden_workspace: torch.Tensor,
    result: torch.Tensor,
    limit: float = 0.0,
    max_group_tiles: int | None = None,
) -> torch.Tensor:
    """Public expert-grouped two/three-projection packed prefill operator."""
    _ensure_builtins()
    if metadata.ndim != 2 or metadata.shape[0] not in (10, 15):
        raise ValueError(
            "grouped packed prefill requires [10,E] or [15,E] metadata"
        )
    request = OperatorRequest(
        operation="moe_topk_grouped",
        device_type=value.device.type,
        packed_formats=tuple(f"p{bits}" for bits in range(8, 17)),
        code_dims=(4, 8, 16),
        codebook_sizes=(
            256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
        ),
        activation=str(activation).strip().lower(),
        top_k=1,
        batch_size=int(value.shape[0]),
    )
    output = REGISTRY.call(
        request,
        value=value,
        token_ids=token_ids,
        group_experts=group_experts,
        group_offsets=group_offsets,
        weights=route_weights,
        metadata=metadata,
        activation=activation,
        beta=float(activation_beta),
        linear_beta=float(activation_linear_beta),
        limit=float(limit),
        hidden_workspace=hidden_workspace,
        result=result,
        max_group_tiles=(
            max(1, (int(group_offsets.diff().max().item()) + 3) // 4)
            if max_group_tiles is None
            else max(1, int(max_group_tiles))
        ),
    )
    if output is None:
        raise RuntimeError("grouped packed MoE operator rejected input")
    return output


def projection_dequant(
    metadata: torch.Tensor,
    output_gu: torch.Tensor,
    output_down: torch.Tensor,
) -> torch.Tensor:
    """Public packed-to-dense BF16 expansion for 2- or 3-projection experts."""
    _ensure_builtins()
    if metadata.ndim != 2 or metadata.shape[0] not in (10, 15):
        raise ValueError("projection dequant requires [10,E] or [15,E] metadata")
    request = OperatorRequest(
        operation="projection_dequant",
        device_type=metadata.device.type,
        packed_formats=tuple(f"p{bits}" for bits in range(8, 17)),
        code_dims=(4, 8, 16),
        codebook_sizes=(
            256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
        ),
        activation="situ",
        top_k=1,
        batch_size=int(output_gu.shape[0]) if output_gu.ndim == 3 else 1,
    )
    resolved = REGISTRY.resolve(request)
    output = resolved.implementation(
        metadata=metadata,
        output_gu=output_gu,
        output_down=output_down,
    )
    if output is None:
        def describe(tensor: torch.Tensor) -> str:
            return (
                f"device={tensor.device},dtype={tensor.dtype},"
                f"shape={tuple(tensor.shape)},"
                f"contiguous={tensor.is_contiguous()}"
            )

        raise RuntimeError(
            "projection dequant operator rejected input: "
            f"operator={resolved.name}; "
            f"metadata[{describe(metadata)}]; "
            f"gate_up[{describe(output_gu)}]; "
            f"down[{describe(output_down)}]"
        )
    return output


def packed_route_slots(
    route_ids: torch.Tensor,
    directory: torch.Tensor,
    *,
    output: torch.Tensor,
    hit_mask: torch.Tensor,
) -> bool:
    """Map Top-K expert IDs to stable packed-slot metadata on the device.

    ``directory`` is ``[expert_count, metadata_rows]`` and remains compact:
    it contains only fixed slot pointers and VQ shape tags, never expanded
    expert indices or dequantized matrices.  The operation is model-agnostic
    and graph-safe because all output buffers are supplied by the caller.
    """
    _ensure_builtins()
    device_type = route_ids.device.type
    try:
        implementation = _PACKED_ROUTE_SLOT_IMPLEMENTATIONS.get(device_type)
        if implementation is None:
            request = OperatorRequest(
                operation="packed_route_slots",
                device_type=device_type,
                activation="none",
                top_k=int(route_ids.numel()),
                batch_size=1,
            )
            implementation = REGISTRY.resolve(request).implementation
            _PACKED_ROUTE_SLOT_IMPLEMENTATIONS[device_type] = implementation
        return bool(
            implementation(
                route_ids=route_ids,
                directory=directory,
                output=output,
                hit_mask=hit_mask,
            )
        )
    except LookupError:
        return False


def packed_h2d_batch(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> bool:
    """Batch independent compact CPU-to-CUDA payload copies.

    Sources stay packed.  The operation only changes submission topology; it
    never concatenates, expands, or dequantizes expert weights.
    """
    _ensure_builtins()
    if not pairs or len(pairs) > 128:
        return False
    sources = [source for source, _target in pairs]
    destinations = [target for _source, target in pairs]
    device = destinations[0].device
    if (
        device.type != "cuda"
        or any(source.device.type != "cpu" for source in sources)
        or any(not source.is_pinned() for source in sources)
        or any(target.device != device for target in destinations)
    ):
        return False
    key = device.type
    try:
        implementation = _PACKED_H2D_BATCH_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="packed_h2d_batch",
                device_type=device.type,
                activation="none",
                top_k=len(pairs),
                batch_size=1,
            )
            implementation = REGISTRY.resolve(request).implementation
            _PACKED_H2D_BATCH_IMPLEMENTATIONS[key] = implementation
        return bool(
            implementation(
                sources=sources,
                destinations=destinations,
            )
        )
    except LookupError:
        return False


def packed_moe_operator_name(
    *,
    device_type: str,
    activation: str,
    top_k: int,
    packed_formats: tuple[str, ...],
    code_dims: tuple[int, ...],
    codebook_sizes: tuple[int, ...],
    batch_size: int = 1,
) -> str:
    """Resolve the public packed MoE backend for CLI diagnostics."""
    _ensure_builtins()
    return REGISTRY.resolve(
        OperatorRequest(
            operation="moe_topk",
            device_type=str(device_type),
            packed_formats=packed_formats,
            code_dims=code_dims,
            codebook_sizes=codebook_sizes,
            activation=str(activation),
            top_k=int(top_k),
            batch_size=int(batch_size),
        )
    ).name


def packed_moe_selected_topk(
    value: torch.Tensor,
    experts,
    route_weights: torch.Tensor,
    *,
    activation: str,
    activation_beta: float,
    activation_linear_beta: float | None,
    limit: float = 0.0,
) -> torch.Tensor | None:
    """Execute selected packed experts through the common ``moe_topk`` op.

    This is the RAM/LRU form of the resident-metadata interface above.  Model
    code supplies logical packed weights; the selected backend owns decoding,
    fusion and workspace policy.
    """
    if not experts:
        return None
    _ensure_builtins()
    # Packed projection-VQ is a generic bitstream format.  Keep the registry
    # key derived from the actual bit width so new heterogeneous manifests do
    # not need another model-specific lookup table.
    def packed_format(bits: int) -> str:
        if bits < 8 or bits > 16:
            raise ValueError(f"unsupported packed VQ width: {bits}")
        return f"p{bits}"
    packed_formats = tuple(
        sorted(
            {
                packed_format(int(weight.bits))
                for pair in experts
                for weight in pair
            }
        )
    )
    code_dims = tuple(
        sorted(
            {
                int(weight.dim)
                for pair in experts
                for weight in pair
            }
        )
    )
    codebook_sizes = tuple(
        sorted(
            {
                int(weight.cb.shape[0])
                for pair in experts
                for weight in pair
            }
        )
    )
    request = OperatorRequest(
        operation="moe_topk",
        device_type=value.device.type,
        packed_formats=packed_formats,
        code_dims=code_dims,
        codebook_sizes=codebook_sizes,
        activation=activation,
        top_k=len(experts),
        batch_size=value.shape[0],
    )
    try:
        return REGISTRY.call(
            request,
            value=value,
            experts=experts,
            weights=route_weights,
            limit=float(limit),
            activation=activation,
            beta=float(activation_beta),
            linear_beta=activation_linear_beta,
        )
    except LookupError:
        return None


def dense_vq_gemv_packed(
    x_rows: torch.Tensor,
    payload: torch.Tensor,
    codebook: torch.Tensor,
    *,
    rows: int,
    blocks: int,
    bits: int,
) -> torch.Tensor | None:
    """Run one ordinary Dense-VQ Linear without expanding packed indices."""
    _ensure_builtins()
    request = OperatorRequest(
        operation="dense_vq_gemv",
        device_type=x_rows.device.type,
        packed_formats=(f"p{int(bits)}",),
        code_dims=(int(codebook.shape[-1]),),
        codebook_sizes=(int(codebook.shape[-2]),),
        activation="none",
        top_k=1,
        batch_size=int(x_rows.shape[0]),
    )
    try:
        return REGISTRY.call(
            request,
            x_rows=x_rows,
            payload=payload,
            codebook=codebook,
            rows=int(rows),
            blocks=int(blocks),
            bits=int(bits),
        )
    except LookupError:
        return None


def dense_vq_mma_packed_m1(
    value: torch.Tensor,
    payload: torch.Tensor,
    codebook: torch.Tensor,
    *,
    rows: int,
    blocks: int,
    bits: int,
) -> torch.Tensor | None:
    """Run one-token packed VQ through the direct Tensor Core prototype."""
    _ensure_builtins()
    request = OperatorRequest(
        operation="dense_vq_mma",
        device_type=payload.device.type,
        packed_formats=(f"p{int(bits)}",),
        code_dims=(int(codebook.shape[-1]),),
        codebook_sizes=(int(codebook.shape[-2]),),
        activation="none",
        top_k=1,
        batch_size=int(value.shape[0]),
    )
    try:
        return REGISTRY.call(
            request,
            value=value,
            payload=payload,
            codebook=codebook,
            rows=int(rows),
            blocks=int(blocks),
            bits=int(bits),
        )
    except LookupError:
        return None


def dense_vq_dequant_packed(
    payload: torch.Tensor,
    codebook: torch.Tensor,
    *,
    rows: int,
    blocks: int,
    bits: int,
    row_ids: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Expand one Dense-VQ matrix, or selected rows, to transient BF16."""
    _ensure_builtins()
    request = OperatorRequest(
        operation="dense_vq_dequant",
        device_type=payload.device.type,
        packed_formats=(f"p{int(bits)}",),
        code_dims=(int(codebook.shape[-1]),),
        codebook_sizes=(int(codebook.shape[-2]),),
        activation="none",
        top_k=1,
        batch_size=1 if row_ids is None else max(1, int(row_ids.numel())),
    )
    try:
        return REGISTRY.call(
            request,
            payload=payload,
            codebook=codebook,
            rows=int(rows),
            blocks=int(blocks),
            bits=int(bits),
            row_ids=row_ids,
        )
    except LookupError:
        return None


def dense_vq_dequant_fp8_packed(
    payload: torch.Tensor,
    codebook: torch.Tensor,
    *,
    rows: int,
    blocks: int,
    bits: int,
    row_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Convert packed Dense-VQ rows directly to E4M3 and tensor scale.

    This is the common CUDA primitive for resident FP8 images and bounded
    row/tile conversion.  No full-size BF16 intermediate is materialized.
    """
    _ensure_builtins()
    request = OperatorRequest(
        operation="dense_vq_dequant_fp8",
        device_type=payload.device.type,
        packed_formats=(f"p{int(bits)}",),
        code_dims=(int(codebook.shape[-1]),),
        codebook_sizes=(int(codebook.shape[-2]),),
        activation="none",
        top_k=1,
        batch_size=1 if row_ids is None else max(1, int(row_ids.numel())),
    )
    try:
        return REGISTRY.call(
            request,
            payload=payload,
            codebook=codebook,
            rows=int(rows),
            blocks=int(blocks),
            bits=int(bits),
            row_ids=row_ids,
        )
    except LookupError:
        return None


def dense_vq_quantize_fp8_codebook(
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Quantize one compact FP32 VQ codebook for tiled E4M3 execution."""
    _ensure_builtins()
    request = OperatorRequest(
        operation="dense_vq_codebook_fp8",
        device_type=codebook.device.type,
        code_dims=(int(codebook.shape[-1]),),
        codebook_sizes=(int(codebook.shape[-2]),),
        activation="none",
        top_k=1,
        batch_size=1,
    )
    try:
        return REGISTRY.call(request, codebook=codebook)
    except LookupError:
        return None


def dense_vq_expand_fp8_tile(
    payload: torch.Tensor,
    fp8_codebook: torch.Tensor,
    output: torch.Tensor,
    *,
    rows: int,
    blocks: int,
    bits: int,
    row_start: int,
    row_count: int,
) -> torch.Tensor | None:
    """Expand a contiguous packed VQ row tile into fixed E4M3 storage."""
    _ensure_builtins()
    request = OperatorRequest(
        operation="dense_vq_expand_fp8_tile",
        device_type=payload.device.type,
        packed_formats=(f"p{int(bits)}",),
        code_dims=(int(fp8_codebook.shape[-1]),),
        codebook_sizes=(int(fp8_codebook.shape[-2]),),
        activation="none",
        top_k=1,
        batch_size=int(row_count),
    )
    try:
        return REGISTRY.call(
            request,
            payload=payload,
            fp8_codebook=fp8_codebook,
            output=output,
            rows=int(rows),
            blocks=int(blocks),
            bits=int(bits),
            row_start=int(row_start),
            row_count=int(row_count),
        )
    except LookupError:
        return None


def packed_moe_selected_rows(
    value: torch.Tensor,
    experts,
    route_weights: torch.Tensor,
    *,
    activation: str,
    activation_beta: float,
    activation_linear_beta: float | None,
    limit: float = 0.0,
) -> torch.Tensor | None:
    """Execute independently routed packed rows through the common CPU op."""
    if (
        value.ndim != 2
        or value.shape[0] < 2
        or route_weights.ndim != 2
        or route_weights.shape[0] != value.shape[0]
        or len(experts) != value.shape[0]
    ):
        return None
    top_k = int(route_weights.shape[1])
    flat = [bundle for row in experts for bundle in row]
    if top_k <= 0 or any(len(row) != top_k for row in experts) or not flat:
        return None
    _ensure_builtins()

    def packed_format(bits: int) -> str:
        if bits < 8 or bits > 16:
            raise ValueError(f"unsupported packed VQ width: {bits}")
        return f"p{bits}"

    request = OperatorRequest(
        operation="moe_topk:rows",
        device_type=value.device.type,
        packed_formats=tuple(sorted({
            packed_format(int(weight.bits))
            for bundle in flat
            for weight in bundle
        })),
        code_dims=tuple(sorted({
            int(weight.dim) for bundle in flat for weight in bundle
        })),
        codebook_sizes=tuple(sorted({
            int(weight.cb.shape[0]) for bundle in flat for weight in bundle
        })),
        activation=activation,
        top_k=top_k,
        batch_size=int(value.shape[0]),
    )
    try:
        return REGISTRY.call(
            request,
            value=value,
            experts=experts,
            weights=route_weights,
            limit=float(limit),
            activation=activation,
            beta=float(activation_beta),
            linear_beta=activation_linear_beta,
        )
    except LookupError:
        return None


def resident_moe_topk(
    value: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    metadata: torch.Tensor,
    *,
    activation: str,
    limit: float,
    codegemm_gu_workspace: torch.Tensor | None,
    codegemm_activation_workspace: torch.Tensor | None,
    codegemm_down_workspace: torch.Tensor | None,
    hidden_workspace: torch.Tensor,
    output_workspace: torch.Tensor,
    result: torch.Tensor,
    packed_formats: tuple[str, ...],
    code_dims: tuple[int, ...],
    codebook_sizes: tuple[int, ...],
) -> torch.Tensor | None:
    """Run mixed resident codebooks without a host-side route split.

    This interface describes storage and math capabilities only.  It is shared
    by every model configuration using a SwiGLU Top-K resident expert layout.
    """
    _ensure_builtins()
    normalized_formats = tuple(sorted(set(packed_formats)))
    normalized_dims = tuple(sorted(set(int(v) for v in code_dims)))
    normalized_sizes = tuple(sorted(set(int(v) for v in codebook_sizes)))
    key = (
        value.device.type,
        normalized_formats,
        normalized_dims,
        normalized_sizes,
    )
    try:
        implementation = _RESIDENT_MOE_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="resident_moe_topk",
                device_type=value.device.type,
                packed_formats=normalized_formats,
                code_dims=normalized_dims,
                codebook_sizes=normalized_sizes,
                activation=activation,
                top_k=int(route_ids.numel()),
                batch_size=int(value.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _RESIDENT_MOE_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            route_ids=route_ids,
            weights=route_weights,
            metadata=metadata,
            limit=float(limit),
            codegemm_gu_workspace=codegemm_gu_workspace,
            codegemm_activation_workspace=(
                codegemm_activation_workspace
            ),
            codegemm_down_workspace=codegemm_down_workspace,
            hidden_workspace=hidden_workspace,
            output_workspace=output_workspace,
            result=result,
            include_k4096=4096 in normalized_sizes,
        )
    except LookupError:
        return None


def create_resident_moe_layer(
    executor,
    experts: tuple[tuple[object, object, object], ...],
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    router_mask: torch.Tensor,
    shared_weights: tuple[object, object, object],
    *,
    activation: str,
    top_k: int,
    normalize_route: bool,
    routed_scaling: float,
):
    """Compose a persistent Router/shared/routed CPU MoE executor.

    Dispatch is based solely on packed formats, codebook geometry, dense
    block format and activation.  The returned executor keeps every source
    tensor compact and can be reused for every decode token in that layer.
    """
    _ensure_builtins()
    if not experts or len(shared_weights) != 3:
        return None
    packed_formats = tuple(
        sorted(
            {f"p{int(weight.bits)}" for bundle in experts for weight in bundle}
            | {
                "e4m3fn-block-major32"
                if getattr(weight, "layout", "row-major")
                == "block-major32"
                else "e4m3fn"
                for weight in shared_weights
            }
            | {str(router_weight.dtype).removeprefix("torch.")}
        )
    )
    code_dims = tuple(
        sorted(
            {int(weight.dim) for bundle in experts for weight in bundle}
            | {int(getattr(weight, "block", 128)) for weight in shared_weights}
        )
    )
    codebook_sizes = tuple(
        sorted(
            {
                int(weight.cb.shape[0])
                for bundle in experts
                for weight in bundle
            }
        )
    )
    request = OperatorRequest(
        operation="resident_moe_layer",
        device_type=router_weight.device.type,
        packed_formats=packed_formats,
        code_dims=code_dims,
        codebook_sizes=codebook_sizes,
        activation=activation,
        top_k=int(top_k),
        batch_size=1,
    )
    try:
        return REGISTRY.call(
            request,
            executor=executor,
            router_weight=router_weight,
            router_bias=router_bias,
            router_mask=router_mask,
            shared_weights=shared_weights,
            top_k=int(top_k),
            normalize_route=bool(normalize_route),
            routed_scaling=float(routed_scaling),
        )
    except LookupError:
        return None


def resident_moe_forward_rows(
    executor,
    value: torch.Tensor,
    *,
    limit: float,
    activation: str,
    beta: float,
    linear_beta: float,
) -> torch.Tensor | None:
    """Reuse one fixed resident MoE executor for decode or multi-row prefill.

    The native layer owns token-sized fixed buffers.  Multi-token callers copy
    each completed row before reusing those buffers, so no second expert image
    and no full dequantized matrix is required.
    """
    if (
        executor is None
        or executor is False
        or not hasattr(executor, "forward_fused_moe")
        or value.is_cuda
        or value.ndim != 2
        or value.shape[0] < 1
    ):
        return None
    output = torch.empty(
        value.shape[0], value.shape[1], dtype=torch.float32, device=value.device
    )
    for row in range(value.shape[0]):
        result = executor.forward_fused_moe(
            value[row : row + 1],
            float(limit),
            str(activation),
            float(beta),
            float(linear_beta),
        )
        if result.numel() != value.shape[1]:
            return None
        output[row].copy_(result)
    return output


def create_latent_resident_moe_layer(
    executor,
    experts: tuple[tuple[object, object, object], ...],
    input_weights: tuple[object, object, object, object],
    output_weights: tuple[object, object],
    route_correction: torch.Tensor,
    route_mask: torch.Tensor,
    routed_norm: torch.Tensor,
    *,
    activation: str,
    scoring: str,
    top_k: int,
    normalize_route: bool,
    routed_scaling: float,
    rms_eps: float,
    limit: float,
    beta: float,
    linear_beta: float | None,
):
    """Create one format-driven full latent-MoE CPU decode executor."""
    from ..kernels import BlockFP8Weight

    _ensure_builtins()
    if not experts or len(input_weights) != 4 or len(output_weights) != 2:
        return None

    def dense_format(weight) -> str | None:
        if isinstance(weight, BlockFP8Weight) and not weight.q.is_cuda:
            return (
                "e4m3fn-block-major32"
                if weight.layout == "block-major32"
                else "e4m3fn"
            )
        if (
            isinstance(weight, torch.Tensor)
            and not weight.is_cuda
            and weight.dtype in (torch.bfloat16, torch.float32)
        ):
            return str(weight.dtype).removeprefix("torch.")
        return None

    dense_formats = {
        dense_format(weight)
        for weight in (*input_weights, *output_weights)
    }
    if None in dense_formats:
        return None
    packed_formats = tuple(sorted(
        {f"p{int(weight.bits)}" for bundle in experts for weight in bundle}
        | dense_formats
    ))
    code_dims = tuple(sorted(
        {int(weight.dim) for bundle in experts for weight in bundle}
        | {128}
    ))
    codebook_sizes = tuple(sorted({
        int(weight.cb.shape[0])
        for bundle in experts
        for weight in bundle
    }))
    request = OperatorRequest(
        operation="latent_resident_moe_layer",
        device_type="cpu",
        packed_formats=packed_formats,
        code_dims=code_dims,
        codebook_sizes=codebook_sizes,
        activation=activation,
        top_k=int(top_k),
        batch_size=1,
    )
    try:
        return REGISTRY.call(
            request,
            executor=executor,
            input_weights=input_weights,
            output_weights=output_weights,
            route_correction=route_correction,
            route_mask=route_mask,
            routed_norm=routed_norm,
            top_k=int(top_k),
            normalize_route=bool(normalize_route),
            routed_scaling=float(routed_scaling),
            rms_eps=float(rms_eps),
            limit=float(limit),
            scoring=str(scoring),
            activation=str(activation),
            beta=float(beta),
            linear_beta=linear_beta,
        )
    except LookupError:
        return None


def create_tensor_parallel(
    kind: str,
    devices: tuple[torch.device, ...],
    spec,
):
    """通过公共能力注册创建有状态 TP executor。"""
    _ensure_builtins()
    normalized = kind.strip().lower()
    activation = str(getattr(spec, "activation", "none")).lower()
    request = OperatorRequest(
        operation=f"tensor_parallel:{normalized}",
        device_type="cuda",
        activation=activation,
        top_k=1,
        batch_size=1,
    )
    return REGISTRY.call(
        request,
        kind=normalized,
        devices=tuple(devices),
        spec=spec,
    )

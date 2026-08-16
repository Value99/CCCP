"""CPU 通用量化算子注册。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import OperatorRegistry
from .spec import OperatorCapability


def _rmsnorm(**kwargs):
    value = kwargs["value"]
    weight = kwargs["weight"]
    source_dtype = value.dtype
    work = value.float()
    work = work * (
        work.square().mean(dim=-1, keepdim=True) + kwargs["eps"]
    ).rsqrt()
    result = weight.to(source_dtype) * work.to(source_dtype)
    output = kwargs.get("output")
    if output is not None:
        output.copy_(result)
        return output
    return result


def _attention_residual(**kwargs):
    prefix = kwargs["prefix"]
    residual = kwargs["residual"]
    projection = kwargs["projection"]
    norm_weight = kwargs["norm_weight"]
    values = torch.cat((residual, prefix.unsqueeze(-2)), dim=-2)
    values_f = values.float()
    inverse = torch.rsqrt(
        values_f.square().mean(dim=-1, keepdim=True)
        + kwargs["eps"]
    )
    residual_inverse = kwargs.get("residual_inverse")
    if residual_inverse is not None:
        cached = residual_inverse.reshape(1, -1, 1)
        inverse[..., :-1, :].copy_(
            torch.where(
                cached > 0,
                cached,
                inverse[..., :-1, :],
            )
        )
        cached.copy_(inverse[..., :-1, :])
    normalized = values_f * inverse
    score_weight = norm_weight.float() * projection.reshape(-1).float()
    scores = (normalized * score_weight).sum(dim=-1)
    probabilities = scores.softmax(dim=-1).unsqueeze(-2)
    result = torch.matmul(
        probabilities,
        values_f,
    ).squeeze(-2).to(values.dtype)
    post_norm_weight = kwargs.get("post_norm_weight")
    if post_norm_weight is not None:
        work = result.float()
        normalized = work * torch.rsqrt(
            work.square().mean(dim=-1, keepdim=True)
            + kwargs["eps"]
        )
        result = (
            post_norm_weight.to(result.dtype)
            * normalized.to(result.dtype)
        )
    output = kwargs.get("output")
    if output is not None:
        output.copy_(result)
        return output
    return result


def _residual_add3(**kwargs):
    return kwargs["residual"] + (
        kwargs["routed"] + kwargs["shared"]
    )


def _causal_latent_prefill(**kwargs):
    from .attention_prefill import causal_latent_prefill

    return causal_latent_prefill(**kwargs)


def _gated_activation(**kwargs):
    gate = kwargs["gate"]
    up = kwargs["up"]
    activation = kwargs["activation"]
    if activation in {"silu", "swiglu"}:
        limit = float(kwargs.get("limit", 0.0))
        if limit > 0.0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        result = F.silu(gate) * up
    else:
        gate_f = gate.float()
        up_f = up.float()
        beta = kwargs["beta"]
        activated = (
            beta
            * torch.tanh(gate_f / beta)
            * torch.sigmoid(gate_f)
        )
        linear_beta = kwargs["linear_beta"]
        if linear_beta is not None:
            up_f = linear_beta * torch.tanh(up_f / linear_beta)
        result = (activated * up_f).to(gate.dtype)
    output = kwargs.get("output")
    if output is not None:
        output.copy_(result)
        return output
    return result


def _vq_gemv(**kwargs):
    from ..cpuext import vq_gemv_cpu

    return vq_gemv_cpu(
        kwargs["x_rows"],
        kwargs["indices"],
        kwargs["codebook"],
    )


def _vq_relayout(**kwargs):
    from ..cpuext import vq_repack_block_major_cpu

    return vq_repack_block_major_cpu(
        kwargs["payload"],
        kwargs["rows"],
        kwargs["blocks"],
        kwargs["bits"],
    )


def _vq_relayout_row_tile(**kwargs):
    from ..cpuext import vq_repack_row_tile_cpu

    return vq_repack_row_tile_cpu(
        kwargs["payload"],
        kwargs["rows"],
        kwargs["blocks"],
        kwargs["bits"],
        kwargs.get("tile_rows", 8),
    )


def _vq_compile_u16_row_tile(**kwargs):
    from ..cpuext import vq_compile_u16_row_tile_cpu

    return vq_compile_u16_row_tile_cpu(
        kwargs["payload"],
        kwargs["rows"],
        kwargs["blocks"],
        kwargs["bits"],
        kwargs.get("tile_rows", 8),
    )


def _vq_compile_q4_0(**kwargs):
    from ..cpuext import vq_compile_q4_0_cpu

    return vq_compile_q4_0_cpu(
        kwargs["payload"],
        kwargs["codebook"],
        kwargs["rows"],
        kwargs["blocks"],
        kwargs["bits"],
    )


def _block_scaled_gemv(**kwargs):
    from ..cpuext import block_fp8_gemv_cpu

    return block_fp8_gemv_cpu(
        kwargs["value"],
        kwargs["weights"],
        kwargs["scales"],
        kwargs["cols"],
        kwargs["block_size"],
        kwargs.get("output"),
        rows=kwargs.get("rows"),
    )


def _block_scaled_gemm(**kwargs):
    from ..cpuext import block_fp8_gemm_cpu

    return block_fp8_gemm_cpu(
        kwargs["value"],
        kwargs["weights"],
        kwargs["scales"],
        kwargs["cols"],
        kwargs["block_size"],
        kwargs.get("output"),
        rows=kwargs.get("rows"),
    )


def _block_scaled_grouped_gemv(**kwargs):
    from ..cpuext import block_fp8_grouped_gemv_cpu

    return block_fp8_grouped_gemv_cpu(
        kwargs["value"],
        kwargs["weight_ptrs"],
        kwargs["scale_ptrs"],
        kwargs["row_offsets"],
        kwargs["total_rows"],
        kwargs["cols"],
        kwargs["block_size"],
        kwargs.get("output"),
        block_major=kwargs.get("block_major", False),
    )


def _dense_grouped_gemv(**kwargs):
    from ..cpuext import bf16_grouped_gemv_cpu

    return bf16_grouped_gemv_cpu(
        kwargs["value"],
        kwargs["weight_ptrs"],
        kwargs["row_offsets"],
        kwargs["total_rows"],
        kwargs["cols"],
        kwargs.get("output"),
    )


def _block_scaled_grouped_gemm(**kwargs):
    from ..cpuext import block_fp8_grouped_gemm_cpu

    return block_fp8_grouped_gemm_cpu(
        kwargs["value"],
        kwargs["weight_ptrs"],
        kwargs["scale_ptrs"],
        kwargs["row_offsets"],
        kwargs["total_rows"],
        kwargs["cols"],
        kwargs["block_size"],
        kwargs.get("output"),
        block_major=kwargs.get("block_major", False),
    )


def _block_scaled_grouped_rows_gemv(**kwargs):
    from ..cpuext import block_fp8_grouped_rows_gemv_cpu

    return block_fp8_grouped_rows_gemv_cpu(
        kwargs["value"],
        kwargs["weight_ptrs"],
        kwargs["scale_ptrs"],
        kwargs["row_offsets"],
        kwargs["total_rows"],
        kwargs["cols"],
        kwargs["block_size"],
        kwargs.get("output"),
        block_major=kwargs.get("block_major", False),
    )


def _vq_gemv_packed_list(**kwargs):
    from ..cpuext import vq_gemv_packed_list_cpu

    return vq_gemv_packed_list_cpu(
        kwargs["x_rows"],
        kwargs["payloads"],
        kwargs["codebook"],
        kwargs["rows"],
        kwargs["blocks"],
        kwargs["bits"],
        allow_direct=kwargs.get("allow_direct", False),
    )


def _packed_moe_topk(**kwargs):
    from ..cpuext import moe_packed_topk_cpu

    return moe_packed_topk_cpu(
        kwargs["value"],
        kwargs["experts"],
        kwargs["weights"],
        kwargs.get("limit", 0.0),
        activation=kwargs["activation"],
        activation_beta=kwargs["beta"],
        activation_linear_beta=kwargs.get("linear_beta"),
    )


def _packed_moe_rows(**kwargs):
    from ..cpuext import moe_packed_rows_cpu

    return moe_packed_rows_cpu(
        kwargs["value"],
        kwargs["experts"],
        kwargs["weights"],
        kwargs.get("limit", 0.0),
        activation=kwargs["activation"],
        activation_beta=kwargs["beta"],
        activation_linear_beta=kwargs.get("linear_beta"),
    )


def _resident_moe_layer(**kwargs):
    from ..cpuext import configure_packed_resident_moe_cpu

    return configure_packed_resident_moe_cpu(
        kwargs["executor"],
        kwargs["router_weight"],
        kwargs["router_bias"],
        kwargs["router_mask"],
        kwargs["shared_weights"],
        top_k=kwargs["top_k"],
        normalize_route=kwargs["normalize_route"],
        routed_scaling=kwargs["routed_scaling"],
    )


def _latent_resident_moe_layer(**kwargs):
    from ..cpuext import configure_packed_latent_moe_cpu

    return configure_packed_latent_moe_cpu(
        kwargs["executor"],
        kwargs["input_weights"],
        kwargs["output_weights"],
        kwargs["route_correction"],
        kwargs["route_mask"],
        kwargs["routed_norm"],
        top_k=kwargs["top_k"],
        normalize_route=kwargs["normalize_route"],
        routed_scaling=kwargs["routed_scaling"],
        rms_eps=kwargs["rms_eps"],
        limit=kwargs["limit"],
        scoring=kwargs["scoring"],
        activation=kwargs["activation"],
        beta=kwargs["beta"],
        linear_beta=kwargs.get("linear_beta"),
    )


def _resident_projection_layer(**kwargs):
    from ..cpuext import make_resident_projection_cpu

    return make_resident_projection_cpu(tuple(kwargs["weights"]))


def _kda_recurrent(**kwargs):
    from ..cpuext import kda_recurrent_cpu

    return kda_recurrent_cpu(**kwargs)


def _short_conv3(**kwargs):
    from ..cpuext import short_conv3_cpu

    return short_conv3_cpu(
        kwargs["query"],
        kwargs["key"],
        kwargs["value"],
        kwargs["states"],
        kwargs["weights"],
    )


def _gated_rmsnorm(**kwargs):
    from ..cpuext import gated_rmsnorm_cpu

    return gated_rmsnorm_cpu(
        kwargs["value"],
        kwargs["gate"],
        kwargs["weight"],
        kwargs["output"],
        kwargs["eps"],
    )


def _route_topk(**kwargs):
    logits = kwargs["logits"]
    try:
        from ..cpuext import route_topk_sigmoid_cpu

        routed = route_topk_sigmoid_cpu(
            logits,
            kwargs["bias"],
            kwargs["mask"],
            kwargs["top_k"],
            kwargs["normalize"],
            kwargs["scaling"],
        )
    except (ImportError, RuntimeError):
        routed = None
    if routed is not None:
        weights, indices = routed
        output_buffers = kwargs.get("output_buffers")
        if output_buffers is not None:
            output_weights, output_indices = output_buffers
            output_weights.copy_(weights)
            output_indices.copy_(indices)
            return output_weights, output_indices
        return weights, indices
    scores = logits.float().sigmoid()
    choice = scores + kwargs["bias"].float()
    choice = choice.masked_fill(~kwargs["mask"], float("-inf"))
    indices = torch.argsort(
        choice,
        dim=-1,
        descending=True,
        stable=True,
    )[:, : kwargs["top_k"]]
    weights = scores.gather(-1, indices)
    if kwargs["normalize"] and kwargs["top_k"] > 1:
        weights = weights / (
            weights.sum(dim=-1, keepdim=True) + 1.0e-20
        )
    weights = weights * kwargs["scaling"]
    output_buffers = kwargs.get("output_buffers")
    if output_buffers is not None:
        output_weights, output_indices = output_buffers
        output_weights.copy_(weights)
        output_indices.copy_(indices)
        return output_weights, output_indices
    return weights, indices


def _linear_route_topk(**kwargs):
    logits, output_weights, output_indices = kwargs["output_buffers"]
    value = kwargs["value"].float()
    weight = kwargs["weight"]
    if not isinstance(weight, torch.Tensor):
        return None
    torch.mm(value, weight.float().t(), out=logits)
    return _route_topk(
        logits=logits,
        bias=kwargs["bias"],
        mask=kwargs["mask"],
        top_k=kwargs["top_k"],
        normalize=kwargs["normalize"],
        scaling=kwargs["scaling"],
        output_buffers=(output_weights, output_indices),
    )


def register(registry: OperatorRegistry) -> None:
    registry.register(
        "cpu.resident_projection_layer.mixed.decode",
        OperatorCapability(
            operation="resident_projection_layer",
            device_types=("cpu",),
            packed_formats=(
                "bf16",
                "e4m3fn",
                "e4m3fn-block-major32",
                "q4_0-linear-block32",
            ),
            code_dims=(32, 128),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _resident_projection_layer,
        priority=120,
    )
    registry.register(
        "cpu.resident_moe_layer.packed_block_fp8.decode",
        OperatorCapability(
            operation="resident_moe_layer",
            device_types=("cpu",),
            packed_formats=(
                *(f"p{bits}" for bits in range(8, 17)),
                "e4m3fn",
                "e4m3fn-block-major32",
                "float32",
                "bfloat16",
            ),
            code_dims=(4, 8, 16, 128),
            codebook_sizes=(
                256, 512, 1024, 2048, 4096,
                8192, 16384, 32768, 65536,
            ),
            activations=("silu", "swiglu", "situ"),
            max_top_k=16,
            batch_sizes=(1,),
        ),
        _resident_moe_layer,
        priority=120,
    )
    registry.register(
        "cpu.latent_resident_moe_layer.packed_block_fp8.decode",
        OperatorCapability(
            operation="latent_resident_moe_layer",
            device_types=("cpu",),
            packed_formats=(
                *(f"p{bits}" for bits in range(8, 17)),
                "e4m3fn",
                "e4m3fn-block-major32",
                "float32",
                "bfloat16",
            ),
            code_dims=(4, 8, 16, 128),
            codebook_sizes=(
                256, 512, 1024, 2048, 4096,
                8192, 16384, 32768, 65536,
            ),
            activations=("silu", "swiglu", "situ"),
            max_top_k=16,
            batch_sizes=(1,),
        ),
        _latent_resident_moe_layer,
        priority=130,
    )
    registry.register(
        "cpu.dense_grouped_gemv.bf16.decode",
        OperatorCapability(
            operation="dense_grouped_gemv",
            device_types=("cpu",),
            packed_formats=("bf16",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _dense_grouped_gemv,
        priority=100,
    )
    registry.register(
        "cpu.block_scaled_gemv.e4m3fn.b128.decode",
        OperatorCapability(
            operation="block_scaled_gemv",
            device_types=("cpu",),
            packed_formats=("e4m3fn", "e4m3fn-block-major32"),
            code_dims=(128,),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _block_scaled_gemv,
        priority=100,
    )
    registry.register(
        "cpu.block_scaled_gemm.e4m3fn.b128.verify",
        OperatorCapability(
            operation="block_scaled_gemm",
            device_types=("cpu",),
            packed_formats=("e4m3fn", "e4m3fn-block-major32"),
            code_dims=(128,),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(2, 17)),
        ),
        _block_scaled_gemm,
        priority=100,
    )
    registry.register(
        "cpu.block_scaled_grouped_gemv.e4m3fn.b128.decode",
        OperatorCapability(
            operation="block_scaled_grouped_gemv",
            device_types=("cpu",),
            packed_formats=("e4m3fn", "e4m3fn-block-major32"),
            code_dims=(128,),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _block_scaled_grouped_gemv,
        priority=100,
    )
    registry.register(
        "cpu.block_scaled_grouped_gemm.e4m3fn.b128.verify",
        OperatorCapability(
            operation="block_scaled_grouped_gemm",
            device_types=("cpu",),
            packed_formats=("e4m3fn", "e4m3fn-block-major32"),
            code_dims=(128,),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(2, 17)),
        ),
        _block_scaled_grouped_gemm,
        priority=100,
    )
    registry.register(
        "cpu.block_scaled_grouped_rows_gemv.e4m3fn.b128.decode",
        OperatorCapability(
            operation="block_scaled_grouped_rows_gemv",
            device_types=("cpu",),
            packed_formats=("e4m3fn", "e4m3fn-block-major32"),
            code_dims=(128,),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(1, 17)),
        ),
        _block_scaled_grouped_rows_gemv,
        priority=100,
    )
    registry.register(
        "cpu.residual_add.three_way.reference",
        OperatorCapability(
            operation="residual_add:three_way",
            device_types=("cpu",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(1, 257)),
        ),
        _residual_add3,
        priority=10,
    )
    registry.register(
        "cpu.route_topk.sigmoid.reference",
        OperatorCapability(
            operation="route_topk",
            device_types=("cpu",),
            activations=("sigmoid",),
            max_top_k=16,
            batch_sizes=tuple(range(1, 257)),
        ),
        _route_topk,
        priority=10,
    )
    registry.register(
        "cpu.linear_route_topk.sigmoid.decode",
        OperatorCapability(
            operation="linear_route_topk",
            device_types=("cpu",),
            activations=("sigmoid",),
            max_top_k=16,
            batch_sizes=tuple(range(1, 257)),
        ),
        _linear_route_topk,
        priority=100,
    )
    registry.register(
        "cpu.gated_activation.reference",
        OperatorCapability(
            operation="gated_activation",
            device_types=("cpu",),
            activations=("silu", "swiglu", "situ"),
            max_top_k=1,
            batch_sizes=tuple(range(1, 257)),
        ),
        _gated_activation,
        priority=10,
    )
    registry.register(
        "cpu.residual_mix.attention.reference",
        OperatorCapability(
            operation="residual_mix:attention",
            device_types=("cpu",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(2, 17)),
        ),
        _attention_residual,
        priority=10,
    )
    registry.register(
        "cpu.normalization.rmsnorm.reference",
        OperatorCapability(
            operation="normalization:rmsnorm",
            device_types=("cpu",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(1, 257)),
        ),
        _rmsnorm,
        priority=10,
    )
    registry.register(
        "cpu.vq_gemv.index_tensor.batch",
        OperatorCapability(
            operation="vq_gemv",
            device_types=("cpu",),
            packed_formats=("u8", "u16"),
            code_dims=(2, 4, 8, 16),
            codebook_sizes=(256, 1024, 4096, 16384, 65536),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(1, 17)),
        ),
        _vq_gemv,
        priority=50,
    )
    registry.register(
        "cpu.vq_gemv.packed_list.decode",
        OperatorCapability(
            operation="vq_gemv:list",
            device_types=("cpu",),
            packed_formats=tuple(f"p{bits}" for bits in range(8, 17)),
            code_dims=(2, 4, 8, 16),
            codebook_sizes=(
                256, 512, 1024, 2048, 4096,
                8192, 16384, 32768, 65536,
            ),
            activations=("none",),
            max_top_k=16,
            batch_sizes=tuple(range(1, 17)),
        ),
        _vq_gemv_packed_list,
        priority=60,
    )
    registry.register(
        "cpu.vq_relayout.compact_block_major",
        OperatorCapability(
            operation="vq_relayout:block_major",
            device_types=("cpu",),
            packed_formats=tuple(f"p{bits}" for bits in range(8, 17)),
            code_dims=(4, 8, 16),
            codebook_sizes=(
                256, 512, 1024, 2048, 4096,
                8192, 16384, 32768, 65536,
            ),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _vq_relayout,
        priority=100,
    )
    registry.register(
        "cpu.vq_relayout.compact_row_tile",
        OperatorCapability(
            operation="vq_relayout:row_tile",
            device_types=("cpu",),
            packed_formats=tuple(f"p{bits}" for bits in range(8, 17)),
            code_dims=(4, 8, 16),
            codebook_sizes=(
                256, 512, 1024, 2048, 4096,
                8192, 16384, 32768, 65536,
            ),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _vq_relayout_row_tile,
        priority=110,
    )
    registry.register(
        "cpu.vq_compile.u16_row_tile",
        OperatorCapability(
            operation="vq_compile:u16_row_tile",
            device_types=("cpu",),
            packed_formats=tuple(f"p{bits}" for bits in range(8, 17)),
            code_dims=(2, 4, 8, 16),
            codebook_sizes=(
                256, 512, 1024, 2048, 4096,
                8192, 16384, 32768, 65536,
            ),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _vq_compile_u16_row_tile,
        priority=120,
    )
    registry.register(
        "cpu.vq_compile.q4_0.linear_block_dot",
        OperatorCapability(
            operation="vq_compile:q4_0",
            device_types=("cpu",),
            packed_formats=tuple(f"p{bits}" for bits in range(8, 17)),
            code_dims=(2, 4, 8, 16),
            codebook_sizes=(
                256, 512, 1024, 2048, 4096,
                8192, 16384, 32768, 65536,
            ),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _vq_compile_q4_0,
        priority=130,
    )
    registry.register(
        "cpu.packed_moe_topk.mixed.persistent_pool",
        OperatorCapability(
            operation="moe_topk",
            device_types=("cpu",),
            packed_formats=tuple(f"p{bits}" for bits in range(8, 17)),
            code_dims=(4, 8, 16),
            codebook_sizes=(
                256, 512, 1024, 2048, 4096,
                8192, 16384, 32768, 65536,
            ),
            activations=("silu", "swiglu", "situ"),
            max_top_k=16,
            batch_sizes=(1,),
        ),
        _packed_moe_topk,
        priority=100,
    )
    registry.register(
        "cpu.packed_moe_rows.mixed.expert_grouped_gemm",
        OperatorCapability(
            operation="moe_topk:rows",
            device_types=("cpu",),
            packed_formats=tuple(f"p{bits}" for bits in range(8, 17)),
            code_dims=(4, 8, 16),
            codebook_sizes=(
                256, 512, 1024, 2048, 4096,
                8192, 16384, 32768, 65536,
            ),
            activations=("silu", "swiglu", "situ"),
            max_top_k=16,
            batch_sizes=tuple(range(2, 4097)),
        ),
        _packed_moe_rows,
        priority=100,
    )
    registry.register(
        "cpu.attention.causal_latent.prefill",
        OperatorCapability(
            operation="attention_step:causal_latent_prefill",
            device_types=("cpu",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(1, 8193)),
        ),
        _causal_latent_prefill,
        priority=100,
    )
    registry.register(
        "cpu.attention.kda_recurrent.avx512",
        OperatorCapability(
            operation="attention_step:kda_recurrent",
            device_types=("cpu",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _kda_recurrent,
        priority=100,
    )
    registry.register(
        "cpu.attention.short_conv3.persistent_pool",
        OperatorCapability(
            operation="attention_step:short_conv3",
            device_types=("cpu",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _short_conv3,
        priority=100,
    )
    registry.register(
        "cpu.attention.gated_rmsnorm.fused",
        OperatorCapability(
            operation="attention_step:gated_rmsnorm",
            device_types=("cpu",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _gated_rmsnorm,
        priority=100,
    )

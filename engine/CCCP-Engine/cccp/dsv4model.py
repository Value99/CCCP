"""CCCP 推理：DeepSeek-V4（CCCP 产物）前向模型。

加载 "cccp-1" 格式（cccp.json + dense.safetensors + experts.L*.safetensors），
前向复用 CCCP/dsv4.py 的公共数学件（hc_pre/hc_post/hc_head/hc_split/rmsnorm/
rope_apply/compressor_*/attn_* 的无权重依赖部分）；权重路径：
  - 大 dense 矩阵（wq_a/wq_b/wkv/wo_a/wo_b/shared/head/embed）：Int4Weight 打包驻留
    （显存/内存），经 _linear 走 LUT 反量化矩阵乘；
  - 小权重（compressor/norms/hc/gate/attn_sink/ape/tid2eid）：f32 原样；
  - routed 专家：ExpertPool 两级 LRU 的 VQWeight（LUT 免还原矩阵乘）。
与 CCCP/dsv4.py 的关系：数值公式一致；为接入 int4/VQ 权重，线性层经本文件的
_linear 分派（F.linear ↔ Int4Weight.matmul_T ↔ VQWeight.matmul_T）。
"""

from __future__ import annotations

import os
import copy
import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .dsv4cache import ContextCapacityError, PagedKV
from .dsv4indexer import IndexerState
from .prefill import (
    prefill_block_size,
    prefill_ranges as _prefill_ranges,
    run_prefill_blocks,
)
from .kernels import (
    BlockFP8Weight,
    Int4Weight,
    ProjectionGroup,
    VQWeight,
    rmsnorm,
)
from .precision import compute_dtype
from .store import CCCPStore


_DENSE_BF16_ELIGIBLE = frozenset({
    "attention", "compressor", "embed", "head", "hyper", "indexer",
    "norm", "shared",
})
_DENSE_BF16_ALIASES = {
    "core": "attention",
    "embedding": "embed",
    "output_head": "head",
    "shared_experts": "shared",
}


def _resolve_native_tensor_fp8_execution(
    mode: str,
    *,
    available: bool,
) -> bool:
    """Resolve the public fixed-projection Tensor-FP8 policy exactly.

    ``auto`` is capability based.  Explicit ``on`` is strict so an
    unsupported machine never grows a hidden BF16 execution image or silently
    changes the requested compute route.
    """

    normalized = str(mode).strip().lower()
    if normalized in ("", "auto"):
        return bool(available)
    if normalized in ("0", "false", "off", "no", "none"):
        return False
    if normalized in ("1", "true", "on", "yes"):
        if not available:
            raise RuntimeError(
                "CCCP_GPU_FP8_EXECUTION=on，但当前硬件/运行时不支持 "
                "Tensor Core E4M3 scaled-MM"
            )
        return True
    raise ValueError(
        "CCCP_GPU_FP8_EXECUTION 仅支持 auto/on/off，"
        f"当前值为 {mode!r}"
    )


def _automatic_prefetch_policy(
    *,
    resident_all: bool,
    packed_device_pool: bool,
    packed_full_gpu: bool,
    extreme_staging: bool,
    route_history_resident: bool,
) -> bool:
    """Choose prefetch only for pools that cannot retain the RAM archive.

    The regular packed hybrid owns one global signature-partitioned VRAM
    arena.  Prefetching an entire previous-token route competes with the
    current layer for those slots and was measured to reduce a 16 GiB launch
    from 5.43 to 2.82 token/s.  Demand staging already overlaps with the
    shared branch; reserve cross-layer prefetch for disk-backed pools and the
    explicit extreme staging layout.
    """

    del packed_device_pool, packed_full_gpu
    return (
        not resident_all
        or (extreme_staging and not route_history_resident)
    )


@dataclass(frozen=True)
class DSV4LayerKVSnapshot:
    """Copied mutable state for one DSV4 layer at a stable prompt boundary."""

    kv: torch.Tensor
    win_pos: torch.Tensor
    compressed_length: int | None
    ckv: torch.Tensor | None
    cscore: torch.Tensor | None
    indexer_length: int | None
    indexer_ckv: torch.Tensor | None
    indexer_cscore: torch.Tensor | None


@dataclass(frozen=True)
class DSV4KVSnapshot:
    """One bounded rollback point; paged payloads remain append-only."""

    pos: int
    layers: tuple[DSV4LayerKVSnapshot, ...]

    @property
    def nbytes(self) -> int:
        tensors: list[torch.Tensor] = []
        for layer in self.layers:
            tensors.extend(
                tensor
                for tensor in (
                    layer.kv,
                    layer.win_pos,
                    layer.ckv,
                    layer.cscore,
                    layer.indexer_ckv,
                    layer.indexer_cscore,
                )
                if tensor is not None
            )
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
        )


def _prefill_sliding_window(
    ring_values: torch.Tensor,
    ring_positions: torch.Tensor,
    current_values: torch.Tensor,
    pos0: int,
    *,
    query_start: int = 0,
    query_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact linear-memory raw-KV windows for block prefill.

    The returned view is ``[B,T,window,D]``.  It contains the same causal
    entries as the decode ring for every query, without materialising a
    quadratic ``[T,T]`` score tensor when the outer block is 8192 tokens.
    """
    batch, window, _ = ring_values.shape
    tokens = current_values.shape[1]
    query_start = max(0, int(query_start))
    if query_count is None:
        query_count = tokens - query_start
    query_count = max(0, min(int(query_count), tokens - query_start))
    device = current_values.device
    positions = torch.arange(
        pos0 + query_start,
        pos0 + query_start + query_count,
        device=device,
    )
    offsets = torch.arange(
        window - 1,
        -1,
        -1,
        device=device,
        dtype=positions.dtype,
    )
    query_windows = positions.view(query_count, 1) - offsets.view(1, window)
    valid_positions = query_windows >= 0
    ring_slots = query_windows.remainder(window)
    prior = ring_values[:, ring_slots]
    prior_positions = ring_positions[:, ring_slots]
    prior_valid = prior_positions == query_windows.view(1, query_count, window)
    current_indices = (query_windows - pos0).clamp(0, tokens - 1)
    current = current_values[:, current_indices]
    current_valid = query_windows >= pos0
    values = torch.where(
        current_valid.view(1, query_count, window, 1),
        current,
        prior,
    )
    valid = valid_positions.view(1, query_count, window) & (
        current_valid.view(1, query_count, window) | prior_valid
    )
    if values.shape[:3] != (batch, query_count, window):
        raise RuntimeError("invalid block-prefill sliding-window shape")
    return values, valid


def _flashmla_prefill_kv_and_indices(
    ring_values: torch.Tensor,
    ring_positions: torch.Tensor,
    current_values: torch.Tensor,
    pos0: int,
    compressed_values: torch.Tensor | None,
    selected_positions: torch.Tensor | None,
    selected_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish exact sparse-Prefill KV rows without a gathered value tensor.

    FlashMLA consumes one compact KV bank and an int32 index list per query.
    The first bank segment keeps the fixed sliding-window ring, the second the
    current causal block, and the optional third the compressed history.  This
    replaces the old ``[T, window/top-k, D]`` gathers and their four einsums.
    """

    if ring_values.ndim != 3 or current_values.ndim != 3:
        raise ValueError("FlashMLA Prefill expects [B,T,D] KV tensors")
    batch, window, width = ring_values.shape
    if batch != 1 or current_values.shape[0] != 1:
        raise ValueError("FlashMLA sparse Prefill currently requires batch=1")
    tokens = int(current_values.shape[1])
    if tokens <= 0 or int(current_values.shape[2]) != int(width):
        raise ValueError("FlashMLA Prefill received an invalid current block")
    device = current_values.device
    query_positions = torch.arange(
        int(pos0), int(pos0) + tokens, device=device, dtype=torch.long
    )
    offsets = torch.arange(
        window - 1, -1, -1, device=device, dtype=torch.long
    )
    key_positions = query_positions[:, None] - offsets[None, :]
    current_mask = key_positions >= int(pos0)
    current_indices = window + key_positions - int(pos0)
    ring_slots = key_positions.remainder(window)
    ring_valid = ring_positions[0].index_select(
        0, ring_slots.reshape(-1)
    ).view(tokens, window) == key_positions
    raw_valid = (
        (key_positions >= 0)
        & (
            current_mask
            | ring_valid
        )
    )
    raw_indices = torch.where(
        current_mask,
        current_indices,
        ring_slots,
    )
    raw_indices = torch.where(
        raw_valid,
        raw_indices,
        torch.full_like(raw_indices, -1),
    )

    banks = [
        ring_values[0].to(torch.bfloat16),
        current_values[0].to(torch.bfloat16),
    ]
    index_parts = [raw_indices]
    if compressed_values is not None:
        if compressed_values.ndim != 3 or compressed_values.shape[0] != 1:
            raise ValueError("compressed FlashMLA KV must be [1,S,D]")
        compressed_count = int(compressed_values.shape[1])
        banks.append(compressed_values[0].to(torch.bfloat16))
        if selected_positions is None:
            raise ValueError("compressed FlashMLA KV requires sparse indices")
        compressed_indices = selected_positions[0].long() + window + tokens
        valid = (
            (selected_positions[0] >= 0)
            & (selected_positions[0] < compressed_count)
        )
        if selected_valid is not None:
            valid &= selected_valid[0]
        compressed_indices = torch.where(
            valid,
            compressed_indices,
            torch.full_like(compressed_indices, -1),
        )
        index_parts.append(compressed_indices)
    kv_bank = torch.cat(banks, dim=0).unsqueeze(1).contiguous()
    indices = torch.cat(index_parts, dim=1).unsqueeze(1).to(torch.int32)
    return kv_bank, indices.contiguous()


def _tp1_token_graph_bucket(
    position: int,
    *,
    hip_runtime: bool = False,
) -> str:
    """Return the fixed-shape TokenGraph bucket for one decode position."""
    compressed_count = (max(0, int(position)) + 1) // 4
    # A normal 128-token answer crosses direct32 near its final token and used
    # to pay for a second 43-layer HIP capture immediately before completion.
    # direct128 still loops over the live compressed count in the fused kernel,
    # so selecting it up front avoids that pause without extra Attention math.
    if compressed_count <= 32 and not hip_runtime:
        return "direct32"
    if compressed_count <= 128:
        return "direct128"
    if compressed_count <= 512:
        return "direct512"
    return "topk512"


def _tp1_direct_cache_reserve_item(max_ctx: int, ratio: int) -> int:
    """Return the last compressed item read by TP1's direct graph buckets."""
    context = max(1, int(max_ctx))
    compression = int(ratio)
    if compression <= 0:
        raise ValueError("ratio must be positive")
    max_items = (context + compression - 1) // compression
    direct_items = 512 if compression == 4 else 16
    return min(max_items, direct_items) - 1


def _tp1_graph_cache_reserve_item(
    max_ctx: int,
    ratio: int,
    sparse_bucket: bool,
) -> int:
    """Return the last compressed item referenced by retained TP1 graphs."""
    compression = int(ratio)
    if bool(sparse_bucket) and compression != 4:
        context = max(1, int(max_ctx))
        return (context + compression - 1) // compression - 1
    return _tp1_direct_cache_reserve_item(max_ctx, compression)


def _dsv4_compressed_page_items(ratio: int) -> int:
    """Keep every direct TokenGraph compressed read inside one KV page."""
    compression = int(ratio)
    if compression <= 0:
        raise ValueError("ratio must be positive")
    return max(16, 4096 // compression)


def _requires_flashmla_splitkv(
    *,
    device_type: str,
    hip_runtime: bool,
    max_ctx: int,
) -> bool:
    """Whether this runtime can enter DSV4's sparse Top-K bucket."""
    return (
        str(device_type) == "cuda"
        and not bool(hip_runtime)
        and int(max_ctx) > 2051
    )


def _flashmla_prefill_batch_enabled(
    tokens: int,
    *,
    runner_available: bool,
) -> bool:
    """Use sparse SplitKV Prefill only for its validated long-batch range."""
    return int(tokens) >= 512 and bool(runner_available)


def _indexer_candidate_capacity(candidate_count: int) -> int:
    """Pad the FP8 Indexer matrix width to the scaled-GEMM alignment."""
    count = int(candidate_count)
    if count <= 0:
        raise ValueError("candidate_count must be positive")
    return (count + 15) // 16 * 16


def _dense_bf16_group(name: str) -> str | None:
    """Return the BF16 residency group for a dense Int4 weight, if eligible."""
    if name == "head.weight":
        return "head"
    if name == "embed.weight":
        return "embed"
    if ".ffn.shared_experts." in name:
        return "shared"
    if name.endswith("_fn"):
        return "hyper"
    if name == "norm.weight" or name.endswith(".attn_norm.weight") \
            or name.endswith(".ffn_norm.weight"):
        return "norm"
    if name.endswith(".q_norm.weight") or name.endswith(".kv_norm.weight"):
        return None
    if name.endswith(".norm.weight"):
        return None
    if name.endswith(".attn.attn_sink"):
        return None
    if ".attn.indexer." in name:
        return "indexer"
    if ".attn.compressor." in name:
        return "compressor"
    if ".attn." in name:
        return "attention"
    # Router and hyperconnection weights are explicitly consumed as FP32.
    return None


def _parse_dense_bf16(value: str | None = None) -> frozenset[str]:
    """Parse CCCP_DENSE_BF16 into the eligible dense residency groups."""
    raw = os.environ.get("CCCP_DENSE_BF16", "none") if value is None else value
    raw = raw.strip().lower()
    if raw in ("", "0", "false", "off", "none"):
        return frozenset()
    if raw in ("1", "true", "all"):
        return _DENSE_BF16_ELIGIBLE
    groups = {
        _DENSE_BF16_ALIASES.get(part.strip().replace("-", "_"),
                                part.strip().replace("-", "_"))
        for part in raw.split(",") if part.strip()
    }
    unknown = groups - _DENSE_BF16_ELIGIBLE
    if unknown:
        valid = ", ".join(sorted(_DENSE_BF16_ELIGIBLE))
        raise ValueError(
            f"unknown CCCP_DENSE_BF16 group(s): {', '.join(sorted(unknown))}; "
            f"valid groups: {valid}"
        )
    return frozenset(groups)


def _cccp_lin(x: torch.Tensor, w) -> torch.Tensor:
    """Route DSV4 projections through the shared batched dispatcher."""
    from .ops import linear_batch

    return linear_batch(
        x,
        w,
        output_dtype=(
            None if isinstance(w, torch.Tensor) else compute_dtype(x.device)
        ),
    )


from . import dsv4 as _dsv4
def _o_proj_cccp(o: torch.Tensor, w: dict, cfg) -> torch.Tensor:
    """分组 LoRA O 的实现：Int4Weight 走逐组 dequant_rows；bf16 常驻张量走原生路径
    （dtype 对齐）。数值与 CCCP.dsv4._o_proj 一致。"""
    wo_a = w["wo_a"]
    if not isinstance(wo_a, (Int4Weight, BlockFP8Weight)):
        if o.dtype != wo_a.dtype:
            o = o.to(wo_a.dtype)
        return _dsv4._o_proj(o, w, cfg)
    B, T = o.shape[0], o.shape[1]
    G = cfg.o_groups
    rank = cfg.o_lora_rank
    o = o.reshape(B, T, G, -1)
    if isinstance(wo_a, BlockFP8Weight):
        # Each group owns an aligned row range in the compact source tensor.
        # The public grouped-row operator reads all independent O groups in a
        # single persistent CPU team; inputs are not broadcast across groups.
        from .ops import linear_grouped_rows

        grouped = w.get("_cpu_wo_a_group")
        if grouped is None:
            grouped = w.setdefault(
                "_cpu_wo_a_group",
                ProjectionGroup(
                    tuple(
                        wo_a.row_view(
                            group * rank,
                            (group + 1) * rank,
                        )
                        for group in range(G)
                    )
                ),
            )
        projected = linear_grouped_rows(
            o.reshape(B * T * G, -1), grouped
        )
        if projected is not None:
            return _cccp_lin(projected.view(B, T, -1), w["wo_b"])
        groups = []
        for group, compact in enumerate(grouped.weights):
            value = o[:, :, group].reshape(-1, o.shape[-1])
            groups.append(
                compact.matmul_T_decode_fused(value).view(B, T, rank)
            )
        return _cccp_lin(torch.cat(groups, dim=-1), w["wo_b"])
    if not o.is_cuda and B * T == 1:
        wo_b = w["wo_b"]
        if isinstance(wo_b, Int4Weight):
            from .cpuext import o_proj_int4_cpu

            fused = o_proj_int4_cpu(
                o.reshape(G, -1),
                wo_a.q,
                wo_a.s,
                wo_a.cols,
                wo_a.gs,
                rank,
                wo_b.q,
                wo_b.s,
                wo_b.cols,
                wo_b.gs,
            )
            if fused is not None:
                return fused.view(B, T, -1)

        from .cpuext import int4_grouped_gemv_cpu

        grouped = int4_grouped_gemv_cpu(
            o.reshape(G, -1),
            wo_a.q,
            wo_a.s,
            wo_a.cols,
            wo_a.gs,
            rank,
        )
        if grouped is not None:
            return _cccp_lin(
                grouped.reshape(B, T, G * rank), w["wo_b"]
            )
    outs = []
    for g in range(G):
        wa_g = wo_a.dequant_rows(g * rank, (g + 1) * rank)  # [rank, D]（Int4Weight.half 时 fp16）
        og = o[:, :, g]
        outs.append((og.half() @ wa_g.t()).float() if wa_g.dtype != og.dtype else og @ wa_g.t())
    o = torch.stack(outs, dim=2)
    return _cccp_lin(o.flatten(2), w["wo_b"])


_dsv4._lin = _cccp_lin              # 线性层钩子：dsv4.py 全部线性层经此分派
_dsv4._o_proj_hook = _o_proj_cccp  # O 投影钩子安装（dsv4.py 的 attn 经此走 Int4 分组反量化）

# HC sinkhorn 融合钩子：20 轮 4×4 双随机归一化原本每轮 4 次小 kernel（每层 attn/ffn
# 两次调用，逐 token ~6500 次 launch），融合后一次 launch。无扩展/非 CUDA/f64 时
# 返回 None 回退原 torch 循环（dspark.py 的 hc_pre/hc_post 同享本钩子）。
from .fusedext import hc_split_fused as _hc_fused

_hc_split_orig = _dsv4.hc_split


def _hc_split_cccp(mixes, scale, base, hc, iters, eps):
    r = _hc_fused(mixes, scale, base, hc, iters, eps)
    if r is not None:
        return r
    return _hc_split_orig(mixes, scale, base, hc, iters, eps)


_dsv4.hc_split = _hc_split_cccp

# RMSNorm 融合钩子：pow/mean/rsqrt/两次乘 ~6 次 launch → 1 次（dsv4 每层 4+ 处）。
from .fusedext import rmsnorm_fused as _rms_fused

_rmsnorm_orig = _dsv4.rmsnorm


def _rmsnorm_cccp(x, w, eps):
    r = _rms_fused(x, w, eps)
    if r is not None:
        return r
    return _rmsnorm_orig(x, w, eps)


_dsv4.rmsnorm = _rmsnorm_cccp

# RoPE 融合钩子：decode 单相位（全部行同一 cos/sin）时 1 次 launch 替代 ~8 次
from .fusedext import rope1_fused as _rope_fused

_rope_orig = _dsv4.rope_apply


def _rope_cccp(x, cos, sin, inverse=False):
    r = _rope_fused(x, cos, sin, inverse)
    if r is not None:
        return r
    return _rope_orig(x, cos, sin, inverse=inverse)


_dsv4.rope_apply = _rope_cccp

from .fusedext import dsv4_attn_decode_fused as _attn_decode_fused

_dsv4._attn_decode_core_hook = _attn_decode_fused

from .fusedext import dsv4_hc_pre_fused as _hc_pre_fused
from .fusedext import dsv4_route_post_fused as _route_post_fused
from .ops import (
    hyper_connection_post as _hyper_connection_post,
    hyper_connection_post_moe as _hyper_connection_post_moe,
    hyper_connection_pre_norm as _hyper_connection_pre_norm,
)

_hc_pre_orig = _dsv4.hc_pre
_hc_post_orig = _dsv4.hc_post


def _hc_pre_cccp(x, fn, scale, base, cfg):
    r = _hc_pre_fused(
        x, fn, scale, base, cfg.hc_sinkhorn_iters, cfg.hc_eps
    )
    if r is not None:
        return r
    return _hc_pre_orig(x, fn, scale, base, cfg)


_dsv4.hc_pre = _hc_pre_cccp


def _hc_post_cccp(out, residual, post, comb, output=None):
    if not residual.is_cuda:
        from .cpuext import hc_post_cpu

        cpu_result = hc_post_cpu(
            out, residual, post, comb, output=output
        )
        if cpu_result is not None:
            return cpu_result
    r = _hyper_connection_post(
        out,
        residual,
        post,
        comb,
        output=output,
    )
    if r is not None:
        return r
    return _hc_post_orig(out, residual, post, comb)


_dsv4.hc_post = _hc_post_cccp


def _hc_pre_norm_cccp(
    x,
    fn,
    scale,
    base,
    norm,
    cfg,
    output_buffers=None,
):
    """HC pre 与随后 RMSNorm 的 BF16 热路径；归约仍在核内使用 FP32。"""
    if not x.is_cuda and x.shape[0] * x.shape[1] == 1:
        from .cpuext import hc_pre_norm_cpu

        xf = x.flatten(2).float()
        mixes = _cccp_lin(xf, fn).float()
        cpu_result = hc_pre_norm_cpu(
            x,
            mixes,
            scale,
            base,
            norm,
            cfg.hc_sinkhorn_iters,
            cfg.rms_eps,
            cfg.hc_eps,
            output_buffers=output_buffers,
        )
        if cpu_result is not None:
            return cpu_result
    r = _hyper_connection_pre_norm(
        x,
        fn,
        scale,
        base,
        norm,
        cfg.hc_sinkhorn_iters,
        cfg.rms_eps,
        output_buffers=output_buffers,
    )
    if r is not None:
        return r
    y, post, comb = _dsv4.hc_pre(x, fn, scale, base, cfg)
    y = _dsv4.rmsnorm(y, norm, cfg.rms_eps)
    dtype = compute_dtype(x.device)
    return y.to(dtype), post.to(dtype), comb.to(dtype)


def _linear(x: torch.Tensor, w) -> torch.Tensor:
    """dense 线性层：Int4Weight 走分块反量化（3D 输入压平再还原），其余按 dtype 对齐 matmul。"""
    if isinstance(w, (Int4Weight, BlockFP8Weight)):
        if x.dim() > 2:
            sh = x.shape
            rows = x.reshape(-1, sh[-1])
            out = (
                w.matmul_T_decode_fused(rows)
                if isinstance(w, BlockFP8Weight)
                else w.matmul_T(rows)
            ).view(*sh[:-1], -1)
        else:
            out = (
                w.matmul_T_decode_fused(x)
                if isinstance(w, BlockFP8Weight)
                else w.matmul_T(x)
            )
        return out.to(compute_dtype(x.device))
    if x.dtype != w.dtype:
        x = x.to(w.dtype)
    return x @ w.t()


def _qkv_cccp(x, w, cfg, cache, pos0, cpu_outputs=None):
    """CPU decode 将共享输入的 Q-rank 与 KV INT4 投影合并到一个并行区。"""
    qkv_group = w.get("qkv_projection_group")
    if cpu_outputs is not None:
        outputs = cpu_outputs
        B, T = x.shape[:2]
        H, hd, rd = (
            cfg.n_heads,
            cfg.head_dim,
            cfg.qk_rope_head_dim,
        )
        cos = cache.cos[pos0:pos0 + T]
        sin = cache.sin[pos0:pos0 + T]
        from .cpuext import q_post_cpu, qkv_pre_cpu

        preprocessed = qkv_pre_cpu(
            outputs[0],
            outputs[1],
            w["q_norm"],
            w["kv_norm"],
            cos,
            sin,
            cfg.rms_eps,
        )
        if preprocessed is not None:
            qr = preprocessed[0].view(B, T, -1)
            kv = preprocessed[1].view(B, T, -1)
            q = _cccp_lin(qr, w["wq_b"]).view(
                B, T, H, hd
            ).float()
            q = q_post_cpu(q, cos, sin, cfg.rms_eps)
            if q is not None:
                return qr, q, kv
        # Keep the exact reference fallback if a future dtype/layout is not
        # accepted by the common post-processing kernel.
        qr = _dsv4.rmsnorm(
            outputs[0].view(B, T, -1),
            w["q_norm"],
            cfg.rms_eps,
        )
        q = _cccp_lin(qr, w["wq_b"]).view(B, T, H, hd).float()
        q *= torch.rsqrt(
            q.square().mean(-1, keepdim=True) + cfg.rms_eps
        )
        q[..., hd - rd:] = _dsv4.rope_apply(
            q[..., hd - rd:],
            cos.view(1, T, 1, -1),
            sin.view(1, T, 1, -1),
        )
        kv = _dsv4.rmsnorm(
            outputs[1].view(B, T, -1),
            w["kv_norm"],
            cfg.rms_eps,
        )
        kv[..., hd - rd:] = _dsv4.rope_apply(
            kv[..., hd - rd:],
            cos.view(1, T, -1),
            sin.view(1, T, -1),
        )
        return qr, q, kv
    if (
        x.is_cuda
        and x.shape[0] * x.shape[1] == 1
        and isinstance(qkv_group, ProjectionGroup)
    ):
        B, T = x.shape[:2]
        q_rows = int(w["wq_a"].shape[0])
        projected = _cccp_lin(x, qkv_group)
        q_projected = projected[..., :q_rows]
        kv_projected = projected[..., q_rows:]
        H, hd, rd = cfg.n_heads, cfg.head_dim, cfg.qk_rope_head_dim
        T = x.shape[1]
        cos = cache.cos[pos0:pos0 + T]
        sin = cache.sin[pos0:pos0 + T]
        qr = _dsv4.rmsnorm(
            q_projected.float(),
            w["q_norm"],
            cfg.rms_eps,
        )
        from .ops import head_rmsnorm_rope

        q = _cccp_lin(qr, w["wq_b"]).view(B, T, H, hd).float()
        enable_head_norm_rope = os.environ.get(
            "CCCP_HEAD_NORM_ROPE", "1"
        ) != "0"
        q_fused = enable_head_norm_rope and head_rmsnorm_rope(
            q.reshape(-1, hd),
            w["q_head_norm"],
            cos.reshape(-1),
            sin.reshape(-1),
            rope_width=rd,
            eps=cfg.rms_eps,
        )
        if not q_fused:
            q *= torch.rsqrt(
                q.square().mean(-1, keepdim=True) + cfg.rms_eps
            )
            q[..., hd - rd:] = _dsv4.rope_apply(
                q[..., hd - rd:],
                cos.view(1, T, 1, -1),
                sin.view(1, T, 1, -1),
            )
        kv = kv_projected.float()
        kv_fused = enable_head_norm_rope and head_rmsnorm_rope(
            kv.reshape(-1, hd),
            w["kv_norm"],
            cos.reshape(-1),
            sin.reshape(-1),
            rope_width=rd,
            eps=cfg.rms_eps,
        )
        if not kv_fused:
            kv = _dsv4.rmsnorm(
                kv,
                w["kv_norm"],
                cfg.rms_eps,
            )
            kv[..., hd - rd:] = _dsv4.rope_apply(
                kv[..., hd - rd:],
                cos.view(1, T, -1),
                sin.view(1, T, -1),
            )
        return qr, q, kv
    if (
        not x.is_cuda
        and x.shape[0] * x.shape[1] == 1
        and os.environ.get("CCCP_CPU_ATTN_MANY", "1") != "0"
        and (
            (
                isinstance(w["wq_a"], Int4Weight)
                and isinstance(w["wkv"], Int4Weight)
                and w["wq_a"].gs == w["wkv"].gs
            )
            or (
                isinstance(w["wq_a"], BlockFP8Weight)
                and isinstance(w["wkv"], BlockFP8Weight)
            )
        )
    ):
        if isinstance(w["wq_a"], Int4Weight):
            from .cpuext import int4_gemv_many_cpu

            outputs = int4_gemv_many_cpu(
                x.flatten(0, 1),
                [w["wq_a"].q, w["wkv"].q],
                [w["wq_a"].s, w["wkv"].s],
                w["wq_a"].gs,
            )
        else:
            from .ops import linear

            group = w.get("_cpu_qkv_group")
            if group is None:
                group = w.setdefault(
                    "_cpu_qkv_group",
                    ProjectionGroup((w["wq_a"], w["wkv"])),
                )
            combined = linear(x.flatten(0, 1), group)
            q_rows = int(w["wq_a"].shape[0])
            outputs = (combined[:, :q_rows], combined[:, q_rows:])
        if outputs is not None:
            B, T = x.shape[:2]
            H, hd, rd = (
                cfg.n_heads,
                cfg.head_dim,
                cfg.qk_rope_head_dim,
            )
            cos = cache.cos[pos0:pos0 + T]
            sin = cache.sin[pos0:pos0 + T]
            if os.environ.get("CCCP_CPU_QKV_POST", "1") != "0":
                from .cpuext import q_post_cpu, qkv_pre_cpu

                preprocessed = qkv_pre_cpu(
                    outputs[0],
                    outputs[1],
                    w["q_norm"],
                    w["kv_norm"],
                    cos,
                    sin,
                    cfg.rms_eps,
                )
                if preprocessed is not None:
                    qr = preprocessed[0].view(B, T, -1)
                    kv = preprocessed[1].view(B, T, -1)
                    if isinstance(w["wq_b"], Int4Weight):
                        from .cpuext import q_int4_post_cpu

                        wq_b = w["wq_b"]
                        q = q_int4_post_cpu(
                            preprocessed[0],
                            wq_b.q,
                            wq_b.s,
                            wq_b.cols,
                            wq_b.gs,
                            cos,
                            sin,
                            H,
                            hd,
                            cfg.rms_eps,
                        )
                        if q is not None:
                            return qr, q, kv
                    q = _cccp_lin(qr, w["wq_b"]).view(
                        B, T, H, hd
                    ).float()
                    q = q_post_cpu(q, cos, sin, cfg.rms_eps)
                    if q is not None:
                        return qr, q, kv
            qr = _dsv4.rmsnorm(
                outputs[0].view(B, T, -1),
                w["q_norm"],
                cfg.rms_eps,
            )
            q = _cccp_lin(qr, w["wq_b"]).view(B, T, H, hd).float()
            q *= torch.rsqrt(
                q.square().mean(-1, keepdim=True) + cfg.rms_eps
            )
            q[..., hd - rd:] = _dsv4.rope_apply(
                q[..., hd - rd:],
                cos.view(1, T, 1, -1),
                sin.view(1, T, 1, -1),
            )
            kv = _dsv4.rmsnorm(
                outputs[1].view(B, T, -1),
                w["kv_norm"],
                cfg.rms_eps,
            )
            kv[..., hd - rd:] = _dsv4.rope_apply(
                kv[..., hd - rd:],
                cos.view(1, T, -1),
                sin.view(1, T, -1),
            )
            return qr, q, kv
    return _dsv4._qkv(x, w, cfg, cache, pos0)


def _compressor_decode_projected(
    x,
    w,
    outputs,
    ratio,
    d,
    rd,
    cos,
    sin,
    eps,
    st,
    pos,
    *,
    state_updated: bool = False,
):
    """Shared state update after a registered grouped projection."""
    B, T = x.shape[:2]
    kv = outputs[0].view(B, T, -1)
    score = outputs[1].view(B, T, -1)
    if not state_updated:
        score = score + w["ape"][pos % ratio]
    coff = kv.shape[-1] // d
    overlap = coff == 2
    should_pool = (pos + 1) % ratio == 0
    if overlap:
        if not state_updated:
            st["ckv"][:, ratio + pos % ratio] = kv[:, 0]
            st["cscore"][:, ratio + pos % ratio] = score[:, 0]
        if not should_pool:
            return None
        kvs = torch.cat(
            [st["ckv"][:, :ratio, :d], st["ckv"][:, ratio:, d:]],
            dim=1,
        )
        scores = torch.cat(
            [
                st["cscore"][:, :ratio, :d],
                st["cscore"][:, ratio:, d:],
            ],
            dim=1,
        )
        probs = scores.float().softmax(dim=1)
        pooled = (kvs.float() * probs).sum(dim=1, keepdim=True)
        st["ckv"][:, :ratio] = st["ckv"][:, ratio:].clone()
        st["cscore"][:, :ratio] = st["cscore"][:, ratio:].clone()
    else:
        if not state_updated:
            st["ckv"][:, pos % ratio] = kv[:, 0]
            st["cscore"][:, pos % ratio] = score[:, 0]
        if not should_pool:
            return None
        probs = st["cscore"].float().softmax(dim=1)
        pooled = (st["ckv"].float() * probs).sum(dim=1, keepdim=True)
    pooled = rmsnorm(pooled, w["norm"], eps)
    pooled[..., d - rd:] = _dsv4.rope_apply(
        pooled[..., d - rd:], cos, sin
    )
    return pooled


def _compressor_decode_cccp(
    x,
    w,
    ratio,
    d,
    rd,
    cos,
    sin,
    eps,
    st,
    pos,
    cpu_outputs=None,
):
    if x.ndim != 3 or int(x.shape[0]) * int(x.shape[1]) != 1:
        raise RuntimeError(
            "compressor_decode is decode-only; multi-token Prefill must use "
            "the batched compressor projection"
        )
    """CPU decode 将 Compressor 的 KV/Gate INT4 投影合并。"""
    if cpu_outputs is not None:
        return _compressor_decode_projected(
            x,
            w,
            cpu_outputs,
            ratio,
            d,
            rd,
            cos,
            sin,
            eps,
            st,
            pos,
        )
    projection_group = w.get("projection_group")
    if (
        x.is_cuda
        and x.shape[0] * x.shape[1] == 1
        and isinstance(projection_group, ProjectionGroup)
    ):
        projected = _cccp_lin(x, projection_group)
        kv_rows = int(w["wkv"].shape[0])
        from .ops import compressed_state_update

        state_updated = (
            os.environ.get("CCCP_COMPRESSED_STATE_UPDATE", "1") != "0"
            and compressed_state_update(
            projected.reshape(1, -1),
            w["ape"],
            st["ckv"],
            st["cscore"],
            ratio=ratio,
            position=pos,
            kv_rows=kv_rows,
            )
        )
        outputs = (
            projected[..., :kv_rows].reshape(-1, kv_rows),
            projected[..., kv_rows:].reshape(
                -1, int(w["wgate"].shape[0])
            ),
        )
        return _compressor_decode_projected(
            x,
            w,
            outputs,
            ratio,
            d,
            rd,
            cos,
            sin,
            eps,
            st,
            pos,
            state_updated=state_updated,
        )
    if (
        not x.is_cuda
        and x.shape[0] * x.shape[1] == 1
        and os.environ.get("CCCP_CPU_ATTN_MANY", "1") != "0"
        and (
            (
                isinstance(w["wkv"], Int4Weight)
                and isinstance(w["wgate"], Int4Weight)
                and w["wkv"].gs == w["wgate"].gs
            )
            or (
                isinstance(w["wkv"], BlockFP8Weight)
                and isinstance(w["wgate"], BlockFP8Weight)
            )
        )
    ):
        if isinstance(w["wkv"], Int4Weight):
            from .cpuext import int4_gemv_many_cpu

            outputs = int4_gemv_many_cpu(
                x.flatten(0, 1),
                [w["wkv"].q, w["wgate"].q],
                [w["wkv"].s, w["wgate"].s],
                w["wkv"].gs,
            )
        else:
            from .ops import linear

            group = w.get("_cpu_compressor_group")
            if group is None:
                group = w.setdefault(
                    "_cpu_compressor_group",
                    ProjectionGroup((w["wkv"], w["wgate"])),
                )
            combined = linear(x.flatten(0, 1), group)
            kv_rows = int(w["wkv"].shape[0])
            outputs = (combined[:, :kv_rows], combined[:, kv_rows:])
        if outputs is not None:
            return _compressor_decode_projected(
                x, w, outputs, ratio, d, rd, cos, sin, eps, st, pos
            )
    return _dsv4.compressor_decode(
        x, w, ratio, d, rd, cos, sin, eps, st, pos
    )


def _compressor_prefill_cccp(
    x: torch.Tensor,
    w: dict,
    ratio: int,
    d: int,
    rd: int,
    cache,
    eps: float,
    st: dict,
    pos0: int,
    *,
    capture_steps: bool = False,
) -> tuple[torch.Tensor | None, int | None, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Project a Prefill block once, then update its ordered compressor state.

    Multi-token Prefill must never call the decode projection once per token.
    Unaligned KV-reuse prefixes and speculative snapshots consume slices of
    the already-projected batch, so they preserve state order without GEMV.
    """
    if x.ndim != 3 or int(x.shape[1]) <= 1:
        raise RuntimeError(
            "batched compressor Prefill requires more than one token"
        )
    if ratio <= 0 or d <= 0 or rd < 0 or rd > d:
        raise ValueError("invalid compressor dimensions")

    tokens = int(x.shape[1])
    projection = w.get("projection_group")
    if isinstance(projection, ProjectionGroup) and tokens <= 16:
        combined = _cccp_lin(x, projection)
        kv_rows = int(w["wkv"].shape[0])
        kv = combined[..., :kv_rows]
        score = combined[..., kv_rows:]
    else:
        # ProjectionGroup's CUDA specialization targets short decode/spec
        # batches. Long Prefill uses two full GEMMs on compact matrices.
        kv = _cccp_lin(x, w["wkv"])
        score = _cccp_lin(x, w["wgate"])

    def projected_step(j: int) -> torch.Tensor | None:
        position = int(pos0) + j
        rope_pos = max(0, position + 1 - ratio)
        return _compressor_decode_projected(
            x[:, j:j + 1],
            w,
            (kv[:, j:j + 1], score[:, j:j + 1]),
            ratio,
            d,
            rd,
            cache.cos[rope_pos].view(1, 1, -1),
            cache.sin[rope_pos].view(1, 1, -1),
            eps,
            st,
            position,
        )

    first_completion = int(pos0) + (
        (ratio - 1 - (int(pos0) % ratio)) % ratio
    )
    first_item = (
        first_completion // ratio
        if first_completion < int(pos0) + tokens
        else None
    )
    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    pooled_parts: list[torch.Tensor] = []

    if capture_steps:
        for j in range(tokens):
            pooled = projected_step(j)
            if pooled is not None:
                pooled_parts.append(pooled)
            snapshots.append((st["ckv"].clone(), st["cscore"].clone()))
        return (
            torch.cat(pooled_parts, dim=1) if pooled_parts else None,
            first_item,
            snapshots,
        )

    # Reach an absolute compressor boundary with already-projected slices.
    # At most ratio-1 state updates are needed; the long body stays batched.
    prefix = 0
    offset = int(pos0) % ratio
    if offset:
        prefix = min(tokens, ratio - offset)
        for j in range(prefix):
            pooled = projected_step(j)
            if pooled is not None:
                pooled_parts.append(pooled)

    remaining = tokens - prefix
    full_groups = remaining // ratio
    body_stop = prefix + full_groups * ratio
    coff = int(kv.shape[-1]) // d
    overlap = coff == 2
    if full_groups:
        kv_groups = kv[:, prefix:body_stop].unflatten(
            1, (full_groups, ratio)
        )
        score_groups = score[:, prefix:body_stop].unflatten(
            1, (full_groups, ratio)
        ) + w["ape"].view(1, 1, ratio, -1)
        if overlap:
            previous_kv = torch.cat(
                (
                    st["ckv"][:, :ratio, :d].unsqueeze(1),
                    kv_groups[:, :-1, :, :d],
                ),
                dim=1,
            )
            previous_score = torch.cat(
                (
                    st["cscore"][:, :ratio, :d].unsqueeze(1),
                    score_groups[:, :-1, :, :d],
                ),
                dim=1,
            )
            pool_kv = torch.cat(
                (previous_kv, kv_groups[..., d:]), dim=2
            )
            pool_score = torch.cat(
                (previous_score, score_groups[..., d:]), dim=2
            )
        else:
            pool_kv = kv_groups
            pool_score = score_groups
        probabilities = pool_score.float().softmax(dim=2)
        pooled = (pool_kv.float() * probabilities).sum(dim=2)
        pooled = rmsnorm(pooled, w["norm"], eps)
        body_pos0 = int(pos0) + prefix
        rope_positions = torch.arange(
            body_pos0,
            body_pos0 + full_groups * ratio,
            ratio,
            device=x.device,
        )
        pooled[..., d - rd:] = _dsv4.rope_apply(
            pooled[..., d - rd:],
            cache.cos[rope_positions].view(1, full_groups, -1),
            cache.sin[rope_positions].view(1, full_groups, -1),
        )
        pooled_parts.append(pooled)
        if overlap:
            st["ckv"][:, :ratio].copy_(kv_groups[:, -1])
            st["cscore"][:, :ratio].copy_(score_groups[:, -1])
            # The decode state machine leaves the completed current group in
            # its write half as well, then overwrites it row by row. Preserve
            # that exact state so speculative snapshots and KV reuse agree.
            st["ckv"][:, ratio:].copy_(kv_groups[:, -1])
            st["cscore"][:, ratio:].copy_(score_groups[:, -1])

    tail = remaining - full_groups * ratio
    if tail:
        tail_kv = kv[:, body_stop:body_stop + tail]
        tail_score = score[:, body_stop:body_stop + tail] + w["ape"][:tail]
        state_offset = ratio if overlap else 0
        st["ckv"][:, state_offset:state_offset + tail].copy_(tail_kv)
        st["cscore"][:, state_offset:state_offset + tail].copy_(tail_score)

    return (
        torch.cat(pooled_parts, dim=1) if pooled_parts else None,
        first_item,
        snapshots,
    )


def _shared_expert_mlp_cccp(x, w, limit):
    """Group compact shared Gate/Up through one public projection."""
    if (
        x.shape[0] == 1
        and isinstance(w["sh_w1"], BlockFP8Weight)
        and isinstance(w["sh_w3"], BlockFP8Weight)
        and isinstance(w["sh_w2"], BlockFP8Weight)
    ):
        from .ops import linear

        group = w.get("_shared_gu_group")
        if group is None:
            group = w.setdefault(
                "_shared_gu_group",
                ProjectionGroup((w["sh_w1"], w["sh_w3"])),
            )
        gu = linear(x, group)
        intermediate = int(w["sh_w1"].shape[0])
        gate, up = gu[:, :intermediate], gu[:, intermediate:]
        if limit > 0:
            up = up.clamp(min=-limit, max=limit)
            gate = gate.clamp(max=limit)
        return linear(F.silu(gate) * up, w["sh_w2"])
    return _dsv4.expert_mlp(
        x,
        w["sh_w1"],
        w["sh_w3"],
        w["sh_w2"],
        limit,
    )


def _dsv4_routed_topology(store, tp_size: int):
    """Return only the DSV4 layer/rank map consumed by the public runtime."""
    from .ops import build_routed_vq_topology_plan

    return build_routed_vq_topology_plan(
        store,
        int(tp_size),
        primary_dense=True,
    )


class DSV4CCCPModel:
    """DeepSeek-V4 CCCP 产物的推理模型（CPU/CUDA，内存显存自动适配由外层 Engine 定）。"""

    def __init__(self, root: str, cache_gb: float = 16.0, max_ctx: int = 2048,
                 device: str = "cpu", vram_cache_gb: float = 4.0,
                 tp_size: int = 1,
                 extreme_fixed_gpu_bytes: int = 0):
        self.tp_size = int(tp_size)
        if self.tp_size <= 0:
            raise ValueError("tp_size must be positive")
        requested = torch.device(device)
        if requested.type == "cuda" and requested.index is None:
            # CUDA_VISIBLE_DEVICES already maps the requested physical card.
            # Public fixed-address/TP operators need a canonical logical
            # index even for TP=1 so graph streams and events bind cuda:0.
            requested = torch.device("cuda", torch.cuda.current_device())
        if requested.type == "cuda" and self.tp_size > 1:
            if self.tp_size > torch.cuda.device_count():
                raise ValueError(
                    f"tp={self.tp_size} exceeds visible CUDA devices"
                )
            self.devices = tuple(
                torch.device("cuda", rank)
                for rank in range(self.tp_size)
            )
            self.device = self.devices[0]
        else:
            self.device = requested
            self.devices = (self.device,)
        self._cpu_numa_interleaved = False
        self._cpu_threads = 0
        if self.device.type == "cpu":
            # RAM/GPU 预设可能携带 BF16；CPU 单 Token GEMV 与融合 HC 的
            # 已验证热路径是 FP32，除非显式开启实验开关，否则在模型构造前纠正。
            if os.environ.get("CCCP_CPU_BF16", "0") != "1":
                os.environ["CCCP_COMPUTE_DTYPE"] = "fp32"
                os.environ["CCCP_DENSE_BF16"] = "none"
            from .cpuext import (
                configure_cpu_threads,
                configure_numa_interleave,
            )

            self._cpu_threads = configure_cpu_threads()
            self._cpu_numa_interleaved = configure_numa_interleave()
        self.store = CCCPStore(root)
        from .ops import ModelOperatorConfig

        self.operator_config = ModelOperatorConfig.from_manifest(
            {
                "model_family": (
                    self.store.man.model_family or "deepseek"
                ),
                "config": self.store.cfg,
            }
        )
        self.packed_operator_name: str | None = None
        if self.store.man.projection_vq:
            from .ops import packed_moe_operator_name

            capabilities = {
                tuple(
                    sorted(
                        capability.items()
                    )
                )
                for layer in self.store.man.expert_files
                for capability in (
                    self.store.man.projection_operator_capabilities(layer)
                )
            }
            names = set()
            for capability_items in capabilities:
                capability = dict(capability_items)
                names.add(
                    packed_moe_operator_name(
                        device_type=self.device.type,
                        activation=(
                            self.operator_config.expert_activation
                        ),
                        top_k=self.operator_config.top_k,
                        **capability,
                    )
                )
            self.packed_operator_name = ",".join(sorted(names))
            print(
                "[cccp] 公共 packed MoE="
                f"{self.packed_operator_name}；"
                f"activation={self.operator_config.expert_activation}；"
                "projection_fused="
                f"{os.environ.get('CCCP_PROJECTION_FUSED', '1')}",
                flush=True,
            )
        self.cfg = self.store.cfg  # dict（DSV4Config.to_json）
        gpu = self.device.type != "cpu"
        self._single_gpu_layer_graph_requested = bool(
            gpu
            and self.tp_size == 1
            and os.environ.get("CCCP_SINGLE_GPU_LAYER_GRAPH", "0") != "0"
        )
        if (
            gpu
            and torch.version.hip is not None
            and self.tp_size == 1
            and os.environ.get("CCCP_PACKED_FULL_GPU", "0") == "1"
        ):
            # Resolve the Windows HIP/full-resident execution mode before the
            # packed pool is constructed.  Setting this after construction is
            # too late: the pool has already selected its pipeline/tensor
            # scheduling topology, which can leave every expert in VRAM while
            # still dispatching 43 layers from Python.  Full-resident HIP has
            # stable expert addresses, so the TP1 parent graph is mandatory.
            self._single_gpu_layer_graph_requested = True
            os.environ["CCCP_SINGLE_GPU_LAYER_GRAPH"] = "1"
            os.environ["CCCP_TOKEN_GRAPH"] = "1"
        self._single_gpu_layer_graph = False
        if not self.store.man.projection_vq and self.tp_size != 1:
            raise ValueError(
                "legacy DeepSeek-V4 archives support only tp=1"
            )
        from .ops import create_routed_vq_runtime

        codebook_runtime = create_routed_vq_runtime(
            self.store,
            device=self.device,
            devices=self.devices,
            tp_size=self.tp_size,
            cache_gb=cache_gb,
            vram_cache_gb=vram_cache_gb,
            startup_gpu_reserve_bytes=extreme_fixed_gpu_bytes,
            topology_plan_factory=_dsv4_routed_topology,
            layer_graph_requested=self._single_gpu_layer_graph_requested,
        )
        self.routed_vq = codebook_runtime.executor
        self._packed_device_pool = (
            codebook_runtime.plan.packed_device_pool
        )
        self._packed_full_gpu = codebook_runtime.plan.packed_full_gpu
        self._hybrid_fixed_token_graph = bool(
            self._single_gpu_layer_graph_requested
            and not self._packed_full_gpu
            and self.routed_vq.fixed_token_graph_candidate
        )
        self._single_gpu_layer_graph = bool(
            self._single_gpu_layer_graph_requested
            and (
                self._packed_full_gpu
                or self._hybrid_fixed_token_graph
            )
        )
        if (
            torch.version.hip is not None
            and self._packed_full_gpu
            and self.tp_size == 1
        ):
            if self.routed_vq.parallelism != "tensor":
                raise RuntimeError(
                    "AMD/HIP full-resident pool must be initialized in "
                    "TP1 tensor mode"
                )
            print(
                "[cccp-amd-plan] residency=GPU-only；pool=TP1-tensor；"
                "decode=TokenGraph-required；eager-layer-fallback=forbidden",
                flush=True,
            )
        self._single_gpu_static_graphs = bool(
            gpu
            and self.tp_size == 1
            and self.store.man.projection_vq
            and os.environ.get("CCCP_STATIC_DECODE_GRAPHS", "1") != "0"
        )
        if self._single_gpu_layer_graph:
            # A one-rank graph is still the same public all-rank dataflow; it
            # merely has no collective peer. Keep fixed TPHidden addresses
            # across layers instead of adding a model-private graph.
            os.environ.setdefault("CCCP_TP_HIDDEN", "1")
            os.environ.setdefault("CCCP_TP_NO_OWNER", "1")
            os.environ.setdefault("CCCP_TP_GRAPH", "1")
            os.environ.setdefault("CCCP_TP_LAYER_GRAPH", "1")
        elif self._single_gpu_layer_graph_requested:
            print(
                "[cccp] packed pool lacks the fixed device-cache graph "
                "capability; TP1 TokenGraph is unavailable",
                flush=True,
            )
        # Benchmark/API diagnostics must distinguish a real all-rank packed
        # pool from the legacy or RAM paths.  Multi-card construction has no
        # silent fallback: preload failure aborts startup before this value can
        # be reported as a successful run.
        self.effective_tp_size = (
            self.tp_size if self._packed_full_gpu else 1
        )
        self.max_ctx = max_ctx
        self._w: dict[str, object] = {}
        self._layers: dict[int, dict] = {}
        self._dense_bf16 = _parse_dense_bf16()
        self._native_tensor_fp8_weights = 0
        self._hc_decode_workspaces: dict[
            tuple[torch.device, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._hc_post_workspaces: dict[
            tuple[torch.device, int],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._prefetch_auto = True
        self._prev_ids: dict[int, list[int]] = {}   # 层 → 上一 token 路由专家（预取用）
        self._profile_enabled = False
        self._profile_records: list[
            tuple[int, tuple[tuple[str, object], ...]]
        ] = []
        self._moe_profile_records: list[
            tuple[int, tuple[tuple[str, object], ...]]
        ] = []
        self.last_layer_profile: dict[str, object] = {}
        self._cpu_resident_experts: dict[
            int, tuple[tuple[VQWeight, VQWeight] | None, ...]
        ] = {}
        self._cpu_moe_layers: dict[int, object] = {}
        self._cpu_fused_resident_moe: dict[int, object | bool] = {}
        self._cpu_head_group = None
        self._tp_shared_mlp = None
        self._heterogeneous_shared_mlp = None
        self._tp_router = None
        self._tp_moe_finalizer = None
        self._tp_collective = None
        self._tp_attention_parallel = None
        self._tp_attention_contexts: tuple[list[dict], ...] | None = None
        self._tp_attention_projection_graphs = None
        self._tp_attention_projection_batches: dict[int, object] = {}
        self._tp_attention_projection_input_events: dict[
            int, tuple[torch.cuda.Event, ...]
        ] = {}
        self._tp_attention_projection_done_events: dict[
            int, tuple[torch.cuda.Event, ...]
        ] = {}
        self._tp_attention_projection_dependencies: list[object] = []
        self._single_gpu_attention_graphs: dict[int, dict] = {}
        self._single_gpu_ffn_graphs: dict[int, dict] = {}
        self._tp_route_weights: tuple[list[dict], ...] | None = None
        self._tp_route_buffers: dict[
            int,
            tuple[
                tuple[torch.Tensor, ...],
                tuple[torch.Tensor, ...],
                tuple[torch.Tensor, ...],
            ],
        ] = {}
        self._tp_route_events: dict[
            int, tuple[torch.cuda.Event, ...]
        ] = {}
        self._tp_route_packed_plan = None
        self._tp_decode_input = None
        self._tp_attention_hidden: dict[int, object] = {}
        self._tp_attention_partials: dict[int, object] = {}
        self._tp_ffn_prefix_hidden: dict[int, object] = {}
        self._tp_layer_output_hidden: dict[int, object] = {}
        self._tp_token_ids: tuple[torch.Tensor, ...] = ()
        self._tp_decode_controls: tuple[object, ...] = ()
        self._tp1_decode_control = None
        self._tp_attention_controlled_batches: dict[
            str, dict[int, object]
        ] = {}
        self._tp_attention_controlled_aux: dict[
            str, dict[int, tuple[tuple[torch.Tensor, ...], ...]]
        ] = {}
        self._tp_attention_controlled_dependencies: list[object] = []
        self._tp_attention_controlled_build_error: str | None = None
        self._tp_hc_layer_plans: dict[str, dict[int, object]] = {}
        self._tp_hc_layer_plan_errors: dict[str, str] = {}
        self._tp_attention_control_ready_events: tuple[
            torch.cuda.Event, ...
        ] = ()
        self._tp_attention_control_rope_rows: tuple[dict, ...] = ()
        self._tp1_token_graphs: dict[str, object] = {}
        self._tp1_token_logits: dict[str, torch.Tensor] = {}
        self._tp1_graph_dependencies: list[object] = []
        self._tp1_graph_build_error: str | None = None
        self._tp1_pool_graph_generation = -1
        self.tp_token_graph_info: dict[str, object] = {}
        self._tp_states_ready = False
        self._tp_stage_profile: dict[str, object] = {}
        self.tp_dataflow = "single"
        self.tp_collectives_per_layer = 0
        self.states: list[dict] | None = None
        c = self.cfg
        ratios = (list(c.get("compress_ratios") or []) + [0] * c["n_layers"])[: c["n_layers"]]
        self.ratios = ratios
        from .dsv4 import RopeCache  # 复用包内频率预计算（纯 torch 无依赖）
        rd = c["qk_rope_head_dim"]
        self.rope_base = RopeCache(rd, max_ctx + 8, c["rope_theta"], None)
        self.rope_cmp = RopeCache(rd, max_ctx + 8, c.get("compress_rope_theta", 160000.0),
                                  c.get("rope_scaling") or None)
        if gpu:
            for rc in (self.rope_base, self.rope_cmp):
                rc.cos = rc.cos.to(self.device)
                rc.sin = rc.sin.to(self.device)

    # ---- 权重访问 ----
    def w(self, name: str):
        wt = self._w.get(name)
        if wt is None:
            wt = self.store.get_dense(name)
            if (
                self.device.type == "cpu"
                and isinstance(wt, BlockFP8Weight)
            ):
                # Select the compact row-major/block-major32 representation
                # with the public, shape-based CPU tuner.  DSV4 supplies only
                # its real activation dtype; the layout and kernel remain
                # model-independent and never materialize BF16/FP32 weights.
                wt = wt.optimize_cpu_layout(
                    input_dtype=compute_dtype(self.device),
                )
            if self.device.type != "cpu":
                group = _dense_bf16_group(name)
                use_bf16 = group in self._dense_bf16
                if isinstance(wt, Int4Weight):
                    if use_bf16:
                        if compute_dtype(self.device) != torch.bfloat16:
                            raise RuntimeError(
                                "CCCP_DENSE_BF16 requires BF16 compute; "
                                "set CCCP_COMPUTE_DTYPE=bf16 on a supported GPU"
                            )
                        wt = wt.dequant_rows(0, wt.shape[0]).to(
                            self.device, dtype=torch.bfloat16
                        )
                    elif group == "hyper":
                        # HC consumes ``fn`` as a dense Tensor (and the fused
                        # path accepts FP32/BF16), not as an Int4Weight object.
                        # Keep the no-CCCP_DENSE_BF16 default functional by
                        # materializing this small matrix in FP32.
                        wt = wt.dequant_rows(0, wt.shape[0]).to(self.device)
                    elif name == "head.weight":
                        # lm_head 每 token 全量乘，常驻 f32（2.1GB，与 GLM 的 lm_head 策略一致）
                        wt = wt.dequant_rows(0, wt.shape[0]).to(self.device)
                    else:
                        # int4 dense GEMM 的 fp16 计算：默认关闭（CCCP_INT4_HALF=1 开启）。
                        # 实测 43 层残差+HC 放大使逐层 hidden rel 差达 1-3%（超 0.5% 门），
                        # 而内存受限卡上提速可忽略；KL 虽不变，从严回 f32。
                        wt = Int4Weight(wt.q.to(self.device), wt.s.to(self.device),
                                        wt.cols, wt.gs,
                                        half=os.environ.get("CCCP_INT4_HALF", "0") == "1")
                else:
                    native_fp8_mode = os.environ.get(
                        "CCCP_GPU_FP8_EXECUTION", "auto"
                    ).strip().lower()
                    # Grouped O-LoRA consumes independent input rows.  Its
                    # existing BF16 grouped operator is already a single
                    # launch, whereas splitting it into scalar scaled-MM
                    # calls would regress.  Keep that capability on BF16
                    # until the public grouped tensor-FP8 row kernel exists.
                    grouped_o_projection = ".attn.wo_a" in name
                    native_fp8_candidate = (
                        isinstance(wt, BlockFP8Weight)
                        and not grouped_o_projection
                    )
                    native_fp8_enabled = (
                        native_fp8_candidate
                        and _resolve_native_tensor_fp8_execution(
                            native_fp8_mode,
                            available=BlockFP8Weight.native_tensor_fp8_available(
                                self.device
                            ),
                        )
                    )
                    if native_fp8_enabled:
                        compact = wt.to(self.device)
                        compiled = compact.compile_gpu_tensor_fp8()
                        if compiled.layout != "tensor-fp8":
                            raise RuntimeError(
                                "native Tensor-FP8 compile returned "
                                f"unexpected layout {compiled.layout!r}"
                            )
                        wt = compiled
                        self._native_tensor_fp8_weights += 1
                    else:
                        wt = wt.to(
                            self.device, dtype=torch.bfloat16
                        ) if use_bf16 else wt.to(self.device)
            self._w[name] = wt
        return wt

    def _prefetch_enabled(self) -> bool:
        if not self.routed_vq.speculative_prefetch:
            return False
        raw = os.environ.get("CCCP_PREFETCH", "auto").strip().lower()
        if raw in ("", "auto"):
            return self._prefetch_auto
        return raw not in ("0", "false", "off", "no")

    def _token_prefetch_enabled(self) -> bool:
        """整轮预取只适用于按层拥有独立缓存容量的专家池。

        极限模式只有一个可复用 Top-K staging 组；整轮预取会让后层覆盖前层，
        因此只在每层 Attention 开始时预取该层上一 token 的路由专家。
        """

        return self._prefetch_enabled() and not bool(
            self.routed_vq.layer_prefetch_only
        )

    def layer(self, i: int) -> dict:
        """一层 dense 权重（attn/hc/norm/gate/compressor/shared），按键名惰性组装。"""
        w = self._layers.get(i)
        if w is not None:
            return w
        p = f"layers.{i}"
        w = {
            "wq_a": self.w(f"{p}.attn.wq_a.weight"),
            "q_norm": self.w(f"{p}.attn.q_norm.weight"),
            "wq_b": self.w(f"{p}.attn.wq_b.weight"),
            "wkv": self.w(f"{p}.attn.wkv.weight"),
            "kv_norm": self.w(f"{p}.attn.kv_norm.weight"),
            "attn_sink": self.w(f"{p}.attn.attn_sink"),
            "wo_a": self.w(f"{p}.attn.wo_a.weight"),
            "wo_b": self.w(f"{p}.attn.wo_b.weight"),
            "attn_norm": self.w(f"{p}.attn_norm.weight"),
            "ffn_norm": self.w(f"{p}.ffn_norm.weight"),
            "gate": (
                self.w(f"{p}.ffn.gate.weight")
                if self.device.type == "cpu"
                else _f32(self.w(f"{p}.ffn.gate.weight"))
            ),
            "sh_w1": self.w(f"{p}.ffn.shared_experts.w1.weight"),
            "sh_w3": self.w(f"{p}.ffn.shared_experts.w3.weight"),
            "sh_w2": self.w(f"{p}.ffn.shared_experts.w2.weight"),
            "hc_attn_fn": self.w(f"{p}.hc_attn_fn"),
            "hc_attn_base": self.w(f"{p}.hc_attn_base"),
            "hc_attn_scale": self.w(f"{p}.hc_attn_scale"),
            "hc_ffn_fn": self.w(f"{p}.hc_ffn_fn"),
            "hc_ffn_base": self.w(f"{p}.hc_ffn_base"),
            "hc_ffn_scale": self.w(f"{p}.hc_ffn_scale"),
        }
        if self.store.has(f"{p}.attn.compressor.wkv.weight"):
            w["cmp"] = {
                "wkv": self.w(f"{p}.attn.compressor.wkv.weight"),
                "wgate": self.w(f"{p}.attn.compressor.wgate.weight"),
                "ape": _f32(self.w(f"{p}.attn.compressor.ape")),  # 需下标切片，须 f32
                "norm": self.w(f"{p}.attn.compressor.norm.weight"),
            }
        if self.ratios[i] == 4:
            w["indexer"] = {
                "wq_b": self.w(f"{p}.attn.indexer.wq_b.weight"),
                "weights_proj": self.w(
                    f"{p}.attn.indexer.weights_proj.weight"
                ),
                "wkv": self.w(
                    f"{p}.attn.indexer.compressor.wkv.weight"
                ),
                "wgate": self.w(
                    f"{p}.attn.indexer.compressor.wgate.weight"
                ),
                "ape": _f32(
                    self.w(f"{p}.attn.indexer.compressor.ape")
                ),
                "norm": self.w(
                    f"{p}.attn.indexer.compressor.norm.weight"
                ),
            }
        if self.store.has(f"{p}.ffn.gate.bias"):
            w["gate_bias"] = self.w(f"{p}.ffn.gate.bias")
        if self.store.has(f"{p}.ffn.gate.tid2eid"):
            w["tid2eid"] = self.w(f"{p}.ffn.gate.tid2eid").long()
        self._layers[i] = w
        return w

    # ---- 前向（数值与 CCCP/dsv4.py 一致；线性层经 _linear 分派） ----
    def _rope(self, i: int):
        return self.rope_cmp if self.ratios[i] else self.rope_base

    def _alloc(self, B: int) -> None:
        self.states = self._allocate_states(B, self.device)

    def _allocate_states(
        self,
        B: int,
        device: torch.device,
    ) -> list[dict]:
        """Allocate one complete MQA/compressor state replica on a TP rank."""
        c = self.cfg
        win, hd = c["sliding_window"], c["head_dim"]
        hot_dtype = compute_dtype(device)
        sparse_splitkv = False
        sparse_features: tuple[str, ...] = ()
        sparse_unavailable_reason: str | None = None
        requires_sparse_splitkv = _requires_flashmla_splitkv(
            device_type=device.type,
            hip_runtime=torch.version.hip is not None,
            max_ctx=self.max_ctx,
        )
        if requires_sparse_splitkv:
            if B != 1 or int(hd) != 512:
                raise RuntimeError(
                    "DSV4 长上下文 CUDA 路线仅支持 batch=1、head_dim=512；"
                    "禁止退回 BF16 全量 Attention"
                )
            from .flashmla_sparse import available as flashmla_available
            from .ops.paged_sparse import cuda_architecture_features

            sparse_splitkv, unavailable_reason = flashmla_available(device)
            if not sparse_splitkv:
                # ``max_ctx`` is the logical, dynamically growing KV limit.
                # It must not prevent an Ada/older GPU from starting while
                # the live context is still inside the exact direct bucket.
                # Keep the reason and fail only if execution actually reaches
                # the sparse-only bucket; no BF16 fallback is introduced.
                sparse_unavailable_reason = (
                    unavailable_reason or "unknown reason"
                )
            else:
                sparse_features = cuda_architecture_features(device)
        if sparse_splitkv:
            from .flashmla_sparse import FlashMLASparseRunner
            from .ops.paged_sparse import (
                IndexerFP8PagedCache,
                Model1FP8PagedCache,
            )
        states = []
        for i in range(c["n_layers"]):
            ratio = self.ratios[i]
            st = {
                # 现有 decode attention 核仍以 FP32 做局部 score/value；
                # SM120 BF16 sparse kernel 接入后再把窗口状态切回 hot_dtype。
                "kv": torch.zeros(
                    B, win, hd, device=device, dtype=torch.float32
                ),
                "win_pos": torch.full(
                    (B, win), -1, dtype=torch.long, device=device
                ),
                "sparse_splitkv_unavailable_reason": (
                    sparse_unavailable_reason
                ),
            }
            if sparse_splitkv:
                st["window_fp8"] = Model1FP8PagedCache.allocate(
                    max_items=int(win),
                    page_items=int(win),
                    device=device,
                )
                st["window_fp8_indices"] = torch.arange(
                    int(win), dtype=torch.int32, device=device
                ).view(1, 1, -1)
                st["sparse_features"] = sparse_features
                # Sparse Prefill also accelerates the five uncompressed
                # sliding-window layers, so every layer owns a runner.  The
                # decode scheduler metadata remains lazy and shape-specific.
                st["sparse_runner"] = FlashMLASparseRunner.create()
            if ratio:
                st["compressed"] = PagedKV(
                    batch=B,
                    page_items=_dsv4_compressed_page_items(ratio),
                    dim=hd,
                    device=device,
                    dtype=compute_dtype(device),
                    max_items=(self.max_ctx + ratio - 1) // ratio,
                )
                if sparse_splitkv:
                    max_compressed = (self.max_ctx + ratio - 1) // ratio
                    st["compressed_fp8"] = Model1FP8PagedCache.allocate(
                        max_items=max_compressed,
                        page_items=min(256, max_compressed),
                        device=device,
                    )
                if ratio == 4:
                    st["indexer"] = IndexerState(
                        batch=B,
                        head_dim=c.get("index_head_dim", 128),
                        rope_dim=c["qk_rope_head_dim"],
                        page_items=max(1, 4096 // ratio),
                        device=device,
                        dtype=compute_dtype(device),
                        max_items=(self.max_ctx + ratio - 1) // ratio,
                    )
                    if sparse_splitkv:
                        candidate_count = (self.max_ctx + ratio - 1) // ratio
                        # FP8 scaled GEMM requires both matrix extents to be
                        # multiples of 16.  ``max_ctx`` is user controlled and
                        # therefore its ratio-4 candidate count is commonly
                        # unaligned (for example 4368 -> 1092).  Keep the
                        # logical cache bound exact, but pad only the fixed
                        # Indexer execution image.  The controlled reducer
                        # masks every row beyond the live position to -inf.
                        candidate_capacity = _indexer_candidate_capacity(
                            candidate_count
                        )
                        indexer_fp8 = IndexerFP8PagedCache.allocate(
                            max_items=candidate_count,
                            head_dim=int(c.get("index_head_dim", 128)),
                            page_items=min(1024, candidate_count),
                            device=device,
                        )
                        st["indexer_fp8"] = indexer_fp8
                        st["indexer_query_fp8"] = torch.empty(
                            int(c.get("index_n_heads", 64)),
                            int(c.get("index_head_dim", 128)),
                            dtype=torch.float8_e4m3fn,
                            device=device,
                        )
                        st["indexer_query_scales"] = torch.empty(
                            int(c.get("index_n_heads", 64)),
                            dtype=torch.float32,
                            device=device,
                        )
                        st["indexer_mm"] = torch.empty(
                            int(c.get("index_n_heads", 64)),
                            candidate_capacity,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        st["indexer_logits"] = torch.empty(
                            1, 1, candidate_capacity,
                            dtype=torch.float32,
                            device=device,
                        )
                        st["indexer_topk_values"] = torch.empty(
                            1, 1, int(c["index_topk"]),
                            dtype=torch.float32,
                            device=device,
                        )
                        st["indexer_topk_indices"] = torch.empty(
                            1, 1, int(c["index_topk"]),
                            dtype=torch.long,
                            device=device,
                        )
                        st["flashmla_query"] = torch.empty(
                            1, 1, int(c["n_heads"]), int(hd),
                            dtype=torch.bfloat16,
                            device=device,
                        )
                coff = 2 if ratio == 4 else 1
                st["ckv"] = torch.zeros(
                    B,
                    coff * ratio,
                    coff * hd,
                    device=device,
                    dtype=hot_dtype,
                )
                st["cscore"] = torch.full((B, coff * ratio, coff * hd), float("-inf"),
                                          device=device, dtype=torch.float32)
            states.append(st)
        return states

    def ensure_position(self, position: int) -> None:
        """Reserve every compressed-layer page before any token state is mutated."""
        if position < 0 or self.states is None:
            return
        try:
            for layer, ratio in enumerate(self.ratios):
                if ratio:
                    self.states[layer]["compressed"].reserve(position // ratio)
                    indexer = self.states[layer].get("indexer")
                    if indexer is not None:
                        indexer.reserve_position(position)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            raise ContextCapacityError(position, exc) from exc

    @torch.no_grad()
    def snapshot_kv(self) -> DSV4KVSnapshot:
        """Copy every in-place mutable field needed to restore one boundary."""
        if self.states is None:
            raise ValueError("cannot snapshot an empty DSV4 KV state")
        if len(self.states) != len(self.ratios):
            raise ValueError("DSV4 KV state/ratio layer count mismatch")

        layers = []
        for ratio, state in zip(self.ratios, self.states):
            indexer = state.get("indexer")
            layers.append(
                DSV4LayerKVSnapshot(
                    kv=state["kv"].clone(),
                    win_pos=state["win_pos"].clone(),
                    compressed_length=(
                        int(state["compressed"].length)
                        if ratio
                        else None
                    ),
                    ckv=state["ckv"].clone() if ratio else None,
                    cscore=(
                        state["cscore"].clone() if ratio else None
                    ),
                    indexer_length=(
                        int(indexer.keys.length)
                        if indexer is not None
                        else None
                    ),
                    indexer_ckv=(
                        indexer.ckv.clone()
                        if indexer is not None
                        else None
                    ),
                    indexer_cscore=(
                        indexer.cscore.clone()
                        if indexer is not None
                        else None
                    ),
                )
            )
        return DSV4KVSnapshot(
            pos=int(self.pos),
            layers=tuple(layers),
        )

    @torch.no_grad()
    def restore_kv(self, snapshot: DSV4KVSnapshot) -> None:
        """Atomically validate, then restore a stable DSV4 prompt boundary."""
        if self.states is None:
            raise ValueError("cannot restore into an empty DSV4 KV state")
        if (
            len(snapshot.layers) != len(self.states)
            or len(self.ratios) != len(self.states)
        ):
            raise ValueError("DSV4 KV snapshot layer count mismatch")

        def require_tensor(
            live: torch.Tensor,
            saved: torch.Tensor | None,
            label: str,
        ) -> None:
            if saved is None:
                raise ValueError(
                    f"DSV4 KV snapshot missing {label}"
                )
            if (
                live.shape != saved.shape
                or live.dtype != saved.dtype
                or live.device != saved.device
            ):
                raise ValueError(
                    f"DSV4 KV snapshot {label} mismatch"
                )

        # Validate every layer before copying any tensor.
        for ratio, state, saved in zip(
            self.ratios,
            self.states,
            snapshot.layers,
        ):
            require_tensor(state["kv"], saved.kv, "raw ring")
            require_tensor(
                state["win_pos"],
                saved.win_pos,
                "win_pos",
            )

            compressed = state.get("compressed")
            if ratio:
                if (
                    compressed is None
                    or saved.compressed_length is None
                    or saved.compressed_length < 0
                    or saved.compressed_length > compressed.length
                ):
                    raise ValueError(
                        "DSV4 KV snapshot compressed length mismatch"
                    )
                require_tensor(
                    state["ckv"],
                    saved.ckv,
                    "compressor ckv",
                )
                require_tensor(
                    state["cscore"],
                    saved.cscore,
                    "compressor cscore",
                )
            elif saved.compressed_length is not None:
                raise ValueError(
                    "DSV4 KV snapshot unexpected compressed state"
                )

            indexer = state.get("indexer")
            if ratio == 4:
                if (
                    indexer is None
                    or saved.indexer_length is None
                    or saved.indexer_length < 0
                    or saved.indexer_length > indexer.keys.length
                ):
                    raise ValueError(
                        "DSV4 KV snapshot Indexer length mismatch"
                    )
                require_tensor(
                    indexer.ckv,
                    saved.indexer_ckv,
                    "Indexer ckv",
                )
                require_tensor(
                    indexer.cscore,
                    saved.indexer_cscore,
                    "Indexer cscore",
                )
            elif saved.indexer_length is not None:
                raise ValueError(
                    "DSV4 KV snapshot unexpected Indexer state"
                )

        for state, saved in zip(self.states, snapshot.layers):
            state["kv"].copy_(saved.kv)
            state["win_pos"].copy_(saved.win_pos)
            # Model1 window cache is a derived, fixed-address representation
            # of the mutable BF16 ring.  Rebuild it after rollback instead of
            # copying a second full snapshot.  Compressed/Indexer FP8 caches
            # are append-only and are bounded by the restored logical lengths.
            window_fp8 = state.get("window_fp8")
            if window_fp8 is not None:
                window_fp8.load_bf16(state["kv"][0])
            if saved.compressed_length is not None:
                state["compressed"].truncate(
                    saved.compressed_length
                )
                state["ckv"].copy_(saved.ckv)
                state["cscore"].copy_(saved.cscore)
            if saved.indexer_length is not None:
                indexer = state["indexer"]
                indexer.keys.truncate(saved.indexer_length)
                indexer.ckv.copy_(saved.indexer_ckv)
                indexer.cscore.copy_(saved.indexer_cscore)

        self.pos = snapshot.pos
        self._spec = None
        self._prev_ids.clear()

    def reset(self) -> None:
        self.states = None
        self._tp1_token_graphs.clear()
        self._tp1_token_logits.clear()
        self._tp1_graph_dependencies.clear()
        self._tp1_graph_build_error = None
        self.tp_token_graph_info = {}
        if self._tp_attention_contexts is not None:
            for rank_contexts in self._tp_attention_contexts:
                for context in rank_contexts:
                    context["state"] = None
        self._tp_states_ready = False
        self.pos = 0
        self._spec = None
        self._prev_ids.clear()

    # ---- Engine 接口（与 CCCP GLMModel 同名：forward/forward_hidden/logits_of/reset_kv/pos） ----
    pos: int = 0

    DSPARK_TARGETS = (40, 41, 42)   # DSpark main_hidden 的取材层（hc 均值隐态）

    def reset_kv(self) -> None:
        self.reset()

    def preload(self) -> None:
        """GPU 路径：全部 dense 权重上显存（int4 打包态 + head f32 常驻）。"""
        if self.device.type == "cpu":
            print(
                f"[cccp] CPU 推理线程：{self._cpu_threads}",
                flush=True,
            )
            if self._cpu_numa_interleaved:
                print(
                    "[cccp] CPU NUMA：专家与 dense 内存跨节点交错分配",
                    flush=True,
                )
            # CPU 首次启动时提前编译/装载融合内核，避免首个 decode 卡顿。
            from .cpuext import prebuild as prebuild_cpu
            prebuild_cpu()
            # CPU 也需要真正的 RAM 模式；仅使用 LRU 会让每批新路由专家
            # 重复承担 zlib 解压和张量构造，即使机器还有数百 GiB 可用内存。
            residency = self.routed_vq.initialize_residency(
                device_type="cpu",
            )
            resident_all = residency.resident_all
            profile_resident = bool(self.routed_vq.compact_full_resident)
            self._prefetch_auto = not (resident_all or profile_resident)
            if resident_all or profile_resident:
                n_experts = self.cfg["n_experts"]
                for layer in range(self.cfg["n_layers"]):
                    experts = self.routed_vq.resident_entries(
                        layer,
                        n_experts,
                    )
                    if any(expert is not None for expert in experts):
                        self._cpu_resident_experts[layer] = experts
                prepared_layers = residency.native_layers
                print(
                    "[cccp-dsv4] CPU 多码本 MoE 执行图完成："
                    f"{prepared_layers} 层；packed 索引保持紧凑，"
                    "Gate/Up/激活/Down/路由归并单次原生调用",
                    flush=True,
                )
            return
        import time
        t0 = time.time()
        extreme_residency = None
        if self.routed_vq.startup_gpu_reserve_bytes > 0:
            # Extreme mode is deliberately expert-first. The pool holds a
            # real CUDA reservation for Dense/context while it fills RAM down
            # to the 1 GiB floor and places overflow experts in VRAM.
            extreme_residency = self.routed_vq.prepare_required_residency(
                device_type="cuda",
            )
        if self._dense_bf16:
            print("[cccp] dense BF16 常驻: "
                  + ",".join(sorted(self._dense_bf16)), flush=True)
        names = self.store.dense_names()
        for name in names:
            self.w(name)
        if self._native_tensor_fp8_weights:
            print(
                "[cccp] 公共 Tensor Core FP8 执行映像："
                f"{self._native_tensor_fp8_weights} 个固定投影；"
                "源检查点未修改，未保留 BF16 副本",
                flush=True,
            )
        # Keep every routing mask at a fixed address before any Attention,
        # FFN or packed-expert graph is built.  Hash-routed layers do not
        # enter the static FFN capture, so lazily moving their first mask
        # during prefill can force the CUDA allocator to synchronize with
        # the persistent mapped-DMA stream.
        if hasattr(self, "cfg") and hasattr(self.store, "available_mask"):
            self._masks = {
                layer: self.store.available_mask(layer).to(self.device)
                for layer in range(int(self.cfg["n_layers"]))
            }
        self._prepare_tp_shared_mlp()
        self._prepare_heterogeneous_shared_mlp()
        self._prepare_single_gpu_static_graphs()
        self._prepare_tp_decode_metadata()
        if extreme_residency is not None:
            self.routed_vq.verify_required_residency()
        vram = torch.cuda.memory_allocated(self.device) / 2**30
        print(f"[cccp] dense 预载完成（{time.time() - t0:.1f}s，显存 {vram:.1f}GB）",
              flush=True)
        # 公共算子注册已在构造期完成解析。这里只报告最终能力，不能再读取
        # grouped 模块历史上的私有 ``_fused`` 状态；该变量在公共化后已删除。
        if os.environ.get("CCCP_GROUPED", "1") != "0":
            backend = self.packed_operator_name or "legacy-grouped-adapter"
            print(
                f"[cccp] 分组GEMM/packed MoE: {backend}",
                flush=True,
            )
        if self._packed_full_gpu:
            # Dense/Attention fixed graphs above are already resident.  The
            # packed pool must compare only its remaining allocations against
            # the currently free VRAM instead of charging Dense twice.
            self.routed_vq.materialize_full_device(dense_resident=True)
            self._prepare_tp_packed_finalizer()
            self._prefetch_auto = False
            return
        residency = (
            self.routed_vq.finalize_residency(
                extreme_residency,
                device_type="cuda",
            )
            if extreme_residency is not None
            else self.routed_vq.initialize_residency(device_type="cuda")
        )
        resident_all = residency.resident_all
        extreme_staging = bool(
            self.routed_vq.fixed_extreme_residency
            and self.routed_vq.extreme_ram_layers
        )
        route_history_resident = bool(
            self.routed_vq.extreme_route_history_resident
        )
        self._prefetch_auto = _automatic_prefetch_policy(
            resident_all=resident_all,
            packed_device_pool=self._packed_device_pool,
            packed_full_gpu=self._packed_full_gpu,
            extreme_staging=extreme_staging,
            route_history_resident=route_history_resident,
        )
        if resident_all:
            if self.routed_vq.profile_hot_cache_enabled:
                # Corpus heat seeds a shared strict LRU. Previous-token
                # speculation uploaded far more experts than demand and
                # displaced useful entries before their layer executed.
                self._prefetch_auto = False
            if os.environ.get("CCCP_PREFETCH", "auto").strip().lower() in ("", "auto"):
                if extreme_staging:
                    if route_history_resident:
                        print(
                            "[cccp-extreme] 一整轮路由可驻留显存：按层保护上一轮"
                            " Top-K，仅传输真实路由变化",
                            flush=True,
                        )
                    else:
                        print(
                            "[cccp-extreme] 启用逐层专家预取：与本层 Attention 重叠，"
                            "禁止整轮 staging 覆盖",
                            flush=True,
                        )
                elif self._prefetch_enabled():
                    print(
                        "[cccp] RAM+VRAM 跨层专家预取已启用："
                        "上一 token 路由与当前层 Attention/共享专家计算重叠",
                        flush=True,
                    )
                elif self.routed_vq.profile_hot_cache_enabled:
                    print(
                        "[cccp] 专家缓存使用按需 strict-LRU；"
                        "关闭上一 token 推测预取，避免无效 H2D 污染热槽",
                        flush=True,
                    )
                else:
                    print(
                        "[cccp] RAM+VRAM 按需专家 staging 已启用；"
                        "跨层预取关闭以避免小显存热槽竞争",
                        flush=True,
                    )

    def _prepare_tp_shared_mlp(self) -> None:
        """Shard every shared expert through the public gated-MLP TP op.

        This is intentionally capability driven: DSV4 only supplies separate
        Gate/Up weights and the clamped SwiGLU parameters.  Weight slicing,
        block-FP8 GEMV, activation and Row-TP reduction remain in ``cccp.ops``.
        """
        if (
            (self.tp_size <= 1 and not self._single_gpu_layer_graph)
            or not self.store.man.projection_vq
            or os.environ.get("CCCP_SHARED_MLP_TP", "1") == "0"
        ):
            return
        from .ops.tensor_parallel import (
            GatedMLPSpec,
            TensorParallelGatedMLP,
        )

        intermediate = int(self.layer(0)["sh_w1"].shape[0])
        executor = TensorParallelGatedMLP(
            self.devices,
            GatedMLPSpec(
                hidden_size=int(self.cfg["hidden"]),
                intermediate_size=intermediate,
                activation=self.operator_config.expert_activation,
                activation_beta=float(self.cfg.get("situ_beta", 4.0)),
                activation_linear_beta=self.cfg.get("situ_linear_beta"),
                activation_limit=float(self.cfg.get("swiglu_limit", 0.0)),
            ),
        )
        for layer in range(int(self.cfg["n_layers"])):
            weights = self.layer(layer)
            executor.add_layer(
                layer,
                0,
                ProjectionGroup(
                    (weights["sh_w1"], weights["sh_w3"])
                ),
                weights["sh_w2"],
            )
        executor.capture()
        self._tp_shared_mlp = executor
        print(
            "[cccp] 公共共享 Dense MLP Column/Row-TP Graph 完成："
            f"{self.cfg['n_layers']} 层×TP{self.tp_size}；"
            "FP8 分片常驻、支持 partial 延迟规约",
            flush=True,
        )

    def _prepare_heterogeneous_shared_mlp(self) -> None:
        """Keep exact shared experts on CPU for GPU-routed overlap.

        This path is capability based: a single CUDA device, a dynamic packed
        expert pool, exact Block-FP8 shared projections, and the public CPU
        fused operator.  It is intentionally disabled for full-resident/TP,
        HIP and Windows paths until their host-mapping contracts provide the
        same routed prelaunch boundary.
        """

        # The current Python-coordinated CPU/GPU branch is a diagnostics-only
        # implementation.  It is exact, but it adds one host rendezvous per
        # layer and regresses H20 Decode.  Keep it explicitly opt-in until the
        # stream-memory-op persistent worker replaces that rendezvous.
        enabled = os.environ.get(
            "CCCP_HETEROGENEOUS_SHARED", "0"
        ).strip().lower()
        if enabled in ("0", "false", "off", "none"):
            return
        if (
            self.device.type != "cuda"
            or torch.version.hip is not None
            or os.name == "nt"
            or self.tp_size != 1
            or not self._packed_device_pool
            or self._packed_full_gpu
            or not self.store.man.projection_vq
        ):
            return
        from . import cpuext
        from .kernels import BlockFP8Weight
        from .ops.heterogeneous import HeterogeneousSharedExpertExecutor

        status = cpuext.extension_status()
        if not bool(status.get("available")):
            if enabled not in ("", "auto"):
                raise RuntimeError(
                    "heterogeneous shared expert requires the fused CPU "
                    f"operator: {status.get('error')}"
                )
            print(
                "[cccp-heterogeneous] CPU shared disabled: "
                f"{status.get('error')}",
                flush=True,
            )
            return
        executor = HeterogeneousSharedExpertExecutor(
            device=self.device,
            hidden_size=int(self.cfg["hidden"]),
            dtype=torch.bfloat16,
        )

        def owned_exact(weight: BlockFP8Weight) -> BlockFP8Weight:
            # Block-major is an exact byte permutation.  Clone both payload
            # and scales because Engine releases the dense mmap after preload.
            optimized = weight.to_block_major()
            return BlockFP8Weight(
                optimized.q.clone(),
                optimized.s.clone(),
                optimized.cols,
                optimized.block,
                rows=optimized.rows,
                layout=optimized.layout,
            )

        retained = 0
        for layer in range(int(self.cfg["n_layers"])):
            prefix = f"layers.{layer}.ffn.shared_experts"
            weights = [
                self.store.get_dense(f"{prefix}.w{index}.weight")
                for index in (1, 3, 2)
            ]
            if not all(isinstance(weight, BlockFP8Weight) for weight in weights):
                if enabled not in ("", "auto"):
                    raise RuntimeError(
                        "heterogeneous shared expert requires exact "
                        "Block-FP8 weights"
                    )
                return
            exact = [owned_exact(weight) for weight in weights]
            retained += sum(weight.nbytes for weight in exact)
            executor.add_layer(layer, exact[0], exact[1], exact[2])
        self._heterogeneous_shared_mlp = executor
        print(
            "[cccp-heterogeneous] shared=CPU exact Block-FP8；"
            "routed=GPU packed；schedule=parallel-DAG；"
            f"threads={status.get('threads')}；"
            f"host_resident={retained / 2**30:.2f}GiB",
            flush=True,
        )

    def _prepare_single_gpu_static_graphs(self) -> None:
        """Capture exact TP1 work around the dynamic packed-expert boundary.

        This follows the same graph boundary used by a quantized ``MUL_MAT_ID``
        runtime: keep routing IDs and all stable dense work on CUDA, then stop
        only where a RAM cache miss may replace compact expert-slot contents.
        Compact routed experts themselves are intentionally excluded because
        their arena addresses are stable while their contents are not.
        """
        if (
            not getattr(self, "_single_gpu_static_graphs", False)
            or getattr(self, "_single_gpu_layer_graph", False)
            or self.device.type != "cuda"
        ):
            return
        from .ops import FixedAddressCudaGraph

        device = self.device
        hidden = int(self.cfg["hidden"])
        base_cfg = self._cfg_obj()
        rope_width = int(base_cfg.qk_rope_head_dim)
        with torch.cuda.device(device):
            pool = torch.cuda.graph_pool_handle()
            source = torch.empty(
                1,
                1,
                int(self.cfg["hc_mult"]),
                hidden,
                dtype=compute_dtype(device),
                device=device,
            )
            token = torch.zeros(1, dtype=torch.long, device=device)
            fixed_cos = torch.ones(
                1,
                rope_width // 2,
                dtype=torch.float32,
                device=device,
            )
            fixed_sin = torch.zeros_like(fixed_cos)
        fixed_cache = SimpleNamespace(cos=fixed_cos, sin=fixed_sin)
        for layer in range(int(self.cfg["n_layers"])):
            weights = self.layer(layer)
            part_names = tuple(
                name for name in ("cmp", "indexer") if name in weights
            )

            def project(
                *,
                item=weights,
                item_cfg=base_cfg,
                cache=fixed_cache,
                names=part_names,
            ):
                normalized, post, comb = _hc_pre_norm_cccp(
                    source,
                    item["hc_attn_fn"],
                    item["hc_attn_scale"],
                    item["hc_attn_base"],
                    item["attn_norm"],
                    item_cfg,
                )
                qr, q, kv = _qkv_cccp(
                    normalized,
                    item,
                    item_cfg,
                    cache,
                    0,
                )
                values = [normalized, post, comb, qr, q, kv]
                for name in names:
                    nested = item[name]
                    values.extend((
                        _cccp_lin(normalized, nested["wkv"]),
                        _cccp_lin(normalized, nested["wgate"]),
                    ))
                return tuple(values)

            graph = FixedAddressCudaGraph(device, project, pool=pool)
            self._single_gpu_attention_graphs[layer] = {
                "graph": graph,
                "source": source,
                "cos": fixed_cos,
                "sin": fixed_sin,
                "parts": part_names,
            }
            # Hash-routed layers may repair unavailable expert IDs with a
            # data-dependent Python branch. Keep those few layers on the
            # ordinary exact path; learned routing is entirely device-side
            # and safe to capture with the shared dense MLP.
            if (
                layer >= int(self.cfg.get("n_hash_layers", 0))
                and os.environ.get("CCCP_STATIC_FFN_GRAPH", "1") != "0"
            ):
                limit = float(self.cfg.get("swiglu_limit", 0.0))

                include_shared = self._heterogeneous_shared_mlp is None

                def ffn(
                    *,
                    item=weights,
                    item_cfg=base_cfg,
                    layer_index=layer,
                    activation_limit=limit,
                    compute_shared=include_shared,
                ):
                    normalized, post, comb = _hc_pre_norm_cccp(
                        source,
                        item["hc_ffn_fn"],
                        item["hc_ffn_scale"],
                        item["hc_ffn_base"],
                        item["ffn_norm"],
                        item_cfg,
                    )
                    rows = normalized.reshape(1, hidden)
                    route_weights, route_ids = self._route_cccp(
                        rows.float(),
                        item,
                        item_cfg,
                        token,
                        layer_index,
                    )
                    if compute_shared:
                        shared = _shared_expert_mlp_cccp(
                            rows,
                            item,
                            activation_limit,
                        )
                        return (
                            normalized,
                            post,
                            comb,
                            shared,
                            route_weights,
                            route_ids,
                        )
                    return (
                        normalized,
                        post,
                        comb,
                        route_weights,
                        route_ids,
                    )

                self._single_gpu_ffn_graphs[layer] = {
                    "graph": FixedAddressCudaGraph(
                        device,
                        ffn,
                        pool=pool,
                    ),
                    "source": source,
                    "includes_shared": include_shared,
                }
        print(
            "[cccp] TP1 fixed decode packets ready: "
            "Attention HC/Norm+projections="
            f"{len(self._single_gpu_attention_graphs)}, "
            "FFN HC/Norm+shared/router="
            f"{len(self._single_gpu_ffn_graphs)}；"
            "dynamic boundary=packed expert slot replacement",
            flush=True,
        )

    @staticmethod
    def _weight_to_device(weight, device: torch.device):
        if isinstance(weight, (BlockFP8Weight, Int4Weight)):
            return weight.to(device)
        if isinstance(weight, torch.Tensor):
            return weight.to(device)
        raise TypeError(f"unsupported TP weight {type(weight)!r}")

    def _prepare_tp_decode_metadata(self) -> None:
        """Prepare all-rank Head/Dense/MoE decode metadata.

        The model layer only describes the HC and compressed-MQA topology.
        Compact FP8 slicing, fixed TPHidden buffers, shared Dense TP and packed
        expert TP are all supplied by the public operator library.
        """
        if (
            (self.tp_size <= 1 and not self._single_gpu_layer_graph)
            or self._tp_shared_mlp is None
            or os.environ.get("CCCP_FULL_TP", "1") == "0"
        ):
            return
        if os.environ.get("CCCP_TP_HIDDEN", "0") == "0":
            raise RuntimeError("DSV4 full TP requires CCCP_TP_HIDDEN=1")
        from .dsv4 import RopeCache
        from .ops import (
            ReplicatedSubgroupTensorParallel,
            shard_linear_input,
            shard_linear_output,
        )

        base_cfg = self._cfg_obj()
        attention_tp = min(
            self.tp_size,
            max(
                1,
                int(
                    os.environ.get(
                        "CCCP_SMALL_OP_TP",
                        str(self.tp_size),
                    )
                ),
            ),
        )
        if (
            self.tp_size % attention_tp
            or int(base_cfg.n_heads) % attention_tp
            or int(base_cfg.o_groups) % attention_tp
        ):
            raise ValueError(
                "Attention subgroup must divide ranks, heads and O groups"
            )
        self._tp_attention_parallel = ReplicatedSubgroupTensorParallel(
            self.devices,
            attention_tp,
        )
        attention_by_rank: list[list[dict]] = []
        route_by_rank: list[list[dict]] = []
        for rank, device in enumerate(self.devices):
            attention_rank = self._tp_attention_parallel.local_rank(rank)
            local_cfg = copy.copy(base_cfg)
            local_cfg.n_heads = int(base_cfg.n_heads) // attention_tp
            local_cfg.o_groups = int(base_cfg.o_groups) // attention_tp
            rope_base = RopeCache(
                int(base_cfg.qk_rope_head_dim),
                self.max_ctx + 8,
                float(base_cfg.rope_theta),
                None,
            )
            rope_cmp = RopeCache(
                int(base_cfg.qk_rope_head_dim),
                self.max_ctx + 8,
                float(getattr(base_cfg, "compress_rope_theta", 160000.0)),
                getattr(base_cfg, "rope_scaling", None),
            )
            for rope in (rope_base, rope_cmp):
                rope.cos = rope.cos.to(device)
                rope.sin = rope.sin.to(device)
            rank_attention = []
            rank_routes = []
            for layer in range(int(self.cfg["n_layers"])):
                source = self.layer(layer)
                weights = {
                    key: self._weight_to_device(source[key], device)
                    for key in ("wq_a", "q_norm", "wkv", "kv_norm")
                }
                grouped_attention = os.environ.get(
                    "CCCP_GROUPED_ATTN_PROJECTIONS", "1"
                ) != "0"
                if (
                    grouped_attention
                    and isinstance(weights["wq_a"], BlockFP8Weight)
                    and isinstance(weights["wkv"], BlockFP8Weight)
                ):
                    weights["qkv_projection_group"] = ProjectionGroup(
                        (weights["wq_a"], weights["wkv"])
                    )
                weights["wq_b"] = shard_linear_output(
                    source["wq_b"], attention_rank, attention_tp, device
                )
                weights["q_head_norm"] = torch.ones(
                    int(base_cfg.head_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
                weights["attn_sink"] = (
                    source["attn_sink"]
                    .chunk(attention_tp, dim=0)[attention_rank]
                    .to(device)
                    .contiguous()
                )
                weights["wo_a"] = shard_linear_output(
                    source["wo_a"], attention_rank, attention_tp, device
                )
                weights["wo_b"] = shard_linear_input(
                    source["wo_b"], attention_rank, attention_tp, device
                )
                for nested in ("cmp", "indexer"):
                    if nested in source:
                        weights[nested] = {
                            key: self._weight_to_device(value, device)
                            for key, value in source[nested].items()
                        }
                        nested_weights = weights[nested]
                        if (
                            grouped_attention
                            and isinstance(
                                nested_weights.get("wkv"),
                                BlockFP8Weight,
                            )
                            and isinstance(
                                nested_weights.get("wgate"),
                                BlockFP8Weight,
                            )
                        ):
                            nested_weights["projection_group"] = (
                                ProjectionGroup(
                                    (
                                        nested_weights["wkv"],
                                        nested_weights["wgate"],
                                    )
                                )
                            )
                rank_attention.append(
                    {
                        "cfg": local_cfg,
                        "weights": weights,
                        "state": None,
                        "rope": rope_cmp if self.ratios[layer] else rope_base,
                        "ratio": int(self.ratios[layer]),
                    }
                )
                route_item = {
                    key: self._weight_to_device(source[key], device)
                    for key in (
                        "hc_attn_fn",
                        "hc_attn_scale",
                        "hc_attn_base",
                        "attn_norm",
                        "hc_ffn_fn",
                        "hc_ffn_scale",
                        "hc_ffn_base",
                        "ffn_norm",
                    )
                }
                route_item["gate_bias"] = self._weight_to_device(
                    source.get(
                        "gate_bias",
                        torch.zeros(
                            int(self.cfg["n_experts"]),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                    ),
                    device,
                )
                route_item["mask"] = self.store.available_mask(layer).to(
                    device
                )
                if "tid2eid" in source:
                    route_item["tid2eid"] = source["tid2eid"].to(device)
                rank_routes.append(route_item)
            attention_by_rank.append(rank_attention)
            route_by_rank.append(rank_routes)
        self._tp_attention_contexts = tuple(attention_by_rank)
        self._tp_route_weights = tuple(route_by_rank)
        self.tp_dataflow = "all-rank-head-dense-packed"

        from .ops import TPHidden, TPPartials
        from .ops.tensor_parallel import (
            ReplicatedLinearSpec,
            TensorParallelReplicatedLinear,
        )

        router = TensorParallelReplicatedLinear(
            self.devices,
            ReplicatedLinearSpec(
                in_features=int(self.cfg["hidden"]),
                out_features=int(self.cfg["n_experts"]),
                input_dtype=torch.bfloat16,
                output_dtype=torch.float32,
            ),
        )
        for layer in range(int(self.cfg["n_layers"])):
            router.add_layer(layer, self.layer(layer)["gate"])
            router.bind_input_hidden(
                layer,
                self._tp_shared_mlp.input_hidden(layer),
            )
        router.capture()
        self._tp_router = router

        top_k = int(self.cfg["top_k"])
        for layer in range(int(self.cfg["n_layers"])):
            logits = list(router.output_hidden(layer).replicas)
            route_weights = []
            route_ids = []
            route_events = []
            for device in self.devices:
                route_weights.append(
                    torch.empty(1, top_k, dtype=torch.float32, device=device)
                )
                route_ids.append(
                    torch.empty(1, top_k, dtype=torch.long, device=device)
                )
                route_events.append(torch.cuda.Event())
            self._tp_route_buffers[layer] = (
                tuple(logits),
                tuple(route_weights),
                tuple(route_ids),
            )
            self._tp_route_events[layer] = tuple(route_events)
            self.routed_vq.bind_hidden_inputs(
                layer,
                self._tp_shared_mlp.input_hidden(layer),
                tuple(route_weights),
                tuple(route_ids),
            )
        hidden_shape = (
            1,
            1,
            int(self.cfg["hc_mult"]),
            int(self.cfg["hidden"]),
        )
        self._tp_decode_input = TPHidden.empty(
            self.devices,
            hidden_shape,
            dtype=compute_dtype(self.devices[0]),
        )
        from .ops import DecodeControl

        self._tp_decode_controls = tuple(
            DecodeControl(device) for device in self.devices
        )
        self._tp_token_ids = tuple(
            control.token for control in self._tp_decode_controls
        )
        if self.tp_size == 1:
            self._tp1_decode_control = self._tp_decode_controls[0]
        for replica in self._tp_decode_input.replicas:
            self._hc_decode_workspace(replica)
        for layer in range(int(self.cfg["n_layers"])):
            self._tp_attention_hidden[layer] = TPHidden.empty(
                self.devices,
                (1, 1, int(self.cfg["hidden"])),
                dtype=compute_dtype(self.devices[0]),
            )
            if self.tp_size > 1:
                self._tp_attention_partials[layer] = TPPartials.empty(
                    self.devices,
                    (1, 1, int(self.cfg["hidden"])),
                )
            self._tp_ffn_prefix_hidden[layer] = TPHidden.empty(
                self.devices,
                hidden_shape,
                dtype=compute_dtype(self.devices[0]),
            )
            self._tp_layer_output_hidden[layer] = TPHidden.empty(
                self.devices,
                hidden_shape,
                dtype=compute_dtype(self.devices[0]),
            )
            for replica in (
                *self._tp_ffn_prefix_hidden[layer].replicas,
                *self._tp_layer_output_hidden[layer].replicas,
            ):
                self._hc_decode_workspace(replica)
        self._prepare_tp_attention_projection_graphs()
        print(
            "[cccp] DSV4 公共固定地址 decode 元数据完成："
            f"Head-TP + shared Dense Column/Row-TP + packed MoE TP，"
            f"{self.cfg['n_layers']} 层×TP{self.tp_size}；"
            f"Attention replicated-subgroup=TP{attention_tp}",
            flush=True,
        )

    def _prepare_tp_attention_projection_graphs(self) -> None:
        """Capture exact dense/norm/RoPE projection sequences per TP rank.

        Only launch scheduling changes: the same public linear, RMSNorm and
        RoPE kernels run in the same mathematical order. Compressor state,
        KV selection and attention remain outside the graph, so context
        length and routing stay dynamic.
        """
        if (
            self._tp_attention_contexts is None
            or self._tp_decode_input is None
            or os.environ.get("CCCP_ATTN_PROJECTION_GRAPH", "1") == "0"
        ):
            return
        from .ops import FixedAddressCudaGraph

        graphs = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                pool = torch.cuda.graph_pool_handle()
            local_input = self._hc_decode_workspace(
                self._tp_decode_input.replicas[rank]
            )[0].view(1, 1, -1)
            local_input.zero_()
            rank_graphs = []
            for context in self._tp_attention_contexts[rank]:
                weights = context["weights"]
                cfg = context["cfg"]
                rope_width = int(cfg.qk_rope_head_dim)
                fixed_cos = torch.ones(
                    1,
                    rope_width // 2,
                    dtype=torch.float32,
                    device=device,
                )
                fixed_sin = torch.zeros_like(fixed_cos)
                fixed_cache = SimpleNamespace(
                    cos=fixed_cos,
                    sin=fixed_sin,
                )
                part_names = tuple(
                    name
                    for name in ("cmp", "indexer")
                    if name in weights
                )

                def project(
                    *,
                    source=local_input,
                    item=weights,
                    item_cfg=cfg,
                    cache=fixed_cache,
                    names=part_names,
                ):
                    qr, q, kv = _qkv_cccp(
                        source,
                        item,
                        item_cfg,
                        cache,
                        0,
                    )
                    values = [qr, q, kv]
                    for name in names:
                        nested = item[name]
                        values.extend((
                            _cccp_lin(source, nested["wkv"]),
                            _cccp_lin(source, nested["wgate"]),
                        ))
                    return tuple(values)

                graph = FixedAddressCudaGraph(
                    device,
                    project,
                    pool=pool,
                )
                context["projection_graph"] = graph
                context["projection_graph_cos"] = fixed_cos
                context["projection_graph_sin"] = fixed_sin
                context["projection_graph_parts"] = part_names
                rank_graphs.append(graph)
            graphs.append(tuple(rank_graphs))
        self._tp_attention_projection_graphs = tuple(graphs)
        if self.tp_size > 1:
            from .fusedext import make_tp_graph_launch_batch

            for layer in range(int(self.cfg["n_layers"])):
                layer_graphs = tuple(
                    graphs[rank][layer] for rank in range(self.tp_size)
                )
                input_events = []
                done_events = []
                for device in self.devices:
                    with torch.cuda.device(device):
                        input_event = torch.cuda.Event()
                        done_event = torch.cuda.Event()
                        stream = torch.cuda.current_stream(device)
                        input_event.record(stream)
                        done_event.record(stream)
                        input_events.append(input_event)
                        done_events.append(done_event)
                with torch.cuda.device(self.devices[0]):
                    source_event = torch.cuda.Event()
                    source_event.record(torch.cuda.current_stream())
                batch = make_tp_graph_launch_batch(
                    [int(device.index) for device in self.devices],
                    [item.graph for item in layer_graphs],
                    [item.stream for item in layer_graphs],
                    done_events,
                    source_event,
                )
                self._tp_attention_projection_batches[layer] = batch
                self._tp_attention_projection_input_events[layer] = tuple(
                    input_events
                )
                self._tp_attention_projection_done_events[layer] = tuple(
                    done_events
                )
                self._tp_attention_projection_dependencies.extend((
                    *layer_graphs,
                    *input_events,
                    *done_events,
                    source_event,
                    batch,
                ))
        print(
            "[cccp] 通用 Attention 固定地址投影 Graph 完成："
            f"{self.cfg['n_layers']} 层×TP{self.tp_size}；"
            f"rank批提交={'on' if self.tp_size > 1 else 'tp1'}",
            flush=True,
        )

    def _prepare_tp_packed_finalizer(self) -> None:
        """Bind packed and shared partials to one public MoE collective."""
        if self._tp_attention_contexts is None:
            return
        from .ops import (
            PackedMoEFinalizerSpec,
            RoutePackedPlanSpec,
            TensorParallelAllRankCollective,
            TensorParallelPackedMoEFinalizer,
            TensorParallelRoutePackedPlan,
        )

        first_router_layer = int(self.cfg.get("n_hash_layers", 0))
        route_layers = tuple(
            range(first_router_layer, int(self.cfg["n_layers"]))
        )
        if route_layers:
            self._tp_route_packed_plan = TensorParallelRoutePackedPlan(
                self.devices,
                RoutePackedPlanSpec(
                    scoring_func=str(self.cfg["scoring_func"]),
                    top_k=int(self.cfg["top_k"]),
                    normalize=bool(self.cfg.get("norm_topk_prob", True)),
                    scaling=float(self.cfg.get("routed_scaling", 1.0)),
                    n_group=int(self.cfg.get("n_group", 1)),
                    topk_group=int(self.cfg.get("topk_group", 1)),
                ),
                self.routed_vq,
                {
                    layer: self._tp_router.output_hidden(layer)
                    for layer in route_layers
                },
                {
                    layer: tuple(
                        self._tp_route_weights[rank][layer]["gate_bias"]
                        for rank in range(self.tp_size)
                    )
                    for layer in route_layers
                },
                {
                    layer: tuple(
                        self._tp_route_weights[rank][layer]["mask"]
                        for rank in range(self.tp_size)
                    )
                    for layer in route_layers
                },
                {
                    layer: (
                        self._tp_route_buffers[layer][1],
                        self._tp_route_buffers[layer][2],
                    )
                    for layer in route_layers
                },
                layers=route_layers,
            )

        finalizer = TensorParallelPackedMoEFinalizer(
            self.devices,
            PackedMoEFinalizerSpec(
                hidden_size=int(self.cfg["hidden"]),
                dtype=torch.bfloat16,
            ),
            self.routed_vq,
        )
        for layer in range(int(self.cfg["n_layers"])):
            finalizer.add_layer(layer)
        self._tp_moe_finalizer = finalizer
        self._tp_collective = TensorParallelAllRankCollective(
            self.devices
        )
        self.tp_collectives_per_layer = 2
        print(
            "[cccp] 通用 packed+shared MoE 单次最终规约完成："
            f"{self.cfg['n_layers']} 层×TP{self.tp_size}",
            flush=True,
        )

    def _ensure_hybrid_tp1_graph_runtime(self) -> None:
        """Lazily bind the dynamic packed cache into the common TP1 graph.

        DSV4 starts with a wider, temporary Prefill arena.  Registered-host UVA
        and the device segmented LRU become authoritative only after that arena
        switches to Decode.  Build expert child graphs at that point, and
        rebuild the parent graph if KV pressure later resizes the same slab.
        """

        if not getattr(self, "_hybrid_fixed_token_graph", False):
            return
        if not self.routed_vq.activate_decode():
            raise RuntimeError(
                "hybrid TP1 TokenGraph requires a Decode arena capability"
            )
        if not self.routed_vq.fixed_token_graph_capable:
            raise RuntimeError(
                "hybrid TP1 TokenGraph requires Linux registered-host UVA, "
                "device segmented LRU and compact Q8 Decode"
            )
        if not self.routed_vq.prepare_fixed_token_graphs(
            activation=self.operator_config.expert_activation,
            activation_beta=float(self.cfg.get("situ_beta", 4.0)),
            activation_linear_beta=self.cfg.get("situ_linear_beta"),
            limit=float(self.cfg.get("swiglu_limit", 0.0)),
        ):
            raise RuntimeError("hybrid fixed expert graph construction failed")
        generation = self.routed_vq.fixed_graph_generation
        if (
            generation == self._tp1_pool_graph_generation
            and self._tp_moe_finalizer is not None
        ):
            return
        if self._tp1_token_graphs:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
            self._tp1_token_graphs.clear()
            self._tp1_token_logits.clear()
            self._tp1_graph_dependencies.clear()
            self.tp_token_graph_info = {}
        self._tp1_graph_build_error = None
        self._tp_route_packed_plan = None
        self._tp_moe_finalizer = None
        self._tp_collective = None
        self._prepare_tp_packed_finalizer()
        self._tp1_pool_graph_generation = generation

    def forward(self, ids: list[int]) -> torch.Tensor:
        """前向一段 token（prefill 或单步 decode），返回最后位置 logits [vocab]。"""
        t = torch.tensor([ids], device=self.device)
        if self.states is None:
            lg = self.prefill(t, full_logits=False)
            self.pos = len(ids)
            return lg.squeeze(0)
        out = []
        for i, tok in enumerate(ids):
            lg = self.decode(torch.tensor([tok], device=self.device), self.pos + i)
            out.append(lg)
        self.pos += len(ids)
        return out[-1].squeeze(0)

    @torch.no_grad()
    def forward_incremental_batch(self, ids_list: list[int]) -> torch.Tensor:
        """Append a non-empty suffix with one layer-first batched pass.

        Unlike :meth:`forward_verify`, this is a committed continuation: it
        does not allocate speculative snapshots or collect DSpark hidden
        states.  Attention, compressors, routed MoE and KV all receive the
        same absolute ``pos0`` and update the existing canonical state once.
        """
        if self.states is None or self.pos <= 0:
            raise RuntimeError(
                "incremental DSV4 batch requires an initialized KV prefix"
            )
        if not ids_list:
            raise ValueError("incremental DSV4 batch requires at least one token")
        from .dsv4 import hc_head

        cfg = self._cfg_obj()
        pos0 = int(self.pos)
        self.last_prefill_block_size = int(len(ids_list))
        ids = torch.tensor([ids_list], device=self.device).long()
        self.ensure_position(pos0 + len(ids_list) - 1)
        h = self._embed(ids).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        for layer in range(cfg.n_layers):
            h = self._block(h, layer, ids, pos0)
        y = hc_head(h, *self._hc_head_w(), cfg)
        y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
        logits = _linear(y[:, -1], self.w("head.weight")).float()
        self.pos = pos0 + len(ids_list)
        return logits.squeeze(0)

    def forward_hidden(self, ids: list[int]) -> torch.Tensor:
        """前向一段 token，返回全部位置的最终 hidden [T, hidden]（已过 final norm）。"""
        t = torch.tensor([ids], device=self.device)
        if self.states is not None and len(ids) == 1:
            raise RuntimeError("forward_hidden 增量模式未实现（投机解码暂未接入 DSV4）")
        from .dsv4 import hc_head
        cfg = self._cfg_obj()
        self._alloc(1)
        self.ensure_position(len(ids) - 1)
        h = self._embed(t).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        for i in range(cfg.n_layers):
            h = self._block(h, i, t, 0)
        y = hc_head(h, *self._hc_head_w(), cfg)
        y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
        self.pos = len(ids)
        return y.squeeze(0)   # [1,T,D] → [T,D]，与 GLMModel/评测脚本口径一致

    def logits_of(self, h: torch.Tensor) -> torch.Tensor:
        """hidden [N, hidden] → logits [N, vocab]。"""
        return self._head_logits(h)

    def _head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Dispatch one-token native BF16 vocab projection by capability."""
        weight = self.w("head.weight")
        if (
            self.device.type == "cpu"
            and hidden.ndim == 2
            and hidden.shape[0] == 1
            and isinstance(weight, torch.Tensor)
            and not weight.is_cuda
            and weight.dtype == torch.bfloat16
        ):
            from .ops import linear

            group = self._cpu_head_group
            if group is None:
                group = ProjectionGroup((weight,))
                self._cpu_head_group = group
            return linear(hidden, group).float()
        return _linear(hidden, weight).float()

    def _cfg_obj(self):
        """把 dict 配置包装为 dsv4.py 函数期望的属性对象。"""
        if getattr(self, "_co", None) is None:
            from .cconfig import DSV4Config
            self._co = DSV4Config.from_json(self.cfg)
        return self._co

    def _expert_mlp_cccp(self, x, gu, dn, weights):
        """VQ 专家 MLP（数值同 dsv4.expert_mlp：up±10、gate≤10、silu(gate)*up）。"""
        limit = self.cfg.get("swiglu_limit", 0.0)
        mi = self.cfg["moe_inter"]
        h = gu.matmul_T(x)                       # [N, 2*mi]
        g, u = h[:, :mi], h[:, mi:]
        if limit:
            u = u.clamp(-limit, limit)
            g = g.clamp(max=limit)
        out = dn.matmul_T(F.silu(g) * u)
        return out * weights

    def _mask(self, layer: int) -> torch.Tensor:
        """该层可用专家布尔掩码（drop 为 False），缓存。"""
        m = getattr(self, "_masks", {}).get(layer)
        if m is None:
            if not hasattr(self, "_masks"):
                self._masks = {}
            m = self.store.available_mask(layer).to(self.device)
            self._masks[layer] = m
        return m

    def _record_route_counts(self, layer: int, indices: torch.Tensor) -> None:
        """按需记录真实路由；仅校准任务启用，正常推理没有 D2H/列表开销。"""
        if os.environ.get("CCCP_ROUTE_COUNTS", "0") == "0":
            return
        self.routed_vq.record_routes(layer, indices)

    def _route_cccp(self, xf: torch.Tensor, w: dict, cfg, ids: torch.Tensor, layer: int):
        """带 drop 掩码的 sqrtsoftplus 路由（数值同 dsv4.gate_route；丢弃专家不可选）。
        learned 层：choice 掩 -inf 后 top-k；hash 层（tid2eid 静态表）：坏槽用
        「未选中且可用」的最高分专家逐个递补。"""
        mask = self._mask(layer)
        gate = w["gate"]
        if (
            not xf.is_cuda
            and xf.shape[0] == 1
            and isinstance(gate, torch.Tensor)
            and gate.dtype == torch.bfloat16
        ):
            gate = w.get("_cpu_gate_group") or w.setdefault(
                "_cpu_gate_group",
                ProjectionGroup((gate,)),
            )
        scores = F.softplus(
            _cccp_lin(
                xf,
                gate.float() if isinstance(gate, torch.Tensor) else gate,
            )
        ).sqrt()
        tid2eid = w.get("tid2eid")
        if tid2eid is not None:
            indices = tid2eid[ids].clone()
            bad = ~mask[indices]
            if bad.any():
                cand = scores.masked_fill(~mask[None, :], -1e30)
                top_cand = cand.topk(cfg.top_k * 2, dim=-1).indices
                for n, k in bad.nonzero().tolist():
                    for c in top_cand[n].tolist():
                        if not (indices[n] == c).any():
                            indices[n, k] = c
                            break
        else:
            fused = _route_post_fused(
                scores,
                w["gate_bias"].float(),
                mask,
                cfg.top_k,
            )
            if fused is not None:
                weights, indices = fused
                if cfg.norm_topk_prob:
                    weights = weights / (
                        weights.sum(dim=-1, keepdim=True) + 1e-20
                    )
                self._record_route_counts(layer, indices)
                return weights * cfg.routed_scaling, indices
            choice = scores + w["gate_bias"].float()
            choice = choice.masked_fill(~mask[None, :], float("-inf"))
            indices = choice.topk(cfg.top_k, dim=-1).indices
        weights = scores.gather(1, indices)
        if cfg.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        self._record_route_counts(layer, indices)
        return weights * cfg.routed_scaling, indices

    def _moe(
        self,
        x: torch.Tensor,
        layer: int,
        ids: torch.Tensor,
        *,
        precomputed_route: tuple[torch.Tensor, torch.Tensor] | None = None,
        precomputed_shared: torch.Tensor | None = None,
        return_parts: bool = False,
    ):
        from .dsv4 import expert_mlp
        c = self.cfg
        w = self.layer(layer)
        B, T, D = x.shape
        output_dtype = compute_dtype(x.device)
        x_rows = x.reshape(B * T, D)
        xf = x_rows.float()
        cfg_obj = self._cfg_obj()
        profile_moe = (
            self._profile_enabled and self.store.man.projection_vq
        )
        moe_events: list[tuple[str, object]] = []

        def mark_moe(name: str) -> None:
            if profile_moe:
                if x_rows.is_cuda:
                    event = torch.cuda.Event(enable_timing=True)
                    event.record(torch.cuda.current_stream(self.device))
                    moe_events.append((name, event))
                else:
                    moe_events.append((name, time.perf_counter()))

        mark_moe("start")
        resident = self._cpu_resident_experts.get(layer)
        cached_layer = None
        if (
            not x_rows.is_cuda
            and resident is not None
            and w.get("tid2eid") is None
            and isinstance(w.get("gate"), torch.Tensor)
            and isinstance(w.get("gate_bias"), torch.Tensor)
            and all(
                isinstance(weight, BlockFP8Weight)
                for weight in (w["sh_w1"], w["sh_w3"], w["sh_w2"])
            )
        ):
            fused_layer = self._cpu_fused_resident_moe.get(layer)
            if fused_layer is None:
                native_layer = self.routed_vq.native_layer(layer)
                from .ops import create_resident_moe_layer

                fused_layer = create_resident_moe_layer(
                    native_layer,
                    tuple(
                        expert
                        for expert in resident
                        if expert is not None
                    ),
                    w["gate"],
                    w["gate_bias"],
                    self._mask(layer),
                    (w["sh_w1"], w["sh_w3"], w["sh_w2"]),
                    activation=self.operator_config.expert_activation,
                    top_k=cfg_obj.top_k,
                    normalize_route=cfg_obj.norm_topk_prob,
                    routed_scaling=cfg_obj.routed_scaling,
                )
                self._cpu_fused_resident_moe[layer] = (
                    fused_layer if fused_layer is not None else False
                )
            if fused_layer is not False:
                from .ops import resident_moe_forward_rows

                fused_output = resident_moe_forward_rows(
                    fused_layer,
                    x_rows,
                    limit=float(c.get("swiglu_limit", 0.0)),
                    activation=self.operator_config.expert_activation,
                    beta=float(c.get("situ_beta", 4.0)),
                    linear_beta=float(c.get("situ_linear_beta", -1.0)),
                )
                if fused_output is not None and fused_output.numel():
                    return fused_output.view(B, T, D).to(output_dtype)
        if B * T == 1 and not x_rows.is_cuda and resident is not None:
            shared_weights = (w["sh_w1"], w["sh_w3"], w["sh_w2"])
            gate = w["gate"]
            if (
                isinstance(gate, Int4Weight)
                and all(
                    isinstance(weight, Int4Weight)
                    for weight in shared_weights
                )
            ):
                cached_layer = self._cpu_moe_layers.get(layer)
                if cached_layer is None:
                    from .cpuext import make_moe_layer_cpu

                    w1, w3, w2 = shared_weights
                    gate_bias = w.get("gate_bias")
                    if gate_bias is None:
                        gate_bias = w.setdefault(
                            "_cpu_gate_bias",
                            torch.zeros(
                                c["n_experts"],
                                dtype=torch.float32,
                                device=x_rows.device,
                            ),
                        )
                    cached_layer = make_moe_layer_cpu(
                        resident,
                        w1.q,
                        w1.s,
                        w3.q,
                        w3.s,
                        w2.q,
                        w2.s,
                        gate.q,
                        gate.s,
                        gate_bias,
                        self._mask(layer),
                        w1.gs,
                        c.get("swiglu_limit", 0.0),
                        cfg_obj.top_k,
                        cfg_obj.norm_topk_prob,
                        cfg_obj.routed_scaling,
                    )
                    if cached_layer is not None:
                        self._cpu_moe_layers[layer] = cached_layer
                if cached_layer is not None and w.get("tid2eid") is None:
                    fused_cpu = cached_layer.forward_learned(x_rows)
                    return fused_cpu.view(B, T, D).to(output_dtype)
        if precomputed_route is None:
            weights, indices = self._route_cccp(
                xf,
                w,
                cfg_obj,
                ids.reshape(-1),
                layer,
            )
        else:
            weights, indices = precomputed_route
            if (
                weights.shape != (B * T, cfg_obj.top_k)
                or indices.shape != (B * T, cfg_obj.top_k)
                or weights.device != x_rows.device
                or indices.device != x_rows.device
            ):
                raise RuntimeError(
                    "fixed FFN packet returned incompatible route buffers"
                )
        mark_moe("route")
        if self.store.man.projection_vq:
            activation = self.operator_config.expert_activation
            limit = float(c.get("swiglu_limit", 0.0))
            if (
                B * T > 1
                and self.device.type == "cpu"
                and not self.routed_vq.compact_full_resident
                and self._prefetch_enabled()
            ):
                # The exact route is already known for this prefill block.
                # Start disk reads for every distinct expert while the shared
                # branch computes; row execution and reduction order stay
                # unchanged, so routing counts/logits remain bit-identical.
                self.routed_vq.prefetch_routes(layer, indices)
            pending_routed = None
            if (
                B * T == 1
                and self._packed_device_pool
            ):
                # 公共双阶段接口先提交 packed DMA，再让默认流计算共享专家；
                # finish_run 仅在 routed kernel 真正读取槽位前建立事件依赖。
                # 这样 RAM→VRAM 与 shared Gate/Up/Down 并行，数学顺序不变。
                pending_routed = self.routed_vq.prepare(
                    layer,
                    x_rows[:1],
                    indices[0],
                    weights[0],
                    activation=activation,
                    activation_beta=float(c.get("situ_beta", 4.0)),
                    activation_linear_beta=c.get("situ_linear_beta"),
                    limit=limit,
                )
            try:
                shared = (
                    precomputed_shared
                    if precomputed_shared is not None
                    else (
                        self._heterogeneous_shared_mlp.run(
                            layer,
                            x_rows,
                            limit=limit,
                        )
                        if (
                            self._heterogeneous_shared_mlp is not None
                            and B * T == 1
                            and pending_routed is not None
                        )
                        else (
                            self._tp_shared_mlp.run(layer, x_rows)
                            if self._tp_shared_mlp is not None and B * T == 1
                            else _shared_expert_mlp_cccp(x_rows, w, limit)
                        )
                    )
                )
            except BaseException:
                if pending_routed is not None:
                    self.routed_vq.cancel(pending_routed)
                raise
            mark_moe("shared")
            if pending_routed is not None:
                routed = self.routed_vq.finish(pending_routed)
                pending_routed = None
            else:
                routed = self.routed_vq.execute(
                    layer,
                    x_rows,
                    indices if B * T > 1 else indices[0],
                    weights if B * T > 1 else weights[0],
                    activation=activation,
                    activation_beta=float(c.get("situ_beta", 4.0)),
                    activation_linear_beta=c.get("situ_linear_beta"),
                    limit=limit,
                )
            mark_moe("routed_batch" if B * T > 1 else "routed")
            # Full-resident packed experts have nothing to prefetch.  Hybrid
            # pools expose the exact host route already resolved for compact
            # slots, so RAM/LRU mode must not pull the same IDs through a
            # second ``tolist`` synchronization.
            if not self._packed_full_gpu and self._prefetch_enabled():
                last_ids = self.routed_vq.last_expert_ids(layer)
                if last_ids is not None:
                    # The public hybrid pool already copied the exact route
                    # once while resolving compact slots.  Reuse that host
                    # metadata instead of adding a second CUDA synchronize.
                    self._prev_ids[layer] = list(last_ids)
                else:
                    self._prev_ids[layer] = [
                        int(expert_id)
                        for expert_id in indices[-1].tolist()
                    ]
            if return_parts and B * T == 1:
                mark_moe("merge")
                if profile_moe:
                    self._moe_profile_records.append(
                        (layer, tuple(moe_events))
                    )
                return (
                    routed.reshape(1, D),
                    shared.reshape(1, D),
                )
            output = (routed.to(shared.dtype) + shared).view(B, T, D).to(
                output_dtype
            )
            mark_moe("merge")
            if profile_moe:
                self._moe_profile_records.append(
                    (layer, tuple(moe_events))
                )
            return output
        if cached_layer is not None:
            fused_cpu = cached_layer.forward(
                x_rows,
                weights[0],
                indices[0],
            )
            return fused_cpu.view(B, T, D).to(output_dtype)
        routed = self.routed_vq.execute(
            layer,
            x_rows,
            indices if B * T > 1 else indices[0],
            weights if B * T > 1 else weights[0],
            activation=self.operator_config.expert_activation,
            activation_beta=float(c.get("situ_beta", 4.0)),
            activation_linear_beta=c.get("situ_linear_beta"),
            limit=float(c.get("swiglu_limit", 0.0)),
        )
        shared = expert_mlp(
            x_rows,
            w["sh_w1"],
            w["sh_w3"],
            w["sh_w2"],
            c.get("swiglu_limit", 0.0),
        )
        return (routed.to(shared.dtype) + shared).view(B, T, D).to(
            output_dtype
        )

    def _hc_decode_workspace(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return one fixed HC workspace for batch-1 CPU/CUDA decode."""
        if (
            hidden.dtype not in (torch.float32, torch.bfloat16)
            or hidden.shape[0] * hidden.shape[1] != 1
        ):
            return None
        width = int(hidden.shape[-1])
        key = (hidden.device, width)
        buffers = self._hc_decode_workspaces.get(key)
        if buffers is None:
            dtype = torch.bfloat16 if hidden.is_cuda else torch.float32
            options = {"dtype": dtype, "device": hidden.device}
            if hidden.is_cuda:
                buffers = (
                    torch.empty((1, width), **options),
                    torch.empty((1, 4), **options),
                    torch.empty((1, 16), **options),
                )
            else:
                buffers = (
                    torch.empty((1, 1, width), **options),
                    torch.empty((1, 1, 4), **options),
                    torch.empty((1, 1, 4, 4), **options),
                )
            self._hc_decode_workspaces[key] = buffers
        return buffers

    def _hc_post_workspace(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return distinct Attention/FFN HC result buffers for decode."""
        if (
            hidden.dtype not in (torch.float32, torch.bfloat16)
            or hidden.shape[0] * hidden.shape[1] != 1
        ):
            return None
        width = int(hidden.shape[-1])
        key = (hidden.device, width)
        buffers = self._hc_post_workspaces.get(key)
        if buffers is None:
            dtype = torch.bfloat16 if hidden.is_cuda else torch.float32
            shape = (1, 1, 4, width)
            buffers = (
                torch.empty(shape, dtype=dtype, device=hidden.device),
                torch.empty(shape, dtype=dtype, device=hidden.device),
            )
            self._hc_post_workspaces[key] = buffers
        return buffers

    def _block(self, h: torch.Tensor, layer: int, ids: torch.Tensor, pos0: int,
               spec: dict | None = None) -> torch.Tensor:
        from .dsv4 import hc_post
        cfg = self._cfg_obj()
        w = self.layer(layer)
        st = self.states[layer]
        profile = self._profile_enabled
        events: list[tuple[str, object]] = []

        def mark(name: str) -> None:
            if profile:
                if h.is_cuda:
                    event = torch.cuda.Event(enable_timing=True)
                    event.record(torch.cuda.current_stream(self.device))
                    events.append((name, event))
                else:
                    events.append((name, time.perf_counter()))

        if h.is_cuda and ids.shape[-1] > 1:
            prepare_prefill_layer = getattr(
                self.routed_vq,
                "prepare_prefill_layer",
                None,
            )
            if callable(prepare_prefill_layer):
                prepare_prefill_layer(layer)
        mark("start")
        hc_workspace = self._hc_decode_workspace(h)
        hc_post_workspace = self._hc_post_workspace(h)
        # 跨层专家预取（B2）：用上一 token 本层路由结果提前装填（时序局部性），
        # attention 计算与专家 读盘/DMA 重叠；未命中回退正常加载，无正确性风险
        prev = self._prev_ids.get(layer)
        if (
            prev
            and ids.shape[-1] == 1
            and self._prefetch_enabled()
            and not self._token_prefetch_enabled()
        ):
            # Extreme staging intentionally uses only this bounded per-layer
            # window.  Normal hybrid decode already queued every layer once at
            # token start; submitting the same 43 routes again doubles host
            # executor traffic without adding useful overlap.
            self.routed_vq.prefetch_routes(layer, prev)
        residual = h
        canonical_short_decode = bool(
            getattr(self, "_canonical_short_decode", False)
        )
        attention_packet = (
            self._single_gpu_attention_graphs.get(int(layer))
            if (
                h.is_cuda
                and h.shape[0] * h.shape[1] == 1
                and spec is None
                and not canonical_short_decode
            )
            else None
        )
        projected_attention = None
        if attention_packet is not None:
            attention_packet["source"].copy_(h)
            rope = self._rope(layer)
            attention_packet["cos"].copy_(rope.cos[pos0:pos0 + 1])
            attention_packet["sin"].copy_(rope.sin[pos0:pos0 + 1])
            packet_outputs = attention_packet["graph"].replay()
            y, post, comb = packet_outputs[:3]
            projected_attention = packet_outputs[3:]
        else:
            y, post, comb = _hc_pre_norm_cccp(
                h,
                w["hc_attn_fn"],
                w["hc_attn_scale"],
                w["hc_attn_base"],
                w["attn_norm"],
                cfg,
                output_buffers=hc_workspace,
            )
        mark("attn_hc_norm")
        a = self._attn_batch(
            y,
            layer,
            pos0,
            spec,
            static_projection_outputs=projected_attention,
        )
        mark("attention")
        h = hc_post(
            a,
            residual,
            post,
            comb,
            output=(
                None if hc_post_workspace is None else hc_post_workspace[0]
            ),
        )
        residual = h
        ffn_packet = (
            self._single_gpu_ffn_graphs.get(int(layer))
            if (
                h.is_cuda
                and h.shape[0] * h.shape[1] == 1
                and spec is None
                and not canonical_short_decode
            )
            else None
        )
        precomputed_route = None
        precomputed_shared = None
        if ffn_packet is not None:
            ffn_packet["source"].copy_(h)
            packet_outputs = ffn_packet["graph"].replay()
            if ffn_packet.get("includes_shared", True):
                y, post, comb, precomputed_shared = packet_outputs[:4]
                precomputed_route = packet_outputs[4:6]
            else:
                y, post, comb = packet_outputs[:3]
                precomputed_route = packet_outputs[3:5]
        else:
            y, post, comb = _hc_pre_norm_cccp(
                h,
                w["hc_ffn_fn"],
                w["hc_ffn_scale"],
                w["hc_ffn_base"],
                w["ffn_norm"],
                cfg,
                output_buffers=hc_workspace,
            )
        mark("ffn_hc_norm")
        moe_result = self._moe(
            y,
            layer,
            ids,
            precomputed_route=precomputed_route,
            precomputed_shared=precomputed_shared,
            return_parts=(
                y.is_cuda
                and y.shape[0] * y.shape[1] == 1
                and self.store.man.projection_vq
            ),
        )
        mark("moe")
        if isinstance(moe_result, tuple):
            routed, shared = moe_result
            output = _hyper_connection_post_moe(
                routed,
                shared,
                residual,
                post,
                comb,
                output=(
                    None
                    if hc_post_workspace is None
                    else hc_post_workspace[1]
                ),
            )
            if output is None:
                combined = (
                    routed.to(shared.dtype) + shared
                ).view(1, 1, -1)
                output = hc_post(
                    combined,
                    residual,
                    post,
                    comb,
                    output=(
                        None
                        if hc_post_workspace is None
                        else hc_post_workspace[1]
                    ),
                )
        else:
            output = hc_post(
                moe_result,
                residual,
                post,
                comb,
                output=(
                    None
                    if hc_post_workspace is None
                    else hc_post_workspace[1]
                ),
            )
        output = output.to(compute_dtype(h.device))
        mark("ffn_hc_post")
        if profile:
            self._profile_records.append((layer, tuple(events)))
        return output

    def start_profile(self) -> None:
        """Start one-token CPU/CUDA stage profiling for the benchmark CLI."""
        self._profile_records = []
        self._moe_profile_records = []
        self.last_layer_profile = {}
        self._tp_stage_profile = {}
        if self.device.type == "cpu" and self.store.man.projection_vq:
            from .cpuext import (
                reset_block_fp8_gemv_profile,
                reset_resident_moe_phase_profile,
                reset_resident_projection_profile,
                reset_three_projection_phase_profile,
            )

            reset_three_projection_phase_profile()
            reset_resident_moe_phase_profile()
            reset_resident_projection_profile()
            reset_block_fp8_gemv_profile()
        self._profile_enabled = True

    def finish_profile(self) -> dict[str, object]:
        """Finish a CLI stage probe and aggregate primary-stream events."""
        self._profile_enabled = False
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        def elapsed_ms(events: tuple, index: int) -> float:
            current = events[index][1]
            following = events[index + 1][1]
            if self.device.type == "cuda":
                return float(current.elapsed_time(following))
            return (float(following) - float(current)) * 1000.0
        stage_names = (
            "attn_hc_norm",
            "attention",
            "ffn_hc_norm",
            "moe",
            "ffn_hc_post",
        )
        totals = {name: 0.0 for name in stage_names}
        layers: list[dict[str, float | int]] = []
        for layer, events in self._profile_records:
            values: dict[str, float | int] = {"layer": layer}
            for index, name in enumerate(stage_names):
                milliseconds = elapsed_ms(events, index)
                values[f"{name}_ms"] = milliseconds
                totals[name] += milliseconds
            values["total_ms"] = sum(
                float(values[f"{name}_ms"])
                for name in stage_names
            )
            layers.append(values)
        top_layers = sorted(
            layers,
            key=lambda item: float(item["total_ms"]),
            reverse=True,
        )[:8]
        moe_stage_names = ("route", "shared", "routed", "merge")
        moe_totals = {name: 0.0 for name in moe_stage_names}
        moe_layers: list[dict[str, float | int]] = []
        for layer, events in self._moe_profile_records:
            values: dict[str, float | int] = {"layer": layer}
            for index, name in enumerate(moe_stage_names):
                milliseconds = elapsed_ms(events, index)
                values[f"{name}_ms"] = milliseconds
                moe_totals[name] += milliseconds
            moe_layers.append(values)
        result: dict[str, object] = {
            "layer_count": len(layers),
            "totals_ms": totals,
            "covered_ms": sum(totals.values()),
            "top_layers": top_layers,
            "layers": layers,
            "moe_totals_ms": moe_totals,
            "moe_layers": moe_layers,
        }
        if self._tp_route_buffers and self.store.man.projection_vq:
            gate_up_counts: dict[str, int] = {}
            down_counts: dict[str, int] = {}
            triple_counts: dict[str, int] = {}
            selected_count = 0
            for layer, buffers in sorted(self._tp_route_buffers.items()):
                route_ids = buffers[2][0].detach().reshape(-1).tolist()
                for expert_id in route_ids:
                    layouts = self.store.man.projection_layouts(
                        int(layer),
                        int(expert_id),
                    )
                    gate_up = (
                        f"{layouts['gate']}+{layouts['up']}"
                    )
                    down = str(layouts["down"])
                    triple = f"{gate_up}->{down}"
                    gate_up_counts[gate_up] = (
                        gate_up_counts.get(gate_up, 0) + 1
                    )
                    down_counts[down] = down_counts.get(down, 0) + 1
                    triple_counts[triple] = (
                        triple_counts.get(triple, 0) + 1
                    )
                    selected_count += 1
            result["routed_projection_layouts"] = {
                "selected_experts": selected_count,
                "gate_up": dict(sorted(
                    gate_up_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )),
                "down": dict(sorted(
                    down_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )),
                "triples": dict(sorted(
                    triple_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )),
            }
        if self._tp_stage_profile:
            result["tensor_parallel"] = self._tp_stage_profile
        if self.device.type == "cpu" and self.store.man.projection_vq:
            from .cpuext import (
                block_fp8_gemv_profile,
                resident_moe_phase_profile,
                resident_projection_profile,
                three_projection_phase_profile,
            )

            result["packed_three_projection"] = (
                three_projection_phase_profile()
            )
            result["block_fp8_gemv"] = block_fp8_gemv_profile()
            result["resident_projection"] = resident_projection_profile()
            result["resident_moe_fused"] = resident_moe_phase_profile()
        self.last_layer_profile = result
        self._profile_records = []
        self._moe_profile_records = []
        return result

    def _attn_batch(
        self,
        y: torch.Tensor,
        layer: int,
        pos0: int,
        spec: dict | None,
        *,
        tp_context: dict | None = None,
        static_projection_outputs: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """批量增量注意力（投机验证用）：T 个 token（positions pos0..pos0+T-1）一次前向。

        数学与 CCCP.dsv4.attn_decode 逐步等价：环形窗自因果掩码（含窗约束）、
        压缩槽可见性 n < (qpos+1)//ratio、sink 在分母、输出末 64 维反旋转。
        Compressor 为每 token 顺序状态机（逐 token 调用 compressor_decode），
        并把每步的 ckv/cscore 快照记入 spec["steps"]（供 spec_commit 回滚）。
        """
        from . import dsv4 as _d
        from .dsv4 import rope_apply
        from .dsv4indexer import (
            hadamard_rotate,
            indexer_scores,
            select_index_positions,
        )

        cfg = (
            tp_context["cfg"]
            if tp_context is not None
            else self._cfg_obj()
        )
        w = (
            tp_context["weights"]
            if tp_context is not None
            else self.layer(layer)
        )
        st = (
            tp_context["state"]
            if tp_context is not None
            else self.states[layer]
        )
        cache = (
            tp_context["rope"]
            if tp_context is not None
            else self._rope(layer)
        )
        ratio = (
            int(tp_context["ratio"])
            if tp_context is not None
            else self.ratios[layer]
        )
        H, hd, rd = cfg.n_heads, cfg.head_dim, cfg.qk_rope_head_dim
        win = cfg.sliding_window
        B, T, _ = y.shape
        dev = y.device
        single_main_decode = (
            T == 1
            and spec is None
            and os.environ.get("CCCP_SINGLE_TOKEN_ATTN_FAST", "1") != "0"
        )
        cpu_attention_outputs = None
        projected_qkv = None
        static_projection = (
            self._single_gpu_attention_graphs.get(int(layer))
            if (
                tp_context is None
                and static_projection_outputs is None
                and not bool(getattr(self, "_canonical_short_decode", False))
            )
            else None
        )
        projection_graph = (
            tp_context.get("projection_graph")
            if tp_context is not None
            else (
                None
                if static_projection is None
                else static_projection["graph"]
            )
        )
        if single_main_decode and static_projection_outputs is not None:
            graph_outputs = static_projection_outputs
            graph_parts = (
                tp_context["projection_graph_parts"]
                if tp_context is not None
                else self._single_gpu_attention_graphs[
                    int(layer)
                ]["parts"]
            )
            projected_qkv = graph_outputs[:3]
            cpu_attention_outputs = {}
            offset = 3
            for name in graph_parts:
                cpu_attention_outputs[
                    "compressor" if name == "cmp" else "indexer"
                ] = graph_outputs[offset:offset + 2]
                offset += 2
        elif single_main_decode and projection_graph is not None:
            if tp_context is not None:
                graph_cos = tp_context["projection_graph_cos"]
                graph_sin = tp_context["projection_graph_sin"]
                graph_parts = tp_context["projection_graph_parts"]
            else:
                static_projection["source"].copy_(y)
                graph_cos = static_projection["cos"]
                graph_sin = static_projection["sin"]
                graph_parts = static_projection["parts"]
            graph_cos.copy_(cache.cos[pos0:pos0 + 1])
            graph_sin.copy_(cache.sin[pos0:pos0 + 1])
            graph_outputs = projection_graph.replay()
            projected_qkv = graph_outputs[:3]
            cpu_attention_outputs = {}
            offset = 3
            for name in graph_parts:
                cpu_attention_outputs[
                    "compressor" if name == "cmp" else "indexer"
                ] = graph_outputs[offset:offset + 2]
                offset += 2
        if single_main_decode and dev.type == "cpu":
            # All projections consuming the same normalized token enter one
            # format-driven resident executor.  It accepts a heterogeneous
            # BF16/block-FP8 list, converts the activation once and returns
            # fixed-address per-projection outputs in source order.
            projection_weights = [w["wq_a"], w["wkv"]]
            projection_groups = [("qkv", 2)]
            if ratio:
                projection_weights.extend((
                    w["cmp"]["wkv"],
                    w["cmp"]["wgate"],
                ))
                projection_groups.append(("compressor", 2))
            if ratio == 4:
                projection_weights.extend((
                    w["indexer"]["wkv"],
                    w["indexer"]["wgate"],
                ))
                projection_groups.append(("indexer", 2))
            projection_group = w.get("_cpu_attention_input_group")
            if projection_group is None:
                projection_group = w.setdefault(
                    "_cpu_attention_input_group",
                    ProjectionGroup(tuple(projection_weights)),
                )
            parts = projection_group.resident_forward_parts(
                y.flatten(0, 1)
            )
            if parts is not None:
                cpu_attention_outputs = {}
                offset = 0
                for name, count in projection_groups:
                    cpu_attention_outputs[name] = tuple(
                        parts[offset:offset + count]
                    )
                    offset += count
        if projected_qkv is None:
            qr, q, kv = _qkv_cccp(
                y,
                w,
                cfg,
                cache,
                pos0,
                cpu_outputs=(
                    cpu_attention_outputs["qkv"]
                    if cpu_attention_outputs is not None
                    else None
                ),
            )
        else:
            qr, q, kv = projected_qkv
        scale = hd ** -0.5
        poss = (
            None
            if single_main_decode
            else torch.arange(pos0, pos0 + T, device=dev)
        )

        if ratio:
            steps = (
                spec.setdefault("steps", {}).setdefault(layer, [])
                if spec is not None
                else None
            )
            if T > 1:
                ck, compressed_start, captured = _compressor_prefill_cccp(
                    y,
                    w["cmp"],
                    ratio,
                    hd,
                    rd,
                    cache,
                    cfg.rms_eps,
                    st,
                    pos0,
                    capture_steps=steps is not None,
                )
                if ck is not None:
                    st["compressed"].write_many(compressed_start, ck)
                if steps is not None:
                    steps.extend(captured)
            else:
                p = pos0
                rope_pos = max(0, p + 1 - ratio)
                cos = cache.cos[rope_pos].view(1, 1, -1)
                sin = cache.sin[rope_pos].view(1, 1, -1)
                ck = _compressor_decode_cccp(
                    y,
                    w["cmp"],
                    ratio,
                    hd,
                    rd,
                    cos,
                    sin,
                    cfg.rms_eps,
                    st,
                    p,
                    cpu_outputs=(
                        cpu_attention_outputs["compressor"]
                        if cpu_attention_outputs is not None
                        else None
                    ),
                )
                if ck is not None:
                    st["compressed"].write(p // ratio, ck[:, 0])
                if steps is not None:
                    steps.append((st["ckv"].clone(), st["cscore"].clone()))

        compressed_count = st["compressed"].length if ratio else 0
        direct_compressed_prefix = False
        if ratio == 4:
            indexer = st["indexer"]
            iw = w["indexer"]
            index_steps = (
                spec.setdefault("indexer_steps", {}).setdefault(layer, [])
                if spec is not None
                else None
            )
            if T > 1:
                index_pooled, index_start, captured = (
                    _compressor_prefill_cccp(
                        y,
                        iw,
                        indexer.ratio,
                        indexer.head_dim,
                        indexer.rope_dim,
                        cache,
                        cfg.rms_eps,
                        indexer.compressor_state,
                        pos0,
                        capture_steps=index_steps is not None,
                    )
                )
                if index_pooled is not None:
                    indexer.keys.write_many(
                        index_start, hadamard_rotate(index_pooled)
                    )
                if index_steps is not None:
                    index_steps.extend(captured)
            else:
                p = pos0
                rope_pos = max(0, p + 1 - ratio)
                index_pooled = _compressor_decode_cccp(
                    y,
                    iw,
                    indexer.ratio,
                    indexer.head_dim,
                    indexer.rope_dim,
                    cache.cos[rope_pos].view(1, 1, -1),
                    cache.sin[rope_pos].view(1, 1, -1),
                    cfg.rms_eps,
                    indexer.compressor_state,
                    p,
                    cpu_outputs=(
                        cpu_attention_outputs["indexer"]
                        if cpu_attention_outputs is not None
                        else None
                    ),
                )
                if index_pooled is not None:
                    indexer.keys.write(
                        p // indexer.ratio,
                        hadamard_rotate(index_pooled)[:, 0],
                    )
                if index_steps is not None:
                    index_steps.append(
                        (indexer.ckv.clone(), indexer.cscore.clone())
                    )
            if indexer.keys.length != compressed_count:
                raise RuntimeError(
                    "Indexer/main compressed KV length mismatch: "
                    f"{indexer.keys.length} != {compressed_count}"
                )
            visible_count = (
                compressed_count
                if single_main_decode
                else (poss + 1) // ratio
            )
            if compressed_count <= cfg.index_topk:
                direct_compressed_prefix = (
                    single_main_decode
                    and compressed_count <= st["compressed"].page_items
                    and os.environ.get("CCCP_DIRECT_KV_PREFIX", "1") != "0"
                )
                if direct_compressed_prefix:
                    selected_positions = None
                    selected_valid = None
                else:
                    selected_positions = torch.arange(
                        compressed_count, device=dev, dtype=torch.long
                    ).view(1, 1, -1).expand(B, T, -1)
                    selected_valid = (
                        selected_positions < visible_count.view(1, T, 1)
                    )
            else:
                iq = _linear(qr, iw["wq_b"]).view(
                    B, T, cfg.index_n_heads, cfg.index_head_dim
                )
                cos = cache.cos[pos0:pos0 + T].view(1, T, 1, -1)
                sin = cache.sin[pos0:pos0 + T].view(1, T, 1, -1)
                iq[..., cfg.index_head_dim - rd:] = rope_apply(
                    iq[..., cfg.index_head_dim - rd:], cos, sin
                )
                iq = hadamard_rotate(iq.to(compute_dtype(dev)))
                if compressed_count <= indexer.keys.page_items:
                    all_index_keys = indexer.keys.contiguous_prefix(
                        compressed_count
                    )
                else:
                    all_index_keys = indexer.keys.gather(
                        torch.arange(
                            compressed_count,
                            device=dev,
                            dtype=torch.long,
                        )
                    )
                index_weights = _linear(y, iw["weights_proj"]) * (
                    cfg.index_head_dim ** -0.5
                    * cfg.index_n_heads ** -0.5
                )
                selection_scores = indexer_scores(
                    iq, all_index_keys, index_weights
                )
                if not single_main_decode:
                    candidate_positions = torch.arange(
                        compressed_count, device=dev
                    )
                    selection_scores = selection_scores.masked_fill(
                        candidate_positions.view(1, 1, -1)
                        >= visible_count.view(1, T, 1),
                        float("-inf"),
                    )
                selected_positions = select_index_positions(
                    selection_scores, cfg.index_topk
                ).long()
                del selection_scores
                del iq, index_weights, all_index_keys
                selected_valid = (
                    None
                    if single_main_decode
                    else selected_positions
                    < visible_count.view(1, T, 1)
                )
        elif compressed_count:
            direct_compressed_prefix = (
                single_main_decode
                and compressed_count <= st["compressed"].page_items
                and os.environ.get("CCCP_DIRECT_KV_PREFIX", "1") != "0"
            )
            if direct_compressed_prefix:
                selected_positions = None
                selected_valid = None
            else:
                selected_positions = torch.arange(
                    compressed_count, device=dev, dtype=torch.long
                ).view(1, 1, -1).expand(B, T, -1)
                selected_valid = (
                    None
                    if single_main_decode
                    else selected_positions < (
                        (poss + 1) // ratio
                    ).view(1, T, 1)
                )
        else:
            if single_main_decode:
                selected_positions = None
                selected_valid = None
            else:
                selected_positions = torch.empty(
                    B, T, 0, dtype=torch.long, device=dev
                )
                selected_valid = torch.empty(
                    B, T, 0, dtype=torch.bool, device=dev
                )

        # Preserve the pre-commit ring.  Full Prefill remains one layer-first
        # outer block, while sparse Attention gathers bounded query tiles from
        # this snapshot instead of materialising [T, top-k, head_dim] (about
        # 3.9 GiB for a 4K DSV4 block) on every layer.
        prefill_ring_values = None
        prefill_ring_positions = None
        if T > 1:
            prefill_ring_values = st["kv"].clone()
            prefill_ring_positions = st["win_pos"].clone()

        if direct_compressed_prefix:
            selected_values = (
                st["compressed"]
                .contiguous_prefix(compressed_count)
                .unsqueeze(1)
                .to(q.dtype)
            )
        elif (
            T == 1
            and selected_positions is not None
            and selected_positions.numel()
        ):
            selected_values = st["compressed"].gather_batched(
                selected_positions.clamp_min(0)
            ).to(q.dtype)
        elif T > 1:
            selected_values = None
        elif single_main_decode:
            selected_values = st["kv"][:, :0].unsqueeze(1)
        else:
            selected_values = torch.empty(
                B, T, 0, hd, device=dev, dtype=q.dtype
            )

        # Commit this chunk so fused T=1 decode and subsequent calls see it.
        if single_main_decode:
            slot = pos0 % win
            st["kv"][:, slot] = kv[:, 0]
            st["win_pos"][:, slot] = pos0
        else:
            recent = min(T, win)
            slots = poss[-recent:] % win
            st["kv"][:, slots] = kv[:, -recent:]
            st["win_pos"][:, slots] = poss[-recent:]

        out_cos = cache.cos[pos0:pos0 + T].view(1, T, 1, -1)
        out_sin = cache.sin[pos0:pos0 + T].view(1, T, 1, -1)
        if T == 1 and dev.type == "cpu":
            from .cpuext import attention_decode_cpu

            fused_cpu = attention_decode_cpu(
                q[:, 0],
                st["kv"],
                st["win_pos"],
                selected_values[:, 0],
                w["attn_sink"],
                out_cos,
                out_sin,
                scale,
            )
            if fused_cpu is not None:
                return _d._o_proj_hook(
                    fused_cpu.unsqueeze(1).flatten(2), w, cfg
                )
        if T == 1 and q.is_cuda:
            from .ops import attention_step

            fused = attention_step(
                "sliding_compressed_mqa_decode",
                q.device.type,
                query=q[:, 0],
                window_kv=st["kv"],
                window_positions=st["win_pos"],
                compressed_kv=selected_values[:, 0],
                sink=w["attn_sink"],
                cos=out_cos,
                sin=out_sin,
                scale=scale,
            )
            if fused is not None:
                return _d._o_proj_hook(
                    fused.unsqueeze(1).flatten(2), w, cfg
                )

        if T > 1:
            assert prefill_ring_values is not None
            assert prefill_ring_positions is not None
            selected_count = (
                int(selected_positions.shape[2])
                if selected_positions is not None
                else 0
            )
            flashmla_prefill = (
                q.is_cuda
                and torch.version.hip is None
                and int(hd) == 512
                and B == 1
                and _flashmla_prefill_batch_enabled(
                    T,
                    runner_available="sparse_runner" in st,
                )
                and os.environ.get("CCCP_FLASHMLA_PREFILL", "1") != "0"
            )
            if flashmla_prefill:
                compressed_bank = (
                    st["compressed"].contiguous_prefix(compressed_count)
                    if compressed_count
                    else None
                )
                flash_kv, flash_indices = (
                    _flashmla_prefill_kv_and_indices(
                        prefill_ring_values,
                        prefill_ring_positions,
                        kv,
                        pos0,
                        compressed_bank,
                        selected_positions,
                        selected_valid,
                    )
                )
                sparse_output = st["sparse_runner"].prefill(
                    query=q[0].to(torch.bfloat16).contiguous(),
                    key_cache=flash_kv,
                    indices=flash_indices,
                    sink=w["attn_sink"].float().contiguous(),
                    scale=float(scale),
                )
                if sparse_output.shape != (T, H, hd):
                    raise RuntimeError(
                        "FlashMLA sparse Prefill returned invalid output"
                    )
                if not getattr(
                    self, "_flashmla_prefill_executor_announced", False
                ):
                    print(
                        "[cccp-prefill] attention="
                        "flashmla.sparse-prefill-fused; "
                        f"outer tokens={T}; topk={flash_indices.shape[-1]}; "
                        "gathered_values=0",
                        flush=True,
                    )
                    self._flashmla_prefill_executor_announced = True
                o = sparse_output.unsqueeze(0)
                o[..., hd - rd:] = rope_apply(
                    o[..., hd - rd:], out_cos, out_sin, inverse=True
                )
                return _d._o_proj_hook(o.flatten(2), w, cfg)

            if (
                q.is_cuda
                and torch.version.hip is None
                and torch.cuda.get_device_capability(q.device) == (9, 0)
                and T >= 512
            ):
                reason = st.get(
                    "sparse_splitkv_unavailable_reason",
                    "FlashMLA sparse Prefill backend unavailable",
                )
                raise RuntimeError(
                    "H20/SM90 长批 Prefill 必须使用 FlashMLA sparse "
                    f"Prefill：{reason}"
                )
            try:
                process_limit_gb = float(
                    os.environ.get("CCCP_VRAM_LIMIT_GB", "0") or 0
                )
            except (TypeError, ValueError):
                process_limit_gb = 0.0
            if process_limit_gb <= 0 and q.is_cuda:
                process_limit_gb = (
                    torch.cuda.get_device_properties(q.device).total_memory
                    / 2**30
                )
            if process_limit_gb >= 30.0:
                attention_batch = T
            else:
                workspace_mib = max(
                    128,
                    int(os.environ.get(
                        "CCCP_PREFILL_ATTENTION_WORKSPACE_MIB", "768"
                    )),
                )
                key_count = max(1, int(win) + selected_count)
                # Gathered BF16 values plus score/softmax tensors.  The
                # estimate is deliberately conservative and aligned so every
                # launch still processes a useful tensor-core-sized batch.
                bytes_per_query = max(
                    1,
                    key_count * int(hd) * q.element_size()
                    + int(H) * key_count * 12
                    + int(H) * int(hd) * 4,
                )
                attention_batch = min(
                    T,
                    max(32, workspace_mib * 2**20 // bytes_per_query),
                )
                if attention_batch >= 32:
                    attention_batch = max(32, attention_batch // 32 * 32)
            if not getattr(self, "_prefill_attention_executor_announced", False):
                print(
                    "[cccp-prefill] attention="
                    "cuda.sparse-gather-batched; "
                    f"outer tokens={T}; query batch={attention_batch}; "
                    "single-token projection=forbidden",
                    flush=True,
                )
                self._prefill_attention_executor_announced = True
            o = torch.empty(
                B, T, H, hd, dtype=q.dtype, device=dev
            )
            for query_start in range(0, T, attention_batch):
                query_stop = min(T, query_start + attention_batch)
                query_count = query_stop - query_start
                query = q[:, query_start:query_stop]
                raw_values, raw_allow = _prefill_sliding_window(
                    prefill_ring_values,
                    prefill_ring_positions,
                    kv,
                    pos0,
                    query_start=query_start,
                    query_count=query_count,
                )
                if selected_count:
                    block_positions = selected_positions[
                        :, query_start:query_stop
                    ]
                    block_selected = st["compressed"].gather_batched(
                        block_positions.clamp_min(0)
                    ).to(q.dtype)
                    block_valid = (
                        selected_valid[:, query_start:query_stop]
                        if selected_valid is not None
                        else torch.ones(
                            B,
                            1,
                            selected_count,
                            device=dev,
                            dtype=torch.bool,
                        )
                    )
                else:
                    block_selected = torch.empty(
                        B,
                        query_count,
                        0,
                        hd,
                        device=dev,
                        dtype=q.dtype,
                    )
                    block_valid = None
                raw_scores = torch.einsum(
                    "bthd,btsd->bhts", query * scale, raw_values
                ).masked_fill(
                    ~raw_allow.unsqueeze(1), float("-inf")
                )
                if selected_count:
                    compressed_scores = torch.einsum(
                        "bthd,btkd->bhtk", query * scale, block_selected
                    ).masked_fill(
                        ~block_valid.unsqueeze(1), float("-inf")
                    )
                else:
                    compressed_scores = torch.empty(
                        B, H, query_count, 0, device=dev, dtype=q.dtype
                    )
                scores = torch.cat(
                    [raw_scores, compressed_scores], dim=-1
                ).float()
                maximum = scores.amax(dim=-1)
                exponential = (
                    scores - maximum.unsqueeze(-1)
                ).exp()
                denominator = exponential.sum(dim=-1) + (
                    w["attn_sink"].view(1, -1, 1) - maximum
                ).exp()
                probability = (
                    exponential / denominator.unsqueeze(-1)
                ).to(raw_values.dtype)
                block_output = torch.einsum(
                    "bhts,btsd->bthd",
                    probability[..., :win],
                    raw_values,
                )
                if selected_count:
                    block_output += torch.einsum(
                        "bhtk,btkd->bthd",
                        probability[..., win:],
                        block_selected,
                    )
                o[:, query_start:query_stop].copy_(block_output)
            o[..., hd - rd:] = rope_apply(
                o[..., hd - rd:], out_cos, out_sin, inverse=True
            )
            return _d._o_proj_hook(o.flatten(2), w, cfg)

        if selected_valid is None:
            selected_valid = torch.ones(
                B,
                1,
                selected_values.shape[2],
                device=dev,
                dtype=torch.bool,
            )
        if poss is None:
            poss = torch.arange(pos0, pos0 + T, device=dev)

        # The T=1 fused path above reads the committed ring directly.  If it
        # cannot run, the fallback must use that same ring without appending
        # the just-committed token a second time.
        if T == 1:
            raw_values = st["kv"]
            raw_positions = st["win_pos"]
        if T > 1:
            raw_scores = torch.einsum(
                "bthd,btsd->bhts", q * scale, raw_values
            )
        else:
            raw_scores = torch.einsum(
                "bthd,bsd->bhts", q * scale, raw_values
            )
            raw_allow = (
                (raw_positions.unsqueeze(1) >= 0)
                & (raw_positions.unsqueeze(1) <= poss.view(1, T, 1))
                & (raw_positions.unsqueeze(1) > poss.view(1, T, 1) - win)
            )
        raw_scores = raw_scores.masked_fill(
            ~raw_allow.unsqueeze(1), float("-inf")
        )
        if selected_values.shape[2]:
            compressed_scores = torch.einsum(
                "bthd,btkd->bhtk", q * scale, selected_values
            ).masked_fill(~selected_valid.unsqueeze(1), float("-inf"))
        else:
            compressed_scores = torch.empty(
                B, H, T, 0, device=dev, dtype=q.dtype
            )

        scores = torch.cat([raw_scores, compressed_scores], dim=-1).float()
        m = scores.amax(dim=-1)
        e = (scores - m.unsqueeze(-1)).exp()
        denom = e.sum(dim=-1) + (w["attn_sink"].view(1, -1, 1) - m).exp()
        probs = (
            e / denom.unsqueeze(-1)
        ).to(raw_values.dtype)
        raw_width = raw_values.shape[2] if T > 1 else raw_values.shape[1]
        o = (
            torch.einsum(
                "bhts,btsd->bthd",
                probs[..., :raw_width],
                raw_values,
            )
            if T > 1
            else torch.einsum(
                "bhts,bsd->bthd",
                probs[..., :raw_width],
                raw_values,
            )
        )
        if selected_values.shape[2]:
            o += torch.einsum(
                "bhtk,btkd->bthd", probs[..., raw_width:], selected_values
            )
        o[..., hd - rd:] = rope_apply(
            o[..., hd - rd:], out_cos, out_sin, inverse=True
        )
        return _d._o_proj_hook(o.flatten(2), w, cfg)

    def _spec_snapshot(self, pos0: int, T: int) -> dict:
        """验证前快照各层将被触碰的状态：环槽（值+win_pos）、压缩槽、
        以及 compressor 每步状态容器（在 _attn_batch 中逐步填充）。"""
        cfg = self._cfg_obj()
        win = cfg.sliding_window
        poss = torch.arange(pos0, pos0 + T, device=self.device)
        spec = {
            "pos0": pos0,
            "T": T,
            "pre": [],
            "steps": {},
            "indexer_steps": {},
        }
        for i in range(cfg.n_layers):
            st = self.states[i]
            slots = poss % win
            pre = {"slots": slots.clone(),
                   "kv": st["kv"][:, slots].clone(),
                   "win_pos": st["win_pos"][:, slots].clone()}
            ratio = self.ratios[i]
            if ratio:
                cn = torch.unique(poss // ratio)
                pre["cslots"] = cn
                pre["compressed_length"] = st["compressed"].length
                pre["compressed_values"] = st["compressed"].gather(cn).clone()
                spec["steps"][i] = []
                if ratio == 4:
                    indexer = st["indexer"]
                    pre["indexer_length"] = indexer.keys.length
                    pre["indexer_values"] = indexer.keys.gather(cn).clone()
                    spec["indexer_steps"][i] = []
            spec["pre"].append(pre)
        return spec

    def spec_commit(self, keep: int) -> None:
        """验证后按接受前缀截断：恢复被拒位置的环槽/压缩槽，compressor 状态
        回滚到「处理完 position keep-1」的快照，model.pos = keep。"""
        spec = getattr(self, "_spec", None)
        assert spec is not None, "spec_commit 前须先 forward_verify"
        pos0, T = spec["pos0"], spec["T"]
        cfg = self._cfg_obj()
        win = cfg.sliding_window
        a = keep - pos0 - 1                     # 最末保留 token 的批内下标
        assert -1 <= a < T
        for i in range(cfg.n_layers):
            st = self.states[i]
            pre = spec["pre"][i]
            for j in range(a + 1, T):           # 被拒位置：恢复旧环槽内容
                st["kv"][:, pre["slots"][j]] = pre["kv"][:, j]
                st["win_pos"][:, pre["slots"][j]] = pre["win_pos"][:, j]
            if self.ratios[i]:
                ratio = self.ratios[i]
                for k, n in enumerate(pre["cslots"].tolist()):
                    if (n + 1) * ratio - 1 >= keep:     # 池化完成于被拒位置 → 恢复
                        st["compressed"].write(
                            n, pre["compressed_values"][:, k]
                        )
                target_length = max(pre["compressed_length"], keep // ratio)
                st["compressed"].truncate(target_length)
                ckv, cscore = spec["steps"][i][a]
                st["ckv"].copy_(ckv)
                st["cscore"].copy_(cscore)
                if ratio == 4:
                    indexer = st["indexer"]
                    for k, n in enumerate(pre["cslots"].tolist()):
                        if (n + 1) * ratio - 1 >= keep:
                            indexer.keys.write(
                                n, pre["indexer_values"][:, k]
                            )
                    indexer_length = max(
                        pre["indexer_length"], keep // ratio
                    )
                    indexer.keys.truncate(indexer_length)
                    index_ckv, index_cscore = spec["indexer_steps"][i][a]
                    indexer.ckv.copy_(index_ckv)
                    indexer.cscore.copy_(index_cscore)
        self._spec = None
        self.pos = keep

    @torch.no_grad()
    def forward_verify(self, ids_list: list[int], pos0: int) -> tuple[torch.Tensor, torch.Tensor]:
        """投机验证：一次批量前向处理 [t1, d1..dk]（positions pos0..pos0+T-1）。

        返回 (logits [T, vocab], main_hidden [T, 3·hidden])；KV 状态前进到
        pos0+T（随后由 spec_commit(keep) 截断到接受前缀）。main_hidden 为
        DSPARK_TARGETS 各层 hc 均值隐态的拼接（供 DSpark 草稿头）。
        """
        from .dsv4 import hc_head
        cfg = self._cfg_obj()
        assert self.states is not None and pos0 > 0, "forward_verify 前须先 prefill"
        T = len(ids_list)
        ids = torch.tensor([ids_list], device=self.device).long()
        self.ensure_position(pos0 + T - 1)
        if self._prev_ids and self._token_prefetch_enabled():
            for l, es in self._prev_ids.items():
                self.routed_vq.prefetch_routes(l, es)
        self._spec = self._spec_snapshot(pos0, T)
        h = self._embed(ids).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        mh = []
        for i in range(cfg.n_layers):
            h = self._block(h, i, ids, pos0, self._spec)
            if i in self.DSPARK_TARGETS:
                mh.append(h.mean(dim=2))
        y = hc_head(h, *self._hc_head_w(), cfg)
        y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
        logits = _linear(y, self.w("head.weight")).float()
        return logits[0], torch.cat(mh, dim=-1)[0]

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.w("embed.weight")
        if isinstance(emb, Int4Weight):
            e = torch.stack([emb.row(int(i)) for i in ids.reshape(-1)])
            return e.view(*ids.shape, -1).to(compute_dtype(self.device))
        return emb[ids]

    @torch.no_grad()
    def prefill_chunked(
        self,
        ids: torch.Tensor,
        chunk_size: int | None = None,
        capture_mh: bool = False,
        progress_callback=None,
        layer_progress_callback=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build long-context state without a sequence-squared attention tensor."""
        from .dsv4 import hc_head

        cfg = self._cfg_obj()
        ids = ids.to(self.device).long()
        B, T = ids.shape
        if chunk_size is None:
            # Normal chat, route calibration and every accelerator share one
            # outer scheduler.  The previous hybrid-only 512-token override
            # silently defeated CCCP_PREFILL_BLOCK_TOKENS=4096 and left the
            # RAM+VRAM path with eight times as many Python/layer submissions.
            # Packed MoE owns its bounded inner micro-batch, so the outer
            # activation block can remain 4096 without expanding experts.
            chunk_size = prefill_block_size()
        self.last_prefill_block_size = int(chunk_size)
        self._alloc(B)
        main_hidden_parts = [] if capture_mh else None
        final_y = None
        for start, end in _prefill_ranges(T, chunk_size):
            self.ensure_position(end - 1)
            chunk_ids = ids[:, start:end]
            h = self._embed(chunk_ids).unsqueeze(2).repeat(
                1, 1, cfg.hc_mult, 1
            )
            chunk_main_hidden = []
            for layer in range(cfg.n_layers):
                try:
                    h = self._block(h, layer, chunk_ids, start)
                except torch.cuda.OutOfMemoryError as error:
                    limit = os.environ.get("CCCP_VRAM_LIMIT_GB")
                    limit_text = (
                        f"（当前进程上限 {limit} GiB）" if limit else ""
                    )
                    raise RuntimeError(
                        "GPU 显存不足，无法容纳 "
                        f"{end - start}-token 完整高速 Prefill 批次"
                        f"{limit_text}；为避免显著降速，启动器未自动拆成"
                        "小批次。请使用更大显存，或缩短本次输入。"
                    ) from error
                if layer_progress_callback is not None:
                    layer_progress_callback(
                        start,
                        end,
                        layer + 1,
                        cfg.n_layers,
                    )
                if capture_mh and layer in self.DSPARK_TARGETS:
                    chunk_main_hidden.append(h.mean(dim=2))
            if capture_mh:
                main_hidden_parts.append(
                    torch.cat(chunk_main_hidden, dim=-1)
                )
            y = hc_head(h, *self._hc_head_w(), cfg)
            final_y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
            if progress_callback is not None:
                progress_callback(end)
        if final_y is None:
            raise ValueError("prefill requires at least one token")
        self.pos = T
        logits = _linear(final_y[:, -1], self.w("head.weight")).float()
        main_hidden = (
            torch.cat(main_hidden_parts, dim=1)
            if main_hidden_parts is not None
            else None
        )
        return logits, main_hidden

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, full_logits: bool = True) -> torch.Tensor:
        from .dsv4 import hc_head
        cfg = self._cfg_obj()
        ids = ids.to(self.device).long()
        B, T = ids.shape
        if T > 512:
            if full_logits:
                raise RuntimeError(
                    "long prefill full_logits would materialize [T, vocab]; "
                    "use full_logits=False"
                )
            logits, _ = self.prefill_chunked(ids)
            return logits
        self.last_prefill_block_size = int(T)
        self._alloc(B)
        self.ensure_position(T - 1)
        h = self._embed(ids).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        for i in range(cfg.n_layers):
            if os.environ.get("CCCP_DEBUG_LAYER_TRACE", "0") != "0":
                print(f"[cccp-debug] prefill layer {i} begin", flush=True)
            h = self._block(h, i, ids, 0)
            if os.environ.get("CCCP_DEBUG_LAYER_TRACE", "0") != "0":
                print(f"[cccp-debug] prefill layer {i} end", flush=True)
        y = hc_head(h, *self._hc_head_w(), cfg)
        y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
        logits = _linear(y, self.w("head.weight")).float()
        return logits if full_logits else logits[:, -1]

    @torch.no_grad()
    def prefill_mh(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """prefill + 捕获 DSpark main_hidden。返回 (logits 末位 [1, vocab],
        main_hidden [1, T, 3·hidden])；同时建立 KV 并置 model.pos = T。"""
        from .dsv4 import hc_head
        cfg = self._cfg_obj()
        ids = ids.to(self.device).long()
        B, T = ids.shape
        if T > 512:
            logits, main_hidden = self.prefill_chunked(
                ids, capture_mh=True
            )
            assert main_hidden is not None
            return logits, main_hidden
        self._alloc(B)
        self.ensure_position(T - 1)
        h = self._embed(ids).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        mh = []
        for i in range(cfg.n_layers):
            h = self._block(h, i, ids, 0)
            if i in self.DSPARK_TARGETS:
                mh.append(h.mean(dim=2))
        y = hc_head(h, *self._hc_head_w(), cfg)
        y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
        logits = _linear(y, self.w("head.weight")).float()
        self.pos = T
        return logits[:, -1], torch.cat(mh, dim=-1)

    @staticmethod
    def _copy_paged_state(source: PagedKV, target: PagedKV) -> None:
        for page_index, page in enumerate(source.pages):
            target.ensure_page(page_index).copy_(page.to(target.device))
        target.length = int(source.length)

    def _sync_tp_attention_states(self) -> None:
        if self._tp_attention_contexts is None or self.states is None:
            raise RuntimeError("DSV4 TP state cannot be initialized")
        if self._tp_states_ready:
            return
        rank_states = [self.states]
        for rank in range(1, self.tp_size):
            rank_states.append(self._allocate_states(1, self.devices[rank]))
        for rank, states in enumerate(rank_states):
            for layer, state in enumerate(states):
                self._tp_attention_contexts[rank][layer]["state"] = state
                if rank == 0:
                    continue
                source = self.states[layer]
                state["kv"].copy_(source["kv"].to(self.devices[rank]))
                state["win_pos"].copy_(
                    source["win_pos"].to(self.devices[rank])
                )
                if self.ratios[layer]:
                    self._copy_paged_state(
                        source["compressed"], state["compressed"]
                    )
                    state["ckv"].copy_(
                        source["ckv"].to(self.devices[rank])
                    )
                    state["cscore"].copy_(
                        source["cscore"].to(self.devices[rank])
                    )
                    if self.ratios[layer] == 4:
                        source_indexer = source["indexer"]
                        target_indexer = state["indexer"]
                        self._copy_paged_state(
                            source_indexer.keys,
                            target_indexer.keys,
                        )
                        target_indexer.ckv.copy_(
                            source_indexer.ckv.to(self.devices[rank])
                        )
                        target_indexer.cscore.copy_(
                            source_indexer.cscore.to(self.devices[rank])
                        )
        for states in rank_states:
            for layer, state in enumerate(states):
                if "window_fp8" not in state:
                    continue
                state["window_fp8"].load_bf16(state["kv"][0])
                ratio = int(self.ratios[layer])
                if ratio:
                    compressed = state["compressed"]
                    rows = torch.cat(
                        [page[0] for page in compressed.pages], dim=0
                    )[:compressed.length]
                    state["compressed_fp8"].load_bf16(rows)
                if ratio == 4:
                    indexer = state["indexer"]
                    rows = torch.cat(
                        [page[0] for page in indexer.keys.pages], dim=0
                    )[:indexer.keys.length]
                    state["indexer_fp8"].load_bf16(rows)
        self._tp_states_ready = True

    def _ensure_tp_position(self, position: int) -> None:
        if self._tp_attention_contexts is None:
            return
        for rank in range(self.tp_size):
            for layer, ratio in enumerate(self.ratios):
                if not ratio:
                    continue
                state = self._tp_attention_contexts[rank][layer]["state"]
                state["compressed"].reserve(position // ratio)
                indexer = state.get("indexer")
                if indexer is not None:
                    indexer.reserve_position(position)

    def _tp_route(
        self,
        layer: int,
        rank: int,
        router_logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .ops import route_topk

        assert self._tp_route_weights is not None
        item = self._tp_route_weights[rank][layer]
        logits, weights, indices = self._tp_route_buffers[layer]
        with torch.cuda.device(self.devices[rank]):
            tid2eid = item.get("tid2eid")
            if tid2eid is None:
                routed = route_topk(
                    router_logits,
                    item["gate_bias"],
                    item["mask"],
                    scoring_func="sqrtsoftplus",
                    top_k=int(self.cfg["top_k"]),
                    normalize=bool(self.cfg.get("norm_topk_prob", True)),
                    scaling=float(self.cfg.get("routed_scaling", 1.0)),
                    output_buffers=(weights[rank], indices[rank]),
                )
                if routed is None:
                    raise RuntimeError(
                        "public sqrtsoftplus Router rejected DSV4 TP inputs"
                    )
                return routed
            scores = F.softplus(router_logits).sqrt()
            selected = tid2eid[token_ids.reshape(-1)].reshape(
                1, int(self.cfg["top_k"])
            )
            selected_weights = scores.gather(1, selected)
            if self.cfg.get("norm_topk_prob", True):
                selected_weights = selected_weights / (
                    selected_weights.sum(dim=-1, keepdim=True) + 1e-20
                )
            selected_weights *= float(
                self.cfg.get("routed_scaling", 1.0)
            )
            logits[rank].copy_(scores)
            weights[rank].copy_(selected_weights)
            indices[rank].copy_(selected)
            return weights[rank], indices[rank]

    @staticmethod
    def _capture_retained_graph(device, function, *, pool=None):
        """Capture a callable and retain its raw graph for parent composition."""
        stream = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        with torch.cuda.device(device), torch.cuda.stream(stream):
            outputs = function()
        current.wait_stream(stream)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.device(device), torch.cuda.graph(
            graph,
            stream=stream,
            pool=pool,
        ):
            outputs = function()
        graph.instantiate()
        torch.cuda.synchronize(device)
        return graph, outputs, stream

    def _tp_state_snapshot(self, rank: int = 0):
        """Preserve one rank's prompt state during graph warm-up/capture."""
        snapshots = []
        assert self._tp_attention_contexts is not None
        for context in self._tp_attention_contexts[int(rank)]:
            state = context["state"]
            item = {
                "state": state,
                "kv": state["kv"].clone(),
                "win_pos": state["win_pos"].clone(),
            }
            for name in ("ckv", "cscore"):
                if name in state:
                    item[name] = state[name].clone()
            for name in ("window_fp8", "compressed_fp8"):
                if name in state:
                    item[f"{name}_storage"] = state[name].storage.clone()
            if "indexer_fp8" in state:
                item["indexer_fp8_values"] = state["indexer_fp8"].values.clone()
                item["indexer_fp8_scales"] = state["indexer_fp8"].scales.clone()
            paged = state.get("compressed")
            if paged is not None:
                item["compressed_length"] = int(paged.length)
                item["compressed_pages"] = tuple(
                    page.clone() for page in paged.pages
                )
            indexer = state.get("indexer")
            if indexer is not None:
                item["indexer_length"] = int(indexer.keys.length)
                item["indexer_pages"] = tuple(
                    page.clone() for page in indexer.keys.pages
                )
                item["indexer_ckv"] = indexer.ckv.clone()
                item["indexer_cscore"] = indexer.cscore.clone()
            snapshots.append(item)
        return snapshots

    @staticmethod
    def _restore_tp_state(snapshots) -> None:
        for item in snapshots:
            state = item["state"]
            state["kv"].copy_(item["kv"])
            state["win_pos"].copy_(item["win_pos"])
            for name in ("ckv", "cscore"):
                if name in item:
                    state[name].copy_(item[name])
            for name in ("window_fp8", "compressed_fp8"):
                key = f"{name}_storage"
                if key in item:
                    state[name].storage.copy_(item[key])
            if "indexer_fp8_values" in item:
                state["indexer_fp8"].values.copy_(item["indexer_fp8_values"])
                state["indexer_fp8"].scales.copy_(item["indexer_fp8_scales"])
            paged = state.get("compressed")
            if paged is not None:
                for target, source in zip(
                    paged.pages, item["compressed_pages"]
                ):
                    target.copy_(source)
                paged.length = item["compressed_length"]
            indexer = state.get("indexer")
            if indexer is not None:
                for target, source in zip(
                    indexer.keys.pages, item["indexer_pages"]
                ):
                    target.copy_(source)
                indexer.keys.length = item["indexer_length"]
                indexer.ckv.copy_(item["indexer_ckv"])
                indexer.cscore.copy_(item["indexer_cscore"])

    def _tp_controlled_attention(
        self,
        source: torch.Tensor,
        layer: int,
        *,
        direct_width: int,
        selected_topk: bool,
        rank: int = 0,
        publish_hidden: bool = True,
    ):
        """Capture-safe rank-local Attention driven by device control.

        ``publish_hidden=False`` ends at the fixed FP32 Row-TP partial.  A
        public all-rank collective can then publish the sum before HC/FFN
        post-processing.  TP1 keeps the original complete local path.
        """
        from . import dsv4 as _d
        from .dsv4 import hc_post, rope_apply
        from .fusedext import (
            dsv4_attn_decode_controlled_fused,
            dsv4_kv_commit_controlled_fused,
            tp_all_rank_reduce_fused,
        )
        from .ops import (
            fused_compressor_cache_store,
            paged_indexer_logits,
            persistent_topk_exact,
            sparse_paged_attention_splitkv,
        )

        assert self._tp_attention_contexts is not None
        assert self._tp_route_weights is not None
        rank = int(rank)
        control = self._tp_decode_controls[rank]
        context = self._tp_attention_contexts[rank][layer]
        state = context["state"]
        weights = context["weights"]
        cfg = context["cfg"]
        ratio = int(context["ratio"])
        route_item = self._tp_route_weights[rank][layer]
        y, attention_post, attention_comb = _hc_pre_norm_cccp(
            source,
            route_item["hc_attn_fn"],
            route_item["hc_attn_scale"],
            route_item["hc_attn_base"],
            route_item["attn_norm"],
            cfg,
            output_buffers=self._hc_decode_workspace(source),
        )
        cache = SimpleNamespace(
            cos=context["control_cos"],
            sin=context["control_sin"],
        )
        qr, query, kv = _qkv_cccp(y, weights, cfg, cache, 0)
        if ratio:
            nested = weights["cmp"]
            projection = nested.get("projection_group")
            if projection is None:
                projection = ProjectionGroup(
                    (nested["wkv"], nested["wgate"])
                )
            projected = _cccp_lin(y, projection).reshape(1, -1)
            paged = state["compressed"]
            compressed_fp8 = state.get("compressed_fp8")
            if not fused_compressor_cache_store(
                projected=projected,
                ape=nested["ape"],
                ckv=state["ckv"],
                cscore=state["cscore"],
                norm=nested["norm"],
                rope_cos=context["control_pool_cos"],
                rope_sin=context["control_pool_sin"],
                page_ptrs=paged.device_page_ptrs(),
                control=control.values,
                model1_cache=(
                    compressed_fp8.storage
                    if compressed_fp8 is not None
                    else None
                ),
                ratio=ratio,
                kv_rows=int(nested["wkv"].shape[0]),
                width=int(cfg.head_dim),
                rope_width=int(cfg.qk_rope_head_dim),
                page_items=int(paged.page_items),
                overlap=bool(int(nested["wkv"].shape[0]) > cfg.head_dim),
                hadamard=False,
                eps=float(cfg.rms_eps),
                cache_format=(
                    "model1-fp8-e4m3-e8m0-rope64"
                    if compressed_fp8 is not None
                    else "bf16"
                ),
                page_layout=(
                    "model1-page-major"
                    if compressed_fp8 is not None
                    else "pointer-pages"
                ),
                architecture_features=tuple(
                    state.get("sparse_features", ())
                ),
            ):
                raise RuntimeError("controlled main compressor was rejected")

        if ratio == 4:
            indexer = state["indexer"]
            nested = weights["indexer"]
            projection = nested.get("projection_group")
            if projection is None:
                projection = ProjectionGroup(
                    (nested["wkv"], nested["wgate"])
                )
            projected = _cccp_lin(y, projection).reshape(1, -1)
            indexer_fp8 = state.get("indexer_fp8")
            if not fused_compressor_cache_store(
                projected=projected,
                ape=nested["ape"],
                ckv=indexer.ckv,
                cscore=indexer.cscore,
                norm=nested["norm"],
                rope_cos=context["control_pool_cos"],
                rope_sin=context["control_pool_sin"],
                page_ptrs=indexer.keys.device_page_ptrs(),
                control=control.values,
                indexer_cache=(
                    indexer_fp8.values
                    if indexer_fp8 is not None
                    else None
                ),
                indexer_scales=(
                    indexer_fp8.scales
                    if indexer_fp8 is not None
                    else None
                ),
                ratio=4,
                kv_rows=int(nested["wkv"].shape[0]),
                width=int(indexer.head_dim),
                rope_width=int(indexer.rope_dim),
                page_items=int(indexer.keys.page_items),
                overlap=True,
                hadamard=True,
                eps=float(cfg.rms_eps),
                cache_format=(
                    "indexer-e4m3-row-scale"
                    if indexer_fp8 is not None
                    else "bf16"
                ),
                page_layout="pointer-pages",
                architecture_features=tuple(
                    state.get("sparse_features", ())
                ),
            ):
                raise RuntimeError("controlled indexer compressor was rejected")

        use_flashmla = ratio == 4 and selected_topk
        selected = None
        if use_flashmla:
            nested = weights["indexer"]
            indexer_fp8 = state.get("indexer_fp8")
            if indexer_fp8 is None or "sparse_runner" not in state:
                raise RuntimeError(
                    "DSV4 topk512 缺少 FP8 Indexer/FlashMLA 状态；"
                    "禁止退回 BF16 全量 Attention"
                )
            candidate_count = int(state["indexer_logits"].shape[-1])
            index_query = _linear(qr, nested["wq_b"]).view(
                1, 1, cfg.index_n_heads, cfg.index_head_dim
            )
            index_weights = (
                _linear(y, nested["weights_proj"]) * (
                    cfg.index_head_dim ** -0.5
                    * cfg.index_n_heads ** -0.5
                )
            ).float().contiguous()
            scores = paged_indexer_logits(
                index_query.to(torch.bfloat16),
                indexer_fp8.values[:candidate_count],
                indexer_fp8.scales[:candidate_count],
                index_weights,
                context["control_cos"],
                context["control_sin"],
                control.values,
                compression_ratio=4,
                page_layout="contiguous-logical-pages",
                cache_format="indexer-e4m3-row-scale",
                query_fp8=state["indexer_query_fp8"],
                query_scales=state["indexer_query_scales"],
                mm_workspace=state["indexer_mm"],
                output=state["indexer_logits"],
                architecture_features=tuple(
                    state.get("sparse_features", ())
                ),
            )
            if scores is None:
                raise RuntimeError(
                    "DSV4 FP8 Indexer 算子不可用；禁止退回 BF16 Indexer"
                )
            topk_result = persistent_topk_exact(
                scores,
                int(cfg.index_topk),
                values=state["indexer_topk_values"],
                indices=state["indexer_topk_indices"],
                architecture_features=tuple(
                    state.get("sparse_features", ())
                ),
            )
            if topk_result is None:
                raise RuntimeError(
                    "DSV4 固定工作区 Top-K 算子不可用；禁止退回 torch.topk"
                )
            selected = topk_result[1]
            compressed_values = None
        elif ratio:
            max_items = (self.max_ctx + ratio - 1) // ratio
            ratio_width = (
                max_items
                if selected_topk
                else (
                    direct_width
                    if ratio == 4
                    else max(1, direct_width // 32)
                )
            )
            width = min(
                max_items,
                ratio_width,
            )
            if width > state["compressed"].page_items:
                fixed_indices = torch.arange(
                    width, dtype=torch.long, device=source.device
                ).view(1, -1)
                compressed_values = state["compressed"].gather_batched(
                    fixed_indices
                )
            else:
                compressed_values = state["compressed"].pages[0][:, :width]
        else:
            compressed_values = torch.empty(
                1,
                0,
                int(cfg.head_dim),
                dtype=torch.bfloat16,
                device=source.device,
            )

        if not dsv4_kv_commit_controlled_fused(
            kv,
            state["kv"],
            state["win_pos"],
            control.values,
            (
                state["window_fp8"].storage
                if "window_fp8" in state
                else None
            ),
        ):
            raise RuntimeError("controlled window KV commit was rejected")
        attended = None
        if use_flashmla and selected is not None:
            from .fusedext import sparse_attention_inverse_rope_fused

            state["flashmla_query"].copy_(query)
            sparse_output = sparse_paged_attention_splitkv(
                state["flashmla_query"],
                state["window_fp8"].storage,
                state["window_fp8_indices"],
                sink=weights["attn_sink"],
                scale=int(cfg.head_dim) ** -0.5,
                cache_format="model1-fp8-e4m3-e8m0-rope64",
                page_layout="model1-page-major",
                compression_ratio=4,
                extra_key_cache=state["compressed_fp8"].storage,
                extra_indices=selected,
                runner=state["sparse_runner"],
                architecture_features=tuple(
                    state.get("sparse_features", ())
                ),
            )
            if sparse_output is None:
                raise RuntimeError(
                    "FlashMLA sparse SplitKV 执行失败；"
                    "禁止退回 BF16 全量 Attention"
                )
            restored = sparse_attention_inverse_rope_fused(
                sparse_output,
                context["control_cos"],
                context["control_sin"],
            )
            if restored is None:
                raise RuntimeError(
                    "FlashMLA 逆 RoPE 融合算子不可用；"
                    "禁止退回 BF16 全量 Attention"
                )
            attended = restored[:, 0]
        if use_flashmla and attended is None:
            raise RuntimeError(
                "DSV4 topk512 未执行 FlashMLA sparse SplitKV；"
                "禁止退回 BF16 全量 Attention"
            )
        if attended is None:
            if compressed_values is None:
                assert selected is not None
                compressed_values = state["compressed"].gather_batched(
                    selected
                )[:, 0]
            attended = dsv4_attn_decode_controlled_fused(
                query[:, 0],
                state["kv"],
                state["win_pos"],
                compressed_values,
                weights["attn_sink"],
                context["control_cos"],
                context["control_sin"],
                control.values,
                int(cfg.head_dim) ** -0.5,
                ratio=ratio,
                selected_topk=selected_topk,
            )
        if attended is None:
            raise RuntimeError("controlled fixed-shape Attention was rejected")
        projected_attention = _d._o_proj_hook(
            attended.unsqueeze(1).flatten(2), weights, cfg
        ).contiguous()
        if not publish_hidden:
            partial = self._tp_attention_partials[layer].contributions[rank]
            partial.copy_(projected_attention)
            return source, y, attention_post, attention_comb

        attention_target = self._tp_attention_hidden[layer].replicas[rank]
        if projected_attention.dtype == attention_target.dtype:
            attention_target.copy_(
                projected_attention.view_as(attention_target)
            )
        else:
            partial = projected_attention.float().contiguous()
            if tp_all_rank_reduce_fused([partial], [attention_target]) is None:
                raise RuntimeError("TP1 Attention publication was rejected")
        prefix = hc_post(
            attention_target.view(1, 1, -1),
            source,
            attention_post,
            attention_comb,
            output=self._tp_ffn_prefix_hidden[layer].replicas[0],
        )
        normalized, ffn_post, ffn_comb = _hc_pre_norm_cccp(
            prefix,
            route_item["hc_ffn_fn"],
            route_item["hc_ffn_scale"],
            route_item["hc_ffn_base"],
            route_item["ffn_norm"],
            cfg,
            output_buffers=self._hc_decode_workspace(prefix),
        )
        self._tp_shared_mlp.input_hidden(layer).replicas[rank].copy_(
            normalized.reshape(1, -1)
        )
        return prefix, ffn_post, ffn_comb

    def _retire_hip_tp1_token_graphs(self, next_bucket: str) -> None:
        """Release the retired HIP bucket before capturing its successor."""
        if not self._tp1_token_graphs:
            return
        previous = ",".join(self._tp1_token_graphs)
        device = self.devices[0]
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
        self._tp1_token_graphs.clear()
        self._tp1_token_logits.clear()
        self._tp1_graph_dependencies.clear()
        self.tp_token_graph_info = {}

        # HIP's graph executable and LLVM allocation are owned by several
        # Python/C++ wrappers. Drop every owner before returning the retired
        # bucket. This runs only when the compressed-context shape crosses a
        # direct32/direct128/direct512 boundary.
        import gc

        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        print(
            "[cccp] TokenGraph bucket 切换："
            f"{previous} -> {next_bucket}；旧图已释放",
            flush=True,
        )

    def _prepare_tp1_token_graphs(self, capture_position: int) -> None:
        """Build full-token CUDA parent graphs for DSV4 TP1."""
        if self._tp1_graph_build_error is not None:
            return
        hip_runtime = torch.version.hip is not None
        dynamic_hybrid = bool(
            getattr(self, "_hybrid_fixed_token_graph", False)
        )
        requested_bucket = _tp1_token_graph_bucket(
            capture_position,
            hip_runtime=hip_runtime,
        )
        if self._tp1_token_graphs:
            if (
                (not hip_runtime and not dynamic_hybrid)
                or requested_bucket in self._tp1_token_graphs
            ):
                return
            self._retire_hip_tp1_token_graphs(requested_bucket)
        if (
            self.tp_size != 1
            or not self._single_gpu_layer_graph
            or os.environ.get("CCCP_TOKEN_GRAPH", "1") == "0"
        ):
            return
        if (
            self._tp1_decode_control is None
            or self._tp_attention_contexts is None
            or self._tp_shared_mlp is None
            or self._tp_router is None
            or self._tp_moe_finalizer is None
        ):
            return
        from .dsv4 import hc_head
        from .fusedext import make_tp_raw_graph_dag_batch
        from .ops import hyper_connection_post_moe

        device = self.devices[0]
        layer_count = int(self.cfg["n_layers"])
        sparse_splitkv_available = any(
            "sparse_runner" in context["state"]
            for context in self._tp_attention_contexts[0]
            if int(context["ratio"]) == 4
        )
        # Every page address referenced by a captured graph must already be
        # stable.  At max_ctx=4096 each ratio-4/128 state fits in one page.
        for context in self._tp_attention_contexts[0]:
            state = context["state"]
            ratio = int(context["ratio"])
            if not ratio:
                continue
            # Graphs below the sparse boundary only reference the largest
            # exact direct bucket.  Reserving to the model's logical max_ctx
            # defeats dynamic KV and can eagerly allocate a million-token
            # cache.  Pages beyond this point are created on demand.
            max_item = _tp1_graph_cache_reserve_item(
                int(self.max_ctx), ratio, sparse_splitkv_available
            )
            state["compressed"].reserve(max_item)
            state["compressed"].device_page_ptrs()
            indexer = state.get("indexer")
            if indexer is not None:
                indexer.keys.reserve(max_item)
                indexer.keys.device_page_ptrs()

        # Shared per-token RoPE rows replace 43 per-layer copies.  The input
        # graph selects them from the immutable tables using device position.
        rope_rows = {}
        for context in self._tp_attention_contexts[0]:
            ratio = int(context["ratio"])
            key = (id(context["rope"]), ratio)
            if key not in rope_rows:
                width = int(context["cfg"].qk_rope_head_dim) // 2
                rope_rows[key] = (
                    torch.empty(1, width, dtype=torch.float32, device=device),
                    torch.empty(1, width, dtype=torch.float32, device=device),
                    torch.empty(1, width, dtype=torch.float32, device=device),
                    torch.empty(1, width, dtype=torch.float32, device=device),
                )
            current_cos, current_sin, pool_cos, pool_sin = rope_rows[key]
            context["control_cos"] = current_cos
            context["control_sin"] = current_sin
            context["control_pool_cos"] = pool_cos
            context["control_pool_sin"] = pool_sin

        capture_started = time.perf_counter()
        graph_pool = torch.cuda.graph_pool_handle()

        def prepare_input():
            token = self._tp1_decode_control.token
            embedded = self._embed(token).unsqueeze(1).unsqueeze(2).repeat(
                1, 1, int(self.cfg["hc_mult"]), 1
            )
            self._tp_decode_input.replicas[0].copy_(embedded)
            position = self._tp1_decode_control.position
            for context in self._tp_attention_contexts[0]:
                ratio = int(context["ratio"])
                key = (id(context["rope"]), ratio)
                current_cos, current_sin, pool_cos, pool_sin = rope_rows[key]
                cache = context["rope"]
                current_cos.copy_(torch.index_select(cache.cos, 0, position))
                current_sin.copy_(torch.index_select(cache.sin, 0, position))
                pool_position = (position + 1 - ratio).clamp_min(0)
                pool_cos.copy_(torch.index_select(cache.cos, 0, pool_position))
                pool_sin.copy_(torch.index_select(cache.sin, 0, pool_position))
            return self._tp_decode_input.replicas[0]

        input_graph, _, input_stream = self._capture_retained_graph(
            device, prepare_input, pool=graph_pool
        )
        dependencies: list[object] = [input_graph, input_stream, rope_rows]
        snapshot = self._tp_state_snapshot()
        self._tp1_decode_control.update(0, int(capture_position))
        torch.cuda.current_stream(device).synchronize()
        bucket_specs = [
            ("direct32", 32, False),
            ("direct128", 128, False),
            ("direct512", 512, False),
        ]
        if (
            (self.max_ctx + 3) // 4 > int(self.cfg["index_topk"])
            and sparse_splitkv_available
        ):
            bucket_specs.append(("topk512", 512, True))
        if hip_runtime or dynamic_hybrid:
            # HIP graph executables are host-memory heavy.  A hybrid cache also
            # owns a large fixed 32-GiB process budget and must not retain three
            # otherwise identical 43-layer parent graphs. Keep one active
            # context bucket and replace it only at a shape boundary; replay
            # still has one parent launch per token. AMD/HIP 按需单 bucket and
            # the dynamic hybrid cache intentionally share this bounded rule.
            bucket_specs = [
                spec for spec in bucket_specs
                if spec[0] == requested_bucket
            ]
            if not bucket_specs:
                return

        try:
            for bucket_name, direct_width, selected_topk in bucket_specs:
                self._restore_tp_state(snapshot)
                layer_batches = []
                bucket_dependencies = []
                source = self._tp_decode_input.replicas[0]
                for layer in range(layer_count):
                    def attention_phase(
                        layer_index=layer,
                        layer_source=source,
                    ):
                        return self._tp_controlled_attention(
                            layer_source,
                            layer_index,
                            direct_width=direct_width,
                            selected_topk=selected_topk,
                        )

                    attention_graph, ffn_aux, attention_stream = (
                        self._capture_retained_graph(
                            device, attention_phase, pool=graph_pool
                        )
                    )
                    prefix, ffn_post, ffn_comb = ffn_aux
                    final_state = self._tp_moe_finalizer.layers[layer]
                    shared_state = self._tp_shared_mlp.layers[layer]
                    router_state = self._tp_router.layers[layer]
                    expert_batch = final_state.graph_batch

                    def final_phase(
                        layer_index=layer,
                        layer_prefix=prefix,
                        post=ffn_post,
                        comb=ffn_comb,
                        state=final_state,
                        shared=shared_state,
                    ):
                        combined = hyper_connection_post_moe(
                            state.expert_contributions[0],
                            shared.contributions[0],
                            layer_prefix,
                            post,
                            comb,
                            output=self._tp_layer_output_hidden[
                                layer_index
                            ].replicas[0],
                        )
                        if combined is None:
                            raise RuntimeError(
                                "TP1 routed/shared/HC finalizer was rejected"
                            )
                        return combined

                    final_graph, _, final_stream = self._capture_retained_graph(
                        device, final_phase, pool=graph_pool
                    )
                    hash_route_graph = None
                    if (
                        self._tp_route_packed_plan is None
                        or not self._tp_route_packed_plan.handles(layer)
                    ):
                        def hash_route_phase(layer_index=layer):
                            return self._tp_route(
                                layer_index,
                                0,
                                self._tp_route_buffers[layer_index][0][0],
                                self._tp1_decode_control.token,
                            )

                        hash_route_graph, _, hash_route_stream = (
                            self._capture_retained_graph(
                                device, hash_route_phase, pool=graph_pool
                            )
                        )
                    launch_stream = torch.cuda.Stream(device=device)
                    done_event = torch.cuda.Event()
                    source_event = torch.cuda.Event()
                    source_event.record(torch.cuda.current_stream(device))
                    branch_dependencies = []
                    if hash_route_graph is None:
                        expert_raw = expert_batch.raw_graphs()[0]
                        router_raw = router_state.graphs[0].raw_cuda_graph()
                    else:
                        expert_raw = (
                            self.routed_vq.fixed_layer_child_graphs(layer)[0][-1]
                            .raw_cuda_graph()
                        )
                        router_raw = hash_route_graph.raw_cuda_graph()
                    # The shared MLP and routed branch consume the same
                    # normalized hidden but have no data dependency.  A
                    # nested route->expert child graph lets the expert begin
                    # as soon as routing finishes while the independent
                    # shared branch is still running.
                    branch_stream = torch.cuda.Stream(device=device)
                    branch_done = torch.cuda.Event()
                    branch_source = torch.cuda.Event()
                    branch_source.record(torch.cuda.current_stream(device))
                    route_expert_batch = make_tp_raw_graph_dag_batch(
                        [int(device.index)],
                        [[
                            [router_raw],
                            [expert_raw],
                        ]],
                        [branch_stream],
                        [branch_done],
                        branch_source,
                    )
                    if route_expert_batch is None:
                        raise RuntimeError(
                            "TP1 route/expert branch graph unavailable"
                        )
                    parallel_stage = [
                        shared_state.graphs[0].raw_cuda_graph(),
                        route_expert_batch.raw_graphs()[0],
                    ]
                    branch_dependencies.extend((
                        branch_stream,
                        branch_done,
                        branch_source,
                        route_expert_batch,
                    ))
                    layer_stages = [
                        [attention_graph.raw_cuda_graph()],
                        parallel_stage,
                        [final_graph.raw_cuda_graph()],
                    ]
                    layer_batch = make_tp_raw_graph_dag_batch(
                        [int(device.index)],
                        [layer_stages],
                        [launch_stream],
                        [done_event],
                        source_event,
                    )
                    if layer_batch is None:
                        raise RuntimeError("TP1 layer parent graph unavailable")
                    layer_batches.append(layer_batch)
                    bucket_dependencies.extend((
                        attention_graph,
                        attention_stream,
                        final_graph,
                        final_stream,
                        launch_stream,
                        done_event,
                        source_event,
                        *branch_dependencies,
                    ))
                    if hash_route_graph is not None:
                        bucket_dependencies.extend((
                            hash_route_graph,
                            hash_route_stream,
                        ))
                    source = self._tp_layer_output_hidden[layer].replicas[0]

                final_source = source

                def head_phase():
                    value = hc_head(
                        final_source,
                        *self._hc_head_w(),
                        self._cfg_obj(),
                    )
                    value = rmsnorm(
                        value,
                        self.w("norm.weight"),
                        float(self.cfg["rms_eps"]),
                    )
                    return self._head_logits(value[:, 0])

                head_graph, logits, head_stream = self._capture_retained_graph(
                    device, head_phase, pool=graph_pool
                )
                token_stream = torch.cuda.Stream(device=device)
                token_done = torch.cuda.Event()
                token_source = torch.cuda.Event()
                token_source.record(torch.cuda.current_stream(device))
                stages = [[input_graph.raw_cuda_graph()]]
                stages.extend(
                    [batch.raw_graphs()[0]] for batch in layer_batches
                )
                stages.append([head_graph.raw_cuda_graph()])
                token_batch = make_tp_raw_graph_dag_batch(
                    [int(device.index)],
                    [[stage for stage in stages]],
                    [token_stream],
                    [token_done],
                    token_source,
                )
                if token_batch is None:
                    raise RuntimeError("TP1 token parent graph unavailable")
                self._tp1_token_graphs[bucket_name] = token_batch
                self._tp1_token_logits[bucket_name] = logits
                dependencies.extend((
                    *layer_batches,
                    *bucket_dependencies,
                    head_graph,
                    head_stream,
                    token_stream,
                    token_done,
                    token_source,
                    token_batch,
                    logits,
                ))
            self._tp1_graph_dependencies = dependencies
            self.tp_dataflow = "tp1-full-token-graph"
            self.tp_collectives_per_layer = 0
            uses_sparse_bucket = "topk512" in self._tp1_token_graphs
            has_flashmla = any(
                "sparse_runner" in context["state"]
                for context in self._tp_attention_contexts[0]
            )
            if uses_sparse_bucket and not has_flashmla:
                raise RuntimeError(
                    "DSV4 topk512 只允许 FlashMLA sparse SplitKV；"
                    "禁止记录或执行 sparse=bf16-exact"
                )
            self.tp_token_graph_info = {
                "layers": layer_count,
                "main_launches_per_token": 1,
                "buckets": tuple(self._tp1_token_graphs),
                "device_control_updates_per_token": 1,
                "python_layer_loop": False,
                "cuda_event_boundaries": 0,
                "indexer_candidate_capacity": (
                    (int(self.max_ctx) + 3) // 4
                ),
                "sparse_topk": int(self.cfg["index_topk"]),
                "sparse_attention": (
                    "flashmla-model1-fp8-splitkv"
                    if uses_sparse_bucket
                    else "not-required-direct"
                ),
            }
            print(
                "[cccp] 公共 TP1 DSV4 TokenGraph 完成："
                f"{layer_count} 个完整 LayerGraph，"
                f"bucket={','.join(self._tp1_token_graphs)}；"
                "每 token 一次主 cudaGraphLaunch；"
                f"sparse={self.tp_token_graph_info['sparse_attention']}；"
                "策略="
                f"{'按需单 bucket' if hip_runtime or dynamic_hybrid else '全 bucket 预捕获'}；"
                f"capture={(time.perf_counter() - capture_started) * 1000:.1f}ms",
                flush=True,
            )
            if hip_runtime:
                print(
                    "[cccp-amd-audit] decode=TP1-TokenGraph；"
                    "operator=cccp.packed_moe_topk_fused；"
                    "experts=GPU-only；H2D=0；CPU-fallback=0",
                    flush=True,
                )
        except Exception as exc:
            self._tp1_token_graphs.clear()
            self._tp1_token_logits.clear()
            self._tp1_graph_dependencies.clear()
            self._tp1_graph_build_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._restore_tp_state(snapshot)
            torch.cuda.synchronize(device)

    def _prepare_tp_controlled_attention_graphs(
        self,
        capture_position: int,
    ) -> None:
        """Capture one fixed-address Attention graph per TP rank and layer.

        The ordinary multi-rank path entered Python once for every rank and
        layer.  In a single-process TP runtime that serialized independent
        GPU work and made TP2 Attention almost exactly twice as slow as TP1.
        These batches submit every rank from one public C++ call; the only
        inter-rank boundary is the existing Row-TP publication afterwards.
        """
        if (
            self._tp_attention_controlled_batches
            or self._tp_attention_controlled_build_error is not None
        ):
            return
        if (
            self.tp_size <= 1
            or not self._packed_full_gpu
            or os.environ.get("CCCP_TP_ATTENTION_GRAPH", "1") == "0"
            or self._tp_attention_contexts is None
            or self._tp_decode_input is None
        ):
            return
        from .fusedext import make_tp_graph_launch_batch

        layer_count = int(self.cfg["n_layers"])
        # Captured page pointers and all rank-local RoPE rows must keep their
        # addresses for the engine lifetime.
        rope_rows_by_rank: list[dict] = []
        control_ready_events = []
        graph_pools = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                graph_pools.append(torch.cuda.graph_pool_handle())
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(device))
                control_ready_events.append(ready)
            rank_rows: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
            for context in self._tp_attention_contexts[rank]:
                state = context["state"]
                ratio = int(context["ratio"])
                if ratio:
                    max_item = (self.max_ctx + ratio - 1) // ratio - 1
                    state["compressed"].reserve(max_item)
                    state["compressed"].device_page_ptrs()
                    indexer = state.get("indexer")
                    if indexer is not None:
                        indexer.keys.reserve(max_item)
                        indexer.keys.device_page_ptrs()
                key = (id(context["rope"]), ratio)
                if key not in rank_rows:
                    width = int(context["cfg"].qk_rope_head_dim) // 2
                    rank_rows[key] = tuple(
                        torch.empty(
                            1,
                            width,
                            dtype=torch.float32,
                            device=device,
                        )
                        for _ in range(4)
                    )
                current_cos, current_sin, pool_cos, pool_sin = rank_rows[key]
                context["control_cos"] = current_cos
                context["control_sin"] = current_sin
                context["control_pool_cos"] = pool_cos
                context["control_pool_sin"] = pool_sin
            rope_rows_by_rank.append(rank_rows)

        snapshots = [
            self._tp_state_snapshot(rank) for rank in range(self.tp_size)
        ]
        bucket_specs = [
            ("direct32", 32, False),
            ("direct128", 128, False),
            ("direct512", 512, False),
        ]
        if (self.max_ctx + 3) // 4 > int(self.cfg["index_topk"]):
            bucket_specs.append(("topk512", 512, True))
        dependencies: list[object] = [
            rope_rows_by_rank,
            control_ready_events,
            graph_pools,
        ]
        try:
            for bucket_name, direct_width, selected_topk in bucket_specs:
                for snapshot in snapshots:
                    self._restore_tp_state(snapshot)
                bucket_batches: dict[int, object] = {}
                bucket_aux: dict[int, tuple[tuple[torch.Tensor, ...], ...]] = {}
                for control in self._tp_decode_controls:
                    control.update(0, int(capture_position))
                for layer in range(layer_count):
                    graphs = []
                    streams = []
                    rank_aux = []
                    for rank, device in enumerate(self.devices):
                        source_hidden = (
                            self._tp_decode_input
                            if layer == 0
                            else self._tp_layer_output_hidden[layer - 1]
                        )
                        source = source_hidden.replicas[rank]

                        def attention_phase(
                            layer_index=layer,
                            layer_source=source,
                            rank_index=rank,
                        ):
                            return self._tp_controlled_attention(
                                layer_source,
                                layer_index,
                                direct_width=direct_width,
                                selected_topk=selected_topk,
                                rank=rank_index,
                                publish_hidden=False,
                            )

                        graph, aux, stream = self._capture_retained_graph(
                            device,
                            attention_phase,
                            pool=graph_pools[rank],
                        )
                        graphs.append(graph)
                        streams.append(stream)
                        rank_aux.append(tuple(aux))
                    source_event = torch.cuda.Event()
                    with torch.cuda.device(self.devices[0]):
                        source_event.record(
                            torch.cuda.current_stream(self.devices[0])
                        )
                    partials = self._tp_attention_partials[layer]
                    batch = make_tp_graph_launch_batch(
                        [int(device.index) for device in self.devices],
                        graphs,
                        streams,
                        list(partials.ready_events),
                        source_event,
                    )
                    if batch is None:
                        raise RuntimeError(
                            "public all-rank Attention graph batch unavailable"
                        )
                    bucket_batches[layer] = batch
                    bucket_aux[layer] = tuple(rank_aux)
                    dependencies.extend((
                        *graphs,
                        *streams,
                        source_event,
                        batch,
                    ))
                self._tp_attention_controlled_batches[bucket_name] = (
                    bucket_batches
                )
                self._tp_attention_controlled_aux[bucket_name] = bucket_aux
            self._tp_attention_control_ready_events = tuple(
                control_ready_events
            )
            self._tp_attention_control_rope_rows = tuple(rope_rows_by_rank)
            self._tp_attention_controlled_dependencies = dependencies
            self.tp_dataflow = "all-rank-controlled-attention"
            print(
                "[cccp] 公共 DSV4 TP Attention Graph 完成："
                f"{layer_count} 层×TP{self.tp_size}，"
                "每层一次全 rank 并发提交",
                flush=True,
            )
        except Exception as exc:
            self._tp_attention_controlled_batches.clear()
            self._tp_attention_controlled_aux.clear()
            self._tp_attention_controlled_dependencies.clear()
            self._tp_attention_controlled_build_error = (
                f"{type(exc).__name__}: {exc}"
            )
            raise
        finally:
            for snapshot in snapshots:
                self._restore_tp_state(snapshot)
            for device in self.devices:
                torch.cuda.synchronize(device)

    def _update_tp_attention_controls(
        self,
        token: int,
        position: int,
        hidden,
    ) -> tuple[torch.cuda.Event, ...]:
        """Publish token/position/RoPE once per rank for all layer graphs."""
        if not self._tp_attention_control_ready_events:
            raise RuntimeError("controlled Attention events are unavailable")
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                stream = torch.cuda.current_stream(device)
                if hidden.ready_events is not None:
                    stream.wait_event(hidden.ready_events[rank])
                self._tp_decode_controls[rank].update(token, position)
                rows = self._tp_attention_control_rope_rows[rank]
                seen: set[tuple[int, int]] = set()
                for context in self._tp_attention_contexts[rank]:
                    ratio = int(context["ratio"])
                    key = (id(context["rope"]), ratio)
                    if key in seen:
                        continue
                    seen.add(key)
                    current_cos, current_sin, pool_cos, pool_sin = rows[key]
                    cache = context["rope"]
                    current_cos.copy_(cache.cos[position:position + 1])
                    current_sin.copy_(cache.sin[position:position + 1])
                    pool_position = max(0, position + 1 - ratio)
                    pool_cos.copy_(cache.cos[pool_position:pool_position + 1])
                    pool_sin.copy_(cache.sin[pool_position:pool_position + 1])
                self._tp_attention_control_ready_events[rank].record(stream)
        return self._tp_attention_control_ready_events

    def _prepare_tp_hc_layer_plans(self, bucket: str) -> None:
        """Bind complete fixed-address HC TP layers to one public call."""
        if (
            bucket in self._tp_hc_layer_plans
            or bucket in self._tp_hc_layer_plan_errors
            or os.environ.get("CCCP_TP_HC_LAYER_PLAN", "0") == "0"
        ):
            return
        batches = self._tp_attention_controlled_batches.get(bucket)
        aux_by_layer = self._tp_attention_controlled_aux.get(bucket)
        if (
            batches is None
            or aux_by_layer is None
            or self._tp_shared_mlp is None
            or self._tp_router is None
            or self._tp_moe_finalizer is None
        ):
            return
        from .ops import TensorParallelHyperConnectionDecodeLayerPlan

        plans: dict[int, object] = {}
        cfg = self._cfg_obj()
        try:
            for layer in range(int(self.cfg["n_layers"])):
                ffn_parameters = []
                ffn_aux_buffers = []
                for rank in range(self.tp_size):
                    item = self._tp_route_weights[rank][layer]
                    ffn_parameters.append((
                        item["hc_ffn_fn"],
                        item["hc_ffn_scale"],
                        item["hc_ffn_base"],
                        item["ffn_norm"],
                    ))
                    workspace = self._hc_decode_workspace(
                        self._tp_ffn_prefix_hidden[layer].replicas[rank]
                    )
                    ffn_aux_buffers.append((workspace[1], workspace[2]))
                plans[layer] = (
                    TensorParallelHyperConnectionDecodeLayerPlan(
                        layer,
                        batches[layer],
                        self._tp_attention_partials[layer],
                        self._tp_attention_hidden[layer],
                        aux_by_layer[layer],
                        self._tp_ffn_prefix_hidden[layer],
                        self._tp_shared_mlp.input_hidden(layer),
                        tuple(ffn_parameters),
                        tuple(ffn_aux_buffers),
                        self._tp_shared_mlp,
                        self._tp_router,
                        self.routed_vq,
                        self._tp_layer_output_hidden[layer],
                        sinkhorn_iters=int(cfg.hc_sinkhorn_iters),
                        eps=float(cfg.rms_eps),
                    )
                )
            self._tp_hc_layer_plans[bucket] = plans
            print(
                "[cccp] 公共 Hyper-Connection 真TP整层计划完成："
                f"{len(plans)} 层×TP{self.tp_size}，bucket={bucket}；"
                "每层一次 Python→C++ 提交",
                flush=True,
            )
        except Exception as exc:
            self._tp_hc_layer_plan_errors[bucket] = (
                f"{type(exc).__name__}: {exc}"
            )
            raise

    def _publish_controlled_attention_lengths(self, position: int) -> None:
        """Keep host-only rollback metadata aligned with device control."""
        assert self._tp_attention_contexts is not None
        for rank_contexts in self._tp_attention_contexts:
            for context in rank_contexts:
                ratio = int(context["ratio"])
                if not ratio:
                    continue
                length = (int(position) + 1) // ratio
                state = context["state"]
                state["compressed"].length = max(
                    state["compressed"].length,
                    length,
                )
                if ratio == 4:
                    state["indexer"].keys.length = max(
                        state["indexer"].keys.length,
                        length,
                    )

    def _decode_tp1_token_graph(
        self,
        ids: torch.Tensor,
        position: int,
    ) -> torch.Tensor | None:
        if self.tp_size != 1 or self._tp1_decode_control is None:
            return None
        self._prepare_tp1_token_graphs(position)
        if not self._tp1_token_graphs:
            return None
        bucket = _tp1_token_graph_bucket(
            position,
            hip_runtime=torch.version.hip is not None,
        )
        if bucket == "topk512" and bucket not in self._tp1_token_graphs:
            assert self._tp_attention_contexts is not None
            reason = next(
                (
                    context["state"].get(
                        "sparse_splitkv_unavailable_reason"
                    )
                    for context in self._tp_attention_contexts[0]
                    if int(context["ratio"]) == 4
                    and context["state"].get(
                        "sparse_splitkv_unavailable_reason"
                    )
                ),
                "FlashMLA sparse SplitKV backend unavailable",
            )
            raise RuntimeError(
                "DSV4 当前上下文已进入 sparse SplitKV 区间，"
                "但本机 CUDA 后端不支持；禁止退回 sparse=bf16-exact："
                f"{reason}"
            )
        graph = self._tp1_token_graphs.get(bucket)
        if graph is None:
            return None
        self.routed_vq.refresh_mapped_cache()
        self._tp1_decode_control.update(int(ids.reshape(-1)[0]), position)
        with torch.cuda.device(self.devices[0]):
            graph.launch_tp1()
        # Host lengths are rollback/diagnostic metadata only; all GPU kernels
        # derive visibility from DecodeControl and never wait on these writes.
        assert self._tp_attention_contexts is not None
        for context in self._tp_attention_contexts[0]:
            ratio = int(context["ratio"])
            if ratio:
                length = (int(position) + 1) // ratio
                state = context["state"]
                state["compressed"].length = max(
                    state["compressed"].length, length
                )
                if ratio == 4:
                    state["indexer"].keys.length = max(
                        state["indexer"].keys.length, length
                    )
        layer_count = int(self.cfg["n_layers"])
        self.routed_vq.record_cache_hits(
            int(self.cfg["top_k"]) * layer_count,
            fused_submissions=layer_count,
            graph_submissions=layer_count,
        )
        return self._tp1_token_logits[bucket]

    def _decode_tp(self, ids: torch.Tensor, pos: int) -> torch.Tensor:
        from .dsv4 import hc_head, hc_post
        from .ops import TPHiddenStageProfiler

        self._ensure_hybrid_tp1_graph_runtime()
        if (
            self._tp_attention_contexts is None
            or self._tp_route_weights is None
            or self._tp_shared_mlp is None
            or self._tp_moe_finalizer is None
            or self._tp_collective is None
            or self._tp_decode_input is None
        ):
            raise RuntimeError("DSV4 full TP metadata is unavailable")
        self._sync_tp_attention_states()
        self._ensure_tp_position(pos)
        hip_full_resident = bool(
            torch.version.hip is not None
            and self._packed_full_gpu
            and self.tp_size == 1
        )
        if self.tp_size == 1 and (
            not self._profile_enabled or hip_full_resident
        ):
            token_graph_logits = self._decode_tp1_token_graph(ids, pos)
            if token_graph_logits is not None:
                return token_graph_logits
        if hip_full_resident:
            raise RuntimeError(
                "AMD/HIP full-resident decode requires the TP1 TokenGraph; "
                "refusing the slow eager layer fallback "
                f"(requested={int(self._single_gpu_layer_graph_requested)}, "
                f"build_error={self._tp1_graph_build_error or 'none'})"
            )
        controlled_bucket = None
        controlled_batches = None
        controlled_aux_by_layer = None
        if self.tp_size > 1:
            self._prepare_tp_controlled_attention_graphs(pos)
            compressed_count = (int(pos) + 1) // 4
            if compressed_count <= 32:
                controlled_bucket = "direct32"
            elif compressed_count <= 128:
                controlled_bucket = "direct128"
            elif compressed_count <= 512:
                controlled_bucket = "direct512"
            else:
                controlled_bucket = "topk512"
            controlled_batches = self._tp_attention_controlled_batches.get(
                controlled_bucket
            )
            controlled_aux_by_layer = self._tp_attention_controlled_aux.get(
                controlled_bucket
            )
        cfg = self._cfg_obj()
        embedded = (
            self._embed(ids)
            .unsqueeze(1)
            .unsqueeze(2)
            .repeat(1, 1, cfg.hc_mult, 1)
        )
        hidden = self._tp_decode_input.copy_from_owner(embedded, 0)
        controlled_input_events = None
        if controlled_batches is not None:
            controlled_input_events = self._update_tp_attention_controls(
                int(ids.reshape(-1)[0]),
                pos,
                hidden,
            )
            assert controlled_bucket is not None
            self._prepare_tp_hc_layer_plans(controlled_bucket)
        hc_layer_plans = (
            self._tp_hc_layer_plans.get(controlled_bucket, {})
            if controlled_bucket is not None
            else {}
        )
        stage_profiler = (
            TPHiddenStageProfiler(True)
            if self._profile_enabled
            else None
        )
        if self.tp_size == 1 and self._tp1_decode_control is not None:
            self._tp1_decode_control.update(
                int(ids.reshape(-1)[0]), pos
            )
        else:
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    self._tp_token_ids[rank].copy_(ids)

        for layer in range(int(self.cfg["n_layers"])):
            hc_layer_plan = hc_layer_plans.get(layer)
            if hc_layer_plan is not None and stage_profiler is None:
                input_events = (
                    controlled_input_events
                    if layer == 0
                    else hidden.ready_events
                )
                if input_events is None:
                    raise RuntimeError(
                        "HC layer plan input events are unavailable"
                    )
                hidden = hc_layer_plan.launch(input_events)
                self.routed_vq.record_cache_hits(int(self.cfg["top_k"]))
                continue
            attention_profile = None
            if stage_profiler is not None:
                hidden, attention_profile = stage_profiler.begin(
                    "attention",
                    hidden,
                    layer=layer,
                )
            attention_tp_partials = self._tp_attention_partials.get(layer)
            controlled_batch = (
                controlled_batches.get(layer)
                if controlled_batches is not None
                else None
            )
            if controlled_batch is not None:
                if attention_tp_partials is None:
                    raise RuntimeError(
                        "fixed TP Attention partials are unavailable"
                    )
                input_events = (
                    controlled_input_events
                    if layer == 0
                    else hidden.ready_events
                )
                if input_events is None:
                    raise RuntimeError(
                        "controlled Attention input events are unavailable"
                    )
                controlled_batch.launch_from_events([
                    event.cuda_event for event in input_events
                ])
                assert controlled_aux_by_layer is not None
                attention_aux = list(controlled_aux_by_layer[layer])
            else:
                projection_batch = (
                    self._tp_attention_projection_batches.get(layer)
                )
                projection_input_events = (
                    self._tp_attention_projection_input_events.get(layer)
                )
                projection_done_events = (
                    self._tp_attention_projection_done_events.get(layer)
                )
                attention_aux = []
                for rank, device in enumerate(self.devices):
                    route_item = self._tp_route_weights[rank][layer]
                    with torch.cuda.device(device):
                        local = hidden.wait_on(device)
                        residual = local
                        y, post, comb = _hc_pre_norm_cccp(
                            local,
                            route_item["hc_attn_fn"],
                            route_item["hc_attn_scale"],
                            route_item["hc_attn_base"],
                            route_item["attn_norm"],
                            cfg,
                            output_buffers=self._hc_decode_workspace(local),
                        )
                        if projection_batch is not None:
                            context = self._tp_attention_contexts[rank][layer]
                            context["projection_graph_cos"].copy_(
                                context["rope"].cos[pos:pos + 1]
                            )
                            context["projection_graph_sin"].copy_(
                                context["rope"].sin[pos:pos + 1]
                            )
                            assert projection_input_events is not None
                            projection_input_events[rank].record(
                                torch.cuda.current_stream(device)
                            )
                        attention_aux.append((residual, y, post, comb))
                if projection_batch is not None:
                    assert projection_input_events is not None
                    projection_batch.launch_from_events([
                        event.cuda_event for event in projection_input_events
                    ])
                for rank, device in enumerate(self.devices):
                    with torch.cuda.device(device):
                        residual, y, post, comb = attention_aux[rank]
                        projection_outputs = None
                        if projection_batch is not None:
                            assert projection_done_events is not None
                            torch.cuda.current_stream(device).wait_event(
                                projection_done_events[rank]
                            )
                            projection_outputs = (
                                self._tp_attention_projection_graphs[rank][
                                    layer
                                ].outputs
                            )
                        projected_attention = self._attn_batch(
                            y,
                            layer,
                            pos,
                            None,
                            tp_context=(
                                self._tp_attention_contexts[rank][layer]
                            ),
                            static_projection_outputs=projection_outputs,
                        ).contiguous()
                        if self.tp_size == 1:
                            self._tp_attention_hidden[
                                layer
                            ].replicas[0].copy_(
                                projected_attention.view_as(
                                    self._tp_attention_hidden[
                                        layer
                                    ].replicas[0]
                                )
                            )
                        else:
                            if attention_tp_partials is None:
                                raise RuntimeError(
                                    "fixed TP Attention partials are unavailable"
                                )
                            attention_tp_partials.contributions[rank].copy_(
                                projected_attention
                            )
                            attention_tp_partials.ready_events[rank].record(
                                torch.cuda.current_stream(device)
                            )
            if self.tp_size == 1:
                attention = self._tp_attention_hidden[layer]
            else:
                assert attention_tp_partials is not None
                if (
                    self._tp_attention_parallel is None
                    or self._tp_attention_parallel.group_size == self.tp_size
                ):
                    attention = self._tp_collective.reduce_from_events(
                        self._tp_moe_finalizer.launch_batch(layer),
                        attention_tp_partials,
                        self._tp_attention_hidden[layer],
                    )
                else:
                    attention = (
                        self._tp_attention_parallel.reduce_from_events(
                            attention_tp_partials,
                            self._tp_attention_hidden[layer],
                        )
                    )
            if stage_profiler is not None:
                attention = stage_profiler.end(
                    attention_profile,
                    attention,
                )

            moe_profile = None
            if stage_profiler is not None:
                attention, moe_profile = stage_profiler.begin(
                    "moe",
                    attention,
                    layer=layer,
                )

            ffn_input = self._tp_shared_mlp.input_hidden(layer)
            prefix_hidden = self._tp_ffn_prefix_hidden[layer]
            ffn_aux = []
            for rank, device in enumerate(self.devices):
                route_item = self._tp_route_weights[rank][layer]
                with torch.cuda.device(device):
                    value = attention.wait_on(device)
                    residual, _, post, comb = attention_aux[rank]
                    prefix = hc_post(
                        value,
                        residual,
                        post,
                        comb,
                        output=prefix_hidden.replicas[rank],
                    )
                    ffn_residual = prefix
                    normalized, ffn_post, ffn_comb = _hc_pre_norm_cccp(
                        prefix,
                        route_item["hc_ffn_fn"],
                        route_item["hc_ffn_scale"],
                        route_item["hc_ffn_base"],
                        route_item["ffn_norm"],
                        cfg,
                        output_buffers=self._hc_decode_workspace(prefix),
                    )
                    ffn_input.replicas[rank].copy_(
                        normalized.reshape(1, -1)
                    )
                    if self.tp_size != 1:
                        ffn_input.ready_events[rank].record(
                            torch.cuda.current_stream(device)
                        )
                    ffn_aux.append(
                        (ffn_residual, ffn_post, ffn_comb)
                    )

            if self._tp_router is None:
                raise RuntimeError("DSV4 TP Router is unavailable")
            if self.tp_size == 1:
                shared_contribution = (
                    self._tp_shared_mlp.launch_partials_tp1(
                        layer, ffn_input
                    )
                )
                if (
                    self._tp_route_packed_plan is not None
                    and self._tp_route_packed_plan.handles(layer)
                ):
                    self._tp_router.run_tp1(layer, ffn_input)
                else:
                    self._tp_route(
                        layer,
                        0,
                        self._tp_route_buffers[layer][0][0],
                        self._tp_token_ids[0],
                    )
                combined_local = self._tp_moe_finalizer.run_tp1(
                    layer, shared_contribution
                )
                combined_hidden = self._tp_moe_finalizer.output_hidden(
                    layer
                )
            else:
                shared_partials = self._tp_shared_mlp.launch_partials(
                    layer, ffn_input
                )
                if (
                    self._tp_route_packed_plan is not None
                    and self._tp_route_packed_plan.handles(layer)
                ):
                    router_logits = self._tp_router.run_hidden(
                        layer, ffn_input
                    )
                    route_events = router_logits.ready_events
                else:
                    for rank in range(self.tp_size):
                        self._tp_route(
                            layer,
                            rank,
                            self._tp_route_buffers[layer][0][rank],
                            self._tp_token_ids[rank],
                        )
                        with torch.cuda.device(self.devices[rank]):
                            self._tp_route_events[layer][rank].record(
                                torch.cuda.current_stream(self.devices[rank])
                            )
                    route_events = self._tp_route_events[layer]
                combined_hidden = self._tp_moe_finalizer.run_from_events(
                    layer,
                    route_events,
                    shared_partials,
                )
            if stage_profiler is not None:
                combined_hidden = stage_profiler.end(
                    moe_profile,
                    combined_hidden,
                )
            self.routed_vq.record_cache_hits(int(self.cfg["top_k"]))
            output = self._tp_layer_output_hidden[layer]
            post_profile = None
            if stage_profiler is not None:
                combined_hidden, post_profile = stage_profiler.begin(
                    "ffn_post",
                    combined_hidden,
                    layer=layer,
                )
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    stream = torch.cuda.current_stream(device)
                    if self.tp_size != 1:
                        stream.wait_event(combined_hidden.ready_events[rank])
                    ffn_residual, ffn_post, ffn_comb = ffn_aux[rank]
                    posted = hc_post(
                        combined_hidden.replicas[rank].view(1, 1, -1),
                        ffn_residual,
                        ffn_post,
                        ffn_comb,
                        output=output.replicas[rank],
                    )
                    if (
                        posted.data_ptr()
                        != output.replicas[rank].data_ptr()
                    ):
                        output.replicas[rank].copy_(posted)
                    if self.tp_size != 1:
                        output.ready_events[rank].record(stream)
            if stage_profiler is not None:
                output = stage_profiler.end(post_profile, output)
            hidden = output

        if stage_profiler is not None:
            self._tp_stage_profile = stage_profiler.result(self.devices)
        if controlled_batches is not None:
            self._publish_controlled_attention_lengths(pos)
        with torch.cuda.device(self.device):
            final_hidden = hidden.wait_on(self.device)
            y = hc_head(final_hidden, *self._hc_head_w(), cfg)
            y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
            return self._head_logits(y[:, 0])

    @torch.inference_mode()
    def decode(self, ids: torch.Tensor, pos: int) -> torch.Tensor:
        from .dsv4 import hc_head
        cfg = self._cfg_obj()
        ids = ids.to(self.device).long()
        if self._tp_attention_contexts is not None:
            return self._decode_tp(ids, pos)
        self.ensure_position(pos)
        if self._prev_ids and self._token_prefetch_enabled():
            for l, es in self._prev_ids.items():
                self.routed_vq.prefetch_routes(l, es)
        h = self._embed(ids).unsqueeze(1).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        for i in range(cfg.n_layers):
            if os.environ.get("CCCP_DEBUG_LAYER_TRACE", "0") != "0":
                print(f"[cccp-debug] decode layer {i} begin", flush=True)
            h = self._block(h, i, ids.view(-1, 1), pos)
            if os.environ.get("CCCP_DEBUG_LAYER_TRACE", "0") != "0":
                print(f"[cccp-debug] decode layer {i} end", flush=True)
        y = hc_head(h, *self._hc_head_w(), cfg)
        y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
        return self._head_logits(y[:, 0])

    def _hc_head_w(self):
        return (
            self.w("hc_head_fn"),
            self.w("hc_head_scale"),
            self.w("hc_head_base"),
        )


def _f32(w):
    """Int4Weight → 就地反量化 f32（共享专家走 f32 精确路径，体积小）。"""
    if isinstance(w, Int4Weight):
        return w.dequant_rows(0, w.shape[0])
    if isinstance(w, BlockFP8Weight):
        return w.dequant_rows(0, w.shape[0], torch.float32)
    return w.float()

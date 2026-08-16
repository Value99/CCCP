"""Reference math and decode primitives for the Kimi K3 runtime.

The production CUDA path may fuse these operations, but every optimized kernel
is validated against the functions in this module.  Tensor conventions follow
the published Kimi implementation:

* KDA state uses V-first layout ``[heads, value_dim, key_dim]``.
* Attention Residuals mix block snapshots with FP32 scores.
* SiTU-GLU keeps its nonlinear math in FP32.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .grouped import activate_gate_up


def rmsnorm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    source_dtype = value.dtype
    work = value.float()
    work = work * torch.rsqrt(
        work.square().mean(dim=-1, keepdim=True) + float(eps)
    )
    return weight.to(source_dtype) * work.to(source_dtype)


def situ_mlp(
    value: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    gate_up = F.linear(value, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = activate_gate_up(
        gate,
        up,
        activation="situ",
        situ_beta=beta,
        situ_linear_beta=linear_beta,
    )
    return F.linear(activated, down_weight)


def attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply one Kimi Attention-Residual mixture."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)
    values_f = values.float()
    variance = values_f.square().mean(dim=-1, keepdim=True)
    normalized = values_f * torch.rsqrt(variance + float(eps))
    score_weight = norm_weight.float() * projection.reshape(-1).float()
    scores = (normalized * score_weight).sum(dim=-1)
    probabilities = scores.softmax(dim=-1).unsqueeze(-2)
    return torch.matmul(probabilities, values_f).squeeze(-2).to(values.dtype)


def route_experts(
    value: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    available: torch.Tensor,
    *,
    top_k: int,
    normalize: bool,
    scaling: float,
    n_group: int = 1,
    topk_group: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kimi sigmoid route with drop-expert masking before Top-K."""
    logits = F.linear(value.float(), gate_weight.float())
    scores = logits.sigmoid()
    choice = scores + correction_bias.float()
    choice = choice.masked_fill(~available, float("-inf"))
    if n_group > 1 and n_group > topk_group:
        if choice.shape[-1] % n_group:
            raise ValueError("expert count must be divisible by n_group")
        grouped = choice.view(
            *choice.shape[:-1],
            n_group,
            choice.shape[-1] // n_group,
        )
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        selected_groups = group_scores.topk(
            int(topk_group),
            dim=-1,
            sorted=False,
        ).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(-1, selected_groups, True)
        choice = choice.masked_fill(
            ~group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(choice),
            float("-inf"),
        )
    indices = choice.topk(int(top_k), dim=-1, sorted=False).indices
    weights = scores.gather(-1, indices)
    if normalize and top_k > 1:
        weights = weights / (
            weights.sum(dim=-1, keepdim=True) + 1e-20
        )
    return weights * float(scaling), indices


def short_conv_step(
    value: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One causal depthwise-convolution update followed by SiLU.

    ``state`` stores the previous ``kernel_size - 1`` projected values as
    ``[channels, history]``.  PyTorch Conv1d uses cross-correlation order, so
    the current value is paired with the final kernel coefficient.
    """
    if value.ndim != 1:
        raise ValueError("short_conv_step expects one flattened token")
    kernel = weight.reshape(weight.shape[0], -1).float()
    if state.shape != (value.numel(), kernel.shape[1] - 1):
        raise ValueError("short convolution state shape mismatch")
    window = torch.cat((state, value[:, None]), dim=-1)
    output = (window.float() * kernel).sum(dim=-1)
    return F.silu(output).to(value.dtype), window[:, 1:].to(state.dtype)


def kda_recurrent_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    *,
    lower_bound: float | None = -5.0,
) -> torch.Tensor:
    """Reference one-token KDA recurrence, updating ``state`` in place."""
    if query.ndim != 2 or key.shape != query.shape:
        raise ValueError("KDA query/key must be [heads, key_dim]")
    heads, key_dim = query.shape
    if value.shape != (heads, state.shape[-2]):
        raise ValueError("KDA value/state shape mismatch")
    if state.shape != (heads, value.shape[-1], key_dim):
        raise ValueError("KDA state must use [heads, value_dim, key_dim]")

    query_f = F.normalize(query.float(), p=2, dim=-1, eps=1e-6)
    key_f = F.normalize(key.float(), p=2, dim=-1, eps=1e-6)
    gate_f = gate.float() + dt_bias[: heads * key_dim].view(
        heads, key_dim
    ).float()
    a = a_log[:heads].float().exp().unsqueeze(-1)
    if lower_bound is None:
        log_decay = -a * F.softplus(gate_f)
    else:
        log_decay = float(lower_bound) * torch.sigmoid(a * gate_f)

    state.mul_(log_decay.exp().unsqueeze(-2))
    prediction = torch.matmul(
        state,
        key_f.unsqueeze(-1),
    ).squeeze(-1)
    delta = (value.float() - prediction) * beta.float().sigmoid().unsqueeze(-1)
    state.add_(delta.unsqueeze(-1) * key_f.unsqueeze(-2))
    output = torch.matmul(
        state,
        query_f.unsqueeze(-1),
    ).squeeze(-1)
    return (output * (1.0 / math.sqrt(key_dim))).to(value.dtype)


def gated_rmsnorm(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return rmsnorm(value, weight, eps) * gate.sigmoid()


__all__ = [
    "attention_residual",
    "gated_rmsnorm",
    "kda_recurrent_step",
    "rmsnorm",
    "route_experts",
    "short_conv_step",
    "situ_mlp",
]

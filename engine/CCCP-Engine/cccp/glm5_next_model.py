"""GLM-5.3 text runtime backed by CCCP's public compact operators.

The topology and cache semantics come from the pinned upstream Transformers
``glm5_next`` implementation.  This adapter only binds released checkpoint
names to CCCP storage and substitutes the two storage-sensitive operations:

* dense projections use :mod:`cccp.ops` Linear with compact block-FP8;
* routed experts use the common packed CPU/CUDA expert pool.

No dispatch depends on a model directory name or an S/M/L suffix.
"""

from __future__ import annotations

import gc
import os
import time
from types import MethodType
from typing import Iterable

import torch
from torch import nn

from .kernels import BlockFP8Weight
from .store import CCCPStore


_SOURCE_ROOT = "model.language_model."


def _glm5_next_kda_raw_inputs(
    log_decay: torch.Tensor,
    beta_probability: torch.Tensor,
    *,
    lower_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert GLM's precomputed KDA gates for the public raw-gate kernel."""

    bound = float(lower_bound)
    if bound >= 0.0:
        raise ValueError("KDA lower_bound must be negative")
    eps = torch.finfo(torch.float32).eps
    gate_probability = (log_decay.float() / bound).clamp(eps, 1.0 - eps)
    beta_probability = beta_probability.float().clamp(eps, 1.0 - eps)
    return torch.logit(gate_probability), torch.logit(beta_probability)


def _glm5_next_delta_rule(fallback):
    """Bind GLM-5.3 KDA tensors to the model-independent ordered scan.

    Transformers' portable fallback materialises a six-dimensional decay
    tensor.  At 4096 tokens that requests another 8 GiB per linear-attention
    layer.  CCCP accepts the same mathematical tensors but executes the
    ordered recurrence through the public CPU/CUDA operator registry.
    """

    del fallback
    neutral_parameters: dict[
        tuple[str, int | None, int, int], tuple[torch.Tensor, torch.Tensor]
    ] = {}

    def run(
        query,
        key,
        value,
        *,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **_kwargs,
    ):
        if (
            query.ndim != 4
            or query.shape[0] != 1
            or key.shape != query.shape
            or g.shape != query.shape
            or value.ndim != 4
            or value.shape[:3] != query.shape[:3]
            or beta.shape != query.shape[:3]
            or not use_qk_l2norm_in_kernel
        ):
            raise RuntimeError(
                "GLM-5.3 KDA requires the public [1,T,H,D] ordered-scan "
                "contract; quadratic Transformers fallback is forbidden"
            )
        from .ops import ordered_recurrent_scan

        lower_bound = -5.0
        raw_gate, raw_beta = _glm5_next_kda_raw_inputs(
            g,
            beta,
            lower_bound=lower_bound,
        )
        heads = int(query.shape[2])
        key_dim = int(query.shape[3])
        value_dim = int(value.shape[3])
        cache_key = (
            query.device.type,
            query.device.index,
            heads,
            key_dim,
        )
        neutral = neutral_parameters.get(cache_key)
        if neutral is None:
            neutral = (
                torch.zeros(heads, dtype=torch.float32, device=query.device),
                torch.zeros(
                    heads,
                    key_dim,
                    dtype=torch.float32,
                    device=query.device,
                ),
            )
            neutral_parameters[cache_key] = neutral
        a_log, dt_bias = neutral
        state = (
            torch.zeros(
                heads,
                value_dim,
                key_dim,
                dtype=torch.float32,
                device=query.device,
            )
            if initial_state is None
            else initial_state[0].transpose(-1, -2).float().contiguous()
        )
        output = ordered_recurrent_scan(
            query[0].contiguous(),
            key[0].contiguous(),
            value[0].contiguous(),
            raw_gate[0].to(query.dtype).contiguous(),
            raw_beta[0].contiguous(),
            a_log,
            dt_bias,
            state,
            lower_bound=lower_bound,
            backend="auto",
        )
        final_state = (
            state.transpose(-1, -2).contiguous().unsqueeze(0)
            if output_final_state
            else None
        )
        return output.unsqueeze(0).to(value.dtype), final_state

    run._cccp_public_kda = True
    return run


def _glm5_next_pool_phase(executor, *, rows: int):
    """Use the common Prefill/Decode arena lifecycle for GLM-5.3."""

    return executor.phase(rows=int(rows))


def _glm5_next_linear_decode_graph_eligible(
    block_type: str,
    hidden_states,
    cache_params,
    *,
    layer: int,
) -> bool:
    """Return whether a recurrent single-token layer has fixed CUDA state."""
    if (
        block_type != "linear_attention"
        or hidden_states.device.type != "cuda"
        or tuple(hidden_states.shape[:2]) != (1, 1)
        or cache_params is None
    ):
        return False
    has_previous = getattr(cache_params, "has_previous_state", None)
    return bool(callable(has_previous) and has_previous(int(layer)))


def _glm5_next_static_pool_states(
    indexer: nn.Module,
    packed_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
    """Build sparse-index pools without data-dependent output slicing.

    Invalid trailing pools remain present and are masked by ``pool_valid`` in
    the official indexer forward pass.  The mathematics is unchanged, while
    the fixed output extent removes the dynamic ``tensor[:, bool_mask]`` that
    prevents full-token scheduling and creates a CPU synchronization point.
    This is Attention topology; codebook execution stays in the public VQ
    runtime.
    """
    keys, gate_scores, valid_keys = torch.split(
        packed_states,
        [int(indexer.head_dim), int(indexer.head_dim), 1],
        dim=-1,
    )
    valid_keys = valid_keys.bool().squeeze(-1)
    batch_size, seq_len = keys.shape[:2]
    pool_size = int(indexer.index_kpool)
    number_of_pools = (int(seq_len) + pool_size - 1) // pool_size
    device = keys.device
    first_key = torch.where(
        valid_keys.any(-1),
        valid_keys.long().argmax(-1),
        torch.full(
            (batch_size,),
            int(seq_len),
            dtype=torch.long,
            device=device,
        ),
    )
    pool_offsets = torch.arange(
        number_of_pools * pool_size,
        device=device,
    ).view(1, number_of_pools, pool_size)
    pool_indices = first_key[:, None, None] + pool_offsets
    batch_indices = torch.arange(device=device, end=batch_size)[:, None, None]
    safe_indices = pool_indices.clamp(0, int(seq_len) - 1)
    grouped_keys = keys[batch_indices, safe_indices]
    grouped_gate_scores = gate_scores[batch_indices, safe_indices]
    grouped_valid_keys = valid_keys[batch_indices, safe_indices]
    grouped_valid_keys = grouped_valid_keys & (pool_indices < int(seq_len))
    pool_valid = grouped_valid_keys.all(-1)
    pool_indices = pool_indices.masked_fill(~grouped_valid_keys, -1)
    logits = (
        grouped_gate_scores.float()
        + indexer.index_kpool_compress_ape.float()[None, None]
    )
    logits = logits.masked_fill(
        ~grouped_valid_keys[..., None],
        float("-inf"),
    )
    probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(
        grouped_keys.dtype
    )
    pool_keys = (probabilities * grouped_keys).sum(dim=2)
    return pool_keys, pool_indices, pool_valid


def _glm5_next_hc_pre_norm(
    value: torch.Tensor,
    hyper_connection: nn.Module,
    norm: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bind GLM's H/C topology to the public fused H/C implementation."""
    from .ops import hyper_connection_pre_norm

    fused = hyper_connection_pre_norm(
        value,
        hyper_connection.fn,
        hyper_connection.scale,
        hyper_connection.base,
        norm.weight,
        sinkhorn_iters=int(hyper_connection.hc_sinkhorn_iters),
        eps=float(norm.variance_epsilon),
    )
    if fused is not None:
        return fused
    post, combine, collapsed = hyper_connection(value)
    return norm(collapsed), post, combine


def _glm5_next_hc_post(
    value: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    combine: torch.Tensor,
) -> torch.Tensor:
    """Bind GLM's H/C residual topology to the public fused post operator."""
    from .ops import hyper_connection_post

    fused = hyper_connection_post(value, residual, post, combine)
    if fused is not None:
        return fused
    dtype = residual.dtype
    return post.to(dtype).unsqueeze(-1) * value.unsqueeze(-2) + torch.matmul(
        combine.to(dtype).transpose(-1, -2), residual
    )


class GLM5NextDecoderLayerRuntime(nn.Module):
    """Topology-only adapter using public H/C, Attention and routed-VQ math."""

    def __init__(self, layer: nn.Module, *, decode_graph_pool=None) -> None:
        super().__init__()
        self.block_type = layer.block_type
        self.self_attn = layer.self_attn
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.attn_hc = layer.attn_hc
        self.ffn_hc = layer.ffn_hc
        self._decode_graph_pool = decode_graph_pool
        self._linear_decode_graph = None
        self._linear_decode_cache_id: int | None = None
        self._linear_decode_hidden: torch.Tensor | None = None
        self._linear_decode_attention_mask: torch.Tensor | None = None
        self._linear_decode_kwargs: dict[str, object] = {}

    def reset_decode_graph(self) -> None:
        self._linear_decode_graph = None
        self._linear_decode_cache_id = None
        self._linear_decode_hidden = None
        self._linear_decode_attention_mask = None
        self._linear_decode_kwargs.clear()

    def _linear_decode_prefix(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_params,
        kwargs: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normalized, post, combine = _glm5_next_hc_pre_norm(
            hidden_states,
            self.attn_hc,
            self.input_layernorm,
        )
        attended = self.self_attn(
            hidden_states=normalized,
            cache_params=cache_params,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = _glm5_next_hc_post(
            attended,
            residual,
            post,
            combine,
        )
        residual = hidden_states
        normalized, post, combine = _glm5_next_hc_pre_norm(
            hidden_states,
            self.ffn_hc,
            self.post_attention_layernorm,
        )
        return normalized, residual, post, combine

    @staticmethod
    def _snapshot_recurrent_cache(cache_params, layer: int):
        state = cache_params.layers[int(layer)]
        snapshots = []
        for name in ("conv_states", "recurrent_states"):
            for tensor in getattr(state, name, ()):
                if isinstance(tensor, torch.Tensor):
                    snapshots.append((tensor, tensor.clone()))
        return snapshots

    def _capture_linear_decode_graph(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_params,
        kwargs: dict[str, object],
    ) -> None:
        from .ops import FixedAddressCudaGraph

        layer = int(self.self_attn.layer_idx)
        self._linear_decode_hidden = torch.empty_like(hidden_states)
        self._linear_decode_hidden.copy_(hidden_states)
        self._linear_decode_attention_mask = (
            torch.empty_like(attention_mask)
            if isinstance(attention_mask, torch.Tensor)
            else None
        )
        if self._linear_decode_attention_mask is not None:
            self._linear_decode_attention_mask.copy_(attention_mask)
        fixed_kwargs: dict[str, object] = {}
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                fixed_value = torch.empty_like(value)
                fixed_value.copy_(value)
                fixed_kwargs[name] = fixed_value
            else:
                fixed_kwargs[name] = value
        self._linear_decode_kwargs = fixed_kwargs

        def function():
            hidden = self._linear_decode_hidden
            if hidden is None:
                raise RuntimeError("linear Decode graph input was released")
            return self._linear_decode_prefix(
                hidden,
                self._linear_decode_attention_mask,
                cache_params,
                self._linear_decode_kwargs,
            )

        snapshots = self._snapshot_recurrent_cache(cache_params, layer)
        try:
            self._linear_decode_graph = FixedAddressCudaGraph(
                hidden_states.device,
                function,
                pool=self._decode_graph_pool,
            )
        finally:
            for target, snapshot in snapshots:
                target.copy_(snapshot)
        self._linear_decode_cache_id = id(cache_params)

    def _forward_linear_decode_graph(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_params,
        kwargs: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self._linear_decode_graph is None
            or self._linear_decode_cache_id != id(cache_params)
        ):
            self.reset_decode_graph()
            self._capture_linear_decode_graph(
                hidden_states,
                attention_mask,
                cache_params,
                kwargs,
            )
        fixed_hidden = self._linear_decode_hidden
        graph = self._linear_decode_graph
        if fixed_hidden is None or graph is None:
            raise RuntimeError("linear Decode graph capture failed")
        fixed_hidden.copy_(hidden_states)
        if self._linear_decode_attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise ValueError("linear Decode attention mask changed type")
            self._linear_decode_attention_mask.copy_(attention_mask)
        for name, target in self._linear_decode_kwargs.items():
            source = kwargs[name]
            if isinstance(target, torch.Tensor):
                if not isinstance(source, torch.Tensor):
                    raise ValueError(f"linear Decode kwarg {name} changed type")
                target.copy_(source)
        return graph.replay()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        layer = int(getattr(self.self_attn, "layer_idx", -1))
        if (
            self._decode_graph_pool is not None
            and _glm5_next_linear_decode_graph_eligible(
                self.block_type,
                hidden_states,
                past_key_values,
                layer=layer,
            )
        ):
            normalized, residual, post, combine = (
                self._forward_linear_decode_graph(
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    kwargs,
                )
            )
            output = self.mlp(normalized)
            return (
                _glm5_next_hc_post(
                    output,
                    residual,
                    post,
                    combine,
                ),
                None,
            )
        residual = hidden_states
        hidden_states, post, combine = _glm5_next_hc_pre_norm(
            hidden_states,
            self.attn_hc,
            self.input_layernorm,
        )
        topk_indices = None
        if self.block_type == "linear_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _, topk_indices = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                prev_topk_indices=prev_topk_indices,
                **kwargs,
            )
        hidden_states = _glm5_next_hc_post(
            hidden_states,
            residual,
            post,
            combine,
        )

        residual = hidden_states
        hidden_states, post, combine = _glm5_next_hc_pre_norm(
            hidden_states,
            self.ffn_hc,
            self.post_attention_layernorm,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = _glm5_next_hc_post(
            hidden_states,
            residual,
            post,
            combine,
        )
        return hidden_states, topk_indices


def _glm5_next_source_names(target_name: str) -> tuple[str, ...]:
    """Map one upstream text-model name to released checkpoint names.

    The rules mirror Transformers' official ``glm5_next`` conversion map.
    Keeping this tiny, explicit mapping beside the adapter prevents archive
    conversion details from leaking into the public operators.
    """

    if target_name.endswith("self_attn.conv1d.weight"):
        prefix = target_name[: -len("conv1d.weight")]
        return tuple(
            _SOURCE_ROOT + prefix + projection + "_conv1d.weight"
            for projection in ("q", "k", "v")
        )

    converted = target_name
    converted = converted.replace(".self_attn.forget_gate.", ".self_attn.")
    hyper_connections = {
        ".attn_hc.fn": ".hc_attn_fn",
        ".attn_hc.base": ".hc_attn_base",
        ".attn_hc.scale": ".hc_attn_scale",
        ".ffn_hc.fn": ".hc_ffn_fn",
        ".ffn_hc.base": ".hc_ffn_base",
        ".ffn_hc.scale": ".hc_ffn_scale",
    }
    for target, source in hyper_connections.items():
        converted = converted.replace(target, source)
    return (_SOURCE_ROOT + converted,)


class CCCPLinear(nn.Module):
    """An ``nn.Linear``-compatible view over a CCCP compact weight."""

    def __init__(
        self,
        weight: torch.Tensor | BlockFP8Weight,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        from .ops import linear, linear_batch

        rows = int(value.numel() // value.shape[-1])
        result = (
            linear(
                value.reshape(1, value.shape[-1]),
                self.weight,
                output_dtype=value.dtype,
            ).reshape(*value.shape[:-1], -1)
            if rows == 1
            else linear_batch(
                value,
                self.weight,
                output_dtype=value.dtype,
            )
        )
        if self.bias is not None:
            result = result + self.bias.to(result.dtype)
        return result


class GLM5NextTopkRouterRuntime(nn.Module):
    """Bind GLM's standard sigmoid Top-K math to the public router operator."""

    def __init__(self, router: nn.Module) -> None:
        super().__init__()
        self.top_k = int(router.top_k)
        self.num_experts = int(router.num_experts)
        self.hidden_dim = int(router.hidden_dim)
        self.routed_scaling_factor = float(router.routed_scaling_factor)
        self.num_group = int(router.num_group)
        self.topk_group = int(router.topk_group)
        self.norm_topk_prob = bool(router.norm_topk_prob)
        self.weight = nn.Parameter(
            router.weight.detach().float(),
            requires_grad=False,
        )
        self.register_buffer(
            "e_score_correction_bias",
            router.e_score_correction_bias.detach().float(),
            persistent=False,
        )
        self._decode_buffers: dict[
            tuple[str, int | None],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    def _decode_route_buffers(
        self,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (value.device.type, value.device.index)
        buffers = self._decode_buffers.get(key)
        if buffers is None:
            buffers = (
                torch.empty(
                    1,
                    self.num_experts,
                    dtype=torch.float32,
                    device=value.device,
                ),
                torch.empty(
                    1,
                    self.top_k,
                    dtype=torch.float32,
                    device=value.device,
                ),
                torch.empty(
                    1,
                    self.top_k,
                    dtype=torch.long,
                    device=value.device,
                ),
                torch.ones(
                    self.num_experts,
                    dtype=torch.bool,
                    device=value.device,
                ),
            )
            self._decode_buffers[key] = buffers
        return buffers

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = hidden_states.reshape(-1, self.hidden_dim)
        if rows.shape[0] == 1:
            from .ops import linear_route_topk

            logits, weights, indices, mask = self._decode_route_buffers(rows)
            routed = linear_route_topk(
                rows,
                self.weight,
                self.e_score_correction_bias,
                mask,
                scoring_func="sigmoid",
                top_k=self.top_k,
                normalize=self.norm_topk_prob,
                scaling=self.routed_scaling_factor,
                n_group=self.num_group,
                topk_group=self.topk_group,
                output_buffers=(logits, weights, indices),
            )
            if routed is not None:
                return logits, routed[0], routed[1]

        router_logits = torch.nn.functional.linear(
            rows.float(),
            self.weight,
        )
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        group_scores = (
            scores_for_choice.view(
                -1,
                self.num_group,
                self.num_experts // self.num_group,
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores,
            k=self.topk_group,
            dim=-1,
            sorted=False,
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                -1,
                self.num_group,
                self.num_experts // self.num_group,
            )
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(
            ~score_mask.bool(),
            float("-inf"),
        )
        indices = torch.topk(
            scores_for_choice,
            k=self.top_k,
            dim=-1,
            sorted=False,
        )[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob:
            weights = weights / (
                weights.sum(dim=-1, keepdim=True) + 1.0e-20
            )
        weights = weights * self.routed_scaling_factor
        return router_logits, weights, indices


class GLM5NextPackedExperts(nn.Module):
    """Thin topology adapter for the common packed routed-expert pool."""

    def __init__(self, executor, *, layer: int, swiglu_limit: float) -> None:
        super().__init__()
        self.executor = executor
        self.layer = int(layer)
        self.swiglu_limit = float(swiglu_limit)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        result = self.executor.execute(
            self.layer,
            hidden_states,
            top_k_index,
            top_k_weights,
            activation="silu",
            activation_beta=1.0,
            activation_linear_beta=None,
            limit=self.swiglu_limit,
        )
        return result.to(hidden_states.dtype)


def _parent_and_leaf(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, leaf = name.rpartition(".")
    return (
        module.get_submodule(parent_name) if parent_name else module,
        leaf,
    )


class GLM5NextCCCPModel:
    """Manifest-driven GLM-5.3 text inference adapter."""

    def __init__(
        self,
        root: str,
        cache_gb: float = 16.0,
        max_ctx: int = 2048,
        device: str = "cpu",
        vram_cache_gb: float = 4.0,
        tp_size: int = 1,
    ) -> None:
        self.tp_size = int(tp_size)
        if self.tp_size <= 0:
            raise ValueError("GLM-5.3 tp_size must be positive")
        self.root = root
        requested = torch.device(device)
        if requested.type == "cuda" and requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())
        if self.tp_size > 1:
            if requested.type != "cuda":
                raise ValueError("GLM-5.3 routed TP requires CUDA")
            primary = int(requested.index or 0)
            self.devices = tuple(
                torch.device("cuda", primary + rank)
                for rank in range(self.tp_size)
            )
            if self.devices[-1].index >= torch.cuda.device_count():
                raise ValueError(
                    f"GLM-5.3 tp={self.tp_size} exceeds visible CUDA devices"
                )
        else:
            self.devices = (requested,)
        self.device = self.devices[0]
        self.store = CCCPStore(root)
        self.cfg = self.store.cfg
        self.max_ctx = int(max_ctx)
        self.effective_tp_size = self.tp_size
        self._cache_gb = float(cache_gb)
        self._vram_cache_gb = float(vram_cache_gb)
        self._model: nn.Module | None = None
        self._lm_head: CCCPLinear | None = None
        self._past_key_values = None
        # Vision is strictly opt-in: text preload never constructs this field.
        self._vision_model: nn.Module | None = None
        self._outer_config = None
        self._multimodal_active = False
        self.last_multimodal_timings: dict[str, float] | None = None
        self._text_config = None
        self._fixed_token_graph = None
        self._fixed_token_graph_cache = None
        self._fixed_token_graph_capacity = 0
        self._device_token_history: torch.Tensor | None = None
        self._token_history: list[int] = []
        self.pos = 0

        from .ops import create_routed_vq_runtime

        codebook_runtime = create_routed_vq_runtime(
            self.store,
            device=self.device,
            devices=self.devices,
            tp_size=self.tp_size,
            cache_gb=self._cache_gb,
            vram_cache_gb=self._vram_cache_gb,
        )
        self.routed_vq = codebook_runtime.executor

    @staticmethod
    def _require_transformers():
        try:
            from transformers import AutoConfig, DynamicCache
            from transformers.models.glm5_next.modeling_glm5_next import (
                Glm5NextTextModel,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "内置 Transformers 缺少 glm5_next；请使用完整 CCCP "
                "离线环境，启动器不应在运行时联网下载依赖"
            ) from error
        return AutoConfig, DynamicCache, Glm5NextTextModel

    def _load_source(self, names: Iterable[str]):
        names = tuple(names)
        missing = [name for name in names if not self.store.has(name)]
        if missing:
            raise KeyError(f"GLM-5.3 dense archive is missing {missing}")
        values = [self.store.get_dense(name) for name in names]
        if len(values) == 1:
            return values[0]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(
                "GLM-5.3 converted convolution weights must be tensors"
            )
        return torch.cat(values, dim=0)

    def _place_weight(self, value, *, matrix: bool):
        if isinstance(value, BlockFP8Weight):
            if not matrix:
                value = value.to(torch.bfloat16)
            elif self.device.type == "cpu":
                value = value.optimize_cpu_layout()
            else:
                value = value.to(self.device)
            return value
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"unsupported GLM-5.3 dense value {type(value)!r}")
        return value.to(self.device)

    def _replace_sparse_experts(self, model: nn.Module) -> None:
        limit = float(getattr(model.config, "swiglu_limit", 10.0))
        for layer, decoder in enumerate(model.layers):
            experts = getattr(getattr(decoder, "mlp", None), "experts", None)
            if experts is None:
                continue
            decoder.mlp.experts = GLM5NextPackedExperts(
                self.routed_vq,
                layer=layer,
                swiglu_limit=limit,
            )

    @staticmethod
    def _replace_sparse_routers(model: nn.Module) -> int:
        replaced = 0
        for decoder in model.layers:
            gate = getattr(getattr(decoder, "mlp", None), "gate", None)
            if gate is None:
                continue
            decoder.mlp.gate = GLM5NextTopkRouterRuntime(gate)
            replaced += 1
        return replaced

    def _replace_linears(self, model: nn.Module) -> int:
        replaced = 0
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            source_names = _glm5_next_source_names(name + ".weight")
            weight = self._place_weight(
                self._load_source(source_names),
                matrix=True,
            )
            bias_names = _glm5_next_source_names(name + ".bias")
            bias = (
                self._place_weight(
                    self._load_source(bias_names),
                    matrix=False,
                )
                if all(self.store.has(item) for item in bias_names)
                else None
            )
            parent, leaf = _parent_and_leaf(model, name)
            setattr(parent, leaf, CCCPLinear(weight, bias))
            replaced += 1
        return replaced

    def _install_public_decoder_layers(self, model: nn.Module) -> None:
        decode_graph_enabled = (
            self.device.type == "cuda"
            and self.tp_size > 1
            and os.environ.get("CCCP_LINEAR_DECODE_GRAPH", "1") != "0"
        )
        for index, layer in enumerate(model.layers):
            decode_graph_pool = None
            if decode_graph_enabled and layer.block_type == "linear_attention":
                with torch.cuda.device(self.device):
                    decode_graph_pool = torch.cuda.graph_pool_handle()
            model.layers[index] = GLM5NextDecoderLayerRuntime(
                layer,
                decode_graph_pool=decode_graph_pool,
            )

    @staticmethod
    def _install_static_sparse_index_pools(model: nn.Module) -> None:
        for decoder in model.layers:
            indexer = getattr(
                getattr(decoder, "self_attn", None),
                "indexer",
                None,
            )
            if indexer is None:
                continue
            indexer.get_pooled_states = MethodType(
                _glm5_next_static_pool_states,
                indexer,
            )

    def _assign_remaining_state(self, model: nn.Module) -> int:
        assigned = 0
        for name, _parameter in list(model.named_parameters()):
            source_names = _glm5_next_source_names(name)
            value = self._place_weight(
                self._load_source(source_names),
                matrix=False,
            )
            if isinstance(value, BlockFP8Weight):
                raise TypeError(f"non-linear parameter stayed compact: {name}")
            parent, leaf = _parent_and_leaf(model, name)
            parent._parameters[leaf] = nn.Parameter(
                value,
                requires_grad=False,
            )
            assigned += 1
        for name, buffer in list(model.named_buffers()):
            source_names = _glm5_next_source_names(name)
            if all(self.store.has(item) for item in source_names):
                value = self._place_weight(
                    self._load_source(source_names),
                    matrix=False,
                )
                if isinstance(value, BlockFP8Weight):
                    raise TypeError(f"buffer stayed compact: {name}")
                parent, leaf = _parent_and_leaf(model, name)
                parent._buffers[leaf] = value
                assigned += 1
            elif buffer.device.type == "meta":
                raise KeyError(f"GLM-5.3 buffer has no archive source: {name}")
        meta = [
            name
            for name, value in list(model.named_parameters())
            + list(model.named_buffers())
            if value.device.type == "meta"
        ]
        if meta:
            raise RuntimeError(f"GLM-5.3 unresolved meta tensors: {meta[:8]}")
        return assigned

    def _build_text_model(self) -> None:
        AutoConfig, _DynamicCache, Glm5NextTextModel = (
            self._require_transformers()
        )
        from transformers.models.glm5_next import modeling_glm5_next

        for name in (
            "chunk_kimi_delta_attention",
            "recurrent_kimi_delta_attention",
        ):
            implementation = getattr(modeling_glm5_next, name)
            if not getattr(implementation, "_cccp_public_kda", False):
                setattr(
                    modeling_glm5_next,
                    name,
                    _glm5_next_delta_rule(implementation),
                )
        outer = AutoConfig.from_pretrained(
            self.root,
            local_files_only=True,
        )
        self._outer_config = outer
        config = outer.get_text_config()
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = Glm5NextTextModel(config)
        self._replace_sparse_experts(model)
        linears = self._replace_linears(model)
        state = self._assign_remaining_state(model)
        routers = self._replace_sparse_routers(model)
        self._install_static_sparse_index_pools(model)
        self._install_public_decoder_layers(model)
        lm_head = self._place_weight(
            self.store.get_dense("lm_head.weight"),
            matrix=True,
        )
        self._lm_head = CCCPLinear(lm_head)
        self._model = model.eval()
        self._text_config = config
        print(
            "[cccp-glm5-next] 官方文本拓扑绑定完成："
            f"公共 Linear={linears}，公共 Router={routers}，"
            f"非线性状态={state}；"
            "视觉塔未载入",
            flush=True,
        )

    @staticmethod
    def _vision_parent_leaf(module: nn.Module, name: str) -> tuple[nn.Module, str]:
        return _parent_and_leaf(module, name)

    @torch.inference_mode()
    def forward_multimodal_prefill(
        self,
        ids: list[int],
        *,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Run one image prompt through the official GLM visual injection path.

        This is deliberately separate from ``forward``: normal text requests
        retain the existing fixed-address TokenGraph and never import or load
        the vision tower.
        """
        if not ids:
            raise ValueError("GLM multimodal prefill requires prompt tokens")
        if self._model is None or self._lm_head is None:
            raise RuntimeError("GLM-5.3 model has not been preloaded")
        from .glm5_next_multimodal import load_vision_model
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        load_started = time.perf_counter()
        vision = load_vision_model(self)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        load_seconds = time.perf_counter() - load_started
        vision_dtype = next(vision.parameters()).dtype
        pixel_values = pixel_values.to(device=self.device, dtype=vision_dtype)
        image_grid_thw = image_grid_thw.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        vision_started = time.perf_counter()
        image_output = vision(pixel_values, grid_thw=image_grid_thw)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        vision_seconds = time.perf_counter() - vision_started
        image_embeds = image_output.pooler_output
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = torch.cat(list(image_embeds), dim=0)
        image_embeds = image_embeds.to(self.device, dtype=self._model.embed_tokens.weight.dtype)
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        inputs_embeds = self._model.embed_tokens(input_ids)
        image_mask = (input_ids == int(self._outer_config.image_token_id)).unsqueeze(-1)
        expected = int(image_mask.sum().item())
        if expected != int(image_embeds.shape[0]):
            raise ValueError(
                f"GLM image placeholder/features mismatch: tokens={expected}, features={image_embeds.shape[0]}"
            )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        self._multimodal_active = True
        self._fixed_token_graph = None
        self._fixed_token_graph_cache = None
        self._fixed_token_graph_capacity = 0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        language_started = time.perf_counter()
        with _glm5_next_pool_phase(self.routed_vq, rows=len(ids)):
            output = self._model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        language_seconds = time.perf_counter() - language_started
        self.last_multimodal_timings = {
            "vision_load_ms": load_seconds * 1000.0,
            "vision_forward_ms": vision_seconds * 1000.0,
            "language_prefill_ms": language_seconds * 1000.0,
        }
        self._past_key_values = output.past_key_values
        self.pos = len(ids)
        self._token_history = list(ids)
        return self._lm_head(output.last_hidden_state[:, -1]).float().squeeze(0)

    @torch.inference_mode()
    def preload_vision(self) -> float:
        """Opt in to loading the image tower without changing text startup."""
        from .glm5_next_multimodal import load_vision_model

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        load_vision_model(self)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - started

    def preload(self) -> None:
        started = time.perf_counter()
        from .ops import configure_dynamic_sdpa_backends

        configure_dynamic_sdpa_backends(self.device)
        if self.device.type == "cpu":
            from .cpuext import prebuild as prebuild_cpu

            prebuild_cpu()
        self._build_text_model()
        self.routed_vq.initialize_residency(
            device_type=self.device.type,
        )
        if self._fixed_token_graph_enabled():
            self._capture_fixed_token_graph()
        print(
            "[cccp-glm5-next] 模型加载完成："
            f"device={self.device}，elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    def prefill_batch_available(self) -> bool:
        return True

    @property
    def device_greedy_supported(self) -> bool:
        """Advertise the public graph's device-token feedback capability."""
        return self._fixed_token_graph is not None

    def prepare_context_capacity(self, tokens: int) -> None:
        """Prepare a fixed-address KV bucket outside a measured/request path."""
        self._ensure_fixed_token_capacity(int(tokens))

    def _fixed_token_graph_enabled(self) -> bool:
        return bool(
            self.device.type == "cuda"
            and self.tp_size == 1
            and self.routed_vq.full_resident
            and os.environ.get("CCCP_FIXED_TOKEN_GRAPH", "1") != "0"
        )

    @torch.inference_mode()
    def _capture_fixed_token_graph(
        self,
        capacity: int | None = None,
    ) -> None:
        """Capture the official topology through the public graph scheduler.

        A four-token sacrificial Prefill initializes every static and
        recurrent cache before capture.  Capturing from an empty cache would
        record the one-time Prefill branch and is not a reusable Decode graph.
        Routed codebook math remains owned by ``RoutedVQExecutor``; this
        adapter only binds the model topology and cache tensors.
        """
        if self._model is None or self._lm_head is None:
            raise RuntimeError("GLM-5.3 graph capture requires a loaded model")
        from transformers import StaticCache
        from .ops import FixedTokenGraph, fixed_token_capacity

        capacity = fixed_token_capacity(
            1 if capacity is None else int(capacity),
            limit=self.max_ctx,
        )

        cache = StaticCache(
            config=self._text_config,
            max_cache_len=capacity,
        )
        self._past_key_values = cache
        self._fixed_token_graph_cache = cache
        self._fixed_token_graph_capacity = capacity
        if self._device_token_history is None:
            self._device_token_history = torch.empty(
                self.max_ctx,
                dtype=torch.long,
                device=self.device,
            )
        warm_ids = torch.zeros(
            (1, min(4, self.max_ctx)),
            dtype=torch.long,
            device=self.device,
        )
        with _glm5_next_pool_phase(
            self.routed_vq,
            rows=int(warm_ids.shape[1]),
        ):
            self._model(
                input_ids=warm_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )

        token = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        position = torch.full(
            (1,),
            int(warm_ids.shape[1]),
            dtype=torch.long,
            device=self.device,
        )

        def token_step():
            with _glm5_next_pool_phase(self.routed_vq, rows=1):
                output = self._model(
                    input_ids=token,
                    past_key_values=cache,
                    use_cache=True,
                    position_ids=position.view(1, 1),
                    return_dict=True,
                )
            hidden = output.last_hidden_state[:, -1]
            logits = self._lm_head(hidden).float()
            return hidden, logits

        started = time.perf_counter()
        self._fixed_token_graph = FixedTokenGraph(
            self.device,
            token=token,
            position=position,
            function=token_step,
        )
        torch.cuda.synchronize(self.device)
        cache.reset()
        self.pos = 0
        self._token_history.clear()
        print(
            "[cccp-glm5-next] 公共固定地址 TokenGraph 完成："
            f"Decode 全拓扑，cache_bucket={capacity}，"
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    @torch.inference_mode()
    def _ensure_fixed_token_capacity(self, required: int) -> None:
        if (
            self._fixed_token_graph is None
            or int(required) <= self._fixed_token_graph_capacity
        ):
            return
        from .ops import fixed_token_capacity

        target = fixed_token_capacity(required, limit=self.max_ctx)
        history = (
            self._device_token_history[:self.pos].cpu().tolist()
            if self._device_token_history is not None and self.pos
            else list(self._token_history)
        )
        self._fixed_token_graph = None
        self._fixed_token_graph_cache = None
        self._past_key_values = None
        gc.collect()
        torch.cuda.empty_cache()
        self._capture_fixed_token_graph(target)
        if history:
            self._forward_eager_hidden([int(item) for item in history])
            self.pos = len(history)
            self._token_history[:] = [int(item) for item in history]
        print(
            "[cccp-glm5-next] TokenGraph KV 桶扩展："
            f"required={required}，capacity={target}",
            flush=True,
        )

    def _forward_fixed_token(
        self,
        token: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fixed_token_graph is None:
            raise RuntimeError("GLM-5.3 fixed token graph is unavailable")
        self._ensure_fixed_token_capacity(self.pos + 1)
        graph = self._fixed_token_graph
        if graph is None:
            raise RuntimeError("GLM-5.3 fixed token graph recapture failed")
        history = self._device_token_history
        if history is not None:
            if isinstance(token, torch.Tensor):
                history[self.pos].copy_(token.reshape(-1)[0])
            else:
                history[self.pos].fill_(int(token))
        hidden, logits = graph.replay(token, self.pos)
        self.pos += 1
        if not isinstance(token, torch.Tensor):
            self._token_history.append(int(token))
        return hidden, logits

    def _forward_eager_hidden(self, ids: list[int]) -> torch.Tensor:
        input_ids = torch.tensor(
            ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        with _glm5_next_pool_phase(self.routed_vq, rows=len(ids)):
            output = self._model(
                input_ids=input_ids,
                past_key_values=self._past_key_values,
                use_cache=True,
                return_dict=True,
            )
        self._past_key_values = output.past_key_values
        if self._device_token_history is not None:
            self._device_token_history[
                self.pos:self.pos + len(ids)
            ].copy_(input_ids[0])
        return output.last_hidden_state.squeeze(0)

    @torch.inference_mode()
    def reset_kv(self) -> None:
        if (
            self._fixed_token_graph_cache is not None
            and self._past_key_values is self._fixed_token_graph_cache
        ):
            self._past_key_values.reset()
        else:
            self._past_key_values = None
            if self._model is not None:
                for decoder in self._model.layers:
                    reset_graph = getattr(decoder, "reset_decode_graph", None)
                    if callable(reset_graph):
                        reset_graph()
        self._token_history.clear()
        self.pos = 0

    def reset(self) -> None:
        self.reset_kv()
        # A media turn is request-local. Rebuild the normal text graph lazily
        # only if a later text request needs it; vision weights remain loaded
        # but are never touched by the text-only forward path.
        if self._multimodal_active:
            self._multimodal_active = False
            if self._fixed_token_graph_enabled():
                self._capture_fixed_token_graph()

    @torch.inference_mode()
    def truncate_kv(self, keep: int) -> None:
        keep = max(0, int(keep))
        if self._past_key_values is None:
            self.pos = 0
            self._token_history.clear()
            return
        if self._past_key_values is self._fixed_token_graph_cache:
            if self._device_token_history is not None and keep:
                retained = [
                    int(item)
                    for item in self._device_token_history[:keep].cpu().tolist()
                ]
            else:
                retained = self._token_history[:keep]
            self._past_key_values.reset()
            self.pos = 0
            self._token_history.clear()
            if retained:
                self._forward_eager_hidden(retained)
                self.pos = len(retained)
                self._token_history.extend(retained)
            return
        crop = getattr(self._past_key_values, "crop", None)
        if not callable(crop):
            raise RuntimeError("glm5_next DynamicCache does not support crop")
        crop(keep)
        self.pos = keep

    @torch.inference_mode()
    def forward_hidden(self, ids: list[int]) -> torch.Tensor:
        if self._multimodal_active:
            if not ids:
                raise ValueError("GLM-5.3 forward requires at least one token")
            input_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
            with _glm5_next_pool_phase(self.routed_vq, rows=len(ids)):
                output = self._model(
                    input_ids=input_ids,
                    past_key_values=self._past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            self._past_key_values = output.past_key_values
            self.pos += len(ids)
            self._token_history.extend(int(item) for item in ids)
            return output.last_hidden_state.squeeze(0)
        if not ids:
            raise ValueError("GLM-5.3 forward requires at least one token")
        if self._model is None or self._lm_head is None:
            raise RuntimeError("GLM-5.3 model has not been preloaded")
        if self.pos + len(ids) > self.max_ctx:
            raise RuntimeError(
                f"GLM-5.3 context {self.pos + len(ids)} exceeds "
                f"max_ctx={self.max_ctx}"
            )
        self._ensure_fixed_token_capacity(self.pos + len(ids))
        if len(ids) == 1 and self._fixed_token_graph is not None:
            hidden, _logits = self._forward_fixed_token(ids[0])
            return hidden
        hidden = self._forward_eager_hidden(ids)
        self.pos += len(ids)
        self._token_history.extend(int(item) for item in ids)
        return hidden

    def logits_of(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._lm_head is None:
            raise RuntimeError("GLM-5.3 lm_head has not been preloaded")
        return self._lm_head(hidden).float()

    @torch.inference_mode()
    def forward(self, ids: list[int]) -> torch.Tensor:
        if len(ids) == 1 and self._fixed_token_graph is not None:
            _hidden, logits = self._forward_fixed_token(ids[0])
            return logits.squeeze(0)
        hidden = self.forward_hidden(ids)
        return self.logits_of(hidden[-1:]).squeeze(0)


__all__ = [
    "CCCPLinear",
    "GLM5NextCCCPModel",
    "GLM5NextDecoderLayerRuntime",
    "GLM5NextPackedExperts",
    "GLM5NextTopkRouterRuntime",
    "_glm5_next_delta_rule",
    "_glm5_next_hc_post",
    "_glm5_next_hc_pre_norm",
    "_glm5_next_kda_raw_inputs",
    "_glm5_next_linear_decode_graph_eligible",
    "_glm5_next_pool_phase",
    "_glm5_next_static_pool_states",
]

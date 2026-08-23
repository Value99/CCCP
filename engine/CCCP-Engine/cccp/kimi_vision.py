"""Configuration-driven Kimi K3 MoonViT3D vision runtime.

This module is intentionally independent from ``transformers`` and from the
text-generation protocol.  It consumes dense weights from a ``CCCPStore``
(``get_dense(name)``), a mapping, or a callable.  Names are formed as
``config.prefix + configured_name``.  The default convention is::

    patch_embed.proj.weight
    blocks.{layer}.norm1.weight
    blocks.{layer}.attn.qkv.weight       # or q/k/v_proj.weight
    blocks.{layer}.attn.proj.weight
    blocks.{layer}.norm2.weight
    blocks.{layer}.mlp.fc1.weight
    blocks.{layer}.mlp.fc2.weight
    merger.ln_q.weight
    merger.mlp.0.weight
    merger.mlp.2.weight

All suffixes are configurable because this repository does not include the
upstream Kimi K3 model or vision configuration.  In particular, checkpoint
owners must set dimensions, qkv layout, normalization, activation, merger
layout, biases, and name suffixes from the actual checkpoint metadata.  The
runtime implements full, non-causal attention together with Kimi's repeated
2D rotary embedding and divided spatial/temporal patch position embedding.

Projection matrices may be ordinary tensors or compact ``BlockFP8Weight``
objects.  Every projection goes through the public ``linear_batch`` operator;
compact weights are never passed to a torch normalization operation.  Norm
vectors are explicitly materialized when a store happens to encode one as
BlockFP8.  Keeping production projections in compact block FP8 is what permits
the expected roughly 431 MiB single-device visual-weight residency; the exact
size is checkpoint/config dependent and is available through
``resident_weight_bytes`` after weights have been loaded.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .kernels import BlockFP8Weight, Int4Weight
from .ops import linear_batch


WeightSource = Mapping[str, Any] | Callable[[str], Any] | Any


@dataclass(frozen=True)
class KimiVisionConfig:
    """Explicit architecture and checkpoint naming for MoonViT3D."""

    in_channels: int = 3
    patch_size: int = 14
    temporal_patch_size: int = 1
    hidden_size: int = 1024
    num_hidden_layers: int = 27
    num_attention_heads: int = 12
    qkv_hidden_size: int = 1536
    intermediate_size: int = 4096
    output_size: int = 7168
    spatial_merge_size: int = 2
    max_spatial_size: int = 512
    max_temporal_size: int = 4
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    norm_type: str = "rmsnorm"
    activation: str = "gelu_tanh"
    qkv_layout: str = "fused"
    mlp_layout: str = "plain"
    use_bias: bool = False
    norm_bias: bool = False
    prefix: str = "vision_tower."
    projector_prefix: str = "mm_projector."
    patch_embed_weight: str = "patch_embed.proj.weight"
    patch_embed_bias: str = "patch_embed.proj.bias"
    block_prefix: str = "encoder.blocks.{layer}."
    norm1_weight: str = "norm0.weight"
    norm1_bias: str = "norm0.bias"
    norm2_weight: str = "norm1.weight"
    norm2_bias: str = "norm1.bias"
    qkv_weight: str = "wqkv.weight"
    qkv_bias: str = "wqkv.bias"
    q_weight: str = "wqkv.weight"
    q_bias: str = "wqkv.bias"
    k_weight: str = "wqkv.weight"
    k_bias: str = "wqkv.bias"
    v_weight: str = "wqkv.weight"
    v_bias: str = "wqkv.bias"
    attn_out_weight: str = "wo.weight"
    attn_out_bias: str = "wo.bias"
    mlp_in_weight: str = "mlp.fc0.weight"
    mlp_in_bias: str = "mlp.fc0.bias"
    mlp_gate_weight: str = "mlp.gate_proj.weight"
    mlp_gate_bias: str = "mlp.gate_proj.bias"
    mlp_up_weight: str = "mlp.up_proj.weight"
    mlp_up_bias: str = "mlp.up_proj.bias"
    mlp_out_weight: str = "mlp.fc1.weight"
    mlp_out_bias: str = "mlp.fc1.bias"
    merger_norm_weight: str = "post_norm.weight"
    merger_norm_bias: str = "post_norm.bias"
    merger_in_weight: str = "../mm_projector.proj.0.weight"
    merger_in_bias: str = "../mm_projector.proj.0.bias"
    merger_out_weight: str = "../mm_projector.proj.2.weight"
    merger_out_bias: str = "../mm_projector.proj.2.bias"
    final_norm_weight: str = "encoder.final_layernorm.weight"
    final_norm_bias: str = "encoder.final_layernorm.bias"
    position_embedding: str = "learned"
    position_embedding_table: str | None = "patch_embed.pos_emb.weight"
    dtype: torch.dtype = torch.bfloat16
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        positive = {
            "in_channels": self.in_channels,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "output_size": self.output_size,
            "spatial_merge_size": self.spatial_merge_size,
            "max_spatial_size": self.max_spatial_size,
            "max_temporal_size": self.max_temporal_size,
        }
        if self.num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be non-negative")
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.qkv_hidden_size % self.num_attention_heads:
            raise ValueError("qkv_hidden_size must be divisible by num_attention_heads")
        if self.qkv_hidden_size // self.num_attention_heads % 4:
            raise ValueError("vision attention head size must be divisible by four")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'")
        if self.activation not in {"gelu", "gelu_tanh", "silu"}:
            raise ValueError("unsupported activation")
        if self.qkv_layout not in {"fused", "separate"}:
            raise ValueError("qkv_layout must be 'fused' or 'separate'")
        if self.mlp_layout not in {"plain", "gated"}:
            raise ValueError("mlp_layout must be 'plain' or 'gated'")
        if self.position_embedding not in {"zero", "learned"}:
            raise ValueError("position_embedding must be 'zero' or 'learned'")
        if self.position_embedding == "learned" and not self.position_embedding_table:
            raise ValueError("learned position_embedding requires position_embedding_table")

    @property
    def patch_dim(self) -> int:
        return (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )


class PatchMergerMLPV2:
    """Public spatial patch merger: normalize, pool merge windows, then MLP."""

    def __init__(self, runtime: "KimiVisionRuntime") -> None:
        self.runtime = runtime

    def __call__(self, value: torch.Tensor, grid_thw: Sequence[int]) -> torch.Tensor:
        return self.runtime._merger_impl(value, grid_thw)


class KimiVisionRuntime:
    """Small dependency-free execution runtime for MoonViT3D + merger."""

    def __init__(self, config: KimiVisionConfig, weights: WeightSource) -> None:
        self.config = config
        self._source = weights
        self.device = torch.device(config.device)
        self._weights: dict[str, torch.Tensor | BlockFP8Weight] = {}
        self.patch_merger = PatchMergerMLPV2(self)

    def _full_name(self, suffix: str) -> str:
        return f"{self.config.prefix}{suffix}"

    def _projector_name(self, suffix: str) -> str:
        return f"{self.config.projector_prefix}{suffix}"

    def _load_source(self, name: str) -> Any:
        source = self._source
        if isinstance(source, Mapping):
            return source[name]
        get_dense = getattr(source, "get_dense", None)
        if callable(get_dense):
            return get_dense(name)
        if callable(source):
            return source(name)
        raise TypeError(
            "weights must be a mapping, callable, or object with get_dense"
        )

    def _weight(
        self,
        suffix: str,
        *,
        norm: bool = False,
        projector: bool = False,
    ) -> torch.Tensor | BlockFP8Weight:
        name = (
            self._projector_name(suffix)
            if projector
            else self._full_name(suffix)
        )
        cached = self._weights.get(name)
        if cached is not None:
            return cached
        try:
            value = self._load_source(name)
        except (KeyError, LookupError) as exc:
            raise KeyError(f"missing Kimi vision weight {name!r}") from exc
        if not isinstance(value, (torch.Tensor, BlockFP8Weight)):
            if isinstance(value, Int4Weight):
                raise TypeError(
                    f"Kimi vision weight {name!r} is Int4Weight; vision projections "
                    "require Tensor or BlockFP8Weight"
                )
            raise TypeError(
                f"Kimi vision weight {name!r} has unsupported type "
                f"{type(value)!r}"
            )
        if norm and isinstance(value, BlockFP8Weight):
            value = value.to(device=self.device, dtype=self.config.dtype)
        elif isinstance(value, BlockFP8Weight):
            value = value.to(self.device)
        else:
            value = value.to(self.device)
        self._weights[name] = value
        return value

    def _tensor_weight(
        self,
        suffix: str,
        *,
        norm: bool = False,
        projector: bool = False,
    ) -> torch.Tensor:
        value = self._weight(suffix, norm=norm, projector=projector)
        if not isinstance(value, torch.Tensor):
            name = (
                self._projector_name(suffix)
                if projector
                else self._full_name(suffix)
            )
            raise TypeError(f"weight {name!r} must be a tensor here")
        return value

    def _block_name(self, layer: int, suffix: str) -> str:
        return self.config.block_prefix.format(layer=layer) + suffix

    def _linear(
        self,
        value: torch.Tensor,
        weight_suffix: str,
        bias_suffix: str | None = None,
        *,
        projector: bool = False,
    ) -> torch.Tensor:
        weight = self._weight(weight_suffix, projector=projector)
        result = linear_batch(value, weight, output_dtype=self.config.dtype)
        if bias_suffix is not None:
            bias = self._tensor_weight(bias_suffix).to(result.dtype)
            result = result + bias
        return result

    def _norm(
        self,
        value: torch.Tensor,
        weight_suffix: str,
        bias_suffix: str,
        *,
        projector: bool = False,
    ) -> torch.Tensor:
        weight = self._tensor_weight(
            weight_suffix,
            norm=True,
            projector=projector,
        )
        if weight_suffix in {"post_norm.weight", "encoder.final_layernorm.weight"}:
            weight = weight.flatten()
        if weight.numel() != value.shape[-1]:
            raise ValueError(
                f"norm {self._full_name(weight_suffix)!r} has "
                f"{weight.numel()} values, expected {value.shape[-1]}"
            )
        bias = None
        if self.config.norm_bias:
            bias = self._tensor_weight(bias_suffix, norm=True).flatten()
        if self.config.norm_type == "layernorm":
            return F.layer_norm(
                value,
                (value.shape[-1],),
                weight.to(value.dtype),
                None if bias is None else bias.to(value.dtype),
                self.config.norm_eps,
            )
        variance = value.float().pow(2).mean(dim=-1, keepdim=True)
        result = value.float() * torch.rsqrt(variance + self.config.norm_eps)
        result = result * weight.float()
        if bias is not None:
            result = result + bias.float()
        return result.to(value.dtype)

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        if self.config.activation == "silu":
            return F.silu(value)
        return F.gelu(
            value,
            approximate="tanh" if self.config.activation == "gelu_tanh" else "none",
        )

    def patchify(
        self,
        pixel_values: torch.Tensor,
        grid_thws: Sequence[Sequence[int]] | torch.Tensor,
    ) -> list[torch.Tensor]:
        """Convert flat ``[N,C,P,P]`` or padded ``[B,C,T,H,W]`` pixels."""
        if pixel_values.ndim == 4:
            channels, patch, patch_w = pixel_values.shape[1:]
            if channels != self.config.in_channels or patch != self.config.patch_size or patch_w != patch:
                raise ValueError("flat pixel_values must have shape [N,C,patch_size,patch_size]")
            grids = grid_thws.tolist() if isinstance(grid_thws, torch.Tensor) else grid_thws
            output = []
            offset = 0
            for raw_grid in grids:
                if len(raw_grid) != 3:
                    raise ValueError("every grid_thws row must contain T, H, W")
                grid_t, grid_h, grid_w = (int(item) for item in raw_grid)
                count = grid_t * grid_h * grid_w
                if min(grid_t, grid_h, grid_w) <= 0 or offset + count > pixel_values.shape[0]:
                    raise ValueError("grid_thws patch count does not match flat pixel_values")
                output.append(pixel_values[offset:offset + count].reshape(count, -1))
                offset += count
            if offset != pixel_values.shape[0]:
                raise ValueError("flat pixel_values has unused patches")
            return output
        if pixel_values.ndim != 5:
            raise ValueError("pixel_values must have shape [N,C,P,P] or [B,C,T,H,W]")
        batch, channels, frames, height, width = pixel_values.shape
        if channels != self.config.in_channels:
            raise ValueError(
                f"pixel channels {channels} != configured {self.config.in_channels}"
            )
        grids = grid_thws.tolist() if isinstance(grid_thws, torch.Tensor) else grid_thws
        if len(grids) != batch:
            raise ValueError("grid_thws must contain one [T,H,W] row per batch item")
        temporal = self.config.temporal_patch_size
        patch = self.config.patch_size
        output: list[torch.Tensor] = []
        for index, raw_grid in enumerate(grids):
            if len(raw_grid) != 3:
                raise ValueError("every grid_thws row must contain T, H, W")
            grid_t, grid_h, grid_w = (int(item) for item in raw_grid)
            if min(grid_t, grid_h, grid_w) <= 0:
                raise ValueError("grid_thws values must be positive")
            used_t, used_h, used_w = (
                grid_t * temporal,
                grid_h * patch,
                grid_w * patch,
            )
            if used_t > frames or used_h > height or used_w > width:
                raise ValueError(
                    "grid_thws exceeds the corresponding pixel_values extent"
                )
            item = pixel_values[index, :, :used_t, :used_h, :used_w]
            item = item.reshape(
                channels,
                grid_t,
                temporal,
                grid_h,
                patch,
                grid_w,
                patch,
            )
            item = item.permute(1, 3, 5, 0, 2, 4, 6).contiguous()
            output.append(item.reshape(grid_t * grid_h * grid_w, -1))
        return output

    def _patch_embed(
        self,
        patches: torch.Tensor,
        grid_thw: Sequence[int],
    ) -> torch.Tensor:
        weight = self._weight(self.config.patch_embed_weight)
        if isinstance(weight, torch.Tensor) and weight.ndim > 2:
            weight = weight.flatten(1)
        if tuple(weight.shape) != (self.config.hidden_size, self.config.patch_dim):
            raise ValueError(
                "patch embedding weight must have logical shape "
                f"[{self.config.hidden_size}, {self.config.patch_dim}], got "
                f"{tuple(weight.shape)}"
            )
        result = linear_batch(
            patches.to(self.device, dtype=self.config.dtype),
            weight,
            output_dtype=self.config.dtype,
        )
        if self.config.use_bias:
            result = result + self._tensor_weight(
                self.config.patch_embed_bias
            ).to(result.dtype)
        table = None
        if self.config.position_embedding_table is not None:
            try:
                table = self._tensor_weight("patch_embed.pos_emb.weight")
            except KeyError:
                table = None
        if table is None:
            return result
        grid_t, grid_h, grid_w = (int(item) for item in grid_thw)
        if table.ndim != 3 or table.shape[-1] != result.shape[-1]:
            raise ValueError("invalid Kimi vision position embedding shape")
        position = table.permute(2, 0, 1).unsqueeze(0)
        if (grid_h, grid_w) != tuple(table.shape[:2]):
            position = F.interpolate(
                position.float(),
                size=(grid_h, grid_w),
                mode="bicubic",
                align_corners=False,
            ).to(result.dtype)
        position = position.squeeze(0).permute(1, 2, 0).reshape(-1, result.shape[-1])
        if grid_t > 1:
            time = torch.arange(grid_t, device=self.device, dtype=torch.float32)
            half = result.shape[-1] // 2
            omega = torch.arange(half, device=self.device, dtype=torch.float32)
            omega = 1.0 / (10000.0 ** (omega / half))
            phase = time[:, None] * omega[None, :]
            temporal = torch.cat((phase.sin(), phase.cos()), dim=-1)
            if temporal.shape[-1] < result.shape[-1]:
                temporal = F.pad(temporal, (0, result.shape[-1] - temporal.shape[-1]))
            temporal = temporal[:, None, :].expand(grid_t, position.shape[0], -1)
            position = position[None, :, :].expand(grid_t, -1, -1) + temporal
            position = position.reshape(-1, result.shape[-1])
        return result + position

    def _rope_cos_sin(self, grid_thw: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build Kimi's repeated spatial 2D rotary frequencies.

        Kimi pairs even rotary dimensions with the width coordinate and odd
        pairs with the height coordinate.  Video frames reuse the same spatial
        frequencies; temporal information is supplied by patch position
        embeddings rather than a third rotary axis.
        """
        grid_t, grid_h, grid_w = (int(item) for item in grid_thw)
        if grid_h > self.config.max_spatial_size or grid_w > self.config.max_spatial_size:
            raise ValueError("vision grid exceeds configured rotary position limit")
        head_dim = self.config.qkv_hidden_size // self.config.num_attention_heads
        pair_count = head_dim // 2
        axis_count = pair_count // 2
        index = torch.arange(axis_count, device=self.device, dtype=torch.float32)
        inverse = self.config.rope_theta ** (-4.0 * index / head_dim)
        y, x = torch.meshgrid(
            torch.arange(grid_h, device=self.device, dtype=torch.float32),
            torch.arange(grid_w, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        phases = torch.empty(grid_h * grid_w, pair_count, device=self.device, dtype=torch.float32)
        phases[:, 0::2] = x.reshape(-1, 1) * inverse
        phases[:, 1::2] = y.reshape(-1, 1) * inverse
        phases = phases.repeat(grid_t, 1)
        return phases.cos(), phases.sin()

    @staticmethod
    def _rotate_pairs(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        pairs = value.float().reshape(value.shape[0], value.shape[1], -1, 2)
        cosine = cos.unsqueeze(0)
        sine = sin.unsqueeze(0)
        real, imag = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack((real * cosine - imag * sine, real * sine + imag * cosine), dim=-1)
        return rotated.flatten(-2).to(value.dtype)

    def _apply_rope(self, query: torch.Tensor, key: torch.Tensor, grid_thw: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._rope_cos_sin(grid_thw)
        return self._rotate_pairs(query, cos, sin), self._rotate_pairs(key, cos, sin)

    def _attention(self, value: torch.Tensor, layer: int, grid_thw: Sequence[int]) -> torch.Tensor:
        cfg = self.config
        prefix = cfg.block_prefix.format(layer=layer)
        bias = (lambda name: prefix + name) if cfg.use_bias else (lambda name: None)
        if cfg.qkv_layout == "fused":
            qkv = self._linear(
                value,
                prefix + cfg.qkv_weight,
                bias(cfg.qkv_bias),
            )
        else:
            raise ValueError("Kimi vision runtime requires fused qkv layout")
        if qkv.shape[-1] != 3 * cfg.qkv_hidden_size:
            raise ValueError("fused qkv projection must output 3 * qkv_hidden_size")
        query, key, projected_value = qkv.chunk(3, dim=-1)
        heads = cfg.num_attention_heads
        head_dim = cfg.qkv_hidden_size // heads
        tokens = value.shape[0]
        query = query.reshape(tokens, heads, head_dim).transpose(0, 1)
        key = key.reshape(tokens, heads, head_dim).transpose(0, 1)
        projected_value = projected_value.reshape(
            tokens, heads, head_dim
        ).transpose(0, 1)
        query, key = self._apply_rope(query, key, grid_thw)
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
        probabilities = torch.softmax(scores / math.sqrt(head_dim), dim=-1)
        attended = torch.matmul(probabilities.to(projected_value.dtype), projected_value)
        attended = attended.transpose(0, 1).reshape(tokens, cfg.qkv_hidden_size)
        return self._linear(
            attended,
            prefix + cfg.attn_out_weight,
            bias(cfg.attn_out_bias),
        )

    def _mlp(self, value: torch.Tensor, layer: int) -> torch.Tensor:
        cfg = self.config
        prefix = cfg.block_prefix.format(layer=layer)
        optional_bias = lambda name: prefix + name if cfg.use_bias else None
        if cfg.mlp_layout == "gated":
            gate = self._linear(
                value,
                prefix + cfg.mlp_gate_weight,
                optional_bias(cfg.mlp_gate_bias),
            )
            up = self._linear(
                value,
                prefix + cfg.mlp_up_weight,
                optional_bias(cfg.mlp_up_bias),
            )
            hidden = self._activate(gate) * up
        else:
            hidden = self._activate(
                self._linear(
                    value,
                    prefix + cfg.mlp_in_weight,
                    optional_bias(cfg.mlp_in_bias),
                )
            )
        return self._linear(
            hidden,
            prefix + cfg.mlp_out_weight,
            optional_bias(cfg.mlp_out_bias),
        )

    def _block(self, value: torch.Tensor, layer: int, grid_thw: Sequence[int]) -> torch.Tensor:
        cfg = self.config
        prefix = cfg.block_prefix.format(layer=layer)
        normalized = self._norm(
            value,
            prefix + cfg.norm1_weight,
            prefix + cfg.norm1_bias,
        )
        value = value + self._attention(normalized, layer, grid_thw)
        normalized = self._norm(
            value,
            prefix + cfg.norm2_weight,
            prefix + cfg.norm2_bias,
        )
        return value + self._mlp(normalized, layer)

    def _merge_tokens(
        self,
        value: torch.Tensor,
        grid_thw: Sequence[int],
    ) -> torch.Tensor:
        grid_t, grid_h, grid_w = (int(item) for item in grid_thw)
        merge = self.config.spatial_merge_size
        if grid_h % merge or grid_w % merge:
            raise ValueError(
                "grid H and W must be divisible by spatial_merge_size"
            )
        expected = grid_t * grid_h * grid_w
        if value.shape != (expected, self.config.hidden_size):
            raise ValueError(
                f"visual token shape {tuple(value.shape)} does not match grid"
            )
        value = value.reshape(grid_t, grid_h, grid_w, self.config.hidden_size)
        value = value.reshape(
            grid_t,
            grid_h // merge,
            merge,
            grid_w // merge,
            merge,
            self.config.hidden_size,
        )
        value = value.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        return value.reshape(-1, merge * merge * self.config.hidden_size)

    def _merger_impl(self, value: torch.Tensor, grid_thw: Sequence[int]) -> torch.Tensor:
        cfg = self.config
        value = self._merge_tokens(value, grid_thw)
        value = self._activate(
            self._linear(
                value,
                "proj.0.weight",
                None,
                projector=True,
            )
        )
        value = self._linear(
            value,
            "proj.2.weight",
            None,
            projector=True,
        )
        value = self._norm(
            value,
            cfg.merger_norm_weight,
            cfg.merger_norm_bias,
            projector=True,
        )
        if value.shape[-1] != cfg.output_size:
            raise ValueError(
                f"merger output width {value.shape[-1]} != {cfg.output_size}"
            )
        return value

    def _position_indices(self, grid_thw: Sequence[int], count: int) -> torch.Tensor:
        grid_t, grid_h, grid_w = (int(item) for item in grid_thw)
        positions = torch.arange(count, device=self.device, dtype=torch.long)
        # Flattened table indexing is an injectable convention, not a claimed K3 layout.
        return positions

    @torch.inference_mode()
    def encode(
        self,
        pixel_values: torch.Tensor,
        grid_thws: Sequence[Sequence[int]] | torch.Tensor,
    ) -> list[torch.Tensor]:
        """Encode images/videos into per-item ``[..., output_size]`` tensors."""
        grids = grid_thws.tolist() if isinstance(grid_thws, torch.Tensor) else grid_thws
        patches = self.patchify(pixel_values, grids)
        encoded: list[torch.Tensor] = []
        for item, grid in zip(patches, grids):
            hidden = self._patch_embed(item, grid)
            for layer in range(self.config.num_hidden_layers):
                hidden = self._block(hidden, layer, grid)
            hidden = self._norm(
                hidden,
                self.config.final_norm_weight,
                self.config.final_norm_bias,
            )
            encoded.append(self.patch_merger(hidden, grid))
        return encoded

    @property
    def resident_weight_bytes(self) -> int:
        """Bytes currently cached on this runtime's single configured device."""
        total = 0
        seen: set[int] = set()
        for value in self._weights.values():
            marker = id(value)
            if marker in seen:
                continue
            seen.add(marker)
            if isinstance(value, BlockFP8Weight):
                total += value.nbytes
            else:
                total += value.numel() * value.element_size()
        return total


MoonViT3D = KimiVisionRuntime

__all__ = ["KimiVisionConfig", "KimiVisionRuntime", "MoonViT3D", "PatchMergerMLPV2"]

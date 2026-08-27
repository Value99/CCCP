"""Qwen3.5 Dense MTP drafter backed by manifest-declared fixed tensors.

The implementation follows Transformers' public ``MtpLayer`` contract but
loads only the MTP attachment from CCCP's fixed tensor archive.  Embeddings,
the language-model head and rotary embeddings stay shared with the main
model.  No routed-expert or model-directory-name condition exists here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from .ops.codebook import (
    DenseBF16Linear,
    DenseBF16LinearGroup,
    DenseBF16SwiGLU,
)


_SOURCE_TO_LAYER = {
    "mtp.pre_fc_norm_embedding.weight": "enorm.weight",
    "mtp.pre_fc_norm_hidden.weight": "hnorm.weight",
    "mtp.fc.weight": "eh_proj.weight",
    "mtp.norm.weight": "post_norm.weight",
}


def _mtp_source_name(layer_name: str) -> str:
    if layer_name in _SOURCE_TO_LAYER.values():
        for source, target in _SOURCE_TO_LAYER.items():
            if target == layer_name:
                return source
    prefix = "mtp_block."
    if layer_name.startswith(prefix):
        return "mtp.layers.0." + layer_name[len(prefix):]
    raise KeyError(f"unsupported Qwen3.5 MTP tensor {layer_name!r}")


def _replace(parent: nn.Module, name: str, module: nn.Module) -> None:
    setattr(parent, name, module)


def _install_projection_groups(layer: nn.Module) -> tuple[int, int]:
    """Fuse projections that consume the exact same activation tensor."""
    groups = 0
    projections = 0
    candidates = (
        (layer.mtp_block.self_attn, ("q_proj", "k_proj", "v_proj")),
        (layer.mtp_block.mlp, ("gate_proj", "up_proj")),
    )
    for owner, names in candidates:
        linears = tuple(getattr(owner, name) for name in names)
        group = DenseBF16LinearGroup(linears)
        for index, name in enumerate(names):
            _replace(owner, name, group.view(index))
        setattr(owner, "_cccp_mtp_projection_group", group)
        groups += 1
        projections += len(names)
    return groups, projections


def _install_cpu_resident_linears(layer: nn.Module) -> int:
    """Use the generic resident BF16 GEMV executor for one-token drafting."""
    installed = 0
    for owner, name in (
        (layer, "eh_proj"),
        (layer.mtp_block.self_attn, "o_proj"),
        (layer.mtp_block.mlp, "down_proj"),
    ):
        _replace(owner, name, DenseBF16Linear(getattr(owner, name)))
        installed += 1
    return installed


def _install_cpu_swiglu(layer: nn.Module) -> bool:
    """Fuse the exact MTP MLP without changing its BF16 weights."""
    mlp = layer.mtp_block.mlp
    group = getattr(mlp, "_cccp_mtp_projection_group", None)
    members = tuple(getattr(group, "cpu_members", ()))
    down = getattr(mlp, "down_proj", None)
    down_linear = down.fallback if isinstance(down, DenseBF16Linear) else down
    if (
        not isinstance(group, DenseBF16LinearGroup)
        or len(members) != 2
        or not all(isinstance(item, nn.Linear) for item in members)
        or not isinstance(down_linear, nn.Linear)
    ):
        return False
    layer.mtp_block.mlp = DenseBF16SwiGLU(
        mlp,
        members[0],
        members[1],
        down_linear,
    )
    return True


@dataclass(frozen=True)
class Qwen35MTPState:
    cache_length: int


class Qwen35MTP(nn.Module):
    """One Qwen MTP layer reused autoregressively for up to five drafts."""

    def __init__(self, model, config) -> None:
        super().__init__()
        from transformers.cache_utils import MtpCache
        from transformers.modeling_layers import MtpLayer
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_impl

        self.owner = model
        self.device = model.device
        self.config = copy.deepcopy(config)
        self.config.num_hidden_layers = 1
        self.config.layer_types = ["full_attention"]
        self.config.num_mtp_layers = 1
        self.config.mtp_layer_types = ["full_attention"]

        with torch.device("meta"):
            layer = MtpLayer(
                self.config,
                qwen_impl.Qwen3_5DecoderLayer,
                qwen_impl.Qwen3_5RMSNorm,
                0,
                True,
            )
        expected = set(dict(layer.named_parameters()))
        source_names = {_mtp_source_name(name) for name in expected}
        declared = int((model.cfg or {}).get("mtp_layers") or 0)
        if declared != 1:
            raise ValueError(
                "Qwen3.5 Dense MTP currently requires manifest config "
                f"mtp_layers=1, got {declared}"
            )
        state = model.archive.load_dense_tensors(source_names, self.device)
        mapped = {
            target: state[_mtp_source_name(target)]
            for target in expected
        }
        missing, unexpected = layer.load_state_dict(
            mapped,
            strict=True,
            assign=True,
        )
        if missing or unexpected:
            raise ValueError(
                "Qwen3.5 MTP fixed tensor mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        groups, projections = _install_projection_groups(layer)
        resident = (
            _install_cpu_resident_linears(layer)
            if self.device.type == "cpu"
            else 0
        )
        fused_swiglu = (
            _install_cpu_swiglu(layer)
            if self.device.type == "cpu"
            else False
        )
        self.layer = layer.eval()
        self.embed_tokens = model.network.model.embed_tokens
        self.shared_head = model.network.lm_head
        self.rotary_emb = model.network.model.rotary_emb
        self.cache = MtpCache(config=self.config)
        self._mask_factory = qwen_impl.create_causal_mask
        self._last_hidden: torch.Tensor | None = None
        print(
            "[cccp-mtp] architecture=qwen3.5-dense; layers=1; "
            f"projection_groups={groups}/{projections}; "
            f"cpu_resident_linears={resident}; "
            f"cpu_fused_swiglu={int(fused_swiglu)}; max_draft=5",
            flush=True,
        )

    def reset(self) -> None:
        from transformers.cache_utils import MtpCache

        self.cache = MtpCache(config=self.config)
        self._last_hidden = None

    @property
    def cache_length(self) -> int:
        return int(self.cache.get_seq_length(0))

    def snapshot(self) -> Qwen35MTPState:
        return Qwen35MTPState(self.cache_length)

    def crop(self, length: int) -> None:
        length = max(0, int(length))
        current = self.cache_length
        if length > current:
            raise ValueError(
                f"Qwen3.5 MTP cache cannot grow by crop: {current}->{length}"
            )
        if length < current:
            # Transformers 5.17+ keeps absolute positive cropping only for
            # compatibility and requires a negative removal count.  The
            # latter also states the rollback intent unambiguously.
            self.cache.crop(-(current - length))

    def _run(
        self,
        token_ids: torch.Tensor,
        previous_hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embed_tokens(token_ids).to(previous_hidden.device)
        position_embeddings = self.rotary_emb(
            embeddings,
            position_ids=positions,
        )
        attention_mask = self._mask_factory(
            config=self.config,
            inputs_embeds=embeddings,
            attention_mask=None,
            past_key_values=self.cache,
            position_ids=positions,
            layer_idx=0,
        )
        hidden = self.layer(
            embeddings,
            previous_hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=positions,
            past_key_values=self.cache,
        )
        logits = self.shared_head(hidden).float()
        self._last_hidden = hidden[:, -1:, :]
        return hidden, logits

    @torch.no_grad()
    def prefill(self, main_hidden: torch.Tensor, token_ids: list[int]) -> None:
        """Build shifted MTP context from an already-prefilled main prompt."""
        self.reset()
        if len(token_ids) <= 1:
            return
        if main_hidden.ndim != 2 or main_hidden.shape[0] != len(token_ids):
            raise ValueError("Qwen3.5 MTP prompt hidden/token length mismatch")
        shifted_ids = torch.as_tensor(
            token_ids[1:], dtype=torch.long, device=self.device
        ).reshape(1, -1)
        previous = main_hidden[:-1].unsqueeze(0)
        positions = torch.arange(
            1, len(token_ids), dtype=torch.long, device=self.device
        ).reshape(1, -1)
        self._run(shifted_ids, previous, positions)

    @torch.no_grad()
    def step(
        self,
        previous_hidden: torch.Tensor,
        token_id: int,
        position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token = torch.tensor(
            [[int(token_id)]], dtype=torch.long, device=self.device
        )
        positions = torch.tensor(
            [[int(position)]], dtype=torch.long, device=self.device
        )
        hidden, logits = self._run(
            token,
            previous_hidden.reshape(1, 1, -1),
            positions,
        )
        return hidden[:, -1, :], logits[:, -1, :]


__all__ = ["Qwen35MTP", "Qwen35MTPState"]

"""Opt-in GLM-5.3-Flash image preprocessing and vision bridge.

This module is intentionally not imported by the normal text-only startup path.
It mirrors the official Transformers GLM5-Next image processor and vision
model contract, while avoiding torchvision so the existing text image stays
unchanged.  Video is deliberately not implemented in this first self-use
prototype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PreparedImage:
    """One official-compatible image patch sequence."""

    pixel_values: torch.Tensor
    grid_thw: tuple[int, int, int]
    token_count: int


def _align(value: int, factor: int) -> int:
    return math.ceil(value / factor) * factor


def smart_resize(
    height: int,
    width: int,
    *,
    num_frames: int = 2,
    temporal_factor: int = 2,
    factor: int = 28,
    min_image_tokens: int = 16,
    max_image_tokens: int = 8000,
) -> tuple[int, int]:
    """The GLM-5.3 official smart_resize calculation."""

    pixels_per_token = temporal_factor * factor**2
    min_pixels = min_image_tokens * pixels_per_token
    max_pixels = max_image_tokens * pixels_per_token
    aligned_frames = max(temporal_factor, round(num_frames / temporal_factor) * temporal_factor)
    aligned_height = _align(height, factor)
    aligned_width = _align(width, factor)
    aligned_budget = aligned_frames * aligned_height * aligned_width

    if aligned_budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = _align(max(1, math.ceil(height * scale)), factor)
        aligned_width = _align(max(1, math.ceil(width * scale)), factor)
        aligned_budget = aligned_frames * aligned_height * aligned_width

    if aligned_budget > max_pixels:
        minimum_pixels = aligned_frames * factor**2
        if max_pixels < minimum_pixels:
            raise ValueError("GLM image token budget cannot fit one aligned patch")
        low, high = 1, height
        best_height, best_width = factor, factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height = _align(content_height, factor)
            candidate_width = _align(content_width, factor)
            if aligned_frames * candidate_height * candidate_width <= max_pixels:
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        aligned_height, aligned_width = best_height, best_width
    return aligned_height, aligned_width


def prepare_image(image: object) -> PreparedImage:
    """Convert one PIL RGB image to official GLM patchified pixels."""

    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("GLM image input must be a PIL image")
    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("GLM image has invalid dimensions")
    target_h, target_w = smart_resize(height, width)
    pixels_per_token = 2 * 28**2
    scale = min(target_h / height, target_w / width)
    if 2 * height * width >= pixels_per_token * 16:
        scale = min(1.0, scale)
    content_h = max(1, min(target_h, math.floor(height * scale)))
    content_w = max(1, min(target_w, math.floor(width * scale)))
    if (content_h, content_w) != (height, width):
        image = image.resize((content_w, content_h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(image, (0, 0))

    # Official processor: rescale, CLIP normalize, then temporal patchify.
    array = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
    array = array.view(target_h, target_w, 3).permute(2, 0, 1).contiguous()
    tensor = array.float().div(255.0)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    patch = 14
    merge = 2
    grid_h, grid_w = target_h // patch, target_w // patch
    tensor = tensor.unsqueeze(0)
    patches = tensor.reshape(1, 3, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    patches = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, 2, -1, -1)
        .reshape(1, grid_h * grid_w, 3 * 2 * patch * patch)
        .squeeze(0)
        .contiguous()
    )
    token_count = (grid_h * grid_w) // (merge * merge)
    return PreparedImage(
        pixel_values=patches,
        grid_thw=(1, grid_h, grid_w),
        token_count=token_count,
    )


def prepare_image_sources(sources: Iterable[str]) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Load and preprocess ordered image sources without importing vision code."""

    from .media import decode_image, load_media

    prepared = []
    for source in sources:
        payload = load_media(source, allowed_mime_prefixes=("image/",))
        prepared.append(prepare_image(decode_image(payload)))
    if not prepared:
        raise ValueError("at least one image is required")
    pixels = torch.cat([item.pixel_values for item in prepared], dim=0)
    grids = torch.tensor([item.grid_thw for item in prepared], dtype=torch.long)
    counts = tuple(item.token_count for item in prepared)
    return pixels, grids, counts


def expand_image_token_ids(
    ids: list[int], *, image_token_id: int, token_counts: Iterable[int]
) -> list[int]:
    """Expand one template image token to the processor's visual token count."""

    counts = iter(int(value) for value in token_counts)
    expanded: list[int] = []
    seen = 0
    for token in ids:
        if int(token) == int(image_token_id):
            try:
                count = next(counts)
            except StopIteration as exc:
                raise ValueError("prompt has more image tokens than image inputs") from exc
            if count <= 0:
                raise ValueError("image token count must be positive")
            expanded.extend([int(token)] * count)
            seen += 1
        else:
            expanded.append(int(token))
    try:
        next(counts)
    except StopIteration:
        if seen == 0:
            raise ValueError("image input has no matching image placeholder")
        return expanded
    raise ValueError("prompt has fewer image tokens than image inputs")


def load_vision_model(owner: object):
    """Lazily build the official vision tower from CCCP dense tensors."""

    vision = getattr(owner, "_vision_model", None)
    if vision is not None:
        return vision
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    outer = AutoConfig.from_pretrained(owner.root, local_files_only=True)
    with torch.device("meta"):
        vision = Glm5NextVisionModel(outer.vision_config)
    parameters = list(vision.named_parameters())
    if not parameters:
        raise RuntimeError("official GLM visual model has no parameters")
    for name, parameter in parameters:
        source = "model.visual." + name
        raw = owner._load_source((source,))
        raw_shape = tuple(getattr(raw, "shape", ()))
        if raw_shape != tuple(parameter.shape):
            raise ValueError(
                f"GLM visual shape mismatch for {source}: archive={raw_shape}, model={tuple(parameter.shape)}"
            )
        value = owner._place_weight(raw, matrix=False)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"visual parameter stayed compact: {name}")
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(f"placed GLM visual parameter changed shape: {name}")
        if value.dtype != torch.bfloat16:
            raise TypeError(f"GLM visual parameter {name} must be BF16, got {value.dtype}")
        parent, leaf = owner._vision_parent_leaf(vision, name)
        parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
    for name, buffer in list(vision.named_buffers()):
        source = "model.visual." + name
        # Official Transformers creates rotary inv_freq as a deterministic
        # non-persistent buffer; it is intentionally absent from safetensors.
        if name == "rotary_pos_emb.inv_freq" and buffer.device.type == "meta":
            dim = int(outer.vision_config.hidden_size // outer.vision_config.num_heads // 2)
            theta = float(getattr(outer.vision_config, "rope_theta", 10000.0))
            value = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            parent, leaf = owner._vision_parent_leaf(vision, name)
            parent._buffers[leaf] = value.to(owner.device)
            continue
        if owner.store.has(source):
            value = owner._place_weight(owner._load_source((source,)), matrix=False)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"visual buffer stayed compact: {name}")
            parent, leaf = owner._vision_parent_leaf(vision, name)
            parent._buffers[leaf] = value
        elif buffer.device.type == "meta":
            raise KeyError(f"GLM visual buffer missing from dense archive: {name}")
    unresolved = [
        name for name, value in list(vision.named_parameters()) + list(vision.named_buffers())
        if value.device.type == "meta"
    ]
    if unresolved:
        raise RuntimeError(f"unresolved GLM visual tensors: {unresolved[:8]}")
    owner._vision_model = vision.eval()
    return owner._vision_model


__all__ = [
    "PreparedImage",
    "expand_image_token_ids",
    "load_vision_model",
    "prepare_image",
    "prepare_image_sources",
    "smart_resize",
]

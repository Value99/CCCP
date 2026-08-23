"""Configuration-driven Kimi K2.5 media preprocessing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .media import decode_image, decode_video, load_media


DEFAULT_MEDIA_PROC_CFG = {
    "in_patch_limit": 16384,
    "patch_size": 14,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "merge_kernel_size": 2,
    "fixed_output_tokens": None,
    "patch_limit_on_one_side": 512,
    "in_patch_limit_each_frame": 4096,
    "in_patch_limit_video": None,
    "sample_fps": 2.0,
    "max_num_frames_each_video": None,
    "temporal_merge_kernel_size": 4,
    "timestamp_mode": "hh:mm:ss.fff",
}


@dataclass(frozen=True)
class MediaBatch:
    pixel_values: np.ndarray
    grid_thws: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"pixel_values": self.pixel_values, "grid_thws": self.grid_thws}


def load_media_config(model_dir: str | Path) -> dict[str, Any]:
    """Read ``preprocessor_config.json`` and its ``media_proc_cfg`` section."""
    root = Path(model_dir)
    path = root / "preprocessor_config.json"
    if not path.exists():
        path = root / "processor_config.json"
    values: Mapping[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        values = values.get("media_proc_cfg", values)
    config = dict(DEFAULT_MEDIA_PROC_CFG)
    config.update(values)
    return config


def _resize_config(
    width: int,
    height: int,
    config: Mapping[str, Any],
    *,
    frame_count: int = 1,
) -> dict[str, int | float | None]:
    patch_size = int(config["patch_size"])
    merge = int(config["merge_kernel_size"])
    limit = int(config["in_patch_limit"])
    if frame_count > 1:
        limit = int(config.get("in_patch_limit_each_frame") or limit)
        total = config.get("in_patch_limit_video")
        if total is not None:
            limit = min(limit, round(int(total) / frame_count))
    side_limit = int(config["patch_limit_on_one_side"])
    scale = min(
        1.0,
        math.sqrt(limit / (max(1, width // patch_size) * max(1, height // patch_size))),
        side_limit * patch_size / width,
        side_limit * patch_size / height,
    )
    new_width = min(max(1, int(width * scale)), side_limit * patch_size)
    new_height = min(max(1, int(height * scale)), side_limit * patch_size)
    factor = merge * patch_size
    pad_width = (factor - new_width % factor) % factor
    pad_height = (factor - new_height % factor) % factor
    token_height = (new_height + pad_height) // factor
    token_width = (new_width + pad_width) // factor
    if token_height * merge > side_limit or token_width * merge > side_limit:
        raise ValueError("resized media exceeds patch side limit")
    return {
        "new_width": new_width,
        "new_height": new_height,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "num_tokens": config["fixed_output_tokens"] or token_height * token_width,
    }


def _image_array(image: Any, resize: Mapping[str, Any]) -> np.ndarray:
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("media image must be a PIL Image")
    image = image.convert("RGB").resize(
        (int(resize["new_width"]), int(resize["new_height"])),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(image, dtype=np.uint8)
    return np.pad(
        array,
        ((0, int(resize["pad_height"])), (0, int(resize["pad_width"])), (0, 0)),
        mode="constant",
    )


def _patchify(frames: np.ndarray, patch_size: int) -> np.ndarray:
    time, height, width, channels = frames.shape
    if channels != 3 or height % patch_size or width % patch_size:
        raise ValueError("frames must be RGB and patch-aligned")
    patches = frames.reshape(
        time, height // patch_size, patch_size,
        width // patch_size, patch_size, channels,
    ).transpose(0, 1, 3, 5, 2, 4)
    return patches.reshape(-1, channels, patch_size, patch_size)


def _timestamp(seconds: float, mode: str) -> str:
    if mode == "hh:mm:ss.fff":
        hours = int(seconds // 3600)
        minutes = int(seconds % 3600 // 60)
        return f"{hours:02d}:{minutes:02d}:{seconds % 60:06.3f}"
    if mode == "mm:ss.fff":
        return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"
    if mode == "mm:ss":
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
    raise ValueError(f"invalid timestamp mode: {mode}")


class KimiMediaProcessor:
    """Small dependency-light equivalent of the official Kimi media processor."""

    def __init__(self, config: Mapping[str, Any] | None = None, *, model_dir: str | Path | None = None):
        values = load_media_config(model_dir) if model_dir is not None else dict(DEFAULT_MEDIA_PROC_CFG)
        if config:
            values.update(config)
        self.config = values

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "KimiMediaProcessor":
        return cls(model_dir=model_dir)

    def process_images(self, images: list[Any]) -> MediaBatch:
        patches = []
        grids = []
        mean = np.asarray(self.config["image_mean"], dtype=np.float32)
        inv_std = 1.0 / np.asarray(self.config["image_std"], dtype=np.float32)
        for image in images:
            resize = _resize_config(image.width, image.height, self.config)
            array = _image_array(image, resize).astype(np.float32) / 255.0
            array = (array - mean) * inv_std
            patches.append(_patchify(array[None, ...], int(self.config["patch_size"])))
            grids.append([1, array.shape[0] // int(self.config["patch_size"]), array.shape[1] // int(self.config["patch_size"])])
        if not patches:
            return MediaBatch(np.empty((0, 3, int(self.config["patch_size"]), int(self.config["patch_size"])), dtype=np.float32), np.empty((0, 3), dtype=np.int64))
        return MediaBatch(np.concatenate(patches), np.asarray(grids, dtype=np.int64))

    def process_sources(self, sources: list[str], **load_kwargs: Any) -> MediaBatch:
        images = [decode_image(load_media(source, **load_kwargs)) for source in sources]
        return self.process_images(images)

    def split_video(self, source: str, **load_kwargs: Any) -> tuple[list[Any], str]:
        frames = decode_video(load_media(source, **load_kwargs), sample_fps=float(self.config["sample_fps"]), max_frames=self.config["max_num_frames_each_video"])
        prompts = []
        chunks = []
        kernel = int(self.config["temporal_merge_kernel_size"])
        for start in range(0, len(frames), kernel):
            chunk = frames[start:start + kernel]
            chunks.extend(frame.image for frame in chunk)
            prompts.append(self.video_chunk_prompt(_timestamp(chunk[0].timestamp, str(self.config["timestamp_mode"]))))
        return chunks, "".join(prompts)

    def process_video_frames(self, frames: list[Any]) -> MediaBatch:
        if not frames:
            raise ValueError("video has no frames")
        resize = _resize_config(frames[0].width, frames[0].height, self.config, frame_count=len(frames))
        array = np.stack([_image_array(frame, resize) for frame in frames]).astype(np.float32) / 255.0
        mean = np.asarray(self.config["image_mean"], dtype=np.float32)
        inv_std = 1.0 / np.asarray(self.config["image_std"], dtype=np.float32)
        array = (array - mean) * inv_std
        return MediaBatch(_patchify(array, int(self.config["patch_size"])), np.asarray([[len(frames), array.shape[1] // int(self.config["patch_size"]), array.shape[2] // int(self.config["patch_size"])]], dtype=np.int64))

    @staticmethod
    def image_prompt() -> str:
        return "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"

    @staticmethod
    def video_chunk_prompt(timestamp: str) -> str:
        return f"{timestamp}<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"


__all__ = ["DEFAULT_MEDIA_PROC_CFG", "KimiMediaProcessor", "MediaBatch", "load_media_config"]

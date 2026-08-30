"""Model-independent storage runtime for ordinary codebook tensors.

This module deliberately knows nothing about routed experts.  A model archive
declares logical tensors in ``tensor_vq`` and the adapter replaces matching
Linear/Embedding modules with the same compact p8--p16 execution objects.
Architecture adapters may therefore reuse the storage and operators without
copying model-family conditions into the public kernel layer.  This module
contains storage objects only; codebook image selection and execution are
published by :mod:`cccp.ops.codebook`.
"""

from __future__ import annotations

import json
import os
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .store import PackedVQWeight, SafeFile


def _internal_tensor_name(name: str) -> str:
    prefix = "model.language_model."
    return "model." + name[len(prefix):] if name.startswith(prefix) else name


def _packing_bits(value: str) -> int:
    normalized = str(value).strip().lower().replace("_", "-")
    if not normalized.startswith("packed-u"):
        raise ValueError(f"unsupported Dense VQ packing {value!r}")
    bits = int(normalized.removeprefix("packed-u"))
    if not 8 <= bits <= 16:
        raise ValueError(f"Dense VQ index width must be p8--p16, got p{bits}")
    return bits


@dataclass(frozen=True)
class DenseVQTensorSpec:
    name: str
    source_name: str
    filename: str
    indices_key: str
    codebook_key: str
    rows: int
    cols: int
    bits: int
    layout_name: str


class DenseVQArchive:
    """Validated lazy access to ``tensor_vq`` and fixed Dense tensors."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        with (self.root / "cccp.json").open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("format") != "cccp-1":
            raise ValueError("Dense VQ archive requires format=cccp-1")
        raw_specs = self.manifest.get("tensor_vq")
        if not isinstance(raw_specs, dict) or not raw_specs:
            raise ValueError("Dense VQ archive has no tensor_vq entries")
        if self.manifest.get("expert_files"):
            raise ValueError("Dense VQ archive must not declare routed experts")
        self.layouts = dict((self.manifest.get("quant") or {}).get("layouts") or {})
        self.specs: dict[str, DenseVQTensorSpec] = {}
        for source_name, item in raw_specs.items():
            shape = tuple(int(value) for value in item.get("shape") or ())
            if len(shape) != 2 or min(shape) <= 0:
                raise ValueError(f"invalid Dense VQ shape for {source_name!r}")
            layout_name = str(item.get("layout") or "")
            layout = self.layouts.get(layout_name)
            if not isinstance(layout, dict):
                raise ValueError(
                    f"Dense VQ tensor {source_name!r} references unknown layout "
                    f"{layout_name!r}"
                )
            dim = int(layout.get("dim") or 0)
            if dim <= 0 or shape[1] % dim:
                raise ValueError(f"invalid code dimension for {source_name!r}")
            internal = _internal_tensor_name(str(source_name))
            if internal in self.specs:
                raise ValueError(f"duplicate Dense VQ tensor {internal!r}")
            self.specs[internal] = DenseVQTensorSpec(
                name=internal,
                source_name=str(source_name),
                filename=str(item["file"]),
                indices_key=str(item["indices_key"]),
                codebook_key=str(item["codebook_key"]),
                rows=shape[0],
                cols=shape[1],
                bits=_packing_bits(item.get("index_packing") or ""),
                layout_name=layout_name,
            )
        self._files: dict[str, SafeFile] = {}
        dense_name = str(self.manifest.get("dense_file") or "dense.safetensors")
        self.dense = SafeFile(str(self.root / dense_name))
        self._dense_name = dense_name

    def _file(self, filename: str) -> SafeFile:
        handle = self._files.get(filename)
        if handle is None:
            path = self.root / filename
            if not path.is_file():
                raise FileNotFoundError(f"Dense VQ shard is missing: {filename}")
            handle = SafeFile(str(path))
            self._files[filename] = handle
        return handle

    def load_weight(self, name: str, device: torch.device) -> PackedVQWeight:
        spec = self.specs[name]
        handle = self._file(spec.filename)
        for key in (spec.indices_key, spec.codebook_key):
            if key not in handle.meta:
                raise ValueError(f"{spec.filename} is missing {key!r}")
        raw = handle.get_tensor(spec.indices_key).view(torch.uint8).reshape(-1)
        codebook = handle.get_tensor(spec.codebook_key).float()
        layout = self.layouts[spec.layout_name]
        dim = int(layout["dim"])
        size = int(layout.get("size") or layout.get("codebook_size") or 0)
        if tuple(codebook.shape) != (size, dim):
            raise ValueError(
                f"Dense VQ codebook shape mismatch for {spec.source_name}: "
                f"{tuple(codebook.shape)} != {(size, dim)}"
            )
        weight = PackedVQWeight(raw, codebook, spec.rows, spec.cols, spec.bits)
        if device.type == "cuda":
            weight.raw = weight.raw.to(device)
            weight.cb = weight.cb.to(device)
        return weight

    @staticmethod
    def source_dense_name(internal: str) -> str:
        if internal.startswith("model."):
            return "model.language_model." + internal[len("model."):]
        return internal

    def load_dense_state(
        self,
        names: set[str],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        available = set(self.dense.keys())
        for internal in sorted(names):
            source = self.source_dense_name(internal)
            if source not in available:
                raise ValueError(f"dense.safetensors is missing {source!r}")
            state[internal] = self.dense.get_tensor(source).to(device)
        return state

    def dense_tensor_names(self, prefix: str = "") -> tuple[str, ...]:
        """Return fixed-tensor names without exposing the archive handle.

        Architecture adapters use this for optional, manifest-declared
        attachments such as an MTP block.  Keeping the lookup here prevents
        adapters from depending on ``SafeFile`` internals or reopening the
        same multi-GiB dense archive through a second mapping.
        """
        return tuple(
            sorted(
                name
                for name in self.dense.keys()
                if not prefix or name.startswith(prefix)
            )
        )

    def load_dense_tensors(
        self,
        names: set[str] | tuple[str, ...],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Materialize exact fixed tensors under their archive names."""
        available = set(self.dense.keys())
        missing = sorted(set(names) - available)
        if missing:
            raise ValueError(
                "dense.safetensors is missing fixed tensors: "
                f"{missing[:8]}"
            )
        return {
            name: self.dense.get_tensor(name).to(device)
            for name in sorted(set(names))
        }

    @property
    def packed_bytes(self) -> int:
        return sum(
            (spec.rows * (spec.cols // int(self.layouts[spec.layout_name]["dim"]))
             * spec.bits) // 8
            for spec in self.specs.values()
        )

    @property
    def dense_file_bytes(self) -> int:
        """On-disk upper bound for fixed tensors loaded beside VQ weights."""
        return int((self.root / self._dense_name).stat().st_size)

    def release_dense_ram_blob(self) -> tuple[int, tuple[str, ...]]:
        paths = [str(self.root / self._dense_name)]
        paths.extend(str(self.root / name) for name in self._files)
        released = self.dense.release_ram_blob()
        for handle in self._files.values():
            released += handle.release_ram_blob()
        return released, tuple(paths)

    def close(self) -> None:
        self.dense.close()
        for handle in self._files.values():
            handle.close()




def _scaled_mm_available(device: torch.device) -> bool:
    """torch._scaled_mm 需要 SM89+ 的 FP8 张量核(RTX40 系起)。

    RTX 20/30 系(sm75/86)上调用会直接 RuntimeError——一键包在任意
    设备必须可用,fp8 快路径按能力门控,不可用回退 int4 路径。
    """
    if (
        device.type != "cuda"
        or torch.version.hip is not None
        or not hasattr(torch, "_scaled_mm")
    ):
        return False
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except (RuntimeError, TypeError, ValueError):
        return False
    return (int(major), int(minor)) >= (8, 9)

class DenseVQLinear(nn.Module):
    """Ordinary Linear backed by one compact manifest tensor."""

    def __init__(self, weight: PackedVQWeight, *, name: str) -> None:
        super().__init__()
        self.name = str(name)
        self.rows = int(weight.rows)
        self.cols = int(weight.cols)
        self.blocks = int(weight.blocks)
        self.bits = int(weight.bits)
        self.layout = str(weight.layout)
        self.source_bits = int(weight.source_bits)
        self.register_buffer("payload", weight.raw, persistent=False)
        self.register_buffer("codebook", weight.cb, persistent=False)
        self.register_buffer(
            "gpu_scales",
            torch.empty(0, dtype=torch.float16, device=weight.raw.device),
            persistent=False,
        )
        self.register_buffer(
            "cpu_prefill_weight",
            torch.empty(0, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "fp8_decode_input",
            torch.empty(
                0,
                dtype=torch.float8_e4m3fn,
                device=weight.raw.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "fp8_decode_scale",
            torch.empty(0, dtype=torch.float32, device=weight.raw.device),
            persistent=False,
        )
        self._cpu_executor = None
        self.codebook_scale = 1.0

    def compile_cpu_prefill_bf16(self) -> bool:
        """Build a persistent BF16 GEMM image without changing Decode Q4."""
        if self.payload.is_cuda or self.layout != "q4_0":
            return False
        if self.cpu_prefill_weight.numel():
            return True
        from .cpuext import q4_0_dequant_cpu

        dense = q4_0_dequant_cpu(self.payload, self.rows, self.cols)
        if dense is None:
            return False
        self.cpu_prefill_weight = dense.to(torch.bfloat16).contiguous()
        return True

    @classmethod
    def from_archive(
        cls,
        archive: DenseVQArchive,
        name: str,
        device: torch.device,
    ) -> "DenseVQLinear":
        return cls(archive.load_weight(name, device), name=name)

    @property
    def in_features(self) -> int:
        return self.cols

    @property
    def out_features(self) -> int:
        return self.rows

    @property
    def packed_nbytes(self) -> int:
        return int(self.payload.numel() * self.payload.element_size())

    def compile_cpu(self) -> bool:
        if self.payload.device.type != "cpu" or self.layout == "q4_0":
            return self.layout == "q4_0"
        weight = PackedVQWeight(
            self.payload,
            self.codebook,
            self.rows,
            self.cols,
            self.bits,
        )
        if not weight.compile_cpu_q4_0():
            return False
        self.payload = weight.raw
        self.codebook = torch.empty(0, dtype=torch.float32)
        self.source_bits = int(weight.source_bits)
        self.layout = "q4_0"
        return True

    def compile_gpu_compact_q8(self) -> bool:
        """Keep packed VQ indices and compile only the shared codebook to Q8."""
        if self.payload.device.type != "cuda":
            return False
        if self.layout == "vq_q8_codebook":
            return True
        from .ops.codebook import quantize_q8_codebook

        codebook, scale = quantize_q8_codebook(self.codebook)
        self.codebook = codebook
        self.codebook_scale = float(scale)
        self.gpu_scales = torch.empty(
            0, dtype=torch.float16, device=self.payload.device
        )
        self.layout = "vq_q8_codebook"
        return True

    def compile_gpu_bf16(self) -> bool:
        """Expand one VQ matrix once into an exact resident BF16 image.

        This path is intentionally capacity-driven by the architecture loader.
        It is useful on high-memory accelerators because every subsequent
        projection goes through the vendor GEMM/GEMV implementation and no
        longer pays codebook lookup or weight-reconstruction overhead.
        """
        if self.payload.device.type != "cuda" or self.layout == "bf16":
            return self.layout == "bf16"
        from .ops import dense_vq_dequant_packed

        dense = dense_vq_dequant_packed(
            self.payload,
            self.codebook,
            rows=self.rows,
            blocks=self.blocks,
            bits=self.bits,
        )
        if dense is None or tuple(dense.shape) != (self.rows, self.cols):
            return False
        self.payload = dense.contiguous()
        self.codebook = torch.empty(
            0, dtype=torch.float32, device=self.payload.device
        )
        self.gpu_scales = torch.empty(
            0, dtype=torch.float16, device=self.payload.device
        )
        self.layout = "bf16"
        return True

    def compile_gpu_fp8(self) -> bool:
        """Compile packed VQ directly into tensor-scaled E4M3 storage.

        This format is intentionally separate from the portable block-FP8
        GEMV representation.  It is consumed by the vendor Tensor Core
        ``scaled_mm`` path on GPUs which advertise native FP8 support.  The
        tensor scale selects the fast H20/Blackwell cuBLAS path; row-wise
        scaled GEMM is deliberately excluded because it benchmarks slower
        than BF16 at batch one on the supported runtime.  Conversion happens
        from packed indices through a compact FP8 codebook; a full temporary
        BF16 matrix is never allocated.
        """
        if self.payload.device.type != "cuda" or self.layout == "fp8_tensor":
            return self.layout == "fp8_tensor"
        from .ops import dense_vq_dequant_fp8_packed

        compiled = dense_vq_dequant_fp8_packed(
            self.payload,
            self.codebook,
            rows=self.rows,
            blocks=self.blocks,
            bits=self.bits,
        )
        if compiled is None:
            return False
        quantized, weight_scale = compiled
        if (
            tuple(quantized.shape) != (self.rows, self.cols)
            or quantized.dtype != torch.float8_e4m3fn
            or tuple(weight_scale.shape) != (1, 1)
            or weight_scale.dtype != torch.float32
        ):
            return False
        self.payload = quantized
        self.gpu_scales = weight_scale.contiguous()
        self.codebook = torch.empty(
            0, dtype=torch.float32, device=self.payload.device
        )
        self.fp8_decode_input = torch.empty(
            (1, self.cols),
            dtype=torch.float8_e4m3fn,
            device=self.payload.device,
        )
        self.fp8_decode_scale = torch.empty(
            (1, 1), dtype=torch.float32, device=self.payload.device
        )
        self.layout = "fp8_tensor"
        return True

    def _gpu_fp8(self, rows: torch.Tensor) -> torch.Tensor:
        from .fusedext import dense_fp8_quantize_rows_fused

        if rows.shape[0] == 1:
            quantized = dense_fp8_quantize_rows_fused(
                rows.contiguous(),
                self.fp8_decode_input,
                self.fp8_decode_scale,
            )
            if quantized is None:
                raise RuntimeError(
                    f"CUDA Dense VQ FP8 activation kernel unavailable for "
                    f"{self.name}"
                )
            scales = self.fp8_decode_scale
        else:
            # The old Prefill path expanded the complete activation to FP32,
            # then launched separate abs/amax/div/clamp/cast kernels for every
            # projection.  At 4096 rows those temporary tensors cost more than
            # the FP8 GEMM itself.  The public quantizer reduces BF16 directly
            # and converts to E4M3 without materializing an FP32 activation.
            source = rows.to(torch.bfloat16).contiguous()
            quantized = torch.empty_like(
                source, dtype=torch.float8_e4m3fn
            )
            scales = torch.empty(
                (1, 1),
                dtype=torch.float32,
                device=source.device,
            )
            fused = dense_fp8_quantize_rows_fused(
                source, quantized, scales
            )
            if fused is None:
                raise RuntimeError(
                    f"CUDA Dense VQ FP8 batch activation kernel unavailable "
                    f"for {self.name}"
                )
            quantized = fused
        return torch._scaled_mm(
            quantized,
            self.payload.t(),
            scale_a=scales,
            scale_b=self.gpu_scales,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

    def _cpu(self, rows: torch.Tensor) -> torch.Tensor:
        from .cpuext import (
            q4_0_dequant_cpu,
            q4_0_gemm_cpu,
            q4_0_gemv_cpu,
            vq_dequant_packed_cpu,
        )
        if self.layout == "q4_0":
            if rows.shape[0] == 1:
                if self._cpu_executor is None:
                    from .cpuext import make_resident_projection_cpu
                    from .kernels import BlockFP8Weight

                    self._cpu_executor = make_resident_projection_cpu((
                        BlockFP8Weight(
                            self.payload,
                            torch.empty(0, dtype=torch.float32),
                            self.cols,
                            rows=self.rows,
                            layout="q4_0",
                        ),
                    ))
                if self._cpu_executor is None:
                    raise RuntimeError(
                        f"CPU resident Q4 GEMV unavailable for {self.name}"
                    )
                source = (
                    rows.contiguous()
                    if rows.dtype in (torch.float32, torch.bfloat16)
                    else rows.float().contiguous()
                )
                return self._cpu_executor.forward_combined(
                    source, source.dtype == torch.float32
                )
            if rows.shape[0] <= 64:
                result = q4_0_gemm_cpu(
                    rows.float().contiguous(),
                    self.payload,
                    self.rows,
                    self.cols,
                )
                if result is None:
                    raise RuntimeError(f"CPU Q4 GEMM unavailable for {self.name}")
                return result
            if self.cpu_prefill_weight.numel():
                return F.linear(
                    rows.to(torch.bfloat16), self.cpu_prefill_weight
                )
            dense = q4_0_dequant_cpu(self.payload, self.rows, self.cols)
            if dense is not None:
                # Long Prefill is a matrix-matrix problem.  Keeping both
                # operands in FP32 silently routes Xeon through ordinary
                # FP32 GEMM and leaves BF16/AMX unused.  The source model and
                # activations are BF16, so a transient BF16 execution image
                # preserves the model's declared compute dtype while letting
                # oneDNN select its tiled BF16 GEMM.  Decode and short batches
                # continue to use the compact Q4 kernels above.
                dense = dense.to(torch.bfloat16)
        else:
            dense = vq_dequant_packed_cpu(
                self.payload,
                self.codebook,
                self.rows,
                self.blocks,
                self.bits,
                self.layout,
            )
        if dense is None:
            raise RuntimeError(f"CPU Dense VQ dequant unavailable for {self.name}")
        if dense.dtype == torch.bfloat16:
            return F.linear(rows.to(torch.bfloat16), dense)
        return F.linear(rows.float(), dense)

    def _gpu(self, rows: torch.Tensor) -> torch.Tensor:
        if self.layout == "bf16":
            return F.linear(rows.to(torch.bfloat16), self.payload)
        if self.layout == "fp8_tensor":
            return self._gpu_fp8(rows)
        if self.layout == "vq_q8_codebook":
            from .fusedext import dense_vq_gemv_packed_q8_codebook_fused

            return dense_vq_gemv_packed_q8_codebook_fused(
                rows,
                self.payload,
                self.codebook,
                self.codebook_scale,
                self.rows,
                self.blocks,
                self.bits,
            )
        from .ops import dense_vq_dequant_packed, dense_vq_gemv_packed
        if rows.shape[0] == 1:
            result = dense_vq_gemv_packed(
                rows.float().contiguous(),
                self.payload,
                self.codebook,
                rows=self.rows,
                blocks=self.blocks,
                bits=self.bits,
            )
            if result is None:
                raise RuntimeError(f"CUDA packed Dense VQ GEMV unavailable for {self.name}")
            return result
        dense = dense_vq_dequant_packed(
            self.payload,
            self.codebook,
            rows=self.rows,
            blocks=self.blocks,
            bits=self.bits,
        )
        if dense is None:
            raise RuntimeError(f"CUDA packed Dense VQ Prefill unavailable for {self.name}")
        return F.linear(rows.to(torch.bfloat16), dense)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.cols:
            raise ValueError(
                f"Dense VQ Linear {self.name} expected {self.cols} columns, "
                f"got {value.shape[-1]}"
            )
        shape = value.shape
        rows = value.reshape(-1, shape[-1])
        result = self._gpu(rows) if rows.is_cuda else self._cpu(rows)
        result = result.reshape(*shape[:-1], self.rows)
        return result.to(value.dtype)


class _DenseVQLinearGroupView(nn.Module):
    """One row slice of a GPU projection group owned by its parent layer."""

    def __init__(self, group: "DenseVQLinearGroup", index: int) -> None:
        super().__init__()
        # Registering the same large group below every view makes module/state
        # walks visit it repeatedly.  The architecture parent owns the group;
        # views deliberately retain only a weak reference.
        object.__setattr__(self, "_group_ref", weakref.ref(group))
        self.index = int(index)
        self.in_features = int(group.cols)
        self.out_features = int(group.row_counts[self.index])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        group = self._group_ref()
        if group is None:
            raise RuntimeError("Dense VQ projection group owner was released")
        return group.project(self.index, value)


class DenseVQLinearGroup(nn.Module):
    """Coalesce Linears which consume the identical activation tensor.

    This is a generic row-concatenated execution object.  Architecture
    adapters decide which projections are mathematically independent and
    share an input; the storage/operator layer never contains model names.
    GPU payload bytes are moved into one buffer, not duplicated. CPU Q4
    members remain separate zero-copy tensors owned by one native resident
    executor, which quantizes the shared activation once and enters one
    OpenMP team for the complete projection group.
    """

    def __init__(self, linears: tuple[DenseVQLinear, ...]) -> None:
        super().__init__()
        if len(linears) < 2:
            raise ValueError("Dense VQ projection group needs at least two Linears")
        first = linears[0]
        device_type = first.payload.device.type
        if device_type not in {"cpu", "cuda"}:
            raise ValueError("Dense VQ projection grouping requires CPU/CUDA")
        if any(
            item.cols != first.cols
            or item.layout != first.layout
            or item.payload.device != first.payload.device
            for item in linears
        ):
            raise ValueError("Dense VQ projection group layouts must match")
        supported = (
            {"q4_0"}
            if device_type == "cpu"
            else {"bf16", "fp8_tensor", "vq_q8_codebook"}
        )
        if first.layout not in supported:
            raise ValueError(
                f"Dense VQ projection grouping does not support {first.layout}"
            )
        self.cols = int(first.cols)
        self.layout = str(first.layout)
        self.row_counts = tuple(int(item.rows) for item in linears)
        self.row_offsets = tuple(
            sum(self.row_counts[:index])
            for index in range(len(self.row_counts))
        )
        self.rows = sum(self.row_counts)
        self._cpu_executor = None
        self.cpu_members = nn.ModuleList()
        self.gpu_members = nn.ModuleList()
        if device_type == "cpu":
            from .cpuext import make_resident_projection_cpu
            from .kernels import BlockFP8Weight

            self.cpu_members.extend(linears)
            # A Q4 image is tile-major in groups of eight output rows.  When
            # every logical projection ends on a tile boundary, concatenate
            # the images once and replace member buffers with zero-copy
            # slices.  Long Prefill can then execute shared-input QKV or
            # Gate/Up as one GEMM without retaining a duplicate payload.
            can_combine = all(item.rows % 8 == 0 for item in linears)
            if can_combine:
                combined_payload = torch.cat(
                    tuple(item.payload.reshape(-1) for item in linears)
                ).contiguous()
                byte_offset = 0
                for item in linears:
                    byte_count = int(item.payload.numel())
                    item.payload = combined_payload.narrow(
                        0, byte_offset, byte_count
                    )
                    byte_offset += byte_count
            else:
                combined_payload = torch.empty(0, dtype=torch.uint8)
            empty_scale = torch.empty(0, dtype=torch.float32)
            weights = tuple(
                BlockFP8Weight(
                    item.payload,
                    empty_scale,
                    item.cols,
                    rows=item.rows,
                    layout="q4_0",
                )
                for item in linears
            )
            self._cpu_executor = make_resident_projection_cpu(weights)
            if self._cpu_executor is None:
                raise RuntimeError(
                    "native CPU Dense VQ projection grouping is unavailable"
                )
            common_scale = None
        elif self.layout == "vq_q8_codebook":
            # Compact projections can use different codebooks and packing
            # widths. Keep their immutable public executors under one owner;
            # views still coalesce the shared-input call at the architecture
            # boundary without inventing a model-specific weight image.
            self.gpu_members.extend(linears)
            common_scale = None
            combined_payload = torch.empty(
                0, dtype=torch.uint8, device=first.payload.device
            )
        elif self.layout == "fp8_tensor":
            # Native tensor-scaled GEMM accepts one scale for the combined B
            # matrix. Normalize each already-compiled projection to the
            # largest member scale, then concatenate. E4M3 values convert to
            # BF16 exactly enough for this one-time scale change, and the
            # architecture still executes the mathematically identical row
            # concatenation in one vendor GEMM.
            common_scale = torch.stack(tuple(
                item.gpu_scales.reshape(()) for item in linears
            )).amax().reshape(1, 1)
            payloads = tuple(
                (
                    item.payload.to(torch.bfloat16)
                    * (item.gpu_scales / common_scale).to(torch.bfloat16)
                ).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                for item in linears
            )
            combined_payload = torch.cat(payloads, dim=0).contiguous()
        else:
            common_scale = None
            combined_payload = torch.cat(
                tuple(item.payload for item in linears), dim=0
            ).contiguous()
        self.register_buffer(
            "payload", combined_payload, persistent=False
        )
        self.register_buffer(
            "cpu_prefill_weight",
            torch.empty(0, dtype=torch.bfloat16),
            persistent=False,
        )
        if self.layout == "fp8_tensor":
            self.register_buffer(
                "gpu_scales", common_scale.contiguous(), persistent=False
            )
        else:
            self.register_buffer(
                "gpu_scales",
                torch.empty(
                    0,
                    dtype=torch.float16,
                    device=self.payload.device,
                ),
                persistent=False,
            )
        self.register_buffer(
            "fp8_decode_input",
            torch.empty(
                (1, self.cols) if self.layout == "fp8_tensor" else 0,
                dtype=torch.float8_e4m3fn,
                device=self.payload.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "fp8_decode_scale",
            torch.empty(
                (1, 1) if self.layout == "fp8_tensor" else 0,
                dtype=torch.float32,
                device=self.payload.device,
            ),
            persistent=False,
        )
        self._cached_input: torch.Tensor | None = None
        self._cached_parts: tuple[torch.Tensor, ...] = ()
        self._remaining: set[int] = set()

    def compile_cpu_prefill_bf16(self) -> bool:
        """Expand a compatible combined Q4 group once for long GEMM."""
        if self.payload.is_cuda or self.layout != "q4_0":
            return False
        if self.cpu_prefill_weight.numel():
            return True
        if not self.payload.numel():
            return False
        from .cpuext import q4_0_dequant_cpu

        dense = q4_0_dequant_cpu(self.payload, self.rows, self.cols)
        if dense is None:
            return False
        self.cpu_prefill_weight = dense.to(torch.bfloat16).contiguous()
        return True

    def view(self, index: int) -> nn.Module:
        if not 0 <= int(index) < len(self.row_counts):
            raise IndexError("Dense VQ projection group index is out of range")
        return _DenseVQLinearGroupView(self, int(index))

    def _forward_combined(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.cols:
            raise ValueError("Dense VQ projection group input mismatch")
        shape = value.shape
        rows = value.reshape(-1, self.cols)
        if not value.is_cuda:
            if self.layout != "q4_0" or self._cpu_executor is None:
                raise ValueError("Dense VQ CPU projection group mismatch")
            if rows.shape[0] == 1:
                combined = self._cpu_executor.forward_combined(
                    rows.contiguous(), rows.dtype == torch.float32
                )
            elif self.payload.numel():
                from .cpuext import (
                    q4_0_dequant_cpu,
                    q4_0_gemm_cpu,
                )

                if rows.shape[0] <= 64:
                    combined = q4_0_gemm_cpu(
                        rows.float().contiguous(),
                        self.payload,
                        self.rows,
                        self.cols,
                    )
                elif self.cpu_prefill_weight.numel():
                    combined = F.linear(
                        rows.to(torch.bfloat16), self.cpu_prefill_weight
                    )
                else:
                    dense = q4_0_dequant_cpu(
                        self.payload, self.rows, self.cols
                    )
                    combined = (
                        None
                        if dense is None
                        else F.linear(
                            rows.to(torch.bfloat16),
                            dense.to(torch.bfloat16),
                        )
                    )
                if combined is None:
                    raise RuntimeError(
                        "Dense VQ combined CPU Prefill GEMM unavailable"
                    )
            else:
                combined = torch.cat(
                    tuple(member._cpu(rows) for member in self.cpu_members),
                    dim=-1,
                )
        elif self.layout == "bf16":
            combined = F.linear(rows.to(torch.bfloat16), self.payload)
        elif self.layout == "fp8_tensor":
            from .fusedext import dense_fp8_quantize_rows_fused

            if rows.shape[0] == 1:
                quantized = dense_fp8_quantize_rows_fused(
                    rows.contiguous(),
                    self.fp8_decode_input,
                    self.fp8_decode_scale,
                )
                if quantized is None:
                    raise RuntimeError(
                        "grouped Dense VQ FP8 activation kernel unavailable"
                    )
                scales = self.fp8_decode_scale
            else:
                source = rows.to(torch.bfloat16).contiguous()
                quantized = torch.empty_like(
                    source, dtype=torch.float8_e4m3fn
                )
                scales = torch.empty(
                    (1, 1),
                    dtype=torch.float32,
                    device=source.device,
                )
                fused = dense_fp8_quantize_rows_fused(
                    source, quantized, scales
                )
                if fused is None:
                    raise RuntimeError(
                        "grouped Dense VQ FP8 batch activation kernel "
                        "unavailable"
                    )
                quantized = fused
            combined = torch._scaled_mm(
                quantized,
                self.payload.t(),
                scale_a=scales,
                scale_b=self.gpu_scales,
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )
        elif self.layout == "vq_q8_codebook":
            combined = torch.cat(
                tuple(member._gpu(rows) for member in self.gpu_members),
                dim=-1,
            )
        else:
            raise RuntimeError(
                f"unsupported Dense VQ group layout: {self.layout}"
            )
        return combined.reshape(*shape[:-1], self.rows).to(value.dtype)

    def project(self, index: int, value: torch.Tensor) -> torch.Tensor:
        index = int(index)
        if self._cached_input is not value or index not in self._remaining:
            combined = self._forward_combined(value)
            self._cached_input = value
            self._cached_parts = tuple(
                combined.narrow(-1, offset, rows)
                for offset, rows in zip(self.row_offsets, self.row_counts)
            )
            self._remaining = set(range(len(self.row_counts)))
        result = self._cached_parts[index]
        self._remaining.remove(index)
        if not self._remaining:
            self._cached_input = None
            self._cached_parts = ()
        return result


class DenseVQSwiGLU(nn.Module):
    """Run one CPU Gate/Up/SwiGLU/Down token in one native worker team.

    Architecture adapters select a mathematically compatible MLP. This
    generic execution object only validates three Q4 projections and keeps
    the original module as its multi-token Prefill path.
    """

    def __init__(
        self,
        fallback: nn.Module,
        gate: DenseVQLinear,
        up: DenseVQLinear,
        down: DenseVQLinear,
    ) -> None:
        super().__init__()
        if (
            any(item.layout != "q4_0" for item in (gate, up, down))
            or gate.cols != up.cols
            or gate.rows != up.rows
            or down.cols != gate.rows
            or down.rows != gate.cols
        ):
            raise ValueError("Dense VQ SwiGLU projection shapes mismatch")
        from .cpuext import make_packed_three_layer_cpu
        from .kernels import BlockFP8Weight

        empty = torch.empty(0, dtype=torch.float32)

        def q4_weight(item: DenseVQLinear) -> BlockFP8Weight:
            return BlockFP8Weight(
                item.payload,
                empty,
                item.cols,
                rows=item.rows,
                layout="q4_0",
            )

        self.fallback = fallback
        self.hidden_size = int(gate.cols)
        self.intermediate_size = int(gate.rows)
        self._executor = make_packed_three_layer_cpu(
            ((q4_weight(gate), q4_weight(up), q4_weight(down)),),
            force_mixed=True,
        )
        if self._executor is None:
            raise RuntimeError("native CPU Dense VQ SwiGLU is unavailable")
        self.register_buffer(
            "_expert_id", torch.zeros(1, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "_route_weight", torch.ones(1, dtype=torch.float32), persistent=False
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        rows = value.reshape(-1, self.hidden_size)
        if rows.shape[0] != 1 or rows.is_cuda:
            return self.fallback(value)
        result = self._executor.forward(
            rows,
            self._expert_id,
            self._route_weight,
            0.0,
            "swiglu",
            1.0,
            -1.0,
        )
        if result is None or result.numel() != self.hidden_size:
            raise RuntimeError("native CPU Dense VQ SwiGLU returned no result")
        return result.reshape(*shape[:-1], self.hidden_size).to(value.dtype)


class DenseBF16SwiGLU(nn.Module):
    """Exact one-token BF16 SwiGLU with one persistent native worker team."""

    def __init__(
        self,
        fallback: nn.Module,
        gate: nn.Linear,
        up: nn.Linear,
        down: nn.Linear,
    ) -> None:
        super().__init__()
        if any(
            item.bias is not None
            or item.weight.is_cuda
            or item.weight.dtype != torch.bfloat16
            for item in (gate, up, down)
        ):
            raise ValueError("Dense BF16 SwiGLU requires CPU BF16/no-bias")
        from .cpuext import make_bf16_swiglu_cpu

        self.fallback = fallback
        self.hidden_size = int(gate.in_features)
        self._executor = make_bf16_swiglu_cpu(
            gate.weight,
            up.weight,
            down.weight,
        )
        if self._executor is None:
            raise RuntimeError("native CPU BF16 SwiGLU is unavailable")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        rows = value.reshape(-1, self.hidden_size)
        if rows.shape[0] != 1 or rows.is_cuda:
            return self.fallback(value)
        source = (
            rows.contiguous()
            if rows.dtype in (torch.float32, torch.bfloat16)
            else rows.float().contiguous()
        )
        result = self._executor.forward(source)
        return result.reshape(*shape[:-1], self.hidden_size).to(value.dtype)


class DenseBF16Linear(nn.Module):
    """Bias-free CPU BF16 Linear with a resident one-token GEMV executor."""

    def __init__(self, fallback: nn.Linear) -> None:
        super().__init__()
        if (
            fallback.bias is not None
            or fallback.weight.is_cuda
            or fallback.weight.dtype != torch.bfloat16
        ):
            raise ValueError("Dense BF16 Linear requires CPU BF16/no-bias")
        from .cpuext import make_resident_projection_cpu

        self.fallback = fallback
        self.in_features = int(fallback.in_features)
        self.out_features = int(fallback.out_features)
        self._executor = make_resident_projection_cpu((fallback.weight,))
        if self._executor is None:
            raise RuntimeError("native CPU Dense BF16 Linear is unavailable")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        rows = value.reshape(-1, self.in_features)
        if rows.shape[0] != 1 or rows.is_cuda:
            return self.fallback(value)
        source = (
            rows.contiguous()
            if rows.dtype in (torch.float32, torch.bfloat16)
            else rows.float().contiguous()
        )
        result = self._executor.forward_combined(
            source, source.dtype == torch.float32
        )
        return result.reshape(*shape[:-1], self.out_features).to(value.dtype)


class DenseBF16LinearGroup(nn.Module):
    """Coalesce ordinary bias-free GPU Linears with an identical input.

    Architecture adapters own the semantic decision to group projections;
    this storage object only concatenates compatible resident weights and
    serves lightweight row views.  It is useful beside Dense VQ projections
    when a manifest keeps tiny control projections in the fixed tensor file.
    """

    def __init__(self, linears: tuple[nn.Linear, ...]) -> None:
        super().__init__()
        if len(linears) < 2:
            raise ValueError("Dense BF16 group needs at least two Linears")
        first = linears[0]
        if first.bias is not None or first.weight.dtype != torch.bfloat16:
            raise ValueError("Dense BF16 grouping requires BF16/no-bias")
        self.cols = int(first.in_features)
        if any(
            item.bias is not None
            or item.in_features != self.cols
            or item.weight.device != first.weight.device
            or item.weight.dtype != torch.bfloat16
            for item in linears
        ):
            raise ValueError("Dense BF16 projection group mismatch")
        self.row_counts = tuple(int(item.out_features) for item in linears)
        self.row_offsets = tuple(
            sum(self.row_counts[:index])
            for index in range(len(self.row_counts))
        )
        self.rows = sum(self.row_counts)
        self.cpu_members = nn.ModuleList()
        if not first.weight.is_cuda:
            self.cpu_members.extend(linears)
        combined_payload = (
            torch.cat(
                tuple(item.weight for item in linears), dim=0
            ).contiguous()
            if first.weight.is_cuda
            else torch.empty(0, dtype=torch.bfloat16)
        )
        self.register_buffer(
            "payload",
            combined_payload,
            persistent=False,
        )
        self._cpu_executor = None
        if not first.weight.is_cuda:
            from .cpuext import make_resident_projection_cpu

            self._cpu_executor = make_resident_projection_cpu(
                tuple(item.weight for item in linears)
            )
            if self._cpu_executor is None:
                raise RuntimeError(
                    "native CPU Dense BF16 projection grouping is unavailable"
                )
        self._cached_input: torch.Tensor | None = None
        self._cached_parts: tuple[torch.Tensor, ...] = ()
        self._remaining: set[int] = set()

    def view(self, index: int) -> nn.Module:
        if not 0 <= int(index) < len(self.row_counts):
            raise IndexError("Dense BF16 group index is out of range")
        return _DenseVQLinearGroupView(self, int(index))

    def project(self, index: int, value: torch.Tensor) -> torch.Tensor:
        index = int(index)
        if self._cached_input is not value or index not in self._remaining:
            shape = value.shape
            rows = value.reshape(-1, self.cols)
            if value.is_cuda:
                combined = F.linear(rows.to(torch.bfloat16), self.payload)
                combined = combined.to(value.dtype)
            elif rows.shape[0] > 1:
                combined = torch.cat(
                    tuple(
                        F.linear(rows.to(torch.bfloat16), item.weight)
                        for item in self.cpu_members
                    ),
                    dim=-1,
                ).to(value.dtype)
            else:
                source = (
                    rows.contiguous()
                    if rows.dtype in (torch.float32, torch.bfloat16)
                    else rows.float().contiguous()
                )
                combined = self._cpu_executor.forward_combined(
                    source, source.dtype == torch.float32
                )
            combined = combined.reshape(*shape[:-1], self.rows)
            self._cached_input = value
            self._cached_parts = tuple(
                combined.narrow(-1, offset, rows)
                for offset, rows in zip(self.row_offsets, self.row_counts)
            )
            self._remaining = set(range(len(self.row_counts)))
        result = self._cached_parts[index]
        self._remaining.remove(index)
        if not self._remaining:
            self._cached_input = None
            self._cached_parts = ()
        return result


def _selected_packed_indices(
    payload: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    rows: int,
    blocks: int,
    bits: int,
) -> torch.Tensor:
    ids = row_ids.reshape(-1).to(torch.int64)
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= rows):
        raise IndexError("Dense VQ embedding token id is out of range")
    if bits == 16:
        return payload.view(torch.uint16).reshape(rows, blocks)[ids]
    block_ids = torch.arange(blocks, device=payload.device, dtype=torch.int64)
    logical = ids[:, None] * blocks + block_ids[None, :]
    bit_offsets = logical * bits
    byte_offsets = torch.div(bit_offsets, 8, rounding_mode="floor")
    shifts = bit_offsets & 7
    last = payload.numel() - 1
    b0 = payload[byte_offsets].to(torch.int64)
    b1 = payload[(byte_offsets + 1).clamp_max(last)].to(torch.int64)
    b2 = payload[(byte_offsets + 2).clamp_max(last)].to(torch.int64)
    words = b0 | (b1 << 8) | (b2 << 16)
    return ((words >> shifts) & ((1 << bits) - 1)).to(torch.int64)


class DenseVQEmbedding(nn.Module):
    """Embedding that expands only the requested vocabulary rows."""

    def __init__(self, weight: PackedVQWeight, *, name: str) -> None:
        super().__init__()
        self.name = str(name)
        self.num_embeddings = int(weight.rows)
        self.embedding_dim = int(weight.cols)
        self.blocks = int(weight.blocks)
        self.bits = int(weight.bits)
        self.layout = "packed"
        self.register_buffer("payload", weight.raw, persistent=False)
        self.register_buffer("codebook", weight.cb, persistent=False)

    @classmethod
    def from_archive(
        cls,
        archive: DenseVQArchive,
        name: str,
        device: torch.device,
    ) -> "DenseVQEmbedding":
        return cls(archive.load_weight(name, device), name=name)

    def compile_gpu_bf16(self) -> bool:
        """Expand the vocabulary once for capture-safe GPU lookup."""
        if self.payload.device.type != "cuda" or self.layout == "bf16":
            return self.layout == "bf16"
        from .ops import dense_vq_dequant_packed

        dense = dense_vq_dequant_packed(
            self.payload,
            self.codebook,
            rows=self.num_embeddings,
            blocks=self.blocks,
            bits=self.bits,
        )
        if dense is None or tuple(dense.shape) != (
            self.num_embeddings,
            self.embedding_dim,
        ):
            return False
        self.payload = dense.contiguous()
        self.codebook = torch.empty(
            0, dtype=torch.float32, device=self.payload.device
        )
        self.layout = "bf16"
        return True

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        shape = token_ids.shape
        ids = token_ids.reshape(-1).to(self.payload.device, dtype=torch.int64)
        if self.layout == "bf16":
            values = F.embedding(ids, self.payload)
        elif self.payload.is_cuda:
            from .ops import dense_vq_dequant_packed
            values = dense_vq_dequant_packed(
                self.payload,
                self.codebook,
                rows=self.num_embeddings,
                blocks=self.blocks,
                bits=self.bits,
                row_ids=ids,
            )
            if values is None:
                raise RuntimeError("CUDA Dense VQ embedding operator unavailable")
        else:
            indices = _selected_packed_indices(
                self.payload,
                ids,
                rows=self.num_embeddings,
                blocks=self.blocks,
                bits=self.bits,
            )
            # 16bit 载荷的行索引是 uint16——torch 高级索引只收
            # long/int/byte/bool,CPU 嵌入查表需转 int64(CUDA 路径
            # 走 fused 算子不经此处;CPU 模式首跑实证)。
            values = self.codebook[indices.long()].reshape(
                -1, self.embedding_dim
            )
            values = values.to(torch.bfloat16)
        return values.reshape(*shape, self.embedding_dim)


class DenseVQPoolStats:
    """Common launcher diagnostics for a model with no expert cache."""

    def __init__(self, device: torch.device, packed_bytes: int) -> None:
        self.full_resident = device.type == "cuda"
        self.compact_full_resident = device.type == "cpu"
        self.gpu_storage_bytes = packed_bytes if self.full_resident else 0
        self.gpu_arena_bytes = self.gpu_storage_bytes
        self.host_expert_bytes = 0
        self._host_pinned_bytes = 0
        self.budget = packed_bytes
        self.bytes = self.gpu_storage_bytes
        self.supports_vram_watch = False
        self.hits = 0
        self.miss = 0


__all__ = [
    "DenseVQArchive",
    "DenseBF16Linear",
    "DenseBF16LinearGroup",
    "DenseBF16SwiGLU",
    "DenseVQEmbedding",
    "DenseVQLinear",
    "DenseVQLinearGroup",
    "DenseVQSwiGLU",
    "DenseVQPoolStats",
    "DenseVQTensorSpec",
]

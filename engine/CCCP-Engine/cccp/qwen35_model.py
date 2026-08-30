"""Text-only Qwen3.5 adapter for manifest-declared Dense VQ archives."""

from __future__ import annotations

import gc
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .ops.codebook import (
    DenseBF16Linear,
    DenseBF16LinearGroup,
    DenseVQArchive,
    DenseVQEmbedding,
    DenseVQLinear,
    DenseVQLinearGroup,
    DenseVQPoolStats,
    DenseVQSwiGLU,
    native_fp8_gemm_available,
    plan_dense_vq_gpu_image,
)


_GIB = 1 << 30
_QWEN35_RUNTIME_LOGGED: set[str] = set()
_QWEN35_VERIFY_LOCAL = threading.local()


@dataclass(frozen=True)
class Qwen35DecodeSnapshot:
    """Mutable recurrent state needed to undo one speculative block."""

    position: int
    linear_states: tuple[dict[str, object] | None, ...]


def _resolve_module(root: torch.nn.Module, path: str):
    current = root
    parts = path.split(".")
    for part in parts[:-1]:
        current = getattr(current, part)
    return current, parts[-1]


def _install_qwen35_projection_groups(
    network: torch.nn.Module,
) -> tuple[int, int]:
    """Group only projections proven to consume the identical tensor.

    The generic Dense VQ operator knows only row-concatenation.  Qwen's
    architecture adapter owns the semantic grouping, keeping model-family
    conditions out of the reusable storage/kernel layer.
    """
    groups = 0
    projections = 0
    for layer in network.model.layers:
        candidates: list[tuple[torch.nn.Module, tuple[str, ...], str]] = [
            (layer.mlp, ("gate_proj", "up_proj"), "_cccp_gate_up_group"),
        ]
        if getattr(layer, "linear_attn", None) is not None:
            candidates.append((
                layer.linear_attn,
                # B/A are ordinary fixed Dense tensors in the current
                # manifest; QKV/Z are the two VQ projections sharing the
                # same hidden input.  Group the capability-compatible subset
                # instead of making the four-member set all-or-nothing.
                ("in_proj_qkv", "in_proj_z"),
                "_cccp_input_group",
            ))
        if getattr(layer, "self_attn", None) is not None:
            candidates.append((
                layer.self_attn,
                ("q_proj", "k_proj", "v_proj"),
                "_cccp_qkv_group",
            ))
        for owner, names, group_name in candidates:
            linears = tuple(getattr(owner, name) for name in names)
            if (
                all(isinstance(item, DenseVQLinear) for item in linears)
                and all(
                    item.layout in {
                        "q4_0", "bf16", "fp8_tensor", "vq_q8_codebook"
                    }
                    for item in linears
                )
            ):
                group = DenseVQLinearGroup(linears)
            elif (
                all(isinstance(item, torch.nn.Linear) for item in linears)
                and not linears[0].weight.is_cuda
                and all(
                    item.bias is None
                    and item.weight.dtype == torch.bfloat16
                    for item in linears
                )
            ):
                group = DenseBF16LinearGroup(linears)
            else:
                continue
            setattr(owner, group_name, group)
            for index, name in enumerate(names):
                setattr(owner, name, group.view(index))
            groups += 1
            projections += len(names)
        linear_attention = getattr(layer, "linear_attn", None)
        if linear_attention is not None:
            fixed_names = ("in_proj_b", "in_proj_a")
            fixed_linears = tuple(
                getattr(linear_attention, name) for name in fixed_names
            )
            if (
                all(isinstance(item, torch.nn.Linear) for item in fixed_linears)
                and all(
                    item.weight.dtype == torch.bfloat16
                    for item in fixed_linears
                )
            ):
                fixed_group = DenseBF16LinearGroup(fixed_linears)
                setattr(
                    linear_attention,
                    "_cccp_ba_group",
                    fixed_group,
                )
                for index, name in enumerate(fixed_names):
                    setattr(linear_attention, name, fixed_group.view(index))
                groups += 1
                projections += len(fixed_names)
    return groups, projections


def _install_qwen35_cpu_mlp_fusion(
    network: torch.nn.Module,
) -> tuple[int, int]:
    """Select generic resident SwiGLU for compatible CPU decode layers."""
    installed = 0
    fixed_native = 0
    for layer in network.model.layers:
        mlp = layer.mlp
        group = getattr(mlp, "_cccp_gate_up_group", None)
        members = getattr(group, "cpu_members", ())
        down = getattr(mlp, "down_proj", None)
        if (
            isinstance(group, DenseVQLinearGroup)
            and len(members) == 2
            and all(isinstance(item, DenseVQLinear) for item in members)
            and isinstance(down, DenseVQLinear)
        ):
            layer.mlp = DenseVQSwiGLU(mlp, members[0], members[1], down)
            installed += 1
        elif (
            isinstance(group, DenseBF16LinearGroup)
            and isinstance(down, torch.nn.Linear)
            and not down.weight.is_cuda
            and down.weight.dtype == torch.bfloat16
            and down.bias is None
        ):
            mlp.down_proj = DenseBF16Linear(down)
            fixed_native += 1
    return installed, fixed_native


def _qwen35_delta_rule(fallback):
    """Dispatch one-token Qwen recurrence to the matching CCCP kernel."""

    def run(
        query,
        key,
        value,
        *,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        **kwargs,
    ):
        passthrough = dict(kwargs)
        del kwargs
        if (
            use_qk_l2norm_in_kernel
            and query.ndim == 4
            and query.shape[0] == 1
        ):
            if query.device.type == "cpu":
                state = (
                    initial_state.squeeze(0).contiguous()
                    if initial_state is not None
                    else torch.zeros(
                        (
                            query.shape[2],
                            query.shape[3],
                            value.shape[3],
                        ),
                        dtype=torch.float32,
                        device=query.device,
                    )
                )
                if query.shape[1] == 1:
                    from .cpuext import qwen35_delta_recurrent_cpu

                    fused = qwen35_delta_recurrent_cpu(
                        query[0, 0].contiguous(),
                        key[0, 0].contiguous(),
                        value[0, 0].contiguous(),
                        g[0, 0].contiguous(),
                        beta[0, 0].contiguous(),
                        state,
                    )
                else:
                    from .cpuext import qwen35_delta_recurrent_batch_cpu

                    fused = qwen35_delta_recurrent_batch_cpu(
                        query[0].contiguous(),
                        key[0].contiguous(),
                        value[0].contiguous(),
                        g[0].contiguous(),
                        beta[0].contiguous(),
                        state,
                    )
            elif query.device.type == "cuda":
                state = (
                    initial_state.squeeze(0).contiguous()
                    if initial_state is not None
                    else torch.zeros(
                        (
                            query.shape[2],
                            query.shape[3],
                            value.shape[3],
                        ),
                        dtype=torch.float32,
                        device=query.device,
                    )
                )
                if query.shape[1] > 8:
                    # 长 prefill:串行 recurrent 批内核(为 MTP verify 的
                    # 小批设计)在 T 大时是纯串行瓶颈(实测 32tok=1.1s,
                    # 28 tok/s);放行原 fla chunk 并行路径(用户实测
                    # prefill 应为 2000 tok/s 级)。decode/verify(≤8)
                    # 路径不变。
                    return fallback(
                        query,
                        key,
                        value,
                        g=g,
                        beta=beta,
                        initial_state=initial_state,
                        output_final_state=output_final_state,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        **passthrough,
                    )
                if query.shape[1] == 1:
                    from .fusedext import qwen35_delta_recurrent_fused

                    # ``value`` comes from a split/repeat view in
                    # Transformers. ``empty_like`` preserves that view's
                    # non-standard stride on some Torch builds, while the
                    # compiled recurrence writes a dense [H,V] matrix.
                    output = torch.empty(
                        value[0, 0].shape,
                        dtype=value.dtype,
                        device=value.device,
                    )
                    result = qwen35_delta_recurrent_fused(
                        query[0, 0].contiguous(),
                        key[0, 0].contiguous(),
                        value[0, 0].contiguous(),
                        g[0, 0].contiguous(),
                        beta[0, 0].contiguous(),
                        state,
                        output,
                    )
                else:
                    from .fusedext import (
                        qwen35_delta_recurrent_batch_checkpoint_fused,
                        qwen35_delta_recurrent_batch_fused,
                    )

                    output = torch.empty(
                        value[0].shape,
                        dtype=value.dtype,
                        device=value.device,
                    )
                    owner = getattr(_QWEN35_VERIFY_LOCAL, "owner", None)
                    checkpoints = (
                        owner._new_recurrent_checkpoints(
                            state, int(query.shape[1])
                        )
                        if owner is not None
                        else None
                    )
                    if checkpoints is None:
                        result = qwen35_delta_recurrent_batch_fused(
                            query[0].contiguous(),
                            key[0].contiguous(),
                            value[0].contiguous(),
                            g[0].contiguous(),
                            beta[0].contiguous(),
                            state,
                            output,
                        )
                    else:
                        result = qwen35_delta_recurrent_batch_checkpoint_fused(
                            query[0].contiguous(),
                            key[0].contiguous(),
                            value[0].contiguous(),
                            g[0].contiguous(),
                            beta[0].contiguous(),
                            state,
                            output,
                            checkpoints,
                        )
                fused = None if result is None else (result, state)
            else:
                fused = None
            if fused is not None:
                output, state = fused
                executor = (
                    (
                        "cpu.recurrent-single"
                        if query.shape[1] == 1
                        else "cpu.recurrent-ordered-batch"
                    )
                    if query.device.type == "cpu"
                    else (
                        "cuda.recurrent-single"
                        if query.shape[1] == 1
                        else "cuda.recurrent-ordered-batch"
                    )
                )
                if executor not in _QWEN35_RUNTIME_LOGGED:
                    _QWEN35_RUNTIME_LOGGED.add(executor)
                    print(
                        "[cccp-qwen35] delta_executor="
                        f"{executor}; tokens={int(query.shape[1])}; "
                        "qk_l2norm=fused; state_update=in-place",
                        flush=True,
                    )
                return (
                    output.reshape_as(value),
                    state.unsqueeze(0) if output_final_state else None,
                )
        return fallback(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    return run


def _qwen35_conv_update(fallback):

    def run(value, state, weight, bias=None, activation="silu", **kwargs):
        del kwargs
        if bias is None and activation in {"silu", "swish"}:
            if value.device.type == "cpu":
                from .cpuext import qwen35_conv1d_update_cpu

                fused = qwen35_conv1d_update_cpu(value, state, weight)
            elif value.device.type == "cuda":
                from .fusedext import qwen35_conv1d_update_fused

                fused = qwen35_conv1d_update_fused(
                    value,
                    state,
                    weight,
                    torch.empty_like(value),
                )
            else:
                fused = None
            if fused is not None:
                return fused
        return fallback(value, state, weight, bias, activation)

    return run


class Qwen35DenseVQModel:
    """CCCP model contract backed by Transformers' public Qwen3.5 math.

    Storage and projection execution are CCCP-native.  The architecture layer
    owns only Qwen's attention/recurrent state machine, which keeps this path
    isolated from DSV4/Kimi/GLM while allowing future Dense VQ manifests to
    reuse ``dense_vq`` unchanged.
    """

    def __init__(
        self,
        root: str,
        *,
        cache_gb: float = 0.0,
        max_ctx: int = 4096,
        device: str = "cpu",
        vram_cache_gb: float = 0.0,
        tp_size: int = 1,
        **_unused,
    ) -> None:
        del cache_gb, vram_cache_gb
        if int(tp_size) != 1:
            raise ValueError("Dense VQ Qwen3.5 currently supports tp=1")
        self.root = str(Path(root).resolve())
        self.device = torch.device(device)
        self.archive = DenseVQArchive(self.root)
        self.store = self.archive
        self.cfg = dict(self.archive.manifest.get("config") or {})
        self.max_ctx = min(
            int(max_ctx),
            int(self.cfg.get("max_position_embeddings") or max_ctx),
        )
        self.pos = 0
        self.network = None
        self.cache = None
        self.codebook_stats = DenseVQPoolStats(
            self.device, self.archive.packed_bytes
        )
        self.effective_tp_size = 1
        self.packed_operator_name = "dense_vq.p8-p16.linear+embedding"
        self._gpu_image = "none"
        self._decode_graph = None
        self._decode_graph_token = None
        self._decode_graph_position = None
        self._decode_graph_logits = None
        self._decode_graph_warm_tokens = 0
        self._verify_graph = None
        self._verify_graph_token = None
        self._verify_graph_positions = None
        self._verify_graph_hidden = None
        self._verify_graph_batch = 0
        self._verify_graph_warm_batch = 0
        self._verify_capture_active = False
        self._verify_batch = 0
        self._verify_state_layers: dict[int, int] = {}
        self._verify_recurrent_checkpoints: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._verify_conv_inputs: dict[int, torch.Tensor] = {}
        self._verify_conv_before: dict[int, torch.Tensor] = {}
        self.mtp = None
        self._loaded = False

    def _configuration(self):
        from transformers import AutoConfig
        from .ops.sdpa import register_transformers_native_gqa_attention

        outer = AutoConfig.from_pretrained(
            self.root,
            local_files_only=True,
            trust_remote_code=False,
        )
        config = getattr(outer, "text_config", outer)
        config.max_position_embeddings = self.max_ctx
        config.use_cache = True
        # The common CCCP adapter keeps native grouped-query attention active
        # with StaticCache masks.  Selection is capability based and is shared
        # by CUDA and CPU; no model-name branch lives in the operator itself.
        config._attn_implementation = (
            register_transformers_native_gqa_attention()
        )
        return config

    def preload(self) -> None:
        if self._loaded:
            return
        started = time.perf_counter()
        if self.device.type == "cuda":
            from .ops import configure_dynamic_sdpa_backends
            from .fusedext import available, last_error

            if not available():
                raise RuntimeError(
                    "Dense VQ GPU operators are unavailable: "
                    f"{last_error() or 'unknown build error'}"
                )
            configure_dynamic_sdpa_backends(self.device)
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_impl

        Qwen3_5ForCausalLM = qwen_impl.Qwen3_5ForCausalLM
        Qwen3_5TextRotaryEmbedding = qwen_impl.Qwen3_5TextRotaryEmbedding

        config = self._configuration()
        with torch.device("meta"):
            network = Qwen3_5ForCausalLM(config)

        count = len(self.archive.specs)
        compiled = 0
        gpu_image = "none"
        gpu_plan: dict[str, int] = {}
        if self.device.type == "cuda":
            linear_bf16_bytes = 0
            linear_fp8_bytes = 0
            linear_compact_bytes = 0
            packed_embedding_bytes = 0
            embedding_bf16_bytes = 0
            for name, spec in self.archive.specs.items():
                parent, attribute = _resolve_module(
                    network, name[: -len(".weight")]
                )
                original = getattr(parent, attribute)
                if isinstance(original, torch.nn.Linear):
                    linear_bf16_bytes += spec.rows * spec.cols * 2
                    linear_fp8_bytes += spec.rows * spec.cols + 4
                    layout = self.archive.layouts[spec.layout_name]
                    code_dim = int(layout["dim"])
                    codebook_size = int(
                        layout.get("size") or layout.get("codebook_size") or 0
                    )
                    linear_compact_bytes += (
                        spec.rows * (spec.cols // code_dim) * spec.bits
                    ) // 8 + codebook_size * code_dim
                elif isinstance(original, torch.nn.Embedding):
                    code_dim = int(
                        self.archive.layouts[spec.layout_name]["dim"]
                    )
                    packed_embedding_bytes += (
                        spec.rows * (spec.cols // code_dim) * spec.bits
                    ) // 8
                    embedding_bf16_bytes += spec.rows * spec.cols * 2
                else:
                    raise TypeError(
                        f"Dense VQ tensor {name!r} targets unsupported module "
                        f"{type(original).__name__}"
                    )
            free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
            gpu_image, gpu_plan = plan_dense_vq_gpu_image(
                free_bytes=int(free_bytes),
                linear_bf16_bytes=linear_bf16_bytes,
                packed_embedding_bytes=packed_embedding_bytes,
                fixed_file_bytes=self.archive.dense_file_bytes,
                max_ctx=self.max_ctx,
                initial_ctx=min(self.max_ctx, 4096),
                config=config,
                linear_fp8_bytes=linear_fp8_bytes,
                linear_compact_bytes=linear_compact_bytes,
                embedding_bf16_bytes=embedding_bf16_bytes,
                fp8_supported=native_fp8_gemm_available(self.device),
            )
            if (
                gpu_image == "compact"
                and int(free_bytes) < int(gpu_plan["compact_planned"])
            ):
                raise RuntimeError(
                    "Dense VQ compact image exceeds planned free VRAM; choose "
                    "CPU or a device with more VRAM"
                )
            image_probe = os.environ.get(
                "CCCP_DENSE_VQ_GPU_IMAGE", "auto"
            ).strip().lower()
            if image_probe not in {"", "auto", "fp8", "bf16", "compact"}:
                raise ValueError(
                    "CCCP_DENSE_VQ_GPU_IMAGE must be auto, fp8, bf16 or compact"
                )
            if image_probe in {"fp8", "bf16", "compact"}:
                required_key = f"{image_probe}_planned"
                if int(free_bytes) < int(gpu_plan[required_key]):
                    raise RuntimeError(
                        f"forced Dense VQ {image_probe.upper()} image exceeds "
                        "planned free VRAM; choose CPU or a device with more VRAM"
                    )
                if image_probe == "fp8" and not native_fp8_gemm_available(
                    self.device
                ):
                    raise RuntimeError(
                        "forced Dense VQ FP8 image requires native NVIDIA "
                        "FP8 scaled GEMM"
                    )
                gpu_image = image_probe
                gpu_plan["planned"] = int(gpu_plan[required_key])
            print(
                "[cccp-dense-vq-plan] "
                f"image={gpu_image}; "
                f"selection={image_probe or 'auto'}; "
                f"free={gpu_plan['free'] / _GIB:.2f}GiB; "
                f"bf16_linear={gpu_plan['linear_bf16'] / _GIB:.2f}GiB; "
                f"fp8_linear={gpu_plan['linear_fp8'] / _GIB:.2f}GiB; "
                f"compact_linear={gpu_plan['linear_compact'] / _GIB:.2f}GiB; "
                f"packed_embedding={gpu_plan['packed_embedding'] / _GIB:.2f}GiB; "
                f"bf16_embedding={gpu_plan['embedding_bf16'] / _GIB:.2f}GiB; "
                f"fixed_upper={gpu_plan['fixed'] / _GIB:.2f}GiB; "
                f"kv={gpu_plan['kv'] / _GIB:.2f}GiB; "
                f"runtime_reserve={gpu_plan['runtime'] / _GIB:.2f}GiB; "
                f"required={gpu_plan['planned'] / _GIB:.2f}GiB",
                flush=True,
            )
        for index, name in enumerate(sorted(self.archive.specs), 1):
            if not name.endswith(".weight"):
                raise ValueError(f"Dense VQ module tensor must end in .weight: {name}")
            module_path = name[: -len(".weight")]
            parent, attribute = _resolve_module(network, module_path)
            original = getattr(parent, attribute)
            if isinstance(original, torch.nn.Embedding):
                replacement = DenseVQEmbedding.from_archive(
                    self.archive, name, self.device
                )
                if (
                    self.device.type == "cuda"
                    and gpu_image in {"bf16", "fp8"}
                    and not replacement.compile_gpu_bf16()
                ):
                    raise RuntimeError(
                        "GPU Dense VQ BF16 embedding expansion failed: " + name
                    )
            elif isinstance(original, torch.nn.Linear):
                replacement = DenseVQLinear.from_archive(
                    self.archive, name, self.device
                )
                if self.device.type == "cpu":
                    if not replacement.compile_cpu():
                        raise RuntimeError(
                            f"CPU Dense VQ Q4 compilation failed: {name}"
                        )
                    compiled += 1
                elif self.device.type == "cuda":
                    if gpu_image == "fp8":
                        if not replacement.compile_gpu_fp8():
                            raise RuntimeError(
                                "GPU Dense VQ FP8 compilation failed: " + name
                            )
                    elif gpu_image == "bf16":
                        if not replacement.compile_gpu_bf16():
                            raise RuntimeError(
                                "GPU Dense VQ BF16 expansion failed: " + name
                            )
                    elif (
                        gpu_image == "compact"
                        and not replacement.compile_gpu_compact_q8()
                    ):
                        raise RuntimeError(
                            f"GPU Dense VQ compact Q8 compilation failed: {name}"
                        )
                    compiled += 1
            else:
                raise TypeError(
                    f"Dense VQ tensor {name!r} targets unsupported module "
                    f"{type(original).__name__}"
                )
            setattr(parent, attribute, replacement)
            if index == count or index % 16 == 0:
                print(
                    f"[cccp-winui-progress] "
                    f"phase={'dense-vq-compile' if compiled else 'dense-vq-load'} "
                    f"current={index} total={count}",
                    flush=True,
                )
                if self.device.type == "cpu":
                    # Each source payload is replaced immediately by its Q4
                    # image.  Collect periodically so a 32-GiB machine never
                    # retains all source VQ payloads and all Q4 images at once.
                    gc.collect()

        dense_names = set(dict(network.named_parameters()))
        state = self.archive.load_dense_state(dense_names, self.device)
        missing, unexpected = network.load_state_dict(
            state,
            strict=False,
            assign=True,
        )
        if missing or unexpected:
            raise ValueError(
                "Dense Qwen fixed tensor mismatch: "
                f"missing={list(missing)[:8]}, unexpected={list(unexpected)[:8]}"
            )
        network.model.rotary_emb = Qwen3_5TextRotaryEmbedding(
            config, device=self.device
        )
        projection_groups, grouped_projections = (
            _install_qwen35_projection_groups(network)
        )
        fused_cpu_mlp_layers, fixed_cpu_mlp_layers = (
            _install_qwen35_cpu_mlp_fusion(network)
            if self.device.type == "cpu"
            else (0, 0)
        )
        if self.device.type == "cuda":
            # Group construction briefly owns both the individual payloads
            # and their single concatenated replacement.  All architecture
            # attributes now point at lightweight views, so release the old
            # module buffers before allocating recurrent/KV state.
            gc.collect()
            torch.cuda.empty_cache()
        fused_delta_layers = 0
        # Transformers 5.14 stores the dispatch functions on each module;
        # 5.15 moved them to module globals. Support both public layouts
        # without version checks, and dispatch by the actual tensor device.
        # Transformers uses the recurrent entry for cached single-token decode
        # and the chunk entry for every multi-token prefill.  Both implement
        # the same ordered recurrence, so leaving either global unpatched
        # silently routes half of inference through its Python/Torch fallback.
        for delta_name in (
            "torch_recurrent_gated_delta_rule",
            "torch_chunk_gated_delta_rule",
            # fla 符号:建模层在 fla 可导入时直接调用,且无设备检查——
            # CPU 张量会直入 Triton 崩溃。同样按张量设备分派(CPU→cpuext
            # 融合,CUDA→fusedext;不支持形状回退原实现)。
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
        ):
            global_delta = getattr(qwen_impl, delta_name, None)
            if global_delta is not None and not getattr(
                global_delta, "_cccp_qwen35_fused", False
            ):
                global_delta = _qwen35_delta_rule(global_delta)
                global_delta._cccp_qwen35_fused = True
                setattr(qwen_impl, delta_name, global_delta)
        global_conv = getattr(qwen_impl, "causal_conv1d_update", None)
        if global_conv is not None and not getattr(
            global_conv, "_cccp_qwen35_fused", False
        ):
            global_conv = _qwen35_conv_update(global_conv)
            global_conv._cccp_qwen35_fused = True
            qwen_impl.causal_conv1d_update = global_conv
        for module in network.modules():
            if module.__class__.__name__ == "Qwen3_5GatedDeltaNet":
                if hasattr(module, "recurrent_gated_delta_rule"):
                    module.recurrent_gated_delta_rule = _qwen35_delta_rule(
                        module.recurrent_gated_delta_rule
                    )
                if hasattr(module, "chunk_gated_delta_rule"):
                    # 实例在构造时绑定了 fla 的 chunk 函数对象(建模层
                    # 423 行),模块级补丁够不到——fla 入口无设备检查,
                    # CPU 张量直入 Triton 崩溃;同样按设备分派。
                    module.chunk_gated_delta_rule = _qwen35_delta_rule(
                        module.chunk_gated_delta_rule
                    )
                norm = getattr(module, "norm", None)
                if (
                    self.device.type == "cpu"
                    and norm is not None
                    and type(norm).__name__ == "FusedRMSNormGated"
                ):
                    # fla 的门控范数也是 Triton-only;CPU 换建模层自带
                    # 的 torch 等价实现(权重同源拷贝,silu 门控语义一致)。
                    torch_norm = qwen_impl.Qwen3_5RMSNormGated(
                        int(norm.weight.shape[0]), eps=float(norm.eps)
                    )
                    torch_norm.weight.data.copy_(norm.weight.data)
                    module.norm = torch_norm
                if hasattr(module, "causal_conv1d_update"):
                    module.causal_conv1d_update = _qwen35_conv_update(
                        module.causal_conv1d_update
                    )
                fused_delta_layers += 1
        network.eval()
        self.network = network
        for layer_index, layer in enumerate(network.model.layers):
            linear_attention = getattr(layer, "linear_attn", None)
            projection = getattr(linear_attention, "in_proj_qkv", None)
            if projection is None:
                continue

            def capture_conv_input(
                _module,
                _inputs,
                output,
                *,
                index=layer_index,
            ):
                if self._verify_capture_active:
                    self._verify_conv_inputs[index] = output

            projection.register_forward_hook(capture_conv_input)

        mtp_layers = int(self.cfg.get("mtp_layers") or 0)
        if mtp_layers:
            from .qwen35_mtp import Qwen35MTP

            self.mtp = Qwen35MTP(self, config)

        if self.device.type == "cpu":
            print(
                f"[cccp-dense-vq] CPU Q4 执行映像完成："
                f"{compiled} 个 Linear；Embedding 保持精确 packed 行读取；"
                f"Qwen Gated Delta 融合={fused_delta_layers} 层；"
                f"同输入投影合并={projection_groups} 组 / "
                f"{grouped_projections} 个原始 Linear；"
                f"SwiGLU 三投影融合={fused_cpu_mlp_layers} 层；"
                f"BF16 常驻 MLP={fixed_cpu_mlp_layers} 层",
                flush=True,
            )
        elif self.device.type == "cuda":
            if gpu_image == "fp8":
                image_bytes = (
                    gpu_plan["linear_fp8"] + gpu_plan["embedding_bf16"]
                )
                self.packed_operator_name = (
                    "dense_vq.fp8-tensor.native-scaled-mm"
                )
                print(
                    f"[cccp-dense-vq] GPU 原生 FP8 常驻执行映像完成："
                    f"{compiled} 个 Linear；packed VQ→E4M3 融合转换；"
                    "无整矩阵 BF16 中间态；"
                    "Decode 激活由融合 kernel 动态缩放",
                    flush=True,
                )
            elif gpu_image == "bf16":
                image_bytes = (
                    gpu_plan["linear_bf16"] + gpu_plan["embedding_bf16"]
                )
                self.packed_operator_name = "dense_vq.bf16-resident.vendor-gemm"
                print(
                    f"[cccp-dense-vq] GPU BF16 常驻执行映像完成："
                    f"{compiled} 个 Linear；运行期使用厂商 GEMM/GEMV；"
                    f"VQ 只在加载期展开",
                    flush=True,
                )
            elif gpu_image == "compact":
                image_bytes = (
                    gpu_plan["linear_compact"]
                    + gpu_plan["packed_embedding"]
                )
                self.packed_operator_name = (
                    "dense_vq.compact-q8-codebook.direct-dp4a"
                )
                print(
                    f"[cccp-dense-vq] GPU 紧凑 VQ 执行映像完成："
                    f"{compiled} 个 Linear；packed 索引保持原格式；"
                    "仅 Q8 小码本常驻；Decode 使用直接 DP4A",
                    flush=True,
                )
            print(
                f"[cccp-dense-vq] Qwen Gated Delta/短卷积 GPU 融合="
                f"{fused_delta_layers} 层",
                flush=True,
            )
            print(
                "[cccp-dense-vq] GPU 同输入投影合并="
                f"{projection_groups} 组 / "
                f"{grouped_projections} 个原始 Linear；"
                "每组一次 GEMM/GEMV",
                flush=True,
            )
            self.codebook_stats.gpu_storage_bytes = int(image_bytes)
            self.codebook_stats.gpu_arena_bytes = int(image_bytes)
            self.codebook_stats.bytes = int(image_bytes)
        self._gpu_image = gpu_image
        self._new_cache(config)
        self._loaded = True
        gc.collect()
        print(
            f"[cccp-dense-vq] Qwen3.5 Dense 加载完成："
            f"{count} 个 VQ 张量；设备={self.device}; "
            f"packed={self.archive.packed_bytes / 2**30:.2f}GiB；"
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    def _new_cache(self, config=None) -> None:
        if config is None:
            config = self._configuration()
        if (
            self.device.type == "cuda"
            and torch.version.hip is None
            and self._gpu_image in {"bf16", "fp8"}
        ):
            from transformers.cache_utils import StaticCache

            self.cache = StaticCache(
                config=config,
                max_cache_len=self.max_ctx,
            )
        else:
            from transformers.cache_utils import DynamicCache

            self.cache = DynamicCache(config=config)
        self.pos = 0
        self._decode_graph = None
        self._decode_graph_token = None
        self._decode_graph_position = None
        self._decode_graph_logits = None
        self._decode_graph_warm_tokens = 0
        self._verify_graph = None
        self._verify_graph_token = None
        self._verify_graph_positions = None
        self._verify_graph_hidden = None
        self._verify_graph_batch = 0
        self._verify_graph_warm_batch = 0
        self._verify_capture_active = False
        self._verify_batch = 0
        self._verify_state_layers = {}
        self._verify_recurrent_checkpoints = {}
        self._verify_conv_inputs = {}
        self._verify_conv_before = {}

    def reset_kv(self) -> None:
        self._new_cache()
        if self.mtp is not None:
            self.mtp.reset()

    def prefill_batch_available(self) -> bool:
        return True

    def _network_logits(
        self,
        ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, object]:
        outputs = self.network.model(
            input_ids=ids,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=positions,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state[:, -1, :]
        logits = self.network.lm_head(hidden).float().squeeze(0)
        return logits, outputs.past_key_values

    def _token_graph_enabled(self) -> bool:
        return bool(
            self.device.type == "cuda"
            and torch.version.hip is None
            and self._gpu_image in {"bf16", "fp8"}
            and os.environ.get("CCCP_TOKEN_GRAPH", "1") != "0"
        )

    def _forward_token_graph(
        self,
        ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run or capture one fixed-address native Qwen decode token."""
        if ids.shape != (1, 1) or not self._token_graph_enabled():
            return None
        if self._decode_graph is not None:
            self._decode_graph_token.copy_(ids)
            self._decode_graph_position.fill_(self.pos)
            self._decode_graph.replay()
            self.pos += 1
            return self._decode_graph_logits
        if self._decode_graph_warm_tokens < 1:
            # One eager token warms vendor GEMM/SDPA plans and allocators.
            # The next real token is captured exactly once and remains part
            # of the canonical KV/recurrent state; no synthetic token is run.
            self._decode_graph_warm_tokens += 1
            return None

        token = ids.clone()
        position = torch.full(
            (1,), self.pos, dtype=torch.long, device=self.device
        )
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(self.device)
        try:
            with torch.cuda.graph(graph):
                logits, updated_cache = self._network_logits(token, position)
        except Exception as exc:
            raise RuntimeError(
                "Qwen3.5 native token graph capture failed; refusing the "
                f"slow per-layer fallback: {type(exc).__name__}: {exc}"
            ) from exc
        self.cache = updated_cache
        self._decode_graph = graph
        self._decode_graph_token = token
        self._decode_graph_position = position
        self._decode_graph_logits = logits
        self.pos += 1
        self.packed_operator_name = (
            f"dense_vq.{self._gpu_image}-resident.native-token-graph"
        )
        print(
            "[cccp-qwen35] decode_executor=native-token-graph; "
            f"image={self._gpu_image}; launches=1/token; "
            "static-kv=enabled; eager-fallback=forbidden",
            flush=True,
        )
        return logits

    @torch.no_grad()
    def forward(self, token_ids) -> torch.Tensor:
        if not self._loaded or self.network is None:
            raise RuntimeError("Qwen3.5 Dense model is not preloaded")
        ids = torch.as_tensor(
            token_ids,
            dtype=torch.long,
            device=self.device,
        ).reshape(1, -1)
        if ids.numel() == 0:
            raise ValueError("Qwen3.5 forward requires at least one token")
        if self.pos + ids.shape[1] > self.max_ctx:
            raise RuntimeError(
                f"Qwen3.5 context exceeds max_ctx={self.max_ctx}"
            )
        positions = torch.arange(
            self.pos,
            self.pos + ids.shape[1],
            dtype=torch.long,
            device=self.device,
        )
        graph_logits = self._forward_token_graph(ids)
        if graph_logits is not None:
            return graph_logits
        logits, self.cache = self._network_logits(ids, positions)
        self.pos += int(ids.shape[1])
        return logits

    @torch.no_grad()
    def forward_hidden(self, token_ids) -> torch.Tensor:
        # The public engine only requests this method for sequential fallback;
        # returning one final hidden row preserves that contract.
        if not self._loaded or self.network is None:
            raise RuntimeError("Qwen3.5 Dense model is not preloaded")
        ids = torch.as_tensor(
            token_ids, dtype=torch.long, device=self.device
        ).reshape(1, -1)
        positions = torch.arange(
            self.pos, self.pos + ids.shape[1],
            dtype=torch.long, device=self.device,
        )
        outputs = self.network.model(
            input_ids=ids,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=positions,
            return_dict=True,
        )
        self.cache = outputs.past_key_values
        self.pos += int(ids.shape[1])
        return outputs.last_hidden_state.squeeze(0)

    def _begin_verify_capture(self, batch: int, *, replay: bool) -> None:
        """Expose fixed recurrent/conv state to the checkpoint kernels."""
        if self.cache is None:
            raise RuntimeError("Qwen3.5 verification requires initialized KV")
        self._verify_batch = int(batch)
        self._verify_state_layers = {}
        self._verify_conv_before = {}
        if not replay:
            self._verify_recurrent_checkpoints = {}
            self._verify_conv_inputs = {}
        for layer_index, layer in enumerate(self.cache.layers):
            recurrent = getattr(layer, "recurrent_states", None)
            if isinstance(recurrent, dict):
                state = recurrent.get(0)
                if isinstance(state, torch.Tensor):
                    self._verify_state_layers[int(state.data_ptr())] = layer_index
            conv = getattr(layer, "conv_states", None)
            if isinstance(conv, dict):
                state = conv.get(0)
                if isinstance(state, torch.Tensor):
                    self._verify_conv_before[layer_index] = state.clone()
        self._verify_capture_active = True
        _QWEN35_VERIFY_LOCAL.owner = self

    def _end_verify_capture(self) -> None:
        self._verify_capture_active = False
        if getattr(_QWEN35_VERIFY_LOCAL, "owner", None) is self:
            del _QWEN35_VERIFY_LOCAL.owner

    def _forward_hidden_verify_eager(
        self,
        ids: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        """Run one verifier block eagerly while retaining commit checkpoints."""
        positions = torch.arange(
            self.pos,
            self.pos + batch,
            dtype=torch.long,
            device=self.device,
        )
        self._begin_verify_capture(batch, replay=False)
        try:
            outputs = self.network.model(
                input_ids=ids,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=positions,
                return_dict=True,
            )
        finally:
            self._end_verify_capture()
        self.cache = outputs.past_key_values
        self.pos += batch
        return outputs.last_hidden_state.squeeze(0)

    def _new_recurrent_checkpoints(
        self,
        state: torch.Tensor,
        tokens: int,
    ) -> torch.Tensor:
        layer_index = self._verify_state_layers.get(int(state.data_ptr()))
        if layer_index is None:
            raise RuntimeError(
                "Qwen3.5 verification state is not registered in the cache"
            )
        checkpoints = torch.empty(
            (int(tokens), *state.shape),
            dtype=torch.float32,
            device=state.device,
        )
        self._verify_recurrent_checkpoints[layer_index] = (
            state,
            checkpoints,
        )
        return checkpoints

    def commit_verified_prefix(self, committed: int) -> None:
        """Commit a causal verification prefix without replaying the model."""
        committed = int(committed)
        batch = int(self._verify_batch)
        if not 1 <= committed <= batch:
            raise ValueError(
                f"Qwen3.5 verification commit {committed} is outside 1..{batch}"
            )
        if committed < batch:
            for state, checkpoints in self._verify_recurrent_checkpoints.values():
                state.copy_(checkpoints[committed - 1])
            for layer_index, before in self._verify_conv_before.items():
                projected = self._verify_conv_inputs.get(layer_index)
                if projected is None:
                    raise RuntimeError(
                        "Qwen3.5 verification conv input checkpoint is missing"
                    )
                target = self.cache.layers[layer_index].conv_states[0]
                source = torch.cat(
                    (
                        before,
                        projected[:, :committed].transpose(1, 2),
                    ),
                    dim=-1,
                )[..., -target.shape[-1]:]
                target.copy_(source)
            target_position = self.pos - (batch - committed)
            for layer in self.cache.layers:
                cumulative = getattr(layer, "cumulative_length", None)
                if isinstance(cumulative, torch.Tensor):
                    cumulative.fill_(target_position)
            self.pos = target_position

    @torch.no_grad()
    def forward_hidden_verify(self, token_ids) -> torch.Tensor:
        """Verify one fixed speculative block with a single CUDA launch.

        The first block of a given size executes eagerly to initialize vendor
        plans.  The next real block is captured in-place, so no synthetic
        token ever enters KV or recurrent state.  Only the steady verification
        size is graphed; shortened final/replay blocks stay eager and cannot
        retain additional graph pools.
        """
        ids = torch.as_tensor(
            token_ids, dtype=torch.long, device=self.device
        ).reshape(1, -1)
        batch = int(ids.shape[1])
        checkpoint_enabled = bool(
            batch >= 2
            and self.device.type == "cuda"
            and torch.version.hip is None
        )
        if not checkpoint_enabled:
            return self.forward_hidden(token_ids)
        enabled = bool(
            self._token_graph_enabled()
            and os.environ.get("CCCP_VERIFY_GRAPH", "1") != "0"
        )
        if not enabled:
            return self._forward_hidden_verify_eager(ids, batch)
        if self._verify_graph is not None:
            if batch != self._verify_graph_batch:
                # A shortened final block still needs recurrent/conv
                # checkpoints when the accepted prefix is partial.  Keep it
                # eager instead of creating a second graph pool.
                return self._forward_hidden_verify_eager(ids, batch)
            self._verify_graph_token.copy_(ids)
            self._verify_graph_positions.copy_(torch.arange(
                self.pos,
                self.pos + batch,
                dtype=torch.long,
                device=self.device,
            ))
            self._begin_verify_capture(batch, replay=True)
            try:
                self._verify_graph.replay()
            finally:
                self._end_verify_capture()
            self.pos += batch
            return self._verify_graph_hidden
        if self._verify_graph_warm_batch != batch:
            self._verify_graph_warm_batch = batch
            return self._forward_hidden_verify_eager(ids, batch)

        token = ids.clone()
        positions = torch.arange(
            self.pos,
            self.pos + batch,
            dtype=torch.long,
            device=self.device,
        )
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(self.device)
        self._begin_verify_capture(batch, replay=False)
        try:
            with torch.cuda.graph(graph):
                outputs = self.network.model(
                    input_ids=token,
                    past_key_values=self.cache,
                    use_cache=True,
                    cache_position=positions,
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state.squeeze(0)
        except Exception as exc:
            raise RuntimeError(
                "Qwen3.5 speculative verification graph capture failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            self._end_verify_capture()
        self.cache = outputs.past_key_values
        self._verify_graph = graph
        self._verify_graph_token = token
        self._verify_graph_positions = positions
        self._verify_graph_hidden = hidden
        self._verify_graph_batch = batch
        self.pos += batch
        print(
            "[cccp-mtp] verifier=cuda-graph; architecture=qwen3.5-dense; "
            f"batch={batch}; launches=1/block; static-kv=enabled",
            flush=True,
        )
        return hidden

    @property
    def supports_direct_verify_commit(self) -> bool:
        return bool(
            self.device.type == "cuda" and torch.version.hip is None
        )

    def logits_of(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.network is None:
            raise RuntimeError("Qwen3.5 Dense model is not preloaded")
        return self.network.lm_head(hidden).float()

    def snapshot_decode_state(self) -> Qwen35DecodeSnapshot:
        """Snapshot recurrent/conv state before a speculative verify block.

        Full-attention Dynamic KV can be cropped by position and Static KV is
        addressed by ``cache_position``.  Only Qwen's linear-attention state
        is destructive, so cloning that compact state is both sufficient and
        much cheaper than copying every full-attention KV page.
        """
        if self.cache is None:
            raise RuntimeError("Qwen3.5 cache is not initialized")
        snapshots: list[dict[str, object] | None] = []
        for layer in self.cache.layers:
            if not hasattr(layer, "recurrent_states"):
                snapshots.append(None)
                continue
            snapshots.append({
                "conv_states": {
                    index: (
                        value.clone() if value is not None else None
                    )
                    for index, value in layer.conv_states.items()
                },
                "recurrent_states": {
                    index: (
                        value.clone() if value is not None else None
                    )
                    for index, value in layer.recurrent_states.items()
                },
                "is_conv_states_initialized": dict(
                    layer.is_conv_states_initialized
                ),
                "is_recurrent_states_initialized": dict(
                    layer.is_recurrent_states_initialized
                ),
                "has_previous_state": dict(layer.has_previous_state),
                "conv_kernel_size": dict(layer.conv_kernel_size),
            })
        return Qwen35DecodeSnapshot(
            position=int(self.pos),
            linear_states=tuple(snapshots),
        )

    def restore_decode_state(
        self,
        snapshot: Qwen35DecodeSnapshot,
    ) -> None:
        """Restore one speculative snapshot without replacing static buffers."""
        if self.cache is None:
            raise RuntimeError("Qwen3.5 cache is not initialized")
        if len(snapshot.linear_states) != len(self.cache.layers):
            raise ValueError("Qwen3.5 speculative cache layout changed")
        for layer, state in zip(self.cache.layers, snapshot.linear_states):
            if state is None:
                keys = getattr(layer, "keys", None)
                if (
                    isinstance(keys, torch.Tensor)
                    and keys.numel()
                    and keys.shape[-2] > snapshot.position
                    and layer.__class__.__name__.startswith("Dynamic")
                ):
                    layer.crop(-(int(keys.shape[-2]) - snapshot.position))
                continue
            for field in ("conv_states", "recurrent_states"):
                target = getattr(layer, field)
                for index, saved in state[field].items():
                    current = target.get(index)
                    if saved is None:
                        target[index] = None
                    elif (
                        isinstance(current, torch.Tensor)
                        and current.shape == saved.shape
                    ):
                        current.copy_(saved)
                    else:
                        target[index] = saved.clone()
            layer.is_conv_states_initialized = dict(
                state["is_conv_states_initialized"]
            )
            layer.is_recurrent_states_initialized = dict(
                state["is_recurrent_states_initialized"]
            )
            layer.has_previous_state = dict(state["has_previous_state"])
            layer.conv_kernel_size = dict(state["conv_kernel_size"])
        self.pos = int(snapshot.position)


    @property
    def supports_mtp(self) -> bool:
        return self.mtp is not None


__all__ = ["Qwen35DecodeSnapshot", "Qwen35DenseVQModel"]

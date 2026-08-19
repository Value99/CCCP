"""CCCP 模型前向：GLM-5.2（MLA + MoE）在 CPU 上的完整推理实现。

数值与 CCCP/modelmath.py 逐行一致（该实现已对照逐元素朴素实现验证，max_diff<1e-8），
差异仅在权重来源：dense 走 Int4Weight 分块反量化，专家走 ExpertPool 的 VQ LUT。
注意力为全量因果注意力（短上下文 <2048 与 DSA top-2048 等价），KV cache 以 f16 存储。
MTP 层与 DSA indexer 不存在于 CCCP 产物中，本文件亦不实现。
"""

from __future__ import annotations

import gc
import math
import os
import time
from collections import Counter

import torch
import torch.nn.functional as F

from .kernels import (
    BlockFP8Weight,
    Int4Weight,
    RopeCache,
    VQWeight,
    merge_attention_scores,
    rmsnorm,
)


def _latent_attention_context_batched(
    qa: torch.Tensor,
    q_rot: torch.Tensor,
    ckv: torch.Tensor,
    krot: torch.Tensor,
    *,
    scale: float,
    pos0: int,
    query_batch: int,
) -> torch.Tensor:
    """Exact latent-MLA attention with a bounded query workspace.

    ``qa`` keeps the complete outer Prefill block.  Only the quadratic score
    and softmax workspace is query-tiled; every tile attends to the complete
    key/value range and therefore produces the same result as the full matrix.
    """
    heads, tokens, _ = qa.shape
    sequence = int(ckv.shape[0])
    query_batch = max(1, min(int(query_batch), int(tokens)))
    output = torch.empty(
        (heads, tokens, int(ckv.shape[1])),
        dtype=ckv.dtype,
        device=qa.device,
    )
    key_positions = (
        torch.arange(sequence, device=qa.device)
        if tokens > 1
        else None
    )
    krot_t = krot.t()
    ckv_t = ckv.t()
    qrot = q_rot if q_rot.dtype == ckv.dtype else q_rot.to(ckv.dtype)
    for start in range(0, tokens, query_batch):
        stop = min(tokens, start + query_batch)
        score_nope = qa[:, start:stop] @ ckv_t
        score_rope = qrot[:, start:stop] @ krot_t
        scores = merge_attention_scores(score_nope, score_rope, scale)
        if key_positions is not None:
            query_positions = torch.arange(
                pos0 + start,
                pos0 + stop,
                device=qa.device,
            )
            causal = key_positions[None, :] > query_positions[:, None]
            scores.masked_fill_(causal[None], float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        output[:, start:stop].copy_(attention.to(ckv.dtype) @ ckv)
    return output


from .precision import compute_dtype
from .store import CCCPStore, ExpertPool


def _linear(
    x: torch.Tensor,
    w,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense linear with packed INT4 decode and a general prefill fallback."""
    if isinstance(w, (Int4Weight, BlockFP8Weight)):
        return w.matmul_T_decode_fused(x, output=output)
    if w.dtype != torch.float32:
        result = (x.to(w.dtype) @ w.t()).float()
        if output is not None:
            output.copy_(result)
            return output
        return result
    return x.float() @ w.t()


def _attention_linear(
    x: torch.Tensor,
    w,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention-only decode dispatch for the fused packed INT4 GEMV."""
    if isinstance(w, Int4Weight):
        return w.matmul_T_decode_fused(x, output=output)
    return _linear(x, w)


def _swiglu_linear(
    x: torch.Tensor,
    gate,
    up,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse packed INT4 Gate/Up GEMVs and FP32 SwiGLU for decode."""
    if (
        isinstance(gate, Int4Weight)
        and isinstance(up, Int4Weight)
        and gate.cols == up.cols
        and gate.gs == up.gs
        and gate.q.shape == up.q.shape
        and gate.s.shape == up.s.shape
    ):
        from .fusedext import int4_swiglu_fused

        fused = int4_swiglu_fused(
            x,
            gate.q,
            gate.s,
            up.q,
            up.s,
            gate.cols,
            gate.gs,
            output=output,
        )
        if fused is not None:
            return fused
    return F.silu(_linear(x, gate)) * _linear(x, up)


def _glm_route(
    logits: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    top_k: int,
    routed_scaling: float,
    output_buffers: tuple[
        torch.Tensor,
        torch.Tensor,
    ] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized GLM route weights and Top-K expert IDs."""
    from .ops import route_topk

    fused = route_topk(
        logits,
        bias,
        mask,
        scoring_func="sigmoid",
        top_k=top_k,
        normalize=True,
        scaling=routed_scaling,
        output_buffers=output_buffers,
    )
    if fused is not None:
        return fused
    probability = logits.sigmoid()
    choice = probability + bias
    choice = choice.masked_fill(~mask, float("-inf"))
    indices = choice.topk(top_k, dim=-1).indices
    weights = probability.gather(1, indices)
    weights = weights / (
        weights.sum(-1, keepdim=True) + 1e-20
    ) * routed_scaling
    return weights, indices


def _create_glm_expert_pool(
    store: CCCPStore,
    *,
    device: str | torch.device,
    cache_gb: float,
    vram_cache_gb: float,
    pin_gb: float,
):
    """Select the common expert pool strictly from device/manifest capability."""
    resolved = torch.device(device)
    if (
        resolved.type == "cpu"
        and getattr(
            store.man,
            "packed_expert_vq",
            getattr(store.man, "projection_vq", False),
        )
        and os.environ.get("CCCP_CPU_PACKED", "1") != "0"
    ):
        from .store import PackedCpuExpertPool

        return PackedCpuExpertPool(store, budget_gb=cache_gb)
    return ExpertPool(
        store,
        vram_cache_gb if resolved.type != "cpu" else cache_gb,
        device=str(resolved),
        ram_gb=cache_gb - pin_gb if resolved.type != "cpu" else 0.0,
        pin_gb=pin_gb,
    )


class GLMModel:
    """CCCP 格式 GLM-5.2 的推理模型（CPU / CUDA 双路径）。

    device="cpu"：dense int4 打包驻留内存，专家缓存于内存（默认）。
    device="cuda"：dense 权重（int4 打包态 ≈9.2GB）与 KV cache 常驻显存，
    专家缓存于显存（vram_cache_gb 预算），未命中从磁盘/内存页缓存上传。
    """

    def __init__(
        self,
        root: str,
        cache_gb: float = 16.0,
        max_ctx: int = 2048,
        device: str = "cpu",
        vram_cache_gb: float = 4.0,
        tp_size: int = 1,
    ):
        self.device = torch.device(device)
        self.store = CCCPStore(root)
        self.cfg = self.store.cfg
        from .ops import ModelOperatorConfig

        self.operator_config = ModelOperatorConfig.from_manifest(
            {
                "model_family": self.store.man.model_family or "glm",
                "config": self.cfg,
            }
        )
        gpu = self.device.type != "cpu"
        # 热专家静态钉住已证伪：路由热度和输入域强相关（编码提示实测命中仅 14%，
        # 平均 profile 的 top-32 覆盖 66% 只是跨域平均），LRU 的会话局部性更优
        pin_gb = float(os.environ.get("CCCP_PIN_GB", "0")) if gpu else 0.0
        self._cache_gb = cache_gb
        self._vram_cache_gb = vram_cache_gb
        self._pin_gb = pin_gb
        self.requested_tp_size = int(tp_size)
        self.effective_tp_size = 1
        self.expert_parallel = None
        if self.requested_tp_size > 1:
            if not gpu:
                raise ValueError("tp_size > 1 requires CUDA")
            from .expert_parallel import GpuResidentExpertParallel

            self.expert_parallel = GpuResidentExpertParallel(
                self.store, self.requested_tp_size, self.device
            )
            self.pool = None
        else:
            self.pool = _create_glm_expert_pool(
                self.store,
                device=self.device,
                cache_gb=cache_gb,
                vram_cache_gb=vram_cache_gb,
                pin_gb=pin_gb,
            )
        # 逻辑上下文可很大，但把整张 RoPE 表一次性放入每层 Graph 的公共
        # 工作集会拖慢短/中上下文。先固定 32K 地址窗口，跨界时成倍扩展并
        # 统一重捕获；这不改变 max_ctx 的逻辑准入上限。
        rope_initial = max(
            2048,
            int(os.environ.get("CCCP_ROPE_INITIAL_CTX", "32768")),
        )
        self.rope = RopeCache(
            self.cfg["qk_rope_head_dim"],
            self.cfg["rope_theta"],
            max_len=min(max_ctx + 8, rope_initial + 8),
        )
        if gpu:
            self.rope.cos = self.rope.cos.to(self.device)
            self.rope.sin = self.rope.sin.to(self.device)
        self.max_ctx = max_ctx
        self._wcache: dict[str, object] = {}
        self._lm_head_int4: Int4Weight | None = None
        self._decode_workspaces: dict[
            tuple[int, str], torch.Tensor
        ] = {}
        self._decode_position = (
            torch.empty(
                1,
                dtype=torch.long,
                device=self.device,
            )
            if gpu
            else None
        )
        self._attention_graphs: dict[
            int,
            tuple[
                torch.cuda.CUDAGraph,
                torch.Tensor,
                int,
            ],
        ] = {}
        self._attention_graph_stream = (
            torch.cuda.Stream(device=self.device)
            if gpu
            else None
        )
        self._attention_graph_failed = False
        self._masks: dict[int, torch.Tensor] = {}
        self._prev_ids: dict[int, list[int]] = {}   # 层 → 上一 token 路由专家（预取用）
        # MLA 潜变量 KV（CCCP_LATENT_KV，默认开）：存 c_kv [S,512] + k_rot [S,64] f16
        # （≈0.09MB/token），注意力用吸收形式（q_nope@Wuk、ctx@Wuv^T）免逐头展开；
        # 旧路径（=0）存逐头全量 K/V f16（5.11MB/token，22GB 卡 ctx 受限）。
        self.latent_kv = (os.environ.get("CCCP_LATENT_KV", "1") != "0"
                          and self.device.type != "cpu")
        # 计算 dtype（精度策略层）：GPU 上半精度张量核（Turing→fp16，Ampere+→bf16），
        # MLA 吸收矩阵与潜变量 KV 按此存储，注意力 einsum 免 .float() 上抛
        self.cdt = compute_dtype(self.device) if gpu else torch.float32
        self._wuk: dict[int, torch.Tensor] = {}   # [H, nope, R] 计算 dtype
        self._wuv: dict[int, torch.Tensor] = {}   # [H, v, R] 计算 dtype
        # 每层 KV cache：latent 模式 (c_kv [S,R] f16, k_rot [S,rd] f16)；
        # 旧模式 (k [H, S, qk_head_dim] f16, v [H, S, v_head_dim] f16)
        self.kv: list[tuple[torch.Tensor, torch.Tensor] | None] = \
            [None] * self.cfg["n_layers"]
        # CUDA 潜变量 KV 使用可扩容复用区；self.kv 继续保存已用区间视图，
        # 保持截断接口不变。decode 直接原位写入一行，不再每层、每 token
        # 分配两个 torch.cat 结果。
        self._latent_buffers: list[
            tuple[torch.Tensor, torch.Tensor] | None
        ] = [None] * self.cfg["n_layers"]
        # FlashInfer 直接复用上述分离 KV 缓冲；关闭、缺依赖或失败时原
        # PyTorch MLA 路径不变。
        self._flashinfer_mla_runner = None
        self._flashinfer_mla_unavailable = False
        self._flashinfer_mla_state = None
        self._direct_mla_bmm = (
            os.environ.get("CCCP_GLM_DIRECT_BMM", "1") != "0"
        )
        try:
            from .fusedext import (
                glm_latent_kv_decode_prepare_fused,
                glm_mla_bmm_decode_fused,
                glm_moe_residual_add_fused,
                glm_norm_qkv_int4_fused,
                glm_residual_norm_router_fused,
                int4_glm_qb_split_fused,
            )

            self._latent_kv_decode_prepare = (
                glm_latent_kv_decode_prepare_fused
            )
            self._mla_bmm_decode = glm_mla_bmm_decode_fused
            self._q_b_split_decode = int4_glm_qb_split_fused
            self._norm_qkv_decode = glm_norm_qkv_int4_fused
            self._residual_norm_router_decode = (
                glm_residual_norm_router_fused
            )
            self._moe_residual_add_decode = (
                glm_moe_residual_add_fused
            )
        except ImportError:
            self._latent_kv_decode_prepare = None
            self._mla_bmm_decode = None
            self._q_b_split_decode = None
            self._norm_qkv_decode = None
            self._residual_norm_router_decode = None
            self._moe_residual_add_decode = None
        self.pos = 0

    def preload(self) -> None:
        """GPU 路径：把全部 dense 权重预载到显存（约 13GB，含 lm_head/router 常驻 f32），
        并把钉住热专家预读到 RAM（消除冷启动拖尾）。"""
        if self.device.type == "cpu":
            if getattr(self.pool, "prefill_rows_supported", False):
                from .cpuext import prebuild as prebuild_cpu

                prebuild_cpu()
                resident_all = self.pool.preload_all()
                if not resident_all:
                    self.pool.preload_pinned()
                if self.pool.compact_full_resident:
                    prepared_layers = self.pool.prepare_native_layers()
                    print(
                        "[cccp-glm] CPU 多码本 MoE 执行图完成："
                        f"{prepared_layers} 层；packed 索引保持紧凑",
                        flush=True,
                    )
            return
        t0 = time.time()
        names = self.store.dense_names()
        for i, name in enumerate(names):
            self.w(name)
            if (i + 1) % 200 == 0:
                print(f"[cccp] 预载 dense {i + 1}/{len(names)}", flush=True)
        vram = torch.cuda.memory_allocated(self.device) / 2**30
        print(f"[cccp] dense 预载完成（{time.time() - t0:.1f}s，显存 {vram:.1f}GB）",
              flush=True)
        if self.latent_kv:
            self._build_absorbed()
        if (
            self.expert_parallel is not None
            and self.expert_parallel.preload_if_fits()
        ):
            self.pool = self.expert_parallel
            self.effective_tp_size = self.requested_tp_size
            return
        if self.expert_parallel is not None:
            print(
                "[cccp] 全显存专家容量/P2P不满足，回退单卡 RAM+显存缓存",
                flush=True,
            )
            self.expert_parallel = None
            self.pool = ExpertPool(
                self.store,
                self._vram_cache_gb,
                device=self.device,
                ram_gb=self._cache_gb - self._pin_gb,
                pin_gb=self._pin_gb,
            )
        resident_all = self.pool.preload_all()
        if resident_all:
            self.pool.pin_host_resident()
        else:   # 内存不够（已警告）时回退热钉住 + LRU
            self.pool.preload_pinned()
        self.pool.build_gpu_arenas()
        self.pool.preload_profile_gpu()

    def _build_absorbed_layer(self, layer: int) -> None:
        """单层的 kv_b_proj → Wuk/Wuv 分解（preload 全量或首次按需懒构建）。"""
        c = self.cfg
        H, R = c["n_heads"], c["kv_lora_rank"]
        nope, vd = c["qk_nope_head_dim"], c["v_head_dim"]
        w = self.w(f"model.layers.{layer}.self_attn.kv_b_proj.weight")
        if isinstance(w, (Int4Weight, BlockFP8Weight)):
            w = w.dequant_rows(0, w.shape[0])
        w = w.float().view(H, nope + vd, R)
        self._wuk[layer] = w[:, :nope].to(self.cdt).to(self.device)
        self._wuv[layer] = w[:, nope:].to(self.cdt).to(self.device)

    def _build_absorbed(self) -> None:
        """预分解各层 kv_b_proj 为吸收形式矩阵 Wuk/Wuv（f16 显存常驻，≈2.3GB/78层）。
        kv_b_proj [H*(nope+v), R] → 每头前 nope 行为 Wuk、后 v 行为 Wuv。"""
        c = self.cfg
        t0 = time.time()
        for layer in range(c["n_layers"]):
            self._build_absorbed_layer(layer)
        vram = torch.cuda.memory_allocated(self.device) / 2**30
        print(f"[cccp] MLA 吸收矩阵预分解完成（{time.time() - t0:.1f}s，"
              f"KV 潜变量模式 ≈0.09MB/token，显存 {vram:.1f}GB）", flush=True)

    # ---- 权重访问（带缓存） ----
    def w(self, name: str):
        wt = self._wcache.get(name)
        if wt is None:
            wt = self.store.get_dense(name)
            if self.device.type != "cpu":
                # GPU 路径：全部 dense 上显存（int4 打包态直接上卡，lm_head/router 解到 f32）
                if isinstance(wt, Int4Weight):
                    if name == "lm_head.weight":
                        packed_lm = Int4Weight(
                            wt.q.to(self.device),
                            wt.s.to(self.device),
                            wt.cols,
                            wt.gs,
                            half=False,
                        )
                        self._lm_head_int4 = packed_lm
                        wt = (
                            wt.dequant_rows(
                                0,
                                wt.shape[0],
                            ).to(self.device)
                            if os.environ.get(
                                "CCCP_LM_HEAD_KEEP_F32",
                                "0",
                            ) != "0"
                            else packed_lm
                        )
                    elif self.f32_resident(name):
                        wt = wt.dequant_rows(0, wt.shape[0]).to(self.device)
                    else:
                        # int4 fp16 计算默认关（CCCP_INT4_HALF=1 开启；见 dsv4model 注释）
                        wt = Int4Weight(
                            wt.q.to(self.device),
                            wt.s.to(self.device),
                            wt.cols,
                            wt.gs,
                            half=os.environ.get(
                                "CCCP_INT4_HALF",
                                "0",
                            ) == "1",
                        )
                elif isinstance(wt, BlockFP8Weight):
                    wt = BlockFP8Weight(
                        wt.q.to(self.device),
                        wt.s.to(self.device),
                        wt.cols,
                        wt.block,
                    )
                else:
                    wt = wt.to(self.device)
            # 高频大矩阵常驻 f32：lm_head（951M，每 token 全量乘）与各层 router
            # （gate.weight 118M）。其余 dense 保持 int4 打包 + 分块反量化以省内存。
            elif isinstance(wt, Int4Weight) and self.f32_resident(name):
                wt = wt.dequant_rows(0, wt.shape[0])
            self._wcache[name] = wt
        return wt

    @staticmethod
    def f32_resident(name: str) -> bool:
        return name == "lm_head.weight" or name.endswith(".mlp.gate.weight")

    def reset_kv(self) -> None:
        self.kv = [None] * self.cfg["n_layers"]
        self._flashinfer_mla_state = None
        self.pos = 0

    def _release_completed_scan_layer(self, layer: int) -> dict[str, int]:
        """Drop weights that a one-way route scan will never revisit.

        This is deliberately guarded by the route-scan environment flag.
        Normal chat and API inference retain their cross-token caches.
        """
        if os.environ.get("CCCP_ROUTE_SCAN_LAYER_LOCAL", "0") == "0":
            return {"dense_objects": 0, "expert_shards": 0}
        layer = int(layer)
        prefix = f"model.layers.{layer}."
        dense_objects = 0
        for name in tuple(self._wcache):
            if name.startswith(prefix):
                self._wcache.pop(name, None)
                dense_objects += 1
        self._wuk.pop(layer, None)
        self._wuv.pop(layer, None)
        self._masks.pop(layer, None)
        self._prev_ids.pop(layer, None)
        release_pool = getattr(self.pool, "release_scan_layer", None)
        expert_shards = (
            int(bool(release_pool(layer)))
            if callable(release_pool)
            else 0
        )
        # Large tensors and SafeFile byte buffers are reference-counted, while
        # the collection also clears the few Python cycles built by callbacks.
        gc.collect()
        return {
            "dense_objects": dense_objects,
            "expert_shards": expert_shards,
        }

    def truncate_kv(self, keep: int) -> None:
        """KV 截断到前 keep 位（MTP 投机验证回滚用）。
        旧格式 k/v [H, S, d] 切 dim1；潜变量格式 ckv/krot [S, d] 切 dim0。"""
        if self.latent_kv:
            self.kv = [(k_[:keep], v_[:keep]) if k_ is not None else None
                       for k_, v_ in self.kv]
        else:
            self.kv = [(k_[:, :keep], v_[:, :keep]) if k_ is not None else None
                       for k_, v_ in self.kv]
        self._flashinfer_mla_state = None
        self.pos = keep

    def _latent_buffer(
        self,
        layer: int,
        required: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回至少能容纳 ``required`` 行的可复用潜变量 KV 缓冲区。"""
        current = self._latent_buffers[layer]
        if current is not None and current[0].shape[0] >= required:
            return current
        if current is not None and self._attention_graphs:
            # 每层 Graph 只捕获自己的 KV 地址。某一层扩容不应使其余
            # 77 层全部失效，否则长上下文边界会退化为逐 token 重捕获。
            self._attention_graphs.pop(layer, None)
        initial = self._latent_initial_capacity(layer)
        old_capacity = 0 if current is None else current[0].shape[0]
        capacity = min(
            self.max_ctx,
            max(required, initial, old_capacity * 2),
        )
        # FlashInfer MLA 使用 page=64；额外尾部只是存储容量，不改变
        # max_ctx 的逻辑上限。
        capacity = ((capacity + 63) // 64) * 64
        ckv = torch.empty(
            capacity,
            self.cfg["kv_lora_rank"],
            dtype=self.cdt,
            device=self.device,
        )
        krot = torch.empty(
            capacity,
            self.cfg["qk_rope_head_dim"],
            dtype=self.cdt,
            device=self.device,
        )
        if current is not None and self.pos:
            used = min(self.pos, current[0].shape[0])
            ckv[:used].copy_(current[0][:used])
            krot[:used].copy_(current[1][:used])
        result = (ckv, krot)
        self._latent_buffers[layer] = result
        return result

    def _ensure_latent_capacity(self, required: int) -> None:
        """Safely grow fixed-address latent KV buffers before decode."""

        if self.device.type != "cuda" or not self.latent_kv:
            return
        growing = [
            layer
            for layer, current in enumerate(self._latent_buffers)
            if current is not None and current[0].shape[0] < required
        ]
        if not growing:
            return
        captured_growth = any(
            layer in self._attention_graphs
            for layer in growing
        )
        if captured_growth:
            # Graph replay is asynchronous.  Finish all users of the old
            # addresses before replacing any captured KV buffer.
            torch.cuda.synchronize(self.device)
            self._attention_graphs.clear()
        for layer in growing:
            ckv, krot = self._latent_buffer(layer, required)
            used = min(self.pos, required)
            self.kv[layer] = (ckv[:used], krot[:used])

    def _latent_initial_capacity(self, layer: int) -> int:
        """Choose a stable KV address window for graph-backed decode."""
        configured = os.environ.get("CCCP_LATENT_KV_INITIAL")
        if configured is not None:
            return max(1, min(self.max_ctx, int(configured)))
        graph_resident = (
            self.device.type == "cuda"
            and layer >= 4
            and os.environ.get("CCCP_ATTENTION_GRAPH", "1") != "0"
            and self.expert_parallel is not None
            and getattr(self.pool, "full_resident", False)
        )
        if not graph_resident:
            return min(self.max_ctx, 2048)
        graph_window = max(
            2048,
            int(
                os.environ.get(
                    "CCCP_LATENT_KV_GRAPH_INITIAL",
                    "32768",
                )
            ),
        )
        return min(self.max_ctx, graph_window)

    def _prepare_flashinfer_mla_decode(self, end: int):
        """单 token 规划一次 FlashInfer MLA，随后 78 层复用。"""
        if (
            not self.latent_kv
            or self.cdt != torch.bfloat16
            or os.environ.get("CCCP_FLASHINFER_MLA", "1") == "0"
            or self._flashinfer_mla_unavailable
        ):
            return None
        from .flashinfer_mla import last_error
        from .ops import attention_step

        if self._flashinfer_mla_runner is None:
            try:
                self._flashinfer_mla_runner = attention_step(
                    "paged_latent_create",
                    self.device.type,
                    device=self.device,
                    max_ctx=self.max_ctx,
                    heads=self.cfg["n_heads"],
                    ckv_dim=self.cfg["kv_lora_rank"],
                    kpe_dim=self.cfg["qk_rope_head_dim"],
                    dtype=self.cdt,
                    qk_head_dim=self.cfg["qk_head_dim"],
                )
            except (ImportError, LookupError, RuntimeError):
                self._flashinfer_mla_runner = None
            if self._flashinfer_mla_runner is None:
                self._flashinfer_mla_unavailable = True
                print(
                    "[cccp] FlashInfer MLA 不可用，回退原 PyTorch MLA："
                    f"{last_error()}",
                    flush=True,
                )
                return None
            print(
                "[cccp] FlashInfer MLA decode 已启用（复用分离 latent KV）",
                flush=True,
            )
        runner = self._flashinfer_mla_runner
        if runner is None:
            return None
        try:
            prepared = attention_step(
                "paged_latent_prepare",
                self.device.type,
                runner=runner,
                length=end,
            )
        except (LookupError, RuntimeError):
            prepared = False
        if not prepared:
            self._flashinfer_mla_unavailable = True
            print(
                "[cccp] FlashInfer MLA 运行失败，回退原 PyTorch MLA："
                f"{last_error()}",
                flush=True,
            )
            return None
        return runner

    def _ensure_rope_capacity(self, required: int) -> None:
        if required <= self.rope.cos.shape[0]:
            return
        if self._attention_graphs and self.device.type == "cuda":
            # Captured kernels may still read the old cos/sin addresses.
            torch.cuda.synchronize(self.device)
        if self.rope.ensure_length(min(self.max_ctx + 8, required + 8)):
            # Attention Graph 直接捕获 cos/sin 地址；扩容后只需在边界
            # 重捕获一次，不能继续重放旧地址。
            self._attention_graphs.clear()

    # ---- 基本件 ----
    def _decode_workspace(
        self,
        layer: int,
        name: str,
        rows: int,
    ) -> torch.Tensor | None:
        """Return a stable FP32 decode output buffer when reuse is enabled."""
        return self._decode_tensor_workspace(
            layer,
            name,
            (1, rows),
            torch.float32,
        )

    def _decode_tensor_workspace(
        self,
        layer: int,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return a stable decode tensor with an arbitrary shape and dtype."""
        if (
            self.device.type == "cpu"
            or os.environ.get(
                "CCCP_DECODE_WORKSPACES",
                "1",
            ) == "0"
        ):
            return None
        key = (layer, name)
        output = self._decode_workspaces.get(key)
        if (
            output is None
            or output.shape != shape
            or output.dtype != dtype
        ):
            output = torch.empty(
                shape,
                dtype=dtype,
                device=self.device,
            )
            self._decode_workspaces[key] = output
        return output

    def _shared_expert_eager(
        self,
        x: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        p = f"model.layers.{layer}.mlp.shared_experts"
        gate = self.w(f"{p}.gate_proj.weight")
        up = self.w(f"{p}.up_proj.weight")
        down = self.w(f"{p}.down_proj.weight")
        reuse = (
            x.shape[0] == 1
            and isinstance(gate, Int4Weight)
            and isinstance(up, Int4Weight)
            and isinstance(down, Int4Weight)
        )
        return _linear(
            _swiglu_linear(
                x,
                gate,
                up,
                output=(
                    self._decode_workspace(
                        layer,
                        "shared_intermediate",
                        int(gate.shape[0]),
                    )
                    if reuse
                    else None
                ),
            ),
            down,
            output=(
                self._decode_workspace(
                    layer,
                    "shared_output",
                    int(down.shape[0]),
                )
                if reuse
                else None
            ),
        )

    def embed(
        self,
        ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        emb = self.w("model.embed_tokens.weight")
        if isinstance(emb, Int4Weight):
            if (
                isinstance(ids, torch.Tensor)
                and ids.is_cuda
                and ids.dtype == torch.long
                and ids.numel() == 1
                and emb.q.is_cuda
            ):
                from .fusedext import int4_embedding_device_fused

                fused = int4_embedding_device_fused(
                    emb.q,
                    emb.s,
                    ids.reshape(1),
                    emb.cols,
                    emb.gs,
                    output=self._decode_workspace(
                        -1,
                        "embedding",
                        emb.cols,
                    ),
                )
                if fused is not None:
                    return fused
                ids = [int(ids.item())]
            if len(ids) == 1 and emb.q.is_cuda:
                from .fusedext import int4_embedding_fused

                fused = int4_embedding_fused(
                    emb.q,
                    emb.s,
                    ids[0],
                    emb.cols,
                    emb.gs,
                    output=self._decode_workspace(
                        -1,
                        "embedding",
                        emb.cols,
                    ),
                )
                if fused is not None:
                    return fused
            return torch.stack([emb.row(i) for i in ids])
        return emb[ids].float()

    def _attention_output(
        self,
        x: torch.Tensor,
        layer: int,
        weight,
    ) -> torch.Tensor:
        output = (
            self._decode_workspace(
                layer,
                "o_proj",
                int(weight.shape[0]),
            )
            if x.shape[0] == 1 and isinstance(weight, Int4Weight)
            else None
        )
        return _attention_linear(x, weight, output=output)

    def _attention(
        self,
        x: torch.Tensor,
        layer: int,
        pos0: int,
        input_norm_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.latent_kv:
            return self._attention_latent(
                x,
                layer,
                pos0,
                input_norm_weight,
            )
        if input_norm_weight is not None:
            x = rmsnorm(x, input_norm_weight, self.cfg["rms_eps"])
        return self._attention_full(x, layer, pos0)

    def _attention_latent(
        self,
        x: torch.Tensor,
        layer: int,
        pos0: int,
        input_norm_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        graph_enabled = (
            os.environ.get("CCCP_ATTENTION_GRAPH", "1") != "0"
            and not self._attention_graph_failed
            and self.expert_parallel is not None
            and getattr(self.pool, "full_resident", False)
            and x.shape[0] == 1
            and layer >= 4
            and self._flashinfer_mla_state is not None
            and os.environ.get("CCCP_GLM_QB_SPLIT", "1") != "0"
            and os.environ.get("CCCP_DECODE_WORKSPACES", "1") != "0"
        )
        if not graph_enabled:
            return self._attention_latent_eager(
                x,
                layer,
                pos0,
                input_norm_weight,
            )
        cached = self._attention_graphs.get(layer)
        if cached is not None and cached[2] == x.data_ptr():
            cached[0].replay()
            # CUDA Graph replays tensor writes but not this Python metadata
            # assignment.  After reset(), rebuild the logical KV views so a
            # short sequential prompt can be truncated or extended safely
            # without requiring one batch-prefill pass first.
            end = pos0 + x.shape[0]
            ckv_buffer, krot_buffer = self._latent_buffer(
                layer,
                end,
            )
            self.kv[layer] = (
                ckv_buffer[:end],
                krot_buffer[:end],
            )
            return cached[1]

        eager_output = self._attention_latent_eager(
            x,
            layer,
            pos0,
            input_norm_weight,
        )
        stream = self._attention_graph_stream
        if stream is None:
            return eager_output
        try:
            current = torch.cuda.current_stream(self.device)
            stream.wait_stream(current)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    graph_output = self._attention_latent_eager(
                        x,
                        layer,
                        pos0,
                        input_norm_weight,
                    )
            current.wait_stream(stream)
            self._attention_graphs[layer] = (
                graph,
                graph_output,
                x.data_ptr(),
            )
            return graph_output
        except Exception as error:
            self._attention_graph_failed = True
            self._attention_graphs.clear()
            print(
                "[cccp] Attention CUDA Graph 捕获失败，"
                f"回退逐算子路径：{error}",
                flush=True,
            )
            return eager_output

    def _attention_latent_eager(
        self,
        x: torch.Tensor,
        layer: int,
        pos0: int,
        input_norm_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MLA 吸收形式注意力：KV 只存潜变量（c_kv [S,R] + k_rot [S,rd]，计算 dtype），
        分数 = (q_nope@Wuk)@c_kv^T + q_rot@k_rot^T，输出 = (attn@c_kv)@Wuv^T。
        与 _attention_full 数学等价（展开点不同，半精度舍入位置略移），
        KV 显存 5.11→0.09 MB/token，长上下文不再顶爆 22GB 卡。
        GEMM 走精度策略层半精度（fp16/bf16 张量核），softmax 保持 f32。"""
        c = self.cfg
        H = c["n_heads"]
        p = f"model.layers.{layer}.self_attn"
        T = x.shape[0]
        dt = self.cdt
        if layer not in self._wuk:
            self._build_absorbed_layer(layer)   # 未走 preload 的路径（自检/调试）懒构建
        q_a_weight = self.w(f"{p}.q_a_proj.weight")
        kv_a_weight = self.w(f"{p}.kv_a_proj_with_mqa.weight")
        fused_qkv = (
            self._norm_qkv_decode(
                x,
                input_norm_weight,
                q_a_weight.q,
                q_a_weight.s,
                kv_a_weight.q,
                kv_a_weight.s,
                q_a_weight.cols,
                q_a_weight.gs,
                c["rms_eps"],
                (
                    self._decode_workspace(
                        layer,
                        "q_a",
                        int(q_a_weight.shape[0]),
                    ),
                    self._decode_workspace(
                        layer,
                        "kv_a",
                        int(kv_a_weight.shape[0]),
                    ),
                )
                if os.environ.get(
                    "CCCP_DECODE_WORKSPACES",
                    "1",
                ) != "0"
                else None
            )
            if (
                T == 1
                and input_norm_weight is not None
                and self._norm_qkv_decode is not None
                and isinstance(q_a_weight, Int4Weight)
                and isinstance(kv_a_weight, Int4Weight)
                and q_a_weight.cols == kv_a_weight.cols
                and q_a_weight.gs == kv_a_weight.gs
            )
            else None
        )
        if fused_qkv is None:
            if input_norm_weight is not None:
                x = rmsnorm(x, input_norm_weight, c["rms_eps"])
            q_a = _attention_linear(x, q_a_weight)
            kv = _attention_linear(x, kv_a_weight)
        else:
            q_a, kv = fused_qkv
        q_norm_weight = self.w(f"{p}.q_a_layernorm.weight")
        q_b_weight = self.w(f"{p}.q_b_proj.weight")
        q_resid = rmsnorm(
            q_a,
            q_norm_weight,
            1e-6,
            output=(
                self._decode_workspace(
                    layer,
                    "q_a_norm",
                    int(q_a.shape[1]),
                )
                if (
                    T == 1
                    and os.environ.get(
                        "CCCP_RMSNORM_WORKSPACES",
                        "1",
                    )
                    != "0"
                )
                else None
            ),
        )
        fused_q_parts = (
            self._q_b_split_decode(
                q_resid,
                q_b_weight.q,
                q_b_weight.s,
                q_b_weight.cols,
                q_b_weight.gs,
                H,
                c["qk_nope_head_dim"],
                c["qk_rope_head_dim"],
                self._decode_tensor_workspace(
                    layer,
                    "q_nope_bf16",
                    (H, 1, c["qk_nope_head_dim"]),
                    dt,
                ),
                self._decode_tensor_workspace(
                    layer,
                    "q_rope_f32",
                    (H, 1, c["qk_rope_head_dim"]),
                    torch.float32,
                ),
            )
            if (
                T == 1
                and isinstance(q_b_weight, Int4Weight)
                and self._q_b_split_decode is not None
            )
            else None
        )
        if fused_q_parts is None:
            q = _attention_linear(
                q_resid,
                q_b_weight,
                output=(
                    self._decode_workspace(
                        layer,
                        "q_b",
                        int(q_b_weight.shape[0]),
                    )
                    if T == 1
                    and isinstance(q_b_weight, Int4Weight)
                    else None
                ),
            )
            q = q.view(
                T,
                H,
                c["qk_head_dim"],
            ).transpose(0, 1)
            q_nope, q_rot = q.split(
                [
                    c["qk_nope_head_dim"],
                    c["qk_rope_head_dim"],
                ],
                dim=-1,
            )
        else:
            q_nope, q_rot = fused_q_parts

        c_raw, k_rot = kv.split(
            [c["kv_lora_rank"], c["qk_rope_head_dim"]],
            dim=-1,
        )

        end = pos0 + T
        ckv_buffer, krot_buffer = self._latent_buffer(layer, end)
        prepared_q_rot = (
            self._latent_kv_decode_prepare(
                c_raw,
                self.w(f"{p}.kv_a_layernorm.weight"),
                q_rot,
                k_rot.view(1, T, c["qk_rope_head_dim"]),
                self.rope.cos,
                self.rope.sin,
                ckv_buffer,
                krot_buffer,
                self._decode_position,
                1e-6,
                (
                    self._decode_tensor_workspace(
                        layer,
                        "prepared_q_rot",
                        (
                            H,
                            1,
                            c["qk_rope_head_dim"],
                        ),
                        dt,
                    )
                    if os.environ.get(
                        "CCCP_ATTENTION_TENSOR_WORKSPACES",
                        "0",
                    ) != "0"
                    or os.environ.get(
                        "CCCP_ATTENTION_GRAPH",
                        "1",
                    ) != "0"
                    else None
                ),
            )
            if (
                T == 1
                and dt == torch.bfloat16
                and self._latent_kv_decode_prepare is not None
            )
            else None
        )
        if prepared_q_rot is not None:
            q_rot = prepared_q_rot
        else:
            c_new = rmsnorm(
                c_raw,
                self.w(f"{p}.kv_a_layernorm.weight"),
                1e-6,
            )
            q_rot, k_rot = self.rope.apply(
                q_rot,
                k_rot.view(1, T, c["qk_rope_head_dim"]),
                pos0,
            )
            ckv_buffer[pos0:end].copy_(c_new)
            krot_buffer[pos0:end].copy_(k_rot[0])
        ckv = ckv_buffer[:end]
        krot = krot_buffer[:end]
        self.kv[layer] = (ckv, krot)
        S = ckv.shape[0]

        scale = math.sqrt(c["qk_head_dim"])
        attention_workspace_enabled = (
            T == 1
            and (
                os.environ.get(
                    "CCCP_ATTENTION_TENSOR_WORKSPACES",
                    "0",
                )
                != "0"
                or os.environ.get(
                    "CCCP_ATTENTION_GRAPH",
                    "1",
                )
                != "0"
            )
        )
        # The contraction is a plain batched matmul.  Calling bmm directly
        # avoids einsum's equation parsing plus its permute/reshape dispatcher
        # path on every layer and decode token.
        if self._direct_mla_bmm:
            qa_input = (
                q_nope
                if q_nope.dtype == dt
                else q_nope.to(dt)
            )
            qa = (
                self._mla_bmm_decode(
                    qa_input,
                    self._wuk[layer],
                    False,
                    self._decode_tensor_workspace(
                        layer,
                        "mla_qa",
                        (H, 1, c["kv_lora_rank"]),
                        dt,
                    )
                )
                if (
                    T == 1
                    and self._mla_bmm_decode is not None
                    and (
                        os.environ.get(
                            "CCCP_GLM_CUBLAS_Q",
                            "0",
                        )
                        != "0"
                        or os.environ.get(
                            "CCCP_GLM_CUBLAS_DECODE",
                            "0",
                        )
                        != "0"
                    )
                )
                else None
            )
            if qa is None:
                qa = torch.bmm(
                    qa_input,
                    self._wuk[layer],
                    out=(
                        self._decode_tensor_workspace(
                            layer,
                            "mla_qa",
                            (H, 1, c["kv_lora_rank"]),
                            dt,
                        )
                        if attention_workspace_enabled
                        else None
                    ),
                )
        else:
            qa = torch.einsum(
                "htn,hnr->htr",
                q_nope.to(dt),
                self._wuk[layer],
            )
        flash_state = self._flashinfer_mla_state if T == 1 else None
        if flash_state is not None:
            from .flashinfer_mla import last_error
            from .ops import attention_step

            page = flash_state.page_size
            flash_out = attention_step(
                "paged_latent_decode",
                self.device.type,
                runner=flash_state,
                query_nope=qa.transpose(0, 1),
                query_rope=q_rot.to(dt).transpose(0, 1),
                latent_cache=ckv_buffer.view(
                    -1,
                    page,
                    c["kv_lora_rank"],
                ),
                rope_cache=krot_buffer.view(
                    -1,
                    page,
                    c["qk_rope_head_dim"],
                ),
            )
            if flash_out is not None:
                ctx = flash_out.transpose(0, 1)
                if self._direct_mla_bmm:
                    out = (
                        self._mla_bmm_decode(
                            ctx,
                            self._wuv[layer],
                            True,
                            self._decode_tensor_workspace(
                                layer,
                                "mla_value_output",
                                (H, 1, c["v_head_dim"]),
                                dt,
                            )
                        )
                        if (
                            T == 1
                            and self._mla_bmm_decode is not None
                            and (
                                os.environ.get(
                                    "CCCP_GLM_CUBLAS_VALUE",
                                    "1",
                                )
                                != "0"
                                or os.environ.get(
                                    "CCCP_GLM_CUBLAS_DECODE",
                                    "0",
                                )
                                != "0"
                            )
                        )
                        else None
                    )
                    if out is None:
                        out = torch.bmm(
                            ctx,
                            self._wuv[layer].transpose(1, 2),
                            out=(
                                self._decode_tensor_workspace(
                                    layer,
                                    "mla_value_output",
                                    (H, 1, c["v_head_dim"]),
                                    dt,
                                )
                                if attention_workspace_enabled
                                else None
                            ),
                        )
                else:
                    out = torch.einsum(
                        "htr,hnr->htn",
                        ctx,
                        self._wuv[layer],
                    )
                out = out.transpose(0, 1).reshape(
                    T,
                    H * c["v_head_dim"],
                )
                return self._attention_output(
                    out,
                    layer,
                    self.w(f"{p}.o_proj.weight"),
                )
            self._flashinfer_mla_unavailable = True
            self._flashinfer_mla_state = None
            print(
                "[cccp] FlashInfer MLA kernel 失败，回退原 PyTorch MLA："
                f"{last_error()}",
                flush=True,
            )
        if T > 1:
            workspace_mib = max(
                128,
                int(os.environ.get(
                    "CCCP_GLM_PREFILL_ATTENTION_WORKSPACE_MIB",
                    "512",
                )),
            )
            # Two BF16 score components, one FP32 merged score, FP32
            # softmax, one BF16 cast and a small causal mask.  This estimate
            # intentionally leaves allocator headroom on 24 GiB cards.
            bytes_per_query = max(
                1,
                H * S * (2 + 2 + 4 + 4 + 2) + S,
            )
            query_batch = min(
                T,
                max(32, workspace_mib * 2**20 // bytes_per_query),
            )
            if query_batch >= 32:
                query_batch = max(32, query_batch // 32 * 32)
            if not getattr(self, "_glm_prefill_attention_announced", False):
                print(
                    "[cccp-prefill] attention=cuda.latent-mla-query-batched; "
                    f"outer tokens={T}; query batch={query_batch}; "
                    "single-token projection=forbidden",
                    flush=True,
                )
                self._glm_prefill_attention_announced = True
        else:
            query_batch = 1
        ctx = _latent_attention_context_batched(
            qa,
            q_rot,
            ckv,
            krot,
            scale=scale,
            pos0=pos0,
            query_batch=query_batch,
        )
        if self._direct_mla_bmm:
            out = (
                self._mla_bmm_decode(
                    ctx,
                    self._wuv[layer],
                    True,
                    self._decode_tensor_workspace(
                        layer,
                        "mla_value_output",
                        (H, 1, c["v_head_dim"]),
                        dt,
                    )
                )
                if (
                    T == 1
                    and self._mla_bmm_decode is not None
                    and (
                        os.environ.get(
                            "CCCP_GLM_CUBLAS_VALUE",
                            "1",
                        )
                        != "0"
                        or os.environ.get(
                            "CCCP_GLM_CUBLAS_DECODE",
                            "0",
                        )
                        != "0"
                    )
                )
                else None
            )
            if out is None:
                out = torch.bmm(
                    ctx,
                    self._wuv[layer].transpose(1, 2),
                    out=(
                        self._decode_tensor_workspace(
                            layer,
                            "mla_value_output",
                            (H, 1, c["v_head_dim"]),
                            dt,
                        )
                        if attention_workspace_enabled
                        else None
                    ),
                )
        else:
            out = torch.einsum(
                "htr,hnr->htn",
                ctx,
                self._wuv[layer],
            )
        out = out.transpose(0, 1).reshape(T, H * c["v_head_dim"])
        return self._attention_output(
            out,
            layer,
            self.w(f"{p}.o_proj.weight"),
        )

    def _attention_full(self, x: torch.Tensor, layer: int, pos0: int) -> torch.Tensor:
        c = self.cfg
        H = c["n_heads"]
        p = f"model.layers.{layer}.self_attn"
        T = x.shape[0]
        q_resid = rmsnorm(_attention_linear(x, self.w(f"{p}.q_a_proj.weight")),
                          self.w(f"{p}.q_a_layernorm.weight"), 1e-6)
        q = _attention_linear(q_resid, self.w(f"{p}.q_b_proj.weight"))
        q = q.view(T, H, c["qk_head_dim"]).transpose(0, 1)
        q_nope, q_rot = q.split([c["qk_nope_head_dim"], c["qk_rope_head_dim"]], dim=-1)

        kv = _attention_linear(
            x, self.w(f"{p}.kv_a_proj_with_mqa.weight")
        )
        k_pass, k_rot = kv.split([c["kv_lora_rank"], c["qk_rope_head_dim"]], dim=-1)
        k_pass = rmsnorm(k_pass, self.w(f"{p}.kv_a_layernorm.weight"), 1e-6)
        k_pass = _attention_linear(
            k_pass, self.w(f"{p}.kv_b_proj.weight")
        )
        k_pass = k_pass.view(T, H, c["qk_nope_head_dim"] + c["v_head_dim"]).transpose(0, 1)
        k_nope, v = k_pass.split([c["qk_nope_head_dim"], c["v_head_dim"]], dim=-1)

        q_rot, k_rot = self.rope.apply(q_rot, k_rot.view(1, T, c["qk_rope_head_dim"]), pos0)
        k_rot = k_rot.expand(H, T, c["qk_rope_head_dim"])
        q_f = torch.cat([q_nope, q_rot], dim=-1)
        k_f = torch.cat([k_nope, k_rot], dim=-1)

        past = self.kv[layer]
        if past is not None:
            k_f = torch.cat([past[0].float(), k_f], dim=1)
            v = torch.cat([past[1].float(), v], dim=1)
        self.kv[layer] = (k_f.half(), v.half())

        scores = (q_f.float() @ k_f.float().transpose(1, 2)) / math.sqrt(c["qk_head_dim"])
        S = scores.shape[-1]
        if T > 1:  # decode 单 token 时全部历史可见，无需掩码
            kpos = torch.arange(S, device=x.device)
            qpos = torch.arange(pos0, pos0 + T, device=x.device)
            causal = kpos[None, :] > qpos[:, None]
            scores = scores.masked_fill(causal[None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v.float()).transpose(0, 1).reshape(T, H * c["v_head_dim"])
        return self._attention_output(
            out,
            layer,
            self.w(f"{p}.o_proj.weight"),
        )

    def _moe(
        self,
        x: torch.Tensor,
        layer: int,
        route_logits: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            # 空批(KV 全命中的续算路径):MoE/路由对 0 行是恒等——直接
            # 返回空张量,避免 topk/get_many/C++ 内核的 0 维 grid 崩溃
            # (同进程二次 generate 实证,第三十轮)。
            if residual is not None:
                return residual
            return x
        c = self.cfg
        p = f"model.layers.{layer}.mlp"
        parallel = self.expert_parallel
        route_start = (
            parallel.profile_event()
            if parallel is not None and parallel.profile_enabled
            else None
        )
        logits = (
            route_logits
            if route_logits is not None
            else _linear(x, self.w(f"{p}.gate.weight")).float()
        )
        mask = self._mask(layer)
        w, idx = _glm_route(
            logits,
            self.w(f"{p}.gate.e_score_correction_bias").float(),
            mask,
            c["top_k"],
            c["routed_scaling"],
            (
                parallel.decode_route_outputs()
                if parallel is not None
                else None
            ),
        )
        self._record_route_counts(layer, idx)

        def merge_outputs(
            routed: torch.Tensor,
            shared: torch.Tensor,
        ) -> torch.Tensor:
            routed = routed.to(shared.dtype)
            fused = (
                self._moe_residual_add_decode(
                    residual,
                    routed,
                    shared,
                )
                if (
                    residual is not None
                    and self._moe_residual_add_decode is not None
                )
                else None
            )
            if fused is not None:
                return fused
            result = routed + shared
            return residual + result if residual is not None else result

        if parallel is not None:
            route_end = parallel.profile_event()
            parallel.profile_cuda("route", route_start, route_end)

            def compute_shared_expert() -> torch.Tensor:
                return self._shared_expert_eager(x, layer)

            overlapped_final = (
                parallel.compute_final_overlap(
                    x,
                    layer,
                    idx,
                    w,
                    compute_shared_expert,
                    residual,
                )
                if residual is not None
                else None
            )
            if overlapped_final is not None:
                return overlapped_final

            shared_start = parallel.profile_event()
            shared = compute_shared_expert()
            shared_end = parallel.profile_event()
            final = (
                parallel.compute_final(
                    x,
                    layer,
                    idx,
                    w,
                    shared,
                    residual,
                )
                if residual is not None
                else None
            )
            routed = (
                parallel.compute(x, layer, idx, w)
                if final is None
                else None
            )
            parallel.profile_cuda(
                "shared_expert",
                shared_start,
                shared_end,
            )
            if final is not None:
                return final
            assert routed is not None
            add_start = parallel.profile_event()
            result = merge_outputs(routed, shared)
            add_end = parallel.profile_event()
            parallel.profile_cuda("final_add", add_start, add_end)
            return result

        if (
            x.shape[0] > 1
            and getattr(self.pool, "prefill_rows_supported", False)
            and callable(getattr(self.pool, "run_rows", None))
        ):
            # Exact routes for the whole block are already known. Use the
            # same model-independent expert-grouped packed-row operator as
            # the other CCCP architectures so every selected expert is
            # dequantized once for all of its routed tokens.
            self.pool.prefetch([
                (int(layer), int(expert))
                for expert in torch.unique(idx).detach().cpu().tolist()
            ])
            shared = self._shared_expert_eager(x, layer)
            routed = self.pool.run_rows(
                layer,
                x,
                idx,
                w,
                activation=self.operator_config.expert_activation,
                activation_beta=float(c.get("situ_beta", 4.0)),
                activation_linear_beta=c.get("situ_linear_beta"),
                limit=float(c.get("swiglu_limit", 0.0)),
            )
            return merge_outputs(routed, shared)

        if (
            x.shape[0] == 1
            and self.device.type == "cpu"
            and getattr(
                self.store.man,
                "packed_expert_vq",
                getattr(self.store.man, "projection_vq", False),
            )
            and hasattr(self.pool, "run_native")
        ):
            shared = self._shared_expert_eager(x, layer)
            expert_ids = [int(expert) for expert in idx[0].tolist()]
            self._prev_ids[layer] = expert_ids
            routed = self.pool.run_native(
                layer,
                x,
                idx[0],
                w[0],
                activation=self.operator_config.expert_activation,
                activation_beta=float(c.get("situ_beta", 4.0)),
                activation_linear_beta=c.get("situ_linear_beta"),
                limit=float(c.get("swiglu_limit", 0.0)),
            )
            if routed is None:
                selected = self.pool.get_many([
                    (int(layer), expert) for expert in expert_ids
                ])
                experts = [
                    selected[(int(layer), expert)]
                    for expert in expert_ids
                ]
                from .ops import packed_moe_selected_topk

                routed = packed_moe_selected_topk(
                    x.float(),
                    experts,
                    w[0].float(),
                    activation=self.operator_config.expert_activation,
                    activation_beta=float(c.get("situ_beta", 4.0)),
                    activation_linear_beta=c.get("situ_linear_beta"),
                    limit=float(c.get("swiglu_limit", 0.0)),
                )
                if routed is None:
                    from .grouped import moe_mlp_grouped_mixed

                    routed = moe_mlp_grouped_mixed(
                        x.float(),
                        experts,
                        w[0].float(),
                        limit=float(c.get("swiglu_limit", 0.0)),
                        activation=self.operator_config.expert_activation,
                        situ_beta=float(c.get("situ_beta", 4.0)),
                        situ_linear_beta=c.get("situ_linear_beta"),
            )
            return merge_outputs(routed.reshape(1, -1), shared)

        if x.shape[0] > 1:
            raise RuntimeError(
                "grouped Prefill executor unavailable; the historical "
                "per-token/per-expert projection implementation was deleted"
            )

        # Decode is intrinsically one row, but every routed expert still has
        # to enter the fused top-k packed operator.  The former generic
        # per-expert projection loop is deliberately absent: an operator
        # regression must fail loudly instead of restoring GEMV per expert.
        if x.shape[0] == 1:
            from .grouped import moe_mlp_grouped_mixed

            shared = self._shared_expert_eager(x, layer)
            eids = idx[0].tolist()
            self._prev_ids[layer] = eids
            got = self.pool.get_many([(layer, expert) for expert in eids])
            experts = [got[(layer, expert)] for expert in eids]
            if all(
                isinstance(gu, VQWeight) and isinstance(dn, VQWeight)
                for gu, dn in experts
            ):
                routed = moe_mlp_grouped_mixed(
                    x,
                    experts,
                    w[0],
                    limit=float(c.get("swiglu_limit", 0.0)),
                    activation=self.operator_config.expert_activation,
                    situ_beta=float(c.get("situ_beta", 4.0)),
                    situ_linear_beta=c.get("situ_linear_beta"),
                )
                return merge_outputs(
                    routed.unsqueeze(0),
                    shared,
                )
            # 长生成后池内专家为紧凑/展开形态(非 VQWeight 包装):单行
            # 走 run_rows(行数无关的分组算子),与 prefill 分支同源——
            # 同进程二次 generate 续算路径的实测触发点(第三十轮)。
            run_rows = getattr(self.pool, "run_rows", None)
            if callable(run_rows):
                routed = run_rows(
                    layer,
                    x,
                    idx,
                    w,
                    activation=self.operator_config.expert_activation,
                    activation_beta=float(c.get("situ_beta", 4.0)),
                    activation_linear_beta=c.get("situ_linear_beta"),
                    limit=float(c.get("swiglu_limit", 0.0)),
                )
                return merge_outputs(routed, shared)
        raise RuntimeError(
            "fused packed top-k decode operator unavailable; the legacy "
            "single-token expert projection implementation was deleted"
        )

    def _dense_mlp(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        p = f"model.layers.{layer}.mlp"
        return _linear(
            _swiglu_linear(
                x,
                self.w(f"{p}.gate_proj.weight"),
                self.w(f"{p}.up_proj.weight"),
            ),
            self.w(f"{p}.down_proj.weight"),
        )

    def _mask(self, layer: int) -> torch.Tensor:
        """该层可用专家掩码（缓存；drop 专家为 False）。"""
        m = self._masks.get(layer)
        if m is None:
            m = self.store.available_mask(layer).to(self.device)
            self._masks[layer] = m
        return m

    def _record_route_counts(
        self,
        layer: int,
        indices: torch.Tensor,
    ) -> None:
        """Record exact selected experts only while route calibration is active."""
        if os.environ.get("CCCP_ROUTE_COUNTS", "0") == "0":
            return
        pool = self.pool
        if pool is None:
            return
        counts = getattr(pool, "route_counts", None)
        if counts is None:
            counts = Counter()
            pool.route_counts = counts
        counts.update(
            (int(layer), int(expert))
            for expert in indices.detach().reshape(-1).cpu().tolist()
        )

    def _forward_layer(
        self,
        x: torch.Tensor,
        layer: int,
        pos0: int,
    ) -> torch.Tensor:
        """Execute one architecture layer for one contiguous token block."""
        c = self.cfg
        eps = c["rms_eps"]
        prev = self._prev_ids.get(layer)
        if (
            prev
            and x.shape[0] == 1
            and os.environ.get("CCCP_PREFETCH", "1") != "0"
        ):
            self.pool.prefetch([(layer, expert) for expert in prev])
        h = self._attention(
            x,
            layer,
            pos0,
            self.w(f"model.layers.{layer}.input_layernorm.weight"),
        )
        if layer not in set(c["moe_layers"]):
            x = x + h
            hn = rmsnorm(
                x,
                self.w(
                    f"model.layers.{layer}."
                    "post_attention_layernorm.weight"
                ),
                eps,
            )
            return x + self._dense_mlp(hn, layer)

        post_norm = self.w(
            f"model.layers.{layer}.post_attention_layernorm.weight"
        )
        route_weight = self.w(f"model.layers.{layer}.mlp.gate.weight")
        fused_post = (
            self._residual_norm_router_decode(
                x,
                h,
                post_norm,
                route_weight,
                eps,
                (
                    self.expert_parallel.decode_norm_output()
                    if self.expert_parallel is not None
                    else None
                ),
                (
                    self._decode_workspace(
                        layer,
                        "post_attention_residual",
                        int(x.shape[1]),
                    ),
                    self._decode_workspace(
                        layer,
                        "route_logits",
                        int(route_weight.shape[0]),
                    ),
                )
                if (
                    self.expert_parallel is not None
                    and os.environ.get("CCCP_DECODE_WORKSPACES", "1") != "0"
                )
                else None,
            )
            if self._residual_norm_router_decode is not None
            else None
        )
        if fused_post is None:
            x = x + h
            hn = rmsnorm(x, post_norm, eps)
            route_logits = None
        else:
            x, hn, route_logits = fused_post
        return self._moe(
            hn,
            layer,
            route_logits=route_logits,
            residual=x,
        )

    def forward_hidden(
        self,
        ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """前向一段 token，返回全部位置的最终 hidden [T, hidden]（已过 final norm）。"""
        c = self.cfg
        eps = c["rms_eps"]
        if self.pos + len(ids) > self.max_ctx:
            raise RuntimeError(f"上下文超限（{self.pos + len(ids)} > {self.max_ctx}），"
                               f"请 /clear 或调大 --max-ctx")
        pos0 = self.pos
        self._ensure_rope_capacity(pos0 + len(ids))
        self._ensure_latent_capacity(pos0 + len(ids))
        if len(ids) == 1 and self._decode_position is not None:
            self._decode_position.fill_(pos0)
        self._flashinfer_mla_state = (
            self._prepare_flashinfer_mla_decode(pos0 + len(ids))
            if len(ids) == 1
            else None
        )
        x = self.embed(ids)
        if self._prev_ids and len(ids) == 1 and os.environ.get("CCCP_PREFETCH", "1") != "0":
            # decode 单步：token 级全层预取（窗口 = 整个 token，见 dsv4model.decode）
            for layer_id, expert_ids in self._prev_ids.items():
                self.pool.prefetch(
                    [(layer_id, expert_id) for expert_id in expert_ids]
                )
        for layer in range(c["n_layers"]):
            x = self._forward_layer(x, layer, pos0)
        self.pos += len(ids)
        return rmsnorm(
            x,
            self.w("model.norm.weight"),
            eps,
            output=(
                self._decode_workspace(
                    -1,
                    "final_norm",
                    int(x.shape[1]),
                )
                if (
                    x.shape[0] == 1
                    and
                    os.environ.get(
                        "CCCP_STATIC_LM_OUTPUT",
                        "0",
                    )
                    != "0"
                    and os.environ.get(
                        "CCCP_RMSNORM_WORKSPACES",
                        "1",
                    )
                    != "0"
                )
                else None
            ),
        )

    def prefill_chunked(
        self,
        ids: torch.Tensor | list[int],
        *,
        chunk_size: int = 4096,
        progress_callback=None,
        layer_progress_callback=None,
    ) -> torch.Tensor:
        """Route-scan prefill with one layer's KV resident at a time.

        The sequence is evaluated layer-first and each layer is split into
        bounded contiguous blocks. This is mathematically identical to the
        regular causal prefill, but finished layer KV is released immediately
        instead of retaining every layer until the whole scan completes.
        """
        if isinstance(ids, torch.Tensor) and ids.ndim == 2:
            if ids.shape[0] != 1:
                raise ValueError("分层 prefill 只支持 batch=1")
            ids = ids.reshape(-1)
        if self.pos != 0:
            raise RuntimeError("分层 prefill 只允许在 reset 后开始")
        total = len(ids)
        if total <= 0:
            raise ValueError("prefill token 不能为空")
        if total > self.max_ctx:
            raise RuntimeError(f"上下文超限（{total} > {self.max_ctx}）")
        chunk_size = max(1, int(chunk_size))
        self._ensure_rope_capacity(total)
        self._ensure_latent_capacity(total)
        self._flashinfer_mla_state = None
        x = self.embed(ids)
        layer_count = int(self.cfg["n_layers"])
        for layer in range(layer_count):
            self.kv[layer] = None
            outputs = []
            for start in range(0, total, chunk_size):
                stop = min(total, start + chunk_size)
                outputs.append(
                    self._forward_layer(x[start:stop], layer, start)
                )
                if layer_progress_callback is not None and stop < total:
                    layer_progress_callback(
                        start,
                        stop,
                        layer + 1,
                        layer_count,
                    )
            next_x = torch.cat(outputs, dim=0)
            del outputs
            x = next_x
            self.kv[layer] = None
            self._release_completed_scan_layer(layer)
            if layer_progress_callback is not None:
                layer_progress_callback(
                    ((total - 1) // chunk_size) * chunk_size,
                    total,
                    layer + 1,
                    layer_count,
                )
        self.pos = total
        if progress_callback is not None:
            progress_callback(total)
        return rmsnorm(
            x,
            self.w("model.norm.weight"),
            self.cfg["rms_eps"],
        )

    def logits_of(self, h: torch.Tensor) -> torch.Tensor:
        """hidden [N, hidden] → logits [N, vocab]（lm_head，int4 分块或 f32 直接乘）。"""
        lm = self.w("lm_head.weight")
        if (
            self._lm_head_int4 is not None
            and os.environ.get("CCCP_LM_HEAD_INT4", "1") != "0"
        ):
            lm = self._lm_head_int4
        if isinstance(lm, Int4Weight):
            return lm.matmul_T_decode_fused(
                h,
                output=(
                    self._decode_workspace(
                        -1,
                        "lm_logits",
                        int(lm.shape[0]),
                    )
                    if (
                        h.shape[0] == 1
                        and os.environ.get(
                            "CCCP_STATIC_LM_OUTPUT",
                            "0",
                        )
                        != "0"
                    )
                    else None
                ),
            )
        return h.float() @ lm.t()

    def forward(
        self,
        ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """前向一段 token（prefill 或单步 decode），返回最后位置的 f32 logits [vocab]。"""
        h = self.forward_hidden(ids)
        return self.logits_of(h[-1:]).squeeze(0)

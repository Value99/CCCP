"""Linux/CUDA 的首选 FlashInfer MLA 适配层。

适配层直接复用 CCCP 的分离 cKV/kPE 缓冲。依赖缺失、显式关闭或执行失败
时返回 ``None``，调用方只能选择 CCCP 自带的 paged latent CUDA 算子；原生
Windows 从不进入本适配层，也不允许静默落入普通 PyTorch BF16 Attention。
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from pathlib import Path

import torch


_WRAPPER_CLS = None
_LAST_ERROR: Exception | None = None


def _ensure_ninja_on_path() -> None:
    """Expose the venv-bundled ninja to FlashInfer's JIT helper."""
    if shutil.which("ninja") is not None:
        return
    executable = "ninja.exe" if os.name == "nt" else "ninja"
    # ``venv/bin/python`` 常是指向系统解释器的符号链接；只调用
    # ``resolve()`` 会跳出虚拟环境并错过同目录安装的 ninja。
    directories = (
        Path(sys.executable).parent,
        Path(sys.executable).resolve().parent,
    )
    for directory in dict.fromkeys(directories):
        candidate = directory / executable
        if candidate.is_file():
            os.environ["PATH"] = (
                str(directory)
                + os.pathsep
                + os.environ.get("PATH", "")
            )
            return


def _wrapper_cls():
    global _WRAPPER_CLS, _LAST_ERROR
    if os.environ.get("CCCP_FLASHINFER_MLA", "1") == "0":
        return None
    if _WRAPPER_CLS is False:
        return None
    if _WRAPPER_CLS is None:
        try:
            _ensure_ninja_on_path()
            from flashinfer.mla import BatchMLAPagedAttentionWrapper

            _WRAPPER_CLS = BatchMLAPagedAttentionWrapper
        except Exception as exc:
            _LAST_ERROR = exc
            _WRAPPER_CLS = False
            return None
    return _WRAPPER_CLS


class FlashInferMLADecode:
    """batch=1、query=1 的固定地址 FlashInfer MLA 执行器。"""

    page_size = 64

    def __init__(
        self,
        *,
        device: torch.device,
        max_ctx: int,
        heads: int,
        ckv_dim: int,
        kpe_dim: int,
        dtype: torch.dtype,
        softmax_scale: float,
    ) -> None:
        wrapper_cls = _wrapper_cls()
        if wrapper_cls is None:
            raise RuntimeError("FlashInfer MLA 不可用")
        self.device = device
        self.max_ctx = max_ctx
        self.heads = heads
        self.ckv_dim = ckv_dim
        self.kpe_dim = kpe_dim
        self.dtype = dtype
        self.softmax_scale = softmax_scale
        self.max_blocks = (
            max_ctx + self.page_size - 1
        ) // self.page_size

        # FlashInfer 建议 128 MiB split-K workspace。只有显式启用时分配。
        workspace = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
        self._qo_gpu = torch.empty(
            2, dtype=torch.int32, device=device
        )
        self._kv_indptr_gpu = torch.empty(
            2, dtype=torch.int32, device=device
        )
        self._kv_indices_gpu = torch.empty(
            self.max_blocks,
            dtype=torch.int32,
            device=device,
        )
        self._kv_len_gpu = torch.empty(
            1, dtype=torch.int32, device=device
        )
        self._qo_cpu = torch.tensor(
            [0, 1], dtype=torch.int32
        ).pin_memory()
        self._qo_cpu_ring = torch.empty(
            max_ctx + 1,
            2,
            dtype=torch.int32,
            pin_memory=True,
        )
        # A device-token decode loop can enqueue several complete tokens
        # without a CPU synchronization.  Keep one pinned metadata source per
        # context length so an async H2D copy is never fed from a CPU buffer
        # that the next token has already overwritten.
        self._kv_indptr_cpu_ring = torch.empty(
            max_ctx + 1,
            2,
            dtype=torch.int32,
            pin_memory=True,
        )
        self._kv_indices_cpu = torch.arange(
            self.max_blocks, dtype=torch.int32
        ).pin_memory()
        self._kv_len_cpu_ring = torch.empty(
            max_ctx + 1,
            1,
            dtype=torch.int32,
            pin_memory=True,
        )
        self._out = torch.empty(
            1,
            heads,
            ckv_dim,
            dtype=dtype,
            device=device,
        )
        self._wrapper = wrapper_cls(
            workspace,
            use_cuda_graph=True,
            qo_indptr=self._qo_gpu,
            kv_indptr=self._kv_indptr_gpu,
            kv_indices=self._kv_indices_gpu,
            kv_len_arr=self._kv_len_gpu,
            backend=os.environ.get(
                "CCCP_FLASHINFER_BACKEND",
                "auto",
            ),
        )
        self._prepared_blocks = 0
        self._prepared_query_length = 0
        self._plan_initialized = False
        self._decode_plan_initialized = False
        self.gpu_plan_hits = 0
        self.gpu_plan_rejections = 0
        self.cpu_plan_calls = 0

    def prepare(self, length: int) -> None:
        if length <= 0 or length > self.max_ctx:
            raise ValueError(
                f"FlashInfer MLA length={length} 超出 1..{self.max_ctx}"
            )
        blocks = (
            length + self.page_size - 1
        ) // self.page_size
        kv_indptr_cpu = self._kv_indptr_cpu_ring[length]
        kv_len_cpu = self._kv_len_cpu_ring[length]
        kv_indptr_cpu[0] = 0
        kv_indptr_cpu[1] = blocks
        kv_len_cpu[0] = length
        gpu_plan = False
        if getattr(
            self,
            "_decode_plan_initialized",
            self._plan_initialized,
        ):
            from .fusedext import (
                flashinfer_mla_batch1_plan_fused,
            )

            try:
                gpu_plan = flashinfer_mla_batch1_plan_fused(
                    self._wrapper._int_workspace_buffer,
                    self._kv_indptr_gpu,
                    self._kv_indices_gpu,
                    self._kv_len_gpu,
                    length,
                    self.page_size,
                    self.heads,
                    self._wrapper._plan_info,
                )
            except RuntimeError:
                # An unknown FlashInfer planner layout is a compatibility
                # miss, not an inference failure.  Its official planner is
                # the numerical source of truth for this token.
                gpu_plan = False
            if gpu_plan:
                self.gpu_plan_hits += 1
            else:
                self.gpu_plan_rejections += 1
        if not gpu_plan:
            self._wrapper.plan(
                self._qo_cpu,
                kv_indptr_cpu,
                self._kv_indices_cpu[:blocks],
                kv_len_cpu,
                num_heads=self.heads,
                head_dim_ckv=self.ckv_dim,
                head_dim_kpe=self.kpe_dim,
                page_size=self.page_size,
                causal=False,
                sm_scale=self.softmax_scale,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
            self.cpu_plan_calls += 1
            self._plan_initialized = True
            self._decode_plan_initialized = True
        self._prepared_blocks = blocks
        self._prepared_query_length = 1

    def prepare_prefill(
        self,
        query_length: int,
        length: int,
    ) -> None:
        """Plan one causal block whose queries end at ``length``."""
        if (
            query_length <= 0
            or length < query_length
            or length > self.max_ctx
        ):
            raise ValueError(
                "FlashInfer MLA prefill lengths must satisfy "
                "0 < query_length <= length <= max_ctx"
            )
        blocks = (length + self.page_size - 1) // self.page_size
        qo_cpu = self._qo_cpu_ring[query_length]
        qo_cpu[0] = 0
        qo_cpu[1] = query_length
        kv_indptr_cpu = self._kv_indptr_cpu_ring[length]
        kv_len_cpu = self._kv_len_cpu_ring[length]
        kv_indptr_cpu[0] = 0
        kv_indptr_cpu[1] = blocks
        kv_len_cpu[0] = length
        self._wrapper.plan(
            qo_cpu,
            kv_indptr_cpu,
            self._kv_indices_cpu[:blocks],
            kv_len_cpu,
            num_heads=self.heads,
            head_dim_ckv=self.ckv_dim,
            head_dim_kpe=self.kpe_dim,
            page_size=self.page_size,
            causal=True,
            sm_scale=self.softmax_scale,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )
        self.cpu_plan_calls += 1
        self._plan_initialized = True
        self._decode_plan_initialized = False
        self._prepared_blocks = blocks
        self._prepared_query_length = query_length

    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
    ) -> torch.Tensor:
        if self._prepared_blocks == 0:
            raise RuntimeError("FlashInfer MLA 尚未 prepare")
        return self._wrapper.run(
            q_nope,
            q_pe,
            ckv_cache[:self._prepared_blocks],
            kpe_cache[:self._prepared_blocks],
            out=self._out,
        )

    def run_prefill(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._prepared_query_length != int(q_nope.shape[0]):
            raise RuntimeError("FlashInfer MLA prefill plan is stale")
        return self._wrapper.run(
            q_nope,
            q_pe,
            ckv_cache[:self._prepared_blocks],
            kpe_cache[:self._prepared_blocks],
            out=output,
        )


def create_runner(
    *,
    device: torch.device,
    max_ctx: int,
    heads: int,
    ckv_dim: int,
    kpe_dim: int,
    dtype: torch.dtype,
    qk_head_dim: int,
) -> FlashInferMLADecode | None:
    global _LAST_ERROR
    if _wrapper_cls() is None:
        return None
    try:
        return FlashInferMLADecode(
            device=device,
            max_ctx=max_ctx,
            heads=heads,
            ckv_dim=ckv_dim,
            kpe_dim=kpe_dim,
            dtype=dtype,
            softmax_scale=1.0 / math.sqrt(qk_head_dim),
        )
    except Exception as exc:
        _LAST_ERROR = exc
        return None


def prepare_runner(
    runner: FlashInferMLADecode,
    length: int,
) -> bool:
    global _LAST_ERROR
    try:
        runner.prepare(length)
        return True
    except Exception as exc:
        _LAST_ERROR = exc
        return False


def prepare_prefill_runner(
    runner: FlashInferMLADecode,
    query_length: int,
    length: int,
) -> bool:
    global _LAST_ERROR
    try:
        runner.prepare_prefill(query_length, length)
        return True
    except Exception as exc:
        _LAST_ERROR = exc
        return False


def decode(
    runner: FlashInferMLADecode,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
) -> torch.Tensor | None:
    global _LAST_ERROR
    try:
        return runner.run(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
        )
    except Exception as exc:
        _LAST_ERROR = exc
        return None


def prefill(
    runner: FlashInferMLADecode,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    global _LAST_ERROR
    try:
        return runner.run_prefill(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            output,
        )
    except Exception as exc:
        _LAST_ERROR = exc
        return None


def last_error() -> Exception | None:
    return _LAST_ERROR

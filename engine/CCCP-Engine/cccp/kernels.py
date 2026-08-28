"""CCCP 数值内核：int4 / VQ 矩阵乘、RMSNorm、交错 RoPE。

两类量化权重的免还原/分块还原矩阵乘：
  - Int4Weight：u8 双半字节 [R, C//2] + f16 组缩放 [R, C//64]；matmul 按行块
    反量化到 f32 后 torch.mm，内存峰值 = 一个行块。
  - VQWeight：u8 码字索引 [R, C//dim] + 层共享码本 f32 [K, dim]；LUT 算法：
    y[r] = Σ_b s[b, idx[r, b]]，其中 s[b, c] = x[bd:bd+dim]·cb[c] 只需算 B×K 次，
    把 O(R·C) 的 matmul 降为 O(B·K + R·B) 的查表加（v 档约快 6 倍）。
RMSNorm / RoPE 与 CCCP/modelmath.py 逐行一致（单测对照过朴素实现）。
"""

from __future__ import annotations

import math
import os
import statistics
import time

import torch

from .precision import compute_dtype

INT4_GROUP = 64
_BLOCK_FP8_LAYOUT_DECISIONS: dict[
    tuple[int, int, int], tuple[bool, float, float]
] = {}


def rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSNorm：f32 算方差，乘权重后回原 dtype。
    CUDA + f32 输入走融合 kernel（1 次 launch 替代 ~6 次），其余回退 torch 表达式。"""
    dt = x.dtype
    if dt == torch.float32 and x.is_cuda:
        fn = _rms_fused()
        if fn is not None:
            r = fn(
                x,
                w if w.dtype == torch.float32 else w.float(),
                eps,
                output=output,
            )
            if r is not None:
                return r
    v = x.float().pow(2).mean(-1, keepdim=True)
    result = (
        w.float() * (x.float() * torch.rsqrt(v + eps))
    ).to(dt)
    if output is not None:
        output.copy_(result)
        return output
    return result


def _rms_fused():
    """fusedext.rmsnorm_fused 的懒导入（避免 kernels 被 CPU-only 场景导入时触发扩展编译）。"""
    global _RMS_FUSED
    if _RMS_FUSED is None:
        try:
            from .fusedext import rmsnorm_fused
            _RMS_FUSED = rmsnorm_fused
        except Exception:
            _RMS_FUSED = False
    return _RMS_FUSED or None


_RMS_FUSED = None


def _int4_gemv_fused():
    """Lazily resolve the direct packed INT4 decode kernel."""
    global _INT4_GEMV_FUSED
    if _INT4_GEMV_FUSED is None:
        try:
            from .fusedext import int4_gemv_fused
            _INT4_GEMV_FUSED = int4_gemv_fused
        except Exception:
            _INT4_GEMV_FUSED = False
    return _INT4_GEMV_FUSED or None


_INT4_GEMV_FUSED = None


def _block_fp8_gemv_fused():
    """Lazily resolve the native E4M3 block-scaled decode kernel."""
    global _BLOCK_FP8_GEMV_FUSED
    if _BLOCK_FP8_GEMV_FUSED is None:
        try:
            from .ops import block_scaled_gemv

            _BLOCK_FP8_GEMV_FUSED = block_scaled_gemv
        except Exception:
            _BLOCK_FP8_GEMV_FUSED = False
    return _BLOCK_FP8_GEMV_FUSED or None


_BLOCK_FP8_GEMV_FUSED = None


def _block_fp8_grouped_gemv_fused():
    """Lazily resolve the public grouped block-FP8 decode operation."""
    global _BLOCK_FP8_GROUPED_GEMV_FUSED
    if _BLOCK_FP8_GROUPED_GEMV_FUSED is None:
        try:
            from .ops import block_scaled_grouped_gemv

            _BLOCK_FP8_GROUPED_GEMV_FUSED = block_scaled_grouped_gemv
        except Exception:
            _BLOCK_FP8_GROUPED_GEMV_FUSED = False
    return _BLOCK_FP8_GROUPED_GEMV_FUSED or None


_BLOCK_FP8_GROUPED_GEMV_FUSED = None


def _glm_rope_qk_fused():
    """Lazily resolve the GLM Q/K RoPE fusion."""
    global _GLM_ROPE_QK_FUSED
    if _GLM_ROPE_QK_FUSED is None:
        try:
            from .fusedext import glm_rope_qk_fused
            _GLM_ROPE_QK_FUSED = glm_rope_qk_fused
        except Exception:
            _GLM_ROPE_QK_FUSED = False
    return _GLM_ROPE_QK_FUSED or None


_GLM_ROPE_QK_FUSED = None


def merge_attention_scores(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Merge latent-MLA score components without changing their GEMMs."""
    if a.is_cuda and a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16:
        global _GLM_MERGE_SCORES_FUSED
        if _GLM_MERGE_SCORES_FUSED is None:
            try:
                from .fusedext import glm_merge_scores_fused
                _GLM_MERGE_SCORES_FUSED = glm_merge_scores_fused
            except Exception:
                _GLM_MERGE_SCORES_FUSED = False
        if _GLM_MERGE_SCORES_FUSED:
            result = _GLM_MERGE_SCORES_FUSED(a, b, scale)
            if result is not None:
                return result
    return a.float() / scale + b.float() / scale


_GLM_MERGE_SCORES_FUSED = None


class RopeCache:
    """RoPE cos/sin 预计算（交错布局，[T, rope_dim//2]）。"""

    def __init__(self, rope_dim: int, theta: float, max_len: int = 8192):
        self.rope_dim = int(rope_dim)
        self.theta = float(theta)
        self.cos, self.sin = self._build(int(max_len))

    def _build(self, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        inv = 1.0 / (
            self.theta
            ** (
                torch.arange(
                    0,
                    self.rope_dim,
                    2,
                    dtype=torch.float32,
                )
                / self.rope_dim
            )
        )
        freqs = torch.outer(
            torch.arange(max_len, dtype=torch.float32),
            inv,
        )
        return freqs.cos(), freqs.sin()

    def ensure_length(self, required: int) -> bool:
        """按需扩展 RoPE 表；返回地址是否发生变化。"""

        required = int(required)
        if required <= self.cos.shape[0]:
            return False
        device = self.cos.device
        capacity = max(required, self.cos.shape[0] * 2)
        cos, sin = self._build(capacity)
        self.cos = cos.to(device)
        self.sin = sin.to(device)
        return True

    def apply(self, q: torch.Tensor, k: torch.Tensor, pos0: int):
        """q: [H, T, D]；k: [1, T, D] → HF apply_rotary_pos_emb_interleave 的 cat 布局。"""
        T = q.shape[1]
        cos = self.cos[pos0:pos0 + T]
        sin = self.sin[pos0:pos0 + T]
        if q.is_cuda and q.dtype == torch.float32 and k.dtype == torch.float32:
            fn = _glm_rope_qk_fused()
            if fn is not None:
                result = fn(q, k, cos, sin)
                if result is not None:
                    return result
        q1, q2 = q[..., 0::2], q[..., 1::2]
        k1, k2 = k[..., 0::2], k[..., 1::2]
        qe = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        ke = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return qe, ke


def dequant_int4(packed: torch.Tensor, scales: torch.Tensor,
                 gs: int = INT4_GROUP, half: bool = False) -> torch.Tensor:
    """int4 行块反量化：packed u8 [r, C//2]，scales f16 [r, C//gs] → f32/f16 [r, C]。

    用 256 字节查找表一次 gather 出 (lo, hi) 两个半字节（连续写，最快路径），
    再按组就地乘缩放。比逐半字节位运算 + 跨步写快约 2 倍（本机实测）。
    half=True：LUT 与输出走 fp16（2080 张量核 matmul 提速 + 写出量减半；
    int4 网格本身 ~6% 误差，fp16 的 0.05% 精度远超所需，无额外损失）。
    """
    r = packed.shape[0]
    cols = packed.shape[1] * 2
    key = f"{packed.device}:h" if half else str(packed.device)
    lut = _LUTS.get(key)
    if lut is None:
        base = _INT4_LUT.to(torch.float16) if half else _INT4_LUT
        lut = base.to(packed.device)
        _LUTS[key] = lut
    w = lut[packed.long()].view(r, cols)
    if half:
        w.view(r, cols // gs, gs).mul_(scales.unsqueeze(-1))
    else:
        w.view(r, cols // gs, gs).mul_(scales.float().unsqueeze(-1))
    return w


def _make_lut() -> torch.Tensor:
    t = torch.arange(256, dtype=torch.int16)
    return (torch.stack((t & 15, t >> 4), 1).to(torch.float32) - 8)


_INT4_LUT = _make_lut()  # [256, 2]：字节 → (低半字节值, 高半字节值)，零点是 8
_LUTS: dict = {"cpu": _INT4_LUT}

# 码本半精度计算副本缓存：键 = (f32 码本 data_ptr, dtype)，值 = (低精度副本, f32 强引用)。
# 强引用防 data_ptr 复用后串码本（同 ExpertPool._cb_dev 的竞态教训）；
# 同层同档专家共享同一码本张量 → 全池天然去重，每 (层,档) 只多一份副本。
_CB_LO: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def cb_compute(cb: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """码本的计算 dtype 副本（fp32 原样返回；半精度按 ptr 去重缓存）。"""
    if dt == torch.float32 or cb.dtype == dt:
        return cb
    key = (cb.data_ptr(), str(dt))
    ent = _CB_LO.get(key)
    if ent is None:
        lo = cb.to(dt)
        ent = (lo, cb)
        _CB_LO[key] = ent
    return ent[0]


def _lut_on(device) -> torch.Tensor:
    """各设备缓存一份 LUT（GPU 推理路径避免跨设备索引）。"""
    key = str(device)
    lut = _LUTS.get(key)
    if lut is None:
        lut = _INT4_LUT.to(device)
        _LUTS[key] = lut
    return lut


class Int4Weight:
    """int4-g64 打包权重；matmul 按行块在线反量化（dense 低内存驻留方案）。
    half=True：反量化与 matmul 走 fp16（Turing 张量核，~2× fp32；权重仍 int4 驻留）。"""

    __slots__ = ("q", "s", "cols", "gs", "half")

    def __init__(self, q: torch.Tensor, s: torch.Tensor, cols: int, gs: int = INT4_GROUP,
                 half: bool = False):
        self.q = q          # u8 [R, C//2]
        self.s = s          # f16 [R, C//gs]
        self.cols = cols
        self.gs = gs
        self.half = half

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.q.shape[0], self.cols])

    @property
    def nbytes(self) -> int:
        return self.q.numel() + self.s.numel() * 2

    def dequant_rows(self, r0: int, r1: int) -> torch.Tensor:
        return dequant_int4(self.q[r0:r1], self.s[r0:r1], self.gs, half=self.half)

    def matmul_T(self, x: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        """y = x @ W.T。x: [T, C] → [T, R] f32（half 时内部 fp16 计算、输出 f32）。

        行块大小自适应： transient 反量化块 ≤64MB（GPU 上 wq_b 级别大矩阵一次
        成型——原固定 512 行会把单个 GEMM 拆成 64 块 × 5 次 launch，WDDM 下
        launch 开销远超计算本身；显存代价仅一块临时缓冲）。
        """
        if (
            not x.is_cuda
            and x.dim() == 2
            and x.shape[0] == 1
            and self.q.dtype == torch.uint8
            and self.s.dtype == torch.float16
        ):
            from .cpuext import int4_gemv_cpu

            fused_cpu = int4_gemv_cpu(
                x, self.q, self.s, self.cols, self.gs
            )
            if fused_cpu is not None:
                return fused_cpu
        R = self.q.shape[0]
        if chunk is None:
            esz = 2 if self.half else 4
            chunk = max(512, min(R, (64 * 2**20) // max(self.cols * esz, 1)))
        if self.half:
            xh = x.half()
            out = torch.empty(x.shape[0], R, dtype=torch.float16, device=x.device)
            for r0 in range(0, R, chunk):
                r1 = min(r0 + chunk, R)
                out[:, r0:r1] = xh @ self.dequant_rows(r0, r1).t()
            return out.float()
        out = torch.empty(x.shape[0], R, dtype=torch.float32, device=x.device)
        for r0 in range(0, R, chunk):
            r1 = min(r0 + chunk, R)
            out[:, r0:r1] = x.float() @ self.dequant_rows(r0, r1).t()
        return out

    def matmul_T_decode_fused(
        self,
        x: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use direct packed INT4 GEMV for a compatible decode row."""
        if (
            x.is_cuda
            and x.dim() == 2
            and x.shape[0] == 1
            and x.dtype in (torch.float32, torch.bfloat16)
            and self.q.dtype == torch.uint8
            and self.s.dtype == torch.float16
            and self.gs == 64
        ):
            fn = _int4_gemv_fused()
            if fn is not None:
                fused = fn(
                    x,
                    self.q,
                    self.s,
                    self.cols,
                    self.gs,
                    output=output,
                )
                if fused is not None:
                    return fused
        return self.matmul_T(x)

    def row(self, r: int) -> torch.Tensor:
        """反量化单行 [C]（embed 查表用）。"""
        return self.dequant_rows(r, r + 1).squeeze(0)


class BlockFP8Weight:
    """原生 E4M3 权重与 128×128 FP32 反量化尺度。

    CCCP ``dense=fp8-native`` 直接保存 FP8 检查点字节，不先展开成
    BF16/F32。矩阵乘按行块临时反量化，常驻显存仍是 1 byte/weight。
    CPU 单 token decode 返回每个权重自己的固定缓冲；同一权重的下一次
    decode 会覆盖该缓冲。
    """

    __slots__ = (
        "q",
        "s",
        "cols",
        "block",
        "rows",
        "layout",
        "_decode_outputs",
        "_fp8_decode_input",
        "_fp8_decode_scale",
    )

    def __init__(
        self,
        q: torch.Tensor,
        s: torch.Tensor,
        cols: int,
        block: int = 128,
        *,
        rows: int | None = None,
        layout: str | None = None,
    ):
        if q.dtype != torch.uint8 or s.dtype != torch.float32:
            raise TypeError("BlockFP8Weight requires uint8 data and f32 scales")
        self.q = q
        self.s = s
        self.cols = cols
        self.block = block
        if layout is None:
            layout = "row-major" if q.ndim == 2 else "block-major32"
        if layout not in (
            "row-major",
            "block-major32",
            "q4_0",
            "tensor-fp8",
        ):
            raise ValueError(f"unsupported block FP8 layout {layout!r}")
        if layout in ("row-major", "tensor-fp8") and q.ndim != 2:
            raise ValueError("row-major block FP8 requires a rank-2 tensor")
        if layout == "block-major32" and q.ndim != 5:
            raise ValueError(
                "block-major32 FP8 requires [RB,4,CB,32,128]"
            )
        if layout == "q4_0":
            if q.ndim != 1 or cols % 32:
                raise ValueError("q4_0 requires a flat image and cols % 32 == 0")
            expected = int(rows) * (int(cols) // 32) * 18
            if q.numel() != expected or s.numel() != 0:
                raise ValueError(
                    f"q4_0 image mismatch: {q.numel()} != {expected}"
                )
        self.layout = layout
        self.rows = int(q.shape[0]) if rows is None else int(rows)
        self._decode_outputs = {}
        self._fp8_decode_input = None
        self._fp8_decode_scale = None
        if layout == "tensor-fp8":
            if s.numel() != 1:
                raise ValueError("tensor-fp8 requires one FP32 weight scale")
            if q.is_cuda:
                self._fp8_decode_input = torch.empty(
                    (1, self.cols),
                    dtype=torch.float8_e4m3fn,
                    device=q.device,
                )
                self._fp8_decode_scale = torch.empty(
                    (1, 1), dtype=torch.float32, device=q.device
                )

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.rows, self.cols])

    @property
    def nbytes(self) -> int:
        return self.q.numel() + self.s.numel() * 4

    @property
    def dtype(self) -> torch.dtype:
        """Logical compute dtype exposed to generic linear dispatch."""
        return torch.bfloat16

    @property
    def device(self) -> torch.device:
        return self.q.device

    def to(
        self,
        device=None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ):
        """Move the compact FP8 payload, or explicitly materialize a dtype.

        A device-only transfer keeps the one-byte weights and FP32 block
        scales compact.  Supplying a floating dtype is an explicit request
        used only by a few very small metadata/norm tensors.
        """
        if isinstance(device, torch.dtype):
            dtype = device
            device = self.q.device
        if dtype is not None and dtype != torch.uint8:
            target = self.q.device if device is None else device
            return self.dequant_rows(0, self.rows, dtype).to(
                target,
                non_blocking=non_blocking,
            )
        target = self.q.device if device is None else device
        target_device = torch.device(target)
        if self.layout == "q4_0" and target_device.type != "cpu":
            return self.dequant_rows(0, self.rows, dtype or torch.bfloat16).to(
                target_device, non_blocking=non_blocking
            )
        if self.layout == "block-major32" and target_device.type != "cpu":
            return self.to_row_major().to(
                target_device,
                non_blocking=non_blocking,
            )
        return BlockFP8Weight(
            self.q.to(target, non_blocking=non_blocking),
            self.s.to(target, non_blocking=non_blocking),
            self.cols,
            self.block,
            rows=self.rows,
            layout=self.layout,
        )

    @staticmethod
    def native_tensor_fp8_available(device: torch.device | str) -> bool:
        """Whether vendor tensor-scaled E4M3 GEMM is available.

        The decision is capability based and shared by every architecture.
        NVIDIA Ada/Hopper/Blackwell use the CUDA 13 scaled-MM path; ROCm and
        older NVIDIA devices keep their existing compact/BF16 executor.
        """
        target = torch.device(device)
        if (
            target.type != "cuda"
            or torch.version.hip is not None
            or not hasattr(torch, "_scaled_mm")
            or not hasattr(torch, "float8_e4m3fn")
        ):
            return False
        try:
            major, minor = torch.cuda.get_device_capability(target)
        except (RuntimeError, TypeError, ValueError):
            return False
        return (int(major), int(minor)) >= (8, 9)

    def compile_gpu_tensor_fp8(self):
        """Replace block scales with one native Tensor Core FP8 image.

        The source E4M3 bytes are re-normalized in bounded row chunks and the
        returned object owns only one byte per weight plus a scalar scale.
        No BF16 copy is retained.  This is a generic execution-image compile,
        not a model conversion and it never changes checkpoint files.
        """
        if self.layout == "tensor-fp8":
            return self
        if (
            self.layout != "row-major"
            or not self.q.is_cuda
            or self.block != 128
            or not self.native_tensor_fp8_available(self.q.device)
        ):
            return self
        weight_scale = self.s.amax().clamp_min(1.0e-12).reshape(1)
        compiled = torch.empty_like(self.q)
        # Bound the temporary BF16 image to roughly 64 MiB.  The source and
        # destination stay compact and are never simultaneously expanded in
        # their entirety.
        chunk_rows = max(
            self.block,
            ((64 * 2**20) // max(self.cols * 2, 1) // self.block)
            * self.block,
        )
        chunk_rows = min(self.rows, chunk_rows)
        for start in range(0, self.rows, chunk_rows):
            stop = min(self.rows, start + chunk_rows)
            first_block = start // self.block
            last_block = (stop + self.block - 1) // self.block
            row_scales = self.s[first_block:last_block].repeat_interleave(
                self.block, dim=0
            )[: stop - start]
            scales = row_scales.repeat_interleave(
                self.block, dim=1
            )[:, : self.cols]
            normalized = (
                self.q[start:stop]
                .view(torch.float8_e4m3fn)
                .to(torch.bfloat16)
                * (scales / weight_scale).to(torch.bfloat16)
            )
            compiled[start:stop].copy_(
                normalized.clamp(-448.0, 448.0)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
        return BlockFP8Weight(
            compiled,
            weight_scale,
            self.cols,
            self.block,
            rows=self.rows,
            layout="tensor-fp8",
        )

    def _tensor_fp8_matmul(
        self,
        x: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.layout != "tensor-fp8" or not x.is_cuda:
            raise ValueError("native tensor FP8 requires a CUDA execution image")
        from .fusedext import dense_fp8_quantize_rows_fused

        source = x.to(torch.bfloat16).contiguous()
        if source.shape[0] == 1:
            quantized = self._fp8_decode_input
            activation_scale = self._fp8_decode_scale
        else:
            quantized = torch.empty_like(
                source, dtype=torch.float8_e4m3fn
            )
            activation_scale = torch.empty(
                (1, 1), dtype=torch.float32, device=source.device
            )
        if quantized is None or activation_scale is None:
            raise RuntimeError("native tensor FP8 workspace is unavailable")
        fused = dense_fp8_quantize_rows_fused(
            source, quantized, activation_scale
        )
        if fused is None:
            raise RuntimeError("native tensor FP8 activation kernel unavailable")
        return self._tensor_fp8_matmul_prequantized(
            fused,
            activation_scale,
            output=output,
        )

    def _tensor_fp8_matmul_prequantized(
        self,
        quantized: torch.Tensor,
        activation_scale: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Consume one shared FP8 activation image without requantizing it."""

        if self.layout != "tensor-fp8" or not quantized.is_cuda:
            raise ValueError("native tensor FP8 requires a CUDA execution image")
        result = torch._scaled_mm(
            quantized,
            self.q.view(torch.float8_e4m3fn).t(),
            scale_a=activation_scale,
            scale_b=self.s.reshape(1, 1),
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        if output is not None:
            output.copy_(result)
            return output
        return result

    def to_row_major(self):
        """Restore compact row-major bytes without dequantizing weights."""
        if self.layout in ("row-major", "tensor-fp8"):
            return self
        if self.layout == "q4_0":
            return self
        row_blocks = int(self.q.shape[0])
        col_blocks = int(self.q.shape[2])
        raw = (
            self.q.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(row_blocks * self.block, col_blocks * self.block)
            [: self.rows, : self.cols]
            .contiguous()
        )
        return BlockFP8Weight(
            raw,
            self.s,
            self.cols,
            self.block,
            rows=self.rows,
            layout="row-major",
        )

    def to_block_major(self):
        """Replace row-major bytes with the common CPU 32x128 tile layout.

        The returned weight has the same logical values and compact byte
        width.  It is a CPU decode layout, not a model-specific format and
        never materializes BF16/F32 weights.
        """
        if self.layout == "block-major32":
            return self
        if self.q.is_cuda or self.block != 128:
            return self
        from .cpuext import block_fp8_to_block_major_cpu

        packed = block_fp8_to_block_major_cpu(self.q, self.block)
        if packed is None:
            return self
        return BlockFP8Weight(
            packed,
            self.s,
            self.cols,
            self.block,
            rows=self.rows,
            layout="block-major32",
        )

    def optimize_cpu_layout(
        self,
        *,
        minimum_speedup: float = 1.03,
        input_dtype: torch.dtype | None = None,
    ):
        """Autotune the common compact CPU layout once per logical shape.

        The first weight of a shape is evaluated in both exact layouts using
        the public native GEMV.  Later layers reuse the decision, so a model
        never pays a per-layer benchmark and small/square projections are not
        forced into a layout that only benefits wide or tall matrices.
        """
        compile_mode = os.environ.get("CCCP_CPU_COMPILE", "off").lower()
        if (
            compile_mode == "q4"
            and os.environ.get("CCCP_CPU_BLOCK_FP8_Q4", "1").strip().lower()
            not in ("0", "false", "off", "none")
        ):
            return self.compile_cpu_q4_0()
        mode = os.environ.get("CCCP_CPU_BLOCK_MAJOR", "auto").lower()
        if mode in ("0", "false", "off", "none"):
            return self
        if mode in ("force", "always"):
            return self.to_block_major()
        if input_dtype is None:
            input_dtype = (
                torch.bfloat16
                if os.environ.get("CCCP_COMPUTE_DTYPE", "bf16").lower()
                in ("bf16", "bfloat16")
                else torch.float32
            )
        if input_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                "block-FP8 CPU layout tuning only supports FP32/BF16 input"
            )
        key = (
            self.rows,
            self.cols,
            torch.get_num_threads(),
            input_dtype,
        )
        decision = _BLOCK_FP8_LAYOUT_DECISIONS.get(key)
        if decision is not None and not decision[0]:
            return self
        blocked = self.to_block_major()
        if blocked is self:
            return blocked
        if decision is not None:
            return blocked
        from .cpuext import block_fp8_gemv_cpu

        if decision is None:
            value = torch.randn(1, self.cols, dtype=input_dtype)
            row_output = torch.empty(self.rows, dtype=input_dtype)
            block_output = torch.empty_like(row_output)

            def measure(weight, output):
                block_fp8_gemv_cpu(
                    value,
                    weight.q,
                    weight.s,
                    weight.cols,
                    weight.block,
                    output,
                    rows=weight.rows,
                )
                samples = []
                for _ in range(5):
                    started = time.perf_counter()
                    block_fp8_gemv_cpu(
                        value,
                        weight.q,
                        weight.s,
                        weight.cols,
                        weight.block,
                        output,
                        rows=weight.rows,
                    )
                    samples.append(time.perf_counter() - started)
                return statistics.median(samples)

            # Alternate order once so the decision does not simply select
            # whichever payload happened to be in LLC last.
            row_seconds = measure(self, row_output)
            block_seconds = measure(blocked, block_output)
            block_seconds = min(
                block_seconds,
                measure(blocked, block_output),
            )
            row_seconds = min(
                row_seconds,
                measure(self, row_output),
            )
            use_blocked = (
                row_seconds / max(block_seconds, 1.0e-12)
                >= float(minimum_speedup)
            )
            decision = (use_blocked, row_seconds, block_seconds)
            _BLOCK_FP8_LAYOUT_DECISIONS[key] = decision
        return blocked if decision[0] else self

    def compile_cpu_q4_0(self):
        """Create a process-local linear Q4 image without touching model files."""
        if self.layout == "q4_0":
            return self
        if self.layout != "row-major" or self.q.is_cuda or self.cols % 32:
            return self
        from .cpuext import block_fp8_compile_q4_0_cpu

        compiled = block_fp8_compile_q4_0_cpu(
            self.q, self.s, self.rows, self.cols, self.block
        )
        if compiled is None:
            return self
        return BlockFP8Weight(
            compiled,
            torch.empty(0, dtype=torch.float32),
            self.cols,
            self.block,
            rows=self.rows,
            layout="q4_0",
        )

    @staticmethod
    def cpu_layout_decisions() -> dict:
        return {
            f"{rows}x{cols}@{threads}/{str(input_dtype).removeprefix('torch.')}": {
                "block_major32": use_blocked,
                "row_major_ms": row_seconds * 1000.0,
                "block_major32_ms": block_seconds * 1000.0,
                "speedup": row_seconds / max(block_seconds, 1.0e-12),
            }
            for (rows, cols, threads, input_dtype), (
                use_blocked,
                row_seconds,
                block_seconds,
            ) in _BLOCK_FP8_LAYOUT_DECISIONS.items()
        }

    def row_slice(self, start: int, stop: int):
        """Return an aligned compact output-row slice.

        Block-FP8 scales restart every ``block`` rows.  Non-aligned starts
        cannot be represented by the current public CUDA kernel and are
        therefore materialized as a small BF16 shard by TP helpers.
        """
        if start < 0 or stop < start or stop > self.shape[0]:
            raise IndexError((start, stop))
        if start % self.block and self.layout != "tensor-fp8":
            raise ValueError("BlockFP8 row slice start must be block-aligned")
        scale_start = start // self.block
        scale_stop = (stop + self.block - 1) // self.block
        if self.layout == "q4_0":
            stride = (self.cols // 32) * 18
            return BlockFP8Weight(
                self.q[start * stride : stop * stride].clone(),
                self.s,
                self.cols,
                self.block,
                rows=stop - start,
                layout="q4_0",
            )
        if self.layout == "tensor-fp8":
            return BlockFP8Weight(
                self.q[start:stop].clone(
                    memory_format=torch.contiguous_format
                ),
                self.s.clone(memory_format=torch.contiguous_format),
                self.cols,
                self.block,
                rows=stop - start,
                layout="tensor-fp8",
            )
        q = (
            self.q[start:stop].clone(memory_format=torch.contiguous_format)
            if self.layout == "row-major"
            else self.q[
                start // self.block : (stop + self.block - 1) // self.block
            ].clone(memory_format=torch.contiguous_format)
        )
        return BlockFP8Weight(
            q,
            self.s[scale_start:scale_stop].clone(
                memory_format=torch.contiguous_format
            ),
            self.cols,
            self.block,
            rows=stop - start,
            layout=self.layout,
        )

    def row_view(self, start: int, stop: int):
        """Return an aligned zero-copy compact output-row view.

        This is for grouped linear operators which consume the result
        immediately while the parent weight remains alive.  Unlike
        :meth:`row_slice`, it neither duplicates FP8 payload bytes nor
        materializes a floating-point matrix.
        """
        if start < 0 or stop < start or stop > self.shape[0]:
            raise IndexError((start, stop))
        if start % self.block and self.layout != "tensor-fp8":
            raise ValueError("BlockFP8 row view start must be block-aligned")
        scale_start = start // self.block
        scale_stop = (stop + self.block - 1) // self.block
        if self.layout == "q4_0":
            stride = (self.cols // 32) * 18
            return BlockFP8Weight(
                self.q[start * stride : stop * stride],
                self.s,
                self.cols,
                self.block,
                rows=stop - start,
                layout="q4_0",
            )
        if self.layout == "tensor-fp8":
            return BlockFP8Weight(
                self.q[start:stop],
                self.s,
                self.cols,
                self.block,
                rows=stop - start,
                layout="tensor-fp8",
            )
        q = (
            self.q[start:stop]
            if self.layout == "row-major"
            else self.q[
                start // self.block : (stop + self.block - 1) // self.block
            ]
        )
        return BlockFP8Weight(
            q,
            self.s[scale_start:scale_stop],
            self.cols,
            self.block,
            rows=stop - start,
            layout=self.layout,
        )

    def column_slice(self, start: int, stop: int):
        """Return an aligned compact input-column slice."""
        if start < 0 or stop < start or stop > self.cols:
            raise IndexError((start, stop))
        if self.layout == "q4_0":
            if start % 32 or stop % 32:
                raise ValueError("q4_0 column slices must be block32 aligned")
            blocks = self.cols // 32
            image = self.q.view(self.rows, blocks, 18)
            sliced = image[:, start // 32 : stop // 32].contiguous().view(-1)
            return BlockFP8Weight(
                sliced,
                self.s,
                stop - start,
                self.block,
                rows=self.rows,
                layout="q4_0",
            )
        if self.layout == "tensor-fp8":
            return BlockFP8Weight(
                self.q[:, start:stop].clone(
                    memory_format=torch.contiguous_format
                ),
                self.s.clone(memory_format=torch.contiguous_format),
                stop - start,
                self.block,
                rows=self.rows,
                layout="tensor-fp8",
            )
        if start % self.block:
            raise ValueError(
                "BlockFP8 column slice start must be block-aligned"
            )
        scale_start = start // self.block
        scale_stop = (stop + self.block - 1) // self.block
        q = (
            self.q[:, start:stop].clone(memory_format=torch.contiguous_format)
            if self.layout == "row-major"
            else self.q[
                :, :, scale_start:scale_stop
            ].clone(memory_format=torch.contiguous_format)
        )
        return BlockFP8Weight(
            q,
            self.s[:, scale_start:scale_stop].clone(
                memory_format=torch.contiguous_format
            ),
            stop - start,
            self.block,
            rows=self.rows,
            layout=self.layout,
        )

    def dequant_rows(
        self,
        r0: int,
        r1: int,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        # In block-major layouts q.shape[0] is the number of physical row
        # tiles, not the logical matrix row count.  Bounds must therefore be
        # checked against ``rows``; using q.shape[0] rejected valid batched
        # prefill projections as soon as they requested a full logical row
        # range (for example 0..2048).
        if r0 < 0 or r1 < r0 or r1 > self.rows:
            raise IndexError((r0, r1))
        if dtype is None:
            dtype = (
                compute_dtype(self.q.device)
                if self.q.is_cuda
                else torch.float32
            )
        if self.layout == "q4_0":
            blocks = self.cols // 32
            image = self.q.view(self.rows, blocks, 18)[r0:r1]
            scale = (
                image[..., :2]
                .contiguous()
                .view(torch.float16)
                .float()
                .reshape(r1 - r0, blocks, 1)
            )
            packed = image[..., 2:]
            low = (packed & 0x0F).to(torch.int16) - 8
            high = (packed >> 4).to(torch.int16) - 8
            values = torch.cat((low, high), dim=-1).float()
            values.mul_(scale)
            return values.reshape(r1 - r0, self.cols).to(dtype or torch.float32)
        if self.layout == "tensor-fp8":
            return (
                self.q[r0:r1]
                .view(torch.float8_e4m3fn)
                .to(dtype)
                * self.s.reshape(1, 1).to(dtype)
            )
        first_block = r0 // self.block
        last_block = (r1 + self.block - 1) // self.block
        scale_rows = self.s[first_block:last_block].repeat_interleave(
            self.block,
            dim=0,
        )
        offset = r0 - first_block * self.block
        scale_rows = scale_rows[offset : offset + (r1 - r0)]
        scales = scale_rows.repeat_interleave(
            self.block,
            dim=1,
        )[:, : self.cols]
        if self.layout == "row-major":
            raw = self.q[r0:r1]
        else:
            # Explicit materialization is reserved for small compatibility
            # factors.  Decode only the requested rows; resident projection
            # matrices stay in the compact block-major layout.
            row_blocks = []
            for row in range(r0, r1):
                block_row = row // self.block
                local = row % self.block
                chunk = local // 32
                within = local % 32
                row_blocks.append(
                    self.q[block_row, chunk, :, within, :]
                    .reshape(-1)[: self.cols]
                )
            raw = torch.stack(row_blocks, dim=0)
        values = raw.view(torch.float8_e4m3fn).to(dtype)
        return values * scales.to(dtype)

    def matmul_T(
        self,
        x: torch.Tensor,
        chunk: int | None = None,
    ) -> torch.Tensor:
        rows = self.rows
        if self.layout == "tensor-fp8" and x.is_cuda:
            return self._tensor_fp8_matmul(x)
        if self.layout == "q4_0" and not x.is_cuda:
            from .cpuext import q4_0_dequant_cpu, q4_0_gemv_cpu

            values = x.float().contiguous()
            if values.shape[0] > 1:
                # Q4 is the low-latency single-token execution image.  Long
                # prompt prefill must not invoke that GEMV once per token:
                # expand the current projection once, consume it with a
                # compiled multi-row GEMM, then immediately release it.
                dense = q4_0_dequant_cpu(
                    self.q,
                    rows,
                    self.cols,
                )
                if dense is not None:
                    result = torch.mm(values, dense.t())
                    del dense
                    return result
            output = torch.empty(values.shape[0], rows, dtype=torch.float32)
            for token in range(values.shape[0]):
                result = q4_0_gemv_cpu(
                    values[token : token + 1],
                    self.q,
                    rows,
                    self.cols,
                    output[token : token + 1],
                )
                if result is None:
                    break
            else:
                return output
        dtype = (
            compute_dtype(x.device)
            if x.is_cuda
            else torch.float32
        )
        if chunk is None:
            element_size = 2 if dtype != torch.float32 else 4
            chunk = max(
                self.block,
                min(
                    rows,
                    (64 * 2**20) // max(self.cols * element_size, 1),
                ),
            )
            chunk = max(
                self.block,
                (chunk // self.block) * self.block,
            )
        x_compute = x.to(dtype)
        out = torch.empty(
            x.shape[0],
            rows,
            dtype=torch.float32,
            device=x.device,
        )
        for r0 in range(0, rows, chunk):
            r1 = min(r0 + chunk, rows)
            out[:, r0:r1] = (
                x_compute @ self.dequant_rows(r0, r1, dtype).t()
            ).float()
        return out

    def matmul_T_decode_fused(
        self,
        x: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.layout == "tensor-fp8" and x.is_cuda:
            return self._tensor_fp8_matmul(x, output=output)
        if (
            self.layout == "q4_0"
            and not x.is_cuda
            and x.shape == (1, self.cols)
            and x.dtype in (torch.float32, torch.bfloat16)
        ):
            from .cpuext import q4_0_gemv_cpu

            value = x.float().contiguous()
            target = output
            if target is None or target.dtype != torch.float32:
                target = self._decode_outputs.get(torch.float32)
                if target is None:
                    target = torch.empty((1, self.rows), dtype=torch.float32)
                    self._decode_outputs[torch.float32] = target
            fused = q4_0_gemv_cpu(
                value, self.q, self.rows, self.cols, target
            )
            if fused is not None:
                return fused
        if (
            x.dim() == 2
            and 1 <= x.shape[0] <= 16
            and x.shape[1] == self.cols
            and x.dtype in (torch.float32, torch.bfloat16)
            and self.q.dtype == torch.uint8
            and self.s.dtype == torch.float32
            and self.block == 128
        ):
            fn = (
                _block_fp8_gemv_fused()
                if x.shape[0] == 1
                else None
            )
            if x.shape[0] > 1:
                try:
                    from .ops import block_scaled_gemm

                    fn = block_scaled_gemm
                except ImportError:
                    fn = None
            if fn is not None:
                kernel_x = (
                    x.to(torch.bfloat16)
                    if (
                        not x.is_cuda
                        and x.dtype == torch.float32
                        and os.environ.get(
                            "CCCP_CPU_BLOCK_FP8_BF16", "0"
                        ) not in ("", "0", "false", "off")
                    )
                    else x
                )
                target = output
                if (
                    target is None
                    and not x.is_cuda
                    and x.shape[0] == 1
                    and not torch.is_grad_enabled()
                ):
                    target = self._decode_outputs.get(kernel_x.dtype)
                    if target is None:
                        target = torch.empty(
                            (1, self.rows),
                            dtype=kernel_x.dtype,
                        )
                        self._decode_outputs[kernel_x.dtype] = target
                fused = fn(
                    kernel_x,
                    self.q,
                    self.s,
                    block_size=self.block,
                    rows=self.rows,
                    cols=self.cols,
                    output=target,
                )
                if fused is not None:
                    return fused
        result = self.matmul_T(x)
        target = output
        if (
            target is None
            and not x.is_cuda
            and x.shape[0] == 1
            and not torch.is_grad_enabled()
        ):
            target = self._decode_outputs.get(result.dtype)
            if target is None:
                target = torch.empty_like(result)
                self._decode_outputs[result.dtype] = target
        if target is not None:
            target.copy_(result)
            return target
        return result

    def row(self, r: int) -> torch.Tensor:
        return self.dequant_rows(r, r + 1).squeeze(0)


class ProjectionGroup:
    """A logical row-concatenation which keeps each projection compact.

    Separate source tensors have independent 128-row FP8 scale origins.  A
    physical ``torch.cat`` would either duplicate them as BF16 or corrupt the
    scale layout at a non-aligned boundary.  This public wrapper selects one
    registered grouped GEMV when available and otherwise concatenates only
    the token-sized outputs.  Its CPU batch-one result is fixed storage and
    remains valid until the next decode call on the same group.
    """

    __slots__ = ("weights", "cols", "_grouped_meta", "_resident_cpu")

    def __init__(self, weights):
        values = tuple(weights)
        if not values:
            raise ValueError("ProjectionGroup requires at least one weight")
        cols = int(values[0].shape[1])
        if any(len(value.shape) != 2 or int(value.shape[1]) != cols
               for value in values):
            raise ValueError("ProjectionGroup column widths must match")
        self.weights = values
        self.cols = cols
        self._grouped_meta = {}
        self._resident_cpu = None

    @property
    def shape(self) -> torch.Size:
        return torch.Size(
            [sum(int(value.shape[0]) for value in self.weights), self.cols]
        )

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def nbytes(self) -> int:
        return sum(
            int(value.nbytes)
            if hasattr(value, "nbytes")
            else value.numel() * value.element_size()
            for value in self.weights
        )

    def to(self, device, non_blocking: bool = False) -> "ProjectionGroup":
        return ProjectionGroup(
            value.to(device, non_blocking=non_blocking)
            for value in self.weights
        )

    def _resident_executor(
        self,
        x: torch.Tensor,
    ):
        if (
            x.is_cuda
            or x.dim() != 2
            or x.shape != (1, self.cols)
            or x.dtype not in (torch.float32, torch.bfloat16)
            or not x.is_contiguous()
        ):
            return None
        if (
            x.dtype == torch.float32
            and all(
                isinstance(weight, BlockFP8Weight)
                and weight.layout != "q4_0"
                for weight in self.weights
            )
            and os.environ.get("CCCP_CPU_BLOCK_FP8_BF16", "0")
            in ("", "0", "false", "off")
        ):
            return None
        return self._resident_layer()

    def _resident_layer(self):
        """Return the format-generic fixed-address CPU projection executor."""
        executor = self._resident_cpu
        if executor is False:
            return None
        if executor is None:
            from .ops import create_resident_projection_layer

            executor = create_resident_projection_layer(self.weights)
            self._resident_cpu = executor if executor is not None else False
        if executor is None or executor is False:
            return None
        return executor

    def resident_forward_parts(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | None:
        """Run mixed compact projections in one fixed-address CPU team."""
        executor = self._resident_executor(x)
        if executor is None:
            return None
        return tuple(executor.forward(x))

    def matmul_T_grouped_rows_fused(
        self,
        x: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Apply weight ``i`` only to matching input row ``i``."""
        if any(
            isinstance(weight, BlockFP8Weight)
            and weight.layout == "tensor-fp8"
            for weight in self.weights
        ):
            return None
        if (
            x.dim() != 2
            or x.shape != (len(self.weights), self.cols)
            or x.dtype not in (torch.float32, torch.bfloat16)
            or not all(
                isinstance(weight, BlockFP8Weight)
                and weight.block == 128
                and weight.cols == self.cols
                and weight.q.device == x.device
                and weight.s.device == x.device
                and weight.q.is_contiguous()
                and weight.s.is_contiguous()
                for weight in self.weights
            )
        ):
            return None
        target = output
        if target is None:
            target = torch.empty(
                (1, int(self.shape[0])), dtype=x.dtype, device=x.device
            )
        if (
            not x.is_cuda
            and any(weight.layout == "q4_0" for weight in self.weights)
        ):
            executor = self._resident_layer()
            if executor is not None:
                combined = executor.forward_grouped(
                    x.contiguous(), target.dtype == torch.float32
                )
                target.copy_(combined)
                return target
        from .ops import block_scaled_grouped_rows_gemv

        kernel_x = (
            x.to(torch.bfloat16)
            if (
                not x.is_cuda
                and x.dtype == torch.float32
                and os.environ.get("CCCP_CPU_BLOCK_FP8_BF16", "0")
                not in ("", "0", "false", "off")
            )
            else x
        )

        total_rows = int(self.shape[0])
        start = 0
        output_offset = 0
        while start < len(self.weights):
            layout = self.weights[start].layout
            stop = start + 1
            while (
                stop < len(self.weights)
                and self.weights[stop].layout == layout
            ):
                stop += 1
            device_key = (
                "rows",
                x.device.type,
                int(x.device.index or 0),
                start,
                stop,
            )
            metadata = self._grouped_meta.get(device_key)
            group = self.weights[start:stop]
            if metadata is None:
                offsets = [0]
                for weight in group:
                    offsets.append(offsets[-1] + int(weight.shape[0]))
                metadata = (
                    torch.tensor(
                        [weight.q.data_ptr() for weight in group],
                        dtype=torch.int64,
                        device=x.device,
                    ),
                    torch.tensor(
                        [weight.s.data_ptr() for weight in group],
                        dtype=torch.int64,
                        device=x.device,
                    ),
                    torch.tensor(
                        offsets, dtype=torch.int32, device=x.device
                    ),
                    offsets[-1],
                )
                self._grouped_meta[device_key] = metadata
            fused = block_scaled_grouped_rows_gemv(
                kernel_x[start:stop],
                metadata[0],
                metadata[1],
                metadata[2],
                total_rows=metadata[3],
                cols=self.cols,
                block_size=128,
                block_major=(layout == "block-major32"),
                output=target.narrow(-1, output_offset, metadata[3]),
            )
            if fused is None:
                return None
            output_offset += metadata[3]
            start = stop
        return target

    def matmul_T_decode_fused(
        self,
        x: torch.Tensor,
        output: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if (
            x.is_cuda
            and all(
                isinstance(weight, BlockFP8Weight)
                and weight.layout == "tensor-fp8"
                for weight in self.weights
            )
        ):
            from .fusedext import dense_fp8_quantize_rows_fused

            source = x.to(torch.bfloat16).contiguous()
            first = self.weights[0]
            if source.shape[0] == 1:
                quantized = first._fp8_decode_input
                activation_scale = first._fp8_decode_scale
            else:
                quantized = torch.empty_like(
                    source, dtype=torch.float8_e4m3fn
                )
                activation_scale = torch.empty(
                    (1, 1), dtype=torch.float32, device=source.device
                )
            if quantized is None or activation_scale is None:
                raise RuntimeError(
                    "native tensor FP8 workspace is unavailable"
                )
            fused = dense_fp8_quantize_rows_fused(
                source, quantized, activation_scale
            )
            if fused is None:
                raise RuntimeError(
                    "native tensor FP8 activation kernel unavailable"
                )
            parts = tuple(
                weight._tensor_fp8_matmul_prequantized(
                    fused,
                    activation_scale,
                )
                for weight in self.weights
            )
            result = torch.cat(parts, dim=-1)
            if output is not None:
                output.copy_(result)
                return output
            return result
        if (
            not torch.is_grad_enabled()
            and not x.is_cuda
            and x.dim() == 2
            and x.shape == (1, self.cols)
            and x.dtype in (torch.float32, torch.bfloat16)
        ):
            executor = self._resident_executor(x)
            if executor is not None:
                all_block_fp8 = all(
                    isinstance(weight, BlockFP8Weight)
                    for weight in self.weights
                )
                all_bf16 = all(
                    isinstance(weight, torch.Tensor)
                    and weight.dtype == torch.bfloat16
                    for weight in self.weights
                )
                target_dtype = (
                    output.dtype
                    if output is not None
                    else output_dtype
                    if output_dtype is not None
                    else x.dtype
                    if all_block_fp8
                    else torch.bfloat16
                    if all_bf16
                    else torch.float32
                )
                combined = executor.forward_combined(
                    x,
                    target_dtype == torch.float32,
                )
                if output is not None:
                    output.copy_(combined)
                    return output
                return combined
        if (
            len(self.weights) >= 1
            and not x.is_cuda
            and x.dim() == 2
            and x.shape == (1, self.cols)
            and x.dtype in (torch.float32, torch.bfloat16)
            and all(
                isinstance(weight, torch.Tensor)
                and not weight.is_cuda
                and weight.dtype == torch.bfloat16
                and weight.dim() == 2
                and int(weight.shape[1]) == self.cols
                and weight.is_contiguous()
                for weight in self.weights
            )
        ):
            from .ops import dense_grouped_gemv

            device_key = ("dense-bf16", "cpu", 0, 0, len(self.weights))
            metadata = self._grouped_meta.get(device_key)
            if metadata is None:
                offsets = [0]
                for weight in self.weights:
                    offsets.append(offsets[-1] + int(weight.shape[0]))
                metadata = (
                    torch.tensor(
                        [weight.data_ptr() for weight in self.weights],
                        dtype=torch.int64,
                    ),
                    torch.tensor(offsets, dtype=torch.int32),
                    offsets[-1],
                )
                self._grouped_meta[device_key] = metadata
            target = output
            if target is None:
                target = torch.empty(
                    (1, metadata[2]), dtype=torch.bfloat16
                )
            fused = dense_grouped_gemv(
                x,
                metadata[0],
                metadata[1],
                total_rows=metadata[2],
                cols=self.cols,
                output=target,
            )
            if fused is not None:
                return fused
        if (
            len(self.weights) > 1
            and x.dim() == 2
            and 1 <= x.shape[0] <= 16
            and x.shape[1] == self.cols
            and x.dtype in (torch.float32, torch.bfloat16)
            and all(
                isinstance(weight, BlockFP8Weight)
                and weight.block == 128
                and weight.cols == self.cols
                and weight.q.device == x.device
                and weight.s.device == x.device
                and weight.q.is_contiguous()
                and weight.s.is_contiguous()
                for weight in self.weights
            )
        ):
            if x.shape[0] == 1:
                fn = _block_fp8_grouped_gemv_fused()
            else:
                try:
                    from .ops import block_scaled_grouped_gemm

                    fn = block_scaled_grouped_gemm
                except ImportError:
                    fn = None
            if fn is not None:
                kernel_x = (
                    x.to(torch.bfloat16)
                    if (
                        not x.is_cuda
                        and x.dtype == torch.float32
                        and os.environ.get(
                            "CCCP_CPU_BLOCK_FP8_BF16", "0"
                        ) not in ("", "0", "false", "off")
                    )
                    else x
                )
                total_rows = int(self.shape[0])
                target = output
                if target is None:
                    target_dtype = (
                        torch.float32
                        if x.is_cuda and x.shape[0] == 1
                        else x.dtype
                    )
                    target = torch.empty(
                        (x.shape[0], total_rows),
                        dtype=target_dtype,
                        device=x.device,
                    )
                segments = []
                start = 0
                while start < len(self.weights):
                    layout = self.weights[start].layout
                    stop = start + 1
                    while (
                        stop < len(self.weights)
                        and self.weights[stop].layout == layout
                    ):
                        stop += 1
                    segments.append((start, stop, layout))
                    start = stop
                output_offset = 0
                completed = True
                for start, stop, layout in segments:
                    device_key = (
                        x.device.type,
                        int(x.device.index or 0),
                        start,
                        stop,
                    )
                    metadata = self._grouped_meta.get(device_key)
                    group = self.weights[start:stop]
                    if metadata is None:
                        rows = [int(weight.shape[0]) for weight in group]
                        offsets = [0]
                        for count in rows:
                            offsets.append(offsets[-1] + count)
                        metadata = (
                            torch.tensor(
                                [weight.q.data_ptr() for weight in group],
                                dtype=torch.int64,
                                device=x.device,
                            ),
                            torch.tensor(
                                [weight.s.data_ptr() for weight in group],
                                dtype=torch.int64,
                                device=x.device,
                            ),
                            torch.tensor(
                                offsets,
                                dtype=torch.int32,
                                device=x.device,
                            ),
                            offsets[-1],
                        )
                        self._grouped_meta[device_key] = metadata
                    segment_output = target.narrow(
                        -1,
                        output_offset,
                        metadata[3],
                    )
                    fused = fn(
                        kernel_x,
                        metadata[0],
                        metadata[1],
                        metadata[2],
                        total_rows=metadata[3],
                        cols=self.cols,
                        block_size=128,
                        block_major=(layout == "block-major32"),
                        output=segment_output,
                    )
                    if fused is None:
                        completed = False
                        break
                    output_offset += metadata[3]
                if completed:
                    return target
        outputs = []
        for weight in self.weights:
            if isinstance(weight, (BlockFP8Weight, Int4Weight)):
                outputs.append(weight.matmul_T_decode_fused(x))
            else:
                outputs.append(
                    torch.nn.functional.linear(
                        x.to(weight.dtype),
                        weight,
                    ).float()
                )
        result = torch.cat(outputs, dim=-1)
        if output is not None:
            output.copy_(result)
            return output
        return result


class VQWeight:
    """VQ 索引态权重：u8 索引 [R, B] + 码本 [K, dim]，LUT 矩阵乘。"""

    __slots__ = ("idx", "cb", "cols", "dim")

    def __init__(self, idx: torch.Tensor, cb: torch.Tensor, cols: int):
        self.idx = idx              # u8 [R, B]，B = cols // dim
        self.cb = cb.float()        # f32 [K, dim]
        self.cols = cols
        self.dim = cb.shape[1]

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.idx.shape[0], self.cols])

    @property
    def nbytes(self) -> int:
        return self.idx.numel() * self.idx.element_size()

    def to(self, device, non_blocking: bool = False) -> "VQWeight":
        """搬移到指定设备（GPU 推理路径，可异步上传）。"""
        return VQWeight(self.idx.to(device, non_blocking=non_blocking),
                        self.cb.to(device, non_blocking=non_blocking), self.cols)

    def dequant(self) -> torch.Tensor:
        """还原为 f32 [R, C]（小矩阵或对照测试用）。"""
        return self.cb[self.idx.reshape(-1).long()].reshape(self.idx.shape[0], self.cols)

    def matmul_T(self, x: torch.Tensor) -> torch.Tensor:
        """LUT 版 y = x @ W.T。x: [T, C] → [T, R] f32。

        s[t, b, c] = x 第 b 块与码字 c 的点积（[T, B, K]），随后按索引查表求和。
        逐 t 循环 gather（峰值 [R,B]）；大码本（k4096）按 token 分块计算 s，
        峰值 [Tc,B,K] f32 封顶 ~256MB（全量 [T,B,K] 在长 prefill 会爆显存）。
        GPU 上内积走精度策略层的半精度（fp16/bf16 张量核，fp32 累加），
        查表求和用 sum(dtype=f32) 保 f32 累加精度——量化噪声比半精度舍入大
        两个数量级，输出分布不受影响（dspark_check 逐字一致验收过）。
        """
        T = x.shape[0]
        R, B = self.idx.shape
        K = self.cb.shape[0]
        dt = compute_dtype(x.device)
        cb = cb_compute(self.cb, dt)
        idxl = self.idx.long()
        barange = torch.arange(B, device=x.device)
        out = torch.empty(T, R, dtype=torch.float32, device=x.device)
        # 分块大小：让 [Tc, B, K] f32 ≤ 256MB
        tchunk = max(1, min(T, (256 * 2**20) // (B * K * 4)))
        for t0 in range(0, T, tchunk):
            t1 = min(t0 + tchunk, T)
            xb = x[t0:t1].to(dt).view(t1 - t0, B, self.dim)
            s = xb @ cb.t()                        # [Tc, B, K]（半精度 GEMM）
            for t in range(t1 - t0):
                # g[r, b] = s[t, b, idx[r, b]]；对 [B, K] 用 (行b, 列idx) 高级索引
                g = s[t][barange.unsqueeze(0), idxl]   # [R, B]
                out[t0 + t] = g.sum(1, dtype=torch.float32)   # f32 累加
        return out

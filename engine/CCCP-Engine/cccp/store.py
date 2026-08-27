"""CCCP 模型仓库层：读取 CCCP 产出的 "cccp-1" 格式（纯文件 I/O，无 mmap）。

目录结构（GLM-5.2-cccp/）：
    cccp.json               清单（config + quant 元信息 + 文件映射）
    dense.safetensors       dense 权重：int4 对（name + name.qs）或 f32 小张量
    vq-codebooks.safetensors 可选专家级跨层码本池和U8分配表
    experts.L*.safetensors  每层专家：共享 cb.gu.{档}，或按连续专家分组的
                            cb.gu.{档}.g{组号} 码本（down 同理）+
                            e{N}.gu{档}(z)/e{N}.dn{档}(z) 索引（z = zlib 熵编码）

为什么不用 safetensors 的 mmap：Windows 下长进程累积大量映射（75 个专家文件 +
CUDA 显存预留）会间歇性触发 access violation（实测层 64 附近必现）。
这里自实现 safetensors 读取（8 字节头长 + JSON 头 + 原始字节），
普通文件读走页缓存，无映射累积问题，返回的张量全部自持内存。
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
import warnings
import zlib
from collections import OrderedDict

import torch

from .expert_slots import ExpertSignature, GpuExpertArenas
from .kernels import BlockFP8Weight, Int4Weight, VQWeight, cb_compute
from .ramcache import active_ram_file

# 专家磁盘加载调参（2026-07-20 实测调优，见 cccp/README 缓存×速度基准）：
#   CCCP_LOAD_WORKERS：并行加载线程数（NVMe 随机读吃队列深度；默认 12）
#   CCCP_READ_BUF_MB ：每文件句柄读缓冲（默认 2MB——大缓冲无益：get_bytes 是
#       5-9MB 整块读，Python 对大 read 直写目标缓冲；且线程局部句柄 × 75 层文件
#       会放大内存占用，16MB×900 句柄实测 OOM 崩溃）
_LOAD_WORKERS = int(os.environ.get("CCCP_LOAD_WORKERS", "12"))
_READ_BUF = int(os.environ.get("CCCP_READ_BUF_MB", "2")) * 1024 * 1024
_EXEC = None
_WINDOWS = os.name == "nt"
_ROCM = torch.version.hip is not None


def _from_readonly_buffer(
    buffer: bytes | bytearray | memoryview,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a read-only tensor view without PyTorch's misleading warning.

    CCCP never mutates these index tensors.  Copying several GiB solely to
    make the Python buffer writable would slow model loading and double peak
    host traffic, so retain the zero-copy view and suppress only this exact
    warning at its source.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable",
            category=UserWarning,
        )
        return torch.frombuffer(buffer, dtype=dtype)
_PF_EXEC = None
_SAFEFILE_THREAD = threading.local()


def _should_batch_h2d(copy_count: int) -> bool:
    """Use one native CUDA batch submission for each routed expert group.

    ``PackedHybridPool`` has already collected every missing expert in the
    current layer before reaching this function. Automatic mode enters the
    compiled extension once for that whole group instead of issuing one
    Python/PyTorch call per expert. Linux/TCC uses ``cudaMemcpyBatchAsync``;
    Windows/WDDM queues the same group as a tight C++ ``cudaMemcpyAsync`` loop
    because the native batch API has twice produced asynchronous illegal
    addresses on consumer drivers. Unavailable extensions or rejected layouts
    still use the Python asynchronous copy-stream fallback without changing
    model results.

    Auto mode on Windows additionally keeps large groups out of the compiled
    submission: batched Prefill chunks submit 100+ ten-MiB copies per group,
    and both recorded ``cudaErrorIllegalAddress`` incidents surfaced in that
    window while decode-sized groups (<= 8 copies) hold a clean multi-run
    record. Those groups still DMA on the same copy stream through the Python
    fallback, which costs milliseconds of submission CPU against a
    bandwidth-bound transfer. ``CCCP_H2D_BATCH_MAX_COPIES`` widens the
    envelope and ``CCCP_H2D_BATCH=1`` restores the previous behaviour.
    """
    # HIPified cudaMemcpyAsync rejects otherwise valid registered-host copies
    # on Windows ROCm (observed at the eighth item of a layer batch). Keep the
    # exact same copy stream and tail event, but submit its items through
    # PyTorch's native HIP copy implementation instead of the CUDA C++ shim.
    if _ROCM:
        return False
    mode = os.environ.get("CCCP_H2D_BATCH", "auto").strip().lower()
    if mode in {"0", "false", "off", "no"}:
        return False
    if mode in {"1", "true", "on", "yes"}:
        return copy_count > 1
    if _WINDOWS:
        try:
            maximum = int(
                os.environ.get("CCCP_H2D_BATCH_MAX_COPIES", "8")
            )
        except (TypeError, ValueError):
            maximum = 8
        if copy_count > max(1, maximum):
            return False
    minimum = int(os.environ.get("CCCP_H2D_BATCH_MIN_COPIES", "2"))
    return copy_count >= max(2, minimum)


def _prefer_direct_pinned_h2d() -> bool:
    """Submit page-locked expert storage directly to CUDA by default.

    The compact archive is registered in place by ``cudaHostRegister``.  A
    second CPU memcpy into a rotating bounce ring defeats that registration
    and makes the host core the transfer bottleneck.  Keep an explicit opt-out
    for drivers on which registration fails; pageable sources still fall back
    to the existing contiguous pinned ring automatically.
    """

    # cudaHostRegister succeeded on Windows ROCm, but PyTorch HIP then
    # rejected direct tensor.copy_ from that externally registered pointer.
    # AMD must use PyTorch-owned pinned staging; NVIDIA retains the measured
    # direct registered-host DMA path unchanged.
    if _ROCM:
        return False
    mode = os.environ.get("CCCP_WDDM_DIRECT_PIN", "auto").strip().lower()
    if mode in {"1", "true", "on", "yes"}:
        return True
    if mode in {"0", "false", "off", "no"}:
        return False
    return True


def _unpack_u12(packed: torch.Tensor, count: int) -> torch.Tensor:
    """把 CCCP 的双 12-bit/3-byte 索引恢复为 u16。

    产物仅在磁盘/RAM blob 中紧凑保存；进入 GPU 专家 arena 前只解包一次，
    因而不会给每次专家计算增加位操作。
    """
    raw = packed.view(torch.uint8).reshape(-1).to(torch.int32)
    if raw.numel() % 3:
        raise ValueError(f"u12 packed bytes 必须为 3 的倍数，实际 {raw.numel()}")
    tri = raw.reshape(-1, 3)
    out = torch.empty(tri.shape[0] * 2, dtype=torch.uint16)
    out[0::2] = (tri[:, 0] | ((tri[:, 1] & 0x0F) << 8)).to(torch.uint16)
    out[1::2] = ((tri[:, 1] >> 4) | (tri[:, 2] << 4)).to(torch.uint16)
    if count > out.numel():
        raise ValueError(f"u12 索引数量不足: need={count}, have={out.numel()}")
    return out[:count]


def _unpack_u14(packed: torch.Tensor, count: int) -> torch.Tensor:
    """把 CCCP 的四 14-bit/7-byte 索引恢复为 u16。"""
    raw = packed.view(torch.uint8).reshape(-1).to(torch.int64)
    if raw.numel() % 7:
        raise ValueError(f"u14 packed bytes 必须为 7 的倍数，实际 {raw.numel()}")
    group = raw.reshape(-1, 7)
    word = torch.zeros(group.shape[0], dtype=torch.int64)
    for byte in range(7):
        word |= group[:, byte] << (8 * byte)
    out = torch.empty(group.shape[0] * 4, dtype=torch.uint16)
    mask = (1 << 14) - 1
    for index in range(4):
        out[index::4] = ((word >> (14 * index)) & mask).to(torch.uint16)
    if count > out.numel():
        raise ValueError(f"u14 索引数量不足: need={count}, have={out.numel()}")
    return out[:count]


def _unpack_u10(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Restore four little-endian 10-bit indices from every five bytes."""
    raw = packed.view(torch.uint8).reshape(-1).to(torch.int64)
    if raw.numel() % 5:
        raise ValueError(
            f"u10 packed bytes must be a multiple of 5, got {raw.numel()}"
        )
    group = raw.reshape(-1, 5)
    word = torch.zeros(group.shape[0], dtype=torch.int64)
    for byte in range(5):
        word |= group[:, byte] << (8 * byte)
    out = torch.empty(group.shape[0] * 4, dtype=torch.uint16)
    for index in range(4):
        out[index::4] = (
            (word >> (10 * index)) & 0x3FF
        ).to(torch.uint16)
    if count > out.numel():
        raise ValueError(
            f"u10 index count is too small: need={count}, "
            f"have={out.numel()}"
        )
    return out[:count]


def _unpack_u9(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Restore consecutive little-endian 9-bit indices.

    Nine-bit archives pack eight indices into nine bytes.  The vectorized
    reference path is used by correctness tests and CPU fallbacks; resident
    inference keeps the payload packed and extracts indices inside the native
    CPU/CUDA kernels.
    """
    raw = packed.view(torch.uint8).reshape(-1).to(torch.int32)
    if raw.numel() * 8 != int(count) * 9:
        raise ValueError(
            "u9 packed bytes do not match the requested index count: "
            f"bytes={raw.numel()}, count={count}"
        )
    bit_offsets = torch.arange(count, dtype=torch.int64) * 9
    byte_offsets = torch.bitwise_right_shift(bit_offsets, 3)
    shifts = torch.bitwise_and(bit_offsets, 7).to(torch.int32)
    # Every valid final index ends at or before the final payload bit.  One
    # zero pad byte keeps the two-byte gather branch-free at the boundary.
    padded = torch.cat((raw, torch.zeros(1, dtype=torch.int32)))
    words = (
        padded[byte_offsets]
        | torch.bitwise_left_shift(padded[byte_offsets + 1], 8)
    )
    return torch.bitwise_and(
        torch.bitwise_right_shift(words, shifts),
        0x1FF,
    ).to(torch.uint16)


def _unpack_odd_width(
    packed: torch.Tensor,
    count: int,
    bits: int,
) -> torch.Tensor:
    """Reference unpacker for row-aligned p11/p13/p15 payloads."""
    if bits not in (11, 13, 15):
        raise ValueError(f"unsupported odd packed width {bits}")
    raw = packed.view(torch.uint8).reshape(-1).to(torch.int64)
    if raw.numel() * 8 != int(count) * bits:
        raise ValueError(
            f"u{bits} packed bytes do not match index count: "
            f"bytes={raw.numel()}, count={count}"
        )
    bit_offsets = torch.arange(count, dtype=torch.int64) * bits
    byte_offsets = torch.bitwise_right_shift(bit_offsets, 3)
    shifts = torch.bitwise_and(bit_offsets, 7)
    # At most 22 bits are needed. Two pad bytes keep the final gather valid;
    # production CPU/CUDA paths read only the bytes actually required.
    padded = torch.cat((raw, torch.zeros(2, dtype=torch.int64)))
    words = (
        padded[byte_offsets]
        | torch.bitwise_left_shift(padded[byte_offsets + 1], 8)
        | torch.bitwise_left_shift(padded[byte_offsets + 2], 16)
    )
    return torch.bitwise_and(
        torch.bitwise_right_shift(words, shifts),
        (1 << bits) - 1,
    ).to(torch.uint16)


def _stored_index_bits(num_bytes: int, count: int) -> int:
    """Infer standard CCCP index width from exact payload length."""
    total_bits = int(num_bytes) * 8
    if count > 0 and total_bits % int(count) == 0:
        bits = total_bits // int(count)
        if 8 <= bits <= 16:
            return bits
    raise ValueError(
        f"cannot infer VQ index width: bytes={num_bytes}, count={count}"
    )


def _safe_arena_budget(
    *,
    requested_bytes: int,
    allocated_bytes: int,
    device_free_bytes: int,
    process_limit_bytes: int,
    reserve_bytes: int,
) -> int:
    """Cap a fixed expert arena before allocation.

    ``device_free_bytes`` already excludes current allocations, while the
    allocator's per-process limit does not.  Respect both ceilings so a large
    user request cannot first OOM and then fall back to an unnecessarily small
    half-sized cache.
    """
    device_room = max(0, int(device_free_bytes) - int(reserve_bytes))
    process_room = max(
        0,
        int(process_limit_bytes) - int(allocated_bytes) - int(reserve_bytes),
    )
    return max(
        0,
        min(int(requested_bytes), device_room, process_room),
    )


def _executor():
    """常驻加载线程池（避免每层每 token 重建线程的生成开销；线程局部句柄复用 fd）。"""
    global _EXEC
    if _EXEC is None:
        from concurrent.futures import ThreadPoolExecutor
        _EXEC = ThreadPoolExecutor(max_workers=_LOAD_WORKERS, thread_name_prefix="cccp-load")
    return _EXEC


def _pf_executor():
    """预取专用小池（4 线程）：与紧急加载池隔离——否则冷启动时 get_many 的
    紧急 miss 排在数百个预取任务后面（实测 1.57→0.07 tok/s 的饥饿事故）。"""
    global _PF_EXEC
    if _PF_EXEC is None:
        from concurrent.futures import ThreadPoolExecutor
        _PF_EXEC = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cccp-prefetch")
    return _PF_EXEC


_STAGE_EXEC = None


def _stage_executor():
    """后台 staging 专用单线程：预取的 RAM→VRAM 装槽+DMA 全部在此线程串行完成
    （单线程队列 = 槽位纪律天然有序，避开上次多线程并行装槽的错配事故），
    主线程推理不被 host memcpy 阻塞（真并行预加载）。"""
    global _STAGE_EXEC
    if _STAGE_EXEC is None:
        from concurrent.futures import ThreadPoolExecutor
        _STAGE_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cccp-stage")
    return _STAGE_EXEC


class PinnedStage:
    """专家上传的 pinned 分段暂存：host memcpy 进 pinned 槽 + 异步 DMA（真 ~10GB/s），
    替代页式上传（~3.7GB/s，驱动内部分段复制）。

    复用安全：每槽一个事件，复写前等该槽上次 DMA 完成；wait() 让计算流只等
    拷贝流尾部事件（前序 DMA 按流序天然先行）。索引张量为 u8（槽也按 u8 存取）。
    """

    def __init__(
        self,
        device,
        n_slots: int = 32,
        slot_mb: int = 12,
        *,
        measure: bool = False,
    ):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.slots = [torch.empty(slot_mb * 2**20, dtype=torch.uint8, pin_memory=True)
                      for _ in range(n_slots)]
        self.events = [torch.cuda.Event() for _ in range(n_slots)]
        self.last = torch.cuda.Event()
        # Direct uploads from permanently page-locked expert tensors bypass the
        # staging slots. Keep sources alive until their DMA event completes so
        # this remains safe for callers that do not otherwise retain the tensor.
        self._pinned_inflight: list[tuple[torch.cuda.Event, list[torch.Tensor]]] = []
        self._measure = bool(measure)
        self._timing: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        # Separate host preparation from device DMA.  The previous single
        # CUDA-event span enclosed Python/pageable-to-pinned staging gaps, so
        # the value labelled H2D could mostly be CPU work rather than PCIe.
        self.transfer_seconds = 0.0
        self.host_staging_seconds = 0.0
        self.direct_upload_bytes = 0
        self.staged_upload_bytes = 0
        self.upload_submissions = 0
        self.upload_copies = 0
        self.batch_submissions = 0
        self.batch_copies = 0
        self.batch_fallbacks = 0
        self.last_batch_submissions = 0
        self.last_batch_copies = 0
        self.i = 0
        backend = (
            "hip-python-async-copy-stream"
            if _ROCM
            else (
                "compiled-async-loop"
                if _WINDOWS
                else "cuda-memcpy-batch"
            )
        )
        print(
            "[cccp-dma] submission=compiled-layer-batch "
            f"backend={backend} scope=one-routed-layer max_copies=128 "
            "fallback=python-async-copy-stream",
            flush=True,
        )

    def upload_batch(self, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """成批上传：多对 (src CPU, dst GPU) 依次装槽 DMA，一次尾部事件。
        字节原生处理（u8/u16 索引通用）：槽按字节存取，dtype 由 dst 自带。
        装槽 host memcpy 串行（并行装槽曾致槽内容错配事故，见 EXPERIENCE §12）。

        跨流安全（NaN 事故的根因修复）：
          1) 批首 copy stream 等默认流当前队尾——被驱逐释放的显存块之后若被
             empty_like 复用为 DMA 目标，该块的最后读者（默认流 kernel）必定
             已在本次 wait 前提交，DMA 按序排在其后，杜绝「计算读旧块 vs
             DMA 写复用块」并发；
          2) 每个 dst record_stream(copy stream)——dst 被驱逐释放时 allocator
             会等 DMA 真正完成才把块给别人（wait_event 只排序不等于完成）。
        """
        if not pairs:
            return
        self._pinned_inflight = [
            item for item in self._pinned_inflight if not item[0].query()
        ]
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        views = []
        direct_sources = []
        pending: list[tuple[torch.Tensor, torch.Tensor]] = []
        pending_slots: set[int] = set()
        submissions = 0
        copies = 0

        def submit_pending() -> None:
            nonlocal submissions, copies
            if not pending:
                return
            copy_count = len(pending)
            timing_start = None
            timing_end = None
            if self._measure:
                timing_start = torch.cuda.Event(enable_timing=True)
                timing_end = torch.cuda.Event(enable_timing=True)
                timing_start.record(self.stream)
            use_batch = _should_batch_h2d(len(pending))
            submitted = False
            if use_batch:
                from .ops import packed_h2d_batch

                with torch.cuda.stream(self.stream):
                    submitted = packed_h2d_batch(pending)
            if not submitted:
                with torch.cuda.stream(self.stream):
                    for source, target in pending:
                        target.copy_(source, non_blocking=True)
                if use_batch:
                    self.batch_fallbacks += 1
            for _source, target in pending:
                target.record_stream(self.stream)
            for slot_index in pending_slots:
                self.events[slot_index].record(self.stream)
            if timing_end is not None:
                timing_end.record(self.stream)
                self._timing.append((timing_start, timing_end))
            self.upload_submissions += 1
            self.upload_copies += copy_count
            if submitted:
                submissions += 1
                copies += copy_count
            pending.clear()
            pending_slots.clear()

        direct_pinned = _prefer_direct_pinned_h2d()
        for src, dst in pairs:
            if src.is_pinned() and direct_pinned:
                self.direct_upload_bytes += int(src.nbytes)
                direct_sources.append(src)
                views.append((src, dst, src.view(torch.uint8).view(-1)))
                pending.append((src, dst))
                if len(pending) == 128:
                    submit_pending()
                continue
            if self.i in pending_slots:
                submit_pending()
            slot = self.slots[self.i]
            ev = self.events[self.i]
            ev.synchronize()
            nb = int(src.nbytes)
            assert nb <= slot.numel(), f"专家张量超槽位（{nb}B > {slot.numel()}B）"
            staging_started = time.perf_counter()
            slot[:nb].copy_(src.view(torch.uint8).view(-1))
            self.host_staging_seconds += time.perf_counter() - staging_started
            self.staged_upload_bytes += int(nb)
            slot_view = slot[:nb]
            pending.append(
                (slot_view, dst.view(torch.uint8).view(-1))
            )
            pending_slots.add(self.i)
            views.append((src, dst, slot_view))
            self.i = (self.i + 1) % len(self.slots)
        submit_pending()
        self.last_batch_submissions = submissions
        self.last_batch_copies = copies
        self.batch_submissions += submissions
        self.batch_copies += copies
        # Correctness waits use a dedicated tail event.  Per-submission timing
        # events above measure only work actually queued on the copy stream.
        self.last.record(self.stream)
        if direct_sources:
            done = torch.cuda.Event()
            done.record(self.stream)
            self._pinned_inflight.append((done, direct_sources))
        if os.environ.get("CCCP_STAGE_VERIFY", "0") != "0":
            # 诊断：校验每对 DMA 落盘内容与源一致（定位错字节/错地址）
            torch.cuda.synchronize()
            for src, dst, view in views:
                s = src.view(torch.uint8).view(-1)
                d = dst.view(torch.uint8).view(-1).cpu()
                if not torch.equal(d, s):
                    neq = (d != s)
                    n = int(neq.sum())
                    first = int(neq.nonzero()[0])
                    slot_eq = torch.equal(view.cpu(), s)
                    print(f"[stage-verify] 不一致: {n}/{s.numel()} 字节 "
                          f"首个@{first} ({first / s.numel():.1%}) 槽位正确={slot_eq}",
                          flush=True)

    def upload(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """src（CPU u8 张量）→ dst（GPU u8 张量）：host memcpy + 异步 DMA。"""
        self.upload_batch([(src, dst)])

    def wait(self) -> None:
        """让当前流等待拷贝流尾部（本批全部 DMA 完成）。"""
        torch.cuda.current_stream().wait_event(self.last)

    def collect_timing(self, *, synchronize: bool = False) -> float:
        """回收已完成的 copy-stream 批次，返回累计纯设备 DMA 秒数。

        主存整理时间由 ``host_staging_seconds`` 单独累计；这里的 CUDA 事件仅
        包住真正提交到 copy stream 的 H2D 操作，不再把 Python 提交间隙算进去。
        """
        if not self._measure:
            return 0.0
        if synchronize and self._timing:
            self._timing[-1][1].synchronize()
        completed = 0
        for _start, end in self._timing:
            if not end.query():
                break
            completed += 1
        for start, end in self._timing[:completed]:
            self.transfer_seconds += start.elapsed_time(end) / 1000.0
        if completed:
            del self._timing[:completed]
        return self.transfer_seconds


_DTYPES = {
    "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16, "I32": torch.int32,
    "I64": torch.int64, "F16": torch.float16, "F32": torch.float32,
    "F64": torch.float64, "BF16": torch.bfloat16, "BOOL": torch.bool,
    "U16": torch.uint16, "U32": torch.uint32, "U64": torch.uint64,
    # Safetensors has no raw-byte view mode.  These entries intentionally
    # expose FP8 payload bytes; the logical Dense decoder applies its scale.
    "F8_E4M3": torch.uint8, "F8_E5M2": torch.uint8,
    "F8_E8M0": torch.uint8,
}
_DTYPE_NBYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


class SafeFile:
    """极简 safetensors 读取器（纯文件 I/O，无 mmap；线程局部句柄支持并发读）。"""

    def __init__(self, path: str):
        self.path = path
        self._all_handles: set = set()
        self._handles_lock = threading.Lock()
        self._ram_blob = active_ram_file(path)
        if self._ram_blob is None:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n).decode("utf-8"))
        else:
            n = struct.unpack_from("<Q", self._ram_blob, 0)[0]
            header = json.loads(
                memoryview(self._ram_blob)[8:8 + n]
                .tobytes()
                .decode("utf-8")
            )
        self.meta = {k: v for k, v in header.items() if k != "__metadata__"}
        self.data_start = 8 + n

    def release_ram_blob(self) -> int:
        released = 0 if self._ram_blob is None else len(self._ram_blob)
        self._ram_blob = None
        return released

    @property
    def ram_blob_nbytes(self) -> int:
        return 0 if self._ram_blob is None else len(self._ram_blob)

    def close(self) -> None:
        with self._handles_lock:
            handles = tuple(self._all_handles)
            self._all_handles.clear()
        for handle in handles:
            handle.close()

    def _fh(self):
        """One current shard handle per reader thread.

        A handle per (thread, shard) leaks roughly ``workers * layers`` file
        descriptors during a full-model pass and exceeds the usual 1024-fd
        limit.  Workers process one layer at a time, so retaining only their
        current shard keeps seek independence and caps descriptors at the
        worker count.
        """
        slot = getattr(_SAFEFILE_THREAD, "slot", None)
        f = None if slot is None else slot[1]
        if slot is None or slot[0] is not self or f.closed:
            if slot is not None and not f.closed:
                previous = slot[0]
                with previous._handles_lock:
                    previous._all_handles.discard(f)
                f.close()
            f = open(self.path, "rb", buffering=_READ_BUF)
            with self._handles_lock:
                self._all_handles.add(f)
            _SAFEFILE_THREAD.slot = (self, f)
        return f

    def keys(self):
        return self.meta.keys()

    def get_bytes(self, name: str) -> bytes | memoryview:
        info = self.meta[name]
        start = self.data_start + info["data_offsets"][0]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        if self._ram_blob is not None:
            return memoryview(self._ram_blob)[start:start + size]
        f = self._fh()
        f.seek(start)
        return f.read(size)

    def get_tensor(self, name: str) -> torch.Tensor:
        """读张量：单次分配 + readinto 直填，省掉 bytes→bytearray 的整块拷贝
        （专家加载热路径：每 token ~5.9GB，省一次全量 memcpy）。"""
        info = self.meta[name]
        start = self.data_start + info["data_offsets"][0]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        if self._ram_blob is not None:
            buf = memoryview(self._ram_blob)[start:start + size]
        else:
            f = self._fh()
            f.seek(start)
            buf = bytearray(size)
            f.readinto(buf)
        t = _from_readonly_buffer(buf, dtype=_DTYPES[info["dtype"]])
        return t.reshape(info["shape"])


class SafeTensorCollection:
    """Lazy, read-only view over one or more safetensors shards.

    Legacy CCCP models keep non-expert tensors in one ``dense.safetensors``.
    Kimi K3 preserves source-native BF16 tensors in many ``dense/`` shards.
    This adapter gives both layouts the same ``keys``/``get_tensor`` contract
    and only opens a shard when one of its tensors is requested.
    """

    def __init__(
        self,
        root: str,
        files: list[str],
        *,
        audit_file: str | None = None,
    ):
        if not files:
            raise ValueError("at least one dense safetensors shard is required")
        self.root = root
        self.files = tuple(files)
        self._handles: dict[str, SafeFile] = {}
        self._locations: dict[str, str] = {}
        self._nbytes: dict[str, int] = {}
        self._logical_entries: dict[str, dict] = {}

        if audit_file is not None and os.path.exists(audit_file):
            with open(audit_file, "r", encoding="utf-8") as handle:
                audit = json.load(handle)
            aliases: dict[str, str] = {}
            for filename in self.files:
                normalized = filename.replace("\\", "/")
                aliases[normalized] = filename
                aliases[os.path.basename(normalized)] = filename
            for shard, item in audit.get("shards", {}).items():
                filename = aliases.get(str(shard).replace("\\", "/"))
                if filename is None:
                    continue
                for name in item.get("tensor_audit", {}):
                    self._locations[name] = filename
                    self._nbytes[name] = int(
                        item["tensor_audit"][name].get("bytes", 0)
                    )
            if audit.get("format") == "kimi-k3-dense-audit-v1":
                for name, entry in audit.get("entries", {}).items():
                    filename = aliases.get(
                        str(entry["shard"]).replace("\\", "/")
                    )
                    if filename is None:
                        raise ValueError(
                            f"dense audit shard is not in manifest: "
                            f"{entry['shard']}"
                        )
                    self._locations[name] = filename
                    self._nbytes[name] = int(entry["stored_bytes"])
                    self._logical_entries[name] = entry

        # Developer fixtures and legacy manifests may not have an audit.
        # Reading headers is cheap and does not touch tensor payloads.
        if not self._locations:
            for shard in self.files:
                safe = self._handle(shard)
                for name in safe.keys():
                    if name in self._locations:
                        raise ValueError(
                            f"duplicate dense tensor {name!r} in "
                            f"{self._locations[name]!r} and {shard!r}"
                        )
                    self._locations[name] = shard
                    info = safe.meta[name]
                    self._nbytes[name] = (
                        int(info["data_offsets"][1])
                        - int(info["data_offsets"][0])
                    )

    def _handle(self, shard: str) -> SafeFile:
        handle = self._handles.get(shard)
        if handle is None:
            handle = SafeFile(os.path.join(self.root, shard))
            self._handles[shard] = handle
        return handle

    def keys(self):
        return self._locations.keys()

    def get_tensor(self, name: str) -> torch.Tensor:
        handle = self._handle(self._locations[name])
        entry = self._logical_entries.get(name)
        if entry is None:
            return handle.get_tensor(name)
        kind = entry["storage_kind"]
        if kind == "source":
            return handle.get_tensor(entry.get("value_key") or name)
        if kind == "fp8":
            return self._decode_fp8(handle, entry)
        if kind == "d3-p12":
            return self._decode_d3(handle, entry)
        raise ValueError(f"unknown dense storage kind {kind!r}")

    def get_block_fp8(self, name: str) -> BlockFP8Weight | None:
        """Return audited block-FP8 without expanding it to BF16."""
        entry = self._logical_entries.get(name)
        if entry is None:
            # Source-exact archives retain an original E4M3 weight plus either
            # an E8M0 ``.scale`` tensor or the standard fine-grained FP8
            # ``.weight_scale_inv`` F32 tensor. Recognize both from headers so
            # the E4M3 payload remains byte-packed in RAM/VRAM.
            shard = self._locations.get(name)
            if shard is None or not name.endswith(".weight"):
                return None
            handle = self._handle(shard)
            info = handle.meta[name]
            scale_names = (
                name[: -len("weight")] + "scale",
                name + "_scale_inv",
            )
            scale_name = next(
                (
                    candidate
                    for candidate in scale_names
                    if self._locations.get(candidate) == shard
                    and handle.meta[candidate].get("dtype")
                    in ("F8_E8M0", "F32")
                ),
                None,
            )
            if info.get("dtype") != "F8_E4M3" or scale_name is None:
                return None
            raw = handle.get_tensor(name)
            scales = handle.get_tensor(scale_name)
            rows, columns = (int(value) for value in info["shape"])
        else:
            if entry["storage_kind"] != "fp8":
                return None
            handle = self._handle(self._locations[name])
            raw = handle.get_tensor(entry["value_key"])
            scales = handle.get_tensor(entry["scale_key"])
            rows, columns = (
                int(value) for value in entry["logical_shape"]
            )
        if scales.dtype == torch.uint8:
            scales = torch.pow(2.0, scales.float() - 127.0)
        if raw.shape != (rows, columns):
            raise ValueError(
                f"dense FP8 payload shape mismatch for {name!r}: "
                f"{tuple(raw.shape)} != {(rows, columns)}"
            )
        return BlockFP8Weight(raw, scales, columns, 128)

    @staticmethod
    def _logical_dtype(entry: dict) -> torch.dtype:
        dtype = str(entry["source_dtype"])
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported dense logical dtype {dtype!r}")
        return _DTYPES[dtype]

    @classmethod
    def _decode_fp8(cls, handle: SafeFile, entry: dict) -> torch.Tensor:
        raw = handle.get_tensor(entry["value_key"])
        scales = handle.get_tensor(entry["scale_key"])
        if scales.dtype == torch.uint8:
            scales = torch.pow(2.0, scales.float() - 127.0)
        rows, columns = (int(value) for value in entry["logical_shape"])
        block = 128
        dtype = cls._logical_dtype(entry)
        output = torch.empty((rows, columns), dtype=dtype)
        for row_start in range(0, rows, block):
            row_stop = min(row_start + block, rows)
            for column_start in range(0, columns, block):
                column_stop = min(column_start + block, columns)
                values = (
                    raw[row_start:row_stop, column_start:column_stop]
                    .view(torch.float8_e4m3fn)
                    .to(dtype)
                )
                output[
                    row_start:row_stop, column_start:column_stop
                ] = values * scales[
                    row_start // block,
                    column_start // block,
                ].to(dtype)
        return output

    @classmethod
    def _decode_d3(cls, handle: SafeFile, entry: dict) -> torch.Tensor:
        packed = handle.get_tensor(entry["index_key"]).to(torch.int32)
        if packed.ndim != 2 or packed.shape[1] % 3:
            raise ValueError(
                f"invalid row-aligned p12 dense tensor "
                f"{entry['logical_name']!r}"
            )
        rows, columns = (int(value) for value in entry["logical_shape"])
        groups = columns // 4
        tri = packed.reshape(rows, -1, 3)
        indices = torch.empty(
            (rows, tri.shape[1] * 2), dtype=torch.int64
        )
        indices[:, 0::2] = (
            tri[:, :, 0] | ((tri[:, :, 1] & 0x0F) << 8)
        )
        indices[:, 1::2] = (
            (tri[:, :, 1] >> 4) | (tri[:, :, 2] << 4)
        )
        indices = indices[:, :groups]
        codebook = handle.get_tensor(entry["codebook_key"]).float()
        return (
            codebook[indices.reshape(-1)]
            .reshape(rows, columns)
            .to(cls._logical_dtype(entry))
        )

    def nbytes(self, name: str) -> int:
        """Return payload bytes without reading the tensor body."""
        value = self._nbytes.get(name)
        if value:
            return value
        shard = self._locations[name]
        info = self._handle(shard).meta[name]
        return int(info["data_offsets"][1]) - int(info["data_offsets"][0])

    def resident_nbytes(self, name: str) -> int:
        """Bytes occupied after decoding one logical Dense tensor."""
        entry = self._logical_entries.get(name)
        if entry is None:
            return self.nbytes(name)
        if entry["storage_kind"] == "fp8":
            # The public BlockFP8 operator consumes the audited uint8 payload
            # and FP32 scales directly on CPU/CUDA.  Placement must therefore
            # budget the compact resident representation, not a hypothetical
            # BF16 expansion.
            return self.nbytes(name)
        elements = 1
        for value in entry["logical_shape"]:
            elements *= int(value)
        return elements * _DTYPE_NBYTES[str(entry["source_dtype"])]

    def ram_blob_paths(self) -> tuple[str, ...]:
        return tuple(
            handle.path
            for handle in self._handles.values()
            if handle.ram_blob_nbytes
        )

    def release_ram_blob(self) -> int:
        released = 0
        for handle in self._handles.values():
            released += handle.release_ram_blob()
        return released

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()


class Manifest:
    """cccp.json 解析。"""

    def __init__(self, root: str):
        with open(os.path.join(root, "cccp.json"), "r", encoding="utf-8") as f:
            m = json.load(f)
        assert m["format"] == "cccp-1", f"不支持的格式: {m['format']}"
        self.root = root
        self.config = m["config"]
        self.quant = m["quant"]
        self.model_family = str(m.get("model_family", ""))
        dense_files = m.get("dense_files")
        if dense_files is None:
            dense_files = [m["dense_file"]]
            self.dense_root = root
        else:
            dense_path = (m.get("nonexpert") or {}).get("path", "dense")
            normalized_path = str(dense_path).strip("/\\")
            prefixed = normalized_path and all(
                str(value).replace("\\", "/").startswith(
                    normalized_path.replace("\\", "/") + "/"
                )
                for value in dense_files
            )
            # Standard Kimi archives store root-relative entries such as
            # ``dense/model-00001...`` while older fixtures may list only the
            # basename and rely on ``nonexpert.path``.
            self.dense_root = (
                root
                if prefixed
                else os.path.join(root, normalized_path)
            )
        self.dense_files = [str(value) for value in dense_files]
        # Keep the legacy attribute for startup estimators and old callers.
        self.dense_file = self.dense_files[0]
        audit_name = m.get("dense_audit_file")
        self.dense_audit_file = (
            os.path.join(root, audit_name) if audit_name else None
        )
        routed = m.get("routed_experts") or {}
        self.routed_layers = int(routed.get("layers", 0) or 0)
        self.routed_experts_per_layer = int(
            routed.get("experts_per_layer", 0) or 0
        )
        self.no_expert_drop = bool(routed.get("no_expert_drop", False))
        routed_layers = routed.get("layer_files") or {}
        # Projection-VQ has two released manifest layouts.  Kimi archives
        # describe each layer under ``routed_experts.layer_files`` while the
        # compact DeepSeek-V4 archive keeps the same information in the
        # top-level ``expert_files``/``layer_audit`` maps and stores the
        # projection descriptions in ``quant.layouts``.  Normalize both here
        # so model and operator code never branch on a model-family name.
        quant_method = str(self.quant.get("method", "")).strip().lower()
        projection_method = (
            quant_method == "projection-vq"
            or quant_method.endswith("-projection-vq")
        )
        heterogeneous = (
            self.quant.get("heterogeneous_expert_tiering") or {}
        )
        heterogeneous_projection_vq = bool(
            projection_method
            and self.quant.get("layouts")
            and heterogeneous.get("precision_levels")
            and heterogeneous.get("layer_expert_levels")
        )
        layer_projection_layouts = (
            self.quant.get("layer_projection_layouts") or {}
        )
        expert_storage = str(
            self.quant.get("expert_storage", "")
        ).strip().lower()
        codebook_policy = self.quant.get("codebook_policy") or {}
        split_projection_vq = bool(
            not routed_layers
            and m.get("expert_files")
            and projection_method
            and expert_storage
            == "split-gate-up-down-private-codebook-v1"
            and layer_projection_layouts
            and self.quant.get("projection_layouts")
            and codebook_policy.get("scope")
            == "per-expert-per-projection"
            and not bool(codebook_policy.get("gate_up_combined", False))
        )
        combined_projection_vq = bool(
            not split_projection_vq
            and not routed_layers
            and m.get("expert_files")
            and projection_method
            and layer_projection_layouts
            and self.quant.get("projection_layouts")
        )
        flat_projection_vq = bool(
            (split_projection_vq or not combined_projection_vq)
            and not routed_layers
            and m.get("expert_files")
            and projection_method
            and (
                self.quant.get("projection_layouts")
                or heterogeneous_projection_vq
            )
        )
        self.projection_vq = bool(
            (
                routed_layers
                and (
                    self.quant.get("projection_layouts")
                    or heterogeneous_projection_vq
                )
            )
            or flat_projection_vq
        )
        self.packed_expert_vq = bool(
            self.projection_vq
            or (
                combined_projection_vq
                and (
                    self.quant.get("vq")
                    or self.quant.get("vq_projections")
                )
                and self.quant.get("layer_kinds")
            )
        )
        # Storage geometry can be either the current per-projection packed
        # format or the earlier combined Gate/Up codebook format.  Both are
        # codebook execution and must enter the same public routed-VQ runtime;
        # only its format-selected storage backend may differ.
        self.expert_codebook_vq = bool(
            self.projection_vq
            or self.quant.get("vq")
            or self.quant.get("vq_projections")
        )
        self.heterogeneous_projection_vq = heterogeneous_projection_vq
        self.combined_projection_vq = combined_projection_vq
        self.split_projection_vq = split_projection_vq
        self.projection_private_codebooks = bool(split_projection_vq)
        self.projection_names: tuple[str, ...] = ()
        self.projection_layout_by_layer: dict[int, dict[str, str]] = {}
        self.projection_layout_by_expert: dict[
            int, tuple[dict[str, str], ...]
        ] = {}
        self.projection_precision_levels: dict[
            str, dict[str, str]
        ] = {}
        self.projection_level_by_expert: dict[
            int, tuple[str, ...]
        ] = {}
        if self.projection_vq:
            if flat_projection_vq:
                layer_audit = m.get("layer_audit") or {}
                self.expert_files = {
                    int(layer): str(filename)
                    for layer, filename in m["expert_files"].items()
                }
                self.expert_audit_files = {
                    int(layer): str(filename)
                    for layer, filename in (
                        m.get("expert_audit_files") or {}
                    ).items()
                }
                self.expert_audit_files.update({
                    int(layer): str(item["audit_path"])
                    for layer, item in layer_audit.items()
                    if isinstance(item, dict) and item.get("audit_path")
                })
                self.routed_layers = len(self.expert_files)
                self.routed_experts_per_layer = int(
                    self.config.get("n_experts", 0)
                )
                self.no_expert_drop = bool(
                    self.quant.get("no_expert_drop", False)
                )
            else:
                self.expert_files = {
                    int(layer): str(item["path"])
                    for layer, item in routed_layers.items()
                }
                self.expert_audit_files = {
                    int(layer): str(item["audit_path"])
                    for layer, item in routed_layers.items()
                }

            if heterogeneous_projection_vq:
                layout_specs = self.quant.get("layouts") or {}
                self.projection_precision_levels = {
                    str(level): {
                        str(projection): str(layout_name)
                        for projection, layout_name in layouts.items()
                    }
                    for level, layouts in heterogeneous[
                        "precision_levels"
                    ].items()
                }
                self.projection_level_by_expert = {
                    int(layer): tuple(str(level) for level in levels)
                    for layer, levels in heterogeneous[
                        "layer_expert_levels"
                    ].items()
                }
            elif split_projection_vq:
                self.projection_layout_by_layer = {
                    int(layer): {
                        ("down" if str(projection) == "dn" else str(projection)):
                            str(layout_name)
                        for projection, layout_name in layouts.items()
                    }
                    for layer, layouts in layer_projection_layouts.items()
                }
                referenced_layouts = {
                    layout
                    for layouts in self.projection_layout_by_layer.values()
                    for layout in layouts.values()
                }
                layout_specs = {}
                for layout_name in referenced_layouts:
                    try:
                        dim_text, size_text = layout_name.split("-", 1)
                        if not dim_text.startswith("d") or not size_text.startswith("k"):
                            raise ValueError
                        dim = int(dim_text[1:])
                        size = int(size_text[1:])
                    except (TypeError, ValueError):
                        raise ValueError(
                            "split projection VQ layout names must use "
                            f"d<dim>-k<size>: {layout_name!r}"
                        ) from None
                    layout_specs[layout_name] = {
                        "dim": dim,
                        "codebook_size": size,
                    }
            elif combined_projection_vq:
                layout_specs = self.quant["projection_layouts"]
                self.projection_layout_by_layer = {
                    int(layer): {
                        ("down" if str(projection) == "dn" else str(projection)):
                            str(layout_name)
                        for projection, layout_name in layouts.items()
                    }
                    for layer, layouts in layer_projection_layouts.items()
                }
            elif flat_projection_vq:
                layout_specs = self.quant.get("layouts") or {}
                self.projection_layout_by_layer = {
                    int(layer): {
                        str(projection): str(layout_name)
                        for projection, layout_name in layouts.items()
                    }
                    for layer, layouts in self.quant[
                        "projection_layouts"
                    ].items()
                }
            else:
                layout_specs = self.quant["projection_layouts"]
                self.projection_layout_by_layer = {
                    int(layer): {
                        str(projection): str(layout_name)
                        for projection, layout_name in item[
                            "projection_layout"
                        ].items()
                    }
                    for layer, item in routed_layers.items()
                }
            self.vq_dims = {
                str(name): (
                    int(item["dim"]),
                    int(item.get("size", item.get("codebook_size", 0))),
                )
                for name, item in layout_specs.items()
            }
            self.projection_layout_specs = {
                str(name): dict(item)
                for name, item in layout_specs.items()
            }
            self.projection_codebook_group_sizes = {
                name: int(item["group_size"])
                for name, item in self.projection_layout_specs.items()
                if item.get("group_size") is not None
            }
            self.projection_codebook_group_counts = {
                name: int(item["groups"])
                for name, item in self.projection_layout_specs.items()
                if item.get("groups") is not None
            }
            if any(
                size <= 0
                for size in self.projection_codebook_group_sizes.values()
            ):
                raise ValueError(
                    "projection VQ codebook group_size must be positive"
                )
            self.index_packing = {
                str(name): str(packing)
                for name, packing in self.quant.get(
                    "index_packing", {}
                ).items()
            }
            self._validate_projection_layouts()
            if heterogeneous_projection_vq:
                self.projection_layout_by_expert = {
                    layer: tuple(
                        self.projection_precision_levels[level]
                        for level in levels
                    )
                    for layer, levels in self.projection_level_by_expert.items()
                }
        else:
            self.heterogeneous_projection_vq = False
            self.split_projection_vq = False
            self.projection_private_codebooks = False
            self.expert_files = {
                int(l): v for l, v in m["expert_files"].items()
            }
            self.expert_audit_files = {
                int(layer): value
                for layer, value in m.get(
                    "expert_audit_files", {}
                ).items()
            }
            self.projection_layout_specs = {}
            self.projection_codebook_group_sizes = {}
            self.projection_codebook_group_counts = {}
            self.index_packing = {}
            # ``quant.vq`` is only a legacy summary of the Gate/Up layout.
            # Current per-projection manifests may omit it because the exact
            # Gate/Up and Down shapes already live in ``vq_projections``.
            # Derive the summary strictly instead of crashing with KeyError;
            # the storage/runtime path remains the existing combined format.
            vq_summary = self.quant.get("vq") or {}
            if not vq_summary:
                projection_summaries = self.quant.get("vq_projections") or {}
                vq_summary = {}
                for kind, projections in projection_summaries.items():
                    if not isinstance(projections, dict):
                        raise ValueError(
                            f"VQ projection summary {kind!r} must be an object"
                        )
                    primary = projections.get("gu")
                    if primary is None:
                        raise ValueError(
                            f"VQ projection summary {kind!r} is missing gu"
                        )
                    try:
                        dim, size = (int(value) for value in primary)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"VQ projection summary {kind!r}.gu must be [dim, size]"
                        ) from None
                    if dim <= 0 or size <= 0:
                        raise ValueError(
                            f"VQ projection summary {kind!r}.gu must be positive"
                        )
                    for projection, shape in projections.items():
                        try:
                            projection_dim, projection_size = (
                                int(value) for value in shape
                            )
                        except (TypeError, ValueError):
                            raise ValueError(
                                f"VQ projection summary {kind!r}.{projection} "
                                "must be [dim, size]"
                            ) from None
                        if projection_dim != dim or projection_size <= 0:
                            raise ValueError(
                                f"VQ projection summary {kind!r}.{projection} "
                                f"is incompatible with dim={dim}"
                            )
                    vq_summary[str(kind)] = (dim, size)
            self.vq_dims = {
                str(kind): tuple(int(value) for value in shape)
                for kind, shape in vq_summary.items()
            }  # 档 -> (dim, k)
            referenced_kinds = {
                str(kind).rstrip("z")
                for kind in (self.quant.get("layer_kinds") or {}).values()
                if str(kind) != "drop"
            }
            missing_kinds = sorted(referenced_kinds - set(self.vq_dims))
            if missing_kinds:
                raise ValueError(
                    "layer_kinds references undefined VQ summaries: "
                    f"{missing_kinds}"
                )
        layout = m["quant"].get("vq_codebook_layout") or {}
        layout_format = layout.get("format")
        if layout_format not in (
            None,
            "cccp-vq-codebook-layout-v1",
            "expert-assigned-codebook-v1",
        ):
            raise ValueError(f"不支持的 VQ 码本布局: {layout_format}")
        if (
            layout_format == "cccp-vq-codebook-layout-v1"
            and layout.get("assignment") != "contiguous-expert-id"
        ):
            raise ValueError(
                f"不支持的 VQ 码本分配规则: {layout.get('assignment')}"
            )
        if (
            layout_format == "expert-assigned-codebook-v1"
            and layout.get("assignment")
            != "per-expert-per-projection"
        ):
            raise ValueError(
                f"不支持的 VQ 码本分配规则: {layout.get('assignment')}"
            )
        self.vq_codebook_layout = layout
        self.vq_codebook_layout_format = layout_format
        self.vq_codebook_group_sizes = {
            str(kind): int(size)
            for kind, size in layout.get("group_size", {}).items()
        }
        if any(size <= 0 for size in self.vq_codebook_group_sizes.values()):
            raise ValueError("VQ 码本 group_size 必须大于 0")
        self.vq_codebook_file = (
            m["quant"].get("vq_codebook_file")
            or layout.get("codebook_file")
        )
        self.int4_group = m["quant"].get("int4_group", 64)
        self.zlib = m["quant"].get("zlib", False)
        # 每层每专家档位串（'v'/'w'/'x'/'d'=drop），量化/repack 时写入；缺省 = 全保留
        self.tiers_per_layer = {
            int(l): s
            for l, s in m.get("tiers_per_layer", {}).items()
        }
        if self.projection_vq:
            self.tiers_per_layer.update(
                {
                    int(layer): str(item.get("tier_string", ""))
                    for layer, item in routed_layers.items()
                    if item.get("tier_string")
                }
            )
        self._audit_tiers: dict[int, str] = {}

    def _validate_projection_layouts(self) -> None:
        supported_projection_sets = (
            {"gate", "up", "down"},
            {"gu", "down"},
        )
        if self.heterogeneous_projection_vq:
            if not self.projection_precision_levels:
                raise ValueError(
                    "heterogeneous projection VQ has no precision_levels"
                )
            for level, layouts in self.projection_precision_levels.items():
                if set(layouts) not in supported_projection_sets:
                    raise ValueError(
                        f"precision level {level!r} must define either "
                        "gate/up/down or gu/down"
                    )
            expected = int(self.routed_experts_per_layer)
            for layer in self.expert_files:
                levels = self.projection_level_by_expert.get(layer)
                if levels is None:
                    raise ValueError(
                        f"L{layer} has no heterogeneous expert level map"
                    )
                if len(levels) != expected:
                    raise ValueError(
                        f"L{layer} expert level count {len(levels)} != "
                        f"n_experts {expected}"
                    )
                unknown = sorted(
                    set(levels) - set(self.projection_precision_levels)
                )
                if unknown:
                    raise ValueError(
                        f"L{layer} references unknown precision levels "
                        f"{unknown}"
                    )
            referenced = {
                layout
                for layouts in self.projection_precision_levels.values()
                for layout in layouts.values()
            }
            projection_sets = {
                frozenset(layouts)
                for layouts in self.projection_precision_levels.values()
            }
        else:
            if not self.projection_layout_by_layer:
                raise ValueError("projection VQ has no per-layer layout map")
            projection_sets = {
                frozenset(layouts)
                for layouts in self.projection_layout_by_layer.values()
            }
            referenced = {
                layout
                for layouts in self.projection_layout_by_layer.values()
                for layout in layouts.values()
            }
        if len(projection_sets) != 1:
            raise ValueError(
                "projection VQ entries must use one projection schema"
            )
        projection_set = set(next(iter(projection_sets)))
        if projection_set not in supported_projection_sets:
            raise ValueError(
                "projection VQ must define either gate/up/down or gu/down"
            )
        self.projection_names = (
            ("gate", "up", "down")
            if projection_set == {"gate", "up", "down"}
            else ("gu", "down")
        )
        missing = sorted(referenced - set(self.projection_layout_specs))
        if missing:
            raise ValueError(
                f"projection VQ references undefined layouts: {missing}"
            )
        for layout in referenced:
            dim, size = self.vq_dims[layout]
            if dim <= 0 or size <= 0 or size & (size - 1):
                raise ValueError(
                    f"projection layout {layout} must have a positive dim "
                    "and power-of-two codebook size"
                )
            expected_bits = size.bit_length() - 1
            packing = self.index_packing.get(layout)
            if packing is None:
                continue
            bits = int(
                packing.removeprefix("packed-u").removeprefix("u")
            )
            if bits < expected_bits:
                raise ValueError(
                    f"projection layout {layout} packing {packing} does not "
                    f"have enough bits for codebook size {size}"
                )

    def projection_layouts(
        self,
        layer: int,
        expert_id: int | None = None,
    ) -> dict[str, str]:
        """Resolve Gate/Up/Down layouts without model-name dispatch."""
        layer = int(layer)
        if not self.heterogeneous_projection_vq:
            return self.projection_layout_by_layer[layer]
        if expert_id is None:
            raise ValueError(
                f"L{layer} uses heterogeneous layouts; expert_id is required"
            )
        levels = self.projection_level_by_expert[layer]
        if expert_id < 0 or expert_id >= len(levels):
            raise IndexError(f"L{layer} expert_id {expert_id} is out of range")
        return self.projection_precision_levels[levels[expert_id]]

    def tier_string(self, layer: int) -> str | None:
        value = self.tiers_per_layer.get(layer)
        if value is not None:
            return value
        cached = self._audit_tiers.get(layer)
        if cached is not None:
            return cached
        audit_name = self.expert_audit_files.get(layer)
        if audit_name is None:
            return None
        with open(
            os.path.join(self.root, audit_name),
            "r",
            encoding="utf-8",
        ) as handle:
            audit = json.load(handle)
        tiers = str(audit.get("tier_string", ""))
        if not tiers:
            return None
        self._audit_tiers[layer] = tiers
        return tiers

    def projection_operator_capability(
        self,
        layer: int,
        expert_id: int | None = None,
    ) -> dict[str, tuple]:
        """Return the exact public operator key for one expert layout."""
        if not self.projection_vq:
            return {}
        formats = {
            "u8": "p8",
            "u16": "p16",
        }
        formats.update(
            {f"packed-u{bits}": f"p{bits}" for bits in range(8, 17)}
        )
        if self.heterogeneous_projection_vq and expert_id is None:
            used_levels = set(self.projection_level_by_expert[int(layer)])
            layout_names = sorted(
                {
                    layout
                    for level in used_levels
                    for layout in self.projection_precision_levels[
                        level
                    ].values()
                }
            )
        else:
            layouts = self.projection_layouts(layer, expert_id)
            layout_names = [
                layouts[projection]
                for projection in self.projection_names
            ]
        packed_formats = []
        code_dims = []
        codebook_sizes = []
        for layout in layout_names:
            packing = self.index_packing.get(layout)
            dim, size = self.vq_dims[layout]
            if packing is None:
                # A few early projection manifests described the exact
                # codebook but omitted the redundant packing table entry.
                # A power-of-two codebook has one unambiguous index width.
                bits = int(size).bit_length() - 1
                if size <= 0 or (1 << bits) != int(size):
                    raise ValueError(
                        f"L{layer} cannot infer packed width "
                        f"from non-power-of-two codebook {layout}"
                    )
                packing = (
                    "u8" if bits == 8
                    else "u16" if bits == 16
                    else f"packed-u{bits}"
                )
            if packing not in formats:
                raise ValueError(
                    f"L{layer} has no public packed format for "
                    f"{layout} -> {packing!r}"
                )
            packed_formats.append(formats[packing])
            code_dims.append(int(dim))
            codebook_sizes.append(int(size))
        return {
            "packed_formats": tuple(packed_formats),
            "code_dims": tuple(code_dims),
            "codebook_sizes": tuple(codebook_sizes),
        }

    def projection_operator_capabilities(
        self,
        layer: int,
    ) -> tuple[dict[str, tuple], ...]:
        """Return every unique public packed capability used by a layer."""
        per_expert = self.projection_layout_by_expert.get(int(layer))
        if per_expert is None:
            return (self.projection_operator_capability(layer),)
        unique: dict[tuple[tuple[str, tuple], ...], dict[str, tuple]] = {}
        for expert in range(len(per_expert)):
            capability = self.projection_operator_capability(layer, expert)
            key = tuple(sorted(capability.items()))
            unique.setdefault(key, capability)
        return tuple(unique.values())


class CCCPStore:
    """dense + 专家文件的 mmap 访问。"""

    def __init__(self, root: str):
        self.root = root
        self.man = Manifest(root)
        self.cfg = self.man.config
        self._dense = SafeTensorCollection(
            self.man.dense_root,
            self.man.dense_files,
            audit_file=self.man.dense_audit_file,
        )
        self._dense_keys = set(self._dense.keys())
        self._expert_handles: dict[int, SafeFile] = {}
        self._expert_keys: dict[int, set[str]] = {}
        self._expert_open_lock = threading.RLock()
        self._cb_cache: dict[
            tuple[str, str, str], torch.Tensor
        ] = {}
        self._cb_lock = threading.RLock()
        self._vq_codebook_pool: SafeFile | None = None
        self._vq_codebook_pool_keys: set[str] = set()
        self._vq_assignments: dict[str, torch.Tensor] = {}
        if (
            self.man.vq_codebook_layout_format
            == "expert-assigned-codebook-v1"
        ):
            pool_name = (
                self.man.vq_codebook_file
                or "vq-codebooks.safetensors"
            )
            self._vq_codebook_pool = SafeFile(
                os.path.join(self.root, pool_name)
            )
            self._vq_codebook_pool_keys = set(
                self._vq_codebook_pool.keys()
            )
            assignment_keys = self.man.vq_codebook_layout.get(
                "assignment_keys",
                {
                    "gu": "assignment.v.gu",
                    "down": "assignment.v.down",
                },
            )
            for projection in ("gu", "down"):
                key = str(assignment_keys[projection])
                if key not in self._vq_codebook_pool_keys:
                    raise KeyError(f"多码本池缺少分配表: {key}")
                assignment = self._vq_codebook_pool.get_tensor(key)
                if (
                    assignment.dtype != torch.uint8
                    or tuple(assignment.shape)
                    != (92, int(self.cfg["n_experts"]))
                ):
                    raise ValueError(
                        f"多码本分配表形状/类型错误: "
                        f"{key} {assignment.dtype} "
                        f"{tuple(assignment.shape)}"
                    )
                self._vq_assignments[projection] = assignment
        self._mtp: SafeFile | None = None
        # 可选热度档案（模型目录 profile.json 或 CCCP_PROFILE_JSON）：层 → 按路由
        # 命中降序的专家号，供 ExpertPool 把最热专家永久钉进内存（LRU 对冷专家的
        # 一次性缓存会污染热集合，实测命中率仅 ~20%，钉住 top-32 ≈66% 路由质量）
        self.heat_ranks: dict[int, list[int]] | None = None
        self.heat_counts: dict[int, dict[int, float]] | None = None
        self.q4_heat_ranks: dict[int, list[int]] | None = None
        self.route_allowlist: dict[int, set[int]] | None = None
        self.profile_path: str | None = None
        self.profile_loaded = False
        pj = os.environ.get("CCCP_PROFILE_JSON") or os.path.join(root, "profile.json")
        if os.path.exists(pj):
            with open(pj, "r", encoding="utf-8") as f:
                pr = json.load(f)
            self.profile_path = os.path.abspath(pj)
            self.profile_loaded = True
            raw_counts = pr.get("counts", {})
            self.heat_counts = {
                int(layer): {
                    int(expert): float(count)
                    for expert, count in counts.items()
                }
                for layer, counts in raw_counts.items()
            }
            self.heat_ranks = {
                int(l): sorted(
                    (int(e) for e in cnt),
                    key=lambda e: (-float(cnt[str(e)]), e),
                )
                for l, cnt in raw_counts.items()
            }
            # 默认按语料命中排序。可选 route-cost A/B 模式再乘模型 manifest
            # 中的 VQ 码本成本；两种模式都不含领域硬编码，也不改变专家集合。
            q4_score_mode = os.environ.get(
                "CCCP_CPU_Q4_HOT_SCORE", "route"
            ).strip().lower()
            if q4_score_mode not in {"route", "route-cost"}:
                raise ValueError("CCCP_CPU_Q4_HOT_SCORE must be route or route-cost")

            def q4_vq_cost(layer: int, expert: int) -> int:
                if q4_score_mode == "route":
                    return 1
                layouts = self.man.projection_layouts(layer, expert)
                return sum(
                    int(self.man.vq_dims[layouts[name]][0])
                    * int(self.man.vq_dims[layouts[name]][1])
                    for name in self.man.projection_names
                )

            self.q4_heat_ranks = {
                int(layer): sorted(
                    (int(expert) for expert in counts),
                    key=lambda expert: (
                        -float(counts[str(expert)])
                        * q4_vq_cost(int(layer), expert),
                        -float(counts[str(expert)]),
                        expert,
                    ),
                )
                for layer, counts in raw_counts.items()
            }
            if os.environ.get("CCCP_ROUTE_PROFILE", "0") != "0":
                raw_allowed = pr.get("allowed_experts")
                if raw_allowed is None and pr.get("strict_route"):
                    raw_allowed = {
                        layer: list(counts)
                        for layer, counts in pr.get("counts", {}).items()
                    }
                if not isinstance(raw_allowed, dict):
                    raise ValueError("路由 Profile 缺少 allowed_experts 对象")
                self.route_allowlist = {
                    int(layer): {int(expert) for expert in experts}
                    for layer, experts in raw_allowed.items()
                }
                top_k = int(self.cfg.get("top_k", 1))
                expert_count = int(self.cfg["n_experts"])
                for layer in self.man.expert_files:
                    allowed = self.route_allowlist.get(int(layer), set())
                    if len(allowed) < top_k:
                        raise ValueError(
                            f"路由 Profile 的 L{layer} 仅有 {len(allowed)} 个专家，"
                            f"少于 top_k={top_k}"
                        )
                    if min(allowed) < 0 or max(allowed) >= expert_count:
                        raise ValueError(f"路由 Profile 的 L{layer} 含越界专家")
        # MTP 附件存在时，把第 78 层注册进专家体系（透明复用 ExpertPool/回退掩码）
        mtp_path = os.path.join(root, "mtp.safetensors")
        l78_path = os.path.join(root, "experts.L78.safetensors")
        if os.path.exists(mtp_path) and os.path.exists(l78_path):
            self._mtp = SafeFile(mtp_path)
            self.man.expert_files[78] = "experts.L78.safetensors"
            self.man.tiers_per_layer[78] = "v" * self.cfg["n_experts"]

    def has_mtp(self) -> bool:
        return self._mtp is not None

    def close(self) -> None:
        """Close lazily opened shard handles without modifying model files."""
        self._dense.close()
        for handle in self._expert_handles.values():
            handle.close()
        if self._vq_codebook_pool is not None:
            self._vq_codebook_pool.close()
        if self._mtp is not None:
            self._mtp.close()

    def release_ram_blobs(self) -> None:
        """Detach SafeFile views after all permanent GPU weights are ready."""
        self._dense.release_ram_blob()
        for handle in self._expert_handles.values():
            handle.release_ram_blob()
        if self._vq_codebook_pool is not None:
            self._vq_codebook_pool.release_ram_blob()
        if self._mtp is not None:
            self._mtp.release_ram_blob()

    def clear_codebook_cache(self) -> int:
        """Release decoded codebooks held by completed route-scan layers."""
        with self._cb_lock:
            released = len(self._cb_cache)
            self._cb_cache.clear()
        return released

    def release_expert_layer(self, layer: int) -> bool:
        """Close one completed expert shard used by a layer-local scan.

        Normal chat keeps shard metadata and handles cached because it revisits
        every layer for each generated token. Route calibration is a single
        layer-first pass, so completed shards cannot provide a later cache hit.
        """
        layer = int(layer)
        with self._expert_open_lock:
            handle = self._expert_handles.pop(layer, None)
            self._expert_keys.pop(layer, None)
        if handle is None:
            return False
        handle.release_ram_blob()
        handle.close()
        return True

    def release_dense_ram_blob(self) -> tuple[int, tuple[str, ...]]:
        """Detach only Dense RAM images after CUDA weights become permanent.

        Packed expert views are intentionally untouched: RAM+GPU inference
        may still use them as the source of asynchronous expert DMA.
        """
        paths = self._dense.ram_blob_paths()
        return self._dense.release_ram_blob(), paths

    def get_mtp(self, name: str):
        """MTP dense 权重：attn.* 为 int4 对，router 等小张量 f32 原样。"""
        assert self._mtp is not None, "模型目录无 MTP 附件（mtp.safetensors）"
        keys = set(self._mtp.keys())
        if name + ".qs" in keys:
            q = self._mtp.get_tensor(name)
            s = self._mtp.get_tensor(name + ".qs")
            return Int4Weight(q, s, q.shape[1] * 2, self.man.int4_group)
        return self._mtp.get_tensor(name).float()

    # ---- dense ----
    def has(self, name: str) -> bool:
        return name in self._dense_keys

    def dense_names(self) -> list[str]:
        """全部 dense 权重名（不含量化缩放伴随键）。"""
        return sorted(
            name
            for name in self._dense_keys
            if not name.endswith(".qs")
            and not (
                name.endswith(".weight_scale_inv")
                and name[: -len("_scale_inv")] in self._dense_keys
            )
            and not (
                name.endswith(".scale")
                and name[: -len("scale")] + "weight"
                in self._dense_keys
            )
        )

    def get_raw(self, name: str) -> torch.Tensor:
        return self._dense.get_tensor(name)

    def dense_nbytes(self, name: str) -> int:
        """Return one dense tensor's stored bytes without reading its payload."""
        return self._dense.nbytes(name)

    def dense_resident_nbytes(self, name: str) -> int:
        """Return decoded in-memory bytes used by placement planning."""
        return self._dense.resident_nbytes(name)

    def get_dense(self, name: str):
        """返回 f32 张量（小权重）或 Int4Weight（打包大权重）。"""
        if name + ".qs" in self._dense_keys:
            q = self._dense.get_tensor(name)
            s = self._dense.get_tensor(name + ".qs")
            if self.man.quant.get("dense") == "fp8-native":
                return BlockFP8Weight(q, s, q.shape[1])
            return Int4Weight(q, s, q.shape[1] * 2, self.man.int4_group)
        audited_fp8 = self._dense.get_block_fp8(name)
        if audited_fp8 is not None:
            return audited_fp8
        value = self._dense.get_tensor(name)
        if self.man.dense_audit_file is not None:
            # Audited mixed Dense formats already declare the exact logical
            # dtype per tensor. New format names must not fall through to the
            # legacy unconditional FP32 conversion.
            return value
        if self.man.quant.get("dense") in (
            "source-native-uncompressed",
            "mixed-source-fp8-d3-p12",
        ):
            return value
        return value.float()

    # ---- 专家 ----
    def _eh(self, layer: int) -> SafeFile:
        h = self._expert_handles.get(layer)
        if h is None:
            # get_many can request several experts from a newly opened layer
            # concurrently.  Publish the handle and its key set atomically.
            with self._expert_open_lock:
                h = self._expert_handles.get(layer)
                if h is None:
                    h = SafeFile(
                        os.path.join(
                            self.root,
                            self.man.expert_files[layer],
                        )
                    )
                    keys = set(h.keys())
                    self._expert_handles[layer] = h
                    self._expert_keys[layer] = keys
        return h

    def expert_kind(self, layer: int, eid: int) -> str:
        """探测专家档位：返回 VQ 档名（可带 z 后缀）或 ``drop``。

        ``p12`` 是 k=4096 索引的磁盘紧凑编码，不属于新的计算档位，所以
        对上层仍报告原档名，加载时再透明解包。
        """
        keys = self._expert_keys.get(layer)
        if keys is None:
            self._eh(layer)
            keys = self._expert_keys[layer]
        if self.man.projection_vq:
            layouts = self.man.projection_layouts(layer, eid)
            present = True
            for projection in self.man.projection_names:
                storage_names = (
                    ("down", "dn") if projection == "down" else (projection,)
                )
                if not any(
                    f"e{eid}.{name}.{layouts[projection]}" in keys
                    for name in storage_names
                ):
                    present = False
                    break
            if present:
                return "projection-vq"
            return "drop"
        for k in self.man.vq_dims:
            if (
                f"e{eid}.gu.{k}" in keys
                and f"e{eid}.down.{k}" in keys
            ):
                return k
            if f"e{eid}.gu{k}p14z" in keys:
                return k + "z"
            if f"e{eid}.gu{k}p14" in keys:
                return k
            if f"e{eid}.gu{k}p12z" in keys:
                return k + "z"
            if f"e{eid}.gu{k}p12" in keys:
                return k
            if f"e{eid}.gu{k}z" in keys:
                return k + "z"
            if f"e{eid}.gu{k}" in keys:
                return k
        return "drop"

    def available_mask(self, layer: int) -> torch.Tensor:
        """该层可用专家布尔掩码 [E]（drop 为 False），用于回退路由掩码。"""
        E = self.cfg["n_experts"]
        s = self.man.tier_string(layer)
        if s is not None:
            if len(s) != E:
                raise ValueError(
                    f"L{layer} tier_string 长度 {len(s)} != n_experts {E}"
                )
            mask = torch.tensor(
                [c.lower() != "d" for c in s],
                dtype=torch.bool,
            )
        else:
            mask = torch.ones(E, dtype=torch.bool)  # 清单无档位串 = 全保留（老产物）
        if self.route_allowlist is not None and layer in self.route_allowlist:
            allowed = torch.zeros(E, dtype=torch.bool)
            allowed[list(sorted(self.route_allowlist[layer]))] = True
            mask &= allowed
        return mask

    @staticmethod
    def _down_codebook_stem(
        keys: set[str],
        kind: str,
    ) -> str:
        standard = f"cb.down.{kind}"
        return (
            standard
            if any(
                key == standard or key.startswith(standard + ".")
                for key in keys
            )
            else f"cb.dn.{kind}"
        )

    def _codebook_reference(
        self,
        layer: int,
        kind: str,
        eid: int | None,
        projection: str,
    ) -> tuple[SafeFile, str, str]:
        """按专属→专家分配→连续分组→共享解析一个投影的码本。"""
        if projection not in ("gu", "down"):
            raise ValueError(projection)
        self._eh(layer)
        keys = self._expert_keys[layer]
        stem = (
            f"cb.gu.{kind}"
            if projection == "gu"
            else self._down_codebook_stem(keys, kind)
        )
        if (
            eid is not None
            and f"{stem}.e{eid}" in keys
        ):
            key = f"{stem}.e{eid}"
            return (
                self._expert_handles[layer],
                key,
                f"L{layer}.e{eid}",
            )
        layout = self.man.vq_codebook_layout
        if (
            self.man.vq_codebook_layout_format
            == "expert-assigned-codebook-v1"
            and kind == str(layout.get("kind", "v"))
            and eid is not None
        ):
            assignment = self._vq_assignments[projection]
            row = layer - 1
            if 0 <= row < assignment.shape[0]:
                codebook = int(assignment[row, eid])
                sentinel = int(
                    layout.get("missing_assignment_sentinel", 255)
                )
                if codebook != sentinel:
                    band_size = int(layout["layer_band_size"])
                    band = row // band_size
                    key = (
                        f"cb.{kind}.band{band:02d}."
                        f"{projection}.{codebook:03d}"
                    )
                    if key not in self._vq_codebook_pool_keys:
                        raise KeyError(
                            "专家分配表引用了不存在的码本: "
                            f"L{layer} e{eid} {projection} -> {key}"
                        )
                    assert self._vq_codebook_pool is not None
                    semantic = f"band{band:02d}.cb{codebook:03d}"
                    return self._vq_codebook_pool, key, semantic
                if not bool(layout.get("legacy_fallback", True)):
                    raise KeyError(
                        f"L{layer} e{eid} {projection} 无多码本分配"
                    )
        group_size = self.man.vq_codebook_group_sizes.get(kind)
        if group_size is not None and eid is not None:
            variant = f"g{eid // group_size:03d}"
            key = f"{stem}.{variant}"
            if key in keys:
                return (
                    self._expert_handles[layer],
                    key,
                    f"L{layer}.{variant}",
                )
        if stem not in keys:
            raise KeyError(
                f"L{layer} e{eid} {projection} 缺少可用码本: {stem}"
            )
        return (
            self._expert_handles[layer],
            stem,
            f"L{layer}.shared",
        )

    def codebook_variants(
        self,
        layer: int,
        kind: str,
        eid: int | None,
    ) -> tuple[str, str]:
        """返回GU/Down稳定语义键，用于RAM/GPU码本缓存。"""
        gu = self._codebook_reference(
            layer, kind, eid, "gu"
        )[2]
        down = self._codebook_reference(
            layer, kind, eid, "down"
        )[2]
        return gu, down

    def codebook_variant(
        self,
        layer: int,
        kind: str,
        eid: int | None,
    ) -> str:
        """旧调用兼容；仅当GU/Down选择相同时返回一个语义键。"""
        gu, down = self.codebook_variants(layer, kind, eid)
        return gu if gu == down else f"{gu}|{down}"

    def codebooks(
        self,
        layer: int,
        kind: str,
        eid: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """依次选择专家专属、自适应分配、连续分组、层共享码本。"""
        gu_handle, gu_key, gu_variant = self._codebook_reference(
            layer, kind, eid, "gu"
        )
        (
            down_handle,
            down_key,
            down_variant,
        ) = self._codebook_reference(
            layer, kind, eid, "down"
        )
        with self._cb_lock:
            gu_cache_key = (gu_handle.path, "gu", gu_key)
            cb_gu = self._cb_cache.get(gu_cache_key)
            if cb_gu is None:
                cb_gu = gu_handle.get_tensor(gu_key).float()
                self._cb_cache[gu_cache_key] = cb_gu
            down_cache_key = (
                down_handle.path,
                "down",
                down_key,
            )
            cb_down = self._cb_cache.get(down_cache_key)
            if cb_down is None:
                cb_down = down_handle.get_tensor(down_key).float()
                self._cb_cache[down_cache_key] = cb_down
        return cb_gu, cb_down

    def _projection_codebook_key(
        self,
        layer: int,
        projection: str,
        eid: int | None,
    ) -> str:
        layout = self.man.projection_layouts(layer, eid)[projection]
        key = f"cb.{projection}.{layout}"
        if self.man.projection_private_codebooks:
            if eid is None:
                raise ValueError(
                    f"L{layer} {projection} private codebook requires expert_id"
                )
            return f"{key}.e{int(eid)}"
        group_size = self.man.projection_codebook_group_sizes.get(layout)
        if group_size is None:
            return key
        if eid is None:
            raise ValueError(
                f"L{layer} {projection} layout {layout} requires expert_id"
            )
        group = int(eid) // group_size
        groups = self.man.projection_codebook_group_counts.get(layout)
        if group < 0 or (groups is not None and group >= groups):
            raise ValueError(
                f"L{layer} e{eid} {projection} codebook group {group} "
                f"is outside layout {layout}"
            )
        return f"{key}.g{group:03d}"

    def projection_codebooks(
        self,
        layer: int,
        eid: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """读取三投影独立码本，不把 gate/up 强行拼成同一码本。"""
        if not self.man.projection_vq:
            raise RuntimeError("当前模型不是三投影 VQ 格式")
        if self.man.combined_projection_vq:
            layouts = self.man.projection_layouts(layer, eid)
            result = []
            with self._cb_lock:
                for projection in self.man.projection_names:
                    handle, key, _variant = self._codebook_reference(
                        layer,
                        layouts[projection],
                        eid,
                        projection,
                    )
                    cache_key = (handle.path, projection, key)
                    codebook = self._cb_cache.get(cache_key)
                    if codebook is None:
                        codebook = handle.get_tensor(key).float()
                        self._cb_cache[cache_key] = codebook
                    result.append(codebook)
            return tuple(result)
        handle = self._eh(layer)
        result = []
        with self._cb_lock:
            for projection in self.man.projection_names:
                key = self._projection_codebook_key(
                    layer,
                    projection,
                    eid,
                )
                cache_key = (handle.path, projection, key)
                codebook = self._cb_cache.get(cache_key)
                if codebook is None:
                    codebook = handle.get_tensor(key).float()
                    self._cb_cache[cache_key] = codebook
                result.append(codebook)
        return tuple(result)

    def projection_codebook_variants(
        self,
        layer: int,
        eid: int | None = None,
    ) -> tuple[str, ...]:
        """返回稳定的三投影码本语义键，供 RAM/VRAM 缓存隔离。"""
        if self.man.combined_projection_vq:
            layouts = self.man.projection_layouts(layer, eid)
            return tuple(
                self._codebook_reference(
                    layer,
                    layouts[projection],
                    eid,
                    projection,
                )[2]
                for projection in self.man.projection_names
            )
        return tuple(
            f"L{layer}." + self._projection_codebook_key(
                layer,
                projection,
                eid,
            )[3:]
            for projection in self.man.projection_names
        )

    def load_expert(self, layer: int, eid: int) -> tuple[VQWeight, VQWeight]:
        """加载一个专家的 (gu, dn) VQ 权重；zlib blob 就地解压。

        注意：z 后缀在量化端是按张量独立决定的（gu/dn 可能一个压缩一个原始），
        这里逐张量探测。
        """
        kind = self.expert_kind(layer, eid)
        if kind == "drop":
            raise KeyError(f"expert {layer}/{eid} 已被量化丢弃")
        base = kind.rstrip("z")
        dim, _ = self.man.vq_dims[base]
        cb_gu, cb_dn = self.codebooks(layer, base, eid)
        h = self._eh(layer)
        keys = self._expert_keys[layer]
        H = self.cfg.get("routed_hidden", self.cfg["hidden"])
        I = self.cfg["moe_inter"]

        def _idx(tag: str, rows: int, cols: int) -> torch.Tensor:
            count = rows * (cols // dim)
            standard_tag = "down" if tag == "dn" else tag
            standard_key = f"e{eid}.{standard_tag}.{base}"
            if standard_key in keys:
                stored = h.get_tensor(standard_key)
                raw = stored.view(torch.uint8).reshape(-1)
                bits = _stored_index_bits(raw.numel(), count)
                if bits == 8:
                    return raw.reshape(rows, cols // dim)
                if bits == 16:
                    return raw.view(torch.uint16).reshape(
                        rows,
                        cols // dim,
                    )
                if bits == 9:
                    unpacked = _unpack_u9(raw, count)
                elif bits == 10:
                    unpacked = _unpack_u10(raw, count)
                elif bits == 12:
                    unpacked = _unpack_u12(raw, count)
                elif bits == 14:
                    unpacked = _unpack_u14(raw, count)
                else:
                    unpacked = _unpack_odd_width(raw, count, bits)
                return unpacked.reshape(rows, cols // dim)
            p14zkey = f"e{eid}.{tag}{base}p14z"
            p14key = f"e{eid}.{tag}{base}p14"
            if p14zkey in keys:
                raw = zlib.decompress(h.get_bytes(p14zkey))
                packed = _from_readonly_buffer(raw, dtype=torch.uint8)
                return _unpack_u14(packed, count).reshape(rows, cols // dim)
            if p14key in keys:
                packed = h.get_tensor(p14key)
                return _unpack_u14(packed, count).reshape(rows, cols // dim)
            p12zkey = f"e{eid}.{tag}{base}p12z"
            p12key = f"e{eid}.{tag}{base}p12"
            if p12zkey in keys:
                raw = zlib.decompress(h.get_bytes(p12zkey))
                packed = _from_readonly_buffer(raw, dtype=torch.uint8)
                return _unpack_u12(packed, count).reshape(rows, cols // dim)
            if p12key in keys:
                packed = h.get_tensor(p12key)
                return _unpack_u12(packed, count).reshape(rows, cols // dim)
            zkey = f"e{eid}.{tag}{base}z"
            if zkey in keys:
                # get_bytes 纯文件读（无 mmap），zlib 解压后即为索引字节
                raw = zlib.decompress(h.get_bytes(zkey))
                # k>256 的档索引为 u16（如 w=8D-k4096），其余 u8
                idt = torch.uint16 if self.man.vq_dims[base][1] > 256 else torch.uint8
                a = _from_readonly_buffer(raw, dtype=idt)  # bytes 直视图，免拷贝
                return a.reshape(rows, cols // dim)
            return h.get_tensor(f"e{eid}.{tag}{base}")

        return (VQWeight(_idx("gu", 2 * I, H), cb_gu, H),
                VQWeight(_idx("dn", H, I), cb_dn, I))

    def load_expert_packed(
        self,
        layer: int,
        eid: int,
    ) -> tuple["PackedVQWeight", ...]:
        """Load an expert without expanding packed on-disk indices.

        Projection archives use row-aligned p9/p10/p12/p14 packing. Expanding
        these tensors to uint16 makes full-VRAM residency impractical. The
        packed representation is consumed directly by the common CPU/CUDA VQ
        operators; legacy ``load_expert`` remains unchanged.
        """
        if self.man.projection_vq:
            handle = self._eh(layer)
            layouts = self.man.projection_layouts(layer, eid)
            codebooks = self.projection_codebooks(layer, eid)
            hidden = int(
                self.cfg.get("routed_hidden", self.cfg["hidden"])
            )
            intermediate = int(self.cfg["moe_inter"])
            shapes = {
                "gate": (intermediate, hidden),
                "up": (intermediate, hidden),
                "down": (hidden, intermediate),
                "gu": (2 * intermediate, hidden),
            }
            weights = []
            for projection, codebook in zip(
                self.man.projection_names,
                codebooks,
            ):
                layout_name = layouts[projection]
                dim, _codebook_size = self.man.vq_dims[layout_name]
                rows, cols = shapes[projection]
                storage_names = (
                    ("down", "dn") if projection == "down" else (projection,)
                )
                key = next(
                    (
                        f"e{eid}.{name}.{layout_name}"
                        for name in storage_names
                        if f"e{eid}.{name}.{layout_name}"
                        in self._expert_keys[layer]
                    ),
                    None,
                )
                if key is None:
                    raise KeyError(
                        f"L{layer} e{eid} is missing {projection} layout "
                        f"{layout_name}"
                    )
                stored = (
                    handle.get_tensor(key)
                    .view(torch.uint8)
                    .reshape(-1)
                )
                count = rows * (cols // dim)
                bits = _stored_index_bits(stored.numel(), count)
                declared = self.man.index_packing.get(layout_name)
                packing_widths = {
                    f"packed-u{width}": width
                    for width in range(8, 17)
                }
                packing_widths.update({"u8": 8, "u16": 16})
                expected = packing_widths.get(declared)
                if expected is not None and bits != expected:
                    raise ValueError(
                        f"{key} 索引位宽与清单不符: "
                        f"{bits} != {expected}"
                    )
                weights.append(
                    PackedVQWeight(
                        stored,
                        codebook,
                        rows,
                        cols,
                        bits,
                    )
                )
            return tuple(weights)

        kind = self.expert_kind(layer, eid)
        if kind == "drop":
            raise KeyError(f"expert {layer}/{eid} 已被量化丢弃")
        base = kind.rstrip("z")
        dim, codebook_size = self.man.vq_dims[base]
        cb_gu, cb_dn = self.codebooks(layer, base, eid)
        handle = self._eh(layer)
        keys = self._expert_keys[layer]
        hidden = int(self.cfg.get("routed_hidden", self.cfg["hidden"]))
        intermediate = int(self.cfg["moe_inter"])

        def packed(
            tag: str,
            rows: int,
            cols: int,
        ) -> PackedVQWeight:
            standard_tag = "down" if tag == "dn" else tag
            standard_key = f"e{eid}.{standard_tag}.{base}"
            if standard_key in keys:
                storage = (
                    handle.get_tensor(standard_key)
                    .view(torch.uint8)
                    .reshape(-1)
                )
                bits = _stored_index_bits(
                    storage.numel(),
                    rows * (cols // dim),
                )
                return PackedVQWeight(
                    storage,
                    cb_gu if tag == "gu" else cb_dn,
                    rows,
                    cols,
                    bits,
                )
            for bits in (14, 12):
                stem = f"e{eid}.{tag}{base}p{bits}"
                if stem + "z" in keys:
                    raw = zlib.decompress(handle.get_bytes(stem + "z"))
                    storage = _from_readonly_buffer(raw, dtype=torch.uint8)
                    return PackedVQWeight(
                        storage,
                        cb_gu if tag == "gu" else cb_dn,
                        rows,
                        cols,
                        bits,
                    )
                if stem in keys:
                    storage = (
                        handle.get_tensor(stem)
                        .view(torch.uint8)
                        .reshape(-1)
                    )
                    return PackedVQWeight(
                        storage,
                        cb_gu if tag == "gu" else cb_dn,
                        rows,
                        cols,
                        bits,
                    )

            stem = f"e{eid}.{tag}{base}"
            dtype = torch.uint16 if codebook_size > 256 else torch.uint8
            if stem + "z" in keys:
                raw = zlib.decompress(handle.get_bytes(stem + "z"))
                indices = _from_readonly_buffer(raw, dtype=dtype)
            else:
                indices = handle.get_tensor(stem)
            return PackedVQWeight(
                indices.view(torch.uint8).reshape(-1),
                cb_gu if tag == "gu" else cb_dn,
                rows,
                cols,
                16 if dtype == torch.uint16 else 8,
            )

        return (
            packed("gu", 2 * intermediate, hidden),
            packed("dn", hidden, intermediate),
        )


class PackedVQWeight:
    """Byte-exact VQ indices plus logical matrix metadata.

    ``bits`` is 8/16 for ordinary indices and 9..15 for row-aligned
    packed indices.  Only the byte payload is staged to CUDA; no dequantized
    expert matrix and no expanded uint16 copy is created.
    """

    __slots__ = (
        "raw",
        "cb",
        "rows",
        "cols",
        "blocks",
        "dim",
        "bits",
        "source_bits",
        "layout",
    )

    def __init__(
        self,
        raw: torch.Tensor,
        cb: torch.Tensor,
        rows: int,
        cols: int,
        bits: int,
    ):
        if not 8 <= bits <= 16:
            raise ValueError(f"unsupported packed VQ width {bits}")
        self.raw = raw.contiguous().view(torch.uint8).reshape(-1)
        self.cb = cb.float()
        self.rows = int(rows)
        self.cols = int(cols)
        self.dim = int(cb.shape[1])
        if self.cols % self.dim:
            raise ValueError("VQ columns must be divisible by code dimension")
        self.blocks = self.cols // self.dim
        self.bits = int(bits)
        self.source_bits = int(bits)
        self.layout = "row-major"
        expected_bits = self.rows * self.blocks * self.bits
        if expected_bits % 8:
            raise ValueError(
                "packed VQ tensor must be byte aligned across complete rows"
            )
        expected = expected_bits // 8
        if self.raw.numel() != expected:
            raise ValueError(
                f"packed VQ payload mismatch: {self.raw.numel()} != {expected}"
            )

    @property
    def nbytes(self) -> int:
        return self.raw.numel()

    @property
    def dtype_tag(self) -> int:
        return {
            8: 0,
            16: 1,
            12: 2,
            14: 3,
            10: 4,
            9: 5,
            11: 6,
            13: 7,
            15: 8,
        }[
            self.bits
        ]

    def optimize_cpu_layout(self) -> bool:
        """Replace row-major indices with compact block-major traversal.

        The byte count and index width stay identical.  This is a CPU-only
        storage transform; CUDA transport continues to use the archive's
        original row-major representation unless explicitly requested.
        """
        if self.layout == "block-major":
            return True
        from .ops import vq_relayout_block_major

        packed = vq_relayout_block_major(
            self.raw,
            rows=self.rows,
            blocks=self.blocks,
            bits=self.bits,
            code_dim=self.dim,
            codebook_size=int(self.cb.shape[0]),
        )
        if packed is None:
            return False
        if packed.numel() != self.raw.numel():
            raise RuntimeError("compact VQ relayout changed payload size")
        self.raw = packed
        self.layout = "block-major"
        return True

    def optimize_cpu_row_tile(self, tile_rows: int = 8) -> bool:
        """Replace row-major indices with compact CPU row-tile traversal."""
        if self.layout == f"row-tile-{int(tile_rows)}":
            return True
        if self.layout != "row-major":
            return False
        from .ops import vq_relayout_row_tile

        packed = vq_relayout_row_tile(
            self.raw,
            rows=self.rows,
            blocks=self.blocks,
            bits=self.bits,
            code_dim=self.dim,
            codebook_size=int(self.cb.shape[0]),
            tile_rows=int(tile_rows),
        )
        if packed is None:
            return False
        if packed.numel() != self.raw.numel():
            raise RuntimeError("compact VQ row-tile changed payload size")
        self.raw = packed
        self.layout = f"row-tile-{int(tile_rows)}"
        return True

    def compile_cpu_u16_row_tile(self, tile_rows: int = 8) -> bool:
        """Compile packed bytes into an exact runtime-only CPU image."""
        if self.layout == f"u16-row-tile-{int(tile_rows)}":
            return True
        if self.layout != "row-major" or int(tile_rows) != 8:
            return False
        from .ops import vq_compile_u16_row_tile

        compiled = vq_compile_u16_row_tile(
            self.raw,
            rows=self.rows,
            blocks=self.blocks,
            bits=self.bits,
            code_dim=self.dim,
            codebook_size=int(self.cb.shape[0]),
            tile_rows=int(tile_rows),
        )
        if compiled is None:
            return False
        if compiled.dtype != torch.uint16 or compiled.numel() != (
            self.rows * self.blocks
        ):
            raise RuntimeError("CPU VQ compilation returned an invalid image")
        self.source_bits = int(self.bits)
        self.raw = compiled.contiguous().view(torch.uint8).reshape(-1)
        self.bits = 16
        self.layout = f"u16-row-tile-{int(tile_rows)}"
        return True

    def compile_cpu_q4_0(self) -> bool:
        """Compile VQ codes to a process-local Q4 block-dot execution image."""
        if self.layout == "q4_0":
            return True
        if self.layout != "row-major" or self.cols % 32:
            return False
        from .ops import vq_compile_q4_0

        compiled = vq_compile_q4_0(
            self.raw,
            self.cb,
            rows=self.rows,
            blocks=self.blocks,
            bits=self.bits,
            code_dim=self.dim,
            codebook_size=int(self.cb.shape[0]),
        )
        expected = self.rows * (self.cols // 32) * 18
        if (
            compiled is None
            or compiled.dtype != torch.uint8
            or compiled.numel() != expected
        ):
            return False
        self.source_bits = int(self.bits)
        self.raw = compiled.contiguous().reshape(-1)
        self.layout = "q4_0"
        return True

    def unpack(self) -> torch.Tensor:
        """Reference unpacker used by CPU tests and correctness probes."""
        if self.layout == "q4_0":
            raise RuntimeError("Q4 execution images no longer contain VQ indices")
        count = self.rows * self.blocks
        if self.bits == 8:
            result = self.raw
        elif self.bits == 16:
            result = self.raw.view(torch.uint16)
        elif self.bits == 9:
            result = _unpack_u9(self.raw, count)
        elif self.bits == 10:
            result = _unpack_u10(self.raw, count)
        elif self.bits == 12:
            result = _unpack_u12(self.raw, count)
        elif self.bits == 14:
            result = _unpack_u14(self.raw, count)
        else:
            result = _unpack_odd_width(self.raw, count, self.bits)
        physical = result.reshape(-1)
        if self.layout == "row-major":
            return physical.reshape(self.rows, self.blocks)
        if self.layout == "block-major":
            return physical.reshape(self.blocks, self.rows).t().contiguous()
        if self.layout in ("row-tile-8", "u16-row-tile-8"):
            logical = torch.empty(
                self.rows,
                self.blocks,
                dtype=physical.dtype,
                device=physical.device,
            )
            for first_row in range(0, self.rows, 8):
                valid = min(8, self.rows - first_row)
                start = first_row * self.blocks
                stop = start + self.blocks * valid
                logical[first_row : first_row + valid].copy_(
                    physical[start:stop].reshape(self.blocks, valid).t()
                )
            return logical
        raise ValueError(f"unsupported packed VQ layout {self.layout!r}")


class PackedCpuExpertPool:
    """Generic CPU LRU that keeps VQ expert indices byte-packed.

    The pool deliberately mirrors only the small subset of ``ExpertPool`` used
    by CPU decode. Packed payloads stay compact in RAM; the common CPU VQ
    backend extracts indices while computing and never creates a resident
    uint16 expansion.
    """

    full_resident = False
    prefetch_default = True
    expanded_index_bytes = 0
    prefill_rows_supported = True

    def __init__(self, store: CCCPStore, budget_gb: float = 16.0):
        self.store = store
        self.device = torch.device("cpu")
        self.gpu = False
        self.budget = max(0, int(float(budget_gb) * 2**30))
        self.cache: OrderedDict[
            tuple[int, int],
            tuple[PackedVQWeight, ...],
        ] = OrderedDict()
        self.pinned: dict[
            tuple[int, int],
            tuple[PackedVQWeight, ...],
        ] = {}
        self.bytes = 0
        self.compact_full_resident = False
        self.hits = 0
        self.miss = 0
        self._pending: OrderedDict = OrderedDict()
        self.layer_local_cache = (
            os.environ.get("CCCP_ROUTE_SCAN_LAYER_LOCAL", "0") != "0"
        )
        self._active_layer: int | None = None
        self.layer_cache_resets = 0
        self._native_layers: dict[int, object | bool] = {}
        self.native_hits = 0
        self.native_fallbacks = 0
        self.block_major_entries = 0
        self.block_major_bytes = 0
        self.compiled_index_bytes = 0
        self.compiled_source_bytes = 0
        self.compiled_linear_bytes = 0
        self.cpu_compile_mode = "off"
        self.prefetch_calls = 0
        self.prefetch_keys = 0
        self.prefill_rows_calls = 0
        self.prefill_rows_tokens = 0
        self.prefill_rows_micro_batches = 0
        self.prefill_rows_fallbacks = 0

    @property
    def host_expert_bytes(self) -> int:
        """Return the compact resident/LRU payload footprint."""
        return int(self.bytes)

    @property
    def compact_resident_entries(self) -> int:
        return len(self.pinned)

    @staticmethod
    def _entry_bytes(entry) -> int:
        return sum(int(weight.nbytes) for weight in entry)

    def _put(self, key, entry) -> None:
        size = self._entry_bytes(entry)
        old = self.cache.pop(key, None)
        if old is not None:
            self.bytes -= self._entry_bytes(old)
        while self.cache and self.bytes + size > self.budget:
            _, victim = self.cache.popitem(last=False)
            self.bytes -= self._entry_bytes(victim)
        if size <= self.budget:
            self.cache[key] = entry
            self.bytes += size

    def _enter_layer(self, layer: int) -> None:
        """Keep only one routed-expert layer resident during route scans."""
        layer = int(layer)
        if not self.layer_local_cache or self._active_layer == layer:
            return
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        self.cache.clear()
        self.bytes = 0
        self.store.clear_codebook_cache()
        if self._active_layer is not None:
            self.layer_cache_resets += 1
        self._active_layer = layer

    def release_scan_layer(self, layer: int) -> bool:
        """Immediately release the completed layer in route-scan mode."""
        layer = int(layer)
        if not self.layer_local_cache or self._active_layer != layer:
            return False
        pending = tuple(self._pending.values())
        self._pending.clear()
        for future in pending:
            future.cancel()
        # A future that was already running cannot be cancelled. Wait for it
        # before closing the shard that its SafeFile reader may still use.
        for future in pending:
            if future.cancelled():
                continue
            try:
                future.result()
            except Exception:
                # Loading errors are surfaced by the synchronous get_many
                # path. Cleanup remains best-effort during cancellation.
                pass
        self.cache.clear()
        self.bytes = 0
        self.store.clear_codebook_cache()
        released = bool(self.store.release_expert_layer(layer))
        self._native_layers.pop(layer, None)
        self._active_layer = None
        self.layer_cache_resets += 1
        return released

    def preload_all(self, reserve_gb: float | None = None) -> bool:
        if os.environ.get("CCCP_FULL_RESIDENT", "1") == "0":
            return False
        # 严格领域 Profile 只允许一小组专家。不要用全模型 69GiB 文件体积
        # 拒绝它的运行时编译；返回 False 后由 preload_pinned 精确编译/常驻白名单。
        if (
            self.store.route_allowlist is not None
            and os.environ.get("CCCP_PROFILE_FULL_LOAD", "0") != "0"
        ):
            return False
        import psutil
        from concurrent.futures import FIRST_COMPLETED, wait

        if reserve_gb is None:
            reserve_gb = float(
                os.environ.get("CCCP_RESIDENT_RESERVE_GB", "2.0")
            )
        total = sum(
            os.path.getsize(os.path.join(self.store.root, filename))
            for filename in self.store.man.expert_files.values()
            if os.path.exists(os.path.join(self.store.root, filename))
        )
        available = int(psutil.virtual_memory().available)
        compile_mode = os.environ.get(
            "CCCP_CPU_COMPILE", "off"
        ).strip().lower()
        if compile_mode not in {"0", "off", "false", "auto", "u16", "q4"}:
            raise ValueError("CCCP_CPU_COMPILE must be off, auto, u16, or q4")
        # ``auto`` preserves the lossless historical u16 execution view.
        # Q4 changes model weights and therefore always requires an explicit
        # CLI choice until a model-specific quality gate accepts it.
        requested_compile = "u16" if compile_mode == "auto" else compile_mode
        compile_enabled = requested_compile in {"u16", "q4"}
        compiled_upper_bound = total * (4 if requested_compile == "q4" else 2)
        compiled_need = compiled_upper_bound + int(reserve_gb * 2**30)
        if compile_enabled and compiled_need > available:
            if compile_mode in {"u16", "q4"}:
                raise MemoryError(
                    "forced CPU VQ compilation cannot fit: "
                    f"upper bound {compiled_upper_bound / 2**30:.1f}GiB + "
                    f"reserve {reserve_gb:.1f}GiB > available "
                    f"{available / 2**30:.1f}GiB"
                )
            compile_enabled = False
            print(
                "[cccp] CPU 在线编译自动回退紧凑索引："
                f"上界 {compiled_upper_bound / 2**30:.1f}GiB + "
                f"预留 {reserve_gb:.1f}GiB > 可用 "
                f"{available / 2**30:.1f}GiB",
                flush=True,
            )
        if total + int(reserve_gb * 2**30) > available:
            print(
                "[cccp] packed CPU专家无法全量常驻："
                f"文件约 {total / 2**30:.1f}GiB + 预留 {reserve_gb:.1f}GiB"
                f" > 可用 {available / 2**30:.1f}GiB，回退紧凑LRU",
                flush=True,
            )
            return False
        n_experts = int(self.store.cfg["n_experts"])
        keys = [
            (int(layer), expert)
            for layer in self.store.man.expert_files
            for expert in range(n_experts)
            if self.store.expert_kind(int(layer), expert) != "drop"
        ]
        started = time.time()
        print(
            f"[cccp] packed CPU专家全量常驻：{len(keys)} 个读取中…",
            flush=True,
        )
        resident_bytes = 0
        layout_mode = os.environ.get(
            "CCCP_CPU_PACKED_LAYOUT", "tile8"
        ).strip().lower()
        self.cpu_compile_mode = requested_compile if compile_enabled else "off"
        executor = _executor()
        key_iterator = iter(keys)
        pending = {}
        configured_window = int(os.environ.get("CCCP_CPU_LOAD_WINDOW", "32"))
        window = (
            1
            if self.cpu_compile_mode == "q4"
            else max(4, configured_window)
        )

        def submit_one() -> bool:
            try:
                key = next(key_iterator)
            except StopIteration:
                return False
            pending[executor.submit(self.store.load_expert_packed, *key)] = key
            return True

        for _ in range(min(window, len(keys))):
            submit_one()
        index = 0
        while pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in completed:
                key = pending.pop(future)
                entry = future.result()
                index += 1
                if compile_enabled:
                    for weight in entry:
                        source_bytes = int(weight.nbytes)
                        compiled = (
                            weight.compile_cpu_q4_0()
                            if self.cpu_compile_mode == "q4"
                            else weight.compile_cpu_u16_row_tile(8)
                        )
                        if not compiled:
                            raise RuntimeError(
                                "CPU VQ compilation unavailable for "
                                f"layer/expert {key}"
                            )
                        self.compiled_source_bytes += source_bytes
                        self.compiled_index_bytes += int(weight.nbytes)
                        if self.cpu_compile_mode == "q4":
                            self.compiled_linear_bytes += int(weight.nbytes)
                elif layout_mode in {"tile8", "row-tile", "row_tile"}:
                    # Repack Gate, Up and Down. The old path stopped after
                    # Gate/Up, leaving the route-reduced Down projection in
                    # row-major order even though it owns the largest share
                    # of cold-cache reads. row-tile-8 is byte-for-byte the
                    # same size and preserves every logical index.
                    for weight in entry:
                        if weight.optimize_cpu_row_tile(8):
                            self.block_major_entries += 1
                            self.block_major_bytes += int(weight.nbytes)
                elif layout_mode not in {"0", "off", "false", "row"}:
                    for projection, weight in enumerate(entry):
                        if (
                            projection < 2
                            and weight.dim == 4
                            and int(weight.cb.shape[0]) <= 4096
                            and weight.optimize_cpu_layout()
                        ):
                            self.block_major_entries += 1
                            self.block_major_bytes += int(weight.nbytes)
                self.pinned[key] = entry
                resident_bytes += self._entry_bytes(entry)
                if index % 2000 == 0:
                    print(
                        f"[cccp] packed CPU专家常驻 {index}/{len(keys)}",
                        flush=True,
                    )
                submit_one()
        self.bytes = resident_bytes
        self.expanded_index_bytes = (
            int(self.compiled_index_bytes)
            if self.cpu_compile_mode == "u16"
            else 0
        )
        self.compact_full_resident = True
        if compile_enabled or layout_mode in {"tile8", "row-tile", "row_tile"}:
            # Relayout replaces each source tensor with an equal-size compact
            # tensor.  Release completed Future references and return the old
            # byte buffers to the OS instead of leaving one model-sized copy
            # in the glibc arena.
            pending.clear()
            future = None
            import ctypes
            import gc

            gc.collect()
            malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
            if malloc_trim is not None:
                malloc_trim(0)
        print(
            "[cccp] packed CPU专家常驻完成："
            f"{len(keys)} 个 / {resident_bytes / 2**30:.1f}GiB / "
            f"{time.time() - started:.1f}s；"
            f"cpu_compile={self.cpu_compile_mode}；"
            f"compiled_index_bytes={self.compiled_index_bytes}",
            flush=True,
        )
        return True

    def preload_pinned(self) -> None:
        """把 Profile 热专家预载进 CPU 紧凑缓存，其余容量继续作为 LRU。

        上游 v1.2.0 在 CPU 路径忽略 ``CCCP_PROFILE_JSON`` 的热度排序。随
        WINUI-EXE 发行的低内存版本把缓存的一部分用于固定热集，使语料生成的
        Profile 对 CPU 推理真正生效，同时绝不突破 ``--cache-gb`` 预算。
        """
        ranks = self.store.heat_ranks or {}
        if not ranks or self.budget <= 0 or self.pinned:
            return None
        full_load = os.environ.get("CCCP_PROFILE_FULL_LOAD", "0") != "0"
        try:
            fraction = (
                1.0 if full_load
                else float(os.environ.get("CCCP_PROFILE_PIN_FRACTION", "0.75"))
            )
        except ValueError:
            fraction = 1.0 if full_load else 0.75
        fraction = min(1.0 if full_load else 0.95, max(0.0, fraction))
        limit = int(self.budget * fraction)
        if limit <= 0:
            return None

        ordered: list[tuple[int, int]] = []
        depth = max((len(items) for items in ranks.values()), default=0)
        # 分层轮转，避免前几层先把预算耗尽。
        for index in range(depth):
            for layer in sorted(ranks):
                if index < len(ranks[layer]):
                    ordered.append((int(layer), int(ranks[layer][index])))

        loaded_bytes = 0
        required_bytes = 0
        loaded = 0
        compile_mode = os.environ.get("CCCP_CPU_COMPILE", "off").strip().lower()
        requested_compile = "u16" if compile_mode == "auto" else compile_mode
        compile_enabled = requested_compile in {"u16", "q4"}
        try:
            q4_hot_per_layer = max(
                int(self.store.cfg.get("top_k", 1)),
                int(os.environ.get("CCCP_CPU_Q4_HOT_PER_LAYER", "8")),
            )
        except ValueError:
            q4_hot_per_layer = max(int(self.store.cfg.get("top_k", 1)), 8)
        q4_ranks = self.store.q4_heat_ranks or ranks
        q4_hot_keys = {
            (int(layer), int(expert))
            for layer, experts in q4_ranks.items()
            for expert in experts[:q4_hot_per_layer]
        }
        layout_mode = os.environ.get("CCCP_CPU_PACKED_LAYOUT", "tile8").strip().lower()
        self.cpu_compile_mode = (
            f"q4-hot{q4_hot_per_layer}"
            if requested_compile == "q4"
            else requested_compile if compile_enabled else "off"
        )
        progress_total = max(1, len(ordered))
        progress_every = max(1, progress_total // 100)
        for progress_current, key in enumerate(ordered, 1):
            if (
                progress_current == 1
                or progress_current == progress_total
                or progress_current % progress_every == 0
            ):
                print(
                    "[cccp-winui-progress] phase=experts "
                    f"current={progress_current} total={progress_total} "
                    f"loaded={loaded} resident_gib={loaded_bytes / 2**30:.2f}",
                    flush=True,
                )
            if key in self.pinned:
                continue
            try:
                entry = self.store.load_expert_packed(*key)
            except (KeyError, ValueError, OSError):
                continue
            compile_this = compile_enabled and (
                requested_compile != "q4" or key in q4_hot_keys
            )
            if compile_this:
                for weight in entry:
                    source_bytes = int(weight.nbytes)
                    compiled = (
                        weight.compile_cpu_q4_0()
                        if requested_compile == "q4"
                        else weight.compile_cpu_u16_row_tile(8)
                    )
                    if not compiled:
                        raise RuntimeError(
                            f"Profile CPU 编译失败: layer/expert {key}"
                        )
                    self.compiled_source_bytes += source_bytes
                    self.compiled_index_bytes += int(weight.nbytes)
                    if requested_compile == "q4":
                        self.compiled_linear_bytes += int(weight.nbytes)
            elif layout_mode in {"tile8", "row-tile", "row_tile"}:
                for weight in entry:
                    if weight.optimize_cpu_row_tile(8):
                        self.block_major_entries += 1
                        self.block_major_bytes += int(weight.nbytes)
            size = self._entry_bytes(entry)
            required_bytes += size
            if size <= 0 or loaded_bytes + size > limit:
                continue
            self.pinned[key] = entry
            loaded_bytes += size
            loaded += 1
        self.bytes = loaded_bytes
        self.expanded_index_bytes = (
            int(self.compiled_index_bytes) if requested_compile == "u16" else 0
        )
        if full_load:
            required = len(set(ordered))
            if loaded != required:
                raise MemoryError(
                    f"Profile 要求全部常驻，但只加载 {loaded}/{required} 个专家；"
                    f"请把 --cache-gb 调大到至少 {required_bytes / 2**30:.2f} GiB 以上"
                )
            self.compact_full_resident = True
        print(
            "[cccp-winui] CPU Profile 热集预载："
            f"{loaded} 个专家 / {loaded_bytes / 2**30:.2f}GiB；"
            f"缓存预算 {self.budget / 2**30:.2f}GiB；"
            f"LRU 剩余 {(self.budget - loaded_bytes) / 2**30:.2f}GiB；"
            f"路由={'白名单' if self.store.route_allowlist is not None else '模型默认'}；"
            f"全部常驻={'是' if full_load else '否'}",
            f"cpu_compile={self.cpu_compile_mode}",
            flush=True,
        )
        return None

    def pin_host_resident(self, budget_gb: float | None = None) -> float:
        return 0.0

    def build_gpu_arenas(self) -> float:
        return 0.0

    def get_many(self, keys: list[tuple[int, int]]) -> dict:
        from concurrent.futures import as_completed

        output = {}
        missing = []
        for key in keys:
            entry = self.pinned.get(key)
            if entry is not None:
                self.hits += 1
                output[key] = entry
                continue
            entry = self.cache.get(key)
            if entry is not None:
                self.hits += 1
                self.cache.move_to_end(key)
                output[key] = entry
                continue
            missing.append(key)
        futures = {}
        for key in missing:
            future = self._pending.pop(key, None)
            if future is None:
                future = _executor().submit(
                    self.store.load_expert_packed,
                    *key,
                )
            futures[future] = key
        for future in as_completed(futures):
            key = futures[future]
            entry = future.result()
            self.miss += 1
            self._put(key, entry)
            output[key] = entry
        return output

    def prefetch(self, keys: list[tuple[int, int]]) -> None:
        if keys:
            self._enter_layer(int(keys[0][0]))
        self.prefetch_calls += 1
        self.prefetch_keys += len(keys)
        while len(self._pending) > 256:
            _, future = self._pending.popitem(last=False)
            future.cancel()
        for key in keys:
            if (
                key in self.pinned
                or key in self.cache
                or key in self._pending
            ):
                continue
            self._pending[key] = _pf_executor().submit(
                self.store.load_expert_packed,
                *key,
            )

    def run_native(
        self,
        layer: int,
        value: torch.Tensor,
        expert_ids: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float = 0.0,
    ) -> torch.Tensor | None:
        """Run one full-resident layer through the common native directory.

        This removes the per-token Python expert-list reconstruction while
        preserving the exact packed tensors held by ``pinned``. Uniform and
        mixed-codebook layers both use one format-driven native execution
        plan; ``None`` is reserved for unsupported/capacity fallbacks.
        """
        if not self.compact_full_resident:
            return None
        cached = self.native_layer(int(layer))
        if cached is None:
            return None
        output = cached.forward(
            value.float().contiguous(),
            expert_ids,
            route_weights.float().contiguous(),
            float(limit),
            str(activation).strip().lower(),
            float(activation_beta),
            (
                -1.0
                if activation_linear_beta is None
                else float(activation_linear_beta)
            ),
        )
        if output.numel():
            self.native_hits += 1
            return output
        self.native_fallbacks += 1
        return None

    def run_rows(
        self,
        layer: int,
        value: torch.Tensor,
        expert_ids: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float = 0.0,
    ) -> torch.Tensor:
        """Run CPU prefill rows in bounded exact-route micro-batches."""
        from .ops import packed_moe_selected_rows
        from .prefill import prefill_moe_batch_size

        if (
            value.is_cuda
            or value.ndim != 2
            or expert_ids.ndim != 2
            or route_weights.shape != expert_ids.shape
            or expert_ids.shape[0] != value.shape[0]
        ):
            raise ValueError("packed CPU prefill route shape mismatch")
        self._enter_layer(int(layer))
        rows = int(value.shape[0])
        top_k = int(expert_ids.shape[1])
        micro_batch = min(
            rows,
            prefill_moe_batch_size(default=256, maximum=4096),
        )
        outputs = []
        self.prefill_rows_calls += 1
        self.prefill_rows_tokens += rows
        for start in range(0, rows, micro_batch):
            stop = min(rows, start + micro_batch)
            ids = expert_ids[start:stop].to("cpu").long().contiguous()
            flat_ids = [int(item) for item in ids.reshape(-1).tolist()]
            selected = self.get_many(list(dict.fromkeys(
                (int(layer), expert_id) for expert_id in flat_ids
            )))
            nested = [
                [
                    selected[(int(layer), int(ids[row, slot]))]
                    for slot in range(top_k)
                ]
                for row in range(stop - start)
            ]
            result = packed_moe_selected_rows(
                value[start:stop],
                nested,
                route_weights[start:stop],
                activation=activation,
                activation_beta=activation_beta,
                activation_linear_beta=activation_linear_beta,
                limit=limit,
            )
            self.prefill_rows_micro_batches += 1
            if result is None:
                self.prefill_rows_fallbacks += 1
                raise RuntimeError("CPU packed prefill rows operator unavailable")
            outputs.append(result)
        return torch.cat(outputs, dim=0)

    def native_layer(self, layer: int):
        """Return one cached format-driven compact resident executor."""
        if not self.compact_full_resident:
            return None
        cached = self._native_layers.get(int(layer))
        if cached is None:
            n_experts = int(self.store.cfg["n_experts"])
            raw_entries = tuple(
                self.pinned.get((int(layer), expert))
                for expert in range(n_experts)
            )
            if self.store.route_allowlist is not None:
                # 原生目录仍按模型原始 expert_id 索引。白名单外槽位引用一个已
                # 常驻的占位专家（不复制张量）；路由 mask 保证这些槽位永不执行。
                fallback = next((entry for entry in raw_entries if entry is not None), None)
                if fallback is None:
                    self._native_layers[int(layer)] = False
                    return None
                entries = tuple(entry if entry is not None else fallback for entry in raw_entries)
            else:
                entries = raw_entries
            projection_counts = {
                len(entry) for entry in entries if entry is not None
            }
            if (
                any(entry is None for entry in entries)
                or len(projection_counts) != 1
                or next(iter(projection_counts), 0) not in {2, 3}
            ):
                self._native_layers[int(layer)] = False
                return None
            from .cpuext import (
                make_packed_three_layer_cpu,
                make_packed_two_layer_cpu,
            )

            cached = (
                make_packed_two_layer_cpu(entries)
                if next(iter(projection_counts)) == 2
                else make_packed_three_layer_cpu(entries)
            )
            self._native_layers[int(layer)] = cached or False
        if cached is False:
            return None
        return cached

    def prepare_native_layers(self) -> int:
        """Build every resident packed layer execution plan at load time.

        Codebook grouping, transposed lookup views and fixed Top-K workspaces
        are process-local runtime data. Preparing them here keeps the first
        decode token on the same native path as steady state and never writes
        a derived model or expands a logical expert matrix.
        """
        if not self.compact_full_resident:
            return 0
        prepared = 0
        for layer in sorted(
            int(value) for value in self.store.man.expert_files
        ):
            if self.native_layer(layer) is not None:
                prepared += 1
        return prepared


class ExpertPool:
    """专家缓存池（两级）：计算设备缓存 + 可选内存前级缓存。

    CPU 模式：单级内存缓存（budget_gb），未命中从磁盘加载。
    GPU 模式：显存主缓存（budget_gb）+ 内存前级缓存（ram_gb，远大于显存），
    显存未命中优先从内存前级上传（PCIe 快），前级也未命中才读磁盘。
    两级均以 VQ 索引态驻留（v 档 ≈9.4MB/专家，w 档 ≈4.7MB），LRU 驱逐。
    """

    def __init__(self, store: CCCPStore, budget_gb: float = 16.0, device: str = "cpu",
                 ram_gb: float = 0.0, pin_gb: float = 0.0):
        self.store = store
        self.device = torch.device(device)
        self.gpu = self.device.type != "cpu"
        self.budget = int(budget_gb * 2**30)
        self.ram_budget = int(ram_gb * 2**30)
        self.cache: OrderedDict[tuple[int, int], tuple[VQWeight, VQWeight]] = OrderedDict()
        self.ram: OrderedDict[tuple[int, int], tuple[VQWeight, VQWeight]] = OrderedDict()
        self.bytes = 0
        self.ram_bytes = 0
        self._host_pinned_bytes = 0
        self.hits = 0
        self.miss = 0
        self.stage = PinnedStage(self.device) if self.gpu else None
        self._stage_dirty = False   # 有预取 DMA 在飞（get/get_many 命中路径需先 wait）
        self._inflight: set = set()  # DMA 在飞的 cache key（落地前禁止驱逐：
        #   驱逐会释放目标显存被分配器复用，而在飞 DMA 继续写 → 随机覆写 KV/权重，
        #   曾致 prefill 后 hidden 全零、logits 全等 argmax 恒 0）
        self._pending: dict = {}    # 后台磁盘加载（预取软提示）：key → Future
        from collections import deque
        import threading
        self._recent: deque = deque(maxlen=256)   # 近期命中统计（0=hit 1=miss），预取自适应
        self._stage_lock = threading.RLock()      # staging/缓存突变互斥（后台 staging 线程安全）
        self._staging: dict = {}                  # 正在后台 staged 的 key → threading.Event
        self._gpu_arenas: GpuExpertArenas | None = None
        self._rebuilding_arenas = False
        self.full_resident = False
        # Expanded legacy VQ entries are resident but are not the byte-packed
        # CPU format consumed by ``prepare_native_layers``.
        self.compact_full_resident = False
        # Both backends have a true row-batched executor.  CUDA expands each
        # unique expert in bounded chunks and submits grouped GEMMs; legacy
        # CPU VQ archives group every routed row by expert and evaluate that
        # expert once for the complete row group.  Neither path may regress
        # to a token-by-token expert projection loop.
        self.prefill_rows_supported = True
        self.prefill_executor = (
            "cuda.chunked-dequant-grouped-gemm"
            if self.gpu
            else "cpu.expert-grouped-vq-rows"
        )
        self.prefill_batch_rows = 0
        self.prefill_batch_submissions = 0
        self.prefill_batch_max = 0
        self.prefill_expert_chunk_capacity = 0
        self.prefill_expert_chunk_submissions = 0
        self.prefill_layer_unique_max = 0
        self._prefill_executor_announced = False
        self._prefill_dequant_workspace = None
        # 热专家钉住区（永不驱逐）：按 profile 热度 top-N 准入，LRU 池只服务冷专家
        self.pinned: dict[tuple[int, int], tuple[VQWeight, VQWeight]] = {}
        self._pin_sets: dict[int, set[int]] = {}
        if pin_gb > 0 and store.heat_ranks:
            H = store.cfg.get("routed_hidden", store.cfg["hidden"])
            I = store.cfg["moe_inter"]
            est = 3 * I * H // 4  # v 档索引字节（上界）
            n_layers = max(1, len(store.man.expert_files))
            pin_n = int(pin_gb * 2**30 // (est * n_layers))
            if pin_n > 0:
                self._pin_sets = {l: set(r[:pin_n]) for l, r in store.heat_ranks.items()}
                print(f"[cccp] 热专家钉住: top-{pin_n}/层 ≈{pin_gb:.0f}GB", flush=True)

    def _hot(self, layer: int, eid: int) -> bool:
        s = self._pin_sets.get(layer)
        return s is not None and eid in s

    def preload_pinned(self) -> None:
        """启动时把钉住专家全部读入 RAM（消除逐 token 填充的冷启动拖尾）。

        只读盘入 RAM 钉住区，不上传显存（显存缓存由 decode 路径按需填充）。
        """
        full_profile = os.environ.get("CCCP_PROFILE_FULL_LOAD", "0") != "0"
        if full_profile:
            allowlist = self.store.route_allowlist
            if not allowlist:
                raise ValueError(
                    "CCCP_PROFILE_FULL_LOAD=1 需要启用含 allowed_experts 的严格路由配置"
                )
            # A generated profile is an exact residency contract, not merely a
            # heat hint.  Load every selected expert, including experts with a
            # zero count in a short calibration sample.  This makes the first
            # real prompt independent of disk latency.
            self._pin_sets = {
                int(layer): {int(expert) for expert in experts}
                for layer, experts in allowlist.items()
            }
        keys = [(l, e) for l, es in self._pin_sets.items() for e in es
                if (l, e) not in self.pinned]
        if not keys:
            return
        import time as _time
        from concurrent.futures import as_completed
        t0 = _time.time()
        fmap = {_executor().submit(self.store.load_expert, *k): k for k in keys}
        n = 0
        loaded_bytes = 0
        progress_every = max(1, len(keys) // 100)
        for fut in as_completed(fmap):
            key = fmap[fut]
            entry = fut.result()
            entry_bytes = entry[0].nbytes + entry[1].nbytes
            if full_profile and self.ram_budget > 0:
                if loaded_bytes + entry_bytes > self.ram_budget:
                    raise MemoryError(
                        "配置要求全部专家驻留 RAM，但专家缓存预算不足："
                        f"已读取 {n}/{len(keys)} 个、"
                        f"当前至少需要 {(loaded_bytes + entry_bytes) / 2**30:.2f} GiB，"
                        f"预算 {self.ram_budget / 2**30:.2f} GiB"
                    )
            self.pinned[key] = entry
            loaded_bytes += entry_bytes
            n += 1
            if n == 1 or n == len(keys) or n % progress_every == 0:
                print(
                    "[cccp-winui-progress] phase=experts "
                    f"current={n} total={len(keys)} "
                    f"resident_gib={loaded_bytes / 2**30:.2f}",
                    flush=True,
                )
        gb = sum(v[0].nbytes + v[1].nbytes for v in self.pinned.values()) / 2**30
        label = "配置专家" if full_profile else "热专家"
        print(
            f"[cccp] {label} RAM 预载完成（{n} 个 / {gb:.2f}GiB，"
            f"{_time.time() - t0:.1f}s）",
            flush=True,
        )
        if full_profile:
            self.full_resident = True

    @property
    def host_expert_bytes(self) -> int:
        """Physical host bytes retaining exact experts (pinned set + RAM LRU)."""
        pinned_bytes = sum(
            gu.nbytes + down.nbytes
            for gu, down in self.pinned.values()
        )
        return int(pinned_bytes + self.ram_bytes)

    def preload_profile_gpu(self) -> int:
        """Warm the selected profile into fixed GPU slots during startup.

        Disk/RAM loading and H2D upload are deliberately separate phases so
        the launcher remains responsive.  When VRAM cannot contain the whole
        profile, only the hottest capacity-safe subset is uploaded; every
        remaining expert stays resident in RAM and follows the exact route.
        """
        if (
            not self.gpu
            or os.environ.get("CCCP_PROFILE_FULL_LOAD", "0") == "0"
            or not self.store.route_allowlist
            or not self.pinned
            or self._gpu_arenas is None
        ):
            return 0

        allowlist = self.store.route_allowlist
        ranks = self.store.heat_ranks or {}
        per_layer: dict[int, list[int]] = {}
        for layer in sorted(allowlist):
            allowed = allowlist[layer]
            ranked = [
                int(expert)
                for expert in ranks.get(layer, ())
                if int(expert) in allowed
            ]
            seen = set(ranked)
            ranked.extend(sorted(int(expert) for expert in allowed if expert not in seen))
            per_layer[int(layer)] = ranked

        # Interleave heat ranks across layers.  A partial-VRAM preload must
        # not fill the arena with the first layers and leave later layers with
        # no hot experts; rank 0 of every layer is considered before rank 1.
        ordered: list[tuple[int, int]] = []
        max_rank = max((len(experts) for experts in per_layer.values()), default=0)
        for rank in range(max_rank):
            for layer, experts in per_layer.items():
                if rank < len(experts):
                    ordered.append((layer, experts[rank]))

        # Keep the hottest experts for each physical index signature.  This is
        # exact capacity accounting; it never invents or removes route experts.
        from collections import Counter

        admitted: list[tuple[int, int]] = []
        used_by_signature: Counter = Counter()
        for key in ordered:
            entry = self.pinned.get(key)
            if entry is None or not self._gpu_arenas.supports(entry):
                continue
            signature = ExpertSignature.of(entry)
            if used_by_signature[signature] >= self._gpu_arenas.capacity(entry):
                continue
            admitted.append(key)
            used_by_signature[signature] += 1

        total = len(ordered)
        if not admitted:
            return 0
        started = time.time()
        batch_size = 64
        uploaded = 0
        for begin in range(0, len(admitted), batch_size):
            batch_keys = admitted[begin:begin + batch_size]
            batch_entries = [self.pinned[key] for key in batch_keys]
            staged = self._stage_ents(batch_keys, batch_entries)
            with self._stage_lock:
                self._inflight.update(batch_keys)
                self._stage_dirty = True
                for key, entry in zip(batch_keys, staged):
                    self._put(key, entry)
                self._wait_stage()
            uploaded += len(batch_keys)
            print(
                "[cccp-winui-progress] phase=expert-upload "
                f"current={uploaded} total={len(admitted)} "
                f"profile_total={total}",
                flush=True,
            )

        suffix = "全部配置专家已在 GPU" if uploaded == total else (
            f"另有 {total - uploaded} 个配置专家驻留 RAM，按需上传"
        )
        print(
            f"[cccp] 配置专家 GPU 预热完成：{uploaded}/{total} 个，"
            f"{time.time() - started:.1f}s；{suffix}",
            flush=True,
        )
        return uploaded

    def _prefill_dequant_chunk_capacity(self, expert_count: int) -> int:
        """Size the GLM Prefill expert expansion from live VRAM headroom."""
        hidden = int(self.store.cfg.get("routed_hidden") or self.store.cfg["hidden"])
        intermediate = int(self.store.cfg["moe_inter"])
        bytes_per_expert = 6 * hidden * intermediate  # BF16 GU [2I,H] + DN [H,I]
        try:
            explicit_limit = float(os.environ.get("CCCP_VRAM_LIMIT_GB", "0"))
        except (TypeError, ValueError):
            explicit_limit = 0.0
        if explicit_limit > 0:
            available = max(
                0,
                int(explicit_limit * 2**30)
                - int(torch.cuda.memory_allocated(self.device)),
            )
        else:
            available, _total = torch.cuda.mem_get_info(self.device)
            available = int(available)
        # Grouped input, activation and output rows remain full-token tensors.
        # Reserve them separately so expanding more experts cannot starve the
        # tensor-core GEMMs that consume the expansion.
        # A 4K/top-8 layer additionally needs roughly 0.4 GiB each for
        # grouped-input, Gate/Up and Down rows, plus FP32 route reduction and
        # allocator slack.  Keep 2.5 GiB outside the dequant slab; this yields
        # about 16--18 legacy GLM experts per chunk on a 24 GiB card.
        safety = max(2560 * 2**20, int(available * 0.25))
        automatic = max(1, (available - safety) // max(1, bytes_per_expert))
        try:
            requested = int(os.environ.get("CCCP_PREFILL_DEQUANT_EXPERTS", "0"))
        except (TypeError, ValueError):
            requested = 0
        if requested > 0:
            automatic = min(automatic, requested)
        return max(1, min(int(expert_count), int(automatic)))

    def _prefill_dequant_workspace_for(
        self,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = int(self.store.cfg.get("routed_hidden") or self.store.cfg["hidden"])
        intermediate = int(self.store.cfg["moe_inter"])
        cached = self._prefill_dequant_workspace
        if cached is not None and int(cached[0].shape[0]) >= int(capacity):
            return cached
        cached = (
            torch.empty(
                int(capacity),
                2 * intermediate,
                hidden,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.empty(
                int(capacity),
                hidden,
                intermediate,
                dtype=torch.bfloat16,
                device=self.device,
            ),
        )
        self._prefill_dequant_workspace = cached
        return cached

    @staticmethod
    def _legacy_prefill_metadata_rows(
        experts: list[tuple[VQWeight, VQWeight]],
    ) -> list[list[int]]:
        """Build [10,E] pointer metadata for combined-GU legacy VQ experts."""
        if not experts:
            return []

        def rows(projection: int) -> list[list[int]]:
            weights = [expert[projection] for expert in experts]
            codebooks = [
                cb_compute(weight.cb, torch.bfloat16).contiguous()
                for weight in weights
            ]
            tags = []
            for weight in weights:
                if weight.idx.dtype == torch.uint8:
                    tags.append(0)
                elif weight.idx.dtype == torch.uint16:
                    tags.append(1)
                else:
                    raise TypeError(
                        f"unsupported legacy VQ index dtype {weight.idx.dtype}"
                    )
            return [
                [weight.idx.data_ptr() for weight in weights],
                [codebook.data_ptr() for codebook in codebooks],
                [int(weight.idx.shape[1]) for weight in weights],
                [int(weight.dim) for weight in weights],
                tags,
            ]

        return rows(0) + rows(1)

    def run_rows(
        self,
        layer: int,
        value: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float = 0.0,
    ) -> torch.Tensor:
        """Run a complete GLM Prefill layer with a row-batched executor.

        CUDA expands each selected expert once per layer chunk, then all of
        its routed token rows execute in two tensor-core grouped GEMMs.  CPU
        groups the complete token block by expert and executes two batched VQ
        projections per unique expert.  Decode GEMV is structurally
        unreachable from either Prefill path.
        """
        if not self.gpu:
            return self._run_rows_cpu(
                layer,
                value,
                route_ids,
                route_weights,
                activation=activation,
                activation_beta=activation_beta,
                activation_linear_beta=activation_linear_beta,
                limit=limit,
            )
        if value.ndim != 2 or value.shape[0] <= 1:
            raise ValueError("GPU grouped Prefill requires value [N,D] with N>1")
        if route_ids.ndim != 2 or route_weights.shape != route_ids.shape:
            raise ValueError("GPU grouped Prefill routes/weights must be matching [N,K]")
        if not self._prefill_executor_announced:
            print(
                "[cccp-prefill] executor=cuda.chunked-dequant-grouped-gemm; "
                "outer token batch=complete; single-token projection=forbidden",
                flush=True,
            )
            self._prefill_executor_announced = True

        rows = int(value.shape[0])
        top_k = int(route_ids.shape[1])
        flat_ids = route_ids.reshape(-1)
        unique_global, unique_counts = torch.unique(
            flat_ids,
            sorted=True,
            return_counts=True,
        )
        unique_ids = [int(item) for item in unique_global.detach().cpu().tolist()]
        unique_count = len(unique_ids)
        local_route_ids = torch.searchsorted(unique_global, route_ids).contiguous()
        flat_local_ids = local_route_ids.reshape(-1)
        token_ids = (
            torch.arange(rows, dtype=torch.long, device=self.device)
            .view(-1, 1)
            .expand(rows, top_k)
            .reshape(-1)
        )
        flat_weights = route_weights.reshape(-1).float()
        input_rows = value.to(torch.bfloat16).contiguous()
        result = torch.zeros(
            rows,
            int(value.shape[1]),
            dtype=torch.float32,
            device=self.device,
        )
        chunk_capacity = self._prefill_dequant_chunk_capacity(unique_count)
        self.prefill_expert_chunk_capacity = max(
            self.prefill_expert_chunk_capacity,
            chunk_capacity,
        )
        self.prefill_layer_unique_max = max(
            self.prefill_layer_unique_max,
            unique_count,
        )
        print(
            f"[cccp-prefill] layer={layer}; token batch={rows}; "
            f"unique experts={unique_count}; expert chunk={chunk_capacity}; "
            "capacity=automatic free-VRAM",
            flush=True,
        )
        from .grouped import activate_gate_up
        from .ops import projection_dequant

        for chunk_start in range(0, unique_count, chunk_capacity):
            chunk_stop = min(unique_count, chunk_start + chunk_capacity)
            keys = [
                (int(layer), expert)
                for expert in unique_ids[chunk_start:chunk_stop]
            ]
            selected = self.get_many(keys)
            experts = [selected[key] for key in keys]
            chunk_count = len(experts)
            gu_buffer, down_buffer = self._prefill_dequant_workspace_for(chunk_count)
            metadata = torch.tensor(
                self._legacy_prefill_metadata_rows(experts),
                dtype=torch.long,
                device=self.device,
            )
            projection_dequant(
                metadata,
                gu_buffer[:chunk_count],
                down_buffer[:chunk_count],
            )

            selected_positions = (
                (flat_local_ids >= chunk_start)
                & (flat_local_ids < chunk_stop)
            ).nonzero(as_tuple=False).reshape(-1)
            chunk_ids = flat_local_ids.index_select(0, selected_positions) - chunk_start
            order = torch.argsort(chunk_ids)
            sorted_ids = chunk_ids.index_select(0, order).contiguous()
            sorted_positions = selected_positions.index_select(0, order).contiguous()
            sorted_tokens = token_ids.index_select(0, sorted_positions).contiguous()
            sorted_weights = flat_weights.index_select(0, sorted_positions).contiguous()
            group_ids = torch.arange(
                chunk_count,
                dtype=torch.long,
                device=self.device,
            )
            offsets = torch.searchsorted(
                sorted_ids,
                group_ids,
                right=True,
            ).to(torch.int32)
            grouped_input = input_rows.index_select(0, sorted_tokens)
            gate_up = torch._grouped_mm(
                grouped_input,
                gu_buffer[:chunk_count].transpose(1, 2),
                offs=offsets,
            )
            gate, up = gate_up.chunk(2, dim=-1)
            if float(limit) > 0.0:
                gate = gate.clamp(max=float(limit))
                up = up.clamp(min=-float(limit), max=float(limit))
            activated = activate_gate_up(
                gate,
                up,
                activation=activation,
                situ_beta=float(activation_beta),
                situ_linear_beta=(
                    None
                    if activation_linear_beta is None
                    else float(activation_linear_beta)
                ),
            )
            down = torch._grouped_mm(
                activated.contiguous(),
                down_buffer[:chunk_count].transpose(1, 2),
                offs=offsets,
            )
            # Converting every routed row to FP32 at once costs
            # T*top_k*hidden*4 bytes (about 770 MiB for GLM 4K).  Keep the
            # exact FP32 route accumulation but bound only this linear
            # workspace; the outer token/expert GEMMs remain fully batched.
            try:
                route_row_batch = max(
                    1,
                    int(os.environ.get("CCCP_PREFILL_ROUTE_ROWS", "2048")),
                )
            except ValueError:
                route_row_batch = 2048
            for route_start in range(0, int(down.shape[0]), route_row_batch):
                route_stop = min(
                    int(down.shape[0]),
                    route_start + route_row_batch,
                )
                weighted = down[route_start:route_stop].float()
                weighted.mul_(
                    sorted_weights[route_start:route_stop].unsqueeze(1)
                )
                result.index_add_(
                    0,
                    sorted_tokens[route_start:route_stop],
                    weighted,
                )
            self.prefill_expert_chunk_submissions += 1

        self.prefill_batch_submissions += 1
        self.prefill_batch_rows += rows
        self.prefill_batch_max = max(self.prefill_batch_max, rows)
        self._prefill_dequant_workspace = None
        return result

    def _run_rows_cpu(
        self,
        layer: int,
        value: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float = 0.0,
    ) -> torch.Tensor:
        """Execute legacy VQ Prefill once per unique expert, never per token."""
        if (
            value.is_cuda
            or value.ndim != 2
            or value.shape[0] <= 1
            or route_ids.ndim != 2
            or route_weights.shape != route_ids.shape
            or route_ids.shape[0] != value.shape[0]
        ):
            raise ValueError("CPU grouped Prefill routes must match value [N,D]")
        if not self._prefill_executor_announced:
            print(
                "[cccp-prefill] executor=cpu.expert-grouped-vq-rows; "
                "outer token batch=complete; single-token projection=forbidden",
                flush=True,
            )
            self._prefill_executor_announced = True

        from .grouped import activate_gate_up
        from .cpuext import vq_gemv_list_cpu

        rows = int(value.shape[0])
        top_k = int(route_ids.shape[1])
        flat_ids = route_ids.to("cpu").long().reshape(-1)
        flat_weights = route_weights.to("cpu").float().reshape(-1)
        token_ids = (
            torch.arange(rows, dtype=torch.long)
            .view(-1, 1)
            .expand(rows, top_k)
            .reshape(-1)
        )
        unique_ids = [int(item) for item in torch.unique(flat_ids).tolist()]
        keys = [(int(layer), expert) for expert in unique_ids]
        selected = self.get_many(keys)
        result = torch.zeros(rows, int(value.shape[1]), dtype=torch.float32)
        source = value.to("cpu").float().contiguous()

        for expert in unique_ids:
            positions = (flat_ids == expert).nonzero(as_tuple=False).reshape(-1)
            expert_tokens = token_ids.index_select(0, positions)
            expert_weights = flat_weights.index_select(0, positions)
            bundle = selected[(int(layer), expert)]
            if len(bundle) != 2 or not all(
                isinstance(weight, VQWeight) for weight in bundle
            ):
                raise RuntimeError(
                    "legacy CPU grouped Prefill requires two VQ projections"
                )
            gate_up, down = bundle
            expert_source = source.index_select(0, expert_tokens)
            hidden = vq_gemv_list_cpu(
                expert_source,
                [gate_up.idx] * int(expert_source.shape[0]),
                gate_up.cb,
            )
            if hidden is None:
                hidden = gate_up.matmul_T(expert_source)
            gate, up = hidden.chunk(2, dim=-1)
            if float(limit) > 0.0:
                gate.clamp_(max=float(limit))
                up.clamp_(min=-float(limit), max=float(limit))
            activated = activate_gate_up(
                gate,
                up,
                activation=activation,
                situ_beta=float(activation_beta),
                situ_linear_beta=(
                    None
                    if activation_linear_beta is None
                    else float(activation_linear_beta)
                ),
            )
            projected = vq_gemv_list_cpu(
                activated,
                [down.idx] * int(activated.shape[0]),
                down.cb,
            )
            if projected is None:
                projected = down.matmul_T(activated)
            projected.mul_(expert_weights.unsqueeze(1))
            result.index_add_(0, expert_tokens, projected)

        self.prefill_batch_submissions += 1
        self.prefill_batch_rows += rows
        self.prefill_batch_max = max(self.prefill_batch_max, rows)
        self.prefill_layer_unique_max = max(
            self.prefill_layer_unique_max,
            len(unique_ids),
        )
        return result

    def preload_all(self, reserve_gb: float | None = None) -> bool:
        """启动时尝试把全部专家常驻 RAM（钉住，永不驱逐，之后零磁盘读）。

        判定：专家文件总量 ×1.05 + 预留 ≤ 当前可用物理内存。
        满足 → 并行全量读入 pinned 区，返回 True；
        不满足 → 打印醒目警告（列出缺口与建议）并返回 False，调用方回退
        热专家钉住 + LRU 按需加载。可用 CCCP_FULL_RESIDENT=0 关闭本行为，
        CCCP_RESIDENT_RESERVE_GB 调整预留（默认 2GB）。
        """
        # A strict generated profile has its own exact expert set.  Do not
        # accidentally read the entire unfiltered model before loading it.
        if (
            os.environ.get("CCCP_PROFILE_FULL_LOAD", "0") != "0"
            and self.store.route_allowlist
        ):
            return False
        if os.environ.get("CCCP_FULL_RESIDENT", "1") == "0":
            return False
        import psutil
        import time as _time
        from concurrent.futures import as_completed
        if reserve_gb is None:
            reserve_gb = float(os.environ.get("CCCP_RESIDENT_RESERVE_GB", "2.0"))
        root = self.store.root
        total = 0
        for fn in self.store.man.expert_files.values():
            p = os.path.join(root, fn)
            if os.path.exists(p):
                total += os.path.getsize(p)
        total_gb = total / 2**30 * 1.05
        avail_gb = psutil.virtual_memory().available / 2**30
        if total_gb + reserve_gb > avail_gb:
            print(f"[cccp] 警告：无法全量常驻专家：专家 {total_gb:.1f}GB + 预留 {reserve_gb:.1f}GB"
                  f" > 可用内存 {avail_gb:.1f}GB（差 {total_gb + reserve_gb - avail_gb:.1f}GB）。\n"
                  f"       将按需加载（LRU + 磁盘读，多轮后命中率上升）。\n"
                  f"       想全常驻：关闭其他占内存程序 / 加内存条 / "
                  f"调小 CCCP_RESIDENT_RESERVE_GB。", flush=True)
            return False
        n_experts = self.store.cfg["n_experts"]
        keys = [(l, e) for l in (int(x) for x in self.store.man.expert_files)
                for e in range(n_experts)
                if self.store.expert_kind(l, e) != "drop"]
        t0 = _time.time()
        print(f"[cccp] 全量专家常驻：{len(keys)} 个 / ≈{total_gb:.1f}GB 读盘中…", flush=True)
        fmap = {_executor().submit(self.store.load_expert, *k): k for k in keys}
        n = 0
        for fut in as_completed(fmap):
            self.pinned[fmap[fut]] = fut.result()
            n += 1
            if n % 2000 == 0:
                print(f"[cccp] 全量常驻 {n}/{len(keys)}", flush=True)
        gb = sum(v[0].nbytes + v[1].nbytes for v in self.pinned.values()) / 2**30
        print(f"[cccp] 全量专家常驻完成（{n} 个 / {gb:.1f}GB，{_time.time() - t0:.0f}s），"
              f"之后推理零磁盘读", flush=True)
        self.full_resident = True
        return True

    def pin_host_resident(self, budget_gb: float | None = None) -> float:
        """把常驻 RAM 专家索引转换为真正的 CUDA page-locked 内存。

        普通 ``preload_all`` 仅表示 Python 强引用常驻，内存仍是 pageable；每次显存
        miss 都要先复制进 PinnedStage 槽，再做 DMA。这里替换 idx 为 pin_memory()
        副本后，上传可直接异步 DMA，省掉逐 token 的 CPU memcpy 与槽位等待。

        CCCP_HOST_PIN_GB 默认 auto：仅当转换后仍能保留足够可用 RAM 时全量启用；
        数字指定 GiB 上限，0 明确关闭。转换逐专家替换，峰值只多一个专家而非再
        复制整个模型。
        """
        if not self.gpu or not self.pinned:
            self._host_pinned_bytes = 0
            return 0.0
        total = sum(g.nbytes + d.nbytes for g, d in self.pinned.values())
        if budget_gb is None:
            raw = os.environ.get("CCCP_HOST_PIN_GB", "auto").strip().lower()
            if raw in ("", "auto"):
                import psutil
                avail = psutil.virtual_memory().available
                reserve = 2 * 2**30
                if avail < total + reserve:
                    self._host_pinned_bytes = 0
                    print("[cccp] 锁页专家内存自动关闭："
                          f"可用 {avail / 2**30:.1f}GB < 专家 {total / 2**30:.1f}GB"
                          f" + 安全余量 {reserve / 2**30:.1f}GB",
                          flush=True)
                    return 0.0
                budget = total
                print(f"[cccp] 锁页专家内存自动启用：{total / 2**30:.1f}GB",
                      flush=True)
            else:
                budget = max(0, int(float(raw) * 2**30))
        else:
            budget = max(0, int(budget_gb * 2**30))
        if budget == 0:
            self._host_pinned_bytes = 0
            return 0.0

        import time as _time
        t0 = _time.time()
        pinned_bytes = 0
        pinned_count = 0
        for key in self.pinned:
            gu, dn = self.pinned[key]
            nb = gu.nbytes + dn.nbytes
            if pinned_bytes + nb > budget:
                continue
            try:
                gu_idx = gu.idx if gu.idx.is_pinned() else gu.idx.pin_memory()
                dn_idx = dn.idx if dn.idx.is_pinned() else dn.idx.pin_memory()
            except RuntimeError as exc:
                print(f"[cccp] 锁页专家内存停止于 {pinned_bytes / 2**30:.1f}GB：{exc}",
                      flush=True)
                break
            self.pinned[key] = (
                VQWeight(gu_idx, gu.cb, gu.cols),
                VQWeight(dn_idx, dn.cb, dn.cols),
            )
            pinned_bytes += nb
            pinned_count += 1
            if pinned_count % 2000 == 0:
                print(f"[cccp] 锁页专家内存 {pinned_count}/{len(self.pinned)} "
                      f"({pinned_bytes / 2**30:.1f}GB)", flush=True)
        print(f"[cccp] 锁页专家内存完成：{pinned_count} 个 / "
              f"{pinned_bytes / 2**30:.1f}GB（{_time.time() - t0:.1f}s）",
              flush=True)
        self._host_pinned_bytes = pinned_bytes
        if pinned_bytes >= total:
            print(
                "[cccp-dma] mode=direct-pinned cpu_bridge=disabled "
                f"locked={pinned_bytes / 2**30:.2f}GiB；"
                "GPU 直接从锁页专家 RAM 发起异步 DMA，不经过 CPU 中转缓冲",
                flush=True,
            )
        else:
            print(
                "[cccp-dma] mode=mixed cpu_bridge=fallback-for-unlocked "
                f"locked={pinned_bytes / 2**30:.2f}GiB "
                f"pageable={(total - pinned_bytes) / 2**30:.2f}GiB；"
                "仅未锁页部分使用 CPU 中转缓冲",
                flush=True,
            )
        return pinned_bytes / 2**30

    def build_gpu_arenas(self) -> float:
        """按显存缓存预算一次性分配稳定的专家索引槽。

        码本仍由 ``_cb_dev`` 按层共享；arena 只保存占显存主体的 GU/DN 索引。
        每种索引 shape/dtype 独立成池，并按模型中该签名的专家数量等比例分配，
        后续 miss 只覆盖槽位视图，不再调用 CUDA allocator。
        """
        if not self.gpu:
            return 0.0
        if self._gpu_arenas is not None:
            return self._gpu_arenas.nbytes / 2**30
        entries = list(self.pinned.values())
        if not entries:
            entries = list(self.ram.values())
        if not entries or self.budget <= 0:
            return 0.0

        # Fixed arenas allocate their full capacity immediately.  Unlike the
        # historical lazy LRU, a requested 20 GiB cache therefore cannot be
        # created blindly after 13+ GiB of dense BF16 weights are resident on a
        # 32 GiB card.  Clamp from the live allocator/device state first.
        allocated_bytes = torch.cuda.memory_allocated(self.device)
        device_free_bytes, device_total_bytes = torch.cuda.mem_get_info(
            self.device
        )
        device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        try:
            process_fraction = torch.cuda.get_per_process_memory_fraction(
                device_index
            )
        except (AttributeError, RuntimeError):
            process_fraction = 1.0
        process_limit_bytes = int(device_total_bytes * process_fraction)
        reserve_gb = float(os.environ.get(
            "CCCP_VRAM_HEADROOM_GB",
            os.environ.get("CCCP_VRAM_RESERVE_GB", "1"),
        ))
        safe_budget = _safe_arena_budget(
            requested_bytes=self.budget,
            allocated_bytes=allocated_bytes,
            device_free_bytes=device_free_bytes,
            process_limit_bytes=process_limit_bytes,
            reserve_bytes=int(reserve_gb * 2**30),
        )
        if safe_budget < self.budget:
            requested_gb = self.budget / 2**30
            self.budget = safe_budget
            print(
                f"[cccp] 固定专家槽分配前封顶：{requested_gb:.1f}GB"
                f" → {safe_budget / 2**30:.1f}GB"
                f"（dense/已分配 {allocated_bytes / 2**30:.1f}GB"
                f" + 运行时余量 {reserve_gb:.1f}GB）",
                flush=True,
            )
        if self.budget <= 0:
            return 0.0

        from collections import Counter

        counts = Counter(ExpertSignature.of(entry) for entry in entries)
        total_model_bytes = sum(
            signature.slot_bytes * count
            for signature, count in counts.items()
        )
        if total_model_bytes <= 0:
            return 0.0
        scale = min(1.0, self.budget / total_model_bytes)
        minimum = max(1, int(self.store.cfg.get("top_k", 6)))
        allocated = {
            signature: min(count, max(minimum, int(count * scale)))
            for signature, count in counts.items()
        }

        def used_bytes() -> int:
            return sum(
                signature.slot_bytes * count
                for signature, count in allocated.items()
            )

        # 极小预算时先收缩到可分配范围；正常服务器预算远高于每签名 top-k。
        while used_bytes() > self.budget:
            candidates = [
                signature
                for signature, count in allocated.items()
                if count > 1
            ]
            if not candidates:
                return 0.0
            largest = max(
                candidates,
                key=lambda signature: signature.slot_bytes * allocated[signature],
            )
            allocated[largest] -= 1

        # 利用取整后的余量，优先补齐当前覆盖率最低的签名。
        while True:
            candidates = [
                signature
                for signature, count in counts.items()
                if allocated[signature] < count
                and used_bytes() + signature.slot_bytes <= self.budget
            ]
            if not candidates:
                break
            next_signature = min(
                candidates,
                key=lambda signature: allocated[signature] / counts[signature],
            )
            allocated[next_signature] += 1

        self._gpu_arenas = GpuExpertArenas(
            allocated.items(),
            self.device,
        )
        gb = self._gpu_arenas.nbytes / 2**30
        detail = ", ".join(
            f"{signature.gu_dtype}:{count}"
            for signature, count in allocated.items()
        )
        print(f"[cccp] 固定专家显存槽：{sum(allocated.values())} 个 / "
              f"{gb:.2f}GB（{detail}）", flush=True)
        return gb

    def _release_gpu_key(self, key) -> None:
        arenas = getattr(self, "_gpu_arenas", None)
        if arenas is not None:
            arenas.release(key)

    @property
    def gpu_storage_bytes(self) -> int:
        """当前已实际分配的专家索引显存（动态 cache 或固定 arena）。"""
        arenas = getattr(self, "_gpu_arenas", None)
        if arenas is None:
            return self.bytes
        dynamic = sum(
            gu.nbytes + dn.nbytes
            for key, (gu, dn) in self.cache.items()
            if not arenas.owns(key)
        )
        return arenas.nbytes + dynamic

    @property
    def gpu_arena_bytes(self) -> int:
        """固定专家 arena 当前真实占用；区别于可动态下调的逻辑预算。"""
        arenas = getattr(self, "_gpu_arenas", None)
        return 0 if arenas is None else arenas.nbytes

    def _drain_staging_for_arena_resize(self, timeout_s: float) -> None:
        """等待缩容前已经提交的后台 staging，等待期间不持有池锁。"""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            with self._stage_lock:
                events = list(self._staging.values())
            if not events:
                return
            for event in events:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0 or not event.wait(remaining):
                    with self._stage_lock:
                        pending = len(self._staging)
                    raise RuntimeError(
                        f"timed out draining {pending} staging expert batches "
                        "before arena resize"
                    )

    def resize_gpu_arenas(
        self,
        budget: int,
        *,
        staging_timeout_s: float = 30.0,
    ) -> tuple[int, int]:
        """在更小预算下物理重建固定专家槽，并返回 (旧字节数, 新字节数)。"""
        budget = max(0, int(budget))
        arenas = getattr(self, "_gpu_arenas", None)
        old_bytes = 0 if arenas is None else arenas.nbytes
        if not self.gpu or arenas is None or budget >= old_bytes:
            self.trim_to(budget)
            return old_bytes, old_bytes

        with self._stage_lock:
            self._rebuilding_arenas = True
        try:
            self._drain_staging_for_arena_resize(staging_timeout_s)
            torch.cuda.synchronize(self.device)
            with self._stage_lock:
                if self._stage_dirty:
                    self._wait_stage()
                if self._staging:
                    raise RuntimeError(
                        "cannot resize expert arenas while staging is active"
                    )
                # cache 中的 VQWeight.idx 是 arena GU/DN 张量的视图；必须先清空，
                # 再删除 arena 本体，才能让 caching allocator 看到真实可释放块。
                self.cache.clear()
                self.bytes = 0
                self._inflight.clear()
                self._gpu_arenas = None
                self.budget = budget
            # 局部变量同样持有旧 arena；不删除它，empty_cache 仍无法归还显存。
            del arenas
            torch.cuda.empty_cache()
            try:
                self.build_gpu_arenas()
            except torch.cuda.OutOfMemoryError:
                # 部分构造产生的临时张量在异常展开后已失去引用；清缓存并只降档一次，
                # 避免在显存压力下反复 OOM。
                self._gpu_arenas = None
                torch.cuda.empty_cache()
                retry_budget = max(2**29, budget // 2)
                if retry_budget >= budget:
                    raise
                self.budget = retry_budget
                self.build_gpu_arenas()
            new_bytes = self.gpu_arena_bytes
            if new_bytes > self.budget:
                raise RuntimeError(
                    "rebuilt expert arenas exceed budget: "
                    f"{new_bytes} > {self.budget}"
                )
            return old_bytes, new_bytes
        finally:
            with self._stage_lock:
                self._rebuilding_arenas = False

    def _touch_gpu_key(self, key) -> None:
        arenas = getattr(self, "_gpu_arenas", None)
        if arenas is not None:
            arenas.touch(key)

    def _lease_gpu_ent(self, key, cpu_ent, *, use_arena: bool = True):
        """返回 arena 目标视图；不支持的签名返回 None 走动态分配回退。"""
        arenas = getattr(self, "_gpu_arenas", None)
        if not use_arena or arenas is None or not arenas.supports(cpu_ent):
            return None
        lease, gu_idx, dn_idx = arenas.lease(key, cpu_ent)
        if lease.replaced is not None:
            old = self.cache.pop(lease.replaced, None)
            if old is not None:
                self.bytes -= old[0].nbytes + old[1].nbytes
            self._inflight.discard(lease.replaced)
        arenas.mark_inflight(key)
        self._inflight.add(key)
        gu, dn = cpu_ent
        return (
            (gu.idx, gu_idx),
            (dn.idx, dn_idx),
            (
                VQWeight(gu_idx, self._cb_dev(gu.cb), gu.cols),
                VQWeight(dn_idx, self._cb_dev(dn.cb), dn.cols),
            ),
        )

    def _evict(self, d: OrderedDict, size_ref: str, budget: int, need: int,
               skip_inflight: bool = False) -> None:
        scanned = 0
        while getattr(self, size_ref) + need > budget and d:
            if skip_inflight and scanned < len(d):
                key = next(iter(d))
                if key in self._inflight:
                    # DMA 在飞的条目不可驱逐（显存复用会被 DMA 覆写），顺延为最新
                    d.move_to_end(key)
                    scanned += 1
                    continue
            elif skip_inflight:
                break   # 全部在飞：宁可暂超预算也不驱逐（安全优先）
            key, (g, dd) = d.popitem(last=False)
            if d is self.cache:
                self.bytes -= g.nbytes + dd.nbytes
                self._release_gpu_key(key)
            else:
                self.ram_bytes -= g.nbytes + dd.nbytes

    def trim_to(self, budget: int) -> None:
        """动态收紧显存缓存预算并立即按 LRU 驱逐到预算内（VramWatch 止血用）。
        先等在飞 DMA 落地：否则驱逐会释放 DMA 目标显存，被覆写后数据随机损坏。"""
        with self._stage_lock:
            if self.gpu and self._stage_dirty:
                self._wait_stage()
            self.budget = max(0, int(budget))
            self._evict(self.cache, "bytes", self.budget, 0)

    def _put(self, key, ent) -> None:
        with self._stage_lock:
            nb = ent[0].nbytes + ent[1].nbytes
            old = self.cache.pop(key, None)
            if old is not None:
                self.bytes -= old[0].nbytes + old[1].nbytes
            self._evict(self.cache, "bytes", self.budget, nb, skip_inflight=self.gpu)
            self.cache[key] = ent
            self.bytes += nb

    def _put_ram(self, key, ent) -> None:
        if self.ram_budget <= 0:
            return
        with self._stage_lock:
            nb = ent[0].nbytes + ent[1].nbytes
            old = self.ram.pop(key, None)
            if old is not None:
                self.ram_bytes -= old[0].nbytes + old[1].nbytes
            self._evict(self.ram, "ram_bytes", self.ram_budget, nb)
            self.ram[key] = ent
            self.ram_bytes += nb

    def _cb_dev(self, cb: torch.Tensor) -> torch.Tensor:
        """层共享码本的设备副本（按 data_ptr 恒等缓存：消除每专家重复的码本小上传）。

        必须强引用 CPU 码本：并行加载时 codebooks() 竞态会产生重复码本张量，
        落选者被 LRU 驱逐释放后其 data_ptr 可能被后续分配复用——若只按裸 ptr
        缓存键，新层码本会命中旧指针，返回**别的层的码本**（GLM 实测 KL 8.9 /
        输出乱码复读的根因）。强引用使 ptr 永不复用，键恒有效。"""
        if not hasattr(self, "_cb_devs"):
            self._cb_devs = {}
        key = cb.data_ptr()
        ent = self._cb_devs.get(key)
        if ent is None:
            d = cb.to(self.device)
            ent = (d, cb)          # (设备副本, CPU 强引用防 ptr 复用)
            self._cb_devs[key] = ent
        return ent[0]

    def _stage_ent(self, key, cpu_ent) -> tuple:
        """CPU 专家 (VQWeight, VQWeight) 经 pinned 分段上传到 GPU（码本随行）。
        CCCP_STAGE_SYNC=1（诊断）：走默认流同步 .to() 直传，绕过 pinned/DMA 机制。
        全程持 _stage_lock：与后台 staging 线程互斥，槽位轮转才不乱。"""
        with self._stage_lock:
            leased = self._lease_gpu_ent(key, cpu_ent)
            if leased is not None:
                gu_pair, dn_pair, out = leased
                if os.environ.get("CCCP_STAGE_SYNC", "0") != "0":
                    gu_pair[1].copy_(gu_pair[0])
                    dn_pair[1].copy_(dn_pair[0])
                    self._gpu_arenas.clear_inflight(key)
                    self._inflight.discard(key)
                else:
                    self.stage.upload_batch([gu_pair, dn_pair])
                return out
            if os.environ.get("CCCP_STAGE_SYNC", "0") != "0":
                return tuple(VQWeight(vq.idx.to(self.device), self._cb_dev(vq.cb), vq.cols)
                             for vq in cpu_ent)
            out = []
            for vq in cpu_ent:
                idx_d = torch.empty_like(vq.idx, device=self.device)
                self.stage.upload(vq.idx, idx_d)
                out.append(VQWeight(idx_d, self._cb_dev(vq.cb), vq.cols))
            return tuple(out)

    def _stage_ents(self, keys: list, cpu_ents: list) -> list:
        """_stage_ent 的成批版：全部索引一次 upload_batch（少 stream/事件开销）。
        全程持 _stage_lock（见上）。"""
        if os.environ.get("CCCP_STAGE_SYNC", "0") != "0":
            return [self._stage_ent(k, e) for k, e in zip(keys, cpu_ents)]
        with self._stage_lock:
            pairs = []
            outputs = []
            from collections import Counter
            signature_counts = Counter(
                ExpertSignature.of(cpu_ent)
                for cpu_ent in cpu_ents
            )
            arenas = getattr(self, "_gpu_arenas", None)
            for key, cpu_ent in zip(keys, cpu_ents):
                use_arena = (
                    arenas is not None
                    and arenas.supports(cpu_ent)
                    and signature_counts[ExpertSignature.of(cpu_ent)]
                    <= arenas.capacity(cpu_ent)
                )
                leased = self._lease_gpu_ent(
                    key,
                    cpu_ent,
                    use_arena=use_arena,
                )
                if leased is not None:
                    gu_pair, dn_pair, out = leased
                    pairs.extend((gu_pair, dn_pair))
                    outputs.append(out)
                    continue
                ent = tuple(torch.empty_like(vq.idx, device=self.device) for vq in cpu_ent)
                pairs.extend((vq.idx, d) for vq, d in zip(cpu_ent, ent))
                outputs.append(tuple(
                    VQWeight(d, self._cb_dev(vq.cb), vq.cols)
                    for vq, d in zip(cpu_ent, ent)
                ))
            self.stage.upload_batch(pairs)
            return outputs

    def prefetch(self, keys: list[tuple[int, int]]) -> None:
        """跨层专家预取（软提示，不阻塞；预测错误无正确性影响，仅 LRU 轻微污染）。

        利用路由时序局部性（相邻 token 专家集实测重合 70-90%）：在计算第 L 层
        attention 期间，把上一 token 第 L 层的专家集预先装填：
          - RAM/pinned 已有的 → pinned 分段异步 DMA 上显存（真 ~10GB/s，与计算重叠）；
          - 仅在磁盘的 → 提交线程池后台加载，get_many 命中时等待结果。
        """
        if not keys:
            return
        # staging 积压闸门：后台线程消费不过来时直接放弃本轮预取——否则 _staging
        # 无界增长、staged 条目占满 inflight 无法驱逐、缓存超预算 OOM（GLM 实测）
        with self._stage_lock:
            if self._rebuilding_arenas:
                return
            staging_backlog = len(self._staging)
        if staging_backlog > 512:
            return
        # _pending 上限：预测错偏的 Future 永不消费会无限堆积（占内存），超cap丢弃最旧
        while len(self._pending) > 256:
            oldest = next(iter(self._pending))   # 插入序最旧
            self._pending.pop(oldest).cancel()   # 尽力取消；已运行的读盘结果随引用丢弃
        # 调试二分：CCCP_PREFETCH_STAGE=0 时只做磁盘预载，不做 RAM→VRAM 异步 DMA
        do_stage = os.environ.get("CCCP_PREFETCH_STAGE", "1") != "0"
        # 自适应 1：近期 miss 率过高（RAM 池装不下工作集）时，磁盘带宽全让给
        # get_many 的紧急 miss，不再为预测预取抢队列（冷启动负优化修复）；
        # 自适应 2：预取池积压 >64 时暂停提交（get_many 绝不阻塞在预取池 backlog 后）
        recent = self._recent
        disk_ok = (len(self._pending) < 64 and
                   (len(recent) < 64 or (sum(recent) / len(recent)) < 0.5))
        stage_keys, stage_ents = [], []
        for key in keys:
            # 无锁快速过滤；提交前 _stage_async 会在锁内再次校验并关闭竞争窗口。
            if key in self.cache or key in self._staging or key in self._pending:
                continue
            cpu_ent = self.pinned.get(key)
            if cpu_ent is None:
                cpu_ent = self.ram.get(key)
                if cpu_ent is not None:
                    self.ram.move_to_end(key)
            if cpu_ent is not None:
                if self.gpu and do_stage:
                    stage_keys.append(key)
                    stage_ents.append(cpu_ent)
            elif disk_ok:
                self._pending[key] = _pf_executor().submit(self.store.load_expert, *key)
        if stage_keys:
            self._stage_async(stage_keys, stage_ents)

    def _stage_async(self, keys: list, cpu_ents: list) -> None:
        """后台 staging（真并行预加载）：单线程队列里完成 装槽+DMA+入缓存，
        主线程推理零阻塞。get_many 经 _staging 事件查重（宁可等待也不重复加载）。"""
        import threading
        fresh_keys, fresh_ents = [], []
        dones = {}
        with self._stage_lock:
            for k, cpu_ent in zip(keys, cpu_ents):
                if k in self.cache or k in self._staging:
                    continue
                ev = threading.Event()
                self._staging[k] = ev
                dones[k] = ev
                fresh_keys.append(k)
                fresh_ents.append(cpu_ent)
        if not fresh_keys:
            return
        keys, cpu_ents = fresh_keys, fresh_ents

        def job():
            try:
                staged = self._stage_ents(keys, cpu_ents)  # 内部持锁（槽位纪律）
                with self._stage_lock:
                    self._inflight.update(keys)
                    self._stage_dirty = True
                    for k, ent in zip(keys, staged):
                        self._put(k, ent)
                self._wait_stage()          # 本批 DMA 落地即解 inflight，防积压超预算
            finally:
                with self._stage_lock:
                    for k, ev in dones.items():
                        self._staging.pop(k, None)       # 先入缓存再解除标记
                        ev.set()                          # 唤醒等待者（走缓存命中）

        _stage_executor().submit(job)

    def _wait_staging_key(self, key):
        """返回已落地缓存；只等待该 key 的后台 staging，不等待整个拷贝流。"""
        with self._stage_lock:
            ent = self.cache.get(key)
            ev = self._staging.get(key)
        if ev is not None:
            ev.wait()
        elif ent is not None:
            return ent
        with self._stage_lock:
            return self.cache.get(key)

    def _wait_stage(self) -> None:
        """有在飞 DMA 时让计算流等待拷贝流尾部（缓存命中路径的安全网）。
        落地后清除 inflight 标记（对应 cache 条目恢复可驱逐）。"""
        with self._stage_lock:
            if self._stage_dirty:
                self.stage.wait()
                self._stage_dirty = False
                arenas = getattr(self, "_gpu_arenas", None)
                if arenas is not None:
                    for key in self._inflight:
                        arenas.clear_inflight(key)
                self._inflight.clear()

    def get(self, layer: int, eid: int) -> tuple[VQWeight, VQWeight]:
        key = (layer, eid)
        ent = self._wait_staging_key(key)
        if ent is not None:
            self.hits += 1
            self._recent.append(0)
            self.cache.move_to_end(key)
            self._touch_gpu_key(key)
            return ent
        self.miss += 1
        self._recent.append(1)
        cpu_ent = self.pinned.get(key)
        if cpu_ent is None:
            cpu_ent = self.ram.get(key)
            if cpu_ent is None:
                fut = self._pending.pop(key, None)
                if fut is not None and fut.done():
                    cpu_ent = fut.result()      # 预取已完成：零等待
                else:
                    if fut is not None:
                        fut.cancel()            # 未完成不在此等待（防预取池 backlog 饥饿）
                    cpu_ent = self.store.load_expert(layer, eid)
                if self._hot(layer, eid):
                    self.pinned[key] = cpu_ent  # 热专家：永久钉住，不占 LRU 预算
                else:
                    self._put_ram(key, cpu_ent)
            else:
                self.ram.move_to_end(key)
        if self.gpu:
            ent = self._stage_ent(key, cpu_ent)
            self._stage_dirty = True
            self._wait_stage()
        else:
            ent = cpu_ent
        self._put(key, ent)
        return ent

    def get_many(self, keys: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[VQWeight, VQWeight]]:
        """批量取专家：未命中项并行磁盘加载（常驻线程池，NVMe 队列深度受益）
        + 异步上传显存。decode 每层 8 专家的读路径由此从串行 ~88ms 降到 ~20ms。
        """
        out: dict[tuple[int, int], tuple[VQWeight, VQWeight]] = {}
        missing: list[tuple[int, int]] = []
        ready_keys: list[tuple[int, int]] = []
        ready_cpu: list[tuple[VQWeight, VQWeight]] = []
        demand_upload = False
        unresolved: list[tuple[tuple[int, int], object | None]] = []
        # Decode top-k arrives as one batch. Snapshot ordinary GPU hits under
        # one lock instead of taking the same RLock once per expert.
        with self._stage_lock:
            for key in keys:
                ent = self.cache.get(key)
                event = self._staging.get(key)
                if ent is None or event is not None:
                    unresolved.append((key, event))
                    continue
                self.hits += 1
                self._recent.append(0)
                self.cache.move_to_end(key)
                self._touch_gpu_key(key)
                out[key] = ent
        for key, event in unresolved:
            ent = None
            if event is not None:
                event.wait()
                with self._stage_lock:
                    ent = self.cache.get(key)
                    if ent is not None:
                        self.hits += 1
                        self._recent.append(0)
                        self.cache.move_to_end(key)
                        self._touch_gpu_key(key)
            if ent is not None:
                out[key] = ent
                continue
            cpu_ent = self.pinned.get(key)
            if cpu_ent is None:
                cpu_ent = self.ram.get(key)
                if cpu_ent is not None:
                    self.ram.move_to_end(key)
            if cpu_ent is not None:
                self.hits += 1
                self._recent.append(0)
                if self.gpu:
                    ready_keys.append(key)
                    ready_cpu.append(cpu_ent)
                else:
                    self._put(key, cpu_ent)
                    out[key] = cpu_ent
            else:
                missing.append(key)
        if ready_keys:
            staged = self._stage_ents(ready_keys, ready_cpu)
            with self._stage_lock:
                self._inflight.update(ready_keys)
                self._stage_dirty = True
                for key, ent in zip(ready_keys, staged):
                    self._put(key, ent)
                    out[key] = ent
            demand_upload = True
        if missing:
            # 后台 staging 查重：正在 staged 的 key 等其完成事件走缓存命中，
            # 绝不重复读盘（等待 ≪ 磁盘加载；事件在入缓存后 set，无竞态窗）
            still = []
            for k in missing:
                ev = self._staging.get(k)
                if ev is not None:
                    ev.wait()
                    ent = self.cache.get(k)
                    if ent is not None:
                        self.hits += 1
                        self._recent.append(0)
                        self.cache.move_to_end(k)
                        self._touch_gpu_key(k)
                        out[k] = ent
                        continue
                still.append(k)
            missing = still
        if missing:
            from concurrent.futures import as_completed
            # 上传与加载重叠：哪个专家先读完就先上传显存，其余仍在后台读盘；
            # 预取已提交的加载直接复用其 Future（不重复读盘）
            futs = {}
            for k in missing:
                fut = self._pending.pop(k, None)
                if fut is not None and fut.done():
                    futs[k] = fut               # 预取已完成：零等待直接取结果
                else:
                    if fut is not None:
                        fut.cancel()            # 未完成的预取：尽力取消，绝不在此等待
                    #（预取池是 backlog 重灾区的慢池；紧急 miss 一律走 12 线程快池）
                    futs[k] = _executor().submit(self.store.load_expert, *k)
            fmap = {f: k for k, f in futs.items()}
            for fut in as_completed(fmap):
                key = fmap[fut]
                cpu_ent = fut.result()
                self.miss += 1
                self._recent.append(1)
                if self._hot(*key):
                    self.pinned[key] = cpu_ent  # 热专家：永久钉住
                else:
                    self._put_ram(key, cpu_ent)
                ent = self._stage_ent(key, cpu_ent) if self.gpu else cpu_ent
                if self.gpu:
                    self._inflight.add(key)
                    self._stage_dirty = True
                    demand_upload = True
                self._put(key, ent)
                out[key] = ent
        if self.gpu and demand_upload:
            with self._stage_lock:
                # 只在本批发起了 DMA 时等待；纯缓存命中不阻塞预取流。
                self._wait_stage()
        return out

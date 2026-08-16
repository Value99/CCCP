"""CCCP 动态显存监视器（后台线程）：把本进程显存占用动态维持在物理显存以内。

动机：WDDM 下分配顶满物理显存会被驱动换页到"共享显存"（系统内存），
带宽从 HBM ~600GB/s 掉到 PCIe 级且伴随整轮同步卡顿——实测共享显存每换页
数 GB，decode 速度可掉数倍。静态预留（引擎初始化的分配器硬上限）只能防
"自己顶满"，防不了：其他进程中途抢占、碎片化、换用更小显卡（如 16GB）。

本监视器每 interval 秒查询一次空闲显存，滞回调节专家显存缓存预算：
  空闲 < low_gb   → 收紧预算 step_gb 并立即 LRU 驱逐 + empty_cache（止血）
  空闲 > high_gb  → 预算放宽 step_gb（不超过初始上限，缓存按需求自然回填）
任何显卡（16GB/22GB/48GB）与任何模型档位下都自动找到可用工作点；
CCCP_VRAM_WATCH=0 可关闭，low/high/interval 可用同名环境变量覆盖。
"""

from __future__ import annotations

import os
import threading
import time

import torch


class VramWatch:
    """后台滞回调节器：pool 为 ExpertPool（需有 budget 属性与 trim_to 方法）。"""

    def __init__(self, pool, max_budget: int, device: int = 0,
                 low_gb: float | None = None, high_gb: float | None = None,
                 step_gb: float = 0.5, interval: float | None = None,
                 min_gb: float = 0.5, quiet: bool = False):
        self.pool = pool
        self.device = device
        self.max_budget = int(max_budget)
        self.min_budget = int(min_gb * 2**30)
        self.low = float(low_gb if low_gb is not None
                         else os.environ.get("CCCP_VRAM_WATCH_LOW_GB", "0.8"))
        self.high = float(high_gb if high_gb is not None
                          else os.environ.get("CCCP_VRAM_WATCH_HIGH_GB", "3.0"))
        self.step = int(step_gb * 2**30)
        self.interval = float(interval if interval is not None
                              else os.environ.get("CCCP_VRAM_WATCH_SEC", "3"))
        self.quiet = quiet
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self.trims = 0      # 累计止血次数（诊断/基准记录用）
        self.grows = 0

    def start(self) -> None:
        if self._th is not None or not torch.cuda.is_available():
            return
        self._th = threading.Thread(target=self._run, name="cccp-vramwatch",
                                    daemon=True)
        self._th.start()
        if not self.quiet:
            print(f"[cccp] 显存动态监测已启动（空闲<{self.low:.1f}GB 收紧 / "
                  f">{self.high:.1f}GB 放宽，{self.interval:.0f}s 周期）", flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2)
            self._th = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                free = torch.cuda.mem_get_info(self.device)[0]
            except Exception:
                continue
            budget = self.pool.budget
            if free < self.low * 2**30 and budget > self.min_budget:
                new = max(self.min_budget, budget - self.step)
                self.pool.trim_to(new)
                torch.cuda.empty_cache()
                self.trims += 1
                if not self.quiet:
                    print(f"[vramwatch] 空闲 {free / 2**30:.2f}GB < {self.low}GB → "
                          f"显存缓存收紧至 {new / 2**30:.1f}GB", flush=True)
            elif (
                free > self.high * 2**30
                and budget < self.max_budget
                and getattr(self.pool, "supports_vram_growth", True)
            ):
                new = min(self.max_budget, budget + self.step)
                self.pool.budget = new      # 只放宽上限，缓存按需自然回填
                self.grows += 1
                if not self.quiet:
                    print(f"[vramwatch] 空闲 {free / 2**30:.2f}GB > {self.high}GB → "
                          f"显存缓存放宽至 {new / 2**30:.1f}GB", flush=True)

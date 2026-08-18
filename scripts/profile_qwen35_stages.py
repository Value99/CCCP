"""Component-level decode timing for Qwen3.5 Dense VQ via CUDA events.

Wraps DenseVQLinear / DenseVQLinearGroup / DenseVQEmbedding forwards and the
Gated-Delta attention entry with paired CUDA events (no CUPTI, works under
CUDA Graph replay disabled), then reports per-component GPU milliseconds per
decode token.  Architecture-agnostic: pure timing, no kernel changes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from cccp.qwen35_model import Qwen35DenseVQModel


class StageTimer:
    def __init__(self) -> None:
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.counts: dict[str, int] = defaultdict(int)

    def wrap(self, obj, name: str, method: str = "forward") -> None:
        original = getattr(obj, method)
        timer = self

        def timed(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            timer.events.append((name, start, end))
            timer.counts[name] += 1
            return result

        setattr(obj, method, timed.__get__(obj, type(obj)))

    def report(self, tokens: int) -> None:
        torch.cuda.synchronize()
        totals: dict[str, float] = defaultdict(float)
        for name, start, end in self.events:
            totals[name] += start.elapsed_time(end)
        grand = sum(
            v for k, v in totals.items() if not k.startswith(">")
        )
        print(f"tokens={tokens}  wrapped-total={grand:.2f} ms "
              f"({grand / tokens:.3f} ms/tok)")
        for name, ms in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"  {ms / tokens:8.3f} ms/tok  {ms:9.1f} ms total  "
                  f"n={self.counts[name]:>6}  {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prefill-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=64)
    args = parser.parse_args()

    model = Qwen35DenseVQModel(
        args.model, device="cuda",
        max_ctx=max(128, args.prefill_tokens + args.decode_tokens + 8),
    )
    model.preload()
    model.forward([248000] * args.prefill_tokens)
    torch.cuda.synchronize()

    # 禁 graph 走 eager,才能分段
    model._decode_graph = None

    timer = StageTimer()
    from cccp.dense_vq import DenseVQLinear, DenseVQLinearGroup, DenseVQEmbedding

    linear_total = [0]

    def wrap_class(cls, label):
        original = cls.forward

        def timed(self, *a, **kw):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = original(self, *a, **kw)
            end.record()
            timer.events.append((label, start, end))
            timer.counts[label] += 1
            return out

        cls.forward = timed

    wrap_class(DenseVQLinear, "linear(int4 GEMV)")
    wrap_class(DenseVQLinearGroup, ">group(total,勿重复计)")

    import time as _time
    started = _time.perf_counter()
    for i in range(args.decode_tokens):
        model.forward([248001 + i])
    torch.cuda.synchronize()
    wall = _time.perf_counter() - started
    print(f"eager wall: {args.decode_tokens / wall:.2f} tok/s "
          f"({wall / args.decode_tokens * 1e3:.3f} ms/tok)")
    timer.report(args.decode_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

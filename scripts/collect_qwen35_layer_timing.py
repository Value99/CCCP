"""Collect per-layer timings recorded inside the fork's decoder loop.

Runs decode with CCCP_QWEN35_LAYER_TIMING=1 so each layer records a same-
stream CUDA event pair (captured by the token graph and replayed every
token), then aggregates layer-type totals and prints the worst layers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from cccp.qwen35_model import Qwen35DenseVQModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, default=48)
    args = parser.parse_args()

    model = Qwen35DenseVQModel(args.model, device="cuda", max_ctx=512)
    model.preload()
    model.forward([248000] * 32)
    torch.cuda.synchronize()

    import time
    started = time.perf_counter()
    for i in range(args.decode_tokens):
        model.forward([248001 + i])
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    print(f"wall: {args.decode_tokens / wall:.2f} tok/s "
          f"({wall / args.decode_tokens * 1e3:.3f} ms/tok)")

    events = getattr(model.network.model, "_cccp_layer_events", None)
    if not events:
        print("no layer events (graph replay may not re-record python-side hooks)")
        return 1
    # 事件对象在每次 forward 被替换;graph 模式下 python 不执行 -> 只剩捕获期一组。
    # 因此直接读最后一组(graph 捕获那次),replay 会重复同样的 kernel 序列。
    torch.cuda.synchronize()
    per_type = defaultdict(float)
    rows = []
    for i, kind, s, e in events:
        ms = s.elapsed_time(e)
        per_type[kind] += ms
        rows.append((ms, i, kind))
    rows.sort(reverse=True)
    n = args.decode_tokens
    total = sum(r[0] for r in rows)
    print(f"captured layers={len(rows)} total={total:.3f} ms")
    for ms, i, kind in rows[:12]:
        print(f"  layer {i:3d} [{kind:<15}] {ms:7.3f} ms")
    for kind, ms in sorted(per_type.items(), key=lambda kv: -kv[1]):
        print(f"  TYPE {kind:<15} {ms:7.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

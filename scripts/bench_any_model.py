"""架构无关实机基准:DSV4/GLM 等经 Engine+adapter 生成,计时 decode。

用法: python bench_any_model.py <model_dir> [--max-new N]
env: CCCP_VQ_INT4_IMAGE=off|on|auto (VQ 码本 int4 快档)
     CCCP_BENCH_NO_EOS=1 时清空 EOS 跑满 max_new(纯计时)。
"""
import os
import sys
import time
from pathlib import Path

import torch

from cccp.chat_adapters import adapter_for_arch
from cccp.chat_adapters.base import ChatMessage, ChatOptions
from cccp.engine import Engine


def main() -> int:
    model = Path(sys.argv[1])
    max_new = 128
    if "--max-new" in sys.argv:
        max_new = int(sys.argv[sys.argv.index("--max-new") + 1])
    warmup = 0
    if "--warmup" in sys.argv:
        warmup = int(sys.argv[sys.argv.index("--warmup") + 1])
    tp = 1
    if "--tp" in sys.argv:
        tp = int(sys.argv[sys.argv.index("--tp") + 1])
    device = "cpu" if "--cpu" in sys.argv else "cuda"
    engine = Engine(str(model), device=device, tp_size=tp)
    if os.environ.get("CCCP_BENCH_NO_EOS", "0") == "1":
        engine.eos = set()
    arch = getattr(engine, "arch", "") or ""
    adapter = adapter_for_arch(arch)
    options = ChatOptions(
        thinking_mode="chat",
        reasoning_effort=None,
        temperature=0.0,
        top_p=1.0,
        max_new=max_new,
    )
    messages = [ChatMessage(
        role="user",
        content="请从 1 数到 40，每个数字单独一行。",
    )]
    plan = adapter.prepare(engine, messages, options, None)
    if warmup > 0:
        # 预热段:填热专家 LRU/算子缓存,不计时。
        engine.generate(plan.input_ids, max_new=warmup, temp=0.0, top_p=1.0)
        if engine.model.device.type == "cuda":
            torch.cuda.synchronize(engine.model.device)
        print(f"[bench-any] warmup={warmup} done", flush=True)
    # 预填单独计时,decode 段取 generate 内部统计或总时。
    started = time.perf_counter()
    if os.environ.get("CCCP_TORCH_PROFILE", "0") == "1":
        from torch.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        ) as prof:
            out = engine.generate(
                plan.input_ids,
                max_new=max_new,
                temp=0.0,
                top_p=1.0,
            )
        prof.export_chrome_trace("/media/tyh20/disk22/dsv4_trace.json")
    else:
        out = engine.generate(
            plan.input_ids,
            max_new=max_new,
            temp=0.0,
            top_p=1.0,
        )
    if engine.model.device.type == "cuda":
        torch.cuda.synchronize(engine.model.device)
    elapsed = time.perf_counter() - started
    parsed = adapter.parse_complete(engine, out, options)
    free_gib = 0.0
    try:
        free, _total = torch.cuda.mem_get_info()
        free_gib = free / 2**30
    except Exception:
        pass
    kv = getattr(engine, "last_kv_stats", None)
    prefill_ms = getattr(kv, "prefill_ms", None) if kv is not None else None
    prompt_n = len(plan.input_ids)
    if prefill_ms:
        prefill_str = (
            f"prefill={prefill_ms:.0f}ms({prompt_n}tok,"
            f"{prompt_n / (prefill_ms / 1000.0):.1f}tok/s)"
        )
    else:
        prefill_str = "prefill=n/a"
    decode_s = max(elapsed - (prefill_ms / 1000.0 if prefill_ms else 0.0), 1e-9)
    print(
        f"[bench-any] arch={arch} dev={device} tokens={len(out)} "
        f"elapsed={elapsed:.2f}s {prefill_str} "
        f"decode={len(out) / decode_s:.2f}tok/s "
        f"wall={len(out) / max(elapsed, 1e-9):.2f}tok/s "
        f"free_vram={free_gib:.1f}GiB",
        flush=True,
    )
    print(f"[bench-any] content={parsed.content[:60]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

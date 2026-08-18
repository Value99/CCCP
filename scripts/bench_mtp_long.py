"""长输出 MTP 轮分解基准:127+ 轮下的相位占比(需 CCCP_MTP_PROFILE=1)。

CCCP_TORCH_PROFILE=1 时用 torch.profiler 抓内核级耗时(CUPTI 可用时)。
"""
import os
import sys
import time
from pathlib import Path

import torch

from cccp.chat_adapters.base import ChatMessage, ChatOptions
from cccp.chat_adapters.qwen35 import Qwen35ChatAdapter
from cccp.engine import Engine


def main() -> int:
    model = Path(sys.argv[1] if len(sys.argv) > 1 else
                 "/media/tyh20/disk22/qwen3.8-27b-cccp-l")
    engine = Engine(str(model), device="cuda")
    adapter = Qwen35ChatAdapter()
    options = ChatOptions(
        thinking_mode="chat",
        reasoning_effort=None,
        temperature=0.0,
        top_p=1.0,
        max_new=160,
    )
    messages = [ChatMessage(
        role="user",
        content="请从 1 数到 60，每个数字占一行，不要省略任何数字。",
    )]
    plan = adapter.prepare(engine, messages, options, None)
    torch_prof = os.environ.get("CCCP_TORCH_PROFILE", "0") == "1"
    started = time.perf_counter()
    if torch_prof:
        from torch.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            output_ids = engine.generate_speculative(
                plan.input_ids, max_new=options.max_new, k=int(os.environ.get("CCCP_MTP_K", "4")),
            )
    else:
        output_ids = engine.generate_speculative(
            plan.input_ids, max_new=options.max_new, k=int(os.environ.get("CCCP_MTP_K", "4")),
        )
    torch.cuda.synchronize(engine.model.device)
    elapsed = time.perf_counter() - started
    stats = dict(engine.spec_stats or {})
    print(
        f"[bench-mtp] tokens={len(output_ids)} elapsed={elapsed:.2f}s "
        f"decode={stats.get('decode_tokens_per_second', 0):.2f}tok/s "
        f"rounds={stats.get('rounds')} "
        f"accepted={stats.get('accepted')}/{stats.get('drafted')}",
        flush=True,
    )
    if torch_prof:
        prof.export_chrome_trace("/media/tyh20/disk22/mtp_trace.json")
        print(prof.key_averages(
            group_by_input_shape=True
        ).table(sort_by="cuda_time_total", row_limit=20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

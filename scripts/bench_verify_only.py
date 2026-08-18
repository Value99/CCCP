"""稳态 verify 计时:绕开生成 EOS,直接循环 forward_hidden_verify。"""
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
        thinking_mode="chat", reasoning_effort=None,
        temperature=0.0, top_p=1.0, max_new=32,
    )
    messages = [ChatMessage(role="user", content="请从 1 数到 60。")]
    plan = adapter.prepare(engine, messages, options, None)
    ids = plan.input_ids

    # 1) 预填 + 两次 batch-5 verify(第 1 次走 eager 暖机+惰性转换,
    #    第 2 次捕获图),之后进入稳态回放。
    engine.model.forward_hidden(ids)
    block = ids[-1:] + [10, 11, 12, 13, 14]
    for _ in range(2):
        hidden = engine.model.forward_hidden_verify(block)
        engine.model.logits_of(hidden)
        engine.reset()

    # 2) 稳态:重新预填,再连测 100 次完整 verify+logits。
    engine.model.forward_hidden(ids)
    torch.cuda.synchronize()
    start = time.perf_counter()
    rounds = 100
    for _ in range(rounds):
        hidden = engine.model.forward_hidden_verify(block)
        logits = engine.model.logits_of(hidden)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        f"[bench-verify] {rounds} rounds batch=5: "
        f"{elapsed / rounds * 1000:.2f} ms/round "
        f"(verify+logits, steady)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

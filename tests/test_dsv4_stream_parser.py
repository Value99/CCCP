from __future__ import annotations

import sys
from pathlib import Path


ENGINE = Path(__file__).resolve().parents[1] / "engine" / "CCCP-Engine"
sys.path.insert(0, str(ENGINE))

from cccp.chat_adapters.base import (  # noqa: E402
    AssistantOutput,
    ChatMessage,
    ChatOptions,
)
from cccp.chat_adapters.dsv4 import DSV4ChatAdapter  # noqa: E402
from cccp.chat_adapters.dsv4_encoding import eos_token  # noqa: E402


def _options() -> ChatOptions:
    return ChatOptions(
        thinking_mode="chat",
        reasoning_effort=None,
        temperature=0.0,
        top_p=1.0,
        max_new=16,
    )


def test_stream_parser_hides_split_eos_marker_from_content_and_deltas():
    parser = DSV4ChatAdapter().new_stream_parser(object(), _options())
    split = len(eos_token) // 2
    first = parser.feed("你好" + eos_token[:split])
    second = parser.feed(eos_token[split:])
    output, final = parser.finish()

    assert output.content == "你好"
    assert eos_token not in output.content
    assert "".join(delta.text or "" for delta in (*first, *second, *final)) == "你好"


def test_commit_does_not_duplicate_generated_eos_in_hot_ledger():
    class Engine:
        _cache_ids: list[int]

        @staticmethod
        def encode(text: str) -> list[int]:
            return [ord(char) for char in text]

    engine = Engine()
    adapter = DSV4ChatAdapter()
    plan = adapter.prepare(
        engine,
        [ChatMessage(role="user", content="你好")],
        _options(),
        None,
    )
    output_ids = engine.encode("你好" + eos_token)
    engine._cache_ids = list(plan.input_ids) + output_ids
    ledger = adapter.commit(
        engine,
        plan,
        output_ids,
        AssistantOutput(reasoning_content=None, content="你好", tool_calls=[]),
    )
    eos_ids = engine.encode(eos_token)

    assert ledger.completed_ids is not None
    assert ledger.completed_ids[-len(eos_ids) :] == eos_ids
    assert ledger.completed_ids[-2 * len(eos_ids) : -len(eos_ids)] != eos_ids

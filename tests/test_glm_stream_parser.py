"""GLM streaming reasoning/content boundaries."""

from pathlib import Path
import sys


ENGINE_ROOT = Path(__file__).resolve().parents[1] / "engine" / "CCCP-Engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from cccp.chat_adapters.base import ChatOptions  # noqa: E402
from cccp.chat_adapters.glm import GLMChatAdapter  # noqa: E402


def _options(mode: str) -> ChatOptions:
    return ChatOptions(
        thinking_mode=mode,
        reasoning_effort=None,
        temperature=0.0,
        top_p=1.0,
        max_new=64,
    )


def test_glm_stream_parser_hides_split_think_marker():
    parser = GLMChatAdapter().new_stream_parser(
        object(),
        _options("thinking"),
    )

    first = parser.feed("先计算 17×19=323</thi")
    second = parser.feed("nk>答案是 323。")
    output, final = parser.finish()

    assert "".join(delta.text for delta in first if delta.kind == "reasoning") == (
        "先计算 17×19=323"
    )
    assert "".join(delta.text for delta in second if delta.kind == "content") == (
        "答案是 323。"
    )
    assert output.reasoning_content == "先计算 17×19=323"
    assert output.content == "答案是 323。"
    assert final == ()
    assert "</think>" not in output.content


def test_glm_unfinished_thinking_never_becomes_visible_content():
    parser = GLMChatAdapter().new_stream_parser(
        object(),
        _options("thinking"),
    )

    deltas = parser.feed("仍在推理")
    output, final = parser.finish()

    assert output.reasoning_content == "仍在推理"
    assert output.content == ""
    assert all(delta.kind == "reasoning" for delta in (*deltas, *final))


def test_glm_chat_mode_streams_plain_content():
    parser = GLMChatAdapter().new_stream_parser(
        object(),
        _options("chat"),
    )

    deltas = parser.feed("你好")
    output, final = parser.finish()

    assert [delta.kind for delta in deltas] == ["content"]
    assert output.reasoning_content is None
    assert output.content == "你好"
    assert final == ()

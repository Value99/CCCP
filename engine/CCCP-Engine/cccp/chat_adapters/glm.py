"""GLM chat adapter preserving CCCP's existing terminal prompt format."""

from __future__ import annotations

from dataclasses import dataclass

from .base import (
    AssistantOutput,
    ChatMessage,
    ChatOptions,
    PromptPlan,
    StreamDelta,
    StreamParser,
    UnsupportedChatCapability,
)


def _strip_think(content: str) -> str:
    """Keep only the final answer from a prior GLM assistant response."""
    if "</think>" in content:
        return content.split("</think>", 1)[1].lstrip()
    return content


def _render_prompt(
    messages: tuple[ChatMessage, ...],
    options: ChatOptions,
) -> str:
    """Render supported messages using the unchanged terminal serialization."""
    parts = ["[gMASK]<sop>"]
    if options.thinking_mode == "thinking":
        parts.append("<|system|>Reasoning Effort: Max")

    for message in messages:
        if message.role in {"system", "developer"}:
            parts.append(f"<|system|>{message.content}")
        elif message.role == "user":
            parts.append(f"<|user|>{message.content}\n")
        elif message.role == "assistant":
            parts.append(
                f"<|assistant|>\n<think></think>{_strip_think(message.content)}"
            )
        else:
            raise UnsupportedChatCapability("glm", f"{message.role} messages")

    parts.append("<|assistant|>")
    parts.append(
        "<think>" if options.thinking_mode == "thinking" else "<think></think>"
    )
    return "".join(parts)


def _reject_tools(messages: tuple[ChatMessage, ...], options: ChatOptions) -> None:
    if options.tools or options.tool_choice not in (None, "none"):
        raise UnsupportedChatCapability("glm", "tools")
    if any(
        message.role == "tool" or message.tool_calls or message.tool_call_id is not None
        for message in messages
    ):
        raise UnsupportedChatCapability("glm", "tools")


class _GLMStreamParser:
    _THINK_END = "</think>"

    def __init__(self, *, thinking: bool) -> None:
        self._thinking = bool(thinking)
        self._phase = "reasoning" if self._thinking else "content"
        self._buffer = ""
        self._reasoning = ""
        self._content = ""
        self._finished = False

    @classmethod
    def _marker_overlap(cls, text: str) -> int:
        """Keep only a suffix that can still become ``</think>``."""
        marker = cls._THINK_END
        for size in range(min(len(text), len(marker) - 1), 0, -1):
            if text.endswith(marker[:size]):
                return size
        return 0

    def _emit_reasoning(self, text: str) -> tuple[StreamDelta, ...]:
        if not text:
            return ()
        self._reasoning += text
        return (StreamDelta(kind="reasoning", text=text),)

    def _emit_content(self, text: str) -> tuple[StreamDelta, ...]:
        if not text:
            return ()
        self._content += text
        return (StreamDelta(kind="content", text=text),)

    def feed(self, text: str) -> tuple[StreamDelta, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished GLM stream parser")
        if not isinstance(text, str):
            raise TypeError("stream parser text must be a string")
        if not text:
            return ()
        if self._phase == "content":
            return self._emit_content(text)

        self._buffer += text
        marker_at = self._buffer.find(self._THINK_END)
        if marker_at >= 0:
            reasoning = self._buffer[:marker_at]
            content = self._buffer[marker_at + len(self._THINK_END):]
            self._buffer = ""
            self._phase = "content"
            return (
                *self._emit_reasoning(reasoning),
                *self._emit_content(content.lstrip()),
            )

        overlap = self._marker_overlap(self._buffer)
        safe = self._buffer[:-overlap] if overlap else self._buffer
        self._buffer = self._buffer[-overlap:] if overlap else ""
        return self._emit_reasoning(safe)

    def finish(self) -> tuple[AssistantOutput, tuple[StreamDelta, ...]]:
        if self._finished:
            raise RuntimeError("GLM stream parser is already finished")
        self._finished = True
        final_deltas = (
            self._emit_reasoning(self._buffer)
            if self._phase == "reasoning"
            else self._emit_content(self._buffer)
        )
        self._buffer = ""
        return (
            AssistantOutput(
                reasoning_content=self._reasoning or None,
                content=self._content,
                tool_calls=[],
            ),
            final_deltas,
        )


@dataclass
class GLMTokenLedger:
    """GLM history plus the exact live token prefix for hot continuation."""

    committed_messages: tuple[ChatMessage, ...] = ()
    completed_ids: list[int] | None = None
    thinking_mode: str | None = None

    def clear(self) -> None:
        self.committed_messages = ()
        self.completed_ids = None
        self.thinking_mode = None


class GLMChatAdapter:
    """GLM prompt planning with exact-token multi-turn KV reuse."""

    name = "glm"

    def prepare(
        self,
        engine: object,
        messages: list[ChatMessage],
        options: ChatOptions,
        hot_ledger: object | None,
    ) -> PromptPlan:
        normalized = tuple(messages)
        _reject_tools(normalized, options)
        if (
            isinstance(hot_ledger, GLMTokenLedger)
            and options.thinking_mode == "chat"
            and hot_ledger.thinking_mode == options.thinking_mode
            and hot_ledger.completed_ids is not None
            and getattr(engine, "_cache_ids", None)
            == hot_ledger.completed_ids
            and len(normalized)
            == len(hot_ledger.committed_messages) + 1
            and normalized[:-1] == hot_ledger.committed_messages
            and normalized[-1].role == "user"
        ):
            suffix = (
                f"<|user|>{normalized[-1].content}\n"
                "<|assistant|><think></think>"
            )
            input_ids = [
                *hot_ledger.completed_ids,
                *engine.encode(suffix),
            ]
            return PromptPlan(
                input_ids=input_ids,
                kv_baseline_len=len(input_ids),
                normalized_messages=normalized,
                canonical_prefix_ids=list(input_ids),
                adapter_state={"thinking_mode": options.thinking_mode},
            )
        rendered = _render_prompt(normalized, options)
        input_ids = list(engine.encode(rendered))
        return PromptPlan(
            input_ids=input_ids,
            kv_baseline_len=len(input_ids),
            normalized_messages=normalized,
            canonical_prefix_ids=list(input_ids),
            adapter_state={"thinking_mode": options.thinking_mode},
        )

    def parse_complete(
        self,
        engine: object,
        output_ids: list[int],
        options: ChatOptions,
    ) -> AssistantOutput:
        text = engine.decode(list(output_ids))
        if "</think>" in text:
            reasoning, content = text.split("</think>", 1)
            return AssistantOutput(
                reasoning_content=reasoning or None,
                content=content.lstrip(),
                tool_calls=[],
            )
        if options.thinking_mode == "thinking":
            return AssistantOutput(
                reasoning_content=text or None,
                content="",
                tool_calls=[],
            )
        return AssistantOutput(
            reasoning_content=None,
            content=text,
            tool_calls=[],
        )

    def new_stream_parser(
        self,
        engine: object,
        options: ChatOptions,
    ) -> StreamParser:
        del engine
        return _GLMStreamParser(
            thinking=options.thinking_mode == "thinking"
        )

    def commit(
        self,
        engine: object,
        plan: PromptPlan,
        output_ids: list[int],
        parsed: AssistantOutput,
    ) -> GLMTokenLedger:
        assistant_message = ChatMessage(
            role="assistant",
            content=parsed.content,
            reasoning_content=parsed.reasoning_content,
        )
        completed_ids = [*plan.input_ids, *output_ids]
        if (
            parsed.reasoning_content is not None
            or getattr(engine, "_cache_ids", None) != completed_ids
        ):
            completed_ids = None
        return GLMTokenLedger(
            committed_messages=plan.normalized_messages + (assistant_message,),
            completed_ids=completed_ids,
            thinking_mode=plan.adapter_state["thinking_mode"],
        )


__all__ = [
    "GLMChatAdapter",
    "GLMTokenLedger",
    "UnsupportedChatCapability",
]

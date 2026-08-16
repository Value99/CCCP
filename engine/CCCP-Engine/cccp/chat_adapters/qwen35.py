"""Qwen3.5 ChatML adapter for text-only Dense VQ archives."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .base import (
    AdapterWarning,
    AssistantOutput,
    ChatMessage,
    ChatOptions,
    PromptPlan,
    StreamDelta,
    StreamParser,
    ToolCall,
    ToolFunction,
    UnsupportedChatCapability,
)


_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"


def _reasoning_instruction(options: ChatOptions) -> str:
    if options.thinking_mode != "thinking":
        return ""
    effort = (options.reasoning_effort or "xhigh").lower()
    if effort in {"max", "high", "xhigh"}:
        return (
            "Reasoning effort is set to xhigh. Please think carefully through "
            "the task, validate key assumptions, consider plausible alternatives, "
            "and prioritize correctness, consistency, and clarity in the final answer."
        )
    if effort == "medium":
        return ""
    if effort == "low":
        return (
            "Reasoning effort is set to low. Keep your thinking brief and focused, "
            "moving directly to the conclusion without unnecessary elaboration."
        )
    raise UnsupportedChatCapability(
        "qwen3_5_dense", f"reasoning_effort={options.reasoning_effort}"
    )


def _tool_name(tool: dict) -> str:
    function = tool.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if not isinstance(name, str) or not name:
        raise ValueError("tool schemas must identify a non-empty function name")
    return name


def _selected_tools(options: ChatOptions) -> tuple[dict, ...]:
    choice = options.tool_choice
    if choice == "none":
        return ()
    if choice == "auto":
        return options.tools
    if choice == "required":
        if not options.tools:
            raise ValueError("tool_choice='required' requires at least one tool")
        return options.tools
    if isinstance(choice, dict):
        wanted = choice["function"]["name"]
        selected = tuple(
            tool for tool in options.tools if _tool_name(tool) == wanted
        )
        if not selected:
            raise ValueError(f"unknown tool in named tool_choice: {wanted!r}")
        return selected
    raise ValueError(f"unsupported normalized tool_choice: {choice!r}")


def _tool_instructions(options: ChatOptions) -> str:
    tools = _selected_tools(options)
    if not tools:
        return ""
    lines = [
        "# Tools",
        "",
        "You have access to the following functions:",
        "",
        "<tools>",
    ]
    lines.extend(json.dumps(tool, ensure_ascii=False) for tool in tools)
    lines.extend([
        "</tools>",
        "",
        "If you choose to call a function ONLY reply in the following format "
        "with NO suffix:",
        "",
        "<tool_call>",
        "<function=example_function_name>",
        "<parameter=example_parameter_1>",
        "value_1",
        "</parameter>",
        "</function>",
        "</tool_call>",
    ])
    if options.tool_choice == "required":
        lines.append("You must call at least one available function.")
    elif isinstance(options.tool_choice, dict):
        lines.append(
            "You must call the function named "
            f"{options.tool_choice['function']['name']}."
        )
    if not options.parallel_tool_calls:
        lines.append("Call at most one function in this response.")
    return "\n".join(lines)


def _ensure_text_only(message: ChatMessage) -> None:
    if message.content_parts:
        raise UnsupportedChatCapability(
            "qwen3_5_dense", "vision content in the text-only archive"
        )


def _render_tool_call(call: ToolCall) -> str:
    try:
        arguments = json.loads(call.function.arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("tool call arguments must be JSON") from exc
    if not isinstance(arguments, dict):
        raise ValueError("Qwen3.5 tool call arguments must be an object")
    body = [f"{_TOOL_OPEN}\n<function={call.function.name}>\n"]
    for name, value in arguments.items():
        rendered = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, separators=(",", ":")
        )
        body.append(f"<parameter={name}>\n{rendered}\n</parameter>\n")
    body.append(f"</function>\n{_TOOL_CLOSE}")
    return "".join(body)


def _render_message(message: ChatMessage) -> str:
    _ensure_text_only(message)
    if message.role == "user":
        return f"{_IM_START}user\n{message.content}{_IM_END}\n"
    if message.role == "assistant":
        reasoning = (message.reasoning_content or "").strip()
        value = (
            f"{_IM_START}assistant\n{_THINK_OPEN}\n{reasoning}\n"
            f"{_THINK_CLOSE}\n\n{message.content}"
        )
        if message.tool_calls:
            separator = "\n\n" if message.content.strip() else ""
            value += separator + "\n".join(
                _render_tool_call(call) for call in message.tool_calls
            )
        return value + f"{_IM_END}\n"
    if message.role == "tool":
        return (
            f"{_IM_START}user\n<tool_response>\n{message.content}\n"
            f"</tool_response>{_IM_END}\n"
        )
    raise UnsupportedChatCapability(
        "qwen3_5_dense", f"{message.role} message position"
    )


def _normalized_messages(messages: list[ChatMessage]) -> tuple[ChatMessage, ...]:
    normalized = tuple(messages)
    if not normalized:
        raise ValueError("at least one chat message is required")
    for message in normalized:
        _ensure_text_only(message)
    return normalized


def _render_prompt(
    messages: tuple[ChatMessage, ...],
    options: ChatOptions,
) -> str:
    reasoning = _reasoning_instruction(options)
    tool_text = _tool_instructions(options)
    first_system = messages[0].content if messages[0].role == "system" else ""
    start = 1 if messages[0].role == "system" else 0
    system_parts = [part for part in (reasoning, tool_text, first_system) if part]
    rendered = (
        f"{_IM_START}system\n" + "\n\n".join(system_parts) + f"{_IM_END}\n"
        if system_parts else ""
    )
    for message in messages[start:]:
        if message.role in {"system", "developer", "latest_reminder"}:
            raise UnsupportedChatCapability(
                "qwen3_5_dense", "non-leading system/developer messages"
            )
        rendered += _render_message(message)
    rendered += f"{_IM_START}assistant\n"
    rendered += (
        f"{_THINK_OPEN}\n"
        if options.thinking_mode == "thinking"
        else f"{_THINK_OPEN}\n\n{_THINK_CLOSE}\n\n"
    )
    return rendered


def _parse_scalar(value: str):
    stripped = value.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


_FUNCTION_RE = re.compile(
    r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL
)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)


def _parse_tool_blocks(raw: str, options: ChatOptions) -> tuple[list[ToolCall], list[AdapterWarning]]:
    allowed = {_tool_name(tool) for tool in _selected_tools(options)}
    calls: list[ToolCall] = []
    warnings: list[AdapterWarning] = []
    for index, block in enumerate(
        re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.DOTALL), 1
    ):
        match = _FUNCTION_RE.fullmatch(block.strip())
        if match is None or match.group(1).strip() not in allowed:
            warnings.append(AdapterWarning(
                code="malformed_tool_call",
                message="Qwen3.5 returned a malformed or unavailable tool call",
            ))
            continue
        name, arguments_raw = match.group(1).strip(), match.group(2)
        arguments = {
            key.strip(): _parse_scalar(value)
            for key, value in _PARAMETER_RE.findall(arguments_raw)
        }
        calls.append(ToolCall(
            id=f"call_qwen35_{index}",
            function=ToolFunction(
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        ))
    if not options.parallel_tool_calls and len(calls) > 1:
        warnings.append(AdapterWarning(
            code="parallel_tool_calls_disabled",
            message="Qwen3.5 returned multiple tool calls while parallel calls were disabled",
        ))
        calls = calls[:1]
    return calls, warnings


class _Qwen35StreamParser:
    """Hide structural markers while retaining stable streaming text."""

    _MARKERS = (_THINK_CLOSE, _TOOL_OPEN, _IM_END)

    def __init__(self, options: ChatOptions) -> None:
        self.options = options
        self.phase = "reasoning" if options.thinking_mode == "thinking" else "content"
        self.pending = ""
        self.reasoning = ""
        self.content = ""
        self.tool_raw = ""
        self.finished = False

    @classmethod
    def _overlap(cls, value: str) -> int:
        return max((
            size
            for marker in cls._MARKERS
            for size in range(1, min(len(marker), len(value) + 1))
            if value.endswith(marker[:size])
        ), default=0)

    def _emit(self, value: str) -> tuple[StreamDelta, ...]:
        if not value:
            return ()
        if self.phase == "reasoning":
            self.reasoning += value
            return (StreamDelta(kind="reasoning", text=value),)
        self.content += value
        return (StreamDelta(kind="content", text=value),)

    def _drain(self, force: bool = False) -> tuple[StreamDelta, ...]:
        deltas: list[StreamDelta] = []
        while self.pending:
            if self.phase == "tools":
                self.tool_raw += self.pending
                self.pending = ""
                break
            matches = [
                (self.pending.find(marker), marker)
                for marker in self._MARKERS
                if self.pending.find(marker) >= 0
            ]
            if matches:
                at, marker = min(matches, key=lambda item: item[0])
                deltas.extend(self._emit(self.pending[:at]))
                self.pending = self.pending[at + len(marker):]
                if marker == _THINK_CLOSE:
                    self.phase = "content"
                    self.pending = self.pending.lstrip("\r\n ")
                elif marker == _TOOL_OPEN:
                    self.phase = "tools"
                    self.tool_raw = marker
                elif marker == _IM_END:
                    self.pending = ""
                continue
            if force:
                stable, self.pending = self.pending, ""
            else:
                overlap = self._overlap(self.pending)
                stable = self.pending[:-overlap] if overlap else self.pending
                self.pending = self.pending[-overlap:] if overlap else ""
            deltas.extend(self._emit(stable))
            break
        return tuple(deltas)

    def feed(self, text: str) -> tuple[StreamDelta, ...]:
        if self.finished:
            raise RuntimeError("cannot feed a finished Qwen3.5 stream parser")
        self.pending += text
        return self._drain()

    def finish(self) -> tuple[AssistantOutput, tuple[StreamDelta, ...]]:
        if self.finished:
            raise RuntimeError("Qwen3.5 stream parser is already finished")
        final = self._drain(force=True)
        calls, warnings = _parse_tool_blocks(self.tool_raw, self.options)
        self.finished = True
        return AssistantOutput(
            reasoning_content=self.reasoning or None,
            content=self.content.rstrip(),
            tool_calls=calls,
            warnings=tuple(warnings),
        ), final


@dataclass
class Qwen35TokenLedger:
    committed_messages: tuple[ChatMessage, ...] = ()
    completed_ids: list[int] | None = None
    signature: str = ""


def _signature(options: ChatOptions) -> str:
    return json.dumps({
        "thinking": options.thinking_mode,
        "effort": options.reasoning_effort,
        "tools": options.tools,
        "choice": options.tool_choice,
        "parallel": options.parallel_tool_calls,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Qwen35ChatAdapter:
    name = "qwen3_5_dense"

    def prepare(
        self,
        engine: object,
        messages: list[ChatMessage],
        options: ChatOptions,
        hot_ledger: object | None,
    ) -> PromptPlan:
        normalized = _normalized_messages(messages)
        signature = _signature(options)
        ledger = hot_ledger if isinstance(hot_ledger, Qwen35TokenLedger) else None
        can_extend = (
            ledger is not None
            and ledger.completed_ids is not None
            and ledger.signature == signature
            and len(normalized) == len(ledger.committed_messages) + 1
            and normalized[:-1] == ledger.committed_messages
            and normalized[-1].role == "user"
            and getattr(engine, "_cache_ids", None) == ledger.completed_ids
        )
        if can_extend:
            suffix = _render_message(normalized[-1])
            suffix += f"{_IM_START}assistant\n"
            suffix += (
                f"{_THINK_OPEN}\n"
                if options.thinking_mode == "thinking"
                else f"{_THINK_OPEN}\n\n{_THINK_CLOSE}\n\n"
            )
            ids = [*ledger.completed_ids, *engine.encode(suffix)]
        else:
            ids = list(engine.encode(_render_prompt(normalized, options)))
        return PromptPlan(
            input_ids=ids,
            kv_baseline_len=len(ids),
            normalized_messages=normalized,
            canonical_prefix_ids=list(ids),
            adapter_state={"signature": signature},
        )

    def parse_complete(
        self,
        engine: object,
        output_ids: list[int],
        options: ChatOptions,
    ) -> AssistantOutput:
        parser = self.new_stream_parser(engine, options)
        parser.feed(engine.decode(list(output_ids)))
        parsed, _ = parser.finish()
        return parsed

    def new_stream_parser(
        self,
        engine: object,
        options: ChatOptions,
    ) -> StreamParser:
        del engine
        return _Qwen35StreamParser(options)

    def commit(
        self,
        engine: object,
        plan: PromptPlan,
        output_ids: list[int],
        parsed: AssistantOutput,
    ) -> Qwen35TokenLedger:
        assistant = ChatMessage(
            role="assistant",
            content=parsed.content,
            # Private reasoning is deliberately not part of the next request.
            reasoning_content=None,
            tool_calls=tuple(parsed.tool_calls),
        )
        completed = [*plan.input_ids, *output_ids]
        if (
            plan.adapter_state.get("signature") is None
            or getattr(engine, "_cache_ids", None) != completed
            or parsed.reasoning_content is not None
        ):
            completed = None
        return Qwen35TokenLedger(
            committed_messages=plan.normalized_messages + (assistant,),
            completed_ids=completed,
            signature=str(plan.adapter_state["signature"]),
        )


__all__ = ["Qwen35ChatAdapter", "Qwen35TokenLedger"]

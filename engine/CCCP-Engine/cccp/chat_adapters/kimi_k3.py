"""Kimi K3 XTML chat adapter for text-only inference.

The structural format follows Moonshot's ``encoding_k3.py``.  Control tokens
are encoded as specials while all message text and attribute values are
encoded as ordinary BPE, preventing user text from injecting protocol tokens.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass

from ..media import media_references_digest

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


OPEN = "<|open|>"
CLOSE = "<|close|>"
SEP = "<|sep|>"
END_OF_MSG = "<|end_of_msg|>"
MEDIA_PAD = "<|media_pad|>"
IMAGE_PLACEHOLDER = (
    "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
)
VIDEO_PLACEHOLDER = (
    "<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
)


def _message_content(message: ChatMessage) -> tuple[str, list[dict[str, object]]]:
    if not message.content_parts:
        return message.content, []
    chunks: list[str] = []
    slots: list[dict[str, object]] = []
    for part in message.content_parts:
        if part.type == "text":
            chunks.append(part.text or "")
            continue
        kind = "image" if part.type in {"image_url", "input_image"} else "video"
        placeholder = IMAGE_PLACEHOLDER if kind == "image" else VIDEO_PLACEHOLDER
        chunks.append(placeholder)
        slots.append({"kind": kind, "source": part.url, "placeholder": placeholder})
    return "".join(chunks), slots


def _media_slots(messages: tuple[ChatMessage, ...]) -> list[dict[str, object]]:
    slots: list[dict[str, object]] = []
    for message in messages:
        _content, message_slots = _message_content(message)
        slots.extend(message_slots)
    return slots


def _escape_attr(value: str) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _open_tag(tag: str, **attrs: str) -> list[tuple[str, bool]]:
    segments = [(OPEN, True), (tag, False)]
    for key, value in attrs.items():
        segments.extend([
            (f" {key}", False),
            ('="', False),
            (_escape_attr(value), False),
            ('"', False),
        ])
    segments.append((SEP, True))
    return segments


def _close_tag(tag: str) -> list[tuple[str, bool]]:
    return [(CLOSE, True), (tag, False), (SEP, True)]


def _message(
    role: str,
    content: str,
) -> list[tuple[str, bool]]:
    return [
        *_open_tag("message", role=role),
        (content, False),
        *_close_tag("message"),
        (END_OF_MSG, True),
    ]


def _message_parts(role: str, message: ChatMessage) -> list[tuple[str, bool]]:
    if not message.content_parts:
        return _message(role, message.content)
    segments = _open_tag("message", role=role)
    for part in message.content_parts:
        if part.type == "text":
            segments.append((part.text or "", False))
        else:
            kind = "image" if part.type in {"image_url", "input_image"} else "video"
            segments.append((IMAGE_PLACEHOLDER if kind == "image" else VIDEO_PLACEHOLDER, True))
    segments.extend(_close_tag("message"))
    segments.append((END_OF_MSG, True))
    return segments


def _thinking_effort_message(effort: str) -> list[tuple[str, bool]]:
    body = (
        "`thinking_effort` guides on how much to think in your "
        "thinking channel (not including the response channel), "
        "supported values include `low`, `medium`, `high`, and `max`.\n"
        f"Now the system is invoked with `thinking_effort={effort}`."
    )
    return [
        *_open_tag(
            "message",
            role="system",
            type="thinking-effort",
        ),
        (body, False),
        *_close_tag("message"),
        (END_OF_MSG, True),
    ]


def _unescape_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _json_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _deep_sort_dict(value: object) -> object:
    if isinstance(value, dict):
        return {key: _deep_sort_dict(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_deep_sort_dict(item) for item in value]
    return value


def _xtml_type(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    return "array"


def _xtml_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _normalize_tool_arguments(arguments: str) -> tuple[dict, str | None]:
    """Split stored JSON arguments into per-key values or a raw JSON block."""
    if not arguments.strip():
        return {}, None
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}, arguments
    if not isinstance(parsed, dict):
        raise UnsupportedChatCapability(
            "kimi_k3",
            "non-object tool call arguments",
        )
    return parsed, None


def _extract_response_schema(response_format: dict) -> object:
    json_schema = response_format.get("json_schema")
    if json_schema is None:
        return None
    if isinstance(json_schema, dict):
        return json_schema.get(
            "schema",
            json_schema.get("json_schema", json_schema),
        )
    return json_schema


def _internal_system_message(
    message_type: str,
    body: str,
) -> list[tuple[str, bool]]:
    return [
        *_open_tag("message", role="system", type=message_type),
        (body.strip(), False),
        *_close_tag("message"),
        (END_OF_MSG, True),
    ]


def _render_tool_declare(tools: tuple[dict, ...]) -> list[tuple[str, bool]]:
    body = (
        "# Tools\n"
        "Here are the available tools, described in JSONSchema.\n\n"
        "```json\n"
        f"{_json_compact(_deep_sort_dict(list(tools)))}\n"
        "```"
    )
    return [
        *_open_tag("message", role="system", type="tool-declare"),
        (body, False),
        *_close_tag("message"),
        (END_OF_MSG, True),
    ]


def _assistant_message(
    message: ChatMessage,
    *,
    thinking: bool,
) -> list[tuple[str, bool]]:
    segments = _open_tag("message", role="assistant")
    if thinking:
        segments.extend(_open_tag("think"))
        if message.reasoning_content:
            segments.append((message.reasoning_content, False))
        segments.extend(_close_tag("think"))
    segments.extend(_open_tag("response"))
    if message.content:
        segments.append((message.content, False))
    segments.extend(_close_tag("response"))
    for index, call in enumerate(message.tool_calls, start=1):
        if index == 1:
            segments.extend(_open_tag("tools"))
        segments.extend(
            _open_tag("call", tool=call.function.name, index=str(index))
        )
        arguments, json_block = _normalize_tool_arguments(
            call.function.arguments
        )
        if json_block is not None:
            segments.extend(_open_tag("json", type="object"))
            segments.append((json_block, False))
            segments.extend(_close_tag("json"))
        else:
            for key, value in arguments.items():
                segments.extend(
                    _open_tag("argument", key=key, type=_xtml_type(value))
                )
                segments.append((_xtml_value(value), False))
                segments.extend(_close_tag("argument"))
        segments.extend(_close_tag("call"))
    if message.tool_calls:
        segments.extend(_close_tag("tools"))
    segments.extend(_close_tag("message"))
    segments.append((END_OF_MSG, True))
    return segments


def _tool_run_messages(
    run: list[ChatMessage],
    pending: tuple[ToolCall, ...],
) -> list[tuple[str, bool]]:
    """Render consecutive tool results in assistant tool_calls order.

    Matching follows ``encoding_k3.py``: an id-matched call is authoritative
    for the tool name, results are sorted by call position, and an unmatched
    run falls back to positional name resolution.
    """
    by_id: dict[str, tuple[int, str]] = {}
    for position, call in enumerate(pending, start=1):
        by_id.setdefault(call.id, (position, call.function.name))
    ordered = list(zip(run, (by_id.get(m.tool_call_id or "") for m in run)))
    if all(match is not None for _, match in ordered):
        ordered.sort(key=lambda item: item[1][0])
    segments: list[tuple[str, bool]] = []
    for index, (message, match) in enumerate(ordered, start=1):
        if match is not None:
            name = match[1]
        elif index <= len(pending):
            name = pending[index - 1].function.name
        else:
            raise UnsupportedChatCapability(
                "kimi_k3",
                "tool messages without a matching tool_call_id",
            )
        segments.extend(
            _open_tag("message", role="tool", tool=name, index=str(index))
        )
        if message.content:
            segments.append((message.content, False))
        segments.extend(_close_tag("message"))
        segments.append((END_OF_MSG, True))
    return segments


def _render_messages(
    messages: list[ChatMessage] | tuple[ChatMessage, ...],
    *,
    thinking: bool,
    pending: tuple[ToolCall, ...] = (),
) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role in {"system", "developer"}:
            segments.extend(_message_parts("system", message))
        elif message.role == "user":
            segments.extend(_message_parts("user", message))
        elif message.role == "assistant":
            pending = message.tool_calls
            segments.extend(_assistant_message(message, thinking=thinking))
        elif message.role == "tool":
            run: list[ChatMessage] = []
            while index < len(messages) and messages[index].role == "tool":
                run.append(messages[index])
                index += 1
            index -= 1
            segments.extend(_tool_run_messages(run, pending))
        else:
            raise UnsupportedChatCapability(
                "kimi_k3",
                f"{message.role} messages",
            )
        index += 1
    return segments


def _request_directives(options: ChatOptions) -> list[tuple[str, bool]]:
    """Request-scoped internal messages rendered at the conversation tail."""
    segments: list[tuple[str, bool]] = []
    if options.tools:
        if options.tool_choice == "required":
            segments.extend(_internal_system_message(
                "tool-choice",
                "The system is invoked with `tool_choice=required`.\n"
                "You MUST call tools in the next message.",
            ))
        elif options.tool_choice == "none":
            segments.extend(_internal_system_message(
                "tool-choice",
                "The system is invoked with `tool_choice=none`.\n"
                "You MUST NOT call any tools in the next message.",
            ))
    response_format = options.response_format
    if response_format is not None:
        format_type = (
            response_format.get("type")
            if isinstance(response_format, dict)
            else response_format
        )
        if format_type == "json_object":
            segments.extend(_internal_system_message(
                "response-format",
                "The system is invoked with `response_format=json_object`.\n"
                "Your response must be raw JSON data without markdown code "
                "blocks (```json) or any additional formatting.",
            ))
        elif format_type == "json_schema":
            schema = _json_compact(
                _deep_sort_dict(_extract_response_schema(response_format))
            )
            segments.extend(_internal_system_message(
                "response-format",
                "The system is invoked with `response_format=json_schema`.\n"
                "Your response must be raw JSON data without markdown code "
                "blocks (```json) or any additional formatting.\n"
                "The JSON data must match the following schema:\n"
                f"```json\n{schema}\n```",
            ))
        else:
            raise UnsupportedChatCapability(
                "kimi_k3",
                f"response_format={format_type}",
            )
    return segments


def _reject_unsupported(
    messages: tuple[ChatMessage, ...],
    options: ChatOptions,
) -> None:
    if isinstance(options.tool_choice, dict):
        raise UnsupportedChatCapability("kimi_k3", "named tool_choice")
    for message in messages:
        if message.role != "assistant":
            continue
        for call in message.tool_calls:
            _normalize_tool_arguments(call.function.arguments)


def _render(
    messages: tuple[ChatMessage, ...],
    options: ChatOptions,
) -> list[tuple[str, bool]]:
    thinking = options.thinking_mode == "thinking"
    segments: list[tuple[str, bool]] = []
    if options.tools:
        segments.extend(_render_tool_declare(options.tools))
    if thinking:
        effort = options.reasoning_effort or "max"
        if effort not in {"low", "medium", "high", "max"}:
            raise UnsupportedChatCapability(
                "kimi_k3",
                f"reasoning_effort={effort}",
            )
        segments.extend(_thinking_effort_message(effort))
    segments.extend(_render_messages(messages, thinking=thinking))
    segments.extend(_request_directives(options))
    segments.extend(_open_tag("message", role="assistant"))
    segments.extend(_open_tag("think" if thinking else "response"))
    return segments


def _encode_segments(
    engine: object,
    segments: list[tuple[str, bool]],
) -> list[int]:
    ids: list[int] = []
    tokenizer = getattr(engine, "tok", None)
    for text, allow_special in segments:
        if not text:
            continue
        if tokenizer is not None:
            encoded = tokenizer.encode(
                text,
                allow_special_tokens=allow_special,
            )
            ids.extend(encoded.ids)
        else:
            ids.extend(engine.encode(text))
    return ids


def _decode_raw(engine: object, output_ids: list[int]) -> str:
    tokenizer = getattr(engine, "tok", None)
    if tokenizer is not None:
        return tokenizer.decode(
            output_ids,
            skip_special_tokens=False,
        )
    return engine.decode(output_ids)


_THINK_TO_RESPONSE = (
    f"{CLOSE}think{SEP}{OPEN}response{SEP}"
)
_RESPONSE_END = f"{CLOSE}response{SEP}"
_TOOLS_HEAD = f"{OPEN}tools{SEP}"
_TOOLS_END = f"{CLOSE}tools{SEP}"
_MESSAGE_CLOSE = f"{CLOSE}message{SEP}"
_ATTR_RE = re.compile(r'(\w+)="((?:[^"&]|&amp;|&quot;)*)"')


def _typed_value(text: str, value_type: str) -> object:
    if value_type == "string":
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_attrs(header: str) -> dict[str, str]:
    return {
        key: _unescape_attr(value)
        for key, value in _ATTR_RE.findall(header)
    }


def _parse_calls(block: str) -> list[ToolCall]:
    """Parse one XTML tools section body into OpenAI-style tool calls."""
    calls: list[ToolCall] = []
    for chunk in block.split(f"{OPEN}call "):
        if not chunk:
            continue
        header, sep, body = chunk.partition(SEP)
        if not sep:
            raise ValueError(f"malformed Kimi tool call segment: {header!r}")
        name = _parse_attrs(header).get("tool", "")
        if not name:
            raise ValueError("Kimi tool call is missing the tool attribute")
        arguments: dict[str, object] = {}
        json_block: str | None = None
        for part in body.split(OPEN):
            if not part:
                continue
            ahead, asep, abody = part.partition(SEP)
            if ahead.startswith(CLOSE):
                continue
            if not asep:
                raise ValueError("malformed Kimi tool argument segment")
            if ahead.startswith("argument"):
                attrs = _parse_attrs(ahead)
                key = attrs.get("key", "")
                if not key:
                    raise ValueError(
                        "Kimi tool argument is missing the key attribute"
                    )
                value_text = abody.split(f"{CLOSE}argument{SEP}")[0]
                arguments[key] = _typed_value(
                    value_text,
                    attrs.get("type", "string"),
                )
            elif ahead.startswith("json"):
                json_block = abody.split(f"{CLOSE}json{SEP}")[0].strip()
            else:
                raise ValueError(f"unexpected Kimi tool segment: {ahead!r}")
        if json_block is not None:
            json.loads(json_block)
            argument_text = json_block
        else:
            argument_text = _json_compact(arguments)
        calls.append(
            ToolCall(
                id=f"call_{secrets.token_hex(12)}",
                function=ToolFunction(name=name, arguments=argument_text),
            )
        )
    return calls


def _parse_text(text: str, *, thinking: bool) -> AssistantOutput:
    reasoning = None
    content = text
    if thinking:
        if _THINK_TO_RESPONSE not in text:
            return AssistantOutput(
                reasoning_content=text or None,
                content="",
                tool_calls=[],
            )
        reasoning, content = text.split(_THINK_TO_RESPONSE, 1)
    content, sep, tail = content.partition(_RESPONSE_END)
    tool_calls: list[ToolCall] = []
    warnings: tuple[AdapterWarning, ...] = ()
    if sep and tail.startswith(_TOOLS_HEAD):
        block = tail[len(_TOOLS_HEAD):]
        end = block.find(_TOOLS_END)
        if end >= 0:
            block = block[:end]
        try:
            tool_calls = _parse_calls(block)
        except ValueError as error:
            warnings = (
                AdapterWarning(
                    code="kimi_tool_parse_fallback",
                    message=str(error),
                ),
            )
            content = content + sep + tail
    return AssistantOutput(
        reasoning_content=reasoning or None,
        content=content,
        tool_calls=tool_calls,
        warnings=warnings,
    )


class _KimiStreamParser:
    def __init__(self, *, thinking: bool):
        self._thinking = thinking
        self._phase = "reasoning" if thinking else "content"
        self._buffer = ""
        self._reasoning = ""
        self._content = ""
        self._tool_calls: list[ToolCall] = []
        self._warnings: list[AdapterWarning] = []
        self._finished = False

    def _drain(
        self,
        marker: str,
        kind: str,
    ) -> tuple[StreamDelta, ...]:
        found = self._buffer.find(marker)
        if found >= 0:
            text = self._buffer[:found]
            self._buffer = self._buffer[found + len(marker):]
            if kind == "reasoning":
                self._reasoning += text
                self._phase = "content"
            else:
                self._content += text
                self._phase = "tail"
            return (StreamDelta(kind=kind, text=text),) if text else ()
        safe = max(0, len(self._buffer) - len(marker) + 1)
        if safe == 0:
            return ()
        text = self._buffer[:safe]
        self._buffer = self._buffer[safe:]
        if kind == "reasoning":
            self._reasoning += text
        else:
            self._content += text
        return (StreamDelta(kind=kind, text=text),) if text else ()

    def _finish_tools(self, block: str) -> tuple[StreamDelta, ...]:
        try:
            calls = _parse_calls(block)
        except ValueError as error:
            self._warnings.append(AdapterWarning(
                code="kimi_tool_parse_fallback",
                message=str(error),
            ))
            self._content += block
            return (StreamDelta(kind="content", text=block),) if block else ()
        self._tool_calls = calls
        if not calls:
            return ()
        return (StreamDelta(kind="tool_calls", tool_calls=tuple(calls)),)

    def _drain_tail(self) -> tuple[StreamDelta, ...]:
        if self._phase == "tail":
            if self._buffer.startswith(_TOOLS_HEAD):
                self._buffer = self._buffer[len(_TOOLS_HEAD):]
                self._phase = "tools"
            elif _TOOLS_HEAD.startswith(self._buffer) or (
                _MESSAGE_CLOSE.startswith(self._buffer)
            ):
                return ()
            elif self._buffer.startswith(CLOSE):
                self._buffer = ""
                self._phase = "done"
                return ()
            elif self._buffer:
                text = self._buffer
                self._buffer = ""
                self._phase = "done"
                self._warnings.append(AdapterWarning(
                    code="kimi_tail_text",
                    message="unexpected text after the response channel",
                ))
                self._content += text
                return (StreamDelta(kind="content", text=text),)
            return ()
        end = self._buffer.find(_TOOLS_END)
        if end < 0:
            return ()
        block = self._buffer[:end]
        self._buffer = ""
        self._phase = "done"
        return self._finish_tools(block)

    def feed(self, text: str) -> tuple[StreamDelta, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished Kimi stream parser")
        self._buffer += text
        deltas: list[StreamDelta] = []
        while True:
            previous = self._phase
            if self._phase == "reasoning":
                deltas.extend(self._drain(
                    _THINK_TO_RESPONSE,
                    "reasoning",
                ))
            elif self._phase == "content":
                deltas.extend(self._drain(
                    _RESPONSE_END,
                    "content",
                ))
            elif self._phase in {"tail", "tools"}:
                deltas.extend(self._drain_tail())
            else:
                self._buffer = ""
                break
            if self._phase == previous:
                break
        return tuple(deltas)

    def finish(self) -> tuple[AssistantOutput, tuple[StreamDelta, ...]]:
        if self._finished:
            raise RuntimeError("Kimi stream parser is already finished")
        self._finished = True
        deltas: tuple[StreamDelta, ...] = ()
        if self._phase == "reasoning" and self._buffer:
            self._reasoning += self._buffer
            deltas = (
                StreamDelta(kind="reasoning", text=self._buffer),
            )
        elif self._phase == "content" and self._buffer:
            self._content += self._buffer
            deltas = (
                StreamDelta(kind="content", text=self._buffer),
            )
        elif self._phase == "tail" and self._buffer:
            if self._buffer.startswith(_TOOLS_HEAD):
                deltas = self._finish_tools(
                    self._buffer[len(_TOOLS_HEAD):]
                )
            elif not _MESSAGE_CLOSE.startswith(self._buffer):
                self._content += self._buffer
                deltas = (
                    StreamDelta(kind="content", text=self._buffer),
                )
        elif self._phase == "tools" and self._buffer:
            deltas = self._finish_tools(self._buffer)
        self._buffer = ""
        return (
            AssistantOutput(
                reasoning_content=self._reasoning or None,
                content=self._content,
                tool_calls=self._tool_calls,
                warnings=tuple(self._warnings),
            ),
            deltas,
        )


@dataclass
class KimiTokenLedger:
    committed_messages: tuple[ChatMessage, ...] = ()
    completed_ids: list[int] | None = None
    thinking_mode: str | None = None
    signature: str | None = None
    media_digest: str | None = None
    media_slots: tuple[dict[str, object], ...] = ()
    media_state: object = None

    def clear(self) -> None:
        self.committed_messages = ()
        self.completed_ids = None
        self.thinking_mode = None
        self.signature = None
        self.media_digest = None
        self.media_slots = ()
        self.media_state = None


def _hot_signature(options: ChatOptions) -> str:
    """Identify request-level values already committed into the prefix."""
    return json.dumps(
        {
            "thinking_mode": options.thinking_mode,
            "reasoning_effort": options.reasoning_effort,
            "tools": options.tools,
            "tool_choice": options.tool_choice,
            "response_format": options.response_format,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _hot_eligible(options: ChatOptions) -> bool:
    # tool-choice/response-format directives only render at the conversation
    # tail, so a request carrying them cannot reuse a committed prefix.
    if options.response_format is not None:
        return False
    if options.tools and options.tool_choice in {"required", "none"}:
        return False
    return True


def _committed_prefix_matches(
    normalized: tuple[ChatMessage, ...],
    committed: tuple[ChatMessage, ...],
) -> bool:
    """消息级前缀匹配：reasoning_content 不参与比较。

    思维链已经固化在 committed token/KDA 状态里，客户端回传历史时是否
    带回 reasoning_content 不影响前缀复用的正确性；角色、正文、工具
    调用和工具结果 id 必须严格一致，其余差异说明对话分叉，不能复用。
    """
    if len(normalized) < len(committed):
        return False
    for new, old in zip(normalized, committed):
        if (
            new.role != old.role
            or new.content != old.content
            or new.tool_calls != old.tool_calls
            or new.tool_call_id != old.tool_call_id
        ):
            return False
    return True


class KimiK3ChatAdapter:
    name = "kimi_k3"

    def prepare(
        self,
        engine: object,
        messages: list[ChatMessage],
        options: ChatOptions,
        hot_ledger: object | None,
    ) -> PromptPlan:
        normalized = tuple(messages)
        _reject_unsupported(normalized, options)
        signature = _hot_signature(options)
        rendered_segments = _render(normalized, options)
        input_ids = _encode_segments(engine, rendered_segments)
        slots = []
        cursor = 0
        references = _media_slots(normalized)
        reference_index = 0
        for text, allow_special in rendered_segments:
            encoded = _encode_segments(engine, [(text, allow_special)])
            search_from = 0
            while True:
                matches = [
                    (IMAGE_PLACEHOLDER, "image"),
                    (VIDEO_PLACEHOLDER, "video"),
                ]
                found = [
                    (text.find(value, search_from), value, kind)
                    for value, kind in matches
                    if text.find(value, search_from) >= 0
                ]
                if not found:
                    break
                offset, placeholder, kind = min(found)
                slot = dict(references[reference_index])
                slot["kind"] = kind
                prefix_text = text[:offset] + placeholder.split(MEDIA_PAD, 1)[0]
                prefix_ids = _encode_segments(
                    engine,
                    [(prefix_text, allow_special)],
                )
                media_pad_ids = _encode_segments(
                    engine,
                    [(MEDIA_PAD, True)],
                )
                if len(media_pad_ids) != 1:
                    raise RuntimeError("Kimi media_pad must encode to exactly one token")
                slot["token_start"] = cursor + len(prefix_ids)
                slot["length"] = 1
                slots.append(slot)
                reference_index += 1
                search_from = offset + len(placeholder)
            cursor += len(encoded)
        slots = tuple(slots)
        digest = media_references_digest(slots)
        base = (
            len(hot_ledger.committed_messages)
            if isinstance(hot_ledger, KimiTokenLedger)
            else 0
        )
        if (
            isinstance(hot_ledger, KimiTokenLedger)
            and hot_ledger.signature == signature
            and hot_ledger.media_digest == digest
            and hot_ledger.media_slots == slots
            and hot_ledger.completed_ids is not None
            and getattr(engine, "_cache_ids", None)
            == hot_ledger.completed_ids
            and len(normalized) > base
            and _committed_prefix_matches(
                normalized,
                hot_ledger.committed_messages,
            )
            and all(
                message.role in {"user", "tool"}
                for message in normalized[base:]
            )
            and _hot_eligible(options)
        ):
            seed: tuple[ToolCall, ...] = ()
            if base and normalized[base - 1].role == "assistant":
                seed = normalized[base - 1].tool_calls
            suffix = [
                (END_OF_MSG, True),
                *_render_messages(
                    normalized[base:],
                    thinking=options.thinking_mode == "thinking",
                    pending=seed,
                ),
                *_open_tag("message", role="assistant"),
                *_open_tag(
                    "think"
                    if options.thinking_mode == "thinking"
                    else "response"
                ),
            ]
            input_ids = [
                *hot_ledger.completed_ids,
                *_encode_segments(engine, suffix),
            ]
        else:
            input_ids = list(input_ids)
        media_state = {
            "media_digest": digest,
            "media_slots": slots,
        } if slots else None
        return PromptPlan(
            input_ids=input_ids,
            kv_baseline_len=len(input_ids),
            normalized_messages=normalized,
            canonical_prefix_ids=list(input_ids),
            adapter_state={
                "thinking_mode": options.thinking_mode,
                "signature": signature,
                "media_digest": digest,
                "media_slots": slots,
                "media_state": media_state,
            },
        )

    def parse_complete(
        self,
        engine: object,
        output_ids: list[int],
        options: ChatOptions,
    ) -> AssistantOutput:
        return _parse_text(
            _decode_raw(engine, list(output_ids)),
            thinking=options.thinking_mode == "thinking",
        )

    def new_stream_parser(
        self,
        engine: object,
        options: ChatOptions,
    ) -> StreamParser:
        del engine
        return _KimiStreamParser(
            thinking=options.thinking_mode == "thinking",
        )

    def commit(
        self,
        engine: object,
        plan: PromptPlan,
        output_ids: list[int],
        parsed: AssistantOutput,
    ) -> KimiTokenLedger:
        completed_ids = [*plan.input_ids, *output_ids]
        if getattr(engine, "_cache_ids", None) != completed_ids:
            completed_ids = None
        return KimiTokenLedger(
            committed_messages=plan.normalized_messages + (
                ChatMessage(
                    role="assistant",
                    content=parsed.content,
                    reasoning_content=parsed.reasoning_content,
                    tool_calls=tuple(parsed.tool_calls),
                ),
            ),
            completed_ids=completed_ids,
            thinking_mode=plan.adapter_state["thinking_mode"],
            signature=plan.adapter_state["signature"],
            media_digest=plan.adapter_state["media_digest"],
            media_slots=plan.adapter_state["media_slots"],
            media_state=plan.adapter_state["media_state"],
        )


__all__ = [
    "KimiK3ChatAdapter",
    "KimiTokenLedger",
]

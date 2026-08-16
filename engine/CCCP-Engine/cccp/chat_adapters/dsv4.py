"""DeepSeek-V4 chat adapter with exact-token hot-conversation reuse."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Any

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
from .dsv4_encoding import (
    dsml_token,
    encode_messages,
    eos_token,
    parse_message_from_completion_text,
    thinking_end_token,
    thinking_start_token,
)

_DSML_START = f"\n\n<{dsml_token}tool_calls>"
_DSML_END = f"</{dsml_token}tool_calls>"
_REQUIRED_TOOL_INSTRUCTION = (
    "You MUST call at least one available tool before giving your response."
)
_NAMED_TOOL_INSTRUCTION = (
    'You MUST call the function "{name}" before giving your response.'
)
_SERIAL_TOOL_INSTRUCTION = "You MUST produce at most one tool call."


def _token_lcp(left: list[int], right: list[int]) -> int:
    """Return the length of the common token prefix."""
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def _find_token_subsequence(values: list[int], target: list[int]) -> int | None:
    """Return the first exact occurrence of ``target`` in ``values``."""
    if not target:
        raise ValueError("empty token subsequence")
    for index in range(len(values) - len(target) + 1):
        if values[index : index + len(target)] == target:
            return index
    return None


def _decode_raw(engine: object, ids: list[int]) -> str:
    """Decode exact model tokens without dropping DSV4 special delimiters."""
    tokenizer = getattr(engine, "tok", None)
    tokenizer_decode = getattr(tokenizer, "decode", None)
    if callable(tokenizer_decode):
        return tokenizer_decode(ids, skip_special_tokens=False)
    return engine.decode(ids)


def _message_to_official(message: ChatMessage) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.reasoning_content is not None:
        result["reasoning_content"] = message.reasoning_content
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        result["tool_call_id"] = message.tool_call_id
    return result


def _tool_name(tool: dict) -> str:
    function = tool.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if not isinstance(name, str) or not name:
        raise ValueError("tool schemas must identify a non-empty function name")
    return name


def _tool_prompt_config(
    options: ChatOptions,
) -> tuple[tuple[dict, ...], str]:
    """Select schemas and deterministic restrictions for one request."""
    choice = options.tool_choice
    instructions: list[str] = []
    tools = options.tools

    if choice == "none":
        return (), ""
    if choice == "auto":
        selected = tools
    elif choice == "required":
        if not tools:
            raise ValueError("tool_choice='required' requires at least one tool")
        selected = tools
        instructions.append(_REQUIRED_TOOL_INSTRUCTION)
    elif isinstance(choice, dict):
        function = choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        selected = tuple(tool for tool in tools if _tool_name(tool) == name)
        if not selected:
            raise ValueError(f"unknown tool in named tool_choice: {name!r}")
        instructions.append(_NAMED_TOOL_INSTRUCTION.format(name=name))
    else:
        raise ValueError(f"unsupported normalized tool_choice: {choice!r}")

    if selected and not options.parallel_tool_calls:
        instructions.append(_SERIAL_TOOL_INSTRUCTION)
    return selected, "\n".join(instructions)


def _normalize_messages(
    messages: list[ChatMessage],
    options: ChatOptions,
) -> tuple[ChatMessage, ...]:
    normalized = tuple(messages)
    prompt_tools, tool_instruction = _tool_prompt_config(options)
    needs_system = bool(prompt_tools or tool_instruction or options.response_format)
    if needs_system and (not normalized or normalized[0].role != "system"):
        normalized = (ChatMessage(role="system"),) + normalized
    return normalized


def _official_messages(
    messages: tuple[ChatMessage, ...],
    options: ChatOptions,
) -> list[dict[str, Any]]:
    official = [_message_to_official(message) for message in messages]
    prompt_tools, tool_instruction = _tool_prompt_config(options)
    if prompt_tools or tool_instruction or options.response_format:
        if not official or official[0].get("role") != "system":
            raise RuntimeError(
                "DSV4 top-level controls require a normalized system message"
            )
        if tool_instruction:
            content = official[0].get("content") or ""
            official[0]["content"] = (
                f"{content}\n\n{tool_instruction}" if content else tool_instruction
            )
        if prompt_tools:
            official[0]["tools"] = list(prompt_tools)
        if options.response_format is not None:
            official[0]["response_format"] = options.response_format
    return official


def _hot_signature(options: ChatOptions) -> str:
    """Identify request-level values already committed into the system prefix."""
    return json.dumps(
        {
            "thinking_mode": options.thinking_mode,
            "reasoning_effort": options.reasoning_effort,
            "tools": options.tools,
            "tool_choice": options.tool_choice,
            "parallel_tool_calls": options.parallel_tool_calls,
            "response_format": options.response_format,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _render_prompt(
    messages: list[dict[str, Any]],
    options: ChatOptions,
    *,
    context: list[dict[str, Any]] | None = None,
) -> str:
    if options.reasoning_effort not in {None, "high", "max"}:
        raise UnsupportedChatCapability(
            "dsv4",
            f"reasoning_effort={options.reasoning_effort}",
        )
    return encode_messages(
        messages,
        thinking_mode=options.thinking_mode,
        context=context,
        reasoning_effort=options.reasoning_effort,
    )


def _split_stable_prompt(rendered: str, options: ChatOptions) -> tuple[str, str]:
    marker = (
        thinking_start_token
        if options.thinking_mode == "thinking"
        else thinking_end_token
    )
    if not rendered.endswith(marker):
        raise ValueError(
            "DSV4 generation prompt must end with the active thinking marker"
        )
    return rendered[: -len(marker)], marker


def _longest_delimiter_prefix(
    text: str,
    delimiters: tuple[str, ...],
) -> int:
    """Return the longest suffix that could become a delimiter."""
    return max(
        (
            length
            for delimiter in delimiters
            for length in range(
                1,
                min(len(delimiter), len(text) + 1),
            )
            if text.endswith(delimiter[:length])
        ),
        default=0,
    )


def _first_delimiter(
    text: str,
    delimiters: tuple[str, ...],
) -> tuple[int, str] | None:
    matches = [
        (position, index, delimiter)
        for index, delimiter in enumerate(delimiters)
        if (position := text.find(delimiter)) >= 0
    ]
    if not matches:
        return None
    position, _index, delimiter = min(matches)
    return position, delimiter


class _DSV4StreamParser:
    """Incremental parser that never exposes stable DSV4 delimiters."""

    _TEXT_DELIMITERS = (
        thinking_start_token,
        thinking_end_token,
        _DSML_START,
        eos_token,
    )

    def __init__(self, options: ChatOptions) -> None:
        self.options = options
        self._raw = ""
        self._reasoning = ""
        self._content = ""
        self._tool_calls: list[ToolCall] = []
        self._warnings: list[AdapterWarning] = []
        self._emitted_reasoning = 0
        self._emitted_content = 0
        self._pending = ""
        self._dsml_raw = ""
        self._state = "reasoning" if options.thinking_mode == "thinking" else "content"
        self._finished = False

    def _emit_text(self, text: str, kind: str) -> StreamDelta | None:
        if not text:
            return None
        if kind == "reasoning":
            self._reasoning += text
            self._emitted_reasoning += len(text)
        else:
            self._content += text
            self._emitted_content += len(text)
        return StreamDelta(kind=kind, text=text)

    def _tool_calls_from_dsml(self, block: str) -> list[ToolCall]:
        if self.options.thinking_mode == "thinking":
            completion_text = (
                self._reasoning + thinking_end_token + self._content + block + eos_token
            )
        else:
            completion_text = self._content + block + eos_token
        official = parse_message_from_completion_text(
            completion_text,
            thinking_mode=self.options.thinking_mode,
        )

        prompt_tools, _instruction = _tool_prompt_config(self.options)
        allowed_names = {_tool_name(tool) for tool in prompt_tools}
        tool_calls: list[ToolCall] = []
        for raw_call in official.get("tool_calls", []):
            function = raw_call["function"]
            name = function["name"]
            if name not in allowed_names:
                raise ValueError(f"tool call uses unavailable function {name!r}")
            tool_calls.append(
                ToolCall(
                    id=f"call_{secrets.token_hex(12)}",
                    function=ToolFunction(
                        name=name,
                        arguments=function["arguments"],
                    ),
                )
            )
        return tool_calls

    def _fallback_dsml(
        self,
        block: str,
        *,
        code: str,
        message: str,
    ) -> tuple[StreamDelta, ...]:
        self._warnings.append(AdapterWarning(code=code, message=message))
        delta = self._emit_text(block, "content")
        return () if delta is None else (delta,)

    def _finish_dsml(self, block: str) -> tuple[StreamDelta, ...]:
        try:
            tool_calls = self._tool_calls_from_dsml(block)
        except (
            AssertionError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return self._fallback_dsml(
                block,
                code="malformed_dsml",
                message=(
                    "DSML tool block could not be parsed and was returned "
                    "as assistant text"
                ),
            )

        if not self.options.parallel_tool_calls and len(tool_calls) > 1:
            return self._fallback_dsml(
                block,
                code="parallel_tool_calls_disabled",
                message=(
                    "Multiple DSML tool calls were returned when parallel "
                    "tool calls were disabled"
                ),
            )

        self._tool_calls.extend(tool_calls)
        if not tool_calls:
            return ()
        return (
            StreamDelta(
                kind="tool_calls",
                tool_calls=tuple(tool_calls),
            ),
        )

    def _drain(self, *, force: bool) -> tuple[StreamDelta, ...]:
        deltas: list[StreamDelta] = []
        while self._pending or self._state == "dsml":
            if self._state == "dsml":
                close_at = self._pending.find(_DSML_END)
                if close_at < 0:
                    if force:
                        block = self._dsml_raw + self._pending
                        self._pending = ""
                        self._dsml_raw = ""
                        deltas.extend(
                            self._fallback_dsml(
                                block,
                                code="malformed_dsml",
                                message=(
                                    "Incomplete DSML tool block was returned "
                                    "as assistant text"
                                ),
                            )
                        )
                        self._state = "content"
                    break

                close_end = close_at + len(_DSML_END)
                block = self._dsml_raw + self._pending[:close_end]
                self._pending = self._pending[close_end:]
                self._dsml_raw = ""
                deltas.extend(self._finish_dsml(block))
                self._state = "content"
                continue

            match = _first_delimiter(
                self._pending,
                self._TEXT_DELIMITERS,
            )
            if match is not None:
                position, delimiter = match
                visible = self._pending[:position]
                self._pending = self._pending[position + len(delimiter) :]
                delta = self._emit_text(visible, self._state)
                if delta is not None:
                    deltas.append(delta)

                if delimiter == thinking_start_token:
                    self._state = "reasoning"
                elif delimiter == thinking_end_token:
                    self._state = "content"
                elif delimiter == _DSML_START:
                    self._state = "dsml"
                    self._dsml_raw = delimiter
                else:
                    # EOS controls generation/KV state but is not assistant
                    # content.  It may arrive split across tokenizer chunks;
                    # delimiter prefix retention above keeps it hidden until
                    # the complete marker is available.
                    self._pending = ""
                    break
                continue

            if force:
                visible = self._pending
                self._pending = ""
            else:
                retain = _longest_delimiter_prefix(
                    self._pending,
                    self._TEXT_DELIMITERS,
                )
                split = len(self._pending) - retain
                if split == 0:
                    break
                visible = self._pending[:split]
                self._pending = self._pending[split:]
            delta = self._emit_text(visible, self._state)
            if delta is not None:
                deltas.append(delta)
            break
        return tuple(deltas)

    def feed(self, text: str) -> tuple[StreamDelta, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished DSV4 stream parser")
        if not isinstance(text, str):
            raise TypeError("stream parser text must be a string")
        if not text:
            return ()
        self._raw += text
        self._pending += text
        return self._drain(force=False)

    def finish(self) -> tuple[AssistantOutput, tuple[StreamDelta, ...]]:
        if self._finished:
            raise RuntimeError("DSV4 stream parser is already finished")
        final_deltas = self._drain(force=True)
        self._finished = True
        return (
            AssistantOutput(
                reasoning_content=self._reasoning or None,
                content=self._content,
                tool_calls=list(self._tool_calls),
                warnings=tuple(self._warnings),
            ),
            final_deltas,
        )


@dataclass(frozen=True)
class _DSV4PlanState:
    has_tools: bool
    thinking_mode: str
    encoding_signature: str


@dataclass
class DSV4TokenLedger:
    """Exact committed token history and its structured message counterpart."""

    completed_ids: list[int] | None = None
    committed_messages: tuple[ChatMessage, ...] = ()

    def clear(self) -> None:
        self.completed_ids = None
        self.committed_messages = ()
        if hasattr(self, "_encoding_signature"):
            del self._encoding_signature


class DSV4ChatAdapter:
    """Official DSV4 prompt planning and completion parsing."""

    name = "dsv4"

    def prepare(
        self,
        engine: object,
        messages: list[ChatMessage],
        options: ChatOptions,
        hot_ledger: object | None,
    ) -> PromptPlan:
        normalized = _normalize_messages(messages, options)
        signature = _hot_signature(options)
        ledger = hot_ledger if isinstance(hot_ledger, DSV4TokenLedger) else None

        hot_count = len(ledger.committed_messages) if ledger is not None else 0
        can_extend_hot = (
            ledger is not None
            and ledger.completed_ids is not None
            and hot_count > 0
            and len(normalized) > hot_count
            and normalized[:hot_count] == ledger.committed_messages
            and getattr(ledger, "_encoding_signature", None) == signature
        )

        if can_extend_hot:
            context = _official_messages(ledger.committed_messages, options)
            suffix = [
                _message_to_official(message) for message in normalized[hot_count:]
            ]
            rendered_suffix = _render_prompt(suffix, options, context=context)
            stable_suffix, _marker = _split_stable_prompt(
                rendered_suffix,
                options,
            )
            suffix_ids = engine.encode(rendered_suffix)
            stable_suffix_ids = engine.encode(stable_suffix)
            suffix_baseline = _token_lcp(suffix_ids, stable_suffix_ids)
            input_ids = list(ledger.completed_ids) + suffix_ids
            baseline_len = len(ledger.completed_ids) + suffix_baseline
        else:
            official = _official_messages(normalized, options)
            rendered = _render_prompt(official, options)
            stable, _marker = _split_stable_prompt(rendered, options)
            input_ids = engine.encode(rendered)
            baseline_len = _token_lcp(input_ids, engine.encode(stable))

        if baseline_len <= 0 or baseline_len > len(input_ids):
            raise RuntimeError(
                f"invalid DSV4 prompt baseline {baseline_len} "
                f"for {len(input_ids)} input tokens"
            )

        return PromptPlan(
            input_ids=list(input_ids),
            kv_baseline_len=baseline_len,
            normalized_messages=normalized,
            canonical_prefix_ids=list(input_ids[:baseline_len]),
            adapter_state=_DSV4PlanState(
                has_tools=bool(_tool_prompt_config(options)[0]),
                thinking_mode=options.thinking_mode,
                encoding_signature=signature,
            ),
        )

    def parse_complete(
        self,
        engine: object,
        output_ids: list[int],
        options: ChatOptions,
    ) -> AssistantOutput:
        raw_text = _decode_raw(engine, list(output_ids))
        if raw_text.endswith(eos_token):
            raw_text = raw_text[: -len(eos_token)]
        parser = self.new_stream_parser(engine, options)
        parser.feed(raw_text)
        parsed, _final_deltas = parser.finish()
        return parsed

    def new_stream_parser(
        self,
        engine: object,
        options: ChatOptions,
    ) -> StreamParser:
        del engine
        return _DSV4StreamParser(options)

    def commit(
        self,
        engine: object,
        plan: PromptPlan,
        output_ids: list[int],
        parsed: AssistantOutput,
    ) -> DSV4TokenLedger:
        state = plan.adapter_state
        if not isinstance(state, _DSV4PlanState):
            raise TypeError("prompt plan was not prepared by DSV4ChatAdapter")
        if not 0 < plan.kv_baseline_len <= len(plan.input_ids):
            raise ValueError(
                f"invalid DSV4 baseline {plan.kv_baseline_len} "
                f"for prompt length {len(plan.input_ids)}"
            )

        expected_live = list(plan.input_ids) + list(output_ids)
        live_ids = getattr(engine, "_cache_ids", None)
        if live_ids is None or list(live_ids) != expected_live:
            raise RuntimeError("live cache IDs do not match committed DSV4 turn")

        eos_ids = engine.encode(eos_token)
        close_ids = engine.encode(thinking_end_token)

        def with_eos_once(values: list[int]) -> list[int]:
            result = list(values)
            if not eos_ids or result[-len(eos_ids) :] != eos_ids:
                result.extend(eos_ids)
            return result

        completed_ids: list[int]
        keep_structured_reasoning = state.has_tools
        if state.thinking_mode == "thinking":
            close_at = _find_token_subsequence(output_ids, close_ids)
            if close_at is None:
                keep_structured_reasoning = False
                completed_ids = with_eos_once(
                    list(plan.input_ids[: plan.kv_baseline_len]) + close_ids
                )
            elif state.has_tools:
                completed_ids = with_eos_once(
                    list(plan.input_ids) + list(output_ids)
                )
            else:
                final_answer_ids = output_ids[close_at + len(close_ids) :]
                completed_ids = with_eos_once(
                    list(plan.input_ids[: plan.kv_baseline_len])
                    + close_ids
                    + list(final_answer_ids)
                )
        else:
            completed_ids = with_eos_once(
                list(plan.input_ids) + list(output_ids)
            )

        assistant_message = ChatMessage(
            role="assistant",
            content=parsed.content,
            reasoning_content=(
                parsed.reasoning_content if keep_structured_reasoning else None
            ),
            tool_calls=tuple(parsed.tool_calls),
        )
        ledger = DSV4TokenLedger(
            completed_ids=completed_ids,
            committed_messages=plan.normalized_messages + (assistant_message,),
        )
        ledger._encoding_signature = state.encoding_signature
        return ledger


__all__ = ["DSV4ChatAdapter", "DSV4TokenLedger"]

"""Transport-independent types for model chat adapters.

This module deliberately does not depend on an HTTP framework or model
implementation.  Transports normalize requests into these values before they
reach an adapter, and adapters return these values before a transport
serializes a response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol


_MESSAGE_ROLES = frozenset(
    {"system", "developer", "user", "assistant", "tool", "latest_reminder"}
)


class UnsupportedChatArchitecture(ValueError):
    """The loaded model architecture has no registered chat adapter."""

    def __init__(self, architecture: str) -> None:
        self.architecture = architecture
        super().__init__(f"unsupported chat architecture: {architecture}")


class UnsupportedChatCapability(ValueError):
    """A request uses a feature for which an architecture has no template."""

    def __init__(self, architecture: str, capability: str) -> None:
        self.architecture = architecture
        self.capability = capability
        super().__init__(
            f"{architecture.upper()} chat adapter does not support {capability}"
        )


@dataclass(frozen=True)
class ToolFunction:
    name: str
    arguments: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tool function name must be a non-empty string")
        if not isinstance(self.arguments, str):
            raise TypeError("tool function arguments must be a JSON string")
        try:
            json.loads(self.arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("tool function arguments must be valid JSON") from exc


@dataclass(frozen=True)
class ToolCall:
    id: str
    function: ToolFunction
    type: str = "function"

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("tool call id must be a string")
        if not isinstance(self.function, ToolFunction):
            raise TypeError("tool call function must be a ToolFunction")
        if self.type != "function":
            raise ValueError("tool call type must be 'function'")


@dataclass(frozen=True)
class ChatContentPart:
    """One ordered, transport-normalized message content part."""

    type: str
    text: str | None = None
    url: str | None = None

    def __post_init__(self) -> None:
        if self.type == "text":
            if not isinstance(self.text, str) or self.url is not None:
                raise ValueError("text content parts require text and no URL")
            return
        if self.type not in {"image_url", "input_image", "video_url", "input_video"}:
            raise ValueError(f"unsupported chat content part type: {self.type!r}")
        if not isinstance(self.url, str) or not self.url or self.text is not None:
            raise ValueError("media content parts require a non-empty URL and no text")


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str = ""
    content_parts: tuple[ChatContentPart, ...] = ()
    reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or self.role not in _MESSAGE_ROLES:
            raise ValueError(f"unsupported chat message role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("chat message content must be a string")
        if not isinstance(self.content_parts, tuple) or not all(
            isinstance(part, ChatContentPart) for part in self.content_parts
        ):
            raise TypeError(
                "content_parts must be a tuple of ChatContentPart values"
            )
        if self.reasoning_content is not None and not isinstance(
            self.reasoning_content, str
        ):
            raise TypeError("reasoning_content must be a string or None")
        if not isinstance(self.tool_calls, tuple):
            raise TypeError("tool_calls must be a tuple of ToolCall values")
        if not all(isinstance(tool_call, ToolCall) for tool_call in self.tool_calls):
            raise TypeError("tool_calls must contain only ToolCall values")
        if self.tool_call_id is not None and not isinstance(self.tool_call_id, str):
            raise TypeError("tool_call_id must be a string or None")


@dataclass(frozen=True)
class ChatOptions:
    thinking_mode: str
    reasoning_effort: str | None
    temperature: float
    top_p: float
    max_new: int | None
    stop: tuple[str, ...] = ()
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    tools: tuple[dict, ...] = ()
    tool_choice: object = None
    parallel_tool_calls: bool = True
    response_format: dict | None = None

    def __post_init__(self) -> None:
        if self.thinking_mode not in {"chat", "thinking"}:
            raise ValueError("thinking_mode must be 'chat' or 'thinking'")
        if self.reasoning_effort is not None and not isinstance(
            self.reasoning_effort, str
        ):
            raise TypeError("reasoning_effort must be a string or None")
        if isinstance(self.max_new, bool) or (
            self.max_new is not None and not isinstance(self.max_new, int)
        ):
            raise TypeError("max_new must be an integer or None")
        if self.max_new is not None and self.max_new < 0:
            raise ValueError("max_new must be non-negative")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be greater than zero")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be between -2 and 2")
        if not isinstance(self.stop, tuple) or not all(
            isinstance(item, str) for item in self.stop
        ):
            raise TypeError("stop must be a tuple of strings")
        if not isinstance(self.tools, tuple) or not all(
            isinstance(tool, dict) for tool in self.tools
        ):
            raise TypeError("tools must be a tuple of dictionaries")
        tool_choice = self.tool_choice
        if tool_choice is None:
            normalized_tool_choice: object = "auto" if self.tools else "none"
        elif isinstance(tool_choice, str):
            if tool_choice not in {"none", "auto", "required"}:
                raise ValueError(
                    "tool_choice must be 'none', 'auto', 'required', "
                    "or a named function"
                )
            normalized_tool_choice = tool_choice
        elif isinstance(tool_choice, dict):
            function = tool_choice.get("function")
            if (
                tool_choice.get("type") != "function"
                or not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function["name"]
            ):
                raise ValueError(
                    "named tool_choice must identify a non-empty function name"
                )
            normalized_tool_choice = {
                "type": "function",
                "function": {"name": function["name"]},
            }
        else:
            raise TypeError(
                "tool_choice must be a string, named-function dictionary, or None"
            )
        object.__setattr__(self, "tool_choice", normalized_tool_choice)
        if type(self.parallel_tool_calls) is not bool:
            raise TypeError("parallel_tool_calls must be a bool")
        if self.response_format is not None and not isinstance(
            self.response_format, dict
        ):
            raise TypeError("response_format must be a dictionary or None")


@dataclass
class PromptPlan:
    input_ids: list[int]
    kv_baseline_len: int
    normalized_messages: tuple[ChatMessage, ...]
    canonical_prefix_ids: list[int] | None
    adapter_state: object = None


@dataclass(frozen=True)
class AdapterWarning:
    code: str
    message: str


@dataclass
class AssistantOutput:
    reasoning_content: str | None
    content: str
    tool_calls: list[ToolCall]
    warnings: tuple[AdapterWarning, ...] = ()


@dataclass(frozen=True)
class StreamDelta:
    kind: str
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


class StreamParser(Protocol):
    """Incrementally convert stable decoded text into assistant deltas."""

    def feed(self, text: str) -> tuple[StreamDelta, ...]:
        """Consume one stable decoded text chunk."""

    def finish(self) -> tuple[AssistantOutput, tuple[StreamDelta, ...]]:
        """Flush buffered text and return the complete parsed output."""


class ChatAdapter(Protocol):
    """Model-specific prompt planning, parsing, and committed-turn handling."""

    def prepare(
        self,
        engine: object,
        messages: list[ChatMessage],
        options: ChatOptions,
        hot_ledger: object | None,
    ) -> PromptPlan:
        """Create the exact prompt and KV baseline for a new turn."""

    def parse_complete(
        self,
        engine: object,
        output_ids: list[int],
        options: ChatOptions,
    ) -> AssistantOutput:
        """Parse a completed model output into a transport-independent result."""

    def new_stream_parser(
        self,
        engine: object,
        options: ChatOptions,
    ) -> StreamParser:
        """Create the adapter's incremental output parser."""

    def commit(
        self,
        engine: object,
        plan: PromptPlan,
        output_ids: list[int],
        parsed: AssistantOutput,
    ) -> object:
        """Commit the generated turn and return adapter-specific state."""

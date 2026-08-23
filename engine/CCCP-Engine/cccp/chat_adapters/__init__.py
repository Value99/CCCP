"""Public, framework-independent chat adapter interface."""

from .base import (
    AdapterWarning,
    AssistantOutput,
    ChatAdapter,
    ChatMessage,
    ChatOptions,
    PromptPlan,
    StreamDelta,
    StreamParser,
    ToolCall,
    ToolFunction,
    UnsupportedChatArchitecture,
    UnsupportedChatCapability,
)


def adapter_for_arch(arch: str) -> ChatAdapter:
    """Look up an adapter once the optional adapter registry is installed.

    Keeping the import local leaves this base package usable before concrete
    adapters and their registry are added in the next implementation step.
    """
    from .registry import adapter_for_arch as lookup

    return lookup(arch)


__all__ = [
    "AdapterWarning",
    "AssistantOutput",
    "ChatAdapter",
    "ChatMessage",
    "ChatOptions",
    "PromptPlan",
    "StreamDelta",
    "StreamParser",
    "ToolCall",
    "ToolFunction",
    "UnsupportedChatArchitecture",
    "UnsupportedChatCapability",
    "adapter_for_arch",
]

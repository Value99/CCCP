"""Explicit architecture-to-chat-adapter registry."""

from __future__ import annotations

from types import MappingProxyType

from .base import ChatAdapter, UnsupportedChatArchitecture
from .dsv4 import DSV4ChatAdapter
from .glm import GLMChatAdapter
from .kimi_k3 import KimiK3ChatAdapter
from .qwen35 import Qwen35ChatAdapter


_ADAPTERS = MappingProxyType(
    {
        "dsv4": DSV4ChatAdapter,
        "glm": GLMChatAdapter,
        "glm5_next": GLMChatAdapter,
        "kimi_k3": KimiK3ChatAdapter,
        "qwen3_5_dense": Qwen35ChatAdapter,
    }
)


def adapter_for_arch(arch: str) -> ChatAdapter:
    try:
        return _ADAPTERS[arch]()
    except KeyError as exc:
        raise UnsupportedChatArchitecture(arch) from exc


__all__ = ["UnsupportedChatArchitecture", "adapter_for_arch"]

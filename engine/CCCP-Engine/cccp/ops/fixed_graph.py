"""Model-independent fixed-address CUDA graph scheduling primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def fixed_token_capacity(
    required: int,
    *,
    limit: int,
    initial: int = 256,
) -> int:
    """Return a bounded power-of-two static-cache bucket.

    Fixed-address graphs require stable cache tensors, but the launcher must
    not reserve the model's entire context window for a short conversation.
    All topology adapters use this common bucket policy and recapture only
    when the live context crosses a bucket boundary.
    """
    required = max(1, int(required))
    limit = max(1, int(limit))
    if required > limit:
        raise ValueError(
            f"fixed token graph requires {required} positions, limit={limit}"
        )
    capacity = min(max(1, int(initial)), limit)
    while capacity < required:
        capacity = min(limit, capacity * 2)
    return capacity


class FixedAddressCudaGraph:
    """Capture one deterministic fixed-address callable for sequential replay.

    The caller owns all mutable inputs and may update their contents before
    ``replay``.  Shapes and addresses must not change.  A shared graph-pool
    handle may be supplied when several graphs are always replayed in their
    capture order, allowing their temporary buffers to alias safely.
    """

    def __init__(
        self,
        device: torch.device,
        function: Callable[[], object],
        *,
        pool=None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("fixed-address graph requires a CUDA device")
        self.stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            function()
        current.wait_stream(self.stream)
        torch.cuda.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(self.device):
            with torch.cuda.graph(
                self.graph,
                pool=pool,
                stream=self.stream,
            ):
                self.outputs = function()
        torch.cuda.synchronize(self.device)

    def replay(self):
        with torch.cuda.device(self.device):
            self.graph.replay()
        return self.outputs


class FixedTokenGraph:
    """Model-independent mutable-input wrapper for one-token graph replay.

    Topology adapters own the token/cache semantics, while this public helper
    owns the fixed-address token and position update contract.  A device token
    can be copied directly into the graph input, avoiding a host round-trip in
    GPU-resident greedy/speculative schedulers.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        token: torch.Tensor,
        position: torch.Tensor,
        function: Callable[[], object],
        pool=None,
        graph_factory: Callable[..., Any] = FixedAddressCudaGraph,
    ) -> None:
        self.device = torch.device(device)
        self.token = token
        self.position = position
        if token.device != self.device or position.device != self.device:
            raise ValueError("fixed token graph inputs must share its device")
        if token.dtype != torch.long or position.dtype != torch.long:
            raise ValueError("fixed token graph inputs must use torch.long")
        self.graph = graph_factory(
            self.device,
            function,
            pool=pool,
        )

    def replay(self, token: int | torch.Tensor, position: int):
        if isinstance(token, torch.Tensor):
            if token.device != self.device or token.dtype != self.token.dtype:
                raise ValueError(
                    "device token must match the fixed graph input device/dtype"
                )
            if token.numel() != self.token.numel():
                raise ValueError("device token shape does not match graph input")
            self.token.copy_(token.reshape_as(self.token))
        else:
            self.token.fill_(int(token))
        self.position.fill_(int(position))
        return self.graph.replay()


__all__ = [
    "FixedAddressCudaGraph",
    "FixedTokenGraph",
    "fixed_token_capacity",
]

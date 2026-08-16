"""Model-independent fixed-address CUDA graph scheduling primitives."""

from __future__ import annotations

from collections.abc import Callable

import torch


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


__all__ = ["FixedAddressCudaGraph"]

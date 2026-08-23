"""Fixed-address device control for single-token CUDA decode graphs.

The control block is deliberately model independent.  A runtime updates the
token and position once, while captured operators read the same device
addresses for the lifetime of the engine.  Optional named scalar slots are
available for paged-state counts and graph-bucket metadata.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


class DecodeControl:
    """One pinned-host to device control publication per decode token."""

    TOKEN = 0
    POSITION = 1

    def __init__(
        self,
        device: torch.device | str,
        *,
        scalar_names: Iterable[str] = (),
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DecodeControl requires a CUDA device")
        names = tuple(str(name) for name in scalar_names)
        if len(names) != len(set(names)):
            raise ValueError("DecodeControl scalar names must be unique")
        if any(name in {"token", "position"} for name in names):
            raise ValueError("token and position are reserved control names")
        self._slots = {
            "token": self.TOKEN,
            "position": self.POSITION,
            **{name: index + 2 for index, name in enumerate(names)},
        }
        self.host = torch.zeros(
            len(self._slots),
            dtype=torch.int64,
            pin_memory=True,
        )
        self.values = torch.zeros(
            len(self._slots),
            dtype=torch.int64,
            device=self.device,
        )

    @property
    def token(self) -> torch.Tensor:
        return self.values[self.TOKEN:self.TOKEN + 1]

    @property
    def position(self) -> torch.Tensor:
        return self.values[self.POSITION:self.POSITION + 1]

    def scalar(self, name: str) -> torch.Tensor:
        try:
            slot = self._slots[str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown decode-control scalar {name!r}") from exc
        return self.values[slot:slot + 1]

    def update(
        self,
        token: int,
        position: int,
        **scalars: int,
    ) -> None:
        if position < 0:
            raise ValueError("decode position must be non-negative")
        unknown = set(scalars) - set(self._slots)
        if unknown:
            raise KeyError(
                "unknown decode-control scalars: "
                + ", ".join(sorted(unknown))
            )
        self.host[self.TOKEN] = int(token)
        self.host[self.POSITION] = int(position)
        for name, value in scalars.items():
            self.host[self._slots[name]] = int(value)
        # non_blocking is a single 16/32-byte H2D publication from pinned RAM.
        self.values.copy_(self.host, non_blocking=True)


__all__ = ["DecodeControl"]

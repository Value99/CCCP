"""Low-overhead diagnostics for the generic tensor-parallel data flow."""

from __future__ import annotations

from dataclasses import dataclass
import statistics

import torch

from .hidden import TPHidden


@dataclass
class _TPStageEvents:
    name: str
    layer: int
    starts: tuple[torch.cuda.Event, ...]
    ends: tuple[torch.cuda.Event, ...] | None = None


class TPHiddenStageProfiler:
    """Measure all-rank graph envelopes without synchronizing each stage.

    A timing event is inserted into every rank's dependency chain before the
    operator.  End events wait for the operator's published TPHidden events.
    All measurements are resolved by one synchronization after the token, so
    the probe does not turn every layer boundary into a CPU/GPU barrier.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._records: list[_TPStageEvents] = []

    def begin(
        self,
        name: str,
        hidden: TPHidden,
        *,
        layer: int = -1,
    ) -> tuple[TPHidden, _TPStageEvents | None]:
        if not self.enabled:
            return hidden, None
        if hidden.ready_events is None:
            raise ValueError("TPHidden CUDA profiling requires ready events")
        starts = []
        for device, ready in zip(hidden.devices, hidden.ready_events):
            if device.type != "cuda":
                raise ValueError("TPHidden CUDA profiling requires CUDA ranks")
            with torch.cuda.device(device):
                stream = torch.cuda.current_stream(device)
                stream.wait_event(ready)
                event = torch.cuda.Event(enable_timing=True)
                event.record(stream)
                starts.append(event)
        record = _TPStageEvents(str(name), int(layer), tuple(starts))
        self._records.append(record)
        return (
            TPHidden(hidden.devices, hidden.replicas, tuple(starts)),
            record,
        )

    def end(
        self,
        record: _TPStageEvents | None,
        hidden: TPHidden,
    ) -> TPHidden:
        if record is None:
            return hidden
        if hidden.ready_events is None:
            raise ValueError("TPHidden CUDA profiling requires ready events")
        ends = []
        for device, ready in zip(hidden.devices, hidden.ready_events):
            with torch.cuda.device(device):
                stream = torch.cuda.current_stream(device)
                stream.wait_event(ready)
                event = torch.cuda.Event(enable_timing=True)
                event.record(stream)
                ends.append(event)
        record.ends = tuple(ends)
        return hidden

    def result(self, devices: tuple[torch.device, ...]) -> dict[str, object]:
        if not self.enabled:
            return {}
        for device in devices:
            torch.cuda.synchronize(device)
        items: list[dict[str, object]] = []
        grouped: dict[str, list[tuple[float, ...]]] = {}
        for record in self._records:
            if record.ends is None:
                raise RuntimeError(
                    f"TPHidden profile stage {record.name!r} was not ended"
                )
            elapsed = tuple(
                float(start.elapsed_time(end))
                for start, end in zip(record.starts, record.ends)
            )
            grouped.setdefault(record.name, []).append(elapsed)
            items.append(
                {
                    "layer": record.layer,
                    "stage": record.name,
                    "rank_ms": list(elapsed),
                    "critical_ms": max(elapsed),
                    "mean_rank_ms": statistics.fmean(elapsed),
                }
            )
        stages: dict[str, object] = {}
        critical_path_ms = 0.0
        for name, calls in grouped.items():
            critical = [max(call) for call in calls]
            rank_values = [value for call in calls for value in call]
            total = sum(critical)
            critical_path_ms += total
            stages[name] = {
                "calls": len(calls),
                "critical_total_ms": total,
                "critical_mean_ms": statistics.fmean(critical),
                "critical_max_ms": max(critical),
                "rank_mean_ms": statistics.fmean(rank_values),
            }
        return {
            "mode": "all_rank_async_cuda_events",
            "synchronizations": 1,
            "critical_path_ms": critical_path_ms,
            "totals": {
                f"{name}_ms": value["critical_total_ms"]
                for name, value in stages.items()
            },
            "stages": stages,
            "items": items,
        }


__all__ = ["TPHiddenStageProfiler"]

"""Record an interaction frame by frame for analysis and comparison.

A :class:`TraceRecorder` keeps every reading's channels and events and,
optionally, the engine outputs of the same frame (numbers and short
vectors; images are left out). It writes JSON Lines, one frame per line
with events and outputs, or a wide CSV of channels and scalar outputs, the
formats behavioural analysis tools read. Recording the same channels from
people and from simulated users makes their traces directly comparable.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .logic import Event
from .runtime import Reading

#: Longest output vector kept (a pose has seven numbers).
_MAX_VECTOR = 16


@dataclass
class TraceRecorder:
    """Accumulate frames of readings and engine outputs.

    Args:
        outputs: Engine output names to keep; all numeric outputs if None.
    """

    outputs: Iterable[str] | None = None
    frames: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self, reading: Reading, outputs: Mapping[str, Any] | None = None
    ) -> None:
        """Add one frame."""
        kept = {}
        wanted = None if self.outputs is None else set(self.outputs)
        for key, value in (outputs or {}).items():
            if wanted is not None and key not in wanted:
                continue
            number = _numeric(value)
            if number is not None:
                kept[key] = number
        self.frames.append(
            {
                "time": reading.time,
                "channels": dict(reading.channels),
                "events": [event.as_dict() for event in reading.events],
                "outputs": kept,
            }
        )

    def record_bridge(self, frame: Any) -> None:
        """Add a :class:`~embodied_ui.engine.BridgeFrame`: the reading sent
        and the engine outputs received in that period."""
        self.record(frame.reading, frame.state.outputs)

    def events(self) -> list[Event]:
        """Every recorded event, in order."""
        return [
            Event(e["time"], e["source"], e["kind"], e["value"], e.get("data", {}))
            for frame in self.frames
            for e in frame["events"]
        ]

    def columns(self) -> list[str]:
        """CSV columns: time, channels, then scalar outputs (vectors split)."""
        channels: dict[str, None] = {}
        outputs: dict[str, None] = {}
        for frame in self.frames:
            channels.update(dict.fromkeys(frame["channels"]))
            for key, value in frame["outputs"].items():
                if isinstance(value, list):
                    outputs.update(
                        dict.fromkeys(f"{key}[{i}]" for i in range(len(value)))
                    )
                else:
                    outputs[key] = None
        return ["time", *channels, *outputs]

    def write_jsonl(self, path: str | Path) -> Path:
        """One JSON object per frame; NaN becomes null."""
        path = _prepare(path)
        with path.open("w", encoding="utf-8") as handle:
            for frame in self.frames:
                handle.write(json.dumps(_clean(frame)) + "\n")
        return path

    def write_csv(self, path: str | Path) -> Path:
        """One row per frame with channels and scalar outputs; events are in
        the JSON Lines file or :meth:`events`."""
        path = _prepare(path)
        columns = self.columns()
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for frame in self.frames:
                row: dict[str, Any] = {"time": frame["time"], **frame["channels"]}
                for key, value in frame["outputs"].items():
                    if isinstance(value, list):
                        row.update({f"{key}[{i}]": v for i, v in enumerate(value)})
                    else:
                        row[key] = value
                writer.writerow([row.get(c, "") for c in columns])
        return path


def _numeric(value: Any) -> float | list[float] | None:
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return float(value)
    array = np.asarray(value) if isinstance(value, (list, tuple, np.ndarray)) else None
    if (
        array is not None
        and array.ndim == 1
        and 0 < array.size <= _MAX_VECTOR
        and np.issubdtype(array.dtype, np.number)
    ):
        return [float(v) for v in array]
    return None


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _prepare(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

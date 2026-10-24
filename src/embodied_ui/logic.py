"""Discrete behaviour of controls: thresholds, debouncing, toggles and touches.

Every state machine is driven by simulation time, not by sample counts, so
behaviour does not change with the control period.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .mechanics import Detents

# Samples are compared against hold times after floating-point accumulation
# of MuJoCo time; this tolerance accepts a hold that ends on a sample.
_TIME_EPS = 1e-9


@dataclass(frozen=True)
class Event:
    """A discrete change reported by a control.

    Args:
        time: MuJoCo time of the sample that produced the event, s.
        source: Device name, or ``device.region`` for touch regions.
        kind: ``press``, ``release``, ``toggle``, ``detent``,
            ``touch_begin``, ``touch_move`` or ``touch_end``.
        value: New state, detent index or touch identifier.
        data: Further values, e.g. ``u``, ``v``, ``force`` or ``delta``.
    """

    time: float
    source: str
    kind: str
    value: float = 0.0
    data: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable event."""
        return {
            "time": self.time,
            "source": self.source,
            "kind": self.kind,
            "value": self.value,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Event:
        """Rebuild an event received over the wire."""
        return cls(
            float(value.get("time", 0.0)),
            str(value["source"]),
            str(value["kind"]),
            float(value.get("value", 0.0)),
            {str(k): float(v) for k, v in dict(value.get("data", {})).items()},
        )


@dataclass(frozen=True)
class Interlock:
    """Accept a rising edge only while another channel is at least ``above``.

    Shift-by-wire reverse, for example, requires the brake pedal.
    """

    channel: str
    above: float = 0.5

    def allows(self, channels: Mapping[str, float]) -> bool:
        """Return whether the interlock condition holds."""
        return float(channels.get(self.channel, -math.inf)) >= self.above


class Threshold:
    """Two-level hysteresis with an optional hold time.

    The state turns on once the value has stayed at or above ``on`` for
    ``hold`` seconds, and off once it has stayed at or below ``off`` for the
    same time. An interlock gates only the on-transition.
    """

    def __init__(self, on: float, off: float, hold: float = 0.0) -> None:
        if not off < on:
            raise ValueError("the release threshold must be below the press threshold")
        if not math.isfinite(hold) or hold < 0:
            raise ValueError("hold must be finite and non-negative")
        self.on, self.off, self.hold = float(on), float(off), float(hold)
        self.reset()

    def reset(self) -> None:
        """Release the state and forget pending transitions."""
        self.state = False
        self._since: float | None = None

    def update(self, time: float, value: float, allowed: bool = True) -> bool:
        """Advance to one sample and return the debounced state."""
        crossing = (not self.state and value >= self.on and allowed) or (
            self.state and value <= self.off
        )
        if not crossing:
            self._since = None
            return self.state
        if self._since is None:
            self._since = time
        if time - self._since + _TIME_EPS >= self.hold:
            self.state = not self.state
            self._since = None
        return self.state


class DetentCounter:
    """Track the detent a control rests in, with hysteresis between cells.

    The index changes only once the position is ``margin`` (fraction of the
    half-cell) past the boundary, so noise at a boundary does not chatter.
    """

    def __init__(self, detents: Detents, margin: float = 0.2) -> None:
        if not 0 <= margin < 1:
            raise ValueError("margin must be in [0, 1)")
        self.detents = detents
        self.margin = float(margin)
        self.index: int | None = None

    def reset(self) -> None:
        """Forget the current detent."""
        self.index = None

    def update(self, q: float) -> tuple[int, int]:
        """Return (index, change) for joint position ``q``."""
        nearest = self.detents.index(q)
        if self.index is None:
            self.index = nearest
            return nearest, 0
        if nearest == self.index:
            return self.index, 0
        current = self.detents.position(self.index)
        target = self.detents.position(nearest)
        boundary = 0.5 * (current + target)
        half = abs(target - current) / 2
        if abs(q - boundary) >= self.margin * half:
            change = nearest - self.index
            self.index = nearest
            return nearest, change
        return self.index, 0


@dataclass(frozen=True)
class Touch:
    """One contact on a touch surface in surface coordinates.

    ``u`` runs left to right and ``v`` top to bottom, both in [0, 1].
    """

    u: float
    v: float
    force: float


class TouchTracker:
    """Turn per-body contacts into touch sequences with stable identifiers.

    A touch begins once its force reaches ``threshold`` and ends below half
    of it, like the hysteresis of capacitive controllers.
    """

    def __init__(self, source: str, threshold: float) -> None:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("touch threshold must be finite and positive")
        self.source = source
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        """End every touch without events."""
        self.active: dict[int, tuple[int, Touch]] = {}
        self._next = 1

    def update(self, time: float, contacts: Mapping[int, Touch]) -> list[Event]:
        """Advance with contacts keyed by the touching body; return events."""
        events: list[Event] = []
        for key in sorted(set(self.active) - set(contacts)):
            events.append(self._end(time, key))
        for key in sorted(contacts):
            touch = contacts[key]
            if key in self.active:
                if touch.force < 0.5 * self.threshold:
                    events.append(self._end(time, key))
                    continue
                identifier, previous = self.active[key]
                self.active[key] = (identifier, touch)
                if (touch.u, touch.v) != (previous.u, previous.v):
                    events.append(self._event(time, "touch_move", identifier, touch))
            elif touch.force >= self.threshold:
                identifier = self._next
                self._next += 1
                self.active[key] = (identifier, touch)
                events.append(self._event(time, "touch_begin", identifier, touch))
        return events

    def touches(self) -> list[tuple[int, Touch]]:
        """Return active touches ordered by identifier."""
        return sorted(self.active.values(), key=lambda item: item[0])

    def _end(self, time: float, key: int) -> Event:
        identifier, touch = self.active.pop(key)
        return self._event(time, "touch_end", identifier, touch)

    def _event(self, time: float, kind: str, identifier: int, touch: Touch) -> Event:
        return Event(
            time,
            self.source,
            kind,
            float(identifier),
            {"u": touch.u, "v": touch.v, "force": touch.force},
        )

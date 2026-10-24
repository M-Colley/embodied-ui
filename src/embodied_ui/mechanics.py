"""Degrees of freedom of a control and how they feel.

Linear springs, viscous damping, Coulomb friction, reflected inertia and end
stops use MuJoCo's own joint terms, which its integrator treats exactly.
Position-dependent feel that MuJoCo has no joint term for (detents, tactile
force curves) and engine-driven forces (force feedback, motorized controls)
are generalized forces the runtime writes to ``qfrc_applied`` on the
control's own degrees of freedom before each physics step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .frames import Vec3, vec3


def _real(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative(value: float, name: str) -> float:
    result = _real(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class Detents:
    """Stable positions with a sinusoidal restoring force between them.

    Inside the cell around a stable position ``p`` with half-width ``w`` (to
    the midpoint of the neighbouring position) the generalized force is
    ``-amplitude * sin(pi * (q - p) / w)``: zero at ``p`` and at the cell
    boundary, an unstable equilibrium, and largest halfway. Periodic detents
    (``period``) model rotary encoders; explicit ``positions`` model gates,
    selectors and bistable toggles.

    Args:
        amplitude: Peak restoring force (N) or torque (N m).
        positions: Explicit stable positions in joint units.
        period: Spacing of periodic positions; exclusive with ``positions``.
        offset: One stable position of a periodic pattern.
    """

    amplitude: float
    positions: tuple[float, ...] = ()
    period: float | None = None
    offset: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "amplitude", _non_negative(self.amplitude, "amplitude")
        )
        positions = tuple(sorted(_real(p, "positions") for p in self.positions))
        if len(set(positions)) != len(positions):
            raise ValueError("detent positions must be distinct")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "offset", _real(self.offset, "offset"))
        if self.period is not None:
            period = _real(self.period, "period")
            if period <= 0:
                raise ValueError("period must be positive")
            object.__setattr__(self, "period", period)
        if (self.period is None) == (len(positions) == 0):
            raise ValueError("give either detent positions or a period")
        if self.period is None and len(positions) < 2:
            raise ValueError("explicit detents need at least two positions")

    def index(self, q: float) -> int:
        """Return the index of the nearest stable position."""
        if self.period is not None:
            return int(math.floor((q - self.offset) / self.period + 0.5))
        return int(np.argmin([abs(q - p) for p in self.positions]))

    def position(self, index: int) -> float:
        """Return the stable position with this index."""
        if self.period is not None:
            return self.offset + index * self.period
        return self.positions[index]

    def _half_width(self, index: int, q: float) -> float:
        if self.period is not None:
            return self.period / 2
        p = self.positions[index]
        neighbour = index + (1 if q >= p else -1)
        if 0 <= neighbour < len(self.positions):
            return abs(self.positions[neighbour] - p) / 2
        # Beyond the outermost position, mirror the inner cell.
        inner = index - 1 if neighbour >= len(self.positions) else index + 1
        return abs(self.positions[inner] - p) / 2

    def force(self, q: float) -> float:
        """Return the generalized detent force at joint position ``q``."""
        index = self.index(q)
        width = self._half_width(index, q)
        x = max(-width, min(width, q - self.position(index)))
        return -self.amplitude * math.sin(math.pi * x / width)


@dataclass(frozen=True)
class ForceCurve:
    """Piecewise-linear force-displacement curve, such as a tactile key.

    The generalized force is ``-interp(q)`` so positive values oppose motion
    towards larger ``q``; values outside the table hold the end values.

    Args:
        points: ``(q, force)`` pairs with strictly increasing ``q``.
    """

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        points = tuple(
            (_real(q, "curve position"), _real(f, "curve force"))
            for q, f in self.points
        )
        if len(points) < 2 or any(b[0] <= a[0] for a, b in zip(points, points[1:])):
            raise ValueError("a force curve needs at least two increasing positions")
        object.__setattr__(self, "points", points)

    def force(self, q: float) -> float:
        """Return the generalized force at ``q``."""
        xs, fs = zip(*self.points)
        return -float(np.interp(q, xs, fs))


@dataclass(frozen=True)
class Servo:
    """A motor that drives the axis to an engine-given setpoint.

    Models motorized faders, active steering wheels and shift-by-wire knobs
    that return to park. The force is ``kp (setpoint - q) - kd qdot``,
    limited to ``max_force``; without a setpoint it is zero.
    """

    kp: float
    kd: float = 0.0
    max_force: float = math.inf

    def __post_init__(self) -> None:
        object.__setattr__(self, "kp", _non_negative(self.kp, "kp"))
        object.__setattr__(self, "kd", _non_negative(self.kd, "kd"))
        limit = float(self.max_force)
        if math.isnan(limit) or limit <= 0:
            raise ValueError("max_force must be positive")
        object.__setattr__(self, "max_force", limit)

    def force(self, setpoint: float, q: float, qdot: float) -> float:
        """Return the servo force for one physics step."""
        value = self.kp * (setpoint - q) - self.kd * qdot
        return max(-self.max_force, min(self.max_force, value))


@dataclass(frozen=True)
class Axis:
    """One degree of freedom of a control.

    Args:
        name: Channel name of the raw position, e.g. ``travel`` or ``angle``.
        kind: ``slide`` (metres) or ``hinge`` (radians).
        direction: Joint axis in the parent frame; positive motion follows it.
        range: Joint limits; ``None`` leaves a hinge unlimited (encoders).
        stiffness: Linear spring, N/m or N m/rad.
        rest: Spring reference. A rest beyond the released end stop preloads
            the control, as in keys and pedals.
        damping: Viscous damping, N s/m or N m s/rad.
        friction: Coulomb friction, N or N m.
        armature: Reflected inertia, kg or kg m^2; stabilizes light parts.
        detents: Stable positions.
        curve: Additional force-displacement curve.
        servo: Engine-driven motor.
        stop_time: Time constant of the end-stop contact, s.
    """

    name: str
    kind: Literal["slide", "hinge"]
    direction: Vec3 = (0.0, 0.0, 1.0)
    range: tuple[float, float] | None = None
    stiffness: float = 0.0
    rest: float = 0.0
    damping: float = 0.0
    friction: float = 0.0
    armature: float = 0.0
    detents: Detents | None = None
    curve: ForceCurve | None = None
    servo: Servo | None = None
    stop_time: float = 0.004

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("axis names must be identifiers")
        if self.kind not in ("slide", "hinge"):
            raise ValueError("kind must be 'slide' or 'hinge'")
        direction = np.asarray(vec3(self.direction, "direction"))
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            raise ValueError("direction must be nonzero")
        object.__setattr__(self, "direction", tuple(float(v) for v in direction / norm))
        if self.range is not None:
            low, high = (_real(v, "range") for v in self.range)
            if not low < high:
                raise ValueError("range must be increasing")
            object.__setattr__(self, "range", (low, high))
        elif self.kind == "slide":
            raise ValueError("slide axes need a range")
        for name in ("stiffness", "damping", "friction", "armature"):
            object.__setattr__(self, name, _non_negative(getattr(self, name), name))
        object.__setattr__(self, "rest", _real(self.rest, "rest"))
        stop = _real(self.stop_time, "stop_time")
        if stop <= 0:
            raise ValueError("stop_time must be positive")
        object.__setattr__(self, "stop_time", stop)

    @property
    def needs_force(self) -> bool:
        """Whether the runtime must apply forces to this axis."""
        return (
            self.detents is not None or self.curve is not None or self.servo is not None
        )

    def normalized(self, q: float) -> float:
        """Map a position to [0, 1] across the range (unlimited: raw).

        End stops are compliant, so a preloaded control rests slightly past
        its stop; the result is clipped to [0, 1].
        """
        if self.range is None:
            return q
        low, high = self.range
        return min(1.0, max(0.0, (q - low) / (high - low)))


def symmetric(limit: float) -> tuple[float, float]:
    """Return ``(-limit, limit)``."""
    limit = _real(limit, "limit")
    if limit <= 0:
        raise ValueError("limit must be positive")
    return (-limit, limit)


def evenly_spaced(low: float, high: float, count: int) -> tuple[float, ...]:
    """Return ``count`` evenly spaced positions from ``low`` to ``high``."""
    if count < 2:
        raise ValueError("count must be at least two")
    return tuple(float(v) for v in np.linspace(low, high, count))


def as_floats(values: Sequence[float], name: str) -> tuple[float, ...]:
    """Validate a sequence of finite numbers."""
    return tuple(_real(v, name) for v in values)

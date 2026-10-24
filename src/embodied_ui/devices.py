"""Physical input and output devices.

Every device has its own frame: x to the user's right, y up and z out of the
panel towards the user. Its ``pose`` places that frame in the ``mount`` body
of the host model. Inputs are dynamic mechanisms (joints, springs, end stops
and contacts) operated through contact by the simulated body; only
engine-driven outputs such as :class:`Prop` are kinematic.

A device declares its mechanism, its continuous channels and an optional
state machine. Subclass :class:`Device`, or compose a :class:`Control`, to
add new kinds; :func:`register` makes them available to manifests.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from .frames import Pose
from .geometry import Box, Capsule, Cylinder, Part, Rgba, Sphere
from .logic import DetentCounter, Event, Interlock, Threshold, Touch, TouchTracker
from .mechanics import Axis, Detents, ForceCurve, Servo, symmetric

DEVICE_TYPES: dict[str, type[Device]] = {}
_D = TypeVar("_D", bound="type[Device]")

_DARK: Rgba = (0.12, 0.12, 0.13, 1.0)
_GREY: Rgba = (0.35, 0.36, 0.38, 1.0)
_RED: Rgba = (0.8, 0.15, 0.12, 1.0)


def register(cls: _D) -> _D:
    """Class decorator: make a device type available to manifests by name."""
    DEVICE_TYPES[cls.__name__] = cls
    return cls


@dataclass(frozen=True)
class Raw:
    """Measured state of one device at one sample.

    Args:
        q: Joint position per axis name.
        qdot: Joint velocity per axis name.
        force: Normal contact force on the moving parts, N (channel ``contact``).
        contacts: Touches keyed by touching body, for touch surfaces.
    """

    q: Mapping[str, float]
    qdot: Mapping[str, float]
    force: float = 0.0
    contacts: Mapping[int, Touch] = field(default_factory=dict)


class Logic(Protocol):
    """Stateful discrete behaviour of one device."""

    def reset(self) -> None:
        """Return to the released state."""
        ...

    def update(
        self,
        time: float,
        own: Mapping[str, float],
        channels: Mapping[str, float],
        raw: Raw,
    ) -> tuple[dict[str, float], list[Event]]:
        """Return additional channels and events for one sample.

        ``own`` holds this device's continuous channels without the device
        prefix; ``channels`` holds every channel computed so far, prefixed.
        """
        ...


@dataclass(frozen=True)
class TouchSpec:
    """Touch sensing on a static surface in the device's xy plane.

    Args:
        size: Width and height of the active area, m.
        threshold: Normal force that begins a touch, N.
        regions: Named soft buttons as ``(u0, v0, u1, v1)`` in [0, 1].
    """

    size: tuple[float, float]
    threshold: float = 0.05
    regions: Mapping[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class DisplaySpec:
    """A texture on the device's front face that engines can update."""

    resolution: tuple[int, int]
    size: tuple[float, float]


@dataclass(frozen=True)
class Mechanism:
    """What a device adds to a MuJoCo model.

    Args:
        axes: Degrees of freedom, nested in order (gimbals, push-and-turn).
        moving: Parts carried by the last axis.
        static: Parts fixed to the device frame.
        touch: Add a touch sensor around the moving parts.
        surface: Touch sensing on the first static part.
        display: Texture on the first static part.
        mocap: Engine-driven pose; the device must be mounted on the world.
        indicator: Off and on colours of all static parts.
    """

    axes: tuple[Axis, ...] = ()
    moving: tuple[Part, ...] = ()
    static: tuple[Part, ...] = ()
    touch: bool = False
    surface: TouchSpec | None = None
    display: DisplaySpec | None = None
    mocap: bool = False
    indicator: tuple[Rgba, Rgba] | None = None

    def __post_init__(self) -> None:
        if self.axes and not self.moving:
            raise ValueError("a device with axes needs moving parts")
        if not self.axes and self.moving and not self.mocap:
            raise ValueError("moving parts need an axis")
        names = [axis.name for axis in self.axes]
        if len(set(names)) != len(names):
            raise ValueError("axis names must be unique within a device")
        if (self.surface or self.display or self.indicator) and not self.static:
            raise ValueError("surfaces, displays and indicators need a static part")


@dataclass(frozen=True)
class Device:
    """Base class of all devices.

    Args:
        name: Unique identifier; channels are ``name.channel``.
        pose: Device frame in the mount body.
        mount: Host body carrying the device, or ``world``.
    """

    name: str
    pose: Pose = field(default_factory=Pose)
    mount: str = "world"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or any(c in self.name for c in ".: \t\n")
        ):
            raise ValueError("device names must be nonempty without '.', ':' or spaces")
        if not isinstance(self.pose, Pose):
            raise TypeError("pose must be a Pose")

    def mechanism(self) -> Mechanism:
        """Return the mechanism to compile."""
        raise NotImplementedError

    def continuous(self, raw: Raw) -> dict[str, float]:
        """Return continuous channels: each axis position and its velocity."""
        values: dict[str, float] = {}
        for axis in self.mechanism().axes:
            values[axis.name] = raw.q[axis.name]
            values[f"{axis.name}_rate"] = raw.qdot[axis.name]
        if self.mechanism().touch:
            values["contact"] = raw.force
        return values

    def logic(self) -> Logic | None:
        """Return a fresh state machine, or ``None`` for continuous devices."""
        return None

    def outputs(self) -> tuple[str, ...]:
        """Return the engine outputs this device consumes, without prefix."""
        outputs: list[str] = []
        mechanism = self.mechanism()
        single = len(mechanism.axes) == 1
        for axis in mechanism.axes:
            outputs.append(f"{axis.name}.force")
            if single:
                outputs.append("force")
            if axis.servo is not None:
                outputs.append(f"{axis.name}.setpoint")
                if single:
                    outputs.append("setpoint")
        if mechanism.indicator is not None:
            outputs.append("on")
        if mechanism.display is not None:
            outputs.append("image")
        if mechanism.mocap:
            outputs.append("pose")
        return tuple(outputs)


def _primary(device: Device) -> Axis:
    return device.mechanism().axes[0]


class _ButtonLogic:
    """Momentary or toggle push-button behaviour on a normalized value."""

    def __init__(
        self,
        name: str,
        make: float,
        release: float,
        hold: float,
        toggle: bool,
        interlock: Interlock | None,
    ) -> None:
        self.name = name
        self.threshold = Threshold(make, release, hold)
        self.toggle = toggle
        self.interlock = interlock
        self.reset()

    def reset(self) -> None:
        self.threshold.reset()
        self.state = False

    def update(
        self,
        time: float,
        own: Mapping[str, float],
        channels: Mapping[str, float],
        raw: Raw,
    ) -> tuple[dict[str, float], list[Event]]:
        allowed = self.interlock is None or self.interlock.allows(channels)
        before = self.threshold.state
        pressed = self.threshold.update(time, own["value"], allowed)
        events: list[Event] = []
        if pressed and not before:
            events.append(Event(time, self.name, "press", 1.0))
            if self.toggle:
                self.state = not self.state
                events.append(Event(time, self.name, "toggle", float(self.state)))
        elif before and not pressed:
            events.append(Event(time, self.name, "release", 0.0))
        state = self.state if self.toggle else pressed
        return {"pressed": float(pressed), "state": float(state)}, events


@register
@dataclass(frozen=True)
class PushButton(Device):
    """Spring-loaded push button pressed along the device's -z.

    The return spring is preloaded: ``preload`` acts at rest and
    ``force`` at full travel. ``make`` and ``release`` are the switching
    points as fractions of travel; ``hold`` debounces both edges.

    Defaults model a fingertip panel button within MIL-STD-1472G/H
    (Figure 14: 10-25 mm diameter, 2-6 mm travel, 2.8-11 N resistance): a
    15 mm cap, 4 mm travel and 4.5 N at the make point. :meth:`key` gives a
    keyboard key within ISO 9241-410.

    Args:
        travel: Stroke, m.
        preload: Spring force at rest, N.
        force: Spring force at full travel, N.
        cap: Moving geometry; defaults to a 15 mm square cap.
        housing: Static geometry around the cap.
        make: Press threshold, fraction of travel.
        release: Release threshold, fraction of travel.
        hold: Seconds a threshold crossing must persist.
        toggle: Latch: each press flips ``state``.
        interlock: Condition for accepting presses.
        curve: Optional tactile force curve replacing the linear spring.
        mass: Moving mass, kg.
    """

    travel: float = 0.004
    preload: float = 3.0
    force: float = 6.0
    cap: Part | None = None
    housing: Part | None = None
    make: float = 0.5
    release: float = 0.35
    hold: float = 0.0
    toggle: bool = False
    interlock: Interlock | None = None
    curve: ForceCurve | None = None
    mass: float = 0.005

    @classmethod
    def key(cls, name: str, **overrides: Any) -> PushButton:
        """A keyboard key within ISO 9241-410:2008 Annex B.

        3 mm travel (preferred 2-4 mm), 0.65 N at the make point (preferred
        0.5-0.8 N) and a preload of half that force (permitted 25-75%).
        """
        settings: dict[str, Any] = {"travel": 0.003, "preload": 0.325, "force": 0.975}
        settings.update(overrides)
        return cls(name, **settings)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (math.isfinite(self.travel) and self.travel > 0):
            raise ValueError("travel must be finite and positive")
        if not 0 <= self.preload <= self.force:
            raise ValueError("need 0 <= preload <= force")
        if not 0 < self.release < self.make <= 1:
            raise ValueError("need 0 < release < make <= 1")

    def mechanism(self) -> Mechanism:
        stiffness = (self.force - self.preload) / self.travel
        rest = -self.preload / stiffness if stiffness > 0 else 0.0
        if self.curve is not None:
            stiffness, rest = 0.0, 0.0
        cap = self.cap or Part(
            Box((0.0075, 0.0075, 0.003)), Pose((0, 0, 0.003)), _GREY, mass=self.mass
        )
        if cap.mass == 0:
            cap = Part(
                cap.shape, cap.pose, cap.rgba, self.mass, cap.collide, cap.friction
            )
        armature = 0.01
        # Damping ratio 0.5 about the spring (or the curve's mean slope).
        slope = stiffness
        if self.curve is not None:
            (q0, f0), (q1, f1) = self.curve.points[0], self.curve.points[-1]
            slope = max((f1 - f0) / (q1 - q0), 1.0)
        axis = Axis(
            "travel",
            "slide",
            (0.0, 0.0, -1.0),
            (0.0, self.travel),
            stiffness=stiffness,
            rest=rest,
            damping=math.sqrt(slope * (cap.mass + armature)),
            armature=armature,
            curve=self.curve,
        )
        static = (self.housing,) if self.housing else ()
        return Mechanism((axis,), (cap,), static, touch=True)

    def continuous(self, raw: Raw) -> dict[str, float]:
        values = super().continuous(raw)
        values["value"] = _primary(self).normalized(raw.q["travel"])
        return values

    def logic(self) -> Logic:
        return _ButtonLogic(
            self.name, self.make, self.release, self.hold, self.toggle, self.interlock
        )


class _DetentLogic:
    def __init__(self, name: str, axis: str, detents: Detents) -> None:
        self.name, self.axis = name, axis
        self.counter = DetentCounter(detents)

    def reset(self) -> None:
        self.counter.reset()

    def update(
        self,
        time: float,
        own: Mapping[str, float],
        channels: Mapping[str, float],
        raw: Raw,
    ) -> tuple[dict[str, float], list[Event]]:
        index, change = self.counter.update(raw.q[self.axis])
        events = (
            [Event(time, self.name, "detent", float(index), {"delta": float(change)})]
            if change
            else []
        )
        return {"detent": float(index)}, events


class _Chain:
    """Run several logic blocks and merge their channels and events."""

    def __init__(self, *blocks: Logic) -> None:
        self.blocks = blocks

    def reset(self) -> None:
        for block in self.blocks:
            block.reset()

    def update(
        self,
        time: float,
        own: Mapping[str, float],
        channels: Mapping[str, float],
        raw: Raw,
    ) -> tuple[dict[str, float], list[Event]]:
        merged: dict[str, float] = {}
        events: list[Event] = []
        for block in self.blocks:
            values, new = block.update(time, own, channels, raw)
            merged.update(values)
            events.extend(new)
        return merged, events


@register
@dataclass(frozen=True)
class ToggleSwitch(Device):
    """Bistable lever or rocker that snaps between two end positions.

    The lever pivots about the device's x axis; up is positive. ``state`` is
    1 in the upper position.

    Args:
        angle: Throw to each side of centre, rad.
        torque: Peak snap torque of the over-centre spring, N m.
        lever: Moving geometry; defaults to a 12 mm bat handle.
    """

    angle: float = 0.35
    torque: float = 0.03
    lever: Part | None = None

    def mechanism(self) -> Mechanism:
        lever = self.lever or Part(
            Capsule(0.003, 0.006), Pose((0, 0, 0.009)), _GREY, mass=0.003
        )
        axis = Axis(
            "angle",
            "hinge",
            (-1.0, 0.0, 0.0),
            symmetric(self.angle),
            damping=2e-4,
            armature=2e-5,
            # Wells beyond the stops preload the lever against them, so each
            # end position holds with about 60% of the peak snap torque.
            detents=Detents(
                self.torque, positions=(-1.25 * self.angle, 1.25 * self.angle)
            ),
        )
        return Mechanism((axis,), (lever,), touch=True)

    def logic(self) -> Logic:
        return _SignLogic(self.name, "angle", 0.25 * self.angle)


class _SignLogic:
    def __init__(self, name: str, axis: str, band: float) -> None:
        self.name, self.axis, self.band = name, axis, band
        self.reset()

    def reset(self) -> None:
        self.state: bool | None = None

    def update(
        self,
        time: float,
        own: Mapping[str, float],
        channels: Mapping[str, float],
        raw: Raw,
    ) -> tuple[dict[str, float], list[Event]]:
        q = raw.q[self.axis]
        events: list[Event] = []
        if self.state is None:
            self.state = q > 0
        elif not self.state and q > self.band:
            self.state = True
            events.append(Event(time, self.name, "toggle", 1.0))
        elif self.state and q < -self.band:
            self.state = False
            events.append(Event(time, self.name, "toggle", 0.0))
        return {"state": float(self.state)}, events


@register
@dataclass(frozen=True)
class Knob(Device):
    """Rotary knob or encoder turned about the device's z axis.

    Clockwise, as seen by the user, is positive. ``value`` counts turns for
    endless encoders and is normalized over ``range`` otherwise. The default
    peak torque (12 mN m detents plus 2 mN m friction) stays below the
    32 mN m that MIL-STD-1472G (Figure 10) allows for continuous knobs.

    Args:
        radius: Knob radius, m.
        height: Knob height, m.
        range: Angle limits, rad; ``None`` for an endless encoder.
        detents: Detents per revolution; 0 for smooth.
        detent_torque: Peak detent torque, N m.
        friction: Coulomb friction, N m.
        push: Add a push-to-select stroke along -z.
    """

    radius: float = 0.015
    height: float = 0.018
    range: tuple[float, float] | None = None
    detents: int = 24
    detent_torque: float = 0.012
    friction: float = 0.002
    push: bool = False

    def mechanism(self) -> Mechanism:
        turn = Axis(
            "angle",
            "hinge",
            (0.0, 0.0, -1.0),
            self.range,
            damping=1e-4,
            friction=self.friction,
            armature=2e-5,
            detents=(
                Detents(self.detent_torque, period=2 * math.pi / self.detents)
                if self.detents
                else None
            ),
        )
        cap = Part(
            Cylinder(self.radius, self.height / 2),
            Pose((0, 0, self.height / 2)),
            _DARK,
            mass=0.01,
        )
        axes: tuple[Axis, ...] = (turn,)
        if self.push:
            axes = (
                Axis(
                    "travel",
                    "slide",
                    (0.0, 0.0, -1.0),
                    (0.0, 0.002),
                    stiffness=1500.0,
                    rest=-0.002,
                    damping=0.2,
                    armature=0.01,
                ),
                turn,
            )
        return Mechanism(axes, (cap,), touch=True)

    def continuous(self, raw: Raw) -> dict[str, float]:
        values = super().continuous(raw)
        angle = raw.q["angle"]
        if self.range is None:
            values["value"] = angle / (2 * math.pi)
        else:
            low, high = self.range
            values["value"] = (angle - low) / (high - low)
        if self.push:
            values["push"] = min(1.0, max(0.0, raw.q["travel"] / 0.002))
        return values

    def logic(self) -> Logic | None:
        blocks: list[Logic] = []
        detents = self.mechanism().axes[-1].detents
        if detents is not None:
            blocks.append(_DetentLogic(self.name, "angle", detents))
        if self.push:
            blocks.append(_PushOnly(self.name))
        return _Chain(*blocks) if blocks else None


class _PushOnly(_ButtonLogic):
    def __init__(self, name: str) -> None:
        super().__init__(name, 0.5, 0.35, 0.0, False, None)

    def update(
        self,
        time: float,
        own: Mapping[str, float],
        channels: Mapping[str, float],
        raw: Raw,
    ) -> tuple[dict[str, float], list[Event]]:
        values, events = super().update(time, {"value": own["push"]}, channels, raw)
        return {"pressed": values["pressed"]}, events


@register
@dataclass(frozen=True)
class Slider(Device):
    """Linear fader moving along the device's x axis from 0 to ``travel``.

    Args:
        travel: Stroke, m.
        positions: Number of detented positions, or 0 for smooth.
        detent_force: Peak detent force, N.
        friction: Coulomb friction, N.
        servo: Motor for engine-set positions (motorized faders).
    """

    travel: float = 0.06
    positions: int = 0
    detent_force: float = 0.5
    friction: float = 0.3
    servo: Servo | None = None

    def mechanism(self) -> Mechanism:
        detents = (
            Detents(
                self.detent_force,
                positions=tuple(
                    self.travel * i / (self.positions - 1)
                    for i in range(self.positions)
                ),
            )
            if self.positions >= 2
            else None
        )
        axis = Axis(
            "position",
            "slide",
            (1.0, 0.0, 0.0),
            (0.0, self.travel),
            damping=0.5,
            friction=self.friction,
            armature=0.02,
            detents=detents,
            servo=self.servo,
        )
        knob = Part(Box((0.006, 0.01, 0.006)), Pose((0, 0, 0.006)), _DARK, mass=0.01)
        rail = Part(
            Box((self.travel / 2 + 0.008, 0.002, 0.001)),
            Pose((self.travel / 2, 0, -0.001)),
            _GREY,
            collide=False,
        )
        return Mechanism((axis,), (knob,), (rail,), touch=True)

    def continuous(self, raw: Raw) -> dict[str, float]:
        values = super().continuous(raw)
        values["value"] = _primary(self).normalized(raw.q["position"])
        return values

    def logic(self) -> Logic | None:
        detents = self.mechanism().axes[0].detents
        return _DetentLogic(self.name, "position", detents) if detents else None


@register
@dataclass(frozen=True)
class Lever(Device):
    """Hinged lever with gated positions: selectors, stalks, gear levers.

    The lever pivots about ``axis`` (device frame) at the origin and carries
    a knob at ``length`` along +z. ``position`` reports the gate index. The
    default 17 degree gates and 150 mN m detents lie within MIL-STD-1472G's
    detented rotary selectors (15-40 degrees, 115-680 mN m, Figure 6).

    Args:
        positions: Gate angles, rad; at least two.
        length: Pivot to knob centre, m.
        gate_torque: Peak detent torque between gates, N m.
        centre: Spring-return angle for momentary stalks, or ``None``.
        stiffness: Return-spring stiffness, N m/rad, when ``centre`` is set.
        axis: Pivot axis in the device frame.
    """

    positions: tuple[float, ...] = (-0.3, 0.0, 0.3)
    length: float = 0.08
    gate_torque: float = 0.15
    centre: float | None = None
    stiffness: float = 0.0
    axis: tuple[float, float, float] = (-1.0, 0.0, 0.0)

    def mechanism(self) -> Mechanism:
        low, high = min(self.positions), max(self.positions)
        margin = 0.05 * (high - low)
        axis = Axis(
            "angle",
            "hinge",
            self.axis,
            (low - margin, high + margin),
            stiffness=self.stiffness if self.centre is not None else 0.0,
            rest=self.centre or 0.0,
            damping=0.002,
            armature=1e-4,
            detents=Detents(self.gate_torque, positions=self.positions),
        )
        shaft = Part(
            Capsule(0.004, self.length / 2),
            Pose((0, 0, self.length / 2)),
            _GREY,
            mass=0.02,
        )
        knob = Part(Sphere(0.012), Pose((0, 0, self.length)), _DARK, mass=0.03)
        return Mechanism((axis,), (shaft, knob), touch=True)

    def continuous(self, raw: Raw) -> dict[str, float]:
        values = super().continuous(raw)
        low, high = min(self.positions), max(self.positions)
        values["value"] = min(1.0, max(0.0, (raw.q["angle"] - low) / (high - low)))
        return values

    def logic(self) -> Logic:
        detents = self.mechanism().axes[0].detents
        assert detents is not None
        return _RenamedDetent(_DetentLogic(self.name, "angle", detents), "position")


class _RenamedDetent:
    def __init__(self, inner: _DetentLogic, channel: str) -> None:
        self.inner, self.channel = inner, channel

    def reset(self) -> None:
        self.inner.reset()

    def update(
        self,
        time: float,
        own: Mapping[str, float],
        channels: Mapping[str, float],
        raw: Raw,
    ) -> tuple[dict[str, float], list[Event]]:
        values, events = self.inner.update(time, own, channels, raw)
        return {self.channel: values["detent"]}, events


@register
@dataclass(frozen=True)
class Pedal(Device):
    """Spring-loaded pedal pressed along the device's -z.

    Args:
        travel: Stroke, m.
        preload: Spring force at rest, N.
        force: Spring force at full travel, N.
        pad: Half extents of the pad, m.
    """

    travel: float = 0.06
    preload: float = 10.0
    force: float = 40.0
    pad: tuple[float, float, float] = (0.04, 0.07, 0.008)

    def mechanism(self) -> Mechanism:
        stiffness = (self.force - self.preload) / self.travel
        axis = Axis(
            "travel",
            "slide",
            (0.0, 0.0, -1.0),
            (0.0, self.travel),
            stiffness=stiffness,
            rest=-self.preload / stiffness,
            damping=10.0,
            armature=0.05,
        )
        pad = Part(Box(self.pad), Pose((0, 0, self.pad[2])), _DARK, mass=0.5)
        return Mechanism((axis,), (pad,), touch=True)

    def continuous(self, raw: Raw) -> dict[str, float]:
        values = super().continuous(raw)
        values["value"] = _primary(self).normalized(raw.q["travel"])
        return values


@register
@dataclass(frozen=True)
class SteeringWheel(Device):
    """Steering wheel turning about the device's z axis (towards the driver).

    Counter-clockwise, as seen by the driver, is positive, as in ISO 8855's
    left-positive steering angle; ``value`` is the angle over
    ``angle_limit``. Engines may apply ``torque`` (self-aligning moment) or,
    with ``servo``, a ``setpoint`` for automated steering.

    Args:
        radius: Rim centreline radius, m.
        grip: Rim tube radius, m.
        angle_limit: Lock-to-centre angle, rad.
        stiffness: Passive centring stiffness, N m/rad.
        damping: N m s/rad.
        friction: Coulomb friction of the column, N m.
        segments: Capsules approximating the rim.
        servo: Motor for engine-commanded steering.
    """

    radius: float = 0.18
    grip: float = 0.015
    angle_limit: float = 7.85
    stiffness: float = 0.2
    damping: float = 0.1
    friction: float = 0.1
    segments: int = 24
    servo: Servo | None = None

    def mechanism(self) -> Mechanism:
        axis = Axis(
            "angle",
            "hinge",
            (0.0, 0.0, 1.0),
            symmetric(self.angle_limit),
            stiffness=self.stiffness,
            damping=self.damping,
            friction=self.friction,
            armature=0.03,
            servo=self.servo,
        )
        chord = 2 * self.radius * math.sin(math.pi / self.segments)
        # Segment centres lie on the chords, inside the centreline circle.
        inner = self.radius * math.cos(math.pi / self.segments)
        rim = []
        for i in range(self.segments):
            angle = 2 * math.pi * (i + 0.5) / self.segments
            rim.append(
                Part(
                    Capsule(self.grip, chord / 2),
                    Pose(
                        (inner * math.cos(angle), inner * math.sin(angle), 0.0),
                        _tangent_quat(angle),
                    ),
                    _DARK,
                    mass=1.2 / self.segments,
                )
            )
        hub = Part(
            Cylinder(0.04, 0.02), Pose((0, 0, -0.02)), _GREY, mass=0.5, collide=False
        )
        return Mechanism((axis,), (*rim, hub), touch=False)

    def continuous(self, raw: Raw) -> dict[str, float]:
        values = super().continuous(raw)
        values["value"] = max(-1.0, min(1.0, raw.q["angle"] / self.angle_limit))
        return values

    def outputs(self) -> tuple[str, ...]:
        return (*super().outputs(), "torque")


def _tangent_quat(angle: float) -> tuple[float, float, float, float]:
    """Quaternion turning +z onto the rim tangent at ``angle`` in the xy plane."""
    # A quarter turn about z x t = (-cos a, -sin a, 0) maps z onto the
    # tangent t = (-sin a, cos a, 0).
    s = math.sqrt(0.5)
    return (s, -s * math.cos(angle), -s * math.sin(angle), 0.0)


@register
@dataclass(frozen=True)
class Joystick(Device):
    """Spring-centred two-axis stick on a gimbal at the device origin.

    ``x`` tilts the stick to the user's right, ``y`` away from the user.

    Args:
        limit: Tilt limit per axis, rad.
        stiffness: Centring stiffness, N m/rad.
        length: Gimbal to handle centre, m.
    """

    limit: float = 0.35
    stiffness: float = 0.3
    length: float = 0.06

    def mechanism(self) -> Mechanism:
        common: dict[str, Any] = {
            "range": symmetric(self.limit),
            "stiffness": self.stiffness,
            "damping": 0.005,
            "armature": 1e-4,
        }
        axes = (
            Axis("x", "hinge", (0.0, 1.0, 0.0), **common),
            Axis("y", "hinge", (-1.0, 0.0, 0.0), **common),
        )
        handle = Part(
            Capsule(0.012, self.length / 2),
            Pose((0, 0, self.length / 2)),
            _DARK,
            mass=0.05,
        )
        return Mechanism(axes, (handle,), touch=True)

    def continuous(self, raw: Raw) -> dict[str, float]:
        values = super().continuous(raw)
        values["x_value"] = raw.q["x"] / self.limit
        values["y_value"] = raw.q["y"] / self.limit
        return values


class _TouchLogic:
    def __init__(self, name: str, spec: TouchSpec) -> None:
        self.name = name
        self.tracker = TouchTracker(name, spec.threshold)
        self.regions = dict(spec.regions)
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self.inside: dict[str, bool] = {region: False for region in self.regions}

    def update(
        self,
        time: float,
        own: Mapping[str, float],
        channels: Mapping[str, float],
        raw: Raw,
    ) -> tuple[dict[str, float], list[Event]]:
        events = self.tracker.update(time, raw.contacts)
        touches = self.tracker.touches()
        values: dict[str, float] = {"touches": float(len(touches))}
        if touches:
            _, first = touches[0]
            values.update(u=first.u, v=first.v, force=first.force)
        else:
            values.update(u=math.nan, v=math.nan, force=0.0)
        for region, (u0, v0, u1, v1) in self.regions.items():
            pressed = any(u0 <= t.u <= u1 and v0 <= t.v <= v1 for _, t in touches)
            if pressed != self.inside[region]:
                kind = "press" if pressed else "release"
                events.append(
                    Event(time, f"{self.name}.{region}", kind, float(pressed))
                )
            self.inside[region] = pressed
            values[f"{region}.pressed"] = float(pressed)
        return values, events


@register
@dataclass(frozen=True)
class TouchSurface(Device):
    """Rigid touch panel in the device's xy plane, optionally a display.

    ``u`` runs to the user's right and ``v`` downwards over the active area.
    Each touching body is one touch; ``u``, ``v`` and ``force`` report the
    oldest active touch and events report all of them. For in-vehicle soft
    keys, Kim et al. (2014) found no gain beyond 17.5 mm keys while driving;
    ISO 9241-410 Annex J asks for at least 5 mm between targets.

    Args:
        size: Width and height of the active area, m.
        thickness: Panel thickness behind the surface, m.
        threshold: Normal force that begins a touch, N.
        regions: Named soft buttons as ``(u0, v0, u1, v1)``.
        resolution: Pixels (width, height) of a display texture, if any.
    """

    size: tuple[float, float] = (0.2, 0.12)
    thickness: float = 0.006
    threshold: float = 0.05
    regions: Mapping[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    resolution: tuple[int, int] | None = None

    def mechanism(self) -> Mechanism:
        width, height = self.size
        panel = Part(
            Box((width / 2, height / 2, self.thickness / 2)),
            Pose((0, 0, -self.thickness / 2)),
            _DARK,
        )
        display = (
            DisplaySpec(tuple(self.resolution), (width, height))  # type: ignore[arg-type]
            if self.resolution
            else None
        )
        return Mechanism(
            static=(panel,),
            surface=TouchSpec((width, height), self.threshold, dict(self.regions)),
            display=display,
        )

    def continuous(self, raw: Raw) -> dict[str, float]:
        return {}

    def logic(self) -> Logic:
        surface = self.mechanism().surface
        assert surface is not None
        return _TouchLogic(self.name, surface)


@register
@dataclass(frozen=True)
class Screen(Device):
    """Display without touch sensing, e.g. an instrument cluster or HUD.

    Args:
        size: Width and height, m.
        resolution: Texture pixels (width, height).
        collide: Whether the bezel takes part in contacts.
    """

    size: tuple[float, float] = (0.3, 0.12)
    resolution: tuple[int, int] = (320, 128)
    collide: bool = False

    def mechanism(self) -> Mechanism:
        width, height = self.size
        panel = Part(
            Box((width / 2, height / 2, 0.003)),
            Pose((0, 0, -0.003)),
            _DARK,
            collide=self.collide,
        )
        return Mechanism(
            static=(panel,),
            display=DisplaySpec(tuple(self.resolution), (width, height)),  # type: ignore[arg-type]
        )

    def continuous(self, raw: Raw) -> dict[str, float]:
        return {}


@register
@dataclass(frozen=True)
class Indicator(Device):
    """Lamp whose colour the engine switches with the ``on`` output."""

    shape: Part | None = None
    off: Rgba = (0.15, 0.05, 0.02, 1.0)
    on: Rgba = (1.0, 0.55, 0.0, 1.0)

    def mechanism(self) -> Mechanism:
        part = self.shape or Part(
            Cylinder(0.005, 0.001), Pose((0, 0, 0.001)), self.off, collide=False
        )
        return Mechanism(static=(part,), indicator=(self.off, self.on))

    def continuous(self, raw: Raw) -> dict[str, float]:
        return {}


@register
@dataclass(frozen=True)
class Speaker(Device):
    """Sound source the engine drives with ``level`` and ``frequency``.

    ``level`` is the sound pressure level in dB SPL at ``reference_distance``
    from the speaker and ``frequency`` the dominant tone in Hz. Listeners
    hear the level spread in a free field
    (:meth:`InterfaceRuntime.sound_level`); room acoustics, masking and
    hearing thresholds belong to the listener's model, not the device.
    """

    radius: float = 0.03
    reference_distance: float = 1.0

    def mechanism(self) -> Mechanism:
        grille = Part(
            Cylinder(self.radius, 0.002), Pose((0, 0, -0.002)), _DARK, collide=False
        )
        return Mechanism(static=(grille,))

    def continuous(self, raw: Raw) -> dict[str, float]:
        return {}

    def outputs(self) -> tuple[str, ...]:
        return ("level", "frequency")


@register
@dataclass(frozen=True)
class Vibrotactor(Device):
    """Vibration actuator behind a contact pad, e.g. in a seat or wheel rim.

    The engine sets ``amplitude`` (peak acceleration of the pad, m/s²) and
    ``frequency`` (Hz). The pad senses contact like a touch surface
    (``touches``, ``u``, ``v``, ``force``), so a tactile model can tell
    whether skin touches it while it vibrates
    (:meth:`InterfaceRuntime.vibration`). The vibration is a signal, not a
    force on the model.
    """

    size: tuple[float, float] = (0.03, 0.03)
    thickness: float = 0.004
    threshold: float = 0.05

    def mechanism(self) -> Mechanism:
        width, height = self.size
        pad = Part(
            Box((width / 2, height / 2, self.thickness / 2)),
            Pose((0, 0, -self.thickness / 2)),
            _GREY,
        )
        return Mechanism(
            static=(pad,), surface=TouchSpec((width, height), self.threshold)
        )

    def continuous(self, raw: Raw) -> dict[str, float]:
        return {}

    def logic(self) -> Logic:
        surface = self.mechanism().surface
        assert surface is not None
        return _TouchLogic(self.name, surface)

    def outputs(self) -> tuple[str, ...]:
        return ("amplitude", "frequency")


@register
@dataclass(frozen=True)
class Prop(Device):
    """Engine-driven rigid object, e.g. a door, a handed-over cup or a mirror.

    Its pose follows the engine's ``pose`` output as a MuJoCo mocap body, so
    the simulated body can touch it but not push it.
    """

    parts: tuple[Part, ...] = (Part(Box((0.05, 0.05, 0.05)), rgba=_RED),)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mount != "world":
            raise ValueError("props are mocap bodies and must be mounted on the world")

    def mechanism(self) -> Mechanism:
        return Mechanism(moving=tuple(self.parts), mocap=True)

    def continuous(self, raw: Raw) -> dict[str, float]:
        return {}


@register
@dataclass(frozen=True)
class Control(Device):
    """A custom device composed from axes and parts.

    Args:
        axes: Degrees of freedom, nested in order.
        moving: Parts on the last axis.
        static: Parts fixed to the device frame.
        touch: Report the contact force on the moving parts.
        normalize: Report ``value`` (first axis over its range) as well.
    """

    axes: tuple[Axis, ...] = ()
    moving: tuple[Part, ...] = ()
    static: tuple[Part, ...] = ()
    touch: bool = True
    normalize: bool = True

    def mechanism(self) -> Mechanism:
        return Mechanism(
            tuple(self.axes), tuple(self.moving), tuple(self.static), self.touch
        )

    def continuous(self, raw: Raw) -> dict[str, float]:
        values = super().continuous(raw)
        if self.normalize and self.axes:
            first = self.axes[0]
            values["value"] = first.normalized(raw.q[first.name])
        return values


def device_type(name: str) -> type[Device]:
    """Return a registered device class by name."""
    try:
        return DEVICE_TYPES[name]
    except KeyError:
        known = ", ".join(sorted(DEVICE_TYPES))
        raise ValueError(f"unknown device type {name!r}; registered: {known}") from None

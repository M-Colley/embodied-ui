"""Engine-independent lockstep between a MuJoCo body and an engine.

An :class:`Engine` advances by exactly one fixed period per command, so one
MuJoCo control period and one engine frame share a clock. Routes turn device
channels into engine controls; engine outputs (force feedback, lamps,
displays, prop poses) flow back into the MuJoCo model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import mujoco
import numpy as np

from .frames import FLU, Frame, Pose, frame
from .logic import Event
from .runtime import InterfaceRuntime, Reading, merge_readings


class SynchronizationError(RuntimeError):
    """MuJoCo and the engine no longer share one clock."""


@dataclass(frozen=True)
class Command:
    """Engine controls and forwarded events for one frame."""

    time: float
    controls: Mapping[str, float]
    events: tuple[Event, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable command (NaN becomes ``None``)."""
        return {
            "time": self.time,
            "controls": {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in self.controls.items()
            },
            "events": [event.as_dict() for event in self.events],
        }


@dataclass(frozen=True)
class EngineState:
    """What an engine reports after a frame.

    Args:
        frame: Engine frame counter.
        time: Engine simulation time, s.
        outputs: Named values: numbers, lists, poses ``[x, y, z, qw, qx, qy,
            qz]`` and images as NumPy arrays.
    """

    frame: int
    time: float
    outputs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionInfo:
    """What a bridge tells an engine when a session starts.

    Args:
        dt: Shared fixed period, s.
        controls: Control names the engine will receive.
        outputs: Outputs the bridge consumes; engines may send more.
        manifest: Interface manifests, for engines that mirror devices.
    """

    dt: float
    controls: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    manifest: Sequence[Mapping[str, Any]] = ()


@runtime_checkable
class Engine(Protocol):
    """A simulator advanced in lockstep, one fixed period per command."""

    dt: float

    def start(self, info: SessionInfo) -> EngineState:
        """Open the session and return the initial state without advancing."""
        ...

    def reset(
        self, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> EngineState:
        """Restore the initial scenario and return its state."""
        ...

    def step(self, command: Command) -> EngineState:
        """Apply one command, advance exactly ``dt`` and return the new state."""
        ...

    def close(self) -> None:
        """End the session and release engine resources."""
        ...


@dataclass(frozen=True)
class Route:
    """Send one channel to one engine control as ``scale * x + offset``.

    Args:
        source: Channel, e.g. ``wheel.value``.
        target: Engine control, e.g. ``vehicle.steer``.
        scale, offset: Affine map.
        clip: Optional bounds applied last.
    """

    source: str
    target: str
    scale: float = 1.0
    offset: float = 0.0
    clip: tuple[float, float] | None = None

    def value(self, channels: Mapping[str, float]) -> float:
        """Return the routed value; NaN (no touch) stays NaN."""
        try:
            x = float(channels[self.source])
        except KeyError:
            raise KeyError(f"route source {self.source!r} is not a channel") from None
        y = self.scale * x + self.offset
        if self.clip is not None and not math.isnan(y):
            y = min(self.clip[1], max(self.clip[0], y))
        return y


@dataclass(frozen=True)
class Routes:
    """Channel and event routing from devices to an engine.

    Args:
        routes: Channel routes.
        events: ``(device, target)`` pairs; events of ``device`` (and of its
            touch regions) are forwarded with ``device`` renamed ``target``.
        passthrough: Also forward every unrouted channel and event under its
            own name, so engines can bind any channel directly.
        feedback: Routes from engine outputs to device outputs, e.g.
            ``Route("lights.left_blinker", "left_lamp.on")``.
    """

    routes: tuple[Route, ...] = ()
    events: tuple[tuple[str, str], ...] = ()
    passthrough: bool = False
    feedback: tuple[Route, ...] = ()

    def controls(self, channels: Sequence[str]) -> tuple[str, ...]:
        """Return the control names a session will send."""
        names = [route.target for route in self.routes]
        if self.passthrough:
            routed = {route.source for route in self.routes}
            names += [c for c in channels if c not in routed]
        return tuple(dict.fromkeys(names))

    def command(self, reading: Reading) -> Command:
        """Build the command for one reading."""
        controls: dict[str, float] = {}
        for route in self.routes:
            controls[route.target] = route.value(reading.channels)
        if self.passthrough:
            routed = {route.source for route in self.routes}
            for name, value in reading.channels.items():
                if name not in routed and name not in controls:
                    controls[name] = value
        events: list[Event] = []
        for event in reading.events:
            device, _, rest = event.source.partition(".")
            renamed = next((t for d, t in self.events if d == device), None)
            if renamed is not None:
                source = f"{renamed}.{rest}" if rest else renamed
                events.append(
                    Event(event.time, source, event.kind, event.value, event.data)
                )
            elif self.passthrough:
                events.append(event)
        return Command(reading.time, controls, tuple(events))


PASSTHROUGH = Routes(passthrough=True)


@dataclass(frozen=True)
class BridgeFrame:
    """One shared period: what was measured, sent and received."""

    reading: Reading
    command: Command | None
    state: EngineState


class Bridge:
    """Advance an engine once per MuJoCo control period.

    The caller steps MuJoCo (applying each runtime's forces before every
    physics step, e.g. with ``InterfaceRuntime.step``) and then calls
    :meth:`step`, which measures the devices, sends one command and applies
    the returned outputs.

    Args:
        runtimes: Bound interfaces of the MuJoCo model.
        engine: Lockstep engine.
        routes: Channel routing; defaults to passing every channel through.
        engine_frame: Convention of poses the engine sends; ``*.pose``
            outputs are converted to ``model_frame``. ``None`` leaves them.
        model_frame: Convention of the MuJoCo world.
        tolerance: Relative clock tolerance per period.
    """

    def __init__(
        self,
        runtimes: InterfaceRuntime | Sequence[InterfaceRuntime],
        engine: Engine,
        routes: Routes = PASSTHROUGH,
        *,
        engine_frame: Frame | str | None = None,
        model_frame: Frame | str = FLU,
        tolerance: float = 1e-6,
    ) -> None:
        self.runtimes = (
            (runtimes,) if isinstance(runtimes, InterfaceRuntime) else tuple(runtimes)
        )
        if not self.runtimes:
            raise ValueError("a bridge needs at least one interface runtime")
        models = {id(r.model) for r in self.runtimes}
        if len(models) != 1:
            raise ValueError("all runtimes must belong to one compiled model")
        dt = float(engine.dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("engine.dt must be finite and positive")
        self.engine = engine
        self.routes = routes
        self.dt = dt
        self.engine_frame = frame(engine_frame) if engine_frame is not None else None
        self.model_frame = frame(model_frame)
        self.tolerance = float(tolerance)
        self._started = False
        self._failed = False
        self._time: float | None = None
        self._engine_time: float | None = None

    @property
    def model(self) -> mujoco.MjModel:
        """Return the shared compiled model."""
        return self.runtimes[0].model

    def _read(self, data: mujoco.MjData) -> Reading:
        return merge_readings([runtime.read(data) for runtime in self.runtimes])

    def info(self) -> SessionInfo:
        """Return the session description sent to the engine."""
        channels = tuple(c for r in self.runtimes for c in r.channels)
        return SessionInfo(
            self.dt,
            self.routes.controls(channels),
            tuple(o for r in self.runtimes for o in r.outputs),
            tuple(r.interface.to_manifest() for r in self.runtimes),
        )

    def reset(
        self,
        data: mujoco.MjData,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> BridgeFrame:
        """Start or reset the engine and anchor both clocks without advancing."""
        self._failed = True
        for runtime in self.runtimes:
            runtime.reset()
        if not self._started:
            state = self.engine.start(self.info())
            self._started = True
            if seed is not None or options:
                state = self.engine.reset(seed, options)
        else:
            state = self.engine.reset(seed, options)
        self._apply(data, state)
        reading = self._read(data)
        self._time = float(data.time)
        self._engine_time = float(state.time)
        self._failed = False
        return BridgeFrame(reading, None, state)

    def step(self, data: mujoco.MjData) -> BridgeFrame:
        """Measure one completed MuJoCo period and advance the engine once."""
        if self._time is None or self._engine_time is None or self._failed:
            raise RuntimeError("reset a healthy bridge before stepping")
        self._failed = True
        now = float(data.time)
        if not self._one_period(now - self._time):
            raise SynchronizationError(
                f"MuJoCo advanced {now - self._time:.6g} s; "
                f"the bridge period is {self.dt} s"
            )
        reading = self._read(data)
        command = self.routes.command(reading)
        state = self.engine.step(command)
        if not self._one_period(float(state.time) - self._engine_time):
            raise SynchronizationError(
                f"the engine advanced {float(state.time) - self._engine_time:.6g} s; "
                f"expected {self.dt} s"
            )
        self._apply(data, state)
        self._time = now
        self._engine_time = float(state.time)
        self._failed = False
        return BridgeFrame(reading, command, state)

    def close(self) -> None:
        """Close the engine session."""
        if self._started:
            self._started = False
            self.engine.close()

    def __enter__(self) -> Bridge:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _one_period(self, elapsed: float) -> bool:
        return math.isclose(
            elapsed, self.dt, rel_tol=self.tolerance, abs_tol=self.dt * self.tolerance
        )

    def _apply(self, data: mujoco.MjData, state: EngineState) -> None:
        outputs = dict(state.outputs)
        for route in self.routes.feedback:
            if route.source in outputs:
                outputs[route.target] = route.value(outputs)
        if self.engine_frame is not None:
            for key, value in outputs.items():
                if key.endswith(".pose") and value is not None:
                    pose = np.asarray(value, dtype=float).ravel()
                    converted = self.engine_frame.pose_to(
                        self.model_frame,
                        Pose(tuple(pose[:3]), tuple(pose[3:])),  # type: ignore[arg-type]
                    )
                    outputs[key] = [*converted.pos, *converted.quat]
        for runtime in self.runtimes:
            runtime.set_outputs(data, outputs)

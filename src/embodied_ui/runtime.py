"""Read channels and events from a compiled model and apply device forces.

Call :meth:`InterfaceRuntime.apply` before every physics step (it writes the
devices' own entries of ``qfrc_applied`` and nothing else), and
:meth:`InterfaceRuntime.read` once per control period after stepping.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from .devices import Device, Logic, Mechanism, Raw
from .logic import Event, Touch
from .mechanics import Axis

if TYPE_CHECKING:
    from .interface import Interface


@dataclass(frozen=True)
class Reading:
    """Channels and events of one sample.

    Channels are ``device.channel`` floats; discrete states are 0 or 1 and
    missing touch coordinates are NaN.
    """

    time: float
    channels: Mapping[str, float]
    events: tuple[Event, ...] = ()

    def __getitem__(self, name: str) -> float:
        return self.channels[name]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable reading (NaN becomes ``None``)."""
        return {
            "time": self.time,
            "channels": {
                k: (None if math.isnan(v) else v) for k, v in self.channels.items()
            },
            "events": [event.as_dict() for event in self.events],
        }


@dataclass
class _Axis:
    axis: Axis
    qpos: int
    dof: int
    feedback: float = 0.0
    setpoint: float | None = None


@dataclass
class _Device:
    device: Device
    mechanism: Mechanism
    axes: list[_Axis]
    touch: list[int]
    logic: Logic | None
    surface: int | None = None
    texture: tuple[int, int, int, int] | None = None
    indicators: list[int] = field(default_factory=list)
    mocap: int | None = None
    image: NDArray[np.uint8] | None = None
    body: int = -1
    signals: dict[str, float] = field(default_factory=dict)


class InterfaceRuntime:
    """Bound interface for one compiled model.

    Attributes:
        channels: Names of every channel a reading contains.
        outputs: Names of every engine output the devices consume.
    """

    def __init__(self, interface: Interface, model: mujoco.MjModel) -> None:
        self.interface = interface
        self.model = model
        self._devices: list[_Device] = []
        own_bodies: set[int] = set()
        for device in interface.devices:
            mechanism = device.mechanism()
            axes = []
            for axis in mechanism.axes:
                joint = model.joint(interface.element(device.name, axis.name))
                axes.append(_Axis(axis, int(joint.qposadr[0]), int(joint.dofadr[0])))
            touch = []
            if mechanism.touch:
                count = sum(
                    1
                    for index in range(len(mechanism.moving))
                    if _exists(
                        model,
                        mujoco.mjtObj.mjOBJ_SENSOR,
                        interface.element(device.name, f"touch{index}"),
                    )
                )
                touch = [
                    int(
                        model.sensor(interface.element(device.name, f"touch{i}")).adr[0]
                    )
                    for i in range(count)
                ]
            bound = _Device(device, mechanism, axes, touch, device.logic())
            base = model.body(interface.element(device.name))
            bound.body = int(base.id)
            own_bodies.add(int(base.id))
            for axis in mechanism.axes:
                own_bodies.add(
                    int(model.body(interface.element(device.name, axis.name)).id)
                )
            if mechanism.surface is not None:
                bound.surface = int(
                    model.geom(interface.element(device.name, "static0")).id
                )
            if mechanism.display is not None:
                texture = model.texture(interface.element(device.name, "display"))
                bound.texture = (
                    int(texture.adr[0]),
                    int(texture.width[0]),
                    int(texture.height[0]),
                    int(texture.nchannel[0]),
                )
            if mechanism.indicator is not None:
                bound.indicators = [
                    int(model.geom(interface.element(device.name, f"static{i}")).id)
                    for i in range(len(mechanism.static))
                ]
            if mechanism.mocap:
                bound.mocap = int(model.body_mocapid[base.id])
            self._devices.append(bound)
        self._own_bodies = own_bodies
        self._previous: dict[str, float] = {}
        self.channels = self._channel_names()
        self.outputs = tuple(
            f"{bound.device.name}.{name}"
            for bound in self._devices
            for name in bound.device.outputs()
        )
        self.reset()

    # Lifecycle -------------------------------------------------------------

    def reset(self) -> None:
        """Release every state machine and clear engine forces and setpoints."""
        self._previous = {}
        for bound in self._devices:
            bound.signals.clear()
            if bound.logic is not None:
                bound.logic.reset()
            for axis in bound.axes:
                axis.feedback = 0.0
                axis.setpoint = None
        self._last_time: float | None = None

    # Physics ---------------------------------------------------------------

    def apply(self, data: mujoco.MjData) -> None:
        """Write detent, force-curve, servo and engine forces for one step."""
        self._check(data)
        for bound in self._devices:
            for axis in bound.axes:
                spec = axis.axis
                if not (spec.needs_force or axis.feedback):
                    continue
                q = float(data.qpos[axis.qpos])
                force = axis.feedback
                if spec.detents is not None:
                    force += spec.detents.force(q)
                if spec.curve is not None:
                    force += spec.curve.force(q)
                if spec.servo is not None and axis.setpoint is not None:
                    force += spec.servo.force(
                        axis.setpoint, q, float(data.qvel[axis.dof])
                    )
                data.qfrc_applied[axis.dof] = force

    def step(self, data: mujoco.MjData, nstep: int = 1) -> None:
        """Advance ``nstep`` physics steps, applying device forces before each."""
        for _ in range(nstep):
            self.apply(data)
            mujoco.mj_step(self.model, data)

    # Sensing ---------------------------------------------------------------

    def read(self, data: mujoco.MjData) -> Reading:
        """Sample every device; call once per control period."""
        self._check(data)
        time = float(data.time)
        if self._last_time is not None and time < self._last_time:
            raise ValueError(
                "MuJoCo time went backwards; reset the runtime after mj_resetData"
            )
        self._last_time = time
        contacts = self._contacts(data)
        continuous: dict[str, float] = {}
        state: list[tuple[_Device, dict[str, float], Raw]] = []
        for bound in self._devices:
            raw = Raw(
                {a.axis.name: float(data.qpos[a.qpos]) for a in bound.axes},
                {a.axis.name: float(data.qvel[a.dof]) for a in bound.axes},
                float(sum(data.sensordata[adr] for adr in bound.touch)),
                contacts.get(bound.surface, {}) if bound.surface is not None else {},
            )
            own = bound.device.continuous(raw)
            continuous.update(
                {f"{bound.device.name}.{k}": float(v) for k, v in own.items()}
            )
            state.append((bound, own, raw))
        merged = dict(self._previous)
        merged.update(continuous)
        events: list[Event] = []
        discrete: dict[str, float] = {}
        for bound, own, raw in state:
            if bound.logic is None:
                continue
            values, new = bound.logic.update(time, own, merged, raw)
            prefixed = {f"{bound.device.name}.{k}": float(v) for k, v in values.items()}
            merged.update(prefixed)
            discrete.update(prefixed)
            events.extend(new)
        self._previous = discrete
        channels = {**continuous, **discrete}
        return Reading(time, channels, tuple(events))

    def _contacts(self, data: mujoco.MjData) -> dict[int, dict[int, Touch]]:
        surfaces = {b.surface: b for b in self._devices if b.surface is not None}
        if not surfaces or data.ncon == 0:
            return {}
        geom1 = np.asarray(data.contact.geom1[: data.ncon])
        geom2 = np.asarray(data.contact.geom2[: data.ncon])
        result: dict[int, dict[int, Touch]] = {}
        wrench = np.zeros(6)
        for surface, bound in surfaces.items():
            indices = np.flatnonzero((geom1 == surface) | (geom2 == surface))
            if len(indices) == 0:
                continue
            spec = bound.mechanism.surface
            assert spec is not None
            width, height = spec.size
            origin = data.geom_xpos[surface]
            rotation = data.geom_xmat[surface].reshape(3, 3)
            # The panel geom sits half its thickness behind the device frame.
            groups: dict[int, list[tuple[float, float, float]]] = {}
            for index in indices:
                other = int(geom2[index] if geom1[index] == surface else geom1[index])
                body = int(self.model.geom_bodyid[other])
                if body in self._own_bodies:
                    continue
                mujoco.mj_contactForce(self.model, data, int(index), wrench)
                local = rotation.T @ (data.contact.pos[index] - origin)
                groups.setdefault(body, []).append(
                    (float(local[0]), float(local[1]), max(0.0, float(wrench[0])))
                )
            touches = {}
            for body, points in groups.items():
                array = np.asarray(points)
                total = float(array[:, 2].sum())
                weights = (
                    array[:, 2] / total
                    if total > 0
                    else np.full(len(array), 1 / len(array))
                )
                x = float(weights @ array[:, 0])
                y = float(weights @ array[:, 1])
                u = min(1.0, max(0.0, x / width + 0.5))
                v = min(1.0, max(0.0, 0.5 - y / height))
                touches[body] = Touch(u, v, total)
            result[surface] = touches
        return result

    # Engine outputs ----------------------------------------------------------

    def set_outputs(self, data: mujoco.MjData, outputs: Mapping[str, Any]) -> None:
        """Apply engine outputs addressed to devices; ignore other keys.

        Recognized keys: ``device.force``/``device.torque`` and
        ``device.axis.force`` (generalized force added every step),
        ``device.setpoint``/``device.axis.setpoint`` (servo target),
        ``device.on`` (indicator), ``device.image`` (H x W x 3 uint8 display
        frame, top row first), ``device.pose`` (mocap position and
        (w, x, y, z) quaternion in the model's frame), and the signals
        ``device.level``/``device.frequency`` (speaker) and
        ``device.amplitude``/``device.frequency`` (vibrotactor), kept for
        perception models (:meth:`signals`).
        """
        self._check(data)
        for bound in self._devices:
            name = bound.device.name
            prefix = name + "."
            single = bound.axes[0] if len(bound.axes) == 1 else None
            for key, value in outputs.items():
                if not key.startswith(prefix):
                    continue
                suffix = key[len(prefix) :]
                if suffix in _SIGNALS and suffix in bound.device.outputs():
                    bound.signals[suffix] = _finite(value, key)
                head, _, tail = suffix.partition(".")
                if suffix in ("force", "torque") and single is not None:
                    single.feedback = _finite(value, key)
                elif suffix == "setpoint" and single is not None:
                    single.setpoint = None if value is None else _finite(value, key)
                elif tail in ("force", "torque", "setpoint"):
                    axis = next((a for a in bound.axes if a.axis.name == head), None)
                    if axis is None:
                        raise KeyError(f"{name} has no axis {head!r}")
                    if tail == "setpoint":
                        axis.setpoint = None if value is None else _finite(value, key)
                    else:
                        axis.feedback = _finite(value, key)
                elif suffix == "on" and bound.indicators:
                    off, on = bound.mechanism.indicator  # type: ignore[misc]
                    colour = on if float(value) >= 0.5 else off
                    for geom in bound.indicators:
                        self.model.geom_rgba[geom] = colour
                elif suffix == "image" and bound.texture is not None:
                    self._set_image(bound, value)
                elif suffix == "pose" and bound.mocap is not None:
                    pose = np.asarray(value, dtype=float).ravel()
                    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                        raise ValueError(f"{key} must be seven finite numbers")
                    data.mocap_pos[bound.mocap] = pose[:3]
                    data.mocap_quat[bound.mocap] = pose[3:] / np.linalg.norm(pose[3:])

    def _set_image(self, bound: _Device, value: Any) -> None:
        assert bound.texture is not None
        address, width, height, channels = bound.texture
        image = np.asarray(value)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("display images must be H x W x 3 (or 4) uint8 arrays")
        if image.shape[:2] != (height, width):
            raise ValueError(
                f"{bound.device.name} displays {width}x{height} pixels, got "
                f"{image.shape[1]}x{image.shape[0]}"
            )
        bound.image = image[:, :, :3].copy()
        # MuJoCo's texture rows run bottom-up in texture coordinates.
        flat = bound.image[::-1].reshape(-1)
        self.model.tex_data[address : address + width * height * channels] = flat

    def image(self, device: str) -> NDArray[np.uint8] | None:
        """Return the last frame shown on a display, top row first."""
        return self._bound(device).image

    def signals(self, device: str) -> dict[str, float]:
        """Return the last signal outputs of a device: ``on`` of a lamp,
        ``level`` and ``frequency`` of a speaker, ``amplitude`` and
        ``frequency`` of a vibrotactor. Empty until the engine sets them."""
        return dict(self._bound(device).signals)

    def position(self, data: mujoco.MjData, device: str) -> NDArray[np.float64]:
        """Return the world position of a device's frame, m."""
        self._check(data)
        return np.array(data.xpos[self._bound(device).body])

    def sound_level(
        self, data: mujoco.MjData, device: str, point: Sequence[float]
    ) -> float:
        """Free-field sound pressure level of a speaker at a world point,
        dB SPL: its level minus 20 log10(distance / reference distance),
        so -6 dB per doubling of distance. ``-inf`` while silent."""
        bound = self._bound(device)
        level = bound.signals.get("level")
        if level is None:
            return -math.inf
        reference = float(getattr(bound.device, "reference_distance", 1.0))
        offset = np.asarray(point, dtype=float) - self.position(data, device)
        distance = float(np.linalg.norm(offset))
        return level - 20.0 * math.log10(max(distance, 1e-3) / reference)

    def vibration(self, device: str) -> float:
        """Peak acceleration a vibrotactor transmits, m/s²: its amplitude
        while the last reading found something touching its pad, else 0."""
        bound = self._bound(device)
        tracker = getattr(bound.logic, "tracker", None)
        touched = tracker is not None and bool(tracker.touches())
        return bound.signals.get("amplitude", 0.0) if touched else 0.0

    def textures(self) -> list[int]:
        """Return texture ids to re-upload to a renderer after display updates."""
        return [
            int(self.model.texture(self.interface.element(b.device.name, "display")).id)
            for b in self._devices
            if b.texture is not None
        ]

    # Helpers ---------------------------------------------------------------

    def _bound(self, device: str) -> _Device:
        for bound in self._devices:
            if bound.device.name == device:
                return bound
        raise KeyError(device)

    def _check(self, data: mujoco.MjData) -> None:
        if data.model is not self.model:
            raise ValueError("data belongs to a different model than this runtime")

    def _channel_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for bound in self._devices:
            zeros = {a.axis.name: 0.0 for a in bound.axes}
            raw = Raw(zeros, zeros, 0.0, {})
            own = bound.device.continuous(raw)
            names += [f"{bound.device.name}.{k}" for k in own]
            logic = bound.device.logic()
            if logic is not None:
                values, _ = logic.update(0.0, own, {}, raw)
                names += [f"{bound.device.name}.{k}" for k in values]
        return tuple(names)


#: Engine outputs that are signals for perception, not forces or visuals.
_SIGNALS = ("on", "level", "frequency", "amplitude")


def _exists(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> bool:
    return mujoco.mj_name2id(model, kind, name) >= 0


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def merge_readings(readings: Sequence[Reading]) -> Reading:
    """Combine readings of several interfaces taken at the same time."""
    if not readings:
        raise ValueError("nothing to merge")
    times = {r.time for r in readings}
    if len(times) != 1:
        raise ValueError("readings must share one time")
    channels: dict[str, float] = {}
    events: list[Event] = []
    for reading in readings:
        overlap = set(channels) & set(reading.channels)
        if overlap:
            raise ValueError(f"duplicate channels {sorted(overlap)}")
        channels.update(reading.channels)
        events.extend(reading.events)
    return Reading(times.pop(), channels, tuple(events))

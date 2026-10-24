"""CARLA: one vehicle in synchronous lockstep, with lights and cameras.

Controls:
    ``vehicle.throttle``, ``vehicle.brake`` in [0, 1]; ``vehicle.steer`` in
    [-1, 1] (positive steers right, CARLA's convention); ``vehicle.gear``;
    ``vehicle.hand_brake``, ``vehicle.reverse``, ``vehicle.manual_gear_shift``
    and ``lights.<light>`` as booleans (>= 0.5). Lights are ``position``,
    ``low_beam``, ``high_beam``, ``brake``, ``left_blinker``,
    ``right_blinker``, ``reverse``, ``fog``, ``interior``, ``special1``,
    ``special2`` and ``hazard`` (both blinkers).

Outputs, in CARLA's left-handed world frame and SI units like every engine's
vehicle outputs: ``vehicle.pose`` ``[x, y, z, qw, qx, qy, qz]``,
``vehicle.velocity``, ``vehicle.acceleration``, ``vehicle.angular_velocity``
(rad/s, left-handed rates), ``vehicle.speed``; CARLA's own
``vehicle.location`` and ``vehicle.rotation`` (pitch, yaw, roll in degrees);
``lights.<light>``; and ``camera.<name>`` images (H x W x 3 uint8, RGB, top
row first) of the frame just simulated.

The engine owns the world's synchronous settings for its lifetime and
restores them on close, like a single ticking client must.
"""

from __future__ import annotations

import importlib
import math
import queue
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..engine import Command, EngineState, SessionInfo, SynchronizationError
from ..frames import Pose, rotation_from_unreal

_LIGHTS = {
    "position": "Position",
    "low_beam": "LowBeam",
    "high_beam": "HighBeam",
    "brake": "Brake",
    "right_blinker": "RightBlinker",
    "left_blinker": "LeftBlinker",
    "reverse": "Reverse",
    "fog": "Fog",
    "interior": "Interior",
    "special1": "Special1",
    "special2": "Special2",
}
_BOOLEAN = ("hand_brake", "reverse", "manual_gear_shift")


def light_state(carla: Any, lights: Mapping[str, bool]) -> Any:
    """Return a ``carla.VehicleLightState`` with the named lights on.

    ``hazard`` switches both blinkers; unknown names raise ``KeyError``.
    """
    on = {name for name, value in lights.items() if value}
    unknown = on - set(_LIGHTS) - {"hazard"}
    if unknown:
        raise KeyError(f"unknown CARLA lights {sorted(unknown)}")
    if "hazard" in on:
        on |= {"left_blinker", "right_blinker"}
    flags = carla.VehicleLightState.NONE
    for light in on - {"hazard"}:
        flags |= getattr(carla.VehicleLightState, _LIGHTS[light])
    return carla.VehicleLightState(flags)


def light_outputs(carla: Any, state: Any) -> dict[str, float]:
    """Return ``lights.<light>`` outputs (0 or 1) of a ``VehicleLightState``."""
    return {
        f"lights.{light}": float(bool(state & getattr(carla.VehicleLightState, flag)))
        for light, flag in _LIGHTS.items()
    }


@dataclass(frozen=True)
class CarlaCamera:
    """An RGB camera rigidly attached to the vehicle.

    The default places it at a driver's eyes in CARLA's Nissan Patrol;
    adjust ``location`` for other vehicles.

    Args:
        name: Output ``camera.<name>``.
        location: Vehicle-frame position (forward, right, up), m.
        rotation: (pitch, yaw, roll), degrees.
        width, height: Image size, pixels.
        fov: Horizontal field of view, degrees.
    """

    name: str
    location: tuple[float, float, float] = (-0.15, -0.38, 1.65)
    rotation: tuple[float, float, float] = (-5.0, 0.0, 0.0)
    width: int = 640
    height: int = 360
    fov: float = 90.0


class CarlaEngine:
    """Lockstep engine for one CARLA vehicle.

    Args:
        host, port: CARLA server.
        dt: Fixed delta seconds shared with MuJoCo.
        vehicle: Actor id of an existing vehicle, or a blueprint filter to
            spawn one at ``spawn_point`` (destroyed on close).
        spawn_point: Index into the map's recommended spawn points.
        cameras: RGB cameras to attach.
        timeout: RPC and tick deadline, s.
        allow_existing_synchronous: Take over a world another client already
            runs synchronously.
        no_rendering: Disable rendering (no camera images).
        max_substep: Maximum physics substep, s.
        max_substeps: Maximum physics substeps per frame.
        settle: Seconds simulated, braked, after spawning and on reset, so
            the vehicle has dropped onto its wheels before the session's
            first frame.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        *,
        dt: float = 0.02,
        vehicle: int | str = "vehicle.*",
        spawn_point: int = 0,
        cameras: tuple[CarlaCamera, ...] = (),
        timeout: float = 10.0,
        allow_existing_synchronous: bool = False,
        no_rendering: bool = False,
        max_substep: float = 0.01,
        max_substeps: int = 10,
        settle: float = 1.0,
    ) -> None:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if dt > max_substep * max_substeps:
            raise ValueError("dt must not exceed max_substep * max_substeps")
        self.host, self.port = host, int(port)
        self.dt = float(dt)
        self.vehicle_request = vehicle
        self.spawn_point = int(spawn_point)
        self.cameras = tuple(cameras)
        self.timeout = float(timeout)
        self.allow_existing_synchronous = allow_existing_synchronous
        self.no_rendering = no_rendering
        self.max_substep = float(max_substep)
        self.max_substeps = int(max_substeps)
        if not math.isfinite(settle) or settle < 0:
            raise ValueError("settle must be finite and non-negative")
        self.settle = float(settle)
        self._carla: Any = None
        self.client: Any = None
        self.world: Any = None
        self.vehicle: Any = None
        self._original: Any = None
        self._own_vehicle = False
        self._spawn_transform: Any = None
        self._sensors: list[tuple[CarlaCamera, Any, queue.Queue[Any]]] = []
        self._frame: int | None = None
        self._control: dict[str, float] = {}
        self._lights: dict[str, bool] = {}

    # Session ---------------------------------------------------------------

    def start(self, info: SessionInfo) -> EngineState:
        if not math.isclose(info.dt, self.dt, rel_tol=1e-9):
            raise ValueError("session dt differs from the engine's")
        if self.world is not None:
            raise RuntimeError("the CARLA engine is already started")
        carla = self._carla = importlib.import_module("carla")
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            settings = self.world.get_settings()
            if settings.synchronous_mode and not self.allow_existing_synchronous:
                raise RuntimeError(
                    "the CARLA world is already synchronous: another client may be "
                    "ticking it; pass allow_existing_synchronous=True to take over"
                )
            self._original = settings
            new = carla.WorldSettings(
                synchronous_mode=True,
                no_rendering_mode=self.no_rendering,
                fixed_delta_seconds=self.dt,
                substepping=True,
                max_substep_delta_time=self.max_substep,
                max_substeps=self.max_substeps,
            )
            self.world.apply_settings(new)
            self._attach_vehicle()
            self._attach_cameras()
            return self._settle()
        except BaseException:
            self.close()
            raise

    def _attach_vehicle(self) -> None:
        if isinstance(self.vehicle_request, int):
            self.vehicle = self.world.get_actor(self.vehicle_request)
            if self.vehicle is None:
                raise ValueError(f"no CARLA actor with id {self.vehicle_request}")
            self._spawn_transform = self.vehicle.get_transform()
            return
        blueprints = self.world.get_blueprint_library().filter(self.vehicle_request)
        if not blueprints:
            raise ValueError(f"no vehicle blueprint matches {self.vehicle_request!r}")
        points = self.world.get_map().get_spawn_points()
        if not 0 <= self.spawn_point < len(points):
            raise ValueError(
                f"spawn point {self.spawn_point} out of range 0..{len(points) - 1}"
            )
        self._spawn_transform = points[self.spawn_point]
        self.vehicle = self.world.spawn_actor(blueprints[0], self._spawn_transform)
        self._own_vehicle = True

    def _attach_cameras(self) -> None:
        carla = self._carla
        library = self.world.get_blueprint_library()
        for camera in self.cameras:
            blueprint = library.find("sensor.camera.rgb")
            blueprint.set_attribute("image_size_x", str(camera.width))
            blueprint.set_attribute("image_size_y", str(camera.height))
            blueprint.set_attribute("fov", str(camera.fov))
            transform = carla.Transform(
                carla.Location(*camera.location), carla.Rotation(*camera.rotation)
            )
            sensor = self.world.spawn_actor(
                blueprint, transform, attach_to=self.vehicle
            )
            images: queue.Queue[Any] = queue.Queue()
            sensor.listen(images.put)
            self._sensors.append((camera, sensor, images))

    def reset(
        self, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> EngineState:
        """Return the vehicle to its start, at rest, and tick once."""
        carla = self._carla
        self._require()
        self.vehicle.set_transform(self._spawn_transform)
        self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        self.vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        self._control = {}
        self._lights = {}
        self.vehicle.set_light_state(carla.VehicleLightState.NONE)
        return self._settle()

    def step(self, command: Command) -> EngineState:
        self._require()
        self._apply(command.controls)
        return self._tick_state()

    def close(self) -> None:
        for _, sensor, _ in self._sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        self._sensors = []
        if self.vehicle is not None and self._carla is not None:
            try:
                self.vehicle.apply_control(
                    self._carla.VehicleControl(brake=1.0, hand_brake=True)
                )
                if self._own_vehicle:
                    self.vehicle.destroy()
            except RuntimeError:
                pass
        self.vehicle = None
        self._own_vehicle = False
        if self.world is not None and self._original is not None:
            self.world.apply_settings(self._original)
        self._original = None
        self.world = None
        self._frame = None

    # Internals -------------------------------------------------------------

    def _require(self) -> None:
        if self.world is None or self.vehicle is None:
            raise RuntimeError("start the CARLA engine first")

    def _apply(self, controls: Mapping[str, float]) -> None:
        carla = self._carla
        for name, value in controls.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            space, _, field = name.partition(".")
            if space == "vehicle":
                if field not in ("throttle", "steer", "brake", "gear", *_BOOLEAN):
                    raise KeyError(f"unknown CARLA vehicle control {name!r}")
                self._control[field] = float(value)
            elif space == "lights":
                if field != "hazard" and field not in _LIGHTS:
                    raise KeyError(f"unknown CARLA light {name!r}")
                self._lights[field] = float(value) >= 0.5
        c = self._control
        control = carla.VehicleControl(
            throttle=min(1.0, max(0.0, c.get("throttle", 0.0))),
            steer=min(1.0, max(-1.0, c.get("steer", 0.0))),
            brake=min(1.0, max(0.0, c.get("brake", 0.0))),
            hand_brake=c.get("hand_brake", 0.0) >= 0.5,
            reverse=c.get("reverse", 0.0) >= 0.5,
            manual_gear_shift=c.get("manual_gear_shift", 0.0) >= 0.5,
            gear=int(round(c.get("gear", 0.0))),
        )
        self.vehicle.apply_control(control)
        if self._lights:
            self.vehicle.set_light_state(light_state(carla, self._lights))

    def _settle(self) -> EngineState:
        """Tick braked for ``settle`` seconds and return the last state."""
        carla = self._carla
        self.vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        state = self._tick_state()
        for _ in range(round(self.settle / self.dt)):
            state = self._tick_state()
        return state

    def _tick_state(self) -> EngineState:
        previous = self._frame
        frame = int(self.world.tick(self.timeout))
        if previous is not None and frame != previous + 1:
            raise SynchronizationError("CARLA frame gap: another client is ticking")
        snapshot = self.world.get_snapshot()
        if snapshot.frame != frame:
            raise SynchronizationError("CARLA snapshot does not match the ticked frame")
        actor = snapshot.find(self.vehicle.id)
        if actor is None:
            raise RuntimeError("the vehicle disappeared from the CARLA world")
        transform = actor.get_transform()
        location = _vector(transform.location)
        rotation = (
            transform.rotation.pitch,
            transform.rotation.yaw,
            transform.rotation.roll,
        )
        velocity = _vector(actor.get_velocity())
        pose = Pose.from_matrix(rotation_from_unreal(rotation), location)
        outputs: dict[str, Any] = {
            "vehicle.location": list(location),
            "vehicle.rotation": list(rotation),
            "vehicle.velocity": list(velocity),
            "vehicle.acceleration": list(_vector(actor.get_acceleration())),
            "vehicle.angular_velocity": [
                math.radians(v) for v in _vector(actor.get_angular_velocity())
            ],
            "vehicle.speed": math.hypot(*velocity),
            "vehicle.pose": [*pose.pos, *pose.quat],
        }
        outputs.update(light_outputs(self._carla, self.vehicle.get_light_state()))
        for camera, _, images in self._sensors:
            outputs[f"camera.{camera.name}"] = self._image(images, frame)
        self._frame = frame
        return EngineState(frame, float(snapshot.timestamp.elapsed_seconds), outputs)

    def _image(self, images: queue.Queue[Any], frame: int) -> np.ndarray:
        while True:
            image = images.get(timeout=self.timeout)
            if image.frame == frame:
                bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                    image.height, image.width, 4
                )
                return bgra[:, :, 2::-1].copy()
            if image.frame > frame:
                raise SynchronizationError("camera image from a later frame")


def _vector(value: Any) -> tuple[float, float, float]:
    return (float(value.x), float(value.y), float(value.z))

"""Convert 6-DOF engine chassis/platform telemetry into model-frame base kinematics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .frames import FLU, Frame, Pose, Vec3, frame, vec3


@dataclass(frozen=True)
class MotionSample:
    """Converted 6-DOF rigid-body kinematics at one control timestamp.

    Vectors are in metres and radians; when extracted with ``in_body_frame=True``
    they are projected into the moving body's local ``model_frame`` axes.
    """

    time: float
    pose: Pose
    lin_vel: Vec3 = (0.0, 0.0, 0.0)
    lin_acc: Vec3 = (0.0, 0.0, 0.0)
    ang_vel: Vec3 = (0.0, 0.0, 0.0)
    ang_acc: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MotionTrajectory:
    """Time-indexed 6-DOF base kinematics ready for replay or D'Alembert loading."""

    times: NDArray[np.float64]
    lin_acc: NDArray[np.float64]
    ang_vel: NDArray[np.float64]
    ang_acc: NDArray[np.float64]
    lin_vel: NDArray[np.float64]
    poses: tuple[Pose, ...]

    def as_base_profile_kwargs(self) -> dict[str, NDArray[np.float64]]:
        """Keyword arguments matching ``RecordedBaseProfile(times=..., ...)``."""
        return {
            "times": self.times.copy(),
            "lin_acc": self.lin_acc.copy(),
            "ang_vel": self.ang_vel.copy(),
            "ang_acc": self.ang_acc.copy(),
        }


def _read_vec3(outputs: Mapping[str, Any], key: str) -> Vec3 | None:
    raw = outputs.get(key)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float).ravel()
    return vec3(tuple(arr), key)


def extract_motion_sample(
    outputs: Mapping[str, Any],
    *,
    time: float = 0.0,
    prefix: str = "vehicle",
    engine_frame: Frame | str | None = None,
    model_frame: Frame | str = FLU,
    in_body_frame: bool = True,
) -> MotionSample:
    """Extract and convert one frame's 6-DOF motion outputs.

    Args:
        outputs: Engine outputs dictionary (e.g. ``state.outputs``).
        time: Sample timestamp [s].
        prefix: Output namespace (``"vehicle"`` reads ``vehicle.pose``, etc.).
        engine_frame: Coordinate convention of ``outputs``; ``None`` assumes
            they are already in ``model_frame``.
        model_frame: Target MuJoCo world/body coordinate convention.
        in_body_frame: Rotate world-frame velocities and accelerations into the
            moving body's local ``model_frame`` axes using ``<prefix>.pose``.
    """
    if not math.isfinite(time):
        raise ValueError("time must be finite")
    target = frame(model_frame)
    source = target if engine_frame is None else frame(engine_frame)

    raw_pose = outputs.get(f"{prefix}.pose")
    if raw_pose is not None:
        flat = np.asarray(raw_pose, dtype=float).ravel()
        if flat.size != 7:
            raise ValueError(f"{prefix}.pose must have 7 elements [x,y,z,qw,qx,qy,qz]")
        pose_src = Pose(vec3(tuple(flat[:3]), "pos"), tuple(flat[3:]))  # type: ignore[arg-type]
        pose_m = source.pose_to(target, pose_src)
    else:
        pose_m = Pose()

    v_src = _read_vec3(outputs, f"{prefix}.velocity") or (0.0, 0.0, 0.0)
    a_src = _read_vec3(outputs, f"{prefix}.acceleration") or (0.0, 0.0, 0.0)
    w_src = _read_vec3(outputs, f"{prefix}.angular_velocity") or (0.0, 0.0, 0.0)
    alpha_raw = _read_vec3(outputs, f"{prefix}.angular_acceleration")

    v_m = source.point_to(target, v_src)
    a_m = source.point_to(target, a_src)
    w_m = source.pseudovector_to(target, w_src)
    alpha_m = (
        np.zeros(3, dtype=float)
        if alpha_raw is None
        else source.pseudovector_to(target, alpha_raw)
    )

    if in_body_frame:
        rot_t = pose_m.rotation().T
        v_m = rot_t @ v_m
        a_m = rot_t @ a_m
        w_m = rot_t @ w_m
        alpha_m = rot_t @ alpha_m

    return MotionSample(
        time=float(time),
        pose=pose_m,
        lin_vel=cast(Vec3, tuple(float(x) for x in v_m)),
        lin_acc=cast(Vec3, tuple(float(x) for x in a_m)),
        ang_vel=cast(Vec3, tuple(float(x) for x in w_m)),
        ang_acc=cast(Vec3, tuple(float(x) for x in alpha_m)),
    )


def extract_motion_trajectory(
    frames: Iterable[Any],
    *,
    prefix: str = "vehicle",
    engine_frame: Frame | str | None = None,
    model_frame: Frame | str = FLU,
    in_body_frame: bool = True,
) -> MotionTrajectory:
    """Convert a sequence of ``BridgeFrame``, ``EngineState`` or trace dicts.

    Derives ``ang_acc`` via finite differences of ``ang_vel`` whenever
    ``<prefix>.angular_acceleration`` is absent from the recorded outputs.
    """
    samples: list[MotionSample] = []
    has_explicit_alpha = True
    for item in frames:
        if isinstance(item, Mapping):
            t = float(item["time"])
            out = item.get("outputs", item)
        elif hasattr(item, "state") and hasattr(item, "reading"):
            t = float(item.reading.time)
            out = item.state.outputs
        elif hasattr(item, "time") and hasattr(item, "outputs"):
            t = float(item.time)
            out = item.outputs
        else:
            raise TypeError(
                "frames items must be BridgeFrame, EngineState, or mapping records"
            )
        if f"{prefix}.angular_acceleration" not in out:
            has_explicit_alpha = False
        samples.append(
            extract_motion_sample(
                out,
                time=t,
                prefix=prefix,
                engine_frame=engine_frame,
                model_frame=model_frame,
                in_body_frame=in_body_frame,
            )
        )

    if len(samples) < 2:
        raise ValueError("at least two frames are required to build a MotionTrajectory")
    times = np.array([s.time for s in samples], dtype=np.float64)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("frame timestamps must be strictly increasing")
    lin_vel = np.array([s.lin_vel for s in samples], dtype=np.float64)
    lin_acc = np.array([s.lin_acc for s in samples], dtype=np.float64)
    ang_vel = np.array([s.ang_vel for s in samples], dtype=np.float64)
    ang_acc = (
        np.array([s.ang_acc for s in samples], dtype=np.float64)
        if has_explicit_alpha
        else np.gradient(ang_vel, times, axis=0)
    )
    return MotionTrajectory(
        times=times,
        lin_acc=lin_acc,
        ang_vel=ang_vel,
        ang_acc=ang_acc,
        lin_vel=lin_vel,
        poses=tuple(s.pose for s in samples),
    )

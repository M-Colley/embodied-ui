"""Poses and coordinate conventions of MuJoCo models and game engines.

Engines disagree on handedness, up axis and units. A :class:`Frame` names
the physical direction of each of its axes, so a conversion between any two
frames is one signed permutation and a unit scale. Rotations transform as
``R' = C R C^T``, which stays a proper rotation even when ``C`` flips
handedness.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import mujoco
import numpy as np
from numpy.typing import NDArray

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

_DIRECTIONS: dict[str, Vec3] = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def vec3(value: Sequence[float], name: str = "vector") -> Vec3:
    """Return a finite 3-vector as a tuple."""
    if len(value) != 3:
        raise ValueError(f"{name} must have three components")
    return cast(Vec3, tuple(_finite(v, name) for v in value))


def unit_quat(value: Sequence[float], name: str = "quaternion") -> Quat:
    """Return a unit (w, x, y, z) quaternion, renormalizing rounding errors.

    Engine exports often carry float32 quaternions, so norms within 1e-3 of
    one are renormalized; anything further off is rejected.
    """
    if len(value) != 4:
        raise ValueError(f"{name} must have four components (w, x, y, z)")
    q = np.array([_finite(v, name) for v in value])
    norm = float(np.linalg.norm(q))
    if not math.isclose(norm, 1.0, abs_tol=1e-3):
        raise ValueError(f"{name} must have unit length, got {norm:.6f}")
    return cast(Quat, tuple(float(v) for v in q / norm))


@dataclass(frozen=True)
class Pose:
    """Position (m) and unit (w, x, y, z) orientation of a child in its parent."""

    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", vec3(self.pos, "pos"))
        object.__setattr__(self, "quat", unit_quat(self.quat, "quat"))

    @classmethod
    def from_matrix(
        cls, rotation: NDArray[np.float64], pos: Sequence[float] | NDArray[np.float64]
    ) -> Pose:
        """Build a pose from a 3x3 rotation matrix and a position."""
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=float).reshape(9))
        return cls(vec3(tuple(pos)), cast(Quat, tuple(quat)))

    @classmethod
    def from_transform(cls, matrix: NDArray[np.float64]) -> Pose:
        """Build a pose from a 4x4 homogeneous rigid transform."""
        matrix = np.asarray(matrix, dtype=float)
        return cls.from_matrix(matrix[:3, :3], matrix[:3, 3])

    def rotation(self) -> NDArray[np.float64]:
        """Return the 3x3 rotation matrix."""
        out = np.zeros(9)
        mujoco.mju_quat2Mat(out, np.asarray(self.quat))
        return out.reshape(3, 3)

    def __mul__(self, other: Pose) -> Pose:
        """Compose: ``(a * b)`` places ``b`` (given in ``a``'s frame) in a's parent."""
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, np.asarray(self.quat), np.asarray(other.quat))
        return Pose(
            tuple(self.apply(other.pos)),  # type: ignore[arg-type]
            cast(Quat, tuple(quat / np.linalg.norm(quat))),
        )

    def inverse(self) -> Pose:
        """Return the inverse transform."""
        quat = np.zeros(4)
        mujoco.mju_negQuat(quat, np.asarray(self.quat))
        pos = -(self.rotation().T @ np.asarray(self.pos))
        return Pose(cast(Vec3, tuple(pos)), cast(Quat, tuple(quat)))

    def apply(self, point: Sequence[float]) -> NDArray[np.float64]:
        """Map a point from this pose's frame to its parent frame."""
        return self.rotation() @ np.asarray(point, dtype=float) + np.asarray(self.pos)


@dataclass(frozen=True)
class Frame:
    """A coordinate convention: the direction of each axis and the length unit.

    A device (button, knob, screen) has its own axes: x to the user's right,
    y up and z out of the panel towards the user. Engines orient objects
    differently, so ``device`` names those three axes in an object's local
    axes; Unity, for example, presses a world-space canvas along local +z.

    Args:
        x, y, z: One of ``forward``, ``backward``, ``left``, ``right``, ``up``
            or ``down`` for each axis.
        unit: Metres per length unit (0.01 for Unreal Engine centimetres).
        name: Label used in manifests and error messages.
        device: Local axes (``+x``, ``-z``, ...) of a device's right, up and
            towards-the-user directions.
    """

    x: str
    y: str
    z: str
    unit: float = 1.0
    name: str = "custom"
    device: tuple[str, str, str] = ("+x", "+y", "+z")

    def __post_init__(self) -> None:
        for axis in (self.x, self.y, self.z):
            if axis not in _DIRECTIONS:
                raise ValueError(f"unknown axis direction {axis!r}")
        if abs(np.linalg.det(self.basis())) != 1.0:
            raise ValueError("frame axes must be three distinct directions")
        if not math.isfinite(self.unit) or self.unit <= 0:
            raise ValueError("unit must be finite and positive")
        if abs(np.linalg.det(self.device_axes())) != 1.0:
            raise ValueError("device axes must be three distinct local axes")

    def basis(self) -> NDArray[np.float64]:
        """Return the columns of this frame's axes in forward-left-up terms."""
        return np.column_stack([_DIRECTIONS[a] for a in (self.x, self.y, self.z)])

    def device_axes(self) -> NDArray[np.float64]:
        """Return D whose columns are a device's axes in local object axes."""
        columns = []
        for axis in self.device:
            if len(axis) != 2 or axis[0] not in "+-" or axis[1] not in "xyz":
                raise ValueError(
                    f"device axis must look like '+x' or '-z', got {axis!r}"
                )
            column = np.zeros(3)
            column["xyz".index(axis[1])] = 1.0 if axis[0] == "+" else -1.0
            columns.append(column)
        return np.column_stack(columns)

    @property
    def right_handed(self) -> bool:
        """Whether x cross y equals z."""
        return bool(np.linalg.det(self.basis()) > 0)

    def matrix_to(self, other: Frame) -> NDArray[np.float64]:
        """Return C with ``v_other = C v_self`` for direction vectors."""
        return other.basis().T @ self.basis()

    def point_to(self, other: Frame, point: Sequence[float]) -> NDArray[np.float64]:
        """Convert a point or displacement, including the unit change."""
        return (
            self.matrix_to(other)
            @ np.asarray(point, dtype=float)
            * (self.unit / other.unit)
        )

    def rotation_to(
        self, other: Frame, rotation: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Convert a rotation matrix expressed in this frame."""
        c = self.matrix_to(other)
        return c @ np.asarray(rotation, dtype=float) @ c.T

    def pose_to(self, other: Frame, pose: Pose) -> Pose:
        """Convert a pose given in this frame's convention."""
        return Pose.from_matrix(
            self.rotation_to(other, pose.rotation()), self.point_to(other, pose.pos)
        )

    def device_pose_to(self, other: Frame, pose: Pose) -> Pose:
        """Convert an engine object's pose to the pose of a device's own axes.

        ``pose`` is the object's position and rotation in this frame; the
        result places the device frame (x right, y up, z towards the user).
        """
        c = self.matrix_to(other)
        rotation = c @ pose.rotation() @ self.device_axes()
        return Pose.from_matrix(rotation, self.point_to(other, pose.pos))

    def device_points(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert N x 3 object-local vertices to device axes in metres."""
        return np.asarray(points, dtype=float) @ self.device_axes() * self.unit

    def pseudovector_to(
        self, other: Frame, value: Sequence[float]
    ) -> NDArray[np.float64]:
        """Convert an angular velocity or torque; it flips with handedness."""
        c = self.matrix_to(other)
        return float(np.linalg.det(c)) * (c @ np.asarray(value, dtype=float))


#: Right-handed robotics convention (REP 103): x forward, y left, z up, metres.
FLU = Frame("forward", "left", "up", name="flu")
#: Unity: left-handed, x right, y up, z forward, metres. A control faces
#: local -z, like a world-space canvas, and is pressed along +z.
UNITY = Frame("right", "up", "forward", name="unity", device=("+x", "+y", "-z"))
#: Unreal Engine: left-handed, x forward, y right, z up, centimetres. A
#: control is pressed along local +x.
UNREAL = Frame(
    "forward", "right", "up", unit=0.01, name="unreal", device=("+y", "+z", "-x")
)
#: CARLA's Python API: Unreal axes in metres.
CARLA = Frame("forward", "right", "up", name="carla", device=("+y", "+z", "-x"))
#: glTF 2.0: right-handed, +y up, +z the asset's front, +x its left.
GLTF = Frame("left", "up", "forward", name="gltf")
#: MyoSuite's seated full body, as in myosuite-carla: x left, y backward, z up.
MYOSUITE_SEATED = Frame("left", "backward", "up", name="myosuite_seated")

FRAMES: dict[str, Frame] = {
    frame.name: frame for frame in (FLU, UNITY, UNREAL, CARLA, GLTF, MYOSUITE_SEATED)
}


def frame(value: Frame | str) -> Frame:
    """Resolve a frame or one of the names in :data:`FRAMES`."""
    if isinstance(value, Frame):
        return value
    try:
        return FRAMES[value]
    except KeyError:
        raise ValueError(
            f"unknown frame {value!r}; known: {', '.join(sorted(FRAMES))}"
        ) from None


def quat_xyzw(quat: Sequence[float]) -> Quat:
    """Reorder an engine (x, y, z, w) quaternion to MuJoCo's (w, x, y, z)."""
    x, y, z, w = (float(v) for v in quat)
    return unit_quat((w, x, y, z))


def rotation_from_unreal(rotation: Sequence[float]) -> NDArray[np.float64]:
    """Return the rotation matrix of an Unreal/CARLA (pitch, yaw, roll) in degrees.

    Columns are the rotated forward, right and up axes in the Unreal frame,
    following ``FRotationMatrix``.
    """
    pitch, yaw, roll = (math.radians(angle) for angle in vec3(rotation, "rotation"))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    rows = np.array(
        [
            [cp * cy, cp * sy, sp],
            [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
            [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp],
        ]
    )
    return rows.T


def rotation_to_unreal(rotation: NDArray[np.float64]) -> tuple[float, float, float]:
    """Return Unreal/CARLA (pitch, yaw, roll) in degrees of a rotation matrix.

    Inverse of :func:`rotation_from_unreal`; the matrix is expressed in the
    Unreal/CARLA frame.
    """
    r = np.asarray(rotation, dtype=float)
    pitch = math.asin(max(-1.0, min(1.0, r[2, 0])))
    yaw = math.atan2(r[1, 0], r[0, 0])
    roll = math.atan2(-r[2, 1], r[2, 2])
    return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))

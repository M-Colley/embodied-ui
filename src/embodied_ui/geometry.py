"""Primitive shapes, engine meshes and their primitive collision fits.

Contacts between a muscle-driven fingertip and a control are most stable on
primitives. Engine meshes are therefore imported for appearance and, by
default, replaced by a fitted primitive for collision; ``fit="mesh"`` keeps
the mesh, which MuJoCo collides as its convex hull.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .frames import Pose, Vec3, frame, vec3

Rgba = tuple[float, float, float, float]
Fit = Literal["mesh", "box", "cylinder", "capsule", "sphere"]


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class Box:
    """Box with half extents in metres."""

    half_size: Vec3

    def __post_init__(self) -> None:
        size = vec3(self.half_size, "half_size")
        for value in size:
            _positive(value, "half_size")
        object.__setattr__(self, "half_size", size)


@dataclass(frozen=True)
class Cylinder:
    """Cylinder along local z."""

    radius: float
    half_length: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive(self.radius, "radius"))
        object.__setattr__(
            self, "half_length", _positive(self.half_length, "half_length")
        )


@dataclass(frozen=True)
class Capsule:
    """Capsule along local z; ``half_length`` excludes the hemispherical caps."""

    radius: float
    half_length: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive(self.radius, "radius"))
        object.__setattr__(
            self, "half_length", _positive(self.half_length, "half_length")
        )


@dataclass(frozen=True)
class Sphere:
    """Sphere."""

    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive(self.radius, "radius"))


@dataclass(frozen=True)
class Mesh:
    """A triangle mesh file (OBJ, STL, PLY, GLB/glTF) exported from an engine.

    Args:
        file: Mesh file; a glTF scene is flattened unless ``node`` is given.
        node: Name of one glTF/scene node to import.
        frame: Convention of the file's object-local vertices, e.g.
            ``"unity"`` for an OBJ written in a Unity object's local space or
            ``"gltf"`` for glTF files; vertices are mapped to device axes.
        scale: Per-axis scale applied in the file's frame before conversion.
        fit: Collision primitive fitted to the mesh, or ``"mesh"`` to keep it.
    """

    file: str
    node: str | None = None
    frame: str = "flu"
    scale: Vec3 = (1.0, 1.0, 1.0)
    fit: Fit = "box"

    def __post_init__(self) -> None:
        object.__setattr__(self, "file", str(self.file))
        object.__setattr__(self, "scale", vec3(self.scale, "scale"))
        frame(self.frame)
        if self.fit not in ("mesh", "box", "cylinder", "capsule", "sphere"):
            raise ValueError(f"unknown fit {self.fit!r}")


Shape = Box | Cylinder | Capsule | Sphere | Mesh


@dataclass(frozen=True)
class Part:
    """One geom of a device.

    Args:
        shape: Primitive or mesh.
        pose: Placement in the device or moving-part frame.
        rgba: Colour.
        mass: Mass in kg; zero for massless visual or static parts.
        collide: Whether the part takes part in contacts.
        friction: Sliding, torsional and rolling friction.
    """

    shape: Shape
    pose: Pose = field(default_factory=Pose)
    rgba: Rgba = (0.2, 0.2, 0.22, 1.0)
    mass: float = 0.0
    collide: bool = True
    friction: Vec3 = (1.0, 0.005, 0.0001)

    def __post_init__(self) -> None:
        if not isinstance(self.shape, (Box, Cylinder, Capsule, Sphere, Mesh)):
            raise TypeError("shape must be a Box, Cylinder, Capsule, Sphere or Mesh")
        if len(self.rgba) != 4 or not all(0 <= float(c) <= 1 for c in self.rgba):
            raise ValueError("rgba must contain four values in [0, 1]")
        mass = float(self.mass)
        if not math.isfinite(mass) or mass < 0:
            raise ValueError("mass must be finite and non-negative")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "friction", vec3(self.friction, "friction"))


@dataclass(frozen=True)
class LoadedMesh:
    """Vertices (metres, target frame) and triangle faces of an imported mesh."""

    vertices: NDArray[np.float64]
    faces: NDArray[np.int32]


def load_mesh(mesh: Mesh, base: Path | None = None) -> LoadedMesh:
    """Load a mesh with trimesh in device axes and metres.

    Faces are re-wound when the conversion flips handedness, so outward
    normals stay outward.
    """
    try:
        import trimesh
    except ImportError as error:  # pragma: no cover - exercised without extra
        raise ImportError(
            "Loading meshes needs trimesh: pip install 'embodied-ui[mesh]'"
        ) from error
    path = Path(mesh.file)
    if base is not None and not path.is_absolute():
        path = base / path
    if not path.exists():
        raise FileNotFoundError(f"mesh file not found: {path}")
    loaded: Any = trimesh.load(str(path), force="scene")
    if mesh.node is not None:
        if mesh.node not in loaded.graph.nodes_geometry:
            raise ValueError(f"{path} has no geometry node {mesh.node!r}")
        transform, geometry_name = loaded.graph[mesh.node]
        geometry = loaded.geometry[geometry_name].copy()
        geometry.apply_transform(transform)
    else:
        geometry = (
            loaded.to_geometry()
            if hasattr(loaded, "to_geometry")
            else loaded.dump(concatenate=True)
        )
    vertices = np.asarray(geometry.vertices, dtype=float) * np.asarray(mesh.scale)
    faces = np.asarray(geometry.faces, dtype=np.int32)
    source = frame(mesh.frame)
    converted = source.device_points(vertices)
    if np.linalg.det(source.device_axes()) < 0:
        faces = faces[:, ::-1].copy()
    if len(faces) == 0:
        raise ValueError(f"{path} contains no triangles")
    return LoadedMesh(converted, faces)


def fit_primitive(vertices: NDArray[np.float64], kind: Fit) -> tuple[Shape, Pose]:
    """Fit an enclosing primitive and return it with its pose in the mesh frame.

    ``box`` is the minimum-volume oriented box, ``sphere`` the minimum
    enclosing sphere and ``cylinder`` the minimum enclosing cylinder found by
    trimesh. ``capsule`` shares the cylinder's axis and radius with its end
    caps inside the cylinder's ends, which suits rounded caps and handles.
    """
    import trimesh

    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("fitting needs at least four 3-D points")
    if kind == "box":
        to_origin, extents = trimesh.bounds.oriented_bounds(points)
        pose = Pose.from_transform(np.linalg.inv(to_origin))
        return Box(tuple(np.maximum(np.asarray(extents) / 2, 1e-6))), pose  # type: ignore[arg-type]
    if kind == "sphere":
        centre, radius = trimesh.nsphere.minimum_nsphere(points)
        return Sphere(float(radius)), Pose(tuple(centre))  # type: ignore[arg-type]
    if kind in ("cylinder", "capsule"):
        hull = trimesh.convex.convex_hull(points)
        result = trimesh.bounds.minimum_cylinder(hull)
        radius = float(result["radius"])
        half = float(result["height"]) / 2
        pose = Pose.from_transform(result["transform"])
        if kind == "cylinder":
            return Cylinder(radius, max(half, 1e-6)), pose
        return Capsule(radius, max(half - radius, 1e-6)), pose
    raise ValueError(f"cannot fit {kind!r}")

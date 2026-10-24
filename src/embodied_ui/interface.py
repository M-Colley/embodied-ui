"""Compose devices into any MuJoCo model and exchange them as manifests.

An :class:`Interface` is a named set of devices. :meth:`Interface.attach`
adds it to an ``mujoco.MjSpec``; :attr:`Interface.transform` is the same
operation for ``ModelBuilder.apply_transform`` in MyoSuite. After compiling,
:meth:`Interface.bind` returns the runtime that reads channels and events.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import json
import math
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import logic as _logic
from . import mechanics as _mechanics
from .devices import Device, Mechanism, device_type
from .frames import Pose, frame
from .geometry import (
    Box,
    Capsule,
    Cylinder,
    Mesh,
    Part,
    Sphere,
    fit_primitive,
    load_mesh,
)

if TYPE_CHECKING:
    import mujoco

    from .runtime import InterfaceRuntime

MANIFEST_FORMAT = "embodied-ui/manifest"
MANIFEST_VERSION = 1
# Touch sites extend this far beyond their geoms: MuJoCo places a contact
# point midway through the penetration, so a flush site misses firm presses.
TOUCH_MARGIN = 0.002


@dataclass(frozen=True)
class Contacts:
    """How device geoms collide with the host model.

    By default device geoms use ``contype``/``conaffinity`` bits. Naming
    ``pairs`` (fnmatch patterns of host body names, e.g. ``"distph*"``)
    instead clears the bits and adds explicit contact pairs with every
    colliding geom of the matching bodies, which keeps dense musculoskeletal
    models from touching controls with unintended segments.
    """

    contype: int = 1
    conaffinity: int = 1
    pairs: tuple[str, ...] = ()
    condim: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", tuple(self.pairs))
        if self.condim not in (1, 3, 4, 6):
            raise ValueError("condim must be 1, 3, 4 or 6")


@dataclass(frozen=True)
class Interface:
    """A named set of devices that can be added to any MuJoCo model.

    Args:
        name: Prefix of every generated element, ``name/device/...``.
        devices: Devices in reading order; interlocks see earlier devices'
            channels from the same sample.
        contacts: Collision policy towards the host.
        assets: Directory that relative mesh paths resolve against.
        sensors: Add joint-position sensors for each axis (for loggers and
            observation pipelines that read ``sensordata``).
    """

    name: str
    devices: tuple[Device, ...]
    contacts: Contacts = field(default_factory=Contacts)
    assets: str | None = None
    sensors: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "devices", tuple(self.devices))
        if not self.name or any(c in self.name for c in "/. \t\n"):
            raise ValueError(
                "interface names must be nonempty without '/', '.' or spaces"
            )
        names = [device.name for device in self.devices]
        if len(set(names)) != len(names):
            raise ValueError("device names must be unique")
        for device in self.devices:
            if not isinstance(device, Device):
                raise TypeError("devices must be Device instances")

    def __getitem__(self, name: str) -> Device:
        for device in self.devices:
            if device.name == name:
                return device
        raise KeyError(name)

    # Element names ---------------------------------------------------------

    def element(self, device: str, *parts: str) -> str:
        """Return the MuJoCo name of a generated element."""
        return "/".join((self.name, device, *parts))

    # Composition -----------------------------------------------------------

    @property
    def transform(self) -> Callable[[mujoco.MjSpec], mujoco.MjSpec]:
        """Return ``attach`` as a function for ``ModelBuilder.apply_transform``."""
        return self.attach

    def attach(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """Add every device to ``spec`` in place and return it."""
        import mujoco

        _check_names(spec, self)
        host_geoms = (
            _pair_targets(spec, self.contacts.pairs) if self.contacts.pairs else []
        )
        moving_bodies: list[mujoco.MjsBody] = []
        for device in self.devices:
            mechanism = device.mechanism()
            mount = _find_body(spec, device.mount)
            if mechanism.mocap and mount is not spec.worldbody:
                raise ValueError(
                    f"{device.name}: mocap devices must be mounted on the world"
                )
            base = mount.add_body(
                name=self.element(device.name),
                pos=device.pose.pos,
                quat=device.pose.quat,
                mocap=mechanism.mocap,
            )
            colliders: list[str] = []
            for index, part in enumerate(mechanism.static):
                colliders += self._add_part(spec, base, device, f"static{index}", part)
            parent = base
            for axis in mechanism.axes:
                parent = parent.add_body(name=self.element(device.name, axis.name))
                _add_joint(spec, parent, self.element(device.name, axis.name), axis)
                moving_bodies.append(parent)
                if device.mount != "world":
                    spec.add_exclude(bodyname1=parent.name, bodyname2=device.mount)
            carrier = parent if mechanism.axes else base
            touch_parts: list[tuple[str, Part]] = []
            for index, part in enumerate(mechanism.moving):
                names = self._add_part(spec, carrier, device, f"part{index}", part)
                colliders += names
                touch_parts += [(name, part) for name in names]
            if mechanism.touch:
                for index, (geom_name, _) in enumerate(touch_parts):
                    _add_touch_site(
                        spec,
                        carrier,
                        geom_name,
                        self.element(device.name, f"touch{index}"),
                    )
            if mechanism.display is not None:
                _add_display(
                    spec,
                    self.element(device.name, "static0"),
                    self.element(device.name, "display"),
                    mechanism,
                )
            if self.sensors:
                for axis in mechanism.axes:
                    spec.add_sensor(
                        name=self.element(device.name, axis.name),
                        type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT,
                        objname=self.element(device.name, axis.name),
                    )
            for name in colliders:
                for host in host_geoms:
                    spec.add_pair(
                        geomname1=name, geomname2=host, condim=self.contacts.condim
                    )
        _set_explicit_inertias(moving_bodies)
        return spec

    def _add_part(
        self,
        spec: mujoco.MjSpec,
        body: mujoco.MjsBody,
        device: Device,
        label: str,
        part: Part,
    ) -> list[str]:
        """Add one part; return the names of its colliding geoms."""
        import mujoco

        name = self.element(device.name, label)
        collide = part.collide
        bits = (
            (0, 0)
            if (self.contacts.pairs or not collide)
            else (
                self.contacts.contype,
                self.contacts.conaffinity,
            )
        )
        shape = part.shape
        pose = part.pose
        if isinstance(shape, Mesh):
            base = Path(self.assets) if self.assets else None
            loaded = load_mesh(shape, base)
            mesh_name = self.element(device.name, label, "mesh")
            spec.add_mesh(
                name=mesh_name,
                uservert=loaded.vertices.ravel().tolist(),
                userface=loaded.faces.ravel().tolist(),
            )
            visual = body.add_geom(
                name=f"{name}/visual" if shape.fit != "mesh" else name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                pos=pose.pos,
                quat=pose.quat,
                rgba=part.rgba,
            )
            _geom_defaults(visual)
            if shape.fit == "mesh":
                visual.mass = part.mass
                visual.contype, visual.conaffinity = bits
                visual.condim = self.contacts.condim
                visual.friction = part.friction
                return [name] if collide else []
            visual.contype = visual.conaffinity = 0
            visual.group = 1
            visual.mass = 0.0
            primitive, local = fit_primitive(loaded.vertices, shape.fit)
            shape, pose = primitive, pose * local
        geom = body.add_geom(name=name, pos=pose.pos, quat=pose.quat, rgba=part.rgba)
        _geom_defaults(geom)
        geom.type, geom.size = _primitive(shape)
        if isinstance(part.shape, Mesh):
            geom.rgba = (0, 0, 0, 0)
            geom.group = 3
        geom.mass = part.mass
        geom.contype, geom.conaffinity = bits
        geom.condim = self.contacts.condim
        geom.friction = part.friction
        return [name] if collide else []

    # Runtime ---------------------------------------------------------------

    def bind(self, model: mujoco.MjModel) -> InterfaceRuntime:
        """Resolve element addresses in a compiled model and return a runtime."""
        from .runtime import InterfaceRuntime

        return InterfaceRuntime(self, model)

    # Manifests -------------------------------------------------------------

    def to_manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable description in this package's frame."""
        return {
            "format": MANIFEST_FORMAT,
            "version": MANIFEST_VERSION,
            "frame": "flu",
            "name": self.name,
            "contacts": _encode(self.contacts),
            "sensors": self.sensors,
            "devices": [_encode(device) for device in self.devices],
        }

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any] | str | Path,
        *,
        target: str = "flu",
        assets: str | Path | None = None,
    ) -> Interface:
        """Build an interface from a manifest dict or JSON file.

        Device poses given in an engine frame (``"frame": "unity"``) are
        converted to ``target`` with that engine's device-axis convention.
        Relative mesh paths resolve against ``assets``, by default the
        manifest file's directory.
        """
        if isinstance(manifest, (str, Path)):
            path = Path(manifest)
            data = json.loads(path.read_text(encoding="utf-8"))
            assets = assets if assets is not None else path.parent
        else:
            data = dict(manifest)
        if data.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"not an {MANIFEST_FORMAT} document")
        if int(data.get("version", 0)) != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest version {data.get('version')!r}")
        source = frame(data.get("frame", "flu"))
        destination = frame(target)
        built = []
        for item in data["devices"]:
            device = _decode(item)
            if not isinstance(device, Device):
                raise ValueError("manifest devices must decode to devices")
            if source != destination:
                device = dataclasses.replace(
                    device, pose=source.device_pose_to(destination, device.pose)
                )
            built.append(device)
        contacts = _decode(data["contacts"]) if "contacts" in data else Contacts()
        return cls(
            str(data["name"]),
            tuple(built),
            contacts,
            str(assets) if assets is not None else None,
            bool(data.get("sensors", True)),
        )

    def save_manifest(self, path: str | Path) -> None:
        """Write :meth:`to_manifest` as JSON."""
        Path(path).write_text(
            json.dumps(self.to_manifest(), indent=2), encoding="utf-8"
        )


# Spec helpers ----------------------------------------------------------------


def _find_body(spec: mujoco.MjSpec, name: str) -> mujoco.MjsBody:
    if name in ("world", "worldbody"):
        return spec.worldbody
    for body in spec.bodies:
        if body.name == name:
            return body
    raise KeyError(f"mount body {name!r} does not exist in the host model")


def _check_names(spec: mujoco.MjSpec, interface: Interface) -> None:
    prefix = interface.name + "/"
    for collection in (spec.bodies, spec.joints, spec.geoms, spec.sites, spec.sensors):
        for element in collection:
            if element.name.startswith(prefix):
                raise ValueError(
                    f"the host already contains {element.name!r}; was the interface "
                    "attached twice?"
                )


def _pair_targets(spec: mujoco.MjSpec, patterns: Sequence[str]) -> list[str]:
    targets: list[str] = []
    for body in spec.bodies:
        if any(fnmatch.fnmatchcase(body.name, pattern) for pattern in patterns):
            for geom in body.geoms:
                if (geom.contype or geom.conaffinity) and geom.name:
                    targets.append(geom.name)
    if not targets:
        raise ValueError(
            f"no named colliding geoms on bodies matching {list(patterns)}"
        )
    return targets


def _geom_defaults(geom: mujoco.MjsGeom) -> None:
    """Override host default classes so devices behave identically anywhere."""
    geom.margin = 0.0
    geom.gap = 0.0
    geom.solref = (0.02, 1.0)
    geom.solimp = (0.9, 0.95, 0.001, 0.5, 2.0)
    geom.solmix = 1.0
    geom.priority = 0
    geom.group = 0
    geom.condim = 3


def _primitive(shape: object) -> tuple[Any, tuple[float, float, float]]:
    import mujoco

    if isinstance(shape, Box):
        return mujoco.mjtGeom.mjGEOM_BOX, shape.half_size
    if isinstance(shape, Cylinder):
        return mujoco.mjtGeom.mjGEOM_CYLINDER, (shape.radius, shape.half_length, 0.0)
    if isinstance(shape, Capsule):
        return mujoco.mjtGeom.mjGEOM_CAPSULE, (shape.radius, shape.half_length, 0.0)
    if isinstance(shape, Sphere):
        return mujoco.mjtGeom.mjGEOM_SPHERE, (shape.radius, 0.0, 0.0)
    raise TypeError(f"not a primitive: {shape!r}")


def _add_joint(
    spec: mujoco.MjSpec, body: mujoco.MjsBody, name: str, axis: _mechanics.Axis
) -> None:
    import mujoco

    # The host's compiler may read hinge angles (range, springref) in
    # degrees, MJCF's default; devices are specified in radians.
    scale = 1.0
    if axis.kind == "hinge" and spec.compiler.degree:
        scale = 180.0 / math.pi
    joint = body.add_joint(
        name=name,
        type=mujoco.mjtJoint.mjJNT_SLIDE
        if axis.kind == "slide"
        else mujoco.mjtJoint.mjJNT_HINGE,
        axis=axis.direction,
        pos=(0.0, 0.0, 0.0),
        stiffness=axis.stiffness,
        springref=axis.rest * scale,
        damping=axis.damping,
        frictionloss=axis.friction,
        armature=axis.armature,
        ref=0.0,
        margin=0.0,
    )
    if axis.range is not None:
        joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.range = (axis.range[0] * scale, axis.range[1] * scale)
        joint.solref_limit = (axis.stop_time, 1.0)
    else:
        joint.limited = mujoco.mjtLimited.mjLIMITED_FALSE
    joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE


def _add_touch_site(
    spec: mujoco.MjSpec, body: mujoco.MjsBody, geom_name: str, name: str
) -> None:
    import mujoco

    geom = next(g for g in body.geoms if g.name == geom_name)
    size = np.asarray(geom.size, dtype=float).copy()
    if geom.type == mujoco.mjtGeom.mjGEOM_BOX:
        size = size + TOUCH_MARGIN
    elif geom.type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        size[0] += TOUCH_MARGIN
        size[1] += TOUCH_MARGIN
    elif geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
        size[0] += TOUCH_MARGIN
    else:
        raise ValueError(
            "touch sensing needs primitive geoms; fit meshes with fit='box'"
        )
    site = body.add_site(
        name=name,
        type=geom.type,
        size=size,
        pos=geom.pos,
        quat=geom.quat,
        rgba=(0.0, 0.0, 0.0, 0.0),
        group=4,
    )
    site.fromto = (math.nan,) * 6
    site.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    spec.add_sensor(
        name=name,
        type=mujoco.mjtSensor.mjSENS_TOUCH,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=name,
    )


def _add_display(
    spec: mujoco.MjSpec, geom_name: str, name: str, mechanism: Mechanism
) -> None:
    import mujoco

    assert mechanism.display is not None
    width, height = mechanism.display.resolution
    spec.add_texture(
        name=name,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=int(width),
        height=int(height),
        nchannel=3,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        rgb1=(0.02, 0.02, 0.03),
        rgb2=(0.02, 0.02, 0.03),
    )
    material = spec.add_material(name=name, rgba=(1.0, 1.0, 1.0, 1.0))
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    material.emission = 0.6
    for geom in spec.geoms:
        if geom.name == geom_name:
            geom.material = name
            geom.rgba = (1.0, 1.0, 1.0, 1.0)
            return
    raise KeyError(geom_name)


def _set_explicit_inertias(bodies: Sequence[mujoco.MjsBody]) -> None:
    """Fix each moving body's inertia from its own geoms, whatever the host's
    compiler settings, and give part-less gimbal bodies a token inertia."""
    import mujoco

    standalone = mujoco.MjSpec()
    measured = []
    for index, body in enumerate(bodies):
        geoms = [g for g in body.geoms if g.mass > 0]
        if not geoms:
            body.explicitinertial = True
            body.mass = 1e-3
            body.ipos = (0.0, 0.0, 0.0)
            body.iquat = (1.0, 0.0, 0.0, 0.0)
            body.inertia = (1e-8, 1e-8, 1e-8)
            continue
        inertial_body = standalone.worldbody.add_body(name=f"b{index}")
        for geom in geoms:
            copy = inertial_body.add_geom(
                type=geom.type,
                size=geom.size,
                pos=geom.pos,
                quat=geom.quat,
                mass=geom.mass,
            )
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError("moving mesh parts need a fitted primitive")
            copy.fromto = (math.nan,) * 6
        measured.append((index, body))
    if not measured:
        return
    model = standalone.compile()
    for index, body in measured:
        inertial = model.body(f"b{index}")
        body.explicitinertial = True
        body.mass = float(inertial.mass[0])
        body.ipos = inertial.ipos
        body.iquat = inertial.iquat
        body.inertia = inertial.inertia


# Manifest encoding -------------------------------------------------------------

_CLASSES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        Pose,
        Part,
        Box,
        Cylinder,
        Capsule,
        Sphere,
        Mesh,
        Contacts,
        _mechanics.Axis,
        _mechanics.Detents,
        _mechanics.ForceCurve,
        _mechanics.Servo,
        _logic.Interlock,
    )
}


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        encoded = {"type": type(value).__name__}
        for item in dataclasses.fields(value):
            encoded[item.name] = _encode(getattr(value, item.name))
        return encoded
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "type" in value:
            name = value["type"]
            cls = _CLASSES.get(name) or device_type(name)
            hints = typing.get_type_hints(cls)
            kwargs = {}
            for key, item in value.items():
                if key == "type":
                    continue
                if key not in hints:
                    raise ValueError(f"{name} has no field {key!r}")
                kwargs[key] = _coerce(_decode(item), hints[key])
            return cls(**kwargs)
        return {str(k): _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return tuple(_decode(v) for v in value)
    return value


def _coerce(value: Any, hint: Any) -> Any:
    """Restore infinities that JSON cannot carry."""
    if value in ("inf", "-inf"):
        return float(value)
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType) and value is None:
        return None
    return value

"""A servo-driven fingertip for tests, examples and idealized-effector baselines.

The probe is a sphere on three world-aligned slide joints and a twist hinge,
each driven by a position servo. It operates devices through contact exactly
like a musculoskeletal finger would, without a body's limits, so it isolates
interface behaviour from motor control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from .frames import Vec3, vec3


@dataclass(frozen=True)
class Probe:
    """Fingertip probe composed into a host spec.

    Args:
        name: Prefix of generated elements.
        pos: Home position in the world, m.
        radius: Tip radius, m (an adult index fingertip is about 8 mm).
        mass: Moving mass, kg.
        stiffness: Translational servo stiffness, N/m.
        twist_stiffness: Twist servo stiffness, N m/rad.
        friction: Sliding, torsional and rolling friction of the tip.
    """

    name: str = "probe"
    pos: Vec3 = (0.0, 0.0, 0.1)
    radius: float = 0.008
    mass: float = 0.2
    stiffness: float = 2000.0
    twist_stiffness: float = 0.5
    friction: Vec3 = (1.0, 0.02, 0.0001)

    def attach(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """Add the probe and its four position servos."""
        body = spec.worldbody.add_body(name=self.name, pos=vec3(self.pos))
        moment = 0.4 * self.mass * self.radius**2
        axes = (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1)))
        for label, axis in axes:
            body.add_joint(
                name=f"{self.name}/{label}",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=axis,
                armature=0.0,
                damping=0.0,
                stiffness=0.0,
            )
        body.add_joint(
            name=f"{self.name}/twist",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=(0, 0, 1),
            armature=1e-4,
        )
        tip = body.add_geom(
            name=f"{self.name}/tip",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(self.radius, 0, 0),
            mass=self.mass,
            rgba=(0.9, 0.7, 0.6, 1),
            friction=vec3(self.friction),
        )
        tip.condim = 4
        tip.contype = tip.conaffinity = 1
        tip.margin = tip.gap = 0.0
        for label, _ in axes:
            self._servo(spec, label, self.stiffness, self.mass)
        self._servo(spec, "twist", self.twist_stiffness, moment + 1e-4)
        return spec

    def _servo(
        self, spec: mujoco.MjSpec, label: str, kp: float, inertia: float
    ) -> None:
        actuator = spec.add_actuator(
            name=f"{self.name}/{label}",
            target=f"{self.name}/{label}",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
        )
        kv = 2.0 * math.sqrt(kp * inertia)  # critically damped
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.gainprm[0] = kp
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.biasprm[0] = 0.0
        actuator.biasprm[1] = -kp
        actuator.biasprm[2] = -kv

    def bind(self, model: mujoco.MjModel) -> ProbeControl:
        """Return a controller for a compiled model."""
        return ProbeControl(self, model)


class ProbeControl:
    """Set the probe's world target position and twist."""

    def __init__(self, probe: Probe, model: mujoco.MjModel) -> None:
        self.probe = probe
        self.home = np.asarray(probe.pos, dtype=float)
        self._actuators = [
            int(model.actuator(f"{probe.name}/{label}").id)
            for label in ("x", "y", "z", "twist")
        ]
        self._joints = [
            int(model.joint(f"{probe.name}/{label}").qposadr[0])
            for label in ("x", "y", "z")
        ]

    def target(self, data: mujoco.MjData, position: Vec3, twist: float = 0.0) -> None:
        """Command a world position (m) and twist (rad)."""
        offset = np.asarray(vec3(position)) - self.home
        for actuator, value in zip(self._actuators, (*offset, twist)):
            data.ctrl[actuator] = value

    def position(self, data: mujoco.MjData) -> NDArray[np.float64]:
        """Return the tip's world position."""
        return self.home + np.array([data.qpos[i] for i in self._joints])

"""Parametric panel mounting poses and multi-device console layouts."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from .devices import Device
from .frames import FLU, Frame, Pose, Vec3, _finite, frame, vec3
from .interface import Contacts, Interface


def panel_pose(
    anchor: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    distance: float,
    azimuth: float = 0.0,
    elevation: float = 0.0,
    tilt: float = 0.0,
    roll: float = 0.0,
    host_frame: Frame | str = FLU,
) -> Pose:
    """Place a device panel facing back towards ``anchor`` in ``host_frame``.

    At ``azimuth = elevation = tilt = roll = 0`` the panel sits ``distance`` m
    forward of ``anchor``, vertical, with device ``+x`` to the user's right,
    ``+y`` up and ``+z`` pointing back towards ``anchor``. Positive ``azimuth``
    sweeps left (rad), positive ``elevation`` raises the panel (rad), positive
    ``tilt`` reclines the top away from the user (``pi/2`` gives a horizontal
    tabletop), and ``roll`` rotates in-plane about device ``+z``.
    """
    d = _finite(distance, "distance")
    if d <= 0.0:
        raise ValueError("distance must be positive")
    az = _finite(azimuth, "azimuth")
    el = _finite(elevation, "elevation")
    ti = _finite(tilt, "tilt")
    ro = _finite(roll, "roll")
    target = frame(host_frame)
    if not target.right_handed:
        raise ValueError("host_frame must be right-handed for MuJoCo device mounting")
    orig = np.asarray(vec3(anchor, "anchor"), dtype=float)

    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)
    pos_flu = (d * ce * ca, d * ce * sa, d * se)

    # Upright panel facing back towards the origin in FLU (+x fwd, +y left, +z up).
    base_rot = np.array(
        [
            [sa, 0.0, -ca],
            [-ca, 0.0, -sa],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    ct, st = math.cos(-ti), math.sin(-ti)
    rx_tilt = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]], dtype=float)
    cr, sr = math.cos(ro), math.sin(ro)
    rz_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    rot_flu = base_rot @ rx_tilt @ rz_roll
    rot_target = FLU.matrix_to(target) @ rot_flu
    pos_target = orig + FLU.point_to(target, pos_flu)
    return Pose.from_matrix(rot_target, cast(Vec3, tuple(float(v) for v in pos_target)))


@dataclass(frozen=True)
class PanelLayout:
    """Mount multiple devices at local ``(u, v, z)`` offsets on a shared panel."""

    pose: Pose
    mount: str = "world"

    def place(
        self,
        device: Device,
        u: float = 0.0,
        v: float = 0.0,
        z: float = 0.0,
        in_plane_angle: float = 0.0,
    ) -> Device:
        """Return a copy of ``device`` positioned at ``(u, v, z)`` on the panel."""
        ang = _finite(in_plane_angle, "in_plane_angle")
        c, s = math.cos(ang), math.sin(ang)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        local = Pose.from_matrix(
            rot, (_finite(u, "u"), _finite(v, "v"), _finite(z, "z"))
        )
        return dataclasses.replace(
            device,
            pose=self.pose * local * device.pose,
            mount=self.mount,
        )

    def interface(
        self,
        name: str,
        placements: Sequence[Device | tuple[Device, float, float]],
        *,
        contacts: Contacts = Contacts(),
        sensors: bool = True,
    ) -> Interface:
        """Build an :class:`Interface` from devices or ``(device, u, v)`` tuples."""
        mounted: list[Device] = []
        for entry in placements:
            if isinstance(entry, Device):
                mounted.append(self.place(entry))
            else:
                dev, u_off, v_off = entry
                mounted.append(self.place(dev, u=u_off, v=v_off))
        return Interface(
            name=name,
            devices=tuple(mounted),
            contacts=contacts,
            sensors=sensors,
        )

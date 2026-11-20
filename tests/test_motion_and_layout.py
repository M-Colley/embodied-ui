"""Tests for 6-DOF engine motion extraction and parametric panel layouts."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from embodied_ui import (
    CARLA,
    FLU,
    MYOSUITE_SEATED,
    UNITY,
    UNREAL,
    Knob,
    PanelLayout,
    Pose,
    PushButton,
    TouchSurface,
    TraceRecorder,
    extract_motion_sample,
    panel_pose,
)
from embodied_ui.backends import LoopbackEngine
from embodied_ui.engine import Bridge
from embodied_ui.frames import rotation_from_unreal

from .conftest import build_scene


def test_extract_motion_sample_is_heading_invariant_and_flips_handedness() -> None:
    # Car in CARLA turned 90 deg right (nose along +y_carla), braking at -4 m/s^2
    # along its nose (world acceleration = (0, -4, 0)) with left yaw rate -0.5 rad/s
    # about +z_carla (positive +z rate in right-handed FLU).
    pose_90 = Pose.from_matrix(rotation_from_unreal((0.0, 90.0, 0.0)), (10.0, 5.0, 0.0))
    outputs = {
        "vehicle.pose": [*pose_90.pos, *pose_90.quat],
        "vehicle.velocity": [0.0, 12.0, 0.0],
        "vehicle.acceleration": [0.0, -4.0, 0.0],
        "vehicle.angular_velocity": [0.0, 0.0, -0.5],
    }

    flu_sample = extract_motion_sample(
        outputs, time=0.2, engine_frame=CARLA, model_frame=FLU, in_body_frame=True
    )
    np.testing.assert_allclose(flu_sample.lin_vel, (12.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(flu_sample.lin_acc, (-4.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(flu_sample.ang_vel, (0.0, 0.0, 0.5), atol=1e-12)

    seated_sample = extract_motion_sample(
        outputs,
        time=0.2,
        engine_frame=CARLA,
        model_frame=MYOSUITE_SEATED,
        in_body_frame=True,
    )
    # In MYOSUITE_SEATED (+x left, +y backward, +z up), braking -4 fwd is +4 along +y.
    np.testing.assert_allclose(seated_sample.lin_vel, (0.0, -12.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(seated_sample.lin_acc, (0.0, 4.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(seated_sample.ang_vel, (0.0, 0.0, 0.5), atol=1e-12)

    # Unreal centimetres scale linear acceleration from cm/s^2 to m/s^2.
    unreal_sample = extract_motion_sample(
        {"vehicle.acceleration": [250.0, 0.0, 0.0]},
        engine_frame=UNREAL,
        model_frame=FLU,
    )
    np.testing.assert_allclose(unreal_sample.lin_acc, (2.5, 0.0, 0.0), atol=1e-12)


def test_trace_recorder_extracts_motion_trajectory_and_angular_accel() -> None:
    dt = 0.02
    scene = build_scene([PushButton("ok")])

    def step_engine(command, t):
        del command
        return {
            "vehicle.pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "vehicle.velocity": [0.0, 0.0, 5.0 * t],
            "vehicle.acceleration": [0.0, 0.0, 5.0],
            "vehicle.angular_velocity": [0.0, -2.0 * t, 0.0],
        }

    recorder = TraceRecorder()
    with Bridge(scene.runtime, LoopbackEngine(dt, step_engine)) as bridge:
        bridge.reset(scene.data)
        for _ in range(6):
            scene.runtime.step(scene.data, round(dt / scene.model.opt.timestep))
            recorder.record_bridge(bridge.step(scene.data))

    traj = recorder.motion_trajectory(engine_frame=UNITY, model_frame=FLU)
    assert traj.times.shape == (6,)
    # Unity +z -> FLU +x; Unity -y yaw rate -> FLU +z yaw rate (+2*t -> alpha_z = +2).
    np.testing.assert_allclose(traj.lin_acc[:, 0], 5.0, atol=1e-12)
    np.testing.assert_allclose(traj.ang_acc[:, 2], 2.0, atol=1e-10)
    kwargs = traj.as_base_profile_kwargs()
    assert set(kwargs) == {"times", "lin_acc", "ang_vel", "ang_acc"}


def test_panel_pose_and_panel_layout_compose_into_mujoco_spec() -> None:
    # Upright screen 0.5 m ahead in FLU: +x=(0,-1,0), +y=(0,0,1), +z=(-1,0,0).
    upright = panel_pose(distance=0.5, host_frame=FLU)
    np.testing.assert_allclose(upright.pos, (0.5, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(
        upright.rotation(),
        np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        atol=1e-12,
    )

    # Tilt = pi/2 reclines the panel into a horizontal tabletop (+z_dev = +z_flu).
    tabletop = panel_pose(distance=0.4, tilt=math.pi / 2.0, host_frame=FLU)
    np.testing.assert_allclose(tabletop.rotation()[:, 2], (0.0, 0.0, 1.0), atol=1e-12)
    np.testing.assert_allclose(tabletop.rotation()[:, 1], (1.0, 0.0, 0.0), atol=1e-12)

    # Seated MyoSuite convention (+x left, +y backward, +z up): forward is -y.
    seated = panel_pose(
        anchor=(0.0, 0.0, 1.1), distance=0.45, host_frame=MYOSUITE_SEATED
    )
    np.testing.assert_allclose(seated.pos, (0.0, -0.45, 1.1), atol=1e-12)
    np.testing.assert_allclose(seated.rotation()[:, 0], (-1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(seated.rotation()[:, 2], (0.0, 1.0, 0.0), atol=1e-12)

    with pytest.raises(ValueError, match="right-handed"):
        panel_pose(distance=0.5, host_frame=UNITY)

    layout = PanelLayout(
        panel_pose(distance=0.42, tilt=math.radians(25.0), host_frame=FLU)
    )
    iface = layout.interface(
        "console",
        [
            TouchSurface("pad", size=(0.16, 0.10)),
            (PushButton("confirm"), -0.12, 0.0),
            (Knob("gain"), 0.12, 0.0),
        ],
    )
    spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    iface.attach(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    runtime = iface.bind(model)
    pos_confirm = runtime.position(data, "confirm")
    pos_gain = runtime.position(data, "gain")
    # Confirm (-u) sits to user's left (+y_flu); gain (+u) sits to right (-y_flu).
    assert pos_confirm[1] > 0.10
    assert pos_gain[1] < -0.10
    manifest = iface.to_manifest()
    assert len(manifest["devices"]) == 3

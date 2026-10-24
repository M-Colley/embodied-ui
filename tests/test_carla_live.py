"""Live CARLA lockstep with interface devices operated by a fingertip probe.

Enable with ``EMBODIED_UI_CARLA=1`` against a running CARLA server (host and
port from ``CARLA_HOST``/``CARLA_PORT``); ``EMBODIED_UI_CARLA_CAMERA=1`` also
checks camera images and needs a rendering server.
"""

import math
import os

import mujoco
import numpy as np
import pytest

from embodied_ui import (
    Indicator,
    Interface,
    Pose,
    PushButton,
    SteeringWheel,
    ToggleSwitch,
)
from embodied_ui.engine import Bridge, Route, Routes
from embodied_ui.probe import Probe

pytestmark = [
    pytest.mark.carla,
    pytest.mark.skipif(
        os.environ.get("EMBODIED_UI_CARLA") != "1", reason="set EMBODIED_UI_CARLA=1"
    ),
]
DT = 0.02
HOST = os.environ.get("CARLA_HOST", "127.0.0.1")
PORT = int(os.environ.get("CARLA_PORT", "2000"))


def cockpit():
    interface = Interface(
        "cockpit",
        (
            PushButton("throttle", Pose((0, 0, 0)), travel=0.01, preload=2, force=6),
            ToggleSwitch("hazard", Pose((0.1, 0, 0))),
            SteeringWheel("wheel", Pose((0.6, 0, 0.3))),
            Indicator("blinker", Pose((0.2, 0, 0))),
        ),
    )
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    probe = Probe(pos=(0, 0, 0.05))
    probe.attach(spec)
    interface.attach(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    return interface, model, data, interface.bind(model), probe.bind(model)


ROUTES = Routes(
    routes=(
        Route("throttle.value", "vehicle.throttle"),
        Route("wheel.value", "vehicle.steer", scale=-1.0),
        Route("hazard.state", "lights.hazard"),
    ),
    feedback=(Route("lights.left_blinker", "blinker.on"),),
)


def test_fingertip_drives_a_carla_vehicle_and_its_lights():
    from embodied_ui.backends.carla import CarlaEngine

    interface, model, data, runtime, probe = cockpit()
    engine = CarlaEngine(
        HOST, PORT, dt=DT, vehicle="vehicle.nissan.patrol", spawn_point=105
    )
    with Bridge(runtime, engine, ROUTES, engine_frame="carla") as bridge:
        start = bridge.reset(data)
        hazard = model.joint("cockpit/hazard/angle").qposadr[0]
        data.qpos[hazard] = 0.35  # switch the hazard lights on by hand
        frames = []
        for k in range(150):
            # Press the throttle key fully from 0.5 s to 2.5 s.
            down = 0.5 < k * DT < 2.5
            probe.target(data, (0, 0, 0.0 if down else 0.03))
            runtime.step(data, 10)
            frames.append(bridge.step(data))
        settings = engine.world.get_settings()
        assert settings.synchronous_mode and math.isclose(
            settings.fixed_delta_seconds, DT
        )
    assert [f.state.frame - start.state.frame for f in frames] == list(range(1, 151))
    elapsed = frames[-1].state.time - start.state.time
    assert elapsed == pytest.approx(150 * DT, rel=1e-6)
    throttle = [f.command.controls["vehicle.throttle"] for f in frames]
    assert max(throttle) > 0.9 and throttle[-1] < 0.05
    speeds = [f.state.outputs["vehicle.speed"] for f in frames]
    assert max(speeds) > 1.0
    assert frames[-1].state.outputs["lights.left_blinker"] == 1.0
    assert frames[-1].state.outputs["lights.right_blinker"] == 1.0
    lamp = model.geom("cockpit/blinker/static0").id
    np.testing.assert_allclose(model.geom_rgba[lamp], Indicator("x").on)
    pose = np.asarray(frames[-1].state.outputs["vehicle.pose"])
    assert pose.shape == (7,) and np.linalg.norm(pose[3:]) == pytest.approx(1)
    # Closing restored the world's asynchronous settings.
    restored = engine._carla.Client(HOST, PORT).get_world().get_settings()
    assert not restored.synchronous_mode


@pytest.mark.skipif(
    os.environ.get("EMBODIED_UI_CARLA_CAMERA") != "1", reason="needs a rendering server"
)
def test_driver_camera_images_match_the_simulated_frame():
    from embodied_ui.backends.carla import CarlaCamera, CarlaEngine

    interface, model, data, runtime, probe = cockpit()
    camera = CarlaCamera("driver", width=160, height=90)
    engine = CarlaEngine(
        HOST,
        PORT,
        dt=DT,
        vehicle="vehicle.nissan.patrol",
        spawn_point=105,
        cameras=(camera,),
    )
    with Bridge(runtime, engine, ROUTES) as bridge:
        bridge.reset(data)
        for _ in range(10):
            runtime.step(data, 10)
            frame = bridge.step(data)
    image = frame.state.outputs["camera.driver"]
    assert image.shape == (90, 160, 3) and image.dtype == np.uint8
    assert image.std() > 1.0  # not a blank frame

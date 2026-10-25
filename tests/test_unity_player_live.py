"""Live: a built Windows player, outside the Editor, in lockstep with Python.

``EmbodiedUI.PlayerTest.PlayerBuild.Build`` (an Editor script in
``unity/EmbodiedUITests``, run with ``-executeMethod``) generates a minimal
bridge scene and builds a Windows 64-bit player in batch mode. The test
then launches the player and drives it from a MuJoCo fingertip.

The player runs with ``-batchmode`` by default. That shows no window, but
unlike ``-nographics`` it still creates a graphics device, so cameras
render. Set ``EMBODIED_UI_UNITY_PLAYER_MODE=windowed`` for a 320 x 240
window instead. Enable with ``EMBODIED_UI_UNITY=<path to Unity.exe>``.
``EMBODIED_UI_UNITY_PLAYER=<path to EmbodiedUIPlayer.exe>`` reuses a
built player instead.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from embodied_ui import Indicator, Interface, Pose, PushButton
from embodied_ui.backends import RemoteEngine
from embodied_ui.engine import Bridge, Route, Routes
from embodied_ui.probe import Probe

UNITY = os.environ.get("EMBODIED_UI_UNITY")
PLAYER = os.environ.get("EMBODIED_UI_UNITY_PLAYER")
MODE = os.environ.get("EMBODIED_UI_UNITY_PLAYER_MODE", "batchmode")
pytestmark = pytest.mark.skipif(
    not (UNITY or PLAYER),
    reason="set EMBODIED_UI_UNITY to Unity.exe (or EMBODIED_UI_UNITY_PLAYER)",
)
PROJECT = Path(__file__).parents[1] / "unity" / "EmbodiedUITests"
PORT = 47477
DT = 0.02


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    if PLAYER:
        return Path(PLAYER), None
    folder = tmp_path_factory.mktemp("unity-player")
    log = folder / "build.log"
    started = time.perf_counter()
    completed = subprocess.run(
        [
            UNITY,
            "-batchmode",
            "-quit",
            "-projectPath",
            str(PROJECT),
            "-buildTarget",
            "Win64",
            "-executeMethod",
            "EmbodiedUI.PlayerTest.PlayerBuild.Build",
            "-embodiedBuildPath",
            str(folder / "build"),
            "-logFile",
            str(log),
        ],
        timeout=3600,
        check=False,
    )
    exe = folder / "build" / "EmbodiedUIPlayer.exe"
    tail = log.read_text(errors="replace")[-4000:] if log.exists() else ""
    assert completed.returncode == 0 and exe.exists(), tail
    return exe, time.perf_counter() - started


@pytest.fixture
def player(executable, tmp_path):
    exe, _ = executable
    log = tmp_path / "player.log"
    args = [str(exe), "-embodiedPort", str(PORT), "-logFile", str(log)]
    if MODE == "batchmode":
        args.append("-batchmode")
    else:
        args += [
            "-screen-fullscreen",
            "0",
            "-screen-width",
            "320",
            "-screen-height",
            "240",
        ]
    process = subprocess.Popen(args)
    yield process, log
    if process.poll() is None:
        process.kill()


def rig():
    interface = Interface(
        "rig", (PushButton("throttle_key"), Indicator("lamp", Pose((0.6, 0, 0))))
    )
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    probe = Probe(pos=(0, 0, 0.04))
    probe.attach(spec)
    interface.attach(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data, interface.bind(model), probe.bind(model)


def test_a_built_player_runs_in_lockstep_with_python(executable, player, tmp_path):
    process, log = player
    model, data, runtime, tip = rig()
    remote = RemoteEngine(port=PORT, dt=DT, timeout=60, connect_timeout=120)
    routes = Routes(routes=(Route("throttle_key.value", "vehicle.throttle"),))
    lamp = model.geom("rig/lamp/static0").id
    lamp_on_while_pressed = False
    frames = []
    with Bridge(runtime, remote, routes, engine_frame="unity") as bridge:
        first = bridge.reset(data)
        assert remote.engine_name == "Unity player" and remote.convention == "unity"
        started = time.perf_counter()
        for k in range(100):
            t = k * DT
            tip.target(data, (0.0, 0.0, 0.0 if 0.4 < t < 1.6 else 0.04))
            runtime.step(data, 10)
            frames.append(bridge.step(data))
            if 0.8 < t < 1.6 and np.allclose(model.geom_rgba[lamp], Indicator("x").on):
                lamp_on_while_pressed = True
        wall = time.perf_counter() - started
    # The player quits by itself once the session is closed.
    assert process.wait(60) == 0

    outputs = [f.state.outputs for f in frames]
    last = outputs[-1]
    # A player, not the Editor, in the requested mode, with a graphics device.
    assert last["unity.is_editor"] == 0.0
    assert last["unity.batch_mode"] == (1.0 if MODE == "batchmode" else 0.0)
    assert last["unity.renders"] == 1.0
    # Frame timing: one frame of exactly the period and one physics step each.
    assert [f.state.frame for f in frames] == list(range(1, 101))
    assert frames[-1].state.time - first.state.time == pytest.approx(100 * DT)
    delta = np.array([o["unity.delta_time"] for o in outputs])
    np.testing.assert_allclose(delta, DT, atol=1e-6)
    steps = np.diff([o["unity.fixed_steps"] for o in outputs])
    assert (steps == 1).all()
    # The fingertip's key reached the Input System action in the player.
    pressed = [o["unity.throttle_input"] for o in outputs[30:75]]
    assert min(pressed) > 0.9
    assert outputs[10]["unity.throttle_input"] == 0.0
    # Outputs came back: telemetry, the lamp and rendered camera images.
    assert max(o["vehicle.speed"] for o in outputs) > 0.5
    z = [o["vehicle.pose"][2] for o in outputs]
    assert z[-1] > z[0] + 0.5
    assert lamp_on_while_pressed
    image = last["camera.main"]
    assert image.shape == (48, 64, 3) and image.dtype == np.uint8
    assert image.std() > 5.0  # a rendered scene, not a blank frame

    out = os.environ.get("EMBODIED_UI_UNITY_PLAYER_EVIDENCE")
    if out:
        _, build_seconds = executable
        summary = {
            "unity": last["unity.version"],
            "mode": MODE,
            "graphics_device": last["unity.graphics_device"],
            "build_seconds": build_seconds,
            "frames": len(frames),
            "delta_time_s": [float(delta.min()), float(delta.max())],
            "physics_steps_per_frame": sorted(set(steps.tolist())),
            "throttle_input_min_while_pressed": min(pressed),
            "max_speed_mps": max(o["vehicle.speed"] for o in outputs),
            "lamp_on_while_pressed": lamp_on_while_pressed,
            "camera_shape": list(image.shape),
            "camera_pixel_std": float(image.std()),
            "lockstep_frames_per_wall_second": len(frames) / wall,
            "player_exit_code": process.returncode,
        }
        Path(out).write_text(json.dumps(summary, indent=2))
        try:
            from PIL import Image
        except ImportError:
            return
        Image.fromarray(image).save(Path(out).with_suffix(".png"))

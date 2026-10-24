"""Live: a MuJoCo fingertip drives a Unity scene through the Unity package.

Starts Unity in batch mode on ``unity/EmbodiedUITests`` to run the package's
PlayMode tests, including ``PythonClientDrivesTheScene``, which waits for
this client. Enable with ``EMBODIED_UI_UNITY=<path to Unity.exe>``.
"""

import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from embodied_ui import (
    Indicator,
    Interface,
    Pose,
    PushButton,
    Speaker,
    TouchSurface,
    TraceRecorder,
    Vibrotactor,
)
from embodied_ui.backends import RemoteEngine, describe
from embodied_ui.engine import Bridge, Route, Routes
from embodied_ui.probe import Probe

UNITY = os.environ.get("EMBODIED_UI_UNITY")
pytestmark = pytest.mark.skipif(not UNITY, reason="set EMBODIED_UI_UNITY to Unity.exe")
PROJECT = Path(__file__).parents[1] / "unity" / "EmbodiedUITests"
PORT = 47475
DT = 0.02


@pytest.fixture
def unity(tmp_path):
    results = tmp_path / "playmode.xml"
    log = tmp_path / "unity.log"
    process = subprocess.Popen(
        [
            UNITY,
            "-batchmode",
            "-projectPath",
            str(PROJECT),
            "-runTests",
            "-testPlatform",
            "PlayMode",
            "-testResults",
            str(results),
            "-logFile",
            str(log),
        ],
        env={**os.environ, "EMBODIED_UI_LIVE_PORT": str(PORT)},
    )
    yield process, results, log
    if process.poll() is None:
        process.kill()


def rig():
    interface = Interface(
        "rig",
        (
            PushButton("throttle_key"),
            TouchSurface("screen", Pose((0.3, 0, 0))),
            Indicator("lamp", Pose((0.6, 0, 0))),
            Speaker("chime", Pose((0.0, 0.5, 0))),
            Vibrotactor("seat", Pose((0.0, -0.5, 0))),
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


def test_a_fingertip_drives_a_unity_scene(unity, tmp_path):
    process, results, log = unity

    # Devices designed in the Unity scene, fetched at runtime.
    described = describe(
        port=PORT, directory=tmp_path / "assets", timeout=60, connect_timeout=300
    )
    start = described["start"]
    np.testing.assert_allclose(start.pose.pos, (0.4, 0.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(start.pose.rotation() @ (0, 0, -1), (1, 0, 0), atol=1e-6)
    spec = described.attach(mujoco.MjSpec())
    cap = spec.compile().geom(f"{described.name}/start/part0")
    np.testing.assert_allclose(sorted(cap.size), [0.003, 0.0075, 0.0075], atol=1e-6)

    interface, model, data, runtime, tip = rig()
    remote = RemoteEngine(port=PORT, dt=DT, timeout=60, connect_timeout=120)
    routes = Routes(
        routes=(Route("throttle_key.value", "vehicle.throttle"),),
        events=(("screen", "screen"),),
    )
    frames = []
    recorder = TraceRecorder()
    chime_db = []
    seat = []
    lamp = model.geom("rig/lamp/static0").id
    lamp_on_while_pressed = False
    with Bridge(runtime, remote, routes, engine_frame="unity") as bridge:
        first = bridge.reset(data)
        assert remote.engine_name == "Unity test scene" and remote.convention == "unity"
        for k in range(150):
            t = k * DT
            if 0.4 < t < 2.0:
                target = (0.0, 0.0, 0.0)  # press the throttle key
            elif 2.2 < t < 2.6:
                target = (0.3, 0.0, 0.006)  # touch the screen centre
            else:
                target = (0.15 if t >= 2.0 else 0.0, 0.0, 0.04)
            tip.target(data, target)
            runtime.step(data, 10)
            frame = bridge.step(data)
            frames.append(frame)
            recorder.record_bridge(frame)
            # Heard 0.5 m from the chime: 6 dB above its 1 m level.
            ear = runtime.position(data, "chime") + (0.5, 0, 0)
            chime_db.append(runtime.sound_level(data, "chime", ear))
            amplitude = runtime.signals("seat")["amplitude"]
            seat.append((amplitude, runtime.vibration("seat")))
            if 1.0 < t < 2.0 and np.allclose(model.geom_rgba[lamp], Indicator("x").on):
                lamp_on_while_pressed = True
    process.wait(180)

    outputs = [f.state.outputs for f in frames]
    assert [f.state.frame for f in frames] == list(range(1, 151))
    assert frames[-1].state.time - first.state.time == pytest.approx(150 * DT)
    delta = np.array([o["unity.delta_time"] for o in outputs])
    np.testing.assert_allclose(delta, DT, atol=1e-6)
    steps = np.diff([o["unity.fixed_steps"] for o in outputs])
    assert (steps == 1).all()
    pressed = [o["unity.throttle_input"] for o in outputs[30:95]]
    assert min(pressed) > 0.9
    assert max(o["vehicle.speed"] for o in outputs) > 0.5
    z = [o["vehicle.pose"][2] for o in outputs]
    assert z[-1] > z[0] + 0.5  # the car moved along Unity's forward axis
    image = outputs[-1]["camera.main"]
    assert image.shape == (48, 64, 3) and image.dtype == np.uint8
    assert lamp_on_while_pressed
    assert max(o["unity.touches"] for o in outputs[105:135]) == 1
    # Sound and vibration from Unity reach the MuJoCo devices; nobody
    # touches the seat pad, so no vibration is felt.
    assert max(chime_db) == pytest.approx(64.0 + 20 * np.log10(2), abs=1e-6)
    assert chime_db[0] == pytest.approx(0.0 + 20 * np.log10(2), abs=1e-6)
    assert max(a for a, _ in seat) == 2.0 and max(f for _, f in seat) == 0.0
    trace = recorder.write_csv(tmp_path / "trace.csv")
    header = trace.read_text().splitlines()[0].split(",")
    assert {"time", "throttle_key.value", "chime.level", "vehicle.speed"} <= set(header)

    root = ET.parse(results).getroot()
    cases = {case.get("name"): case.get("result") for case in root.iter("test-case")}
    assert cases == {
        "ACSharpClientStepsTheSceneInLockstep": "Passed",
        "PythonClientDrivesTheScene": "Passed",
    }, cases
    version = (
        (PROJECT / "ProjectSettings" / "ProjectVersion.txt").read_text().splitlines()[0]
    )
    summary = {
        "unity": version.split(":", 1)[1].strip(),
        "test_framework_engine": root.get("engine-version") or "",
        "frames": len(frames),
        "delta_time_s": [float(delta.min()), float(delta.max())],
        "physics_steps_per_frame": sorted(set(steps.tolist())),
        "max_speed_mps": max(o["vehicle.speed"] for o in outputs),
        "throttle_input_min_while_pressed": min(pressed),
        "touches_reported": max(o["unity.touches"] for o in outputs[105:135]),
        "chime_db_at_0_5_m": max(chime_db),
        "seat_amplitude_mps2": max(a for a, _ in seat),
        "trace_columns": len(header),
        "camera_shape": list(image.shape),
        "playmode": cases,
    }
    out = os.environ.get("EMBODIED_UI_UNITY_EVIDENCE")
    if out:
        Path(out).write_text(json.dumps(summary, indent=2))
        Path(out).with_suffix(".xml").write_bytes(results.read_bytes())

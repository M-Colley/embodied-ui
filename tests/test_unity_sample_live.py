"""Live: Python drives the package's WheelCollider sample in Unity's PlayMode.

Unity compiles a package's ``Samples~`` folder only after the sample is
imported. Like Package Manager's import, this test copies
``Samples~/WheelVehicle`` to ``Assets/Samples/<package>/<version>/<sample>``
in ``unity/EmbodiedUITests``; it adds ``SampleTests/WheelVehicle`` as the
sample's ``Tests`` folder, with its own test assembly. The copies are
ignored by git. Unity then runs that PlayMode test in batch mode, and a
``RemoteEngine`` drives the car with ``vehicle.throttle``,
``vehicle.brake`` and ``vehicle.steer``. Enable with
``EMBODIED_UI_UNITY=<path to Unity.exe>``.
"""

import json
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from embodied_ui.backends import RemoteEngine
from embodied_ui.engine import Command, SessionInfo

UNITY = os.environ.get("EMBODIED_UI_UNITY")
pytestmark = pytest.mark.skipif(not UNITY, reason="set EMBODIED_UI_UNITY to Unity.exe")
ROOT = Path(__file__).parents[1] / "unity"
PROJECT = ROOT / "EmbodiedUITests"
PACKAGE = ROOT / "io.github.m-colley.embodied-ui"
PORT = 47476
DT = 0.02
CONTROLS = ("vehicle.throttle", "vehicle.brake", "vehicle.steer")
# Seconds at which each phase ends: settle, accelerate, turn right, brake.
SETTLE, ACCELERATE, TURN, BRAKE = 0.5, 3.5, 5.5, 8.0


def import_sample() -> Path:
    """Copy the sample where Package Manager would import it, with its tests."""
    package = json.loads((PACKAGE / "package.json").read_text())
    sample = next(s for s in package["samples"] if s["path"] == "Samples~/WheelVehicle")
    destination = (
        PROJECT
        / "Assets"
        / "Samples"
        / package["displayName"]
        / package["version"]
        / sample["displayName"]
    )
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(PACKAGE / sample["path"], destination)
    shutil.copytree(PROJECT / "SampleTests" / "WheelVehicle", destination / "Tests")
    return destination


@pytest.fixture
def unity(tmp_path):
    import_sample()
    results = tmp_path / "playmode.xml"
    process = subprocess.Popen(
        [
            UNITY,
            "-batchmode",
            "-projectPath",
            str(PROJECT),
            "-runTests",
            "-testPlatform",
            "PlayMode",
            "-assemblyNames",
            "EmbodiedUI.Samples.WheelVehicle.Tests",
            "-testResults",
            str(results),
            "-logFile",
            str(tmp_path / "unity.log"),
        ],
        env={**os.environ, "EMBODIED_UI_SAMPLE_PORT": str(PORT)},
    )
    yield process, results
    if process.poll() is None:
        process.kill()


def profile(t):
    """(throttle, brake, steer) at time ``t``; steer is positive to the right."""
    if t < SETTLE:
        return 0.0, 0.0, 0.0
    if t < ACCELERATE:
        return 1.0, 0.0, 0.0
    if t < TURN:
        return 0.3, 0.0, 1.0
    return 0.0, 1.0, 0.0


def heading(pose):
    """Yaw of a Unity pose in degrees; positive turns right (towards +x)."""
    w, x, y, z = pose[3:]
    forward = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    return math.degrees(math.atan2(forward[0], forward[2]))


def test_python_drives_the_wheel_vehicle_sample(unity, tmp_path):
    process, results = unity
    engine = RemoteEngine(port=PORT, dt=DT, timeout=60, connect_timeout=900)
    first = engine.start(SessionInfo(DT, CONTROLS))
    assert engine.engine_name == "Wheel vehicle sample"
    assert engine.convention == "unity"
    states, sent = [], []
    try:
        for k in range(round(BRAKE / DT)):
            throttle, brake, steer = profile(k * DT)
            controls = dict(zip(CONTROLS, (throttle, brake, steer)))
            states.append(engine.step(Command(k * DT, controls)))
            sent.append((throttle, brake, steer))
    finally:
        engine.close()
    process.wait(300)

    n = len(states)
    out = [s.outputs for s in states]
    t = (np.arange(n) + 1) * DT
    # One lockstep frame per command, each lasting exactly the period.
    assert [s.frame for s in states] == list(range(1, n + 1))
    assert states[-1].time - first.time == pytest.approx(n * DT)
    np.testing.assert_allclose([o["unity.delta_time"] for o in out], DT, atol=1e-6)
    assert (np.diff([o["unity.fixed_steps"] for o in out]) == 1).all()
    # The controls reached the sample's wheel colliders in the same frame.
    sample = np.array([[1500.0, 3000.0, 35.0]]) * np.array(sent)
    np.testing.assert_allclose([o["wheels.motor_torque"] for o in out], sample[:, 0])
    np.testing.assert_allclose([o["wheels.brake_torque"] for o in out], sample[:, 1])
    np.testing.assert_allclose(
        [o["wheels.steer_angle"] for o in out], sample[:, 2], atol=1e-4
    )
    assert all(o["wheels.grounded"] == 4 for o, ti in zip(out, t) if ti > 0.3)

    speed = np.array([o["vehicle.speed"] for o in out])
    yaw = np.array([heading(o["vehicle.pose"]) for o in out])
    yaw_rate = np.array([o["vehicle.angular_velocity"][1] for o in out])

    def at(time):
        return int(round(time / DT)) - 1

    # Accelerates straight: 1500 N m on two 0.35 m wheels limit a 1500 kg
    # car to 5.7 m/s^2.
    assert speed[at(SETTLE)] < 0.1
    assert 5.0 < speed[at(ACCELERATE)] < 5.72 * (ACCELERATE - SETTLE)
    assert abs(yaw[at(ACCELERATE)] - yaw[at(SETTLE)]) < 1.0
    # Turns right with full right steer.
    turned = yaw[at(TURN)] - yaw[at(ACCELERATE)]
    assert turned > 30.0
    assert (yaw_rate[at(ACCELERATE) + 25 : at(TURN)] > 0).all()
    # Brakes to a stop.
    braking = slice(at(TURN), at(TURN + 0.5))
    deceleration = -np.diff(speed[braking]).mean() / DT
    assert deceleration > 3.0
    assert speed[-1] < 0.3

    root = ET.parse(results).getroot()
    cases = {case.get("name"): case.get("result") for case in root.iter("test-case")}
    assert cases == {"PythonClientDrivesTheWheelVehicle": "Passed"}, cases

    evidence = os.environ.get("EMBODIED_UI_UNITY_SAMPLE_EVIDENCE")
    if evidence:
        summary = {
            "frames": n,
            "delta_time_s": [
                min(o["unity.delta_time"] for o in out),
                max(o["unity.delta_time"] for o in out),
            ],
            "speed_after_accelerating_mps": float(speed[at(ACCELERATE)]),
            "heading_change_while_accelerating_deg": float(
                yaw[at(ACCELERATE)] - yaw[at(SETTLE)]
            ),
            "heading_change_while_turning_deg": float(turned),
            "speed_after_turning_mps": float(speed[at(TURN)]),
            "mean_deceleration_first_0_5_s_mps2": float(deceleration),
            "final_speed_mps": float(speed[-1]),
            "playmode": cases,
        }
        Path(evidence).write_text(json.dumps(summary, indent=2))
        Path(evidence).with_suffix(".xml").write_bytes(results.read_bytes())

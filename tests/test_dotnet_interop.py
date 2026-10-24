"""The Unity package's C# protocol layer against the Python client.

Builds ``dotnet/EmbodiedUI.Protocol.Harness`` (which compiles the package's
``Runtime/Protocol`` sources unchanged) and drives its toy engine through a
real bridge. Skipped when the .NET SDK is unavailable.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from embodied_ui import Indicator, Interface, Pose, Screen
from embodied_ui.backends import RemoteEngine, describe
from embodied_ui.engine import Bridge, Route, Routes

from .conftest import Scene

pytestmark = pytest.mark.dotnet
HARNESS = Path(__file__).parents[1] / "dotnet" / "EmbodiedUI.Protocol.Harness"
DT = 0.02


@pytest.fixture(scope="module")
def harness():
    if shutil.which("dotnet") is None:
        pytest.skip("the .NET SDK is not installed")
    build = subprocess.run(
        ["dotnet", "build", "-c", "Release", "--nologo", "-v", "q"],
        cwd=HARNESS,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    dll = HARNESS / "bin" / "Release" / "net9.0" / "EmbodiedUI.Protocol.Harness.dll"
    selftest = subprocess.run(
        ["dotnet", str(dll), "selftest"], capture_output=True, text=True, timeout=120
    )
    assert selftest.returncode == 0, selftest.stdout + selftest.stderr
    process = subprocess.Popen(
        ["dotnet", str(dll), "serve", "0"], stdout=subprocess.PIPE, text=True
    )
    line = process.stdout.readline()
    assert line.startswith("PORT "), line
    yield int(line.split()[1])
    process.kill()
    process.wait(10)


def test_describe_builds_the_engine_designed_interface(harness, tmp_path):
    interface = describe(port=harness, directory=tmp_path, timeout=10)
    assert interface.name == "toy"
    button = interface["b"]
    # 0.4 m along Unity +z (forward) is 0.4 m along +x in FLU; pressing
    # (device -z) pushes forward.
    np.testing.assert_allclose(button.pose.pos, (0.4, 0, 0), atol=1e-12)
    np.testing.assert_allclose(
        button.pose.rotation() @ (0, 0, -1), (1, 0, 0), atol=1e-12
    )


def test_bridge_drives_the_csharp_engine(harness, tmp_path):
    import mujoco

    from embodied_ui.probe import Probe

    described = describe(port=harness, directory=tmp_path, timeout=10)
    # Mix engine-designed devices with devices defined in Python, and face
    # the button upwards so the probe can press it from above.
    button = described["b"]
    interface = Interface(
        "toy",
        (
            type(button)("b", Pose((0, 0, 0))),
            Indicator("lamp", Pose((0.2, 0, 0))),
            Screen("screen", Pose((0, 0.3, 0)), resolution=(8, 4)),
        ),
    )
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    probe = Probe(pos=(0, 0, 0.04))
    probe.attach(spec)
    interface.attach(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    scene = Scene(interface, model, data, interface.bind(model), probe.bind(model))

    remote = RemoteEngine(port=harness, dt=DT, timeout=10)
    routes = Routes(
        routes=(Route("b.value", "vehicle.throttle"),), events=(("b", "b"),)
    )
    bridge = Bridge(scene.runtime, remote, routes, engine_frame="unity")
    start = bridge.reset(data)
    assert remote.engine_name == "dotnet toy" and remote.convention == "unity"
    assert start.state.frame == 0
    frames = []
    for k in range(40):
        down = 0.1 < k * DT < 0.5
        scene.probe.target(data, (0, 0, 0.006 if down else 0.03))
        scene.runtime.step(data, 10)
        frames.append(bridge.step(data))
    assert [f.state.frame for f in frames] == list(range(1, 41))
    assert frames[-1].state.time == pytest.approx(40 * DT)
    presses = [e for f in frames for e in f.command.events if e.kind == "press"]
    assert len(presses) == 1
    # The C# engine toggled its lamp on the press and reports it back.
    lamp = model.geom("toy/lamp/static0").id
    np.testing.assert_allclose(model.geom_rgba[lamp], Indicator("x").on)
    assert frames[-1].state.outputs["vehicle.speed"] > 0
    image = scene.runtime.image("screen")
    assert image.shape == (4, 8, 3) and image.max() > 0
    bridge.close()

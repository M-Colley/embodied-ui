"""Live: the Unity Editor's manifest exporter, loaded and compiled in MuJoCo.

Runs the package's EditMode tests (``Tests/Editor``) in a batch-mode Editor
on ``unity/EmbodiedUITests``. ``ExportWritesTheScenesDevicesAndMeshes``
builds a scene of ``EmbodiedDevice`` components and exports it with
``ManifestExporter.Export``, the routine behind *Tools > Embodied UI >
Export Manifest*. ``EMBODIED_UI_MANIFEST_OUT`` asks it to keep the export.
Python then loads that manifest, compiles it in MuJoCo and presses the
exported button. Enable with ``EMBODIED_UI_UNITY=<path to Unity.exe>``.
"""

import json
import math
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from embodied_ui import Interface, Pose
from embodied_ui.frames import FLU, UNITY
from embodied_ui.probe import Probe

UNITY_EXE = os.environ.get("EMBODIED_UI_UNITY")
pytestmark = pytest.mark.skipif(
    not UNITY_EXE, reason="set EMBODIED_UI_UNITY to Unity.exe"
)
PROJECT = Path(__file__).parents[1] / "unity" / "EmbodiedUITests"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    pytest.importorskip("trimesh")
    folder = tmp_path_factory.mktemp("unity-editor")
    manifest = folder / "export" / "interface.json"
    results = folder / "editmode.xml"
    subprocess.run(
        [
            UNITY_EXE,
            "-batchmode",
            "-projectPath",
            str(PROJECT),
            "-runTests",
            "-testPlatform",
            "EditMode",
            "-assemblyNames",
            "EmbodiedUI.Editor.Tests",
            "-testResults",
            str(results),
            "-logFile",
            str(folder / "unity.log"),
        ],
        env={**os.environ, "EMBODIED_UI_MANIFEST_OUT": str(manifest)},
        timeout=1800,
        check=False,
    )
    return manifest, results


def test_the_editor_exporter_writes_a_manifest_mujoco_compiles(exported, tmp_path):
    manifest, results = exported
    root = ET.parse(results).getroot()
    cases = {case.get("name"): case.get("result") for case in root.iter("test-case")}
    assert cases == {
        "TheMenuItemCallsTheExportRoutine": "Passed",
        "ExportWritesTheScenesDevicesAndMeshes": "Passed",
        "DuplicateDeviceNamesAreRejected": "Passed",
        "PosesAreInWorldCoordinatesWithoutAReference": "Passed",
    }, cases

    # Load with embodied-ui's manifest loader, converting Unity's frame.
    interface = Interface.from_manifest(manifest, target="flu")
    assert interface.name == "untitled"  # the test scene was never saved
    assert [d.name for d in interface.devices] == [
        "hazard_lamp",
        "screen",
        "start",
        "volume",
    ]
    start = interface["start"]
    # 0.5 m up and 0.4 m ahead of the dashboard origin, pressed forward.
    np.testing.assert_allclose(start.pose.pos, (0.4, 0.0, 0.5), atol=1e-6)
    np.testing.assert_allclose(start.pose.rotation() @ (0, 0, -1), (1, 0, 0), atol=1e-6)
    assert start.travel == pytest.approx(0.004)
    tilt = (math.cos(math.radians(10)), math.sin(math.radians(10)), 0.0, 0.0)
    expected = UNITY.device_pose_to(FLU, Pose((-0.15, 0.45, 0.45), tilt))
    np.testing.assert_allclose(interface["screen"].pose.pos, expected.pos, atol=1e-6)
    np.testing.assert_allclose(
        interface["screen"].pose.rotation(), expected.rotation(), atol=1e-6
    )
    assert interface["screen"].regions == {"ok": (0.0, 0.0, 0.5, 0.5)}
    assert interface["volume"].push and interface["volume"].detents == 12

    # Compile it in MuJoCo with a servo fingertip in front of the button.
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    probe = Probe(pos=(0.35, 0.0, 0.5))
    probe.attach(spec)
    interface.attach(spec)
    model = spec.compile()
    cap = model.geom("untitled/start/part0")
    assert cap.type[0] == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(sorted(cap.size), [0.003, 0.0075, 0.0075], atol=1e-6)
    lamp = model.geom("untitled/hazard_lamp/static0")
    assert lamp.type[0] == mujoco.mjtGeom.mjGEOM_SPHERE
    assert lamp.size[0] == pytest.approx(0.01, abs=1e-5)

    # The exported button works: the fingertip presses and releases it.
    data = mujoco.MjData(model)
    runtime = interface.bind(model)
    tip = probe.bind(model)
    face = 0.4 - 0.003  # the cap's outer face, 3 mm in front of its centre
    events = []
    for k in range(60):
        t = k * 0.02
        x = face - 0.008 + 0.006 if 0.2 < t < 0.8 else 0.35
        tip.target(data, (x, 0.0, 0.5))
        runtime.step(data, 10)
        events += [(e.source, e.kind) for e in runtime.read(data).events]
    assert events == [("start", "press"), ("start", "release")]

    out = os.environ.get("EMBODIED_UI_UNITY_EDITOR_EVIDENCE")
    if out:
        summary = {
            "editmode": cases,
            "manifest_name": interface.name,
            "devices": [d.name for d in interface.devices],
            "start_pose_flu": list(start.pose.pos),
            "cap_box_half_sizes_m": sorted(float(s) for s in cap.size),
            "lamp_sphere_radius_m": float(lamp.size[0]),
            "probe_events": [list(e) for e in events],
        }
        Path(out).write_text(json.dumps(summary, indent=2))
        Path(out).with_suffix(".xml").write_bytes(results.read_bytes())

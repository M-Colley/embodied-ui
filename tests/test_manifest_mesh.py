import json
import math

import mujoco
import numpy as np
import pytest

from embodied_ui import (
    Axis,
    Box,
    Contacts,
    Control,
    Detents,
    ForceCurve,
    Interface,
    Interlock,
    Knob,
    Mesh,
    Part,
    Pose,
    PushButton,
    Servo,
    SteeringWheel,
    TouchSurface,
)
from embodied_ui.frames import FLU, UNITY
from embodied_ui.geometry import fit_primitive, load_mesh

trimesh = pytest.importorskip("trimesh")


def full_interface(assets=None):
    return Interface(
        "cockpit",
        (
            PushButton(
                "start",
                Pose((0.1, 0.2, 0.3), (0, 1, 0, 0)),
                toggle=True,
                interlock=Interlock("brake.value", 0.3),
                curve=ForceCurve(((0.0, 0.5), (0.002, 0.7), (0.004, 1.0))),
            ),
            Knob("volume", range=None, push=True),
            SteeringWheel("wheel", servo=Servo(5.0, 0.5)),
            TouchSurface(
                "screen", regions={"ok": (0.0, 0.0, 0.5, 0.5)}, resolution=(32, 16)
            ),
            Control(
                "custom",
                axes=(
                    Axis(
                        "tilt",
                        "hinge",
                        (1, 0, 0),
                        (-0.2, 0.2),
                        detents=Detents(0.1, positions=(-0.2, 0.2)),
                    ),
                ),
                moving=(Part(Box((0.01, 0.01, 0.01)), mass=0.01),),
            ),
        ),
        Contacts(contype=2, conaffinity=2),
        assets,
    )


def test_manifest_round_trip_is_lossless():
    interface = full_interface()
    document = json.loads(json.dumps(interface.to_manifest()))
    rebuilt = Interface.from_manifest(document)
    assert rebuilt == interface


def test_manifest_rejects_unknown_fields_and_versions():
    document = full_interface().to_manifest()
    document["devices"][0]["colour"] = "red"
    with pytest.raises(ValueError):
        Interface.from_manifest(document)
    with pytest.raises(ValueError):
        Interface.from_manifest({**full_interface().to_manifest(), "version": 99})


def test_unity_manifest_poses_are_converted():
    # A Unity object 1 m forward and 0.5 m right, unrotated: its control
    # faces the user standing behind it.
    document = {
        "format": "embodied-ui/manifest",
        "version": 1,
        "frame": "unity",
        "name": "dash",
        "devices": [
            {
                "type": "PushButton",
                "name": "b",
                "pose": {"type": "Pose", "pos": [0.5, 0.0, 1.0], "quat": [1, 0, 0, 0]},
            }
        ],
    }
    interface = Interface.from_manifest(document)
    pose = interface["b"].pose
    np.testing.assert_allclose(pose.pos, (1.0, -0.5, 0.0))
    # Pressing (device -z) pushes forward, away from the user.
    np.testing.assert_allclose(pose.rotation() @ (0, 0, -1), (1, 0, 0), atol=1e-12)
    expected = UNITY.device_pose_to(FLU, Pose((0.5, 0, 1)))
    assert pose == expected


def write_box(path, extents, transform=None):
    mesh = trimesh.creation.box(extents=extents)
    if transform is not None:
        mesh.apply_transform(transform)
    mesh.export(path)
    return mesh


def test_box_fit_recovers_a_rotated_box(tmp_path):
    rotation = trimesh.transformations.rotation_matrix(0.4, (0, 0, 1))
    rotation[:3, 3] = (0.01, 0.02, 0.03)
    write_box(tmp_path / "cap.obj", (0.03, 0.02, 0.01), rotation)
    loaded = load_mesh(Mesh(str(tmp_path / "cap.obj")))
    shape, pose = fit_primitive(loaded.vertices, "box")
    assert sorted(np.asarray(shape.half_size) * 2) == pytest.approx(
        [0.01, 0.02, 0.03], abs=1e-6
    )
    np.testing.assert_allclose(pose.pos, (0.01, 0.02, 0.03), atol=1e-9)
    sphere, centre = fit_primitive(loaded.vertices, "sphere")
    assert sphere.radius == pytest.approx(
        math.sqrt(0.03**2 + 0.02**2 + 0.01**2) / 2, rel=1e-6
    )
    cylinder, _ = fit_primitive(
        trimesh.creation.cylinder(0.01, 0.04, sections=64).vertices, "cylinder"
    )
    assert cylinder.radius == pytest.approx(0.01, rel=0.01)
    assert cylinder.half_length == pytest.approx(0.02, rel=0.01)


def test_unity_mesh_vertices_map_to_device_axes_with_outward_normals(tmp_path):
    # In Unity object space, +z points into the panel: a cap 5 mm thick
    # standing out of the panel spans z in [-0.005, 0].
    transform = np.eye(4)
    transform[2, 3] = -0.0025
    write_box(tmp_path / "cap.obj", (0.02, 0.01, 0.005), transform)
    loaded = load_mesh(Mesh(str(tmp_path / "cap.obj"), frame="unity"))
    assert loaded.vertices[:, 2].min() == pytest.approx(0, abs=1e-9)
    assert loaded.vertices[:, 2].max() == pytest.approx(0.005)
    mesh = trimesh.Trimesh(loaded.vertices, loaded.faces)
    assert mesh.volume > 0  # outward winding kept despite the handedness flip


def test_mesh_parts_compile_with_fitted_colliders_and_touch_sensing(tmp_path):
    write_box(tmp_path / "cap.obj", (0.015, 0.015, 0.006))
    cap = Part(Mesh("cap.obj", fit="box"), Pose((0, 0, 0.003)), mass=0.005)
    interface = Interface("ui", (PushButton("b", cap=cap),), assets=str(tmp_path))
    model = interface.attach(mujoco.MjSpec()).compile()
    collider = model.geom("ui/b/part0")
    visual = model.geom("ui/b/part0/visual")
    assert collider.type[0] == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(
        sorted(collider.size), sorted((0.0075, 0.0075, 0.003)), atol=1e-6
    )
    assert model.geom_contype[visual.id] == 0 and model.nmesh == 1
    assert model.sensor("ui/b/touch0").type[0] == mujoco.mjtSensor.mjSENS_TOUCH


def test_manifest_file_resolves_assets_relative_to_itself(tmp_path):
    write_box(tmp_path / "cap.obj", (0.015, 0.015, 0.006))
    cap = Part(Mesh("cap.obj", fit="box"), mass=0.005)
    Interface("ui", (PushButton("b", cap=cap),)).save_manifest(tmp_path / "ui.json")
    interface = Interface.from_manifest(tmp_path / "ui.json")
    assert interface.assets == str(tmp_path)
    interface.attach(mujoco.MjSpec()).compile()

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from embodied_ui.frames import (
    CARLA,
    FLU,
    FRAMES,
    GLTF,
    MYOSUITE_SEATED,
    UNITY,
    UNREAL,
    Frame,
    Pose,
    frame,
    quat_xyzw,
    rotation_from_unreal,
    unit_quat,
)


def wxyz(rotation: Rotation) -> tuple[float, ...]:
    x, y, z, w = rotation.as_quat()
    return (w, x, y, z)


def test_pose_composition_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        ra, rb = Rotation.random(random_state=rng), Rotation.random(random_state=rng)
        pa, pb = rng.normal(size=3), rng.normal(size=3)
        a, b = Pose(tuple(pa), wxyz(ra)), Pose(tuple(pb), wxyz(rb))
        c = a * b
        np.testing.assert_allclose(c.rotation(), (ra * rb).as_matrix(), atol=1e-12)
        np.testing.assert_allclose(c.pos, ra.apply(pb) + pa, atol=1e-12)
        identity = a * a.inverse()
        np.testing.assert_allclose(identity.pos, 0, atol=1e-12)
        np.testing.assert_allclose(identity.rotation(), np.eye(3), atol=1e-12)


def test_quaternions_are_validated_and_renormalized():
    assert unit_quat((1.0005, 0, 0, 0)) == (1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        unit_quat((1.1, 0, 0, 0))
    assert quat_xyzw((0, 0, 0, 1)) == (1.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_round_trips_are_exact(name):
    rng = np.random.default_rng(1)
    source = FRAMES[name]
    for target in FRAMES.values():
        rotation = Rotation.random(random_state=rng)
        pose = Pose(tuple(rng.normal(size=3)), wxyz(rotation))
        back = target.pose_to(source, source.pose_to(target, pose))
        np.testing.assert_allclose(back.pos, pose.pos, atol=1e-9)
        np.testing.assert_allclose(back.rotation(), pose.rotation(), atol=1e-9)
        assert np.linalg.det(source.pose_to(target, pose).rotation()) == pytest.approx(
            1
        )


def test_physical_directions_agree_across_engines():
    # Unity forward and right, Unreal centimetres, glTF front.
    np.testing.assert_allclose(UNITY.point_to(FLU, (0, 0, 1)), (1, 0, 0))
    np.testing.assert_allclose(UNITY.point_to(FLU, (1, 0, 0)), (0, -1, 0))
    np.testing.assert_allclose(UNREAL.point_to(FLU, (100, 200, 300)), (1, -2, 3))
    np.testing.assert_allclose(GLTF.point_to(UNITY, (0, 0, 1)), (0, 0, 1))
    np.testing.assert_allclose(MYOSUITE_SEATED.point_to(FLU, (0, 1, 0)), (-1, 0, 0))
    assert UNITY.right_handed is False and UNREAL.right_handed is False
    assert FLU.right_handed and GLTF.right_handed


def test_unity_left_turn_is_a_positive_flu_yaw():
    # Unity's Quaternion.Euler(0, -90, 0) turns forward to the left.
    unity = Pose(
        (0, 0, 0), quat_xyzw((0, -math.sin(math.pi / 4), 0, math.cos(math.pi / 4)))
    )
    flu = UNITY.pose_to(FLU, unity)
    np.testing.assert_allclose(flu.rotation() @ (1, 0, 0), (0, 1, 0), atol=1e-12)


def test_unreal_rotator_matches_carla_axes():
    # Yaw 90 turns forward to the right in Unreal and CARLA.
    np.testing.assert_allclose(
        rotation_from_unreal((0, 90, 0)) @ (1, 0, 0), (0, 1, 0), atol=1e-12
    )
    np.testing.assert_allclose(
        rotation_from_unreal((90, 0, 0)) @ (1, 0, 0), (0, 0, 1), atol=1e-12
    )
    pose = Pose.from_matrix(rotation_from_unreal((0, 90, 0)), (1, 2, 3))
    flu = CARLA.pose_to(FLU, pose)
    np.testing.assert_allclose(flu.pos, (1, -2, 3))
    np.testing.assert_allclose(flu.rotation() @ (1, 0, 0), (0, -1, 0), atol=1e-12)


def test_pseudovectors_flip_with_handedness():
    # A left yaw rate in Unity is negative about +y; in FLU positive about +z.
    np.testing.assert_allclose(UNITY.pseudovector_to(FLU, (0, -1, 0)), (0, 0, 1))
    np.testing.assert_allclose(
        FLU.pseudovector_to(MYOSUITE_SEATED, (0, 0, 1)), (0, 0, 1)
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Device right, up and towards-the-user axes for an unrotated object
        # whose engine "forward" points away from the user.
        (UNITY, [(0, -1, 0), (0, 0, 1), (-1, 0, 0)]),
        (UNREAL, [(0, -1, 0), (0, 0, 1), (-1, 0, 0)]),
        (FLU, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
    ],
)
def test_device_axes(source, expected):
    pose = source.device_pose_to(FLU, Pose())
    np.testing.assert_allclose(
        pose.rotation().T, np.array(expected, dtype=float), atol=1e-12
    )
    assert np.linalg.det(pose.rotation()) == pytest.approx(1)


def test_frame_validation():
    with pytest.raises(ValueError):
        Frame("forward", "forward", "up")
    with pytest.raises(ValueError):
        Frame("forward", "left", "up", device=("+x", "+x", "+z"))
    with pytest.raises(ValueError):
        frame("nope")
    assert frame("unity") is UNITY


def test_unreal_rotator_inverse():
    from embodied_ui.frames import rotation_to_unreal

    rng = np.random.default_rng(3)
    for _ in range(50):
        angles = (rng.uniform(-80, 80), rng.uniform(-179, 179), rng.uniform(-179, 179))
        np.testing.assert_allclose(
            rotation_to_unreal(rotation_from_unreal(angles)), angles, atol=1e-9
        )

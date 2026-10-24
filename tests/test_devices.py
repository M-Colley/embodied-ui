"""Device mechanics and logic, operated through contact by a servo fingertip."""

import math

import mujoco
import numpy as np
import pytest

from embodied_ui import (
    Contacts,
    Indicator,
    Interface,
    Interlock,
    Joystick,
    Knob,
    Lever,
    Pedal,
    Pose,
    Prop,
    PushButton,
    Screen,
    Servo,
    Slider,
    SteeringWheel,
    ToggleSwitch,
    TouchSurface,
)
from embodied_ui.probe import Probe

from .conftest import build_scene


def events(readings, kind=None):
    return [
        (e.source, e.kind, e.value)
        for r in readings
        for e in r.events
        if kind in (None, e.kind)
    ]


def press_path(x, y, depth, top, until):
    """Hover, press to ``depth`` below ``top`` until ``until`` seconds, lift."""
    return lambda t: ((x, y, top + 0.02 if t < 0.1 or t > until else top - depth),)


def test_push_button_spring_matches_its_force_specification():
    scene = build_scene([PushButton("b", travel=0.004, preload=1.5, force=3.0)])
    for travel, expected in ((0.0, 1.5), (0.002, 2.25), (0.004, 3.0)):
        scene.set_qpos("b", "travel", travel)
        dof = scene.model.joint("ui/b/travel").dofadr[0]
        assert -scene.data.qfrc_passive[dof] == pytest.approx(expected, rel=1e-9)


def test_push_button_press_and_release_events():
    scene = build_scene([PushButton("b")], probe_pos=(0, 0, 0.04))
    readings = scene.run(0.8, press_path(0, 0, 0.008, 0.006 + 0.008, 0.5))
    assert events(readings) == [("b", "press", 1.0), ("b", "release", 0.0)]
    pressed = [r for r in readings if r["b.pressed"]]
    assert all(r["b.contact"] > 2.0 for r in pressed[2:])
    assert readings[-1]["b.value"] == pytest.approx(0, abs=0.01)


def test_toggle_button_latches_on_alternate_presses():
    scene = build_scene([PushButton("b", toggle=True)], probe_pos=(0, 0, 0.04))
    top = 0.014

    def twice(t):
        down = 0.1 < t < 0.3 or 0.5 < t < 0.7
        return ((0, 0, top - 0.008 if down else top + 0.02),)

    readings = scene.run(0.9, twice)
    assert [e for e in events(readings, "toggle")] == [
        ("b", "toggle", 1.0),
        ("b", "toggle", 0.0),
    ]
    assert readings[20]["b.state"] == 1.0 and readings[-1]["b.state"] == 0.0


def test_hold_time_rejects_brief_contacts():
    scene = build_scene([PushButton("b", hold=0.3)], probe_pos=(0, 0, 0.04))
    readings = scene.run(0.6, press_path(0, 0, 0.008, 0.014, 0.25))
    assert events(readings) == []
    scene.runtime.reset()
    readings = scene.run(1.0, press_path(0, 0, 0.008, 0.014, 0.7))
    presses = [r.time for r in readings for e in r.events if e.kind == "press"]
    assert len(presses) == 1 and presses[0] > 0.35


def test_interlock_gates_presses_on_another_channel():
    devices = [
        Slider("brake", Pose((0.1, 0, 0))),
        PushButton("reverse", interlock=Interlock("brake.value", 0.5)),
    ]
    scene = build_scene(devices, probe_pos=(0, 0, 0.04))
    readings = scene.run(0.6, press_path(0, 0, 0.008, 0.014, 0.4))
    assert events(readings, "press") == []
    scene.set_qpos("brake", "position", 0.05)
    scene.runtime.reset()
    readings = scene.run(0.6, press_path(0, 0, 0.008, 0.014, 0.4))
    assert events(readings, "press") == [("reverse", "press", 1.0)]


@pytest.mark.parametrize("degree", [True, False])
def test_toggle_switch_is_bistable_in_either_angle_unit(degree):
    scene = build_scene([ToggleSwitch("t", angle=0.35)], degree=degree)
    for start, expected in ((0.05, 0.35), (-0.05, -0.35), (0.3, 0.35)):
        mujoco.mj_resetData(scene.model, scene.data)
        scene.set_qpos("t", "angle", start)
        scene.runtime.reset()
        readings = scene.run(0.5)
        assert scene.qpos("t", "angle") == pytest.approx(expected, abs=0.01)
        assert readings[-1]["t.state"] == float(expected > 0)


def test_toggle_switch_flips_when_pushed():
    # Positive angles tilt the bat up (+y); push it up from below.
    scene = build_scene([ToggleSwitch("t", angle=0.35)], probe_pos=(0, -0.03, 0.012))
    scene.set_qpos("t", "angle", -0.35)
    scene.runtime.reset()
    readings = scene.run(0.6, lambda t: ((0, -0.03 + 0.06 * min(1, t / 0.4), 0.012),))
    assert events(readings, "toggle") == [("t", "toggle", 1.0)]
    assert scene.qpos("t", "angle") == pytest.approx(0.35, abs=0.02)


def test_knob_settles_into_detents_and_counts_them():
    scene = build_scene([Knob("k", detents=24)])
    period = 2 * math.pi / 24
    for fraction, expected in ((0.3, 0), (0.45, 0), (0.55, 1), (-0.7, -1)):
        mujoco.mj_resetData(scene.model, scene.data)
        scene.set_qpos("k", "angle", fraction * period)
        scene.runtime.reset()
        readings = scene.run(0.4)
        assert scene.qpos("k", "angle") == pytest.approx(expected * period, abs=0.01)
        assert readings[-1]["k.detent"] == expected


def test_knob_turned_by_fingertip_twist_reports_detent_steps():
    scene = build_scene([Knob("k", detents=24)], probe_pos=(0, 0, 0.03))
    top = 0.018 + 0.008

    def turn(t):
        twist = 0.0 if t < 0.2 else -min(1.0, (t - 0.2) / 0.5) * 1.0
        return ((0, 0, top - 0.002), twist)

    readings = scene.run(0.9, turn)
    steps = sum(
        e.data["delta"] for r in readings for e in r.events if e.kind == "detent"
    )
    # Clockwise (as seen by the user) is positive; one radian is 3.8 detents.
    assert steps == readings[-1]["k.detent"]
    assert 3 <= steps <= 4


def test_slider_and_lever_rest_in_their_nearest_gate():
    scene = build_scene(
        [
            Slider("s", positions=5, friction=0.0),
            Lever("l", Pose((0.2, 0, 0)), positions=(-0.3, 0.0, 0.3)),
        ]
    )
    scene.set_qpos("s", "position", 0.0175)
    scene.set_qpos("l", "angle", 0.2)
    scene.runtime.reset()
    readings = scene.run(0.6)
    assert scene.qpos("s", "position") == pytest.approx(0.015, abs=0.001)
    assert readings[-1]["s.detent"] == 1
    assert scene.qpos("l", "angle") == pytest.approx(0.3, abs=0.01)
    assert readings[-1]["l.position"] == 2


def test_pedal_spring_and_normalized_value():
    scene = build_scene([Pedal("p", travel=0.06, preload=10, force=40)])
    scene.set_qpos("p", "travel", 0.03)
    dof = scene.model.joint("ui/p/travel").dofadr[0]
    assert -scene.data.qfrc_passive[dof] == pytest.approx(25)
    assert scene.runtime.read(scene.data)["p.value"] == pytest.approx(0.5)


def test_steering_wheel_torque_and_servo_outputs():
    scene = build_scene([SteeringWheel("w", servo=Servo(kp=20, kd=2))])
    scene.runtime.set_outputs(scene.data, {"w.torque": 0.5})
    scene.run(0.5)
    # Equilibrium with the centring spring (0.2 N m/rad) beyond friction.
    assert 0.5 < scene.qpos("w", "angle") < 2.5
    scene.runtime.set_outputs(scene.data, {"w.torque": 0.0, "w.setpoint": -1.0})
    readings = scene.run(1.5)
    assert scene.qpos("w", "angle") == pytest.approx(-1.0, abs=0.05)
    assert readings[-1]["w.value"] == pytest.approx(-1.0 / 7.85, abs=0.01)


def test_steering_rim_is_a_closed_circle():
    wheel = SteeringWheel("w", radius=0.18, segments=24)
    rim = wheel.mechanism().moving[:24]
    for part in rim:
        centre = np.asarray(part.pose.pos)
        tangent = part.pose.rotation()[:, 2]
        assert np.dot(centre, tangent) == pytest.approx(0, abs=1e-12)
        assert np.linalg.norm(centre) == pytest.approx(0.18 * math.cos(math.pi / 24))


def test_joystick_recentres():
    scene = build_scene([Joystick("j")])
    scene.set_qpos("j", "x", 0.2)
    scene.set_qpos("j", "y", -0.2)
    readings = scene.run(1.0)
    assert (
        abs(readings[-1]["j.x_value"]) < 0.05 and abs(readings[-1]["j.y_value"]) < 0.05
    )


def test_touch_surface_reports_position_regions_and_identities():
    regions = {"left": (0.0, 0.0, 0.5, 1.0), "right": (0.5, 0.0, 1.0, 1.0)}
    scene = build_scene(
        [TouchSurface("s", size=(0.2, 0.12), regions=regions)],
        probe_pos=(0.05, 0.03, 0.03),
    )

    def tap(x, y):
        return lambda t: ((x, y, 0.007 if 0.05 < t < 0.3 else 0.03),)

    readings = scene.run(0.4, tap(0.05, 0.03))
    touching = [r for r in readings if r["s.touches"] == 1]
    assert touching
    assert touching[-1]["s.u"] == pytest.approx(0.75, abs=0.01)
    assert touching[-1]["s.v"] == pytest.approx(0.25, abs=0.01)
    assert [e[1:] for e in events(readings) if e[1] != "touch_move"] == [
        ("touch_begin", 1.0),
        ("press", 1.0),
        ("touch_end", 1.0),
        ("release", 0.0),
    ] or [e[1:] for e in events(readings) if e[1] != "touch_move"] == [
        ("touch_begin", 1.0),
        ("press", 1.0),
        ("release", 0.0),
        ("touch_end", 1.0),
    ]
    assert ("s.right", "press", 1.0) in events(readings)
    readings = scene.run(0.4, tap(-0.05, -0.03))
    begins = [e for e in events(readings) if e[1] == "touch_begin"]
    assert begins == [("s", "touch_begin", 2.0)]
    assert ("s.left", "press", 1.0) in events(readings)
    assert math.isnan(readings[-1]["s.u"])


def test_indicator_display_and_prop_outputs():
    scene = build_scene(
        [
            Indicator("lamp", Pose((0.2, 0, 0))),
            Screen("screen", Pose((0, 0.2, 0)), resolution=(8, 4)),
            Prop("cup", Pose((0, -0.3, 0.1))),
        ]
    )
    geom = scene.model.geom("ui/lamp/static0").id
    scene.runtime.set_outputs(scene.data, {"lamp.on": 1})
    np.testing.assert_allclose(scene.model.geom_rgba[geom], Indicator("x").on)
    scene.runtime.set_outputs(scene.data, {"lamp.on": 0})
    np.testing.assert_allclose(scene.model.geom_rgba[geom], Indicator("x").off)

    image = np.zeros((4, 8, 3), np.uint8)
    image[0, :, 0] = 255  # top row red
    scene.runtime.set_outputs(scene.data, {"screen.image": image})
    texture = scene.model.texture("ui/screen/display")
    data = scene.model.tex_data[texture.adr[0] : texture.adr[0] + 8 * 4 * 3].reshape(
        4, 8, 3
    )
    assert (data[-1, :, 0] == 255).all() and (data[0, :, 0] == 0).all()
    np.testing.assert_array_equal(scene.runtime.image("screen"), image)
    with pytest.raises(ValueError):
        scene.runtime.set_outputs(
            scene.data, {"screen.image": np.zeros((5, 8, 3), np.uint8)}
        )

    scene.runtime.set_outputs(scene.data, {"cup.pose": [0.1, 0.2, 0.3, 0, 0, 0, 1]})
    mujoco.mj_forward(scene.model, scene.data)
    np.testing.assert_allclose(
        scene.data.xpos[scene.model.body("ui/cup").id], (0.1, 0.2, 0.3)
    )
    assert scene.runtime.outputs == (
        "lamp.on",
        "screen.image",
        "cup.pose",
    )


def test_devices_follow_a_moving_mount_and_ignore_it_in_contacts():
    import mujoco as mj

    spec = mj.MjSpec()
    spec.compiler.degree = False
    arm = spec.worldbody.add_body(name="dash", pos=(0, 0, 0.5))
    arm.add_joint(name="dash_slide", type=mj.mjtJoint.mjJNT_SLIDE, axis=(1, 0, 0))
    arm.add_geom(
        name="dash_geom", type=mj.mjtGeom.mjGEOM_BOX, size=(0.1, 0.1, 0.01), mass=1
    )
    interface = Interface("ui", (PushButton("b", Pose((0, 0, 0.0)), mount="dash"),))
    interface.attach(spec)
    model = spec.compile()
    data = mj.MjData(model)
    data.qpos[0] = 0.3
    mj.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[model.body("ui/b").id], (0.3, 0, 0.5))
    excluded = {
        (int(a), int(b))
        for a, b in zip(model.exclude_signature >> 16, model.exclude_signature & 0xFFFF)
    }
    assert excluded


def test_explicit_contact_pairs():
    spec = mujoco.MjSpec()
    Probe(pos=(0, 0, 0.05)).attach(spec)
    interface = Interface("ui", (PushButton("b"),), Contacts(pairs=("probe",)))
    interface.attach(spec)
    model = spec.compile()
    cap = model.geom("ui/b/part0").id
    assert model.geom_contype[cap] == 0 and model.geom_conaffinity[cap] == 0
    assert model.npair == 1
    with pytest.raises(ValueError):
        Interface("ui", (PushButton("b"),), Contacts(pairs=("nothing*",))).attach(
            mujoco.MjSpec()
        )


def test_attaching_twice_and_bad_mounts_are_rejected():
    interface = Interface("ui", (PushButton("b"),))
    spec = interface.attach(mujoco.MjSpec())
    with pytest.raises(ValueError):
        interface.attach(spec)
    with pytest.raises(KeyError):
        Interface("ui", (PushButton("b", mount="nope"),)).attach(mujoco.MjSpec())
    with pytest.raises(ValueError):
        Prop("p", mount="dash")
    with pytest.raises(ValueError):
        Interface("ui", (PushButton("b"), PushButton("b")))


def test_inertia_does_not_depend_on_host_compiler_settings():
    masses = []
    for inertiafromgeom in (
        mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO,
        mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE,
    ):
        spec = mujoco.MjSpec()
        spec.compiler.inertiafromgeom = inertiafromgeom
        Interface("ui", (PushButton("b"), Joystick("j", Pose((0.1, 0, 0))))).attach(
            spec
        )
        model = spec.compile()
        masses.append(model.body_mass.copy())
    np.testing.assert_allclose(masses[0], masses[1])


def test_channel_and_output_listing_matches_readings():
    scene = build_scene(
        [
            PushButton("b"),
            Knob("k", Pose((0.1, 0, 0)), push=True),
            TouchSurface("s", Pose((0, 0.2, 0)), regions={"ok": (0, 0, 1, 1)}),
        ]
    )
    reading = scene.runtime.read(scene.data)
    assert set(reading.channels) == set(scene.runtime.channels)
    assert "k.pressed" in reading.channels and "s.ok.pressed" in reading.channels


def test_key_preset_follows_iso_9241_410():
    key = PushButton.key("k")
    scene = build_scene([key])
    scene.set_qpos("k", "travel", key.make * key.travel)
    dof = scene.model.joint("ui/k/travel").dofadr[0]
    assert -scene.data.qfrc_passive[dof] == pytest.approx(0.65)
    assert 0.0015 <= key.travel <= 0.006 and 0.25 <= key.preload / 0.65 <= 0.75


def test_default_panel_button_actuates_within_mil_std_1472():
    button = PushButton("b")
    make_force = button.preload + (button.force - button.preload) * button.make
    assert 2.8 <= make_force <= 11.0 and 0.002 <= button.travel <= 0.006

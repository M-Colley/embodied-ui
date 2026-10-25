"""A MyoSuite hand presses an embodied-ui button through contact.

The hand is myo_sim's ``myohand``: MyoSuite's right forearm and hand, 23
joints and 39 muscles, loaded as an editable ``MjSpec``. A default
``PushButton`` is composed into the same spec with ``Interface.attach``.
Its explicit contact pairs are limited to the index distal phalanx.

The muscles follow a fixed open-loop activation pattern; no policy is used:

* **Posture.** The wrist flexors FCR and FCU hold 5 % activation
  throughout. This braces the wrist against its flexion stop (45 degrees in
  this model). The hand first relaxes for 1 s in this posture.
* **Press.** The index finger's superficial flexor FDS2 ramps from 0 to
  30 % over 0.25 s, holds for 0.5 s, returns to 0 over 0.1 s and stays off.

The button faces the relaxed index fingertip, 2 mm from it, pressed along
the direction the fingertip moves when the index MCP joint flexes.

Without the wrist hold, the wrist yields to the button's reaction and the
fingertip slides off the cap. In a pronated forearm it also slides off.
VALIDATION.md (2026-10-25) records these runs.

The test skips when myo_sim (installed with MyoSuite) is missing.
"""

import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from embodied_ui import Contacts, Interface, Pose, PushButton

myo_sim = pytest.importorskip("myo_sim", reason="needs MyoSuite's myo_sim models")
pytestmark = pytest.mark.filterwarnings("ignore:Attaching a spec with an empty prefix")

DISTAL = ("distph2_coll_r", "distph2_coll_2_r")  # index distal phalanx colliders
HOLD = {"FCR_r": 0.05, "FCU_r": 0.05}
PRESS = "FDS2_r"
LEVEL = 0.3
RAMP, PLATEAU, FALL = 0.25, 0.5, 0.1
RELAX = 1.0
GAP = 0.002
PERIOD = 0.02


def activation(t, level):
    """Ramp to ``level``, hold, return to zero, stay off."""
    if t < RAMP:
        return level * t / RAMP
    if t < RAMP + PLATEAU:
        return level
    if t < RAMP + PLATEAU + FALL:
        return level * (RAMP + PLATEAU + FALL - t) / FALL
    return 0.0


def controls(model, level):
    ctrl = np.zeros(model.nu)
    for name, value in HOLD.items():
        ctrl[model.actuator(name).id] = value
    ctrl[model.actuator(PRESS).id] = level
    return ctrl


def transfer(source_model, source, model, data):
    """Copy the hand's state by joint and actuator name into a larger model."""
    for j in range(source_model.njnt):
        joint = model.joint(source_model.joint(j).name)
        data.qpos[joint.qposadr[0]] = source.qpos[source_model.jnt_qposadr[j]]
        data.qvel[joint.dofadr[0]] = source.qvel[source_model.jnt_dofadr[j]]
    for a in range(source_model.nu):
        actuator = model.actuator(source_model.actuator(a).name).id
        data.act[model.actuator_actadr[actuator]] = source.act[
            source_model.actuator_actadr[a]
        ]
    mujoco.mj_forward(model, data)


def compose(pose):
    panel = Interface(
        "panel", (PushButton("key", pose),), Contacts(pairs=("distph2_r",))
    )
    spec = myo_sim.load_spec("myohand")
    panel.attach(spec)
    model = spec.compile()
    return panel, model


def clearance(model, data):
    cap = model.geom("panel/key/part0").id
    fromto = np.zeros(6)
    return min(
        mujoco.mj_geomDistance(model, data, model.geom(g).id, cap, 0.1, fromto)
        for g in DISTAL
    )


@pytest.fixture(scope="module")
def hand_and_button():
    # Relax the hand alone with the wrist hold.
    hand = myo_sim.load_spec("myohand").compile()
    rest = mujoco.MjData(hand)
    while rest.time < RELAX - 1e-9:
        rest.ctrl[:] = controls(hand, 0.0)
        mujoco.mj_step(hand, rest)
    pad = hand.geom("distph2_coll_2_r").id
    tip = rest.geom_xpos[pad].copy()
    jac = np.zeros((3, hand.nv))
    mujoco.mj_jac(hand, rest, jac, None, tip, hand.geom_bodyid[pad])
    normal = jac[:, hand.joint("mcp2_flexion_r").dofadr[0]]
    normal /= np.linalg.norm(normal)
    # Device frame: -z (the press direction) along the fingertip's motion.
    z = -normal
    x = np.cross((0.0, 0.0, 1.0), z)
    x /= np.linalg.norm(x)
    rotation = np.column_stack([x, np.cross(z, x), z])
    # Place the cap 2 mm from the closest distal-phalanx collider.
    far = tip + 0.03 * normal
    _, model = compose(Pose.from_matrix(rotation, far))
    data = mujoco.MjData(model)
    transfer(hand, rest, model, data)
    origin = far - normal * (clearance(model, data) - GAP)
    panel, model = compose(Pose.from_matrix(rotation, origin))
    return hand, rest, panel, model


def run(hand_and_button, level, seconds=2.0):
    hand, rest, panel, model = hand_and_button
    data = mujoco.MjData(model)
    transfer(hand, rest, model, data)
    assert clearance(model, data) == pytest.approx(GAP, abs=1e-6)
    runtime = panel.bind(model)
    substeps = round(PERIOD / model.opt.timestep)
    cap = model.geom("panel/key/part0").id
    distal = {model.geom(g).id for g in DISTAL}
    mcp = model.joint("mcp2_flexion_r").qposadr[0]
    wrench = np.zeros(6)
    rows = []
    while data.time < seconds - 1e-9:
        data.ctrl[:] = controls(model, activation(data.time, level))
        runtime.step(data, substeps)
        reading = runtime.read(data)
        # Normal force of the fingertip contacts from MuJoCo's solver.
        solver = 0.0
        for i in range(data.ncon):
            pair = {int(data.contact.geom1[i]), int(data.contact.geom2[i])}
            if cap in pair and pair & distal:
                mujoco.mj_contactForce(model, data, i, wrench)
                solver += wrench[0]
        rows.append((reading, solver, float(data.qpos[mcp])))
    return rows


def test_an_open_loop_muscle_pattern_presses_and_releases_the_button(
    hand_and_button,
):
    rows = run(hand_and_button, LEVEL)
    readings = [r for r, _, _ in rows]
    events = [(e.kind, e.time) for r in readings for e in r.events]
    assert [kind for kind, _ in events] == ["press", "release"], events
    pressed_at, released_at = events[0][1], events[1][1]
    by_time = {round(r.time, 6): r for r in readings}
    # The press happens while FDS2 is active; the release follows its
    # switch-off, not a slip of the fingertip off the cap.
    assert 0.1 < pressed_at < RAMP + PLATEAU
    assert RAMP + PLATEAU < released_at < RAMP + PLATEAU + FALL + 0.3
    assert max(mcp for _, _, mcp in rows) < 0.2
    # Contact forces: the default button needs 4.5 N at its make point
    # (half travel) and 4.05 N at its release point (35 % travel), and
    # bottoms out at 6 N.
    at_press = by_time[round(pressed_at, 6)]["key.contact"]
    at_release = by_time[round(released_at, 6)]["key.contact"]
    peak = max(r["key.contact"] for r in readings)
    assert 4.0 < at_press < 5.5
    assert 3.0 < at_release < 4.6
    assert max(r["key.value"] for r in readings) == pytest.approx(1.0, abs=0.01)
    assert 6.0 < peak < 10.0
    # The button's touch sensor reads the solver's contact force.
    for reading, solver, _ in rows:
        assert reading["key.contact"] == pytest.approx(solver, rel=1e-6, abs=1e-9)
    # After the release the cap is back at rest; the relaxed finger may lean
    # on it, but with less than the 3 N preload.
    late = [r for r in readings if r.time > 1.3]
    assert all(r["key.value"] < 0.02 and r["key.pressed"] == 0 for r in late)
    assert max(r["key.contact"] for r in late) < 3.0

    out = os.environ.get("EMBODIED_UI_HAND_EVIDENCE")
    if out:
        _, rest, _, model = hand_and_button
        summary = {
            "model": "myo_sim myohand",
            "myo_sim": getattr(myo_sim, "__version__", ""),
            "mujoco": mujoco.__version__,
            "pattern": {
                "hold": HOLD,
                "press": PRESS,
                "level": LEVEL,
                "ramp_s": RAMP,
                "plateau_s": PLATEAU,
                "fall_s": FALL,
                "relax_s": RELAX,
            },
            "gap_m": GAP,
            "events": [{"kind": k, "time_s": round(t, 3)} for k, t in events],
            "contact_at_press_N": at_press,
            "contact_at_release_N": at_release,
            "peak_contact_N": peak,
            "max_mcp_flexion_rad": max(mcp for _, _, mcp in rows),
            "max_contact_after_release_N": max(r["key.contact"] for r in late),
            "trace": [
                {
                    "time_s": round(r.time, 3),
                    "activation": round(activation(r.time - PERIOD, LEVEL), 4),
                    "value": round(r["key.value"], 4),
                    "contact_N": round(r["key.contact"], 4),
                    "pressed": r["key.pressed"],
                }
                for r in readings
            ],
        }
        Path(out).write_text(json.dumps(summary, indent=2))


def test_without_the_flexor_the_button_stays_up(hand_and_button):
    # Control: the same posture and placement with FDS2 off.
    rows = run(hand_and_button, 0.0)
    readings = [r for r, _, _ in rows]
    assert not [e for r in readings for e in r.events]
    assert max(r["key.value"] for r in readings) < 0.02
    assert max(r["key.contact"] for r in readings) < 3.0

"""Auditory and tactile output devices, and interaction traces."""

import csv
import json
import math

import numpy as np
import pytest

from embodied_ui import (
    Indicator,
    Pose,
    PushButton,
    Speaker,
    TraceRecorder,
    Vibrotactor,
)
from embodied_ui.backends import LoopbackEngine
from embodied_ui.engine import Bridge

from .conftest import build_scene

DT = 0.02


def test_speaker_level_falls_6_db_per_doubling_of_distance():
    scene = build_scene([Speaker("chime", Pose((0.5, 0, 0)))])
    assert scene.runtime.outputs == ("chime.level", "chime.frequency")
    assert scene.runtime.sound_level(scene.data, "chime", (0, 0, 0)) == -math.inf
    scene.runtime.set_outputs(
        scene.data, {"chime.level": 70.0, "chime.frequency": 1000}
    )
    assert scene.runtime.signals("chime") == {"level": 70.0, "frequency": 1000.0}
    at_1m = scene.runtime.sound_level(scene.data, "chime", (1.5, 0, 0))
    at_2m = scene.runtime.sound_level(scene.data, "chime", (2.5, 0, 0))
    assert at_1m == pytest.approx(70.0)
    assert at_1m - at_2m == pytest.approx(20 * math.log10(2))
    np.testing.assert_allclose(scene.runtime.position(scene.data, "chime"), (0.5, 0, 0))
    scene.runtime.reset()
    assert scene.runtime.signals("chime") == {}


def test_vibration_reaches_only_a_touching_body():
    scene = build_scene([Vibrotactor("seat")], probe_pos=(0, 0, 0.03))
    scene.runtime.set_outputs(
        scene.data, {"seat.amplitude": 1.5, "seat.frequency": 250}
    )
    readings = scene.run(0.1, lambda t: ((0, 0, 0.03),))
    assert readings[-1]["seat.touches"] == 0 and scene.runtime.vibration("seat") == 0.0
    readings = scene.run(0.3, lambda t: ((0, 0, 0.004),))
    assert readings[-1]["seat.touches"] == 1
    assert scene.runtime.vibration("seat") == 1.5
    with pytest.raises(KeyError):
        scene.runtime.vibration("wheel")


def test_lamp_state_is_kept_for_perception():
    scene = build_scene([Indicator("lamp")])
    scene.runtime.set_outputs(scene.data, {"lamp.on": True, "lamp.level": 3.0})
    assert scene.runtime.signals("lamp") == {"on": 1.0}


def test_trace_recorder_writes_bridge_frames(tmp_path):
    scene = build_scene([PushButton("b"), Speaker("chime", Pose((0.2, 0, 0)))])

    def update(command, time):
        return {
            "chime.level": 60.0 if time > 0.05 else 0.0,
            "vehicle.pose": [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "camera": np.zeros((4, 4, 3), np.uint8),
            "label": "text",
        }

    recorder = TraceRecorder()
    with Bridge(scene.runtime, LoopbackEngine(DT, update)) as bridge:
        bridge.reset(scene.data)
        for _ in range(5):
            scene.runtime.step(scene.data, round(DT / scene.model.opt.timestep))
            recorder.record_bridge(bridge.step(scene.data))
    assert len(recorder.frames) == 5
    assert set(recorder.frames[-1]["outputs"]) == {"chime.level", "vehicle.pose"}

    rows = list(csv.DictReader(recorder.write_csv(tmp_path / "t.csv").open(newline="")))
    assert len(rows) == 5 and float(rows[-1]["chime.level"]) == 60.0
    assert rows[0]["vehicle.pose[1]"] == "2.0" and "b.value" in rows[0]
    lines = recorder.write_jsonl(tmp_path / "t.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    assert first["time"] == pytest.approx(DT) and "b.pressed" in first["channels"]
    assert recorder.events() == []


def test_trace_recorder_filters_outputs_and_keeps_events():
    scene = build_scene([PushButton("b")], probe_pos=(0, 0, 0.04))
    recorder = TraceRecorder(outputs=("b.force",))
    readings = scene.run(
        0.8, lambda t: ((0, 0, 0.034 if t < 0.1 or t > 0.5 else 0.006),)
    )
    for reading in readings:
        recorder.record(reading, {"b.force": 0.5, "other": 1.0})
    kinds = [e.kind for e in recorder.events()]
    assert kinds == ["press", "release"]
    assert recorder.columns()[-1] == "b.force"

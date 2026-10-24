import math

import numpy as np
import pytest

from embodied_ui.logic import (
    DetentCounter,
    Event,
    Interlock,
    Threshold,
    Touch,
    TouchTracker,
)
from embodied_ui.mechanics import Axis, Detents, ForceCurve, Servo


def test_periodic_detents_restore_towards_the_nearest_position():
    detents = Detents(0.01, period=0.5, offset=0.1)
    for k in range(-3, 4):
        p = 0.1 + 0.5 * k
        assert detents.force(p) == pytest.approx(0, abs=1e-12)
        assert detents.force(p + 0.25) == pytest.approx(0, abs=1e-12)  # boundary
        assert detents.force(p + 0.1) < 0 < detents.force(p - 0.1)
        assert detents.force(p + 0.125) == pytest.approx(-0.01)
        assert detents.index(p + 0.2) == k and detents.position(k) == pytest.approx(p)


def test_explicit_detents_and_outer_cells():
    detents = Detents(1.0, positions=(0.3, -0.3, 0.0))
    assert detents.positions == (-0.3, 0.0, 0.3)
    assert detents.force(0.15) == pytest.approx(0, abs=1e-12)
    assert detents.force(0.35) < 0  # beyond the outermost well
    assert detents.force(-0.35) > 0
    with pytest.raises(ValueError):
        Detents(1.0)
    with pytest.raises(ValueError):
        Detents(1.0, positions=(0.0,))
    with pytest.raises(ValueError):
        Detents(1.0, positions=(0.0, 1.0), period=0.5)


def test_force_curve_and_servo():
    curve = ForceCurve(((0.0, 0.5), (0.002, 0.6), (0.003, 0.3), (0.004, 1.0)))
    assert curve.force(0.0025) == pytest.approx(-0.45)
    assert curve.force(1.0) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        ForceCurve(((0.0, 0.0), (0.0, 1.0)))
    servo = Servo(kp=10, kd=1, max_force=2)
    assert servo.force(1.0, 0.0, 0.0) == 2
    assert servo.force(0.1, 0.0, 0.5) == pytest.approx(0.5)


def test_axis_validation():
    with pytest.raises(ValueError):
        Axis("travel", "slide")  # slides need a range
    with pytest.raises(ValueError):
        Axis("bad name", "hinge")
    axis = Axis("a", "hinge", (0, 0, 2), (-1, 1))
    assert axis.direction == (0, 0, 1)
    assert axis.normalized(2.0) == 1.0 and axis.normalized(0.0) == 0.5


def test_threshold_hold_is_independent_of_the_sample_period():
    for dt in (0.001, 0.02):
        threshold = Threshold(0.5, 0.3, hold=0.1)
        times = np.arange(0, 0.2 + dt / 2, dt)
        states = [threshold.update(t, 1.0) for t in times]
        first = times[states.index(True)]
        assert first == pytest.approx(0.1, abs=1e-9)


def test_threshold_hysteresis_and_interlock():
    threshold = Threshold(0.5, 0.3)
    assert not threshold.update(0.0, 0.6, allowed=False)
    assert threshold.update(0.1, 0.6)
    assert threshold.update(0.2, 0.4)  # inside the band
    assert not threshold.update(0.3, 0.3)
    with pytest.raises(ValueError):
        Threshold(0.3, 0.5)
    assert Interlock("brake.value", 0.3).allows({"brake.value": 0.4})
    assert not Interlock("brake.value", 0.3).allows({})


def test_detent_counter_needs_a_margin_past_the_boundary():
    counter = DetentCounter(Detents(1.0, period=1.0), margin=0.2)
    assert counter.update(0.0) == (0, 0)
    assert counter.update(0.55) == (0, 0)  # 0.05 past the boundary < 0.1
    assert counter.update(0.65) == (1, 1)
    assert counter.update(0.45) == (1, 0)
    assert counter.update(2.1) == (2, 1)


def test_touch_tracker_identifies_touches():
    tracker = TouchTracker("screen", threshold=0.1)
    events = tracker.update(0.0, {7: Touch(0.2, 0.3, 0.05)})
    assert events == []  # below threshold
    events = tracker.update(0.1, {7: Touch(0.2, 0.3, 0.2), 9: Touch(0.8, 0.8, 0.5)})
    assert [(e.kind, e.value) for e in events] == [
        ("touch_begin", 1.0),
        ("touch_begin", 2.0),
    ]
    events = tracker.update(0.2, {7: Touch(0.25, 0.3, 0.2), 9: Touch(0.8, 0.8, 0.04)})
    assert [(e.kind, e.value) for e in events] == [
        ("touch_move", 1.0),
        ("touch_end", 2.0),
    ]
    events = tracker.update(0.3, {})
    assert [(e.kind, e.value) for e in events] == [("touch_end", 1.0)]
    assert tracker.touches() == []


def test_event_round_trip():
    event = Event(1.0, "screen.ok", "press", 1.0, {"u": 0.5})
    assert Event.from_dict(event.as_dict()) == event
    assert math.isfinite(event.as_dict()["time"])

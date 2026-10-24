"""The README quick start: a fingertip presses a button through a bridge."""

import mujoco

from embodied_ui import Interface, Knob, Pose, PushButton, TouchSurface
from embodied_ui.backends import LoopbackEngine
from embodied_ui.engine import Bridge
from embodied_ui.probe import Probe

panel = Interface(
    "panel",
    (
        PushButton("start"),
        Knob("volume", Pose((0.06, 0, 0))),
        TouchSurface("screen", Pose((0, 0.12, 0)), regions={"ok": (0, 0, 0.5, 0.5)}),
    ),
)
spec = mujoco.MjSpec()
probe = Probe(pos=(0, 0, 0.05))  # or your MyoSuite body
probe.attach(spec)
panel.attach(spec)  # or builder.apply_transform(panel.transform)
model = spec.compile()
data = mujoco.MjData(model)
runtime = panel.bind(model)

bridge = Bridge(runtime, LoopbackEngine(0.02))  # or RemoteEngine / CarlaEngine
bridge.reset(data)
tip = probe.bind(model)
for k in range(50):
    tip.target(data, (0, 0, 0.0 if 10 < k < 30 else 0.05))
    runtime.step(data, 10)  # applies detents and feedback, steps physics
    frame = bridge.step(data)
    for event in frame.reading.events:
        print(f"{event.time:.2f} s  {event.source}  {event.kind}")

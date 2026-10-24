# embodied-ui

Physical user interfaces for simulated bodies. embodied-ui adds buttons,
switches, knobs, sliders, levers, pedals, steering wheels, joysticks,
touchscreens, displays and lamps to any MuJoCo model. A simulated body, such
as a MyoSuite musculoskeletal model, a robot hand or a learned agent,
operates them through contact. Their channels and events go to a game engine
in lockstep: Unity, CARLA, or anything that speaks the `embodied-ui/1`
protocol.

```
 MuJoCo (body + devices)            embodied-ui                 engine
 ───────────────────────     ─────────────────────────     ────────────────
 finger presses a cap  ──▶  runtime.read(data)       ──▶  Unity Input System
 joint travel, contact       channels + events             virtual device and
                             Routes (scale, rename)        touchscreen, CARLA
                                                           vehicle control
 lamp colour, display  ◀──  runtime.set_outputs(...)  ◀──  lamps, images,
 texture, prop pose,         Bridge (one period, one       poses, torques,
 wheel torque                frame, clocks checked)        telemetry
```

It is engine-independent and model-independent. It does not register
environments, does not depend on MyoSuite or CARLA, and imports engine
clients lazily. It works on CPU MuJoCo with `MjSpec`, and with MyoSuite's
`ModelBuilder` through `ModelBuilder.apply_transform(interface.transform)`.

## Design

- **Inputs are mechanisms, not mocap.** A button is a sliding body with a
  preloaded spring and an end stop. A knob is a hinge with detents, and a
  touchscreen is a rigid panel that reads contact positions. The simulated
  body must push, turn and touch them. Forces, travel, debouncing and
  interlocks then behave as they do on hardware.
- **Engine-driven outputs are kinematic or applied.** Lamps change colour,
  displays update a texture, props follow engine poses as mocap bodies, and
  force feedback (wheel torque, motorized faders) is applied to the device's
  own degree of freedom. The rest of the body is never touched.
- **Geometry may come from the engine.** Meshes exported from Unity or glTF
  are converted to MuJoCo's axes. Each gets a fitted primitive collider (box,
  cylinder, capsule or sphere, via [trimesh](https://trimesh.org)), because
  fingertip contacts are most stable on primitives.
- **One clock.** `Bridge.step` requires MuJoCo to have advanced by exactly one
  period, sends one command, and requires the engine to report exactly one
  period later. A mismatch raises `SynchronizationError`.

## Install

```bash
pip install -e .                 # numpy and mujoco only
pip install -e ".[mesh]"         # engine meshes and primitive fitting
pip install -e ".[test,lint]"    # development
```

CARLA needs its own version-matched client wheel. The Unity package is in
`unity/io.github.m-colley.embodied-ui`; install it in Unity's Package Manager
from a git URL ending in `?path=unity/io.github.m-colley.embodied-ui`, or from
disk.

## Quick start

```python
import mujoco
from embodied_ui import Interface, Knob, Pose, PushButton, TouchSurface
from embodied_ui.backends import LoopbackEngine
from embodied_ui.engine import Bridge
from embodied_ui.probe import Probe

panel = Interface("panel", (
    PushButton("start"),
    Knob("volume", Pose((0.06, 0, 0))),
    TouchSurface("screen", Pose((0, 0.12, 0)), regions={"ok": (0, 0, 0.5, 0.5)}),
))
spec = mujoco.MjSpec()
probe = Probe(pos=(0, 0, 0.05))          # or your MyoSuite body
probe.attach(spec)
panel.attach(spec)                       # or builder.apply_transform(panel.transform)
model = spec.compile()
data = mujoco.MjData(model)
runtime = panel.bind(model)

bridge = Bridge(runtime, LoopbackEngine(0.02))   # or RemoteEngine / CarlaEngine
bridge.reset(data)
tip = probe.bind(model)
for k in range(50):
    tip.target(data, (0, 0, 0.0 if 10 < k < 30 else 0.05))
    runtime.step(data, 10)               # applies detents and feedback, steps physics
    frame = bridge.step(data)
    for event in frame.reading.events:
        print(event.time, event.source, event.kind)
```

`examples/carla_fingertip.py` drives a CARLA vehicle this way.

## Devices

Every device has its own frame: x to the user's right, y up and z out of the
panel towards the user. `pose` places it in its `mount` body (`world` by
default). Channels are `device.channel` floats; events are `press`,
`release`, `toggle`, `detent`, `touch_begin`, `touch_move` and `touch_end`.

| Device | Degrees of freedom | Channels (besides raw positions, rates and `contact` force) | Engine outputs | Defaults and sources |
| --- | --- | --- | --- | --- |
| `PushButton` | slide | `value`, `pressed`, `state` (toggle mode) | `force` | 15 mm cap, 4 mm travel, 4.5 N at the make point, within MIL-STD-1472G/H fingertip buttons (10-25 mm, 2-6 mm, 2.8-11 N). `PushButton.key()`: ISO 9241-410 key, 3 mm and 0.65 N (preferred 2-4 mm, 0.5-0.8 N). Hysteresis, hold time, toggle, interlock, tactile force curves. |
| `ToggleSwitch` | hinge | `state` | `force` | Over-centre snap modelled as detent wells beyond the stops |
| `Knob` | hinge (+ push) | `value` (turns or range), `detent`, `pressed` | `force` | 24 detents per turn; 12 mN m detents plus 2 mN m friction, below MIL-STD-1472G's 32 mN m for continuous knobs |
| `Slider` | slide | `value`, `detent` | `force`, `setpoint` (servo) | Optional detents; motorized faders through `Servo` |
| `Lever` | hinge | `value`, `position` (gate) | `force` | Gates 17 degrees apart with 150 mN m detents, within MIL-STD-1472G detented selectors (15-40 degrees, 115-680 mN m); spring return for stalks |
| `Pedal` | slide | `value` | `force` | Preloaded spring, 10-40 N over 60 mm |
| `SteeringWheel` | hinge | `value` (angle over lock) | `torque`, `setpoint` | Rim of capsules, ±450 degree lock, centring spring, column friction |
| `Joystick` | 2 hinges | `x_value`, `y_value` | per-axis `force` | Spring-centred gimbal |
| `TouchSurface` | none | `touches`, `u`, `v`, `force`, `<region>.pressed` | `image` (with `resolution`) | One touch per touching body. In-vehicle soft keys: no gain beyond 17.5 mm (Kim et al., 2014); ≥ 5 mm between targets (ISO 9241-410 Annex J) |
| `Screen` | none | none | `image` | Texture written into `model.tex_data` |
| `Indicator` | none | none | `on` | Off and on colours |
| `Prop` | mocap | none | `pose` | Engine-driven object the body can touch but not push |
| `Control` | any | per axis, `value` | per axis | Compose your own from `Axis`, `Detents`, `ForceCurve`, `Servo` and `Part` |

Register custom classes with `@register` so manifests can name them.
Linear springs, damping, Coulomb friction, armature and end stops use
MuJoCo's own joint terms. Hinge angles are converted when the host compiles
in degrees. Detents, force curves, servos and engine forces are written to
the device's own `qfrc_applied` entries before each physics step. Call
`runtime.apply(data)`, or use `runtime.step(data, n)`, every step.
`Contacts(pairs=("distph*",))` limits collisions to named segments of a dense
musculoskeletal model.

## Engines

| Backend | Use |
| --- | --- |
| `LoopbackEngine(dt, update)` | In-process tests, offline studies, and tasks written in Python, such as a secondary task that draws a display and reacts to touches |
| `RemoteEngine(host, port, dt=...)` | Any `embodied-ui/1` server: the Unity package, `EngineServer` (which serves any Python engine on another machine), or your own |
| `CarlaEngine(...)` | One CARLA vehicle in synchronous mode: `vehicle.*` controls, `lights.*` with hazard, telemetry, driver-eye RGB cameras. It restores the world's settings on close |

`Routes` map channels to engine controls with scale, offset and clipping.
They also rename device events and route engine outputs back to devices
(`feedback`). With no routes, every channel and event passes through under
its own name. Poses from engines are converted with `engine_frame`.

### Coordinate conventions

`embodied_ui.frames` names each axis direction, so any two conventions
convert exactly. This includes handedness flips: rotations as `C R Cᵀ`,
angular rates as pseudovectors. Built in: `flu` (x forward, y left, z up),
`unity`, `unreal` (centimetres), `carla`, `gltf` and `myosuite_seated`. A
device's own axes in each engine follow that engine's habit. In Unity, a
control's local +z points into the panel, as for a world-space canvas. In
Unreal and CARLA, local +x does.

## Unity

The package `io.github.m-colley.embodied-ui` (Unity 2022.3 or newer, Input
System, Newtonsoft JSON) contains:

- **EmbodiedBridge**: the lockstep server. It handles one request at the
  start of each frame, before Unity updates its clock, and sets
  `Time.captureDeltaTime` to the shared period. It sends the state after
  rendering.
- **Virtual input hardware.** Every channel becomes a control of an
  `EmbodiedUI` Input System device. For example, `start.pressed` binds as
  `<EmbodiedUI>/start_pressed`, so existing input actions work unchanged.
  Touches arrive on a virtual `Touchscreen` for uGUI and UI Toolkit.
- **Outputs.** `EmbodiedPoseOutput`, `EmbodiedVehicleOutput` (SI telemetry),
  `EmbodiedCameraOutput` (driver views or `screen.image`),
  `EmbodiedLampOutput`, `EmbodiedValueOutput` and `SetOutput` from scripts.
- **Designing devices in Unity.** Put an `EmbodiedDevice` component on a
  control, optionally with its mesh. The Python side then fetches the scene's
  devices at runtime with `embodied_ui.backends.describe(port=...)`, or loads
  a manifest exported with *Tools > Embodied UI > Export Manifest*:

  ```python
  dashboard = describe(port=47474, target="myosuite_seated")  # an Interface
  ```

- `EmbodiedEventListener` for UnityEvents, and a WheelCollider vehicle
  sample.

The engine-independent C# protocol layer (`Runtime/Protocol`) also builds with
.NET. `dotnet/EmbodiedUI.Protocol.Harness` runs its self-tests and serves a
toy engine, which the Python test suite drives through a real bridge.

## Other engines

[PROTOCOL.md](PROTOCOL.md) specifies the `embodied-ui/1` wire format: framing,
messages, the vehicle output schema and conformance tests. The Python
`EngineServer` and the C# `LockstepServer` are reference servers for
implementers in Unreal, Godot or other simulators.

## Mix and match with myosuite-carla

[myosuite-carla](https://github.com/M-Colley/myosuite-carla) can use these
devices and engines:

- `FullBodyDriverConfig(interfaces=(dashboard,))` composes devices into the
  MyoSuite full-body driver; its environment reports them in
  `info["interface"]`.
- `myosuite_carla.interfaces.interface_hook` routes them to CARLA's lights,
  with feedback to lamps.
- `EngineVehicleSession` lets the full-body driver and its scenarios drive a
  Unity car.
- `VehicleSessionEngine` lets any MuJoCo body with embodied-ui controls drive
  CARLA through `CarlaSession`, or the fitted vehicle surrogate offline.

## A simulated human to operate them

[embodied-human](https://github.com/M-Colley/embodied-human) builds a modular
simulated human on top of these devices: eyes, foveated vision, attention,
memory, proprioception and motor modules. It includes SuRT and DRT tasks on
embodied-ui touch surfaces, lamps and buttons, and a benchmark against
published human data.

## Validation

See [VALIDATION.md](VALIDATION.md). In short:

- **Offline (Python):** the offline suite covers device mechanics through
  contact with a servo fingertip, logic, frames, manifests, meshes, the
  bridge and the TCP protocol.
- **C# protocol layer:** it builds with .NET 9, passes its self-tests, and
  interoperates with the Python client.
- **CARLA:** the live tests pass against CARLA 0.10.0 (UE5.5), including
  driver-eye cameras.
- **Unity 6.6 (6000.6.7f1):** the package compiles without errors and its
  PlayMode tests pass in batch mode. A live test drives a Unity scene from a
  MuJoCo fingertip, checking lockstep timing, input, physics, lamps,
  touches, camera images and `describe`.

## Development

```bash
python -m pytest tests -q
python -m ruff check src tests examples
python -m mypy src/embodied_ui --ignore-missing-imports
python -m build
dotnet build dotnet/EmbodiedUI.Protocol.Harness -c Release
EMBODIED_UI_CARLA=1 python -m pytest tests/test_carla_live.py   # with a CARLA server
EMBODIED_UI_UNITY="<path to Unity.exe>" python -m pytest tests/test_unity_live.py
```

## Licence

Apache-2.0. Standards are cited for their published numbers only; no
standard text is reproduced.

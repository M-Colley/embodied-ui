# Validation

What has and has not been verified, with the commands and versions used.
A skipped test or a test double is not evidence of a live pass.

## 2026-10-24, first PC (Windows 11, CPU MuJoCo)

Python 3.12.13, MuJoCo 3.17.0, NumPy 2.5.4, trimesh 5.1.1, SciPy 1.17.1;
.NET SDK 9.0.317 with Newtonsoft.Json 13.0.4; CARLA 0.10.0 (UE5.5 shipping
package) with Town10HD_Opt and `vehicle.nissan.patrol` at spawn point 105.

### Offline suite

```bash
python -m pytest tests -q
```

- **Result:** 67 passed and 2 skipped (the live CARLA tests), on MuJoCo
  3.17.0 ([JUnit](evidence/offline-mujoco-3.17.xml)) and on MuJoCo 3.7.0
  ([JUnit](evidence/offline-mujoco-3.7.xml)). This includes the .NET interop
  tests.
- **Devices operated through contact:** a servo fingertip (`Probe`) presses,
  pushes, turns and touches the devices. The tests check:
  - spring forces against their specification;
  - hysteresis and hold times independent of the sample period;
  - toggles, interlocks and bistable switches, with host models compiled in
    degrees and in radians;
  - detent settling and counting, including turning a knob by fingertip
    twist;
  - touch positions within 0.01 of the true (u, v), with regions and touch
    identities;
  - force feedback and servo setpoints, lamps, display textures (row order)
    and mocap props.
- **Composition:**
  - mounting on moving bodies;
  - explicit contact pairs;
  - inertia independent of the host's compiler settings;
  - manifests, including a lossless JSON round trip and Unity-frame
    conversion;
  - engine meshes with fitted colliders, outward winding and touch sensing.
- **Frames:** exact round trips between all conventions; Unity, Unreal and
  CARLA directions; pseudovectors; the Unreal rotator and its inverse.
- **Bridge:**
  - passthrough and routed commands, event renaming and feedback routes;
  - pose conversion;
  - detection of MuJoCo and engine clock slips;
  - the TCP protocol with attachments, NaN as `null`, errors, version
    checks, and malformed or oversized frames.

**Minimum MuJoCo version.** MuJoCo 3.6.0 produces almost no contact force
when a geom on a body hinged about the contact normal is pressed. A knob
pressed from above, for example, gives 0.001 N where 4.94 N is expected.
3.7.0 through 3.17.0 give 4.94 N (bisected on 3.6.0, 3.7.0, 3.8.0, 3.8.1,
3.11.0, 3.14.0 and 3.17.0). The package therefore requires `mujoco>=3.7`.

### C# protocol layer (Unity package `Runtime/Protocol`)

```bash
dotnet build dotnet/EmbodiedUI.Protocol.Harness -c Release
dotnet dotnet/EmbodiedUI.Protocol.Harness/bin/Release/net9.0/EmbodiedUI.Protocol.Harness.dll selftest
```

- **Self-tests:** the package's unmodified protocol sources build with
  warnings as errors at C# 9, the language level of Unity 2022.3 and 6. Ten
  self-tests pass ([output](evidence/dotnet-protocol-selftest.txt)).
- **Interop:** `tests/test_dotnet_interop.py` starts the harness's toy engine
  and checks it end to end:
  - it fetches the engine's devices with `describe`;
  - it mixes them with Python-defined devices;
  - a fingertip presses a button through a real bridge for 40 frames;
  - frame numbers and times are exact;
  - the press event reaches C#, which toggles a lamp that lights in MuJoCo;
  - the throttle route moves the toy car;
  - an image attachment reaches the MuJoCo display.

### CARLA, live

The server was `CarlaUnreal.exe -nullrhi -nosound -unattended -notrace
-carla-rpc-port=2000`. For cameras it was `-RenderOffScreen -quality-level=Low`
instead.

```bash
EMBODIED_UI_CARLA=1 EMBODIED_UI_CARLA_CAMERA=1 python -m pytest tests/test_carla_live.py
python examples/carla_fingertip.py --camera --out evidence/carla-fingertip.json
```

- **Live suite:** 2 passed ([JUnit](evidence/carla-live-suite.xml)).
  - The fingertip's throttle key drove the vehicle.
  - A hazard switch set both blinkers, and CARLA's reported light state lit
    a MuJoCo lamp through a feedback route.
  - CARLA's frame numbers and time advanced exactly once per period.
  - The world's asynchronous settings were restored on close.
  - Driver-eye camera images (160 x 90) arrived for the simulated frame.
- **Example run** ([report](evidence/carla-fingertip.json),
  [driver view](evidence/carla-fingertip.png)):
  - 200 lockstep frames;
  - throttle key held from 0.5 s to 2.5 s;
  - top speed 7.42 m/s;
  - hazard lights on from 1.0 s, lamp lit at the end;
  - 32.5 frames per wall-clock second with a 640 x 360 camera rendered
    every frame.
- **A mistake during the session:** stopping only CARLA's launcher process
  left the headless server running. A camera test then crashed it, because
  CARLA cameras need a rendering server. Stop
  `CarlaUnreal-Win64-Shipping.exe` too.

### Mix and match with myosuite-carla (live)

[myosuite-carla](https://github.com/M-Colley/myosuite-carla) recorded this
run in its own VALIDATION.md and in `evidence/ue55-interface-hazard.json`:

- **Setup:** the MyoSuite full-body driver (fast preset) carried an
  embodied-ui hazard switch and lamp on its steering-wheel hub. Its reflex
  driver drove CARLA through the muscles.
- **Result:** after 300 periods the vehicle was at 8.14 m/s. Once the switch
  was flipped at 2.02 s, CARLA's light state was 48 (both blinkers) and the
  lamp was lit from that state.

### Unity 6.6, live (2026-10-24, later the same day)

Unity 6000.6.7f1 was installed. The test project `unity/EmbodiedUITests`
references the package from disk with Input System 1.20.1, Newtonsoft JSON
3.2.2 and Test Framework 1.8.0, all bundled with the editor.

```bash
Unity.exe -batchmode -projectPath unity/EmbodiedUITests -runTests -testPlatform PlayMode -testResults results.xml
EMBODIED_UI_UNITY="<path to Unity.exe>" python -m pytest tests/test_unity_live.py
```

**Compilation.** The package compiles with no errors. Unity 6.6 reported one
warning, the deprecated `FindObjectsByType` sort overload, which is now
handled.

**PlayMode tests.** Both pass ([results](evidence/unity-live.xml),
[summary](evidence/unity-live.json)).

- **`ACSharpClientStepsTheSceneInLockstep`:** a background-thread client sends
  hello, 20 steps, reset and close. Every frame lasts 0.02 s with exactly
  one physics step. An Input System action bound to
  `<EmbodiedUI>/vehicle_throttle` reads the control and accelerates a
  rigid body, a light follows it, and touches begin and end on the
  virtual touchscreen. Device events reach `EventReceived`, and camera
  images arrive as attachments.
- **`PythonClientDrivesTheScene` with `tests/test_unity_live.py`:**
  - `describe` returns the scene's designed button. Its Unity pose converts
    to (0.4, 0, 1.0) m in MuJoCo, it is pressed forward, and its cube mesh
    gets a fitted 15 x 15 x 6 mm box collider.
  - A servo fingertip presses an embodied-ui key for 1.6 s. Over 150
    lockstep frames, Unity's input read 1.0 throughout the press and the
    car reached 7.7 m/s.
  - Unity's light lit the MuJoCo lamp, a tap on the MuJoCo touch surface
    became one Unity touch, and 64 x 48 camera images arrived every frame.
  - Unity's frame time was 0.0199999921 s (the float32 value of 0.02)
    every frame.

**Two defects found and fixed.**
- **Hook point.** The bridge first hooked into the `Initialization` phase,
  after Unity's `TimeUpdate`, so the first frame of a session ran with the
  real elapsed time (0.15 ms).
- **Focus.** The Input System disabled the virtual device because a
  batch-mode editor is unfocused. The bridge now sets `backgroundBehavior`
  to `IgnoreFocus` (and, in the Editor, routes all device input to the Game
  view) during a session, restoring the settings afterwards.

### Unity 6.6, multimodal outputs (2026-10-24, evening)

Same editor and test project, rerun after adding sound and vibration:

- **New outputs.** `EmbodiedAudioOutput` and `EmbodiedVibrationOutput`
  report `chime.level`, `chime.frequency`, `seat.amplitude` and
  `seat.frequency`; the test car sets them at full throttle.
- **C# test.** `ACSharpClientStepsTheSceneInLockstep` checks that the chime
  is silent in the welcome and at 64 dB in the last frame, and that the
  seat vibrates at 2 m/s².
- **Live test.** The MuJoCo rig gains a `Speaker` and a `Vibrotactor`.
  - 0.5 m from the speaker, the free-field level is 70.0 dB, which is
    64 dB + 20 log10 2.
  - The seat signal arrives at 2 m/s², and the felt vibration stays 0
    because nothing touches the pad.
  - A `TraceRecorder` wrote all 150 frames as a CSV with 43 columns.
- **Result.** Both PlayMode tests pass
  ([summary](evidence/unity-live.json)). The log has no compiler warnings
  or errors.
- **Offline suite.** With these changes it has 75 passed and 3 skipped
  (the CARLA and Unity live tests) on MuJoCo 3.17.0. The new tests cover:
  - the speaker's free-field level;
  - vibration felt only through contact with the pad;
  - lamp signals;
  - trace files in CSV and JSON Lines;
  - the Unity package layout.

**Defect found and fixed.** All output components shared one file,
`EmbodiedOutputs.cs`. Unity attaches a component in the Inspector, and
restores it from a saved scene, only if its class and file names match.
Adding components from code, as the tests do, hid the problem. Each
component now has its own file and `.meta`, and
`tests/test_unity_package.py` checks this offline.

## 2026-10-25, first PC: Editor exporter, sample, player and a MyoSuite hand

Same PC and Unity 6000.6.7f1 editor (Input System 1.20.1, Test Framework
1.8.0, NUnit engine 3.5.0.0). Windows build support (Mono, x86_64) is
installed with the editor. Python 3.12.13, MuJoCo 3.17.0, NumPy 2.5.4,
trimesh 5.1.1, MyoSuite 3.0.0 from its `dev` branch (commit c0af9c6) with
myo_sim 0.2.3. Ruff 0.17.2 and mypy 2.4.0. The GPU was an NVIDIA GeForce
RTX 5080.

```bash
python -m pytest tests -q --junitxml evidence/offline-2026-10-25.xml
python -m ruff check src tests examples
python -m mypy src/embodied_ui --ignore-missing-imports
EMBODIED_UI_UNITY="<path to Unity.exe>" python -m pytest tests/test_unity_live.py \
  tests/test_unity_editor_live.py tests/test_unity_sample_live.py \
  tests/test_unity_player_live.py tests/test_myosuite_hand.py \
  --junitxml evidence/unity-live-suite-2026-10-25.xml
EMBODIED_UI_UNITY_PLAYER=<built EmbodiedUIPlayer.exe> EMBODIED_UI_UNITY_PLAYER_MODE=windowed \
  python -m pytest tests/test_unity_player_live.py
```

The `EMBODIED_UI_*_EVIDENCE` variables named in each test wrote the JSON
summaries below.

**Checks.**
- **Offline suite:** 77 passed and 6 skipped (the CARLA and Unity live
  tests) ([JUnit](evidence/offline-2026-10-25.xml)). The MyoSuite hand test
  is offline and runs here. With `myo_sim` made unimportable, the suite has
  75 passed and 7 skipped.
- **Lint and types:** ruff passes, and mypy finds no issues in 16 files.
- **Live suite:** all 6 tests pass in 42 s
  ([JUnit](evidence/unity-live-suite-2026-10-25.xml)). This includes the
  rerun of `tests/test_unity_live.py`, now limited to the `EmbodiedUI.Tests`
  assembly. None of the Unity logs has a compiler warning or error.

### Editor manifest exporter (EditMode)

The package now has an EditMode test assembly, `Tests/Editor`
(`EmbodiedUI.Editor.Tests`, Editor only, constrained to
`UNITY_INCLUDE_TESTS`). `tests/test_unity_editor_live.py` runs it with
`-runTests -testPlatform EditMode` and loads the export in Python.

- **Unity side** ([NUnit](evidence/unity-editor-export.xml)). All four tests
  pass:
  - The menu item *Tools > Embodied UI > Export Manifest...* exists and
    calls the public routine `ManifestExporter.Export(path, reference)`.
  - A new scene has a mesh button, a knob, a tilted touch surface and a
    sphere lamp, posed relative to a rotated reference. The export lists
    them in name order. Its poses are relative to the reference, its
    parameters are as set, and the indicator's mesh part has zero mass.
  - The meshes lie beside the manifest, in device axes and metres with the
    scale applied: the cube cap spans ±7.5 x ±7.5 x ±3 mm with 12 faces,
    and the lamp sphere has a 10 mm radius.
  - Duplicate device names are rejected and nothing is written. Without a
    reference, poses are in world coordinates.
- **Python side** ([summary](evidence/unity-editor-export.json)):
  - `Interface.from_manifest` loads the exported file and converts it from
    Unity's frame. The button lies at (0.4, 0, 0.5) m and is pressed
    forward; the tilted surface matches `UNITY.device_pose_to(FLU, ...)`.
  - The interface compiles in MuJoCo. The cap gets a fitted 3 x 7.5 x
    7.5 mm (half-size) box collider and the lamp a 10.0 mm sphere.
  - A servo fingertip then presses and releases the exported button.

**Two defects found and fixed.**
- **Unsaved scenes.** An unsaved scene has an empty name, so the manifest
  had `"name": ""`, which `Interface` rejects. Such exports are now named
  `untitled`, in `ManifestBuilder`, which also serves `describe`.
- **Missing folders.** `Export` failed with `DirectoryNotFoundException`
  when the target folder did not exist and no device had a mesh. It now
  creates the folder.

### WheelCollider sample (PlayMode, driven from Python)

The sample now has its own assembly definition,
`EmbodiedUI.Samples.WheelVehicle`, so that test and user assemblies can
reference it. `tests/test_unity_sample_live.py` imports it as Package
Manager would:
- It copies `Samples~/WheelVehicle` to
  `Assets/Samples/Embodied UI/0.1.0/Wheel Vehicle` in the test project.
- It adds `unity/EmbodiedUITests/SampleTests/WheelVehicle` as the sample's
  `Tests` folder, with its own test assembly. The copies are ignored by git.

The PlayMode test builds a 1500 kg car with four wheel colliders (0.35 m
radius) on flat ground and adds the sample's `EmbodiedWheelVehicle`. A
`RemoteEngine` then sends `vehicle.throttle`, `vehicle.brake` and
`vehicle.steer` for 400 frames
([summary](evidence/unity-wheel-vehicle.json),
[NUnit](evidence/unity-wheel-vehicle.xml)):

- **Lockstep:** every frame lasted 0.02 s with one physics step, and frame
  numbers and times were exact.
- **Controls:** in the same frame, the wheel colliders carried the sample's
  torques and angle: 1500 N m per unit throttle, 3000 N m per unit brake,
  35 degrees per unit steer. All four wheels stayed grounded.
- **Accelerates:** after 3 s of full throttle the car reached 8.81 m/s and
  its heading had not changed. A torque-limited bound is 5.7 m/s², which
  would allow at most 17 m/s.
- **Turns:** with full right steer and 0.3 throttle for 2 s, the heading
  turned 67.3 degrees to the right, and the yaw rate stayed positive.
- **Brakes:** under full brake the car slowed by 3.86 m/s² on average over
  the first 0.5 s and stopped (0.0006 m/s after 2.5 s).

### A Windows player outside the Editor

`unity/EmbodiedUITests/Assets/Player` holds a car driven by an Input System
action on `<EmbodiedUI>/vehicle_throttle`, a lamp, a probe output and a
bootstrap. The bootstrap reads `-embodiedPort`, waits for the client and
quits when the session ends. Its Editor script
`EmbodiedUI.PlayerTest.PlayerBuild.Build` generates the scene into
`Assets/Generated` (ignored by git) and builds a Windows 64-bit Mono player:

```bash
Unity.exe -batchmode -quit -projectPath unity/EmbodiedUITests -buildTarget Win64 \
  -executeMethod EmbodiedUI.PlayerTest.PlayerBuild.Build -embodiedBuildPath <folder>
```

- **Build:** succeeded with 0 errors and 0 warnings and a 98.7 MB player.
  The first build took 39 s and an incremental one 9.6 s, including Editor
  start-up.
- **Rendering mode.** The player runs with `-batchmode` by default. This
  shows no window, but unlike `-nographics` it still creates a graphics
  device, here Direct3D 12, so cameras render. A windowed run (320 x 240,
  `-screen-fullscreen 0`) passes the same test. The bootstrap sets
  `Application.runInBackground`, so an unfocused window keeps running.
- **Batch-mode run** ([summary](evidence/unity-player.json),
  [camera frame](evidence/unity-player.png)):
  - The player reported `isEditor` 0 and `isBatchMode` 1.
  - All 100 lockstep frames lasted 0.02 s with one physics step each.
  - The fingertip's key reached the player's input action (1.0 throughout
    the press, 0 before it). The car reached 5.73 m/s and the lamp lit the
    MuJoCo indicator.
  - 64 x 48 camera images arrived every frame; their pixel standard
    deviation was 34, so they are rendered frames, not blanks.
  - The run reached 402 lockstep frames per wall-clock second, and the
    player exited with code 0 after `close`.
- **Windowed run** ([summary](evidence/unity-player-windowed.json)): the
  same results at 61 frames per second, which matches vertical sync on a
  60 Hz display.

### A MyoSuite hand presses a button through contact

`tests/test_myosuite_hand.py` loads myo_sim's `myohand` (right forearm and
hand, 23 joints, 39 muscles) with `myo_sim.load_spec`. It composes a default
`PushButton` into the same `MjSpec` with `Interface.attach`, using
`Contacts(pairs=("distph2_r",))`. The test skips when `myo_sim` is missing.

**Open-loop pattern**, with no policy:
- FCR and FCU hold 5 % activation throughout, which braces the wrist
  against its 45-degree flexion stop.
- The hand relaxes for 1 s.
- FDS2, the index finger's superficial flexor, ramps from 0 to 30 % in
  0.25 s, holds for 0.5 s, falls to 0 in 0.1 s and stays off.

**Placement.** The button faces the relaxed fingertip 2 mm from the nearest
distal-phalanx collider (`mj_geomDistance`). It is pressed along the
direction the fingertip moves when the MCP joint flexes (`mj_jac`).

**Results** ([summary and trace](evidence/myosuite-hand-button.json)):
- **Events:** one press at 0.26 s while FDS2 was active, and one release at
  0.92 s, after its switch-off. The MCP joint never flexed beyond
  0.016 rad, so the fingertip did not slide off the cap.
- **Forces:**
  - 4.85 N at the press, where the spring gives 4.5 N at the make point.
  - 3.74 N at the release, where it gives 4.05 N at 35 % travel.
  - The peak was 6.52 N with the cap at its end stop, where the spring
    gives 6 N.
  - The button's touch sensor matched the solver's normal contact force to
    within 10⁻⁶ relative in every frame.
- **After release:** the relaxed finger rested on the cap with at most
  1.80 N, below the 3 N preload, so the cap stayed up.
- **Control:** with FDS2 off, the button was neither pressed nor moved, and
  its contact force stayed below 3 N.

**Exploration before choosing the pattern** (scripts, not tests). Each
trial ran 1.5-2 s. FDS2 or FDP2 alone, both together, or the index
intrinsics (LU_RB2, RI2, UI_UB2) were ramped as above.
- **Wrist free:**
  - Weak ramps (5-20 %) never reached the make point (at most 3.3 N).
  - Ramps strong enough to press (20-60 %, depending on the muscles) did
    press the button. The wrist then extended by 0.2-0.3 rad under the
    button's reaction, and the fingertip slid off the cap's edge
    0.06-0.26 s later, while the muscles were still active.
  - The intrinsics alone never exceeded 1.7 N.
- **Other postures:**
  - With the wrist held at its extension stop (ECRL, ECRB and ECU at 10 %
    or 30 %), FDS2 at 30-60 % also slid off.
  - In a pronated forearm (PT and PQ at 30 %, or after 8 s of relaxation),
    FDS2 at 30-60 % slid off too.
- **Braced wrist (FCR and FCU at 5 %):** FDS2 at 25-35 % pressed and
  released reliably. This held over 1.0-1.5 s of relaxation and 1-3 mm
  gaps:
  - press at 0.22-0.30 s with 4.6-5.1 N;
  - release at 0.88-0.92 s with 3.6-4.0 N;
  - peak 5.2-7.8 N.
- **Limits of the braced pattern:** FDS2 at 60 % or more slid off after
  2 s of relaxation. FDS2 and FDP2 together slid off at 30 %.
- **Drift:** in the braced posture, gravity slowly pronates the forearm.
  `pro_sup` rose by about 0.25 rad over a trial. The myohand has no
  supinator or biceps, and the open-loop pattern does not correct the
  drift.

The pattern shows that muscle-driven fingers operate the devices through
contact as the servo fingertip does. It is not a model of how people press
buttons.

### Continuous integration and an independent rerun

- **CI.** `.github/workflows/tests.yml` runs the offline suite (with the
  .NET interop test), ruff, mypy and the build on Python 3.10 and 3.12.
- **Python 3.10 rehearsal.** A fresh 3.10 environment, installed as the
  workflow installs it, passed 75 tests with 7 skipped (no MyoSuite, CARLA
  or Unity).
- **Typing fix.** Its mypy flagged `memoryview(array)` in `protocol.py`:
  NumPy arrays are not typed as buffers before Python 3.12. Attachments are
  C-contiguous, so `array.data` gives the same view, now used.
- **Independent rerun** of this section's work, outside the session that
  wrote it:
  - the offline suite: 77 passed, 6 skipped
    ([JUnit](evidence/offline-2026-10-25.xml));
  - all six live tests (the four Unity tests and both MyoSuite hand tests),
    which passed in 39 s.

## Not yet validated

- **Unity:**
  - The menu item was not clicked interactively. Its save dialog cannot run
    in batch mode, so the tests call the routine it calls.
  - Only a Windows 64-bit Mono player was built. IL2CPP, other platforms
    and Unity 2022.3, the declared minimum, have not been tested.
- **Unreal:** no Unreal or Godot server exists. PROTOCOL.md specifies one.
- **Musculoskeletal operation:**
  - One MyoSuite hand pressed one button, with a hand-tuned open-loop
    pattern and a braced wrist.
  - Learned or reflex control, the other devices (knobs, switches, touch
    surfaces) and an unbraced wrist have not been shown.
  - The hand test was run on MuJoCo 3.17.0 only.

## 2026-10-26: Python 3.14

A fresh Python 3.14.4 environment (uv) with `pip install -e ".[test,lint]"`
passed 75 tests and skipped 7: the live CARLA, Unity and MyoSuite-hand
tests that CI also skips. Ruff, mypy and the build pass. The CI workflow
now also runs on Python 3.14.

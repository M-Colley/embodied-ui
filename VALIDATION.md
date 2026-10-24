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

### Not yet validated

- **Unity scripts:** `Runtime/Unity`, `Editor` and the sample (the bridge
  component, virtual input device, touchscreen injection, outputs, describe
  with real meshes, exporter) have not been compiled or run in a Unity
  Editor. Only Unity Hub metadata was present on this PC, without an editor.
- **Unreal:** no Unreal or Godot server exists. PROTOCOL.md specifies one.
- **Musculoskeletal operation:** no MyoSuite hand has yet operated these
  devices with learned or reflex control; the tests use the servo fingertip.

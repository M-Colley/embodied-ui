# Embodied UI for Unity

Lockstep server for [embodied-ui](https://github.com/M-Colley/embodied-ui):
simulated MuJoCo bodies (MyoSuite muscles, robot hands, learned agents)
press, turn and touch controls whose channels arrive in Unity as Input System
hardware. Unity reports poses, lamps, displays, camera images and vehicle
telemetry back.

## Setup

1. Add the package. In *Package Manager > Add package from git URL*, enter
   `https://github.com/M-Colley/embodied-ui.git?path=unity/io.github.m-colley.embodied-ui`.
   It depends on the Input System and Newtonsoft JSON.
2. Add **Embodied Bridge** to one GameObject. Set the port (default 47474) and
   optionally a *reference* transform, the origin of poses and devices.
3. Bind input actions to `<EmbodiedUI>/<device>_<channel>`, for example
   `<EmbodiedUI>/start_pressed` or `<EmbodiedUI>/wheel_value`. Alternatively,
   read `EmbodiedBridge.Active.GetControl("wheel.value")` from scripts. Touch
   surfaces drive a virtual touchscreen; route them with
   `EmbodiedTouch.ToScreen`.
4. Report outputs: **Pose Output** (`cup.pose`), **Vehicle Output**
   (`vehicle.*`), **Camera Output** (`camera.driver` or `screen.image`),
   **Lamp Output** (`lamp.on`), **Value Output** (`wheel.torque`), or
   `EmbodiedBridge.Active.SetOutput(name, value)`.
5. Optionally design controls here. Add **Embodied Device** to a control,
   with its local +Z pointing into the panel. Python then builds the devices
   with `embodied_ui.backends.describe(port=47474)`, or from
   *Tools > Embodied UI > Export Manifest*.

## How a frame runs

Each request is handled at the start of a frame, before Unity updates its
clock. On `step`, the controls are written to the virtual device; with
**Drive Input Updates**, Input System events are processed once, so FixedUpdate
and Update read the same values. The frame then runs with
`Time.captureDeltaTime` equal to the session period and `physicsSubsteps`
fixed steps. After rendering, the outputs are sent. A reset invokes
`EmbodiedBridge.ResetRequested`, runs one frame to apply it, and reports frame
0. If no request arrives within **Request Timeout**, the game runs freely
again.

## Status

The `Runtime/Protocol` assembly has no Unity dependency. It is built and
tested with .NET 9 against the Python client. The Unity-specific scripts
have not yet been compiled in a Unity Editor; see the repository's
VALIDATION.md.

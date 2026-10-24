# The `embodied-ui/1` lockstep protocol

This document is for people who implement an engine server: Unreal, Godot, a
custom simulator or another language. The Python client is
`embodied_ui.backends.RemoteEngine`. The reference servers are
`embodied_ui.backends.EngineServer` (Python) and `LockstepServer` in the Unity
package's `Runtime/Protocol` (C#, no Unity dependency).

## Roles and timing

The **client** is the MuJoCo side. It connects over TCP to the **server**,
the engine, which listens (default port 47474). A session shares one fixed
period `dt`:

1. The client advances MuJoCo by `dt` and measures the devices.
2. It sends one `step` with the measured controls and events.
3. The engine applies them, advances exactly `dt`, and replies with one
   `state`.
4. The client applies the outputs to MuJoCo, then continues with step 1.

The engine must not advance between replies. Engine time is the session
clock: `frame * dt` from the `welcome`. Clients reject a state whose time is
not exactly one period after the previous one.

## Framing

Each message is one or more frames. A frame is a 4-byte little-endian
unsigned length followed by that many bytes. Frames above 256 MiB are
invalid.

- The first frame is a UTF-8 JSON object with a string `type`.
- If the header has `"attachments": [{"name", "dtype", "shape"}, ...]`, one
  binary frame follows per attachment, in order. Each holds C-order,
  little-endian data of `dtype` (`uint8`, `uint16`, `int32`, `float32`,
  `float64`) and `shape`. The receiver checks the byte counts.
- NaN and infinities are not JSON. They travel as `null`, and clients read a
  `null` number as NaN (for example `screen.u` while nothing touches).

## Messages

| Client sends | Fields | Server replies |
| --- | --- | --- |
| `hello` | `protocol`, `dt`, `controls` (names), `outputs` (names the client consumes), `manifest` (list of interface manifests) | `welcome` or `error` |
| `describe` (may come first, on its own connection) | `protocol` | `description` |
| `step` | `frame` (the server's current frame), `time`, `controls` (name to number or null), `events` (list) | `state` or `error` |
| `reset` | `seed` (integer or null), `options` (object) | `state` or `error` |
| `close` | none | `bye` |

Reply fields:

- **`welcome`:** `protocol`, `engine` (a name), `dt` (must equal the
  requested one), `convention` (`unity`, `unreal`, `carla` or `flu`),
  `frame`, `time` and `outputs`.
- **`state`:** `frame`, `time` and `outputs`. Attachments carry images and
  other arrays.
- **`description`:** `protocol`, `engine`, `convention` and `manifest` (an
  `embodied-ui/manifest`), with mesh files as `uint8` attachments named by
  relative path, e.g. `meshes/start.obj`.
- **`error`:** `message`. The server keeps the connection and waits for the
  next request.

A `step` whose `frame` differs from the server's frame count must be
rejected. A `reset` returns frame 0 and time 0. It may let the engine run
one internal frame to apply the reset, but must not report it.

An event is `{"time", "source", "kind", "value", "data"}`:

| `kind` | `value` | `data` |
| --- | --- | --- |
| `press`, `release` | 1 or 0 | none |
| `toggle` | new state | none |
| `detent` | detent index | `delta` |
| `touch_begin`, `touch_move`, `touch_end` | touch id | `u` (0 left to 1 right), `v` (0 top to 1 bottom), `force` in N |

The protocol version is `embodied-ui/1`. Peers accept the same major version
(`embodied-ui/1.x`) and reject others with an `error`.

## Output conventions

- A **pose** is `[x, y, z, qw, qx, qy, qz]` in the announced `convention`,
  relative to the engine's reference object. The client converts it.
- An **image** is an H x W x 3 `uint8` attachment, RGB, top row first.
- **Device outputs** are named `device.output`: `lamp.on`, `screen.image`,
  `wheel.torque` (N m), `slider.setpoint`, `cup.pose`.
- The **vehicle schema**, shared by `CarlaEngine`, the Unity package and
  `myosuite_carla.interfaces.EngineVehicleSession`, is in the convention's
  axes and SI units, for the world or the reference object:

  | Output | Content |
  | --- | --- |
  | `vehicle.pose` | pose |
  | `vehicle.velocity` | m/s |
  | `vehicle.acceleration` | m/s² |
  | `vehicle.angular_velocity` | rad/s; the convention's handedness applies to the rates |
  | `vehicle.speed` | m/s |

  Vehicle controls are `vehicle.throttle`, `vehicle.brake` (0 to 1),
  `vehicle.steer` (-1 to 1, positive to the right), `vehicle.gear`, and the
  booleans `vehicle.hand_brake`, `vehicle.reverse` and
  `vehicle.manual_gear_shift` (≥ 0.5 is true).

## Conformance

A server conforms if the Python test pattern in
`tests/test_dotnet_interop.py` passes against it:

1. `describe` returns a manifest.
2. A bridge drives 40 frames with exact frame numbers and times.
3. Events reach the engine.
4. Outputs (a lamp state, a vehicle speed, an image attachment) arrive.

Run your server, then:

```python
from embodied_ui.backends import RemoteEngine, describe
interface = describe(port=PORT)          # if you implement describe
engine = RemoteEngine(port=PORT, dt=0.02)
```

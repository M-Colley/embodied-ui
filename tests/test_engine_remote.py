import math
import socket
import struct
import threading

import mujoco
import numpy as np
import pytest

from embodied_ui import (
    Indicator,
    Pose,
    Prop,
    PushButton,
    Screen,
    SteeringWheel,
    TouchSurface,
)
from embodied_ui.backends import EngineServer, LoopbackEngine, RemoteEngine
from embodied_ui.engine import Bridge, Command, Route, Routes, SynchronizationError
from embodied_ui.protocol import (
    ProtocolError,
    RemoteError,
    receive_message,
    send_message,
)

from .conftest import build_scene

DT = 0.02


def devices():
    return [
        PushButton("b"),
        SteeringWheel("wheel", Pose((0.5, 0, 0))),
        Indicator("lamp", Pose((0.2, 0, 0))),
        Screen("screen", Pose((0, 0.3, 0)), resolution=(8, 4)),
        TouchSurface("touch", Pose((0, -0.3, 0))),
        Prop("cup", Pose((0, 0, 0.5))),
    ]


def advance(scene, bridge):
    scene.runtime.step(scene.data, round(DT / scene.model.opt.timestep))
    return bridge.step(scene.data)


def test_passthrough_bridge_sends_every_channel_and_applies_outputs():
    scene = build_scene(devices())

    def update(command, time):
        return {
            "lamp.on": float(time > 0.05),
            "wheel.torque": 0.2,
            "echo": command.controls["b.value"],
        }

    engine = LoopbackEngine(DT, update)
    with Bridge(scene.runtime, engine) as bridge:
        start = bridge.reset(scene.data)
        assert start.command is None and engine.info is not None
        assert set(engine.info.controls) == set(scene.runtime.channels)
        assert "screen.image" in engine.info.outputs
        assert engine.info.manifest[0]["name"] == "ui"
        frames = [advance(scene, bridge) for _ in range(5)]
    assert [f.state.frame for f in frames] == [1, 2, 3, 4, 5]
    assert frames[-1].command.controls["b.value"] == pytest.approx(0.0, abs=0.01)
    lamp = scene.model.geom("ui/lamp/static0").id
    np.testing.assert_allclose(scene.model.geom_rgba[lamp], Indicator("x").on)
    assert scene.qpos("wheel", "angle") > 0
    assert engine.closed


def test_routes_scale_clip_rename_events_and_feed_back():
    scene = build_scene(devices(), probe_pos=(0, -0.3, 0.03))
    routes = Routes(
        routes=(
            Route("wheel.value", "vehicle.steer", scale=-1.0),
            Route("b.value", "vehicle.brake", scale=2.0, clip=(0.0, 1.0)),
        ),
        events=(("touch", "ui.panel"),),
        feedback=(Route("lights.left_blinker", "lamp.on"),),
    )
    engine = LoopbackEngine(DT, lambda c, t: {"lights.left_blinker": 1.0})
    bridge = Bridge(scene.runtime, engine, routes)
    bridge.reset(scene.data)
    scene.set_qpos("wheel", "angle", 1.0)
    frames = []
    for k in range(15):
        scene.probe.target(scene.data, (0, -0.3, 0.006 if k > 2 else 0.03))
        frames.append(advance(scene, bridge))
    command = frames[0].command
    assert set(command.controls) == {"vehicle.steer", "vehicle.brake"}
    assert command.controls["vehicle.steer"] < 0
    begins = [e for f in frames for e in f.command.events if e.kind == "touch_begin"]
    assert begins and begins[0].source == "ui.panel"
    lamp = scene.model.geom("ui/lamp/static0").id
    np.testing.assert_allclose(scene.model.geom_rgba[lamp], Indicator("x").on)
    assert engine.info.controls == ("vehicle.steer", "vehicle.brake")


def test_engine_poses_are_converted_from_the_engine_frame():
    scene = build_scene(devices())
    # Unity: 1 m forward, 0.2 m up.
    engine = LoopbackEngine(DT, lambda c, t: {"cup.pose": [0, 0.2, 1, 1, 0, 0, 0]})
    bridge = Bridge(scene.runtime, engine, engine_frame="unity")
    bridge.reset(scene.data)
    advance(scene, bridge)
    mujoco.mj_forward(scene.model, scene.data)
    np.testing.assert_allclose(
        scene.data.xpos[scene.model.body("ui/cup").id], (1, 0, 0.2), atol=1e-12
    )


def test_clock_mismatches_are_detected():
    scene = build_scene(devices())
    bridge = Bridge(scene.runtime, LoopbackEngine(DT))
    with pytest.raises(RuntimeError):
        bridge.step(scene.data)
    bridge.reset(scene.data)
    scene.runtime.step(scene.data, 5)  # half a period
    with pytest.raises(SynchronizationError):
        bridge.step(scene.data)
    with pytest.raises(RuntimeError):  # failed bridges must be reset
        bridge.step(scene.data)

    class Slow(LoopbackEngine):
        def step(self, command):
            state = super().step(command)
            return type(state)(state.frame, state.time * 2, state.outputs)

    bridge = Bridge(scene.runtime, Slow(DT))
    bridge.reset(scene.data)
    with pytest.raises(SynchronizationError):
        advance(scene, bridge)


def test_bridge_over_tcp_carries_images_and_errors():
    scene = build_scene(devices())
    image = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)

    def update(command, time):
        if command.controls.get("b.pressed") is None:
            raise AssertionError("controls missing")
        return {"screen.image": image, "lamp.on": 1.0, "touch.u": math.nan}

    server = EngineServer(
        LoopbackEngine(DT, update), port=0, convention="flu", name="loopback"
    ).start_background()
    remote = RemoteEngine(port=server.port, dt=DT, timeout=5)
    bridge = Bridge(scene.runtime, remote)
    bridge.reset(scene.data)
    assert remote.engine_name == "loopback" and remote.convention == "flu"
    frame = advance(scene, bridge)
    assert frame.state.frame == 1 and frame.state.time == pytest.approx(DT)
    np.testing.assert_array_equal(frame.state.outputs["screen.image"], image)
    assert math.isnan(frame.state.outputs["touch.u"])
    np.testing.assert_array_equal(scene.runtime.image("screen"), image)
    bridge.close()
    server.join(5)
    assert server.error is None


def test_remote_engine_errors_and_version_checks():
    server = EngineServer(LoopbackEngine(DT), port=0).start_background()
    remote = RemoteEngine(port=server.port, dt=0.01, timeout=5)
    from embodied_ui.engine import SessionInfo

    with pytest.raises(RemoteError, match="dt"):
        remote.start(SessionInfo(0.01))
    server.join(5)

    server = EngineServer(LoopbackEngine(DT), port=0).start_background()
    with socket.create_connection(("127.0.0.1", server.port), timeout=5) as sock:
        send_message(sock, {"type": "hello", "protocol": "embodied-ui/2", "dt": DT})
        reply, _ = receive_message(sock)
        assert reply["type"] == "error" and "embodied-ui/1" in reply["message"]
        send_message(sock, {"type": "step"})
        reply, _ = receive_message(sock)
        assert reply["type"] == "error" and "hello" in reply["message"]
        send_message(sock, {"type": "close"})
        assert receive_message(sock)[0]["type"] == "bye"
    server.join(5)


def test_framing_rejects_oversized_and_malformed_frames():
    left, right = socket.socketpair()
    with left, right:
        left.sendall(struct.pack("<I", 2**31))
        with pytest.raises(ProtocolError):
            receive_message(right)
    left, right = socket.socketpair()
    with left, right:
        payload = b"[1, 2]"
        left.sendall(struct.pack("<I", len(payload)) + payload)
        with pytest.raises(ProtocolError):
            receive_message(right)
    left, right = socket.socketpair()
    with left, right:
        array = np.ones((2, 3), np.float32)
        thread = threading.Thread(
            target=send_message, args=(left, {"type": "x", "v": math.inf}, {"a": array})
        )
        thread.start()
        header, arrays = receive_message(right)
        thread.join()
        assert header["v"] is None
        np.testing.assert_array_equal(arrays["a"], array)


def test_command_serialization_replaces_nan():
    command = Command(0.1, {"u": math.nan, "b": 1.0})
    assert command.as_dict()["controls"] == {"u": None, "b": 1.0}

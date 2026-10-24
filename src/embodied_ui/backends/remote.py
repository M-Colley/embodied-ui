"""Engines in another process or on another machine, over TCP.

:class:`RemoteEngine` is the client for any server of the ``embodied-ui/1``
protocol: the Unity package, the reference :class:`EngineServer` (which
serves any Python :class:`~embodied_ui.engine.Engine`, including CARLA), or
an implementation for another engine.
"""

from __future__ import annotations

import math
import re
import socket
import tempfile
import threading
import time as _time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..engine import Command, Engine, EngineState, SessionInfo
from ..interface import Interface
from ..logic import Event
from ..protocol import (
    DEFAULT_PORT,
    PROTOCOL,
    ProtocolError,
    check_protocol,
    expect,
    names,
    outputs_from,
    receive_message,
    send_message,
    split_outputs,
)


class RemoteEngine:
    """Lockstep client of an engine server.

    Args:
        host, port: Server address.
        dt: Proposed period; the server must confirm it.
        timeout: Deadline for each reply, s. Rendering engines may need
            several seconds for their first frame.
        connect_timeout: How long to retry connecting while the engine
            starts, s.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        *,
        dt: float,
        timeout: float = 30.0,
        connect_timeout: float = 60.0,
    ) -> None:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.host, self.port = host, int(port)
        self.dt = float(dt)
        self.timeout = float(timeout)
        self.connect_timeout = float(connect_timeout)
        self.engine_name = ""
        self.convention = "flu"
        self.welcome: dict[str, Any] = {}
        self._socket: socket.socket | None = None
        self._frame = 0

    def _connect(self) -> socket.socket:
        deadline = _time.monotonic() + self.connect_timeout
        while True:
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=self.timeout
                )
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return sock
            except OSError:
                if _time.monotonic() >= deadline:
                    raise
                _time.sleep(0.25)

    def _request(
        self, header: Mapping[str, Any], *types: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._socket is None:
            raise RuntimeError("start the remote engine first")
        send_message(self._socket, header)
        reply, arrays = receive_message(self._socket)
        expect(reply, *types)
        return reply, arrays

    def _state(
        self, reply: Mapping[str, Any], arrays: Mapping[str, Any]
    ) -> EngineState:
        frame = int(reply["frame"])
        self._frame = frame
        return EngineState(frame, float(reply["time"]), outputs_from(reply, arrays))

    def start(self, info: SessionInfo) -> EngineState:
        if self._socket is not None:
            raise RuntimeError("the remote engine is already started")
        if not math.isclose(info.dt, self.dt, rel_tol=1e-9):
            raise ValueError("session dt differs from the engine's")
        self._socket = self._connect()
        try:
            reply, arrays = self._request(
                {
                    "type": "hello",
                    "protocol": PROTOCOL,
                    "dt": self.dt,
                    "controls": list(info.controls),
                    "outputs": list(info.outputs),
                    "manifest": list(info.manifest),
                },
                "welcome",
            )
            check_protocol(reply.get("protocol"))
            if not math.isclose(
                float(reply.get("dt", math.nan)), self.dt, rel_tol=1e-9
            ):
                raise ProtocolError(
                    f"engine runs at dt={reply.get('dt')}, requested {self.dt}"
                )
            self.engine_name = str(reply.get("engine", ""))
            self.convention = str(reply.get("convention", "flu"))
            self.welcome = reply
            return self._state(reply, arrays)
        except BaseException:
            self._drop()
            raise

    def reset(
        self, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> EngineState:
        return self._state(
            *self._request(
                {"type": "reset", "seed": seed, "options": dict(options or {})}, "state"
            )
        )

    def step(self, command: Command) -> EngineState:
        payload = command.as_dict()
        payload.update(type="step", frame=self._frame)
        return self._state(*self._request(payload, "state"))

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._request({"type": "close"}, "bye")
        except (OSError, ProtocolError, ConnectionError):
            pass
        finally:
            self._drop()

    def _drop(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None


_ASSET_NAME = re.compile(r"^[A-Za-z0-9_\-][A-Za-z0-9_.\-/]*$")


def describe(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    *,
    target: str = "flu",
    directory: str | Path | None = None,
    timeout: float = 30.0,
    connect_timeout: float = 60.0,
) -> Interface:
    """Ask a running engine for the devices designed in it.

    The engine replies with an ``embodied-ui/manifest`` and mesh files as
    attachments; meshes are written to ``directory`` (a new temporary
    directory by default) and the manifest is converted to ``target``.
    """
    client = RemoteEngine(
        host, port, dt=1.0, timeout=timeout, connect_timeout=connect_timeout
    )
    client._socket = client._connect()
    try:
        reply, arrays = client._request(
            {"type": "describe", "protocol": PROTOCOL}, "description"
        )
        check_protocol(reply.get("protocol"))
        folder = (
            Path(directory)
            if directory is not None
            else Path(tempfile.mkdtemp(prefix="embodied-ui-"))
        )
        folder.mkdir(parents=True, exist_ok=True)
        for name, array in arrays.items():
            if not _ASSET_NAME.match(name) or ".." in name.split("/"):
                raise ProtocolError(f"refusing to write asset {name!r}")
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(array.astype("uint8", copy=False).tobytes())
        return Interface.from_manifest(reply["manifest"], target=target, assets=folder)
    finally:
        client.close()


class EngineServer:
    """Serve a Python engine over the protocol, one client at a time.

    Use it to run CARLA or a Python task on another machine, or as the
    reference implementation when writing a server for another engine.

    Args:
        engine: Engine to serve.
        host, port: Listening address; port 0 picks a free port.
        convention: Pose convention announced to clients.
        name: Engine name announced to clients.
    """

    def __init__(
        self,
        engine: Engine,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        *,
        convention: str = "flu",
        name: str = "python",
    ) -> None:
        self.engine = engine
        self.convention = convention
        self.name = name
        self._listener = socket.create_server((host, port))
        self.port = int(self._listener.getsockname()[1])
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def serve_one(self) -> None:
        """Accept one client and serve it until it closes."""
        connection, _ = self._listener.accept()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        started = False
        with connection:
            while True:
                try:
                    header, _ = receive_message(connection)
                except ConnectionError:
                    break
                kind = header["type"]
                try:
                    if kind == "hello":
                        check_protocol(header.get("protocol"))
                        dt = float(header["dt"])
                        if not math.isclose(dt, self.engine.dt, rel_tol=1e-9):
                            raise ProtocolError(
                                f"this engine runs at dt={self.engine.dt}, not {dt}"
                            )
                        state = self.engine.start(
                            SessionInfo(
                                dt,
                                names(header.get("controls", [])),
                                names(header.get("outputs", [])),
                                tuple(header.get("manifest", [])),
                            )
                        )
                        started = True
                        self._reply(connection, "welcome", state)
                    elif kind == "step":
                        self._require(started)
                        controls = {
                            str(k): (math.nan if v is None else float(v))
                            for k, v in dict(header.get("controls") or {}).items()
                        }
                        events = tuple(
                            Event.from_dict(e) for e in header.get("events") or []
                        )
                        state = self.engine.step(
                            Command(float(header.get("time", 0.0)), controls, events)
                        )
                        self._reply(connection, "state", state)
                    elif kind == "reset":
                        self._require(started)
                        state = self.engine.reset(
                            header.get("seed"), header.get("options")
                        )
                        self._reply(connection, "state", state)
                    elif kind == "describe":
                        check_protocol(header.get("protocol"))
                        manifest = getattr(self.engine, "manifest", None)
                        if manifest is None:
                            raise ProtocolError("this engine describes no devices")
                        send_message(
                            connection,
                            {
                                "type": "description",
                                "protocol": PROTOCOL,
                                "engine": self.name,
                                "convention": self.convention,
                                "manifest": manifest,
                            },
                        )
                    elif kind == "close":
                        send_message(connection, {"type": "bye"})
                        break
                    else:
                        raise ProtocolError(f"unknown message type {kind!r}")
                except Exception as error:  # report to the client, keep serving
                    self.error = error
                    send_message(
                        connection,
                        {
                            "type": "error",
                            "message": f"{type(error).__name__}: {error}",
                        },
                    )
        if started:
            self.engine.close()

    def _require(self, started: bool) -> None:
        if not started:
            raise ProtocolError("send hello first")

    def _reply(self, connection: socket.socket, kind: str, state: EngineState) -> None:
        values, arrays = split_outputs(state.outputs)
        header: dict[str, Any] = {
            "type": kind,
            "frame": state.frame,
            "time": state.time,
            "outputs": values,
        }
        if kind == "welcome":
            header.update(
                protocol=PROTOCOL,
                engine=self.name,
                dt=self.engine.dt,
                convention=self.convention,
            )
        send_message(connection, header, arrays)

    def start_background(self) -> EngineServer:
        """Serve one client on a daemon thread and return immediately."""

        def run() -> None:
            try:
                self.serve_one()
            except BaseException as error:  # pragma: no cover - surfaced via .error
                self.error = error

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def join(self, timeout: float | None = None) -> None:
        """Wait for the background client to finish."""
        if self._thread is not None:
            self._thread.join(timeout)

    def close(self) -> None:
        """Stop listening."""
        self._listener.close()

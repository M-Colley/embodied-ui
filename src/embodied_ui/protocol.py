"""Wire format of the ``embodied-ui/1`` lockstep protocol.

Every message is a frame: a 4-byte little-endian unsigned length followed by
that many bytes. The first frame of a message is a UTF-8 JSON object with a
``type``; binary attachments (camera images, display frames) follow as one
further frame each, described in the header's ``attachments`` list. NaN and
infinities are not valid JSON and travel as ``null``.

The client (the MuJoCo side) connects to the engine, which listens:

=========  ===========================================  =================
client     fields                                       server reply
=========  ===========================================  =================
hello      protocol, dt, controls, outputs, manifest    welcome | error
describe   protocol                                     description
step       frame, time, controls, events                state | error
reset      seed, options                                state | error
close      -                                            bye
=========  ===========================================  =================

``describe`` may precede ``hello`` on a separate connection: the engine
returns the devices designed in it as an ``embodied-ui/manifest`` with mesh
files as ``uint8`` attachments named by relative path.

``welcome`` and ``state`` carry ``frame``, ``time`` and ``outputs``;
``welcome`` also names the ``engine``, its ``dt`` and the ``convention``
of its poses (``unity``, ``unreal``, ``carla`` or ``flu``). Poses are
``[x, y, z, qw, qx, qy, qz]`` in that convention.
"""

from __future__ import annotations

import json
import math
import socket
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

PROTOCOL = "embodied-ui/1"
DEFAULT_PORT = 47474
MAX_FRAME = 256 * 1024 * 1024
_LENGTH = struct.Struct("<I")
_DTYPES = {"uint8", "uint16", "int32", "float32", "float64"}


class ProtocolError(RuntimeError):
    """A peer sent something this protocol does not allow."""


class RemoteError(RuntimeError):
    """The engine reported an error."""


def sanitize(value: Any) -> Any:
    """Replace non-finite floats by ``None`` and NumPy scalars by Python ones."""
    if isinstance(value, Mapping):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _send_frame(sock: socket.socket, payload: bytes | memoryview) -> None:
    if len(payload) > MAX_FRAME:
        raise ProtocolError(f"frame of {len(payload)} bytes exceeds {MAX_FRAME}")
    sock.sendall(_LENGTH.pack(len(payload)))
    sock.sendall(payload)


def _receive_exact(sock: socket.socket, size: int) -> bytearray:
    buffer = bytearray(size)
    view = memoryview(buffer)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:], size - received)
        if count == 0:
            raise ConnectionError("peer closed the connection")
        received += count
    return buffer


def _receive_frame(sock: socket.socket) -> bytearray:
    (size,) = _LENGTH.unpack(_receive_exact(sock, _LENGTH.size))
    if size > MAX_FRAME:
        raise ProtocolError(f"peer announced a frame of {size} bytes")
    return _receive_exact(sock, size)


def send_message(
    sock: socket.socket,
    header: Mapping[str, Any],
    attachments: Mapping[str, NDArray[Any]] | None = None,
) -> None:
    """Send a JSON header and its binary attachments."""
    payload = dict(sanitize(header))
    arrays: list[NDArray[Any]] = []
    if attachments:
        descriptors = []
        for name, array in attachments.items():
            array = np.ascontiguousarray(array)
            if array.dtype.name not in _DTYPES:
                raise ProtocolError(
                    f"attachment {name!r} has unsupported dtype {array.dtype}"
                )
            array = array.astype(array.dtype.newbyteorder("<"), copy=False)
            descriptors.append(
                {"name": name, "dtype": array.dtype.name, "shape": list(array.shape)}
            )
            arrays.append(array)
        payload["attachments"] = descriptors
    _send_frame(
        sock, json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    )
    for array in arrays:
        _send_frame(sock, array.data.cast("B"))


def receive_message(
    sock: socket.socket,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Receive one message; return its header and attachments by name."""
    try:
        header = json.loads(_receive_frame(sock).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"invalid JSON header: {error}") from error
    if not isinstance(header, dict) or not isinstance(header.get("type"), str):
        raise ProtocolError("message headers must be objects with a string 'type'")
    arrays: dict[str, NDArray[Any]] = {}
    for descriptor in header.pop("attachments", None) or []:
        dtype = str(descriptor.get("dtype"))
        if dtype not in _DTYPES:
            raise ProtocolError(f"unsupported attachment dtype {dtype!r}")
        shape = tuple(int(n) for n in descriptor.get("shape", ()))
        data = _receive_frame(sock)
        expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if len(data) != expected:
            raise ProtocolError(
                f"attachment {descriptor.get('name')!r} has {len(data)} bytes; "
                f"shape {shape} needs {expected}"
            )
        arrays[str(descriptor["name"])] = np.frombuffer(
            bytes(data), dtype=np.dtype(dtype).newbyteorder("<")
        ).reshape(shape)
    return header, arrays


def expect(header: Mapping[str, Any], *types: str) -> None:
    """Raise if a reply is an error or of an unexpected type."""
    kind = header.get("type")
    if kind == "error":
        raise RemoteError(str(header.get("message", "unspecified engine error")))
    if kind not in types:
        raise ProtocolError(f"expected {' or '.join(types)}, got {kind!r}")


def check_protocol(value: Any) -> None:
    """Accept the same major protocol version."""
    if not isinstance(value, str) or value.split(".")[0] != PROTOCOL:
        raise ProtocolError(f"peer speaks {value!r}; this side speaks {PROTOCOL}")


def outputs_from(
    header: Mapping[str, Any], arrays: Mapping[str, NDArray[Any]]
) -> dict[str, Any]:
    """Merge JSON outputs and attachments; ``null`` becomes NaN."""
    outputs: dict[str, Any] = {}
    for key, value in dict(header.get("outputs") or {}).items():
        outputs[str(key)] = math.nan if value is None else value
    outputs.update(arrays)
    return outputs


def split_outputs(
    outputs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Separate array outputs (sent as attachments) from JSON values."""
    values: dict[str, Any] = {}
    arrays: dict[str, NDArray[Any]] = {}
    for key, value in outputs.items():
        if (
            isinstance(value, np.ndarray)
            and value.dtype.name in _DTYPES
            and value.ndim >= 2
        ):
            arrays[key] = value
        else:
            values[key] = value
    return values, arrays


def names(values: Sequence[Any]) -> tuple[str, ...]:
    """Validate a list of names from a peer."""
    if not isinstance(values, (list, tuple)) or not all(
        isinstance(v, str) for v in values
    ):
        raise ProtocolError("expected a list of names")
    return tuple(values)

"""An in-process engine: tests, offline studies and Python-made tasks."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from ..engine import Command, EngineState, SessionInfo

#: ``update(command, time) -> outputs`` for one frame.
Update = Callable[[Command, float], Mapping[str, Any]]


class LoopbackEngine:
    """A lockstep engine implemented in Python.

    Without ``update`` it echoes every control as an output, which makes
    routing visible in tests. With ``update`` it runs Python-implemented
    tasks, e.g. a secondary task that draws a display and reacts to touches.

    Args:
        dt: Fixed period, s.
        update: Function from the command and new engine time to outputs.
        reset: Function returning initial outputs for a seed.
        manifest: Devices this engine describes to ``describe`` requests.
    """

    def __init__(
        self,
        dt: float,
        update: Update | None = None,
        reset: Callable[[int | None], Mapping[str, Any]] | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.dt = float(dt)
        self._update = update
        self._reset = reset
        self.manifest = dict(manifest) if manifest is not None else None
        self.info: SessionInfo | None = None
        self.commands: list[Command] = []
        self.frame = 0
        self.closed = False

    def start(self, info: SessionInfo) -> EngineState:
        if not math.isclose(info.dt, self.dt, rel_tol=1e-9):
            raise ValueError("session dt differs from the engine's")
        self.info = info
        return self.reset()

    def reset(
        self, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> EngineState:
        self.frame = 0
        self.commands = []
        outputs = dict(self._reset(seed)) if self._reset else {}
        return EngineState(0, 0.0, outputs)

    def step(self, command: Command) -> EngineState:
        if self.info is None or self.closed:
            raise RuntimeError("start the engine before stepping")
        self.commands.append(command)
        self.frame += 1
        time = self.frame * self.dt
        outputs = (
            dict(self._update(command, time))
            if self._update
            else dict(command.controls)
        )
        return EngineState(self.frame, time, outputs)

    def close(self) -> None:
        self.closed = True

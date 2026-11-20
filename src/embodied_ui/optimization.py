"""Parametric UI layout search spaces, rollout metrics, and BO/MOBO study harness."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .devices import Device, PushButton, TouchSurface
from .frames import FLU, Frame, Pose, _finite
from .geometry import Box, Cylinder, Part, Sphere
from .interface import Contacts, Interface
from .layout import PanelLayout, panel_pose


@dataclass(frozen=True)
class ParameterBound:
    """Continuous decision variable bound ``[low, high]``."""

    key: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("ParameterBound.key must be non-empty")
        lo = _finite(self.low, f"{self.key}.low")
        hi = _finite(self.high, f"{self.key}.high")
        if lo >= hi:
            raise ValueError(f"{self.key} requires low < high, got ({lo}, {hi})")
        object.__setattr__(self, "low", lo)
        object.__setattr__(self, "high", hi)

    def to_wire(self, order: int) -> dict[str, Any]:
        return {
            "key": self.key,
            "init": {"low": self.low, "high": self.high},
            "optSeqOrder": int(order),
        }


@dataclass(frozen=True)
class ObjectiveBound:
    """Bounded scalar objective ``[low, high]`` with optimization direction."""

    key: str
    low: float
    high: float
    minimize: bool = False

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("ObjectiveBound.key must be non-empty")
        lo = _finite(self.low, f"{self.key}.low")
        hi = _finite(self.high, f"{self.key}.high")
        if lo >= hi:
            raise ValueError(f"{self.key} requires low < high, got ({lo}, {hi})")
        object.__setattr__(self, "low", lo)
        object.__setattr__(self, "high", hi)

    def clamp(self, value: float) -> float:
        v = _finite(value, self.key)
        return min(self.high, max(self.low, v))

    def to_wire(self, order: int) -> dict[str, Any]:
        return {
            "key": self.key,
            "init": {
                "low": self.low,
                "high": self.high,
                "minimize": 1 if self.minimize else 0,
            },
            "optSeqOrder": int(order),
        }


def _scale_part_xy(part: Part, scale: float) -> Part:
    shape = part.shape
    if isinstance(shape, Box):
        hx, hy, hz = shape.half_size
        new_shape: Any = Box((hx * scale, hy * scale, hz))
    elif isinstance(shape, Cylinder):
        new_shape = Cylinder(shape.radius * scale, shape.half_length)
    elif isinstance(shape, Sphere):
        new_shape = Sphere(shape.radius * scale)
    else:
        return part
    return dataclasses.replace(part, shape=new_shape)


@dataclass(frozen=True)
class LayoutCandidate:
    """Concrete 3D panel placement and target-scale configuration."""

    distance: float
    azimuth: float = 0.0
    elevation: float = 0.0
    tilt: float = 0.0
    roll: float = 0.0
    target_scale: float = 1.0

    def __post_init__(self) -> None:
        d = _finite(self.distance, "distance")
        ts = _finite(self.target_scale, "target_scale")
        if d <= 0.0:
            raise ValueError("distance must be positive")
        if ts <= 0.0:
            raise ValueError("target_scale must be positive")
        for label in ("azimuth", "elevation", "tilt", "roll"):
            _finite(getattr(self, label), label)

    def pose(
        self,
        anchor: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        host_frame: Frame | str = FLU,
    ) -> Pose:
        """Compute panel :class:`Pose` facing ``anchor`` in ``host_frame``."""
        return panel_pose(
            anchor,
            distance=self.distance,
            azimuth=self.azimuth,
            elevation=self.elevation,
            tilt=self.tilt,
            roll=self.roll,
            host_frame=host_frame,
        )

    def panel_layout(
        self,
        anchor: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        host_frame: Frame | str = FLU,
        mount: str = "world",
    ) -> PanelLayout:
        """Build a :class:`PanelLayout` for this candidate's pose."""
        return PanelLayout(self.pose(anchor, host_frame=host_frame), mount=mount)

    def scale_device(self, device: Device) -> Device:
        """Scale interactive surface/cap lateral dimensions by ``target_scale``."""
        s = self.target_scale
        if abs(s - 1.0) <= 1e-12:
            return device
        if isinstance(device, TouchSurface):
            w, h = device.size
            return dataclasses.replace(device, size=(w * s, h * s))
        if isinstance(device, PushButton):
            base_cap = device.cap or Part(
                Box((0.0075, 0.0075, 0.003)),
                Pose((0.0, 0.0, 0.003)),
                (0.35, 0.36, 0.38, 1.0),
                mass=device.mass,
            )
            return dataclasses.replace(device, cap=_scale_part_xy(base_cap, s))
        return device

    def build_interface(
        self,
        name: str,
        placements: Sequence[Device | tuple[Device, float, float]],
        *,
        anchor: Sequence[float] = (0.0, 0.0, 0.0),
        host_frame: Frame | str = FLU,
        mount: str = "world",
        scale_offsets: bool = True,
        contacts: Contacts = Contacts(),
        sensors: bool = True,
    ) -> Interface:
        """Mount and scale ``placements`` onto a compiled :class:`Interface`."""
        layout = self.panel_layout(anchor, host_frame=host_frame, mount=mount)
        offset_factor = self.target_scale if scale_offsets else 1.0
        scaled_entries: list[Device | tuple[Device, float, float]] = []
        for item in placements:
            if isinstance(item, Device):
                scaled_entries.append(self.scale_device(item))
            else:
                dev, u_off, v_off = item
                scaled_entries.append(
                    (
                        self.scale_device(dev),
                        float(u_off) * offset_factor,
                        float(v_off) * offset_factor,
                    )
                )
        return layout.interface(
            name,
            scaled_entries,
            contacts=contacts,
            sensors=sensors,
        )


@dataclass(frozen=True)
class LayoutSpace:
    """Parametric search space over 3D panel pose and target scale."""

    distance: tuple[float, float] = (0.35, 0.75)
    azimuth: tuple[float, float] | None = (-0.52, 0.52)
    elevation: tuple[float, float] | None = (-0.35, 0.25)
    tilt: tuple[float, float] | None = (0.0, 0.78)
    roll: tuple[float, float] | None = None
    target_scale: tuple[float, float] | None = (0.75, 1.50)
    fixed_azimuth: float = 0.0
    fixed_elevation: float = 0.0
    fixed_tilt: float = 0.0
    fixed_roll: float = 0.0
    fixed_target_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.distance[0] <= 0.0:
            raise ValueError("distance lower bound must be positive")
        if self.target_scale is not None and self.target_scale[0] <= 0.0:
            raise ValueError("target_scale lower bound must be positive")
        # Validate all active bounds via ParameterBound.
        self.parameter_bounds()

    def parameter_bounds(self) -> tuple[ParameterBound, ...]:
        """Return ordered :class:`ParameterBound` specs for active dimensions."""
        specs: list[ParameterBound] = [
            ParameterBound("distance", self.distance[0], self.distance[1])
        ]
        for key in ("azimuth", "elevation", "tilt", "roll", "target_scale"):
            rng = getattr(self, key)
            if rng is not None:
                specs.append(ParameterBound(key, rng[0], rng[1]))
        return tuple(specs)

    def midpoint(self) -> LayoutCandidate:
        """Return the midpoint :class:`LayoutCandidate` of the search box."""
        mid = {b.key: 0.5 * (b.low + b.high) for b in self.parameter_bounds()}
        return self.decode(mid)

    def decode(self, values: Mapping[str, float]) -> LayoutCandidate:
        """Validate and decode a parameter mapping into a :class:`LayoutCandidate`."""
        bounds = {b.key: b for b in self.parameter_bounds()}
        unknown = set(values) - set(bounds)
        missing = set(bounds) - set(values)
        if missing or unknown:
            raise ValueError(
                f"Parameter keys mismatch: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        checked: dict[str, float] = {}
        for key, spec in bounds.items():
            v = _finite(values[key], key)
            if not (spec.low - 1e-9 <= v <= spec.high + 1e-9):
                raise ValueError(
                    f"Parameter {key}={v} outside [{spec.low}, {spec.high}]"
                )
            checked[key] = min(spec.high, max(spec.low, v))

        return LayoutCandidate(
            distance=checked["distance"],
            azimuth=checked.get("azimuth", self.fixed_azimuth),
            elevation=checked.get("elevation", self.fixed_elevation),
            tilt=checked.get("tilt", self.fixed_tilt),
            roll=checked.get("roll", self.fixed_roll),
            target_scale=checked.get("target_scale", self.fixed_target_scale),
        )


@dataclass(frozen=True)
class ErgonomicRolloutMetrics:
    """Multi-objective performance, biomechanical effort, and comfort summary."""

    throughput_bps: float
    mean_effort: float
    peak_fatigue: float = 0.0
    mean_neck_moment_nm: float = 0.0
    motion_sickness_index: float = 0.0
    completion_rate: float = 1.0

    @classmethod
    def from_rollout(
        cls,
        *,
        duration_s: float,
        index_of_difficulty_bits: Sequence[float] = (),
        activations: np.ndarray | Sequence[Sequence[float]] | None = None,
        fatigue: np.ndarray | Sequence[Sequence[float]] | None = None,
        neck_moments_nm: np.ndarray | Sequence[float] | None = None,
        motion_sickness_index: float = 0.0,
        completion_rate: float = 1.0,
    ) -> ErgonomicRolloutMetrics:
        """Aggregate step-level rollout traces into scalar optimization metrics."""
        dur = _finite(duration_s, "duration_s")
        if dur <= 0.0:
            raise ValueError("duration_s must be positive")
        bits_sum = float(np.sum(np.asarray(index_of_difficulty_bits, dtype=float)))
        tp = max(0.0, bits_sum / dur)

        effort = 0.0
        if activations is not None:
            act_arr = np.asarray(activations, dtype=float)
            if act_arr.size > 0:
                effort = float(np.mean(np.square(act_arr)))

        fat_peak = 0.0
        if fatigue is not None:
            fat_arr = np.asarray(fatigue, dtype=float)
            if fat_arr.size > 0:
                fat_peak = float(np.max(fat_arr))

        neck_mean = 0.0
        if neck_moments_nm is not None:
            neck_arr = np.asarray(neck_moments_nm, dtype=float)
            if neck_arr.size > 0:
                neck_mean = float(np.mean(np.abs(neck_arr)))

        return cls(
            throughput_bps=tp,
            mean_effort=effort,
            peak_fatigue=fat_peak,
            mean_neck_moment_nm=neck_mean,
            motion_sickness_index=_finite(
                motion_sickness_index, "motion_sickness_index"
            ),
            completion_rate=_finite(completion_rate, "completion_rate"),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "throughput_bps": float(self.throughput_bps),
            "mean_effort": float(self.mean_effort),
            "peak_fatigue": float(self.peak_fatigue),
            "mean_neck_moment_nm": float(self.mean_neck_moment_nm),
            "motion_sickness_index": float(self.motion_sickness_index),
            "completion_rate": float(self.completion_rate),
        }


DEFAULT_ERGONOMIC_OBJECTIVES: tuple[ObjectiveBound, ...] = (
    ObjectiveBound("throughput_bps", 0.0, 20.0, minimize=False),
    ObjectiveBound("mean_effort", 0.0, 1.0, minimize=True),
    ObjectiveBound("peak_fatigue", 0.0, 1.0, minimize=True),
    ObjectiveBound("mean_neck_moment_nm", 0.0, 30.0, minimize=True),
    ObjectiveBound("motion_sickness_index", 0.0, 100.0, minimize=True),
)


RolloutEvaluator = Callable[
    [LayoutCandidate, int], ErgonomicRolloutMetrics | Mapping[str, float]
]


@dataclass
class LayoutOptimizationHarness:
    """Adapter bridging :class:`LayoutSpace` rollouts to BO/MOBO study backends.

    Instances satisfy the ``(parameters, iteration) -> objectives`` callable
    signature of ``bo_client.run_connected_study`` / ``run_subprocess_study``
    and also provide a zero-dependency loopback NDJSON client via
    :meth:`run_connected_loop`.
    """

    space: LayoutSpace
    rollout_fn: RolloutEvaluator
    objectives: tuple[ObjectiveBound, ...] = DEFAULT_ERGONOMIC_OBJECTIVES
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("objectives must be non-empty")
        self.objectives = tuple(self.objectives)

    def __call__(
        self, parameters: Mapping[str, float], iteration: int = 0
    ) -> dict[str, float]:
        """Evaluate ``parameters`` and return clamped objective values."""
        candidate = self.space.decode(parameters)
        raw_out = self.rollout_fn(candidate, int(iteration))
        raw_map = (
            raw_out.as_dict()
            if isinstance(raw_out, ErgonomicRolloutMetrics)
            else dict(raw_out)
        )
        clamped: dict[str, float] = {}
        for spec in self.objectives:
            if spec.key not in raw_map:
                raise KeyError(
                    f"Rollout output missing objective {spec.key!r}; "
                    f"got {sorted(raw_map)}"
                )
            clamped[spec.key] = spec.clamp(float(raw_map[spec.key]))

        self.history.append(
            {
                "iteration": int(iteration),
                "parameters": {k: float(v) for k, v in parameters.items()},
                "candidate": candidate,
                "objectives": dict(clamped),
            }
        )
        return clamped

    def to_init_message(
        self,
        *,
        optimizer_backend: str = "botorch",
        num_sampling_iterations: int = 4,
        num_optimization_iterations: int = 8,
        seed: int = 3,
        user_id: str = "sim",
        condition_id: str = "layout",
        group_id: str = "0",
    ) -> dict[str, Any]:
        """Build the ``type='init'`` NDJSON handshake message for the optimizer."""
        params = self.space.parameter_bounds()
        return {
            "type": "init",
            "config": {
                "optimizerBackend": optimizer_backend,
                "nParameters": len(params),
                "nObjectives": len(self.objectives),
                "numSamplingIterations": int(num_sampling_iterations),
                "numOptimizationIterations": int(num_optimization_iterations),
                "batchSize": 1,
                "numRestarts": 8,
                "rawSamples": 256,
                "mcSamples": 64,
                "seed": int(seed),
                "warmStart": False,
                "warmStartObjectiveFormat": "auto",
                "initialParametersDataPath": "",
                "initialObjectivesDataPath": "",
            },
            "parameters": [p.to_wire(i) for i, p in enumerate(params)],
            "objectives": [o.to_wire(i) for i, o in enumerate(self.objectives)],
            "user": {
                "userId": user_id,
                "conditionId": condition_id,
                "groupId": group_id,
            },
        }

    def run_connected_loop(
        self,
        conn: Any,
        *,
        optimizer_backend: str = "botorch",
        num_sampling_iterations: int = 4,
        num_optimization_iterations: int = 8,
        seed: int = 3,
    ) -> list[dict[str, Any]]:
        """Drive a connected NDJSON optimizer socket until ``optimization_finished``."""

        def _send(payload: Mapping[str, Any]) -> None:
            conn.sendall((json.dumps(dict(payload)) + "\n").encode("utf-8"))

        _send(
            self.to_init_message(
                optimizer_backend=optimizer_backend,
                num_sampling_iterations=num_sampling_iterations,
                num_optimization_iterations=num_optimization_iterations,
                seed=seed,
            )
        )
        reader = conn.makefile("r", encoding="utf-8")
        records: list[dict[str, Any]] = []
        for line in reader:
            stripped = line.strip()
            if not stripped:
                continue
            msg = json.loads(stripped)
            mtype = msg.get("type")
            if mtype == "parameters":
                it = int(msg["iteration"])
                params = {
                    str(k): float(v) for k, v in (msg.get("values") or {}).items()
                }
                objs = self(params, it)
                _send({"type": "objectives", "values": objs})
                records.append(
                    {"iteration": it, "parameters": params, "objectives": objs}
                )
            elif mtype == "optimization_finished":
                return records
        raise ConnectionError("Optimizer socket closed before optimization_finished")

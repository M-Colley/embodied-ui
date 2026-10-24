"""Shared helpers: a host scene with a servo fingertip and the interface."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import pytest

from embodied_ui import Interface, InterfaceRuntime
from embodied_ui.probe import Probe, ProbeControl


@dataclass
class Scene:
    interface: Interface
    model: mujoco.MjModel
    data: mujoco.MjData
    runtime: InterfaceRuntime
    probe: ProbeControl

    def run(self, seconds: float, target=None, period: float = 0.02):
        """Step in control periods; return the readings."""
        readings = []
        substeps = round(period / self.model.opt.timestep)
        for k in range(round(seconds / period)):
            if target is not None:
                self.probe.target(self.data, *target(k * period))
            self.runtime.step(self.data, substeps)
            readings.append(self.runtime.read(self.data))
        return readings

    def qpos(self, device: str, axis: str) -> float:
        return float(
            self.data.qpos[
                self.model.joint(self.interface.element(device, axis)).qposadr[0]
            ]
        )

    def set_qpos(self, device: str, axis: str, value: float) -> None:
        self.data.qpos[
            self.model.joint(self.interface.element(device, axis)).qposadr[0]
        ] = value
        mujoco.mj_forward(self.model, self.data)


def build_scene(devices, probe_pos=(0.0, 0.0, 0.08), degree=True, **kwargs) -> Scene:
    interface = Interface("ui", tuple(devices), **kwargs)
    spec = mujoco.MjSpec()
    spec.compiler.degree = degree
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    probe = Probe(pos=probe_pos)
    probe.attach(spec)
    interface.attach(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return Scene(interface, model, data, interface.bind(model), probe.bind(model))


@pytest.fixture
def scene_factory():
    return build_scene

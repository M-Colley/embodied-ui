"""Tests for parametric UI layout search spaces, metrics, and BO/MOBO harness."""

from __future__ import annotations

import json
import socket
import threading

import mujoco
import numpy as np
import pytest

from embodied_ui import (
    ErgonomicRolloutMetrics,
    LayoutCandidate,
    LayoutOptimizationHarness,
    LayoutSpace,
    ObjectiveBound,
    PushButton,
    TouchSurface,
)


def test_layout_space_decode_and_compiled_interface_scaling() -> None:
    space = LayoutSpace(
        distance=(0.35, 0.65),
        azimuth=(-0.4, 0.4),
        elevation=(-0.3, 0.2),
        tilt=(0.0, 0.6),
        roll=None,
        target_scale=(0.8, 1.4),
    )
    bounds = space.parameter_bounds()
    assert [b.key for b in bounds] == [
        "distance",
        "azimuth",
        "elevation",
        "tilt",
        "target_scale",
    ]

    mid = space.midpoint()
    assert mid.distance == pytest.approx(0.50)
    assert mid.target_scale == pytest.approx(1.10)
    assert mid.roll == pytest.approx(0.0)

    placements = [
        (PushButton.key("k0"), -0.02, 0.0),
        (PushButton.key("k1"), 0.02, 0.0),
        (TouchSurface("pad", size=(0.12, 0.08)), 0.0, 0.06),
    ]
    iface = mid.build_interface("panel", placements, anchor=(0.0, 0.0, 1.0))
    pad_dev = iface.devices[2]
    assert isinstance(pad_dev, TouchSurface)
    assert pad_dev.size == pytest.approx((0.12 * 1.10, 0.08 * 1.10))

    spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    iface.attach(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    rt = iface.bind(model)
    rt.step(data, nstep=5)
    reading = rt.read(data)
    assert "k0.pressed" in reading.channels

    with pytest.raises(ValueError, match="outside"):
        space.decode(
            {
                "distance": 0.9,
                "azimuth": 0.0,
                "elevation": 0.0,
                "tilt": 0.0,
                "target_scale": 1.0,
            }
        )


def test_ergonomic_metrics_and_harness_socket_loop() -> None:
    space = LayoutSpace(
        distance=(0.4, 0.7),
        azimuth=(-0.3, 0.3),
        elevation=None,
        tilt=(0.0, 0.5),
        target_scale=(0.8, 1.3),
    )

    def _rollout(candidate: LayoutCandidate, iteration: int) -> ErgonomicRolloutMetrics:
        del iteration
        bits = [3.5 * candidate.target_scale, 3.2 * candidate.target_scale]
        acts = np.full((20, 6), 0.25 + 0.3 * candidate.distance)
        fat = np.linspace(0.0, 0.15 * candidate.distance, 20)
        neck = np.full(20, 4.0 + 3.0 * abs(candidate.azimuth))
        return ErgonomicRolloutMetrics.from_rollout(
            duration_s=2.0,
            index_of_difficulty_bits=bits,
            activations=acts,
            fatigue=fat,
            neck_moments_nm=neck,
            motion_sickness_index=2.5,
        )

    objectives = (
        ObjectiveBound("throughput_bps", 0.0, 10.0, minimize=False),
        ObjectiveBound("mean_effort", 0.0, 1.0, minimize=True),
        ObjectiveBound("mean_neck_moment_nm", 0.0, 20.0, minimize=True),
    )
    harness = LayoutOptimizationHarness(space, _rollout, objectives=objectives)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    received_init: dict[str, object] = {}

    def _mock_optimizer() -> None:
        conn, _ = server.accept()
        with conn:
            f_in = conn.makefile("r", encoding="utf-8")
            init_line = f_in.readline()
            init_msg = json.loads(init_line)
            received_init.update(init_msg)
            for it in range(3):
                cand = {
                    "distance": 0.45 + 0.05 * it,
                    "azimuth": -0.1 + 0.1 * it,
                    "tilt": 0.2,
                    "target_scale": 1.0 + 0.1 * it,
                }
                conn.sendall(
                    (
                        json.dumps(
                            {"type": "parameters", "iteration": it, "values": cand}
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                reply = json.loads(f_in.readline())
                assert reply["type"] == "objectives"
                assert set(reply["values"]) == {
                    "throughput_bps",
                    "mean_effort",
                    "mean_neck_moment_nm",
                }
            conn.sendall(
                (json.dumps({"type": "optimization_finished"}) + "\n").encode()
            )
        server.close()

    worker = threading.Thread(target=_mock_optimizer, daemon=True)
    worker.start()

    with socket.create_connection(("127.0.0.1", port), timeout=5.0) as client_conn:
        records = harness.run_connected_loop(
            client_conn,
            num_sampling_iterations=2,
            num_optimization_iterations=1,
        )

    worker.join(timeout=5.0)
    assert received_init["type"] == "init"
    assert len(records) == 3
    assert len(harness.history) == 3
    assert (
        records[2]["objectives"]["throughput_bps"]
        > records[0]["objectives"]["throughput_bps"]
    )

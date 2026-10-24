"""Drive a CARLA vehicle with a fingertip on interface devices, in lockstep.

A servo fingertip presses a throttle key; a hazard toggle switch turns on the
hazard lights, whose state CARLA reports back to a lamp in the MuJoCo model.
Writes a JSON report and, with --camera, the driver's view as a PNG.

    python examples/carla_fingertip.py --out evidence/carla-fingertip.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import mujoco
import numpy as np

from embodied_ui import (
    Indicator,
    Interface,
    Pose,
    PushButton,
    SteeringWheel,
    ToggleSwitch,
)
from embodied_ui.backends.carla import CarlaCamera, CarlaEngine
from embodied_ui.engine import Bridge, Route, Routes
from embodied_ui.probe import Probe

DT = 0.02


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--spawn-point", type=int, default=105)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument(
        "--camera", action="store_true", help="needs a rendering server"
    )
    parser.add_argument("--out", type=Path, default=Path("carla-fingertip.json"))
    args = parser.parse_args()

    interface = Interface(
        "cockpit",
        (
            PushButton("throttle", travel=0.01, preload=2.0, force=6.0),
            ToggleSwitch("hazard", Pose((0.1, 0, 0))),
            SteeringWheel("wheel", Pose((0.6, 0, 0.3))),
            Indicator("blinker", Pose((0.2, 0, 0))),
        ),
    )
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    probe = Probe(pos=(0, 0, 0.05))
    probe.attach(spec)
    interface.attach(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    runtime = interface.bind(model)
    tip = probe.bind(model)
    routes = Routes(
        routes=(
            Route("throttle.value", "vehicle.throttle"),
            Route("wheel.value", "vehicle.steer", scale=-1.0),
            Route("hazard.state", "lights.hazard"),
        ),
        feedback=(Route("lights.left_blinker", "blinker.on"),),
    )
    cameras = (CarlaCamera("driver", width=640, height=360),) if args.camera else ()
    engine = CarlaEngine(
        args.host,
        args.port,
        dt=DT,
        vehicle="vehicle.nissan.patrol",
        spawn_point=args.spawn_point,
        cameras=cameras,
    )
    rows = []
    started = time.perf_counter()
    with Bridge(runtime, engine, routes, engine_frame="carla") as bridge:
        start = bridge.reset(data)
        version = engine.client.get_server_version()
        town = engine.world.get_map().name
        hazard = model.joint("cockpit/hazard/angle").qposadr[0]
        steps = round(args.seconds / DT)
        for k in range(steps):
            t = k * DT
            # Flip the hazard switch with the fingertip's twin: set it by
            # hand at 1 s; press the throttle key from 0.5 s to 2.5 s.
            if k == round(1.0 / DT):
                data.qpos[hazard] = 0.35
            tip.target(data, (0, 0, 0.0 if 0.5 < t < 2.5 else 0.03))
            runtime.step(data, 10)
            frame = bridge.step(data)
            out = frame.state.outputs
            rows.append(
                {
                    "frame": frame.state.frame - start.state.frame,
                    "time": round(frame.state.time - start.state.time, 6),
                    "key_travel_mm": round(1000 * frame.reading["throttle.travel"], 3),
                    "key_contact_n": round(frame.reading["throttle.contact"], 3),
                    "throttle": round(frame.command.controls["vehicle.throttle"], 4),
                    "speed": round(out["vehicle.speed"], 4),
                    "hazard_switch": frame.reading["hazard.state"],
                    "left_blinker": out["lights.left_blinker"],
                    "right_blinker": out["lights.right_blinker"],
                }
            )
        image = frame.state.outputs.get("camera.driver")
    wall = time.perf_counter() - started
    report = {
        "server": {
            "version": version,
            "map": town,
            "host": args.host,
            "port": args.port,
        },
        "client": {"python": platform.python_version(), "mujoco": mujoco.__version__},
        "dt": DT,
        "frames": len(rows),
        "wall_seconds": round(wall, 3),
        "frames_per_wall_second": round(len(rows) / wall, 1),
        "max_speed": max(r["speed"] for r in rows),
        "lamp_on_at_end": bool(
            np.allclose(
                model.geom_rgba[model.geom("cockpit/blinker/static0").id],
                Indicator("x").on,
            )
        ),
        "rows": rows[::5],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    if image is not None:
        from PIL import Image

        Image.fromarray(image).save(args.out.with_suffix(".png"))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

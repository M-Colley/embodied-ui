# Package instructions

This is Mark Colley's standalone `embodied-ui` package: physical user
interfaces for MuJoCo bodies and the `embodied-ui/1` lockstep protocol to
game engines. MyoSuite, myosuite-carla, CARLA and Unity are separate; keep
adapters for myosuite-carla in that repository.

- **Engine-independent, model-independent.** Keep the core free of engine,
  MyoSuite and CARLA imports, and import backends lazily. Devices compose
  into any `MjSpec` and through MyoSuite's
  `ModelBuilder.apply_transform(interface.transform)`. Do not register
  environments.
- **Inputs are mechanisms.** Joints, springs, end stops and contacts, never
  mocap. Device forces may only be written to the device's own
  `qfrc_applied` entries. Give every default parameter a cited source in the
  docstring and README.
- **Keep the protocol in step.** `protocol.py`, PROTOCOL.md, the C#
  `Runtime/Protocol` sources and the .NET harness must agree. Bump the major
  version for incompatible changes. The C# protocol layer has no
  UnityEngine dependency and must build at C# 9.
- **Unity package.** Every file and folder outside `~` folders needs a
  committed `.meta` file with a stable GUID.
- **Checks.**
  - Test: `python -m pytest tests -q`.
  - Lint: `python -m ruff check src tests examples`.
  - Types: `python -m mypy src/embodied_ui --ignore-missing-imports`.
  - Build: `python -m build` and
    `dotnet build dotnet/EmbodiedUI.Protocol.Harness -c Release`.
  - Minimum supported MuJoCo is 3.7 (3.6.0 has a hinge-contact bug); test
    changes on it.
- **Live validation** needs a CARLA server
  (`EMBODIED_UI_CARLA=1`, plus `EMBODIED_UI_CARLA_CAMERA=1` with rendering)
  or a Unity Editor. Record builds, commands and results in VALIDATION.md;
  doubles and skipped tests are not evidence of a live pass.
- Keep commit authorship with the human owner. Do not add AI co-author
  trailers.

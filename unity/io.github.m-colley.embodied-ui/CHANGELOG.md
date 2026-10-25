# Changelog

## Unreleased

- Audio and vibration outputs for embodied-ui speakers and vibrotactors.
- Each output component in its own file, so it can be added in the
  Inspector and restored from saved scenes.
- Declares the audio module dependency.
- Manifest export names an unsaved scene's interface `untitled`; an empty
  name could not be loaded in Python. `ManifestExporter.Export` creates the
  target folder.
- EditMode tests of the manifest exporter (`Tests/Editor`).
- The Wheel Vehicle sample has its own assembly definition,
  `EmbodiedUI.Samples.WheelVehicle`, so tests and other assemblies can
  reference it.

## 0.1.0

- Lockstep bridge for the embodied-ui/1 protocol with virtual Input System
  device, virtual touchscreen, outputs (pose, vehicle, camera, lamp, value),
  device authoring with runtime describe and manifest export, event
  listeners and a WheelCollider vehicle sample.

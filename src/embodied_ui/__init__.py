"""Physical user interfaces for simulated bodies, bridged to game engines.

Devices (buttons, switches, knobs, sliders, levers, pedals, steering wheels,
joysticks, touch surfaces, screens, lamps, speakers, vibrotactors and
engine-driven props) are compiled into any MuJoCo model, operated through
contact by a simulated body, and routed in lockstep to Unity, CARLA or any
engine speaking the ``embodied-ui/1`` protocol.
"""

from .devices import (
    DEVICE_TYPES,
    Control,
    Device,
    Indicator,
    Joystick,
    Knob,
    Lever,
    Pedal,
    Prop,
    PushButton,
    Screen,
    Slider,
    Speaker,
    SteeringWheel,
    ToggleSwitch,
    TouchSurface,
    Vibrotactor,
    register,
)
from .frames import (
    CARLA,
    FLU,
    FRAMES,
    GLTF,
    MYOSUITE_SEATED,
    UNITY,
    UNREAL,
    Frame,
    Pose,
)
from .geometry import Box, Capsule, Cylinder, Mesh, Part, Sphere
from .interface import Contacts, Interface
from .layout import PanelLayout, panel_pose
from .logic import Event, Interlock
from .mechanics import Axis, Detents, ForceCurve, Servo
from .motion import (
    MotionSample,
    MotionTrajectory,
    extract_motion_sample,
    extract_motion_trajectory,
)
from .optimization import (
    DEFAULT_ERGONOMIC_OBJECTIVES,
    ErgonomicRolloutMetrics,
    LayoutCandidate,
    LayoutOptimizationHarness,
    LayoutSpace,
    ObjectiveBound,
    ParameterBound,
)
from .runtime import InterfaceRuntime, Reading, merge_readings
from .trace import TraceRecorder

__version__ = "0.1.0"

__all__ = [
    "Axis",
    "Box",
    "CARLA",
    "Capsule",
    "Contacts",
    "Control",
    "Cylinder",
    "DEFAULT_ERGONOMIC_OBJECTIVES",
    "DEVICE_TYPES",
    "Detents",
    "Device",
    "ErgonomicRolloutMetrics",
    "Event",
    "FLU",
    "FRAMES",
    "ForceCurve",
    "Frame",
    "GLTF",
    "Indicator",
    "Interface",
    "InterfaceRuntime",
    "Interlock",
    "Joystick",
    "Knob",
    "LayoutCandidate",
    "LayoutOptimizationHarness",
    "LayoutSpace",
    "Lever",
    "MYOSUITE_SEATED",
    "Mesh",
    "MotionSample",
    "MotionTrajectory",
    "ObjectiveBound",
    "PanelLayout",
    "ParameterBound",
    "Part",
    "Pedal",
    "Pose",
    "Prop",
    "PushButton",
    "Reading",
    "Screen",
    "Servo",
    "Slider",
    "Speaker",
    "Sphere",
    "SteeringWheel",
    "ToggleSwitch",
    "TouchSurface",
    "TraceRecorder",
    "UNITY",
    "UNREAL",
    "Vibrotactor",
    "extract_motion_sample",
    "extract_motion_trajectory",
    "merge_readings",
    "panel_pose",
    "register",
]

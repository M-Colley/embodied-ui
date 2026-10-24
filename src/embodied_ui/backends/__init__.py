"""Engine backends: in-process, remote (Unity and others) and CARLA.

Backends import their engine's client library lazily, so importing this
package never requires CARLA or a running engine.
"""

from .loopback import LoopbackEngine
from .remote import EngineServer, RemoteEngine, describe

__all__ = ["CarlaEngine", "EngineServer", "LoopbackEngine", "RemoteEngine", "describe"]


def __getattr__(name: str) -> object:
    if name == "CarlaEngine":
        from .carla import CarlaEngine

        return CarlaEngine
    raise AttributeError(name)

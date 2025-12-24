import time
from dataclasses import dataclass, field
from typing import List, Tuple


class UAVControllerSim:
    """
    Thin wrapper that mimics a UAV controller. In demo mode it only tracks state changes.
    """

    def __init__(self, uav_id: str = "sim-uav-1", mode: str = "AUTO", connection_state: str = "disconnected"):
        self.uav_id = uav_id
        self.mode = mode
        self.connection_state = connection_state

    def arm(self) -> bool:
        self.connection_state = "armed"
        return True

    def takeoff(self) -> bool:
        self.mode = "TAKEOFF"
        self.connection_state = "flying"
        return True

    def fly_route(self, route: List[Tuple[float, float, float]]) -> bool:
        # In simulation we only acknowledge the request.
        self.mode = "AUTO"
        return bool(route)

    def land(self) -> bool:
        self.mode = "LAND"
        self.connection_state = "landed"
        return True


class UAVController:
    """
    Adapter around UAVControllerSim so the domain layer exposes a stable controller API.
    """

    def __init__(self, uav_id: str = "sim-uav-1", mode: str = "AUTO", connection_state: str = "disconnected", backend=None):
        self._backend = backend or UAVControllerSim(uav_id=uav_id, mode=mode, connection_state=connection_state)

    @property
    def uav_id(self) -> str:
        return self._backend.uav_id

    @property
    def mode(self) -> str:
        return self._backend.mode

    @property
    def connection_state(self) -> str:
        return self._backend.connection_state

    def arm(self) -> bool:
        return self._backend.arm()

    def takeoff(self) -> bool:
        return self._backend.takeoff()

    def fly_route(self, route: List[Tuple[float, float, float]]) -> bool:
        return self._backend.fly_route(route)

    def land(self) -> bool:
        return self._backend.land()


@dataclass
class Mission:
    id: str
    area: List[Tuple[float, float, float]]
    altitude: float
    speed: float
    status: str = "planned"
    start_time: str | None = None
    end_time: str | None = None

    def start(self):
        self.status = "running"
        self.start_time = str(int(time.time() * 1000))

    def pause(self):
        self.status = "paused"

    def stop(self):
        self.status = "completed"
        self.end_time = str(int(time.time() * 1000))


@dataclass
class Record:
    timestamp: str | int
    model: str
    lat: float
    lon: float
    altitude: float
    battery: float
    signal: float

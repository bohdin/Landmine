import json
import uuid
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from core.database import SessionLocal
from core.mission_planner import plan_lawnmower_path
from core.mission_simulator import MissionSimulator
from core.mission_validation import validate_polygon
from core.mission_flow import run_mission
from core.processing import MissionProcessor
from core.repository import (
    create_mission,
    get_latest_run_for_mission,
    get_mission,
    get_mines,
    get_risk_zones,
    list_missions,
)
from web.models import detector, set_thresholds

router = APIRouter()
simulator = MissionSimulator(SessionLocal)
processor = MissionProcessor(SessionLocal, detector)


class MissionCreateRequest(BaseModel):
    polygon: List[List[float]]
    altitude: float = Field(default=0.4, ge=0.0)
    speed: float = Field(default=1.0, gt=0)
    track_spacing_m: float = Field(default=5.0, gt=0)
    waypoint_step_m: float = Field(default=2.0, gt=0)
    capture_every_m: float = Field(default=2.0, gt=0)

    @validator("altitude", pre=True, always=True)
    def clamp_alt(cls, v):
        return max(0.3, min(0.5, float(v)))

    @validator("polygon")
    def ensure_polygon(cls, v):
        if len(v) < 3:
            raise ValueError("polygon must have at least 3 points")
        return v

    @validator("capture_every_m")
    def ensure_capture_step(cls, v, values):
        waypoint_step = values.get("waypoint_step_m", 2.0)
        if v < waypoint_step:
            return waypoint_step
        return v


class MissionRunRequest(BaseModel):
    model: str = "ensemble"
    threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    draw_all: bool = False


@router.post("/missions")
def create_mission_endpoint(body: MissionCreateRequest):
    latlon = [(p[0], p[1]) for p in body.polygon]
    ok, reason = validate_polygon(latlon)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)
    polygon = [(p[0], p[1], body.altitude if len(p) < 3 else p[2]) for p in body.polygon]
    route = plan_lawnmower_path(
        [(p[0], p[1]) for p in polygon],
        track_spacing_m=body.track_spacing_m,
        waypoint_step_m=body.waypoint_step_m,
        altitude=body.altitude,
    )
    mission_id = str(uuid.uuid4())
    with SessionLocal() as session:
        create_mission(
            session,
            mission_id=mission_id,
            polygon=polygon,
            route=route,
            altitude=body.altitude,
            speed=body.speed,
            lane_spacing=body.track_spacing_m,
            track_spacing_m=body.track_spacing_m,
            waypoint_step_m=body.waypoint_step_m,
            capture_every_m=body.capture_every_m,
            seed=int(uuid.UUID(mission_id)) & 0x7FFFFFFF,
        )
        session.commit()
    return {
        "id": mission_id,
        "route": route,
        "polygon": polygon,
        "track_spacing_m": body.track_spacing_m,
        "waypoint_step_m": body.waypoint_step_m,
        "capture_every_m": body.capture_every_m,
    }


@router.get("/missions")
def list_missions_endpoint():
    with SessionLocal() as session:
        return list_missions(session)


@router.get("/missions/{mission_id}")
def get_mission_endpoint(mission_id: str):
    with SessionLocal() as session:
        mission = get_mission(session, mission_id)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return {
            "id": mission.id,
            "polygon": json.loads(mission.polygon_json),
            "route": json.loads(mission.route_json) if mission.route_json else [],
            "status": mission.status,
            "altitude": mission.altitude,
            "speed": mission.speed,
            "lane_spacing": mission.lane_spacing,
            "track_spacing_m": getattr(mission, "track_spacing_m", None),
            "waypoint_step_m": getattr(mission, "waypoint_step_m", None),
            "capture_every_m": getattr(mission, "capture_every_m", None),
            "start_time": mission.start_time,
            "end_time": mission.end_time,
        }


@router.post("/missions/{mission_id}/simulate")
def simulate_mission_endpoint(mission_id: str):
    try:
        result = simulator.simulate(mission_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/missions/{mission_id}/process")
def process_mission_endpoint(mission_id: str, model: str = "ensemble", threshold: float = 0.4):
    try:
        set_thresholds(threshold, threshold)
        result = processor.process(mission_id=mission_id, model=model)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/missions/{mission_id}/run")
def run_mission_endpoint(mission_id: str, body: MissionRunRequest):
    set_thresholds(body.threshold, body.threshold)
    try:
        return run_mission(
            SessionLocal,
            mission_id=mission_id,
            detector=detector,
            model=body.model,
            threshold=body.threshold,
            draw_all=body.draw_all,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/missions/{mission_id}/risk-map")
def risk_map_endpoint(mission_id: str):
    with SessionLocal() as session:
        zones = get_risk_zones(session, mission_id)
    return {"mission_id": mission_id, "zones": zones}


@router.get("/missions/{mission_id}/exports")
def exports_endpoint(mission_id: str):
    with SessionLocal() as session:
        run = get_latest_run_for_mission(session, mission_id)
    if not run:
        raise HTTPException(status_code=404, detail="No runs for mission")
    exports = processor._build_exports(mission_id, run.id)
    return {"mission_id": mission_id, "run_id": run.id, **exports}


@router.post("/missions/{mission_id}/refine")
def refine_mission_endpoint(mission_id: str):
    with SessionLocal() as session:
        mission = get_mission(session, mission_id)
        zones = get_risk_zones(session, mission_id)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        if not zones:
            raise HTTPException(status_code=400, detail="No risk zones to refine")

        polygon = json.loads(mission.polygon_json)
        current_spacing = getattr(mission, "track_spacing_m", None) or mission.lane_spacing or 5.0
        new_spacing = max(1.0, current_spacing * 0.7)
        waypoint_step = getattr(mission, "waypoint_step_m", None) or 2.0
        route = plan_lawnmower_path(
            [(p[0], p[1]) for p in polygon],
            track_spacing_m=new_spacing,
            waypoint_step_m=waypoint_step,
            altitude=mission.altitude,
        )
        mission.route_json = json.dumps(route)
        mission.track_spacing_m = new_spacing
        mission.lane_spacing = new_spacing
        session.commit()
        mines = get_mines(session, mission_id)

    return {
        "id": mission_id,
        "polygon": polygon,
        "route": route,
        "track_spacing_m": new_spacing,
        "waypoint_step_m": waypoint_step,
        "capture_every_m": getattr(mission, "capture_every_m", None),
        "mines": mines,
    }

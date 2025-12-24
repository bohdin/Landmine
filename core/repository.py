import json
import time
from typing import Iterable, List, Tuple

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core.db_models import (
    DetectionModel,
    FrameModel,
    MetalSample,
    MissionModel,
    MissionTargetModel,
    RiskZoneModel,
    RunModel,
    TelemetrySample,
)
from core.database import run_with_retry


def mission_to_dict(m: MissionModel):
    return {
        "id": m.id,
        "polygon": json.loads(m.polygon_json),
        "route": json.loads(m.route_json) if m.route_json else [],
        "altitude": m.altitude,
        "speed": m.speed,
        "lane_spacing": m.lane_spacing,
        "track_spacing_m": getattr(m, "track_spacing_m", None),
        "waypoint_step_m": getattr(m, "waypoint_step_m", None),
        "capture_every_m": getattr(m, "capture_every_m", None),
        "seed": getattr(m, "seed", None),
        "status": m.status,
        "start_time": m.start_time,
        "end_time": m.end_time,
        "created_at": m.created_at,
    }


def create_mission(
    session: Session,
    mission_id: str,
    polygon: List[Tuple[float, float, float]],
    route: List[Tuple[float, float, float]],
    altitude: float,
    speed: float,
    lane_spacing: float,
    track_spacing_m: float,
    waypoint_step_m: float,
    capture_every_m: float,
    seed: int | None = None,
):
    mission = MissionModel(
        id=mission_id,
        polygon_json=json.dumps(polygon),
        route_json=json.dumps(route),
        altitude=altitude,
        speed=speed,
        lane_spacing=lane_spacing,
        track_spacing_m=track_spacing_m,
        waypoint_step_m=waypoint_step_m,
        capture_every_m=capture_every_m,
        seed=seed,
        status="planned",
        created_at=int(time.time() * 1000),
    )
    session.merge(mission)
    return mission


def list_missions(session: Session):
    missions = session.scalars(select(MissionModel)).all()
    return [mission_to_dict(m) for m in missions]


def get_mission(session: Session, mission_id: str):
    mission = session.get(MissionModel, mission_id)
    return mission


def get_run(session: Session, run_id: str):
    return session.get(RunModel, run_id)


def get_latest_run_for_mission(session: Session, mission_id: str):
    return session.scalars(
        select(RunModel).where(RunModel.mission_id == mission_id).order_by(RunModel.started_at.desc())
    ).first()


def save_targets(session: Session, mission_id: str, targets: List[Tuple[float, float, float, float]]):
    session.execute(delete(MissionTargetModel).where(MissionTargetModel.mission_id == mission_id))
    for lat, lon, alt, signature in targets:
        session.add(
            MissionTargetModel(
                mission_id=mission_id,
                lat=lat,
                lon=lon,
                alt=alt,
                signature=signature,
            )
        )


def save_mines(session: Session, mission_id: str, mines: List[Tuple[float, float, float, float]]):
    save_targets(session, mission_id, mines)


def get_mines(session: Session, mission_id: str):
    mines = session.scalars(select(MissionTargetModel).where(MissionTargetModel.mission_id == mission_id)).all()
    return [
        {
            "lat": m.lat,
            "lon": m.lon,
            "alt": m.alt,
            "signature": m.signature,
        }
        for m in mines
    ]


def start_run(session: Session, mission_id: str, run_id: str, started_at: int):
    run = RunModel(id=run_id, mission_id=mission_id, started_at=started_at, status="simulated")
    session.add(run)
    return run


def end_run(session: Session, run_id: str, ended_at: int):
    run = session.get(RunModel, run_id)
    if run:
        run.ended_at = ended_at
        run.status = "completed"
    return run


def bulk_insert_telemetry(session: Session, run_id: str, samples: Iterable[dict]):
    run_with_retry(lambda: session.bulk_insert_mappings(TelemetrySample, [dict(run_id=run_id, **s) for s in samples]))


def bulk_insert_metal(session: Session, run_id: str, samples: Iterable[dict]):
    run_with_retry(lambda: session.bulk_insert_mappings(MetalSample, [dict(run_id=run_id, **s) for s in samples]))


def bulk_insert_frames(session: Session, run_id: str, frames: Iterable[dict]):
    run_with_retry(lambda: session.bulk_insert_mappings(FrameModel, [dict(run_id=run_id, **f) for f in frames]))


def bulk_insert_detections(session: Session, run_id: str, detections: Iterable[dict]):
    run_with_retry(lambda: session.bulk_insert_mappings(DetectionModel, [dict(run_id=run_id, **d) for d in detections]))


def save_risk_zones(session: Session, mission_id: str, run_id: str, zones: List[dict]):
    run_with_retry(lambda: session.execute(delete(RiskZoneModel).where(RiskZoneModel.mission_id == mission_id)))
    for z in zones:
        session.add(
            RiskZoneModel(
                mission_id=mission_id,
                run_id=run_id,
                center_lat=z["center_lat"],
                center_lon=z["center_lon"],
                geometry_json=json.dumps(z["geometry"]),
                score=z["score"],
                detections_count=z["detections_count"],
            )
        )


def get_risk_zones(session: Session, mission_id: str):
    zones = session.scalars(select(RiskZoneModel).where(RiskZoneModel.mission_id == mission_id)).all()
    return [
        {
            "id": z.id,
            "center_lat": z.center_lat,
            "center_lon": z.center_lon,
            "geometry": json.loads(z.geometry_json),
            "score": z.score,
            "detections_count": z.detections_count,
        }
        for z in zones
    ]


def get_frames(session: Session, run_id: str):
    frames = session.scalars(select(FrameModel).where(FrameModel.run_id == run_id).order_by(FrameModel.timestamp)).all()
    return [
        {
            "id": f.id,
            "timestamp": f.timestamp,
            "lat": f.lat,
            "lon": f.lon,
            "alt": f.alt,
            "image_path": f.image_path,
            "patch_sha256": f.patch_sha256,
            "triggered_by_metal": f.triggered_by_metal,
            "frame_index": f.frame_index,
            "tile_id": f.tile_id,
        }
        for f in frames
    ]


def get_telemetry(session: Session, run_id: str):
    samples = session.scalars(
        select(TelemetrySample).where(TelemetrySample.run_id == run_id).order_by(TelemetrySample.timestamp)
    ).all()
    return [
        {
            "timestamp": s.timestamp,
            "lat": s.lat,
            "lon": s.lon,
            "alt": s.alt,
            "battery": s.battery,
            "signal": s.signal,
            "yaw": s.yaw,
        }
        for s in samples
    ]


def get_metal(session: Session, run_id: str):
    samples = session.scalars(
        select(MetalSample).where(MetalSample.run_id == run_id).order_by(MetalSample.timestamp)
    ).all()
    return [
        {
            "timestamp": s.timestamp,
            "lat": s.lat,
            "lon": s.lon,
            "alt": s.alt,
            "value": s.value,
            "adaptive_threshold": s.adaptive_threshold,
        }
        for s in samples
    ]


def get_detections(session: Session, run_id: str):
    detections = session.scalars(
        select(DetectionModel).where(DetectionModel.run_id == run_id).order_by(DetectionModel.id)
    ).all()
    return [
        {
            "id": d.id,
            "frame_id": d.frame_id,
            "model": d.model,
            "score": d.score,
            "bbox": json.loads(d.bbox_json),
            "confirmed_by_metal": d.confirmed_by_metal,
            "risk_score": d.risk_score,
            "lat": d.lat,
            "lon": d.lon,
            "alt": d.alt,
            "label": getattr(d, "label", "review"),
        }
        for d in detections
    ]

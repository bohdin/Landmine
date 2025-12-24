import json
import os
from pathlib import Path
from typing import List

from core.detect_api import DetectAPI
from core.domain import Mission, Record, UAVController
from core.mission_planner import plan_lawnmower_path
from core.mission_simulator import MissionSimulator
from core.mission_validation import validate_polygon
from core.processing import MissionProcessor, nearest_telemetry
from core.repository import get_frames, get_mission, get_risk_zones, get_telemetry


def _frame_to_url(image_path: str) -> str:
    name = Path(image_path).name
    return f"/test_images/{name}"


def _record_to_dict(record: Record):
    return {
        "timestamp": record.timestamp,
        "model": record.model,
        "lat": record.lat,
        "lon": record.lon,
        "altitude": record.altitude,
        "battery": record.battery,
        "signal": record.signal,
    }


def run_mission(
    session_factory,
    mission_id: str,
    detector,
    model: str = "ensemble",
    threshold: float = 0.4,
    draw_all: bool = False,
    history_limit: int = 20,
):
    detect_api = DetectAPI(detector)
    simulator = MissionSimulator(session_factory)
    processor = MissionProcessor(session_factory, detector)

    with session_factory() as session:
        mission_db = get_mission(session, mission_id)
        if not mission_db:
            raise ValueError("Mission not found")
        polygon = json.loads(mission_db.polygon_json)
        track_spacing_m = getattr(mission_db, "track_spacing_m", None) or mission_db.lane_spacing or 5.0
        waypoint_step_m = getattr(mission_db, "waypoint_step_m", None) or 2.0
        route = json.loads(mission_db.route_json) if mission_db.route_json else plan_lawnmower_path(
            [(p[0], p[1]) for p in polygon],
            track_spacing_m=track_spacing_m,
            waypoint_step_m=waypoint_step_m,
            altitude=mission_db.altitude,
        )
        ok, reason = validate_polygon([(p[0], p[1]) for p in polygon])
        if not ok:
            raise ValueError(reason or "Invalid polygon")

        mission = Mission(
            id=mission_id,
            area=polygon,
            altitude=mission_db.altitude,
            speed=mission_db.speed,
            status=mission_db.status,
        )
        mission.start()
        mission_db.status = mission.status
        mission_db.start_time = int(mission.start_time) if mission.start_time else None
        if not mission_db.route_json and route:
            mission_db.route_json = json.dumps(route)
        session.commit()

    controller = UAVController()
    controller.arm()
    controller.takeoff()
    controller.fly_route(route)

    sim_result = simulator.simulate(mission_id)
    run_id = sim_result.get("run_id")

    records: List[Record] = []
    history_candidates = []

    if run_id:
        with session_factory() as session:
            frames = get_frames(session, run_id)
            telemetry = get_telemetry(session, run_id)
        for fr in frames:
            img_path = fr["image_path"]
            if not Path(img_path).exists():
                continue
            image_bytes = Path(img_path).read_bytes()
            result = detect_api.detect_api(image_bytes, model=model, draw_all=draw_all)
            preds = result["predictions"]
            result_img = result["image"]

            result_path = detect_api.save_image(result_img, "web/static/results")
            result_url = f"/static/results/{os.path.basename(result_path)}"
            original_url = _frame_to_url(img_path)

            top_score = 0.0
            if model in preds:
                scores = [box[4] for box in preds.get(model, [])]
                top_score = max(scores) if scores else 0.0
            history_candidates.append((top_score, original_url, result_url))

            telem = nearest_telemetry(fr["timestamp"], telemetry)
            record = Record(
                timestamp=fr["timestamp"],
                model=model,
                lat=fr["lat"],
                lon=fr["lon"],
                altitude=fr["alt"],
                battery=telem["battery"] if telem else 0.0,
                signal=telem["signal"] if telem else 0.0,
            )
            records.append(record)

    history_candidates.sort(key=lambda item: item[0], reverse=True)
    history_added = 0
    for score, original_url, result_url in history_candidates[:history_limit]:
        if score <= 0:
            continue
        detect_api.add_history_entry(
            model=model,
            threshold=threshold,
            original_url=original_url,
            result_url=result_url,
        )
        history_added += 1

    process_result = processor.process(mission_id=mission_id, run_id=run_id, model=model)
    with session_factory() as session:
        risk_zones = get_risk_zones(session, mission_id)

    controller.land()

    return {
        "mission_id": mission_id,
        "run_id": run_id,
        "records_count": len(records),
        "records_sample": [_record_to_dict(r) for r in records[:5]],
        "history_added": history_added,
        "risk_zones": risk_zones,
        "detections": process_result.get("detections"),
        "zones": process_result.get("zones"),
        "exports": process_result.get("exports"),
        "sample_detection": process_result.get("sample_detection"),
        "risk_level": process_result.get("risk_level"),
        "priority_zone": process_result.get("priority_zone"),
        "data_quality": process_result.get("data_quality"),
        "recommendations": process_result.get("recommendations"),
        "preview_path": sim_result.get("preview_path"),
    }

import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from core.geo import haversine_m, meters_per_degree
from core.repository import (
    bulk_insert_detections,
    get_detections,
    get_frames,
    get_mines,
    get_metal,
    get_mission,
    get_risk_zones,
    get_run,
    get_telemetry,
    save_risk_zones,
)


def nearest_telemetry(frame_ts: int, telemetry: List[dict]):
    if not telemetry:
        return None
    return min(telemetry, key=lambda t: abs(t["timestamp"] - frame_ts))


def metal_window(frame_ts: int, metal_samples: List[dict], window_ms: int = 500):
    return [m for m in metal_samples if abs(m["timestamp"] - frame_ts) <= window_ms]


def cluster_detections(detections: List[dict], eps_m: float = 5.0, min_samples: int = 1):
    clusters: List[List[dict]] = []
    visited = set()
    for i, det in enumerate(detections):
        if i in visited:
            continue
        neighbors = []
        for j, other in enumerate(detections):
            if i == j:
                continue
            dist = haversine_m(det["lat"], det["lon"], other["lat"], other["lon"])
            if dist <= eps_m:
                neighbors.append(j)
        if len(neighbors) + 1 < min_samples:
            continue
        cluster_idxs = set([i] + neighbors)
        expanded = True
        while expanded:
            expanded = False
            for j in list(cluster_idxs):
                if j in visited:
                    continue
                visited.add(j)
                for k, other in enumerate(detections):
                    if k in cluster_idxs:
                        continue
                    dist = haversine_m(detections[j]["lat"], detections[j]["lon"], other["lat"], other["lon"])
                    if dist <= eps_m:
                        cluster_idxs.add(k)
                        expanded = True
        clusters.append([detections[k] for k in cluster_idxs])
    return clusters


def make_square_geometry(center_lat: float, center_lon: float, buffer_m: float):
    m_per_lat, m_per_lon = meters_per_degree(center_lat)
    d_lat = buffer_m / m_per_lat
    d_lon = buffer_m / m_per_lon
    return [
        [center_lat + d_lat, center_lon - d_lon],
        [center_lat + d_lat, center_lon + d_lon],
        [center_lat - d_lat, center_lon + d_lon],
        [center_lat - d_lat, center_lon - d_lon],
        [center_lat + d_lat, center_lon - d_lon],
    ]


class MissionProcessor:
    def __init__(self, session_factory, detector, exports_dir: str | Path = "data/exports"):
        self.session_factory = session_factory
        self.detector = detector
        self.exports_dir = Path(exports_dir)
        os.makedirs(self.exports_dir, exist_ok=True)

    def _evaluate_data_quality(self, frames: List[dict], telemetry: List[dict], metal: List[dict]):
        issues = []
        strong_metal = [m for m in metal if m["value"] > m.get("adaptive_threshold", 0)]
        if len(frames) < 3:
            issues.append("low_frame_count")
        if len(telemetry) < 10:
            issues.append("low_telemetry_count")
        if not strong_metal:
            issues.append("no_metal_events")
        status = "ok" if not issues else "needs_rescan"
        return {
            "status": status,
            "issues": issues,
            "frames": len(frames),
            "telemetry": len(telemetry),
            "metal_events": len(strong_metal),
        }

    def _summarize_risk(self, mission_params: dict, zones: List[dict], data_quality: dict):
        if not zones:
            risk_level = "clear" if data_quality.get("status") == "ok" else "uncertain"
            recs = []
            if data_quality.get("issues"):
                recs.append("Data quality issues: " + ", ".join(data_quality["issues"]))
                recs.append("Rescan recommended with tighter lane spacing or reduced speed.")
            else:
                recs.append("No risk zones detected; mission can be closed.")
            return risk_level, None, recs

        top_zone = max(zones, key=lambda z: z["score"])
        risk_level = "danger" if top_zone["score"] >= 0.8 else "warn"
        recs = [
            f"Priority zone {top_zone['id']} (score={top_zone['score']:.2f}) at ({top_zone['center_lat']:.5f}, {top_zone['center_lon']:.5f}).",
        ]
        recommended_speed = max(0.5, mission_params.get("speed", 1.0) * 0.8)
        recs.append(f"Plan a repeat pass with 5 m buffer and speed {recommended_speed:.2f} m/s for confirmation.")
        if data_quality.get("issues"):
            recs.append("Data issues observed: " + ", ".join(data_quality["issues"]))
        return risk_level, top_zone, recs

    def process(self, mission_id: str, run_id: str | None = None, model: str = "ensemble"):
        with self.session_factory() as session:
            mission = get_mission(session, mission_id)
            if not mission:
                raise ValueError("Mission not found")
            run = get_run(session, run_id) if run_id else None
            if not run:
                from core.repository import get_latest_run_for_mission

                run = get_latest_run_for_mission(session, mission_id)
            if not run:
                raise ValueError("No run to process")
            frames = get_frames(session, run.id)
            telemetry = get_telemetry(session, run.id)
            metal = get_metal(session, run.id)
            mission_params = {
                "altitude": mission.altitude,
                "speed": mission.speed,
                "lane_spacing": mission.lane_spacing,
                "track_spacing_m": getattr(mission, "track_spacing_m", None),
                "waypoint_step_m": getattr(mission, "waypoint_step_m", None),
                "capture_every_m": getattr(mission, "capture_every_m", None),
            }
            mines = get_mines(session, mission_id)
            sensor_radius_m = max(1.2, getattr(mission, "capture_every_m", 2.0))

        detections_payload = []
        frame_lookup = {f["id"]: f for f in frames}
        for fr in frames:
            if mines:
                min_dist = min(haversine_m(fr["lat"], fr["lon"], m["lat"], m["lon"]) for m in mines)
                if min_dist > sensor_radius_m:
                    continue
            img = cv2.imread(fr["image_path"])
            if img is None:
                continue
            preds = self.detector.predict(img, mode=model)
            boxes = preds.get(model, [])
            for (x1, y1, x2, y2, score) in boxes:
                metal_ctx = metal_window(fr["timestamp"], metal)
                confirmed = any(m["value"] > m["adaptive_threshold"] for m in metal_ctx)
                label = "confirmed" if confirmed else "review"
                boost = 0.3 if confirmed else 0.05
                detections_payload.append(
                    {
                        "frame_id": fr["id"],
                        "model": model,
                        "score": float(score),
                        "bbox_json": json.dumps([float(x1), float(y1), float(x2), float(y2)]),
                        "confirmed_by_metal": confirmed,
                        "risk_score": float(score + boost),
                        "label": label,
                        "lat": fr["lat"],
                        "lon": fr["lon"],
                        "alt": fr["alt"],
                    }
                )

        with self.session_factory() as session:
            bulk_insert_detections(session, run.id, detections_payload)
            session.commit()

        with self.session_factory() as session:
            detections = get_detections(session, run.id)
            clusters = cluster_detections(detections, eps_m=5.0, min_samples=1)
            zones = []
            for idx, cl in enumerate(clusters):
                if not cl:
                    continue
                center_lat = sum(d["lat"] for d in cl) / len(cl)
                center_lon = sum(d["lon"] for d in cl) / len(cl)
                score = sum(d["risk_score"] for d in cl) / len(cl)
                geometry = make_square_geometry(center_lat, center_lon, buffer_m=5.0)
                zones.append(
                    {
                        "id": idx,
                        "center_lat": center_lat,
                        "center_lon": center_lon,
                        "geometry": geometry,
                        "score": score,
                        "detections_count": len(cl),
                    }
                )
            save_risk_zones(session, mission_id, run.id, zones)
            mission = get_mission(session, mission_id)
            if mission:
                mission.status = "processed"
            session.commit()

        # choose a sample detection to preview
        sample = None
        if detections:
            top = sorted(detections, key=lambda d: d["risk_score"], reverse=True)[0]
            frame = frame_lookup.get(top["frame_id"])
            if frame:
                img_path = frame["image_path"]
                fname = os.path.basename(img_path)
                image_url = f"/test_images/{fname}"
                sample = {
                    "image_url": image_url,
                    "score": top["score"],
                    "risk_score": top["risk_score"],
                    "confirmed_by_metal": top["confirmed_by_metal"],
                    "model": top["model"],
                    "timestamp": frame["timestamp"],
                }

        exports = self._build_exports(mission_id, run.id)
        data_quality = self._evaluate_data_quality(frames, telemetry, metal)
        risk_level, priority_zone, recs = self._summarize_risk(mission_params, zones, data_quality)
        return {
            "mission_id": mission_id,
            "run_id": run.id,
            "detections": len(detections_payload),
            "zones": exports["risk_zones_count"],
            "exports": exports,
            "sample_detection": sample,
            "risk_level": risk_level,
            "priority_zone": priority_zone,
            "data_quality": data_quality,
            "recommendations": recs,
        }

    def _build_exports(self, mission_id: str, run_id: str):
        with self.session_factory() as session:
            mission = get_mission(session, mission_id)
            detections = get_detections(session, run_id)
            zones = get_risk_zones(session, mission_id)

        geojson_path = self.exports_dir / f"mission_{mission_id}_{run_id}.geojson"
        csv_path = self.exports_dir / f"mission_{mission_id}_{run_id}.csv"
        pdf_path = self.exports_dir / f"mission_{mission_id}_{run_id}.pdf"

        # GeoJSON
        features = []
        if mission:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[p[1], p[0]] for p in json.loads(mission.polygon_json)]],
                    },
                    "properties": {"id": mission.id, "type": "mission_area"},
                }
            )
        for det in detections:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [det["lon"], det["lat"]]},
                    "properties": {
                        "score": det["score"],
                        "confirmed_by_metal": det["confirmed_by_metal"],
                        "risk_score": det["risk_score"],
                    },
                }
            )
        for zone in zones:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[pt[1], pt[0]] for pt in zone["geometry"]]]},
                    "properties": {
                        "score": zone["score"],
                        "detections_count": zone["detections_count"],
                        "type": "risk_zone",
                    },
                }
            )
        geojson = {"type": "FeatureCollection", "features": features}
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        # CSV
        import csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "frame_id", "model", "score", "confirmed_by_metal", "risk_score", "lat", "lon", "alt"],
            )
            writer.writeheader()
            for det in detections:
                writer.writerow({k: det.get(k) for k in writer.fieldnames})

        # PDF summary
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 800, "Mission Risk Report")
        c.setFont("Helvetica", 11)
        c.drawString(50, 780, f"Mission: {mission_id}")
        c.drawString(50, 765, f"Run: {run_id}")
        c.drawString(50, 750, f"Detections: {len(detections)}")
        c.drawString(50, 735, f"Risk zones: {len(zones)}")
        y = 710
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Zones:")
        y -= 15
        c.setFont("Helvetica", 10)
        for zone in zones:
            c.drawString(50, y, f"Zone {zone['id']} score={zone['score']:.2f} detections={zone['detections_count']}")
            y -= 14
            if y < 50:
                c.showPage()
                y = 780
                c.setFont("Helvetica", 10)
        c.save()

        return {
            "geojson": str(geojson_path),
            "csv": str(csv_path),
            "pdf": str(pdf_path),
            "risk_zones_count": len(zones),
        }

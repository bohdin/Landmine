import csv
import hashlib
import json
import math
import os
import random
import time
import uuid
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import numpy as np
from sqlalchemy.orm import Session

from core.db_models import MissionTargetModel
from core.geo import (
    haversine_m,
    jitter_lat_lon,
    polygon_area_m2,
    point_in_polygon,
    tile_id_for_point,
)
from core.mission_planner import plan_lawnmower_path
from core.repository import (
    bulk_insert_frames,
    bulk_insert_metal,
    bulk_insert_telemetry,
    end_run,
    get_mission,
    save_targets,
    start_run,
)


def compute_adaptive_threshold(window: List[float]):
    if not window:
        return 0.1
    mu = mean(window) if window else 0.0
    sigma = pstdev(window) if len(window) > 1 else 0.0
    return mu + 1.0 * sigma + 0.1


def random_point_in_polygon_rng(rng: random.Random, polygon: List[Tuple[float, float]]):
    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    for _ in range(10000):
        lat = rng.uniform(min_lat, max_lat)
        lon = rng.uniform(min_lon, max_lon)
        if point_in_polygon(lat, lon, polygon):
            return lat, lon
    return sum(lats) / len(lats), sum(lons) / len(lons)


class MissionSimulator:
    def __init__(
        self,
        session_factory,
        patches_dir: str | Path = "data/test_images/images",
        tile_size_m: float = 2.0,
        base_frame_interval_ms: int = 6000,  # рідше кадри
        trigger_cooldown_ms: int = 3500,
        max_frames: int = 30,
        telemetry_interval_ms: int = 1500,  # ще рідше телеметрія
        metal_interval_ms: int = 600,  # рідше метал
        max_telemetry: int = 400,
        max_metal: int = 800,
    ):
        self.session_factory = session_factory
        self.patches_dir = Path(patches_dir)
        self.tile_size_m = tile_size_m
        self.patch_pool = self._load_patches()
        self.patch_hash_cache: Dict[str, str] = {}
        self.base_frame_interval_ms = base_frame_interval_ms
        self.trigger_cooldown_ms = trigger_cooldown_ms
        self.max_frames = max_frames
        self.telemetry_interval_ms = telemetry_interval_ms
        self.metal_interval_ms = metal_interval_ms
        self.max_telemetry = max_telemetry
        self.max_metal = max_metal

    def _load_patches(self):
        return [p for p in self.patches_dir.glob("*.jpg")]

    def _get_patch_sha(self, path: Path) -> str:
        if path.as_posix() in self.patch_hash_cache:
            return self.patch_hash_cache[path.as_posix()]
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        digest = h.hexdigest()
        self.patch_hash_cache[path.as_posix()] = digest
        return digest

    def _ensure_targets(self, session: Session, mission_id: str, polygon_latlon: List[Tuple[float, float, float]], altitude: float, seed: int):
        existing = session.query(MissionTargetModel).filter(MissionTargetModel.mission_id == mission_id).all()
        if existing:
            return [(t.lat, t.lon, t.alt, t.signature) for t in existing]

        area = polygon_area_m2([(p[0], p[1]) for p in polygon_latlon])
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        rho = rng.uniform(0.0005, 0.002)  # менша щільність
        expected = max(1, area * rho)
        n = max(1, int(np_rng.poisson(expected)))
        targets: List[Tuple[float, float, float, float]] = []
        for _ in range(n):
            lat, lon = random_point_in_polygon_rng(rng, [(p[0], p[1]) for p in polygon_latlon])
            signature = rng.uniform(0.8, 1.4)
            targets.append((lat, lon, altitude, signature))

        save_targets(session, mission_id, targets)
        session.commit()
        return targets

    def _pick_patch_for_tile(self, tile_id: str) -> Path:
        if not self.patch_pool:
            raise RuntimeError("Patch pool is empty")
        rnd = random.Random(tile_id)
        return rnd.choice(self.patch_pool)

    def _sample_route_points(self, route: List[Tuple[float, float, float]], step_m: float):
        if len(route) < 2 or step_m <= 0:
            return []
        points: List[dict] = []
        total_dist = 0.0
        next_sample = 0.0
        prev_lat, prev_lon, prev_alt = route[0]
        points.append({"lat": prev_lat, "lon": prev_lon, "alt": prev_alt, "distance_m": 0.0})
        next_sample += step_m
        for idx in range(1, len(route)):
            lat, lon, alt = route[idx]
            seg_dist = haversine_m(prev_lat, prev_lon, lat, lon)
            if seg_dist <= 0:
                prev_lat, prev_lon, prev_alt = lat, lon, alt
                continue
            while next_sample <= total_dist + seg_dist + 1e-6:
                frac = (next_sample - total_dist) / seg_dist
                samp_lat = prev_lat + (lat - prev_lat) * frac
                samp_lon = prev_lon + (lon - prev_lon) * frac
                samp_alt = prev_alt + (alt - prev_alt) * frac
                points.append({"lat": samp_lat, "lon": samp_lon, "alt": samp_alt, "distance_m": next_sample})
                next_sample += step_m
            total_dist += seg_dist
            prev_lat, prev_lon, prev_alt = lat, lon, alt
        last = route[-1]
        if points[-1]["lat"] != last[0] or points[-1]["lon"] != last[1]:
            points.append({"lat": last[0], "lon": last[1], "alt": last[2], "distance_m": total_dist})
        return points

    def simulate(self, mission_id: str):
        with self.session_factory() as session:
            mission = get_mission(session, mission_id)
            if not mission:
                raise ValueError(f"Mission {mission_id} not found")
            polygon = json.loads(mission.polygon_json)
            mission_altitude = mission.altitude
            mission_speed = mission.speed
            track_spacing_m = getattr(mission, "track_spacing_m", None) or mission.lane_spacing or 5.0
            waypoint_step_m = getattr(mission, "waypoint_step_m", None) or 2.0
            capture_every_m = getattr(mission, "capture_every_m", None) or 2.0
            seed = mission.seed
            if seed is None:
                seed = int(uuid.UUID(mission_id)) & 0x7FFFFFFF
                mission.seed = seed
            if getattr(mission, "track_spacing_m", None) is None:
                mission.track_spacing_m = track_spacing_m
            if getattr(mission, "waypoint_step_m", None) is None:
                mission.waypoint_step_m = waypoint_step_m
            if getattr(mission, "capture_every_m", None) is None:
                mission.capture_every_m = capture_every_m
            route = json.loads(mission.route_json) if mission.route_json else plan_lawnmower_path(
                [(p[0], p[1]) for p in polygon],
                track_spacing_m=track_spacing_m,
                waypoint_step_m=waypoint_step_m,
                altitude=mission_altitude,
            )
            if not mission.route_json:
                mission.route_json = json.dumps(route)
            mission.status = "running"
            mission.start_time = int(time.time() * 1000)
            targets = self._ensure_targets(session, mission_id, polygon, mission_altitude, seed)
            session.commit()

        if len(route) < 2:
            return {"mission_id": mission_id, "message": "Route too short", "frames": 0}

        origin = (route[0][0], route[0][1])
        telemetry_samples = []
        metal_samples = []
        frames = []
        preview_trail = []

        run_id = str(uuid.uuid4())
        start_ts = int(time.time() * 1000)
        frame_dir = Path("data") / "runs" / mission_id / run_id
        os.makedirs(frame_dir, exist_ok=True)

        last_trigger_ts = start_ts
        metal_window: List[float] = []
        telemetry_interval = self.telemetry_interval_ms
        frame_index = 0
        cooldown_ms = self.trigger_cooldown_ms
        max_frames = self.max_frames
        max_telemetry = self.max_telemetry
        max_metal = self.max_metal
        next_telemetry_ts = start_ts

        with self.session_factory() as session:
            start_run(session, mission_id, run_id, started_at=start_ts)
            session.commit()

        current_ts = start_ts
        battery = 100.0
        signal = 1.0
        rng = random.Random(seed + 17)
        capture_points = self._sample_route_points(route, capture_every_m)
        if not capture_points:
            return {"mission_id": mission_id, "message": "No capture points", "frames": 0}
        frame_stride = max(1, int(math.ceil(len(capture_points) / max_frames)))
        sensor_radius_m = max(1.2, capture_every_m)

        prev_lat = capture_points[0]["lat"]
        prev_lon = capture_points[0]["lon"]

        for idx, point in enumerate(capture_points):
            lat = point["lat"]
            lon = point["lon"]
            alt = point["alt"]

            seg_dist = haversine_m(prev_lat, prev_lon, lat, lon) if idx > 0 else 0.0
            travel_ms = int((seg_dist / max(0.1, mission_speed)) * 1000)
            current_ts += travel_ms

            orig_lat = lat
            orig_lon = lon
            drift_x = rng.uniform(-0.15, 0.15)
            drift_y = rng.uniform(-0.15, 0.15)
            lat, lon = jitter_lat_lon(lat, lon, drift_x, drift_y)

            nearest_target = None
            max_signal = 0.0
            for (t_lat, t_lon, _, signature) in targets:
                dist = haversine_m(lat, lon, t_lat, t_lon)
                if nearest_target is None or dist < nearest_target:
                    nearest_target = dist
                contribution = signature * math.exp(-((dist) ** 2) / (2 * (0.8 ** 2)))
                max_signal = max(max_signal, contribution)
            noise = rng.normalvariate(0.05, 0.02)
            metal_value = max(0.0, noise + max_signal)
            metal_window.append(metal_value)
            if len(metal_window) > 40:
                metal_window.pop(0)
            adaptive_threshold = compute_adaptive_threshold(metal_window)

            if current_ts >= next_telemetry_ts and len(telemetry_samples) < max_telemetry:
                telemetry_samples.append(
                    {
                        "timestamp": current_ts,
                        "lat": lat,
                        "lon": lon,
                        "alt": mission_altitude,
                        "battery": max(0.0, battery),
                        "signal": max(0.0, signal),
                        "yaw": rng.uniform(-math.pi, math.pi),
                    }
                )
                battery -= 0.002
                signal = max(0.1, signal - rng.uniform(0.0005, 0.002))
                next_telemetry_ts += telemetry_interval
                if len(telemetry_samples) % 20 == 0:
                    preview_trail.append({"lat": lat, "lon": lon})

            if len(metal_samples) < max_metal:
                metal_samples.append(
                    {
                        "timestamp": current_ts,
                        "lat": lat,
                        "lon": lon,
                        "alt": mission_altitude,
                        "value": float(metal_value),
                        "adaptive_threshold": float(adaptive_threshold),
                    }
                )

            triggered = False
            if metal_value > adaptive_threshold and (current_ts - last_trigger_ts) >= cooldown_ms:
                triggered = True
                last_trigger_ts = current_ts

            target_trigger = nearest_target is not None and nearest_target <= sensor_radius_m
            should_capture = (idx % frame_stride == 0) or triggered or target_trigger

            if should_capture and len(frames) < max_frames:
                tile_id = tile_id_for_point(lat, lon, origin, tile_size_m=self.tile_size_m)
                patch_path = self._pick_patch_for_tile(tile_id)
                patch_sha = self._get_patch_sha(patch_path)
                frames.append(
                    {
                        "timestamp": current_ts,
                        "lat": lat,
                        "lon": lon,
                        "alt": mission_altitude,
                        "image_path": str(patch_path),
                        "patch_sha256": patch_sha,
                        "triggered_by_metal": triggered or target_trigger,
                        "frame_index": frame_index,
                        "tile_id": tile_id,
                    }
                )
                frame_index += 1

            if len(frames) >= max_frames and len(telemetry_samples) >= max_telemetry and len(metal_samples) >= max_metal:
                break

            prev_lat, prev_lon = orig_lat, orig_lon

        with self.session_factory() as session:
            bulk_insert_telemetry(session, run_id, telemetry_samples)
            bulk_insert_metal(session, run_id, metal_samples)
            bulk_insert_frames(session, run_id, frames)
            end_run(session, run_id, ended_at=current_ts)
            mission = get_mission(session, mission_id)
            if mission:
                mission.status = "simulated"
                mission.end_time = current_ts
            session.commit()

        frame_index_path = frame_dir / "frame_index.csv"
        with open(frame_index_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_index",
                    "timestamp",
                    "lat",
                    "lon",
                    "alt",
                    "image_path",
                    "patch_sha256",
                    "triggered_by_metal",
                    "tile_id",
                ],
            )
            writer.writeheader()
            for fr in frames:
                writer.writerow(fr)

        mines_payload = [
            {"lat": t[0], "lon": t[1], "alt": t[2], "signature": t[3]}
            for t in targets
        ]

        return {
            "mission_id": mission_id,
            "run_id": run_id,
            "frames": len(frames),
            "telemetry": len(telemetry_samples),
            "metal_samples": len(metal_samples),
            "frame_index_path": str(frame_index_path),
            "targets_count": len(targets),
            "mines": mines_payload,
            "preview_path": preview_trail,
        }

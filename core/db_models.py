from sqlalchemy import Boolean, Column, Float, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship

from core.database import Base


class MissionModel(Base):
    __tablename__ = "missions"

    id = Column(String, primary_key=True, index=True)
    polygon_json = Column(Text, nullable=False)  # list of [lat, lon, alt]
    route_json = Column(Text, nullable=True)     # planned coverage path
    altitude = Column(Float, default=0.4)
    speed = Column(Float, default=1.0)
    lane_spacing = Column(Float, default=1.0)  # legacy, kept for compatibility
    track_spacing_m = Column(Float, default=5.0)
    waypoint_step_m = Column(Float, default=2.0)
    capture_every_m = Column(Float, default=2.0)
    seed = Column(Integer, nullable=True)
    status = Column(String, default="planned")
    start_time = Column(Integer, nullable=True)
    end_time = Column(Integer, nullable=True)
    created_at = Column(Integer, nullable=False)

    targets = relationship("MissionTargetModel", cascade="all, delete-orphan")


class MissionTargetModel(Base):
    __tablename__ = "mission_targets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mission_id = Column(String, ForeignKey("missions.id"), index=True, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    alt = Column(Float, nullable=False, default=0.0)
    signature = Column(Float, default=1.0)  # relative strength for metal simulator (mine signature)


class RunModel(Base):
    __tablename__ = "runs"

    id = Column(String, primary_key=True, index=True)
    mission_id = Column(String, ForeignKey("missions.id"), index=True, nullable=False)
    status = Column(String, default="simulated")
    started_at = Column(Integer, nullable=False)
    ended_at = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)


class TelemetrySample(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.id"), index=True, nullable=False)
    timestamp = Column(Integer, index=True, nullable=False)  # ms
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    alt = Column(Float, nullable=False)
    battery = Column(Float, nullable=False)
    signal = Column(Float, nullable=False)
    yaw = Column(Float, nullable=True)


class MetalSample(Base):
    __tablename__ = "metal_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.id"), index=True, nullable=False)
    timestamp = Column(Integer, index=True, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    alt = Column(Float, nullable=False)
    value = Column(Float, nullable=False)
    adaptive_threshold = Column(Float, nullable=False)


class FrameModel(Base):
    __tablename__ = "frames"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.id"), index=True, nullable=False)
    timestamp = Column(Integer, index=True, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    alt = Column(Float, nullable=False)
    image_path = Column(Text, nullable=False)
    patch_sha256 = Column(String, nullable=True)
    triggered_by_metal = Column(Boolean, default=False)
    frame_index = Column(Integer, nullable=False, default=0)
    tile_id = Column(String, nullable=True)


class DetectionModel(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.id"), index=True, nullable=False)
    frame_id = Column(Integer, ForeignKey("frames.id"), index=True, nullable=False)
    model = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    bbox_json = Column(Text, nullable=False)  # [x1, y1, x2, y2]
    confirmed_by_metal = Column(Boolean, default=False)
    risk_score = Column(Float, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    alt = Column(Float, nullable=False)
    label = Column(String, nullable=False, default="review")  # "confirmed" or "review"


class RiskZoneModel(Base):
    __tablename__ = "risk_zones"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mission_id = Column(String, ForeignKey("missions.id"), index=True, nullable=False)
    run_id = Column(String, ForeignKey("runs.id"), index=True, nullable=False)
    center_lat = Column(Float, nullable=False)
    center_lon = Column(Float, nullable=False)
    geometry_json = Column(Text, nullable=False)  # polygon lat/lon
    score = Column(Float, nullable=False)
    detections_count = Column(Integer, nullable=False)

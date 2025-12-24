import math
import random
from typing import Iterable, List, Tuple


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    m_per_lat = 111132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    m_per_lon = (math.pi / 180) * 6367449 * math.cos(lat_rad)
    return m_per_lat, m_per_lon


def to_local_xy(lat: float, lon: float, origin: tuple[float, float]) -> tuple[float, float]:
    m_per_lat, m_per_lon = meters_per_degree(origin[0])
    dx = (lon - origin[1]) * m_per_lon
    dy = (lat - origin[0]) * m_per_lat
    return dx, dy


def to_lat_lon(x: float, y: float, origin: tuple[float, float]) -> tuple[float, float]:
    m_per_lat, m_per_lon = meters_per_degree(origin[0])
    lat = origin[0] + y / m_per_lat
    lon = origin[1] + x / m_per_lon
    return lat, lon


def polygon_area_m2(polygon: Iterable[tuple[float, float]]) -> float:
    pts = list(polygon)
    if len(pts) < 3:
        return 0.0
    # Convert to local meters relative to first point for a reasonable approximation
    origin = (pts[0][0], pts[0][1])
    local = [to_local_xy(lat, lon, origin) for lat, lon in pts]
    area = 0.0
    for i in range(len(local)):
        x1, y1 = local[i]
        x2, y2 = local[(i + 1) % len(local)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def point_in_polygon(lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm; polygon is list of (lat, lon)."""
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        lat1, lon1 = polygon[i]
        lat2, lon2 = polygon[(i + 1) % n]
        cond = ((lon1 > lon) != (lon2 > lon)) and (
            lat < (lat2 - lat1) * (lon - lon1) / (lon2 - lon1 + 1e-9) + lat1
        )
        if cond:
            inside = not inside
    return inside


def random_point_in_polygon(polygon: List[Tuple[float, float]]):
    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    for _ in range(10000):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        if point_in_polygon(lat, lon, polygon):
            return lat, lon
    # fallback: return centroid-ish
    return sum(lats) / len(lats), sum(lons) / len(lons)


def jitter_lat_lon(lat: float, lon: float, dx_m: float, dy_m: float) -> tuple[float, float]:
    origin = (lat, lon)
    m_per_lat, m_per_lon = meters_per_degree(lat)
    return lat + dy_m / m_per_lat, lon + dx_m / m_per_lon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def tile_id_for_point(lat: float, lon: float, origin: tuple[float, float], tile_size_m: float = 2.0) -> str:
    x, y = to_local_xy(lat, lon, origin)
    col = int(math.floor(x / tile_size_m))
    row = int(math.floor(y / tile_size_m))
    return f"{row}_{col}"

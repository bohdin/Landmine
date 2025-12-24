from typing import List, Tuple

from core.geo import point_in_polygon, to_lat_lon, to_local_xy


def plan_lawnmower_path(
    polygon: List[Tuple[float, float]],
    track_spacing_m: float = 5.0,
    waypoint_step_m: float = 2.0,
    altitude: float = 0.4,
) -> List[Tuple[float, float, float]]:
    """
    Simple lawnmower coverage planner that walks across the bounding box and keeps only points inside the polygon.
    """
    if len(polygon) < 3:
        return []
    if track_spacing_m <= 0 or waypoint_step_m <= 0:
        return []

    origin = (polygon[0][0], polygon[0][1])
    local = [to_local_xy(lat, lon, origin) for lat, lon in polygon]
    xs = [p[0] for p in local]
    ys = [p[1] for p in local]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    route: List[Tuple[float, float, float]] = []
    lane_idx = 0
    x = min_x
    while x <= max_x + 1e-6:
        lane_points: List[Tuple[float, float, float]] = []
        y = min_y
        while y <= max_y + 1e-6:
            lat, lon = to_lat_lon(x, y, origin)
            if point_in_polygon(lat, lon, polygon):
                lane_points.append((lat, lon, altitude))
            y += waypoint_step_m
        if lane_idx % 2 == 1:
            lane_points.reverse()
        route.extend(lane_points)
        x += track_spacing_m
        lane_idx += 1
    return route

from core.geo import point_in_polygon, to_local_xy
from core.mission_planner import plan_lawnmower_path


def test_lawnmower_path_spacing_and_inside():
    polygon = [
        (48.5, 32.25),
        (48.50008, 32.25),
        (48.50008, 32.2501),
        (48.5, 32.2501),
    ]
    route = plan_lawnmower_path(polygon, track_spacing_m=1.0, waypoint_step_m=1.0, altitude=0.4)
    assert route, "Route should not be empty"
    origin = (polygon[0][0], polygon[0][1])
    xs = []
    for lat, lon, alt in route:
        assert abs(alt - 0.4) < 1e-6
        assert point_in_polygon(lat, lon, [(p[0], p[1]) for p in polygon])
        x, _ = to_local_xy(lat, lon, origin)
        xs.append(round(x, 1))
    xs = sorted(set(xs))
    diffs = [abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
    assert any(0.8 <= d <= 1.2 for d in diffs), f"Lane spacing deviations look wrong: {diffs}"

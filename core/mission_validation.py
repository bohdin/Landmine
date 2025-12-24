from typing import List, Tuple

from core.geo import polygon_area_m2


def segments_intersect(p1, p2, p3, p4):
    def orient(a, b, c):
        return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])

    def on_segment(a, b, c):
        return min(a[0], c[0]) - 1e-12 <= b[0] <= max(a[0], c[0]) + 1e-12 and min(a[1], c[1]) - 1e-12 <= b[1] <= max(a[1], c[1]) + 1e-12

    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)

    if o1 == 0 and on_segment(p1, p3, p2):
        return True
    if o2 == 0 and on_segment(p1, p4, p2):
        return True
    if o3 == 0 and on_segment(p3, p1, p4):
        return True
    if o4 == 0 and on_segment(p3, p2, p4):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def validate_polygon(latlon_polygon: List[Tuple[float, float]], min_area_m2: float = 1.0):
    if len(latlon_polygon) < 3:
        return False, "Polygon must contain at least 3 points"
    area = polygon_area_m2(latlon_polygon)
    if area < min_area_m2:
        return False, f"Polygon area too small ({area:.2f} m^2)"
    n = len(latlon_polygon)
    for i in range(n):
        a1 = latlon_polygon[i]
        a2 = latlon_polygon[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            b1 = latlon_polygon[j]
            b2 = latlon_polygon[(j + 1) % n]
            if segments_intersect(a1, a2, b1, b2):
                return False, "Polygon edges intersect"
    return True, None

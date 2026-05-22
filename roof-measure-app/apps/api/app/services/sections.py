"""Polygon-Aufteilung in Dachflächen-Sektionen.

Strategie:
1. Polygon + Linien nach UTM-32N projizieren (metrische Koordinaten).
2. Endpunkte jeder First/Kehle nach Nähe auf das nächstgelegene Target snappen
   (Boundary oder Endpunkt einer anderen Linie). Damit kreuzen sich Linien exakt
   topologisch, statt knapp daneben.
3. `unary_union` der gesnappten Linien + Polygon-Rand → erzeugt Topologie mit
   sauberen Schnittpunkten.
4. `polygonize` findet alle minimalen geschlossenen Faces.
5. Filter auf Faces innerhalb des Original-Polygons.
"""

from __future__ import annotations

from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, polygonize, unary_union

_TO_UTM = Transformer.from_crs(4326, 25832, always_xy=True)
_FROM_UTM = Transformer.from_crs(25832, 4326, always_xy=True)


def _latlon_to_utm(lat: float, lon: float) -> tuple[float, float]:
    x, y = _TO_UTM.transform(lon, lat)
    return x, y


def _utm_to_latlon(x: float, y: float) -> tuple[float, float]:
    lon, lat = _FROM_UTM.transform(x, y)
    return lat, lon


SNAP_M = 3.0  # großzügig: snappt auch Walmgiebel-Spitzen die ~1-2m daneben liegen
MIN_AREA_M2 = 0.5
NEAR_VERTEX_M = 0.01
EXTEND_M = 0.3  # winzige Linien-Verlängerung über den Snap-Punkt hinaus


def _snap_endpoint(p: Point, targets: list) -> tuple[Point, bool]:
    best = p
    best_d = SNAP_M
    for t in targets:
        if t.is_empty:
            continue
        d = p.distance(t)
        if d < best_d:
            _, near = nearest_points(p, t)
            best = near
            best_d = d
    return best, best is not p


def _snap_line(line: LineString, targets: list) -> tuple[LineString, list[Point], bool, bool]:
    coords = list(line.coords)
    if len(coords) < 2:
        return line, [], False, False
    p1, p2 = Point(coords[0]), Point(coords[-1])
    s1, snapped1 = _snap_endpoint(p1, targets)
    s2, snapped2 = _snap_endpoint(p2, targets)
    snap_pts = [s1] if snapped1 else []
    if snapped2:
        snap_pts.append(s2)
    return LineString([(s1.x, s1.y), (s2.x, s2.y)]), snap_pts, snapped1, snapped2


def _extend_line(line: LineString, extend1: bool, extend2: bool) -> LineString:
    """Verlängert Endpunkte minimal über das Original hinaus, falls gesnappt wurde —
    damit Schnitterkennung in unary_union zuverlässig greift."""
    if not (extend1 or extend2):
        return line
    coords = list(line.coords)
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    dx, dy = x2 - x1, y2 - y1
    import math

    length = math.hypot(dx, dy)
    if length < 1e-6:
        return line
    ux, uy = dx / length, dy / length
    new1 = (x1 - ux * EXTEND_M, y1 - uy * EXTEND_M) if extend1 else (x1, y1)
    new2 = (x2 + ux * EXTEND_M, y2 + uy * EXTEND_M) if extend2 else (x2, y2)
    return LineString([new1, new2])


def _insert_points_into_line(line_coords: list[tuple[float, float]], points: list[Point]) -> list[tuple[float, float]]:
    """Füge Punkte als Vertices in die Polygon-Boundary ein, falls sie auf einer
    Kante liegen aber noch nicht als Vertex existieren. Dadurch wird sichergestellt,
    dass `unary_union` saubere Topologie aufbaut."""
    result = list(line_coords)
    for pt in points:
        # Existiert der Punkt schon als Vertex?
        if any(((pt.x - x) ** 2 + (pt.y - y) ** 2) ** 0.5 < NEAR_VERTEX_M for x, y in result):
            continue
        # Finde die Edge zu der der Punkt am nächsten ist und füge dort ein.
        best_idx = -1
        best_d = float("inf")
        for i in range(len(result) - 1):
            ax, ay = result[i]
            bx, by = result[i + 1]
            seg = LineString([(ax, ay), (bx, by)])
            d = seg.distance(pt)
            if d < best_d:
                best_d = d
                best_idx = i
        if best_idx >= 0 and best_d < NEAR_VERTEX_M * 100:
            result.insert(best_idx + 1, (pt.x, pt.y))
    return result


def compute_sections(
    polygon_latlon: list[tuple[float, float]],
    ridges_latlon: list[list[tuple[float, float]]],
    valleys_latlon: list[list[tuple[float, float]]],
) -> list[dict]:
    if len(polygon_latlon) < 3:
        return []

    poly_utm = [_latlon_to_utm(lat, lon) for lat, lon in polygon_latlon]
    polygon = Polygon(poly_utm)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return []

    raw_lines: list[LineString] = []
    for r in ridges_latlon:
        if len(r) == 2:
            raw_lines.append(LineString([_latlon_to_utm(lat, lon) for lat, lon in r]))
    for v in valleys_latlon:
        if len(v) == 2:
            raw_lines.append(LineString([_latlon_to_utm(lat, lon) for lat, lon in v]))

    boundary = polygon.exterior
    boundary_coords = list(boundary.coords)
    boundary_ls = LineString(boundary_coords)

    # Iteratives Snapping: in jeder Iteration werden Endpunkte aller Linien an die
    # aktuell besten Targets gesnappt — nach 2-3 Passes konvergiert das.
    lines = list(raw_lines)
    for _ in range(3):
        new_lines: list[LineString] = []
        for i, line in enumerate(lines):
            targets = [boundary_ls] + [l for j, l in enumerate(lines) if j != i]
            s_line, _, snapped1, snapped2 = _snap_line(line, targets)
            new_lines.append(_extend_line(s_line, snapped1, snapped2))
        lines = new_lines

    # Punkte sammeln, die auf der Boundary liegen → als Vertices ergänzen, damit
    # polygonize die Boundary an diesen Stellen sauber splittet.
    points_on_boundary: list[Point] = []
    for line in lines:
        for c in (line.coords[0], line.coords[-1]):
            pt = Point(c)
            if pt.distance(boundary_ls) < NEAR_VERTEX_M * 100:  # ~1cm Toleranz
                # Exact projection auf boundary
                _, proj = nearest_points(pt, boundary_ls)
                points_on_boundary.append(proj)

    boundary_coords = _insert_points_into_line(boundary_coords, points_on_boundary)
    boundary_ls = LineString(boundary_coords)

    merged = unary_union([boundary_ls, *lines])
    sub_polygons = list(polygonize(merged))

    sections: list[dict] = []
    for sp in sub_polygons:
        if sp.area < MIN_AREA_M2:
            continue
        # Nur Faces innerhalb des Originalpolygons. Buffer minimal, falls
        # die Sub-Polygon-Repräsentativ-Punkte exakt auf der Boundary liegen.
        rep = sp.representative_point()
        if not polygon.buffer(0.01).contains(rep):
            continue
        coords = list(sp.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) < 3:
            continue
        sections.append(
            {
                "points": [_utm_to_latlon(x, y) for x, y in coords],
                "area_m2_2d": float(sp.area),
            }
        )

    return sections

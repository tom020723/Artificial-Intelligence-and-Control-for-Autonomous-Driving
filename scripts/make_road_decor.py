"""Write map-style decoration (solid background, grass, roadside trees) for sumo-gui.

The output is a separate SUMO additional file (``highway.decor.xml``) made only of
polygons. It is loaded only when the environment opens sumo-gui (off with SUMO_GUI_DECOR=0),
never by training or headless evaluation, so it cannot change simulation results.

    python scripts/make_road_decor.py env/sumo/highway.net.xml
    sumo-gui -c env/sumo/highway.sumocfg -a env/sumo/highway.scenery.xml,env/sumo/highway.decor.xml

Geometry is computed on a 0.5 m grid: the distance from every cell to the nearest
road surface gives the grass band (closed loops of road are filled, so the empty
land inside the intersection becomes grass) and the line where trees are planted.
Lanes closed to all traffic (drawn orange by sumo-gui) are covered with grass,
together with the junction corners that exist only for them.
"""
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

CELL = 0.25              # grid resolution (m)
GRASS_WIDTH = 14.0       # grass band beside the road (m)
TREE_OFFSET = 5.0        # distance from road edge to tree centers (m)
TREE_SPACING = 14.0      # distance between neighboring trees (m)
TREE_RADIUS = 2.3        # crown radius (m)
DEAD_END_CLEARANCE = 12.0  # no trees in front of the open road ends
BACKGROUND_EXTENT = 5000.0  # half-size of the background sheet (m)

BACKGROUND_COLOR = "238,236,229"   # light warm gray, like a paper map
GRASS_COLOR = "168,196,150"        # same green as the existing traffic islands
TREE_COLOR = "86,140,82"
TREE_TOP_COLOR = "112,163,100"

# Lanes closed to all traffic (disallow="all") are drawn in orange by sumo-gui.
# They are covered with grass; the road file itself is left unchanged.
CLOSED_LANE_GROW = 0.7     # cover grows this much, swallowing thin leftovers (m)
CLOSED_STUB_REACH = 14.0   # junction area this close to a closed lane is covered (m)
CLOSED_LANE_GAP = 0.15     # keep the cover this far outside open lanes (m)

# Junction outlines from netconvert can twist (sumo-gui then leaves holes where
# the outline overlaps itself), leave narrow slits next to a lane, or bulge out
# past the lanes. sumo-gui's "real world" view does not draw the lanes inside a
# junction, so those holes show through. The road surface is tidied on the grid:
# junction area that no lane sweeps is covered with grass, and a sheet of
# asphalt is laid under the rest (slits narrower than 2 * GAP_FILL_RADIUS closed,
# shrunk by ASPHALT_INSET so it never shows past the road edge). Open lanes are
# never covered.
ASPHALT_COLOR = "0,0,0"            # sumo-gui "real world" road color
GAP_FILL_RADIUS = 1.5      # (m)
ASPHALT_INSET = 0.4        # (m)
JUNCTION_FILL_RADIUS = 2.0 # gaps between junction lanes narrower than twice this stay road (m)
EXCESS_LANE_GAP = 0.15     # junction trimming stops this far outside open lanes (m)
ASPHALT_LAYER = -0.6           # above the scenery islands (-1), below the lanes

# Negative layers stay underneath the road. The closed-lane cover has to be drawn
# above the black bars sumo-gui puts at lane ends; no vehicle ever drives there,
# and trees stand at least TREE_OFFSET - TREE_RADIUS away from open lanes.
BACKGROUND_LAYER = -5
GRASS_LAYER = -4
COVER_LAYER = 200
TREE_LAYER = 201


def _lane_pieces(net, closed):
    """(polyline, half width) of lanes open to traffic, or of closed lanes (disallow="all")."""
    for edge in net.getEdges(withInternal=True):
        for lane in edge.getLanes():
            if (not lane.getPermissions()) == closed:
                yield np.asarray(lane.getShape(), float), lane.getWidth() / 2


def _junction_shapes(net):
    return [node.getShape() for node in net.getNodes() if len(node.getShape()) >= 3]


def _distance_fields(net, margin):
    """Distances (m) from each grid cell to open lanes, closed lanes and junction areas.

    Each field is <= 0 on its surface. The junction field is negative inside the
    junction polygons. The last result marks the cells in or within 1 m of a
    junction whose outline folds back on itself in a sharp spike.
    """
    from matplotlib.path import Path as MplPath

    xmin, ymin, xmax, ymax = net.getBoundary()
    xs = np.arange(xmin - margin, xmax + margin + CELL, CELL)
    ys = np.arange(ymin - margin, ymax + margin + CELL, CELL)
    junctions = _junction_shapes(net)
    open_lanes = _segment_field(xs, ys, _lane_pieces(net, closed=False))
    closed_lanes = _segment_field(xs, ys, _lane_pieces(net, closed=True))
    junction = _segment_field(xs, ys, ((np.asarray(list(s) + [s[0]], float), 0.0) for s in junctions))
    gx, gy = np.meshgrid(xs, ys)
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    twisted = np.zeros(junction.shape, bool)
    for shape in junctions:
        inside = MplPath(shape).contains_points(cells).reshape(junction.shape)
        junction[inside] = -np.abs(junction[inside])
        if _has_spike(shape):
            outline = _segment_field(xs, ys, [(np.asarray(list(shape) + [shape[0]], float), 0.0)])
            twisted |= inside | (outline <= 1.0)
    return xs, ys, open_lanes, closed_lanes, junction, twisted


def _has_spike(shape, max_angle=15.0):
    """True if the polygon folds back on itself: some corner is sharper than max_angle degrees."""
    pts = [np.asarray(p[:2], float) for p in shape]
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    n = len(pts)
    for i in range(n):
        u, v = pts[i - 1] - pts[i], pts[(i + 1) % n] - pts[i]
        lu, lv = math.hypot(*u), math.hypot(*v)
        if lu > 1.0 and lv > 1.0 and math.degrees(math.acos(np.clip(np.dot(u, v) / (lu * lv), -1, 1))) < max_angle:
            return True
    return False


def _segment_field(xs, ys, pieces):
    """Distance from each grid cell to the nearest thick polyline (<= 0 on it)."""
    field = np.full((len(ys), len(xs)), np.inf)
    reach = GRASS_WIDTH + 2.0
    for line, half in pieces:
        for (x0, y0), (x1, y1) in zip(line[:-1], line[1:]):
            lo_x, hi_x = min(x0, x1) - half - reach, max(x0, x1) + half + reach
            lo_y, hi_y = min(y0, y1) - half - reach, max(y0, y1) + half + reach
            i0, i1 = np.searchsorted(xs, [lo_x, hi_x])
            j0, j1 = np.searchsorted(ys, [lo_y, hi_y])
            if i0 >= i1 or j0 >= j1:
                continue
            gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1])
            dx, dy = x1 - x0, y1 - y0
            length2 = dx * dx + dy * dy
            t = 0.0 if length2 == 0 else np.clip(((gx - x0) * dx + (gy - y0) * dy) / length2, 0, 1)
            dist = np.hypot(gx - (x0 + t * dx), gy - (y0 + t * dy)) - half
            np.minimum(field[j0:j1, i0:i1], dist, out=field[j0:j1, i0:i1])
    return np.minimum(field, 1e3)


def _disk_offsets(radius):
    r = radius / CELL
    n = int(math.ceil(r))
    return [(i, j) for i in range(-n, n + 1) for j in range(-n, n + 1) if i * i + j * j <= r * r + 1e-9]


def _dilate(mask, radius):
    out = np.zeros_like(mask)
    for i, j in _disk_offsets(radius):
        out |= np.roll(np.roll(mask, i, axis=0), j, axis=1)
    return out


def _erode(mask, radius):
    return ~_dilate(~mask, radius)


def _mask_field(mask, passes=4):
    """Field that is <= 0 on a boolean grid mask, blurred so its outline is smooth
    instead of following the grid's staircase."""
    value = mask.astype(float)
    for _ in range(passes):
        value = sum(np.roll(np.roll(value, i, axis=0), j, axis=1)
                    for i in (-1, 0, 1) for j in (-1, 0, 1)) / 9
    return 0.5 - value


def _simplify(points, tolerance):
    """Douglas-Peucker line simplification."""
    points = np.asarray(points, float)
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = points[a], points[b]
        seg = q - p
        rel = points[a + 1:b] - p
        norm = math.hypot(*seg)
        dist = (np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm if norm
                else np.hypot(rel[:, 0], rel[:, 1]))
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return points[keep]


def _fmt(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _circle(cx, cy, r, n=14):
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
            for k in range(n)]


def _round_corners(ring, rounds=2):
    """Chaikin corner cutting: replaces each corner of a closed ring by a short curve."""
    ring = np.asarray(ring, float)
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    for _ in range(rounds):
        nxt = np.roll(ring, -1, axis=0)
        ring = np.stack([0.75 * ring + 0.25 * nxt, 0.25 * ring + 0.75 * nxt], axis=1).reshape(-1, 2)
    return ring


def _outlines(xs, ys, field, level, holes=False, smooth=False):
    """Outer rings of the region field <= level (holes are filled), or its holes.

    smooth rounds the corners, for outlines traced from a grid mask."""
    from contourpy import FillType, contour_generator

    gen = contour_generator(xs, ys, field, fill_type=FillType.OuterOffset)
    points, offsets = gen.filled(-np.inf, level)
    outlines = []
    for pts, off in zip(points, offsets):
        rings = [pts[a:b] for a, b in zip(off[:-1], off[1:])]
        for ring in (rings[1:] if holes else rings[:1]):
            if len(ring) >= 3:
                ring = _simplify(ring, 0.3 if smooth else 0.15)
                outlines.append(_round_corners(ring) if smooth and len(ring) >= 3 else ring)
    return outlines


def _tree_spots(net, xs, ys, field):
    """Points on the TREE_OFFSET line around the road, TREE_SPACING apart."""
    from contourpy import contour_generator

    dead_ends = np.array([node.getCoord()[:2] for node in net.getNodes()
                          if node.getType() == "dead_end"], float).reshape(-1, 2)
    spots = []
    for line in contour_generator(xs, ys, field).lines(TREE_OFFSET):
        seg = np.hypot(*np.diff(line, axis=0).T)
        along = np.concatenate([[0], np.cumsum(seg)])
        if along[-1] < TREE_SPACING:
            continue
        for s in np.arange(TREE_SPACING / 2, along[-1], TREE_SPACING):
            x = np.interp(s, along, line[:, 0])
            y = np.interp(s, along, line[:, 1])
            if len(dead_ends) and np.min(np.hypot(*(dead_ends - (x, y)).T)) < DEAD_END_CLEARANCE:
                continue
            if all(math.hypot(x - u, y - v) >= 0.8 * TREE_SPACING for u, v in spots):
                spots.append((x, y))
    return spots


def write_decor(net_path, out_path=None):
    """Write the decoration file next to the network (or to out_path) and return its path."""
    import sumolib

    net_path = Path(net_path)
    out_path = Path(out_path) if out_path else net_path.with_name("highway.decor.xml")
    net = sumolib.net.readNet(str(net_path), withInternal=True)
    xs, ys, open_lanes, closed_lanes, junction, twisted = _distance_fields(net, GRASS_WIDTH + 4.0)
    # Grass keeps the outline of the whole road layout (closed lanes included), so
    # the land the closed ring used to enclose stays a single green park.
    grass = np.minimum(np.minimum(open_lanes, closed_lanes), junction)
    # Trees line the roads that are actually used.
    roads = np.minimum(open_lanes, junction)
    # Hide closed lanes and the junction corners that exist only because of them,
    # stopping just outside the open lanes so the used road keeps its edge.
    closed_area = np.minimum(closed_lanes, np.maximum(junction, closed_lanes - CLOSED_STUB_REACH))
    cover = np.maximum(closed_area - CLOSED_LANE_GROW, CLOSED_LANE_GAP - open_lanes)
    # Tidy the visible road surface. A junction whose outline has a sharp spike keeps
    # only the area its lanes sweep (gaps between lanes narrower than
    # 2 * JUNCTION_FILL_RADIUS count as swept); the rest of its outline sticks out
    # past the lanes and is covered with grass. sumo-gui fills such outlines
    # differently from a point-in-polygon test, so the trimmed zone reaches a
    # little past the outline and over its small holes. Other junctions keep
    # their shape.
    on_lane = open_lanes <= 0
    in_junction = (junction <= 0) & (cover > 0)
    swept = _erode(_dilate(on_lane, JUNCTION_FILL_RADIUS), JUNCTION_FILL_RADIUS)
    trim_zone = _erode(_dilate(twisted, GAP_FILL_RADIUS), GAP_FILL_RADIUS) & (cover > 0)
    excess = trim_zone & ~swept & (open_lanes > EXCESS_LANE_GAP) & ~(in_junction & ~twisted)
    surface = ((on_lane & (cover > 0)) | in_junction) & ~excess
    closed_surface = _erode(_dilate(surface, GAP_FILL_RADIUS), GAP_FILL_RADIUS)
    asphalt = _erode(closed_surface, ASPHALT_INSET) | (closed_surface & ~surface)
    xmin, ymin, xmax, ymax = net.getBoundary()
    cx, cy, e = (xmin + xmax) / 2, (ymin + ymax) / 2, BACKGROUND_EXTENT

    lines = ["<additional>",
             "    <!-- Generated by scripts/make_road_decor.py; GUI decoration only. -->",
             f'    <poly id="decor_background" color="{BACKGROUND_COLOR}" fill="1" '
             f'layer="{BACKGROUND_LAYER}" shape="{_fmt([(cx - e, cy - e), (cx + e, cy - e), (cx + e, cy + e), (cx - e, cy + e)])}"/>']
    for k, outline in enumerate(_outlines(xs, ys, grass, GRASS_WIDTH)):
        lines.append(f'    <poly id="decor_grass_{k}" color="{GRASS_COLOR}" fill="1" '
                     f'layer="{GRASS_LAYER}" shape="{_fmt(outline)}"/>')
    asphalt_field = _mask_field(asphalt)
    for k, outline in enumerate(_outlines(xs, ys, asphalt_field, 0.0, smooth=True)):
        lines.append(f'    <poly id="decor_asphalt_{k}" color="{ASPHALT_COLOR}" fill="1" '
                     f'layer="{ASPHALT_LAYER}" shape="{_fmt(outline)}"/>')
    # SUMO polygons have no holes: put the land enclosed by the road back on top.
    for k, outline in enumerate(_outlines(xs, ys, asphalt_field, 0.0, holes=True, smooth=True)):
        lines.append(f'    <poly id="decor_island_{k}" color="{GRASS_COLOR}" fill="1" '
                     f'layer="{ASPHALT_LAYER + 0.2:.1f}" shape="{_fmt(outline)}"/>')
    for k, outline in enumerate(_outlines(xs, ys, cover, 0.0)):
        lines.append(f'    <poly id="decor_closed_{k}" color="{GRASS_COLOR}" fill="1" '
                     f'layer="{COVER_LAYER}" shape="{_fmt(outline)}"/>')
    for k, outline in enumerate(_outlines(xs, ys, _mask_field(excess), 0.0, smooth=True)):
        lines.append(f'    <poly id="decor_excess_{k}" color="{GRASS_COLOR}" fill="1" '
                     f'layer="{COVER_LAYER}" shape="{_fmt(outline)}"/>')
    for k, (x, y) in enumerate(_tree_spots(net, xs, ys, roads)):
        lines.append(f'    <poly id="decor_tree_{k}" color="{TREE_COLOR}" fill="1" '
                     f'layer="{TREE_LAYER}" shape="{_fmt(_circle(x, y, TREE_RADIUS))}"/>')
        lines.append(f'    <poly id="decor_tree_{k}_top" color="{TREE_TOP_COLOR}" fill="1" '
                     f'layer="{TREE_LAYER + 0.1:.1f}" '
                     f'shape="{_fmt(_circle(x - 0.5, y + 0.5, TREE_RADIUS * 0.6, 10))}"/>')
    lines.append("</additional>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def gui_decor_options(sumocfg):
    """sumo-gui options that add the decoration to the additional files of sumocfg.

    The decoration is generated from the configuration's own network, so it fits
    whichever road (current road, evaluation suite copy, preview) is being shown.
    An existing file is reused while it is newer than the network and this script,
    because the GUI environment restarts SUMO for every episode.
    """
    sumocfg = Path(sumocfg).resolve()
    root = ET.parse(sumocfg).getroot()
    net_node = root.find("input/net-file")
    add_node = root.find("input/additional-files")
    net = sumocfg.parent / net_node.get("value")
    additional = [] if add_node is None else [
        str(sumocfg.parent / name.strip()) for name in add_node.get("value").split(",") if name.strip()]
    decor = sumocfg.with_name("highway.decor.xml")
    sources = (net, Path(__file__))
    if not decor.is_file() or decor.stat().st_mtime < max(path.stat().st_mtime for path in sources):
        write_decor(net, decor)
    return ["--additional-files", ",".join(additional + [str(decor)])]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("net", type=Path, help="SUMO network (highway.net.xml)")
    parser.add_argument("-o", "--output", type=Path, help="Output file (default: highway.decor.xml beside the network)")
    args = parser.parse_args()
    print(f"Wrote {write_decor(args.net, args.output)}")


if __name__ == "__main__":
    main()

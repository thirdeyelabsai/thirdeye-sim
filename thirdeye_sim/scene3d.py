"""The static 3D scene for the GPU renderer, built once from the road map and cached.

Everything that does not move becomes triangles with a colour and a normal, in metres (z up):
- the ground, parks and water;
- roads (lanes and junctions; underpasses darker, bridges with concrete parapets), lane dashes, solid edge
  lines (yellow down the middle of a two-way road), zebra crossings and stop lines;
- footpaths and corners raised 15 cm with kerbs;
- buildings extruded to a height: OSM's floor count where given, otherwise a guess from the building type and
  footprint (towers for apartments and offices, two to four floors for houses), 3.2 m a floor;
plus instances (boxes and spheres placed by position, heading, size and colour) for trees in parks and on open
ground and the black water tanks on Indian rooftops, line segments for the map (roads drawn at least a few
pixels wide whatever the zoom), and road names to label it.

    from thirdeye_sim.scene3d import load_scene
    scene = load_scene(roadmap)      # dict of numpy arrays, rebuilt when the map or this file changes
"""
import json, math, os, pickle, random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from .paths import CACHE, data_file
FLOOR = 3.2
TILE = 150.0   # metres: the scene is stored in squares this size so only those in view are drawn

COL = {
    "land": (0.60, 0.58, 0.47), "green": (0.37, 0.50, 0.28), "water": (0.26, 0.44, 0.55),
    "road": (0.235, 0.24, 0.255), "road_minor": (0.29, 0.29, 0.30), "tunnel": (0.17, 0.17, 0.18),
    "junction": (0.245, 0.25, 0.265), "paving": (0.66, 0.64, 0.60), "kerb": (0.80, 0.78, 0.74),
    "white": (0.93, 0.93, 0.90), "yellow": (0.95, 0.76, 0.18), "parapet": (0.72, 0.71, 0.68),
    "wall": (0.70, 0.69, 0.66), "stripe_red": (0.80, 0.14, 0.12), "stripe_white": (0.94, 0.94, 0.92),
}
BUILDING = [(0.86, 0.81, 0.71), (0.79, 0.75, 0.68), (0.91, 0.89, 0.85), (0.72, 0.68, 0.62), (0.80, 0.63, 0.51),
            (0.89, 0.83, 0.64), (0.84, 0.78, 0.74)]
GLASS = [(0.55, 0.63, 0.70), (0.47, 0.56, 0.62), (0.62, 0.68, 0.72)]
TREE = [(0.26, 0.42, 0.20), (0.31, 0.46, 0.22), (0.22, 0.37, 0.19), (0.35, 0.47, 0.25), (0.28, 0.40, 0.17)]
MAP = {"land": (0.91, 0.90, 0.87), "green": (0.76, 0.87, 0.69), "water": (0.64, 0.79, 0.92), "building": (0.84, 0.82, 0.78),
       "motorway": (0.95, 0.60, 0.32), "trunk": (0.95, 0.60, 0.32), "primary": (0.98, 0.78, 0.42), "secondary": (0.99, 0.88, 0.55),
       "tertiary": (1.0, 0.97, 0.80), "other": (1.0, 1.0, 1.0)}
MAP_PX = {"motorway": 3.5, "trunk": 3.5, "primary": 3.0, "secondary": 2.6, "tertiary": 2.0, "other": 1.2}


def _ring(poly):
    pts = [(float(x), float(y)) for x, y in poly]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def _area(pts):
    return 0.5 * sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))


class Mesh:
    """Triangles with a colour, a normal and whether they catch shadows: rows of 10 floats."""
    def __init__(self):
        self.rows = []

    def tri(self, a, b, c, col, n=(0.0, 0.0, 1.0), recv=1.0):
        for p in (a, b, c):
            self.rows.append((*p, *col, *n, recv))

    def flat(self, poly, z, col, recv=1.0):
        import mapbox_earcut as earcut
        pts = _ring(poly)
        if len(pts) < 3 or abs(_area(pts)) < 0.05:
            return
        idx = earcut.triangulate_float64(np.array(pts, dtype=np.float64), np.array([len(pts)], dtype=np.uint32))
        for k in range(0, len(idx), 3):
            a, b, c = (pts[i] for i in idx[k:k + 3])
            self.tri((*a, z), (*b, z), (*c, z), col, recv=recv)

    def walls(self, poly, z0, z1, col, recv=0.0):
        pts = _ring(poly)
        if len(pts) < 3:
            return
        ccw = _area(pts) > 0
        for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
            L = math.hypot(bx - ax, by - ay)
            if L < 1e-3:
                continue
            n = ((by - ay) / L, -(bx - ax) / L, 0.0) if ccw else (-(by - ay) / L, (bx - ax) / L, 0.0)
            self.tri((ax, ay, z0), (bx, by, z0), (bx, by, z1), col, n, recv)
            self.tri((ax, ay, z0), (bx, by, z1), (ax, ay, z1), col, n, recv)

    def strip(self, a, b, hw, z, col):
        """A flat bar from a to b, hw metres either side of the line."""
        (ax, ay), (bx, by) = a, b
        L = math.hypot(bx - ax, by - ay)
        if L < 1e-3:
            return
        nx, ny = -(by - ay) / L * hw, (bx - ax) / L * hw
        p = [(ax + nx, ay + ny, z), (bx + nx, by + ny, z), (bx - nx, by - ny, z), (ax - nx, ay - ny, z)]
        self.tri(p[0], p[1], p[2], col); self.tri(p[0], p[2], p[3], col)

    def line(self, pts, hw, z, col, dash=None, gap=None):
        """A painted line along a polyline, solid or dashed."""
        pos, on = 0.0, True
        for a, b in zip(pts, pts[1:]):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            if dash is None:
                self.strip(a, b, hw, z, col)
                continue
            u = 0.0
            while u < L - 1e-6:
                step = min((dash if on else gap) - pos, L - u)
                if on:
                    p = (a[0] + (b[0] - a[0]) * u / L, a[1] + (b[1] - a[1]) * u / L)
                    q = (a[0] + (b[0] - a[0]) * (u + step) / L, a[1] + (b[1] - a[1]) * (u + step) / L)
                    self.strip(p, q, hw, z, col)
                u += step; pos += step
                if pos >= (dash if on else gap) - 1e-6:
                    pos, on = 0.0, not on

    def array(self):
        return np.array(self.rows, dtype=np.float32).reshape(-1, 10)


def _offset(line, d):
    """The polyline shifted d metres to its left."""
    out = []
    for k, (x, y) in enumerate(line):
        a, b = line[max(0, k - 1)], line[min(len(line) - 1, k + 1)]
        L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
        out.append((x - (b[1] - a[1]) / L * d, y + (b[0] - a[0]) / L * d))
    return out


def _building_tags():
    """Tags of each building, in the order the road map stored them."""
    with open(data_file("osm_context.json")) as f:
        ctx = json.load(f)
    return [w.get("tags", {}) for w in ctx["elements"] if w.get("geometry") and len(w["geometry"]) >= 4 and "building" in w.get("tags", {})]


def _height(tags, area, rng):
    try:
        if "height" in tags:
            return max(3.0, float(str(tags["height"]).split()[0]))
        if "building:levels" in tags:
            return max(1.0, float(tags["building:levels"])) * FLOOR + 0.6
    except ValueError:
        pass
    t = tags.get("building", "yes")
    if t == "apartments":
        floors = rng.randint(8, 18) if area > 500 else rng.randint(4, 8)
    elif t in ("office", "hotel"):
        floors = rng.randint(5, 13)
    elif t in ("commercial", "retail"):
        floors = rng.randint(2, 6)
    elif t in ("house", "detached", "residential", "school"):
        floors = rng.randint(2, 4)
    elif t in ("roof", "construction"):
        floors = 1
    elif area < 100:
        floors = rng.randint(1, 3)
    elif area < 400:
        floors = rng.randint(2, 4)
    elif area < 1500:
        floors = rng.randint(3, 6)
    else:
        floors = rng.randint(4, 10)
    return floors * FLOOR + 0.6


def build(m):
    rng = random.Random(7)
    ground, roofs = Mesh(), Mesh()
    inst = []   # (x, y, z0, heading, l, w, h, r, g, b, emit) for boxes
    spheres = []
    x0, y0, x1, y1 = m.bounds
    pad = 3000
    ground.flat([(x0 - pad, y0 - pad), (x1 + pad, y0 - pad), (x1 + pad, y1 + pad), (x0 - pad, y1 + pad)], 0.0, COL["land"])
    for p in m.greens:
        ground.flat(p, 0.03, COL["green"])
    for p in m.water:
        ground.flat(p, 0.035, COL["water"])
    # roads, each kind on its own thin layer so overlapping surfaces never fight over which is on top (that
    # flickers): underpasses below the street, bridges above it, and on the street junctions, then junction
    # lanes, minor roads and main roads
    def road_z(level, layer):
        return {-1: 0.035, 0: 0.08, 1: 0.115}[max(-1, min(1, level))] + 0.0025 * layer

    def paint_z(level):
        return {-1: 0.055, 0: 0.12, 1: 0.14}[max(-1, min(1, level))]

    for k, p in enumerate(m.junctions):
        lv = m.levels["junctions"][k]
        ground.flat(p, road_z(lv, 0), COL["tunnel"] if lv < 0 else COL["junction"])
    reverse = {l["edge"] for l in m.lanes}
    for l in m.lanes:
        if len(l["poly"]) < 3:
            continue
        minor = l["cls"] in ("residential", "living_street", "unclassified", "service")
        col = COL["tunnel"] if l["level"] < 0 else COL["junction"] if l["internal"] else COL["road_minor"] if minor else COL["road"]
        ground.flat(l["poly"], road_z(l["level"], 1 if l["internal"] else 2 if minor else 3), col)
    by_slot = {(l["edge"], l["index"]) for l in m.lanes if not l["internal"]}
    for l in m.lanes:
        if l["internal"] or len(l["line"]) < 2:
            continue
        line, hw = l["line"], l["w"] / 2
        if (l["edge"], l["index"] - 1) not in by_slot:          # kerb side: solid white edge line
            ground.line(_offset(line, hw - 0.25), 0.07, paint_z(l["level"]), COL["white"])
        if (l["edge"], l["index"] + 1) not in by_slot:          # middle of the road
            two_way = (l["edge"][1:] if l["edge"].startswith("-") else "-" + l["edge"]) in reverse
            ground.line(_offset(line, -hw + (0.12 if two_way else 0.25)), 0.07, paint_z(l["level"]), COL["yellow"] if two_way else COL["white"])
        if l["level"] > 0:   # a bridge: concrete parapets either side
            for side in (hw + 0.15, -hw - 0.15):
                edge = _offset(line, side)
                for a, b in zip(edge, edge[1:]):
                    L = math.hypot(b[0] - a[0], b[1] - a[1])
                    if L < 1e-3:
                        continue
                    n = (-(b[1] - a[1]) / L, (b[0] - a[0]) / L, 0.0)
                    if side < 0:
                        n = (-n[0], -n[1], 0.0)
                    roofs.tri((*a, 0.0), (*b, 0.0), (*b, 0.9), COL["parapet"], n, 0.0)
                    roofs.tri((*a, 0.0), (*b, 0.9), (*a, 0.9), COL["parapet"], n, 0.0)
    for k, line in enumerate(m.markings):
        ground.line(line, 0.07, paint_z(m.levels["markings"][k]), COL["white"], dash=3.0, gap=4.5)
    for k, c in enumerate(m.crossings):
        cz = paint_z(m.levels["crossings"][k])
        (ax, ay), (bx, by) = c["line"][0], c["line"][-1]
        L = math.hypot(bx - ax, by - ay) or 1
        ux, uy = (bx - ax) / L, (by - ay) / L
        u = 0.3
        while u + 0.5 < L:
            ground.strip((ax + ux * (u + 0.25) - uy * c["w"] * 0.45, ay + uy * (u + 0.25) + ux * c["w"] * 0.45),
                         (ax + ux * (u + 0.25) + uy * c["w"] * 0.45, ay + uy * (u + 0.25) - ux * c["w"] * 0.45), 0.25, cz, COL["white"])
            u += 1.0
    for sg in m.signals:
        for st in sg["stops"]:
            ground.strip(st["a"], st["b"], 0.2, paint_z(m.lanes[st["lane"]]["level"]), COL["white"])
    # footpaths and corners, raised with kerbs
    for k, p in enumerate(m.walks):
        if m.levels["walks"][k] < 0:
            continue
        ground.flat(p, 0.16, COL["paving"])
        roofs.walls(p, 0.0, 0.16, COL["kerb"], recv=1.0)
    for wa in m.walkareas:
        if wa.get("poly") and len(wa["poly"]) >= 3:
            ground.flat(wa["poly"], 0.155, COL["paving"])
    # buildings
    tags = getattr(m, "building_tags", None) or _building_tags()
    for k, p in enumerate(m.buildings):
        pts = _ring(p)
        if len(pts) < 3:
            continue
        area = abs(_area(pts))
        if area < 8:
            continue
        t = tags[k] if k < len(tags) else {}
        r = random.Random(k * 7919)
        h = _height(t, area, r)
        office = t.get("building") in ("office", "commercial", "hotel") or (h > 20 and r.random() < 0.4)
        col = r.choice(GLASS) if office else r.choice(BUILDING)
        shade = 0.92 + 0.12 * r.random()
        col = tuple(min(1.0, c * shade) for c in col)
        roof = tuple(c * 0.9 for c in col) if not office else tuple(min(1.0, c * 1.05) for c in r.choice(BUILDING))
        roofs.walls(pts, 0.0, h, col)
        roofs.flat(pts, h, roof, recv=0.0)
        if 50 < area < 2000 and h < 30 and r.random() < 0.8:   # black plastic water tanks on the roof
            cx = sum(x for x, _ in pts) / len(pts); cy = sum(y for _, y in pts) / len(pts)
            for _ in range(1 if area < 400 else 2):
                inst.append((cx + r.uniform(-2, 2), cy + r.uniform(-2, 2), h, 0.0, 1.3, 1.3, 1.4, 0.08, 0.08, 0.09, 0.0))
    # the wall round the play area (the edge the map was cut at): concrete, striped red and white where a road
    # meets it. Its inner face lies on the edge, where our car stops.
    area = getattr(m, "play_area", None) or []
    for k in range(len(area)):
        (ax, ay), (bx, by) = area[k], area[(k + 1) % len(area)]
        L = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / L, (by - ay) / L
        ox, oy = uy * 0.3, -ux * 0.3          # outward, half the wall's thickness
        h = math.atan2(uy, ux)
        segs, u = [], 0.0   # [from, to, kind] along the edge: plain wall (up to 30 m a piece) or a stripe over a road
        while u < L - 1e-6:
            step = min(1.5, L - u)
            mx, my = ax + ux * (u + step / 2), ay + uy * (u + step / 2)
            if m.on_road(mx - ox * 3, my - oy * 3):   # (just inside the edge)
                segs.append([u, u + step, "road"])
            elif segs and segs[-1][2] == "wall" and u + step - segs[-1][0] <= 30:
                segs[-1][1] = u + step
            else:
                segs.append([u, u + step, "wall"])
            u += step
        for n, (a0, a1, kind) in enumerate(segs):
            cx, cy = ax + ux * (a0 + a1) / 2 + ox, ay + uy * (a0 + a1) / 2 + oy
            if kind == "wall":
                inst.append((cx, cy, 0.0, h, a1 - a0, 0.6, 1.3, *COL["wall"], 0.0))
            else:
                inst.append((cx, cy, 0.0, h, a1 - a0, 0.6, 1.1, *COL["stripe_red" if n % 2 == 0 else "stripe_white"], 0.0))
    # speed breakers (yellow and black, across the road), rumble strips (three white bars), potholes (dark ragged
    # patches) and bus shelters on the kerb side (roadmap.py: from OpenStreetMap where it maps them, else assumed)
    for b in getattr(m, "bumps", ()):
        (ax, ay), (bx, by) = b["a"], b["b"]
        L = math.hypot(bx - ax, by - ay)
        if L < 1:
            continue
        ux, uy = (bx - ax) / L, (by - ay) / L
        h = math.atan2(uy, ux)
        if b["kind"] == "rumble":
            for k in (-1, 0, 1):
                inst.append(((ax + bx) / 2 - uy * 0.4 * k, (ay + by) / 2 + ux * 0.4 * k, 0.08, h, L, 0.15, 0.03, *COL["white"], 0.0))
            continue
        n, u = 0, 0.0
        while u < L - 1e-6:
            step = min(0.6, L - u)
            inst.append((ax + ux * (u + step / 2), ay + uy * (u + step / 2), 0.08, h, step, 0.55, 0.07,
                         *((0.95, 0.78, 0.10) if n % 2 == 0 else (0.10, 0.10, 0.10)), 0.0))
            u += step; n += 1
    for p in getattr(m, "potholes", ()):
        r = random.Random(p["seed"])
        ring = [(p["x"] + math.cos(a) * p["r"] * r.uniform(0.7, 1.15), p["y"] + math.sin(a) * p["r"] * r.uniform(0.7, 1.15))
                for a in [k * 2 * math.pi / 9 for k in range(9)]]
        ground.flat(ring, 0.0925, (0.13, 0.13, 0.135))
    for st in getattr(m, "bus_stops", ()):
        x, y, h, w = st["x"], st["y"], st["h"], st["w"]
        nx, ny = -math.sin(h), math.cos(h)   # to the left: the kerb side
        cx, cy = x + nx * (w / 2 + 1.7), y + ny * (w / 2 + 1.7)
        c, sn = math.cos(h), math.sin(h)
        roofc, post, glass = (0.16, 0.36, 0.66), (0.55, 0.57, 0.6), (0.62, 0.70, 0.74)
        inst.append((cx, cy, 2.45, h, 3.4, 1.6, 0.12, *roofc, 0.0))                                 # roof
        inst.append((cx + nx * 0.7, cy + ny * 0.7, 0.3, h, 3.2, 0.08, 1.7, *glass, 0.0))           # back panel
        for k in (-1, 1):
            inst.append((cx + c * 1.6 * k, cy + sn * 1.6 * k, 0.0, h, 0.1, 0.1, 2.45, *post, 0.0))   # posts
        inst.append((cx + nx * 0.4, cy + ny * 0.4, 0.0, h, 2.4, 0.45, 0.45, 0.35, 0.36, 0.38, 0.0))  # bench
        sx, sy = cx + c * 2.6 - nx * 0.5, cy + sn * 2.6 - ny * 0.5
        inst.append((sx, sy, 0.0, h, 0.08, 0.08, 2.6, *post, 0.0))                                  # sign
        inst.append((sx, sy, 2.2, h, 0.06, 0.5, 0.4, 0.80, 0.15, 0.12, 0.0))
    # trees: in parks, and scattered over open ground away from roads, paths and buildings
    from shapely.geometry import Point, Polygon
    from shapely.strtree import STRtree
    blockers = []
    for l in m.lanes:
        if len(l["poly"]) >= 3:
            q = Polygon(l["poly"])
            if q.is_valid:
                blockers.append(q.buffer(2.0))
    for p in m.buildings + m.walks + m.junctions:
        if len(p) >= 3:
            q = Polygon(p)
            if q.is_valid:
                blockers.append(q.buffer(1.5))
    tree = STRtree(blockers)
    greens = [Polygon(p) for p in m.greens if len(p) >= 3]
    greens = [g for g in greens if g.is_valid]
    gtree = STRtree(greens)

    def free(x, y):
        pt = Point(x, y)
        return not any(blockers[i].contains(pt) for i in tree.query(pt))

    def plant(x, y, big=1.0):
        r = rng.uniform(1.8, 3.1) * big
        h = rng.uniform(3.0, 5.0) * big
        col = tuple(min(1.0, c * rng.uniform(0.9, 1.12)) for c in rng.choice(TREE))
        inst.append((x, y, 0.0, 0.0, 0.5, 0.5, h, 0.35, 0.27, 0.18, 0.0))                       # trunk
        spheres.append((x, y, h - r * 0.4, rng.uniform(0, 6.28), 2 * r, 2 * r * rng.uniform(0.85, 1.0), r * 1.5, *col, 0.0))

    for g in greens:
        gx0, gy0, gx1, gy1 = g.bounds
        step = 9.0
        n = 0
        y = gy0 + rng.uniform(0, step)
        while y < gy1 and n < 500:
            x = gx0 + rng.uniform(0, step)
            while x < gx1 and n < 500:
                px, py = x + rng.uniform(-3, 3), y + rng.uniform(-3, 3)
                if g.contains(Point(px, py)) and rng.random() < 0.6 and free(px, py):
                    plant(px, py); n += 1
                x += step
            y += step
    # rows along main roads: beyond the footpath on the kerb side, and on the median of a divided road
    for l in m.lanes:
        if l["internal"] or l["cls"].replace("_link", "") not in ("motorway", "trunk", "primary", "secondary", "tertiary"):
            continue
        kerb = (l["edge"], l["index"] - 1) not in by_slot
        median = (l["edge"], l["index"] + 1) not in by_slot and (l["edge"][1:] if l["edge"].startswith("-") else "-" + l["edge"]) not in reverse
        for side, ok, off in ((1, kerb, (4.0, 6.5)), (-1, median, (1.6, 2.6))):
            if not ok:
                continue
            u = rng.uniform(0, 12)
            line = l["line"]
            from .actors import _point
            while u < l["len"]:
                x, y, h = _point(line, u)
                d = l["w"] / 2 + rng.uniform(*off)
                px, py = x - math.sin(h) * d * side, y + math.cos(h) * d * side
                if free(px, py):
                    plant(px, py, rng.uniform(0.75, 1.05))
                u += rng.uniform(9, 15)
    step = 22.0
    y = y0
    while y < y1:
        x = x0
        while x < x1:
            px, py = x + rng.uniform(-8, 8), y + rng.uniform(-8, 8)
            if rng.random() < 0.35 and free(px, py) and not len(gtree.query(Point(px, py), predicate="within")):
                plant(px, py, rng.uniform(0.8, 1.25))
            x += step
        y += step
    # the map: roads as segments drawn at least a few pixels wide, then names
    mp = []      # x, y, nx, ny, r, g, b, real half width, min half width px
    for l in m.lanes:
        if len(l["line"]) < 2:
            continue
        cls = l["cls"].replace("_link", "")
        if l["internal"] and l["from_lane"] is not None:
            cls = m.lanes[l["from_lane"]]["cls"].replace("_link", "")
        key = cls if cls in MAP_PX else "other"
        for a, b in zip(l["line"], l["line"][1:]):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            if L < 1e-3:
                continue
            nx, ny = -(b[1] - a[1]) / L, (b[0] - a[0]) / L
            for (px, py), s in ((a, 1), (b, 1), (b, -1), (a, 1), (b, -1), (a, -1)):
                mp.append((px, py, nx * s, ny * s, *MAP[key], l["w"] / 2, MAP_PX[key]))
    mapfill = Mesh()
    for p in m.greens:
        mapfill.flat(p, 0.0, MAP["green"])
    for p in m.water:
        mapfill.flat(p, 0.0, MAP["water"])
    for p in m.buildings:
        mapfill.flat(p, 0.0, MAP["building"])
    for p in m.junctions:
        mapfill.flat(p, 0.0, MAP["other"])
    rank = {"motorway": 0, "trunk": 0, "primary": 1, "secondary": 2, "tertiary": 3}
    labels, seen = [], {}
    for l in sorted((l for l in m.lanes if not l["internal"] and l["name"]), key=lambda l: -l["len"]):
        cls = l["cls"].replace("_link", "")
        if cls not in rank or l["len"] < 60:
            continue
        mid = l["line"][len(l["line"]) // 2] if len(l["line"]) > 2 else ((l["line"][0][0] + l["line"][1][0]) / 2, (l["line"][0][1] + l["line"][1][1]) / 2)
        if any(math.hypot(mid[0] - x, mid[1] - y) < 700 for x, y in seen.get(l["name"], [])):
            continue
        seen.setdefault(l["name"], []).append(mid)
        (ax, ay), (bx, by) = l["line"][0], l["line"][-1]
        ang = math.degrees(math.atan2(by - ay, bx - ax))
        if ang > 90: ang -= 180
        if ang < -90: ang += 180
        labels.append((l["name"], mid[0], mid[1], ang, rank[cls]))
    base = ground.array()[:6]   # the ground plane under everything, always drawn
    out = {"base": base, "labels": labels}
    for key, arr, per in (("ground", ground.array()[6:], 3), ("solid", roofs.array(), 3), ("map_fill", mapfill.array(), 3),
                          ("map_lines", np.array(mp, dtype=np.float32).reshape(-1, 9), 6),
                          ("boxes", np.array(inst, dtype=np.float32).reshape(-1, 11), 1), ("spheres", np.array(spheres, dtype=np.float32).reshape(-1, 11), 1)):
        out[key], out[key + "_tiles"] = _tiled(arr, per)
    return out


def _tiled(arr, per):
    """Sort primitives (triangles, segment quads or instances: per rows each) by the TILE-metre square their
    centre falls in, so a frame can draw only the squares in view. Returns the sorted rows and
    {(ix, iy): (first row, rows)}; primitives over 100 m across come first, under the key "big", to be
    drawn always (a park's long triangle can reach into view from a square that is not)."""
    n = len(arr) // per
    if n == 0:
        return arr, {}
    prim = arr[:n * per].reshape(n, per, arr.shape[1])
    xs, ys = prim[:, :, 0], prim[:, :, 1]
    big = (xs.max(1) - xs.min(1) > 100) | (ys.max(1) - ys.min(1) > 100)
    ix = np.floor(xs.mean(1) / TILE).astype(np.int64)
    iy = np.floor(ys.mean(1) / TILE).astype(np.int64)
    ix[big], iy[big] = -10 ** 6, -10 ** 6
    order = np.lexsort((ix, iy))
    prim, ix, iy = prim[order], ix[order], iy[order]
    cut = np.flatnonzero((np.diff(ix) != 0) | (np.diff(iy) != 0)) + 1
    index = {}
    for s, e in zip(np.concatenate([[0], cut]), np.concatenate([cut, [n]])):
        key = "big" if ix[s] == -10 ** 6 else (int(ix[s]), int(iy[s]))
        index[key] = (int(s * per), int((e - s) * per))
    return np.ascontiguousarray(prim.reshape(-1, arr.shape[1])), index


def load_scene(m, rebuild=False):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "scene3d.pkl")
    src = [p for p in (os.path.join(CACHE, "gurgaon_map.pkl"), data_file("gurgaon_map.pkl")) if os.path.exists(p)] + [os.path.abspath(__file__)]
    if rebuild or not os.path.exists(path) or os.path.getmtime(path) < max(os.path.getmtime(s) for s in src if os.path.exists(s)):
        with open(path + f".{os.getpid()}", "wb") as f:   # (then renamed: parallel workers never read half a file)
            pickle.dump(build(m), f)
        os.replace(path + f".{os.getpid()}", path)
    with open(path, "rb") as f:
        return pickle.load(f)

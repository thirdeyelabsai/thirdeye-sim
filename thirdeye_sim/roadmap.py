"""The road map the game is played on: Gurgaon's roads, footpaths, crossings, signals and buildings.

Built once from the cleaned SUMO network (gurgaon/gurgaon.net.xml) and OpenStreetMap context
(gurgaon/osm_context.json: buildings, parks, water), then cached as a pickle beside this file. Coordinates
are the network's: metres, x east, y north.

What the map holds:
- lanes: every vehicle lane, with its centre line, width, speed limit, road name and the lanes it leads to;
  drawn as asphalt and used to tell whether a car is on the road and which way traffic runs.
- junctions: junction outlines (asphalt).
- walks: footpath lanes and the footpath areas at junction corners.
- crossings: zebra crossings, as a centre line across the road plus a width.
- markings: dashed lines between lanes of the same direction.
- signals: each controller's phases (duration, light per link) and, per link, the lane it controls.
- sidewalks, walkareas: the footpath network pedestrians walk on: footpath centre lines, and the corner
  areas that join footpaths and crossings (each with the points where they meet).
For road users, each lane also carries its road class and priority, the turn it makes inside a junction,
the signal link that controls it, the junction lanes it conflicts with (foes), and where zebra crossings
cross it.
- buildings, greens, water: OpenStreetMap outlines for context.
"""
import json, math, os, pickle
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
from .paths import CACHE, CHECKOUT, GURGAON, data_file
CELL = 50.0   # metres, the grid used to find what is near a point
START_EDGE = "47420041#4"   # Sunset Boulevard, 3 lanes, a straight 1.4 km on our morning route
# Named places to start a drive at (longitude, latitude): the car starts on the main road nearest to it.
# "sunset" is the start of our morning route (START_EDGE).
STARTS = {"medanta": (77.09453, 28.46353)}   # Medanta on Sunset Boulevard, Sector 43 (OSM)


class Grid:
    """A coarse spatial index: which items have any part in each 50 m cell."""

    def __init__(self):
        self.cells = defaultdict(list)

    def add(self, idx, pts):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        for cx in range(int(min(xs) // CELL), int(max(xs) // CELL) + 1):
            for cy in range(int(min(ys) // CELL), int(max(ys) // CELL) + 1):
                self.cells[(cx, cy)].append(idx)

    def near(self, x0, y0, x1, y1):
        out = set()
        for cx in range(int(x0 // CELL), int(x1 // CELL) + 1):
            for cy in range(int(y0 // CELL), int(y1 // CELL) + 1):
                out.update(self.cells.get((cx, cy), ()))
        return out


def _outline(line, width):
    """The lane's surface. Round ends, so consecutive lane pieces and junctions meet without gaps."""
    from shapely.geometry import LineString
    poly = LineString(line).buffer(width / 2, cap_style=1, join_style=1, quad_segs=4)
    return [tuple(map(float, p)) for p in poly.exterior.coords] if not poly.is_empty and poly.geom_type == "Polygon" else []


def _offset(line, d):
    """The polyline shifted d metres to its left."""
    out = []
    for k, (x, y) in enumerate(line):
        a, b = line[max(0, k - 1)], line[min(len(line) - 1, k + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy) or 1
        out.append((x - dy / n * d, y + dx / n * d))
    return out


def _turn_speed(line, lat_accel=2.5):
    """The fastest comfortable speed through the lane's tightest bend (m/s), from the radius of each corner."""
    v = 30.0
    for (ax, ay), (bx, by), (cx, cy) in zip(line, line[1:], line[2:]):
        a, b, c = math.hypot(bx - ax, by - ay), math.hypot(cx - bx, cy - by), math.hypot(cx - ax, cy - ay)
        area = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2
        if area > 1e-6 and a > 0.3 and b > 0.3:
            v = min(v, math.sqrt(lat_accel * a * b * c / (4 * area)))
    return max(3.0, v)


def _foes_and_crossings(net, m, lane_index):
    """Junction lanes that cross or merge with each other (only one may be in use at a time), and where each
    zebra crossing cuts across the vehicle lanes."""
    from shapely.geometry import LineString
    by_junction = defaultdict(list)
    for i, l in enumerate(m["lanes"]):
        if l["internal"]:
            by_junction[l["junction"]].append(i)
    for ids in by_junction.values():
        shapes = {i: LineString(m["lanes"][i]["line"]).buffer(1.1) for i in ids}
        for a in ids:
            for b in ids:
                if a < b and m["lanes"][a]["from_lane"] != m["lanes"][b]["from_lane"] and shapes[a].intersects(shapes[b]):
                    m["lanes"][a]["foes"].append(b); m["lanes"][b]["foes"].append(a)
    lines = {i: LineString(l["line"]) for i, l in enumerate(m["lanes"])}
    grid = Grid()
    for i, l in enumerate(m["lanes"]):
        grid.add(i, l["line"])
    for k, c in enumerate(m["crossings"]):
        zebra = LineString(c["line"]).buffer(c["w"] / 2)
        (ax, ay), (bx, by) = c["line"][0], c["line"][-1]
        c["lanes"] = []
        for i in grid.near(min(ax, bx) - 5, min(ay, by) - 5, max(ax, bx) + 5, max(ay, by) + 5):
            if lines[i].intersects(zebra):
                s0 = lines[i].project(lines[i].intersection(zebra).centroid)
                c["lanes"].append((i, s0))
                m["lanes"][i]["cross"].append((k, s0))


def _dir_at(line, s):
    for (ax, ay), (bx, by) in zip(line, line[1:]):
        L = math.hypot(bx - ax, by - ay)
        if s <= L:
            return math.atan2(by - ay, bx - ax)
        s -= L
    (ax, ay), (bx, by) = line[-2], line[-1]
    return math.atan2(by - ay, bx - ax)


MOTORWAYS = ("motorway", "motorway_link")


def _overlap_zones(m, lane_polys):
    """Stretches where lanes of two different roads lie on top of each other on the map (a slip road drawn over
    a main road, two OSM ways for one carriageway, a crossing with no junction). Each becomes a zone only one
    vehicle may be in at a time: zones[k] = [(lane, s_from, s_to), ...], and lane["zones"] = [(k, s_from, s_to)].
    Junction lanes count too (a turn at a side street drawn across a main road beside it), except where they
    overlap lanes of their own junction right at its mouth, which the junction's right of way already handles.
    A road across a motorway is no zone: it passes over or under it (lane["over"])."""
    from shapely.geometry import LineString
    for l in m["lanes"]:
        l["zones"], l["over"] = [], []
    # lanes joined end to end (directly or through one junction lane) touch at the joint: not an overlap
    linked = {}
    for i, l in enumerate(m["lanes"]):
        for n in l["next"]:
            for t in [n] + m["lanes"][n]["next"]:
                linked.setdefault(i, set()).add(t); linked.setdefault(t, set()).add(i)
    body = {}
    def flat(i):   # the lane's surface with square ends, a little narrower: only true overlaps count
        if i not in body:
            l = m["lanes"][i]
            body[i] = LineString(l["line"]).buffer(l["w"] / 2 - 0.4, cap_style=2)
        return body[i]
    lanes = m["lanes"]
    # the junction each road lane starts from and ends at, and the level a junction lane lies at (the road it
    # comes from: junction lanes are all built at street level)
    starts, ends = {}, {}
    for i, l in enumerate(lanes):
        for n in l["next"]:
            if l["internal"] and not lanes[n]["internal"]:
                starts[n] = l["junction"]
            if not l["internal"] and lanes[n]["internal"]:
                ends[i] = lanes[n]["junction"]
    def at_mouth(i, j, p):
        """Whether junction lane j overlaps road lane i only where i meets j's junction, within its first or last
        3 m (p: the overlap's span on i). A turn that sweeps further over the road is drawn across it."""
        J = lanes[j]["junction"]
        return (starts.get(i) == J and max(p) < 3.0) or (ends.get(i) == J and min(p) > lanes[i]["len"] - 3.0)
    def level(i):
        l = lanes[i]
        return lanes[l["from_lane"]]["level"] if l["internal"] and l["from_lane"] is not None else l["level"]
    def road_cls(i):   # (a junction lane: the class of the road it comes from)
        l = lanes[i]
        return lanes[l["from_lane"]]["cls"] if l["internal"] and l["from_lane"] is not None else l["cls"]
    grid = Grid()
    cands = [i for i, l in enumerate(lanes) if len(l["line"]) >= 2]
    for i in cands:
        grid.add(i, lanes[i]["line"])
    zones = []
    for i in cands:
        a = lanes[i]
        if i not in lane_polys or not lane_polys[i].is_valid:
            continue
        A = LineString(a["line"])
        for j in grid.near(*A.bounds):
            b = lanes[j]
            if j <= i or b["edge"] == a["edge"] or b["edge"].lstrip("-") == a["edge"].lstrip("-") \
                    or level(j) != level(i) or j in linked.get(i, ()):
                continue
            if a["internal"] and b["internal"] and a["junction"] == b["junction"]:
                continue
            inter = flat(i).intersection(flat(j))
            if inter.area < 3.0:
                continue
            B = LineString(b["line"])
            pa = [A.project(type(inter.centroid)(x, y)) for x, y in inter.envelope.exterior.coords] if inter.geom_type != "Point" else []
            pb = [B.project(type(inter.centroid)(x, y)) for x, y in inter.envelope.exterior.coords] if inter.geom_type != "Point" else []
            if not pa:
                continue
            if (b["internal"] and not a["internal"] and at_mouth(i, j, pa)) or (a["internal"] and not b["internal"] and at_mouth(j, i, pb)):
                continue
            mid = inter.centroid
            ha = _dir_at(a["line"], A.project(mid)); hb = _dir_at(b["line"], B.project(mid))
            ang = abs(math.degrees(math.atan2(math.sin(ha - hb), math.cos(ha - hb))))
            mid = inter.centroid
            ha, hb = _dir_at(a["line"], A.project(mid)), _dir_at(b["line"], B.project(mid))
            same = abs(math.atan2(math.sin(ha - hb), math.cos(ha - hb))) < math.pi / 2
            if not same and (road_cls(i) in MOTORWAYS or road_cls(j) in MOTORWAYS):
                # a road across a motorway with no junction: OSM's highway=motorway is grade separated by
                # definition, so one passes over the other (Gurgaon's OSM tags no bridge on NH-48, where the
                # crossing roads go under it); not an at-grade crossing of the expressway
                a["over"].append(j); b["over"].append(i)
                continue
            k = len(zones)
            za, zb = (i, max(0.0, min(pa) - 3), max(pa) + 3), (j, max(0.0, min(pb) - 3), max(pb) + 3)
            zones.append([za, zb])
            m.setdefault("zone_same", []).append(same)
            a["zones"].append((k, za[1], za[2])); b["zones"].append((k, zb[1], zb[2]))
    m["zones"] = zones
    m.setdefault("zone_same", [])
    # the junction lanes into and out of a road that passes over or under a motorway pass with it (they may lie
    # too little on the motorway's lanes to count above, but a vehicle on them still reaches over them)
    prev = {}
    for i, l in enumerate(lanes):
        for n in l["next"]:
            prev.setdefault(n, []).append(i)
    for i, l in enumerate(lanes):
        if l["internal"] or not l["over"] or l["cls"] in MOTORWAYS:
            continue
        for j in prev.get(i, []) + l["next"]:
            if lanes[j]["internal"]:
                for k in list(l["over"]):
                    if k not in lanes[j]["over"]:
                        lanes[j]["over"].append(k); lanes[k]["over"].append(j)


# Indian road features. OpenStreetMap maps a few speed breakers, rumble strips and bus stops; Gurgaon has far
# more breakers than it maps, and no potholes at all, so ASSUMED ones are added (marked "assumed", not measured):
# a breaker on 35% of approaches to a junction on smaller roads, 15 to 22 m before it; a pothole every 300 m of
# lane on average on secondary, tertiary and residential roads.
SMALL_ROADS = ("tertiary", "residential", "unclassified", "living_street")
POTHOLE_ROADS = ("secondary", "tertiary", "residential", "unclassified")
BUMP_SHARE, POTHOLE_EVERY = 0.35, 300.0


def _road_features(net, m):
    """Speed breakers (m["bumps"]: across a whole road), potholes (m["potholes"]) and bus stops (m["bus_stops"]),
    and on each lane where they lie: lane["bumps"] = [s], lane["potholes"] = [(s, lat, r)], lane["stops"] = [s]."""
    import random
    import xml.etree.ElementTree as ET
    rng = random.Random(4242)
    lanes = m["lanes"]
    for l in lanes:
        l["bumps"], l["potholes"], l["stops"] = [], [], []
    by_edge = {}
    for i, l in enumerate(lanes):
        if not l["internal"] and l["level"] == 0:
            by_edge.setdefault(l["edge"], []).append(i)
    grid = Grid()
    for e, ls in by_edge.items():
        for i in ls:
            grid.add(i, lanes[i]["line"])

    def across(e, frac):
        """A bump across all the lanes of road e at fraction frac of its length; also on its other direction."""
        ls = sorted(by_edge[e], key=lambda i: lanes[i]["index"])
        pts = []
        for i in ls:
            l = lanes[i]
            s = frac * l["len"]
            x, y, h = _point_on(l["line"], s)
            pts += [(x - math.sin(h) * l["w"] / 2, y + math.cos(h) * l["w"] / 2), (x + math.sin(h) * l["w"] / 2, y - math.cos(h) * l["w"] / 2)]
            l["bumps"].append(s)
        a, b = max(pts, key=lambda p: (p[0] - pts[0][0]) ** 2 + (p[1] - pts[0][1]) ** 2), pts[0]
        return a, b

    def snap(x, y, within=12.0):
        best = None
        for i in grid.near(x - within, y - within, x + within, y + within):
            l = lanes[i]
            s, lat = _project_on(l["line"], x, y)
            if 0 < s < l["len"] and abs(lat) < within and (best is None or abs(lat) < best[0]):
                best = (abs(lat), i, s)
        return best

    bumps, stops = [], []
    root = ET.parse(os.path.join(GURGAON, "gurgaon.osm")).getroot()
    for n in root.iter("node"):
        t = {x.get("k"): x.get("v") for x in n.iter("tag")}
        kind = "bump" if t.get("traffic_calming") in ("bump", "hump", "table") else "rumble" if t.get("traffic_calming") == "rumble_strip" else \
            "stop" if t.get("highway") == "bus_stop" else None
        if kind is None:
            continue
        x, y = net.convertLonLat2XY(float(n.get("lon")), float(n.get("lat")))
        hit = snap(x, y, 6.0 if kind != "stop" else 25.0)
        if hit is None:
            continue
        _, i, s = hit
        e = lanes[i]["edge"]
        if kind == "stop":   # in the kerb lane of that road (lane 0 is the kerb side)
            k = min(by_edge[e], key=lambda j: lanes[j]["index"])
            s = s * lanes[k]["len"] / max(1e-6, lanes[i]["len"])
            if 12 < s < lanes[k]["len"] - 12:
                lanes[k]["stops"].append(s)
                x, y, h = _point_on(lanes[k]["line"], s)
                stops.append({"lane": k, "s": s, "x": x, "y": y, "h": h, "w": lanes[k]["w"], "source": "osm"})
            continue
        a, b = across(e, s / lanes[i]["len"])
        bumps.append({"a": a, "b": b, "kind": kind, "source": "osm"})
        rev = e[1:] if e.startswith("-") else "-" + e
        if rev in by_edge:
            a, b = across(rev, 1 - s / lanes[i]["len"])
            bumps.append({"a": a, "b": b, "kind": kind, "source": "osm"})
    done = set()
    for e, ls in sorted(by_edge.items()):
        l0 = lanes[ls[0]]
        if l0["cls"].replace("_link", "") not in SMALL_ROADS or l0["len"] < 45 or e in done:
            continue
        if rng.random() < BUMP_SHARE and not l0["bumps"]:
            d = rng.uniform(15, 22)
            a, b = across(e, 1 - d / l0["len"])
            bumps.append({"a": a, "b": b, "kind": "bump", "source": "assumed"})
            rev = e[1:] if e.startswith("-") else "-" + e
            if rev in by_edge:
                a, b = across(rev, d / lanes[by_edge[rev][0]]["len"])
                bumps.append({"a": a, "b": b, "kind": "bump", "source": "assumed"})
                done.add(rev)
    potholes = []
    for e, ls in sorted(by_edge.items()):
        for i in ls:
            l = lanes[i]
            if l["cls"].replace("_link", "") not in POTHOLE_ROADS:
                continue
            n = 0
            want = l["len"] / POTHOLE_EVERY
            n = int(want) + (1 if rng.random() < want % 1 else 0)
            for _ in range(n):
                s = rng.uniform(8, max(9, l["len"] - 8))
                lat = rng.uniform(-1, 1) * max(0.0, l["w"] / 2 - 0.6)
                r = rng.uniform(0.3, 0.65)
                x, y, h = _point_on(l["line"], s)
                l["potholes"].append((s, lat, r))
                potholes.append({"x": x - math.sin(h) * lat, "y": y + math.cos(h) * lat, "r": r, "seed": rng.random(), "source": "assumed"})
    for l in lanes:
        l["bumps"].sort(); l["potholes"].sort(); l["stops"].sort()
    m["bumps"], m["potholes"], m["bus_stops"] = bumps, potholes, stops


def _point_on(line, s):
    for (ax, ay), (bx, by) in zip(line, line[1:]):
        L = math.hypot(bx - ax, by - ay)
        if s <= L or (bx, by) == tuple(line[-1]):
            u = max(0.0, min(1.0, s / L)) if L > 0 else 0.0
            return ax + (bx - ax) * u, ay + (by - ay) * u, math.atan2(by - ay, bx - ax)
        s -= L
    (ax, ay), (bx, by) = line[-2], line[-1]
    return bx, by, math.atan2(by - ay, bx - ax)


def _project_on(line, x, y):
    """(distance along, offset to the left) of point x, y from a polyline."""
    best, acc = None, 0.0
    for (ax, ay), (bx, by) in zip(line, line[1:]):
        L = math.hypot(bx - ax, by - ay) or 1e-9
        u = max(0.0, min(L, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / L))
        px, py = ax + (bx - ax) * u / L, ay + (by - ay) * u / L
        d = math.hypot(x - px, y - py)
        if best is None or d < best[0]:
            side = 1 if (bx - ax) * (y - ay) - (by - ay) * (x - ax) > 0 else -1
            best = (d, acc + u, side * d)
        acc += L
    return best[1], best[2]


def _footpaths(net, m):
    """Footpath centre lines and the corner areas joining them to each other and to the crossings."""
    sidewalks, walkareas, sw_index, wa_index = [], [], {}, {}
    for e in net.getEdges(withInternal=True):
        if e.getFunction() == "walkingarea" and e.getLanes() and len(e.getLanes()[0].getShape()) >= 3:
            wa_index[e.getID()] = len(walkareas)
            walkareas.append({"poly": [tuple(p) for p in e.getLanes()[0].getShape()], "ports": []})
    levels = _osm_levels()
    for e in net.getEdges(withInternal=False):
        for l in e.getLanes():
            if l.allows("pedestrian") and not l.allows("passenger") and len(l.getShape()) >= 2:
                sw_index[e.getID()] = len(sidewalks)
                sidewalks.append({"line": [tuple(p) for p in l.getShape()], "w": l.getWidth(), "wa": [None, None],
                                  "level": _edge_level(levels, e.getID(), e.getName()), "bad": False})
    cr_index = {}
    k = 0
    for e in net.getEdges(withInternal=True):
        if e.getFunction() == "crossing":
            for l in e.getLanes():
                cr_index[e.getID()] = k; m["crossings"][k]["wa"] = [None, None]; m["crossings"][k]["edge"] = e.getID(); k += 1
    for e in net.getEdges(withInternal=True):
        if e.getID() not in wa_index:
            continue
        w = wa_index[e.getID()]
        poly = walkareas[w]["poly"]
        def nearer_end(line):
            d0 = min(math.hypot(line[0][0] - x, line[0][1] - y) for x, y in poly)
            d1 = min(math.hypot(line[-1][0] - x, line[-1][1] - y) for x, y in poly)
            return 0 if d0 <= d1 else 1
        for o in list(e.getIncoming()) + list(e.getOutgoing()):
            if o.getID() in sw_index:
                j = sw_index[o.getID()]; end = nearer_end(sidewalks[j]["line"])
                sidewalks[j]["wa"][end] = w; walkareas[w]["ports"].append(("sw", j, end))
            elif o.getID() in cr_index:
                j = cr_index[o.getID()]; end = nearer_end(m["crossings"][j]["line"])
                m["crossings"][j]["wa"][end] = w; walkareas[w]["ports"].append(("cr", j, end))
    # a footpath or corner area lying on a traffic lane of the same level is a mapping error: pedestrians keep off it
    from shapely.geometry import Polygon as _P
    lane_polys = {i: _P(l["poly"]) for i, l in enumerate(m["lanes"]) if len(l["poly"]) >= 3 and l["level"] == 0}
    _overlap_zones(m, lane_polys)
    lane_polys = {i: _P(l["poly"]) for i, l in enumerate(m["lanes"]) if len(l["poly"]) >= 3 and l["level"] == 0}
    lgrid = Grid()
    for i in lane_polys:
        lgrid.add(i, m["lanes"][i]["poly"])
    for wa in walkareas:
        P = _P(wa["poly"])
        wa["bad"] = False
        if not P.is_valid:
            continue
        for i in lgrid.near(*P.bounds):
            if i in lane_polys and lane_polys[i].is_valid and P.intersection(lane_polys[i].buffer(-0.3)).area > 0.3:
                wa["bad"] = True
                break
    from shapely.geometry import LineString
    grid = Grid()
    for i, l in enumerate(m["lanes"]):
        grid.add(i, l["line"])
    for s in sidewalks:
        L = LineString(s["line"])
        for i in grid.near(*L.bounds):
            l = m["lanes"][i]
            if l["internal"] or l["level"] != s["level"]:
                continue
            near = LineString(l["line"]).intersection(L.buffer((l["w"] + s["w"]) / 2 - 0.5))
            if near.length > 5:
                s["bad"] = True
                break
    m["sidewalks"], m["walkareas"] = sidewalks, walkareas


def _osm_levels():
    """Road level per OSM way: its layer tag, else 1 for a bridge and -1 for a tunnel, else 0 (ground)."""
    import xml.etree.ElementTree as ET
    out = {}
    for _, el in ET.iterparse(os.path.join(GURGAON, "gurgaon.osm")):
        if el.tag == "way":
            t = {x.get("k"): x.get("v") for x in el.findall("tag")}
            if "highway" in t:
                try:
                    lvl = int(t["layer"])
                except (KeyError, ValueError):
                    lvl = 1 if t.get("bridge") not in (None, "no") else -1 if t.get("tunnel") not in (None, "no") else 0
                out[el.get("id")] = lvl
            el.clear()
    return out


def _edge_level(levels, edge_id, name=""):
    """The OSM layer of the way the edge comes from; without one, a road named as an underpass is below ground
    (-1) and one named as a flyover above (1). OSM here tags almost no road bridges or tunnels, but names them."""
    import re
    m = re.match(r"-?(\d+)", edge_id)
    lvl = levels.get(m.group(1), 0) if m else 0
    if lvl == 0 and name:
        low = name.lower()
        lvl = -1 if "underpass" in low else 1 if "flyover" in low else 0
    return lvl


def build():
    import sumolib
    net = sumolib.net.readNet(os.path.join(GURGAON, "gurgaon.net.xml"), withInternal=True,
                              withPedestrianConnections=True, withPrograms=True)
    m = {"lanes": [], "junctions": [], "walks": [], "crossings": [], "markings": [], "signals": [],
         "buildings": [], "greens": [], "water": [], "levels": {"junctions": [], "walks": [], "markings": [], "crossings": []}}
    lane_index = {}
    levels = _osm_levels()
    for e in net.getEdges(withInternal=True):
        fn = e.getFunction()
        lvl = 0 if fn else _edge_level(levels, e.getID(), e.getName())
        if fn == "walkingarea":
            for l in e.getLanes():
                if len(l.getShape()) >= 3:
                    m["walks"].append([tuple(p) for p in l.getShape()]); m["levels"]["walks"].append(0)
            continue
        if fn == "crossing":
            for l in e.getLanes():
                m["crossings"].append({"line": [tuple(p) for p in l.getShape()], "w": l.getWidth()}); m["levels"]["crossings"].append(0)
            continue
        vehicle = []
        for l in e.getLanes():
            line = [tuple(p) for p in l.getShape()]
            if len(line) < 2:
                continue
            if l.allows("pedestrian") and not l.allows("passenger") and not l.allows("motorcycle"):
                if fn == "":
                    m["walks"].append(_outline(line, l.getWidth())); m["levels"]["walks"].append(lvl)
                continue
            if not (l.allows("passenger") or l.allows("motorcycle")):
                continue
            lane_index[l.getID()] = len(m["lanes"])
            m["lanes"].append({"id": l.getID(), "edge": e.getID(), "internal": fn != "", "line": line, "w": l.getWidth(),
                               "poly": _outline(line, l.getWidth()), "speed": l.getSpeed(),
                               # the drawn line's length: SUMO gives every lane of a curved road the length of its
                               # centre line, so an outer lane's end would sit metres short of where s says it is
                               "len": sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(line, line[1:])),
                               "name": e.getName() if fn == "" else "", "index": l.getIndex(), "next": [], "next_dir": [],
                               "cls": (e.getType() or "").replace("highway.", ""), "prio": e.getPriority(),
                               "junction": l.getID()[1:].rsplit("_", 2)[0] if fn else e.getToNode().getID(),
                               "from_lane": None, "dir": "", "tl": None, "foes": [], "cross": [], "vturn": _turn_speed(line),
                               "level": lvl})
            vehicle.append(l)
        if fn == "" and len(vehicle) > 1:   # dashes between neighbouring lanes going the same way
            vehicle.sort(key=lambda l: l.getIndex())
            for a, b in zip(vehicle, vehicle[1:]):
                pa, pb = a.getShape(), b.getShape()
                if len(pa) == len(pb):
                    m["markings"].append([((x1 + x2) / 2, (y1 + y2) / 2) for (x1, y1), (x2, y2) in zip(pa, pb)])
                else:
                    m["markings"].append(_offset([tuple(p) for p in pa], -a.getWidth() / 2))
                m["levels"]["markings"].append(lvl)
    for e in net.getEdges(withInternal=False):
        for l in e.getLanes():
            if l.getID() not in lane_index:
                continue
            for c in l.getOutgoing():
                via = c.getViaLaneID()
                to = c.getToLane().getID()
                nxt = [i for i in (via, to) if i and i in lane_index]
                if nxt:
                    m["lanes"][lane_index[l.getID()]]["next"].append(lane_index[nxt[0]])
                    m["lanes"][lane_index[l.getID()]]["next_dir"].append(c.getDirection())
                if via in lane_index and to in lane_index:
                    v = m["lanes"][lane_index[via]]
                    v.update(next=[lane_index[to]], next_dir=["s"], dir=c.getDirection(), from_lane=lane_index[l.getID()],
                             level=m["lanes"][lane_index[l.getID()]]["level"])
    for n in net.getNodes():
        sh = n.getShape()
        if len(sh) >= 3:
            m["junctions"].append([tuple(p[:2]) for p in sh])
            inc = [_edge_level(levels, e.getID(), e.getName()) for e in n.getIncoming()]
            m["levels"]["junctions"].append(max(set(inc), key=inc.count) if inc else 0)
    for tl in net.getTrafficLights():
        progs = tl.getPrograms()
        if not progs:
            continue
        prog = progs[sorted(progs)[0]]
        links = defaultdict(list)
        for inc, out, idx in tl.getConnections():
            links[inc.getID()].append(idx)
            for c in inc.getOutgoing():
                if c.getToLane().getID() == out.getID() and c.getViaLaneID() in lane_index:
                    m["lanes"][lane_index[c.getViaLaneID()]]["tl"] = (len(m["signals"]), idx)
        stops = []
        for lid, idxs in links.items():
            if lid not in lane_index:
                continue
            line = m["lanes"][lane_index[lid]]["line"]
            (ax, ay), (bx, by) = line[-2], line[-1]
            d = math.hypot(bx - ax, by - ay) or 1
            w = m["lanes"][lane_index[lid]]["w"] / 2
            stops.append({"lane": lane_index[lid], "links": idxs,
                          "a": (bx + (ay - by) / d * w, by + (bx - ax) / d * w), "b": (bx - (ay - by) / d * w, by - (bx - ax) / d * w)})
        m["signals"].append({"id": tl.getID(), "phases": [(float(p.duration), p.state) for p in prog.getPhases()], "stops": stops})
    _foes_and_crossings(net, m, lane_index)
    _road_features(net, m)
    _footpaths(net, m)
    # the play area: the box the map was downloaded and cut to (clip_osm.py), with a wall round it in the game
    lon0, lat0, lon1, lat1 = map(float, net._location["origBoundary"].split(","))
    m["play_area"] = [tuple(net.convertLonLat2XY(lon, lat)) for lon, lat in ((lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1))]
    from shapely.geometry import Polygon as _Poly
    area = _Poly(m["play_area"])
    with open(os.path.join(GURGAON, "osm_context.json")) as f:
        ctx = json.load(f)
    for w in ctx["elements"]:
        g = w.get("geometry")
        if not g or len(g) < 4:
            continue
        poly = _Poly([tuple(net.convertLonLat2XY(p["lon"], p["lat"])) for p in g]).buffer(0)
        if not poly.intersects(area):
            continue
        poly = poly.intersection(area)   # parks and buildings the edge cuts through end at the wall
        parts = [poly] if poly.geom_type == "Polygon" else [q for q in getattr(poly, "geoms", []) if q.geom_type == "Polygon"]
        t = w.get("tags", {})
        for q in parts:
            if q.area < 1.0:
                continue
            pts = [tuple(c) for c in q.exterior.coords]
            if "building" in t:
                m["buildings"].append(pts)
                m.setdefault("building_tags", []).append(t)   # (heights and kinds, for the 3D scene)
            elif t.get("natural") == "water":
                m["water"].append(pts)
            else:
                m["greens"].append(pts)
    xmin, ymin, xmax, ymax = net.getBoundary()
    m["bounds"] = (xmin, ymin, xmax, ymax)
    m["lane_ids"] = lane_index
    return m


class RoadMap:
    def __init__(self, data):
        self.__dict__.update(data)
        self.grids = {}
        for key in ("lanes", "junctions", "walks", "crossings", "markings", "buildings", "greens", "water", "sidewalks"):
            g = Grid()
            for i, item in enumerate(getattr(self, key)):
                if key == "lanes":
                    pts = item["poly"] or item["line"]
                elif key in ("crossings", "sidewalks"):
                    pts = item["line"]
                else:
                    pts = item
                if pts:
                    g.add(i, pts)
            self.grids[key] = g
        # the lanes beside each lane of a road: (left, right); higher lane numbers lie to the right
        by_slot = {(l["edge"], l["index"]): i for i, l in enumerate(self.lanes) if not l["internal"]}
        self.beside = {i: (by_slot.get((l["edge"], l["index"] - 1)), by_slot.get((l["edge"], l["index"] + 1)))
                       for i, l in enumerate(self.lanes) if not l["internal"]}
        self._group_signals()
        from shapely.geometry import Polygon
        from shapely.prepared import prep
        self._road = {}
        for i, l in enumerate(self.lanes):
            if len(l["poly"]) >= 3:
                self._road[("lane", i)] = prep(Polygon(l["poly"]).buffer(0.25))   # a wheel on the kerb line still counts
        for i, j in enumerate(self.junctions):
            p = Polygon(j)
            if p.is_valid and p.area > 0:
                self._road[("junction", i)] = prep(p.buffer(0.5))

    def _group_signals(self, reach=70.0):
        """OpenStreetMap draws a big crossing as several small junctions close together, each with its own
        lights (the Golf Course Road and Sunset Boulevard crossing has three). Lights whose stop lines lie within
        reach metres of each other are one junction: signal_group gives each its group, they keep one clock,
        and only the first stop line on each approach counts. A stop line whose lane comes straight out of a
        junction of the same group is inner (stop["inner"]); the junction lanes after it get "tl_inner", and
        nobody stops there."""
        group = list(range(len(self.signals)))
        def root(k):
            while group[k] != k:
                group[k] = group[group[k]]
                k = group[k]
            return k
        for a, sa in enumerate(self.signals):
            for b in range(a + 1, len(self.signals)):
                if any(math.hypot(p["a"][0] - q["a"][0], p["a"][1] - q["a"][1]) < reach for p in sa["stops"] for q in self.signals[b]["stops"]):
                    group[root(b)] = root(a)
        self.signal_group = [root(k) for k in range(len(self.signals))]
        prev = {}
        for i, l in enumerate(self.lanes):
            for n in l["next"]:
                prev.setdefault(n, []).append(i)
        for k, sg in enumerate(self.signals):
            for st in sg["stops"]:
                st["inner"] = any(self.lanes[p]["internal"] and self.lanes[p]["tl"] and self.signal_group[self.lanes[p]["tl"][0]] == self.signal_group[k]
                                  for p in prev.get(st["lane"], ()))
                if st["inner"]:
                    for j in self.lanes[st["lane"]]["next"]:
                        self.lanes[j]["tl_inner"] = True

    @classmethod
    def load(cls, rebuild=False):
        if not CHECKOUT:   # an installed copy: the map ships built
            with open(data_file("gurgaon_map.pkl"), "rb") as f:
                return cls(pickle.load(f))
        os.makedirs(CACHE, exist_ok=True)
        path = os.path.join(CACHE, "gurgaon_map.pkl")
        src = [os.path.join(GURGAON, f) for f in ("gurgaon.net.xml", "osm_context.json")] + [os.path.abspath(__file__)]
        if rebuild or not os.path.exists(path) or os.path.getmtime(path) < max(os.path.getmtime(s) for s in src):
            data = build()
            with open(path + f".{os.getpid()}", "wb") as f:   # (then renamed: parallel workers never read half a file)
                pickle.dump(data, f)
            os.replace(path + f".{os.getpid()}", path)
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def on_road(self, x, y):
        """Whether the point is on asphalt (a vehicle lane or a junction)."""
        from shapely.geometry import Point
        p = Point(x, y)
        for i in self.grids["lanes"].near(x, y, x, y):
            if ("lane", i) in self._road and self._road[("lane", i)].contains(p):
                return True
        for i in self.grids["junctions"].near(x, y, x, y):
            if ("junction", i) in self._road and self._road[("junction", i)].contains(p):
                return True
        return False

    def nearest_lane(self, x, y, heading=None, within=6.0, level=None):
        """(lane index, distance, the lane's direction there in radians) of the closest lane, or None.

        With a heading, lanes running within 60 degrees of it win over closer lanes running the other way. With a
        level, a lane on that level wins whenever the point is on it (within half its width and 1 m): a car on
        the street above an underpass stays on the street."""
        best = None
        for i in self.grids["lanes"].near(x - within, y - within, x + within, y + within):
            line = self.lanes[i]["line"]
            for (ax, ay), (bx, by) in zip(line, line[1:]):
                dx, dy = bx - ax, by - ay
                L2 = dx * dx + dy * dy or 1e-9
                u = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
                d = math.hypot(ax + u * dx - x, ay + u * dy - y)
                if d > within:
                    continue
                ang = math.atan2(dy, dx)
                score = d
                if heading is not None and abs(math.atan2(math.sin(ang - heading), math.cos(ang - heading))) > math.radians(60):
                    score += 100
                if level is not None and self.lanes[i]["level"] != level:
                    score += 1000 if d < self.lanes[i]["w"] / 2 + 1.0 else 0   # (only once off our level's lanes does it switch)
                if best is None or score < best[0]:
                    best = (score, i, d, ang)
        return None if best is None else best[1:]

    def random_start(self, rng):
        """A random place to start a drive: on a main road (primary, secondary or tertiary) at street level,
        at least 25 m from either end of its lane and 120 m from the wall round the map, facing the way the lane
        runs. Longer lanes are picked more often, so starts spread evenly along the roads."""
        cands = self.__dict__.get("_start_lanes")
        if cands is None:
            cands = [i for i, l in enumerate(self.lanes) if not l["internal"] and l["level"] == 0 and l["len"] >= 60
                     and l["cls"] in ("primary", "secondary", "tertiary")]
            self._start_lanes = cands
        area = getattr(self, "play_area", None)
        for _ in range(200):
            i = rng.choices(cands, [self.lanes[c]["len"] for c in cands])[0]
            l = self.lanes[i]
            s0 = rng.uniform(25.0, l["len"] - 25.0)
            for (ax, ay), (bx, by) in zip(l["line"], l["line"][1:]):
                L = math.hypot(bx - ax, by - ay)
                if s0 <= L:
                    h = math.atan2(by - ay, bx - ax)
                    x, y = ax + math.cos(h) * s0, ay + math.sin(h) * s0
                    break
                s0 -= L
            else:
                continue
            if area and min(abs((qx - px) * (y - py) - (qy - py) * (x - px)) / math.hypot(qx - px, qy - py)
                            for (px, py), (qx, qy) in zip(area, area[1:] + area[:1])) < 120:
                continue
            return x, y, h
        return self.start_pose("medanta")

    def near_wall(self, margin=4.0):
        """The lanes that come within margin metres of the wall round the map (a road along the map's edge, or
        one running into it): routes keep off them, as a car following them would scrape the wall."""
        out = self.__dict__.get("_near_wall")
        if out is None:
            area = getattr(self, "play_area", None)
            out = set()
            if area:
                from shapely.geometry import LineString, Polygon
                wall = Polygon(area).exterior
                out = {i for i, l in enumerate(self.lanes) if LineString(l["line"]).distance(wall) < margin}
            self._near_wall = out
        return out

    def too_tight(self, radius=4.0):
        """The junction lanes that turn (over 45 degrees) tighter than radius metres: netconvert draws small
        residential corners at about 2.6 m, tighter than a car can turn (3.7 m for ours: a 2.6 m wheelbase and
        35 degrees of steering), so a route through one asks the impossible and routes keep off them."""
        out = self.__dict__.get("_too_tight")
        if out is None:
            out = set()
            for i, l in enumerate(self.lanes):
                if not l["internal"] or len(l["line"]) < 3:
                    continue
                hs = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(l["line"], l["line"][1:])]
                turn = abs(sum(math.atan2(math.sin(h2 - h1), math.cos(h2 - h1)) for h1, h2 in zip(hs, hs[1:])))
                if turn > math.radians(45) and l["len"] / turn < radius:
                    out.add(i)
            self._too_tight = out
        return out

    def start_pool(self, n=256):
        """n random starts (random_start), always the same ones: a drive's seed picks one, so the browser knows
        where a drive begins before the simulator has loaded, and the same seed starts in the same place."""
        pool = self.__dict__.get("_start_pool")
        if pool is None:
            import random
            rng = random.Random(2026)
            pool = self._start_pool = [tuple(round(v, 3) for v in self.random_start(rng)) for _ in range(n)]
        return pool

    def start_pose(self, place="sunset", edge=START_EDGE, ahead=20.0):
        """Where a drive starts: beside a named place (STARTS), on the nearest main-road lane, facing the way
        it runs; or ("sunset") on Sunset Boulevard, a 3-lane main road on our morning route, in the middle
        lane, 20 m past the junction. A tuple (x, y, heading) is used as it is (the browser version passes poses
        worked out in advance)."""
        if isinstance(place, (tuple, list)):
            return tuple(place)
        if place in STARTS:
            from .geo import from_lonlat
            x, y = from_lonlat(*STARTS[place])
            best = None
            for i in self.grids["lanes"].near(x - 300, y - 300, x + 300, y + 300):
                l = self.lanes[i]
                if l["internal"] or l["len"] < 30 or not l["name"] or l["cls"] in ("residential", "living_street", "service"):
                    continue   # a named main road, not a slip road
                line, acc = l["line"], 0.0
                for (ax, ay), (bx, by) in zip(line, line[1:]):
                    L = math.hypot(bx - ax, by - ay) or 1e-9
                    u = max(0.0, min(L, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / L))
                    d = math.hypot(ax + (bx - ax) * u / L - x, ay + (by - ay) * u / L - y)
                    if best is None or d < best[0]:
                        best = (d, i, acc + u)
                    acc += L
            if best is not None:
                _, i, s0 = best
                l = self.lanes[i]
                s0 = max(3.0, min(l["len"] - 3.0, s0))
                for (ax, ay), (bx, by) in zip(l["line"], l["line"][1:]):
                    L = math.hypot(bx - ax, by - ay)
                    if s0 <= L:
                        h = math.atan2(by - ay, bx - ax)
                        return ax + math.cos(h) * s0, ay + math.sin(h) * s0, h
                    s0 -= L
        lanes = sorted((l for l in self.lanes if l["edge"] == edge and not l["internal"]), key=lambda l: l["index"])
        lane = lanes[len(lanes) // 2]
        line, u = lane["line"], ahead
        for (ax, ay), (bx, by) in zip(line, line[1:]):
            L = math.hypot(bx - ax, by - ay)
            if u <= L:
                h = math.atan2(by - ay, bx - ax)
                return ax + math.cos(h) * u, ay + math.sin(h) * u, h
            u -= L
        (ax, ay), (bx, by) = line[-2], line[-1]
        return bx, by, math.atan2(by - ay, bx - ax)

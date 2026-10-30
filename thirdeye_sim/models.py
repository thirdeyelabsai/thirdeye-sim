"""What road users and signals look like: boxes, shared by the desktop renderer (render3d.py) and the browser
version (web.py), so both draw the same thing.

A road user is a few boxes in its own frame: (ahead, left, z0, length, width, height, r, g, b, emit), with
ahead and left in metres from its centre and emit from 0 (lit by the sun) to 1 (glowing: lamps). Signals
come as boxes in world coordinates: (x, y, z0, heading, length, width, height, r, g, b, emit).
"""
import math

from .signals import Signals

CAR_COLOURS = [(0.93, 0.93, 0.92)] * 7 + [(0.72, 0.73, 0.75)] * 4 + [(0.42, 0.43, 0.46)] * 3 + [(0.12, 0.12, 0.13)] * 2 + \
              [(0.70, 0.12, 0.10), (0.16, 0.30, 0.62), (0.55, 0.45, 0.35), (0.80, 0.70, 0.55)]
CLOTHES = [(0.95, 0.95, 0.92), (0.20, 0.35, 0.70), (0.95, 0.55, 0.15), (0.55, 0.12, 0.18), (0.18, 0.50, 0.30), (0.15, 0.15, 0.16),
           (0.85, 0.75, 0.30), (0.60, 0.30, 0.55)]
HELMETS = [(0.1, 0.1, 0.1), (0.92, 0.92, 0.9), (0.75, 0.1, 0.1), (0.15, 0.25, 0.6)]
TRUCK_CAB = [(0.95, 0.55, 0.10), (0.20, 0.40, 0.75), (0.85, 0.80, 0.20), (0.75, 0.15, 0.12)]
TARP = [(0.35, 0.40, 0.30), (0.20, 0.35, 0.60), (0.55, 0.40, 0.25), (0.80, 0.55, 0.15)]
BUS = [(0.16, 0.36, 0.70), (0.80, 0.30, 0.12)]
GLASS = (0.10, 0.12, 0.15)
HEAD = (1.0, 0.95, 0.80)
AMBER = (1.0, 0.62, 0.08)
EGO = (0.80, 0.10, 0.09)
LIT = {"green": (0.15, 0.95, 0.35), "yellow": (1.0, 0.80, 0.0), "red": (1.0, 0.12, 0.08)}


def _pick(seq, k):
    return seq[(k * 2654435761) % 4294967296 % len(seq)]


def _tail(braking):
    return ((1.0, 0.08, 0.06), 1.0) if braking else ((0.45, 0.05, 0.05), 0.3)


def car_boxes(L, W, col, braking, signal, blink, crashed=False):
    if crashed and blink:
        col = (1.0, 0.45, 0.1)
    out = [(0, 0, 0.25, L, W, 0.78, *col, 0.0),
           (-0.06 * L, 0, 1.03, 0.50 * L, 0.88 * W, 0.42, *GLASS, 0.0),
           (-0.09 * L, 0, 1.45, 0.36 * L, 0.80 * W, 0.05, *col, 0.0)]
    tail, glow = _tail(braking)
    for side in (1, -1):
        out.append((L / 2 - 0.05, side * (W / 2 - 0.3), 0.6, 0.12, 0.38, 0.14, *HEAD, 0.9))
        out.append((-L / 2 + 0.05, side * (W / 2 - 0.28), 0.68, 0.12, 0.40 if braking else 0.34, 0.15, *tail, glow))
        if (signal == side or signal == 2) and blink:   # (2: hazard lights, both sides)
            out.append((L / 2 - 0.1, side * (W / 2 - 0.05), 0.65, 0.22, 0.14, 0.13, *AMBER, 1.0))
            out.append((-L / 2 + 0.1, side * (W / 2 - 0.05), 0.70, 0.22, 0.14, 0.13, *AMBER, 1.0))
    return out


def user_boxes(u, t):
    """The boxes of one road user (pedestrian or vehicle) in its own frame."""
    k, kind = u.id, u.kind
    blink = (t * 1.6) % 1.0 < 0.5
    if kind == "pedestrian":
        return [(0, 0, 0.0, 0.26, 0.36, 0.82, *(_pick(CLOTHES, k + 3) if k % 3 else (0.2, 0.22, 0.3)), 0.0),
                (0, 0, 0.82, 0.3, 0.46, 0.6, *_pick(CLOTHES, k), 0.0),
                (0, 0, 1.42, 0.22, 0.22, 0.24, 0.12, 0.09, 0.07, 0.0)]
    if kind == "cow":   # white or grey with a hump, as Gurgaon's cows mostly are
        col = _pick([(0.92, 0.91, 0.88), (0.80, 0.78, 0.74), (0.62, 0.58, 0.52), (0.55, 0.40, 0.28)], k)
        out = [(0, 0, 0.62, 1.55, 0.6, 0.72, *col, 0.0), (0.45, 0, 1.2, 0.35, 0.36, 0.22, *col, 0.0),   # body, hump
               (0.95, 0, 0.95, 0.42, 0.3, 0.32, *col, 0.0), (1.12, 0, 0.98, 0.12, 0.22, 0.16, 0.25, 0.2, 0.18, 0.0)]   # head, nose
        for side in (1, -1):
            out.append((1.0, side * 0.14, 1.25, 0.08, 0.06, 0.16, 0.85, 0.8, 0.7, 0.0))                      # horns
            for ax in (0.55, -0.55):
                out.append((ax, side * 0.2, 0.0, 0.14, 0.14, 0.64, *col, 0.0))                              # legs
        out.append((-0.82, 0, 0.75, 0.08, 0.06, 0.5, *col, 0.0))                                           # tail
        return out
    if kind == "dog":
        col = _pick([(0.62, 0.45, 0.25), (0.15, 0.13, 0.12), (0.78, 0.66, 0.45), (0.52, 0.5, 0.48)], k)
        out = [(0, 0, 0.3, 0.62, 0.22, 0.24, *col, 0.0), (0.38, 0, 0.42, 0.24, 0.18, 0.18, *col, 0.0),
               (-0.36, 0, 0.45, 0.16, 0.05, 0.05, *col, 0.0)]
        for side in (1, -1):
            for ax in (0.22, -0.22):
                out.append((ax, side * 0.07, 0.0, 0.07, 0.07, 0.32, *col, 0.0))
        return out
    braking = getattr(u, "acc", 0.0) < -1.2
    signal = 0
    if getattr(u, "lc", None):
        signal = 1 if u.lc[1] > 0 else -1
    if getattr(u, "hazard", False):
        signal = 2
    L, W = u.length, u.width
    tail, glow = _tail(braking)
    if kind == "car":
        return car_boxes(L, W, _pick(CAR_COLOURS, k), braking, signal, blink, u.crashed)
    if kind == "auto":
        out = [(0.1, 0, 0.22, L, W, 0.72, 0.12, 0.50, 0.22, 0.0),
               (L / 2 - 0.25, 0, 0.5, 0.5, W * 0.7, 0.75, 0.95, 0.80, 0.15, 0.0),
               (-0.15, 0, 0.94, L * 0.78, W * 1.02, 0.80, 0.07, 0.07, 0.07, 0.0),
               (L / 2 - 0.05, 0, 0.55, 0.1, 0.3, 0.14, *HEAD, 0.9)]
        for side in (1, -1):
            out.append((-L / 2 + 0.08, side * (W / 2 - 0.2), 0.55, 0.1, 0.25, 0.12, *tail, glow))
            if (signal == side or signal == 2) and blink:
                out.append((-L / 2 + 0.1, side * (W / 2 - 0.05), 0.75, 0.16, 0.12, 0.12, *AMBER, 1.0))
                out.append((L / 2 - 0.3, side * (W * 0.35 - 0.05), 0.75, 0.16, 0.12, 0.12, *AMBER, 1.0))
        return out
    if kind in ("two_wheeler", "cycle"):
        frame = (0.10, 0.10, 0.11) if kind == "two_wheeler" else (0.25, 0.25, 0.28)
        out = [(0, 0, 0.3, L, 0.28, 0.45 if kind == "two_wheeler" else 0.3, *frame, 0.0),
               (L / 2 - 0.45, 0, 0.9, 0.08, 0.68, 0.05, 0.2, 0.2, 0.2, 0.0),
               (-0.15, 0, 0.75, 0.42, 0.50, 0.55, *_pick(CLOTHES, k), 0.0),
               (-0.08, 0, 1.30, 0.30, 0.28, 0.27, *(_pick(HELMETS, k) if kind == "two_wheeler" else (0.12, 0.09, 0.07)), 0.0)]
        if kind == "two_wheeler":
            out.append((L / 2 - 0.05, 0, 0.7, 0.1, 0.16, 0.12, *HEAD, 0.9))
            out.append((-L / 2 + 0.05, 0, 0.7, 0.1, 0.16, 0.10, *tail, glow))
        return out
    if kind == "bus":
        col = _pick(BUS, k)
        out = [(0, 0, 0.4, L, W, 1.15, *col, 0.0),
               (0, 0, 1.55, L * 0.97, W * 1.01, 0.95, *GLASS, 0.0),
               (0, 0, 2.5, L, W, 0.45, *col, 0.0),
               (-0.5, 0, 2.95, 2.6, 1.6, 0.32, 0.80, 0.80, 0.80, 0.0)]
        for side in (1, -1):
            out.append((L / 2 - 0.05, side * (W / 2 - 0.35), 0.75, 0.1, 0.4, 0.2, *HEAD, 0.9))
            out.append((-L / 2 + 0.05, side * (W / 2 - 0.3), 0.9, 0.1, 0.35, 0.25, *tail, glow))
        return out
    if kind == "truck":
        out = [(L / 2 - 1.05, 0, 0.5, 2.1, W, 2.3, *_pick(TRUCK_CAB, k), 0.0),
               (L / 2 - 0.15, 0, 1.6, 0.25, W * 0.85, 0.75, *GLASS, 0.0),
               (-1.1, 0, 0.9, L - 2.3, W, 2.6, *_pick(TARP, k + 1), 0.0)]
        for side in (1, -1):
            out.append((-L / 2 + 0.05, side * (W / 2 - 0.3), 0.9, 0.1, 0.35, 0.2, *tail, glow))
        return out
    return []


def ego_boxes(e):
    return car_boxes(e.spec.length, e.spec.width, EGO, e.accel < -1.0, 0, False)


def signal_boxes(world, cx, cy, reach=200.0):
    """Signals near (cx, cy): a pole, a lit head, and the stop line glowing in the light's colour. Inner stop
    lines (inside a junction drawn as several) have no lights."""
    rows, t = [], world.t
    for k, sg in enumerate(world.map.signals):
        st = world.signals.state(k, t)
        for stop in sg["stops"]:
            (ax, ay), (bx, by) = stop["a"], stop["b"]
            if stop["inner"] or abs(ax - cx) > reach or abs(ay - cy) > reach:
                continue
            col = LIT[Signals.light(st, stop["links"])]
            hh = math.atan2(by - ay, bx - ax)
            mx, my = (ax + bx) / 2, (ay + by) / 2
            L = max(1e-6, math.hypot(bx - ax, by - ay))
            rows.append((mx, my, 0.13, hh, L, 0.35, 0.02, *col, 0.9))
            px, py = ax + (ax - bx) / L * 0.6, ay + (ay - by) / L * 0.6
            rows.append((px, py, 0.0, hh, 0.16, 0.16, 3.6, 0.25, 0.25, 0.27, 0.0))
            rows.append((px, py, 3.3, hh, 0.45, 0.45, 1.0, 0.08, 0.08, 0.09, 0.0))
            rows.append((px, py, 4.3, hh, 0.36, 0.36, 0.06, *col, 1.0))
    return rows


def in_tunnel(m, u, cache):
    """Whether a vehicle on an underpass is under the street right now (the street surface covers it), so it
    is out of sight from above. cache: a dict kept by the caller."""
    l = m.lanes[u.lane]
    if l["level"] >= 0:
        return False
    if u.lane not in cache:   # which 2 m steps of this underpass lane lie under a street-level road surface
        from shapely.geometry import Point
        from .actors import _point
        covered = []
        for k in range(int(l["len"] // 2) + 1):
            x, y, _ = _point(l["line"], k * 2.0)
            pt = Point(x, y)
            under = any(m.lanes[i]["level"] == 0 and ("lane", i) in m._road and m._road[("lane", i)].contains(pt)
                        for i in m.grids["lanes"].near(x, y, x, y)) or \
                any(m.levels["junctions"][i] == 0 and ("junction", i) in m._road and m._road[("junction", i)].contains(pt)
                    for i in m.grids["junctions"].near(x, y, x, y))
            covered.append(under)
        cache[u.lane] = covered
    c = cache[u.lane]
    return c[min(len(c) - 1, max(0, int(u.s // 2)))] if c else False


def place(x, y, h, boxes):
    """Boxes in a road user's frame to world rows (x, y, z0, heading, length, width, height, r, g, b, emit)."""
    c, s = math.cos(h), math.sin(h)
    return [(x + c * ax - s * ay, y + s * ax + c * ay, z0, h, l, wd, ht, r, g, b, em) for ax, ay, z0, l, wd, ht, r, g, b, em in boxes]

"""Road users around our car: vehicles that follow lanes and pedestrians that walk footpaths.

Only our car's surroundings are simulated. What is visible is the screen plus everything within SENSE (50 m)
of our car all around, the most a car's own sensors would see; the zone is that plus a 20 m band (MARGIN) and
3 s of driving ahead. Each type is kept at the density we measured around our car on the morning drive
(gurgaon/step3/measured_density.json), per lane-km of road in the zone: missing road users appear in the band,
outside anything visible, extra ones are taken away there, and anyone 10 m past the zone is removed. Nobody
appears or vanishes in view, on screen or within the car's sensing range. A vehicle appearing on a road whose
light ahead is red joins the back of the queue, stopped, so queues are there before they come into view. At
the start the whole zone, including what is visible, is filled.

Vehicles (types and base properties from gurgaon/vtypes.add.xml, step 2):
- follow the lane centre line, choosing a turn at each junction (straight 60%, the rest spread over the turns);
- keep a safe gap with the Intelligent Driver Model: speed up towards a desired speed (the driver's own pace on
  that kind of road, see ROAD_KMH and PACE, wandering over time with the odd spurt), slow for the vehicle
  ahead, never move into it;
- slow for bends to keep sideways acceleration under 2.5 m/s^2;
- stop at red and at yellow when they can stop in time;
- enter a junction only when no vehicle is on a lane that crosses or merges with theirs, our car is not
  about to enter one, and there is room on the road they are turning into (one user at a time per conflict,
  so junctions cannot jam into overlaps);
- where two roads are drawn over each other running opposite ways, use the stretch one direction at a time,
  taking turns with whoever has waited longest;
- give way to pedestrians on a zebra crossing;
- treat our car like any other vehicle ahead of them, wherever it is;
- change lanes to get past a slower or stopped vehicle (our car included) when the lane beside is quicker and
  its gaps allow (MOBIL with Indian settings: a driver weighs the braking it forces on the vehicle it cuts in
  front of lightly, accepts short gaps, and will make that vehicle brake at up to 2 to 4 m/s^2). The vehicle
  behind in the new lane slows for it as soon as it starts moving across. After the change it picks its next
  turn from the new lane, so a blocked vehicle may end up on another street;
- when our car honks or flashes its headlights, the road user ahead may make way (moving over, or squeezing
  to the kerb and speeding up) after a reaction time; a vehicle we hold up for 1.5 s honks at us, but not
  while we wait at a red light or behind a queue;
- two-wheelers and cycles ride off the lane centre (cycles at the kerb, a quarter of two-wheelers on a lane
  line) and squeeze past a slower vehicle in the same lane, our car included, when there is room beside it.
  A vehicle sticking out over a lane line is seen by the lane beside.

Pedestrians walk the footpaths at about 1.3 m/s, turn at corners, and cross at zebra crossings only when no
vehicle will reach the crossing within 4 s (and, at a signal, only while traffic across it has red). A
pedestrian whose next step would touch a vehicle or our car waits. When we honk, pedestrians near the road ahead
wait 3 s before stepping out, and those on a crossing hurry.
"""
import json, math, os, random
import xml.etree.ElementTree as ET

from .paths import data_file
from .signals import Signals

HERE = os.path.dirname(os.path.abspath(__file__))
SENSE = 50.0     # metres all around our car that it could sense (cameras, lidar): always simulated, never spawned in
MARGIN = 20.0    # the band outside everything visible where road users appear and are taken away
LOOK = 3.0       # seconds of driving ahead added to the zone
GONE = 10.0      # how far past the zone a road user goes before it is removed (so the edge does not flicker)
REAL_NAME = {"auto": "autorickshaw", "cycle": "bicycle"}
# Our density box (30 m ahead and behind, 10 m to each side) covers about 6 lanes (our side of a main road and
# the near half of the other) and 2 footpaths; dividing the measured counts by that gives road users per
# lane-km and per km of footpath.
BOX_LANE_KM, BOX_FOOTPATH_KM = 0.06 * 6, 0.06 * 2
# How busy each road class is compared with the main roads we measured on (an assumption, to check).
CLASS_SHARE = {"motorway": 1.0, "trunk": 1.0, "primary": 1.0, "secondary": 1.0, "tertiary": 0.6, "unclassified": 0.3,
               "residential": 0.2, "living_street": 0.1}
TURN_WEIGHT = {"s": 6.0, "l": 2.0, "r": 2.0, "L": 1.0, "R": 1.0, "T": 0.05}
# Pedestrians on the drive were often walking along the road edge, not on a footpath; the footpaths here
# carry 3.5 times the footpath density so the count around our car matches.
PED_CALIBRATION = 3.5
# How fast people drive. A car's free-flow speed (km/h) on each kind of road, from our team's driving in
# Gurgaon (assumed, not measured: cars mostly 45 to 60 on main roads), and each driver's own pace as a multiple of it, drawn per
# driver by type (mean, spread, lowest, highest); anything over the road's limit is overspeeding, which every
# type does sometimes. On top, each driver's speed wanders (a few km/h over ten seconds or so) and now and
# then spurts or eases off for a few seconds.
ROAD_KMH = {"motorway": 85, "trunk": 62, "primary": 58, "secondary": 56, "tertiary": 48, "unclassified": 36,
            "residential": 28, "living_street": 15, "service": 20}
PACE = {"cow": (0.05, 0.0, 0.05, 0.05), "car": (1.0, 0.1, 0.8, 1.35), "two_wheeler": (0.88, 0.12, 0.65, 1.35), "auto": (0.75, 0.08, 0.6, 0.95),
        "bus": (0.85, 0.08, 0.7, 1.0), "truck": (0.8, 0.08, 0.65, 0.95), "cycle": (0.3, 0.05, 0.2, 0.4)}
TOP_KMH = {"car": 120, "two_wheeler": 90, "auto": 55, "bus": 80, "truck": 80, "cycle": 25}   # (animals: ANIMALS)
WANDER_SD, WANDER_TAU = 0.07, 10.0     # the slow wander of a driver's pace: spread, and how long it lasts (s)
SPURT_EVERY = 120.0                     # seconds, on average, between a driver's spurts or easings-off
# chaos (world.chaos, 1 = as measured): a stress setting, not a measurement. Above 1, every scripted event is due
# that many times as often, and drivers care less about the braking they cause (MOBIL politeness / chaos), need
# less gain to change lanes (threshold / chaos), force harder braking on others (x chaos, at most 6 m/s^2),
# spurt and wander more, and honk sooner. The measured densities stay as they are.
# style (world.style): "gurgaon" as measured, or "generic", traffic as simulators usually model it, for comparison:
# everyone keeps to the middle of a lane (no filtering, no riding the line), lane changes with MOBIL's standard
# politeness 0.5 and threshold 0.2 m/s^2 (Treiber and Kesting), nobody honks, and no scripted events.
# How likely each kind of road user ahead is to make way when our car honks or flashes its headlights (an
# assumption, to check against the footage); flashing works a little less often. Each honk is a new chance.
YIELD = {"two_wheeler": 0.75, "cycle": 0.85, "auto": 0.6, "car": 0.5, "bus": 0.3, "truck": 0.3, "cow": 0.0}
NAME = {"two_wheeler": "two-wheeler", "auto": "auto", "car": "car", "bus": "bus", "truck": "truck", "cycle": "cycle",
        "cow": "cow", "dog": "dog"}
# animals (events.py puts them on the road; their sizes are typical, not measured): a cow stands in a lane for 20
# to 60 s (assumed) while traffic goes round it, then walks off the road; a dog runs across like a pedestrian, only
# faster
ANIMALS = {"cow": {"length": 2.1, "width": 0.75, "max_speed": 1.0, "accel": 0.5, "decel": 2.0, "emergency": 3.0, "tau": 1.0,
                   "min_gap": 1.0, "speed_factor": [1.0, 0.0, 1.0, 1.0]}}
ANIMAL_KINDS = ("cow", "dog")
# Sideways speed during a lane change at normal speed (m/s); a car crossing a 3.5 m lane takes about 3 s.
LAT_RATE = {"car": 1.1, "auto": 1.0, "two_wheeler": 1.4, "bus": 0.7, "truck": 0.7, "cycle": 0.5, "cow": 0.2}


def load_types():
    """Vehicle types from vtypes.add.xml: size, limits, reaction time, gaps and desired-speed spread."""
    out = {}
    for vt in ET.parse(data_file("vtypes.add.xml")).getroot().iter("vType"):
        if vt.get("vClass") == "pedestrian":
            continue
        sf = [float(x) for x in vt.get("speedFactor", "normc(1,0,1,1)")[6:-1].split(",")]
        out[vt.get("id")] = {"length": float(vt.get("length")), "width": float(vt.get("width")),
                             "max_speed": float(vt.get("maxSpeed")), "accel": float(vt.get("accel")),
                             "decel": float(vt.get("decel")), "emergency": float(vt.get("emergencyDecel")),
                             "tau": float(vt.get("tau")), "min_gap": float(vt.get("minGap")), "speed_factor": sf}
        if vt.get("id") in TOP_KMH:   # top speeds here, not SUMO's (step 2 capped everyone at 70 km/h)
            out[vt.get("id")]["max_speed"] = TOP_KMH[vt.get("id")] / 3.6
    return out


def _point(line, s):
    """(x, y, heading) at distance s along a polyline."""
    for (ax, ay), (bx, by) in zip(line, line[1:]):
        L = math.hypot(bx - ax, by - ay)
        if s <= L or (bx, by) == line[-1]:
            u = max(0.0, min(1.0, s / L)) if L > 0 else 0.0
            return ax + (bx - ax) * u, ay + (by - ay) * u, math.atan2(by - ay, bx - ax)
        s -= L
    (ax, ay), (bx, by) = line[-2], line[-1]
    return bx, by, math.atan2(by - ay, bx - ax)


def a_or_an(name):
    """'an auto', 'a car': the article for a road user's name."""
    return ("an " if name[:1] in "aeiou" else "a ") + name


class Vehicle:
    def __init__(self, uid, kind, spec, lane, s, rng, roadmap):
        self.id, self.kind, self.spec = uid, kind, spec
        self.lane, self.s, self.v = lane, s, 0.0
        mean, sd, lo, hi = PACE[kind]
        self.factor = max(lo, min(hi, rng.gauss(mean, sd)))   # this driver's pace (see ROAD_KMH)
        self.wander = 1.0                                      # its slow wander, multiplying the pace
        self.spurt, self.spurt_until = 1.0, 0.0                # a spurt (above 1) or easing off (below 1) for a while
        w = roadmap.lanes[lane]["w"]
        self.max_lat = max(0.0, w / 2 - spec["width"] / 2 - 0.25)
        if kind == "cycle":
            self.home = self.max_lat              # at the kerb (left of the lane)
        elif kind == "two_wheeler":   # (an assumption, to check against close passes in the footage)
            r = rng.random()
            self.home = rng.uniform(0.3, 1.0) * self.max_lat if r < 0.4 else rng.uniform(-1.0, 1.0) * self.max_lat if r < 0.75 \
                else rng.choice((-1, 1)) * (w / 2 - spec["width"] / 2 + 0.3)   # on a lane line
        else:
            self.home = 0.0
        self.lat = self.home
        self.lat_v = 0.0                        # sideways speed this step (m/s, + to the left)
        self.plan = [lane]
        self.reserved = None
        self.stretches = {}                     # overlap stretches it holds: {stretch: the lane of its part}
        self.zone_wait = None                   # (overlap stretch, since when) while waiting to enter one
        self.zone_asked = -1.0                  # when it last asked for one from the front of its queue
        self.stopped_for = 0.0
        self.crashed = False
        self.acc = 0.0
        # how this driver changes lanes
        self.polite = rng.uniform(0.0, 0.3)     # weight on the braking it causes behind (0 = none at all)
        self.gain = rng.uniform(0.2, 0.6) + (0.4 if kind in ("bus", "truck") else 0.0)   # m/s^2 it must win
        self.push = rng.uniform(1.5, 2.5) if kind in ("bus", "truck") else rng.uniform(2.0, 4.0)   # hardest braking it forces
        self.lat_rate = LAT_RATE[kind]
        self.lc = None                          # while changing lane: (lane it is leaving, that lane's offset)
        self.lc_target = 0.0
        self.lc_next = 0.0                      # earliest time of its next lane change
        self.make_way = None                    # (when, whether it will, "horn" or "dipper") after we signal it
        self.aside_until = self.boost_until = 0.0
        self.honk_t, self.held, self.next_honk = -9.0, 0.0, 0.0
        # an event (events.py) can take this vehicle over: {"speed": m/s, "lat": lateral target, "side": pass on,
        # "brake": (decel, down to m/s, hold s), "parked": True, "wrong_way": True, "red": True, "cut_in": side}
        self.script = None
        self.x = self.y = self.heading = 0.0

    @property
    def length(self):
        return self.spec["length"]

    @property
    def width(self):
        return self.spec["width"]


class Pedestrian:
    def __init__(self, uid, sidewalk, u, direction, rng):
        self.id, self.kind = uid, "pedestrian"
        self.where = ("sw", sidewalk)      # ("sw", footpath) | ("wa", corner area) | ("cr", crossing)
        self.u, self.dir = u, direction    # distance along the footpath or crossing, and which way
        self.speed = self.base = max(0.8, rng.gauss(1.3, 0.2))
        self.hold_until = self.hurry_until = 0.0   # after our horn: wait before stepping out, or hurry across
        self.script = None                         # an event walking it ("free": events.py places it)
        self.lat = rng.uniform(-0.5, 0.5)
        self.path = None                   # inside a corner area: straight line from a to b
        self.after = None                  # and the second leg, when the corner is not convex
        self.came_from = None
        self.waiting = 0.0
        self.x = self.y = self.heading = 0.0
        self.length = self.width = 0.5
        self.crashed = False


class _Ghost:
    """The part of a lane-changing vehicle still in the lane it is leaving, as the vehicles in that lane see it."""
    __slots__ = ("owner", "id", "kind", "lane", "s", "lat", "v", "length", "width")

    def __init__(self, v, lane, s, lat):
        self.owner, self.id, self.kind, self.lane = v, v.id, v.kind, lane
        self.s, self.lat, self.v, self.length, self.width = s, lat, v.v, v.length, v.width


def _boxes_touch(a, b):
    """Whether two rectangles (x, y, heading, length, width) overlap (separating-axis test)."""
    def corners(x, y, h, L, W):
        c, n = math.cos(h), math.sin(h)
        return [(x + c * dx - n * dy, y + n * dx + c * dy) for dx, dy in ((L / 2, W / 2), (L / 2, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2))]
    A, B = corners(*a), corners(*b)
    for h in (a[2], a[2] + math.pi / 2, b[2], b[2] + math.pi / 2):
        c, n = math.cos(h), math.sin(h)
        pa, pb = [x * c + y * n for x, y in A], [x * c + y * n for x, y in B]
        if max(pa) < min(pb) or max(pb) < min(pa):
            return False
    return True


class _Body:
    """Our car seen from a lane: distance along it, offset, size and speed."""
    __slots__ = ("s", "lat", "length", "width", "v")

    def __init__(self, s, lat, length, width, v):
        self.s, self.lat, self.length, self.width, self.v = s, lat, length, width, v


def _project(line, x, y):
    """(distance along a polyline, offset to its left) of the polyline point nearest to (x, y)."""
    best, acc = None, 0.0
    for (ax, ay), (bx, by) in zip(line, line[1:]):
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy) or 1e-9
        u = max(0.0, min(L, ((x - ax) * dx + (y - ay) * dy) / L))
        d = math.hypot(ax + dx * u / L - x, ay + dy * u / L - y)
        if best is None or d < best[0]:
            best = (d, acc + u, ((y - ay) * dx - (x - ax) * dy) / L)
        acc += L
    return best[1], best[2]


class _Proxy:
    """A road user's body as seen by others this step, keeping a link to the road user itself."""
    __slots__ = ("u", "uid", "x", "y", "heading", "length", "width", "v", "level", "lane")

    def __init__(self, u, b):
        self.u = u
        self.lane = getattr(u, "lane", None)
        self.uid, self.x, self.y, self.heading, self.length, self.width, self.v, self.level = \
            b.uid, b.x, b.y, b.heading, b.length, b.width, b.v, b.level

    def __eq__(self, other):
        return other is self.u or other is self


def apart(roadmap, a_lane, b_lane):
    """Whether two lanes pass over or under each other (an untagged flyover or underpass)."""
    return a_lane is not None and b_lane is not None and b_lane in roadmap.lanes[a_lane]["over"]


def level_of(roadmap, u):
    if hasattr(u, "where"):   # a pedestrian (or a dog: anything that walks rather than drives)
        kind, j = u.where
        return roadmap.sidewalks[j]["level"] if kind == "sw" else 0   # (corners, crossings and free walks: ground)
    return roadmap.lanes[u.lane]["level"]


class Traffic:
    def __init__(self, world, seed=0, density_scale=1.0):
        self.world, self.map = world, world.map
        self.rng = random.Random(seed)
        self.types = load_types()
        with open(data_file("step3/measured_density.json")) as f:
            measured = json.load(f)["day_0844"]
        self.per_lane_m = {k: measured[REAL_NAME.get(k, k)] / BOX_LANE_KM / 1000 * density_scale for k in self.types}
        self.ped_per_m = measured["pedestrian"] / BOX_FOOTPATH_KM / 1000 * density_scale * PED_CALIBRATION
        self.vehicles, self.peds = [], []
        self.next_id = 0
        self.on_crossing = {}     # crossing -> pedestrians on it
        self.clock = 0.0
        self._sample_cache = {}
        self._yields, self._steps = {}, 0
        self.removed_stuck = 0
        self.lane_changes = 0
        self.stuck_why = {}
        self._last_refusal = ""
        bad_wa = {k for k, wa in enumerate(self.map.walkareas) if wa.get("bad")}
        self._bad_sidewalks = {j for j, sw in enumerate(self.map.sidewalks) if sw["bad"]}
        self._bad_crossings = {k for k, c in enumerate(self.map.crossings) if any(w in bad_wa or w is None for w in c.get("wa", [None, None]))}
        self._bad_wa = bad_wa

    # where our car can see -----------------------------------------------------------------------------
    def in_view(self, x, y, margin=8.0):
        """Whether a point is on screen or within margin metres of it (the view is a rectangle in our car's frame)."""
        e = self.world.ego
        ahead, behind, half = self.world.view
        dx, dy = x - e.x, y - e.y
        lon = dx * math.cos(e.heading) + dy * math.sin(e.heading)
        lat = -dx * math.sin(e.heading) + dy * math.cos(e.heading)
        return -behind - margin < lon < ahead + margin and abs(lat) < half + margin

    def visible(self, x, y, margin=8.0):
        """Whether a point is on screen or within our car's sensing range (with margin): nobody appears there."""
        e = self.world.ego
        return math.hypot(x - e.x, y - e.y) < SENSE + margin or self.in_view(x, y, margin)

    def in_zone(self, x, y, extra=0.0):
        """Whether a point is in the simulated zone (grown by extra metres)."""
        e = self.world.ego
        dx, dy = x - e.x, y - e.y
        if math.hypot(dx, dy) < SENSE + MARGIN + extra or self.in_view(x, y, MARGIN + extra):
            return True
        lon = dx * math.cos(e.heading) + dy * math.sin(e.heading)
        lat = -dx * math.sin(e.heading) + dy * math.cos(e.heading)
        return 0 <= lon < SENSE + MARGIN + LOOK * max(0.0, e.speed) + extra and abs(lat) < SENSE + MARGIN + extra

    def hidden_from(self):
        """How far ahead, behind and to the side the visible region reaches: (ahead, behind, side) metres."""
        a, b, h = self.world.view
        return max(a, SENSE), max(b, SENSE), max(h, SENSE)

    def _reach(self):
        """Half-width of a square around our car that holds the whole zone (for grid lookups)."""
        a, b, h = self.world.view
        return max(SENSE, a, b, h) + MARGIN + LOOK * max(0.0, self.world.ego.speed) + 5.0

    # spawning --------------------------------------------------------------------------------------------
    def _samples(self, i):
        """Points every 8 m along a lane (or footpath), cached: (s, x, y)."""
        key = ("lane", i) if isinstance(i, int) else i
        c = self._sample_cache.get(key)
        if c is None:
            line = self.map.lanes[i]["line"] if key[0] == "lane" else self.map.sidewalks[key[1]]["line"]
            L = _len(line)
            c = [(u, *_point(line, u)[:2]) for u in [min(L, 4 + 8 * k) for k in range(max(1, int(L // 8)))]]
            self._sample_cache[key] = c
        return c

    def _zone(self):
        """Lane points in the zone, each standing for 8 m of lane: (lane, s, x, y, weight)."""
        e, r = self.world.ego, self._reach()
        pts = []
        for i in self.map.grids["lanes"].near(e.x - r, e.y - r, e.x + r, e.y + r):
            l = self.map.lanes[i]
            if l["internal"] or not l["next"]:
                continue
            share = self._share(l)
            for s0, x, y in self._samples(i):
                if self.in_zone(x, y):
                    pts.append((i, s0, x, y, 8.0 * share))
        return pts

    def _foot_zone(self):
        e, r = self.world.ego, self._reach()
        pts = []
        for j in self.map.grids["sidewalks"].near(e.x - r, e.y - r, e.x + r, e.y + r):
            if j in self._bad_sidewalks:
                continue
            for u, x, y in self._samples(("sw", j)):
                if self.in_zone(x, y):
                    pts.append((j, u, x, y))
        return pts

    def _queue_spot(self, lane, spec):
        """Where a vehicle joins the queue on a lane whose light ahead is red: just behind the last one
        waiting (or at the stop line), as s along the lane; None if the light is not red."""
        l = self.map.lanes[lane]
        if not l["next"]:
            return None
        nxt = next((n for n, d in zip(l["next"], l["next_dir"]) if d == "s"), l["next"][0])
        tl = self.map.lanes[nxt]["tl"]
        if not tl or self.map.lanes[nxt].get("tl_inner") or Signals.light(self.world.signals.state(tl[0], self.world.t), [tl[1]]) != "red":
            return None
        tail = l["len"] - 1.0
        for u in self.vehicles:
            if u.lane == lane and u.v < 1.0 and u.s - u.length / 2 < tail + 0.1 and l["len"] - u.s < 150:
                tail = min(tail, u.s - u.length / 2 - spec["min_gap"])
        return tail - spec["length"] / 2

    def _clear(self, x, y, length, level, room=6.0):
        for u in self.vehicles:
            if abs(u.x - x) < 25 and abs(u.y - y) < 25 and self.map.lanes[u.lane]["level"] == level and \
                    math.hypot(u.x - x, u.y - y) < (u.length + length) / 2 + room:
                return False
        e = self.world.ego
        return math.hypot(e.x - x, e.y - y) > (e.spec.length + length) / 2 + room + 4

    @staticmethod
    def _share(lane):
        return CLASS_SHARE.get(lane["cls"].replace("_link", ""), 0.3)

    def _temper(self, v, spawn=False):
        """v's lane-changing manner and line at the current chaos setting and style (see the notes at the top).
        A vehicle just appearing starts on its line; one already driving moves over to it."""
        polite, gain, push = getattr(v, "manner", (v.polite, v.gain, v.push))
        if not hasattr(v, "own_home"):
            v.own_home = v.home
        if self.world.style == "generic":
            polite, gain, v.home = 0.5, 0.2, 0.0
        else:
            v.home = v.own_home
        if spawn and v.script is None:
            v.lat = v.home
        c = self.world.chaos
        if c == 1.0:
            v.polite, v.gain, v.push = polite, gain, push
        else:
            v.polite, v.gain, v.push = polite / c, gain / c, min(6.0, push * c)

    def set_style(self, style):
        """Switch between "gurgaon" (as measured) and "generic" traffic while driving."""
        self.world.style = style
        for v in self.vehicles:
            self._temper(v)

    def set_chaos(self, c):
        """Change the chaos setting while driving: event rates at once, every driver's manner too."""
        self.world.chaos = max(0.25, min(4.0, c))
        for v in self.vehicles:
            self._temper(v)

    def _spawn_vehicle(self, kind, pts, allow_view=False):
        if not pts:
            return False
        weights = [p[4] for p in pts]
        spec = self.types[kind]
        for _ in range(12):
            i, s0, x, y, _w = self.rng.choices(pts, weights)[0]
            l = self.map.lanes[i]
            if kind == "bus" and l["cls"] == "residential":
                continue
            s0 = max(spec["length"] / 2 + 0.5, min(l["len"] - spec["length"] / 2 - 0.5, s0 + self.rng.uniform(-4, 4)))
            stopped = False
            q = self._queue_spot(i, spec)
            if q is not None and q - 150 < s0 < q + 30 and q > spec["length"] / 2:   # the light ahead is red: join the queue
                s0, stopped = q, True
            x, y, _ = _point(l["line"], s0)
            if (not allow_view and self.visible(x, y)) or not self.in_zone(x, y) or \
                    not self._clear(x, y, spec["length"], l["level"], room=0.5 if stopped else 6.0):
                continue
            v = Vehicle(self.next_id, kind, spec, i, s0, self.rng, self.map)
            inside = [z for z, a, b in l["zones"] if a - spec["length"] / 2 <= s0 <= b + spec["length"] / 2 and not self.map.zone_same[z]]
            if inside and not all(self._take_zone(v, z) for z in inside[:1]):
                continue   # an overlap stretch someone else is using
            v.v = 0.0 if stopped else self._desired(v) * 0.8
            v.manner = (v.polite, v.gain, v.push)   # as drawn; set_chaos scales from these
            self._temper(v, spawn=True)
            self.next_id += 1
            self.vehicles.append(v)
            self._place(v)
            return True
        return False

    def _spawn_ped(self, pts, allow_view=False):
        for _ in range(8):
            if not pts:
                return False
            j, u, x, y = self.rng.choice(pts)
            if not allow_view and self.visible(x, y, margin=4):
                continue
            p = Pedestrian(self.next_id, j, u, self.rng.choice((1, -1)), self.rng)
            self.next_id += 1
            self.peds.append(p)
            self._place_ped(p)
            return True
        return False

    def _top_up(self, allow_view=False, most=4, trim=False):
        """Bring each type to its measured density in the zone: add the missing ones in the band outside what
        is visible, and with trim, take extra ones away there (slow vehicles otherwise gather behind a car that
        passes them, and fast ones leave it)."""
        pts = self._zone()
        lane_m = sum(p[4] for p in pts)
        spots = pts if allow_view else [p for p in pts if not self.visible(p[2], p[3])]
        for kind, per_m in self.per_lane_m.items():
            want = lane_m * per_m
            mine = [v for v in self.vehicles if v.kind == kind and self.in_zone(v.x, v.y)]
            missing = want - len(mine)
            n = int(missing) + (1 if self.rng.random() < missing % 1 else 0) if missing > 0 else 0
            for _ in range(min(most, n)):
                self._spawn_vehicle(kind, spots, allow_view)
            if trim and missing < -1:
                hidden = [v for v in mine if not self.visible(v.x, v.y, margin=6) and v.reserved is None and not v.stretches and v.script is None]
                for v in self.rng.sample(hidden, min(len(hidden), int(-missing - 0.5))):
                    self.vehicles.remove(v)
        feet = self._foot_zone()
        want = len(feet) * 8.0 * self.ped_per_m * 0.5   # the measured box also counts the far side of each road
        have = sum(1 for p in self.peds if self.in_zone(p.x, p.y))
        for _ in range(min(most, max(0, int(round(want - have))))):
            self._spawn_ped(feet, allow_view)

    def fill(self):
        """At the start of a drive, fill the whole zone, including what is visible."""
        self._top_up(allow_view=True, most=10 ** 6)

    def _manage(self):
        e = self.world.ego
        def parked(v):   # stopped at the end of a dead end: it has reached where it was going
            l = self.map.lanes[v.lane]
            return not l["next"] and v.s > l["len"] - v.length - 1 and v.v < 0.5
        def stuck(v):    # a safety valve, as in SUMO: stopped for 30 s off screen means a jam the rules did not untie
            if v.stopped_for > 30 and v.script is None and not self.visible(v.x, v.y, margin=6):
                self.removed_stuck += 1
                self.stuck_why[getattr(v, "why", "?")] = self.stuck_why.get(getattr(v, "why", "?"), 0) + 1
                return True
            return False
        # out of the zone: gone (an event's road user gets more room, it was placed out there on purpose)
        self.vehicles = [v for v in self.vehicles if self.in_zone(v.x, v.y, GONE + (40.0 if v.script is not None else 0.0))
                         and v.plan and not parked(v) and not stuck(v)]
        self.peds = [p for p in self.peds if self.in_zone(p.x, p.y, GONE + (40.0 if p.script is not None else 0.0))]
        self._top_up(trim=True)

    # vehicles --------------------------------------------------------------------------------------------
    def _ahead(self, v, key, reach):
        """The road features of one kind (lane["bumps"], ["potholes"], ["stops"]) on v's plan within reach
        metres: (distance from v's front, the feature)."""
        out, off = [], -v.s - v.length / 2
        for k, i in enumerate(v.plan[:3]):
            l = self.map.lanes[i]
            for f in l[key]:
                d = off + (f[0] if isinstance(f, tuple) else f)
                if -v.length < d < reach:
                    out.append((d, f))
            off += l["len"]
            if off > reach:
                break
        return out

    def _road_features(self, v, v0):
        """Slow for a speed breaker (cross it at about 15 km/h, two-wheelers a little faster) and ease off for a
        pothole in our line (ASSUMED behaviour, like the potholes themselves: roadmap.py)."""
        cross = 5.5 if v.kind in ("two_wheeler", "cycle") else 4.2
        for d, _f in self._ahead(v, "bumps", 45.0):
            v0 = min(v0, math.sqrt(cross ** 2 + 2 * 1.5 * max(0.0, d - 1.0)) if d > 0 else cross)
        for d, (s, lat, r) in self._ahead(v, "potholes", 20.0):
            if abs(v.lat - lat) < r + v.width / 2:
                v0 = min(v0, max(3.0, 0.6 * v0))
        return v0

    def _bus_stop(self, v):
        """Distance to the stop a bus will stop at (the next one in its lane it has not served yet), dwelling
        8 to 20 s there; None when there is none."""
        if self.world.t < getattr(v, "dwell_until", 0.0):
            return 0.0
        served = v.__dict__.setdefault("served", set())
        for d, s in self._ahead(v, "stops", 60.0):
            if any(abs(round(s) - x[1]) < 2 for x in served):
                continue
            # arrived: stopped as close as it comes to anything (its following gap), not on the exact spot
            if d < v.spec["min_gap"] + 1.5 and v.v < 0.3 and v.plan and s in self.map.lanes[v.lane]["stops"]:
                v.dwell_until = self.world.t + self.rng.uniform(8.0, 20.0)
                served.add((v.lane, round(s)))
                return 0.0
            return max(0.0, d)
        return None

    def _pickup(self, v, dt):
        """Autos stop for passengers: about once every 3 minutes on the move, in the kerb lane away from
        junctions, an auto pulls to the kerb with its hazards on for 6 to 15 s (an ASSUMED rate, not measured)."""
        l = self.map.lanes[v.lane]
        if v.script or v.kind != "auto" or l["internal"] or self.world.style != "gurgaon" or v.v < 2.0 or \
                self.map.beside.get(v.lane, (None, None))[0] is not None or l["len"] - v.s < 30:
            return
        if self.rng.random() < dt * self.world.chaos / 180.0:
            v.script = {"speed": 0.0, "lat": v.max_lat, "pickup": self.world.t + self.rng.uniform(6.0, 15.0)}
            v.hazard = True

    def _desired(self, v):
        if v.script and "speed" in v.script:
            return v.script["speed"]
        l = self.map.lanes[v.lane]
        boost = 1.2 if self.world.t < v.boost_until else 1.0   # pushed along by our horn
        road = ROAD_KMH.get(l["cls"].replace("_link", ""), 30) * (0.85 if l["cls"].endswith("_link") else 1.0) / 3.6
        spurt = v.spurt if self.world.t < v.spurt_until else 1.0
        return min(v.spec["max_speed"], v.factor * v.wander * spurt * boost * road, self._curve_limit(v))

    def _bends(self, lane):
        """The bends along a lane: (distance along it, comfortable speed there), for corners slower than 90 km/h."""
        cache = self.__dict__.setdefault("_bend_cache", {})
        if lane not in cache:
            line, out, acc = self.map.lanes[lane]["line"], [], 0.0
            for (ax, ay), (bx, by), (cx, cy) in zip(line, line[1:], line[2:]):
                a, b, c = math.hypot(bx - ax, by - ay), math.hypot(cx - bx, cy - by), math.hypot(cx - ax, cy - ay)
                acc += a
                area = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2
                if area > 1e-6 and a > 0.3 and b > 0.3:
                    v = math.sqrt(2.5 * a * b * c / (4 * area))
                    if v < 25.0:
                        out.append((acc, max(3.0, v)))
            cache[lane] = out
        return cache[lane]

    def _curve_limit(self, v):
        """How fast v may go for the bends ahead on its road: slow at the bend, braking gently (1.5 m/s^2) before
        it. (A junction's lane is short: its tightest corner counts all along it.)"""
        l = self.map.lanes[v.lane]
        if l["internal"]:
            return l["vturn"]
        lim = 1e9
        for s_b, v_b in self._bends(v.lane):
            d = s_b - v.s
            if d < -3.0:
                continue
            if d > 150.0:
                break
            lim = min(lim, math.sqrt(v_b * v_b + 2 * 1.5 * max(0.0, d)))
        return lim

    def _drift(self, v, dt):
        """The slow wander of a driver's pace (a mean-reverting random walk), and its occasional spurts."""
        c = self.world.chaos
        v.wander += (1.0 - v.wander) * dt / WANDER_TAU + WANDER_SD * math.sqrt(c) * math.sqrt(2 * dt / WANDER_TAU) * self.rng.gauss(0, 1)
        if self.world.t >= v.spurt_until and self.rng.random() < dt * c / SPURT_EVERY:
            v.spurt = self.rng.uniform(1.15, 1.3) if self.rng.random() < 0.6 else self.rng.uniform(0.7, 0.85)
            v.spurt_until = self.world.t + self.rng.uniform(3.0, 8.0)

    def _home(self, v):
        """Where in its lane v rides: its usual line, or squeezed to the kerb to let us by."""
        if v.script and "lat" in v.script:
            return v.script["lat"]
        return v.max_lat if self.world.t < v.aside_until else v.home

    # our horn and headlight flash ----------------------------------------------------------------------------
    def ego_signal(self, kind):
        """Our car honks (kind "horn") or flashes its headlights ("dipper"). The road user ahead in our path
        may make way after a reaction time (see _make_way); with the horn, pedestrians near the road ahead
        wait before stepping out, and those on a crossing hurry."""
        e, t = self.world.ego, self.world.t
        c, n = math.cos(e.heading), math.sin(e.heading)
        reach = 40.0 if kind == "horn" else 70.0
        best = None
        for u in self.vehicles:
            dx, dy = u.x - e.x, u.y - e.y
            lon, lat = dx * c + dy * n, -dx * n + dy * c
            dh = abs(math.atan2(math.sin(u.heading - e.heading), math.cos(u.heading - e.heading)))
            if 0 < lon < reach and abs(lat) < (e.spec.width + u.width) / 2 + 0.6 and dh < math.radians(45) and (best is None or lon < best[0]):
                best = (lon, u)
        if best is not None and best[1].make_way is None:
            u = best[1]
            chance = YIELD[u.kind] * (0.8 if kind == "dipper" else 1.0)
            u.make_way = (t + self.rng.uniform(0.4, 1.2), self.rng.random() < chance, kind)
        if kind == "horn":
            for p in self.peds:
                dx, dy = p.x - e.x, p.y - e.y
                lon, lat = dx * c + dy * n, -dx * n + dy * c
                if -3 < lon < 30 and abs(lat) < 15:
                    p.hold_until = t + 3.0
                    if p.where[0] == "cr":
                        p.hurry_until = t + 4.0

    def _make_way(self, v, gap, lead_v, v0, by_lane):
        """React to our horn or flash: move into the lane beside if there is a gap, otherwise squeeze to the
        kerb and speed up a little (both for 8 s), or ignore it. Returns whether it changed lane."""
        _when, will, kind = v.make_way
        v.make_way = None
        if not will:
            self.world.note(f"the {NAME[v.kind]} ahead ignored our {kind}")
            return False
        t = self.world.t
        v.boost_until = t + 8.0
        if v.kind != "cycle" and self._change_lane(v, gap, lead_v, v0, by_lane, aside=True):
            self.world.note(f"the {NAME[v.kind]} ahead moved over for our {kind}")
            return True
        v.aside_until = t + 8.0
        self.world.note(f"the {NAME[v.kind]} ahead moved to the kerb for our {kind}")
        return False

    def _extend_plan(self, v, ahead=90.0):
        dist = self.map.lanes[v.plan[0]]["len"] - v.s
        for i in v.plan[1:]:
            dist += self.map.lanes[i]["len"]
        while dist < ahead:
            last = self.map.lanes[v.plan[-1]]
            if not last["next"]:
                return
            if last["internal"]:
                nxt = last["next"][0]
            else:
                choices = [(n, TURN_WEIGHT.get(d, 1.0) * self._share(self.map.lanes[self.map.lanes[n]["next"][0]] if self.map.lanes[n]["next"] else self.map.lanes[n]))
                           for n, d in zip(last["next"], last["next_dir"])
                           if not (v.kind == "bus" and self.map.lanes[n]["cls"] == "residential")]
                if not choices:
                    return
                tot = sum(w for _, w in choices)
                r = self.rng.uniform(0, tot)
                for n, w in choices:
                    r -= w
                    if r <= 0:
                        break
                nxt = n
            v.plan.append(nxt)
            dist += self.map.lanes[nxt]["len"]

    def _place(self, v):
        l = self.map.lanes[v.lane]
        x, y, h = _point(l["line"], v.s)
        tilt = max(-0.25, min(0.25, math.atan2(v.lat_v, max(v.v, 3.0))))   # nose turned the way it is moving across
        v.x, v.y, v.heading = x - math.sin(h) * v.lat, y + math.cos(h) * v.lat, h + tilt

    def _overlap_lat(self, a_lat, a_w, b_lat, b_w, margin=0.3):
        return abs(a_lat - b_lat) < (a_w + b_w) / 2 + margin

    def _obstacle(self, v, by_lane, lights):
        """Distance (bumper to bumper) and speed of the nearest thing v must not drive into, along its plan."""
        best_gap, best_v = 1e9, 0.0
        off = -v.s          # where each plan lane starts, measured from v
        look = max(40.0, v.v * 5)
        mine = v.lc_target if v.lc else v.lat   # mid lane change, it already keeps clear of the new lane's traffic
        # a two-wheeler squeezing past our car (events.py) rides the gap between two vehicles side by side, about
        # 1.7 m, with little to spare on either side (the body guard still keeps it from touching anyone)
        margin = 0.1 if v.script and "side" in v.script else 0.3
        v.zone_seen = False   # whether v is still waiting for an overlap stretch (see _wait_zone)
        for k, i in enumerate(v.plan):
            l = self.map.lanes[i]
            if off > look:
                break
            for u in by_lane.get(i, ()):
                if u is v or getattr(u, "owner", None) is v or (k == 0 and u.s <= v.s):
                    continue
                # a vehicle moving across into this lane counts from the moment it starts
                if not (getattr(u, "lc", None) and u.lane == i) and not self._overlap_lat(mine, v.width, u.lat, u.width, margin):
                    continue
                gap = off + u.s - (u.length + v.length) / 2
                if gap < best_gap:
                    best_gap, best_v = gap, u.v
                    v.why = "queue" if u.v < 0.1 else "following"
            for z, z0, z1 in l["zones"]:
                if self.map.zone_same[z]:
                    # two roads drawn over each other running the same way: follow whoever is ahead in v's way
                    # on either, as if they were one road
                    for lane_b, b0, b1 in self.map.zones[z]:
                        if lane_b == i:
                            continue
                        for u in by_lane.get(lane_b, ()):
                            if not (b0 - u.length <= u.s <= b1 + u.length):
                                continue
                            o = getattr(u, "owner", None) or u   # (one leaning over a line: the vehicle itself)
                            if o is v:
                                continue
                            # where it really is from v: the stretches map the two roads onto each other only
                            # roughly, and a vehicle reached through two of them could seem both ahead of v and
                            # behind it. Only one in v's way counts (beside it, as in another lane, it does not),
                            # and when each is ahead of the other (side by side, closing at an angle where a slip
                            # road joins), the one that appeared first goes.
                            dx, dy = o.x - v.x, o.y - v.y
                            rel = dx * math.cos(v.heading) + dy * math.sin(v.heading)
                            if abs(dy * math.cos(v.heading) - dx * math.sin(v.heading)) > (o.width + v.width) / 2 + 0.3:
                                continue
                            if rel > 0 and o.id > v.id and dx * math.cos(o.heading) + dy * math.sin(o.heading) < 0:
                                continue
                            if rel > 0 or (rel == 0 and o.id < v.id):
                                gap = rel - (o.length + v.length) / 2
                                if gap < best_gap:
                                    best_gap, best_v = max(0.0, gap), o.v
                                    v.why = "queue" if o.v < 0.1 else "following"
                    continue
                if z in v.stretches:
                    continue
                d0 = off + z0 - v.length / 2 - 0.5
                if d0 < -v.length:   # already in it (appeared there): fine
                    continue
                if d0 < v.v * 1.5 + 4:
                    if self._take_zone(v, z):
                        continue
                    v.zone_asked = self.world.t   # at the front and refused (see _take_zone's turns)
                if d0 < 15:
                    self._wait_zone(v, z)
                if d0 < best_gap:
                    best_gap, best_v = max(0.0, d0), 0.0
                    v.why = "overlap stretch"
            for cr, s0 in l["cross"]:
                if self.on_crossing.get(cr):
                    gap = off + s0 - self.map.crossings[cr]["w"] / 2 - 1.0 - v.length / 2
                    if -v.length < gap < best_gap:
                        best_gap, best_v = max(0.0, gap), 0.0
                        v.why = "pedestrian on crossing"
            if k + 1 < len(v.plan):
                nxt = self.map.lanes[v.plan[k + 1]]
                end = off + l["len"] - v.length / 2 - 0.3
                if nxt["internal"] and not l["internal"]:
                    blocked = False
                    if nxt["tl"] and not nxt.get("tl_inner") and not (v.script and v.script.get("red")):
                        light = Signals.light(lights[nxt["tl"][0]], [nxt["tl"][1]])
                        if light == "red" or (light == "yellow" and end > v.v * v.v / (2 * v.spec["decel"])):
                            blocked = True
                            if k == 0 and v.reserved == v.plan[1]:
                                v.reserved = None   # gives the junction back while it waits
                    if not blocked and v.reserved != v.plan[k + 1]:
                        # claim the junction once its entry is close, even from before a connecting road shorter
                        # than the vehicle (big crossings are several small junctions joined by such stubs: waiting
                        # to be on the stub first, a vehicle could stop short of it for good)
                        near = end < v.v * 1.5 + 4
                        self._last_refusal = "waiting to enter the junction"
                        if not (near and self._reserve(v, v.plan[k + 1], v.plan[k + 2] if k + 2 < len(v.plan) else None)):
                            blocked = True
                    if blocked and end < best_gap:
                        best_gap, best_v = max(0.0, end), 0.0
                        v.why = "signal" if nxt["tl"] and Signals.light(lights[nxt["tl"][0]], [nxt["tl"][1]]) != "green" else self._last_refusal
                        break
            elif not l["next"]:
                end = off + l["len"] - v.length / 2
                if end < best_gap:
                    best_gap, best_v = max(0.0, end), 0.0
                    v.why = "dead end"
            off += l["len"]
        if v.lc:   # until it is out of the lane it is leaving, whatever is ahead there still counts
            old, shift = v.lc
            s_old = getattr(v, "s_old", v.s)
            for u in by_lane.get(old, ()):
                if u is v or getattr(u, "owner", None) is v or u.s <= s_old:
                    continue
                if self._overlap_lat(v.lat + shift, v.width, u.lat, u.width):
                    gap = u.s - s_old - (u.length + v.length) / 2
                    if gap < best_gap:
                        best_gap, best_v = gap, u.v
                        v.why = "queue" if u.v < 0.1 else "following"
        if not v.zone_seen:
            v.zone_wait = None
        return best_gap, best_v

    def _wait_zone(self, v, z):
        """Note that v is waiting to enter overlap stretch z, keeping the time it started waiting."""
        if not (v.zone_wait and v.zone_wait[0] == z):
            v.zone_wait = (z, self.world.t)
        v.zone_seen = True

    def _idm(self, v, gap, lead_v, v0):
        """The Intelligent Driver Model's acceleration for v behind a leader gap metres ahead (1e9: none)."""
        sp = v.spec
        a, b = sp["accel"], sp["decel"]
        free = 1 - (v.v / max(0.5, v0)) ** 4
        if gap >= 1e8:
            return a * free
        s_star = sp["min_gap"] + max(0.0, v.v * sp["tau"] + v.v * (v.v - lead_v) / (2 * math.sqrt(a * b)))
        return a * (free - (s_star / max(gap, 0.05)) ** 2)

    def _neighbours(self, v, n, s_n, lat, by_lane):
        """Who v would have ahead of and behind it in lane n at distance s_n and offset lat: (gap, speed) for
        the one ahead, (gap, speed, vehicle) for the one behind (vehicle None for our car); gaps are bumper to
        bumper, 1e9 when nobody is there."""
        ln = self.map.lanes[n]
        lead, foll = (1e9, 0.0), (1e9, 0.0, None)
        for u in by_lane.get(n, ()):
            if u is v or getattr(u, "owner", None) is v or not self._overlap_lat(lat, v.width, u.lat, u.width):
                continue
            g = abs(u.s - s_n) - (u.length + v.length) / 2
            if u.s >= s_n and g < lead[0]:
                lead = (g, u.v)
            elif u.s < s_n and g < foll[0]:
                foll = (g, u.v, getattr(u, "owner", u))
        if lead[0] > 1e8:   # nobody left in this lane: the first vehicle on the lanes it leads to
            rest = ln["len"] - s_n
            for nx in ln["next"]:
                for u in by_lane.get(nx, ()):
                    g = rest + u.s - (u.length + v.length) / 2
                    if g < lead[0]:
                        lead = (g, u.v)
        e = self.world.ego
        if abs(e.x - v.x) < 60 and abs(e.y - v.y) < 60:   # our car, if it is in that lane
            dh = math.atan2(math.sin(e.heading - v.heading), math.cos(e.heading - v.heading))
            if abs(dh) < math.radians(45):
                s_e, lat_e = _project(ln["line"], e.x, e.y)
                if -5 < s_e < ln["len"] + 5 and abs(lat_e - lat) < (e.spec.width + v.width) / 2 + 0.3:
                    g = abs(s_e - s_n) - (e.spec.length + v.length) / 2
                    sp = max(0.0, e.speed * math.cos(dh))
                    if s_e >= s_n and g < lead[0]:
                        lead = (g, sp)
                    elif s_e < s_n and g < foll[0]:
                        foll = (g, sp, None)
        return lead, foll

    def _change_lane(self, v, gap, lead_v, v0, by_lane, aside=False, only=None):
        """Start a lane change when the lane beside is clearly quicker and its gaps allow. Returns whether it did.

        MOBIL (Kesting, Treiber and Helbing 2007): change when the acceleration v gains, minus politeness times
        the braking it causes the vehicle behind in the new lane, beats a threshold, and that vehicle never has
        to brake harder than v is willing to push it (2 to 4 m/s^2 here; MOBIL's usual safe limit is 4).
        With aside (making way for our car, or an event's cut-in) only the gaps count, and the kerb side is tried
        first; only limits it to one side (1 left, -1 right)."""
        l = self.map.lanes[v.lane]
        if v.kind == "cycle" or l["internal"] or v.lc or v.reserved is not None or v.stretches or v.crashed:
            return False
        if not aside and (self.world.t < v.lc_next or v.why not in ("queue", "following", "path crossed by our car")):
            return False   # only a vehicle ahead is worth going round (not a red light or a busy junction)
        left, right = self.map.beside.get(v.lane, (None, None))
        if (left is None and right is None) or l["len"] - v.s < max(20.0, v.v * 3.0) or v.s < v.length:
            return False   # no lane beside, or too close to the junction to finish the change
        a_cur = self._idm(v, gap, lead_v, v0)
        if not aside and self._idm(v, 1e9, 0.0, v0) - a_cur < v.gain:
            return False   # the vehicle ahead hardly holds it up
        best = None
        for side, n in ((1, left), (-1, right)):
            if n is None or (only is not None and side != only):
                continue
            ln = self.map.lanes[n]
            if not ln["next"] or (v.kind == "bus" and ln["cls"] == "residential"):
                continue
            s_n = _project(ln["line"], v.x, v.y)[0]
            if s_n < v.length or ln["len"] - s_n < max(20.0, v.v * 3.0):
                continue
            if any(a - 10 < s_n + 40 and b + 10 > s_n - 10 for _z, a, b in ln["zones"]):
                continue   # an overlap stretch nearby: stay out of it
            mx = max(0.0, ln["w"] / 2 - v.width / 2 - 0.25)
            lat = max(-mx, min(mx, v.home))
            (g_f, v_f), (g_r, v_r, f) = self._neighbours(v, n, s_n, lat, by_lane)
            # no gap to fit into (a little more room in front of our car, whose driver may not brake for it)
            if g_f < max(1.5, 0.5 * v.v) or g_r < max(2.0, 0.6 * v_r) + (1.0 if f is None else 0.0):
                continue
            if v_r > v.v and (v_r - v.v) ** 2 / (2 * max(0.5, g_r - 1.0)) > v.push:
                continue   # the one behind would have to brake harder than this driver dares make it
            if v.v > v_f and (v.v - v_f) ** 2 / (2 * max(0.5, g_f - 1.0)) > 3.0:
                continue   # it would have to brake hard itself for the one ahead there
            if v.v < 1.0 and (g_f < 12.0 or gap < 2.0):
                continue   # pulling out from a standstill needs a clear stretch and room to angle out
            loss = 0.0
            if f is not None:
                loss = max(0.0, f.acc - self._idm(f, g_r, v.v, max(0.5, self._desired(f))))
            gain = self._idm(v, g_f, v_f, v0) - a_cur - v.polite * loss
            if aside:
                gain = 2.0 + side   # any gap will do, the kerb side first
            if (aside or gain > v.gain) and (best is None or gain > best[0]):
                best = (gain, side, n, s_n, lat)
        if best is None:
            return False
        _gain, side, n, s_n, lat = best
        old, ln = v.lane, self.map.lanes[n]
        shift = side * (l["w"] + ln["w"]) / 2   # where the new lane's centre sits seen from the old one
        v.s_old = v.s
        v.lc, v.lc_target = (old, shift), lat
        v.lat -= shift
        v.lane, v.s, v.plan = n, s_n, [n]
        v.max_lat = mx = max(0.0, ln["w"] / 2 - v.width / 2 - 0.25)
        self._extend_plan(v)   # its next turn now comes from the new lane
        v.lc_next = self.world.t + self.rng.uniform(5.0, 10.0)
        v.pc, v._shared_key = None, None
        self._yields.pop(v.id, None)
        self.lane_changes += 1
        by_lane[old] = [_Ghost(v, old, v.s_old, v.lat + shift) if u is v else u for u in by_lane.get(old, ())]
        by_lane.setdefault(n, []).append(v)
        return True

    def _shift(self, v, by_lane, dt):
        """Move across during a lane change; it is over once the body has left the old lane."""
        old, shift = v.lc
        l = self.map.lanes[v.lane]
        rate = v.lat_rate * min(1.0, (v.v + 0.6) / 4.0)   # slower when barely moving: it has to angle out
        left = l["len"] - v.s
        if left < 15:   # finish before the junction
            rate = max(rate, abs(v.lat - v.lc_target) / max(0.3, left / max(v.v, 0.5)))
        step = rate * dt
        new = v.lat + max(-step, min(step, v.lc_target - v.lat))
        if self._room(v, new, by_lane):
            v.lat_v, v.lat = (new - v.lat) / dt, new
        if abs(v.lat + shift) >= self.map.lanes[old]["w"] / 2 + v.width / 2 + 0.1 or abs(v.lat - v.lc_target) < 0.05:
            v.lc = None

    def _ego_waiting(self):
        """Whether our car is stopped for a reason: a red or yellow light or a vehicle just ahead of it (worked
        out once a step, only when someone is held up by us)."""
        if getattr(self, "_wait_step", None) != self._steps:
            gap, lead = self.world.gap_ahead(look=15.0)
            self._wait_step, self._wait_val = self._steps, gap < 12.0 and lead < 1.0
        return self._wait_val

    def _wrong_way(self, v, by_lane, dt):
        """A vehicle going against the traffic, backwards along its lane near the edge. It slows for anything
        coming at it in its way, and once something is under 2.5 s away it pulls in to the edge and stops
        until the way is clear (our car included)."""
        l = self.map.lanes[v.lane]
        speed = v.script.get("ww_speed", 5.0)
        ahead = 1e9
        lanes = [(v.lane, 0.0)]
        if abs(v.lat) + v.width / 2 > l["w"] / 2:   # leaning over a line: the lane beside counts too
            side = self.map.beside.get(v.lane, (None, None))[0 if v.lat > 0 else 1]
            if side is not None:
                lanes.append((side, l["w"] if v.lat > 0 else -l["w"]))
        for lane, shift in lanes:
            for u in by_lane.get(lane, ()):   # things coming towards it: lower s, overlapping sideways
                if u is v or getattr(u, "owner", None) is v or not (v.s - 40 < u.s < v.s):
                    continue
                if self._overlap_lat(v.lat - shift, v.width, u.lat, u.width, 0.2):
                    ahead = min(ahead, (v.s - u.s - (u.length + v.length) / 2) / max(1.0, abs(v.v) + max(0.0, u.v)))
        e = self.world.ego
        if abs(e.x - v.x) < 45 and abs(e.y - v.y) < 45:
            s_e, lat_e = _project(l["line"], e.x, e.y)
            if v.s - 45 < s_e < v.s and self._overlap_lat(v.lat, v.width, lat_e, e.spec.width, 0.2):
                ahead = min(ahead, (v.s - s_e - (e.spec.length + v.length) / 2) / max(1.0, abs(v.v) + max(0.0, e.speed)))
        if ahead < 2.5:   # pull in to the edge (mostly onto the shoulder) and wait
            v.script["lat"] = v.script.get("aside", v.max_lat) * (1 if v.script.get("edge", 1) > 0 else -1)
            want = 0.0
        else:
            want = speed
        cur = -v.v
        cur = min(want, cur + 2.0 * dt) if cur < want else max(want, cur - 5.0 * dt)
        v.v, v.acc = -cur, 0.0
        target = v.script.get("lat", v.lat)
        step = 0.8 * dt
        new = v.lat + max(-step, min(step, target - v.lat))
        v.lat_v, v.lat = -(new - v.lat) / dt, new
        v.s -= cur * dt
        if v.s < 0:
            back = self._prev().get(v.lane)
            if back:   # back the way a car going straight on would have come
                back = [b for b in back if self.map.lanes[b]["internal"] and self.map.lanes[b]["dir"] == "s"] or back
            if not back:
                v.plan = []   # nowhere to go: taken away (off screen) by _manage
                v.s = 0.0
            else:
                v.lane = back[0]
                v.s += self.map.lanes[v.lane]["len"]
                v.plan = [v.lane]
        self._place(v)
        v.heading += math.pi

    def _prev(self):
        cache = self.__dict__.get("_prev_lanes")
        if cache is None:
            cache = {}
            for i, l in enumerate(self.map.lanes):
                for n in l["next"]:
                    cache.setdefault(n, []).append(i)
            self._prev_lanes = cache
        return cache

    def rejoin_footpath(self, p):
        """A pedestrian back from a free walk steps onto the nearest footpath."""
        best = None
        for j in self.map.grids["sidewalks"].near(p.x - 15, p.y - 15, p.x + 15, p.y + 15):
            if j in self._bad_sidewalks:
                continue
            u, lat = _project(self.map.sidewalks[j]["line"], p.x, p.y)
            if best is None or abs(lat) < best[0]:
                best = (abs(lat), j, u)
        if best is None:
            return False
        p.where, p.u, p.dir, p.script = ("sw", best[1]), best[2], self.rng.choice((1, -1)), None
        self._place_ped(p)
        return True

    def spawn_at(self, kind, lane, s, lat=None, speed=None):
        """Put a vehicle of a kind at distance s along a lane (offset lat), if the spot is free and out of view.
        Returns it, or None."""
        l = self.map.lanes[lane]
        spec = self.types.get(kind) or ANIMALS[kind]
        if not (spec["length"] / 2 < s < l["len"] - spec["length"] / 2):
            return None
        v = Vehicle(self.next_id, kind, spec, lane, s, self.rng, self.map)
        if lat is not None:
            v.lat = v.home = lat
        v.v = self._desired(v) if speed is None else speed
        self._place(v)
        if self.visible(v.x, v.y) or not self._clear(v.x, v.y, spec["length"], l["level"], room=3.0):
            return None
        v.manner = (v.polite, v.gain, v.push)
        self._temper(v)
        self.next_id += 1
        self.vehicles.append(v)
        return v

    def _path_conflict(self, v, near):
        """Distance to the first road user (or our car) that v's own path runs into within a few seconds, and
        its speed. A safety net under the lane rules: it catches roads that cross on the map without a junction,
        a pedestrian off the footpath, and our car wherever it is. When two vehicles wait for each other, the one
        that appeared first goes."""
        D = max(8.0, v.v * 2.5 + 6.0)
        # road users the other rules already keep in order are left to them: those on our own plan (the lane
        # rules), on junction lanes that conflict with ours (one user per conflict), on lanes sharing an overlap
        # stretch with us (the zone rule) and in the lanes beside ours. What remains is our car, pedestrians,
        # vehicles that left our lane on a different turn, and vehicles on lanes the map gives no relation to ours.
        # Right in front of us (the first 3 m) only our own plan and the lanes beside it are left to the lane
        # rules (two-wheelers pass close by those on purpose): any other body there is never driven into, whatever
        # lane it is on (at small junctions the map draws lanes over each other's ends).
        key = tuple(v.plan[:4])
        if getattr(v, "_shared_key", None) != key:
            shared = set(key)
            for i in key:
                li = self.map.lanes[i]
                for z, _a, _b in li["zones"]:
                    shared.update(p[0] for p in self.map.zones[z])
                if li["internal"]:
                    shared.update(li["foes"])   # one user per conflict: the junction rule orders these
                shared.update(x for x in self.map.beside.get(i, ()) if x is not None)
            v._shared_key, v._shared = key, shared
            v._own = set(key) | {x for i in key for x in self.map.beside.get(i, ()) if x is not None}
        shared, own = v._shared, v._own
        k, lane, s0, d = 0, v.lane, v.s, 0.0
        level = self.map.lanes[v.lane]["level"]
        reach = v.width / 2 + 0.25
        while d < D:
            d += 1.5
            s0 += 1.5
            while s0 > self.map.lanes[lane]["len"]:
                s0 -= self.map.lanes[lane]["len"]; k += 1
                if k >= len(v.plan):
                    return 1e9, 0.0
                lane = v.plan[k]
            x, y, h = _point(self.map.lanes[lane]["line"], s0)
            x, y = x - math.sin(h) * v.lat, y + math.cos(h) * v.lat
            over = self.map.lanes[lane]["over"]
            for u in near.get((int(x // 10), int(y // 10)), ()):
                if getattr(u, "u", None) is v or u.level != level:
                    continue
                if over and getattr(u, "lane", None) in over:   # on a flyover or underpass across our path
                    continue
                if getattr(u, "lane", None) in shared and u.uid != -1 and (d > v.length / 2 + 3.0 or u.lane in own):
                    continue
                if u.uid >= 0 and self._yields.get(u.uid) == v.id and v.id < u.uid:
                    continue
                c, n = math.cos(u.heading), math.sin(u.heading)
                ax, ay = u.x - c * u.length / 2, u.y - n * u.length / 2
                bx, by = u.x + c * u.length / 2, u.y + n * u.length / 2
                wx, wy = bx - ax, by - ay
                t = max(0.0, min(1.0, ((x - ax) * wx + (y - ay) * wy) / (wx * wx + wy * wy or 1e-9)))
                if math.hypot(ax + t * wx - x, ay + t * wy - y) < reach + u.width / 2:
                    # already alongside another vehicle (its body overlaps where we are now): the lane rules
                    # handle it; our car and pedestrians are never passed over this way
                    if d <= v.length / 2 + 0.5 and u.uid != -1 and not isinstance(getattr(u, "u", None), Pedestrian):
                        continue
                    self._yields[v.id] = u.uid
                    return max(0.0, d - v.length / 2 - 0.8), max(0.0, u.v * math.cos(u.heading - h))
        self._yields.pop(v.id, None)
        return 1e9, 0.0

    def _junction_lanes(self, junction):
        cache = self.__dict__.setdefault("_jl", {})
        if not cache:
            for i, l in enumerate(self.map.lanes):
                if l["internal"]:
                    cache.setdefault(l["junction"], []).append(i)
        return cache.get(junction, ())

    def _zone_group(self, z, lane):
        """z and every overlap stretch that runs into it along lane: a vehicle passes through them in one go, so
        it takes them together (one at a time, four vehicles on a grid of two lanes over two lanes each held one
        and waited for the next, for good)."""
        spans = {k: (a, b) for k, a, b in self.map.lanes[lane]["zones"] if not self.map.zone_same[k]}
        group, todo = {z}, [z]
        while todo:
            a, b = spans[todo.pop()]
            for k, (c, d) in spans.items():
                if k not in group and c < b + 8.0 and d > a - 8.0:
                    group.add(k)
                    todo.append(k)
        return group

    def _take_zone(self, v, z):
        """Claim an overlap stretch, with those it runs into (_zone_group), all or none: nobody on another part
        of them holds them or is inside them. Vehicles on the same part share them and simply follow each other
        (a queue at a red light inside one stays a queue)."""
        parts = {lane for lane, _a, _b in self.map.zones[z]}
        mine = next((i for i in [v.lane] + v.plan if i in parts), None)
        group = self._zone_group(z, mine) - set(v.stretches)
        t = self.world.t
        since = v.zone_wait[1] if v.zone_wait and v.zone_wait[0] in group else t
        for k in group:
            parts = {lane for lane, _a, _b in self.map.zones[k]}
            # someone behind v on its own part already holds it: it cannot empty for the other part until v
            # goes, so v goes too (waiting its turn here was a circular wait)
            held_behind = any(u is not v and u.stretches.get(k) == mine and not (u.lane == v.lane and u.s > v.s) and
                              u.lane not in v.plan[1:] for u in self.vehicles)
            for u in self.vehicles:
                if u is v:
                    continue
                if k in u.stretches and u.stretches[k] != mine:
                    return False
                for lane, a, b in self.map.zones[k]:
                    if lane != mine and u.lane == lane and a - u.length / 2 <= u.s <= b + u.length / 2:
                        return False
                # take turns: someone from another part has been waiting longer, so let the stretch empty for
                # them. Only someone at the front, who asked just now: a vehicle queued further back keeps its place
                # in time but cannot go yet, and honouring it held up the very queue in front of it.
                if not held_behind and u.zone_wait and u.zone_wait[0] in group and u.zone_wait[1] < since and \
                        u.zone_asked >= t - 0.6 and next((i for i in [u.lane] + u.plan if i in parts), None) != mine:
                    return False
        v.stretches.update({k: mine for k in group})
        v.zone_wait = None
        return True

    def _hash(self):
        """Everyone (and our car) by 10 m cell, each entered in every cell its body reaches."""
        class Body:
            __slots__ = ("uid", "x", "y", "heading", "length", "width", "v", "level", "lane")
        out = {}
        e = self.world.ego
        items = []
        for u in self.vehicles + self.peds:
            b = Body(); b.uid, b.x, b.y, b.heading, b.length, b.width = u.id, u.x, u.y, u.heading, u.length, u.width
            b.v = getattr(u, "v", 0.0) if u.kind != "pedestrian" else 0.0
            b.level = level_of(self.map, u)
            items.append((u, b))
        me = Body(); me.uid, me.x, me.y, me.heading, me.length, me.width = -1, e.x, e.y, e.heading, e.spec.length, e.spec.width
        me.v, me.level, me.lane = max(0.0, e.speed), self.world.ego_level, self.world.status.get("lane")
        items.append((None, me))
        for u, b in items:
            r = b.length / 2 + 3
            for cx in range(int((b.x - r) // 10), int((b.x + r) // 10) + 1):
                for cy in range(int((b.y - r) // 10), int((b.y + r) // 10) + 1):
                    out.setdefault((cx, cy), []).append(b if u is None else _Proxy(u, b))
        return out

    def _conflict_end(self, lane, foe):
        """How far along junction lane foe its path stops crossing or touching lane's (lanes 1.1 m either side of
        their centre lines, as _foes_and_crossings finds foes); the foe's whole length for a merge."""
        cache = self.__dict__.setdefault("_cend", {})
        key = (lane, foe)
        if key not in cache:
            from shapely.geometry import LineString, Point
            a, b = LineString(self.map.lanes[lane]["line"]), LineString(self.map.lanes[foe]["line"])
            hit = b.intersection(a.buffer(2.2))
            pts = [] if hit.is_empty else [c for g in getattr(hit, "geoms", [hit]) for c in g.coords]
            cache[key] = max((b.project(Point(c)) for c in pts), default=self.map.lanes[foe]["len"])
        return cache[key]

    def _reserve(self, v, lane, after):
        """Claim a junction lane: no vehicle on (or holding) a lane that crosses or merges with it, room on the
        road beyond, and our car not in the way."""
        l = self.map.lanes[lane]
        foes = set(l["foes"])
        t = self.world.t
        if getattr(v, "jwait", None) is None or v.jwait[0] != lane:
            v.jwait = (lane, t)   # waiting for this junction lane since
        v.jwait_seen = t          # (still asking: a turn is kept only by one still trying)
        self._last_refusal = "junction busy"
        inside = self.map.lanes[v.lane]["internal"] and t - v.jwait[1] > 3.0
        for u in self.vehicles:
            if u is v:
                continue
            if u.lane in foes and u.s - u.length / 2 > self._conflict_end(lane, u.lane):
                continue   # already through the crossing area, on its way out of the junction (SUMO's junction model too)
            if u.reserved in foes or u.lane in foes:
                # two stopped inside a junction, each on a lane the other needs: the one that appeared first goes
                if inside and u.lane in foes and self.map.lanes[u.lane]["internal"] and u.v < 0.5 and \
                        getattr(u, "jwait", None) is not None and v.id < u.id:
                    continue
                v.jwait_why = "junction busy"
                return False
            # take turns: someone still waiting for a crossing lane has waited longer, so let them go first
            # (without this a busy junction can keep one driver waiting for minutes)
            # (only one still asking and refused because the junction is busy keeps its turn: not one held by a red
            # light, a full road beyond, or anything else, which would hold everyone up with it)
            jw = getattr(u, "jwait", None)
            if jw is not None and jw[0] in foes and jw[1] < v.jwait[1] - 0.5 and u.reserved is None and \
                    u.v < 1.0 and jw[0] in u.plan[:3] and t - getattr(u, "jwait_seen", -9) < 0.6 and \
                    getattr(u, "jwait_why", "") == "junction busy":
                self._last_refusal = "letting another go first"
                v.jwait_why = self._last_refusal
                return False
        self._last_refusal = v.jwait_why = "no room beyond junction"
        if after is not None:
            room = v.length + v.spec["min_gap"] + 1.0
            for u in self.vehicles:
                if u is not v and u.lane == after and u.s - u.length / 2 < room and self._overlap_lat(v.lat, v.width, u.lat, u.width):
                    return False
        self._last_refusal = v.jwait_why = "our car in junction"
        e = self.world.ego
        if math.hypot(e.x - v.x, e.y - v.y) < l["len"] + 15:
            from shapely.geometry import LineString, Point
            if LineString(l["line"]).distance(Point(e.x, e.y)) < 3.0:
                dh = abs(math.atan2(math.sin(e.heading - v.heading), math.cos(e.heading - v.heading)))
                if dh > math.radians(30) or e.speed < 0.5:
                    return False
        mine = self.world.status.get("lane")   # our car on, or about to enter, a crossing or merging lane
        if mine is not None:
            if mine in foes:
                return False
            if e.speed > 1.5:   # any lane it can reach within 3 s, through a junction and a short stub too
                ml, reach = self.map.lanes[mine], max(6.0, e.speed * 3.0)
                todo = [(n, ml["len"] - _project(ml["line"], e.x, e.y)[0]) for n in ml["next"]]
                while todo:
                    n, d = todo.pop()
                    if d >= reach:
                        continue
                    if n in foes:
                        return False
                    todo.extend((k, d + self.map.lanes[n]["len"]) for k in self.map.lanes[n]["next"])
        if after is not None:   # an overlap stretch just past the junction: hold it before entering, so
            for z, a, _b in self.map.lanes[after]["zones"]:   # nobody stops inside the junction waiting for it
                if a < 15 and not self.map.zone_same[z] and z not in v.stretches:
                    self._last_refusal = "overlap stretch beyond junction"
                    if not self._take_zone(v, z):
                        self._wait_zone(v, z)
                        v.zone_asked = self.world.t
                        return False
        v.reserved, v.jwait = lane, None
        return True

    def _lateral(self, v, by_lane, dt):
        """Two-wheelers and cycles: squeeze past a slower vehicle ahead when there is room beside it.

        A two-wheeler may ride over the lane line (up to 0.3 m past it) when the next lane is clear beside it,
        and comes back inside its lane before a junction."""
        l = self.map.lanes[v.lane]
        generic = self.world.style == "generic"   # (lane discipline: nobody squeezes past)
        if v.kind not in ("two_wheeler", "cycle") or l["internal"] or generic:
            home = self._home(v)
            if (v.kind not in ("two_wheeler", "cycle") or generic) and abs(v.lat - home) > 0.01:   # settle back to its line
                step = 0.6 * dt
                new = v.lat + max(-step, min(step, home - v.lat))
                if self._room(v, new, by_lane):
                    v.lat_v, v.lat = (new - v.lat) / dt, new
            return
        inner = max(0.0, l["w"] / 2 - v.width / 2 - 0.25)
        outer = l["w"] / 2 - v.width / 2 + 0.3 if v.kind == "two_wheeler" else inner
        left, right = self.map.beside.get(v.lane, (None, None))   # over a line only where another lane lies
        target = max(-(outer if right is not None else inner), min(outer if left is not None else inner, self._home(v)))
        lead = None
        for u in by_lane.get(v.lane, ()):
            if u is not v and 0 < u.s - v.s < 14 and self._overlap_lat(v.lat, v.width, u.lat, u.width) and u.v < v.v + 1:
                if lead is None or u.s < lead.s:
                    lead = u
        e = self.world.ego   # our car is squeezed past like anyone else
        if abs(e.x - v.x) < 20 and abs(e.y - v.y) < 20 and \
                abs(math.atan2(math.sin(e.heading - v.heading), math.cos(e.heading - v.heading))) < math.radians(45):
            s_e, lat_e = _project(l["line"], e.x, e.y)
            if 0 < s_e - v.s < 14 and self._overlap_lat(v.lat, v.width, lat_e, e.spec.width) and e.speed < v.v + 1 \
                    and (lead is None or s_e < lead.s):
                lead = _Body(s_e, lat_e, e.spec.length, e.spec.width, max(0.0, e.speed))
        near_end = l["len"] - v.s < 12
        if lead is not None and v.kind == "two_wheeler" and not near_end:
            for side in ((v.script["side"],) if v.script and "side" in v.script else (1, -1)):
                t = lead.lat + side * ((lead.width + v.width) / 2 + (0.5 if isinstance(lead, _Body) else 0.25))
                if abs(t) <= outer:
                    target = t
                    break
        step = min(0.8, 0.3 + 0.2 * v.v) * dt   # feet down and shuffling at a standstill, quicker on the move
        new = v.lat + max(-step, min(step, target - v.lat))
        if near_end:
            new = max(-max(inner, abs(v.lat) - step), min(max(inner, abs(v.lat) - step), new))
        if not self._room(v, new, by_lane):
            return
        v.lat_v, v.lat = (new - v.lat) / dt, new

    def _would_touch(self, v, ds, cells):
        """The vehicle whose body moving v ds further along its plan would touch (one it is not touching already),
        or None."""
        lane, s0, k = v.lane, v.s + ds, 0
        while s0 > self.map.lanes[lane]["len"] and k + 1 < len(v.plan):
            s0 -= self.map.lanes[lane]["len"]; k += 1; lane = v.plan[k]
        x, y, h = _point(self.map.lanes[lane]["line"], min(s0, self.map.lanes[lane]["len"]))
        new = (x - math.sin(h) * v.lat, y + math.cos(h) * v.lat, h, v.length, v.width)
        now = (v.x, v.y, v.heading, v.length, v.width)
        level = self.map.lanes[v.lane]["level"]
        over = set(self.map.lanes[v.lane]["over"]) | set(self.map.lanes[lane]["over"])   # (passing over or under)
        cx, cy = int(v.x // 8), int(v.y // 8)
        for i in (cx - 1, cx, cx + 1):
            for j in (cy - 1, cy, cy + 1):
                for u in cells.get((i, j), ()):
                    if u is v or u.crashed or self.map.lanes[u.lane]["level"] != level or u.lane in over:
                        continue
                    body = (u.x, u.y, u.heading, u.length, u.width)
                    if _boxes_touch(new, body) and not _boxes_touch(now, body):
                        return u
        return None

    def _room(self, v, lat, by_lane):
        """Whether v fits at lateral position lat without touching anyone in its lane or the lanes beside."""
        lanes = [(v.lane, 0.0)]
        left, right = self.map.beside.get(v.lane, (None, None))
        w = self.map.lanes[v.lane]["w"]
        if left is not None:
            lanes.append((left, -w))      # a lane to the left sits at +w: our position seen from it is lat - w
        if right is not None:
            lanes.append((right, w))
        for lane, shift in lanes:
            for u in by_lane.get(lane, ()):
                if u is v or getattr(u, "owner", None) is v or abs(u.s - v.s) > (u.length + v.length) / 2 + 0.5:
                    continue
                if self._overlap_lat(lat + shift, v.width, u.lat, u.width, 0.2) and \
                        not self._overlap_lat(v.lat + shift, v.width, u.lat, u.width, 0.2):
                    return False
        # anyone else close by: on another road drawn across or alongside this one (the map has some)
        mine = {lane for lane, _ in lanes}
        x, y, h = _point(self.map.lanes[v.lane]["line"], v.s)
        n_x, n_y, level = -math.sin(h), math.cos(h), self.map.lanes[v.lane]["level"]
        new = (x + n_x * lat, y + n_y * lat, h, v.length + 0.2, v.width + 0.2)
        now = (x + n_x * v.lat, y + n_y * v.lat, h, v.length + 0.2, v.width + 0.2)
        for u in self.vehicles:
            if u is v or u.lane in mine or abs(u.x - x) > 8 or abs(u.y - y) > 8 or self.map.lanes[u.lane]["level"] != level \
                    or u.lane in self.map.lanes[v.lane]["over"]:
                continue
            body = (u.x, u.y, u.heading, u.length, u.width)
            if _boxes_touch(new, body) and not _boxes_touch(now, body):
                return False
        return not (self._hits_ego(v, lat) and not self._hits_ego(v, v.lat))

    def _hits_ego(self, v, lat, margin=None):
        """Whether v, moved to lateral position lat, would come within margin of our car. A two-wheeler or cycle
        moving across turns its nose up to 14 degrees, swinging its front corner about 0.25 m wider: it keeps
        0.45 m."""
        if margin is None:
            margin = 0.45 if v.kind in ("two_wheeler", "cycle") else 0.25
        e = self.world.ego
        if abs(e.x - v.x) > 10 or abs(e.y - v.y) > 10:
            return False
        x, y, h = _point(self.map.lanes[v.lane]["line"], v.s)
        x, y = x - math.sin(h) * lat, y + math.cos(h) * lat
        return _boxes_touch((x, y, h, v.length + 2 * margin, v.width + 2 * margin), (e.x, e.y, e.heading, e.spec.length, e.spec.width))

    def _step_vehicles(self, dt):
        lights = [self.world.signals.state(k, self.world.t) for k in range(len(self.map.signals))]
        by_lane = {}
        for v in self.vehicles:
            by_lane.setdefault(v.lane, []).append(v)
            if v.lc:
                old, shift = v.lc
                v.s_old = _project(self.map.lanes[old]["line"], v.x, v.y)[0]
                by_lane.setdefault(old, []).append(_Ghost(v, old, v.s_old, v.lat + shift))
            elif not self.map.lanes[v.lane]["internal"] and abs(v.lat) + v.width / 2 > self.map.lanes[v.lane]["w"] / 2:
                # riding over a lane line: the lane beside sees the part of it that sticks out
                w = self.map.lanes[v.lane]["w"]
                side = self.map.beside.get(v.lane, (None, None))[0 if v.lat > 0 else 1]
                if side is not None:
                    shift = w if v.lat > 0 else -w   # (a lane to the left sits at +w)
                    by_lane.setdefault(side, []).append(_Ghost(v, side, v.s * self.map.lanes[side]["len"] / max(1e-6, self.map.lanes[v.lane]["len"]), v.lat - shift))
        self._steps += 1
        near = self._hash() if self._steps % 2 == 0 else None
        cells = {}
        for u in self.vehicles:
            cells.setdefault((int(u.x // 8), int(u.y // 8)), []).append(u)
        for v in list(self.vehicles):   # (a copy: a stuck wrong-way vehicle may leave during the loop)
            if v.crashed:
                if self.world.t - getattr(v, "crashed_at", 0.0) < 5.0:
                    continue
                v.crashed = False   # shaken, but drives on
            if v.script and v.script.get("parked"):
                v.v, v.lat_v = 0.0, 0.0
                if self.world.t >= v.script.get("until", 1e18):   # a stop for passengers over: drive on
                    v.script, v.hazard = None, False
                if v.script and self.world.t >= v.script.get("leave_at", 1e18):   # a cow done standing walks off
                    goal = v.script["leave_lat"]
                    v.lat += max(-0.5 * dt, min(0.5 * dt, goal - v.lat))
                    v.script["turn"] = 1.3 if goal > v.lat else -1.3   # (facing the way it walks)
                self._place(v)
                if v.script and "pose" in v.script:   # moved by an event off its lane (a U-turn)
                    v.x, v.y, v.heading = v.script["pose"]
                elif v.script and "turn" in v.script:   # standing at an angle to the lane (a cow), or turning round
                    if "spin" in v.script:
                        v.script["turn"] = max(0.0, v.script["turn"] - v.script["spin"] * dt)
                    v.heading += v.script["turn"]
                if v.script and "leave_at" in v.script and abs(v.script["leave_lat"] - v.lat) < 0.05 and \
                        not self.visible(v.x, v.y):
                    self.vehicles.remove(v)   # off the road and out of sight
                continue
            if v.script and v.script.get("wrong_way"):
                self._wrong_way(v, by_lane, dt)
                # pulled in and stopped for long (a queue at a light coming at it, or face to face with a cycle
                # that waits for it in turn): it gives up, leaving the area out of sight, else turning round
                # (over 1.5 s) to go with the traffic
                v.script["still"] = v.script.get("still", 0.0) + dt if v.v < 0.1 else 0.0
                if v.script["still"] > 10.0 and not self.visible(v.x, v.y, margin=6):
                    self.vehicles.remove(v)
                elif v.script["still"] > 8.0:
                    v.v, v.script = 0.0, {"parked": True, "turn": math.pi, "spin": math.pi / 1.5, "until": self.world.t + 1.5}
                continue
            l_now = self.map.lanes[v.lane]
            if not l_now["next"] and v.s > l_now["len"] - 10 and not self.visible(v.x, v.y, margin=6):
                self.vehicles.remove(v)   # the road runs on past the map's edge: it drives off the map
                continue
            self._pickup(v, dt)
            if v.script and "pickup" in v.script:   # pulled up (or, kept from the kerb, stopped where it is after 4 s)
                v.script.setdefault("since", self.world.t)
                if v.v < 0.6 and (abs(v.lat - v.max_lat) < 0.3 or self.world.t - v.script["since"] > 4.0):   # (0.5 m/s: the least desired speed)
                    v.script = {"parked": True, "until": v.script["pickup"]}
            self._extend_plan(v)
            self._drift(v, dt)
            v.lat_v = 0.0
            for attempt in (0, 1):
                v.why = ""
                gap, lead_v = self._obstacle(v, by_lane, lights)
                if near is not None:
                    v.pc = self._path_conflict(v, near)
                elif getattr(v, "pc", None):
                    v.pc = (v.pc[0] - v.v * dt, v.pc[1]) if v.pc[0] < 1e8 else v.pc
                if getattr(v, "pc", None) and v.pc[0] < gap:
                    gap, lead_v = v.pc
                    v.why = "path crossed by " + ("our car" if self._yields.get(v.id) == -1 else "another road user")
                sp = v.spec
                v0 = max(0.5, self._desired(v))
                if len(v.plan) > 1:   # slow in time for a bend in the next lane
                    nxt = self.map.lanes[v.plan[1]]
                    d = self.map.lanes[v.lane]["len"] - v.s
                    v0 = min(v0, math.sqrt(nxt["vturn"] ** 2 + 2 * sp["decel"] * 0.6 * max(0.0, d)))
                v0 = self._road_features(v, v0)
                stop = self._bus_stop(v) if v.kind == "bus" else None   # a bus stops at the stop in its lane
                if stop is not None and stop < gap:
                    gap, lead_v, v.why = stop, 0.0, "bus stop"
                if attempt:
                    break
                if v.script and v.script.get("cut_in") and self._change_lane(v, gap, lead_v, v0, by_lane, aside=True, only=v.script["cut_in"]):
                    v.script.pop("cut_in")
                    continue   # moved across: look again from the new lane
                if v.make_way is not None and self.world.t >= v.make_way[0]:
                    if self._make_way(v, gap, lead_v, v0, by_lane):
                        continue   # moved over: look again from the new lane
                    break
                # twice a second, a driver held up by the vehicle ahead looks at the lanes beside
                if (self._steps + v.id) % 30 or not self._change_lane(v, gap, lead_v, v0, by_lane):
                    break
            if v.why == "path crossed by our car" and v.v < 1.0 and gap < 8.0:   # held up by us: honks
                v.held += dt
                # ...but not while we wait at a red light or behind a queue: only when we hold everyone up
                if v.held > 1.5 / self.world.chaos and self.world.t >= v.next_honk and not self._ego_waiting() and self.world.style == "gurgaon":
                    v.honk_t, v.next_honk = self.world.t, self.world.t + self.rng.uniform(2.5, 6.0) / self.world.chaos
                    self.world.note(f"{a_or_an(NAME[v.kind])} honked at us")
            else:
                v.held = 0.0
            a = sp["accel"]
            acc = max(-sp["emergency"], min(a, self._idm(v, gap, lead_v, v0)))
            if v.script and "brake" in v.script:   # a hard stop on cue: brakes down to a speed, then waits
                decel, floor, hold = v.script["brake"]
                if v.v > floor + 0.05:
                    acc = min(acc, -decel)
                elif self.world.t < v.script.setdefault("held_until", self.world.t + hold):
                    acc = min(acc, 0.0) if floor > 0.1 else -decel
                else:
                    v.script.pop("brake")
            v.acc = acc
            nv = max(0.0, v.v + acc * dt)
            if gap < 1e8:
                nv = min(nv, max(0.0, (gap - 0.05) / dt))   # never into the one ahead
            v.v = nv
            red = gap < 1e8 and lead_v == 0.0 and len(v.plan) > 1 and self.map.lanes[v.plan[1]]["tl"] is not None and \
                Signals.light(lights[self.map.lanes[v.plan[1]]["tl"][0]], [self.map.lanes[v.plan[1]]["tl"][1]]) == "red"
            # only the front of a line counts as stuck (waiting behind a stopped vehicle, or at a red, is not)
            head = getattr(v, "why", "") not in ("queue", "following", "signal")
            v.stopped_for = v.stopped_for + dt if (nv < 0.1 and not red and head) else 0.0
            if v.lc:
                self._shift(v, by_lane, dt)
            else:
                self._lateral(v, by_lane, dt)
            u = self._would_touch(v, v.v * dt, cells) if v.v > 0 else None
            if u is not None and not (getattr(u, "guard", None) == v.id and v.id < u.id):
                v.v = 0.0   # the last guard: never into anyone's body (the lane rules measure gaps along the
                            # lane's centre, which runs short on the inside of a tight bend); when two hold each
                            # other up this way, the one that appeared first goes
                v.guard = u.id
            else:
                v.guard = None
            v.s += v.v * dt
            for z in list(v.stretches):   # give each overlap stretch back once past our part of it
                part = [p for p in self.map.zones[z] if p[0] == v.lane]
                if not part and self.map.lanes[v.lane]["internal"] is False and all(p[0] not in v.plan for p in self.map.zones[z]):
                    del v.stretches[z]
                elif part and v.s - v.length / 2 > part[0][2]:
                    del v.stretches[z]
            while v.plan and v.s > self.map.lanes[v.lane]["len"]:
                v.s -= self.map.lanes[v.lane]["len"]
                if v.reserved == v.lane:
                    v.reserved = None
                v.plan.pop(0)
                v.lc = None
                if not v.plan:
                    break
                v.lane = v.plan[0]
                nw = self.map.lanes[v.lane]["w"]
                v.max_lat = max(0.0, nw / 2 - v.width / 2 - 0.25)   # (a two-wheeler keeps its line into a junction)
            if v.plan:
                self._place(v)

    # pedestrians -----------------------------------------------------------------------------------------
    def _place_ped(self, p):
        kind, j = p.where
        if kind == "wa":
            (ax, ay), (bx, by) = p.path
            L = math.hypot(bx - ax, by - ay) or 1e-6
            u = min(p.u, L) / L
            p.x, p.y, p.heading = ax + (bx - ax) * u, ay + (by - ay) * u, math.atan2(by - ay, bx - ax)
            return
        line = self.map.sidewalks[j]["line"] if kind == "sw" else self.map.crossings[j]["line"]
        x, y, h = _point(line, p.u)
        if p.dir < 0:
            h += math.pi
        p.x, p.y, p.heading = x - math.sin(h) * p.lat, y + math.cos(h) * p.lat, h

    def _crossing_clear(self, cr, lights):
        c = self.map.crossings[cr]
        lanes = {i: s0 for i, s0 in c.get("lanes", [])}
        signal = [self.map.lanes[i]["tl"] for i in lanes if self.map.lanes[i]["tl"]]
        if signal and any(Signals.light(lights[k], [idx]) != "red" for k, idx in signal):
            return False
        for v in self.vehicles:
            off = -v.s
            for i in v.plan[:4]:
                if i in lanes:
                    d = off + lanes[i]
                    reach = v.length / 2 + c["w"] / 2 + 0.5
                    on_it = -reach < d < reach
                    coming = d > 0 and v.v > 0.5 and d < max(10.0, v.v * 4.0)
                    if on_it or coming:
                        return False
                off += self.map.lanes[i]["len"]
                if off > 60:
                    break
        e = self.world.ego
        (ax, ay), (bx, by) = c["line"][0], c["line"][-1]
        mx, my = (ax + bx) / 2, (ay + by) / 2
        d = math.hypot(e.x - mx, e.y - my)
        toward = (mx - e.x) * math.cos(e.heading) + (my - e.y) * math.sin(e.heading)
        if d < 6 or (toward > 0 and d < max(10.0, e.speed * 4.0)):
            return False
        return True

    def _blocked(self, p, x, y):
        """Whether a pedestrian stepping to (x, y) would touch a vehicle or our car: then it waits a moment."""
        for v in self.vehicles:
            if abs(v.x - x) > 7 or abs(v.y - y) > 7:
                continue
            c, n = math.cos(v.heading), math.sin(v.heading)
            dx, dy = x - v.x, y - v.y
            lon, lat = dx * c + dy * n, -dx * n + dy * c
            if abs(lon) < v.length / 2 + 0.6 and abs(lat) < v.width / 2 + 0.6:
                return True
        e = self.world.ego
        c, n = math.cos(e.heading), math.sin(e.heading)
        dx, dy = x - e.x, y - e.y
        return abs(dx * c + dy * n) < e.spec.length / 2 + 0.6 and abs(-dx * n + dy * c) < e.spec.width / 2 + 0.6

    def _step_peds(self, dt, lights):
        m = self.map
        self.on_crossing = {}
        for p in self.peds:
            kind, j = p.where
            if kind == "cr":
                self.on_crossing.setdefault(j, []).append(p)
        for p in self.peds:
            p.speed = p.base * (1.7 if self.world.t < p.hurry_until else 1.0)
            kind, j = p.where
            if kind == "free":   # walking where an event sends it (events.py moves it)
                continue
            if kind == "wa":
                (ax, ay), (bx, by) = p.path
                L = math.hypot(bx - ax, by - ay)
                if p.u + p.speed * dt >= L and p.after is not None:
                    p.path, p.after, p.u = p.after, None, 0.0
                    self._place_ped(p)
                    continue
                if p.u + p.speed * dt >= L:
                    nk, nj, end = p.next
                    if nk == "cr" and not p.waiting_ok:
                        if self._crossing_clear(nj, lights) and self.world.t >= p.hold_until:
                            p.waiting_ok = True
                        else:
                            p.waiting += dt
                            if p.waiting > 45:   # gives up and walks on along the footpath instead
                                self._choose(p, p.came_from, avoid_crossing=True)
                            continue
                    p.where, p.u, p.dir, p.waiting = (nk, nj), (0.0 if end == 0 else _len(m.crossings[nj]["line"] if nk == "cr" else m.sidewalks[nj]["line"])), (1 if end == 0 else -1), 0.0
                    if nk == "cr":
                        self.on_crossing.setdefault(nj, []).append(p)
                else:
                    (ax, ay), (bx, by) = p.path
                    L = math.hypot(bx - ax, by - ay) or 1e-6
                    u = min(L, p.u + p.speed * dt) / L
                    if not self._blocked(p, ax + (bx - ax) * u, ay + (by - ay) * u):
                        p.u += p.speed * dt
                self._place_ped(p)
                continue
            line = m.sidewalks[j]["line"] if kind == "sw" else m.crossings[j]["line"]
            L = _len(line)
            nx, ny, _ = _point(line, max(0.0, min(L, p.u + p.dir * p.speed * dt)))
            if self._blocked(p, nx, ny):
                continue
            p.u += p.dir * p.speed * dt
            if p.u >= L or p.u <= 0:
                end = 1 if p.u >= L else 0
                p.u = max(0.0, min(L, p.u))
                wa = (m.sidewalks[j]["wa"] if kind == "sw" else m.crossings[j].get("wa", [None, None]))[end]
                if wa is None or wa in self._bad_wa:
                    p.dir = -p.dir      # a dead end: turn round
                else:
                    self._choose(p, wa, came=(kind, j, end))
            self._place_ped(p)

    def _choose(self, p, wa, came=None, avoid_crossing=False):
        """At a corner, pick where to go next: along a footpath mostly, across a road about a quarter of the time."""
        if wa is None:
            return
        ports = [q for q in self.map.walkareas[wa]["ports"] if (came is None or q != came)
                 and not (q[0] == "sw" and q[1] in self._bad_sidewalks) and not (q[0] == "cr" and q[1] in self._bad_crossings)]
        if avoid_crossing:
            ports = [q for q in ports if q[0] == "sw"] or ports
        if not ports:
            p.dir = -p.dir
            return
        cross = [q for q in ports if q[0] == "cr"]
        walk = [q for q in ports if q[0] == "sw"]
        nxt = self.rng.choice(cross) if cross and (not walk or self.rng.random() < 0.25) else self.rng.choice(walk)
        nk, nj, end = nxt
        line = self.map.crossings[nj]["line"] if nk == "cr" else self.map.sidewalks[nj]["line"]
        target = line[0] if end == 0 else line[-1]
        p.path = ((p.x, p.y), target)
        via = self._via(wa, (p.x, p.y), target)
        if via is not None:      # the straight line would cut across the road: walk round the corner
            p.path, p.after = ((p.x, p.y), via), (via, target)
        else:
            p.after = None
        p.where, p.u, p.next, p.came_from, p.waiting, p.waiting_ok = ("wa", wa), 0.0, nxt, wa, 0.0, False

    def _via(self, wa, a, b):
        """A point inside the corner area to walk through when the straight line from a to b leaves it."""
        from shapely.geometry import LineString, Polygon
        cache = self.__dict__.setdefault("_wa_poly", {})
        if wa not in cache:
            poly = Polygon(self.map.walkareas[wa]["poly"])
            cache[wa] = poly.buffer(0.4) if poly.is_valid else None
        poly = cache[wa]
        if poly is None or poly.contains(LineString([a, b])):
            return None
        q = poly.buffer(-0.5).representative_point() if not poly.buffer(-0.5).is_empty else poly.representative_point()
        return (q.x, q.y)

    # one step --------------------------------------------------------------------------------------------
    def step(self, dt):
        self.clock += dt
        lights = [self.world.signals.state(k, self.world.t) for k in range(len(self.map.signals))]
        self._step_peds(dt, lights)
        self._step_vehicles(dt)
        if int(self.clock / 0.5) != int((self.clock - dt) / 0.5):   # top up the zone twice a second
            self._manage()

    def all(self):
        return self.vehicles + self.peds


def _len(line):
    return sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(line, line[1:]))

"""The game world: the road map, our car (the ego), the signals and the clock, stepped together.

step() moves time forward and reports how the ego is doing: whether all four corners are on the road,
which road and lane it is on, whether it is driving against the lane's direction, and whether it has just
crossed a stop line on red. Each such moment goes into the log with its time.
"""
import math

from .actors import Traffic, a_or_an, apart, level_of
from .events import Events
from .roadmap import RoadMap
from .signals import Signals
from .vehicle import Car


def _crosses(p0, p1, a, b):
    """Whether the segment p0 to p1 crosses the segment a to b."""
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return side(a, b, p0) * side(a, b, p1) < 0 and side(p0, p1, a) * side(p0, p1, b) < 0


def _corners(u):
    c, n = math.cos(u.heading), math.sin(u.heading)
    hl, hw = u.length / 2, u.width / 2
    return [(u.x + c * dx - n * dy, u.y + n * dx + c * dy) for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def _rects_touch(a, b):
    """Separating-axis test for two rectangles given by their corners."""
    for poly in (a, b):
        for k in range(4):
            (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % 4]
            nx, ny = y2 - y1, x1 - x2
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def _edge_hits(p, q, rect):
    """Whether the segment p to q (a bumper) touches the rectangle rect (corners): crosses an edge or lies inside."""
    def inside(pt):
        sides = [(b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]) for a, b in zip(rect, rect[1:] + rect[:1])]
        return all(v >= 0 for v in sides) or all(v <= 0 for v in sides)
    return inside(p) or inside(q) or any(_crosses(p, q, a, b) for a, b in zip(rect, rect[1:] + rect[:1]))


STOPPED = 0.05   # m/s: below this a vehicle counts as stopped when classifying a collision


class World:
    def __init__(self, roadmap=None, seed=0, traffic=True, events=True, start="random", chaos=1.0, style="gurgaon"):
        self.map = roadmap or RoadMap.load()
        self.chaos = chaos   # 1 = as measured; more: events more often, drivers pushier (see Traffic.set_chaos)
        self.style = style   # "gurgaon" (as measured) or "generic" (lane-disciplined traffic, for comparison: Traffic.set_style)
        self.scripted = True   # whether scripted events fire (False: the traffic model alone; events in it are still logged)
        self.start = start   # where drives start: "random" (a main road, by seed), a name from roadmap.STARTS, "sunset" (our morning route), or a pose
        self.with_traffic = traffic
        self.with_events = events
        self.signals = Signals(self.map.signals, seed, self.map.signal_group)
        self.seed = seed
        self.reset()

    def reset(self, seed=None):
        """Start the drive again: with a seed, a new draw of traffic, signal timings and events; without, the same one."""
        if seed is not None and seed != self.seed:
            self.seed = seed
            self.signals = Signals(self.map.signals, seed, self.map.signal_group)
        if self.start == "random":   # a place picked by the seed from a fixed pool (the same seed, the same place)
            pool = self.map.start_pool()
            x, y, h = pool[self.seed % len(pool)]
        else:
            x, y, h = self.map.start_pose(self.start)
        self.ego = Car(x, y, h)
        self.t = 0.0
        self.log = []
        self.crashes = []
        self._last = {}
        self.status = {}
        self._front = self._jolt_front = self._front_point()
        self._last_hole, self.jolt = {}, (-9.0, 0.0)
        self._off_since = None
        self.ego_level = 0
        self.signals_given = []   # (time, "horn" or "dipper") of our recent signals
        self.view = getattr(self, "view", (37.5, 12.5, 40.0))   # metres visible ahead, behind and to each side
        self.traffic = Traffic(self, self.seed) if self.with_traffic else None
        self.events = Events(self, self.seed) if self.with_traffic and self.with_events else None
        self.update_status()
        if self.traffic:
            self.traffic.fill()

    def _front_point(self):
        e = self.ego
        return (e.x + math.cos(e.heading) * e.spec.length / 2, e.y + math.sin(e.heading) * e.spec.length / 2)

    def _road_jolts(self, front):
        """Our car over a speed breaker or through a pothole: too fast, and it jolts (self.jolt: when and how hard,
        for the camera), loses some speed and goes in the log."""
        grid = self.__dict__.get("_feature_grid")
        if grid is None:   # speed breakers and potholes by 20 m cell
            grid = self._feature_grid = {}
            for b in getattr(self.map, "bumps", ()):
                for x, y in (b["a"], b["b"], ((b["a"][0] + b["b"][0]) / 2, (b["a"][1] + b["b"][1]) / 2)):
                    grid.setdefault((int(x // 20), int(y // 20)), set()).add(("bump", id(b)))
            self._features = {id(b): b for b in getattr(self.map, "bumps", ())}
            for p in getattr(self.map, "potholes", ()):
                grid.setdefault((int(p["x"] // 20), int(p["y"] // 20)), set()).add(("hole", id(p)))
                self._features[id(p)] = p
        e = self.ego
        kmh = abs(e.speed) * 3.6
        cell = (int(e.x // 20), int(e.y // 20))
        near = set()
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                near |= grid.get((cell[0] + i, cell[1] + j), set())
        c, n = math.cos(e.heading), math.sin(e.heading)
        wheels = [(e.x + c * 1.3 - n * s * 0.78, e.y + n * 1.3 + c * s * 0.78) for s in (1, -1)]
        for kind, key in near:
            f = self._features[key]
            if kind == "bump" and _crosses(self._jolt_front, front, f["a"], f["b"]):
                if kmh > 25:
                    self._jolt(min(1.0, (kmh - 15) / 35), f"hit a speed breaker at {kmh:.0f} km/h", 0.8)
            elif kind == "hole" and any(math.hypot(wx - f["x"], wy - f["y"]) < f["r"] for wx, wy in wheels):
                if kmh > 20 and self.t - self._last_hole.get(key, -9) > 2:
                    self._last_hole[key] = self.t
                    self._jolt(min(0.7, (kmh - 10) / 50), f"went through a pothole at {kmh:.0f} km/h", 0.9)
        self._jolt_front = front

    def _jolt(self, strength, what, keep):
        self.jolt = (self.t, strength)
        self.ego.speed *= keep
        self.note(what)

    def _inside(self, corners):
        """Whether all the corners lie inside the play area (the wall round the map)."""
        area = getattr(self.map, "play_area", None)
        if not area:
            return True
        n = len(area)
        return all(all((area[(k + 1) % n][0] - area[k][0]) * (y - area[k][1]) - (area[(k + 1) % n][1] - area[k][1]) * (x - area[k][0]) >= 0
                       for k in range(n)) for x, y in corners)

    def step(self, dt, throttle=0.0, brake=0.0, steer=0.0, emergency=False):
        grip = 1.0 if self.status.get("on_road", 4) >= 3 else 0.0
        before = (self.ego.x, self.ego.y, self.ego.heading)
        self.ego.step(dt, throttle, brake, steer, emergency, grip)
        if not self._inside(self.ego.corners()):   # the wall at the edge of the map stops the car
            self.ego.x, self.ego.y, self.ego.heading = before
            if abs(self.ego.speed) > 0.5:
                self.note("hit the wall at the edge of the map")
            self.ego.speed = 0.0
        self.t += dt
        front = self._front_point()
        for k, sg in enumerate(self.map.signals):
            for stop in sg["stops"]:
                if not stop["inner"] and _crosses(self._front, front, stop["a"], stop["b"]):
                    light = Signals.light(self.signals.state(k, self.t), stop["links"])
                    if light == "red" and self.ego.speed > 0:
                        self.note("ran a red light")
        self._front = front
        self._road_jolts(front)
        if self.traffic:
            self.traffic.step(dt)
            self._check_crash()
        if self.events:
            self.events.step(dt)
        self.update_status()

    def signal(self, kind):
        """Our car honks (kind "horn") or flashes its headlights ("dipper"); the traffic around reacts."""
        self.signals_given = [g for g in self.signals_given if self.t - g[0] < 2.0] + [(self.t, kind)]
        if self.traffic:
            self.traffic.ego_signal(kind)

    def signal_ahead(self):
        """The traffic light we are coming up to (within 80 m ahead, on our side of the road): (signal, stop,
        light, seconds until it changes), or None."""
        e = self.ego
        c, n = math.cos(e.heading), math.sin(e.heading)
        best = None
        for k, sg in enumerate(self.map.signals):
            for st in sg["stops"]:
                if st["inner"]:
                    continue
                mx, my = (st["a"][0] + st["b"][0]) / 2, (st["a"][1] + st["b"][1]) / 2
                lon, lat = (mx - e.x) * c + (my - e.y) * n, -(mx - e.x) * n + (my - e.y) * c
                if not (-2 < lon < 80 and abs(lat) < 7):
                    continue
                line = self.map.lanes[st["lane"]]["line"]
                h = math.atan2(line[-1][1] - line[-2][1], line[-1][0] - line[-2][0])
                if abs(math.atan2(math.sin(h - e.heading), math.cos(h - e.heading))) > math.radians(40):
                    continue
                if best is None or lon + abs(lat) < best[0]:
                    best = (lon + abs(lat), k, st)
        if best is None:
            return None
        _, k, st = best
        light = Signals.light(self.signals.state(k, self.t), st["links"])
        return k, st, light, self.signals.until_change(k, self.t, st["links"])

    def _collision_type(self, their_speed, mine, theirs):
        """nuPlan's collision types, in its order, and whether our car is at fault: our car stopped (not at
        fault); the other stopped (at fault); our front bumper on it (at fault); its front on our rear bumper (not
        at fault); else a side collision, at fault only while our car is across two lanes or off the road."""
        e = self.ego
        if abs(e.speed) < STOPPED:
            return "stopped_ego", False
        if their_speed < STOPPED:   # (measured from how far it moved: walkers keep no speed of their own)
            return "stopped_track", True
        if _edge_hits(mine[0], mine[1], theirs):
            return "active_front", True
        if _edge_hits(mine[2], mine[3], theirs):
            return "active_rear", False
        lane = self.status.get("lane")
        across = self.status.get("on_road", 4) < 4   # (a wheel off the road)
        if lane is not None and not across:
            from .actors import _project
            l = self.map.lanes[lane]
            across = abs(_project(l["line"], e.x, e.y)[1]) + e.spec.width / 2 > l["w"] / 2 + 0.1
        return "active_lateral", across

    def _check_crash(self):
        """Our car touching any road user is a crash: it goes in the log and the other road user stops for 5 s."""
        e = self.ego
        mine = e.corners()
        seen, self._near_pos = getattr(self, "_near_pos", {}), {}   # (where each road user near us was last step)
        for u in self.traffic.all():
            if abs(u.x - e.x) > 8 or abs(u.y - e.y) > 8:
                continue
            self._near_pos[id(u)] = (u.x, u.y, self.t)
            if level_of(self.map, u) != self.ego_level or apart(self.map, self.status.get("lane"), getattr(u, "lane", None)):
                continue
            theirs = _corners(u)
            if _rects_touch(mine, theirs):
                if not u.crashed:
                    self.note(f"crashed into {a_or_an(u.kind.replace('_', '-'))}")
                    dx, dy = u.x - e.x, u.y - e.y
                    x0, y0, t0 = seen.get(id(u), (u.x, u.y, self.t - 1.0))
                    kind, fault = self._collision_type(math.hypot(u.x - x0, u.y - y0) / max(1e-6, self.t - t0), mine, theirs)
                    self.crashes.append({"t": round(self.t, 1), "kind": u.kind, "ahead": round(dx * math.cos(e.heading) + dy * math.sin(e.heading), 1),
                                         "left": round(-dx * math.sin(e.heading) + dy * math.cos(e.heading), 1), "our_kmh": round(e.speed * 3.6),
                                         "their_kmh": round(getattr(u, "v", 0) * 3.6), "their_heading": round(math.degrees(u.heading - e.heading)) % 360,
                                         "type": kind, "at_fault": fault})
                u.crashed = True
                u.crashed_at = self.t
                u.v = 0.0

    def gap_ahead(self, look=50.0, lights=True):
        """Distance to the nearest thing our car must stop for (a road user in its path, or, with lights, a red
        or yellow stop line on the road it is driving), and that thing's speed."""
        e = self.ego
        c, n = math.cos(e.heading), math.sin(e.heading)
        best, speed = 1e9, 0.0
        waiting_for_us = {i for i, who in self.traffic._yields.items() if who == -1} if self.traffic else set()
        for u in (self.traffic.all() if self.traffic else ()):
            if level_of(self.map, u) != self.ego_level or apart(self.map, self.status.get("lane"), getattr(u, "lane", None)):
                continue
            if u.id in waiting_for_us:   # it has stopped to let us go
                continue
            dx, dy = u.x - e.x, u.y - e.y
            lon, lat = dx * c + dy * n, -dx * n + dy * c
            if 0 < lon < look and abs(lat) < (e.spec.width + u.width) / 2 + 0.3:
                g = lon - (e.spec.length + u.length) / 2
                if g < best:
                    best, speed = g, (getattr(u, "v", 0.0) if u.kind != "pedestrian" else 0.0) * math.cos(u.heading - e.heading)
        mine = self.status.get("lane")
        for k, sg in enumerate(self.map.signals if lights else ()):
            st = self.signals.state(k, self.t)
            for stop in sg["stops"]:
                if stop["inner"] or (mine is not None and stop["lane"] != mine):   # only our own lane's stop line
                    continue
                mx, my = (stop["a"][0] + stop["b"][0]) / 2, (stop["a"][1] + stop["b"][1]) / 2
                dx, dy = mx - e.x, my - e.y
                lon, lat = dx * c + dy * n, -dx * n + dy * c
                if not (0 < lon < look and abs(lat) < 2.5):
                    continue
                line = self.map.lanes[stop["lane"]]["line"]
                h = math.atan2(line[-1][1] - line[-2][1], line[-1][0] - line[-2][0])
                if abs(math.atan2(math.sin(h - e.heading), math.cos(h - e.heading))) > math.radians(40):
                    continue
                light = Signals.light(st, stop["links"])
                g = lon - e.spec.length / 2 - 0.5
                if light == "red" or (light == "yellow" and g > e.speed ** 2 / (2 * e.spec.decel)):
                    if g < best:
                        best, speed = max(0.0, g), 0.0
        return best, speed

    def note(self, what):
        """Log a moment, at most once every 3 s per kind."""
        if self.t - self._last.get(what, -1e9) > 3:
            self.log.append((round(self.t, 1), what))
        self._last[what] = self.t

    def update_status(self):
        e = self.ego
        on = sum(self.map.on_road(x, y) for x, y in e.corners())
        lane = self.map.nearest_lane(e.x, e.y, e.heading if e.speed >= 0 else e.heading + math.pi, level=self.ego_level)
        st = {"on_road": on, "speed_kmh": e.speed * 3.6, "road": "", "limit_kmh": None, "wrong_way": False, "lane": None}
        if lane:
            i, d, ang = lane
            l = self.map.lanes[i]
            st.update(lane=i, road=l["name"] or self._name_near(i), limit_kmh=round(l["speed"] * 3.6))
            if d < l["w"] / 2 + 0.5:
                self.ego_level = l["level"]
            diff = abs(math.atan2(math.sin(ang - e.heading), math.cos(ang - e.heading)))
            st["wrong_way"] = diff > math.radians(120) and e.speed > 1 and d < l["w"] / 2 + 0.3 and on == 4
        if on < 4 and abs(e.speed) > 0.5:
            if self._off_since is None:
                self._off_since = self.t
            if self.t - self._off_since > 0.5:
                self.note("left the road" if on == 0 else "wheels off the road")
        else:
            self._off_since = None
        if st["wrong_way"]:
            self.note("driving against the traffic")
        # over the speed limit: noted each time we go over (back under means 1 km/h below, so it does not flicker)
        lim, was = st["limit_kmh"], self.status.get("over_limit", False)
        st["over_limit"] = bool(lim) and st["speed_kmh"] > (lim - 1 if was else lim + 1)
        if st["over_limit"] and not was:
            self.note(f"went over the {lim} km/h limit")
        self.status = st

    def _name_near(self, i):
        """A lane without a name: a junction lane takes the road it leads to; an unnamed road says what kind it is."""
        l = self.map.lanes[i]
        for j in l["next"]:
            if self.map.lanes[j]["name"]:
                return self.map.lanes[j]["name"]
        if l["internal"]:
            return "junction"
        return {"primary": "Main road", "secondary": "Main road", "tertiary": "Road", "residential": "Residential street",
                "living_street": "Residential street", "service": "Service road"}.get(l["cls"].replace("_link", ""), "Road")

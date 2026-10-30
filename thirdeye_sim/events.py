"""Events around our car, at the rates measured in the footage.

Each event has a rate (per minute of driving, from the 100 labelled clips in event-rates/labels/truth_v04.json),
a condition for when it can happen, a script for the road user involved, a rule for calling it off, and an
outcome: done, called off, or crash.

Normal traffic already produces some of these on its own (overtakes, cut-ins, two-wheelers passing close).
A monitor watches everyone around our car with the same definitions the labels use and counts those; the
director only adds what is missing, so the totals match the measured rates. Its estimate of the normal rate
starts from what test drives showed (PRIOR, worth five minutes of driving) and learns from this drive.

When an event starts, world.events.banner says what is about to happen; when it ends, how it went. Every event,
started by the director or seen in normal traffic, goes into world.events.records with its outcome: the ground
truth labels for the RL environment, which passes new ones out in its info.
"""
import math, random

from .actors import ANIMAL_KINDS, NAME, _point, _project, a_or_an

# per minute of driving: the human labels on 100 clips (6.7 minutes, day and night). Rough: most were seen 1 to 4
# times. Red-light jumping was not seen in those clips: a placeholder until more footage is labelled.
RATES = {"two_wheeler_close": 2.1, "overtaken_right": 1.2, "overtaken_left": 0.75, "pedestrian_crossing": 0.6,
         "wrong_way_in_lane": 0.45, "cut_in_left": 0.3, "cut_in_right": 0.15, "lead_brake": 0.3, "slow_vehicle_ahead": 0.3,
         "pull_out": 0.3, "pedestrian_on_road": 0.3, "red_light_violation": 0.1}
PLACEHOLDER = {"red_light_violation"}
# Indian scenes our labels do not cover yet (no animal or U-turn label in the 100 clips): ASSUMED rates, per
# minute, to be replaced by measured ones once more footage is labelled
RATES.update({"cow_on_road": 0.1, "dog_crossing": 0.15, "u_turn": 0.15})
ASSUMED = {"cow_on_road", "dog_crossing", "u_turn"}
# what normal traffic gave per minute in the test drives (middle lane), before any events
PRIOR = {"two_wheeler_close": 0.35, "overtaken_right": 1.2, "overtaken_left": 1.0, "cut_in_left": 0.2, "cut_in_right": 0.25,
         "pedestrian_crossing": 0.1, "lead_brake": 0.05}
PRIOR_MINUTES = 5.0
PATIENCE = 20.0          # seconds a due event waits for its condition before it is dropped
LEAD = 2.0               # the banner warns this many seconds ahead, no more (a driver has about that long)
MAX_ACTIVE = 3
AT_ONCE = {"two_wheeler_close": 2}   # events that may run side by side (the most frequent one)
MIX = [("car", 0.60), ("two_wheeler", 0.27), ("auto", 0.09), ("truck", 0.02), ("bus", 0.02)]
SIDE = {1: "left", -1: "right"}
# the word shown above the road user an event is about (with a ring under it on the road)
TAG = {"two_wheeler_close": "SQUEEZING PAST", "overtaken_right": "OVERTAKING", "overtaken_left": "OVERTAKING",
       "cut_in_left": "CUTTING IN", "cut_in_right": "CUTTING IN", "lead_brake": "BRAKING HARD", "pedestrian_crossing": "CROSSING",
       "wrong_way_in_lane": "WRONG WAY", "pull_out": "PULLING OUT", "slow_vehicle_ahead": "SLOW",
       "pedestrian_on_road": "WALKING IN ROAD", "red_light_violation": "JUMPING THE RED", "cow_on_road": "COW",
       "dog_crossing": "DOG", "u_turn": "U-TURN"}
VIEW_MARGIN = 9.0        # the traffic keeps new road users 8 m outside what is visible (Traffic.visible)


def banner_text(b):
    """What the banner (Events.banner) says: (tag, text), the tag being the seconds to go or how it ended."""
    state, e = b["state"], b["event"]
    if state == "coming":
        return (f"IN {b['eta']:.0f} s" if b["eta"] >= 0.5 else "NOW"), e.text
    if state == "done":
        return "DONE", e.past
    if state == "crash":
        return "CRASH", e.past
    return "CALLED OFF", f"{e.text}  ({b.get('detail', '').replace('called off: ', '')})"


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Event:
    """One scripted event: start() checks its condition and sets it up; update() returns an outcome when over."""
    def __init__(self, d, label):
        self.d, self.label, self.w = d, label, d.world
        self.actor, self.side, self.text = None, 0, ""
        self.t0 = self.w.t
        self.eta = 3.0

    # helpers ------------------------------------------------------------------------------------------------
    def rel(self, u):
        e = self.w.ego
        dx, dy = u.x - e.x, u.y - e.y
        c, n = math.cos(e.heading), math.sin(e.heading)
        return dx * c + dy * n, -dx * n + dy * c, _wrap(u.heading - e.heading)

    def ego_lane(self):
        i = self.w.status.get("lane")
        if i is None or self.w.map.lanes[i]["internal"]:
            return None
        return i

    def s_on(self, lane):
        e = self.w.ego
        return _project(self.w.map.lanes[lane]["line"], e.x, e.y)

    def pick_kind(self, mix=MIX):
        r = self.d.rng.random() * sum(p for _, p in mix)
        for k, p in mix:
            r -= p
            if r <= 0:
                return k
        return mix[0][0]

    def gone(self):
        u = self.actor
        tr = self.w.traffic
        return u is None or (u not in tr.vehicles and u not in tr.peds)

    def crashed(self):
        return self.actor is not None and getattr(self.actor, "crashed", False)

    def release(self):
        u = self.actor
        if u is None:
            return
        if u.kind == "pedestrian":
            if u.where[0] == "free":
                self.w.traffic.rejoin_footpath(u)
            u.script = None
        else:
            u.script = None
            u.lc_next = self.w.t + 4.0
        self.d.scripted.pop(id(u), None)

    def along(self, lane, s, d):
        """(lane, s) d metres on from distance s along lane (back when d < 0), straight on through junctions;
        None if the road ends or the spot falls inside a junction."""
        m, prev = self.w.map, self.w.traffic._prev()
        s += d
        for _ in range(10):
            l = m.lanes[lane]
            if 0 <= s <= l["len"]:
                return None if l["internal"] else (lane, s)
            if s > l["len"]:
                if not l["next"]:
                    return None
                s -= l["len"]
                straight = [n for n, dr in zip(l["next"], l["next_dir"]) if dr == "s"] if not l["internal"] else []
                lane = (straight or l["next"])[0]
            else:
                back = prev.get(lane)
                if not back:
                    return None
                lane = ([b for b in back if m.lanes[b]["internal"] and m.lanes[b]["dir"] == "s"] or back)[0]
                s += m.lanes[lane]["len"]
        return None

    def behind_spot(self, lane, extra=(4.0, 12.0)):
        """(lane, s) just behind what the screen shows (and the margin kept around it), level with us on lane."""
        s_e, _ = self.s_on(lane)
        return self.along(lane, s_e, -(self.w.traffic.hidden_from()[1] + VIEW_MARGIN + self.d.rng.uniform(*extra)))

    def ahead_spot(self, lane, extra=(5.0, 20.0)):
        s_e, _ = self.s_on(lane)
        return self.along(lane, s_e, self.w.traffic.hidden_from()[0] + VIEW_MARGIN + self.d.rng.uniform(*extra))

    def spawn(self, kind, spot, **kw):
        return self.w.traffic.spawn_at(kind, *spot, **kw) if spot else None

    def find(self, test):
        """The nearest unscripted vehicle around us passing test(u, lon, lat, dh)."""
        best = None
        for u in self.w.traffic.vehicles:
            if u.script is not None or u.crashed:
                continue
            lon, lat, dh = self.rel(u)
            if test(u, lon, lat, dh) and (best is None or abs(lon) < best[0]):
                best = (abs(lon), u)
        return best[1] if best else None

    def take(self, u, script):
        u.script = script
        if u.kind != "pedestrian":
            u.lc_next = self.w.t + 60.0   # no lane changes of its own while the event runs it
        self.actor = u
        self.d.scripted[id(u)] = self.label

    def no(self, why):
        """start() could not set the event up now: why (kept for checking how often each condition blocks it)."""
        self.why_not = why
        return False

    def start(self):
        return False

    def update(self, dt):
        return None


class TwoWheelerClose(Event):
    """A two-wheeler from behind squeezes past us with 0.5 to 0.85 m between it and our side."""
    def start(self):
        L = self.ego_lane()
        e = self.w.ego
        if L is None or e.speed + 3.0 > self.w.traffic.types["two_wheeler"]["max_speed"]:
            return self.no("not on a lane, or too fast for a two-wheeler to pass")
        m = self.w.map
        left, right = m.beside.get(L, (None, None))
        s_e, lat_e = self.s_on(L)
        w = m.lanes[L]["w"]
        room = {1: w / 2 - lat_e - e.spec.width / 2, -1: w / 2 + lat_e - e.spec.width / 2}
        sides = [s for s in (1, -1) if (left if s > 0 else right) is not None or room[s] > 1.1]
        if not sides:
            return self.no("not sides")
        side = self.d.rng.choice(sides)
        mine = [s_ for s_ in (1, -1) if room[s_] > 1.1]   # sides with room to pass us inside our own lane
        u = self.find(lambda u, lon, lat, dh: u.kind == "two_wheeler" and -40 < lon < -5 and abs(dh) < 0.6
                      and ((u.lane == L and mine) or u.lane in (left, right)) and u.lane is not None)
        if u is not None:
            side = (1 if u.lane == left else -1) if u.lane != L else self.d.rng.choice(mine)
        else:   # from behind in our lane when there is room beside us, otherwise in the lane on that side
            lane = L if room[side] > 1.1 else (left if side > 0 else right)
            if lane is None:
                return self.no("lane is None")
            u = self.spawn("two_wheeler", self.behind_spot(lane, (2.0, 6.0)), speed=e.speed + 4)
        if u is None:
            return self.no("u is None")
        self.side, self.clear = side, self.d.rng.uniform(0.5, 0.85)
        self.take(u, {"speed": e.speed + 5, "side": side, "lat": u.lat})
        self.text = f"a two-wheeler will squeeze past on your {SIDE[side]}"
        self.past = f"a two-wheeler squeezed past on your {SIDE[side]}"
        return True

    def update(self, dt):
        if self.gone():
            return "called off: we went another way"
        if self.crashed():
            return "crash"
        u, e = self.actor, self.w.ego
        lon, lat, _ = self.rel(u)
        ahead = lon - (e.spec.length + u.length) / 2
        self.eta = max(0.0, -ahead / max(0.5, u.v - e.speed))
        # aim for our side: where our car is, seen from its lane, plus the gap
        s_e, lat_e = _project(self.w.map.lanes[u.lane]["line"], e.x, e.y)
        u.script["lat"] = lat_e + self.side * (e.spec.width / 2 + self.clear + u.width / 2)
        u.script["speed"] = max(e.speed + 5.0, 6.0)
        gap = abs(lat) - (e.spec.width + u.width) / 2
        if abs(lon) < (e.spec.length + u.length) / 2 and gap < 0.15 and lat * self.side > 0:
            u.script = {"speed": max(0.0, e.speed - 3.0)}
            return "called off: you moved towards it"
        if ahead > 0.5:   # past our front bumper: it has squeezed by
            return "done"
        if self.w.t - self.t0 > 25:
            return "called off: no way past"
        return None


class Overtaken(Event):
    """A faster vehicle in the lane beside us passes us on that side."""
    def start(self):
        L = self.ego_lane()
        e = self.w.ego
        side = 1 if self.label.endswith("left") else -1
        if L is None or e.speed < 2:
            return self.no("L is None or e.speed < 2")
        lane = self.w.map.beside.get(L, (None, None))[0 if side > 0 else 1]
        if lane is None:
            return self.no("lane is None")
        u = self.find(lambda u, lon, lat, dh: u.lane == lane and u.kind != "cycle" and -45 < lon < -6 and abs(dh) < 0.6)
        if u is None:
            u = self.spawn(self.pick_kind(), self.behind_spot(lane), speed=e.speed + 3)
        if u is None:
            return self.no("u is None")
        self.side, self.boost = side, self.d.rng.uniform(3.5, 6.0)
        self.take(u, {"speed": e.speed + self.boost})
        u.lc_next = self.w.t + 60
        self.text = f"{a_or_an(NAME[u.kind])} will overtake you on the {SIDE[side]}"
        self.past = f"{a_or_an(NAME[u.kind])} overtook you on the {SIDE[side]}"
        return True

    def update(self, dt):
        if self.gone():
            return "called off: we went another way"
        if self.crashed():
            return "crash"
        u, e = self.actor, self.w.ego
        lon, _, _ = self.rel(u)
        ahead = lon - (e.spec.length + u.length) / 2
        self.eta = max(0.0, -ahead / max(0.5, u.v - e.speed))
        u.script["speed"] = min(u.spec["max_speed"], e.speed + self.boost)
        if ahead > 4.0:
            return "done"
        if self.w.t - self.t0 > 20:
            return "called off: you kept pace"
        return None


class CutIn(Event):
    """A vehicle from the lane beside overtakes and cuts into our lane 4 to 12 m ahead of us."""
    def start(self):
        L = self.ego_lane()
        e = self.w.ego
        side = 1 if self.label.endswith("left") else -1
        if L is None or e.speed < 3:
            return self.no("L is None or e.speed < 3")
        lane = self.w.map.beside.get(L, (None, None))[0 if side > 0 else 1]
        if lane is None:
            return self.no("lane is None")
        u = self.find(lambda u, lon, lat, dh: u.lane == lane and u.kind in ("car", "auto", "two_wheeler") and -45 < lon < 8 and abs(dh) < 0.6)
        if u is None:
            u = self.spawn(self.pick_kind(MIX[:3]), self.behind_spot(lane), speed=e.speed + 3)
        if u is None:
            return self.no("u is None")
        self.side, self.lane, self.phase = side, L, "pass"
        self.take(u, {"speed": e.speed + 4})
        u.lc_next = self.w.t + 60
        self.text = f"{a_or_an(NAME[u.kind])} will cut in from your {SIDE[side]}"
        self.past = f"{a_or_an(NAME[u.kind])} cut in from your {SIDE[side]}"
        return True

    def update(self, dt):
        if self.gone():
            return "called off: we went another way"
        if self.crashed():
            return "crash"
        u, e = self.actor, self.w.ego
        lon, lat, _ = self.rel(u)
        ahead = lon - (e.spec.length + u.length) / 2
        if self.phase == "pass":
            self.eta = max(0.0, (4.0 - ahead) / max(0.5, u.v - e.speed))
            u.script["speed"] = e.speed + 4.0
            if 4.0 < ahead < 12.0:
                self.phase, self.t_cut = "cut", self.w.t
                u.script = {"speed": e.speed + 1.0, "cut_in": -self.side}
            elif self.w.t - self.t0 > 20:
                return "called off: you kept pace"
        else:
            self.eta = 0.0
            if "cut_in" not in u.script and not u.lc and abs(lat) < 1.0:
                return "done"
            if "cut_in" in u.script:
                u.script["speed"] = e.speed + 1.0
            if self.w.t - self.t_cut > 10 or ahead > 25:
                return "called off: no gap"
        return None


class LeadBrake(Event):
    """The vehicle directly ahead of us brakes hard (4 to 7 m/s^2), most often to a stop, waits, moves on."""
    def start(self):
        e = self.w.ego
        u = self.find(lambda u, lon, lat, dh: 3 < lon - (e.spec.length + u.length) / 2 < 30 and abs(lat) < 1.2 and abs(dh) < 0.5 and u.v > 4.0)
        if u is None:
            return self.no("u is None")
        lon, _, _ = self.rel(u)
        gap = lon - (e.spec.length + u.length) / 2
        if gap < 5.0 or (e.speed > 1.0 and gap / e.speed < 0.8):
            return self.no("gap < 5.0 or (e.speed > 1.0 and gap / e.speed < 0.8)")  # too close to be avoidable: wait for a fairer moment
        self.take(u, {})
        self.when = self.w.t + 1.0
        self.eta = 1.0
        self.text = f"the {NAME[u.kind]} ahead will brake hard"
        self.past = f"the {NAME[u.kind]} ahead braked hard"
        return True

    def update(self, dt):
        if self.gone():
            return "called off: we went another way"
        if self.crashed():
            return "crash"
        u = self.actor
        self.eta = max(0.0, self.when - self.w.t) if self.when is not None else 0.0
        if self.when is not None and self.w.t >= self.when:
            stop = self.d.rng.random() < 0.7
            u.script = {"brake": (self.d.rng.uniform(4.0, 7.0), 0.0 if stop else self.d.rng.uniform(1.0, 3.0), self.d.rng.uniform(2.0, 5.0))}
            self.when = None
        if self.when is None and "brake" not in u.script:
            return "done"
        if self.w.t - self.t0 > 15:
            return "done"
        return None


class PedestrianCrossing(Event):
    """Someone crosses the road ahead where people do: near a junction (where traffic slows and stops for the
    lights) or between vehicles queued in front of us, and less often anywhere along the road. They wait at the
    kerb on our screen and step out timed to be in our lane 1 to 2 s before we get there, expecting us to slow
    down, as pedestrians here do. Lane by lane, they step into a lane only if other traffic in it is at least
    their accepted gap away (0.8 to 1.4 s, half that after waiting 2.5 s; a vehicle crawling below 1.5 m/s does
    not stop them) and never right under our wheels (0.8 s); otherwise they wait, on a lane line if they are part
    way over. Caught in a lane with something close, they hurry. (Earlier they waited for
    a gap our car never left, so they almost never crossed in front of us: 0 of 52 in test drives.)"""
    def start(self):
        L = self.ego_lane()
        e = self.w.ego
        if L is None or e.speed > 20:
            return self.no("not on a lane, or above 72 km/h")
        m, tr = self.w.map, self.w.traffic
        c, n = math.cos(e.heading), math.sin(e.heading)
        near = self.w.traffic.hidden_from()[0] + VIEW_MARGIN + 2
        far_ = near + 40
        mid = (near + far_) / 2
        spots = {1: [], -1: []}
        for j in m.grids["sidewalks"].near(e.x + c * mid - 40, e.y + n * mid - 40, e.x + c * mid + 40, e.y + n * mid + 40):
            if j in tr._bad_sidewalks:
                continue
            for u, x, y in tr._samples(("sw", j)):
                dx, dy = x - e.x, y - e.y
                lon, lat = dx * c + dy * n, -dx * n + dy * c
                if near < lon < far_ and 3 < abs(lat) < 25:
                    spots[1 if lat > 0 else -1].append((lon, lat, x, y))
        cands = []
        line, s_e = m.lanes[L]["line"], _project(m.lanes[L]["line"], e.x, e.y)[0]
        for side in (1, -1):
            for lon, lat, x, y in spots[side]:
                far = [q for q in spots[-side] if abs(q[0] - lon) < 6]
                s_spot = _project(line, x, y)[0]
                if not far or not (s_e + near - 5 < s_spot < m.lanes[L]["len"] - 3):
                    continue   # only along our own road, before the junction we are heading for (we will pass it)
                # where people cross: by a junction, or where traffic is queued, much more than mid-block
                w = 1.0
                if m.grids["junctions"].near(x - 25, y - 25, x + 25, y + 25):
                    w = 4.0
                if any(abs(u.x - x) < 20 and abs(u.y - y) < 20 and u.v < 2.0 for u in tr.vehicles):
                    w = max(w, 4.0)
                cands.append((w, side, lon, lat, x, y, min(far, key=lambda q: abs(q[0] - lon))))
        if not cands:
            # no footpaths mapped here (OpenStreetMap has them on few Gurgaon roads, and people cross anyway):
            # from 1 m beyond one edge of our road to 1 m beyond the other
            left = right = L
            while m.beside.get(left, (None, None))[0] is not None:
                left = m.beside[left][0]
            while m.beside.get(right, (None, None))[1] is not None:
                right = m.beside[right][1]
            def edge(lane, d, px, py):
                ln = m.lanes[lane]["line"]
                lx, ly, lh = _point(ln, _project(ln, px, py)[0])
                off = m.lanes[lane]["w"] / 2 + 1.0
                return lx - math.sin(lh) * off * d, ly + math.cos(lh) * off * d
            def rel(px, py):
                return (px - e.x) * c + (py - e.y) * n, -(px - e.x) * n + (py - e.y) * c
            for k in range(int(near), int(far_), 5):
                if s_e + k > m.lanes[L]["len"] - 3:
                    break
                cx, cy, _ = _point(line, s_e + k)
                w = 4.0 if m.grids["junctions"].near(cx - 25, cy - 25, cx + 25, cy + 25) else 1.0
                if any(abs(u.x - cx) < 20 and abs(u.y - cy) < 20 and u.v < 2.0 for u in tr.vehicles):
                    w = max(w, 4.0)
                ends = {1: edge(left, 1, cx, cy), -1: edge(right, -1, cx, cy)}
                for side in (1, -1):
                    (x, y), (x2, y2) = ends[side], ends[-side]
                    cands.append((w, side, *rel(x, y), x, y, (*rel(x2, y2), x2, y2)))
        if not cands:
            return self.no("no place to cross ahead")
        w_, side, lon, lat, x, y, far = self.d.rng.choices(cands, [c_[0] for c_ in cands])[0]
        from .actors import Pedestrian
        p = Pedestrian(tr.next_id, None, 0.0, 1, tr.rng)
        if tr.visible(x, y, margin=2):
            return self.no("spot already visible")
        tr.next_id += 1
        p.where, p.x, p.y = ("free", None), x, y
        self.a = (x, y)                                    # straight across, square to the road
        self.b = (x - n * (far[1] - lat), y + c * (far[1] - lat))
        p.heading = math.atan2(self.b[1] - y, self.b[0] - x)
        tr.peds.append(p)
        self.take(p, {"free": True})
        self.L = math.hypot(self.b[0] - x, self.b[1] - y)
        self.lanes = self._lanes()
        self.side, self.phase, self.u = side, "kerb", 0.0
        self.walk = self.d.rng.uniform(1.2, 1.6)
        self.accept = self.d.rng.uniform(0.8, 1.4)         # the gap (s) to other traffic this person steps into
        self.due_since = None
        self.lead = self.d.rng.uniform(1.0, 2.0)           # in our lane this long before we get there
        self.text = f"a pedestrian will cross ahead from your {SIDE[side]}"
        self.past = f"a pedestrian crossed ahead from your {SIDE[side]}"
        return True

    def _lanes(self):
        """The lanes on the way across: (distance across where it starts, where it ends), in order."""
        out, cur = [], None
        m = self.w.map
        for k in range(int(self.L) + 1):
            f = k / max(1.0, self.L)
            x, y = self.a[0] + (self.b[0] - self.a[0]) * f, self.a[1] + (self.b[1] - self.a[1]) * f
            hit = m.nearest_lane(x, y, within=1.8)
            on = hit is not None and hit[1] < m.lanes[hit[0]]["w"] / 2
            if on and cur is None:
                cur = [k, k]
            elif on:
                cur[1] = k
            elif cur is not None:
                out.append(tuple(cur)); cur = None
        if cur is not None:
            out.append(tuple(cur))
        return out

    def _point(self, u):
        f = min(1.0, max(0.0, u / max(0.1, self.L)))
        return self.a[0] + (self.b[0] - self.a[0]) * f, self.a[1] + (self.b[1] - self.a[1]) * f

    def _soonest(self, x, y, ours=False):
        """How soon anything coming will reach the point (x, y): seconds, from every vehicle (and with ours, our
        car) whose path runs over it."""
        best = 1e9
        e = self.w.ego
        movers = ([(e.x, e.y, e.heading, e.spec.length, max(0.0, e.speed))] if ours else []) + \
                 [(u.x, u.y, u.heading, u.length, max(0.0, u.v)) for u in self.w.traffic.vehicles if abs(u.x - x) < 70 and abs(u.y - y) < 70]
        for ux, uy, h, length, v in movers:
            c, n = math.cos(h), math.sin(h)
            lon, lat = (x - ux) * c + (y - uy) * n, -(x - ux) * n + (y - uy) * c
            if abs(lat) < 2.2 and -length / 2 < lon < 70:
                if lon < length / 2 + 0.5:
                    best = 0.0   # right there
                elif v > 1.5 or lon < length / 2 + 2.0:   # (a vehicle crawling up a queue does not stop anyone)
                    best = min(best, (lon - length / 2) / max(0.3, v))
        return best

    def _our_time(self, u=None):
        e = self.w.ego
        x, y = self._point(self.u if u is None else u)
        lon = (x - e.x) * math.cos(e.heading) + (y - e.y) * math.sin(e.heading) - e.spec.length / 2
        return lon / max(0.5, e.speed) if lon > 0 else 1e9

    def _our_line(self):
        """How far across the crossing our car's path runs (where we would hit someone on it)."""
        e = self.w.ego
        c, n = math.cos(e.heading), math.sin(e.heading)
        lat = lambda x, y: -(x - e.x) * n + (y - e.y) * c
        la, lb = lat(*self.a), lat(*self.b)
        return self.L * la / (la - lb) if abs(la - lb) > 1e-6 else self.L / 2

    def update(self, dt):
        if self.gone():   # (left behind us, out of the simulated area, after crossing in front of us: still done)
            return getattr(self, "result", None) or "called off: we went another way"
        if self.crashed():
            return "crash"
        p, tr, e = self.actor, self.w.traffic, self.w.ego
        ours = self._our_time()
        mx, my = (self.a[0] + self.b[0]) / 2, (self.a[1] + self.b[1]) / 2
        if getattr(self, "result", None) is None and \
                (mx - e.x) * math.cos(e.heading) + (my - e.y) * math.sin(e.heading) < -(e.spec.length / 2 + 2):
            # we have driven past where it crosses: it happened in front of us (they are out in the road, and go
            # on to the far kerb), or not at all
            if self.u <= 1.0:
                return "called off: you went past"
            self.result = "done"
        if self.phase == "kerb":
            # steps out timed to be in our lane self.lead seconds before we reach it (or, with us slow or stopped,
            # once we are within 25 m), on our screen, when other traffic in the first lane leaves its gap
            mine = max(0.0, min(self.L, self._our_line()))
            walk_in = max(0.0, mine - 1.0) / self.walk
            ours_there = self._our_time(mine)
            self.eta = max(0.0, ours_there - walk_in - self.lead) if ours_there < 1e8 else 9.0
            first = self._point(self.lanes[0][0]) if self.lanes else self.b
            due = ours_there <= walk_in + self.lead or math.hypot(mx - e.x, my - e.y) < 25
            if due and self.due_since is None:
                self.due_since = self.w.t
            gap = self.accept * (0.5 if due and self.w.t - self.due_since > 2.5 else 1.0)   # impatient after a while
            # (not waiting to be on our screen: from the far side they must set off before then, and come into
            # view already crossing)
            if due and self._soonest(*first) >= gap and \
                    self._our_time(self.lanes[0][0] if self.lanes else 0.0) > 0.8:
                self.phase = "cross"
            elif self.w.t - self.t0 > 45:
                return "called off: never found a gap"
            return None
        self.eta = 0.0
        # the lane it is about to step into, if any, must leave its gap; in a lane with something close, hurry
        speed = self.walk
        nxt = next((ln for ln in self.lanes if self.u < ln[0] - 0.2 and ln[0] - self.u < 0.9), None)
        inside = next((ln for ln in self.lanes if ln[0] - 0.2 <= self.u <= ln[1] + 0.2), None)
        if nxt is not None and (self._soonest(*self._point(nxt[0] + 1.0)) < self.accept or self._our_time(nxt[0] + 1.0) < 0.8):
            speed = 0.0
        elif inside is not None and self._soonest(*self._point(self.u), ours=True) < 1.5:
            speed = self.walk * 1.8
        step = speed * dt
        x, y = self._point(self.u + step)
        if step > 0 and not tr._blocked(p, x, y):
            self.u += step
            p.x, p.y = x, y
        if self.u >= self.L:
            return "done"
        if self.w.t - self.t0 > 70:
            return getattr(self, "result", None) or "called off: never found a gap"
        return None


class WrongWay(Event):
    """A vehicle comes against the traffic in our lane, along its left edge, at 15 to 25 km/h; it pulls in to
    the edge and stops when something is under 2.5 s away, and goes on once we have passed."""
    def start(self):
        L = self.ego_lane()
        e = self.w.ego
        if L is None or e.speed > 22:
            return self.no("not on a lane, or above 80 km/h")
        spot = self.ahead_spot(L)
        if spot is None:
            return self.no("spot is None")
        l = self.w.map.lanes[spot[0]]
        kerb = self.w.map.beside.get(spot[0], (None, None))[0] is None
        # in the kerb lane anything comes; between lanes of traffic only a two-wheeler squeezes through
        kind = self.pick_kind([("two_wheeler", 0.6), ("car", 0.25), ("auto", 0.15)]) if kerb else "two_wheeler"
        spec = self.w.traffic.types[kind]
        edge = max(0.0, l["w"] / 2 - spec["width"] / 2 - 0.25)
        u = self.spawn(kind, spot, lat=edge * 0.8, speed=0.0)
        if u is None:
            return self.no("u is None")
        speed = self.d.rng.uniform(4.0, 7.0)
        # when it gives way: in the kerb lane it pulls right off onto the kerb edge; between lanes it rides the line
        aside = l["w"] / 2 + spec["width"] / 2 + 0.15 if kerb else l["w"] / 2 - 0.1
        self.take(u, {"wrong_way": True, "ww_speed": speed, "lat": edge * 0.8, "edge": 1, "aside": aside})
        u.v = -speed
        self.text = f"{a_or_an(NAME[kind])} is coming the wrong way in your lane"
        self.past = f"{a_or_an(NAME[kind])} came the wrong way in your lane"
        return True

    def update(self, dt):
        if self.gone():
            return "done" if getattr(self, "passed", False) else "called off: turned off"
        if self.crashed():
            return "crash"
        u, e = self.actor, self.w.ego
        lon, _, _ = self.rel(u)
        self.eta = max(0.0, (lon - (e.spec.length + u.length) / 2) / max(0.5, e.speed - u.v))
        if lon < -(e.spec.length + u.length) / 2 - 3:
            self.passed = True
            return "done"
        if self.w.t - self.t0 > 45:
            return "called off: never reached us"
        return None

    def release(self):
        pass   # it carries on against the traffic until it leaves the area


class PullOut(Event):
    """A vehicle parked half on the kerb ahead pulls out into the kerb lane when we are 2 to 3 s away; if we are
    already closer than 1.5 s it waits for us to pass, then pulls out behind us."""
    def start(self):
        L = self.ego_lane()
        if L is None:
            return self.no("L is None")
        m = self.w.map
        k = L                     # the kerb lane of our road
        while m.beside.get(k, (None, None))[0] is not None:
            k = m.beside[k][0]
        spot = self.ahead_spot(k, (5.0, 25.0))
        if spot is None or m.lanes[spot[0]]["len"] - spot[1] < 15:
            return self.no("spot is None or m.lanes[spot[0]]['len'] - spot[1] < 15")
        kl = m.lanes[spot[0]]
        kind = self.pick_kind([("car", 0.6), ("auto", 0.4)])
        u = self.spawn(kind, spot, lat=kl["w"] / 2 + 0.45, speed=0.0)
        if u is None:
            return self.no("u is None")
        self.take(u, {"parked": True})
        self.trigger, self.phase, self.waited = self.d.rng.uniform(2.2, 3.2), "parked", False
        self.text = f"a parked {NAME[kind]} will pull out ahead on your left"
        self.past = f"a parked {NAME[kind]} pulled out ahead on your left"
        return True

    def _go(self):
        self.phase = "out"
        self.actor.script, self.actor.home = {"speed": 4.0}, 0.0

    def update(self, dt):
        if self.gone():
            return "called off: we went another way"
        if self.crashed():
            return "crash"
        u, e = self.actor, self.w.ego
        lon, lat, _ = self.rel(u)
        gap = lon - (e.spec.length + u.length) / 2
        tta = gap / max(0.5, e.speed)
        if self.phase == "parked":
            self.eta = max(0.0, tta - self.trigger)
            if gap < -2:                       # we are past it: it pulls out behind us
                self._go()
            elif (tta <= self.trigger or gap < 10) and self.w.traffic.in_view(u.x, u.y, margin=-2):   # on our screen
                if tta < 1.5 and e.speed > 2:
                    self.waited = True         # we are too close and quick: it waits
                else:
                    self._go()
            return None
        self.eta = 0.0
        if abs(u.lat) < 0.8:
            return "called off: you came too fast" if self.waited else "done"
        if self.w.t - self.t0 > 40:
            return "called off: stuck"
        return None


class SlowVehicle(Event):
    """A cycle-rickshaw or cart at walking pace in our lane, near its left edge (they keep to the kerb lane, so
    only when we are in it)."""
    def start(self):
        L = self.ego_lane()
        if L is None or self.w.map.beside.get(L, (None, None))[0] is not None:
            return self.no("L is None or self.w.map.beside.get(L, (None, None))[0] is not None")
        speed = self.d.rng.uniform(2.0, 3.2)
        u = self.spawn("cycle", self.ahead_spot(L), speed=speed)
        if u is None:
            return self.no("u is None")
        self.take(u, {"speed": speed})
        self.text = "a slow cycle-rickshaw is in your lane ahead"
        self.past = "a slow cycle-rickshaw was in your lane"
        return True

    def update(self, dt):
        if self.gone():
            return "done"
        if self.crashed():
            return "crash"
        lon, _, _ = self.rel(self.actor)
        self.eta = max(0.0, lon / max(0.5, self.w.ego.speed - self.actor.v))
        if lon < -4:
            return "done"
        if self.w.t - self.t0 > 90:
            return "done"
        return None

    def release(self):
        pass   # it stays slow: a cycle-rickshaw does not speed up once we have passed


class PedestrianOnRoad(Event):
    """Someone walking along the edge of the road in the kerb lane for 25 to 50 m, then back onto the footpath."""
    def start(self):
        L = self.ego_lane()
        if L is None:
            return self.no("L is None")
        m = self.w.map
        k = L
        while m.beside.get(k, (None, None))[0] is not None:
            k = m.beside[k][0]
        spot = self.ahead_spot(k, (5.0, 15.0))
        walk = self.d.rng.uniform(25, 50)
        d = self.d.rng.choice((1, -1))
        if spot is None or not (walk < spot[1] < self.w.map.lanes[spot[0]]["len"] - walk):
            return self.no("spot is None or not (walk < spot[1] < self.w.map.lanes[spot[0]]['len']")
        k, s = spot
        kl = m.lanes[k]
        x, y, h = _point(kl["line"], s)
        off = kl["w"] / 2 - 0.2
        from .actors import Pedestrian
        tr = self.w.traffic
        p = Pedestrian(tr.next_id, None, 0.0, 1, tr.rng)
        tr.next_id += 1
        p.where = ("free", None)
        p.x, p.y = x - math.sin(h) * off, y + math.cos(h) * off
        if tr.visible(p.x, p.y, margin=2):
            return self.no("spot already visible")
        tr.peds.append(p)
        self.take(p, {"free": True})
        self.lane, self.s, self.end, self.dir, self.off = k, s, s + d * walk, d, off
        self.text = "a pedestrian is walking in the road ahead"
        self.past = "a pedestrian walked in the road ahead"
        return True

    def update(self, dt):
        if self.gone():
            return "called off: we went another way"
        if self.crashed():
            return "crash"
        p = self.actor
        kl = self.w.map.lanes[self.lane]
        step = p.base * dt
        x, y, h = _point(kl["line"], self.s + self.dir * step)
        nx, ny = x - math.sin(h) * self.off, y + math.cos(h) * self.off
        if not self.w.traffic._blocked(p, nx, ny):
            self.s += self.dir * step
            p.x, p.y, p.heading = nx, ny, h if self.dir > 0 else h + math.pi
        lon, _, _ = self.rel(p)
        self.eta = max(0.0, lon / max(0.5, self.w.ego.speed))
        if (self.dir > 0 and self.s >= self.end) or (self.dir < 0 and self.s <= self.end) or self.w.t - self.t0 > 60:
            return "done"
        return None


class RedLightJump(Event):
    """A vehicle on a crossing road keeps going through its red light while we approach on green."""
    def start(self):
        L = self.ego_lane()
        if L is None:
            return self.no("L is None")
        w, m = self.w, self.w.map
        from .signals import Signals
        for k, sg in enumerate(m.signals):
            mine = [st for st in sg["stops"] if st["lane"] == L and not st["inner"]]
            if not mine:
                continue
            s_e, _ = self.s_on(L)
            if m.lanes[L]["len"] - s_e > 80:
                continue
            state = w.signals.state(k, w.t)
            if Signals.light(state, mine[0]["links"]) == "red":
                continue
            for st in sg["stops"]:
                if st["lane"] == L or st["inner"] or Signals.light(state, st["links"]) != "red":
                    continue
                lane = st["lane"]
                cands = [u for u in w.traffic.vehicles if u.lane == lane and u.script is None and u.v > 3.0
                         and m.lanes[lane]["len"] - u.s < 40]
                if cands:
                    u = max(cands, key=lambda u: u.s)
                    self.take(u, {"red": True})
                    lon, lat, _ = self.rel(u)
                    self.side = 1 if lat > 0 else -1
                    self.lane = lane
                    self.text = f"{a_or_an(NAME[u.kind])} will jump the red light from your {SIDE[self.side]}"
                    self.past = f"{a_or_an(NAME[u.kind])} jumped the red light from your {SIDE[self.side]}"
                    return True
        return self.no("return True")

    def update(self, dt):
        if self.gone():
            return "called off: we went another way"
        if self.crashed():
            return "crash"
        u = self.actor
        self.eta = max(0.0, (self.w.map.lanes[self.lane]["len"] - u.s) / max(1.0, u.v)) if u.lane == self.lane else 0.0
        if u.lane != self.lane and not self.w.map.lanes[u.lane]["internal"]:
            return "done"
        if self.w.t - self.t0 > 15:
            return "called off: it stopped"
        return None


class CowOnRoad(Event):
    """A cow standing in our lane ahead, at an angle to it, as cows do; traffic goes round it. After 20 to 60 s it
    walks off the road."""
    def start(self):
        L = self.ego_lane()
        if L is None:
            return self.no("L is None")
        spot = self.ahead_spot(L, (5.0, 20.0))
        w = self.w.map.lanes[spot[0]]["w"] if spot else 3.5
        u = self.spawn("cow", spot, lat=self.d.rng.uniform(-0.3, 0.3) * w, speed=0.0)
        if u is None:
            return self.no("u is None")
        # it stands for a while, then walks off the road on the nearer side (both assumed, STANDARDS.md): a cow
        # that stood for good held up everyone behind it on a road with no room to go round
        m, k = self.w.map, spot[0]
        n_left = n_right = 0
        while m.beside.get(k, (None, None))[0] is not None and n_left < 8:
            k, n_left = m.beside[k][0], n_left + 1
        k = spot[0]
        while m.beside.get(k, (None, None))[1] is not None and n_right < 8:
            k, n_right = m.beside[k][1], n_right + 1
        side, n = (1, n_left) if n_left <= n_right else (-1, n_right)
        self.take(u, {"parked": True, "turn": self.d.rng.uniform(-1.2, 1.2),
                      "leave_at": self.w.t + self.d.rng.uniform(20.0, 60.0), "leave_lat": side * (n * w + w / 2 + 1.2)})
        self.text = "a cow is standing in the road ahead"
        self.past = "a cow stood in the road"
        return True

    def update(self, dt):
        if self.gone():
            return "done"
        if self.crashed():
            return "crash"
        lon, _, _ = self.rel(self.actor)
        self.eta = max(0.0, lon / max(0.5, self.w.ego.speed))
        if lon < -5:
            return "done"
        if self.w.t - self.t0 > 90:
            return "called off: never reached"
        return None

    def release(self):
        pass   # it stands until its time is up, then walks off the road (actors.py)


class DogCrossing(Event):
    """A dog at the roadside runs across in front of us when we are about to reach it: there is little time."""
    SPEED = 4.0

    def start(self):
        L = self.ego_lane()
        if L is None or self.w.ego.speed < 3:
            return self.no("L is None or we are nearly still")
        m = self.w.map
        spot = self.ahead_spot(L, (0.0, 10.0))
        if spot is None:
            return self.no("spot is None")
        k, s = spot
        left = right = k   # the road's two edges, from the lanes beside ours
        while m.beside.get(left, (None, None))[0] is not None:
            left = m.beside[left][0]
        while m.beside.get(right, (None, None))[1] is not None:
            right = m.beside[right][1]
        x, y, h = _point(m.lanes[k]["line"], s)
        nx, ny = -math.sin(h), math.cos(h)
        side = self.d.rng.choice((1, -1))
        def edge(lane, d):   # a point 3 m beyond the outer edge of lane, on side d
            lx, ly, lh = _point(m.lanes[lane]["line"], _project(m.lanes[lane]["line"], x, y)[0])
            off = m.lanes[lane]["w"] / 2 + 3.0
            return lx - math.sin(lh) * off * d, ly + math.cos(lh) * off * d
        a, b = (edge(left, 1), edge(right, -1)) if side == 1 else (edge(right, -1), edge(left, 1))
        from .actors import Pedestrian
        tr = self.w.traffic
        p = Pedestrian(tr.next_id, None, 0.0, 1, tr.rng)
        tr.next_id += 1
        p.kind, p.where = "dog", ("free", None)
        p.length, p.width = 0.8, 0.3
        p.x, p.y, p.heading = a[0], a[1], math.atan2(b[1] - a[1], b[0] - a[0])
        if tr.visible(p.x, p.y, margin=0):
            return self.no("spot already visible")
        tr.peds.append(p)
        self.take(p, {"free": True})
        self.a, self.b, self.u, self.running = a, b, 0.0, False
        self.across = (x, y)
        self.text = "a dog will dart across the road ahead"
        self.past = "a dog ran across the road"
        return True

    def update(self, dt):
        if self.gone():
            return "called off: we went another way"
        if self.crashed():
            return "crash"
        p, e = self.actor, self.w.ego
        ax, ay = self.across
        lon = (ax - e.x) * math.cos(e.heading) + (ay - e.y) * math.sin(e.heading)
        self.eta = max(0.0, lon / max(0.5, e.speed))
        L = math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])
        if not self.running and (self.eta < (L / 2) / self.SPEED + 0.4 or lon < 8):
            self.running = True   # off it goes, timed to cross just in front of us
        if self.running:
            self.u = min(L, self.u + self.SPEED * dt)
            p.x = self.a[0] + (self.b[0] - self.a[0]) * self.u / L
            p.y = self.a[1] + (self.b[1] - self.a[1]) * self.u / L
            if self.u >= L:
                self.w.traffic.peds.remove(p)
                return "done"
        if lon < -15 and not self.running:
            self.w.traffic.peds.remove(p)
            return "called off: we went past"
        return None

    def release(self):
        self.d.scripted.pop(id(self.actor), None)


class UTurn(Event):
    """A vehicle on the other carriageway turns round through a gap in the median ahead and comes out into our
    road, across its lanes: we slow down or stop for it."""
    def start(self):
        L = self.ego_lane()
        if L is None:
            return self.no("L is None")
        m = self.w.map
        M = L
        while m.beside.get(M, (None, None))[1] is not None:   # our road's lane next to the median
            M = m.beside[M][1]
        gap = self.ahead_spot(M, (5.0, 25.0))
        if gap is None:
            return self.no("no spot ahead")
        mk, ms = gap
        gx, gy, gh = _point(m.lanes[mk]["line"], ms)
        best = None   # the nearest lane of the other carriageway, on our right, running the other way
        for j in m.grids["lanes"].near(gx - 25, gy - 25, gx + 25, gy + 25):
            lj = m.lanes[j]
            if lj["internal"] or lj["level"] != m.lanes[mk]["level"]:
                continue
            sj, latj = _project(lj["line"], gx, gy)
            if not (20 < sj < lj["len"] - 5):
                continue
            jx, jy, jh = _point(lj["line"], sj)
            side = -(jx - gx) * math.sin(gh) + (jy - gy) * math.cos(gh)   # + to the left of our road
            if abs(_wrap(jh - gh)) > math.radians(150) and -25 < side < -3 and (best is None or -side < best[0]):
                best = (-side, j, sj)
        if best is None:
            return self.no("no other carriageway beside us")
        _, j, sj = best
        kind = self.pick_kind([("car", 0.6), ("auto", 0.25), ("two_wheeler", 0.15)])
        u = self.spawn(kind, (j, max(5.0, sj - 22.0)), speed=5.0)
        if u is None:
            return self.no("u is None")
        self.take(u, {"speed": 5.0})
        self.j, self.sj, self.mk, self.ms = j, sj, mk, ms
        self.curve, self.k = None, 0.0
        self.text = f"{a_or_an(NAME[kind])} will make a U-turn ahead"
        self.past = f"{a_or_an(NAME[kind])} made a U-turn ahead"
        return True

    def update(self, dt):
        if self.gone():
            return "called off: we went another way"
        if self.crashed():
            return "crash"
        u, m = self.actor, self.w.map
        lon, _, _ = self.rel(u)
        self.eta = max(0.0, lon / max(0.5, self.w.ego.speed))
        if self.curve is None:
            if u.lane != self.j or u.s < self.sj - 1.0:
                if self.w.t - self.t0 > 30:
                    return "called off: never reached the gap"
                return None
            # the turn: a curve from where it is to our road's median lane, facing our way
            tx, ty, th = _point(m.lanes[self.mk]["line"], self.ms)
            d = math.hypot(tx - u.x, ty - u.y)
            p0, p3 = (u.x, u.y), (tx, ty)
            p1 = (u.x + math.cos(u.heading) * d * 0.7, u.y + math.sin(u.heading) * d * 0.7)
            p2 = (tx - math.cos(th) * d * 0.7, ty - math.sin(th) * d * 0.7)
            self.curve, self.k, self.len = (p0, p1, p2, p3), 0.0, d * 1.6
            u.lane, u.s, u.plan, u.v, u.lat = self.mk, self.ms, [self.mk], 0.0, 0.0   # it blocks our road where it turns
            u.script = {"parked": True, "pose": (u.x, u.y, u.heading)}
        self.k = min(1.0, self.k + 3.0 * dt / self.len)
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = self.curve
        t, r = self.k, 1 - self.k
        x = r ** 3 * x0 + 3 * r * r * t * x1 + 3 * r * t * t * x2 + t ** 3 * x3
        y = r ** 3 * y0 + 3 * r * r * t * y1 + 3 * r * t * t * y2 + t ** 3 * y3
        dx = 3 * r * r * (x1 - x0) + 6 * r * t * (x2 - x1) + 3 * t * t * (x3 - x2)
        dy = 3 * r * r * (y1 - y0) + 6 * r * t * (y2 - y1) + 3 * t * t * (y3 - y2)
        u.script["pose"] = (x, y, math.atan2(dy, dx))
        if self.k >= 1.0:
            u.script = {"speed": 4.0}   # off along our road, slowly at first
            u.v = 2.0
            self.w.traffic._extend_plan(u)
            return "done"
        return None


MAKERS = {"cow_on_road": CowOnRoad, "dog_crossing": DogCrossing, "u_turn": UTurn, "two_wheeler_close": TwoWheelerClose, "overtaken_right": Overtaken, "overtaken_left": Overtaken,
          "cut_in_left": CutIn, "cut_in_right": CutIn, "lead_brake": LeadBrake, "pedestrian_crossing": PedestrianCrossing,
          "wrong_way_in_lane": WrongWay, "pull_out": PullOut, "slow_vehicle_ahead": SlowVehicle,
          "pedestrian_on_road": PedestrianOnRoad, "red_light_violation": RedLightJump}


class Monitor:
    """Sees events around our car with the definitions the labels use (as in event-rates/scripts/track_events.py),
    from everyone's true positions, 15 times a second."""
    HALF_LEN, HALF_W = 2.15, 0.9

    def __init__(self, d):
        self.d, self.hist, self.cool = d, {}, {}

    def _fire(self, u, label, extra=""):
        key = (id(u), label)
        t = self.d.world.t
        if t < self.cool.get(key, -1e9):
            return
        self.cool[key] = t + (20.0 if label in ("wrong_way_in_lane", "pedestrian_crossing") else 10.0)
        self.d.seen(u, label, extra)

    def step(self):
        w = self.d.world
        e = w.ego
        c, n = math.cos(e.heading), math.sin(e.heading)
        t = w.t
        alive = set()
        for u in w.traffic.all():
            if u.kind in ANIMAL_KINDS:   # (animals are events of their own, not road users the labels count)
                continue
            dx, dy = u.x - e.x, u.y - e.y
            if abs(dx) > 45 or abs(dy) > 45:
                continue
            lon, lat = dx * c + dy * n, -dx * n + dy * c
            dh = _wrap(u.heading - e.heading)
            h = self.hist.setdefault(id(u), [])
            h.append((t, lon, lat, dh))
            while h and h[0][0] < t - 4.0:
                h.pop(0)
            alive.add(id(u))
            ped = u.kind == "pedestrian"
            same = abs(dh) < math.radians(45)
            if ped:
                if 3 < lon < 20 and abs(lat) > 2.5 and any(abs(q[2]) > 2.5 and q[2] * lat < 0 and 3 < q[1] < 20 for q in h):
                    self._fire(u, "pedestrian_crossing")
                continue
            small = u.kind in ("two_wheeler", "cycle")
            gap = abs(lat) - self.HALF_W - u.width / 2
            # passing us: from behind our rear to ahead of our front within 3 s, within 8 m to the side
            if same and lon > self.HALF_LEN + u.length / 2 and abs(lat) < 8:
                behind = [q for q in h if q[0] > t - 3.0 and q[1] < -self.HALF_LEN - u.length / 2]
                if behind:
                    side = "left" if sum(q[2] for q in h if q[0] >= behind[-1][0]) > 0 else "right"
                    close = min(abs(q[2]) for q in h if q[0] >= behind[-1][0]) - self.HALF_W - u.width / 2
                    self._fire(u, "two_wheeler_close" if small and close < 1.0 else f"overtaken_{side}")
            if small and abs(lon) < self.HALF_LEN + u.length / 2 and gap < 1.0:
                self._fire(u, "two_wheeler_close")
            # into our lane 2 to 15 m ahead from beside it
            if same and 2 < lon < 15 and abs(lat) < 0.8:
                was = [q for q in h if q[0] > t - 3.0 and abs(q[2]) > 1.75 + 0.8]
                if was:
                    self._fire(u, f"cut_in_{'left' if was[-1][2] > 0 else 'right'}")
            # against us in our lane for a second: facing against the way its own lane runs (traffic turning
            # across our lane, or oncoming while we turn, does not count)
            l = w.map.lanes[u.lane]
            against = abs(_wrap(_point(l["line"], u.s)[2] - u.heading)) > math.radians(135)
            if against and 3 < lon < 40 and abs(lat) < 1.75 and abs(dh) > math.radians(135):
                if all(abs(q[3]) > math.radians(135) for q in h if q[0] > t - 1.0) and h[0][0] < t - 1.0:
                    self._fire(u, "wrong_way_in_lane")
            # the vehicle directly ahead braking hard
            if same and 0 < lon < 30 and abs(lat) < 1.5 and getattr(u, "acc", 0.0) < -3.5 and getattr(u, "v", 0.0) > 3.0:
                self._fire(u, "lead_brake")
        for k in [k for k in self.hist if k not in alive]:
            del self.hist[k]


class Events:
    def __init__(self, world, seed=0):
        self.world = world
        self.rng = random.Random(seed + 991)
        self.monitor = Monitor(self)
        self.records = []        # every event, started or seen: dicts with t, label, source, actor, outcome
        self.active, self.pending = [], {}
        self.scripted = {}       # id(road user) -> label of the event running it
        self.seen_count = {k: 0 for k in RATES}
        self.done_count = {k: 0 for k in RATES}
        self.tried = {k: 0 for k in RATES}     # scripted events that fell due (started or not)
        self.blocked = {}        # (label, the condition that stopped it starting) -> times
        self.banner = None       # {"text", "state" (coming / done / called off / crash), "eta", "t"}
        self.recent = []         # (time it ended, road user, state) of events just over, for a flash on screen
        self._steps = 0
        self._next_try = 0.0

    def seen(self, u, label, extra=""):
        """The monitor saw an event; one that a script is running is that event's, not normal traffic's."""
        if id(u) in self.scripted:
            return
        self.seen_count[label] = self.seen_count.get(label, 0) + 1
        self.records.append({"t": round(self.world.t, 2), "label": label, "source": "traffic", "actor": u.kind,
                             "outcome": "seen"})
        who = NAME.get(u.kind, u.kind)
        side = label.rsplit("_", 1)[-1]
        self.world.note({"two_wheeler_close": f"{a_or_an(who)} squeezed past close", "overtaken_left": f"{a_or_an(who)} overtook you on the left",
                         "overtaken_right": f"{a_or_an(who)} overtook you on the right", "cut_in_left": f"{a_or_an(who)} cut in from your left",
                         "cut_in_right": f"{a_or_an(who)} cut in from your right", "lead_brake": f"the {who} ahead braked hard",
                         "pedestrian_crossing": "a pedestrian crossed in front of you",
                         "wrong_way_in_lane": f"{a_or_an(who)} came the wrong way"}.get(label, label.replace("_", " ")))

    def rate_so_far(self, label):
        """How often this event has happened so far, per minute: seen in normal traffic plus scripted ones that
        went through, starting from what normal traffic gave in the test drives (worth PRIOR_MINUTES)."""
        minutes = self.world.t / 60.0
        happened = self.seen_count.get(label, 0) + self.done_count.get(label, 0)
        return (PRIOR.get(label, 0.0) * PRIOR_MINUTES + happened) / (PRIOR_MINUTES + minutes)

    def due_rate(self, label):
        """How often (per minute) to make this event fall due: what the traffic does not produce of its rate,
        divided by the share of scripted ones that play out (some find no chance, or are called off). Without
        that division the director settled at about half the measured rate."""
        w = self.world
        target = RATES[label] * w.chaos
        if self.rate_so_far(label) >= target:
            return 0.0
        minutes = w.t / 60.0
        traffic = (PRIOR.get(label, 0.0) * PRIOR_MINUTES + self.seen_count.get(label, 0)) / (PRIOR_MINUTES + minutes)
        works = (self.done_count.get(label, 0) + 1.0) / (self.tried.get(label, 0) + 1.0)   # (starts at 1: no failures yet)
        return max(0.0, target - traffic) / max(0.25, works)

    def step(self, dt):
        w = self.world
        self._steps += 1
        if self._steps % 4 == 0:
            self.monitor.step()
        for label in RATES:   # each event falls due at the rate still missing (called-off ones are made up)
            want = self.due_rate(label) if w.style == "gurgaon" and w.scripted else 0.0
            if label not in self.pending and self.rng.random() < want / 60.0 * dt:
                self.pending[label] = w.t + PATIENCE
                self.tried[label] = self.tried.get(label, 0) + 1
        if w.t >= self._next_try:
            self._next_try = w.t + 0.5
            for label in sorted(self.pending, key=self.pending.get):
                if w.t > self.pending[label]:
                    del self.pending[label]
                    self.records.append({"t": round(w.t, 2), "label": label, "source": "scripted", "actor": None, "outcome": "no chance"})
                    continue
                if len(self.active) >= MAX_ACTIVE or sum(a.label == label for a in self.active) >= AT_ONCE.get(label, 1):
                    continue
                ev = MAKERS[label](self, label)
                if not ev.start():
                    key = (label, getattr(ev, "why_not", "?"))
                    self.blocked[key] = self.blocked.get(key, 0) + 1
                else:
                    del self.pending[label]
                    self.active.append(ev)
        for ev in list(self.active):
            out = ev.update(dt)
            # did it play out where we could see it: its road user on screen as it happened
            if ev.eta <= LEAD and ev.actor is not None and w.traffic.in_view(ev.actor.x, ev.actor.y, margin=0):
                ev.shown = True
            if ev.eta <= LEAD:
                ev.highlight = True   # its road user gets a ring and a tag on screen until it is over
            if out:
                self._finish(ev, out)
        # the banner: how the last event went (for 2.5 s), else the event about to happen within LEAD seconds
        if self.banner and self.banner["state"] != "coming" and w.t - self.banner["t"] < 2.5:
            return
        soon = [ev for ev in self.active if ev.eta <= LEAD and getattr(ev, "shown", False)]
        if soon:   # the soonest, and of those the nearest to us
            ev = min(soon, key=lambda ev: (round(ev.eta * 2), math.hypot(ev.actor.x - w.ego.x, ev.actor.y - w.ego.y)))
            ev.announced = True
            self.banner = {"text": ev.text, "state": "coming", "eta": ev.eta, "t": w.t, "event": ev}
        else:
            self.banner = None

    def _finish(self, ev, outcome):
        w = self.world
        self.active.remove(ev)
        ev.release()
        if outcome == "done":
            self.done_count[ev.label] = self.done_count.get(ev.label, 0) + 1
        state = "crash" if outcome == "crash" else "done" if outcome == "done" else "called off"
        shown = getattr(ev, "shown", False)
        self.records.append({"t": round(ev.t0, 2), "t_end": round(w.t, 2), "label": ev.label, "source": "scripted",
                             "actor": ev.actor.kind if ev.actor is not None else None, "outcome": outcome,
                             "on_screen": shown, "placeholder_rate": ev.label in PLACEHOLDER})
        # say how it went only for what we were warned about or saw happen; a quiet call-off stays quiet
        self.recent = [r for r in self.recent if w.t - r[0] < 2.0]
        if getattr(ev, "highlight", False) and ev.actor is not None:
            self.recent.append((w.t, ev.actor, state))
        if getattr(ev, "announced", False) or (shown and state != "called off"):
            self.banner = {"text": ev.text, "state": state, "detail": outcome, "eta": 0.0, "t": w.t, "event": ev}
            w.note(ev.past if state == "done" else f"crash: {ev.past}" if state == "crash" else f"{outcome}: {ev.text}")

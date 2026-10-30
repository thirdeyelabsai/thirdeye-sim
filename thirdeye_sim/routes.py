"""A route for a driving task: a path through the lane graph from where a drive starts, and how far along it our
car has got. It gives a planner what a navigation system would (how far to go, which way the road bends, the
next turn) and the evaluation what it needs (progress, completion, how far off the route the car strays).

    route = Route(world.map, lane, s, length=800.0, rng=random.Random(0))
    route.update(x, y)        # progress (m, never goes back), deviation (m)
    route.guidance(x, y, h)   # remaining m, bearing to a point 15 m on, offset from the route, next turn
"""
import math

TURNS = {"l": 1, "L": 1, "r": -1, "R": -1}   # SUMO's link directions: + left, - right ("t", a U-turn, is never taken)


class Route:
    def __init__(self, m, lane, s0, length, rng, straight=0.7):
        """From distance s0 along lane, about length metres on: straight on through a junction with probability
        straight where the road allows, else any way but back; never onto a lane by the wall round the map, nor
        round a corner tighter than a car can turn."""
        self.m, self.s0 = m, s0   # (s0: where on its first lane the route starts)
        wall = (m.near_wall() | m.too_tight()) if hasattr(m, "near_wall") else set()   # (nor round a corner too tight to drive)
        best = None
        for _ in range(30):   # draw again if the way ends early (a dead end, the wall at the map's edge)
            lanes, total, cur = [lane], m.lanes[lane]["len"] - s0, lane
            while total < length:
                l = m.lanes[cur]
                # (a junction lane counts as by the wall when every road out of it is)
                nexts = [(n, d) for n, d in zip(l["next"], l["next_dir"]) if d != "t" and n not in wall and
                         not (m.lanes[n]["internal"] and m.lanes[n]["next"] and all(k in wall for k in m.lanes[n]["next"]))]
                if not nexts:
                    break
                on = [n for n, d in nexts if d == "s"]
                cur = rng.choice(on) if on and rng.random() < straight else rng.choice([n for n, _ in nexts])
                lanes.append(cur)
                total += m.lanes[cur]["len"]
            if best is None or total > best[0]:
                best = (total, lanes)
            if total >= length:
                break
        lanes = best[1]
        self.lanes = lanes
        self.edges = {m.lanes[i]["edge"] for i in lanes if not m.lanes[i]["internal"]}
        # the route as one polyline, from where the drive starts
        pts, acc = [], 0.0
        first = m.lanes[lane]["line"]
        for (ax, ay), (bx, by) in zip(first, first[1:]):
            L = math.hypot(bx - ax, by - ay)
            if acc + L > s0:
                u = max(0.0, s0 - acc) / L if L else 0.0
                if not pts:
                    pts.append((ax + (bx - ax) * u, ay + (by - ay) * u))
                pts.append((bx, by))
            acc += L
        for i in lanes[1:]:
            for p in m.lanes[i]["line"]:
                if not pts or math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 0.05:
                    pts.append(tuple(p))
        self.pts = pts
        self.cum = [0.0]
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            self.cum.append(self.cum[-1] + math.hypot(bx - ax, by - ay))
        self.length = self.cum[-1]
        # where each turn is: (distance along the route, +1 left / -1 right)
        self.turns, acc = [], m.lanes[lane]["len"] - s0
        for i, n in zip(lanes, lanes[1:]):
            l = m.lanes[i]
            if n in l["next"]:
                d = l["next_dir"][l["next"].index(n)]
                if d in TURNS:
                    self.turns.append((acc, TURNS[d]))
            acc += m.lanes[n]["len"]
        self.k = 0                 # the polyline segment our car was last nearest
        self.progress = 0.0        # the furthest it has got (m)
        self.deviation = 0.0       # how far it is from the route line (m)
        self.offset = 0.0          # signed: + to the left of the route

    def _nearest(self, x, y):
        best = None
        for k in range(max(0, self.k - 20), min(len(self.pts) - 1, self.k + 60)):
            (ax, ay), (bx, by) = self.pts[k], self.pts[k + 1]
            L2 = (bx - ax) ** 2 + (by - ay) ** 2 or 1e-9
            u = max(0.0, min(1.0, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / L2))
            px, py = ax + (bx - ax) * u, ay + (by - ay) * u
            d = math.hypot(x - px, y - py)
            if best is None or d < best[0]:
                side = 1 if (bx - ax) * (y - ay) - (by - ay) * (x - ax) > 0 else -1
                best = (d, k, self.cum[k] + u * math.sqrt(L2), side * d)
        return best

    def update(self, x, y):
        d, k, s, off = self._nearest(x, y)
        self.k, self.deviation, self.offset = k, d, off
        if d < 6.0:
            self.progress = max(self.progress, s)
        return self.progress, self.deviation

    def point(self, s):
        """(x, y, heading) at distance s along the route."""
        s = max(0.0, min(self.length, s))
        for k in range(len(self.cum) - 1):
            if self.cum[k + 1] >= s:
                (ax, ay), (bx, by) = self.pts[k], self.pts[k + 1]
                L = self.cum[k + 1] - self.cum[k] or 1e-9
                u = (s - self.cum[k]) / L
                return ax + (bx - ax) * u, ay + (by - ay) * u, math.atan2(by - ay, bx - ax)
        (ax, ay), (bx, by) = self.pts[-2], self.pts[-1]
        return bx, by, math.atan2(by - ay, bx - ax)

    @property
    def completion(self):
        return min(1.0, self.progress / max(1e-6, self.length))

    def guidance(self, x, y, h, ahead=15.0):
        """What a navigation system tells the driver: metres left, the bearing to the route 15 m on (radians,
        + to the left), the offset from the route line (m, + left), and the next turn (metres to it, +1 left,
        -1 right, 0 none within 300 m)."""
        _, _, s_here, _ = self._nearest(x, y)
        tx, ty, _ = self.point(s_here + ahead)
        bearing = math.atan2(ty - y, tx - x) - h
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        nxt = next(((d - s_here, side) for d, side in self.turns if d > s_here - 2), None)
        to_turn, side = (min(300.0, max(0.0, nxt[0])), nxt[1]) if nxt and nxt[0] < 300 else (300.0, 0)
        return max(0.0, self.length - self.progress), bearing, self.offset, to_turn, side

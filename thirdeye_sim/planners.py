"""Baseline drivers for the environment (env.py), to score others against (evaluate.py).

IDMPlanner is the usual rule-based baseline (nuPlan ships one): it follows the route with pure pursuit and sets
its speed with the Intelligent Driver Model (Treiber and Kesting's city-car parameters: v0 = the speed limit,
T = 1.0 s, s0 = 2 m, a = 1.5 m/s^2, b = 2.0 m/s^2, delta = 4), keeping its gap to whatever is in its path and
stopping for red lights; it slows to 15 km/h for speed breakers, as a car with a map would. It sees the true
positions of everyone (as nuPlan's IDM planner sees tracked objects), but does nothing Indian roads ask of a
driver beyond that: no reading of two-wheelers filtering, no horn, no edging through gaps.

LaneKeeper is a floor: it follows the route at a steady 30 km/h and never brakes for anyone.

    planner = IDMPlanner(env.unwrapped);  action = planner.act()   (call after every reset)
"""
import math

from .env import pursue

IDM = {"T": 1.0, "s0": 2.0, "a": 1.5, "b": 2.0, "delta": 4}


class IDMPlanner:
    name = "IDM planner"

    def __init__(self, env):
        self.env = env

    def bumps(self):
        """Distances along the route to its speed breakers (computed once per route)."""
        r = self.env.route
        if getattr(self, "_for", None) is not r:
            m = self.env.world.map
            acc, out = -r.s0, []   # (the route starts s0 along its first lane)
            for i in r.lanes:
                for s in m.lanes[i]["bumps"]:
                    if acc + s > 0:
                        out.append(acc + s)
                acc += m.lanes[i]["len"]
            self._for, self._bumps = r, sorted(out)
        return self._bumps

    def act(self):
        env = self.env
        w, e, r = env.world, env.world.ego, env.route
        _, _, s_here, _ = r._nearest(e.x, e.y)
        tx, ty, _ = r.point(s_here + max(6.0, 0.9 * e.speed))
        steer = pursue(e, tx, ty)
        limit = (w.status.get("limit_kmh") or 50) / 3.6
        v0 = limit
        # bends from under the car to 40 m on: keep sideways acceleration under 2 m/s^2
        for d in (0.0, 5.0, 10.0, 20.0, 30.0, 40.0):
            h1, h2 = r.point(s_here + d - 5)[2], r.point(s_here + d + 5)[2]
            k = abs(math.atan2(math.sin(h2 - h1), math.cos(h2 - h1))) / 10.0
            if k > 1e-3:
                v0 = min(v0, math.sqrt(2.0 / k + 2 * IDM["b"] * max(0.0, d - 5)))
        for b in self.bumps():   # speed breakers: cross at 15 km/h
            d = b - s_here - e.spec.length / 2
            if -2 < d < 60:
                v0 = min(v0, math.sqrt(4.2 ** 2 + 2 * IDM["b"] * max(0.0, d)))
        gap, lead = w.gap_ahead(look=80.0)
        v = max(0.0, e.speed)
        free = 1 - (v / max(0.1, v0)) ** IDM["delta"]
        if gap < 1e8:
            s_star = IDM["s0"] + max(0.0, v * IDM["T"] + v * (v - lead) / (2 * math.sqrt(IDM["a"] * IDM["b"])))
            acc = IDM["a"] * (free - (s_star / max(0.1, gap)) ** 2)
        else:
            acc = IDM["a"] * free
        a = acc / e.spec.accel if acc >= 0 else acc / e.spec.decel
        return [max(-1.0, min(1.0, a)), steer]


class LaneKeeper:
    name = "lane keeping only"

    def __init__(self, env):
        self.env = env

    def act(self):
        env = self.env
        e, r = env.world.ego, env.route
        _, _, s_here, _ = r._nearest(e.x, e.y)
        tx, ty, _ = r.point(s_here + max(6.0, 0.9 * e.speed))
        a = (8.3 - e.speed) / 1.5
        a = a / e.spec.accel if a >= 0 else a / e.spec.decel
        return [max(-1.0, min(1.0, a)), pursue(e, tx, ty)]


PLANNERS = {"idm": IDMPlanner, "lane_keeper": LaneKeeper}

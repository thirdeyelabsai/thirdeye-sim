"""A simple driver that follows a polyline: pure pursuit for steering, a target speed for the pedal.

It aims at a point a little way ahead on the path (further at speed) and steers towards it, and it slows
for bends so the sideways acceleration stays under 2 m/s^2. Used to test the world without a person at the
keyboard; the road users of step 2 follow lanes the same way.
"""
import math, os

from .paths import data_file


class Follower:
    def __init__(self, path, speed_factor=0.7):
        self.path = self._resample(path, 1.0)   # a point every metre, so the aim point sits at the right distance
        self.k = 0
        self.speed_factor = speed_factor

    @staticmethod
    def _resample(path, step):
        out = [path[0]]
        for (ax, ay), (bx, by) in zip(path, path[1:]):
            L = math.hypot(bx - ax, by - ay)
            for k in range(1, int(L // step) + 1):
                out.append((ax + (bx - ax) * k * step / L, ay + (by - ay) * k * step / L))
            if L % step > 1e-6:
                out.append((bx, by))
        return out

    @classmethod
    def our_route(cls, roadmap, start=None, lane="kerb"):
        """The lanes of our morning drive from the start edge on, joined into one path: the kerb lane of each
        road, or with lane="middle" the middle one."""
        from .roadmap import START_EDGE
        with open(data_file("step3/probe_route.txt")) as f:
            route = f.read().split()
        start = start or START_EDGE
        route = route[route.index(start):] if start in route else route
        pts = []
        for e in route:
            lanes = sorted((l for l in roadmap.lanes if l["edge"] == e and not l["internal"]), key=lambda l: l["index"])
            if lanes:
                pts += lanes[len(lanes) // 2 if lane == "middle" else 0]["line"]
        return cls(pts)

    def control(self, car, limit_kmh, world=None):
        p, n = self.path, len(self.path)
        while self.k < n - 2 and math.hypot(p[self.k + 1][0] - car.x, p[self.k + 1][1] - car.y) < \
                math.hypot(p[self.k][0] - car.x, p[self.k][1] - car.y) + 0.5:
            self.k += 1
        look = max(5.0, car.speed * 0.9)
        j, acc = self.k, 0.0
        while j < n - 1 and acc < look:
            acc += math.hypot(p[j + 1][0] - p[j][0], p[j + 1][1] - p[j][1]); j += 1
        tx, ty = p[j]
        alpha = math.atan2(ty - car.y, tx - car.x) - car.heading
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        Ld = max(1.0, math.hypot(tx - car.x, ty - car.y))
        delta = math.atan(2 * car.spec.wheelbase * math.sin(alpha) / Ld)
        limit = car.spec.max_steer if car.speed < 0.5 else min(car.spec.max_steer, math.atan(car.spec.max_lat_accel * car.spec.wheelbase / car.speed ** 2))
        steer = max(-1.0, min(1.0, delta / max(limit, 1e-3)))
        # slow for the sharpest bend in the next 30 m
        j, acc, bend = self.k, 0.0, 0.0
        while j < n - 2 and acc < 30:
            a1 = math.atan2(p[j + 1][1] - p[j][1], p[j + 1][0] - p[j][0]); a2 = math.atan2(p[j + 2][1] - p[j + 1][1], p[j + 2][0] - p[j + 1][0])
            seg = math.hypot(p[j + 1][0] - p[j][0], p[j + 1][1] - p[j][1]) or 1e-3
            bend = max(bend, abs(math.atan2(math.sin(a2 - a1), math.cos(a2 - a1))) / max(seg, 3.0))
            acc += seg; j += 1
        target = (limit_kmh or 50) / 3.6 * self.speed_factor
        if bend > 0:
            target = min(target, math.sqrt(2.0 / bend))
        if world is not None:   # keep a safe gap to whatever is ahead, and stop at red lights
            gap, lead = world.gap_ahead()
            if gap < 1e8:
                target = min(target, lead + math.sqrt(2 * 3.0 * max(0.0, gap - 2.5 - 0.9 * car.speed)))   # about a second behind
        done = self.k >= n - 2
        if done:
            target = 0.0
        throttle = 1.0 if car.speed < target - 0.5 else 0.0
        brake = min(1.0, (car.speed - target) / 2) if car.speed > target + 0.3 else 0.0
        if target < 0.5:   # hold the car still rather than creep
            throttle, brake = 0.0, 1.0 if car.speed > 0.05 else 0.0   # (a held brake at a standstill would reverse)
        return throttle, brake, steer, done

"""Car dynamics: a kinematic bicycle model with the car type's limits from step 2.

Position (x, y) is the centre of the car, heading is radians anticlockwise from east, speed is m/s along
the heading (negative when reversing). Controls each step: throttle and brake from 0 to 1, steer from -1
(full right) to 1 (full left), and an emergency brake.

Steering is limited the way a driver would limit it: the wheel turns at most 35 degrees, and at speed only
as far as keeps sideways acceleration under 7 m/s^2 (so a car at 50 km/h cannot turn on the spot). The
wheel turns at a finite rate and returns to centre when let go.
"""
import math
from dataclasses import dataclass


@dataclass
class CarSpec:
    length: float = 4.3
    width: float = 1.8
    wheelbase: float = 2.6
    max_speed: float = 27.78        # 100 km/h
    accel: float = 2.5              # m/s^2 from standstill
    decel: float = 8.0              # brake pedal to the floor (ABS on dry asphalt)
    emergency_decel: float = 9.0
    reverse_speed: float = 2.8      # 10 km/h
    max_steer: float = math.radians(35)
    max_lat_accel: float = 7.0       # near a car's grip limit on dry asphalt
    steer_rate: float = math.radians(200)   # how fast the wheel turns, per second
    rolling: float = 0.15           # m/s^2 lost when coasting
    drag: float = 0.0004            # per (m/s)^2


class Car:
    def __init__(self, x, y, heading, spec=None):
        self.spec = spec or CarSpec()
        self.x, self.y, self.heading = x, y, heading
        self.speed, self.steer, self.accel = 0.0, 0.0, 0.0
        self._reverse_hold = 0.0

    def step(self, dt, throttle=0.0, brake=0.0, steer=0.0, emergency=False, grip=1.0):
        s = self.spec
        v = self.speed
        if v >= -0.01 and brake > 0 and throttle == 0 and v < 0.05:   # holding brake at a standstill selects reverse
            self._reverse_hold += dt
        elif brake == 0:
            self._reverse_hold = 0.0
        a = 0.0
        if self._reverse_hold > 0.4:
            a = -s.accel * 0.6 * brake if v > -s.reverse_speed else 0.0
        else:
            if throttle > 0:
                a += s.accel * throttle * max(0.0, 1 - (v / s.max_speed) ** 2)
            if brake > 0 and v > 0:
                a -= s.decel * brake
            if v < 0 and throttle > 0:   # throttle while rolling backwards brakes first
                a = s.decel * throttle
        if emergency:
            a = -math.copysign(s.emergency_decel, v) if abs(v) > 0.05 else 0.0
        resist = (s.rolling + s.drag * v * v) * (1 if grip >= 1 else 6)   # off the road the ground drags
        if abs(v) > 0.01:
            a -= math.copysign(resist, v)
        nv = v + a * dt
        if v > 0 > nv or v < 0 < nv:   # braking stops the car, it does not reverse it
            nv = 0.0
        self.speed, self.accel = nv, a
        limit = s.max_steer
        if abs(nv) > 0.5:
            limit = min(limit, math.atan(s.max_lat_accel * s.wheelbase / (nv * nv)))
        target = max(-1.0, min(1.0, steer)) * limit
        rate = s.steer_rate * (1 if steer else 2)
        self.steer += max(-rate * dt, min(rate * dt, target - self.steer))
        self.heading = math.atan2(math.sin(self.heading + nv / s.wheelbase * math.tan(self.steer) * dt),
                                  math.cos(self.heading + nv / s.wheelbase * math.tan(self.steer) * dt))
        self.x += nv * math.cos(self.heading) * dt
        self.y += nv * math.sin(self.heading) * dt

    def corners(self):
        s, c, n = self.spec, math.cos(self.heading), math.sin(self.heading)
        hl, hw = s.length / 2, s.width / 2
        return [(self.x + c * dx - n * dy, self.y + n * dx + c * dy) for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]

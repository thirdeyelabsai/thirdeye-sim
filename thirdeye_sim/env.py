"""Driving in Gurgaon as a Gymnasium environment: follow a route through measured Indian traffic.

    import gymnasium as gym, thirdeye_sim.env
    env = gym.make("ThirdEye/Gurgaon-v0")                    # or with any of the options below
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

It follows the conventions of the environments AV researchers already use (STANDARDS.md): Gymnasium's API and
seeding, highway-env's layout for the road users around the car and its action types, MetaDrive's names for
what info reports. Units are SI; the car's frame is ISO 8855 (x ahead, y to the left).

The task: a route of about route_m metres from a random start on a main road (or options={"start": pose}),
both drawn from the episode's seed. The episode ends on reaching the route's end, a crash, leaving the road
entirely or straying off the route (30 m, or 8 m for 5 s); it is cut off after max_seconds.

Observation (observation="dict", the default; "flat" gives the same numbers as one vector, for MLP policies):
  "ego"         speed (m/s), heading against the route (rad), share of the car's corners on the road (0..1),
                speed limit (m/s), against the traffic (0/1), the light ahead (0 none, 1 green, 2 yellow,
                3 red), distance to its stop line (m, 100 when none), acceleration (m/s^2)
  "route"       metres left, bearing to the route 15 m on (rad, + left), offset from the route (m, + left),
                metres to the next turn (300 when none near), its side (+1 left, -1 right, 0 none)
  "road_users"  the nearest road users (rows), each [presence, x, y, vx, vy, cos_h, sin_h, class]: highway-env's
                Kinematics features in our car's frame, velocity relative to ours, normalised by highway-env's
                default ranges (x, y over 100 m, vx, vy over 40 m/s) unless normalize=False; class: 1 car,
                2 two-wheeler, 3 auto, 4 bus, 5 truck, 6 cycle, 7 pedestrian, 8 cow, 9 dog (over 9 when normalised)
  "hazard"      only with privileged=True: the simulator's own event hazard in eight zones round the car
                (hazard.py), privileged because it is the event generator's plan, not something a car can see

Actions (action=):
  "continuous"  [acceleration, steering] in [-1, 1], highway-env's ContinuousAction order: +1 is full throttle
                (up to 2.5 m/s^2), -1 full braking (8 m/s^2); steering +1 is full lock to the left (35 deg,
                less at speed, as a driver would)
  "discrete"    highway-env's DiscreteMetaAction: 0 LANE_LEFT, 1 IDLE, 2 LANE_RIGHT, 3 FASTER, 4 SLOWER, over a
                controller that follows the route (target speed in 2.5 m/s steps)

Reward: each term is in info["rewards"] (MetaDrive's terms where it has them): driving (metres of route made
good, x1), speed (0.1 x speed / limit per step), success (+10 at the route's end), crash (-5), out_of_road (-5),
red_light (-5 each), wrong_way (-1 per second), over_limit (-0.1 per second), jolt (-1 per speed breaker or
pothole taken too fast).

Info: crash, crash_vehicle, crash_human, crash_animal, crash_object (the wall at the map's edge), out_of_road,
off_route, arrive_dest, max_step, cost (1 on a step with a crash or leaving the road: MetaDrive's safe-RL cost),
red_light, wrong_way, over_limit, route_completion (0..1), speed_kmh, events (new ground-truth event records,
events.py), rewards, log.

Options at make(): chaos (1 = as measured), style ("gurgaon" or "generic"), scripted (scripted events on),
route_m, max_seconds, dt, vehicles (rows of road users), normalize, privileged, render_mode ("rgb_array").
"""
import math
import random

import gymnasium as gym
import numpy as np

from .actors import _project
from .hazard import hazard
from .routes import Route
from .world import World

KINDS = {"car": 1, "two_wheeler": 2, "auto": 3, "bus": 4, "truck": 5, "cycle": 6, "pedestrian": 7, "cow": 8, "dog": 9}
VEHICLES = ("car", "two_wheeler", "auto", "bus", "truck", "cycle")
LIGHT = {"green": 1, "yellow": 2, "red": 3}
SUB = 1 / 60                       # the simulator's step (s)
X_RANGE, V_RANGE = 100.0, 40.0     # highway-env's default feature ranges (5 x and 2 x its 20 m/s top speed)
REWARD = {"driving": 1.0, "speed": 0.1, "success": 10.0, "crash": -5.0, "out_of_road": -5.0, "red_light": -5.0,
          "wrong_way": -1.0, "over_limit": -0.1, "jolt": -1.0}


class GurgaonDriving(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, observation="dict", action="continuous", vehicles=8, route_m=800.0, max_seconds=180.0, dt=0.1,
                 chaos=1.0, style="gurgaon", scripted=True, normalize=True, privileged=False, render_mode=None, world=None):
        assert observation in ("dict", "flat") and action in ("continuous", "discrete")
        self.obs_kind, self.act_kind, self.n = observation, action, vehicles
        self.route_m, self.max_seconds, self.dt = route_m, max_seconds, dt
        self.normalize, self.privileged, self.render_mode = normalize, privileged, render_mode
        self.world = world or World(chaos=chaos, style=style)
        self.world.scripted = scripted
        f32 = np.float32
        ego = gym.spaces.Box(np.array([-5, -math.pi, 0, 0, 0, 0, 0, -10], f32), np.array([40, math.pi, 1, 40, 1, 3, 100, 10], f32), dtype=f32)
        route = gym.spaces.Box(np.array([0, -math.pi, -50, 0, -1], f32), np.array([5000, math.pi, 50, 300, 1], f32), dtype=f32)
        lo = np.full((vehicles, 8), -1.0 if normalize else -max(X_RANGE, V_RANGE), f32)
        hi = -lo
        lo[:, 0], hi[:, 0], lo[:, 7], hi[:, 7] = 0, 1, 0, (1 if normalize else 9)
        spaces = {"ego": ego, "route": route, "road_users": gym.spaces.Box(lo, hi, dtype=f32)}
        if privileged:
            spaces["hazard"] = gym.spaces.Box(0, 1, shape=(8,), dtype=f32)
        self._dict_space = gym.spaces.Dict(spaces)
        self.observation_space = self._dict_space if observation == "dict" else gym.spaces.flatten_space(self._dict_space)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=f32) if action == "continuous" else gym.spaces.Discrete(5)
        self.route = None
        self._view = None

    # an episode ------------------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)   # seeds self.np_random; the world, start and route are all drawn from it
        options = options or {}
        self.world.start = options.get("start", "random")
        for _ in range(20):   # a start whose route runs its full length before the map's edge (all from the seed)
            self.world.reset(seed=int(self.np_random.integers(2 ** 31)))
            w, e = self.world, self.world.ego
            lane = w.map.nearest_lane(e.x, e.y, e.heading, within=6.0)
            s0 = _project(w.map.lanes[lane[0]]["line"], e.x, e.y)[0]
            self.route = Route(w.map, lane[0], s0, self.route_m, random.Random(int(self.np_random.integers(2 ** 31))))
            if self.route.length >= 0.9 * self.route_m or "start" in options:
                break
        self.route.update(e.x, e.y)
        self._logged, self._known, self._crashes = len(w.log), (len(w.events.records) if w.events else 0), 0
        self._target_speed, self._lane_offset, self._off_since = 8.0, 0, None
        return self._obs(), self._info({}, [], [], False, False, False)

    def step(self, action):
        w, e = self.world, self.world.ego
        throttle, brake, steer = self._controls(action)
        before = self.route.progress
        for _ in range(max(1, round(self.dt / SUB))):
            w.step(SUB, throttle=throttle, brake=brake, steer=steer)
        self.route.update(e.x, e.y)
        new = [what for _, what in w.log[self._logged:]]
        self._logged = len(w.log)
        crashes = w.crashes[self._crashes:]
        self._crashes = len(w.crashes)
        st = w.status
        wall = any("hit the wall" in x for x in new)
        crashed = bool(crashes) or wall
        out = st["on_road"] == 0
        arrived = self.route.length - self.route.progress < 5.0
        self._off_since = (self._off_since if self._off_since is not None else w.t) if self.route.deviation > 8.0 else None
        off_route = self.route.deviation > 30.0 or (self._off_since is not None and w.t - self._off_since > 5.0)
        terms = {"driving": REWARD["driving"] * (self.route.progress - before),
                 "speed": REWARD["speed"] * max(0.0, e.speed) / max(1.0, (st.get("limit_kmh") or 50) / 3.6),
                 "success": REWARD["success"] if arrived else 0.0,
                 "crash": REWARD["crash"] if crashed else 0.0,
                 "out_of_road": REWARD["out_of_road"] if out else 0.0,
                 "red_light": REWARD["red_light"] * new.count("ran a red light"),
                 "wrong_way": REWARD["wrong_way"] * self.dt * bool(st.get("wrong_way")),
                 "over_limit": REWARD["over_limit"] * self.dt * bool(st.get("over_limit")),
                 "jolt": REWARD["jolt"] * sum(1 for x in new if "speed breaker" in x or "pothole" in x)}
        terminated = crashed or out or arrived or off_route
        truncated = not terminated and w.t >= self.max_seconds
        return self._obs(), float(sum(terms.values())), terminated, truncated, self._info(terms, crashes, new, arrived, off_route, truncated)

    # controls --------------------------------------------------------------------------------------------------
    def _controls(self, action):
        if self.act_kind == "continuous":
            a, s = float(np.clip(action[0], -1, 1)), float(np.clip(action[1], -1, 1))
            return max(0.0, a), max(0.0, -a), s
        return self._meta(int(action))

    def _meta(self, k):
        """highway-env's meta-actions over a simple controller: steer for a point on the route (shifted a lane's
        width for each lane change asked for), keep a target speed, and keep a safe gap to whatever is ahead."""
        w, e = self.world, self.world.ego
        lane = w.status.get("lane")
        left, right = w.map.beside.get(lane, (None, None)) if lane is not None else (None, None)
        if k == 0 and left is not None:
            self._lane_offset += 1
        elif k == 2 and right is not None:
            self._lane_offset -= 1
        elif k == 3:
            self._target_speed = min(25.0, self._target_speed + 2.5)
        elif k == 4:
            self._target_speed = max(0.0, self._target_speed - 2.5)
        if (left is None and self._lane_offset > 0) or (right is None and self._lane_offset < 0):
            self._lane_offset = 0
        _, _, s_here, _ = self.route._nearest(e.x, e.y)
        tx, ty, th = self.route.point(s_here + max(6.0, 0.9 * e.speed))
        width = w.map.lanes[lane]["w"] if lane is not None else 3.5
        tx, ty = tx - math.sin(th) * width * self._lane_offset, ty + math.cos(th) * width * self._lane_offset
        steer = pursue(e, tx, ty)
        gap, lead = w.gap_ahead()
        target = self._target_speed
        if gap < 1e8:
            target = min(target, lead + math.sqrt(2 * 3.0 * max(0.0, gap - 2.5 - 0.9 * e.speed)))
        a = (target - e.speed) / 1.5
        return (min(1.0, a / e.spec.accel), 0.0, steer) if a >= 0 else (0.0, min(1.0, -a / e.spec.decel), steer)

    # what the car knows ----------------------------------------------------------------------------------------
    def _obs(self):
        w, e = self.world, self.world.ego
        st = w.status
        remaining, bearing, offset, to_turn, side = self.route.guidance(e.x, e.y, e.heading)
        _, _, s_here, _ = self.route._nearest(e.x, e.y)
        rh = self.route.point(s_here)[2]
        ahead = w.signal_ahead()
        light, stop = 0, 100.0
        if ahead:
            light = LIGHT[ahead[2]]
            sx, sy = (ahead[1]["a"][0] + ahead[1]["b"][0]) / 2, (ahead[1]["a"][1] + ahead[1]["b"][1]) / 2
            stop = min(100.0, max(0.0, (sx - e.x) * math.cos(e.heading) + (sy - e.y) * math.sin(e.heading)))
        ego = [e.speed, math.atan2(math.sin(e.heading - rh), math.cos(e.heading - rh)), st["on_road"] / 4,
               (st.get("limit_kmh") or 50) / 3.6, float(st.get("wrong_way", False)), light, stop, e.accel]
        route = [min(5000.0, remaining), bearing, max(-50.0, min(50.0, offset)), to_turn, side]
        users = np.zeros((self.n, 8), dtype=np.float32)
        if w.traffic:
            c, n = math.cos(e.heading), math.sin(e.heading)
            rows = []
            for u in w.traffic.all():
                dx, dy = u.x - e.x, u.y - e.y
                x, y = dx * c + dy * n, -dx * n + dy * c
                if x * x + y * y > X_RANGE * X_RANGE:
                    continue
                v = getattr(u, "v", None)
                v = getattr(u, "speed", 0.0) if v is None else v
                rel = u.heading - e.heading
                rows.append((x * x + y * y, [1.0, x, y, v * math.cos(rel) - e.speed, v * math.sin(rel),
                                             math.cos(rel), math.sin(rel), KINDS.get(u.kind, 1)]))
            rows.sort(key=lambda r: r[0])
            for i, (_, r) in enumerate(rows[:self.n]):
                users[i] = r
            if self.normalize:
                users[:, 1:3] = np.clip(users[:, 1:3] / X_RANGE, -1, 1)
                users[:, 3:5] = np.clip(users[:, 3:5] / V_RANGE, -1, 1)
                users[:, 7] /= 9.0
        space = self._dict_space
        obs = {"ego": np.clip(np.array(ego, np.float32), space["ego"].low, space["ego"].high),
               "route": np.clip(np.array(route, np.float32), space["route"].low, space["route"].high),
               "road_users": users}
        if self.privileged:
            obs["hazard"] = np.array(hazard(w), dtype=np.float32)
        return obs if self.obs_kind == "dict" else gym.spaces.flatten(space, obs)

    def _info(self, terms, crashes, new, arrived, off_route, truncated):
        w = self.world
        kinds = [c["kind"] for c in crashes]
        wall = any("hit the wall" in x for x in new)
        crash_vehicle = any(k in VEHICLES for k in kinds)
        crash_human, crash_animal = "pedestrian" in kinds, any(k in ("cow", "dog") for k in kinds)
        out = w.status.get("on_road", 4) == 0
        events = w.events.records[self._known:] if w.events else []
        self._known += len(events)
        return {"crash": crash_vehicle or crash_human or crash_animal or wall, "crash_vehicle": crash_vehicle,
                "crash_human": crash_human, "crash_animal": crash_animal, "crash_object": wall, "out_of_road": out,
                "off_route": off_route, "arrive_dest": arrived, "max_step": truncated,
                "cost": float(bool(kinds) or wall or out), "red_light": new.count("ran a red light"),
                "wrong_way": bool(w.status.get("wrong_way")), "over_limit": bool(w.status.get("over_limit")),
                "route_completion": self.route.completion if self.route else 0.0, "speed_kmh": w.ego.speed * 3.6,
                "events": [dict(r) for r in events], "rewards": dict(terms), "log": list(new)}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._view is None:
            import pygame
            pygame.font.init()
            from .render3d import Renderer3D
            self._view = Renderer3D(self.world, size=(960, 600), scale=1.0)
        self._view.draw()
        W, H = self._view.fb_size
        img = np.frombuffer(self._view.out.read(components=3), dtype=np.uint8).reshape(H, W, 3)
        return img[::-1].copy()


def pursue(e, tx, ty):
    """Pure pursuit: the steering (-1..1, + left) that brings the car e through the point (tx, ty)."""
    alpha = math.atan2(ty - e.y, tx - e.x) - e.heading
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
    d = max(1.0, math.hypot(tx - e.x, ty - e.y))
    limit = e.spec.max_steer if e.speed < 0.5 else min(e.spec.max_steer, math.atan(e.spec.max_lat_accel * e.spec.wheelbase / e.speed ** 2))
    return max(-1.0, min(1.0, math.atan(2 * e.spec.wheelbase * math.sin(alpha) / d) / max(limit, 1e-3)))


# no max_episode_steps: the env cuts an episode off itself at max_seconds (as highway-env does with its
# "duration"), so the time limit stays the same whatever dt, and max_seconds=... at make() is never overridden
gym.register(id="ThirdEye/Gurgaon-v0", entry_point="thirdeye_sim.env:GurgaonDriving")

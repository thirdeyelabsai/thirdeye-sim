# Dream: the Gurgaon driving environment

A Gymnasium environment for driving a car along a route on Gurgaon's real road network (OpenStreetMap), in
traffic that behaves the way Indian traffic does and with events at the rates we measured from Indian dashcam
footage. It follows the conventions AV researchers already use, so existing code runs on it unchanged:
Gymnasium's API and seeding, highway-env's observation and action layouts, MetaDrive's info keys and reward
terms, and nuPlan-style and CARLA-style planner scores. [STANDARDS.md](../STANDARDS.md) lists every standard,
what we follow exactly, and what we add.

## Install

```
pip install "thirdeyelabs[sim]"        # gymnasium, numpy, shapely (installs thirdeye-sim)
pip install "thirdeye-sim[render]"     # adds the 3D view (pygame, moderngl, pyproj)
```

Python 3.10 or later. The road map ships built; rebuilding it from the SUMO network (in a checkout of the
repository) needs `.[map]`.

## Quick start

```python
import gymnasium as gym
import thirdeye_sim.env            # registers ThirdEye/Gurgaon-v0

env = gym.make("ThirdEye/Gurgaon-v0")
obs, info = env.reset(seed=0)
while True:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
print(info["route_completion"], info["rewards"])
```

Vector environments work as usual (on macOS, guard the script with `if __name__ == "__main__":`):

```python
envs = gym.make_vec("ThirdEye/Gurgaon-v0", num_envs=8, vectorization_mode="async")
```

## The task

Each episode draws a start on a main road and a route of about `route_m` metres from it, both from the episode's
seed: the same seed gives the same start, route, traffic and events, in any process. Routes keep off the wall
round the map and off junction corners tighter than a car can turn.

The episode **terminates** when the car reaches the route's end, crashes (a road user, or the wall at the map's
edge), leaves the road entirely, or strays off the route (30 m from it, or 8 m for 5 s). It is **truncated** at
`max_seconds`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `observation` | `"dict"` | `"dict"`, or `"flat"` for the same numbers as one vector |
| `action` | `"continuous"` | `"continuous"` or `"discrete"` (below) |
| `vehicles` | 8 | rows of road users in the observation |
| `route_m` | 800 | route length (m) |
| `max_seconds` | 180 | time limit (s); the env applies it itself, whatever `dt` |
| `dt` | 0.1 | seconds per env step (the simulator steps at 1/60 s inside) |
| `style` | `"gurgaon"` | `"gurgaon"`: measured Indian behaviour and events; `"generic"`: lane discipline, no filtering, no honking, no scripted events, for comparison |
| `chaos` | 1.0 | 1.0 is as measured; other values scale event rates and how pushy drivers are, as a stress setting, never presented as measured |
| `scripted` | True | scripted events on (False gives the pure traffic model) |
| `normalize` | True | normalise road-user features as highway-env does |
| `privileged` | False | add the `"hazard"` observation (below) |
| `render_mode` | None | `"rgb_array"` returns the 3D view (needs `.[render]`) |

## Observation

A `Dict` (or the same values flattened):

| Key | Shape | Contents |
|---|---|---|
| `ego` | 8 | speed (m/s), heading against the route (rad), share of the car's corners on the road (0 to 1), speed limit (m/s), against the traffic (0 or 1), the light ahead (0 none, 1 green, 2 yellow, 3 red), distance to its stop line (m, 100 when none), acceleration (m/s²) |
| `route` | 5 | metres left, bearing to the route 15 m on (rad, + left), offset from the route (m, + left), metres to the next turn (300 when none near), its side (+1 left, -1 right, 0 none) |
| `road_users` | `vehicles` x 8 | the nearest road users within 100 m, nearest first, each `[presence, x, y, vx, vy, cos_h, sin_h, class]` |
| `hazard` | 8 | only with `privileged=True` (below) |

`road_users` uses highway-env's Kinematics features in the car's frame (ISO 8855: x ahead, y to the left), with
velocity relative to ours. With `normalize=True`, x and y are divided by 100 m and vx, vy by 40 m/s (highway-env's
default ranges), clipped to [-1, 1]. `class` is our one added feature: 1 car, 2 two-wheeler, 3 auto-rickshaw,
4 bus, 5 truck, 6 cycle, 7 pedestrian, 8 cow, 9 dog (divided by 9 when normalised). Empty rows are all zeros.

`hazard` is the simulator's own event hazard: for eight zones round the car (front, then anticlockwise), the
chance that the event generator involves that zone in the next 3 s, from its rates and the events already
under way. It is privileged information (the generator's plan, which no real car can see), useful for studying
what a planner gains from knowing it. It is not a prediction of real-world risk: we have not yet measured how
the scene changes the rates.

## Actions

**Continuous** (highway-env's ContinuousAction order): `[acceleration, steering]`, each in [-1, 1]. +1 is full
throttle (up to 2.5 m/s²), -1 full braking (8 m/s²). Steering +1 is full lock to the left (35 degrees, less at
speed, as a driver would).

**Discrete** (highway-env's DiscreteMetaAction): 0 LANE_LEFT, 1 IDLE, 2 LANE_RIGHT, 3 FASTER, 4 SLOWER, over a
controller that follows the route and keeps a target speed in 2.5 m/s steps.

## Reward

The reward is the sum of these terms, each reported in `info["rewards"]` (MetaDrive's terms where it has them):

| Term | Weight | When |
|---|---|---|
| `driving` | +1 | per metre of route made good |
| `speed` | +0.1 | x speed / limit, per step |
| `success` | +10 | at the route's end |
| `crash` | -5 | a crash |
| `out_of_road` | -5 | leaving the road |
| `red_light` | -5 | each red light run |
| `wrong_way` | -1 | per second against the traffic |
| `over_limit` | -0.1 | per second over the limit |
| `jolt` | -1 | each speed breaker or pothole taken too fast |

## Info

`crash`, `crash_vehicle`, `crash_human`, `crash_animal`, `crash_object` (the wall), `out_of_road`, `off_route`,
`arrive_dest`, `max_step`, `cost` (1 on a step with a crash or leaving the road: MetaDrive's safe-RL cost),
`red_light`, `wrong_way`, `over_limit`, `route_completion` (0 to 1), `speed_kmh`, `events` (new ground-truth
event records, each saying whether the event was scripted or arose from traffic), `rewards`, `log`.

## Scoring a planner

```
python -m thirdeye_sim.evaluate --episodes 20 --out results/benchmark
```

It drives every planner in `thirdeye_sim/planners.py` over the same 20 seeded routes in Gurgaon traffic and in
generic traffic, and reports two scores beside the raw counts, each with a 95% confidence interval over routes:

- **nuPlan-style closed-loop score** (reactive agents), our implementation of nuPlan's CLS. Collisions are
  classified as nuPlan does (our car stopped, the other stopped, front, rear, side), and only at-fault ones
  count. The reference progress is the IDM planner's, since we have no human drive of these routes.
- **CARLA Leaderboard 2.1-style driving score**: route completion times the infraction penalty.

`thirdeye_sim/evaluate.py` says exactly where each differs from the original. To score your own planner, add a
class with `act()` to `PLANNERS` in `planners.py`.

### Baselines (30 Oct 2026, 100 routes of 800 m per cell)

| Planner / traffic | nuPlan-style CLS | CARLA-style DS | routes finished | routes ending in a crash | collisions per km | mean speed |
|---|---|---|---|---|---|---|
| IDM planner / generic | 0.73 ± 0.06 | 83.2 ± 6.3 | 78 of 100 | 14 | 0.18 | 19 km/h |
| IDM planner / Gurgaon | 0.62 ± 0.07 | 72.8 ± 7.3 | 64 of 100 | 27 | 0.39 | 17 km/h |
| lane keeping only / generic | 0.13 ± 0.06 | 29.8 ± 5.9 | 14 of 100 | 84 | 2.31 | 27 km/h |
| lane keeping only / Gurgaon | 0.05 ± 0.04 | 23.1 ± 4.6 | 7 of 100 | 93 | 2.97 | 27 km/h |

The IDM planner is the usual rule-based baseline (nuPlan ships one): pure pursuit along the route and the
Intelligent Driver Model for speed, with Treiber and Kesting's city parameters. It reacts only to what is in its
path, as nuPlan's does. On the same 100 routes it crashes about twice as often in Gurgaon traffic as in
generic traffic, and finishes fewer routes. The full table, with red lights, hard braking and how each episode
ended, is in `results/benchmark.md`.

## Speed

On an Apple M5 laptop, with random actions and their frequent resets: one env runs at 174 to 271 env steps per
second (each step is 0.1 s of driving), and 8 async vector envs at 556 to 677 in total, depending on the traffic
round the car and the machine's load. One env uses about 210 MB of memory and starts in under a second.

## What is measured and what is assumed

Event rates come from labelled dashcam clips, with exact Poisson 95% confidence intervals (STANDARDS.md,
section 4). Some things are assumed until we can measure them, and are marked as such in the code: lane counts
on roads OpenStreetMap leaves untagged (two each way on secondary and tertiary roads), speed breakers and bus
stops where OSM maps none, potholes, the rates of the cow, dog and U-turn events, and how long a cow stands.

Known map limits: OpenStreetMap tags almost no bridges or tunnels in Gurgaon. We treat a road crossing a motorway
without a junction as passing under or over it (OSM defines motorways as grade separated), but the 3D view still
draws both at street level there.

## Licences

Code: Apache-2.0. Road network: © OpenStreetMap contributors, ODbL. Event rates and labels: CC BY 4.0.

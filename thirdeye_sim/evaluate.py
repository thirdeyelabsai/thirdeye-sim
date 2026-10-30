"""Score a driving policy in the environment, closed loop, the way the AV benchmarks do (STANDARDS.md).

    .venv-sumo/bin/python -m thirdeye_sim.evaluate --episodes 20 --out results/benchmark
    (every planner in planners.py, in Gurgaon traffic and in generic traffic, 20 seeded routes each)

For each episode (a seeded route of 800 m, traffic reacting to the car) it records raw counts and two scores:

  nuPlan-style closed-loop score (reactive agents), our implementation of nuPlan's CLS:
    multipliers  no at-fault collision (0 with a road user, 0.5 with the wall, else 1; at fault as nuPlan
                 classifies collisions: the other was stopped, our front bumper hit it, or a side collision with
                 our car across two lanes; not when ours was stopped or was hit from behind), drivable area (0 if the car ever left the road entirely), driving direction (0 if
                 it drove over 6 m against the traffic, 0.5 over 2 m), making progress (0 if under 0.2 of the
                 reference progress)
    weighted     ego progress (against the reference, capped at 1) x5, time to collision within bound (min TTC to
                 what is ahead never under 0.95 s while moving) x5, speed-limit compliance (1 minus the mean
                 overspeed over 2.23 m/s) x4, comfort (nuPlan's limits: longitudinal acceleration -4.05 to 2.40,
                 lateral 4.89 m/s^2, yaw rate 0.95 rad/s, jerk 4.13 m/s^3) x2
    The reference is the IDM planner's progress on the same seed (nuPlan uses the human driver's; we have no
    human drive of these routes), so the IDM planner's own ego progress is 1 by definition. Comfort is judged on
    acceleration and yaw rate smoothed over 0.5 s (nuPlan smooths too). A route may take up to 300 s.
  CARLA Leaderboard 2.1-style driving score: route completion x 1 / (1 + sum of coefficient x count), with
    pedestrian 1.0 (animals counted as pedestrians), vehicle 0.70, static (the wall) 0.60, red light 0.40,
    timeout 0.4. One difference: our episodes end at the first collision, CARLA's route runs on.

Results: the mean of each over the episodes with a 95% confidence interval, and counts per kilometre driven.
"""
import argparse, json, math, os, statistics, sys, time
from multiprocessing import Pool

import gymnasium as gym
import numpy as np

from . import env as _env   # registers ThirdEye/Gurgaon-v0
from .planners import PLANNERS

COMFORT = {"lon_min": -4.05, "lon_max": 2.40, "lat": 4.89, "yaw": 0.95, "jerk": 4.13}   # nuPlan
TTC_MIN, OVERSPEED = 0.95, 2.23
CARLA = {"pedestrian": 1.0, "vehicle": 0.70, "static": 0.60, "red_light": 0.40, "timeout": 0.4}
HARD_BRAKES = (3.0, 4.4)   # m/s^2: no standard threshold exists, so two (about 0.3 g and 0.45 g)
TIME_LIMIT = 300.0         # s for a route (800 m at Gurgaon's average of about 23 km/h takes about 2 minutes)
SMOOTH = 5                 # samples (0.5 s) to smooth acceleration and yaw rate over before the comfort limits
T95 = {5: 2.571, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def episode(args):
    """One seeded episode: raw measurements and both scores' parts."""
    planner_key, style, seed, route_m = args
    env = gym.make("ThirdEye/Gurgaon-v0", style=style, route_m=route_m, max_seconds=TIME_LIMIT).unwrapped
    env.reset(seed=seed)
    planner = PLANNERS[planner_key](env)
    w, e = env.world, env.world.ego
    m = {"collisions": {"vehicle": 0, "pedestrian": 0, "animal": 0, "static": 0}, "at_fault": False, "static_hit": False,
         "red_lights": 0, "off_road": False, "wrong_way_m": 0.0, "overspeed": [], "min_ttc": 1e9,
         "hard_brakes": {str(h): 0 for h in HARD_BRAKES}, "comfortable": True, "jolts": 0, "distance": 0.0}
    from collections import deque
    accs, yaws = deque(maxlen=SMOOTH), deque(maxlen=SMOOTH)   # (comfort on smoothed signals, as nuPlan)
    prev_acc, prev_h, braking = 0.0, e.heading, {h: False for h in HARD_BRAKES}
    info, steps = {}, 0
    while True:
        x0, y0 = e.x, e.y
        _, _, terminated, truncated, info = env.step(np.array(planner.act(), dtype=np.float32))
        steps += 1
        dt = env.dt
        d = math.hypot(e.x - x0, e.y - y0)
        m["distance"] += d
        for c in w.crashes[-1:] if info["crash"] else []:
            kind = "pedestrian" if c["kind"] == "pedestrian" else "animal" if c["kind"] in ("cow", "dog") else "vehicle"
            m["collisions"][kind] += 1
            m["at_fault"] |= c["at_fault"]
            m.setdefault("collision_types", []).append(c["type"])
        if info["crash_object"]:
            m["collisions"]["static"] += 1
            m["static_hit"] = True
        m["red_lights"] += info["red_light"]
        m["off_road"] |= info["out_of_road"]
        m["wrong_way_m"] += d if info["wrong_way"] else 0.0
        limit = (w.status.get("limit_kmh") or 50) / 3.6
        m["overspeed"].append(max(0.0, e.speed - limit))
        gap, lead = w.gap_ahead(look=80.0, lights=False)   # (time to collision: road users only, as nuPlan)
        if gap < 1e8 and e.speed > 0.2 and e.speed > lead + 0.1:
            m["min_ttc"] = min(m["min_ttc"], gap / (e.speed - lead))
        accs.append(e.accel)
        yaws.append(math.atan2(math.sin(e.heading - prev_h), math.cos(e.heading - prev_h)) / dt)
        acc, yaw = statistics.fmean(accs), statistics.fmean(yaws)
        jerk = (acc - prev_acc) / dt
        if len(accs) == SMOOTH and (not (COMFORT["lon_min"] <= acc <= COMFORT["lon_max"]) or abs(e.speed * yaw) > COMFORT["lat"] or
                                    abs(yaw) > COMFORT["yaw"] or abs(jerk) > COMFORT["jerk"]):
            m["comfortable"] = False
        for h in HARD_BRAKES:   # hard braking: on the car's own deceleration, not smoothed
            if e.accel <= -h and not braking[h]:
                m["hard_brakes"][str(h)] += 1
            braking[h] = e.accel <= -h
        prev_acc, prev_h = acc, e.heading
        m["jolts"] += sum(1 for x in info["log"] if "speed breaker" in x or "pothole" in x)
        if terminated or truncated:
            break
    m.update(planner=planner_key, style=style, seed=seed, steps=steps, seconds=round(w.t, 1),
             progress=env.route.progress, route_length=env.route.length, completion=env.route.completion,
             arrived=info["arrive_dest"], timeout=info["max_step"], ended=next((k for k in ("crash", "out_of_road", "off_route", "arrive_dest", "max_step") if info.get(k)), "?"),
             mean_overspeed=statistics.fmean(m.pop("overspeed")) if steps else 0.0,
             min_ttc=m["min_ttc"] if m["min_ttc"] < 1e9 else None)
    env.close()
    return m


def scores(ep, reference):
    """nuPlan-style CLS and CARLA LB 2.1-style DS of one episode (reference: the IDM planner's progress)."""
    c = ep["collisions"]
    road_users = c["vehicle"] + c["pedestrian"] + c["animal"]
    fault = 0.0 if (road_users and ep["at_fault"]) else 0.5 if c["static"] else 1.0
    drivable = 0.0 if ep["off_road"] else 1.0
    direction = 0.0 if ep["wrong_way_m"] > 6 else 0.5 if ep["wrong_way_m"] > 2 else 1.0
    ratio = min(1.0, ep["progress"] / max(1.0, reference))
    progressing = 1.0 if ratio >= 0.2 else 0.0
    ttc = 1.0 if ep["min_ttc"] is None or ep["min_ttc"] >= TTC_MIN else 0.0
    speed = max(0.0, 1.0 - ep["mean_overspeed"] / OVERSPEED)
    comfort = 1.0 if ep["comfortable"] else 0.0
    cls = fault * drivable * direction * progressing * (5 * ratio + 5 * ttc + 4 * speed + 2 * comfort) / 16
    penalty = 1.0 / (1.0 + CARLA["pedestrian"] * (c["pedestrian"] + c["animal"]) + CARLA["vehicle"] * c["vehicle"] +
                     CARLA["static"] * c["static"] + CARLA["red_light"] * ep["red_lights"] + CARLA["timeout"] * ep["timeout"])
    return {"nuplan_cls": cls, "carla_ds": 100 * ep["completion"] * penalty, "ego_progress": ratio, "ttc_ok": ttc,
            "speed_compliance": speed, "comfortable": comfort, "no_at_fault_collision": fault}


def ci(xs):
    """Mean and the half-width of its 95% confidence interval (Student's t)."""
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    t = T95.get(n, T95[min(T95, key=lambda k: abs(k - n))] if n < 30 else 1.96)
    return statistics.fmean(xs), t * statistics.stdev(xs) / math.sqrt(n)


def summarise(eps):
    """Per planner and style: scores (mean and 95% CI) and counts per km."""
    ref = {(e["style"], e["seed"]): e["progress"] for e in eps if e["planner"] == "idm"}
    out = {}
    for key in sorted({(e["planner"], e["style"]) for e in eps}):
        group = [e for e in eps if (e["planner"], e["style"]) == key]
        sc = [scores(e, ref.get((e["style"], e["seed"]), e["progress"])) for e in group]
        km = sum(e["distance"] for e in group) / 1000
        row = {name: ci([s[name] for s in sc]) for name in sc[0]}
        row.update({"episodes": len(group), "km": km, "arrived": sum(e["arrived"] for e in group),
                    "ended": {k: sum(e["ended"] == k for e in group) for k in ("arrive_dest", "crash", "out_of_road", "off_route", "max_step")},
                    "per_km": {"collisions": sum(sum(e["collisions"].values()) for e in group) / max(km, 1e-9),
                               "red_lights": sum(e["red_lights"] for e in group) / max(km, 1e-9),
                               "hard_brakes_3.0": sum(e["hard_brakes"]["3.0"] for e in group) / max(km, 1e-9),
                               "hard_brakes_4.4": sum(e["hard_brakes"]["4.4"] for e in group) / max(km, 1e-9),
                               "jolts": sum(e["jolts"] for e in group) / max(km, 1e-9)},
                    "mean_speed_kmh": 3.6 * sum(e["distance"] for e in group) / max(1e-9, sum(e["seconds"] for e in group))})
        out[f"{key[0]} / {key[1]}"] = row
    return out


def markdown(summary, meta):
    f = lambda m: f"{m[0]:.2f} ± {m[1]:.2f}"
    lines = [f"# Planner benchmark ({meta['episodes']} seeded routes of {meta['route_m']:.0f} m per cell)", "",
             "| Planner / traffic | nuPlan-style CLS | CARLA-style DS | routes finished | collisions per km | red lights per km | hard brakes per km (3.0 / 4.4 m/s²) | mean speed |",
             "|---|---|---|---|---|---|---|---|"]
    for k, r in summary.items():
        p = r["per_km"]
        lines.append(f"| {k} | {f(r['nuplan_cls'])} | {r['carla_ds'][0]:.1f} ± {r['carla_ds'][1]:.1f} | {r['arrived']} of {r['episodes']} | "
                     f"{p['collisions']:.2f} | {p['red_lights']:.2f} | {p['hard_brakes_3.0']:.1f} / {p['hard_brakes_4.4']:.1f} | {r['mean_speed_kmh']:.0f} km/h |")
    lines += ["", "± is the half-width of a 95% confidence interval over the routes. How episodes ended:", ""]
    for k, r in summary.items():
        lines.append(f"- {k}: " + ", ".join(f"{n} {name.replace('_', ' ')}" for name, n in r["ended"].items() if n))
    lines += ["", "Scores are our implementations of the nuPlan closed-loop score and the CARLA Leaderboard 2.1 driving score "
              "(thirdeye_sim/evaluate.py says where they differ); the reference progress is the IDM planner's."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--route", type=float, default=800.0)
    ap.add_argument("--planners", default=",".join(PLANNERS))
    ap.add_argument("--styles", default="generic,gurgaon")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/benchmark")
    a = ap.parse_args()
    jobs = [(p, s, seed, a.route) for p in a.planners.split(",") for s in a.styles.split(",") for seed in range(a.episodes)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        eps = pool.map(episode, jobs)
    summary = summarise(eps)
    meta = {"episodes": a.episodes, "route_m": a.route, "seconds": round(time.time() - t0), "python": sys.version.split()[0]}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"meta": meta, "summary": summary, "episodes": eps}, open(a.out + ".json", "w"), indent=1, default=float)
    open(a.out + ".md", "w").write(markdown(summary, meta))
    print(markdown(summary, meta))
    print(f"({meta['seconds']} s)")


if __name__ == "__main__":
    main()

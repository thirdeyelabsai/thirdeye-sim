"""Check the simulated traffic against our dashcam footage: how often each event happens, and how fast cars go.

    python -m thirdeye_sim.realism --drives 48 --out results/realism

The reference is thirdeye_sim/data/measured.json, summaries of our labelled footage (tools/measured_from_footage.py).

Events. In the footage, a person labelled random 4-second clips with the one main event that starts in each (the
highest in the taxonomy's list when several do). The simulator is counted the same way: each drive is cut into
4-second windows, and a window with several events counts only the highest in that list. Each measured rate has
an exact Poisson 95% interval; a simulated rate inside it matches. The drives are the IDM baseline planner
(planners.py) on seeded routes in Gurgaon traffic, once with the event generator on and once with it off. The
generator aims at the measured rates, so the run with it off is the independent check of the traffic model.

Speeds. Our car's GPS speed once a second, against the speed of every car going our way within 60 m of our car
in the simulated traffic, once a second.

Not checked: the Waymo Open Sim Agents Challenge features that need every road user's path in metres (their
speeds and accelerations, gaps, time to collision, distance to the road edge). Our footage comes from one
camera, so we cannot measure them yet.
"""
import argparse, json, math, os, sys, time
from multiprocessing import Pool

import gymnasium as gym
import numpy as np

from . import env as _env   # registers ThirdEye/Gurgaon-v0
from .events import ASSUMED, PLACEHOLDER, RATES
from .planners import IDMPlanner

HERE = os.path.dirname(os.path.abspath(__file__))
NEAR = 60.0                 # m: cars this close to ours, going our way, are the traffic we drive in
SAME_WAY = math.radians(45)
TIME_LIMIT, ROUTE = 300.0, 3000.0
# our event labels that are a different name in the taxonomy (for the priority order only)
AS = {"cow_on_road": "animal_on_road", "dog_crossing": "animal_on_road", "u_turn": "crossing_vehicle"}
NAMES = {"two_wheeler_close": "Two-wheeler squeezing past", "overtaken_right": "Overtaken on the right",
         "overtaken_left": "Overtaken on the left", "pedestrian_crossing": "Pedestrian crossing in front",
         "wrong_way_in_lane": "Vehicle coming the wrong way", "lead_brake": "Car ahead braking hard",
         "slow_vehicle_ahead": "Slow vehicle ahead", "cut_in_left": "Cut-in from the left", "pull_out": "Vehicle pulling out",
         "pedestrian_on_road": "Pedestrian walking in the road", "cut_in_right": "Cut-in from the right",
         "red_light_violation": "Red light jumped", "cow_on_road": "Cow standing in the road",
         "dog_crossing": "Dog running across", "u_turn": "U-turn across the road"}


def measured():
    with open(os.path.join(HERE, "data", "measured.json")) as f:
        return json.load(f)


def poisson_interval(n, hours, level=0.95):
    """Exact (Garwood) interval for a Poisson count n over hours, per hour."""
    def cdf(k, lam):
        s, t = 0.0, math.exp(-lam)
        for i in range(k + 1):
            s += t
            t *= lam / (i + 1)
        return s
    def solve(f, lo=0.0, hi=50.0 + 10.0 * n):
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if f(mid) else (mid, hi)
        return (lo + hi) / 2
    a = (1 - level) / 2
    low = 0.0 if n == 0 else solve(lambda lam: 1 - cdf(n - 1, lam) >= a)
    high = solve(lambda lam: cdf(n, lam) <= a)
    return low / hours, high / hours


def drive(args):
    """One seeded drive: when each event happened, and the speeds of the cars going our way near us."""
    scripted, seed = args
    env = gym.make("ThirdEye/Gurgaon-v0", style="gurgaon", scripted=scripted, route_m=ROUTE, max_seconds=TIME_LIMIT).unwrapped
    env.reset(seed=seed)
    planner, w, e = IDMPlanner(env), env.world, env.world.ego
    kmh, ours, nxt = [], [], 1.0
    while True:
        _, _, terminated, truncated, _ = env.step(np.array(planner.act(), np.float32))
        if w.t >= nxt:
            nxt += 1.0
            ours.append(e.speed * 3.6)
            for u in w.traffic.vehicles:
                dh = abs(math.atan2(math.sin(u.heading - e.heading), math.cos(u.heading - e.heading)))
                if u.kind == "car" and not u.script and dh < SAME_WAY and math.hypot(u.x - e.x, u.y - e.y) < NEAR:
                    kmh.append(max(0.0, u.v) * 3.6)
        if terminated or truncated:
            break
    happened = [(r.get("t_end", r["t"]), r["label"], r["source"]) for r in w.events.records
                if r["outcome"] in ("seen", "done", "crash")]   # (as the monitor saw it, or as a scripted one played out)
    return {"scripted": scripted, "seed": seed, "seconds": w.t, "events": happened, "car_kmh": kmh, "our_kmh": ours}


def main_events(events, priority, clip):
    """One event per window of clip seconds, the highest in the labellers' list, as the footage was labelled."""
    rank = lambda lab: priority.index(AS.get(lab, lab)) if AS.get(lab, lab) in priority else len(priority)
    win = {}
    for t, label, source in events:
        k = int(t // clip)
        if k not in win or rank(label) < rank(win[k][0]):
            win[k] = (label, source)
    return list(win.values())


def speed_summary(s, step, top):
    s = sorted(s)
    q = lambda p: s[int(p * (len(s) - 1))]
    hist = [0] * (top // step + 1)
    for x in s:
        hist[min(top // step, int(x // step))] += 1
    return {"hours": round(len(s) / 3600, 2), "mean_kmh": round(sum(s) / len(s), 1), "median_kmh": round(q(0.5), 1),
            "p85_kmh": round(q(0.85), 1), "stopped_share": round(sum(x < 2 for x in s) / len(s), 4),
            "hist": [round(h / len(s), 4) for h in hist]}


def compare(drives, ref):
    ev = ref["events"]
    hours_ref = ev["clips"] * ev["clip_seconds"] / 3600
    on = [d for d in drives if d["scripted"]]
    off = [d for d in drives if not d["scripted"]]
    h_on, h_off = sum(d["seconds"] for d in on) / 3600, sum(d["seconds"] for d in off) / 3600
    count = lambda ds: [x for d in ds for x in main_events(d["events"], ev["priority"], ev["clip_seconds"])]
    c_on, c_off = count(on), count(off)
    rows = []
    for label in RATES:
        n = ev["counts"].get(label, 0)
        low, high = poisson_interval(n, hours_ref)
        sim = sum(1 for lab, _ in c_on if lab == label) / h_on
        alone = sum(1 for lab, _ in c_off if lab == label) / h_off
        kind = "assumed" if label in ASSUMED else "placeholder" if label in PLACEHOLDER else "measured"
        rows.append({"event": label, "name": NAMES.get(label, label), "rate": kind,
                     "measured_per_hour": None if kind == "assumed" else round(n / hours_ref, 1),
                     "low": None if kind == "assumed" else round(low, 1), "high": None if kind == "assumed" else round(high, 1),
                     "sim_per_hour": round(sim, 1),
                     "sim_from_traffic_per_hour": round(sum(1 for lab, src in c_on if lab == label and src == "traffic") / h_on, 1),
                     "traffic_alone_per_hour": round(alone, 1),
                     "match": None if kind == "assumed" else low <= sim <= high,
                     "traffic_alone_match": None if kind == "assumed" else low <= alone <= high})
    sp = ref["speeds"]
    speeds = {"real_play_area": sp["play_area"], "real_delhi_ncr": sp["delhi_ncr"],
              "sim_cars": speed_summary([x for d in on for x in d["car_kmh"]], sp["step_kmh"], sp["top_kmh"]),
              "sim_our_car": speed_summary([x for d in on for x in d["our_kmh"]], sp["step_kmh"], sp["top_kmh"])}
    checked = [r for r in rows if r["match"] is not None]
    return {"events": rows, "speeds": speeds,
            "summary": {"events_checked": len(checked), "events_matching": sum(r["match"] for r in checked),
                        "events_matching_traffic_alone": sum(r["traffic_alone_match"] for r in checked),
                        "hours_driven": round(h_on, 2), "hours_driven_traffic_alone": round(h_off, 2),
                        "drives": len(on), "measured_clips": ev["clips"], "measured_hours": round(hours_ref, 3)}}


def markdown(res):
    s = res["summary"]
    out = [f"# Realism check ({s['drives']} drives, {s['hours_driven']} h of driving)", "",
           f"{s['events_matching']} of {s['events_checked']} measured events happen in the simulator at a rate inside "
           f"the range our footage gives ({s['events_matching_traffic_alone']} of {s['events_checked']} with the event "
           f"generator off). Per hour of driving:", "",
           "| Event | Real roads | Range | Simulator | Traffic alone | Match |", "|---|---|---|---|---|---|"]
    for r in res["events"]:
        if r["rate"] == "assumed":
            continue
        out.append(f"| {r['name']} | {r['measured_per_hour']:.0f} | {r['low']:.0f} to {r['high']:.0f} | "
                   f"{r['sim_per_hour']:.0f} | {r['traffic_alone_per_hour']:.0f} | {'yes' if r['match'] else 'no'} |")
    out += ["", "Assumed rates (no measurement yet): " + ", ".join(
        f"{r['name'].lower()} {r['sim_per_hour']:.0f}" for r in res["events"] if r["rate"] == "assumed"), "",
        "| Speeds | Median km/h | Mean km/h | Stopped |", "|---|---|---|---|"]
    for k, name in (("real_play_area", "Our dashcam car, play area"), ("real_delhi_ncr", "Our dashcam car, Delhi NCR"),
                    ("sim_cars", "Cars going our way, simulator"), ("sim_our_car", "Our car in the simulator (IDM planner)")):
        v = res["speeds"][k]
        out.append(f"| {name} | {v['median_kmh']} | {v['mean_kmh']} | {v['stopped_share']:.0%} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drives", type=int, default=48)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/realism")
    a = ap.parse_args()
    t0 = time.time()
    with Pool(a.workers) as pool:
        drives = pool.map(drive, [(s, k) for s in (True, False) for k in range(a.drives)])
    res = compare(drives, measured())
    res["meta"] = {"drives": a.drives, "seconds": round(time.time() - t0), "python": sys.version.split()[0]}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w").write(markdown(res))
    print(markdown(res))


if __name__ == "__main__":
    main()

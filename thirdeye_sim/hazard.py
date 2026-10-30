"""The simulator's own event hazard around our car: for each of eight zones (front, front-left, left, back-left,
back, back-right, right, front-right), the chance that one of its events involves that zone in the next few
seconds. It is what the event generator (events.py) will do, at its rates (measured or assumed, times chaos)
plus the events already under way, not a prediction of real-world risk: we have no measured effect of the scene
on the rates yet. Agents may use it as a privileged input, labelled as such (STANDARDS.md).

    hazard(world, horizon=3.0) -> [8 floats in 0..1], zone 0 dead ahead, then anticlockwise
"""
import math

from .events import RATES

ZONES = ["front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right"]
# where each event involves our car (zone indices above): the share of its rate that falls on each zone
WHERE = {
    "two_wheeler_close": {2: 0.3, 3: 0.2, 5: 0.2, 6: 0.3}, "overtaken_left": {2: 0.6, 3: 0.4},
    "overtaken_right": {6: 0.6, 5: 0.4}, "cut_in_left": {1: 1.0}, "cut_in_right": {7: 1.0}, "lead_brake": {0: 1.0},
    "slow_vehicle_ahead": {0: 1.0}, "pull_out": {1: 0.7, 0: 0.3}, "pedestrian_crossing": {0: 0.5, 1: 0.25, 7: 0.25},
    "pedestrian_on_road": {1: 0.7, 0: 0.3}, "wrong_way_in_lane": {0: 1.0}, "red_light_violation": {1: 0.5, 7: 0.5},
    "cow_on_road": {0: 1.0}, "dog_crossing": {0: 0.4, 1: 0.3, 7: 0.3}, "u_turn": {7: 0.6, 0: 0.4},
}


def zone_of(world, x, y):
    e = world.ego
    a = math.atan2(y - e.y, x - e.x) - e.heading
    a = math.atan2(math.sin(a), math.cos(a))
    return int(((math.degrees(a) + 22.5) % 360) // 45)


def hazard(world, horizon=3.0):
    p = [0.0] * 8
    ev = world.events
    if ev is None or world.style != "gurgaon" or not world.scripted:
        return p
    for label, rate in RATES.items():   # the background chance from the rates
        per_s = rate * world.chaos / 60.0
        for z, share in WHERE.get(label, {}).items():
            q = 1 - math.exp(-per_s * share * horizon)
            p[z] = 1 - (1 - p[z]) * (1 - q)
    for a in ev.active:   # events under way: near certain as they come due
        if a.actor is None:
            continue
        z = zone_of(world, a.actor.x, a.actor.y)
        q = math.exp(-max(0.0, getattr(a, "eta", 3.0)) / horizon)
        p[z] = 1 - (1 - p[z]) * (1 - q)
    return [round(v, 4) for v in p]

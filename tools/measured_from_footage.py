"""Summarise our labelled dashcam footage into what the realism check compares against
(thirdeye_sim/data/measured.json): how often each event was the main event of a labelled clip, and how fast our
car drove. Only these summaries ship with the package; the clips, labels and GPS tracks stay with us.

    python tools/measured_from_footage.py ~/Desktop/thirdeye/event-rates
"""
import json, os, sys

ER = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Desktop/thirdeye/event-rates")
OUT = os.path.join(os.path.dirname(__file__), "..", "thirdeye_sim", "data", "measured.json")
BOX = (77.054, 77.109, 28.4565, 28.4725)   # the simulator's play area (lon, lon, lat, lat): gurgaon/clip_osm.py
STEP, TOP = 5, 80                          # speed histogram: 5 km/h steps, the last one TOP km/h and over

labels = json.load(open(os.path.join(ER, "labels", "truth_v04.json")))
taxonomy = json.load(open(os.path.join(ER, "taxonomy.json")))
counts = {}
for v in labels.values():
    counts[v["label"]] = counts.get(v["label"], 0) + 1

def speeds(inside):
    """Our car's GPS speed once a second, from one-minute clips in which it ever went faster than walking
    (3 km/h: GPS jitter reads up to about 2 km/h when parked), so parked recordings are left out."""
    out, clips = [], 0
    for v in json.load(open(os.path.join(ER, "runs", "gps_index.json"))).values():
        sp = [s for s in v.get("speed", []) if s is not None and 0 <= s <= 120]
        if len(sp) < 30 or max(sp) < 3:
            continue
        lon, lat = v["start"]
        if inside and not (BOX[0] <= lon <= BOX[1] and BOX[2] <= lat <= BOX[3]):
            continue
        out += sp; clips += 1
    s = sorted(out)
    q = lambda p: s[int(p * (len(s) - 1))]
    hist = [0] * (TOP // STEP + 1)
    for x in s:
        hist[min(TOP // STEP, int(x // STEP))] += 1
    return {"clips": clips, "hours": round(len(s) / 3600, 2), "mean_kmh": round(sum(s) / len(s), 1),
            "median_kmh": q(0.5), "p85_kmh": q(0.85), "stopped_share": round(sum(x < 2 for x in s) / len(s), 4),
            "hist": [round(h / len(s), 4) for h in hist]}

data = {
    "about": "Summaries of ThirdEye Labs' dashcam footage from Delhi NCR (CC BY 4.0), the reference for "
             "python -m thirdeye_sim.realism. Made by tools/measured_from_footage.py.",
    "events": {"clips": len(labels), "clip_seconds": 4, "taxonomy": taxonomy["version"], "rule": taxonomy["rule"],
               "priority": [c["key"] for c in taxonomy["classes"]], "counts": counts},
    "speeds": {"step_kmh": STEP, "top_kmh": TOP, "stopped_kmh": 2,
               "play_area": speeds(True), "delhi_ncr": speeds(False)},
}
json.dump(data, open(OUT, "w"), indent=1)
print(f"wrote {os.path.normpath(OUT)}: {len(labels)} clips, {data['speeds']['play_area']['clips']} driving minutes in the play area, "
      f"{data['speeds']['delhi_ncr']['clips']} in Delhi NCR")

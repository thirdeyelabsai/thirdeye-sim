"""The data behind the methodology page and the blog post: how AI video models compared with our labellers, the
label counts, the sample clips, one simulated drive cut into 4-second windows, and every benchmark route.
Writes web/public/data/methodology.json (summaries only; the clips' labels, not the clips).

    .venv-sumo/bin/python tools/methodology_data.py ~/Desktop/thirdeye/event-rates
"""
import glob, json, os, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
ER = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Desktop/thirdeye/event-rates")
OUT = os.path.join(ROOT, "web", "public", "data", "methodology.json")

# the clips shown on the pages (blurred copies in web/public/samples), with the event the frame shows best
SAMPLES = ["day_0844_011", "day_0844_002", "day_0844_004", "day_0844_098", "day_0844_171", "day_0844_229"]
MODELS = {"apac.amazon.nova-pro-v1:0": "Amazon Nova Pro", "qwen.qwen3-vl-235b-a22b": "Qwen3-VL 235B",
          "global.amazon.nova-2-lite-v1:0": "Amazon Nova 2 Lite", "nvidia/Cosmos-Reason2-8B": "NVIDIA Cosmos-Reason2 8B",
          "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL 8B"}


def main():
    from thirdeye_sim.realism import NAMES, drive, main_events, measured
    truth = json.load(open(os.path.join(ER, "labels", "truth_v04.json")))
    tax = json.load(open(os.path.join(ER, "taxonomy.json")))
    ren = tax.get("renamed", {})
    label_names = {c["key"]: c["label"] for c in tax["classes"]}

    # AI video models on the same 6 s clips (4 s to label, a second of context either side): each model's best setup
    best = {}
    for f in sorted(glob.glob(os.path.join(ER, "runs", "grid", "*_w6_*.jsonl"))):
        preds, model = {}, None
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            model = r.get("model", model)
            lab = (r.get("parsed") or {}).get("label")
            if lab:
                preds[r["id"].replace("_w6_", "_")] = ren.get(lab, lab)
        if model not in MODELS or len([k for k in truth if k in preds]) < 100:
            continue
        right = sum(preds.get(k) == v["label"] for k, v in truth.items())
        events = [k for k, v in truth.items() if v["label"] != "nothing"]
        row = {"model": MODELS[model], "right": right, "clips": len(truth),
               "events_right": sum(preds.get(k) == truth[k]["label"] for k in events), "events": len(events),
               "said_nothing": sum(preds.get(k) == "nothing" for k in events), "setup": os.path.basename(f)[:-6], "preds": preds}
        if model not in best or right > best[model]["right"]:
            best[model] = row
    models = sorted(best.values(), key=lambda r: -r["right"])
    top = models[0]
    nothing = sum(v["label"] == "nothing" for v in truth.values())

    counts = {}
    for v in truth.values():
        counts[v["label"]] = counts.get(v["label"], 0) + 1
    samples = [{"id": c, "label": truth[c]["label"], "name": label_names.get(truth[c]["label"], truth[c]["label"]),
                "actor": truth[c].get("actor"), "night": c.startswith("night"),
                "ai_label": top["preds"].get(c), "ai_name": label_names.get(top["preds"].get(c), top["preds"].get(c)),
                "ai_model": top["model"]} for c in SAMPLES]

    # one simulated drive (event generator on), its events and the main event in each 4 s window
    ref = measured()["events"]
    # the busiest two minutes of a drive, timed from their start: the first seed whose drive has ten events in
    # two minutes (some drives end early in a crash, and the generator takes a minute or so to get going)
    for seed in range(1, 40):
        d = drive((True, seed))
        ev = sorted(d["events"])
        t0 = max((t for t, _, _ in ev), key=lambda a: sum(a <= t < a + 120 for t, _, _ in ev), default=0.0)
        t0 = 4 * int(t0 // 4)
        if sum(t0 <= t < t0 + 120 for t, _, _ in ev) >= 10:
            break
    first = [(round(t - t0, 1), lab, src) for t, lab, src in ev if t0 <= t < t0 + 120]
    mains = {}
    for t, lab, src in first:
        k = int(t // 4)
        mains.setdefault(k, []).append(lab)
    windows = [{"window": k, "events": labs, "main": main_events([(4 * k + 1, l, "") for l in labs], ref["priority"], 4)[0][0]}
               for k, labs in sorted(mains.items())]

    bench = json.load(open(os.path.join(ROOT, "results", "benchmark.json")))
    routes = [{"traffic": e["style"], "seed": e["seed"], "ended": e["ended"],
               "hit": next((k for k, v in e["collisions"].items() if v), None), "at_fault": e["at_fault"],
               "completion": round(e["completion"], 2)}
              for e in bench["episodes"] if e["planner"] == "idm"]

    data = {
        "labelling": {"clips": len(truth), "clip_seconds": 4, "nothing": nothing, "taxonomy": tax["version"],
                      "rule": tax["rule"], "priority": [{"key": c["key"], "name": c["label"]} for c in tax["classes"]],
                      "counts": counts,
                      "models": [{k: r[k] for k in ("model", "right", "clips", "events_right", "events", "said_nothing")} for r in models]},
        "samples": samples,
        "window_demo": {"seconds": 120, "events": [{"t": t, "label": lab, "name": NAMES.get(lab, lab), "source": src} for t, lab, src in first],
                        "windows": windows},
        "routes": routes,
    }
    json.dump(data, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(models)} models (best {top['model']} {top['right']}/{len(truth)}), {len(samples)} samples, "
          f"{len(first)} events in the demo drive, {len(routes)} routes")
    for r in models:
        print(f"  {r['model']:28s} {r['right']:3d}/100 right, events {r['events_right']}/{r['events']}, said nothing on {r['said_nothing']} events ({r['setup']})")
    for s in samples:
        print(f"  {s['id']}: {s['name']} | {s['ai_model']} said {s['ai_name']}")


if __name__ == "__main__":
    main()

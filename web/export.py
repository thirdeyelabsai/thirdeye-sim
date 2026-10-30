"""Write what the browser version loads into web/public/data (gzipped files end in z, not .gz, so that
servers pass them on as they are instead of unpacking them on the way):

  sim.zip        the simulator's Python code and the two data files it reads, for Pyodide
  map.pklz       the road map the simulator needs (lanes, junctions, footpaths, signals), without buildings
                 and without lane outlines, which web.py rebuilds from the centre lines in a fraction of a second
  meta.json      start poses, the map projection (SUMO's netOffset and UTM zone), road labels, places
  scene/*.binz   the 3D city in 600 m squares, quantised (positions to 2 cm, heights to 0.1 mm on the
                 ground and 1 cm above it), plus big.binz for the few pieces over 100 m across

Chunk file layout (little-endian; each array starts on a 4-byte boundary):
  "TEC2", then uint32 counts: ground vertices, solid vertices, boxes, spheres, map fill vertices, map line
  vertices; float32 origin x, y
  ground    int16 x, y (2 cm), int16 z (0.1 mm), uint8 rgb
  solid     int16 x, y (2 cm), int16 z (1 cm), uint8 rgb, int8 normal xyz (/127), uint8 shadow (0 or 1)
  boxes     int16 x, y (2 cm), int16 z (1 cm), uint16 yaw (2 pi / 65536), uint16 size xyz (1 cm), uint8 rgb
  spheres   the same as boxes
  map fill  int16 x, y (2 cm), uint8 rgb                         (the minimap's parks, water, buildings)
  map lines int16 x, y (2 cm), int8 side xy (/127), uint8 rgb, uint16 half width (1 cm), uint8 least
            width (0.1 px)                                         (its roads, at least a few pixels wide)
Each array is stored whole before the next (all x, then all y, ...), which compresses far better.

    .venv-sumo/bin/python web/export.py
"""
import gzip, io, json, math, os, pickle, re, sys, zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..")
sys.path.insert(0, SIM)
from thirdeye_sim.geo import NET, from_lonlat   # noqa: E402
from thirdeye_sim.places import load_places      # noqa: E402
from thirdeye_sim.roadmap import STARTS, RoadMap  # noqa: E402
from thirdeye_sim.scene3d import TILE, load_scene  # noqa: E402

OUT = os.path.join(HERE, "public", "data")
CHUNK = 4   # tiles per chunk side: 600 m
KEYS = ("ground", "solid", "boxes", "spheres", "map_fill", "map_lines")
WIDTH = {"ground": 10, "solid": 10, "boxes": 11, "spheres": 11, "map_fill": 10, "map_lines": 9}   # floats per row
CORE = ["__init__.py", "actors.py", "events.py", "models.py", "paths.py", "roadmap.py", "signals.py", "vehicle.py", "world.py", "web.py"]
DATA = ["gurgaon/vtypes.add.xml", "gurgaon/step3/measured_density.json"]
SIM_KEYS = ["lanes", "junctions", "walks", "crossings", "signals", "levels", "zone_same", "zones", "sidewalks", "walkareas",
            "bounds", "lane_ids", "play_area"]


def q(a, unit, dtype=np.int16):
    v = np.round(np.asarray(a, dtype=np.float64) / unit)
    info = np.iinfo(dtype)
    assert v.min() >= info.min and v.max() <= info.max, (unit, v.min(), v.max())
    return v.astype(dtype)


def rgb(a):
    return np.clip(np.round(np.asarray(a) * 255), 0, 255).astype(np.uint8)


def chunk_bytes(ground, solid, boxes, spheres, fill, lines, ox, oy):
    parts = [b"TEC2", np.array([len(ground), len(solid), len(boxes), len(spheres), len(fill), len(lines)], np.uint32).tobytes(),
             np.array([ox, oy], np.float32).tobytes()]

    def add(arr):
        b = np.ascontiguousarray(arr).tobytes()
        parts.append(b + b"\0" * (-len(b) % 4))

    if len(ground):
        add(q(ground[:, 0] - ox, 0.02)); add(q(ground[:, 1] - oy, 0.02)); add(q(ground[:, 2], 1e-4)); add(rgb(ground[:, 3:6]))
    if len(solid):
        add(q(solid[:, 0] - ox, 0.02)); add(q(solid[:, 1] - oy, 0.02)); add(q(solid[:, 2], 0.01)); add(rgb(solid[:, 3:6]))
        add(q(solid[:, 6:9], 1 / 127, np.int8)); add((solid[:, 9] > 0.5).astype(np.uint8))
    for inst in (boxes, spheres):
        if len(inst):
            add(q(inst[:, 0] - ox, 0.02)); add(q(inst[:, 1] - oy, 0.02)); add(q(inst[:, 2], 0.01))
            add(q(np.mod(inst[:, 3], 2 * math.pi), 2 * math.pi / 65536, np.uint16)); add(q(inst[:, 4:7], 0.01, np.uint16))
            add(rgb(inst[:, 7:10]))
    if len(fill):
        add(q(fill[:, 0] - ox, 0.02)); add(q(fill[:, 1] - oy, 0.02)); add(rgb(fill[:, 3:6]))
    if len(lines):
        add(q(lines[:, 0] - ox, 0.02)); add(q(lines[:, 1] - oy, 0.02)); add(q(lines[:, 2:4], 1 / 127, np.int8))
        add(rgb(lines[:, 4:7])); add(q(lines[:, 7], 0.01, np.uint16)); add(q(lines[:, 8], 0.1, np.uint8))
    return b"".join(parts)


def main():
    os.makedirs(os.path.join(OUT, "scene"), exist_ok=True)
    m = RoadMap.load()
    sc = load_scene(m)

    # the simulator's code and the files it reads
    with zipfile.ZipFile(os.path.join(OUT, "sim.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in CORE:
            z.write(os.path.join(SIM, "thirdeye_sim", f), "thirdeye_sim/" + f)
        for f in DATA:
            z.write(os.path.join(SIM, f), f)
    raw = pickle.load(open(os.path.join(SIM, "thirdeye_sim", "cache", "gurgaon_map.pkl"), "rb"))
    trimmed = {k: raw[k] for k in SIM_KEYS}
    trimmed["lanes"] = [{k: v for k, v in l.items() if k != "poly"} for l in raw["lanes"]]   # rebuilt on load (web.py)
    trimmed.update(markings=[], buildings=[], greens=[], water=[])
    with gzip.open(os.path.join(OUT, "map.pklz"), "wb", compresslevel=9) as f:
        pickle.dump(trimmed, f, protocol=4)

    # the city, in chunks
    groups = {}
    for key in KEYS:
        per = {"ground": 3, "solid": 3, "map_fill": 3, "map_lines": 6}.get(key, 1)
        for tile, (first, n) in sc[key + "_tiles"].items():
            c = "big" if tile == "big" else (tile[0] // CHUNK, tile[1] // CHUNK)
            groups.setdefault(c, {}).setdefault(key, []).append(sc[key][first:first + n])
    index = []
    for c, parts in groups.items():
        arrs = [np.concatenate(parts[k]) if k in parts else np.zeros((0, WIDTH[k]), np.float32) for k in KEYS]
        if c == "big":
            ox, oy, name = 0.0, 0.0, "big"
            blob = b"TEF2" + np.array([len(a) for a in arrs], np.uint32).tobytes() + b"".join(a.astype(np.float32).tobytes() for a in arrs)
        else:
            ox, oy, name = (c[0] + 0.5) * CHUNK * TILE, (c[1] + 0.5) * CHUNK * TILE, f"{c[0]}_{c[1]}"
            blob = chunk_bytes(*arrs, ox, oy)
        with gzip.open(os.path.join(OUT, "scene", name + ".binz"), "wb", compresslevel=9) as f:
            f.write(blob)
        index.append({"name": name, "x": ox, "y": oy, "half": CHUNK * TILE / 2,
                      "bytes": os.path.getsize(os.path.join(OUT, "scene", name + ".binz"))})

    head = open(NET).read(4000)
    loc = re.search(r'<location netOffset="([-\d.]+),([-\d.]+)".*?projParameter="([^"]+)"', head)
    starts = {name: m.start_pose(name) for name in list(STARTS) + ["sunset"]}
    meta = {"starts": starts, "net_offset": [float(loc.group(1)), float(loc.group(2))], "proj": loc.group(3),
            "base": sc["base"][:, :6].tolist(), "chunk": CHUNK * TILE, "chunks": index,
            "labels": sc["labels"],
            "places": [{k: p[k] for k in ("name", "x", "y", "rank", "colour")} for p in load_places()],
            "play_area": [list(p) for p in m.play_area], "start_pool": [list(p) for p in m.start_pool()], "attribution": "© OpenStreetMap contributors"}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))
    # what the landing and realism pages show: the realism check and the planner benchmark (python -m
    # thirdeye_sim.realism and thirdeye_sim.evaluate write them), without the per-episode details
    res = os.path.join(HERE, "..", "results")
    realism = json.load(open(os.path.join(res, "realism.json")))
    json.dump(realism, open(os.path.join(OUT, "realism.json"), "w"))
    bench = json.load(open(os.path.join(res, "benchmark.json")))
    json.dump({"meta": bench["meta"], "summary": bench["summary"]}, open(os.path.join(OUT, "benchmark.json"), "w"))
    total = sum(e["bytes"] for e in index)
    print(f"scene: {len(index)} chunks, {total / 1e6:.1f} MB gzipped (largest {max(e['bytes'] for e in index) / 1e6:.2f} MB)")
    for f in ("sim.zip", "map.pklz", "meta.json"):
        print(f"{f}: {os.path.getsize(os.path.join(OUT, f)) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

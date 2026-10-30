"""Build the wheel and source package for PyPI: copy the data an installed copy needs from the map build folder
into thirdeye_sim/data, build, and take the copies away again (a checkout reads gurgaon/ itself).

    .venv-sumo/bin/python tools/build_package.py      (writes dist/)
"""
import os, re, shutil, subprocess, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from thirdeye_sim.paths import CACHE, DATA, GURGAON   # noqa: E402
from thirdeye_sim.roadmap import RoadMap               # noqa: E402

FILES = ["vtypes.add.xml", "step3/measured_density.json", "step3/probe_route.txt", "osm_places.json", "osm_context.json"]


def main():
    RoadMap.load()   # (makes sure the built map is current)
    made = []
    for rel in FILES:
        dst = os.path.join(DATA, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(GURGAON, rel), dst); made.append(dst)
    shutil.copy(os.path.join(CACHE, "gurgaon_map.pkl"), os.path.join(DATA, "gurgaon_map.pkl")); made.append(os.path.join(DATA, "gurgaon_map.pkl"))
    head = open(os.path.join(GURGAON, "gurgaon.net.xml")).read(4000)   # the map's projection and offset, for geo.py
    loc = re.search(r"<location [^>]*/>", head).group(0)
    with open(os.path.join(DATA, "location.xml"), "w") as f:
        f.write(f"<net>\n    {loc}\n</net>\n")
    made.append(os.path.join(DATA, "location.xml"))
    try:
        shutil.rmtree(os.path.join(ROOT, "build"), ignore_errors=True)
        subprocess.run([sys.executable, "-m", "build", ROOT], check=True)
    finally:
        for p in made:
            os.remove(p)
        shutil.rmtree(os.path.join(DATA, "step3"), ignore_errors=True)
        shutil.rmtree(os.path.join(ROOT, "build"), ignore_errors=True)


if __name__ == "__main__":
    main()

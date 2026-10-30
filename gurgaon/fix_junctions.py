"""Apply junction_fixes.json to the plain-XML network files (p.nod.xml, p.tll.xml) before the final build.

join: nodes linked by a drivable edge shorter than JOIN_BELOW_M (as built, after the junction shapes cut
the edges) are one real junction that the OSM import
split (a car has nowhere to stand between them, so it waits for all of them to clear at once). Each linked
group becomes one junction.
signals: each group of nodes becomes one signal controller; netconvert builds its plan. Any plan the group's
nodes had before is dropped, so the new controller replaces it.

    ../.venv-sumo/bin/python fix_junctions.py <plain prefix, e.g. /tmp/x/p> <the imported .net.xml>
"""
import json, os, sys
import xml.etree.ElementTree as ET

import sumolib

JOIN_BELOW_M = 5.0

prefix = sys.argv[1]
fixes = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "junction_fixes.json")))
nodes = ET.parse(prefix + ".nod.xml")
by_id = {n.get("id"): n for n in nodes.getroot().iter("node")}
old_tls = set()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from thirdeye_sim.geo import from_lonlat   # (the imported network's own location has netconvert's sign slip)

def node_for(entry):
    """A node of junction_fixes.json: by its id, or, when netconvert has named the merged junction differently
    (wider roads, other merges), the junction nearest where it was (within 25 m)."""
    if entry["id"] in by_id:
        return by_id[entry["id"]]
    x, y = from_lonlat(*entry["at"])
    best = min(by_id.values(), key=lambda n: (float(n.get("x")) - x) ** 2 + (float(n.get("y")) - y) ** 2)
    if (float(best.get("x")) - x) ** 2 + (float(best.get("y")) - y) ** 2 > 25 ** 2:
        raise SystemExit(f"junction_fixes.json: no junction near {entry}")
    return best

for tl, entries in fixes.get("signals", {}).items():
    for entry in entries:
        n = node_for(entry)
        i = n.get("id")
        old_tls.add(n.get("tl", i) if n.get("type") == "traffic_light" else None)
        n.set("type", "traffic_light"); n.set("tl", tl); n.set("tlType", "actuated")
parent = {}
def find(i):
    while parent.get(i, i) != i:
        i = parent[i]
    return i
# edge lengths as built (after the junction shapes cut the edges), from the imported network
for e in sumolib.net.readNet(sys.argv[2]).getEdges():
    a, b = e.getFromNode().getID(), e.getToNode().getID()
    if not e.isSpecial() and e.allows("passenger") and e.getLength() < JOIN_BELOW_M and a in by_id and b in by_id:
        parent[find(a)] = find(b)
groups = {}
for i in parent:
    groups.setdefault(find(i), set()).add(i)
for g in groups.values():
    if len(g) > 1:
        x = sum(float(by_id[i].get("x")) for i in g) / len(g)
        y = sum(float(by_id[i].get("y")) for i in g) / len(g)
        # without a position netconvert places the joined junction at its 'invalid' point, far off the map
        ET.SubElement(nodes.getroot(), "join", nodes=" ".join(sorted(g)), x=f"{x:.2f}", y=f"{y:.2f}")
print("joined", sum(len(g) > 1 for g in groups.values()), "groups of nodes linked by edges under", JOIN_BELOW_M, "m")
nodes.write(prefix + ".nod.xml")
tll = ET.parse(prefix + ".tll.xml")
root = tll.getroot()
for el in list(root):
    if el.get("id") in old_tls or el.get("tl") in old_tls:
        root.remove(el)
tll.write(prefix + ".tll.xml")
print("signal groups:", {k: len(v) for k, v in fixes.get("signals", {}).items()}, "| old plans dropped:", sorted(t for t in old_tls if t))

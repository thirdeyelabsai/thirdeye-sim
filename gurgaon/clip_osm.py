"""Cut the OpenStreetMap download at its bounding box, so the game's map ends cleanly at one edge.

Overpass returns every road that touches the box in full, so roads ran on past the box as loose fragments:
main roads that stop in empty ground, with none of the streets that join them out there. This keeps only the
parts of roads inside the box, ending each road that crosses the edge at a new node exactly on it (where
the game puts its boundary wall). A road that leaves the box and comes back becomes two ways; the second gets
a new id. Node tags (traffic signals, crossings) are kept for the nodes inside.

    ../.venv-sumo/bin/python clip_osm.py gurgaon.osm gurgaon_play.osm
"""
import sys
import xml.etree.ElementTree as ET

BOX = (28.4565, 77.0540, 28.4725, 77.1090)   # south, west, north, east: the Overpass query's box
NEW_ID = 9_000_000_000                         # ids for the nodes and ways this adds


def clip_segment(a, b, box):
    """Liang-Barsky: the part of segment a-b (lon, lat) inside the box, as parameters (t0, t1), or None."""
    s, w, n, e = box
    (x0, y0), (x1, y1) = a, b
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - w), (dx, e - x0), (-dy, y0 - s), (dy, n - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return t0, t1


def main(src, dst, box=BOX):
    root = ET.parse(src).getroot()
    nodes = {n.get("id"): n for n in root.iter("node")}
    pos = {i: (float(n.get("lon")), float(n.get("lat"))) for i, n in nodes.items()}
    out = ET.Element("osm", version="0.6", generator="clip_osm.py")
    note = ET.SubElement(out, "note")
    note.text = "The data included in this document is from www.openstreetmap.org. The data is made available under ODbL."
    s, w, n, e = box
    ET.SubElement(out, "bounds", minlat=str(s), minlon=str(w), maxlat=str(n), maxlon=str(e))
    kept_nodes, new_nodes, ways = set(), [], []
    next_id = NEW_ID

    def boundary_node(a, b, t):
        nonlocal next_id
        next_id += 1
        lon, lat = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        new_nodes.append((str(next_id), lon, lat))
        return str(next_id)

    for way in root.iter("way"):
        tags = [t for t in way.iter("tag")]
        if not any(t.get("k") == "highway" for t in tags):
            continue
        refs = [nd.get("ref") for nd in way.iter("nd") if nd.get("ref") in pos]
        runs, run = [], []
        for ra, rb in zip(refs, refs[1:]):
            a, b = pos[ra], pos[rb]
            c = clip_segment(a, b, box)
            if c is None:
                if run:
                    runs.append(run); run = []
                continue
            t0, t1 = c
            if not run:
                run = [ra] if t0 == 0 else [boundary_node(a, b, t0)]
            if t1 < 1:
                run.append(boundary_node(a, b, t1))
                runs.append(run); run = []
            else:
                run.append(rb)
        if run:
            runs.append(run)
        for k, r in enumerate(r for r in runs if len(r) >= 2):
            wid = way.get("id")
            if k:
                next_id += 1
                wid = str(next_id)
            ways.append((wid, r, tags))
            kept_nodes.update(r)
    for i in sorted(i for i in kept_nodes if i in nodes):
        out.append(nodes[i])
    for i, lon, lat in new_nodes:
        if i in kept_nodes:
            ET.SubElement(out, "node", id=i, lat=f"{lat:.7f}", lon=f"{lon:.7f}", version="1")
    for wid, r, tags in ways:
        el = ET.SubElement(out, "way", id=wid, version="1")
        for ref in r:
            ET.SubElement(el, "nd", ref=ref)
        for t in tags:
            el.append(t)
    ET.ElementTree(out).write(dst, encoding="UTF-8", xml_declaration=True)
    print(f"{len(ways)} road ways ({sum(1 for w in ways if int(w[0]) > NEW_ID)} new pieces), "
          f"{len(kept_nodes)} nodes, {sum(1 for i, *_ in new_nodes if i in kept_nodes)} on the edge")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

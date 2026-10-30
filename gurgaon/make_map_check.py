"""Draw the SUMO road network over satellite imagery, with our drive's GPS track, to check the map by eye.

Each lane is drawn on its centre line, coloured by road class; signal-controlled junctions are red dots.
Where the lanes do not sit on the asphalt in the imagery, or a road we drove is missing, the network needs
fixing before any traffic goes on it. Writes map_check.html (a local page; the imagery loads from Esri).

    ../.venv-sumo/bin/python make_map_check.py
"""
import csv, glob, json, os

import sumolib

HERE = os.path.dirname(os.path.abspath(__file__))
COLOUR = {"motorway": "#d7301f", "primary": "#fc8d59", "secondary": "#fdcc8a", "tertiary": "#91bfdb", "residential": "#d9d9d9"}

net = sumolib.net.readNet(os.path.join(HERE, "gurgaon.net.xml"))
lanes = []
for e in net.getEdges():
    if e.isSpecial():
        continue
    kind = e.getType().split(".")[-1].replace("_link", "")
    colour = COLOUR.get(kind, "#d9d9d9")
    for lane in e.getLanes():
        if lane.allows("passenger") or lane.allows("motorcycle"):
            lanes.append({"c": colour, "p": [[round(lat, 6), round(lon, 6)] for lon, lat in (net.convertXY2LonLat(x, y) for x, y in lane.getShape())],
                          "n": f"{e.getName() or kind} ({e.getLaneNumber()} lanes)"})
signals = []
for tl in net.getTrafficLights():
    for node in {c[0].getEdge().getToNode() for c in tl.getConnections()}:
        lon, lat = net.convertXY2LonLat(*node.getCoord())
        signals.append([round(lat, 6), round(lon, 6)])
track = []
for f in sorted(glob.glob(os.path.expanduser("~/Desktop/thirdeye/event-rates/scenes/day_0844_*/gps.csv"))):
    track += [[float(r["lat"]), float(r["lon"])] for r in csv.DictReader(open(f))]

page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gurgaon map check</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<style>html,body,#m{height:100%;margin:0;background:#111;font:13px -apple-system,sans-serif}
.k{position:absolute;z-index:1000;top:10px;right:10px;background:rgba(20,20,20,.85);color:#eee;padding:8px 10px;border-radius:8px;line-height:1.7}
.k i{display:inline-block;width:18px;height:4px;margin-right:6px;vertical-align:middle}</style></head><body><div id="m"></div>
<div class="k"><b>SUMO network over satellite</b><br><i style="background:#d7301f"></i>motorway<br><i style="background:#fc8d59"></i>primary
<br><i style="background:#fdcc8a"></i>secondary<br><i style="background:#91bfdb"></i>tertiary<br><i style="background:#d9d9d9"></i>residential
<br><i style="background:#ff0;height:3px"></i>our morning drive<br><span style="color:#f33">&#9679;</span> signal in the network</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>
const D = __DATA__;
const m = L.map('m', {preferCanvas: true}).setView([28.4645, 77.0815], 15);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {maxZoom: 20, maxNativeZoom: 19, attribution: 'Imagery &copy; Esri'}).addTo(m);
for (const l of D.lanes) L.polyline(l.p, {color: l.c, weight: 2, opacity: .9}).bindTooltip(l.n).addTo(m);
L.polyline(D.track, {color: '#ff0', weight: 3, opacity: .8, dashArray: '6 6'}).addTo(m);
for (const s of D.signals) L.circleMarker(s, {radius: 5, color: '#f33', fillOpacity: 1}).addTo(m);
</script></body></html>"""
out = os.path.join(HERE, "map_check.html")
open(out, "w").write(page.replace("__DATA__", json.dumps({"lanes": lanes, "signals": signals, "track": track})))
print(out, len(lanes), "lanes,", len(signals), "signals,", len(track), "GPS points")

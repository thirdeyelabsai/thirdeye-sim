"""Named places from OpenStreetMap (gurgaon/osm_places.json: hospitals, malls, metro stations, offices, hotels,
schools, restaurants, sector names...), placed on our map and ranked by how much they help someone find
their way:
1 landmarks (hospitals, malls, stations, colleges, stadiums, cinemas), shown whenever they are in view;
2 offices, hotels, schools, petrol pumps, temples, parks, police, when zoomed in a little;
3 restaurants, cafes, banks, clinics, homes, only up close;
0 areas (sectors and colonies), written large on the full map only.

    from thirdeye_sim.places import load_places
    for p in load_places():  p["name"], p["x"], p["y"], p["cat"], p["rank"], p["colour"]
"""
import json, math, os
from functools import lru_cache

from .geo import from_lonlat, play_box
from .paths import data_file

SOURCE = data_file("osm_places.json")
COLOUR = {"health": (214, 60, 56), "shop": (150, 92, 205), "transit": (40, 112, 222), "school": (176, 112, 58),
          "hotel": (218, 86, 150), "food": (238, 138, 36), "bank": (28, 148, 150), "fuel": (96, 98, 110), "civic": (60, 82, 166),
          "worship": (200, 150, 40), "leisure": (58, 150, 82), "park": (70, 160, 80), "office": (112, 116, 130),
          "homes": (110, 128, 150), "area": (90, 88, 80)}


def _kind(t):
    """(category, rank) of a place from its OSM tags."""
    a, b = t.get("amenity"), t.get("building")
    if a == "hospital" or b == "hospital":
        return "health", 1
    if a == "clinic":
        return "health", 3
    if t.get("shop") == "mall":
        return "shop", 1
    if t.get("shop") in ("supermarket", "department_store") or b == "retail":
        return "shop", 2
    if t.get("railway") in ("station", "halt") or t.get("public_transport") == "station" or a == "bus_station":
        return "transit", 1
    if a in ("college", "university"):
        return "school", 1
    if a == "school":
        return "school", 2
    if t.get("tourism") == "hotel" or b == "hotel":
        return "hotel", 2
    if a in ("restaurant", "cafe", "fast_food"):
        return "food", 3
    if a == "bank":
        return "bank", 3
    if a == "fuel":
        return "fuel", 2
    if a in ("police", "townhall", "courthouse", "library", "community_centre", "marketplace"):
        return "civic", 2
    if a == "place_of_worship":
        return "worship", 2
    if a == "cinema" or t.get("tourism") in ("attraction", "museum") or t.get("leisure") in ("stadium", "golf_course"):
        return "leisure", 1
    if t.get("leisure") in ("park", "sports_centre"):
        return "park", 2
    if t.get("place") or t.get("landuse") == "residential":
        return "area", 0
    if "office" in t or b in ("office", "commercial"):
        return "office", 2
    if b == "apartments":
        return "homes", 3
    return "office", 3


@lru_cache(maxsize=1)
def load_places():
    if not os.path.exists(SOURCE):
        return []
    out = []
    with open(SOURCE) as f:
        elements = json.load(f)["elements"]
    for e in elements:
        t = e.get("tags", {})
        name = t.get("name:en") or t.get("name")
        if not name or any(ord(ch) > 0x24F for ch in name):   # (names our font cannot write)
            continue
        lon, lat = (e["lon"], e["lat"]) if "lon" in e else (e["center"]["lon"], e["center"]["lat"]) if "center" in e else (None, None)
        w, so, e, n = play_box()
        if lon is None or not (w <= lon <= e and so <= lat <= n):
            continue   # (beyond the wall at the edge of the map)
        x, y = from_lonlat(lon, lat)
        cat, rank = _kind(t)
        if any(p["name"] == name and math.hypot(p["x"] - x, p["y"] - y) < 150 for p in out):
            continue   # the same place mapped twice (a building and a point inside it)
        out.append({"name": name, "x": x, "y": y, "cat": cat, "rank": rank, "colour": COLOUR[cat]})
    return out

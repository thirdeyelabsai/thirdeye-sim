"""Map coordinates (metres, as in gurgaon.net.xml) to latitude and longitude and back.

SUMO stores the network in UTM zone 43 shifted by a fixed offset; both are read from the net file's
<location> element.
"""
import os, re
from functools import lru_cache

from .paths import data_file

NET = data_file("gurgaon.net.xml")   # (an installed copy has only its <location> element, in location.xml)
if not os.path.exists(NET):
    NET = data_file("location.xml")


@lru_cache(maxsize=1)
def _transform():
    from pyproj import Transformer
    with open(NET) as f:
        head = f.read(4000)
    loc = re.search(r'<location netOffset="([-\d.]+),([-\d.]+)".*?projParameter="([^"]+)"', head)
    dx, dy, proj = float(loc.group(1)), float(loc.group(2)), loc.group(3)
    return dx, dy, Transformer.from_crs(proj, "EPSG:4326", always_xy=True), Transformer.from_crs("EPSG:4326", proj, always_xy=True)


def to_lonlat(x, y):
    dx, dy, fwd, _ = _transform()
    return fwd.transform(x - dx, y - dy)


def from_lonlat(lon, lat):
    dx, dy, _, back = _transform()
    x, y = back.transform(lon, lat)
    return x + dx, y + dy


@lru_cache(maxsize=1)
def play_box():
    """(west, south, east, north) of the play area: the box the map was downloaded and cut to."""
    with open(NET) as f:
        head = f.read(4000)
    return tuple(map(float, re.search(r'origBoundary="([^"]+)"', head).group(1).split(",")))

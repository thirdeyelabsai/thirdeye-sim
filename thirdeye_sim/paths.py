"""Where the package finds its data. In a checkout of the repository: the map build folder gurgaon/ beside the
package, and caches in thirdeye_sim/cache. In an installed copy: thirdeye_sim/data (tools/build_package.py fills
it), and caches in ~/.cache/thirdeye_sim."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GURGAON = os.path.join(HERE, "..", "gurgaon")
CHECKOUT = os.path.exists(os.path.join(GURGAON, "gurgaon.net.xml"))   # (the map's sources are here: it can be rebuilt)
CACHE = os.path.join(HERE, "cache") if CHECKOUT else os.path.join(os.path.expanduser("~"), ".cache", "thirdeye_sim")


def data_file(rel):
    """A data file by its path in gurgaon/ (as "vtypes.add.xml" or "step3/measured_density.json"): from data/ in
    an installed copy, else from gurgaon/ (a checkout, or the browser version, which carries the few it needs)."""
    p = os.path.join(DATA, rel)
    return p if not CHECKOUT and os.path.exists(p) else os.path.join(GURGAON, rel)

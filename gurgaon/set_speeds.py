"""Set Gurgaon speed limits by road class in a plain-XML edge file (from netconvert --plain-output-prefix).

The OSM import gives untagged roads netconvert's defaults (100 km/h on primary and secondary roads, 80 on
tertiary), far above Gurgaon's limits. An edge still at its class default had no maxspeed tag in OSM, so it
gets the class limit below; edges whose OSM way carries its own maxspeed keep it.

    ../.venv-sumo/bin/python set_speeds.py p.edg.xml
"""
import collections, re, sys

DEFAULT = {"motorway": 39.44, "motorway_link": 22.22, "primary": 27.78, "primary_link": 22.22, "secondary": 27.78,
           "secondary_link": 22.22, "tertiary": 22.22, "tertiary_link": 22.22, "unclassified": 13.89, "residential": 13.89}
KMH = {"motorway": 80, "motorway_link": 60, "primary": 60, "primary_link": 50, "secondary": 60, "secondary_link": 50,
       "tertiary": 50, "tertiary_link": 50, "unclassified": 30, "residential": 30}

path = sys.argv[1]
changed = collections.Counter()


def edge(m):
    tag = m.group(0)
    t, s = re.search(r'type="highway\.([a-z_]+)"', tag), re.search(r' speed="([0-9.]+)"', tag)
    if not (t and s) or t.group(1) not in DEFAULT or abs(float(s.group(1)) - DEFAULT[t.group(1)]) > 0.02:
        return tag
    changed[t.group(1)] += 1
    return tag.replace(s.group(0), f' speed="{KMH[t.group(1)] / 3.6:.2f}"')


out = re.sub(r"<edge [^>]*>", edge, open(path).read())
out = re.sub(r'(<lane [^>]*?) speed="(27\.78|22\.22|13\.89|39\.44)"', r"\1", out)   # lanes inherit the edge's limit
open(path, "w").write(out)
print("edges set to class limits:", dict(changed))

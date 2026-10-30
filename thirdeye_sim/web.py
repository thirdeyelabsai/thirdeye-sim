"""The simulator as the browser version runs it: the same World, in Pyodide inside a web worker.

Each animation frame the page sends the keys held and how many 1/60 s steps are due; frame() answers with
everything the page draws: one float32 array and a small JSON string.

Float array:
  [t, users, boxes, signal rows]
  users        6 floats each: id (-1 for our car), x, y, heading, first box, box count
  boxes        10 floats each, in the road user's own frame: ahead, left, z0, length, width, height, r, g, b, emit
  signal rows  11 floats each, in world coordinates: x, y, z0, heading, length, width, height, r, g, b, emit
JSON: our car's status, the light ahead and its countdown, the event banner, event tags and rings, horn
and honk effects, new log lines, and the distances of road users who honked since the last frame (for sound).
"""
import gzip, json, math, pickle
from array import array

from .events import TAG, banner_text
from .models import ego_boxes, in_tunnel, signal_boxes, user_boxes
from .roadmap import RoadMap, _outline
from .world import World

DT = 1 / 60


class Bridge:
    def __init__(self, map_path, start, seed=0):
        with gzip.open(map_path, "rb") as f:
            data = pickle.load(f)
        for l in data["lanes"]:
            if "poly" not in l:   # left out of the download: the outline follows from the centre line and width
                l["poly"] = _outline(l["line"], l["w"])
        m = RoadMap(data)
        self.world = World(roadmap=m, seed=seed, start=start if isinstance(start, str) else tuple(start))
        self.tunnel = {}
        self._fresh()

    def _fresh(self):
        self.logged, self.heard, self.held = 0, -1.0, 0.0

    def reset(self, seed=None, start=None):
        if start is not None:
            self.world.start = start if isinstance(start, str) else tuple(start)
        self.world.reset(seed)
        self._fresh()

    def set_chaos(self, c):
        """1 = as measured; more: events more often, drivers pushier (actors.py, the note on chaos)."""
        if self.world.traffic:
            self.world.traffic.set_chaos(float(c))
        else:
            self.world.chaos = float(c)

    def set_style(self, style):
        """"gurgaon" (as measured) or "generic" (lane-disciplined traffic, no events), for comparison."""
        if self.world.traffic:
            self.world.traffic.set_style(str(style))
        else:
            self.world.style = str(style)

    def set_scripted(self, on):
        """Scripted events on or off (off: the traffic model alone)."""
        self.world.scripted = bool(on)

    def set_view(self, ahead, behind, half):
        """How far the screen reaches around our car (metres): traffic appears only outside it."""
        self.world.view = (min(130.0, ahead), min(130.0, behind), min(130.0, half))

    def step(self, n, throttle=0.0, braking=False, steer=0.0, emergency=False, horn=False, dipper=False):
        """n steps of 1/60 s with the controls held. braking: the brake is held; as in the desktop game a tap
        brakes gently and holding it brakes harder, reaching full braking after 0.3 s."""
        w = self.world
        if horn:
            w.signal("horn")
        if dipper:
            w.signal("dipper")
        for _ in range(int(n)):
            self.held = self.held + DT if braking else 0.0
            w.step(DT, throttle, min(1.0, 0.5 + self.held / 0.6) if self.held else 0.0, steer, bool(emergency))

    def frame(self):
        w, t = self.world, self.world.t
        users, boxes = array("f"), array("f")
        e = w.ego
        mine = ego_boxes(e)
        users.extend((-1, e.x, e.y, e.heading, 0, len(mine)))
        for b in mine:
            boxes.extend(b)
        n = 1
        for u in (w.traffic.all() if w.traffic else ()):
            if not hasattr(u, "where") and in_tunnel(w.map, u, self.tunnel):   # (vehicles only)
                continue   # under the street in an underpass: out of sight from above
            bs = user_boxes(u, t)
            users.extend((u.id, u.x, u.y, u.heading, len(boxes) // 10, len(bs)))
            for b in bs:
                boxes.extend(b)
            n += 1
        sig = signal_boxes(w, e.x, e.y)
        out = array("f", (t, n, len(boxes) // 10, len(sig)))
        out.extend(users); out.extend(boxes)
        for r in sig:
            out.extend(r)
        return out.tobytes(), json.dumps(self._info())

    def _info(self):
        w, t, e, st = self.world, self.world.t, self.world.ego, self.world.status
        info = {"t": t, "jolt": [w.jolt[0], w.jolt[1]], "chaos": w.chaos, "style": w.style, "scripted": w.scripted, "start": [e.x, e.y] if t == 0 else None, "speed_kmh": e.speed * 3.6, "limit_kmh": st.get("limit_kmh"), "road": st.get("road", ""),
                "on_road": st.get("on_road", 4), "wrong_way": st.get("wrong_way", False), "over_limit": st.get("over_limit", False),
                "accel": e.accel}
        ahead = w.signal_ahead()
        info["light"] = {"colour": ahead[2], "left": ahead[3]} if ahead else None
        ev = w.events
        b = ev.banner if ev else None
        info["banner"] = None
        if b:
            tag, text = banner_text(b)
            info["banner"] = {"tag": tag, "text": text, "state": b["state"]}
        info["tags"], info["rings"] = [], []
        if ev:
            for a in ev.active:
                if getattr(a, "highlight", False) and a.actor is not None:
                    info["tags"].append({"id": a.actor.id, "text": TAG.get(a.label, a.label.upper()), "tall": a.actor.kind in ("bus", "truck"),
                                         "r": max(a.actor.length, a.actor.width) / 2})
            for t_end, u, state in ev.recent:
                if 0 <= t - t_end < 1.5:
                    info["rings"].append({"x": u.x, "y": u.y, "r": max(u.length, u.width) / 2, "age": t - t_end, "state": state})
        info["signals_given"] = [{"kind": k, "age": t - t0} for t0, k in getattr(w, "signals_given", []) if t - t0 < 1.0]
        honking, heard = [], []
        for u in (w.traffic.vehicles if w.traffic else ()):
            age = t - getattr(u, "honk_t", -9)
            if 0 <= age < 0.6:
                honking.append({"id": u.id, "age": age, "ahead": u.length / 2})
            if self.heard < getattr(u, "honk_t", -9) <= t:
                heard.append(math.hypot(u.x - e.x, u.y - e.y))
        if heard:
            self.heard = t
        info["honking"], info["heard"] = honking, heard
        info["log"] = [[round(tt, 1), what] for tt, what in w.log[self.logged:]]
        self.logged = len(w.log)
        return info

"""The dashboard over the 3D view, drawn with pygame into textures: speedometer with the road's speed-limit
sign, road name, clock and coordinates; horn, dipper and brake lamps; the event log; warnings; the minimap's
frame, compass, scale and road names; and the full map's latitude/longitude grid.

Sizes are in window pixels and multiplied by the screen's scale, so text stays sharp on a Retina display.
Each panel is redrawn and uploaded only when what it shows has changed.
"""
import math

import pygame
import pygame.gfxdraw

from .events import banner_text
from .geo import from_lonlat, to_lonlat
from .places import load_places
from .scene3d import TILE

PANEL = (16, 18, 22, 215)
TEXT = (236, 236, 232)
DIM = (160, 163, 168)
WARN = (255, 92, 72)


def _font(size, bold=False):
    return pygame.font.SysFont("helveticaneue,helvetica,arial", size, bold=bold)


def _arc(surf, cx, cy, r0, r1, a0, a1, col, steps=48):
    """A thick anti-aliased arc between radii r0 and r1, angles in degrees anticlockwise from east."""
    if abs(a1 - a0) < 0.5:
        return
    pts = [(cx + math.cos(math.radians(a0 + (a1 - a0) * k / steps)) * r1, cy - math.sin(math.radians(a0 + (a1 - a0) * k / steps)) * r1) for k in range(steps + 1)]
    pts += [(cx + math.cos(math.radians(a1 - (a1 - a0) * k / steps)) * r0, cy - math.sin(math.radians(a1 - (a1 - a0) * k / steps)) * r0) for k in range(steps + 1)]
    pygame.gfxdraw.aapolygon(surf, pts, col)
    pygame.gfxdraw.filled_polygon(surf, pts, col)


class Hud:
    def __init__(self, renderer):
        pygame.font.init()
        self.r = renderer
        self.s = s = renderer.scale
        self.f_big = _font(int(34 * s), True)
        self.f_mid = _font(int(15 * s), True)
        self.f = _font(int(13 * s))
        self.f_small = _font(int(11 * s))
        self.f_label = _font(int(12 * s), True)
        self.tex = {}          # name -> (texture, key of what it shows)
        self.help_until = 25.0   # the key list shows for the first 25 s, then on K
        self.show_help = False
        self.banner_on = True    # B: what an event is about to do (hide it to drive without warnings)
        self.f_place = _font(int(12 * s), True)
        self.f_area = _font(int(15 * s), True)
        self.places = load_places()
        self.place_tiles = {}
        for p in self.places:
            self.place_tiles.setdefault((int(p["x"] // TILE), int(p["y"] // TILE)), []).append(p)

    # textures ---------------------------------------------------------------------------------------------
    def _upload(self, name, surf, key):
        ctx = self.r.ctx
        tex, old = self.tex.get(name, (None, None))
        if tex is None or tex.size != surf.get_size():
            if tex is not None:
                tex.release()
            tex = ctx.texture(surf.get_size(), 4)
            tex.filter = (0x2601, 0x2601)   # linear
        if key != old:
            tex.write(pygame.image.tostring(surf, "RGBA", True))
        self.tex[name] = (tex, key)
        return tex

    def _blit(self, tex, x, y, alpha=1.0):
        """Draw a texture with its top-left corner at (x, y) framebuffer pixels."""
        W, H = self.r.fb_size
        w, h = tex.size
        x0, x1 = x / W * 2 - 1, (x + w) / W * 2 - 1
        y0, y1 = 1 - (y + h) / H * 2, 1 - y / H * 2
        import numpy as np
        q = np.array([(x0, y0, 0, 0), (x1, y0, 1, 0), (x1, y1, 1, 1), (x0, y0, 0, 0), (x1, y1, 1, 1), (x0, y1, 0, 1)], dtype="f4")
        self.r.uibuf.write(q.tobytes())
        tex.use(0)
        self.r.ui["u_tex"].value = 0
        self.r.ui["u_alpha"].value = alpha
        self.r.vao_ui.render(vertices=6)

    def _cached(self, name, key, make):
        if name in self.tex and self.tex[name][1] == key:
            return self.tex[name][0]
        return self._upload(name, make(), key)

    # panels -----------------------------------------------------------------------------------------------
    def _speed_panel(self):
        w, s = self.r.world, self.s
        st, e = w.status, w.ego
        kmh = abs(st.get("speed_kmh", 0.0))
        lon, lat = to_lonlat(e.x, e.y)
        recent = [k for t0, k in w.signals_given if w.t - t0 < 0.6]
        key = (round(kmh), st.get("limit_kmh"), st.get("road"), int(w.t), round(lat, 5), round(lon, 5), tuple(recent), e.accel < -1.0)

        def make():
            S = pygame.Surface((int(330 * s), int(150 * s)), pygame.SRCALPHA)
            pygame.draw.rect(S, PANEL, S.get_rect(), border_radius=int(14 * s))
            cx, cy, r = 72 * s, 72 * s, 56 * s
            _arc(S, cx, cy, r - 7 * s, r, -45, 225, (60, 63, 70))
            limit = st.get("limit_kmh") or 50
            frac = min(1.0, kmh / 120.0)
            col = (90, 210, 120) if kmh <= limit else (255, 170, 40) if kmh <= limit * 1.1 else WARN
            _arc(S, cx, cy, r - 7 * s, r, 225 - 270 * frac, 225, col)
            for v in range(0, 121, 20):
                a = math.radians(225 - 270 * v / 120)
                t = self.f_small.render(str(v), True, DIM)
                S.blit(t, (cx + math.cos(a) * (r - 19 * s) - t.get_width() / 2, cy - math.sin(a) * (r - 19 * s) - t.get_height() / 2))
            t = self.f_big.render(f"{kmh:.0f}", True, TEXT)
            S.blit(t, (cx - t.get_width() / 2, cy - t.get_height() / 2 + 4 * s))
            t = self.f_small.render("km/h", True, DIM)
            S.blit(t, (cx - t.get_width() / 2, cy + 22 * s))
            # the speed-limit sign
            if st.get("limit_kmh"):
                sx, sy, sr = int(292 * s), int(36 * s), int(22 * s)
                pygame.gfxdraw.filled_circle(S, sx, sy, sr, (225, 35, 30)); pygame.gfxdraw.aacircle(S, sx, sy, sr, (225, 35, 30))
                pygame.gfxdraw.filled_circle(S, sx, sy, int(sr * 0.76), (250, 250, 248)); pygame.gfxdraw.aacircle(S, sx, sy, int(sr * 0.76), (250, 250, 248))
                t = self.f_mid.render(str(st["limit_kmh"]), True, (20, 20, 20))
                S.blit(t, (sx - t.get_width() / 2, sy - t.get_height() / 2))
            road = st.get("road") or "off the mapped roads"
            t = self.f_mid.render(road if len(road) < 24 else road[:22] + "...", True, TEXT)
            S.blit(t, (140 * s, 22 * s))
            t = self.f.render(f"{int(w.t // 60):02d}:{int(w.t % 60):02d}", True, DIM)
            S.blit(t, (140 * s, 46 * s))
            t = self.f.render(f"{lat:.5f} N   {lon:.5f} E", True, DIM)
            S.blit(t, (140 * s, 66 * s))
            for k, (name, on, col) in enumerate((("HORN", "horn" in recent, (255, 190, 40)), ("DIPPER", "dipper" in recent, (250, 245, 210)),
                                                  ("BRAKE", e.accel < -1.0, (255, 70, 60)))):
                x0 = (140 + 62 * k) * s
                rect = pygame.Rect(x0, 104 * s, 56 * s, 24 * s)
                pygame.draw.rect(S, col if on else (48, 51, 57), rect, border_radius=int(6 * s))
                t = self.f_small.render(name, True, (20, 20, 20) if on else DIM)
                S.blit(t, (rect.centerx - t.get_width() / 2, rect.centery - t.get_height() / 2))
            return S
        return self._cached("speed", key, make)

    def _log_panel(self):
        log = self.r.world.log[-7:]
        if not log:
            return None
        key = tuple(log)

        def make():
            s = self.s
            S = pygame.Surface((int(330 * s), int((34 + 21 * len(log)) * s)), pygame.SRCALPHA)
            pygame.draw.rect(S, PANEL, S.get_rect(), border_radius=int(12 * s))
            S.blit(self.f_mid.render("Events", True, TEXT), (14 * s, 8 * s))
            for k, (t, what) in enumerate(log):
                col = WARN if what.startswith(("crashed", "ran", "left the", "driving against", "wheels")) else (255, 196, 90) if "ahead" in what or "honked" in what else TEXT
                S.blit(self.f.render(f"{int(t // 60):02d}:{int(t % 60):02d}", True, DIM), (14 * s, (32 + 21 * k) * s))
                S.blit(self.f.render(what[:42], True, col), (60 * s, (32 + 21 * k) * s))
            return S
        return self._cached("log", key, make)

    def _warning(self):
        w = self.r.world
        st = w.status
        crash = w.crashes[-1] if w.crashes and w.t - w.crashes[-1]["t"] < 3 else None
        warn = f"CRASH: {crash['kind'].replace('_', '-')}" if crash else "OFF THE ROAD" if st["on_road"] == 0 else \
            "WHEELS OFF THE ROAD" if st["on_road"] < 4 else "WRONG WAY" if st["wrong_way"] else \
            f"OVER THE {st['limit_kmh']} KM/H LIMIT" if st.get("over_limit") else ""
        if not warn:
            return None

        def make():
            t = self.f_big.render(warn, True, (255, 255, 255))
            S = pygame.Surface((t.get_width() + int(40 * self.s), t.get_height() + int(14 * self.s)), pygame.SRCALPHA)
            pygame.draw.rect(S, (205, 40, 32, 230), S.get_rect(), border_radius=int(S.get_height() / 2))
            S.blit(t, (20 * self.s, 7 * self.s))
            return S
        return self._cached("warn", warn, make)

    def _banner(self):
        """What the current event is about to do, and then how it went."""
        ev = self.r.world.events
        b = ev.banner if ev else None
        if not b or not self.banner_on:
            return None
        tag, text = banner_text(b)
        col = {"coming": (255, 176, 40), "done": (70, 190, 110), "crash": (215, 50, 40)}.get(b["state"], (120, 124, 132))
        key = (tag, text)

        def make():
            s = self.s
            t1 = self.f_mid.render(tag, True, (20, 20, 20))
            t2 = self.f_mid.render(text, True, (240, 240, 236))
            pad = int(12 * s)
            S = pygame.Surface((t1.get_width() + t2.get_width() + 4 * pad, t2.get_height() + int(18 * s)), pygame.SRCALPHA)
            pygame.draw.rect(S, (16, 18, 22, 225), S.get_rect(), border_radius=int(S.get_height() / 2))
            tag_rect = pygame.Rect(int(6 * s), int(6 * s), t1.get_width() + 2 * pad, S.get_height() - int(12 * s))
            pygame.draw.rect(S, col, tag_rect, border_radius=int(tag_rect.height / 2))
            S.blit(t1, (tag_rect.x + pad, tag_rect.centery - t1.get_height() / 2))
            S.blit(t2, (tag_rect.right + pad, S.get_height() / 2 - t2.get_height() / 2))
            return S
        return self._cached("banner", key, make)

    # place names --------------------------------------------------------------------------------------------
    def _place_tex(self, p):
        """A label for the 3D view: a coloured dot and the name on a dark pill, with a pointer below."""
        def make():
            s = self.s
            t = self.f_place.render(p["name"] if len(p["name"]) < 34 else p["name"][:32] + "...", True, (245, 245, 240))
            pad, dot = int(7 * s), int(5 * s)
            w, h = t.get_width() + 3 * pad + 2 * dot, t.get_height() + int(8 * s)
            S = pygame.Surface((w, h + int(6 * s)), pygame.SRCALPHA)
            pygame.draw.rect(S, (18, 20, 24, 205), pygame.Rect(0, 0, w, h), border_radius=int(h / 2))
            pygame.draw.polygon(S, (18, 20, 24, 205), [(w / 2 - 5 * s, h), (w / 2 + 5 * s, h), (w / 2, h + 6 * s)])
            pygame.gfxdraw.filled_circle(S, int(pad + dot), int(h / 2), dot, (*p["colour"], 255))
            pygame.gfxdraw.aacircle(S, int(pad + dot), int(h / 2), dot, (*p["colour"], 255))
            S.blit(t, (2 * pad + 2 * dot, h / 2 - t.get_height() / 2))
            return S
        return self._cached("place:" + p["name"], "place", make)

    def _place_labels(self, avoid):
        """Names of the places in view, most important and nearest first, none overlapping."""
        r = self.r
        W, H = r.fb_size
        e = r.world.ego
        most = 3 if r.zoom <= 2 else 2 if r.zoom <= 5 else 1
        cands = [p for t in r.in_view for p in self.place_tiles.get(t, ()) if 0 < p["rank"] <= most]
        cands.sort(key=lambda p: (p["rank"], (p["x"] - e.x) ** 2 + (p["y"] - e.y) ** 2))
        placed, n = list(avoid), 0
        m = r.mvp
        for p in cands:
            cx = m[0, 0] * p["x"] + m[0, 1] * p["y"] + m[0, 2] * 3.0 + m[0, 3]
            cy = m[1, 0] * p["x"] + m[1, 1] * p["y"] + m[1, 2] * 3.0 + m[1, 3]
            cw = m[3, 0] * p["x"] + m[3, 1] * p["y"] + m[3, 2] * 3.0 + m[3, 3]
            if cw <= 0.1 or abs(cx / cw) > 0.95 or abs(cy / cw) > 0.95:
                continue
            sx, sy = (cx / cw + 1) / 2 * W, (1 - cy / cw) / 2 * H
            tex = self._place_tex(p)
            w, h = tex.size
            rect = (sx - w / 2, sy - h, w, h)
            if any(rect[0] < q[0] + q[2] and q[0] < rect[0] + w and rect[1] < q[1] + q[3] and q[1] < rect[1] + h for q in placed):
                continue
            placed.append(rect)
            self._blit(tex, rect[0], rect[1], 0.95)
            n += 1
            if n >= 14:
                break

    def _place_marks(self, S, cx, cy, hw, hh, pw, ph, most, areas=False):
        """Places on a map surface: area names large and faint (full map), then dots with names."""
        placed = []
        def free(rect):
            return not any(rect.colliderect(q) for q in placed)
        if areas:
            for p in self.places:
                if p["rank"] != 0:
                    continue
                sx, sy = (p["x"] - (cx - hw)) / (2 * hw) * pw, (1 - (p["y"] - (cy - hh)) / (2 * hh)) * ph
                if not (0 <= sx <= pw and 0 <= sy <= ph):
                    continue
                t = self.f_area.render(p["name"].upper(), True, (120, 116, 106))
                rect = t.get_rect(center=(sx, sy))
                if free(rect):
                    t.set_alpha(150)
                    S.blit(t, rect); placed.append(rect)
        for p in sorted(self.places, key=lambda p: p["rank"]):
            if not 0 < p["rank"] <= most:
                continue
            sx, sy = (p["x"] - (cx - hw)) / (2 * hw) * pw, (1 - (p["y"] - (cy - hh)) / (2 * hh)) * ph
            if not (0 <= sx <= pw and 0 <= sy <= ph):
                continue
            name = p["name"] if len(p["name"]) < 30 else p["name"][:28] + "..."
            t = self.f_small.render(name, True, tuple(int(c * 0.75) for c in p["colour"]))
            r_ = max(3, int(4 * self.s))
            rect = pygame.Rect(sx - r_, sy - t.get_height() / 2, t.get_width() + 2 * r_ + 4, t.get_height())
            if not free(rect):
                continue
            pygame.gfxdraw.filled_circle(S, int(sx), int(sy), r_ + 1, (255, 255, 255))
            pygame.gfxdraw.filled_circle(S, int(sx), int(sy), r_, p["colour"])
            halo = self.f_small.render(name, True, (255, 255, 255))
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                S.blit(halo, (sx + r_ + 4 + dx, sy - t.get_height() / 2 + dy))
            S.blit(t, (sx + r_ + 4, sy - t.get_height() / 2))
            placed.append(rect)

    # traffic light countdown --------------------------------------------------------------------------------
    def _our_signal(self):
        return self.r.world.signal_ahead()

    def _countdown(self, sig):
        """A countdown like the timers on Gurgaon's signals: the light's colour and the seconds left."""
        _, st, light, left = sig
        col = {"red": (225, 45, 38), "yellow": (240, 180, 20), "green": (40, 180, 80)}[light]
        key = (light, int(math.ceil(left)))

        def make():
            s = self.s
            t1 = self.f_mid.render(light.upper(), True, (255, 255, 255))
            t2 = self.f_big.render(f"{int(math.ceil(left))} s", True, (255, 255, 255))
            S = pygame.Surface((int(150 * s), t2.get_height() + int(14 * s)), pygame.SRCALPHA)
            pygame.draw.rect(S, (*col, 235), S.get_rect(), border_radius=int(12 * s))
            S.blit(t1, (12 * s, S.get_height() / 2 - t1.get_height() / 2))
            S.blit(t2, (S.get_width() - t2.get_width() - 12 * s, S.get_height() / 2 - t2.get_height() / 2))
            return S
        return self._cached("countdown", key, make)

    def _head_timer(self, sig):
        """The same seconds shown on the signal head itself, as the timer boxes beside Gurgaon's lights do."""
        _, st, light, left = sig
        (ax, ay), (bx, by) = st["a"], st["b"]
        L = math.hypot(bx - ax, by - ay) or 1.0
        px, py = ax + (ax - bx) / L * 0.6, ay + (ay - by) / L * 0.6   # the pole, as render3d places it
        p = self._screen(px, py, 4.8)
        if p is None:
            return
        col = {"red": (255, 70, 60), "yellow": (255, 200, 40), "green": (70, 230, 110)}[light]

        def make():
            t = self.f_mid.render(f"{int(math.ceil(left))}", True, col)
            S = pygame.Surface((t.get_width() + int(12 * self.s), t.get_height() + int(6 * self.s)), pygame.SRCALPHA)
            pygame.draw.rect(S, (12, 12, 14, 235), S.get_rect(), border_radius=int(5 * self.s))
            S.blit(t, (6 * self.s, 3 * self.s))
            return S
        tex = self._cached("headtimer", (light, int(math.ceil(left))), make)
        self._blit(tex, p[0] - tex.size[0] / 2, p[1] - tex.size[1] - 4 * self.s)

    # the road user an event is about ----------------------------------------------------------------------
    def _tag_tex(self, text, arrow=None):
        """An amber tag; with arrow (screen angle in degrees, 0 = right, 90 = up) a pointer in front of the text."""
        def make():
            s = self.s
            t = self.f_mid.render(text, True, (20, 20, 20))
            pad = int(10 * s)
            aw = int(18 * s) if arrow is not None else 0
            S = pygame.Surface((t.get_width() + 2 * pad + aw, t.get_height() + int(10 * s)), pygame.SRCALPHA)
            pygame.draw.rect(S, (255, 176, 40, 240), S.get_rect(), border_radius=int(S.get_height() / 2))
            if arrow is not None:
                cx, cy, r = pad + aw / 2 - 2 * s, S.get_height() / 2, 7 * s
                a = math.radians(arrow)
                pts = [(cx + math.cos(a) * r, cy - math.sin(a) * r), (cx + math.cos(a + 2.4) * r, cy - math.sin(a + 2.4) * r),
                       (cx + math.cos(a - 2.4) * r, cy - math.sin(a - 2.4) * r)]
                pygame.gfxdraw.filled_polygon(S, pts, (20, 20, 20)); pygame.gfxdraw.aapolygon(S, pts, (20, 20, 20))
            S.blit(t, (pad + aw, S.get_height() / 2 - t.get_height() / 2))
            return S
        return self._cached(f"tag:{text}:{arrow}", "tag", make)

    def _screen(self, x, y, z):
        """Where a world point lands on screen (framebuffer pixels, from the top left), or None behind us."""
        m, (W, H) = self.r.mvp, self.r.fb_size
        cx = m[0, 0] * x + m[0, 1] * y + m[0, 2] * z + m[0, 3]
        cy = m[1, 0] * x + m[1, 1] * y + m[1, 2] * z + m[1, 3]
        cw = m[3, 0] * x + m[3, 1] * y + m[3, 2] * z + m[3, 3]
        if cw <= 0.1:
            return None
        return (cx / cw + 1) / 2 * W, (1 - cy / cw) / 2 * H

    def _event_tags(self):
        """A tag above each road user an event is about; one off screen gets the tag at the screen edge, pointing
        at it."""
        r = self.r
        ev_sys = r.world.events
        if not ev_sys or not self.banner_on:
            return
        from .events import TAG
        W, H = r.fb_size
        s = self.s
        e = r.world.ego
        for ev in ev_sys.active:
            u = ev.actor
            if not getattr(ev, "highlight", False) or u is None:
                continue
            text = TAG.get(ev.label, ev.label.upper())
            z = 3.5 if u.kind in ("bus", "truck") else 2.2
            p = self._screen(u.x, u.y, z)
            if p is not None and 40 * s < p[0] < W - 40 * s and 60 * s < p[1] < H - 40 * s:
                tex = self._tag_tex(text)
                self._blit(tex, p[0] - tex.size[0] / 2, p[1] - tex.size[1] - 6 * s)
                continue
            # off screen: from our car towards it, stopped at the screen edge
            me = self._screen(e.x, e.y, 1.0)
            d = math.hypot(u.x - e.x, u.y - e.y) or 1.0
            toward = self._screen(e.x + (u.x - e.x) / d * 4.0, e.y + (u.y - e.y) / d * 4.0, 1.0)
            if me is None or toward is None:
                continue
            dx, dy = toward[0] - me[0], toward[1] - me[1]
            n = math.hypot(dx, dy) or 1.0
            dx, dy = dx / n, dy / n
            k = min((W / 2 - 90 * s) / abs(dx) if abs(dx) > 1e-6 else 1e9, (H / 2 - 60 * s) / abs(dy) if abs(dy) > 1e-6 else 1e9)
            ang = round(math.degrees(math.atan2(-dy, dx)) / 15) * 15
            tex = self._tag_tex(text, ang)
            self._blit(tex, W / 2 + dx * k - tex.size[0] / 2, H / 2 + dy * k - tex.size[1] / 2)

    def _help(self):
        def make():
            text = "arrows drive   H horn   F dipper   space emergency brake   + - zoom   T tilt   C north up   [ ] minimap   M map   B event banner   R restart   K keys   Esc quit"
            t = self.f.render(text, True, (215, 215, 215))
            S = pygame.Surface((t.get_width() + int(24 * self.s), t.get_height() + int(12 * self.s)), pygame.SRCALPHA)
            pygame.draw.rect(S, (16, 18, 22, 190), S.get_rect(), border_radius=int(10 * self.s))
            S.blit(t, (12 * self.s, 6 * self.s))
            return S
        return self._cached("help", "help", make)

    def _minimap_overlay(self):
        """Frame, north arrow, our car, road names and the scale over the minimap."""
        r, s = self.r, self.s
        cx, cy, hw, hh, px, py, pw, ph = r.map_proj
        e = r.world.ego
        key = (round(cx / (hw / 120)), round(cy / (hw / 120)), hw, round(math.degrees(e.heading) / 3))

        def make():
            S = pygame.Surface((pw, ph + int(22 * s)), pygame.SRCALPHA)
            self._labels(S, cx, cy, hw, hh, pw, ph, max_rank=3 if hw <= 500 else 2 if hw <= 1000 else 1)
            if hw <= 500:
                self._place_marks(S, cx, cy, hw, hh, pw, ph, most=1)
            pygame.draw.rect(S, (20, 22, 26), pygame.Rect(0, 0, pw, ph), max(2, int(3 * s)), border_radius=int(4 * s))
            self._marker(S, pw / 2, ph / 2, e.heading, 9 * s)
            nx, ny = pw - 18 * s, 20 * s
            pygame.gfxdraw.filled_circle(S, int(nx), int(ny), int(12 * s), (20, 22, 26, 220))
            t = self.f_label.render("N", True, (255, 255, 255)); S.blit(t, (nx - t.get_width() / 2, ny - t.get_height() / 2))
            bar = self._nice(hw * 0.4)
            L = bar / (2 * hw) * pw
            y = ph - 14 * s
            pygame.draw.rect(S, (20, 22, 26), pygame.Rect(10 * s, y, L, 4 * s))
            t = self.f_small.render(f"{bar:.0f} m" if bar < 1000 else f"{bar / 1000:g} km", True, (20, 22, 26))
            S.blit(t, (14 * s + L, y - 6 * s))
            t = self.f_small.render("[ ] zoom    M full map", True, (230, 230, 230))
            box = pygame.Rect(0, ph + 2 * s, pw, 20 * s)
            pygame.draw.rect(S, (16, 18, 22, 200), box, border_radius=int(6 * s))
            S.blit(t, (box.centerx - t.get_width() / 2, box.centery - t.get_height() / 2))
            return S
        return self._cached("mini", key, make)

    @staticmethod
    def _nice(x):
        p = 10 ** math.floor(math.log10(x))
        return max(m * p for m in (1, 2, 5) if m * p <= x)

    def _marker(self, S, x, y, heading, size):
        """Our car on a map: a red arrow with a white edge, pointing where we head (north is up)."""
        a = heading
        pts = [(x + math.cos(a) * size * 1.3, y - math.sin(a) * size * 1.3),
               (x + math.cos(a + 2.5) * size, y - math.sin(a + 2.5) * size),
               (x - math.cos(a) * size * 0.35, y + math.sin(a) * size * 0.35),
               (x + math.cos(a - 2.5) * size, y - math.sin(a - 2.5) * size)]
        pygame.gfxdraw.filled_polygon(S, pts, (230, 40, 35)); pygame.gfxdraw.aapolygon(S, pts, (255, 255, 255))
        pygame.draw.polygon(S, (255, 255, 255), pts, max(1, int(1.5 * self.s)))

    def _labels(self, S, cx, cy, hw, hh, pw, ph, max_rank):
        for name, x, y, ang, rank in self.r.labels:
            if rank > max_rank:
                continue
            sx, sy = (x - (cx - hw)) / (2 * hw) * pw, (1 - (y - (cy - hh)) / (2 * hh)) * ph
            if not (0 <= sx <= pw and 0 <= sy <= ph):
                continue
            t = self.f_label.render(name, True, (60, 58, 52))
            halo = self.f_label.render(name, True, (255, 255, 255))
            box = pygame.Surface((t.get_width() + 4, t.get_height() + 4), pygame.SRCALPHA)
            for dx, dy in ((0, 0), (4, 0), (0, 4), (4, 4), (2, 0), (0, 2), (4, 2), (2, 4)):
                box.blit(halo, (dx, dy))
            box.blit(t, (2, 2))
            rot = pygame.transform.rotozoom(box, ang, 1.0)
            S.blit(rot, (sx - rot.get_width() / 2, sy - rot.get_height() / 2))

    def _big_map_overlay(self):
        r, s = self.r, self.s
        cx, cy, hw, hh, px, py, pw, ph = r.map_proj
        mx, my = getattr(r, "cursor", (0, 0))   # window pixels, from the top left
        key = (cx, cy, hw, mx, my)

        def make():
            S = pygame.Surface((pw, ph), pygame.SRCALPHA)
            # latitude and longitude lines, labelled at the left and bottom edges
            lon0, lat0 = to_lonlat(cx - hw, cy - hh)
            lon1, lat1 = to_lonlat(cx + hw, cy + hh)
            step = next(st for st in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1) if (lon1 - lon0) / st < 12)
            to_px = lambda x, y: ((x - (cx - hw)) / (2 * hw) * pw, (1 - (y - (cy - hh)) / (2 * hh)) * ph)
            v = math.floor(lon0 / step) * step
            while v <= lon1:
                a, b = to_px(*from_lonlat(v, lat0)), to_px(*from_lonlat(v, lat1))
                pygame.draw.aaline(S, (40, 60, 90, 120), a, b)
                t = self.f_small.render(f"{v:.3f} E", True, (40, 60, 90))
                S.blit(t, (a[0] + 4 * s, ph - 40 * s))
                v += step
            v = math.floor(lat0 / step) * step
            while v <= lat1:
                a, b = to_px(*from_lonlat(lon0, v)), to_px(*from_lonlat(lon1, v))
                pygame.draw.aaline(S, (40, 60, 90, 120), a, b)
                t = self.f_small.render(f"{v:.3f} N", True, (40, 60, 90))
                S.blit(t, (8 * s, a[1] - t.get_height() - 2 * s))
                v += step
            self._place_marks(S, cx, cy, hw, hh, pw, ph, most=3 if hw < 400 else 2 if hw < 1200 else 1, areas=hw >= 350)
            self._labels(S, cx, cy, hw, hh, pw, ph, max_rank=3 if hw < 1200 else 2 if hw < 2500 else 1)
            e = r.world.ego
            ex, ey = to_px(e.x, e.y)
            self._marker(S, ex, ey, e.heading, 12 * s)
            wx = cx - hw + mx * s / pw * 2 * hw
            wy = cy + hh - my * s / ph * 2 * hh
            lon, lat = to_lonlat(wx, wy)
            bar = pygame.Rect(0, 0, pw, int(34 * s))
            pygame.draw.rect(S, (16, 18, 22, 220), bar)
            text = f"Gurgaon   cursor {lat:.5f} N, {lon:.5f} E      wheel zoom   drag to move   C centre on our car   M or Esc close   (paused)"
            S.blit(self.f.render(text, True, TEXT), (14 * s, 9 * s))
            sc = self._nice(hw * 0.25)
            L = sc / (2 * hw) * pw
            pygame.draw.rect(S, (20, 22, 26), pygame.Rect(pw - L - 30 * s, ph - 30 * s, L, 5 * s))
            t = self.f_small.render(f"{sc:.0f} m" if sc < 1000 else f"{sc / 1000:g} km", True, (20, 22, 26))
            S.blit(t, (pw - L - 30 * s, ph - 48 * s))
            return S
        return self._cached("bigmap", key, make)

    # one frame --------------------------------------------------------------------------------------------
    def draw(self, dt, big_map=False):
        r, ctx = self.r, self.r.ctx
        import moderngl
        r.out.use()
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        W, H = r.fb_size
        s = self.s
        if big_map:
            self._blit(self._big_map_overlay(), 0, 0)
            ctx.disable(moderngl.BLEND)
            return
        mini_rect = (r.map_proj[4], H - r.map_proj[5] - r.map_proj[7] - 30 * s, r.map_proj[6], r.map_proj[7] + 30 * s)
        log = self._log_panel()
        avoid = [(0, 0, 360 * s, 250 * s), mini_rect, (W / 2 - 400 * s, 0, 800 * s, 120 * s)]
        if log is not None:
            avoid.append((W - log.size[0] - 16 * s, 0, log.size[0] + 16 * s, log.size[1] + 24 * s))
        self._place_labels(avoid)
        self._event_tags()
        sig = self._our_signal()
        if sig is not None:
            self._head_timer(sig)
        self._blit(self._speed_panel(), 16 * s, 16 * s)
        if sig is not None:
            self._blit(self._countdown(sig), 16 * s, 176 * s)
        if log is not None:
            self._blit(log, W - log.size[0] - 16 * s, 16 * s)
        warn = self._warning()
        top = 20 * s
        if warn is not None:
            self._blit(warn, W / 2 - warn.size[0] / 2, top)
            top += warn.size[1] + 10 * s
        ban = self._banner()
        if ban is not None:
            self._blit(ban, W / 2 - ban.size[0] / 2, top)
        mini = self._minimap_overlay()
        cx, cy, hw, hh, px, py, pw, ph = r.map_proj
        self._blit(mini, px, H - py - ph)
        if self.show_help or r.world.t < self.help_until:
            hp = self._help()
            fade = 1.0 if self.show_help and r.world.t >= self.help_until else min(1.0, max(0.0, self.help_until - r.world.t))
            self._blit(hp, 16 * s, H - hp.size[1] - 16 * s, fade if not self.show_help else 1.0)
        ctx.disable(moderngl.BLEND)

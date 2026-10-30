"""Draw the world top-down with pygame, from just above and behind our car.

By default the view turns with the car so it always points up the screen, sitting low in the frame so
more of the road ahead is visible; C switches to a fixed north-up map.

The static map (land, parks, buildings, footpaths, roads, lane dashes, zebra crossings) is drawn in square
tiles once per zoom level and cached, so a frame only blits tiles and draws what moves: signal lights at
the stop lines, our car and the dashboard.
"""
import math
from collections import OrderedDict

import pygame
import pygame.gfxdraw

from .signals import Signals

COL = {"land": (62, 72, 58), "green": (76, 104, 62), "water": (59, 95, 122), "building": (139, 131, 120),
       "building_edge": (110, 103, 94), "walk": (156, 151, 142), "road": (56, 57, 60), "dash": (215, 215, 210),
       "zebra": (228, 228, 224), "ego": (220, 40, 40), "glass": (30, 30, 34),
       "car": (215, 215, 215), "two_wheeler": (255, 191, 0), "auto": (51, 192, 77), "bus": (59, 115, 242),
       "truck": (160, 102, 58), "cycle": (184, 77, 204), "pedestrian": (255, 122, 217),
       "green_light": (46, 204, 64), "yellow_light": (255, 204, 0), "red_light": (255, 59, 48)}
ZOOMS = [5, 8, 12, 16, 22, 30]   # pixels per metre
TILE_PX = 384


class Renderer:
    def __init__(self, world, size=(1280, 800)):
        self.world = world
        self.size = size
        self.zoom = 3
        self.rotate = True
        self.tiles = OrderedDict()
        pygame.font.init()
        self.font = pygame.font.SysFont("Helvetica", 15)
        self.big = pygame.font.SysFont("Helvetica", 30, bold=True)

    @property
    def s(self):
        return ZOOMS[self.zoom]

    def zoom_by(self, d):
        self.zoom = max(0, min(len(ZOOMS) - 1, self.zoom + d))

    # static map tiles ------------------------------------------------------------------------------------
    def _tile(self, i, j):
        key = (i, j, self.zoom)
        if key in self.tiles:
            self.tiles.move_to_end(key)
            return self.tiles[key]
        s, m = self.s, self.world.map
        x0, y1 = i * TILE_PX / s, -j * TILE_PX / s          # world x of the left edge, world y of the top edge
        x1, y0 = x0 + TILE_PX / s, y1 - TILE_PX / s
        surf = pygame.Surface((TILE_PX, TILE_PX))
        surf.fill(COL["land"])
        P = lambda pts: [(int(round((x - x0) * s)), int(round((y1 - y) * s))) for x, y in pts]

        def polys(key, colour, edge=None, level=None):
            lv = m.levels.get(key)
            for k in m.grids[key].near(x0, y0, x1, y1):
                if level is not None and lv is not None and lv[k] != level:
                    continue
                pts = getattr(m, key)[k]
                if len(pts) >= 3:
                    q = P(pts)
                    pygame.gfxdraw.filled_polygon(surf, q, colour)
                    pygame.gfxdraw.aapolygon(surf, q, edge or colour)

        polys("greens", COL["green"])
        polys("water", COL["water"])
        polys("buildings", COL["building"], COL["building_edge"])
        lanes = [k for k in m.grids["lanes"].near(x0, y0, x1, y1) if len(m.lanes[k]["poly"]) >= 3]
        w = max(1, int(round(0.12 * s)))
        for level in sorted({m.lanes[k]["level"] for k in lanes} | {0}):
            # tunnels a shade darker, bridges outlined so they read as passing over what is below
            road = tuple(max(0, c - 12) for c in COL["road"]) if level < 0 else COL["road"]
            polys("walks", COL["walk"], level=level)
            polys("junctions", road, level=level)
            for k in lanes:
                if m.lanes[k]["level"] == level:
                    q = P(m.lanes[k]["poly"])
                    if level > 0:
                        pygame.draw.polygon(surf, (20, 20, 22), q, max(2, int(0.35 * s)))
                    pygame.gfxdraw.filled_polygon(surf, q, road); pygame.gfxdraw.aapolygon(surf, q, road)
            for k in m.grids["markings"].near(x0, y0, x1, y1):
                if m.levels["markings"][k] == level:
                    self._dashes(surf, P, m.markings[k], 3.0, 4.5, w)
            if level == 0:
                for k in m.grids["crossings"].near(x0, y0, x1, y1):
                    self._zebra(surf, P, m.crossings[k])
        self.tiles[key] = surf
        cap = max(40, int(160e6 / (TILE_PX * TILE_PX * 4)))
        while len(self.tiles) > cap:
            self.tiles.popitem(last=False)
        return surf

    def _dashes(self, surf, P, line, dash, gap, width):
        pos, on = 0.0, True
        for (ax, ay), (bx, by) in zip(line, line[1:]):
            L = math.hypot(bx - ax, by - ay)
            u = 0.0
            while u < L:
                step = min((dash if on else gap) - pos, L - u)
                if on:
                    a = (ax + (bx - ax) * u / L, ay + (by - ay) * u / L)
                    b = (ax + (bx - ax) * (u + step) / L, ay + (by - ay) * (u + step) / L)
                    pygame.draw.line(surf, COL["dash"], *P([a, b]), width)
                u += step; pos += step
                if pos >= (dash if on else gap) - 1e-6:
                    pos, on = 0.0, not on

    def _zebra(self, surf, P, c):
        (ax, ay), (bx, by) = c["line"][0], c["line"][-1]
        L = math.hypot(bx - ax, by - ay) or 1
        ux, uy = (bx - ax) / L, (by - ay) / L          # across the road
        nx, ny = -uy * c["w"] / 2 * 0.9, ux * c["w"] / 2 * 0.9
        u = 0.3
        while u + 0.5 < L:
            p0, p1 = (ax + ux * u, ay + uy * u), (ax + ux * (u + 0.5), ay + uy * (u + 0.5))
            q = P([(p0[0] + nx, p0[1] + ny), (p1[0] + nx, p1[1] + ny), (p1[0] - nx, p1[1] - ny), (p0[0] - nx, p0[1] - ny)])
            pygame.gfxdraw.filled_polygon(surf, q, COL["zebra"])
            u += 1.0

    # one frame --------------------------------------------------------------------------------------------
    def draw(self, screen, hud=True):
        W, H = screen.get_size()
        e, s = self.world.ego, self.s
        # tell the traffic what is on screen, so nobody appears in view
        if self.rotate:
            self.world.view = (0.75 * H / s, 0.25 * H / s, 0.5 * W / s)
        else:
            r = 0.5 * math.hypot(W, H) / s + 12
            self.world.view = (r, r, r)
        if self.rotate:
            ahead = 0.25 * H / s   # our car sits a quarter of the screen below the centre
            cx, cy = e.x + math.cos(e.heading) * ahead, e.y + math.sin(e.heading) * ahead
            D = int(math.hypot(W, H)) + 4
            canvas = pygame.Surface((D, D))
            self._scene(canvas, cx, cy)
            turned = pygame.transform.rotozoom(canvas, 90 - math.degrees(e.heading), 1.0)
            screen.fill(COL["land"])
            screen.blit(turned, turned.get_rect(center=(W // 2, H // 2)))
        else:
            look = min(12.0, max(0.0, e.speed) * 0.8)
            self._scene(screen, e.x + math.cos(e.heading) * look, e.y + math.sin(e.heading) * look)
        if hud:
            self._hud(screen)

    def _scene(self, surf, cx, cy):
        """The map, signal lights and our car, north up, with (cx, cy) at the centre of surf."""
        W, H = surf.get_size()
        e, s = self.world.ego, self.s
        ox, oy = int(round(cx * s)) - W // 2, int(round(-cy * s)) - H // 2   # global pixel of the surface's corner
        i0, i1 = math.floor(ox / TILE_PX), math.floor((ox + W) / TILE_PX)
        j0, j1 = math.floor(oy / TILE_PX), math.floor((oy + H) / TILE_PX)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                surf.blit(self._tile(i, j), (i * TILE_PX - ox, j * TILE_PX - oy))
        # draw one tile of the ring just outside the view per frame, nearest the way we are heading first, so
        # tiles are ready before they come into view instead of stalling a frame
        hx, hy = math.cos(e.heading), -math.sin(e.heading)
        mid_i, mid_j = (i0 + i1) / 2, (j0 + j1) / 2
        ring = [(i, j) for i in range(i0 - 2, i1 + 3) for j in range(j0 - 2, j1 + 3)
                if not (i0 <= i <= i1 and j0 <= j <= j1) and (i, j, self.zoom) not in self.tiles]
        if ring:
            self._tile(*max(ring, key=lambda ij: (ij[0] - mid_i) * hx + (ij[1] - mid_j) * hy))
        S = lambda x, y: (x * s - ox, -y * s - oy)
        m, t = self.world.map, self.world.t
        x0, x1, y0, y1 = cx - W / s / 2, cx + W / s / 2, cy - H / s / 2, cy + H / s / 2
        for k, sg in enumerate(m.signals):
            st = self.world.signals.state(k, t)
            for stop in sg["stops"]:
                if x0 - 20 < stop["a"][0] < x1 + 20 and y0 - 20 < stop["a"][1] < y1 + 20:
                    colour = COL[Signals.light(st, stop["links"]) + "_light"]
                    pygame.draw.line(surf, colour, S(*stop["a"]), S(*stop["b"]), max(2, int(0.5 * s)))
        if self.world.traffic:
            for u in self.world.traffic.all():
                if x0 - 10 < u.x < x1 + 10 and y0 - 10 < u.y < y1 + 10:
                    self._actor(surf, S, u)
        self._car(surf, S, e, COL["ego"])

    def _actor(self, surf, S, u):
        if u.kind == "pedestrian":
            x, y = S(u.x, u.y)
            r = max(2, int(0.3 * self.s))
            pygame.gfxdraw.filled_circle(surf, int(x), int(y), r, COL["pedestrian"]); pygame.gfxdraw.aacircle(surf, int(x), int(y), r, (60, 20, 50))
            return
        c, n = math.cos(u.heading), math.sin(u.heading)
        hl, hw = u.length / 2, u.width / 2
        body = [S(u.x + c * dx - n * dy, u.y + n * dx + c * dy) for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
        pygame.gfxdraw.filled_polygon(surf, body, (255, 120, 0) if u.crashed else COL[u.kind]); pygame.gfxdraw.aapolygon(surf, body, (25, 25, 25))
        a, b = hl - min(1.0, u.length * 0.22), hl - min(1.6, u.length * 0.36)
        w = hw - 0.12
        glass = [S(u.x + c * dx - n * dy, u.y + n * dx + c * dy) for dx, dy in ((a, w), (a, -w), (b, -w), (b, w))]
        pygame.gfxdraw.filled_polygon(surf, glass, COL["glass"])

    def _car(self, screen, S, car, colour):
        q = [S(x, y) for x, y in car.corners()]
        pygame.gfxdraw.filled_polygon(screen, q, colour); pygame.gfxdraw.aapolygon(screen, q, (20, 20, 20))
        c, n, sp = math.cos(car.heading), math.sin(car.heading), car.spec
        a = sp.length / 2 - 1.0; b = sp.length / 2 - 1.7; w = sp.width / 2 - 0.15
        glass = [(car.x + c * dx - n * dy, car.y + n * dx + c * dy) for dx, dy in ((a, w), (a, -w), (b, -w), (b, w))]
        pygame.gfxdraw.filled_polygon(screen, [S(x, y) for x, y in glass], COL["glass"])

    def _compass(self, screen):
        """A small arrow showing which way north is on screen."""
        W, H = screen.get_size()
        cxp, cyp, r = W - 40, H - 46, 16
        ang = math.radians(90 + (90 - math.degrees(self.world.ego.heading) if self.rotate else 0))   # north, anticlockwise from screen right
        pygame.gfxdraw.filled_circle(screen, cxp, cyp, r + 6, (15, 15, 15, 200))
        tip = (cxp + math.cos(ang) * r, cyp - math.sin(ang) * r)
        base = [(cxp + math.cos(ang + a) * r * 0.55, cyp - math.sin(ang + a) * r * 0.55) for a in (2.5, -2.5)]
        pygame.gfxdraw.filled_polygon(screen, [tip, *base], (255, 90, 70))
        n = self.font.render("N", True, (240, 240, 240))
        screen.blit(n, (cxp + math.cos(ang) * (r + 14) - n.get_width() / 2, cyp - math.sin(ang) * (r + 14) - n.get_height() / 2))

    def _hud(self, screen):
        st, W = self.world.status, screen.get_width()
        box = pygame.Surface((300, 118), pygame.SRCALPHA); box.fill((15, 15, 15, 200)); screen.blit(box, (12, 12))
        screen.blit(self.big.render(f"{abs(st['speed_kmh']):.0f} km/h", True, (240, 240, 240)), (24, 20))
        lim = f"limit {st['limit_kmh']} km/h" if st["limit_kmh"] else ""
        screen.blit(self.font.render(f"{st['road'] or 'off the map roads'}", True, (220, 220, 220)), (24, 60))
        screen.blit(self.font.render(lim, True, (200, 200, 200)), (24, 80))
        screen.blit(self.font.render(f"time {int(self.world.t // 60):02d}:{int(self.world.t % 60):02d}", True, (170, 170, 170)), (24, 100))
        crash = self.world.crashes[-1] if self.world.crashes and self.world.t - self.world.crashes[-1]["t"] < 3 else None
        warn = f"CRASH: {crash['kind'].replace('_', '-')}" if crash else "OFF THE ROAD" if st["on_road"] == 0 else \
            "WHEELS OFF THE ROAD" if st["on_road"] < 4 else "WRONG WAY" if st["wrong_way"] else ""
        if warn:
            t = self.big.render(warn, True, (255, 80, 60)); screen.blit(t, (W // 2 - t.get_width() // 2, 18))
        log = self.world.log[-6:]
        if log:
            box = pygame.Surface((300, 22 + 20 * len(log)), pygame.SRCALPHA); box.fill((15, 15, 15, 200)); screen.blit(box, (W - 312, 12))
            screen.blit(self.font.render("Log", True, (230, 230, 230)), (W - 300, 16))
            for k, (tt, what) in enumerate(log):
                screen.blit(self.font.render(f"{int(tt // 60):02d}:{int(tt % 60):02d}  {what}", True, (255, 150, 130)), (W - 300, 36 + 20 * k))
        self._compass(screen)
        help_ = "arrows drive   space emergency brake   hold down when stopped to reverse   + - zoom   C turn view / north up   R restart   Esc quit"
        t = self.font.render(help_, True, (200, 200, 200))
        box = pygame.Surface((t.get_width() + 16, 24), pygame.SRCALPHA); box.fill((15, 15, 15, 170))
        screen.blit(box, (12, screen.get_height() - 36)); screen.blit(t, (20, screen.get_height() - 32))

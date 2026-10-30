"""Draw the world on the GPU (OpenGL 3.3 through moderngl): a 3D scene seen from above and behind our car.

The static scene (scene3d.py) is uploaded once; each frame only the camera, the road users and the signal
lights change. Shadows: everything that stands up (buildings, trees, vehicles) is flattened onto the ground
along the sun's direction into a screen-sized mask, which darkens the ground where it falls. Far ground
fades into the haze. The picture is drawn with 4x multisampling and then resolved to the window.

Camera: it follows our car and turns with it (C: fixed north-up); + and - move it nearer or further, T
cycles the tilt (straight down, three-quarter, low). The minimap in the corner is drawn from the same data
with roads at least a few pixels wide, north up; [ and ] zoom it. M opens the full map with a
latitude/longitude grid (wheel zooms, drag pans).

Positions are blended between the last two simulation steps (alpha), so motion stays smooth at any
refresh rate.
"""
import math, time

import moderngl
import numpy as np

from .geo import to_lonlat
from .scene3d import TILE, load_scene
from .models import ego_boxes, in_tunnel, place, signal_boxes, user_boxes

ZOOMS = [20.0, 28.0, 40.0, 56.0, 78.0, 110.0, 160.0, 240.0]   # camera distance from its target, metres
TILTS = [64.0, 90.0, 46.0]                                     # degrees above the horizon
MINIMAP = [120.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]        # half-width of the minimap, metres
FOG = (0.79, 0.79, 0.75)
SUN = np.array([0.45, -0.40, 0.80]); SUN = SUN / np.linalg.norm(SUN)
SHADOW = (-SUN[0] / SUN[2], -SUN[1] / SUN[2])                  # ground offset per metre of height

LIT_VS = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos; in vec3 in_col; in vec3 in_nrm; in float in_recv;
out vec3 v_col; out vec3 v_nrm; out float v_recv; out vec3 v_world;
void main() { gl_Position = u_mvp * vec4(in_pos, 1.0); v_col = in_col; v_nrm = in_nrm; v_recv = in_recv; v_world = in_pos; }
"""
LIGHT = """
vec3 light(vec3 col, vec3 n, vec3 L, float amb) {
    // a cool sky from above, a warm sun from the south-east
    float d = max(dot(normalize(n), L), 0.0);
    vec3 sky = mix(vec3(0.50, 0.53, 0.60), vec3(0.62, 0.64, 0.68), clamp(n.z, 0.0, 1.0));
    return col * mix(sky + vec3(1.00, 0.93, 0.80) * 0.52 * d, vec3(1.0), amb);
}
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
"""
LIT_FS = """
#version 330
uniform vec3 u_light; uniform sampler2D u_shadow; uniform vec2 u_vp; uniform vec3 u_eye; uniform vec3 u_fog;
uniform vec2 u_fogr; uniform float u_amb; uniform int u_patches;
in vec3 v_col; in vec3 v_nrm; in float v_recv; in vec3 v_world;
out vec4 f;
""" + LIGHT + """
void main() {
    vec3 col = v_col;
    if (u_patches == 1)   // open ground: dry grass and dust in patches
        col = mix(col, col * vec3(0.80, 0.98, 0.72), smoothstep(0.42, 0.72, noise(v_world.xy * 0.018) * 0.7 + noise(v_world.xy * 0.09) * 0.3));
    vec3 c = light(col, v_nrm, u_light, u_amb);
    if (v_nrm.z > 0.9 && u_amb < 0.99)   // faint grain and patches on flat ground, asphalt and paving
        c *= 0.93 + 0.05 * noise(v_world.xy * 0.45) + 0.04 * noise(v_world.xy * 0.035);
    float sh = v_recv * texture(u_shadow, gl_FragCoord.xy / u_vp).r;
    c *= mix(vec3(1.0), vec3(0.58, 0.62, 0.74), sh);
    float k = clamp((length(v_world - u_eye) - u_fogr.x) / (u_fogr.y - u_fogr.x), 0.0, 1.0);
    f = vec4(mix(c, u_fog, k), 1.0);
}
"""
INST_VS = """
#version 330
uniform mat4 u_mvp; uniform vec2 u_sdir; uniform int u_shadowpass;
in vec3 in_pos; in vec3 in_nrm;
in vec3 i_pos; in float i_head; in vec3 i_size; in vec3 i_col; in float i_emit;
out vec3 v_col; out vec3 v_nrm; out float v_emit; out vec3 v_world;
void main() {
    vec3 p = in_pos * i_size;
    float c = cos(i_head), s = sin(i_head);
    p = vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z) + i_pos;
    v_nrm = vec3(c * in_nrm.x - s * in_nrm.y, s * in_nrm.x + c * in_nrm.y, in_nrm.z);
    if (u_shadowpass == 1) p = vec3(p.xy + p.z * u_sdir, 0.1);
    v_world = p; v_col = i_col; v_emit = i_emit;
    gl_Position = u_mvp * vec4(p, 1.0);
}
"""
INST_FS = """
#version 330
uniform vec3 u_light; uniform vec3 u_eye; uniform vec3 u_fog; uniform vec2 u_fogr; uniform float u_amb; uniform int u_shadowpass;
in vec3 v_col; in vec3 v_nrm; in float v_emit; in vec3 v_world;
out vec4 f;
""" + LIGHT + """
void main() {
    if (u_shadowpass == 1) { f = vec4(1.0); return; }
    vec3 c = mix(light(v_col, v_nrm, u_light, u_amb), v_col * 1.15, v_emit);
    float k = clamp((length(v_world - u_eye) - u_fogr.x) / (u_fogr.y - u_fogr.x), 0.0, 1.0);
    f = vec4(mix(c, u_fog, k * (1.0 - v_emit)), 1.0);
}
"""
SHADOW_VS = """
#version 330
uniform mat4 u_mvp; uniform vec2 u_sdir;
in vec3 in_pos;
void main() { gl_Position = u_mvp * vec4(in_pos.xy + in_pos.z * u_sdir, 0.1, 1.0); }
"""
SHADOW_FS = """
#version 330
out vec4 f;
void main() { f = vec4(1.0); }
"""
FX_VS = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos; in vec2 in_uv; in vec4 in_col; in float in_kind;
out vec2 v_uv; out vec4 v_col; flat out float v_kind;
void main() { gl_Position = u_mvp * vec4(in_pos, 1.0); v_uv = in_uv; v_col = in_col; v_kind = in_kind; }
"""
FX_FS = """
#version 330
in vec2 v_uv; in vec4 v_col; flat in float v_kind;
out vec4 f;
void main() {
    float a;
    float r = length(v_uv);
    if (v_kind < 0.5) a = smoothstep(0.70, 0.86, r) * (1.0 - smoothstep(0.90, 1.0, r));      // ring
    else if (v_kind < 1.5) a = (1.0 - v_uv.x) * (1.0 - abs(v_uv.y)) * smoothstep(0.0, 0.08, v_uv.x);   // beam
    else a = 1.0 - smoothstep(0.0, 1.0, r);                                                   // soft disc
    f = vec4(v_col.rgb, v_col.a * a);
}
"""
LINE_VS = """
#version 330
uniform mat4 u_mvp; uniform float u_mpp; uniform float u_case;
in vec2 in_pos; in vec2 in_n; in vec3 in_col; in float in_hw; in float in_px;
out vec3 v_col;
void main() {
    float w = max(in_hw, (in_px + u_case) * u_mpp);
    gl_Position = u_mvp * vec4(in_pos + in_n * w, 0.0, 1.0);
    v_col = u_case > 0.0 ? in_col * 0.72 : in_col;
}
"""
LINE_FS = """
#version 330
in vec3 v_col; out vec4 f;
void main() { f = vec4(v_col, 1.0); }
"""
UI_VS = """
#version 330
in vec2 in_pos; in vec2 in_uv; out vec2 v_uv;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); v_uv = in_uv; }
"""
UI_FS = """
#version 330
uniform sampler2D u_tex; uniform float u_alpha;
in vec2 v_uv; out vec4 f;
void main() { vec4 c = texture(u_tex, v_uv); f = vec4(c.rgb, c.a * u_alpha); }
"""


def _perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy) / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0], [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]], dtype="f8")


def _look_at(eye, target, up):
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def _ortho(l, r, b, t):
    return np.array([[2 / (r - l), 0, 0, -(r + l) / (r - l)], [0, 2 / (t - b), 0, -(t + b) / (t - b)], [0, 0, -0.001, 0], [0, 0, 0, 1]], dtype="f8")


def _mat(m):
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()


def _cube():
    v = []
    faces = [((1, 0, 0), [(0.5, -0.5, 0), (0.5, 0.5, 0), (0.5, 0.5, 1), (0.5, -0.5, 1)]),
             ((-1, 0, 0), [(-0.5, 0.5, 0), (-0.5, -0.5, 0), (-0.5, -0.5, 1), (-0.5, 0.5, 1)]),
             ((0, 1, 0), [(0.5, 0.5, 0), (-0.5, 0.5, 0), (-0.5, 0.5, 1), (0.5, 0.5, 1)]),
             ((0, -1, 0), [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, -0.5, 1), (-0.5, -0.5, 1)]),
             ((0, 0, 1), [(-0.5, -0.5, 1), (0.5, -0.5, 1), (0.5, 0.5, 1), (-0.5, 0.5, 1)])]
    for n, (a, b, c, d) in faces:
        for p in (a, b, c, a, c, d):
            v.append((*p, *n))
    return np.array(v, dtype="f4")


def _icosphere():
    t = (1 + 5 ** 0.5) / 2
    vs = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    vs = [np.array(p, dtype="f8") / np.linalg.norm(p) for p in vs]
    fs = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
          (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    out = []
    for a, b, c in fs:   # one subdivision, flat shaded for a low-poly look
        A, B, C = vs[a], vs[b], vs[c]
        ab, bc, ca = [(p + q) / np.linalg.norm(p + q) for p, q in ((A, B), (B, C), (C, A))]
        for tri in ((A, ab, ca), (B, bc, ab), (C, ca, bc), (ab, bc, ca)):
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); n /= np.linalg.norm(n)
            for p in tri:
                out.append((p[0] * 0.5, p[1] * 0.5, p[2] * 0.5 + 0.5, *n))
    return np.array(out, dtype="f4")


class Renderer3D:
    def __init__(self, world, ctx=None, size=(1280, 800), scale=1.0):
        self.world = world
        self.ctx = ctx or moderngl.create_standalone_context(require=330)
        self.offscreen = ctx is None
        self.scale = scale                 # framebuffer pixels per window pixel (2 on a Retina screen)
        self.zoom, self.tilt, self.rotate = 3, 0, True
        self.mini = 1
        self.big_map = False
        self.map_center, self.map_half = None, 1500.0
        self.cam_yaw = None
        self.prev = {}
        self.fx_events = []                # (start time, kind, x, y, heading)
        self.hud = None
        self.last_frame = time.perf_counter()
        ctx = self.ctx
        self.lit = ctx.program(vertex_shader=LIT_VS, fragment_shader=LIT_FS)
        self.inst = ctx.program(vertex_shader=INST_VS, fragment_shader=INST_FS)
        self.shadow = ctx.program(vertex_shader=SHADOW_VS, fragment_shader=SHADOW_FS)
        self.fx = ctx.program(vertex_shader=FX_VS, fragment_shader=FX_FS)
        self.line = ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
        self.ui = ctx.program(vertex_shader=UI_VS, fragment_shader=UI_FS)
        sc = load_scene(world.map)
        self.labels = sc["labels"]
        self.tiles = {k: sc[k + "_tiles"] for k in ("ground", "solid", "map_fill", "map_lines", "boxes", "spheres")}
        fmt = (["3f 3f 3f 1f", "in_pos", "in_col", "in_nrm", "in_recv"])
        self.vao_base = ctx.vertex_array(self.lit, [(ctx.buffer(sc["base"].tobytes()), *fmt)])
        self.vbo_ground = ctx.buffer(sc["ground"].tobytes())
        self.vbo_solid = ctx.buffer(sc["solid"].tobytes())
        self.vao_ground = ctx.vertex_array(self.lit, [(self.vbo_ground, *fmt)])
        self.vao_solid = ctx.vertex_array(self.lit, [(self.vbo_solid, *fmt)])
        self.vao_solid_shadow = ctx.vertex_array(self.shadow, [(self.vbo_solid, "3f 28x", "in_pos")])
        self.vao_mapfill = ctx.vertex_array(self.lit, [(ctx.buffer(sc["map_fill"].tobytes()), *fmt)])
        self.vao_maplines = ctx.vertex_array(self.line, [(ctx.buffer(sc["map_lines"].tobytes()), "2f 2f 3f 1f 1f", "in_pos", "in_n", "in_col", "in_hw", "in_px")])
        self.cube = ctx.buffer(_cube().tobytes())
        self.sphere = ctx.buffer(_icosphere().tobytes())
        ifmt = "3f 1f 3f 3f 1f/i"
        inames = ("i_pos", "i_head", "i_size", "i_col", "i_emit")
        # trees and rooftop tanks: the ones in view are copied into these each time the squares in view change
        self.all_boxes, self.all_spheres = sc["boxes"], sc["spheres"]
        self.static_boxes = ctx.buffer(reserve=44 * 4096, dynamic=True); self.n_static_boxes = 0
        self.static_spheres = ctx.buffer(reserve=44 * 4096, dynamic=True); self.n_spheres = 0
        self.vao_sboxes = ctx.vertex_array(self.inst, [(self.cube, "3f 3f", "in_pos", "in_nrm"), (self.static_boxes, ifmt, *inames)])
        self.vao_trees = ctx.vertex_array(self.inst, [(self.sphere, "3f 3f", "in_pos", "in_nrm"), (self.static_spheres, ifmt, *inames)])
        self.in_view = frozenset()
        self.dyn = ctx.buffer(reserve=11 * 4 * 4096, dynamic=True)
        self.vao_dyn = ctx.vertex_array(self.inst, [(self.cube, "3f 3f", "in_pos", "in_nrm"), (self.dyn, ifmt, *inames)])
        self.fxbuf = ctx.buffer(reserve=10 * 4 * 6 * 256, dynamic=True)
        self.vao_fx = ctx.vertex_array(self.fx, [(self.fxbuf, "3f 2f 4f 1f", "in_pos", "in_uv", "in_col", "in_kind")])
        self.uibuf = ctx.buffer(reserve=4 * 4 * 6 * 64, dynamic=True)
        self.vao_ui = ctx.vertex_array(self.ui, [(self.uibuf, "2f 2f", "in_pos", "in_uv")])
        self.blank = ctx.texture((1, 1), 1, b"\x00")
        self._targets(size)
        from .hud import Hud
        self.hud = Hud(self)

    # framebuffers ------------------------------------------------------------------------------------------
    def _targets(self, size):
        """Multisampled scene buffer, the shadow mask and (offscreen) the picture, at framebuffer resolution."""
        ctx = self.ctx
        self.size = size
        W, H = int(size[0] * self.scale), int(size[1] * self.scale)
        self.fb_size = (W, H)
        samples = min(4, ctx.max_samples)
        self.msaa = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 4, samples=samples)],
                                    depth_attachment=ctx.depth_renderbuffer((W, H), samples=samples))
        self.mask_tex = ctx.texture((W, H), 1)
        self.mask = ctx.framebuffer(color_attachments=[self.mask_tex])
        if self.offscreen:
            self.out_tex = ctx.texture((W, H), 4)
            self.out = ctx.framebuffer(color_attachments=[self.out_tex])
        else:
            self.out = ctx.screen

    def resize(self, size):
        if tuple(size) != tuple(self.size):
            self._targets(size)

    # camera -------------------------------------------------------------------------------------------------
    def zoom_by(self, d):
        self.zoom = max(0, min(len(ZOOMS) - 1, self.zoom + d))

    def _pose(self, u, alpha):
        """Where a road user is drawn: between its last two simulated positions."""
        p = self.prev.get(id(u))
        if p is None or alpha >= 1:
            return u.x, u.y, u.heading
        x0, y0, h0 = p
        dh = math.atan2(math.sin(u.heading - h0), math.cos(u.heading - h0))
        if abs(u.x - x0) > 8 or abs(u.y - y0) > 8:
            return u.x, u.y, u.heading
        return x0 + (u.x - x0) * alpha, y0 + (u.y - y0) * alpha, h0 + dh * alpha

    def snapshot(self):
        """Call before each simulation step: remember where everyone was."""
        w = self.world
        self.prev = {id(u): (u.x, u.y, u.heading) for u in (w.traffic.all() if w.traffic else ())}
        self.prev[id(w.ego)] = (w.ego.x, w.ego.y, w.ego.heading)

    def _camera(self, alpha, dt):
        e = self.world.ego
        ex, ey, eh = self._pose(e, alpha)
        yaw_goal = eh if self.rotate else math.pi / 2
        if self.cam_yaw is None:
            self.cam_yaw = yaw_goal
        k = 1 - math.exp(-dt * 6.0)
        self.cam_yaw += math.atan2(math.sin(yaw_goal - self.cam_yaw), math.cos(yaw_goal - self.cam_yaw)) * k
        # at speed the camera pulls back a little to show more road ahead; our car stays at the same spot on screen
        self.cam_speed = getattr(self, "cam_speed", 0.0)
        self.cam_speed += (max(0.0, e.speed) - self.cam_speed) * (1 - math.exp(-dt * 1.5))
        d, pitch = ZOOMS[self.zoom] * (1 + min(0.4, self.cam_speed / 35.0)), math.radians(TILTS[self.tilt])
        fwd = np.array([math.cos(self.cam_yaw), math.sin(self.cam_yaw), 0.0])
        ahead = d * (0.28 if TILTS[self.tilt] >= 89 else 0.18)
        target = np.array([ex, ey, 0.0]) + fwd * ahead
        t0, hard = getattr(self.world, "jolt", (-9.0, 0.0))   # a speed breaker or pothole: the view shakes briefly
        age = self.world.t - t0
        if 0 <= age < 0.5:
            k = hard * (1 - age / 0.5) * 0.35
            target = target + np.array([math.sin(age * 71.0), math.cos(age * 53.0), 0.0]) * k
        eye = target - fwd * d * math.cos(pitch) + np.array([0, 0, d * math.sin(pitch)])
        W, H = self.fb_size
        proj = _perspective(42.0, W / H, max(1.0, ZOOMS[self.zoom] * 0.05), 4000.0)   # (a near plane that moves flickers)
        view = _look_at(eye, target, fwd if TILTS[self.tilt] >= 89 else np.array([0.0, 0.0, 1.0]))
        self.eye, self.target, self.mvp = eye, target, proj @ view
        self.fog = (d * 1.6, d * 4.0)
        self._visible(ex, ey, eh)

    def _visible(self, ex, ey, eh):
        """Where the screen meets the ground (cut off at the haze): tells the traffic how far the screen
        reaches around our car, so nobody appears in view, and picks the squares of the scene to draw."""
        inv = np.linalg.inv(self.mvp)
        ahead = behind = half = 0.0
        c, n = math.cos(eh), math.sin(eh)
        foot = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            a = inv @ np.array([sx, sy, -1.0, 1.0]); a = a[:3] / a[3]
            b = inv @ np.array([sx, sy, 1.0, 1.0]); b = b[:3] / b[3]
            d = b - a
            t = -a[2] / d[2] if d[2] < -1e-6 else 1e9
            p = a + d * min(t, 1.0) if t < 1e8 else a + d / np.linalg.norm(d) * self.fog[1]
            dist = math.hypot(p[0] - self.eye[0], p[1] - self.eye[1])
            if dist > self.fog[1]:   # beyond the haze nothing can be seen
                s = self.fog[1] / dist
                p = np.array([self.eye[0] + (p[0] - self.eye[0]) * s, self.eye[1] + (p[1] - self.eye[1]) * s, 0])
            foot.append((p[0], p[1]))
            dx, dy = p[0] - ex, p[1] - ey
            lon, lat = dx * c + dy * n, -dx * n + dy * c
            ahead, behind, half = max(ahead, lon), max(behind, -lon), max(half, abs(lat))
        self.world.view = (min(130.0, ahead), min(130.0, behind), min(130.0, half))
        self.footprint = foot
        self._pick_tiles(self._tiles_in(foot, 80.0))

    @staticmethod
    def _tiles_in(poly, margin):
        """The TILE-metre squares within margin of a convex ground polygon (given in order around it)."""
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        area = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]))
        sign = 1 if area > 0 else -1
        edges = []
        for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1]):
            L = math.hypot(bx - ax, by - ay) or 1e-9
            edges.append((ax, ay, -(by - ay) / L * sign, (bx - ax) / L * sign))   # inward normal
        reach = margin + TILE * 0.7072
        out = set()
        for i in range(math.floor((min(xs) - margin) / TILE), math.floor((max(xs) + margin) / TILE) + 1):
            for j in range(math.floor((min(ys) - margin) / TILE), math.floor((max(ys) + margin) / TILE) + 1):
                cx, cy = (i + 0.5) * TILE, (j + 0.5) * TILE
                if all((cx - ax) * nx + (cy - ay) * ny > -reach for ax, ay, nx, ny in edges):
                    out.add((i, j))
        return frozenset(out)

    def _pick_tiles(self, tiles):
        """Copy the trees and roof tanks of the squares in view into the instance buffers (only on a change)."""
        if tiles == self.in_view:
            return
        self.in_view = tiles
        for key, src, attr, buf in (("boxes", self.all_boxes, "n_static_boxes", "static_boxes"), ("spheres", self.all_spheres, "n_spheres", "static_spheres")):
            idx = self.tiles[key]
            parts = [src[a:a + n] for a, n in (idx[t] for t in tiles if t in idx)]
            arr = np.concatenate(parts) if parts else np.zeros((0, 11), dtype="f4")
            b = getattr(self, buf)
            if arr.nbytes > b.size:
                b.orphan(arr.nbytes * 2)
            if len(arr):
                b.write(arr.tobytes())
            setattr(self, attr, len(arr))

    def _runs(self, key, tiles):
        """(first vertex, vertices) runs to draw for the squares given, neighbours in a row merged."""
        idx = self.tiles[key]
        spans = sorted(idx[t] for t in tiles if t in idx)
        if "big" in idx:
            spans.insert(0, idx["big"])
        out = []
        for a, n in spans:
            if out and out[-1][0] + out[-1][1] == a:
                out[-1] = (out[-1][0], out[-1][1] + n)
            else:
                out.append((a, n))
        return out

    def _render(self, vao, key, tiles):
        for a, n in self._runs(key, tiles):
            vao.render(first=a, vertices=n)

    # dynamic instances --------------------------------------------------------------------------------------
    def _road_users(self, alpha):
        w, rows, t = self.world, [], self.world.t
        for u in (w.traffic.all() if w.traffic else ()):
            x, y, h = self._pose(u, alpha)
            if (int(x // TILE), int(y // TILE)) not in self.in_view:
                continue
            if not hasattr(u, "where") and self._in_tunnel(u):
                continue   # under the street in an underpass: out of sight from above
            rows += place(x, y, h, user_boxes(u, t))
        rows += place(*self._pose(w.ego, alpha), ego_boxes(w.ego))
        rows += signal_boxes(w, self.target[0], self.target[1])
        return np.array(rows, dtype="f4").reshape(-1, 11)

    def _in_tunnel(self, u):
        return in_tunnel(self.world.map, u, self.__dict__.setdefault("_tunnel", {}))

    def _effects(self, alpha):
        """Horn rings, headlight flashes and the rings of road users honking at us, as world-space quads."""
        w, t, rows = self.world, self.world.t, []

        def quad(cx, cy, h, l, wd, z, col, kind, ahead=0.0):
            c, s = math.cos(h), math.sin(h)
            corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)] if kind != 1 else [(0, -1), (1, -1), (1, 1), (0, 1)]
            pts = []
            for u, v in corners:
                lx, ly = (u * l if kind == 1 else u * l) + ahead, v * wd
                pts.append(((cx + c * lx - s * ly, cy + s * lx + c * ly, z), (u, v)))
            for i in (0, 1, 2, 0, 2, 3):
                (p, uv) = pts[i]
                rows.append((*p, *uv, *col, kind))

        e = w.ego
        ex, ey, eh = self._pose(e, alpha)
        fx, fy = ex + math.cos(eh) * e.spec.length / 2, ey + math.sin(eh) * e.spec.length / 2
        for t0, kind in list(getattr(w, "signals_given", [])):
            age = t - t0
            if kind == "horn":
                for k in range(3):
                    a = age - k * 0.18
                    if 0 <= a < 0.7:
                        r = 3 + a * 18
                        quad(fx, fy, eh, r, r, 1.2, (1.0, 0.95, 0.75, 0.55 * (1 - a / 0.7)), 0)
            elif kind == "dipper" and (age < 0.16 or 0.30 < age < 0.46):
                for side in (1, -1):
                    c, s = math.cos(eh), math.sin(eh)
                    ox, oy = fx - s * side * 0.6, fy + c * side * 0.6
                    quad(ox, oy, eh + side * 0.04, 26, 3.2, 0.9, (1.0, 0.97, 0.85, 0.55), 1)
                    quad(ox, oy, eh, 1.2, 1.2, 0.95, (1.0, 1.0, 0.9, 0.9), 2)
        if w.events:   # the road user an event is about: a pulsing amber ring, then a flash of how it went
            pulse = 0.5 + 0.5 * math.sin(t * 7.0)
            for ev in w.events.active:
                if getattr(ev, "highlight", False) and ev.actor is not None:
                    x, y, h = self._pose(ev.actor, alpha)
                    r = max(ev.actor.length, ev.actor.width) / 2 + 1.2 + 0.25 * pulse
                    quad(x, y, h, r, r, 0.3, (1.0, 0.72, 0.1, 0.32 + 0.18 * pulse), 0)
            for t_end, u, state in w.events.recent:
                age = t - t_end
                if 0 <= age < 1.5:
                    col = {"done": (0.3, 1.0, 0.45), "crash": (1.0, 0.2, 0.15)}.get(state, (0.75, 0.75, 0.8))
                    r = max(u.length, u.width) / 2 + 1.3 + age * 2.0
                    quad(u.x, u.y, u.heading, r, r, 0.3, (*col, 0.55 * (1 - age / 1.5)), 0)
        for u in (w.traffic.vehicles if w.traffic else ()):
            age = t - getattr(u, "honk_t", -9)
            if 0 <= age < 0.6:
                x, y, h = self._pose(u, alpha)
                r = 2 + age * 14
                quad(x + math.cos(h) * u.length / 2, y + math.sin(h) * u.length / 2, h, r, r, 1.2, (1.0, 0.6, 0.2, 0.5 * (1 - age / 0.6)), 0)
        return np.array(rows, dtype="f4").reshape(-1, 10)

    # one frame ----------------------------------------------------------------------------------------------
    def draw(self, alpha=1.0):
        now = time.perf_counter()
        dt, self.last_frame = min(0.1, now - self.last_frame), now
        ctx = self.ctx
        if self.big_map:
            self._draw_map(full=True)
            self.hud.draw(dt, big_map=True)
            return
        self._camera(alpha, dt)
        W, H = self.fb_size
        dyn = self._road_users(alpha)
        if len(dyn) * 44 > self.dyn.size:
            self.dyn.orphan(len(dyn) * 44 * 2)
        self.dyn.write(dyn.tobytes())
        sdir = SHADOW
        # 1. shadow mask: everything that stands up, flattened onto the ground along the sun
        self.mask.use(); self.mask.clear(0, 0, 0, 0)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.shadow["u_mvp"].write(_mat(self.mvp)); self.shadow["u_sdir"].value = sdir
        self._render(self.vao_solid_shadow, "solid", self.in_view)
        self.inst["u_mvp"].write(_mat(self.mvp)); self.inst["u_sdir"].value = sdir; self.inst["u_shadowpass"].value = 1
        self.vao_trees.render(instances=self.n_spheres)
        self.vao_sboxes.render(instances=self.n_static_boxes)
        self.vao_dyn.render(instances=len(dyn))
        # 2. the scene
        self.msaa.use(); self.msaa.clear(*FOG, 1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        self.mask_tex.use(0)
        for prog in (self.lit, self.inst):
            prog["u_mvp"].write(_mat(self.mvp)); prog["u_light"].value = tuple(SUN); prog["u_eye"].value = tuple(self.eye)
            prog["u_fog"].value = FOG; prog["u_fogr"].value = self.fog; prog["u_amb"].value = 0.0
        self.lit["u_shadow"].value = 0; self.lit["u_vp"].value = (W, H)
        self.lit["u_patches"].value = 1
        self.vao_base.render()
        self.lit["u_patches"].value = 0
        self._render(self.vao_ground, "ground", self.in_view)
        self._render(self.vao_solid, "solid", self.in_view)
        self.inst["u_shadowpass"].value = 0
        self.vao_sboxes.render(instances=self.n_static_boxes)
        self.vao_trees.render(instances=self.n_spheres)
        self.vao_dyn.render(instances=len(dyn))
        fx = self._effects(alpha)
        if len(fx):
            ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            self.msaa.depth_mask = False
            self.fxbuf.orphan(max(self.fxbuf.size, fx.nbytes)); self.fxbuf.write(fx.tobytes())
            self.fx["u_mvp"].write(_mat(self.mvp))
            self.vao_fx.render(vertices=len(fx))
            self.msaa.depth_mask = True
            ctx.disable(moderngl.BLEND)
        ctx.copy_framebuffer(self.out, self.msaa)
        # 3. minimap and dashboard
        self.out.use()
        self._draw_map(full=False)
        self.hud.draw(dt)

    def _map_view(self, full):
        """(centre x, centre y, half width, pixel rectangle) of the minimap or the full map."""
        W, H = self.fb_size
        s = self.scale
        if full:
            if self.map_center is None:
                self.map_center = (self.world.ego.x, self.world.ego.y)
            return self.map_center[0], self.map_center[1], self.map_half, (0, 0, W, H)
        side = int(230 * s)
        pad = int(16 * s)
        return self.world.ego.x, self.world.ego.y, MINIMAP[self.mini], (W - side - pad, pad + int(26 * s), side, side)

    def _draw_map(self, full):
        ctx = self.ctx
        cx, cy, half, (px, py, pw, ph) = self._map_view(full)
        self.out.use()
        self.out.viewport = (px, py, pw, ph)
        self.out.scissor = (px, py, pw, ph)
        hw, hh = half, half * ph / pw
        proj = _ortho(cx - hw, cx + hw, cy - hh, cy + hh)
        self.map_proj = (cx, cy, hw, hh, px, py, pw, ph)
        ctx.disable(moderngl.DEPTH_TEST)
        self.out.clear(0.91, 0.90, 0.87, 1.0, viewport=(px, py, pw, ph))
        self.blank.use(0)
        self.lit["u_mvp"].write(_mat(proj)); self.lit["u_amb"].value = 1.0; self.lit["u_light"].value = (0.0, 0.0, 1.0)
        self.lit["u_shadow"].value = 0; self.lit["u_vp"].value = (1.0, 1.0); self.lit["u_eye"].value = (cx, cy, 0.0)
        self.lit["u_fogr"].value = (1e9, 2e9)
        tiles = self._tiles_in([(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)], 20.0)
        self._render(self.vao_mapfill, "map_fill", tiles)
        mpp = 2 * hw / pw
        self.line["u_mvp"].write(_mat(proj)); self.line["u_mpp"].value = mpp
        self.line["u_case"].value = 1.0; self._render(self.vao_maplines, "map_lines", tiles)
        self.line["u_case"].value = 0.0; self._render(self.vao_maplines, "map_lines", tiles)
        self.out.viewport = (0, 0, *self.fb_size)
        self.out.scissor = None

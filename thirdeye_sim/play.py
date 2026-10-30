"""Drive our car around Gurgaon.

    cd ~/Desktop/thirdeye/sim && .venv-sumo/bin/python -m thirdeye_sim.play

Arrow keys drive (up accelerates; down brakes, half strength at first and full within 0.3 s, and held at a
standstill reverses), space is an emergency stop, H sounds the horn (hold to keep honking), F flashes the
headlights (the dipper), + and - zoom, T tilts the camera, C switches between a view that turns with the car
and north up, [ and ] zoom the minimap, M opens the full map (paused; wheel zooms, drag moves), P pauses,
K shows the keys, B hides or shows the event banner, , and . make traffic calmer or busier (chaos), R restarts,
Esc quits. Drives start in front of Medanta on
Sunset Boulevard (Sector 43); --start sunset starts at our morning route instead.

The simulation always steps 60 times a simulated second; the picture is drawn as often as the screen
refreshes, blending positions between steps, so it stays smooth on a 60 or 120 Hz display.

    --classic                                  the earlier flat pygame view
    --shot out.png --seconds 120 --every 20    no window: the autopilot drives our morning route and a frame
                                               is saved every 20 s (out_01.png, ...); add --map for the full map
"""
import argparse, json, math, os, time

import pygame

from .autopilot import Follower
from .world import World

DT = 1 / 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--every", type=float, default=20, help="with --shot, seconds between saved frames")
    ap.add_argument("--zoom", type=int)
    ap.add_argument("--tilt", type=int)
    ap.add_argument("--map", action="store_true", help="with --shot, also save the full map")
    ap.add_argument("--classic", action="store_true")
    ap.add_argument("--start", default="random", help="where to start: random (a main road; R picks another), medanta (Sunset Boulevard, Sector 43) or sunset")
    ap.add_argument("--chaos", type=float, default=1.0, help="1 = traffic and events as measured; more is busier (, and . change it)")
    ap.add_argument("--frames", type=int, help="quit after this many frames and print timings (a test)")
    ap.add_argument("--keytest", action="store_true", help="with --frames, press keys through the window and report (a test)")
    args = ap.parse_args()
    if args.shot:
        return shots(args)
    if args.classic:
        return classic(args)
    play(args)


def _save_session(world, drives):
    """Keep what happened in each drive (events with outcomes and whether they were on screen, the log,
    crashes) in sim/logs/, so a drive can be looked at afterwards."""
    if world.t > 5:
        drives.append({"start": world.start, "seconds": round(world.t, 1), "events": world.events.records if world.events else [],
                       "log": world.log, "crashes": world.crashes})
    if drives:
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
        os.makedirs(folder, exist_ok=True)
        json.dump(drives, open(os.path.join(folder, time.strftime("session_%Y%m%d_%H%M%S.json", time.localtime(SESSION))), "w"), indent=1)


SESSION = time.time()


def shots(args):
    """Autopilot drive without a window, saving frames from the GPU renderer."""
    from .render3d import Renderer3D
    pygame.init()
    world = World(start="sunset")   # the autopilot drives our morning route from its start
    view = Renderer3D(world, size=(1280, 800))
    if args.zoom is not None:
        view.zoom = args.zoom
    if args.tilt is not None:
        view.tilt = args.tilt
    pilot = Follower.our_route(world.map, lane="middle")
    n = 0
    while world.t < args.seconds:
        th, br, st, done = pilot.control(world.ego, world.status["limit_kmh"], world)
        view.snapshot()
        world.step(DT, throttle=th, brake=br, steer=st)
        if int(round(world.t * 60)) % 6 == 0:   # the camera eases after the car, so it needs frames between saves
            view.draw()
        if world.t >= (n + 1) * args.every - 1e-9:
            n += 1
            if n % 2 == 0:
                world.signal("horn")
                for _ in range(20):
                    view.snapshot(); world.step(DT, throttle=th, brake=br, steer=st)
            view.draw()
            _save(view, args.shot.replace(".png", f"_{n:02d}.png"))
        if done:
            break
    if args.map:
        view.big_map, view.map_half = True, 1200.0
        view.draw()
        _save(view, args.shot.replace(".png", "_map.png"))
        view.big_map, view.mini = False, 3
    print(f"{world.t:.0f} s, {world.status['road']}, log: {world.log[-8:]}")


def _save(view, path):
    W, H = view.fb_size
    img = pygame.image.frombuffer(view.out.read(components=3), (W, H), "RGB")
    pygame.image.save(pygame.transform.flip(img, False, True), path)


CHAOS_STEPS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


def play(args):
    """The game window (pyglet, so it is sharp on a Retina screen), drawn on the GPU."""
    import moderngl
    import pyglet
    from pyglet.window import key, mouse
    from .render3d import Renderer3D
    from .sound import Sounds
    config = pyglet.gl.Config(major_version=3, minor_version=3, forward_compatible=True, double_buffer=True, depth_size=24)
    window = pyglet.window.Window(1440, 900, "ThirdEye Gurgaon driving", resizable=True, config=config, vsync=True)
    ctx = moderngl.create_context()
    # pyglet may report the window in screen points or in pixels; work in points, draw in pixels
    dpi = max(1.0, float(getattr(window, "scale", 1.0)))
    unit = window.get_framebuffer_size()[0] / window.get_size()[0]       # pixels per pyglet unit
    scale = unit if unit > 1.01 else dpi                                   # pixels per point
    pts = lambda v: v * unit / scale                                       # pyglet units to points
    win = (round(pts(window.get_size()[0])), round(pts(window.get_size()[1])))
    world = World(start=args.start, chaos=args.chaos)
    view = Renderer3D(world, ctx=ctx, size=win, scale=scale)
    if args.zoom is not None:
        view.zoom = args.zoom
    down = set()   # keys held now (our own record: a pushed KeyStateHandler would be replaced by on_key_press below)
    sounds = Sounds()
    state = {"quit": False, "paused": False, "drag": None}
    drives = []

    def to_world(x, y):   # a window point (pyglet units, y up) on the full map
        cx, cy, hw, hh, *_ = view.map_proj
        return cx - hw + pts(x) / win[0] * 2 * hw, cy - hh + pts(y) / win[1] * 2 * hh

    @window.event
    def on_key_press(k, mod):
        down.add(k)
        if k == key.ESCAPE:
            if view.big_map:
                view.big_map = False
            else:
                state["quit"] = True
        elif k in (key.PLUS, key.EQUAL, key.NUM_ADD):
            view.zoom_by(-1)
        elif k in (key.MINUS, key.NUM_SUBTRACT):
            view.zoom_by(1)
        elif k == key.T:
            view.tilt = (view.tilt + 1) % 3
        elif k == key.C:
            if view.big_map:
                view.map_center = (world.ego.x, world.ego.y)
            else:
                view.rotate = not view.rotate
        elif k == key.BRACKETLEFT:
            view.mini = max(0, view.mini - 1)
        elif k == key.BRACKETRIGHT:
            view.mini = min(5, view.mini + 1)
        elif k == key.M:
            view.big_map = not view.big_map
            view.map_center = (world.ego.x, world.ego.y)
        elif k == key.K:
            view.hud.show_help = not view.hud.show_help
        elif k == key.B:
            view.hud.banner_on = not view.hud.banner_on
        elif k == key.P:
            state["paused"] = not state["paused"]
        elif k == key.R:
            _save_session(world, drives)
            world.reset(world.seed + 1 if world.start == "random" else None); view.prev = {}; view.cam_yaw = None
        elif k == key.F and not view.big_map:
            world.signal("dipper")
        elif k in (key.COMMA, key.PERIOD) and world.traffic:   # chaos down or up a notch
            steps = CHAOS_STEPS
            i = min(range(len(steps)), key=lambda j: abs(steps[j] - world.chaos))
            world.traffic.set_chaos(steps[max(0, min(len(steps) - 1, i + (1 if k == key.PERIOD else -1)))])
            world.note(f"traffic set to {world.chaos:g}x the measured events" if world.chaos != 1 else "traffic set to as measured")
        return pyglet.event.EVENT_HANDLED   # (pyglet would otherwise close the window on Esc)

    @window.event
    def on_key_release(k, mod):
        down.discard(k)

    @window.event
    def on_deactivate():   # switching to another app: let go of every key
        down.clear()

    @window.event
    def on_close():
        state["quit"] = True

    @window.event
    def on_resize(w, h):
        nonlocal win
        win = (round(pts(w)), round(pts(h)))
        view.resize(win)
        return pyglet.event.EVENT_HANDLED

    @window.event
    def on_mouse_motion(x, y, dx, dy):
        view.cursor = (pts(x), win[1] - pts(y))

    @window.event
    def on_mouse_scroll(x, y, sx, sy):
        if view.big_map and sy:
            f = 0.8 ** sy
            wx, wy = to_world(x, y)
            cx, cy = view.map_center
            view.map_half = max(80.0, min(8000.0, view.map_half * f))
            view.map_center = (wx + (cx - wx) * f, wy + (cy - wy) * f)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, mod):
        view.cursor = (pts(x), win[1] - pts(y))
        if view.big_map and buttons & mouse.LEFT:
            mpp = 2 * view.map_half / win[0]
            cx, cy = view.map_center
            view.map_center = (cx - pts(dx) * mpp, cy - pts(dy) * mpp)

    held = 0.0          # how long the brake key has been down: a tap brakes gently, holding it brakes hard
    next_horn = 0.0
    acc, last = 0.0, time.perf_counter()
    heard = -1.0        # time of the last road-user honk we played
    frames, frame_ms = 0, []
    while not state["quit"]:
        frames += 1
        if args.frames and frames > args.frames:
            frame_ms.sort()
            print(f"window {win}, framebuffer {window.get_framebuffer_size()}, scale {scale:.1f}; frame median "
                  f"{frame_ms[len(frame_ms) // 2]:.1f} ms, 95th {frame_ms[int(len(frame_ms) * 0.95)]:.1f} ms; simulated {world.t:.1f} s")
            break
        now = time.perf_counter()
        frame, last = min(0.1, now - last), now
        window.dispatch_events()
        if args.keytest:
            script = {30: ("on_key_press", key.UP), 200: ("on_key_press", key.LEFT), 260: ("on_key_release", key.LEFT),
                      300: ("on_key_release", key.UP), 301: ("on_key_press", key.H), 330: ("on_key_release", key.H),
                      331: ("on_key_press", key.DOWN), 420: ("on_key_release", key.DOWN)}
            if frames in script:
                window.dispatch_event(script[frames][0], script[frames][1], 0)
                print(f"frame {frames}: {script[frames][0]} {key.symbol_string(script[frames][1])} | speed {world.ego.speed * 3.6:.1f} km/h, "
                      f"heading {math.degrees(world.ego.heading):.1f}, log {world.log[-1:] if world.log else ''}")
        running = not state["paused"] and not view.big_map
        if running and (key.H in down) and world.t >= next_horn:
            world.signal("horn"); sounds.our_horn()
            next_horn = world.t + 0.45
        if running:
            acc += frame
            steer = (1.0 if (key.LEFT in down) else 0.0) - (1.0 if (key.RIGHT in down) else 0.0)
            steps = 0
            while acc >= DT and steps < 6:
                held = held + DT if (key.DOWN in down) else 0.0
                view.snapshot()
                world.step(DT, throttle=1.0 if (key.UP in down) else 0.0, brake=min(1.0, 0.5 + held / 0.6) if held else 0.0,
                           steer=steer, emergency=key.SPACE in down)
                acc -= DT; steps += 1
            if steps == 6:
                acc = 0.0
            for u in world.traffic.vehicles if world.traffic else ():
                if heard < u.honk_t <= world.t:
                    sounds.their_horn(math.hypot(u.x - world.ego.x, u.y - world.ego.y))
                    heard = u.honk_t
        view.draw(alpha=acc / DT if running else 1.0)
        window.flip()
        frame_ms.append((time.perf_counter() - now) * 1000)
    if not args.frames:   # (not for the quick self-tests)
        _save_session(world, drives)
    window.close()


def classic(args):
    from .render import Renderer
    pygame.init()
    world = World()
    screen = pygame.display.set_mode((1280, 800), pygame.RESIZABLE)
    pygame.display.set_caption("ThirdEye Gurgaon driving")
    view = Renderer(world)
    if args.zoom is not None:
        view.zoom = args.zoom
    clock = pygame.time.Clock()
    held = 0.0
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    view.zoom_by(1)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    view.zoom_by(-1)
                elif ev.key == pygame.K_r:
                    world.reset()
                elif ev.key == pygame.K_c:
                    view.rotate = not view.rotate
                elif ev.key == pygame.K_h:
                    world.signal("horn")
                elif ev.key == pygame.K_f:
                    world.signal("dipper")
        k = pygame.key.get_pressed()
        steer = (1.0 if k[pygame.K_LEFT] else 0.0) - (1.0 if k[pygame.K_RIGHT] else 0.0)
        held = held + DT if k[pygame.K_DOWN] else 0.0
        world.step(DT, throttle=1.0 if k[pygame.K_UP] else 0.0, brake=min(1.0, 0.5 + held / 0.6) if held else 0.0,
                   steer=steer, emergency=bool(k[pygame.K_SPACE]))
        view.draw(screen)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()

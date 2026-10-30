// ThirdEye Gurgaon in the browser. The simulator (Python, in a worker) steps the world 60 times a second;
// this page draws it with the desktop game's camera and shaders, and sends the keys held. Positions are
// blended between the last two simulated steps, so motion stays smooth at any refresh rate.
import * as THREE from "three";
import { City, FOG, type Meta, uniforms } from "./city";
import { Frame, Moving, pose, type Pose } from "./frame";
import { makeLonLat } from "./geo";
import { BigMap } from "./bigmap";
import { Hud } from "./hud";
import { MINIMAP, Minimap } from "./minimap";
import { PlaceLabels } from "./places";
import { CHAOS, Panel, newSeed, readSettings } from "./settings";
import { screenSize } from "./screen";
import { Sounds } from "./sound";
import { isTouch, touchControls } from "./touch";

const DT = 1 / 60;
const ZOOMS = [20, 28, 40, 56, 78, 110, 160, 240];   // camera distance from its target, metres
const TILTS = [64, 90, 46];                          // degrees above the horizon
// the drive's settings, mirrored in the address bar (settings.ts): ?start=random|medanta|sunset, ?seed=N (the
// same traffic and events), ?chaos=1.5, ?traffic=generic, ?events=off
const settings = readSettings();

const $ = (id: string) => document.getElementById(id)!;
const say = (text: string) => { $("progress").textContent = text; };

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.autoClear = false;
$("view").appendChild(renderer.domElement);
const camera = new THREE.PerspectiveCamera(42, 1, 1, 4000);
const mask = new THREE.WebGLRenderTarget(1, 1, { depthBuffer: false });
uniforms.u_shadow.value = mask.texture;
const fogColour = new THREE.Color(FOG.x, FOG.y, FOG.z);

function resize() {
  const [W, H] = screenSize();
  renderer.setSize(W, H);
  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  mask.setSize(size.x, size.y);
  uniforms.u_vp.value.copy(size);
  camera.aspect = W / H;
}
// no zooming the page on a phone (iPhone browsers ignore user-scalable=no): pinch and double tap do nothing;
// the full map handles its own pinch
document.addEventListener("gesturestart", (e) => e.preventDefault());
document.addEventListener("touchmove", (e) => { if (e.touches.length > 1) e.preventDefault(); }, { passive: false });
let lastTap = 0;
document.addEventListener("touchend", (e) => {
  const now = performance.now();
  if (now - lastTap < 350 && !(e.target as HTMLElement).closest("#panel")) e.preventDefault();
  lastTap = now;
}, { passive: false });
window.addEventListener("resize", resize);
window.visualViewport?.addEventListener("resize", resize);   // (a phone turned round, a browser bar shown or hidden)
resize();

// camera, as render3d.py's _camera: behind and above our car, turning with it, pulling back a little at speed
const cam = { zoom: 3, tilt: 0, rotate: true, yaw: null as number | null, speed: 0, fogFar: 300 };
let shake: [number, number] = [-9, 0];   // the last jolt (speed breaker or pothole): when, how hard
function placeCamera(me: Pose, speed: number, dt: number, t = 0) {
  const goal = cam.rotate ? me.h : Math.PI / 2;
  if (cam.yaw === null) cam.yaw = goal;
  cam.yaw += Math.atan2(Math.sin(goal - cam.yaw), Math.cos(goal - cam.yaw)) * (1 - Math.exp(-dt * 6));
  cam.speed += (Math.max(0, speed) - cam.speed) * (1 - Math.exp(-dt * 1.5));
  const d = ZOOMS[cam.zoom] * (1 + Math.min(0.4, cam.speed / 35)), pitch = (TILTS[cam.tilt] * Math.PI) / 180;
  const fwd = new THREE.Vector3(Math.cos(cam.yaw), Math.sin(cam.yaw), 0);
  const target = new THREE.Vector3(me.x, me.y, 0).addScaledVector(fwd, d * (TILTS[cam.tilt] >= 89 ? 0.28 : 0.18));
  const age = t - shake[0];
  if (age >= 0 && age < 0.5) {   // the view shakes briefly
    const k = shake[1] * (1 - age / 0.5) * 0.35;
    target.x += Math.sin(age * 71) * k; target.y += Math.cos(age * 53) * k;
  }
  const eye = target.clone().addScaledVector(fwd, -d * Math.cos(pitch)).add(new THREE.Vector3(0, 0, d * Math.sin(pitch)));
  camera.near = Math.max(1, ZOOMS[cam.zoom] * 0.05);   // (a near plane that moves flickers)
  camera.up.copy(TILTS[cam.tilt] >= 89 ? fwd : new THREE.Vector3(0, 0, 1));
  camera.position.copy(eye);
  camera.lookAt(target);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  uniforms.u_fogr.value.set(d * 1.6, d * 4.0);
  cam.fogFar = d * 4.0;
}

// how far the screen reaches around our car (render3d.py's _visible): the traffic appears only outside it
function visible(me: Pose): [number, number, number] {
  let ahead = 0, behind = 0, half = 0;
  const c = Math.cos(me.h), n = Math.sin(me.h), eye = camera.position;
  for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
    const a = new THREE.Vector3(sx, sy, -1).unproject(camera), b = new THREE.Vector3(sx, sy, 1).unproject(camera);
    const d = b.clone().sub(a);
    const t = d.z < -1e-6 ? -a.z / d.z : 1e9;
    const p = t < 1e8 ? a.clone().addScaledVector(d, Math.min(t, 1)) : a.clone().addScaledVector(d.normalize(), cam.fogFar);
    const dist = Math.hypot(p.x - eye.x, p.y - eye.y);
    if (dist > cam.fogFar) { const s = cam.fogFar / dist; p.set(eye.x + (p.x - eye.x) * s, eye.y + (p.y - eye.y) * s, 0); }
    const dx = p.x - me.x, dy = p.y - me.y, lon = dx * c + dy * n, lat = -dx * n + dy * c;
    ahead = Math.max(ahead, lon); behind = Math.max(behind, -lon); half = Math.max(half, Math.abs(lat));
  }
  return [ahead, behind, half];
}

// keys
const down = new Set<string>();
const sounds = new Sounds();
const once = { dipper: false, reset: false };
const state = { paused: false, dipperAt: -9, nextHorn: 0 };
let hud: Hud;
let mini: Minimap;
let big: BigMap | undefined;
let panel: Panel | undefined;
let worker: Worker | undefined;
let lastPose: Pose = { x: 0, y: 0, h: 0 };

// a change of settings, from the panel or a key: tell the simulator and update the panel and the address bar
function change(p: Partial<typeof settings>) {
  Object.assign(settings, p);
  worker?.postMessage({ type: "set", ...p });
  panel?.sync(settings);
}

function key(k: string) {
  if (k === "escape") {   // closes what is open, else back to the landing page
    if (big?.open) big.toggle(lastPose);
    else if (!$("help").hidden) $("help").hidden = true;
    else if (panel?.open) panel.toggle();
    else location.href = import.meta.env.BASE_URL;
  }
  else if (k === "m") big?.toggle(lastPose);
  else if (k === "+" || k === "=") cam.zoom = Math.max(0, cam.zoom - 1);
  else if (k === "-" || k === "_") cam.zoom = Math.min(ZOOMS.length - 1, cam.zoom + 1);
  else if (k === "t") cam.tilt = (cam.tilt + 1) % TILTS.length;
  else if (k === "c") cam.rotate = !cam.rotate;
  else if (k === "f") once.dipper = true;
  else if (k === "b") hud.bannerOn = !hud.bannerOn;
  else if (k === "p") state.paused = !state.paused;
  else if (k === "r") restartDrive();
  else if (k === "k") $("help").hidden = !$("help").hidden;
  else if (k === "," || k === ".") change({ chaos: Math.max(CHAOS.min, Math.min(CHAOS.max, settings.chaos + (k === "." ? CHAOS.step : -CHAOS.step))) });
  else if (k === "[") mini.zoom = Math.max(0, mini.zoom - 1);
  else if (k === "]") mini.zoom = Math.min(MINIMAP.length - 1, mini.zoom + 1);
}

// keys held (and phone buttons, which press the same keys)
function press(k: string, on: boolean) {
  sounds.wake();
  if (!on) { down.delete(k); return; }
  if (down.has(k)) return;
  down.add(k);
  key(k);
}
window.addEventListener("keydown", (e: KeyboardEvent) => {
  const k = e.key.toLowerCase();
  if (["arrowup", "arrowdown", "arrowleft", "arrowright", " "].includes(k)) e.preventDefault();
  if (e.repeat) return;
  press(k, true);
});
window.addEventListener("keyup", (e: KeyboardEvent) => press(e.key.toLowerCase(), false));
window.addEventListener("blur", () => down.clear());   // switching away: let go of every key
const held = (...keys: string[]) => keys.some((k) => down.has(k));

let restartDrive = () => {};

async function main() {
  // the simulator starts loading Python before anything else (sim.worker.ts)
  const sim = new Worker(new URL("./sim.worker.ts", import.meta.url), { type: "module" });
  worker = sim;
  const meta: Meta = await (await fetch(`${import.meta.env.BASE_URL}data/meta.json`)).json();
  hud = new Hud(makeLonLat(meta.net_offset, meta.proj));
  hud.onSpeeding = () => sounds.chime();
  // where a drive starts: a named place, or the seed's pick from the pool of random starts (the simulator picks
  // the same one), so the city around it can load while Python boots
  const startOf = (s: typeof settings) =>
    s.start === "random" ? meta.start_pool[s.seed % meta.start_pool.length] : meta.starts[s.start] ?? meta.starts.medanta;
  const named = settings.start === "random" ? null : startOf(settings);
  let begin = startOf(settings);
  big = new BigMap(meta, makeLonLat(meta.net_offset, meta.proj));
  if (isTouch()) touchControls(press);
  const city = new City(meta);
  mini = new Minimap(meta);
  const places = new PlaceLabels(meta);
  const moving = new Moving();
  city.scene.add(moving.mesh, moving.fx);
  city.mask.add(moving.shadow);

  let prev: Frame | null = null, cur: Frame | null = null, waiting = true, simText = "Loading Python", cityText = "";
  let ready: (pose: number[]) => void;
  const simReady = new Promise<number[]>((r) => (ready = r));
  sim.onmessage = (e) => {
    const m = e.data;
    if (m.type === "progress") { simText = m.text; say(`${cityText}   ${simText}`); }
    else if (m.type === "error") { say(`Something went wrong: ${m.text}`); console.error(m.text); }
    else if (m.type === "ready") ready(m.pose);
    else if (m.type === "frame") {
      prev = cur;
      cur = new Frame(m.buf, m.info);
      if (prev && cur.t < prev.t) prev = null;   // a restart
      for (const d of cur.info.heard) sounds.theirHorn(d);
      waiting = false;
    }
  };
  sim.postMessage({ type: "init", start: named ?? "random", seed: settings.seed, chaos: settings.chaos, style: settings.style, scripted: settings.scripted });
  const near = (x: number, y: number) => city.loadAround(x, y, 600, (done, total) => {
    cityText = `City ${Math.round((100 * done) / total)}%`;
    say(`${cityText}   ${simText}`);
  });
  // the city around the start loads while Python boots; then it shows, the camera circling, until the traffic
  // is ready
  await near(begin[0], begin[1]);
  $("loading").classList.add("gone");
  $("waiting").hidden = false;
  document.body.classList.add("booting");   // (the dashboard appears with the traffic)
  // the rest of Gurgaon, nearest first, in the background once the traffic is running (before, it would slow
  // Python's download on a phone)
  simReady.then(() => { $("waiting").hidden = true; document.body.classList.remove("booting"); city.loadAround(begin[0], begin[1], 1e9); });
  // a new drive: a new seed, and wherever was chosen (a named start is the same place, random is somewhere new)
  const restart = (start = settings.start) => {
    settings.start = start;
    settings.seed = newSeed();
    once.reset = true;
    begin = startOf(settings);
    city.loadAround(begin[0], begin[1], 1e9);   // its city first
    panel?.sync(settings);
  };
  panel = new Panel(settings, { change, restart, banners: (on) => { hud.bannerOn = on; } });
  restartDrive = () => restart();

  let clock = 0, last = performance.now();
  const frame = (now: number) => {
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (!cur) {   // the traffic is not ready yet: show the city, circling slowly above the start
      const a = now / 9000;
      placeCamera({ x: begin[0], y: begin[1], h: begin[2] + a }, 0, dt);
      renderer.setRenderTarget(mask); renderer.setClearColor(0x000000, 0); renderer.clear(); renderer.render(city.mask, camera);
      renderer.setRenderTarget(null); renderer.setClearColor(fogColour, 1); renderer.clear(); renderer.render(city.scene, camera);
      return;
    }
    const running = !state.paused && !big!.open;
    if (running) clock = Math.min(clock + dt, cur.t + DT);
    if (once.reset) {
      once.reset = false;
      const where = settings.start === "random" ? "random" : startOf(settings);
      sim.postMessage({ type: "reset", seed: settings.seed, start: where });
      waiting = true; prev = null; clock = 0; cam.yaw = null; hud.reset();
    }
    const alpha = prev && cur.t > prev.t ? Math.min(1, Math.max(0, (clock - prev.t) / (cur.t - prev.t))) : 1;
    const me = pose(prev, cur, -1, alpha)!;
    lastPose = me;
    if (big!.open) { big!.draw(renderer, city, me); return; }
    shake = cur.info.jolt;
    placeCamera(me, cur.info.speed_kmh / 3.6, dt, clock);
    const view = visible(me);

    if (!waiting && running) {
      const n = Math.min(6, Math.round((clock + dt - cur.t) / DT));
      if (n > 0) {
        let horn = false;
        if (held("h") && cur.t >= state.nextHorn) {
          horn = true; state.nextHorn = cur.t + 0.45; sounds.ourHorn();
        }
        if (once.dipper) state.dipperAt = cur.t;
        sim.postMessage({
          type: "tick", n, view, horn, dipper: once.dipper,
          throttle: held("arrowup", "w") ? 1 : 0, braking: held("arrowdown", "s"),
          steer: (held("arrowleft", "a") ? 1 : 0) - (held("arrowright", "d") ? 1 : 0), emergency: held(" "),
        });
        once.dipper = false;
        waiting = true;
      }
    }

    moving.update(prev, cur, alpha);
    hud.update(cur, me, { horn: held("h"), dipper: cur.t - state.dipperAt < 0.5, brake: held("arrowdown", "s", " ") });
    const [W, H] = screenSize();
    hud.placeTags(cur, (id) => pose(prev, cur!, id, alpha), me, camera, W, H);

    renderer.setRenderTarget(mask);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(city.mask, camera);
    renderer.setRenderTarget(null);
    renderer.setClearColor(fogColour, 1);
    renderer.clear();
    renderer.render(city.scene, camera);
    places.update(me, cam.zoom, Math.max(...view) + 40, camera, W, H);
    mini.draw(renderer, city, me);
  };
  requestAnimationFrame(frame);
}

main().catch((err) => { say(`Something went wrong: ${err}`); console.error(err); });

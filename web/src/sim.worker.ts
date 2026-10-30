// The simulator, in Python, in this worker: Pyodide runs thirdeye_sim/web.py (the same code as the desktop
// game and the Gymnasium environment). The page sends a tick with the controls and the number of 1/60 s
// steps due; the worker answers with the frame (see web.py for the layout).

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";

type Init = { type: "init"; start: number[] | string; seed: number; chaos: number; style: string; scripted: boolean };
type Tick = {
  type: "tick"; n: number; throttle: number; braking: boolean; steer: number; emergency: boolean;
  horn: boolean; dipper: boolean; view: [number, number, number];
};
type Reset = { type: "reset"; seed: number | null; start: number[] | string | null };
type Set = { type: "set"; chaos?: number; style?: string; scripted?: boolean };

let bridge: any = null;

function say(text: string) {
  postMessage({ type: "progress", text });
}

// Python, the simulator's code and the road map start loading as soon as the worker exists, before the page
// has said where to start; init waits for them
const loading = (async () => {
  const sim = fetch(`${import.meta.env.BASE_URL}data/sim.zip`).then((r) => r.arrayBuffer());
  const map = fetch(`${import.meta.env.BASE_URL}data/map.pklz`).then((r) => r.arrayBuffer());
  say("Loading Python");
  const { loadPyodide } = await import(/* @vite-ignore */ PYODIDE + "pyodide.mjs");
  const py = await loadPyodide({ indexURL: PYODIDE });
  say("Loading the simulator");
  const [, zip, m] = await Promise.all([py.loadPackage(["shapely"]), sim, map]);
  py.FS.mkdirTree("/sim");
  py.unpackArchive(zip, "zip", { extractDir: "/sim" });
  py.FS.writeFile("/sim/map.pklz", new Uint8Array(m));
  py.runPython("import sys; sys.dont_write_bytecode = True; sys.path.insert(0, '/sim')");
  return py;
})();

async function boot(msg: Init) {
  const py = await loading;
  say("Building Gurgaon's roads");
  const Bridge = py.pyimport("thirdeye_sim.web").Bridge;
  bridge = Bridge("/sim/map.pklz", typeof msg.start === "string" ? msg.start : py.toPy(msg.start), msg.seed);
  apply(msg);
  postMessage({ type: "ready", pose: [bridge.world.ego.x, bridge.world.ego.y] });
  send();
}

// the drive's settings (settings.ts): style first, since chaos tempers drivers from it
function apply(m: { chaos?: number; style?: string; scripted?: boolean }) {
  if (m.style !== undefined) bridge.set_style(m.style);
  if (m.chaos !== undefined) bridge.set_chaos(m.chaos);
  if (m.scripted !== undefined) bridge.set_scripted(m.scripted);
}

function send() {
  const out = bridge.frame();
  const raw = out.get(0).toJs();       // bytes: a Uint8Array copy
  const info = out.get(1);
  out.destroy();
  const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  postMessage({ type: "frame", buf, info }, [buf]);
}

onmessage = async (e: MessageEvent<Init | Tick | Reset | Set>) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      await boot(msg);
    } else if (msg.type === "tick" && bridge) {
      bridge.set_view(...msg.view);
      bridge.step(msg.n, msg.throttle, msg.braking, msg.steer, msg.emergency, msg.horn, msg.dipper);
      send();
    } else if (msg.type === "set" && bridge) {
      apply(msg);
    } else if (msg.type === "reset" && bridge) {
      bridge.reset(msg.seed ?? undefined, msg.start ?? undefined);   // (JS null reaches Python as JsNull, undefined as None)
      send();
    }
  } catch (err) {
    postMessage({ type: "error", text: String(err) });
  }
};

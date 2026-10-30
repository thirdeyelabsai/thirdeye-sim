// The landing and realism pages: fills their numbers and draws the events chart from what export.py writes
// (data/realism.json from python -m thirdeye_sim.realism, data/benchmark.json from thirdeye_sim.evaluate).
import "./site.css";

type EventRow = {
  event: string; name: string; rate: "measured" | "placeholder" | "assumed";
  measured_per_hour: number | null; low: number | null; high: number | null;
  sim_per_hour: number; traffic_alone_per_hour: number; match: boolean | null;
};
type Speeds = { median_kmh: number; mean_kmh: number; stopped_share: number; hours: number; clips?: number };
type Realism = {
  events: EventRow[];
  speeds: { real_play_area: Speeds; real_delhi_ncr: Speeds; sim_cars: Speeds; sim_our_car: Speeds };
  summary: { events_checked: number; events_matching: number; events_matching_traffic_alone: number;
             hours_driven: number; drives: number; measured_clips: number };
};
type Bench = { summary: Record<string, { episodes: number; arrived: number; ended: Record<string, number> }> };

const SVG = "http://www.w3.org/2000/svg";
const base = import.meta.env.BASE_URL;
const pct = (x: number) => `${Math.round(x * 100)}%`;
const int = (x: number) => `${Math.round(x)}`;

function fill(key: string, text: string) {
  document.querySelectorAll<HTMLElement>(`[data-fill="${key}"]`).forEach((el) => { el.textContent = text; });
}

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent?: Element) {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  parent?.appendChild(n);
  return n;
}

/** Each measured event: the range our footage gives (band), its middle (tick) and the simulator (dot). */
function drawEvents(box: HTMLElement, rows: EventRow[]) {
  const data = rows.filter((r) => r.rate !== "assumed" && r.low !== null && r.high !== null)
    .sort((a, b) => (b.measured_per_hour ?? 0) - (a.measured_per_hour ?? 0) || b.sim_per_hour - a.sim_per_hour);
  const draw = () => {
    box.replaceChildren();
    const W = Math.max(300, box.clientWidth);
    const narrow = W < 620;
    const rowH = narrow ? 46 : 34, top = 6, bottom = 34;
    const H = top + data.length * rowH + bottom;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "How often each event happens per hour: the range on real roads and the simulator" }, box);
    let left = 0;
    if (!narrow) {   // the widest event name sets the left margin
      const probe = el("text", { "font-size": 13 }, svg);
      for (const r of data) { probe.textContent = r.name; left = Math.max(left, probe.getComputedTextLength()); }
      probe.remove();
      left += 20;
    }
    const right = 14;
    const max = Math.max(...data.map((r) => Math.max(r.high ?? 0, r.sim_per_hour))) * 1.05;
    const step = max > 150 ? 50 : max > 60 ? 20 : 10;
    const top_ = Math.ceil(max / step) * step;
    const x = (v: number) => left + (v / top_) * (W - left - right);
    const y = (i: number) => top + i * rowH + (narrow ? 30 : rowH / 2);
    for (let v = 0; v <= top_; v += step) {
      el("line", { x1: x(v), x2: x(v), y1: top, y2: H - bottom, stroke: "var(--rule)", "stroke-width": 1 }, svg);
      const t = el("text", { x: x(v), y: H - bottom + 20, "text-anchor": "middle", "font-size": 12, fill: "var(--ink-subtle)" }, svg);
      t.textContent = String(v);
    }
    const unit = el("text", { x: W - right, y: H - 2, "text-anchor": "end", "font-size": 12, fill: "var(--ink-subtle)" }, svg);
    unit.textContent = "times per hour of driving";
    const tip = document.createElement("div");
    tip.className = "tip"; tip.hidden = true; box.appendChild(tip);
    data.forEach((r, i) => {
      const g = el("g", {}, svg);
      const label = el("text", { x: narrow ? x(0) : left - 20, y: narrow ? y(i) - 15 : y(i) + 4.5, "text-anchor": narrow ? "start" : "end",
        "font-size": 13, fill: "var(--ink)" }, g);
      label.textContent = r.name;
      el("rect", { x: x(r.low!), y: y(i) - 6, width: Math.max(3, x(r.high!) - x(r.low!)), height: 12,
        fill: "var(--gilt-wash)", stroke: "oklch(0.45 0.12 55 / 0.45)", "stroke-width": 1 }, g);
      el("line", { x1: x(r.measured_per_hour!), x2: x(r.measured_per_hour!), y1: y(i) - 8, y2: y(i) + 8, stroke: "var(--gilt)", "stroke-width": 2 }, g);
      el("circle", { cx: x(r.sim_per_hour), cy: y(i), r: 6, fill: r.match ? "var(--ink)" : "var(--caution)", stroke: "var(--background)", "stroke-width": 2 }, g);
      const hit = el("rect", { x: 0, y: y(i) - rowH / 2 - (narrow ? 8 : 0), width: W, height: rowH, fill: "transparent" }, g);
      const show = (ev: PointerEvent) => {
        tip.replaceChildren();
        const b = document.createElement("b"); b.textContent = r.name; tip.appendChild(b);
        for (const line of [`Real roads: ${int(r.measured_per_hour!)} an hour (between ${int(r.low!)} and ${int(r.high!)})`,
                            `Simulator: ${int(r.sim_per_hour)} an hour`]) {
          const d = document.createElement("div"); d.textContent = line; tip.appendChild(d);
        }
        tip.hidden = false;
        const rect = box.getBoundingClientRect();
        const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
        tip.style.left = `${Math.max(0, Math.min(box.clientWidth - tip.offsetWidth, px + 14))}px`;
        tip.style.top = `${Math.max(0, py - tip.offsetHeight - 10)}px`;
      };
      hit.addEventListener("pointermove", show);
      hit.addEventListener("pointerleave", () => { tip.hidden = true; });
    });
  };
  draw();
  let last = box.clientWidth;
  new ResizeObserver(() => { if (box.clientWidth !== last) { last = box.clientWidth; draw(); } }).observe(box);
}

async function load<T>(name: string): Promise<T | null> {
  try {
    const r = await fetch(`${base}data/${name}`);
    return r.ok ? ((await r.json()) as T) : null;
  } catch { return null; }
}

(async () => {
  const [real, bench] = await Promise.all([load<Realism>("realism.json"), load<Bench>("benchmark.json")]);
  if (real) {
    const s = real.summary, sp = real.speeds;
    fill("matching", int(s.events_matching));
    fill("checked", int(s.events_checked));
    fill("matching-alone", int(s.events_matching_traffic_alone));
    fill("clips", int(s.measured_clips));
    fill("drives", int(s.drives));
    fill("hours", `${Math.round(s.hours_driven)}`);
    fill("real-speed", int(sp.real_play_area.median_kmh));
    fill("sim-speed", int(sp.sim_cars.median_kmh));
    fill("real-stopped", pct(sp.real_play_area.stopped_share));
    fill("sim-stopped", pct(sp.sim_cars.stopped_share));
    fill("our-stopped", pct(sp.sim_our_car.stopped_share));
    document.querySelectorAll<HTMLElement>("[data-chart=events]").forEach((box) => drawEvents(box, real.events));
    const assumed = real.events.filter((r) => r.rate === "assumed").map((r) => r.name.toLowerCase());
    fill("assumed", assumed.join(", ").replace(/, ([^,]*)$/, " and $1"));
  }
  if (bench) {
    const g = bench.summary["idm / gurgaon"], o = bench.summary["idm / generic"];
    if (g && o) {
      fill("crash-gurgaon", int(g.ended.crash ?? 0));
      fill("crash-generic", int(o.ended.crash ?? 0));
      fill("routes", int(g.episodes));
    }
  }
})();

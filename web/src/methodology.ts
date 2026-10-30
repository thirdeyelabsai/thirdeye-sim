// The methodology page's figures, all drawn from data the repository writes: data/methodology.json
// (tools/methodology_data.py), data/realism.json (python -m thirdeye_sim.realism) and data/benchmark.json
// (thirdeye_sim.evaluate), plus the blurred sample clips in samples/.
import { drawEvents, type EventRow, type Realism } from "./site";

type Model = { model: string; right: number; clips: number; events_right: number; events: number; said_nothing: number };
type Sample = { id: string; label: string; name: string; night: boolean; ai_label: string | null; ai_name: string | null; ai_model: string };
type Method = {
  labelling: { clips: number; nothing: number; rule: string; counts: Record<string, number>; models: Model[] };
  samples: Sample[];
  window_demo: { seconds: number; events: { t: number; label: string; name: string; source: string }[];
                 windows: { window: number; events: string[]; main: string }[] };
  routes: { traffic: string; seed: number; ended: string; hit: string | null; at_fault: boolean; completion: number }[];
};
type Cell = { nuplan_cls: [number, number]; carla_ds: [number, number]; episodes: number; arrived: number;
              per_km: { collisions: number }; mean_speed_kmh: number };

const SVG = "http://www.w3.org/2000/svg";
const base = import.meta.env.BASE_URL;
const q = <T extends Element>(sel: string) => Array.from(document.querySelectorAll<T>(sel));
const svgEl = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent?: Element) => {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  parent?.appendChild(n);
  return n;
};
const html = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const load = async <T,>(name: string): Promise<T | null> => {
  try { const r = await fetch(`${base}data/${name}`); return r.ok ? ((await r.json()) as T) : null; } catch { return null; }
};
/** Redraw on width changes only. */
const follow = (box: HTMLElement, draw: () => void) => {
  draw();
  let last = box.clientWidth;
  new ResizeObserver(() => { if (box.clientWidth !== last) { last = box.clientWidth; draw(); } }).observe(box);
};
const pct = (x: number) => `${Math.round(x * 100)}%`;

/** A video that plays while it is in view (muted, looping), with its poster until then. */
function clip(src: string, poster: string, label: string) {
  const v = html("video");
  v.muted = true; v.loop = true; v.playsInline = true; v.preload = "none";
  v.poster = poster; v.src = src; v.setAttribute("aria-label", label);
  return v;
}
const watching = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const v = e.target as HTMLVideoElement;
    if (e.isIntersecting) v.play().catch(() => {}); else v.pause();
  }
}, { threshold: 0.35 });

function gallery(box: HTMLElement, m: Method, names: Record<string, string>) {
  for (const s of m.samples) {
    const ours = names[s.label] ?? s.name, theirs = s.ai_label ? (names[s.ai_label] ?? s.ai_name ?? s.ai_label) : "no answer";
    const card = html("figure", "card");
    const v = clip(`${base}samples/${s.id}.mp4`, `${base}samples/${s.id}.jpg`, `${ours}, ${s.night ? "at night" : "by day"}`);
    watching.observe(v);
    card.appendChild(v);
    const cap = html("figcaption");
    cap.appendChild(html("span", "t-label", s.night ? "Night" : "Day"));
    cap.appendChild(html("b", "", ours));
    const ai = html("span", `ai ${s.ai_label === s.label ? "yes" : "no"}`, `${s.ai_label === s.label ? "✓" : "✗"} ${s.ai_model}: ${theirs}`);
    cap.appendChild(ai);
    card.appendChild(cap);
    box.appendChild(card);
  }
}

function pairs(box: HTMLElement, names: Record<string, string>) {
  const list: [string, string][] = [["day_0844_011", "two_wheeler_close"], ["day_0844_171", "pedestrian_crossing"], ["day_0844_229", "wrong_way_in_lane"]];
  for (const [real, ev] of list) {
    const row = html("figure", "pair");
    row.appendChild(html("figcaption", "t-h4", names[ev] ?? ev));
    const two = html("div", "two");
    for (const [src, poster, tag] of [[`samples/${real}.mp4`, `samples/${real}.jpg`, "On the road, from our dashcam"],
                                      [`samples/dream_${ev}.mp4`, `samples/dream_${ev}.jpg`, "In Dream, from above"]]) {
      const cell = html("div", "cell");
      const v = clip(base + src, base + poster, `${names[ev] ?? ev}: ${tag}`);
      watching.observe(v);
      cell.appendChild(v);
      cell.appendChild(html("span", "t-label", tag));
      two.appendChild(cell);
    }
    row.appendChild(two);
    box.appendChild(row);
  }
}

function models(box: HTMLElement, m: Method) {
  const rows = m.labelling.models;
  const nothing = m.labelling.nothing / m.labelling.clips;
  follow(box, () => {
    box.replaceChildren();
    const W = Math.max(300, box.clientWidth), narrow = W < 620;
    const rowH = narrow ? 58 : 46, top = 26, bottom = 28;
    const H = top + rows.length * rowH + bottom;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Share of clips each AI model labelled as our labeller did" }, box);
    let left = 0;
    if (!narrow) {
      const probe = svgEl("text", { "font-size": 13 }, svg);
      for (const r of rows) { probe.textContent = r.model; left = Math.max(left, probe.getComputedTextLength()); }
      probe.remove(); left += 20;
    }
    const right = 44;
    const x = (v: number) => left + v * (W - left - right);
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      svgEl("line", { x1: x(v), x2: x(v), y1: top - 6, y2: H - bottom, stroke: "var(--rule)" }, svg);
      const t = svgEl("text", { x: x(v), y: H - bottom + 18, "text-anchor": "middle", "font-size": 12, fill: "var(--ink-subtle)" }, svg);
      t.textContent = pct(v);
    }
    svgEl("line", { x1: x(nothing), x2: x(nothing), y1: top - 10, y2: H - bottom, stroke: "var(--caution)", "stroke-width": 1.5, "stroke-dasharray": "4 4" }, svg);
    const bl = svgEl("text", { x: x(nothing) + 6, y: top - 12, "font-size": 12, fill: "var(--caution)" }, svg);
    bl.textContent = `Answering "nothing" every time: ${pct(nothing)}`;
    rows.forEach((r, i) => {
      const y0 = top + i * rowH + (narrow ? 22 : 8);
      const name = svgEl("text", { x: narrow ? x(0) : left - 20, y: narrow ? y0 - 6 : y0 + 13, "text-anchor": narrow ? "start" : "end", "font-size": 13, fill: "var(--ink)" }, svg);
      name.textContent = r.model;
      const a = r.right / r.clips, b = r.events_right / r.events;
      svgEl("rect", { x: x(0), y: y0, width: x(a) - x(0), height: 13, fill: "var(--ink)" }, svg);
      svgEl("rect", { x: x(0), y: y0 + 16, width: Math.max(1, x(b) - x(0)), height: 9, fill: "var(--gilt)" }, svg);
      const ta = svgEl("text", { x: x(a) + 6, y: y0 + 11, "font-size": 12, fill: "var(--ink)" }, svg); ta.textContent = pct(a);
      const tb = svgEl("text", { x: x(b) + 6, y: y0 + 24, "font-size": 11, fill: "var(--ink-subtle)" }, svg); tb.textContent = pct(b);
    });
  });
}

function modelsTable(tbody: HTMLElement, m: Method) {
  for (const r of m.labelling.models) {
    const tr = html("tr");
    for (const v of [r.model, `${r.right}`, `${r.events_right}`, `${r.said_nothing}`]) tr.appendChild(html("td", "", v));
    tbody.appendChild(tr);
  }
}

function ratesTable(tbody: HTMLElement, m: Method, real: Realism) {
  const rows = real.events.filter((r) => r.rate !== "assumed")
    .map((r) => ({ r, n: m.labelling.counts[r.event] ?? 0 })).sort((a, b) => b.n - a.n);
  for (const { r, n } of rows) {
    const tr = html("tr");
    for (const v of [r.name, `${n}`, `${Math.round(r.measured_per_hour ?? 0)}`, `${Math.round(r.low ?? 0)} to ${Math.round(r.high ?? 0)}`]) tr.appendChild(html("td", "", v));
    tbody.appendChild(tr);
  }
  const tr = html("tr", "muted");
  for (const v of ["Nothing notable", `${m.labelling.nothing}`, "", ""]) tr.appendChild(html("td", "", v));
  tbody.appendChild(tr);
}

function timeline(box: HTMLElement, m: Method, names: Record<string, string>) {
  const d = m.window_demo, nw = Math.round(d.seconds / 4);
  const byWin = new Map(d.windows.map((w) => [w.window, w]));
  let chosen = d.windows.reduce((a, w) => (w.events.length > a.events.length ? w : a), d.windows[0]).window;
  const panel = html("div", "win-panel");
  const draw = () => {
    box.replaceChildren();
    const W = Math.max(300, box.clientWidth), H = 92, pad = 4;
    const cw = (W - 2 * pad) / nw;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Two minutes of a simulated drive cut into 4-second windows" }, box);
    const x = (t: number) => pad + (t / d.seconds) * (W - 2 * pad);
    for (let k = 0; k < nw; k++) {
      const w = byWin.get(k);
      const r = svgEl("rect", { x: pad + k * cw + 1, y: 40, width: cw - 2, height: 26,
        fill: k === chosen ? "var(--ink)" : w ? "var(--gilt-wash)" : "var(--surface-raised)", stroke: w ? "oklch(0.45 0.12 55 / 0.45)" : "var(--rule)" }, svg);
      r.style.cursor = w ? "pointer" : "default";
      const pick = () => { if (w) { chosen = k; draw(); } };
      r.addEventListener("pointerenter", pick); r.addEventListener("click", pick);
    }
    for (const e of d.events) {
      const traffic = e.source === "traffic";
      svgEl("circle", { cx: x(e.t), cy: 24, r: 5, fill: traffic ? "var(--gilt)" : "var(--background)", stroke: "var(--gilt)", "stroke-width": 2 }, svg);
      svgEl("line", { x1: x(e.t), x2: x(e.t), y1: 30, y2: 40, stroke: "var(--gilt)", "stroke-width": 1 }, svg);
    }
    for (const s of [0, 30, 60, 90, 120]) {
      const t = svgEl("text", { x: Math.min(W - pad, Math.max(pad, x(s))), y: H - 6, "text-anchor": s === 0 ? "start" : s === 120 ? "end" : "middle", "font-size": 12, fill: "var(--ink-subtle)" }, svg);
      t.textContent = `${s} s`;
    }
    box.appendChild(panel);
    const w = byWin.get(chosen)!;
    panel.replaceChildren();
    panel.appendChild(html("p", "t-label", `Window ${chosen + 1}, ${chosen * 4} to ${chosen * 4 + 4} s`));
    const ul = html("ul");
    const seen = new Set<string>();
    for (const lab of w.events) {
      const li = html("li", lab === w.main && !seen.has(lab) ? "main" : "", names[lab] ?? lab);
      if (lab === w.main && !seen.has(lab)) li.appendChild(html("span", "badge", "counts"));
      seen.add(lab);
      ul.appendChild(li);
    }
    panel.appendChild(ul);
    if (w.events.length > 1) panel.appendChild(html("p", "t-small", "Several events started in this window, so the one highest in the labellers' list is the one that counts."));
  };
  follow(box, draw);
}

function realism(box: HTMLElement, note: HTMLElement, real: Realism) {
  const rows = real.events as EventRow[];
  const checked = rows.filter((r) => r.rate !== "assumed");
  const show = (field: "sim_per_hour" | "traffic_alone_per_hour") => {
    drawEvents(box, rows, field);
    const n = checked.filter((r) => r[field] >= r.low! && r[field] <= r.high!).length;
    note.textContent = field === "sim_per_hour"
      ? `${n} of ${checked.length} inside their range with the event generator on.`
      : `${n} of ${checked.length} inside their range from the traffic alone, with no help from the generator.`;
    q<HTMLButtonElement>("[data-m-field]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mField === field)));
  };
  q<HTMLButtonElement>("[data-m-field]").forEach((b) => b.addEventListener("click", () => show(b.dataset.mField as "sim_per_hour")));
  show("sim_per_hour");
}

function speeds(box: HTMLElement, real: Realism) {
  const a = real.speeds.real_play_area.hist, b = real.speeds.sim_cars.hist;
  follow(box, () => {
    box.replaceChildren();
    const W = Math.max(300, box.clientWidth), H = 240, top = 10, bottom = 30, left = 40, right = 10;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Share of time at each speed, real roads against Dream" }, box);
    const max = Math.ceil(Math.max(...a, ...b) * 20) / 20;
    const x = (i: number) => left + ((i * 5 + 2.5) / 85) * (W - left - right);
    const y = (v: number) => H - bottom - (v / max) * (H - top - bottom);
    for (let v = 0; v <= max + 1e-9; v += 0.05) {
      svgEl("line", { x1: left, x2: W - right, y1: y(v), y2: y(v), stroke: "var(--rule)" }, svg);
      const t = svgEl("text", { x: left - 8, y: y(v) + 4, "text-anchor": "end", "font-size": 12, fill: "var(--ink-subtle)" }, svg); t.textContent = pct(v);
    }
    for (const s of [0, 20, 40, 60, 80]) {
      const t = svgEl("text", { x: left + (s / 85) * (W - left - right), y: H - 8, "text-anchor": "middle", "font-size": 12, fill: "var(--ink-subtle)" }, svg);
      t.textContent = s === 80 ? "80+ km/h" : `${s}`;
    }
    for (const [h, color] of [[a, "var(--ink)"], [b, "var(--gilt)"]] as [number[], string][]) {
      svgEl("polyline", { points: h.map((v, i) => `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    }
  });
}

function units(box: HTMLElement, m: Method) {
  const tip = html("div", "tip"); tip.hidden = true;
  const ENDED: Record<string, [string, string]> = {
    arrive_dest: ["Finished", "finish"], crash: ["Crashed", "crash"], max_step: ["Ran out of time", "slow"],
    out_of_road: ["Left the road", "crash"], off_route: ["Lost the route", "slow"],
  };
  for (const traffic of ["generic", "gurgaon"]) {
    const routes = m.routes.filter((r) => r.traffic === traffic).sort((a, b) => a.seed - b.seed);
    const crashed = routes.filter((r) => r.ended === "crash").length;
    const col = html("div", "unit-col");
    col.appendChild(html("p", "t-h4", traffic === "generic" ? "Orderly traffic" : "Gurgaon traffic"));
    col.appendChild(html("p", "t-small", `${crashed} of ${routes.length} routes ended in a crash`));
    const grid = html("div", "unit-grid");
    for (const r of routes) {
      const [word, cls] = ENDED[r.ended] ?? [r.ended, "slow"];
      const u = html("button", `unit ${cls}`);
      u.type = "button";
      const what = r.ended === "crash" ? `${word} into ${r.hit === "animal" ? "an animal" : r.hit === "pedestrian" ? "a pedestrian" : "a vehicle"}${r.at_fault ? " (its fault, as nuPlan counts it)" : " (not its fault)"}` : word;
      u.setAttribute("aria-label", `Route ${r.seed + 1}: ${what}`);
      const show = () => {
        tip.textContent = `Route ${r.seed + 1}: ${what}, ${Math.round(r.completion * 100)}% of the route done`;
        tip.hidden = false;
        const br = box.getBoundingClientRect(), ur = u.getBoundingClientRect();
        tip.style.left = `${Math.max(0, Math.min(box.clientWidth - tip.offsetWidth, ur.left - br.left - tip.offsetWidth / 2 + ur.width / 2))}px`;
        tip.style.top = `${ur.top - br.top - tip.offsetHeight - 8}px`;
      };
      u.addEventListener("pointerenter", show); u.addEventListener("focus", show); u.addEventListener("click", show);
      u.addEventListener("pointerleave", () => { tip.hidden = true; });
      grid.appendChild(u);
    }
    col.appendChild(grid);
    box.appendChild(col);
  }
  const legend = html("ul", "legend unit-legend");
  for (const [cls, word] of [["finish", "Finished"], ["crash", "Crashed"], ["slow", "Ran out of time"]]) {
    const li = html("li"); li.appendChild(html("i", `unit ${cls}`)); li.appendChild(document.createTextNode(word)); legend.appendChild(li);
  }
  box.appendChild(legend);
  box.appendChild(tip);
}

function benchTable(tbody: HTMLElement, bench: { summary: Record<string, Cell> }) {
  for (const [key, name] of [["idm / generic", "Orderly traffic"], ["idm / gurgaon", "Gurgaon traffic"]]) {
    const c = bench.summary[key];
    if (!c) continue;
    const tr = html("tr");
    for (const v of [name, `${c.nuplan_cls[0].toFixed(2)} ± ${c.nuplan_cls[1].toFixed(2)}`, `${c.carla_ds[0].toFixed(1)} ± ${c.carla_ds[1].toFixed(1)}`,
                     `${c.arrived} of ${c.episodes}`, c.per_km.collisions.toFixed(2), `${Math.round(c.mean_speed_kmh)} km/h`]) tr.appendChild(html("td", "", v));
    tbody.appendChild(tr);
  }
}

(async () => {
  const [m, real, bench] = await Promise.all([load<Method>("methodology.json"), load<Realism>("realism.json"),
                                               load<{ summary: Record<string, Cell> }>("benchmark.json")]);
  const names: Record<string, string> = {};
  for (const r of real?.events ?? []) names[r.event] = r.name;
  names.nothing = "Nothing notable";
  if (m) {
    const best = m.labelling.models[0];
    q<HTMLElement>("[data-m=best-right]").forEach((e) => (e.textContent = `${best.right}`));
    q<HTMLElement>("[data-m=nothing]").forEach((e) => (e.textContent = `${m.labelling.nothing}`));
    q<HTMLElement>("[data-m=gallery]").forEach((b) => gallery(b, m, names));
    q<HTMLElement>("[data-m=pairs]").forEach((b) => pairs(b, names));
    q<HTMLElement>("[data-m=models]").forEach((b) => models(b, m));
    q<HTMLElement>("[data-m=models-table] tbody").forEach((b) => modelsTable(b, m));
    q<HTMLElement>("[data-m=timeline]").forEach((b) => timeline(b, m, names));
    q<HTMLElement>("[data-m=routes]").forEach((b) => units(b, m));
    if (real) q<HTMLElement>("[data-m=rates-table] tbody").forEach((b) => ratesTable(b, m, real));
  }
  if (real) {
    const note = document.querySelector<HTMLElement>("[data-m=toggle-note]");
    q<HTMLElement>("[data-m=events]").forEach((b) => note && realism(b, note, real));
    q<HTMLElement>("[data-m=speeds]").forEach((b) => speeds(b, real));
  }
  if (bench) q<HTMLElement>("[data-m=bench-table] tbody").forEach((b) => benchTable(b, bench));
})();

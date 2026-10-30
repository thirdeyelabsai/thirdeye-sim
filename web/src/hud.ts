// The dashboard, as HTML over the 3D view: speed and limit, road and position, horn, dipper and brake
// lamps, the countdown of the light ahead, the event banner and log, and a tag over each road user an event
// is about (at the screen edge, pointing at it, when it is off screen).
import * as THREE from "three";
import type { Frame, Pose } from "./frame";

const $ = (id: string) => document.getElementById(id)!;
const LIGHT = { red: "#e12d26", yellow: "#f0b414", green: "#28b450" };
const MAX_KMH = 120;

function arc(fromDeg: number, toDeg: number) {
  const r = 46, cx = 60, cy = 60;
  const p = (d: number) => [cx + r * Math.cos((d * Math.PI) / 180), cy + r * Math.sin((d * Math.PI) / 180)];
  const [x0, y0] = p(fromDeg), [x1, y1] = p(toDeg);
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 ${toDeg - fromDeg > 180 ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

const START = 135, SWEEP = 270;   // the dial runs from bottom-left round to bottom-right

export class Hud {
  bannerOn = true;
  onSpeeding?: () => void;
  private wasOver = false;
  private shown = "";
  private logCount = 0;
  private tags = new Map<number, HTMLDivElement>();

  constructor(private lonlat: (x: number, y: number) => [number, number]) {
    document.querySelector("#dial .track")!.setAttribute("d", arc(START, START + SWEEP));
    const ticks = document.querySelector("#dial .ticks")!;
    for (let v = 0; v <= MAX_KMH; v += 20) {
      const d = ((START + (SWEEP * v) / MAX_KMH) * Math.PI) / 180;
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", (60 + 33 * Math.cos(d)).toFixed(1));
      t.setAttribute("y", (63 + 33 * Math.sin(d)).toFixed(1));
      t.textContent = String(v);
      ticks.appendChild(t);
    }
  }

  reset() {
    this.logCount = 0;
    $("log").querySelector("ol")!.innerHTML = "";
  }

  update(cur: Frame, me: Pose, lamps: { horn: boolean; dipper: boolean; brake: boolean }) {
    const i = cur.info;
    const kmh = Math.max(0, i.speed_kmh);
    $("dial").querySelector(".num")!.textContent = String(Math.round(kmh));
    const fill = document.querySelector("#dial .fill")!;
    fill.setAttribute("d", kmh > 0.5 ? arc(START, START + (SWEEP * Math.min(kmh, MAX_KMH)) / MAX_KMH) : "");
    fill.setAttribute("style", `stroke: ${i.limit_kmh && kmh > i.limit_kmh + 5 ? "#d7322a" : "#46be6e"}`);
    $("road").textContent = i.road || " ";
    const t = Math.floor(i.t);
    $("clock").textContent = `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`;
    const [lon, lat] = this.lonlat(me.x, me.y);
    $("coords").textContent = `${lat.toFixed(5)} N   ${lon.toFixed(5)} E`;
    const lim = $("limit");
    lim.classList.toggle("none", !i.limit_kmh);
    lim.querySelector("span")!.textContent = i.limit_kmh ? String(i.limit_kmh) : "";
    $("lamp-horn").classList.toggle("on", lamps.horn);
    $("lamp-dipper").classList.toggle("on", lamps.dipper);
    $("lamp-brake").classList.toggle("on", lamps.brake);

    const cd = $("countdown");
    cd.hidden = !i.light;
    if (i.light) {
      (cd.querySelector("i") as HTMLElement).style.background = LIGHT[i.light.colour];
      cd.querySelector("b")!.textContent = `${Math.ceil(i.light.left)}`;
    }
    const ch = $("chaos");
    ch.hidden = i.chaos === 1;
    ch.textContent = `${i.chaos > 1 ? "Busier" : "Calmer"} traffic: ${i.chaos}x`;
    ch.title = `${i.chaos}x the measured events (Settings)`;
    const warn = $("warn");
    const w = i.wrong_way ? "Driving against the traffic" : i.on_road === 0 ? "Off the road" : i.on_road < 4 ? "Wheels off the road" : "";
    warn.hidden = !w;
    warn.textContent = w;
    // over the limit: the sign flashes, the speed turns red, a pill says so, and a chime sounds each time we go over
    lim.classList.toggle("over", i.over_limit);
    $("dial").classList.toggle("over", i.over_limit);
    const sp = $("speeding");
    sp.hidden = !i.over_limit;
    if (i.over_limit) sp.textContent = `Over the ${i.limit_kmh} km/h limit`;
    if (i.over_limit && !this.wasOver) this.onSpeeding?.();
    this.wasOver = i.over_limit;

    const b = $("banner");
    const bn = this.bannerOn ? i.banner : null;
    b.hidden = !bn;
    if (bn) {
      const key = bn.tag + "|" + bn.text;
      if (key !== this.shown) {
        this.shown = key;
        b.querySelector(".tag")!.textContent = bn.tag;
        b.querySelector(".text")!.textContent = bn.text;
        b.className = { coming: "", done: "done", crash: "crash" }[bn.state] ?? "off";
      }
    }

    if (i.log.length) {
      const ol = $("log").querySelector("ol")!;
      for (const [tt, what] of i.log) {
        const li = document.createElement("li");
        const s = Math.floor(tt);
        li.innerHTML = `<time>${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}</time><span></span>`;
        li.querySelector("span")!.textContent = what;
        ol.appendChild(li);
        this.logCount++;
      }
      while (ol.children.length > 6) ol.removeChild(ol.firstChild!);
    }
  }

  // tags over the road users events are about; off screen, at the edge towards them
  placeTags(cur: Frame, poseOf: (id: number) => Pose | null, me: Pose, camera: THREE.Camera, W: number, H: number) {
    const seen = new Set<number>();
    const placed: [number, number][] = [];   // where tags already sit: another moves up clear of them
    const free = (x: number, y: number) => {
      for (let k = 0; k < 4 && placed.some(([px, py]) => Math.abs(px - x) < 90 && Math.abs(py - y) < 20); k++) y -= 20;
      placed.push([x, y]);
      return y;
    };
    const v = new THREE.Vector3();
    const toScreen = (x: number, y: number, z: number) => {
      v.set(x, y, z).project(camera);
      return v.z < 1 ? [((v.x + 1) / 2) * W, ((1 - v.y) / 2) * H] : null;
    };
    for (const tag of this.bannerOn ? cur.info.tags : []) {
      const q = poseOf(tag.id);
      if (!q) continue;
      seen.add(tag.id);
      let el = this.tags.get(tag.id);
      if (!el) {
        el = document.createElement("div");
        $("tags").appendChild(el);
        this.tags.set(tag.id, el);
      }
      const p = toScreen(q.x, q.y, tag.tall ? 3.5 : 2.2);
      if (p && p[0] > 40 && p[0] < W - 40 && p[1] > 60 && p[1] < H - 40) {
        el.textContent = tag.text;
        el.style.left = `${p[0]}px`;
        el.style.top = `${free(p[0], p[1] - 6)}px`;
        continue;
      }
      const a = toScreen(me.x, me.y, 1), d = Math.hypot(q.x - me.x, q.y - me.y) || 1;
      const b = toScreen(me.x + ((q.x - me.x) / d) * 4, me.y + ((q.y - me.y) / d) * 4, 1);
      if (!a || !b) { el.style.left = "-999px"; continue; }
      const dx = b[0] - a[0], dy = b[1] - a[1], n = Math.hypot(dx, dy) || 1;
      const ux = dx / n, uy = dy / n;
      const k = Math.min(ux ? Math.abs((ux > 0 ? W - 70 - a[0] : a[0] - 70) / ux) : 1e9, uy ? Math.abs((uy > 0 ? H - 50 - a[1] : a[1] - 80) / uy) : 1e9);
      const arrow = Math.abs(ux) > Math.abs(uy) ? (ux > 0 ? "▶" : "◀") : uy > 0 ? "▼" : "▲";
      el.textContent = `${arrow} ${tag.text}`;
      el.style.left = `${a[0] + ux * k}px`;
      el.style.top = `${free(a[0] + ux * k, a[1] + uy * k)}px`;
    }
    for (const [id, el] of this.tags) if (!seen.has(id)) { el.remove(); this.tags.delete(id); }
  }
}

// Names of real places beside the road (OpenStreetMap), as the desktop game: landmarks whenever they are in
// view, offices, schools and the like when zoomed in a little, cafes and shops only up close; most important
// and nearest first, at most 14, none overlapping.
import * as THREE from "three";
import type { Meta } from "./city";
import type { Pose } from "./frame";

const MAX = 14;

export class PlaceLabels {
  private pool: HTMLDivElement[] = [];
  private widths = new Map<string, number>();
  private measure = document.createElement("canvas").getContext("2d")!;

  constructor(private meta: Meta) {
    const box = document.createElement("div");
    box.id = "places";
    document.body.appendChild(box);
    for (let i = 0; i < MAX; i++) {
      const el = document.createElement("div");
      el.innerHTML = "<i></i><span></span>";
      box.appendChild(el);
      this.pool.push(el);
    }
    this.measure.font = "500 12px Inter, system-ui, sans-serif";
  }

  private width(name: string) {
    let w = this.widths.get(name);
    if (w === undefined) { w = this.measure.measureText(name).width + 30; this.widths.set(name, w); }
    return w;
  }

  update(me: Pose, zoom: number, reach: number, camera: THREE.Camera, W: number, H: number) {
    const most = zoom <= 2 ? 3 : zoom <= 5 ? 2 : 1;
    const near = this.meta.places
      .filter((p) => p.rank > 0 && p.rank <= most && Math.abs(p.x - me.x) < reach && Math.abs(p.y - me.y) < reach)
      .sort((a, b) => a.rank - b.rank || Math.hypot(a.x - me.x, a.y - me.y) - Math.hypot(b.x - me.x, b.y - me.y));
    const placed: number[][] = [];
    const v = new THREE.Vector3();
    let n = 0;
    for (const p of near) {
      if (n >= MAX) break;
      v.set(p.x, p.y, 3).project(camera);
      if (v.z >= 1 || Math.abs(v.x) > 0.95 || Math.abs(v.y) > 0.95) continue;
      const sx = ((v.x + 1) / 2) * W, sy = ((1 - v.y) / 2) * H;
      const name = p.name.length < 34 ? p.name : p.name.slice(0, 32) + "...";
      const w = this.width(name), h = 26;
      const r = [sx - w / 2, sy - h, w, h];
      if (placed.some((q) => r[0] < q[0] + q[2] && q[0] < r[0] + w && r[1] < q[1] + q[3] && q[1] < r[1] + h)) continue;
      placed.push(r);
      const el = this.pool[n++];
      if (el.dataset.name !== name) {
        el.dataset.name = name;
        el.querySelector("span")!.textContent = name;
        (el.querySelector("i") as HTMLElement).style.background = `rgb(${p.colour.join(",")})`;
      }
      el.style.transform = `translate(${r[0].toFixed(1)}px, ${r[1].toFixed(1)}px)`;
      el.hidden = false;
    }
    for (let i = n; i < MAX; i++) this.pool[i].hidden = true;
  }
}

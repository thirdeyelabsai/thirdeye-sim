// The full map (M), as the desktop game's: north up, the whole play area at once if wanted, with a latitude and
// longitude grid, area names, landmarks and our car. Drag to move it, wheel or pinch to zoom; M or Esc closes.
// The drive pauses while it is open.
import * as THREE from "three";
import { type City, type Meta, mapUniforms } from "./city";
import type { Pose } from "./frame";
import { screenSize } from "./screen";

const LAND = new THREE.Color(0.91, 0.9, 0.87);

export class BigMap {
  open = false;
  private cx = 0;
  private cy = 0;
  private half = 1500;   // half the width shown, metres
  private cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 1000);
  private canvas = document.createElement("canvas");
  private ctx = this.canvas.getContext("2d")!;
  private pointers = new Map<number, [number, number]>();
  private fromLL: (lon: number, lat: number) => [number, number];

  constructor(private meta: Meta, private lonlat: (x: number, y: number) => [number, number]) {
    this.cam.up.set(0, 1, 0);
    this.canvas.id = "bigmap";
    this.canvas.hidden = true;
    document.body.appendChild(this.canvas);
    // longitude and latitude to map metres: over a few kilometres the projection is affine to well under a
    // metre, so fit it on three points
    const [x0, y0, x1, y1] = [0, 0, 5000, 5000];
    const a = lonlat(x0, y0), b = lonlat(x1, y0), c = lonlat(x0, y1);
    const m = [[(b[0] - a[0]) / (x1 - x0), (c[0] - a[0]) / (y1 - y0)], [(b[1] - a[1]) / (x1 - x0), (c[1] - a[1]) / (y1 - y0)]];
    const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    this.fromLL = (lon, lat) => {
      const dl = lon - a[0], dt = lat - a[1];
      return [x0 + (m[1][1] * dl - m[0][1] * dt) / det, y0 + (-m[1][0] * dl + m[0][0] * dt) / det];
    };
    const c2 = this.canvas;
    c2.addEventListener("wheel", (e) => { e.preventDefault(); this.zoomAt(e.clientX, e.clientY, 0.85 ** -Math.sign(e.deltaY)); }, { passive: false });
    c2.addEventListener("pointerdown", (e) => { c2.setPointerCapture(e.pointerId); this.pointers.set(e.pointerId, [e.clientX, e.clientY]); });
    c2.addEventListener("pointermove", (e) => {
      const was = this.pointers.get(e.pointerId);
      if (!was) return;
      if (this.pointers.size === 1) {
        const mpp = (2 * this.half) / screenSize()[0];
        this.cx -= (e.clientX - was[0]) * mpp; this.cy += (e.clientY - was[1]) * mpp;
      } else if (this.pointers.size === 2) {   // pinch
        const [p, q] = [...this.pointers.values()];
        const other = p === was ? q : p;
        const before = Math.hypot(was[0] - other[0], was[1] - other[1]), after = Math.hypot(e.clientX - other[0], e.clientY - other[1]);
        if (before > 10) this.zoomAt((e.clientX + other[0]) / 2, (e.clientY + other[1]) / 2, before / after);
      }
      this.pointers.set(e.pointerId, [e.clientX, e.clientY]);
    });
    const lift = (e: PointerEvent) => this.pointers.delete(e.pointerId);
    c2.addEventListener("pointerup", lift); c2.addEventListener("pointercancel", lift);
  }

  toggle(me: Pose) {
    this.open = !this.open;
    this.canvas.hidden = !this.open;
    document.body.classList.toggle("bigmap", this.open);
    if (this.open) { this.cx = me.x; this.cy = me.y; }
  }

  private zoomAt(sx: number, sy: number, f: number) {
    const [W, H] = screenSize(), mpp = (2 * this.half) / W;
    const wx = this.cx + (sx - W / 2) * mpp, wy = this.cy - (sy - H / 2) * mpp;
    this.half = Math.max(80, Math.min(4000, this.half * f));
    this.cx = wx + (this.cx - wx) * f; this.cy = wy + (this.cy - wy) * f;
  }

  draw(renderer: THREE.WebGLRenderer, city: City, me: Pose) {
    const [W, H] = screenSize(), dpr = renderer.getPixelRatio();
    const hw = this.half, hh = (hw * H) / W;
    Object.assign(this.cam, { left: -hw, right: hw, top: hh, bottom: -hh });
    this.cam.position.set(this.cx, this.cy, 500);
    this.cam.lookAt(this.cx, this.cy, 0);
    this.cam.updateProjectionMatrix();
    this.cam.updateMatrixWorld();
    mapUniforms.u_mpp.value = (2 * hw) / (W * dpr);
    renderer.setRenderTarget(null);
    renderer.setViewport(0, 0, W, H);
    renderer.setClearColor(LAND, 1);
    renderer.clear();
    renderer.render(city.map, this.cam);
    this.overlay(me, hw, hh, dpr);
  }

  private overlay(me: Pose, hw: number, hh: number, dpr: number) {
    const c = this.canvas, g = this.ctx, [W, H] = screenSize();
    if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const px = (x: number, y: number) => [((x - (this.cx - hw)) / (2 * hw)) * W, (1 - (y - (this.cy - hh)) / (2 * hh)) * H];
    // latitude and longitude lines, labelled at the left and bottom edges
    const [lon0, lat0] = this.lonlat(this.cx - hw, this.cy - hh), [lon1, lat1] = this.lonlat(this.cx + hw, this.cy + hh);
    const step = [0.001, 0.002, 0.005, 0.01, 0.02].find((s) => (lon1 - lon0) / s < 9) ?? 0.05;
    g.strokeStyle = "rgba(60,70,90,0.22)"; g.lineWidth = 1; g.fillStyle = "#4a5466"; g.font = "500 11px Inter, system-ui, sans-serif";
    for (let v = Math.ceil(lon0 / step) * step; v < lon1; v += step) {
      const [ax, ay] = px(...this.fromLL(v, lat0)), [bx, by] = px(...this.fromLL(v, lat1));
      g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke();
      g.textAlign = "center"; g.fillText(`${v.toFixed(3)} E`, ax, H - 26);
    }
    for (let v = Math.ceil(lat0 / step) * step; v < lat1; v += step) {
      const [ax, ay] = px(...this.fromLL(lon0, v)), [bx, by] = px(...this.fromLL(lon1, v));
      g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke();
      g.textAlign = "left"; g.fillText(`${v.toFixed(3)} N`, 8, ay - 4);
    }
    const label = (text: string, x: number, y: number, font: string, fill: string, halo = "#fff") => {
      g.font = font; g.lineWidth = 3; g.strokeStyle = halo; g.strokeText(text, x, y); g.fillStyle = fill; g.fillText(text, x, y);
    };
    g.textAlign = "center"; g.textBaseline = "middle";
    // area names, large and faint; road names; then landmarks, more of them closer in
    for (const p of this.meta.places) {
      if (p.rank !== 0) continue;
      const [sx, sy] = px(p.x, p.y);
      if (sx > 0 && sx < W && sy > 0 && sy < H) label(p.name.toUpperCase(), sx, sy, "700 13px Inter, system-ui, sans-serif", "rgba(110,104,92,0.75)", "rgba(255,255,255,0.6)");
    }
    const roads = hw <= 600 ? 3 : hw <= 1500 ? 2 : 1;
    for (const [name, x, y, ang, rank] of this.meta.labels) {
      if (rank > roads) continue;
      const [sx, sy] = px(x, y);
      if (sx < 0 || sx > W || sy < 0 || sy > H) continue;
      g.save(); g.translate(sx, sy); g.rotate((-ang * Math.PI) / 180);
      label(name, 0, 0, "600 12px Inter, system-ui, sans-serif", "#3c3a34"); g.restore();
    }
    const most = hw <= 400 ? 3 : hw <= 1200 ? 2 : 1;
    g.textAlign = "left";
    for (const p of this.meta.places) {
      if (p.rank < 1 || p.rank > most) continue;
      const [sx, sy] = px(p.x, p.y);
      if (sx < 4 || sx > W - 4 || sy < 4 || sy > H - 4) continue;
      g.fillStyle = `rgb(${p.colour.join(",")})`; g.beginPath(); g.arc(sx, sy, 4, 0, 2 * Math.PI); g.fill();
      label(p.name, sx + 7, sy, "500 12px Inter, system-ui, sans-serif", "#2c2b27");
    }
    // the edge of the play area (the wall in the game)
    g.beginPath();
    this.meta.play_area.forEach(([x, y], i) => { const [sx, sy] = px(x, y); if (i) g.lineTo(sx, sy); else g.moveTo(sx, sy); });
    g.closePath(); g.lineWidth = 2.5; g.strokeStyle = "rgba(200,40,30,0.7)"; g.setLineDash([10, 6]); g.stroke(); g.setLineDash([]);
    // our car
    const [mx, my] = px(me.x, me.y), s = 11;
    const pt = (a: number, r: number) => [mx + Math.cos(a) * r, my - Math.sin(a) * r];
    const arrow = [pt(me.h, s * 1.3), pt(me.h + 2.5, s), pt(me.h + Math.PI, s * 0.35), pt(me.h - 2.5, s)];
    g.beginPath(); arrow.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath();
    g.fillStyle = "#e62823"; g.fill(); g.lineWidth = 2; g.strokeStyle = "#fff"; g.stroke();
    // the cursor's position, and how to use the map
    g.textAlign = "center"; g.textBaseline = "alphabetic";
    g.fillStyle = "rgba(16,18,22,0.85)"; g.beginPath(); g.roundRect(W / 2 - 230, 14, 460, 30, 15); g.fill();
    g.fillStyle = "#eee"; g.font = "500 12.5px Inter, system-ui, sans-serif";
    g.fillText("Drag to move, wheel or pinch to zoom.  M or Esc to go back to the drive.", W / 2, 34);
  }
}

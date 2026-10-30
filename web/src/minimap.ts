// The minimap in the corner (as the desktop game): north up, centred on our car, drawn from the same map
// layers, with road names, landmarks, our car as a red arrow, north and a scale bar on top. [ and ] zoom it.
import * as THREE from "three";
import { type City, type Meta, mapUniforms } from "./city";
import type { Pose } from "./frame";
import { screenSize } from "./screen";

export const MINIMAP = [120, 250, 500, 1000, 2000, 4000];   // half-width, metres
// where it sits (CSS pixels): bottom right, or smaller at the top right on a phone (the pedals are below)
function rect() {
  const [W, H] = screenSize();
  const small = document.body.classList.contains("touch") || W < 760;
  const side = small ? (H < 420 ? 108 : 140) : 230;   // (a short phone screen: room for the horn below it)
  const top = W < 500 ? 140 : 16;   // (a phone held upright: below the speed panel)
  return { side, right: 16, bottom: small ? H - top - side : 52 };
}
const LAND = new THREE.Color(0.91, 0.9, 0.87);
const INK = "#14161a";

function nice(x: number) {
  const p = 10 ** Math.floor(Math.log10(x));
  return Math.max(...[1, 2, 5].map((m) => m * p).filter((v) => v <= x));
}

export class Minimap {
  zoom = 1;
  private cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 1000);
  private canvas = document.createElement("canvas");
  private ctx = this.canvas.getContext("2d")!;
  private hint = document.createElement("div");

  constructor(private meta: Meta) {
    this.cam.up.set(0, 1, 0);
    this.canvas.id = "minimap";
    Object.assign(this.canvas.style, { position: "fixed", pointerEvents: "none" });
    this.hint.id = "minihint";
    this.hint.textContent = "[ ] zoom";
    document.body.append(this.canvas, this.hint);
  }

  draw(renderer: THREE.WebGLRenderer, city: City, me: Pose) {
    const dpr = renderer.getPixelRatio();
    const hw = MINIMAP[this.zoom];
    const { side: SIDE, right, bottom } = rect();
    Object.assign(this.canvas.style, { right: `${right}px`, bottom: `${bottom}px`, width: `${SIDE}px`, height: `${SIDE}px` });
    this.cam.left = -hw; this.cam.right = hw; this.cam.top = hw; this.cam.bottom = -hw;
    this.cam.position.set(me.x, me.y, 500);
    this.cam.lookAt(me.x, me.y, 0);
    this.cam.updateProjectionMatrix();
    this.cam.updateMatrixWorld();
    mapUniforms.u_mpp.value = (2 * hw) / (SIDE * dpr);
    // draw the map exactly where its frame is on screen, measured, whatever the page's offsets
    const box = this.canvas.getBoundingClientRect(), view = renderer.domElement.getBoundingClientRect();
    const x = box.left - view.left, y = view.bottom - box.bottom;
    renderer.setScissorTest(true);
    renderer.setViewport(x, y, SIDE, SIDE);
    renderer.setScissor(x, y, SIDE, SIDE);
    renderer.setClearColor(LAND, 1);
    renderer.clear();
    renderer.render(city.map, this.cam);
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, ...screenSize());
    this.overlay(me, hw, dpr, SIDE);
  }

  private overlay(me: Pose, hw: number, dpr: number, SIDE: number) {
    const c = this.canvas, g = this.ctx, S = SIDE * dpr;
    if (c.width !== S) { c.width = S; c.height = S; }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, SIDE, SIDE);
    const toPx = (x: number, y: number) => [((x - (me.x - hw)) / (2 * hw)) * SIDE, (1 - (y - (me.y - hw)) / (2 * hw)) * SIDE];
    // road names, more of them closer in
    const most = hw <= 500 ? 3 : hw <= 1000 ? 2 : 1;
    g.font = "600 10px Inter, system-ui, sans-serif";
    g.textAlign = "center"; g.textBaseline = "middle";
    for (const [name, x, y, ang, rank] of this.meta.labels) {
      if (rank > most) continue;
      const [sx, sy] = toPx(x, y);
      if (sx < 0 || sx > SIDE || sy < 0 || sy > SIDE) continue;
      g.save(); g.translate(sx, sy); g.rotate((-ang * Math.PI) / 180);
      g.lineWidth = 3; g.strokeStyle = "#fff"; g.strokeText(name, 0, 0);
      g.fillStyle = "#3c3a34"; g.fillText(name, 0, 0);
      g.restore();
    }
    // landmarks up close
    if (hw <= 500) {
      g.textAlign = "left";
      for (const p of this.meta.places) {
        if (p.rank !== 1) continue;
        const [sx, sy] = toPx(p.x, p.y);
        if (sx < 4 || sx > SIDE - 4 || sy < 4 || sy > SIDE - 4) continue;
        g.fillStyle = `rgb(${p.colour.join(",")})`;
        g.beginPath(); g.arc(sx, sy, 3.5, 0, 2 * Math.PI); g.fill();
        g.lineWidth = 3; g.strokeStyle = "#fff"; g.strokeText(p.name, sx + 6, sy);
        g.fillStyle = "#3c3a34"; g.fillText(p.name, sx + 6, sy);
      }
    }
    // our car: a red arrow with a white edge, pointing where we head
    const a = me.h, s = 9, cx = SIDE / 2, cy = SIDE / 2;
    const pt = (ang: number, r: number) => [cx + Math.cos(ang) * r, cy - Math.sin(ang) * r];
    const arrow = [pt(a, s * 1.3), pt(a + 2.5, s), pt(a + Math.PI, s * 0.35), pt(a - 2.5, s)];
    g.beginPath(); arrow.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath();
    g.fillStyle = "#e62823"; g.fill(); g.lineWidth = 1.5; g.strokeStyle = "#fff"; g.stroke();
    // north, scale, frame
    g.fillStyle = "rgba(20,22,26,0.86)"; g.beginPath(); g.arc(SIDE - 18, 20, 12, 0, 2 * Math.PI); g.fill();
    g.fillStyle = "#fff"; g.textAlign = "center"; g.font = "700 11px Inter, system-ui, sans-serif"; g.fillText("N", SIDE - 18, 20.5);
    const bar = nice(hw * 0.4), L = (bar / (2 * hw)) * SIDE;
    g.fillStyle = INK; g.fillRect(10, SIDE - 14, L, 4);
    g.textAlign = "left"; g.font = "600 10px Inter, system-ui, sans-serif";
    g.fillText(bar < 1000 ? `${bar} m` : `${bar / 1000} km`, 14 + L, SIDE - 13);
    g.lineWidth = 3; g.strokeStyle = INK; g.strokeRect(1.5, 1.5, SIDE - 3, SIDE - 3);
  }
}

// One frame from the worker (layout in thirdeye_sim/web.py), and what the page draws from two of them:
// road users placed between their last two simulated poses, so motion stays smooth at any refresh rate.
import * as THREE from "three";
import { CUBE, MAT, instanced } from "./city";

export type Info = {
  t: number; jolt: [number, number]; chaos: number; start: number[] | null; speed_kmh: number; limit_kmh: number | null; road: string; on_road: number; wrong_way: boolean; over_limit: boolean; accel: number;
  light: { colour: "red" | "yellow" | "green"; left: number } | null;
  banner: { tag: string; text: string; state: string } | null;
  tags: { id: number; text: string; tall: boolean; r: number }[];
  rings: { x: number; y: number; r: number; age: number; state: string }[];
  signals_given: { kind: string; age: number }[];
  honking: { id: number; age: number; ahead: number }[];
  heard: number[];
  log: [number, string][];
};

export type Pose = { x: number; y: number; h: number };

export class Frame {
  t: number;
  poses = new Map<number, Pose & { first: number; n: number }>();
  boxes: Float32Array;
  signals: Float32Array;
  info: Info;

  constructor(buf: ArrayBuffer, info: string) {
    const f = new Float32Array(buf);
    this.t = f[0];
    const users = f[1], nBoxes = f[2], nSig = f[3];
    let o = 4;
    for (let i = 0; i < users; i++, o += 6) this.poses.set(f[o], { x: f[o + 1], y: f[o + 2], h: f[o + 3], first: f[o + 4], n: f[o + 5] });
    this.boxes = f.subarray(o, o + nBoxes * 10);
    o += nBoxes * 10;
    this.signals = f.subarray(o, o + nSig * 11);
    this.info = JSON.parse(info);
  }
}

function lerpAngle(a: number, b: number, k: number) {
  return a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * k;
}

// where a road user is drawn: between its poses in the two frames (as render3d.py's _pose)
export function pose(prev: Frame | null, cur: Frame, id: number, alpha: number): Pose | null {
  const b = cur.poses.get(id);
  if (!b) return null;
  const a = prev?.poses.get(id);
  if (!a || alpha >= 1 || Math.abs(b.x - a.x) > 8 || Math.abs(b.y - a.y) > 8) return { x: b.x, y: b.y, h: b.h };
  return { x: a.x + (b.x - a.x) * alpha, y: a.y + (b.y - a.y) * alpha, h: lerpAngle(a.h, b.h, alpha) };
}

const MAX_BOXES = 4096;
const MAX_FX = 256;

export class Moving {
  geo = instanced(CUBE, MAX_BOXES);
  mesh = new THREE.Mesh(this.geo, MAT.inst);
  shadow = new THREE.Mesh(this.geo, MAT.instShadow);
  fxGeo = new THREE.BufferGeometry();
  fx = new THREE.Mesh(this.fxGeo, MAT.fx);
  private fxPos = new Float32Array(MAX_FX * 6 * 3);
  private fxUv = new Float32Array(MAX_FX * 6 * 2);
  private fxCol = new Float32Array(MAX_FX * 6 * 4);
  private fxKind = new Float32Array(MAX_FX * 6);
  private nFx = 0;

  constructor() {
    for (const m of [this.mesh, this.shadow, this.fx]) m.frustumCulled = false;
    this.fx.renderOrder = 10;
    this.fxGeo.setAttribute("position", new THREE.BufferAttribute(this.fxPos, 3));
    this.fxGeo.setAttribute("fx_uv", new THREE.BufferAttribute(this.fxUv, 2));
    this.fxGeo.setAttribute("fx_col", new THREE.BufferAttribute(this.fxCol, 4));
    this.fxGeo.setAttribute("fx_kind", new THREE.BufferAttribute(this.fxKind, 1));
  }

  // road users, our car and the signals near it, as boxes
  update(prev: Frame | null, cur: Frame, alpha: number) {
    const P = this.geo.getAttribute("i_pos") as THREE.InstancedBufferAttribute, H = this.geo.getAttribute("i_head") as THREE.InstancedBufferAttribute;
    const S = this.geo.getAttribute("i_size") as THREE.InstancedBufferAttribute, C = this.geo.getAttribute("i_col") as THREE.InstancedBufferAttribute;
    const E = this.geo.getAttribute("i_emit") as THREE.InstancedBufferAttribute;
    const p = P.array as Float32Array, h = H.array as Float32Array, s = S.array as Float32Array, c = C.array as Float32Array, e = E.array as Float32Array;
    let k = 0;
    const b = cur.boxes;
    for (const [id, u] of cur.poses) {
      const q = pose(prev, cur, id, alpha)!;
      const cs = Math.cos(q.h), sn = Math.sin(q.h);
      for (let j = u.first; j < u.first + u.n && k < MAX_BOXES; j++, k++) {
        const r = j * 10, ax = b[r], ay = b[r + 1];
        p[3 * k] = q.x + cs * ax - sn * ay; p[3 * k + 1] = q.y + sn * ax + cs * ay; p[3 * k + 2] = b[r + 2];
        h[k] = q.h;
        s[3 * k] = b[r + 3]; s[3 * k + 1] = b[r + 4]; s[3 * k + 2] = b[r + 5];
        c[3 * k] = b[r + 6]; c[3 * k + 1] = b[r + 7]; c[3 * k + 2] = b[r + 8];
        e[k] = b[r + 9];
      }
    }
    const sg = cur.signals;
    for (let r = 0; r + 11 <= sg.length && k < MAX_BOXES; r += 11, k++) {
      p[3 * k] = sg[r]; p[3 * k + 1] = sg[r + 1]; p[3 * k + 2] = sg[r + 2]; h[k] = sg[r + 3];
      s[3 * k] = sg[r + 4]; s[3 * k + 1] = sg[r + 5]; s[3 * k + 2] = sg[r + 6];
      c[3 * k] = sg[r + 7]; c[3 * k + 1] = sg[r + 8]; c[3 * k + 2] = sg[r + 9]; e[k] = sg[r + 10];
    }
    this.geo.instanceCount = k;
    for (const a of [P, H, S, C, E]) { a.clearUpdateRanges(); a.addUpdateRange(0, k * a.itemSize); a.needsUpdate = true; }
    this.effects(prev, cur, alpha);
  }

  private quad(cx: number, cy: number, hd: number, l: number, wd: number, z: number, col: number[], kind: number, ahead = 0) {
    if (this.nFx >= MAX_FX) return;
    const c = Math.cos(hd), s = Math.sin(hd);
    const corners = kind !== 1 ? [[-1, -1], [1, -1], [1, 1], [-1, 1]] : [[0, -1], [1, -1], [1, 1], [0, 1]];
    const pts = corners.map(([u, v]) => {
      const lx = u * l + ahead, ly = v * wd;
      return [cx + c * lx - s * ly, cy + s * lx + c * ly, z, u, v];
    });
    let o = this.nFx * 6;
    for (const i of [0, 1, 2, 0, 2, 3]) {
      const [x, y, zz, u, v] = pts[i];
      this.fxPos.set([x, y, zz], 3 * o); this.fxUv.set([u, v], 2 * o); this.fxCol.set(col, 4 * o); this.fxKind[o] = kind;
      o++;
    }
    this.nFx++;
  }

  // horn rings, headlight flashes, rings round event road users, honk rings (as render3d.py's _effects)
  private effects(prev: Frame | null, cur: Frame, alpha: number) {
    this.nFx = 0;
    const info = cur.info, t = cur.t;
    const me = pose(prev, cur, -1, alpha)!;
    const fx = me.x + Math.cos(me.h) * 2.15, fy = me.y + Math.sin(me.h) * 2.15;
    for (const g of info.signals_given) {
      if (g.kind === "horn") {
        for (let k = 0; k < 3; k++) {
          const a = g.age - k * 0.18;
          if (a >= 0 && a < 0.7) { const r = 3 + a * 18; this.quad(fx, fy, me.h, r, r, 1.2, [1.0, 0.95, 0.75, 0.55 * (1 - a / 0.7)], 0); }
        }
      } else if (g.kind === "dipper" && (g.age < 0.16 || (g.age > 0.3 && g.age < 0.46))) {
        for (const side of [1, -1]) {
          const ox = fx - Math.sin(me.h) * side * 0.6, oy = fy + Math.cos(me.h) * side * 0.6;
          this.quad(ox, oy, me.h + side * 0.04, 26, 3.2, 0.9, [1.0, 0.97, 0.85, 0.55], 1);
          this.quad(ox, oy, me.h, 1.2, 1.2, 0.95, [1.0, 1.0, 0.9, 0.9], 2);
        }
      }
    }
    const pulse = 0.5 + 0.5 * Math.sin(t * 7.0);
    for (const tag of info.tags) {
      const q = pose(prev, cur, tag.id, alpha);
      if (!q) continue;
      const r = tag.r + 1.2 + 0.25 * pulse;
      this.quad(q.x, q.y, q.h, r, r, 0.3, [1.0, 0.72, 0.1, 0.32 + 0.18 * pulse], 0);
    }
    for (const ring of info.rings) {
      const col = ring.state === "done" ? [0.3, 1.0, 0.45] : ring.state === "crash" ? [1.0, 0.2, 0.15] : [0.75, 0.75, 0.8];
      const r = ring.r + 1.3 + ring.age * 2.0;
      this.quad(ring.x, ring.y, 0, r, r, 0.3, [...col, 0.55 * (1 - ring.age / 1.5)], 0);
    }
    for (const hk of info.honking) {
      const q = pose(prev, cur, hk.id, alpha);
      if (!q) continue;
      const r = 2 + hk.age * 14;
      this.quad(q.x + Math.cos(q.h) * hk.ahead, q.y + Math.sin(q.h) * hk.ahead, q.h, r, r, 1.2, [1.0, 0.6, 0.2, 0.5 * (1 - hk.age / 0.6)], 0);
    }
    this.fxGeo.setDrawRange(0, this.nFx * 6);
    for (const name of ["position", "fx_uv", "fx_col", "fx_kind"]) (this.fxGeo.getAttribute(name) as THREE.BufferAttribute).needsUpdate = true;
  }
}

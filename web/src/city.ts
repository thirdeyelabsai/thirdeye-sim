// The city (from web/export.py): the ground plane, then 600 m chunks of roads, buildings, trees and tanks,
// loaded nearest first. Each chunk becomes meshes in two scenes: the picture, and the shadow mask (everything
// that stands up, flattened onto the ground along the sun).
import * as THREE from "three";
import { FX_FS, FX_VS, INST_FS, INST_VS, LINE_VS, LIT_FS, LIT_VS, MAPFILL_FS, MAPFILL_VS, SHADOW_FS, SHADOW_VS } from "./shaders";

export type Meta = {
  starts: Record<string, [number, number, number]>;
  net_offset: [number, number];
  proj: string;
  base: number[][];
  chunk: number;
  chunks: { name: string; x: number; y: number; half: number; bytes: number }[];
  labels: [string, number, number, number, number][];
  places: { name: string; x: number; y: number; rank: number; colour: number[] }[];
  play_area: number[][];
  start_pool: [number, number, number][];
  attribution: string;
};

export const FOG = new THREE.Vector3(0.79, 0.79, 0.75);
const SUN = new THREE.Vector3(0.45, -0.4, 0.8).normalize();
export const SDIR = new THREE.Vector2(-SUN.x / SUN.z, -SUN.y / SUN.z);   // ground offset per metre of height

// shared by every material, set once per frame
export const uniforms = {
  u_light: { value: SUN },
  u_fog: { value: FOG },
  u_fogr: { value: new THREE.Vector2(100, 300) },
  u_amb: { value: 0 },
  u_sdir: { value: SDIR },
  u_shadow: { value: null as THREE.Texture | null },
  u_vp: { value: new THREE.Vector2(1, 1) },
};

function lit(ground: boolean, patches: number) {
  return new THREE.ShaderMaterial({
    vertexShader: LIT_VS, fragmentShader: LIT_FS, defines: ground ? { GROUND: 1 } : {},
    uniforms: { ...uniforms, u_patches: { value: patches } },
    side: THREE.DoubleSide,   // (as the desktop renderer: paint strips and some walls wind either way)
  });
}

function inst(shadowpass: number) {
  return new THREE.ShaderMaterial({
    vertexShader: INST_VS, fragmentShader: INST_FS,
    uniforms: { ...uniforms, u_shadowpass: { value: shadowpass } },
    side: THREE.DoubleSide,
    ...(shadowpass ? { depthTest: false, depthWrite: false } : {}),
  });
}

export const mapUniforms = { u_mpp: { value: 1 } };   // metres per pixel on the minimap

function line(casing: number) {
  return new THREE.ShaderMaterial({
    vertexShader: LINE_VS, fragmentShader: MAPFILL_FS, depthTest: false, depthWrite: false, side: THREE.DoubleSide,
    uniforms: { ...mapUniforms, u_case: { value: casing } },
  });
}

export const MAT = {
  base: lit(true, 1),
  ground: lit(true, 0),
  solid: lit(false, 0),
  inst: inst(0),
  instShadow: inst(1),
  shadow: new THREE.ShaderMaterial({
    vertexShader: SHADOW_VS, fragmentShader: SHADOW_FS, uniforms: { ...uniforms },
    depthTest: false, depthWrite: false, side: THREE.DoubleSide,
  }),
  mapFill: new THREE.ShaderMaterial({ vertexShader: MAPFILL_VS, fragmentShader: MAPFILL_FS, depthTest: false, depthWrite: false, side: THREE.DoubleSide }),
  mapCase: line(1),
  mapLine: line(0),
  fx: new THREE.ShaderMaterial({
    vertexShader: FX_VS, fragmentShader: FX_FS, transparent: true, depthWrite: false,
    blending: THREE.CustomBlending, blendSrc: THREE.SrcAlphaFactor, blendDst: THREE.OneFactor,
  }),
};

// a unit box standing on z = 0 and a once-subdivided icosphere (both 1 m across), as in render3d.py
function cube(): THREE.BufferGeometry {
  const faces: [number[], number[][]][] = [
    [[1, 0, 0], [[0.5, -0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 1], [0.5, -0.5, 1]]],
    [[-1, 0, 0], [[-0.5, 0.5, 0], [-0.5, -0.5, 0], [-0.5, -0.5, 1], [-0.5, 0.5, 1]]],
    [[0, 1, 0], [[0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, 0.5, 1], [0.5, 0.5, 1]]],
    [[0, -1, 0], [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, -0.5, 1], [-0.5, -0.5, 1]]],
    [[0, 0, 1], [[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]]],
  ];
  const pos: number[] = [], nrm: number[] = [];
  for (const [n, [a, b, c, d]] of faces) for (const p of [a, b, c, a, c, d]) { pos.push(...p); nrm.push(...n); }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.Float32BufferAttribute(nrm, 3));
  return g;
}

function icosphere(): THREE.BufferGeometry {
  const t = (1 + Math.sqrt(5)) / 2;
  const vs = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    .map((p) => new THREE.Vector3(...p).normalize());
  const fs = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  const pos: number[] = [], nrm: number[] = [];
  const mid = (p: THREE.Vector3, q: THREE.Vector3) => p.clone().add(q).normalize();
  for (const [a, b, c] of fs) {   // one subdivision, flat shaded for a low-poly look
    const A = vs[a], B = vs[b], C = vs[c], ab = mid(A, B), bc = mid(B, C), ca = mid(C, A);
    for (const tri of [[A, ab, ca], [B, bc, ab], [C, ca, bc], [ab, bc, ca]]) {
      const n = tri[1].clone().sub(tri[0]).cross(tri[2].clone().sub(tri[0])).normalize();
      for (const p of tri) { pos.push(p.x * 0.5, p.y * 0.5, p.z * 0.5 + 0.5); nrm.push(n.x, n.y, n.z); }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.Float32BufferAttribute(nrm, 3));
  return g;
}

export const CUBE = cube();
export const SPHERE = icosphere();

// instances: position, heading, size, colour, glow
export function instanced(shape: THREE.BufferGeometry, n: number): THREE.InstancedBufferGeometry {
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute("position", shape.getAttribute("position"));
  g.setAttribute("normal", shape.getAttribute("normal"));
  g.setAttribute("i_pos", new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3));
  g.setAttribute("i_head", new THREE.InstancedBufferAttribute(new Float32Array(n), 1));
  g.setAttribute("i_size", new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3));
  g.setAttribute("i_col", new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3));
  g.setAttribute("i_emit", new THREE.InstancedBufferAttribute(new Float32Array(n), 1));
  g.instanceCount = n;
  return g;
}

async function gunzip(buf: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

// reads the chunk layout written by export.py
class Reader {
  off = 0;
  constructor(public buf: ArrayBuffer) {}
  take<T>(Kind: { new (b: ArrayBuffer, o: number, n: number): T; BYTES_PER_ELEMENT: number }, n: number): T {
    const a = new Kind(this.buf, this.off, n);
    this.off += Math.ceil((n * Kind.BYTES_PER_ELEMENT) / 4) * 4;
    return a;
  }
}

export class City {
  scene = new THREE.Scene();      // the picture
  mask = new THREE.Scene();       // the shadow mask
  map = new THREE.Scene();        // the minimap, seen from straight above
  loaded = new Set<string>();
  bytes = 0;

  constructor(public meta: Meta) {
    const b = meta.base;
    const pos = new Float32Array(b.flatMap((r) => r.slice(0, 3)));
    const col = new Float32Array(b.flatMap((r) => r.slice(3, 6)));
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("col", new THREE.BufferAttribute(col, 3));
    const base = new THREE.Mesh(g, MAT.base);
    base.frustumCulled = false;
    base.renderOrder = -2;
    this.scene.add(base);
  }

  // load chunks nearest (x, y) first, a few at a time; onProgress(done, total) after each
  async loadAround(x: number, y: number, within: number, onProgress?: (done: number, total: number) => void) {
    const todo = this.meta.chunks
      .filter((c) => !this.loaded.has(c.name) && (c.name === "big" || Math.hypot(c.x - x, c.y - y) < within))
      .sort((a, b) => (a.name === "big" ? -1 : b.name === "big" ? 1 : Math.hypot(a.x - x, a.y - y) - Math.hypot(b.x - x, b.y - y)));
    let done = 0;
    const next = async (): Promise<void> => {
      const c = todo.shift();
      if (!c) return;
      this.loaded.add(c.name);
      for (let attempt = 0; attempt < 3; attempt++) {
        try {   // (a dropped connection on a phone should not leave a hole in the city)
          const res = await fetch(`${import.meta.env.BASE_URL}data/scene/${c.name}.binz`);
          if (!res.ok) throw new Error(`${res.status}`);
          const raw = await gunzip(await res.arrayBuffer());
          this.bytes += c.bytes;
          if (c.name === "big") this.addBig(raw); else this.addChunk(raw, c);
          break;
        } catch (err) {
          if (attempt === 2) { this.loaded.delete(c.name); console.warn(`chunk ${c.name} not loaded: ${err}`); }
          else await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
        }
      }
      onProgress?.(++done, done + todo.length);
      return next();
    };
    await Promise.all([next(), next(), next(), next()]);
  }

  private addChunk(buf: ArrayBuffer, c: { x: number; y: number; half: number }) {
    const r = new Reader(buf);
    r.off = 4;
    const [nG, nS, nB, nP, nF, nL] = Array.from(r.take(Uint32Array, 6));
    const [ox, oy] = Array.from(r.take(Float32Array, 2));
    const sphere = new THREE.Sphere(new THREE.Vector3(c.x, c.y, 0), c.half * 1.42 + 150);
    if (nG) {
      const x = r.take(Int16Array, nG), y = r.take(Int16Array, nG), z = r.take(Int16Array, nG), col = r.take(Uint8Array, nG * 3);
      const pos = new Float32Array(nG * 3);
      for (let i = 0; i < nG; i++) { pos[3 * i] = ox + x[i] * 0.02; pos[3 * i + 1] = oy + y[i] * 0.02; pos[3 * i + 2] = z[i] * 1e-4; }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("col", new THREE.BufferAttribute(col, 3, true));
      g.boundingSphere = sphere;
      const m = new THREE.Mesh(g, MAT.ground);
      m.renderOrder = -1;
      this.scene.add(m);
    }
    if (nS) {
      const x = r.take(Int16Array, nS), y = r.take(Int16Array, nS), z = r.take(Int16Array, nS), col = r.take(Uint8Array, nS * 3);
      const nrm = r.take(Int8Array, nS * 3), recv = r.take(Uint8Array, nS);
      const pos = new Float32Array(nS * 3);
      for (let i = 0; i < nS; i++) { pos[3 * i] = ox + x[i] * 0.02; pos[3 * i + 1] = oy + y[i] * 0.02; pos[3 * i + 2] = z[i] * 0.01; }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("col", new THREE.BufferAttribute(col, 3, true));
      g.setAttribute("normal", new THREE.BufferAttribute(nrm, 3, true));
      g.setAttribute("recv", new THREE.BufferAttribute(recv, 1, false));
      g.boundingSphere = sphere;
      this.scene.add(new THREE.Mesh(g, MAT.solid));
      const s = new THREE.Mesh(g, MAT.shadow);
      this.mask.add(s);
    }
    for (const [n, shape] of [[nB, CUBE], [nP, SPHERE]] as [number, THREE.BufferGeometry][]) {
      if (!n) continue;
      const x = r.take(Int16Array, n), y = r.take(Int16Array, n), z = r.take(Int16Array, n), yaw = r.take(Uint16Array, n);
      const size = r.take(Uint16Array, n * 3), col = r.take(Uint8Array, n * 3);
      const g = instanced(shape, n);
      const P = g.getAttribute("i_pos").array as Float32Array, H = g.getAttribute("i_head").array as Float32Array;
      const S = g.getAttribute("i_size").array as Float32Array, C = g.getAttribute("i_col").array as Float32Array;
      for (let i = 0; i < n; i++) {
        P[3 * i] = ox + x[i] * 0.02; P[3 * i + 1] = oy + y[i] * 0.02; P[3 * i + 2] = z[i] * 0.01;
        H[i] = (yaw[i] * 2 * Math.PI) / 65536;
        for (let k = 0; k < 3; k++) { S[3 * i + k] = size[3 * i + k] * 0.01; C[3 * i + k] = col[3 * i + k] / 255; }
      }
      g.boundingSphere = sphere;
      this.scene.add(new THREE.Mesh(g, MAT.inst));
      this.mask.add(new THREE.Mesh(g, MAT.instShadow));
    }
    if (nF) {
      const x = r.take(Int16Array, nF), y = r.take(Int16Array, nF), col = r.take(Uint8Array, nF * 3);
      const pos = new Float32Array(nF * 3);
      for (let i = 0; i < nF; i++) { pos[3 * i] = ox + x[i] * 0.02; pos[3 * i + 1] = oy + y[i] * 0.02; }
      this.addMapFill(pos, new THREE.BufferAttribute(col, 3, true), sphere);
    }
    if (nL) {
      const x = r.take(Int16Array, nL), y = r.take(Int16Array, nL), n = r.take(Int8Array, nL * 2), col = r.take(Uint8Array, nL * 3);
      const hw = r.take(Uint16Array, nL), px = r.take(Uint8Array, nL);
      const pos = new Float32Array(nL * 3), H = new Float32Array(nL), P = new Float32Array(nL);
      for (let i = 0; i < nL; i++) { pos[3 * i] = ox + x[i] * 0.02; pos[3 * i + 1] = oy + y[i] * 0.02; H[i] = hw[i] * 0.01; P[i] = px[i] * 0.1; }
      this.addMapLines(pos, new THREE.BufferAttribute(n, 2, true), new THREE.BufferAttribute(col, 3, true), H, P, sphere);
    }
  }

  private addMapFill(pos: Float32Array, col: THREE.BufferAttribute, sphere: THREE.Sphere | null) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("col", col);
    const m = new THREE.Mesh(g, MAT.mapFill);
    if (sphere) g.boundingSphere = sphere; else m.frustumCulled = false;
    m.renderOrder = 0;
    this.map.add(m);
  }

  private addMapLines(pos: Float32Array, n: THREE.BufferAttribute, col: THREE.BufferAttribute, hw: Float32Array, px: Float32Array, sphere: THREE.Sphere | null) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("l_n", n);
    g.setAttribute("col", col);
    g.setAttribute("l_hw", new THREE.BufferAttribute(hw, 1));
    g.setAttribute("l_px", new THREE.BufferAttribute(px, 1));
    for (const [mat, order] of [[MAT.mapCase, 1], [MAT.mapLine, 2]] as [THREE.ShaderMaterial, number][]) {
      const m = new THREE.Mesh(g, mat);
      if (sphere) g.boundingSphere = sphere; else m.frustumCulled = false;
      m.renderOrder = order;
      this.map.add(m);
    }
  }

  private addBig(buf: ArrayBuffer) {   // the few pieces over 100 m across, as plain float32 rows
    const r = new Reader(buf);
    r.off = 4;
    const [nG, nS, nB, nP, nF, nL] = Array.from(r.take(Uint32Array, 6));
    const rows = (n: number, w: number) => r.take(Float32Array, n * w);
    for (const [n, mat, ground] of [[nG, MAT.ground, true], [nS, MAT.solid, false]] as [number, THREE.ShaderMaterial, boolean][]) {
      if (!n) continue;
      const a = rows(n, 10);
      const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), nrm = new Float32Array(n * 3), recv = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < 3; k++) { pos[3 * i + k] = a[10 * i + k]; col[3 * i + k] = a[10 * i + 3 + k]; nrm[3 * i + k] = a[10 * i + 6 + k]; }
        recv[i] = a[10 * i + 9];
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("col", new THREE.BufferAttribute(col, 3));
      if (!ground) { g.setAttribute("normal", new THREE.BufferAttribute(nrm, 3)); g.setAttribute("recv", new THREE.BufferAttribute(recv, 1)); }
      const m = new THREE.Mesh(g, mat);
      m.frustumCulled = false;
      if (ground) m.renderOrder = -1;
      this.scene.add(m);
      if (!ground) { const s = new THREE.Mesh(g, MAT.shadow); s.frustumCulled = false; this.mask.add(s); }
    }
    for (const [n, shape] of [[nB, CUBE], [nP, SPHERE]] as [number, THREE.BufferGeometry][]) {
      if (!n) continue;
      const a = rows(n, 11);
      const g = instanced(shape, n);
      const P = g.getAttribute("i_pos").array as Float32Array, H = g.getAttribute("i_head").array as Float32Array;
      const S = g.getAttribute("i_size").array as Float32Array, C = g.getAttribute("i_col").array as Float32Array;
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < 3; k++) { P[3 * i + k] = a[11 * i + k]; S[3 * i + k] = a[11 * i + 4 + k]; C[3 * i + k] = a[11 * i + 7 + k]; }
        H[i] = a[11 * i + 3];
      }
      const m = new THREE.Mesh(g, MAT.inst); m.frustumCulled = false; this.scene.add(m);
      const s = new THREE.Mesh(g, MAT.instShadow); s.frustumCulled = false; this.mask.add(s);
    }
    if (nF) {
      const a = rows(nF, 10), pos = new Float32Array(nF * 3), col = new Float32Array(nF * 3);
      for (let i = 0; i < nF; i++) for (let k = 0; k < 3; k++) { pos[3 * i + k] = k < 2 ? a[10 * i + k] : 0; col[3 * i + k] = a[10 * i + 3 + k]; }
      this.addMapFill(pos, new THREE.BufferAttribute(col, 3), null);
    }
    if (nL) {
      const a = rows(nL, 9), pos = new Float32Array(nL * 3), n = new Float32Array(nL * 2), col = new Float32Array(nL * 3);
      const hw = new Float32Array(nL), px = new Float32Array(nL);
      for (let i = 0; i < nL; i++) {
        pos[3 * i] = a[9 * i]; pos[3 * i + 1] = a[9 * i + 1]; n[2 * i] = a[9 * i + 2]; n[2 * i + 1] = a[9 * i + 3];
        for (let k = 0; k < 3; k++) col[3 * i + k] = a[9 * i + 4 + k];
        hw[i] = a[9 * i + 7]; px[i] = a[9 * i + 8];
      }
      this.addMapLines(pos, new THREE.BufferAttribute(n, 2), new THREE.BufferAttribute(col, 3), hw, px, null);
    }
  }
}

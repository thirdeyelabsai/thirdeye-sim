// The desktop renderer's shaders (thirdeye_sim/render3d.py), for three.js ShaderMaterial: the same light,
// ground grain, flattened shadows and haze, so both versions look alike. Coordinates are the simulator's:
// metres, x east, y north, z up.

const LIGHT = /* glsl */ `
uniform vec3 u_light; uniform vec3 u_fog; uniform vec2 u_fogr; uniform float u_amb;
vec3 light(vec3 col, vec3 n, vec3 L, float amb) {
  // a cool sky from above, a warm sun from the south-east
  float d = max(dot(normalize(n), L), 0.0);
  vec3 sky = mix(vec3(0.50, 0.53, 0.60), vec3(0.62, 0.64, 0.68), clamp(n.z, 0.0, 1.0));
  return col * mix(sky + vec3(1.00, 0.93, 0.80) * 0.52 * d, vec3(1.0), amb);
}
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
float haze(vec3 world) { return clamp((length(world - cameraPosition) - u_fogr.x) / (u_fogr.y - u_fogr.x), 0.0, 1.0); }
`;

// ground, roads, buildings: triangles with a colour per vertex (GROUND: flat and always catching shadows)
export const LIT_VS = /* glsl */ `
attribute vec3 col;
#ifndef GROUND
attribute float recv;
#endif
varying vec3 v_col; varying vec3 v_nrm; varying float v_recv; varying vec3 v_world;
void main() {
  v_col = col; v_world = position;
#ifdef GROUND
  v_nrm = vec3(0.0, 0.0, 1.0); v_recv = 1.0;
#else
  v_nrm = normal; v_recv = recv;
#endif
  gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
}`;

export const LIT_FS = /* glsl */ `
uniform sampler2D u_shadow; uniform vec2 u_vp; uniform int u_patches;
varying vec3 v_col; varying vec3 v_nrm; varying float v_recv; varying vec3 v_world;
${LIGHT}
void main() {
  vec3 col = v_col;
  if (u_patches == 1)   // open ground: dry grass and dust in patches
    col = mix(col, col * vec3(0.80, 0.98, 0.72), smoothstep(0.42, 0.72, noise(v_world.xy * 0.018) * 0.7 + noise(v_world.xy * 0.09) * 0.3));
  vec3 c = light(col, v_nrm, u_light, u_amb);
  if (v_nrm.z > 0.9 && u_amb < 0.99)   // faint grain and patches on flat ground, asphalt and paving
    c *= 0.93 + 0.05 * noise(v_world.xy * 0.45) + 0.04 * noise(v_world.xy * 0.035);
  float sh = v_recv * texture2D(u_shadow, gl_FragCoord.xy / u_vp).r;
  c *= mix(vec3(1.0), vec3(0.58, 0.62, 0.74), sh);
  gl_FragColor = vec4(mix(c, u_fog, haze(v_world)), 1.0);
}`;

// boxes and spheres placed many times: trees, rooftop tanks, road users, signals
export const INST_VS = /* glsl */ `
uniform vec2 u_sdir; uniform int u_shadowpass;
attribute vec3 i_pos; attribute float i_head; attribute vec3 i_size; attribute vec3 i_col; attribute float i_emit;
varying vec3 v_col; varying vec3 v_nrm; varying float v_emit; varying vec3 v_world;
void main() {
  vec3 p = position * i_size;
  float c = cos(i_head), s = sin(i_head);
  p = vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z) + i_pos;
  v_nrm = vec3(c * normal.x - s * normal.y, s * normal.x + c * normal.y, normal.z);
  if (u_shadowpass == 1) p = vec3(p.xy + p.z * u_sdir, 0.1);
  v_world = p; v_col = i_col; v_emit = i_emit;
  gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
}`;

export const INST_FS = /* glsl */ `
uniform int u_shadowpass;
varying vec3 v_col; varying vec3 v_nrm; varying float v_emit; varying vec3 v_world;
${LIGHT}
void main() {
  if (u_shadowpass == 1) { gl_FragColor = vec4(1.0); return; }
  vec3 c = mix(light(v_col, v_nrm, u_light, u_amb), v_col * 1.15, v_emit);
  gl_FragColor = vec4(mix(c, u_fog, haze(v_world) * (1.0 - v_emit)), 1.0);
}`;

// buildings flattened onto the ground along the sun, into the shadow mask
export const SHADOW_VS = /* glsl */ `
uniform vec2 u_sdir;
void main() { gl_Position = projectionMatrix * viewMatrix * vec4(position.xy + position.z * u_sdir, 0.1, 1.0); }`;

export const SHADOW_FS = /* glsl */ `
void main() { gl_FragColor = vec4(1.0); }`;

// horn rings, headlight flashes, event rings: quads blended on top
export const FX_VS = /* glsl */ `
attribute vec2 fx_uv; attribute vec4 fx_col; attribute float fx_kind;
varying vec2 v_uv; varying vec4 v_col; varying float v_kind;
void main() { v_uv = fx_uv; v_col = fx_col; v_kind = fx_kind; gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0); }`;

export const FX_FS = /* glsl */ `
varying vec2 v_uv; varying vec4 v_col; varying float v_kind;
void main() {
  float a;
  float r = length(v_uv);
  if (v_kind < 0.5) a = smoothstep(0.70, 0.86, r) * (1.0 - smoothstep(0.90, 1.0, r));                 // ring
  else if (v_kind < 1.5) a = (1.0 - v_uv.x) * (1.0 - abs(v_uv.y)) * smoothstep(0.0, 0.08, v_uv.x);   // beam
  else a = 1.0 - smoothstep(0.0, 1.0, r);                                                               // soft disc
  gl_FragColor = vec4(v_col.rgb, v_col.a * a);
}`;

// the minimap (as render3d.py): parks, water and buildings flat, then roads at their real width but never
// thinner than a few pixels, drawn twice (a darker casing a pixel wider, then the road)
export const MAPFILL_VS = /* glsl */ `
attribute vec3 col; varying vec3 v_col;
void main() { v_col = col; gl_Position = projectionMatrix * viewMatrix * vec4(position.xy, 0.0, 1.0); }`;

export const MAPFILL_FS = /* glsl */ `
varying vec3 v_col;
void main() { gl_FragColor = vec4(v_col, 1.0); }`;

export const LINE_VS = /* glsl */ `
uniform float u_mpp; uniform float u_case;
attribute vec2 l_n; attribute vec3 col; attribute float l_hw; attribute float l_px;
varying vec3 v_col;
void main() {
  float w = max(l_hw, (l_px + u_case) * u_mpp);
  gl_Position = projectionMatrix * viewMatrix * vec4(position.xy + l_n * w, 0.0, 1.0);
  v_col = u_case > 0.0 ? col * 0.72 : col;
}`;

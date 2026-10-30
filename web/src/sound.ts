// Horns, made up the same way as the desktop game's (thirdeye_sim/sound.py): two buzzy pulse tones about a
// major third apart with the ring of a vibrating disc near 2.4 kHz. Ours is an SUV's dual-tone horn (415 and
// 510 Hz); road users honking at us get a higher, shorter beep, quieter the further away they are.
const RATE = 44100;

function peak(x: Float32Array, f0: number, q: number, gainDb: number) {
  const a = 10 ** (gainDb / 40), w = (2 * Math.PI * f0) / RATE, al = Math.sin(w) / (2 * q);
  const b0 = 1 + al * a, b1 = -2 * Math.cos(w), b2 = 1 - al * a, a0 = 1 + al / a, a1 = -2 * Math.cos(w), a2 = 1 - al / a;
  const y = new Float32Array(x.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const out = (b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
    x2 = x1; x1 = x[i]; y2 = y1; y1 = out; y[i] = out;
  }
  return y;
}

function discHorn(freqs: number[], seconds: number, attack = 0.015, release = 0.06) {
  const n = Math.floor(RATE * seconds);
  let wave = new Float32Array(n);
  for (const f of freqs) {
    for (let i = 0; i < n; i++) {
      const t = i / RATE;
      const phase = (f * t * (1 + 0.003 * Math.sin(2 * Math.PI * 5 * t))) % 1;
      wave[i] += phase < 0.32 ? 1 : -0.45;   // a narrow pulse: the disc slapping its stop
    }
  }
  const mean = wave.reduce((a, b) => a + b, 0) / n;
  for (let i = 0; i < n; i++) wave[i] -= mean;
  wave = peak(wave, 2400, 3, 9);
  let m = 0;
  for (const v of wave) m = Math.max(m, Math.abs(v));
  for (let i = 0; i < n; i++) wave[i] = Math.tanh((1.6 * wave[i]) / (m || 1));
  m = 0;
  for (let i = 0; i < n; i++) {
    const t = i / RATE;
    wave[i] *= Math.min(1, t / attack) * Math.min(1, (seconds - t) / release);
    m = Math.max(m, Math.abs(wave[i]));
  }
  for (let i = 0; i < n; i++) wave[i] = (wave[i] / (m || 1)) * 0.4;
  return wave;
}

export class Sounds {
  private ctx: AudioContext | null = null;
  private horn: AudioBuffer | null = null;
  private beep: AudioBuffer | null = null;

  // browsers allow sound only after the player touches a key or the screen
  wake() {
    if (this.ctx) return;
    this.ctx = new AudioContext();
    const make = (w: Float32Array) => {
      const b = this.ctx!.createBuffer(1, w.length, RATE);
      b.copyToChannel(w as Float32Array<ArrayBuffer>, 0);
      return b;
    };
    this.horn = make(discHorn([415, 510], 0.5));
    this.beep = make(discHorn([560, 680], 0.3));
  }

  private play(buf: AudioBuffer | null, volume: number) {
    if (!this.ctx || !buf) return;
    const src = this.ctx.createBufferSource(), gain = this.ctx.createGain();
    gain.gain.value = volume;
    src.buffer = buf;
    src.connect(gain).connect(this.ctx.destination);
    src.start();
  }

  // going over the speed limit: a soft two-note chime
  chime() {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    for (const [f, at] of [[988, 0], [740, 0.13]]) {
      const o = this.ctx.createOscillator(), g = this.ctx.createGain();
      o.frequency.value = f;
      g.gain.setValueAtTime(0, t + at);
      g.gain.linearRampToValueAtTime(0.18, t + at + 0.01);
      g.gain.exponentialRampToValueAtTime(0.001, t + at + 0.35);
      o.connect(g).connect(this.ctx.destination);
      o.start(t + at); o.stop(t + at + 0.4);
    }
  }

  ourHorn() { this.play(this.horn, 1); }
  theirHorn(distance: number) { this.play(this.beep, Math.max(0.05, Math.min(1, 12 / Math.max(12, distance)))); }
}

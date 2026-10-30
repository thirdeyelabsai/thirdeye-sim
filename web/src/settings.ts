// The drive's settings, kept in the address bar so a link replays the same drive: where it starts and the
// seed (the same traffic and events), the traffic style, chaos, and whether scripted events run. The panel
// changes them while driving.

export type Style = "gurgaon" | "generic";
export type Settings = { start: string; seed: number; chaos: number; style: Style; scripted: boolean };

const STARTS: [string, string][] = [["random", "A random main road"], ["medanta", "Near Medanta"], ["sunset", "Sunset Boulevard"]];
export const CHAOS = { min: 0.5, max: 3, step: 0.25 };

export function newSeed() {
  return Math.floor(Math.random() * 2 ** 31);
}

export function readSettings(): Settings {
  const p = new URLSearchParams(location.search);
  const num = (k: string, d: number) => (p.has(k) && Number.isFinite(Number(p.get(k))) ? Number(p.get(k)) : d);
  return {
    start: STARTS.some(([k]) => k === p.get("start")) ? p.get("start")! : "random",
    seed: num("seed", newSeed()),
    chaos: Math.max(CHAOS.min, Math.min(CHAOS.max, num("chaos", 1))),
    style: p.get("traffic") === "generic" ? "generic" : "gurgaon",
    scripted: p.get("events") !== "off",
  };
}

export function link(s: Settings) {
  const p = new URLSearchParams();
  if (s.start !== "random") p.set("start", s.start);
  p.set("seed", String(s.seed));
  if (s.chaos !== 1) p.set("chaos", String(s.chaos));
  if (s.style !== "gurgaon") p.set("traffic", s.style);
  if (!s.scripted) p.set("events", "off");
  return `${location.origin}${location.pathname}?${p}`;
}

export function chaosText(c: number) {
  return c === 1 ? "As measured" : `${c}x the measured events${c > 1 ? ", pushier drivers" : ", calmer drivers"}`;
}

type Hooks = { change: (p: Partial<Settings>) => void; restart: (start: string) => void; banners: (on: boolean) => void };

export class Panel {
  private root = document.createElement("div");
  private button = document.createElement("button");

  constructor(private s: Settings, private hooks: Hooks) {
    this.button.id = "gear";
    this.button.textContent = "Settings";
    this.button.onclick = () => this.toggle();
    this.root.id = "panel";
    this.root.className = "panel";
    this.root.hidden = true;
    this.root.innerHTML = `
      <h3>Drive settings</h3>
      <label class="row">Traffic</label>
      <div class="seg"><button data-style="gurgaon">Gurgaon, measured</button><button data-style="generic">Generic simulator</button></div>
      <p class="hint" id="style-hint"></p>
      <label class="row" for="chaos-range">Chaos <span id="chaos-val"></span></label>
      <input type="range" id="chaos-range" min="${CHAOS.min}" max="${CHAOS.max}" step="${CHAOS.step}" />
      <label class="check"><input type="checkbox" id="scripted" /> Scripted events (the rarer moments, at measured rates)</label>
      <label class="check"><input type="checkbox" id="banners" checked /> Event banners and tags</label>
      <label class="row" for="start">Start</label>
      <div class="line"><select id="start">${STARTS.map(([k, t]) => `<option value="${k}">${t}</option>`).join("")}</select>
        <button id="restart">New drive</button></div>
      <button id="share">Copy a link to this drive</button>
      <p class="hint">The link replays the same start, traffic and events.</p>`;
    document.body.append(this.button, this.root);
    const q = <T extends HTMLElement>(sel: string) => this.root.querySelector(sel) as T;
    this.root.querySelectorAll<HTMLButtonElement>(".seg button").forEach((b) => {
      b.onclick = () => this.set({ style: b.dataset.style as Style });
    });
    q<HTMLInputElement>("#chaos-range").oninput = (e) => this.set({ chaos: Number((e.target as HTMLInputElement).value) });
    q<HTMLInputElement>("#scripted").onchange = (e) => this.set({ scripted: (e.target as HTMLInputElement).checked });
    q<HTMLInputElement>("#banners").onchange = (e) => hooks.banners((e.target as HTMLInputElement).checked);
    q<HTMLButtonElement>("#restart").onclick = () => hooks.restart(q<HTMLSelectElement>("#start").value);
    q<HTMLButtonElement>("#share").onclick = async () => {
      const b = q<HTMLButtonElement>("#share");
      try { await navigator.clipboard.writeText(link(this.s)); b.textContent = "Link copied"; }
      catch { prompt("Copy this link", link(this.s)); }
      setTimeout(() => (b.textContent = "Copy a link to this drive"), 2000);
    };
    // keys typed into the panel stay in the panel; Escape closes it
    this.root.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Escape") this.root.hidden = true;
    });
    this.sync(s);
  }

  get open() { return !this.root.hidden; }
  toggle() { this.root.hidden = !this.root.hidden; }

  private set(p: Partial<Settings>) {
    Object.assign(this.s, p);
    this.hooks.change(p);
    this.sync(this.s);
  }

  // show the current settings (also after a key changed them)
  sync(s: Settings) {
    this.s = s;
    const q = <T extends HTMLElement>(sel: string) => this.root.querySelector(sel) as T;
    this.root.querySelectorAll<HTMLButtonElement>(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.style === s.style));
    q("#style-hint").textContent = s.style === "gurgaon"
      ? "Road users and events at the rates we measured from Delhi NCR dashcam footage."
      : "Traffic as most simulators model it: everyone in lane, standard lane changes, no honking, no events.";
    q<HTMLInputElement>("#chaos-range").value = String(s.chaos);
    q("#chaos-val").textContent = chaosText(s.chaos);
    q<HTMLInputElement>("#scripted").checked = s.scripted;
    q<HTMLInputElement>("#scripted").disabled = s.style === "generic";
    q<HTMLSelectElement>("#start").value = s.start;
    history.replaceState(null, "", link(s));
  }
}

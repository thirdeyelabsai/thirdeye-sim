// Phone controls: steering on the left thumb, pedals on the right, horn and headlight flash above them. Each
// button holds a key while a finger is on it, so several can be pressed at once (steer while braking).

type Button = { key: string; label: string; cls: string; tap?: boolean };

const BUTTONS: Button[] = [
  { key: "arrowleft", label: "◀", cls: "steer left" },
  { key: "arrowright", label: "▶", cls: "steer right" },
  { key: "arrowdown", label: "Brake", cls: "pedal brake" },
  { key: "arrowup", label: "Go", cls: "pedal go" },
  { key: "h", label: "Horn", cls: "small horn" },
  { key: "f", label: "Flash", cls: "small flash", tap: true },
  { key: "m", label: "Map", cls: "small map", tap: true },
];

export function isTouch() {
  return matchMedia("(pointer: coarse)").matches || new URLSearchParams(location.search).has("touch");
}

export function touchControls(press: (key: string, on: boolean) => void) {
  const box = document.createElement("div");
  box.id = "touch";
  for (const b of BUTTONS) {
    const el = document.createElement("button");
    el.className = b.cls;
    el.textContent = b.label;
    const up = () => { el.classList.remove("down"); press(b.key, false); };
    el.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      el.classList.add("down");
      press(b.key, true);
      if (b.tap) setTimeout(up, 120);
    });
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("lostpointercapture", up);
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    box.appendChild(el);
  }
  document.body.appendChild(box);
  document.body.classList.add("touch");
}

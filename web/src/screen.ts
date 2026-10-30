// The size of the screen the game draws on: the 3D view's own box, which the page locks to the visible area
// (style.css). On phones, innerWidth can disagree with what is actually visible (notches, browser bars), so
// nothing measures the window directly.
export function screenSize(): [number, number] {
  const el = document.getElementById("view");
  const w = el?.clientWidth || document.documentElement.clientWidth || innerWidth;
  const h = el?.clientHeight || document.documentElement.clientHeight || innerHeight;
  return [w, h];
}

// Map coordinates (metres, as in the SUMO network) to latitude and longitude: undo SUMO's netOffset, then
// the inverse UTM projection on WGS84 (Krueger's series, accurate to well under a millimetre in the zone).
const A = 6378137, F = 1 / 298.257223563, K0 = 0.9996;
const N = F / (2 - F);
const AA = (A / (1 + N)) * (1 + (N * N) / 4 + (N ** 4) / 64);
const BETA = [N / 2 - (2 / 3) * N * N + (37 / 96) * N ** 3, (1 / 48) * N * N + (1 / 15) * N ** 3, (17 / 480) * N ** 3];
const DELTA = [2 * N - (2 / 3) * N * N - 2 * N ** 3, (7 / 3) * N * N - (8 / 5) * N ** 3, (56 / 15) * N ** 3];

export function makeLonLat(netOffset: [number, number], proj: string) {
  const zone = Number(/\+zone=(\d+)/.exec(proj)?.[1] ?? 43);
  const south = /\+south/.test(proj);
  const lon0 = ((zone - 1) * 6 - 180 + 3) * (Math.PI / 180);
  return (x: number, y: number): [number, number] => {
    const E = x - netOffset[0] - 500000, Nn = y - netOffset[1] - (south ? 10000000 : 0);
    const xi = Nn / (K0 * AA), eta = E / (K0 * AA);
    let xi1 = xi, eta1 = eta;
    for (let j = 1; j <= 3; j++) {
      xi1 -= BETA[j - 1] * Math.sin(2 * j * xi) * Math.cosh(2 * j * eta);
      eta1 -= BETA[j - 1] * Math.cos(2 * j * xi) * Math.sinh(2 * j * eta);
    }
    const chi = Math.asin(Math.sin(xi1) / Math.cosh(eta1));
    let lat = chi;
    for (let j = 1; j <= 3; j++) lat += DELTA[j - 1] * Math.sin(2 * j * chi);
    const lon = lon0 + Math.atan2(Math.sinh(eta1), Math.cos(xi1));
    return [(lon * 180) / Math.PI, (lat * 180) / Math.PI];
  };
}

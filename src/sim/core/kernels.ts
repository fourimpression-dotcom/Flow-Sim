// SPH smoothing kernels (Müller, Charypar & Gross 2003, "Particle-Based Fluid
// Simulation for Interactive Applications"). These are the literal reference
// formulas: written as plain math, not optimized/unrolled, so a future TSL
// (GPU) implementation can be checked against them term-for-term.
//
// Convention: h = smoothing radius, r = distance between two particles
// (0 <= r), all kernels are zero outside r > h.
// Each kernel exposes a *Coefficient() function (depends only on h — compute
// once per step) and the per-pair evaluation function (takes the coefficient
// so it isn't recomputed for every particle pair).

/** Poly6 kernel, used for density estimation. */
export function poly6Coefficient(h: number): number {
  return 315 / (64 * Math.PI * Math.pow(h, 9));
}

export function poly6(rSq: number, h: number, coefficient: number): number {
  const hSq = h * h;
  if (rSq >= hSq) return 0;
  const diff = hSq - rSq;
  return coefficient * diff * diff * diff;
}

/**
 * Spiky kernel gradient magnitude, used for the pressure force. Spiky (rather
 * than poly6) is used here because poly6's gradient vanishes at r -> 0,
 * which lets particles clump; spiky stays strongly repulsive at short range.
 *
 * grad W_spiky(r, h) = coefficient * (h - r)^2 * (rVec / r)
 * Callers multiply the returned scalar by the unit vector rVec/r themselves.
 */
export function spikyGradientCoefficient(h: number): number {
  return -45 / (Math.PI * Math.pow(h, 6));
}

export function spikyGradientMagnitude(r: number, h: number, coefficient: number): number {
  if (r <= 0 || r >= h) return 0;
  const diff = h - r;
  return coefficient * diff * diff;
}

/** Viscosity kernel Laplacian, used for the viscous force. */
export function viscosityLaplacianCoefficient(h: number): number {
  return 45 / (Math.PI * Math.pow(h, 6));
}

export function viscosityLaplacian(r: number, h: number, coefficient: number): number {
  if (r >= h) return 0;
  return coefficient * (h - r);
}

/**
 * Tait equation of state: relates density back to pressure for weakly-
 * compressible SPH. Negative pressure is clamped to 0 — the standard
 * free-surface-flow trick that avoids unphysical inter-particle attraction
 * in low-density (near-surface) regions.
 */
export function taitPressure(
  density: number,
  restDensity: number,
  stiffness: number,
  gamma: number
): number {
  const ratio = density / restDensity;
  const pressure = stiffness * (Math.pow(ratio, gamma) - 1);
  return pressure > 0 ? pressure : 0;
}

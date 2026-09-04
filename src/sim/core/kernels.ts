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
 * Akinci, Akinci & Teschner 2013 ("Versatile Surface Tension and Adhesion
 * for SPH Fluids") cohesion kernel, used for the pairwise surface-tension
 * force. Negative for r < h/2 (short range) and positive for h/2 < r < h
 * (longer range), continuous through the switch at r = h/2 and reaching
 * exactly 0 at r = h. Combined with the force formula's sign (see
 * CpuSphBackend.computeForces), this makes cohesion self-stabilizing:
 * particles that get too close are pushed apart (the negative branch),
 * while particles drifting apart within the kernel support are pulled back
 * together (the positive branch) — unlike reusing a density kernel such as
 * poly6 (which is purely positive and peaks at r=0), there's no risk of
 * attraction blowing up as particles approach each other.
 */
export function cohesionKernelCoefficient(h: number): number {
  return 32 / (Math.PI * Math.pow(h, 9));
}

export function cohesionKernel(r: number, h: number, coefficient: number): number {
  if (r <= 0 || r > h) return 0;
  const hr = h - r;
  const term = hr * hr * hr * r * r * r;
  if (2 * r > h) {
    return coefficient * term;
  }
  const h6 = Math.pow(h, 6);
  return coefficient * (2 * term - h6 / 64);
}

/**
 * Boundary-adhesion (wetting) kernel — distinct from cohesion above.
 * Cohesion is a fluid-fluid pairwise force (kernel of particle-pair
 * distance r, evaluated in the fluid neighbor search); adhesion is a
 * fluid-*solid* force, evaluated once per particle from its signed
 * distance to the collision mesh, not from any particle pair. Akinci et
 * al. 2013 define adhesion the same way as cohesion — as a pairwise force
 * against sampled boundary particles — but this simulation's boundary is a
 * signed distance field, not a particle set, so there is no boundary
 * particle to pair against; this kernel is a distance-based approximation
 * built for that: zero at both ends of [distMin, distMax] (the collision
 * margin and the outer edge of the adhesion band) with zero slope at both
 * ends too (an unclamped bump would start/stop the pull abruptly as a
 * particle crosses into or out of the band), peaking in between.
 */
export function adhesionKernel(dist: number, distMin: number, distMax: number): number {
  if (dist <= distMin || dist >= distMax) return 0;
  const t = (dist - distMin) / (distMax - distMin);
  const bump = t * (1 - t);
  return 16 * bump * bump; // normalized so the peak (t=0.5) is 1
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

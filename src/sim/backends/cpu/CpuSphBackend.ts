import type { CollisionField, SphComputeBackend, SphDiagnostics, SphParams, Vec3Tuple } from "../../core/types";
import {
  adhesionKernel,
  cohesionKernel,
  cohesionKernelCoefficient,
  poly6,
  poly6Coefficient,
  spikyGradientCoefficient,
  spikyGradientMagnitude,
  taitPressure,
  viscosityLaplacian,
  viscosityLaplacianCoefficient,
} from "../../core/kernels";
import { reflectVelocityAgainstNormal } from "../../core/collision";
import { sampleGradient, sampleSignedDistance, type SignedDistanceField } from "../../geometry/signedDistanceField";
import { SpatialGrid } from "./spatialGrid";

/**
 * Reference CPU implementation of SphComputeBackend: a plain per-particle
 * loop over typed arrays, run on the main thread. Implements the fixed
 * 10-stage pipeline (neighbor search -> density/pressure -> fluid forces ->
 * mesh adhesion force -> velocity integration -> wall friction -> XSPH
 * smoothing -> position integration -> mesh collision -> domain boundary)
 * declared by the SphComputeBackend contract, using the shared
 * kernel/collision formulas from core/kernels.ts and core/collision.ts
 * verbatim — this is the pairing a future GPU backend must reproduce.
 */
export class CpuSphBackend implements SphComputeBackend {
  private count = 0;
  private positions = new Float32Array(0);
  private velocities = new Float32Array(0);
  private densities = new Float32Array(0);
  private pressures = new Float32Array(0);
  private forces = new Float32Array(0);
  // Scratch, recomputed every step: the XSPH-smoothed velocity used only to
  // advect position (see computeXsphCorrection). Not part of persistent
  // particle state, so removeParticleAt never needs to touch it.
  private advectionVelocities = new Float32Array(0);
  private grid: SpatialGrid | null = null;
  private collisionField: SignedDistanceField | null = null;
  private readonly neighborScratch: number[] = [];

  get particleCount(): number {
    return this.count;
  }

  init(initialPositions: Float32Array, params: SphParams, initialVelocities?: Float32Array): void {
    this.count = initialPositions.length / 3;
    this.positions = initialPositions.slice();
    this.velocities = initialVelocities ? initialVelocities.slice() : new Float32Array(this.count * 3);
    this.densities = new Float32Array(this.count);
    this.pressures = new Float32Array(this.count);
    this.forces = new Float32Array(this.count * 3);
    this.advectionVelocities = new Float32Array(this.count * 3);
    this.grid = new SpatialGrid(params.smoothingRadius);
  }

  setCollisionField(field: CollisionField | null): void {
    this.collisionField = field as SignedDistanceField | null;
  }

  step(params: SphParams): void {
    if (!this.grid) {
      throw new Error("CpuSphBackend.step() called before init()");
    }

    this.grid.build(this.positions, this.count);
    this.computeDensityPressure(params);
    this.computeForces(params);
    this.computeAdhesionForce(params);
    this.integrateVelocity(params);
    this.applyWallFriction(params);
    this.computeXsphCorrection(params);
    this.integratePosition(params);
    this.enforceMeshCollision(params);
    this.enforceBoundary(params);
  }

  getPositions(): Float32Array {
    return this.positions;
  }

  getDiagnostics(): SphDiagnostics {
    let sumDensity = 0;
    let maxSpeedSq = 0;
    const min: Vec3Tuple = [Infinity, Infinity, Infinity];
    const max: Vec3Tuple = [-Infinity, -Infinity, -Infinity];

    for (let i = 0; i < this.count; i++) {
      sumDensity += this.densities[i]!;

      const vx = this.velocities[i * 3]!;
      const vy = this.velocities[i * 3 + 1]!;
      const vz = this.velocities[i * 3 + 2]!;
      const speedSq = vx * vx + vy * vy + vz * vz;
      if (speedSq > maxSpeedSq) maxSpeedSq = speedSq;

      for (let axis = 0; axis < 3; axis++) {
        const p = this.positions[i * 3 + axis]!;
        if (p < min[axis]!) min[axis] = p;
        if (p > max[axis]!) max[axis] = p;
      }
    }

    return {
      meanDensity: this.count > 0 ? sumDensity / this.count : 0,
      maxSpeed: Math.sqrt(maxSpeedSq),
      minPosition: min,
      maxPosition: max,
    };
  }

  dispose(): void {
    this.grid = null;
    this.collisionField = null;
  }

  private computeDensityPressure(params: SphParams): void {
    const grid = this.grid!;
    const h = params.smoothingRadius;
    const hSq = h * h;
    const poly6Coeff = poly6Coefficient(h);
    const scratch = this.neighborScratch;

    for (let i = 0; i < this.count; i++) {
      const px = this.positions[i * 3]!;
      const py = this.positions[i * 3 + 1]!;
      const pz = this.positions[i * 3 + 2]!;

      grid.queryNeighborCells(px, py, pz, scratch);

      let density = 0;
      for (let k = 0; k < scratch.length; k++) {
        const j = scratch[k]!;
        const dx = px - this.positions[j * 3]!;
        const dy = py - this.positions[j * 3 + 1]!;
        const dz = pz - this.positions[j * 3 + 2]!;
        const rSq = dx * dx + dy * dy + dz * dz;
        if (rSq >= hSq) continue;
        density += params.particleMass * poly6(rSq, h, poly6Coeff);
      }

      this.densities[i] = density;
      this.pressures[i] = taitPressure(density, params.restDensity, params.stiffness, params.gamma);
    }
  }

  private computeForces(params: SphParams): void {
    const grid = this.grid!;
    const h = params.smoothingRadius;
    const hSq = h * h;
    const spikyCoeff = spikyGradientCoefficient(h);
    const viscCoeff = viscosityLaplacianCoefficient(h);
    const cohesionCoeff = cohesionKernelCoefficient(h);
    const massSq = params.particleMass * params.particleMass;
    const scratch = this.neighborScratch;

    for (let i = 0; i < this.count; i++) {
      const px = this.positions[i * 3]!;
      const py = this.positions[i * 3 + 1]!;
      const pz = this.positions[i * 3 + 2]!;
      const pi = this.pressures[i]!;
      const densityI = this.densities[i]!;
      const vix = this.velocities[i * 3]!;
      const viy = this.velocities[i * 3 + 1]!;
      const viz = this.velocities[i * 3 + 2]!;

      grid.queryNeighborCells(px, py, pz, scratch);

      let fx = 0;
      let fy = 0;
      let fz = 0;

      for (let k = 0; k < scratch.length; k++) {
        const j = scratch[k]!;
        if (j === i) continue;

        const dx = px - this.positions[j * 3]!;
        const dy = py - this.positions[j * 3 + 1]!;
        const dz = pz - this.positions[j * 3 + 2]!;
        const rSq = dx * dx + dy * dy + dz * dz;
        if (rSq >= hSq || rSq <= 0) continue;

        const r = Math.sqrt(rSq);
        const densityJ = this.densities[j]!;

        // Pressure force (Müller symmetrized form): F_i += -m_j (p_i+p_j)/(2 rho_j) * gradW
        const gradMag = spikyGradientMagnitude(r, h, spikyCoeff);
        const pressureFactor = (-params.particleMass * (pi + this.pressures[j]!) * gradMag) / (2 * densityJ * r);
        fx += pressureFactor * dx;
        fy += pressureFactor * dy;
        fz += pressureFactor * dz;

        // Viscosity force: F_i += mu * m_j * (v_j - v_i)/rho_j * laplacianW
        const lap = viscosityLaplacian(r, h, viscCoeff);
        const viscFactor = (params.viscosity * params.particleMass * lap) / densityJ;
        fx += viscFactor * (this.velocities[j * 3]! - vix);
        fy += viscFactor * (this.velocities[j * 3 + 1]! - viy);
        fz += viscFactor * (this.velocities[j * 3 + 2]! - viz);

        // Surface tension cohesion (Akinci et al. 2013): pairwise force
        // along (xi-xj), density-normalized (2 rho0 / (rho_i+rho_j)) so it
        // doesn't over-attract in the lower-density free-surface region.
        // cohesionKernel is negative at short range and positive at longer
        // range within its support (see its own doc comment) — applying it
        // regardless of sign (not just when positive) is what makes this
        // self-stabilizing rather than only ever attractive.
        if (params.surfaceTensionCoefficient > 0) {
          const cohesion = cohesionKernel(r, h, cohesionCoeff);
          const densityNorm = (2 * params.restDensity) / (densityI + densityJ);
          const cohesionFactor = (-params.surfaceTensionCoefficient * massSq * cohesion * densityNorm) / r;
          fx += cohesionFactor * dx;
          fy += cohesionFactor * dy;
          fz += cohesionFactor * dz;
        }
      }

      this.forces[i * 3] = fx;
      this.forces[i * 3 + 1] = fy;
      this.forces[i * 3 + 2] = fz;
    }
  }

  /**
   * Adhesion/wetting force: pulls particles near the collision mesh toward
   * its surface, letting water cling to and run along a wall instead of
   * separating from it as soon as nothing is physically pushing it there
   * (mesh collision otherwise only ever pushes particles *out*, never in).
   *
   * This is a fluid-*solid* force, entirely separate from the fluid-*fluid*
   * surface-tension cohesion force in computeForces above — different
   * method, different kernel (adhesionKernel, not cohesionKernel), different
   * coefficient (SphParams.adhesionCoefficient, not
   * .surfaceTensionCoefficient), and it adds onto whatever computeForces
   * already produced rather than being folded into that loop, so the two
   * stay independently readable, tunable, and toggleable.
   *
   * Akinci et al. 2013 define adhesion the same way as cohesion: a pairwise
   * force against sampled boundary particles. This simulation's boundary is
   * a signed distance field, not a particle set, so there's no boundary
   * particle to pair against — adhesionKernel is a distance-based
   * approximation built for that instead (see its own doc comment). The
   * band it's evaluated over starts at the collision margin (where
   * enforceMeshCollision's push-out takes over) and extends one
   * smoothingRadius beyond, so this force and that push-out never act on
   * the same particle at the same time.
   */
  private computeAdhesionForce(params: SphParams): void {
    const field = this.collisionField;
    if (!field || params.adhesionCoefficient <= 0) return;

    const margin = params.smoothingRadius * 0.5;
    const outer = margin + params.smoothingRadius;

    for (let i = 0; i < this.count; i++) {
      const px = this.positions[i * 3]!;
      const py = this.positions[i * 3 + 1]!;
      const pz = this.positions[i * 3 + 2]!;

      const distance = sampleSignedDistance(field, px, py, pz);
      const pull = adhesionKernel(distance, margin, outer);
      if (pull <= 0) continue;

      const [nx, ny, nz] = sampleGradient(field, px, py, pz);
      const magnitude = params.adhesionCoefficient * params.particleMass * pull;

      // Pulls toward the surface: nx/ny/nz point outward, so subtract.
      this.forces[i * 3] = this.forces[i * 3]! - magnitude * nx;
      this.forces[i * 3 + 1] = this.forces[i * 3 + 1]! - magnitude * ny;
      this.forces[i * 3 + 2] = this.forces[i * 3 + 2]! - magnitude * nz;
    }
  }

  private integrateVelocity(params: SphParams): void {
    const dt = params.timeStep;
    const [gx, gy, gz] = params.gravity;
    const minDensity = 1e-6;
    const maxSpeedSq = params.maxSpeed * params.maxSpeed;

    for (let i = 0; i < this.count; i++) {
      const density = Math.max(this.densities[i]!, minDensity);

      const ax = this.forces[i * 3]! / density + gx;
      const ay = this.forces[i * 3 + 1]! / density + gy;
      const az = this.forces[i * 3 + 2]! / density + gz;

      let vx = this.velocities[i * 3]! + ax * dt;
      let vy = this.velocities[i * 3 + 1]! + ay * dt;
      let vz = this.velocities[i * 3 + 2]! + az * dt;

      // Safety valve, not normal behavior: see SphParams.maxSpeed.
      const speedSq = vx * vx + vy * vy + vz * vz;
      if (speedSq > maxSpeedSq) {
        const scale = params.maxSpeed / Math.sqrt(speedSq);
        vx *= scale;
        vy *= scale;
        vz *= scale;
      }

      this.velocities[i * 3] = vx;
      this.velocities[i * 3 + 1] = vy;
      this.velocities[i * 3 + 2] = vz;
    }
  }

  /**
   * Wall friction: damps the velocity component *tangential* to the
   * collision mesh for particles near it, over the same band as
   * computeAdhesionForce (reusing adhesionKernel purely as a generic
   * distance-based proximity weight, not because friction and adhesion are
   * the same effect — they stay entirely separate mechanisms, with
   * separate coefficients). Without this, mesh collision only ever
   * reflects the *normal* velocity component, leaving tangential motion
   * completely unaffected — which, once adhesion is holding water near a
   * wall, reads as it sliding down frictionlessly.
   *
   * This is a velocity correction (like XSPH), not a force: the damping
   * factor 1/(1 + k*weight*dt) is the same form as one step of implicit
   * (backward-Euler) exponential decay, which is unconditionally stable —
   * unlike an explicit drag *force* of the same strength, it can never
   * overshoot and reverse the tangential velocity no matter how large
   * wallFrictionCoefficient is, so there's no stability ceiling to
   * calibrate against here the way there was for cohesion/adhesion.
   */
  private applyWallFriction(params: SphParams): void {
    const field = this.collisionField;
    if (!field || params.wallFrictionCoefficient <= 0) return;

    const margin = params.smoothingRadius * 0.5;
    const outer = margin + params.smoothingRadius;
    const dt = params.timeStep;

    for (let i = 0; i < this.count; i++) {
      const px = this.positions[i * 3]!;
      const py = this.positions[i * 3 + 1]!;
      const pz = this.positions[i * 3 + 2]!;

      const distance = sampleSignedDistance(field, px, py, pz);
      const weight = adhesionKernel(distance, margin, outer);
      if (weight <= 0) continue;

      const [nx, ny, nz] = sampleGradient(field, px, py, pz);
      const vx = this.velocities[i * 3]!;
      const vy = this.velocities[i * 3 + 1]!;
      const vz = this.velocities[i * 3 + 2]!;

      const vn = vx * nx + vy * ny + vz * nz;
      const tx = vx - vn * nx;
      const ty = vy - vn * ny;
      const tz = vz - vn * nz;

      const damping = 1 / (1 + params.wallFrictionCoefficient * weight * dt);

      this.velocities[i * 3] = vn * nx + tx * damping;
      this.velocities[i * 3 + 1] = vn * ny + ty * damping;
      this.velocities[i * 3 + 2] = vn * nz + tz * damping;
    }
  }

  /**
   * XSPH velocity smoothing (Monaghan 1992): blends each particle's
   * velocity toward its local density-weighted neighborhood average,
   *
   *   v_i* = v_i + epsilon * sum_j (2 m_j)/(rho_i+rho_j) * (v_j - v_i) * W_poly6(r_ij, h)
   *
   * and uses only that result (advectionVelocities) to move the particle
   * next — the velocity stored in this.velocities, which pressure/viscosity
   * and next step's integration read, is left exactly as
   * integrateVelocity produced it. Keeping the two separate means XSPH acts
   * purely on advection (making neighboring particles move more coherently,
   * reducing jitter/clumping) without also quietly adding extra dissipation
   * to the force balance the way folding it into the stored velocity would.
   */
  private computeXsphCorrection(params: SphParams): void {
    const grid = this.grid!;
    const h = params.smoothingRadius;
    const hSq = h * h;
    const poly6Coeff = poly6Coefficient(h);
    const scratch = this.neighborScratch;
    const epsilon = params.xsphEpsilon;

    for (let i = 0; i < this.count; i++) {
      const px = this.positions[i * 3]!;
      const py = this.positions[i * 3 + 1]!;
      const pz = this.positions[i * 3 + 2]!;
      const vix = this.velocities[i * 3]!;
      const viy = this.velocities[i * 3 + 1]!;
      const viz = this.velocities[i * 3 + 2]!;
      const densityI = this.densities[i]!;

      grid.queryNeighborCells(px, py, pz, scratch);

      let cx = 0;
      let cy = 0;
      let cz = 0;

      for (let k = 0; k < scratch.length; k++) {
        const j = scratch[k]!;
        if (j === i) continue;

        const dx = px - this.positions[j * 3]!;
        const dy = py - this.positions[j * 3 + 1]!;
        const dz = pz - this.positions[j * 3 + 2]!;
        const rSq = dx * dx + dy * dy + dz * dz;
        if (rSq >= hSq) continue;

        const densityAvg = (densityI + this.densities[j]!) * 0.5;
        if (densityAvg <= 0) continue;
        const weight = (params.particleMass / densityAvg) * poly6(rSq, h, poly6Coeff);

        cx += weight * (this.velocities[j * 3]! - vix);
        cy += weight * (this.velocities[j * 3 + 1]! - viy);
        cz += weight * (this.velocities[j * 3 + 2]! - viz);
      }

      this.advectionVelocities[i * 3] = vix + epsilon * cx;
      this.advectionVelocities[i * 3 + 1] = viy + epsilon * cy;
      this.advectionVelocities[i * 3 + 2] = viz + epsilon * cz;
    }
  }

  private integratePosition(params: SphParams): void {
    const dt = params.timeStep;

    for (let i = 0; i < this.count; i++) {
      this.positions[i * 3] = this.positions[i * 3]! + this.advectionVelocities[i * 3]! * dt;
      this.positions[i * 3 + 1] = this.positions[i * 3 + 1]! + this.advectionVelocities[i * 3 + 1]! * dt;
      this.positions[i * 3 + 2] = this.positions[i * 3 + 2]! + this.advectionVelocities[i * 3 + 2]! * dt;
    }
  }

  /**
   * Pushes any particle penetrating the obstacle's signed-distance-field
   * surface back out along the surface normal, and reflects the inward
   * velocity component. `margin` is treated as an effective particle
   * radius: particles are kept at least that far outside the surface rather
   * than exactly at distance 0, so they don't visually sink into the mesh.
   */
  private enforceMeshCollision(params: SphParams): void {
    const field = this.collisionField;
    if (!field) return;

    const margin = params.smoothingRadius * 0.5;

    for (let i = 0; i < this.count; i++) {
      const px = this.positions[i * 3]!;
      const py = this.positions[i * 3 + 1]!;
      const pz = this.positions[i * 3 + 2]!;

      const distance = sampleSignedDistance(field, px, py, pz);
      if (distance >= margin) continue;

      const [nx, ny, nz] = sampleGradient(field, px, py, pz);
      const penetration = margin - distance;

      this.positions[i * 3] = px + nx * penetration;
      this.positions[i * 3 + 1] = py + ny * penetration;
      this.positions[i * 3 + 2] = pz + nz * penetration;

      const [vx, vy, vz] = reflectVelocityAgainstNormal(
        this.velocities[i * 3]!,
        this.velocities[i * 3 + 1]!,
        this.velocities[i * 3 + 2]!,
        nx,
        ny,
        nz,
        params.boundaryDamping
      );
      this.velocities[i * 3] = vx;
      this.velocities[i * 3 + 1] = vy;
      this.velocities[i * 3 + 2] = vz;
    }
  }

  private enforceBoundary(params: SphParams): void {
    const { domainMin, domainMax, boundaryDamping, deleteParticlesAtFloor } = params;
    const axisNormals: Vec3Tuple[] = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];

    // A plain for-loop doesn't work here once particles can be removed
    // mid-pass: removing index i swaps the last active particle into its
    // slot, which then still needs to be checked itself rather than
    // skipped, so i is only advanced when nothing was removed.
    let i = 0;
    while (i < this.count) {
      if (deleteParticlesAtFloor && this.positions[i * 3 + 1]! < domainMin[1]!) {
        this.removeParticleAt(i);
        continue;
      }

      for (let axis = 0; axis < 3; axis++) {
        const idx = i * 3 + axis;
        const pos = this.positions[idx]!;
        const min = domainMin[axis]!;
        const max = domainMax[axis]!;
        const [nx, ny, nz] = axisNormals[axis]!;

        if (pos < min) {
          this.positions[idx] = min;
        } else if (pos > max) {
          this.positions[idx] = max;
        } else {
          continue;
        }

        const outward = pos < min ? 1 : -1;
        const [vx, vy, vz] = reflectVelocityAgainstNormal(
          this.velocities[i * 3]!,
          this.velocities[i * 3 + 1]!,
          this.velocities[i * 3 + 2]!,
          nx * outward,
          ny * outward,
          nz * outward,
          boundaryDamping
        );
        this.velocities[i * 3] = vx;
        this.velocities[i * 3 + 1] = vy;
        this.velocities[i * 3 + 2] = vz;
      }

      i++;
    }
  }

  /** Removes particle i by swapping in the last active particle and shrinking the active count by one. */
  private removeParticleAt(i: number): void {
    const last = this.count - 1;
    if (i !== last) {
      for (let axis = 0; axis < 3; axis++) {
        this.positions[i * 3 + axis] = this.positions[last * 3 + axis]!;
        this.velocities[i * 3 + axis] = this.velocities[last * 3 + axis]!;
        this.forces[i * 3 + axis] = this.forces[last * 3 + axis]!;
      }
      this.densities[i] = this.densities[last]!;
      this.pressures[i] = this.pressures[last]!;
    }
    this.count--;
  }
}

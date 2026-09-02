import type { CollisionField, SphComputeBackend, SphDiagnostics, SphParams, Vec3Tuple } from "../../core/types";
import {
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
 * 8-stage pipeline (neighbor search -> density/pressure -> forces ->
 * velocity integration -> XSPH smoothing -> position integration -> mesh
 * collision -> domain boundary) declared by the SphComputeBackend contract,
 * using the shared kernel/collision formulas from core/kernels.ts and
 * core/collision.ts verbatim — this is the pairing a future GPU backend
 * must reproduce.
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
    this.integrateVelocity(params);
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
    const scratch = this.neighborScratch;

    for (let i = 0; i < this.count; i++) {
      const px = this.positions[i * 3]!;
      const py = this.positions[i * 3 + 1]!;
      const pz = this.positions[i * 3 + 2]!;
      const pi = this.pressures[i]!;
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
      }

      this.forces[i * 3] = fx;
      this.forces[i * 3 + 1] = fy;
      this.forces[i * 3 + 2] = fz;
    }
  }

  private integrateVelocity(params: SphParams): void {
    const dt = params.timeStep;
    const [gx, gy, gz] = params.gravity;
    const minDensity = 1e-6;

    for (let i = 0; i < this.count; i++) {
      const density = Math.max(this.densities[i]!, minDensity);

      const ax = this.forces[i * 3]! / density + gx;
      const ay = this.forces[i * 3 + 1]! / density + gy;
      const az = this.forces[i * 3 + 2]! / density + gz;

      this.velocities[i * 3] = this.velocities[i * 3]! + ax * dt;
      this.velocities[i * 3 + 1] = this.velocities[i * 3 + 1]! + ay * dt;
      this.velocities[i * 3 + 2] = this.velocities[i * 3 + 2]! + az * dt;
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

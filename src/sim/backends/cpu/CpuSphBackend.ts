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
 * 6-stage pipeline (neighbor search -> density/pressure -> forces ->
 * integrate -> mesh collision -> domain boundary) declared by the
 * SphComputeBackend contract, using the shared kernel/collision formulas
 * from core/kernels.ts and core/collision.ts verbatim — this is the pairing
 * a future GPU backend must reproduce.
 */
export class CpuSphBackend implements SphComputeBackend {
  private count = 0;
  private positions = new Float32Array(0);
  private velocities = new Float32Array(0);
  private densities = new Float32Array(0);
  private pressures = new Float32Array(0);
  private forces = new Float32Array(0);
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
    this.integrate(params);
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

  private integrate(params: SphParams): void {
    const dt = params.timeStep;
    const [gx, gy, gz] = params.gravity;
    const minDensity = 1e-6;

    for (let i = 0; i < this.count; i++) {
      const density = Math.max(this.densities[i]!, minDensity);

      const ax = this.forces[i * 3]! / density + gx;
      const ay = this.forces[i * 3 + 1]! / density + gy;
      const az = this.forces[i * 3 + 2]! / density + gz;

      const vx = this.velocities[i * 3]! + ax * dt;
      const vy = this.velocities[i * 3 + 1]! + ay * dt;
      const vz = this.velocities[i * 3 + 2]! + az * dt;

      this.velocities[i * 3] = vx;
      this.velocities[i * 3 + 1] = vy;
      this.velocities[i * 3 + 2] = vz;

      this.positions[i * 3] = this.positions[i * 3]! + vx * dt;
      this.positions[i * 3 + 1] = this.positions[i * 3 + 1]! + vy * dt;
      this.positions[i * 3 + 2] = this.positions[i * 3 + 2]! + vz * dt;
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
    const { domainMin, domainMax, boundaryDamping } = params;
    const axisNormals: Vec3Tuple[] = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];

    for (let i = 0; i < this.count; i++) {
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
    }
  }
}

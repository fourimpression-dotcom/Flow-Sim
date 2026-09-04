import type { CollisionField, SphComputeBackend, SphDiagnostics, SphParams } from "./core/types";

const MAX_SUBSTEPS_PER_UPDATE = 8;

/**
 * Backend-agnostic simulation orchestrator. Depends only on SphComputeBackend
 * (not on CpuSphBackend specifically), so swapping to a future GPU backend is
 * a one-line change at the call site that constructs this class.
 *
 * Runs physics at a fixed timestep (params.timeStep) using an accumulator, so
 * results are independent of the render frame rate. If a frame takes too
 * long to keep up, substeps are capped rather than spiraling — the
 * simulation runs in slow motion instead of freezing the UI.
 */
export class SphSimulation {
  private readonly backend: SphComputeBackend;
  private readonly params: SphParams;
  // The scenario-computed coefficients (see scenario.ts — both are derived
  // from particle spacing, not fixed constants), kept aside so toggling
  // either back on restores the right value rather than some arbitrary
  // default. Two separate fields for two separate forces — see
  // SphParams.surfaceTensionCoefficient / .adhesionCoefficient.
  private readonly baseSurfaceTensionCoefficient: number;
  private readonly baseAdhesionCoefficient: number;
  private readonly baseWallFrictionCoefficient: number;
  private accumulatedSeconds = 0;

  constructor(
    backend: SphComputeBackend,
    params: SphParams,
    initialPositions: Float32Array,
    initialVelocities?: Float32Array
  ) {
    this.backend = backend;
    this.params = params;
    this.baseSurfaceTensionCoefficient = params.surfaceTensionCoefficient;
    this.baseAdhesionCoefficient = params.adhesionCoefficient;
    this.baseWallFrictionCoefficient = params.wallFrictionCoefficient;
    this.backend.init(initialPositions, params, initialVelocities);
  }

  get particleCount(): number {
    return this.backend.particleCount;
  }

  setCollisionField(field: CollisionField | null): void {
    this.backend.setCollisionField(field);
  }

  /** Toggles whether particles that fall below the floor are deleted rather than bounced back in — takes effect on the next step. */
  setDeleteParticlesAtFloor(enabled: boolean): void {
    this.params.deleteParticlesAtFloor = enabled;
  }

  /** Sets the fluid's dynamic viscosity coefficient directly (not a multiplier — viscosity, unlike surface tension/adhesion, isn't spacing-derived) — takes effect on the next step. */
  setViscosity(value: number): void {
    this.params.viscosity = value;
  }

  /** Toggles surface-tension cohesion (fluid-fluid) on/off, restoring the scenario's own spacing-calibrated coefficient rather than a fixed value — takes effect on the next step. */
  setSurfaceTensionEnabled(enabled: boolean): void {
    this.params.surfaceTensionCoefficient = enabled ? this.baseSurfaceTensionCoefficient : 0;
  }

  /**
   * Scales wall adhesion (fluid-solid) strength — a separate force from
   * surface tension, see SphParams.adhesionCoefficient — as a multiplier on
   * the scenario's own spacing-calibrated coefficient (1 = as calibrated, 0
   * = off) rather than replacing it with a fixed value. Takes effect on the
   * next step.
   */
  setAdhesionStrength(multiplier: number): void {
    this.params.adhesionCoefficient = this.baseAdhesionCoefficient * multiplier;
  }

  /**
   * Scales wall-friction strength — a separate effect from adhesion, see
   * SphParams.wallFrictionCoefficient — as a multiplier on the scenario's
   * own damping rate (1 = as calibrated, 0 = off). Takes effect on the
   * next step.
   */
  setWallFrictionStrength(multiplier: number): void {
    this.params.wallFrictionCoefficient = this.baseWallFrictionCoefficient * multiplier;
  }

  /** Advances the simulation to cover `deltaSeconds` of wall-clock time. */
  update(deltaSeconds: number): void {
    this.accumulatedSeconds += deltaSeconds;

    let substeps = 0;
    while (this.accumulatedSeconds >= this.params.timeStep && substeps < MAX_SUBSTEPS_PER_UPDATE) {
      this.backend.step(this.params);
      this.accumulatedSeconds -= this.params.timeStep;
      substeps++;
    }
  }

  getPositions(): Float32Array {
    return this.backend.getPositions();
  }

  getDiagnostics(): SphDiagnostics {
    return this.backend.getDiagnostics();
  }

  dispose(): void {
    this.backend.dispose();
  }
}

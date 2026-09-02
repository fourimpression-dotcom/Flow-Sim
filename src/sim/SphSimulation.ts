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
  // The scenario-computed coefficient (see scenario.ts — it's derived from
  // particle spacing, not a fixed constant), kept aside so toggling surface
  // tension back on restores the right value rather than some arbitrary
  // default.
  private readonly baseSurfaceTensionCoefficient: number;
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

  /** Toggles surface-tension cohesion on/off, restoring the scenario's own spacing-calibrated coefficient rather than a fixed value — takes effect on the next step. */
  setSurfaceTensionEnabled(enabled: boolean): void {
    this.params.surfaceTensionCoefficient = enabled ? this.baseSurfaceTensionCoefficient : 0;
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

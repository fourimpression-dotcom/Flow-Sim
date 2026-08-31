// Physics-layer types. Deliberately free of any rendering-library (Three.js)
// or execution-backend (Worker/GPU) dependency, since this is the "shared
// spec" that both the CPU backend and the future GPU backend implement.

export type Vec3Tuple = [number, number, number];

/** Opaque, backend-specific representation of static collision geometry. */
export type CollisionField = unknown;

/**
 * Describes a block of water to release: where (center), how much (a box of
 * these per-axis dimensions), and which way it's moving at t=0. Used to
 * build scenario initial conditions from a user-specified source rather
 * than the fixed corner-of-the-domain default.
 */
export interface WaterBlockSource {
  center: Vec3Tuple;
  /** Per-axis side lengths [sizeX, sizeY, sizeZ] of the release region (m). */
  size: Vec3Tuple;
  /** Particle spacing (m) — set explicitly rather than derived from size. */
  spacing: number;
  /** Uniform initial velocity applied to every particle in the block (m/s). */
  initialVelocity: Vec3Tuple;
}

/** Physical + numerical parameters for a WCSPH (Tait equation of state) simulation. */
export interface SphParams {
  /** Mass per particle (kg). Same for all particles in this phase. */
  particleMass: number;
  /** Smoothing radius h (m). Kernel support radius. */
  smoothingRadius: number;
  /** Rest density rho0 (kg/m^3), e.g. 1000 for water. */
  restDensity: number;
  /** Tait equation-of-state stiffness B = rho0 * c^2 / gamma. */
  stiffness: number;
  /** Tait equation-of-state exponent (commonly 7). */
  gamma: number;
  /** Dynamic viscosity coefficient mu. */
  viscosity: number;
  /** Gravity acceleration vector (m/s^2), e.g. [0, -9.81, 0]. */
  gravity: Vec3Tuple;
  /** Fixed physics timestep (s). Chosen from a CFL-style bound on smoothingRadius/soundSpeed. */
  timeStep: number;
  /** Velocity restitution/damping factor applied on boundary collision, in [0, 1]. */
  boundaryDamping: number;
  /** Axis-aligned simulation domain (world units, m). */
  domainMin: Vec3Tuple;
  domainMax: Vec3Tuple;
}

/** Lightweight scalar summary of simulation state, for verification/monitoring. */
export interface SphDiagnostics {
  meanDensity: number;
  maxSpeed: number;
  minPosition: Vec3Tuple;
  maxPosition: Vec3Tuple;
}

/**
 * Contract that every compute backend (CPU today, GPU/TSL later) must satisfy.
 * SphSimulation depends only on this interface, never on a concrete backend —
 * swapping CPU for GPU later means injecting a different implementation here,
 * with no changes to simulation orchestration, scenario setup, or rendering.
 *
 * The pipeline a step() call must perform, in order, is fixed by this contract:
 *   1. neighbor search (rebuild spatial structure for current positions)
 *   2. density + pressure (Tait EOS)
 *   3. force accumulation (pressure force + viscosity force)
 *   4. integration (semi-implicit Euler)
 *   5. mesh obstacle collision (signed-distance-field boundary, if set)
 *   6. domain AABB boundary handling
 * A GPU backend must reproduce this same stage sequence and the same
 * per-stage math (see core/kernels.ts, core/collision.ts) — that is what
 * "shared physics model" means here, since GPU compute code cannot
 * literally execute this CPU code.
 */
export interface SphComputeBackend {
  readonly particleCount: number;

  /**
   * Allocates buffers and seeds initial particle positions. Velocities
   * start at zero unless initialVelocities (same flat xyz-per-particle
   * layout as initialPositions) is provided.
   */
  init(initialPositions: Float32Array, params: SphParams, initialVelocities?: Float32Array): void;

  /**
   * Sets (or clears, with null) the solid obstacle particles collide with,
   * in addition to the domain AABB walls. The representation is
   * backend-specific (the CPU backend expects a SignedDistanceField; see
   * geometry/signedDistanceField.ts) — typed as CollisionField (opaque) here
   * so SphSimulation never needs to know which backend, or which collision
   * representation, is active.
   */
  setCollisionField(field: CollisionField | null): void;

  /** Advances the simulation by exactly params.timeStep seconds. */
  step(params: SphParams): void;

  /**
   * Returns the current particle positions as a flat [x0,y0,z0,x1,y1,z1,...] array.
   * Backends may return a live reference (mutated in place each step) rather
   * than a copy — callers that need a snapshot must copy it themselves.
   */
  getPositions(): Float32Array;

  getDiagnostics(): SphDiagnostics;

  dispose(): void;
}

import type { SphParams, Vec3Tuple, WaterBlockSource } from "./types";

export interface SphScenario {
  params: SphParams;
  /** Flat [x0,y0,z0,x1,y1,z1,...] initial particle positions. */
  initialPositions: Float32Array;
  /** Flat [vx0,vy0,vz0,...] initial particle velocities, same layout as initialPositions. */
  initialVelocities: Float32Array;
}

export interface ObstacleBounds {
  min: Vec3Tuple;
  max: Vec3Tuple;
}

export interface CreateDamBreakScenarioOptions {
  obstacle?: ObstacleBounds;
  /**
   * Explicit water release region + initial velocity, overriding the
   * default "block dropped in a corner of the domain" placement. The domain
   * is still auto-sized to fit both this and the obstacle (if any), plus
   * room for the water to travel given its initial velocity.
   */
  waterSource?: WaterBlockSource;
}

const MAX_PARTICLES = 20000;

// Surface-tension cohesion (see core/kernels.ts's cohesionKernel) scales
// extremely steeply with particle spacing: working through the force
// formula in CpuSphBackend.computeForces, the resulting acceleration is
// proportional to surfaceTensionCoefficient * spacing^3 for a fixed
// coefficient. A single fixed coefficient that looked fine at one spacing
// was unstable at a coarser one and imperceptible at a finer one when
// tested — so the coefficient itself is derived per scenario as
// SURFACE_TENSION_BASE / spacing^3, keeping the actual physical effect
// roughly constant across the full range of particle spacings this app
// produces. SURFACE_TENSION_BASE itself was calibrated empirically (swept
// over several orders of magnitude against this file's own dam-break
// scenario, checking density/pressure/energy stability and no mesh
// penetration) to a value with a wide safety margin below where
// instability actually starts.
const SURFACE_TENSION_BASE = 0.93;

/**
 * SPH release scenario: a block of water given an initial position, size,
 * and velocity, that falls/flies from t=0 under gravity (and, once
 * released, ordinary SPH pressure/viscosity forces). With no obstacle, it's
 * a plain box (used to sanity-check the solver itself). With an obstacle
 * bounding box (a loaded STEP shape) and no explicit waterSource, the
 * classic "block dropped in a corner, falls onto the shape" dam-break
 * default is used.
 */
export function createDamBreakScenario(options: CreateDamBreakScenarioOptions = {}): SphScenario {
  const restDensity = 1000; // kg/m^3 (water)
  const { obstacle, waterSource } = options;

  let domainMin: Vec3Tuple;
  let domainMax: Vec3Tuple;
  let blockMin: Vec3Tuple;
  let blockMax: Vec3Tuple;
  let spacing: number;
  let initialVelocity: Vec3Tuple = [0, 0, 0];

  if (waterSource) {
    const halfX = waterSource.size[0] / 2;
    const halfY = waterSource.size[1] / 2;
    const halfZ = waterSource.size[2] / 2;
    blockMin = [waterSource.center[0] - halfX, waterSource.center[1] - halfY, waterSource.center[2] - halfZ];
    blockMax = [waterSource.center[0] + halfX, waterSource.center[1] + halfY, waterSource.center[2] + halfZ];
    initialVelocity = waterSource.initialVelocity;
    spacing = Math.max(waterSource.spacing, 1e-6);

    const domain = computeDomainAroundSource(waterSource, obstacle);
    domainMin = domain.min;
    domainMax = domain.max;
  } else if (obstacle) {
    const size: Vec3Tuple = [
      obstacle.max[0] - obstacle.min[0],
      obstacle.max[1] - obstacle.min[1],
      obstacle.max[2] - obstacle.min[2],
    ];
    const maxDim = Math.max(size[0], size[1], size[2], 1e-6);
    const pad = maxDim * 0.6;
    // The floor sits just under the model, not under whichever axis happens
    // to be largest: a wide/flat model (e.g. 500x20x300mm) has maxDim from
    // its width, and padding *below* by 60% of that would put the floor 15x
    // the model's own height beneath it. Based on the model's own Y extent instead.
    const padBelow = Math.max(size[1] * 0.1, 1e-6);

    domainMin = [obstacle.min[0] - pad, obstacle.min[1] - padBelow, obstacle.min[2] - pad];
    // Extra headroom above (+Y) so the water block starts clear of the obstacle and falls onto it.
    domainMax = [obstacle.max[0] + pad, obstacle.max[1] + pad * 2.2, obstacle.max[2] + pad];

    spacing = maxDim / 12;

    // Centered over the obstacle in X/Z (not anchored to its min corner) so
    // the water falls onto the middle of the shape by default.
    const obstacleCenterX = (obstacle.min[0] + obstacle.max[0]) / 2;
    const obstacleCenterZ = (obstacle.min[2] + obstacle.max[2]) / 2;
    const blockSize = maxDim * 0.5;
    blockMin = [obstacleCenterX - blockSize / 2, domainMax[1] - blockSize - spacing, obstacleCenterZ - blockSize / 2];
    blockMax = [obstacleCenterX + blockSize / 2, domainMax[1] - spacing, obstacleCenterZ + blockSize / 2];
  } else {
    spacing = 0.04;
    domainMin = [0, 0, 0];
    domainMax = [1, 1, 0.5];
    blockMin = [spacing, spacing, spacing];
    blockMax = [0.35, 0.65, domainMax[2] - spacing];
  }

  spacing = clampSpacingForParticleBudget(blockMin, blockMax, spacing);
  const smoothingRadius = spacing * 1.3; // h; ~1.2-1.5x spacing gives ~30-40 neighbors in 3D
  const surfaceTensionCoefficient = SURFACE_TENSION_BASE / (spacing * spacing * spacing);

  const positions: number[] = [];
  for (let x = blockMin[0]; x <= blockMax[0]; x += spacing) {
    for (let y = blockMin[1]; y <= blockMax[1]; y += spacing) {
      for (let z = blockMin[2]; z <= blockMax[2]; z += spacing) {
        positions.push(x, y, z);
      }
    }
  }

  const particleCount = positions.length / 3;
  const velocities = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount; i++) {
    velocities[i * 3] = initialVelocity[0];
    velocities[i * 3 + 1] = initialVelocity[1];
    velocities[i * 3 + 2] = initialVelocity[2];
  }

  const particleMass = restDensity * Math.pow(spacing, 3);

  // Artificial speed of sound: chosen ~10x the expected max velocity so
  // density variation stays within ~1% (standard weakly-compressible SPH
  // practice). Accounts for both free-fall across the domain and any
  // initial launch speed, whichever is larger.
  const domainHeight = domainMax[1] - domainMin[1];
  const fallSpeed = Math.sqrt(2 * 9.81 * domainHeight);
  const launchSpeed = Math.hypot(initialVelocity[0], initialVelocity[1], initialVelocity[2]);
  const expectedMaxSpeed = Math.max(fallSpeed, launchSpeed) * 1.2;
  const soundSpeed = Math.max(10, 10 * expectedMaxSpeed);
  const gamma = 7;
  const stiffness = (restDensity * soundSpeed * soundSpeed) / gamma;

  // CFL-style bound: dt <~ 0.4 * h / soundSpeed. Kept as a fixed physics
  // timestep; SphSimulation substeps to keep pace with real time.
  const timeStep = Math.min(0.0004, (0.4 * smoothingRadius) / soundSpeed);

  const params: SphParams = {
    particleMass,
    smoothingRadius,
    restDensity,
    stiffness,
    gamma,
    viscosity: 0.6,
    // Mild smoothing: enough to visibly reduce jitter without noticeably
    // over-damping the flow (Monaghan's original paper allows up to ~0.5;
    // real-time SPH tends to use less).
    xsphEpsilon: 0.1,
    surfaceTensionCoefficient,
    gravity: [0, -9.81, 0],
    timeStep,
    boundaryDamping: 0.5,
    domainMin,
    domainMax,
    // Off by default (sealed tank); toggled at runtime via SphSimulation.setDeleteParticlesAtFloor.
    deleteParticlesAtFloor: false,
  };

  return { params, initialPositions: Float32Array.from(positions), initialVelocities: velocities };
}

/**
 * Computes the WaterBlockSource equivalent to createDamBreakScenario's own
 * default placement (used when no explicit waterSource is given) — for
 * pre-filling a "customize the water source" form with the same starting
 * point the simulation would otherwise use by default. Mirrors the
 * obstacle/no-obstacle formulas above; keep the two in sync if either
 * changes.
 */
export function computeDefaultWaterSource(obstacle?: ObstacleBounds): WaterBlockSource {
  if (obstacle) {
    const size: Vec3Tuple = [
      obstacle.max[0] - obstacle.min[0],
      obstacle.max[1] - obstacle.min[1],
      obstacle.max[2] - obstacle.min[2],
    ];
    const maxDim = Math.max(size[0], size[1], size[2], 1e-6);
    const pad = maxDim * 0.6;
    const domainMaxY = obstacle.max[1] + pad * 2.2;
    const spacing = maxDim / 12;
    const blockSize = maxDim * 0.5;
    const obstacleCenterX = (obstacle.min[0] + obstacle.max[0]) / 2;
    const obstacleCenterZ = (obstacle.min[2] + obstacle.max[2]) / 2;

    return {
      center: [obstacleCenterX, domainMaxY - spacing - blockSize / 2, obstacleCenterZ],
      size: [blockSize, blockSize, blockSize],
      spacing,
      initialVelocity: [0, 0, 0],
    };
  }

  const spacing = 0.04;
  const domainMax: Vec3Tuple = [1, 1, 0.5];
  const blockMin: Vec3Tuple = [spacing, spacing, spacing];
  const blockMax: Vec3Tuple = [0.35, 0.65, domainMax[2] - spacing];

  return {
    center: [
      (blockMin[0] + blockMax[0]) / 2,
      (blockMin[1] + blockMax[1]) / 2,
      (blockMin[2] + blockMax[2]) / 2,
    ],
    size: [blockMax[0] - blockMin[0], blockMax[1] - blockMin[1], blockMax[2] - blockMin[2]],
    spacing,
    initialVelocity: [0, 0, 0],
  };
}

/**
 * Builds a domain that contains the water source's release region, the
 * obstacle (if any), and enough room in the direction of initialVelocity
 * for the water to actually travel before hitting a wall, then pads that
 * combined region on every side.
 */
function computeDomainAroundSource(
  source: WaterBlockSource,
  obstacle: ObstacleBounds | undefined
): { min: Vec3Tuple; max: Vec3Tuple } {
  const halfX = source.size[0] / 2;
  const halfY = source.size[1] / 2;
  const halfZ = source.size[2] / 2;
  const min: Vec3Tuple = [source.center[0] - halfX, source.center[1] - halfY, source.center[2] - halfZ];
  const max: Vec3Tuple = [source.center[0] + halfX, source.center[1] + halfY, source.center[2] + halfZ];
  const include = (p: Vec3Tuple): void => {
    for (let axis = 0; axis < 3; axis++) {
      if (p[axis]! < min[axis]!) min[axis] = p[axis]!;
      if (p[axis]! > max[axis]!) max[axis] = p[axis]!;
    }
  };

  if (obstacle) {
    include(obstacle.min);
    include(obstacle.max);
  }

  const maxSourceDim = Math.max(source.size[0], source.size[1], source.size[2], 1e-6);
  const speed = Math.hypot(...source.initialVelocity);
  const travelDistance = Math.max(maxSourceDim * 4, speed * 0.4);
  const direction: Vec3Tuple = speed > 1e-6
    ? [source.initialVelocity[0] / speed, source.initialVelocity[1] / speed, source.initialVelocity[2] / speed]
    : [0, -1, 0];
  include([
    source.center[0] + direction[0] * travelDistance,
    source.center[1] + direction[1] * travelDistance,
    source.center[2] + direction[2] * travelDistance,
  ]);

  const size: Vec3Tuple = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const pad = Math.max(size[0], size[1], size[2], 1e-6) * 0.3;

  return {
    min: [min[0] - pad, min[1] - pad, min[2] - pad],
    max: [max[0] + pad, max[1] + pad, max[2] + pad],
  };
}

/** Scales spacing up (fewer, larger particles) if the naive count would exceed MAX_PARTICLES. */
function clampSpacingForParticleBudget(blockMin: Vec3Tuple, blockMax: Vec3Tuple, spacing: number): number {
  const sizeX = Math.max(blockMax[0] - blockMin[0], 0);
  const sizeY = Math.max(blockMax[1] - blockMin[1], 0);
  const sizeZ = Math.max(blockMax[2] - blockMin[2], 0);

  const estimatedCount = (sizeX / spacing + 1) * (sizeY / spacing + 1) * (sizeZ / spacing + 1);
  if (estimatedCount <= MAX_PARTICLES) return spacing;

  const scale = Math.cbrt(estimatedCount / MAX_PARTICLES);
  return spacing * scale;
}

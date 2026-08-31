import type { Vec3Tuple } from "../core/types";

/**
 * A dense voxel grid of signed distances to a triangle mesh surface
 * (negative = inside the solid, positive = outside). This is the "shared
 * data" a mesh-collision boundary stage reads from — the CPU backend samples
 * it with the trilinear interpolation below; a future GPU backend would
 * upload the same `distances` array as a 3D texture and use hardware
 * trilinear sampling to do the mathematically identical thing.
 */
export interface SignedDistanceField {
  /** World-space position of grid point (0,0,0). */
  origin: Vec3Tuple;
  /** World-space size of one grid cell (cubic cells). */
  cellSize: number;
  /** Grid point counts [nx, ny, nz]. Cell count is (nx-1)*(ny-1)*(nz-1). */
  dims: Vec3Tuple;
  /** Flat array of length nx*ny*nz, signed distance at each grid point. */
  distances: Float32Array;
}

function linearIndex(field: SignedDistanceField, ix: number, iy: number, iz: number): number {
  return ix + iy * field.dims[0] + iz * field.dims[0] * field.dims[1];
}

/** Large sentinel distance for points outside the grid — treated as "far away, definitely outside". */
export const SDF_OUTSIDE_DISTANCE = 1e6;

/**
 * Trilinear-interpolated signed distance at an arbitrary world-space point.
 * Points outside the grid bounds return SDF_OUTSIDE_DISTANCE.
 */
export function sampleSignedDistance(field: SignedDistanceField, x: number, y: number, z: number): number {
  const { origin, cellSize, dims, distances } = field;

  const fx = (x - origin[0]) / cellSize;
  const fy = (y - origin[1]) / cellSize;
  const fz = (z - origin[2]) / cellSize;

  const ix = Math.floor(fx);
  const iy = Math.floor(fy);
  const iz = Math.floor(fz);

  if (ix < 0 || iy < 0 || iz < 0 || ix >= dims[0] - 1 || iy >= dims[1] - 1 || iz >= dims[2] - 1) {
    return SDF_OUTSIDE_DISTANCE;
  }

  const tx = fx - ix;
  const ty = fy - iy;
  const tz = fz - iz;

  const c000 = distances[linearIndex(field, ix, iy, iz)]!;
  const c100 = distances[linearIndex(field, ix + 1, iy, iz)]!;
  const c010 = distances[linearIndex(field, ix, iy + 1, iz)]!;
  const c110 = distances[linearIndex(field, ix + 1, iy + 1, iz)]!;
  const c001 = distances[linearIndex(field, ix, iy, iz + 1)]!;
  const c101 = distances[linearIndex(field, ix + 1, iy, iz + 1)]!;
  const c011 = distances[linearIndex(field, ix, iy + 1, iz + 1)]!;
  const c111 = distances[linearIndex(field, ix + 1, iy + 1, iz + 1)]!;

  const c00 = c000 + (c100 - c000) * tx;
  const c10 = c010 + (c110 - c010) * tx;
  const c01 = c001 + (c101 - c001) * tx;
  const c11 = c011 + (c111 - c011) * tx;

  const c0 = c00 + (c10 - c00) * ty;
  const c1 = c01 + (c11 - c01) * ty;

  return c0 + (c1 - c0) * tz;
}

/**
 * Outward surface normal at a point, estimated as the normalized gradient of
 * the signed distance field via central differences.
 */
export function sampleGradient(field: SignedDistanceField, x: number, y: number, z: number): Vec3Tuple {
  const eps = field.cellSize * 0.5;

  const dx = sampleSignedDistance(field, x + eps, y, z) - sampleSignedDistance(field, x - eps, y, z);
  const dy = sampleSignedDistance(field, x, y + eps, z) - sampleSignedDistance(field, x, y - eps, z);
  const dz = sampleSignedDistance(field, x, y, z + eps) - sampleSignedDistance(field, x, y, z - eps);

  const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (length < 1e-12) {
    return [0, 1, 0]; // degenerate gradient (e.g. deep inside/outside); arbitrary but stable fallback
  }
  return [dx / length, dy / length, dz / length];
}

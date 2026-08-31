import type { Vec3Tuple } from "../core/types";
import type { SignedDistanceField } from "./signedDistanceField";
import { closestPointOnTriangle } from "./triangleDistance";

export interface TriangleSoup {
  /** Flat [x0,y0,z0,x1,y1,z1,...] vertex positions. */
  positions: Float32Array;
  /** Triangle vertex indices, 3 per triangle. */
  indices: Uint32Array;
}

export interface BuildSdfOptions {
  /** Extra world-space margin added around the mesh bounding box. */
  padding?: number;
  /** Target grid cell count along the mesh's longest bounding-box axis. */
  resolution?: number;
}

/**
 * Voxelizes a triangle mesh into a signed distance field: for each grid
 * point, find the closest point on any triangle (accelerated by bucketing
 * triangles into a uniform grid — brute-force over all triangles per grid
 * point would be O(gridPoints * triangleCount), too slow for real meshes),
 * and sign the distance using that triangle's face normal. This assumes a
 * reasonably watertight mesh with consistent (outward-facing) winding, which
 * is what OCCT/STEP tessellation produces for solids.
 */
export function buildSignedDistanceField(mesh: TriangleSoup, options: BuildSdfOptions = {}): SignedDistanceField {
  const padding = options.padding ?? 0;
  const resolution = Math.max(4, options.resolution ?? 32);

  const bounds = computeBounds(mesh.positions);
  const origin: Vec3Tuple = [bounds.min[0] - padding, bounds.min[1] - padding, bounds.min[2] - padding];
  const extent: Vec3Tuple = [
    bounds.max[0] - bounds.min[0] + 2 * padding,
    bounds.max[1] - bounds.min[1] + 2 * padding,
    bounds.max[2] - bounds.min[2] + 2 * padding,
  ];

  const maxDim = Math.max(extent[0], extent[1], extent[2], 1e-6);
  const cellSize = maxDim / resolution;

  const dims: Vec3Tuple = [
    Math.max(3, Math.ceil(extent[0] / cellSize) + 1),
    Math.max(3, Math.ceil(extent[1] / cellSize) + 1),
    Math.max(3, Math.ceil(extent[2] / cellSize) + 1),
  ];

  const triangleCount = mesh.indices.length / 3;
  const triangleNormals = computeTriangleNormals(mesh, triangleCount);
  const buckets = bucketTriangles(mesh, triangleCount, origin, cellSize, dims);

  const [nx, ny, nz] = dims;
  const distances = new Float32Array(nx * ny * nz);
  const closest: [number, number, number] = [0, 0, 0];
  const candidateSet = new Set<number>();

  for (let iz = 0; iz < nz; iz++) {
    const wz = origin[2] + iz * cellSize;
    for (let iy = 0; iy < ny; iy++) {
      const wy = origin[1] + iy * cellSize;
      for (let ix = 0; ix < nx; ix++) {
        const wx = origin[0] + ix * cellSize;

        collectNearbyTriangles(buckets, dims, ix, iy, iz, triangleCount, candidateSet);

        let bestDistSq = Infinity;
        let bestTri = -1;
        let bestX = wx;
        let bestY = wy;
        let bestZ = wz;

        for (const tri of candidateSet) {
          const i0 = mesh.indices[tri * 3]!;
          const i1 = mesh.indices[tri * 3 + 1]!;
          const i2 = mesh.indices[tri * 3 + 2]!;

          closestPointOnTriangle(
            wx,
            wy,
            wz,
            mesh.positions[i0 * 3]!,
            mesh.positions[i0 * 3 + 1]!,
            mesh.positions[i0 * 3 + 2]!,
            mesh.positions[i1 * 3]!,
            mesh.positions[i1 * 3 + 1]!,
            mesh.positions[i1 * 3 + 2]!,
            mesh.positions[i2 * 3]!,
            mesh.positions[i2 * 3 + 1]!,
            mesh.positions[i2 * 3 + 2]!,
            closest
          );

          const dx = wx - closest[0];
          const dy = wy - closest[1];
          const dz = wz - closest[2];
          const distSq = dx * dx + dy * dy + dz * dz;
          if (distSq < bestDistSq) {
            bestDistSq = distSq;
            bestTri = tri;
            bestX = closest[0];
            bestY = closest[1];
            bestZ = closest[2];
          }
        }

        const idx = ix + iy * nx + iz * nx * ny;
        if (bestTri < 0) {
          // No triangle found within the search radius (mesh has a large gap
          // relative to grid resolution). Treat as far outside rather than
          // leaving an undefined value.
          distances[idx] = 1e6;
          continue;
        }

        const dist = Math.sqrt(bestDistSq);
        const nx3 = triangleNormals[bestTri * 3]!;
        const ny3 = triangleNormals[bestTri * 3 + 1]!;
        const nz3 = triangleNormals[bestTri * 3 + 2]!;
        const sign = (wx - bestX) * nx3 + (wy - bestY) * ny3 + (wz - bestZ) * nz3 >= 0 ? 1 : -1;
        distances[idx] = sign * dist;
      }
    }
  }

  return { origin, cellSize, dims, distances };
}

function computeBounds(positions: Float32Array): { min: Vec3Tuple; max: Vec3Tuple } {
  const min: Vec3Tuple = [Infinity, Infinity, Infinity];
  const max: Vec3Tuple = [-Infinity, -Infinity, -Infinity];
  const vertexCount = positions.length / 3;
  for (let i = 0; i < vertexCount; i++) {
    for (let axis = 0; axis < 3; axis++) {
      const v = positions[i * 3 + axis]!;
      if (v < min[axis]!) min[axis] = v;
      if (v > max[axis]!) max[axis] = v;
    }
  }
  return { min, max };
}

function computeTriangleNormals(mesh: TriangleSoup, triangleCount: number): Float32Array {
  const normals = new Float32Array(triangleCount * 3);
  for (let t = 0; t < triangleCount; t++) {
    const i0 = mesh.indices[t * 3]!;
    const i1 = mesh.indices[t * 3 + 1]!;
    const i2 = mesh.indices[t * 3 + 2]!;

    const ax = mesh.positions[i0 * 3]!;
    const ay = mesh.positions[i0 * 3 + 1]!;
    const az = mesh.positions[i0 * 3 + 2]!;
    const abx = mesh.positions[i1 * 3]! - ax;
    const aby = mesh.positions[i1 * 3 + 1]! - ay;
    const abz = mesh.positions[i1 * 3 + 2]! - az;
    const acx = mesh.positions[i2 * 3]! - ax;
    const acy = mesh.positions[i2 * 3 + 1]! - ay;
    const acz = mesh.positions[i2 * 3 + 2]! - az;

    let cx = aby * acz - abz * acy;
    let cy = abz * acx - abx * acz;
    let cz = abx * acy - aby * acx;
    const len = Math.sqrt(cx * cx + cy * cy + cz * cz) || 1;
    cx /= len;
    cy /= len;
    cz /= len;

    normals[t * 3] = cx;
    normals[t * 3 + 1] = cy;
    normals[t * 3 + 2] = cz;
  }
  return normals;
}

/** Buckets each triangle into every grid cell its (padded) AABB overlaps. */
function bucketTriangles(
  mesh: TriangleSoup,
  triangleCount: number,
  origin: Vec3Tuple,
  cellSize: number,
  dims: Vec3Tuple
): number[][] {
  const [nx, ny, nz] = dims;
  const cellCountX = Math.max(1, nx - 1);
  const cellCountY = Math.max(1, ny - 1);
  const cellCountZ = Math.max(1, nz - 1);
  const buckets: number[][] = new Array(cellCountX * cellCountY * cellCountZ);
  for (let i = 0; i < buckets.length; i++) buckets[i] = [];

  const cellIndex = (x: number, y: number, z: number): [number, number, number] => [
    clamp(Math.floor((x - origin[0]) / cellSize), 0, cellCountX - 1),
    clamp(Math.floor((y - origin[1]) / cellSize), 0, cellCountY - 1),
    clamp(Math.floor((z - origin[2]) / cellSize), 0, cellCountZ - 1),
  ];

  for (let t = 0; t < triangleCount; t++) {
    const i0 = mesh.indices[t * 3]!;
    const i1 = mesh.indices[t * 3 + 1]!;
    const i2 = mesh.indices[t * 3 + 2]!;

    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (const vi of [i0, i1, i2]) {
      const vx = mesh.positions[vi * 3]!;
      const vy = mesh.positions[vi * 3 + 1]!;
      const vz = mesh.positions[vi * 3 + 2]!;
      if (vx < minX) minX = vx;
      if (vy < minY) minY = vy;
      if (vz < minZ) minZ = vz;
      if (vx > maxX) maxX = vx;
      if (vy > maxY) maxY = vy;
      if (vz > maxZ) maxZ = vz;
    }

    const [cx0, cy0, cz0] = cellIndex(minX, minY, minZ);
    const [cx1, cy1, cz1] = cellIndex(maxX, maxY, maxZ);

    for (let cz = cz0; cz <= cz1; cz++) {
      for (let cy = cy0; cy <= cy1; cy++) {
        for (let cx = cx0; cx <= cx1; cx++) {
          buckets[cx + cy * cellCountX + cz * cellCountX * cellCountY]!.push(t);
        }
      }
    }
  }

  return buckets;
}

/**
 * Expands a search radius outward from grid point (ix,iy,iz) over the
 * triangle-bucket grid (one smaller than the point grid in each dimension)
 * until at least one candidate triangle is found, then stops (the true
 * nearest triangle may be one cell further out in a rare edge case, but
 * closestPointOnTriangle over the found candidates already gives a very
 * good — not necessarily perfectly exact — distance for boundary purposes).
 */
function collectNearbyTriangles(
  buckets: number[][],
  dims: Vec3Tuple,
  ix: number,
  iy: number,
  iz: number,
  triangleCount: number,
  out: Set<number>
): void {
  out.clear();
  const cellCountX = Math.max(1, dims[0] - 1);
  const cellCountY = Math.max(1, dims[1] - 1);
  const cellCountZ = Math.max(1, dims[2] - 1);
  const cx = clamp(ix, 0, cellCountX - 1);
  const cy = clamp(iy, 0, cellCountY - 1);
  const cz = clamp(iz, 0, cellCountZ - 1);

  const maxRadius = Math.max(cellCountX, cellCountY, cellCountZ);
  let firstHitRadius = -1;

  for (let radius = 1; radius <= maxRadius; radius++) {
    for (let dz = -radius; dz <= radius; dz++) {
      const z = cz + dz;
      if (z < 0 || z >= cellCountZ) continue;
      for (let dy = -radius; dy <= radius; dy++) {
        const y = cy + dy;
        if (y < 0 || y >= cellCountY) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const x = cx + dx;
          if (x < 0 || x >= cellCountX) continue;
          // Only visit the outer shell of this radius (inner cells were already visited at smaller radii).
          if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== radius) continue;
          const bucket = buckets[x + y * cellCountX + z * cellCountX * cellCountY]!;
          for (const tri of bucket) out.add(tri);
        }
      }
    }

    if (out.size > 0 && firstHitRadius < 0) {
      firstHitRadius = radius;
    }
    // Search exactly one ring beyond the first hit: a shell being non-empty
    // only means *some* triangle is that close, not that it's the nearest
    // one overall — the true closest triangle can be bucketed one ring out.
    if (firstHitRadius >= 0 && radius >= firstHitRadius + 1) {
      return;
    }
  }

  if (out.size === 0) {
    // Pathological fallback: mesh has a gap larger than the whole grid. Check everything.
    for (let t = 0; t < triangleCount; t++) out.add(t);
  }
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

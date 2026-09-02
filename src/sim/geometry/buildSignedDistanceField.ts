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
  /**
   * World-space distance beyond which an exact nearest-triangle distance
   * stops being computed (a large sentinel magnitude is used instead, sign
   * still correct via ray-parity). Collision code only ever compares the
   * sampled distance against a small margin, so points already known to be
   * farther than that don't need an exact value — but without a cap, a grid
   * point deep inside or far outside the mesh forces the nearest-triangle
   * search to keep expanding ring by ring across most of the grid before
   * giving up, which is what actually made high-resolution builds so slow.
   * Omit for the old unbounded-search behavior.
   */
  maxSearchDistance?: number;
}

/**
 * Voxelizes a triangle mesh into a signed distance field: for each grid
 * point, find the closest point on any triangle (accelerated by bucketing
 * triangles into a uniform grid — brute-force over all triangles per grid
 * point would be O(gridPoints * triangleCount), too slow for real meshes)
 * for the distance magnitude, and sign it via ray-parity (even-odd rule)
 * against the mesh rather than by trusting any triangle's face normal —
 * see collectRowCrossings for why. This assumes a reasonably watertight
 * mesh, which is what OCCT/STEP tessellation produces for solids.
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
  const buckets = bucketTriangles(mesh, triangleCount, origin, cellSize, dims);
  const searchRadiusCap =
    options.maxSearchDistance !== undefined
      ? Math.max(2, Math.ceil(options.maxSearchDistance / cellSize) + 2)
      : undefined;

  const [nx, ny, nz] = dims;
  const cellCountX = Math.max(1, nx - 1);
  const cellCountY = Math.max(1, ny - 1);
  const cellCountZ = Math.max(1, nz - 1);
  const distances = new Float32Array(nx * ny * nz);
  const closest: [number, number, number] = [0, 0, 0];
  const candidateSet = new Set<number>();
  // Ray origin sits one full cell below the grid's own x-range, so every
  // grid point's +X ray starts strictly outside the mesh.
  const rayOriginX = origin[0] - cellSize;

  // With a search cap, most grid points (anything farther from the surface
  // than maxSearchDistance — typically the bulk of the grid's volume) would
  // still spend up to O(searchRadiusCap^3) work in collectNearbyTriangles
  // just to confirm "nothing here" one point at a time. Precomputing once
  // which cells are within searchRadiusCap of any triangle turns that
  // per-point confirmation into an O(1) lookup, skipping the ring search
  // entirely for points already known to be far.
  const nearMask =
    searchRadiusCap !== undefined
      ? computeNearMask(buckets, cellCountX, cellCountY, cellCountZ, searchRadiusCap)
      : null;

  for (let iz = 0; iz < nz; iz++) {
    const wz = origin[2] + iz * cellSize;
    for (let iy = 0; iy < ny; iy++) {
      const wy = origin[1] + iy * cellSize;

      // Sign is decided per grid-point row by casting a ray in +X from
      // outside the mesh and counting surface crossings (even-odd rule):
      // odd crossings so far along the ray means the point is inside.
      // Unlike trusting a triangle's face normal, ray-triangle intersection
      // doesn't care which way the triangle winds, so this is immune to the
      // occasional STEP-tessellation triangle with a flipped normal — a
      // normal-based vote turned out to still be wrong for any point that
      // sits squarely in front of the flipped triangle itself, since nothing
      // else is closer to outvote it there.
      const crossingXs = collectRowCrossings(mesh, buckets, cellCountX, cellCountY, cellCountZ, iy, iz, wy, wz, rayOriginX);

      for (let ix = 0; ix < nx; ix++) {
        const wx = origin[0] + ix * cellSize;

        const cellX = clamp(ix, 0, cellCountX - 1);
        const cellY = clamp(iy, 0, cellCountY - 1);
        const cellZ = clamp(iz, 0, cellCountZ - 1);
        const maybeNear = !nearMask || nearMask[cellX + cellY * cellCountX + cellZ * cellCountX * cellCountY] === 1;

        if (maybeNear) {
          collectNearbyTriangles(buckets, dims, ix, iy, iz, triangleCount, searchRadiusCap, candidateSet);
        } else {
          candidateSet.clear();
        }

        let bestDistSq = Infinity;
        let bestTri = -1;

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
          }
        }

        const idx = ix + iy * nx + iz * nx * ny;

        let crossingsBeforePoint = 0;
        for (const x of crossingXs) {
          if (x < wx) crossingsBeforePoint++;
          else break;
        }
        const sign = crossingsBeforePoint % 2 === 1 ? -1 : 1;

        if (bestTri < 0) {
          // No triangle found within the search radius — either genuinely
          // far from the surface (maxSearchDistance capped the search, and
          // collision code only ever checks distance against a small
          // margin out here, so the exact magnitude doesn't matter) or, if
          // uncapped, a real gap in the mesh. The sign is still taken from
          // ray-parity rather than assumed "outside": a particle that
          // somehow ends up deep inside solid geometry must still be
          // pushed back out.
          distances[idx] = sign * 1e6;
          continue;
        }

        const dist = Math.sqrt(bestDistSq);
        distances[idx] = sign * dist;
      }
    }
  }

  return { origin, cellSize, dims, distances };
}

/**
 * Casts a ray from (rayOriginX, wy, wz) in the +X direction and returns the
 * sorted, de-duplicated x-coordinates where it crosses the mesh surface.
 * Shared by every grid point in this (iy,iz) row, since they all share the
 * same ray — only how far along it each point sits differs.
 */
function collectRowCrossings(
  mesh: TriangleSoup,
  buckets: number[][],
  cellCountX: number,
  cellCountY: number,
  cellCountZ: number,
  iy: number,
  iz: number,
  wy: number,
  wz: number,
  rayOriginX: number
): number[] {
  const cy = clamp(iy, 0, cellCountY - 1);
  const cz = clamp(iz, 0, cellCountZ - 1);

  const candidates = new Set<number>();
  for (let cx = 0; cx < cellCountX; cx++) {
    const bucket = buckets[cx + cy * cellCountX + cz * cellCountX * cellCountY]!;
    for (const tri of bucket) candidates.add(tri);
  }

  const hits: number[] = [];
  for (const tri of candidates) {
    const i0 = mesh.indices[tri * 3]!;
    const i1 = mesh.indices[tri * 3 + 1]!;
    const i2 = mesh.indices[tri * 3 + 2]!;

    const t = rayTriangleIntersectX(
      rayOriginX, wy, wz,
      mesh.positions[i0 * 3]!, mesh.positions[i0 * 3 + 1]!, mesh.positions[i0 * 3 + 2]!,
      mesh.positions[i1 * 3]!, mesh.positions[i1 * 3 + 1]!, mesh.positions[i1 * 3 + 2]!,
      mesh.positions[i2 * 3]!, mesh.positions[i2 * 3 + 1]!, mesh.positions[i2 * 3 + 2]!
    );
    if (t !== null) hits.push(rayOriginX + t);
  }

  hits.sort((a, b) => a - b);

  // Merge near-duplicate crossings: a ray passing through an edge or vertex
  // shared by multiple triangles hits each of them individually, which
  // would otherwise be counted as separate crossings and corrupt the
  // even-odd parity.
  const mergeEps = 1e-7;
  const merged: number[] = [];
  for (const x of hits) {
    if (merged.length === 0 || x - merged[merged.length - 1]! > mergeEps) {
      merged.push(x);
    }
  }
  return merged;
}

/**
 * Moller-Trumbore ray-triangle intersection for a ray from (ox,oy,oz) in
 * the +X direction. Returns the x-distance to the hit, or null if the ray
 * misses the triangle (or is parallel to it). Intentionally does not cull
 * back-facing triangles — intersection existence must not depend on
 * triangle winding, since that's precisely what may be wrong.
 */
function rayTriangleIntersectX(
  ox: number, oy: number, oz: number,
  v0x: number, v0y: number, v0z: number,
  v1x: number, v1y: number, v1z: number,
  v2x: number, v2y: number, v2z: number
): number | null {
  const e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
  const e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

  // h = dir × e2, dir = (1,0,0)
  const hx = 0, hy = -e2z, hz = e2y;
  const a = e1x * hx + e1y * hy + e1z * hz;
  if (Math.abs(a) < 1e-12) return null;

  const f = 1 / a;
  const sx = ox - v0x, sy = oy - v0y, sz = oz - v0z;
  const u = f * (sx * hx + sy * hy + sz * hz);
  if (u < 0 || u > 1) return null;

  // q = s × e1
  const qx = sy * e1z - sz * e1y;
  const qy = sz * e1x - sx * e1z;
  const qz = sx * e1y - sy * e1x;
  const v = f * qx; // dir · q, dir = (1,0,0)
  if (v < 0 || u + v > 1) return null;

  const t = f * (e2x * qx + e2y * qy + e2z * qz);
  if (t <= 1e-9) return null;
  return t;
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
 * Multi-source BFS (26-connectivity, i.e. Chebyshev distance — matching the
 * cube-shell rings collectNearbyTriangles searches) that marks every bucket
 * cell within `cap` rings of a triangle-occupied cell. A cell left
 * unmarked is guaranteed to have no triangle within `cap` rings, so
 * buildSignedDistanceField's main loop can skip collectNearbyTriangles's
 * per-point ring expansion entirely for grid points in such a cell — this
 * is what turns the O(cap^3) "confirm nothing's nearby" cost, paid
 * separately by every one of the (typically many) far-from-the-surface grid
 * points, into a single O(cellCount) precompute shared by all of them.
 */
function computeNearMask(
  buckets: number[][],
  cellCountX: number,
  cellCountY: number,
  cellCountZ: number,
  cap: number
): Uint8Array {
  const total = cellCountX * cellCountY * cellCountZ;
  const near = new Uint8Array(total);
  const dist = new Int16Array(total).fill(-1);
  const queue = new Int32Array(total);
  let head = 0;
  let tail = 0;

  for (let i = 0; i < total; i++) {
    if (buckets[i]!.length > 0) {
      near[i] = 1;
      dist[i] = 0;
      queue[tail++] = i;
    }
  }

  const planeSize = cellCountX * cellCountY;

  while (head < tail) {
    const idx = queue[head++]!;
    const d = dist[idx]!;
    if (d >= cap) continue;

    const z = Math.floor(idx / planeSize);
    const rem = idx - z * planeSize;
    const y = Math.floor(rem / cellCountX);
    const x = rem - y * cellCountX;

    for (let dz = -1; dz <= 1; dz++) {
      const nz = z + dz;
      if (nz < 0 || nz >= cellCountZ) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= cellCountY) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0 && dz === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= cellCountX) continue;

          const nIdx = nx + ny * cellCountX + nz * planeSize;
          if (dist[nIdx]! >= 0) continue;
          dist[nIdx] = d + 1;
          near[nIdx] = 1;
          queue[tail++] = nIdx;
        }
      }
    }
  }

  return near;
}

/**
 * Expands a search radius outward from grid point (ix,iy,iz) over the
 * triangle-bucket grid (one smaller than the point grid in each dimension)
 * until at least one candidate triangle is found, then stops (the true
 * nearest triangle may be one cell further out in a rare edge case, but
 * closestPointOnTriangle over the found candidates already gives a very
 * good — not necessarily perfectly exact — distance for boundary purposes).
 *
 * searchRadiusCap (in grid cells) stops the expansion early once the point
 * is confirmed farther than that from any triangle, rather than continuing
 * to expand — without it, a grid point deep inside or far outside the mesh
 * (which is most of the grid volume for a solid with any padding around it)
 * would expand rings all the way out to roughly its true distance from the
 * surface before finding anything, which for a fine grid is a huge amount
 * of wasted work for a point whose exact distance doesn't even matter
 * (see BuildSdfOptions.maxSearchDistance).
 */
function collectNearbyTriangles(
  buckets: number[][],
  dims: Vec3Tuple,
  ix: number,
  iy: number,
  iz: number,
  triangleCount: number,
  searchRadiusCap: number | undefined,
  out: Set<number>
): void {
  out.clear();
  const cellCountX = Math.max(1, dims[0] - 1);
  const cellCountY = Math.max(1, dims[1] - 1);
  const cellCountZ = Math.max(1, dims[2] - 1);
  const cx = clamp(ix, 0, cellCountX - 1);
  const cy = clamp(iy, 0, cellCountY - 1);
  const cz = clamp(iz, 0, cellCountZ - 1);

  const fullExtentRadius = Math.max(cellCountX, cellCountY, cellCountZ);
  const maxRadius = searchRadiusCap !== undefined ? Math.min(searchRadiusCap, fullExtentRadius) : fullExtentRadius;
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

  // Falling back to checking every triangle is only reasonable for a
  // genuine gap larger than the *entire* grid — a rare, real problem with
  // the mesh. When the search stopped early because of searchRadiusCap
  // instead, an empty result is the normal, common case (this point is
  // just farther from the surface than collision code cares about) and
  // should stay empty, not degrade into an O(triangleCount) scan for every
  // such point.
  if (out.size === 0 && maxRadius === fullExtentRadius) {
    for (let t = 0; t < triangleCount; t++) out.add(t);
  }
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

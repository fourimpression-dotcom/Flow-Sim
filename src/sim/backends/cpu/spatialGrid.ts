import type { Vec3Tuple } from "../../core/types";

/**
 * Uniform grid neighbor search, rebuilt every step around the current
 * particle cloud's own bounding box — not the full simulation domain. A
 * fixed domain-sized grid is wrong whenever particles are much more
 * localized than the domain allows (e.g. a small custom water source inside
 * a domain sized for a much bigger obstacle): cell size would have to be
 * coarsened to keep the domain-wide cell count bounded, which then crams
 * the whole (tightly-packed) particle cloud into a handful of cells and
 * turns neighbor search into near-brute-force. Sizing the grid to the data
 * instead keeps cell size close to the true smoothing radius regardless of
 * how big the domain is.
 *
 * CPU-specific: a GPU backend will need a fundamentally different (parallel
 * bucket-sort style) neighbor search, so this class intentionally lives
 * under backends/cpu rather than core — only its *contract* (given a query
 * point, hand back candidate neighbor indices) is expected to have a GPU
 * counterpart, not this implementation.
 *
 * queryNeighborCells() returns candidates from the surrounding 3x3x3 cells;
 * callers must still filter by actual distance <= h themselves.
 */

/**
 * Upper bound on total bucket count, as a safety net for the (now rare)
 * case where the particle cloud itself is spread very thin relative to the
 * smoothing radius (e.g. water spread across a huge domain). Cell size is
 * grown (never shrunk) to fit this budget; see clampCellSizeForBudget().
 */
const MAX_GRID_CELLS = 2_000_000;

export class SpatialGrid {
  private readonly baseCellSize: number;
  private cellSize: number;
  private origin: Vec3Tuple = [0, 0, 0];
  private dims: [number, number, number] = [1, 1, 1];
  private buckets: number[][] = [[]];
  /** Indices of buckets touched by the last build() call, so it can clear just those instead of the whole grid. */
  private activeBucketIndices: number[] = [];

  /** cellSize is normally the SPH smoothing radius h; see clampCellSizeForBudget for when it's coarsened. */
  constructor(cellSize: number) {
    this.baseCellSize = Math.max(cellSize, 1e-9);
    this.cellSize = this.baseCellSize;
  }

  private cellCoord(x: number, y: number, z: number): [number, number, number] {
    const ix = clamp(Math.floor((x - this.origin[0]) / this.cellSize), 0, this.dims[0] - 1);
    const iy = clamp(Math.floor((y - this.origin[1]) / this.cellSize), 0, this.dims[1] - 1);
    const iz = clamp(Math.floor((z - this.origin[2]) / this.cellSize), 0, this.dims[2] - 1);
    return [ix, iy, iz];
  }

  private linearIndex(ix: number, iy: number, iz: number): number {
    return ix + iy * this.dims[0] + iz * this.dims[0] * this.dims[1];
  }

  /**
   * Rebuilds the grid around the current particle positions' own bounding
   * box and repopulates it. Cheap relative to the previous domain-sized
   * approach: the bucket count now tracks how spread out the particles
   * actually are, not the (possibly much larger) simulation domain.
   */
  build(positions: Float32Array, count: number): void {
    if (count === 0) {
      this.activeBucketIndices.length = 0;
      return;
    }

    const min: Vec3Tuple = [Infinity, Infinity, Infinity];
    const max: Vec3Tuple = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < count; i++) {
      for (let axis = 0; axis < 3; axis++) {
        const v = positions[i * 3 + axis]!;
        if (v < min[axis]!) min[axis] = v;
        if (v > max[axis]!) max[axis] = v;
      }
    }

    const extent: Vec3Tuple = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    this.cellSize = clampCellSizeForBudget(extent, this.baseCellSize);
    this.origin = min;
    this.dims = [
      Math.max(1, Math.ceil(extent[0] / this.cellSize) + 1),
      Math.max(1, Math.ceil(extent[1] / this.cellSize) + 1),
      Math.max(1, Math.ceil(extent[2] / this.cellSize) + 1),
    ];

    const totalCells = this.dims[0] * this.dims[1] * this.dims[2];
    if (this.buckets.length !== totalCells) {
      this.buckets = new Array(totalCells);
      for (let i = 0; i < totalCells; i++) {
        this.buckets[i] = [];
      }
      this.activeBucketIndices = [];
    } else {
      for (const index of this.activeBucketIndices) {
        this.buckets[index]!.length = 0;
      }
      this.activeBucketIndices.length = 0;
    }

    for (let i = 0; i < count; i++) {
      const [ix, iy, iz] = this.cellCoord(positions[i * 3]!, positions[i * 3 + 1]!, positions[i * 3 + 2]!);
      const index = this.linearIndex(ix, iy, iz);
      const bucket = this.buckets[index]!;
      if (bucket.length === 0) {
        this.activeBucketIndices.push(index);
      }
      bucket.push(i);
    }
  }

  /** Appends candidate neighbor indices near (x,y,z) into `out` (cleared first). */
  queryNeighborCells(x: number, y: number, z: number, out: number[]): void {
    out.length = 0;
    const [cx, cy, cz] = this.cellCoord(x, y, z);

    for (let dz = -1; dz <= 1; dz++) {
      const iz = cz + dz;
      if (iz < 0 || iz >= this.dims[2]) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const iy = cy + dy;
        if (iy < 0 || iy >= this.dims[1]) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const ix = cx + dx;
          if (ix < 0 || ix >= this.dims[0]) continue;
          const bucket = this.buckets[this.linearIndex(ix, iy, iz)]!;
          for (let k = 0; k < bucket.length; k++) {
            out.push(bucket[k]!);
          }
        }
      }
    }
  }
}

/**
 * Grows cellSize (never shrinks it) so the naive cell count fits within
 * MAX_GRID_CELLS. Growing cellSize keeps queryNeighborCells' 3x3x3 search
 * correct — a particle within the smoothing radius h can only be more than
 * one cell away if cellSize < h, so cell size may only ever increase past
 * the requested (== h) value, never drop below it.
 */
function clampCellSizeForBudget(extent: Vec3Tuple, cellSize: number): number {
  const estimatedCells =
    (extent[0] / cellSize + 1) * (extent[1] / cellSize + 1) * (extent[2] / cellSize + 1);
  if (estimatedCells <= MAX_GRID_CELLS) return cellSize;

  const scale = Math.cbrt(estimatedCells / MAX_GRID_CELLS);
  return cellSize * scale;
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

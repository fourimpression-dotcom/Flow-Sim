import type { Vec3Tuple } from "./types";

/**
 * Reflects the inward component of a velocity against a surface normal, with
 * a restitution/damping factor — the shared formula behind every boundary
 * response in this simulation (domain AABB walls, and mesh collision). If
 * the velocity is already moving away from the surface (v.n >= 0), it is
 * returned unchanged.
 *
 * For an axis-aligned normal like (1,0,0), this reduces to exactly
 * `v' = -restitution * v` on that axis — the same formula the domain-wall
 * boundary stage uses, just generalized to an arbitrary normal direction.
 */
export function reflectVelocityAgainstNormal(
  vx: number,
  vy: number,
  vz: number,
  nx: number,
  ny: number,
  nz: number,
  restitution: number
): Vec3Tuple {
  const vn = vx * nx + vy * ny + vz * nz;
  if (vn >= 0) {
    return [vx, vy, vz];
  }
  const factor = (1 + restitution) * vn;
  return [vx - factor * nx, vy - factor * ny, vz - factor * nz];
}

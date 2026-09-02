// Shared between the main thread and the SDF-building worker. Kept free of
// DOM/WebWorker lib dependencies so both tsconfig projects (app / worker)
// can import it.
import type { BuildSdfOptions, TriangleSoup } from "./buildSignedDistanceField";
import type { SignedDistanceField } from "./signedDistanceField";

export interface SdfBuildRequest {
  id: number;
  mesh: TriangleSoup;
  options: BuildSdfOptions;
}

export type SdfBuildResponse =
  | { id: number; ok: true; field: SignedDistanceField }
  | { id: number; ok: false; error: string };

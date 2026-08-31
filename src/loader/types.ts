// Shared between the main thread and the worker. Kept free of DOM/WebWorker
// lib dependencies so both tsconfig projects (app / worker) can import it.
export interface LoadedMesh {
  name: string;
  positions: Float32Array;
  normals: Float32Array | null;
  indices: Uint32Array;
  color: [number, number, number] | null;
}

export interface StepLoadRequest {
  id: number;
  buffer: ArrayBuffer;
}

export type StepLoadResponse =
  | { id: number; ok: true; meshes: LoadedMesh[] }
  | { id: number; ok: false; error: string };

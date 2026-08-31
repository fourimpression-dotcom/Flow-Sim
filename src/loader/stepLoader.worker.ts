/// <reference lib="webworker" />
import occtimportjs from "occt-import-js";
import type { OcctModule } from "occt-import-js";
import type { LoadedMesh, StepLoadRequest, StepLoadResponse } from "./types";

declare const self: DedicatedWorkerGlobalScope;

let occtPromise: Promise<OcctModule> | null = null;

function getOcct(): Promise<OcctModule> {
  if (!occtPromise) {
    // The .wasm binary is copied to public/ (see scripts/copy-wasm.mjs) and
    // served from the site root — but "root" means Vite's configured base
    // path (e.g. "/repo-name/" on GitHub Pages project sites), not always
    // "/". A hardcoded "/occt-import-js.wasm" 404s once the base isn't "/".
    occtPromise = occtimportjs({
      locateFile: () => `${import.meta.env.BASE_URL}occt-import-js.wasm`,
    });
  }
  return occtPromise;
}

self.onmessage = async (event: MessageEvent<StepLoadRequest>) => {
  const { id, buffer } = event.data;

  try {
    const occt = await getOcct();
    const fileBuffer = new Uint8Array(buffer);
    const result = occt.ReadStepFile(fileBuffer, null);

    if (!result.success) {
      throw new Error("Failed to parse the STEP file");
    }

    const meshes: LoadedMesh[] = result.meshes.map((mesh) => ({
      name: mesh.name,
      positions: Float32Array.from(mesh.attributes.position.array),
      normals: mesh.attributes.normal ? Float32Array.from(mesh.attributes.normal.array) : null,
      indices: Uint32Array.from(mesh.index.array),
      color: mesh.color,
    }));

    const transferList: Transferable[] = [];
    for (const mesh of meshes) {
      transferList.push(mesh.positions.buffer, mesh.indices.buffer);
      if (mesh.normals) transferList.push(mesh.normals.buffer);
    }

    const response: StepLoadResponse = { id, ok: true, meshes };
    self.postMessage(response, transferList);
  } catch (err) {
    const response: StepLoadResponse = {
      id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(response);
  }
};

/// <reference lib="webworker" />
import { buildSignedDistanceField } from "./buildSignedDistanceField";
import type { SdfBuildRequest, SdfBuildResponse } from "./sdfWorkerTypes";

declare const self: DedicatedWorkerGlobalScope;

self.onmessage = (event: MessageEvent<SdfBuildRequest>) => {
  const { id, mesh, options } = event.data;

  try {
    const field = buildSignedDistanceField(mesh, options);
    const response: SdfBuildResponse = { id, ok: true, field };
    self.postMessage(response, [field.distances.buffer]);
  } catch (err) {
    const response: SdfBuildResponse = {
      id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(response);
  }
};

import type { BuildSdfOptions, TriangleSoup } from "./buildSignedDistanceField";
import type { SignedDistanceField } from "./signedDistanceField";
import type { SdfBuildRequest, SdfBuildResponse } from "./sdfWorkerTypes";

interface PendingEntry {
  resolve: (field: SignedDistanceField) => void;
  reject: (error: Error) => void;
}

/**
 * Main-thread handle to the SDF-building worker. buildSignedDistanceField is
 * an O(resolution^3) synchronous computation that can take a long time for a
 * complex mesh at high resolution — running it directly on the main thread
 * freezes the tab for however long that takes. This keeps it off the UI
 * thread the same way StepLoader keeps STEP parsing off it.
 */
export class SdfBuilder {
  private readonly worker: Worker;
  private nextId = 0;
  private readonly pending = new Map<number, PendingEntry>();

  constructor() {
    this.worker = new Worker(new URL("./buildSignedDistanceField.worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (event: MessageEvent<SdfBuildResponse>) => {
      const data = event.data;
      const entry = this.pending.get(data.id);
      if (!entry) return;
      this.pending.delete(data.id);

      if (data.ok) {
        entry.resolve(data.field);
      } else {
        entry.reject(new Error(data.error));
      }
    };
    this.worker.onerror = (event: ErrorEvent) => {
      for (const entry of this.pending.values()) {
        entry.reject(new Error(event.message));
      }
      this.pending.clear();
    };
  }

  async build(mesh: TriangleSoup, options: BuildSdfOptions): Promise<SignedDistanceField> {
    const id = this.nextId++;
    // Copy (not transfer) the mesh buffers: the caller's TriangleSoup (e.g.
    // main.ts's pendingObstacleMesh) is reused across multiple rebuilds —
    // whenever the water source spacing changes, for instance — and
    // transferring would detach its buffers, corrupting it for the next call.
    const positions = mesh.positions.slice();
    const indices = mesh.indices.slice();

    return new Promise<SignedDistanceField>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const request: SdfBuildRequest = { id, mesh: { positions, indices }, options };
      this.worker.postMessage(request, [positions.buffer, indices.buffer]);
    });
  }

  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
  }
}

import type { LoadedMesh, StepLoadRequest, StepLoadResponse } from "./types";

interface PendingEntry {
  resolve: (meshes: LoadedMesh[]) => void;
  reject: (error: Error) => void;
}

/**
 * Main-thread handle to the STEP-parsing worker. Keeps occt-import-js (WASM)
 * off the UI thread; each loadStepFile() call is a fire-and-forget request
 * matched back up by id when the worker replies.
 */
export class StepLoader {
  private readonly worker: Worker;
  private nextId = 0;
  private readonly pending = new Map<number, PendingEntry>();

  constructor() {
    this.worker = new Worker(new URL("./stepLoader.worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (event: MessageEvent<StepLoadResponse>) => {
      const data = event.data;
      const entry = this.pending.get(data.id);
      if (!entry) return;
      this.pending.delete(data.id);

      if (data.ok) {
        entry.resolve(data.meshes);
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

  async loadStepFile(file: File): Promise<LoadedMesh[]> {
    const buffer = await file.arrayBuffer();
    const id = this.nextId++;

    return new Promise<LoadedMesh[]>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const request: StepLoadRequest = { id, buffer };
      this.worker.postMessage(request, [buffer]);
    });
  }

  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
  }
}

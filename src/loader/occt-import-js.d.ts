// occt-import-js does not ship its own .d.ts, so this declares just the
// surface this project uses. Extend as more of the API is needed.
declare module "occt-import-js" {
  export interface OcctMeshAttributeArray {
    array: number[];
  }

  export interface OcctMeshAttributes {
    position: OcctMeshAttributeArray;
    normal?: OcctMeshAttributeArray;
  }

  export interface OcctMesh {
    name: string;
    attributes: OcctMeshAttributes;
    index: OcctMeshAttributeArray;
    color: [number, number, number] | null;
  }

  export interface OcctReadResult {
    success: boolean;
    meshes: OcctMesh[];
  }

  export interface OcctModule {
    ReadStepFile(buffer: Uint8Array, params: unknown): OcctReadResult;
    ReadBrepFile(buffer: Uint8Array, params: unknown): OcctReadResult;
    ReadIgesFile(buffer: Uint8Array, params: unknown): OcctReadResult;
  }

  export interface OcctInitOptions {
    locateFile?: (path: string, prefix: string) => string;
  }

  export default function occtimportjs(options?: OcctInitOptions): Promise<OcctModule>;
}

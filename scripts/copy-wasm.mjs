// occt-import-js ships an Emscripten-built .wasm file that must be served
// as a static asset. This copies it into public/ so the dev server and
// production build can both find it at a stable, absolute URL ("/occt-import-js.wasm").
import { copyFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = fileURLToPath(new URL("..", import.meta.url));
const src = path.join(rootDir, "node_modules", "occt-import-js", "dist", "occt-import-js.wasm");
const destDir = path.join(rootDir, "public");
const dest = path.join(destDir, "occt-import-js.wasm");

if (!existsSync(src)) {
  console.warn(
    `[copy-wasm] occt-import-js.wasm not found at ${src}. ` +
      "Skipping copy — check the installed occt-import-js version's dist layout " +
      "and update scripts/copy-wasm.mjs if the path changed."
  );
  process.exit(0);
}

await mkdir(destDir, { recursive: true });
await copyFile(src, dest);
console.log(`[copy-wasm] Copied occt-import-js.wasm -> ${path.relative(rootDir, dest)}`);

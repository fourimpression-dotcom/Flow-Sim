import * as THREE from "three/webgpu";
import { StepLoader } from "./loader/StepLoader";
import type { LoadedMesh } from "./loader/types";
import { Viewer } from "./scene/Viewer";
import { SphSimulation } from "./sim/SphSimulation";
import { CpuSphBackend } from "./sim/backends/cpu/CpuSphBackend";
import { computeDefaultWaterSource, createDamBreakScenario, type ObstacleBounds } from "./sim/core/scenario";
import type { WaterBlockSource } from "./sim/core/types";
import { buildSignedDistanceField, type TriangleSoup } from "./sim/geometry/buildSignedDistanceField";
import type { SignedDistanceField } from "./sim/geometry/signedDistanceField";
import { errorMessage, requireElement } from "./dom";

const canvas = requireElement<HTMLCanvasElement>("#viewer-canvas");
const fileInput = requireElement<HTMLInputElement>("#file-input");
const modelOpacityInput = requireElement<HTMLInputElement>("#model-opacity");
const playButton = requireElement<HTMLButtonElement>("#play-button");
const pauseButton = requireElement<HTMLButtonElement>("#pause-button");
const stopButton = requireElement<HTMLButtonElement>("#stop-button");
const timeScaleInput = requireElement<HTMLInputElement>("#time-scale");
const applySourceButton = requireElement<HTMLButtonElement>("#apply-source-button");
const clearSourceButton = requireElement<HTMLButtonElement>("#clear-source-button");
const sourceCenterXInput = requireElement<HTMLInputElement>("#source-center-x");
const sourceCenterYInput = requireElement<HTMLInputElement>("#source-center-y");
const sourceCenterZInput = requireElement<HTMLInputElement>("#source-center-z");
const sourceSizeXInput = requireElement<HTMLInputElement>("#source-size-x");
const sourceSizeYInput = requireElement<HTMLInputElement>("#source-size-y");
const sourceSizeZInput = requireElement<HTMLInputElement>("#source-size-z");
const sourceSpacingInput = requireElement<HTMLInputElement>("#source-spacing");
const sourceDirXInput = requireElement<HTMLInputElement>("#source-dir-x");
const sourceDirYInput = requireElement<HTMLInputElement>("#source-dir-y");
const sourceDirZInput = requireElement<HTMLInputElement>("#source-dir-z");
const sourceSpeedInput = requireElement<HTMLInputElement>("#source-speed");
const statusEl = requireElement<HTMLSpanElement>("#status");
const dropHint = requireElement<HTMLDivElement>("#drop-hint");
const app = requireElement<HTMLDivElement>("#app");

// STEP files (and OCCT's STEP reader, which occt-import-js wraps)
// conventionally work in millimeters regardless of the file's declared
// unit. Our physics assumes SI units (gravity in m/s^2, density in
// kg/m^3), so mesh coordinates are rescaled to meters right after loading —
// applied once here so rendering, collision, and physics all stay in one
// consistent coordinate space.
const STEP_UNIT_TO_METERS = 0.001;

// The custom-water-source form takes length values in millimeters (matching
// the STEP-file convention above, and more convenient for small parts) —
// the physics engine itself still works in meters throughout, so values are
// converted at the form boundary only.
const MM_TO_METERS = 0.001;
const METERS_TO_MM = 1000;

function setStatus(text: string): void {
  statusEl.textContent = text;
}

async function main(): Promise<void> {
  setStatus("Initializing renderer...");
  const viewer = await Viewer.create(canvas);
  const loader = new StepLoader();
  setStatus("Idle");

  const particleGeometry = new THREE.BufferGeometry();
  // Always keep a valid (if empty) position attribute — rendering a Points
  // object whose geometry has no attributes at all can throw inside the
  // renderer, which would kill the animation loop entirely.
  particleGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
  const particleMaterial = new THREE.PointsMaterial({
    color: 0x4fb0ff,
    size: 0.014,
    sizeAttenuation: true,
  });
  const points = new THREE.Points(particleGeometry, particleMaterial);
  points.visible = false;
  // The renderer frustum-culls Points using geometry.boundingSphere, which
  // is computed once (lazily, from wherever the initial water block is) and
  // never automatically recomputed as positionAttribute.needsUpdate keeps
  // being set. Left on, that stale sphere can end up outside the camera
  // frustum while the actual (moved) particles are still on-screen, making
  // the water vanish depending on zoom/orientation. The particle count here
  // is small enough that skipping culling entirely is cheap.
  points.frustumCulled = false;
  viewer.scene.add(points);

  // Obstacle state from the last loaded STEP file, if any. The collision
  // field is built lazily (only once simulation is actually turned on) so
  // just browsing STEP files with simulation off stays fast.
  let obstacleBounds: ObstacleBounds | undefined;
  let pendingObstacleMesh: TriangleSoup | undefined;
  let collisionField: SignedDistanceField | null = null;
  // Three states: stopped (simulation === null, no water), paused
  // (simulation set but isPlaying false — particles frozen in place), and
  // playing (simulation set and isPlaying true — physics stepping each
  // frame). Play starts fresh from stopped, or resumes in place from
  // paused (never resets the water); Pause freezes in place; Stop
  // clears the water entirely. None of the three ever touch the camera.
  let simulation: SphSimulation | null = null;
  let isPlaying = false;

  // User-specified water source (center/size/direction/speed via the form),
  // overriding the default "drop a block in the corner" placement. The form
  // starts pre-filled with the values equivalent to that default (see
  // populateSourceForm below) so the user has a sensible starting point to
  // tweak from rather than arbitrary placeholders.
  let customWaterSource: WaterBlockSource | null = null;
  populateSourceForm(computeDefaultWaterSource());

  function startSimulation(reframeCamera: boolean): void {
    const scenario = createDamBreakScenario({
      obstacle: obstacleBounds,
      waterSource: customWaterSource ?? undefined,
    });
    simulation = new SphSimulation(
      new CpuSphBackend(),
      scenario.params,
      scenario.initialPositions,
      scenario.initialVelocities
    );
    simulation.setCollisionField(collisionField);

    particleMaterial.size = scenario.params.smoothingRadius * 0.4;

    const positionAttribute = new THREE.BufferAttribute(simulation.getPositions(), 3);
    positionAttribute.setUsage(THREE.DynamicDrawUsage);
    particleGeometry.setAttribute("position", positionAttribute);
    // Only reveal the points once the geometry actually holds this
    // simulation's data — showing it earlier (e.g. right when the toggle is
    // checked, before an async SDF build finishes) would render a stale or
    // empty attribute for a frame or more.
    points.visible = true;

    // The floor grid always follows the current domain, even when the
    // camera doesn't — otherwise the grid keeps showing wherever it was last
    // framed (e.g. a STEP model's footprint) while the water actually
    // settles on the domain floor, which can be a different size/position.
    const box = new THREE.Box3(
      new THREE.Vector3(...scenario.params.domainMin),
      new THREE.Vector3(...scenario.params.domainMax)
    );
    // Reset (same domain, just fresh water) deliberately leaves the camera
    // where the user left it; a fresh start or a newly loaded obstacle
    // reframes since the domain itself may have changed size/position.
    if (reframeCamera) {
      viewer.frameToBox(box);
    } else {
      viewer.updateGridForBox(box);
    }
  }

  async function ensureCollisionFieldAndStart(reframeCamera: boolean): Promise<void> {
    if (pendingObstacleMesh && !collisionField) {
      setStatus("Building collision field...");
      // Let the status text actually paint before the synchronous SDF build blocks the main thread.
      await new Promise(requestAnimationFrame);

      const bounds = obstacleBounds!;
      const sizeX = bounds.max[0] - bounds.min[0];
      const sizeY = bounds.max[1] - bounds.min[1];
      const sizeZ = bounds.max[2] - bounds.min[2];
      const padding = Math.max(sizeX, sizeY, sizeZ, 1e-6) * 0.2;

      collisionField = buildSignedDistanceField(pendingObstacleMesh, { padding, resolution: 32 });
    }
    startSimulation(reframeCamera);
  }

  async function handleFile(file: File): Promise<void> {
    if (!/\.(step|stp)$/i.test(file.name)) {
      setStatus("Please select a STEP file (.step / .stp)");
      return;
    }

    dropHint.style.display = "none";
    setStatus(`Loading: ${file.name}`);

    try {
      const rawMeshes = await loader.loadStepFile(file);
      const meshes = transformMeshesFromStep(rawMeshes, STEP_UNIT_TO_METERS);
      viewer.setMeshes(meshes);

      const merged = mergeMeshesForCollision(meshes);
      obstacleBounds = computeMeshBounds(merged.positions);
      pendingObstacleMesh = merged;
      collisionField = null; // invalidate any previously built field; rebuilt lazily
      // Refresh the form to the new default (obstacle-relative) placement,
      // so it stays a sensible starting point for this model's scale/position.
      populateSourceForm(computeDefaultWaterSource(obstacleBounds));

      const triangleCount = meshes.reduce((sum, m) => sum + m.indices.length / 3, 0);
      setStatus(`${file.name} (${meshes.length} mesh${meshes.length === 1 ? "" : "es"} / ${triangleCount.toLocaleString()} triangles)`);

      if (simulation !== null) {
        // Water is currently shown (playing or paused): rebuild for the new
        // obstacle, keeping whatever play/pause state was in effect.
        await ensureCollisionFieldAndStart(true);
      }
    } catch (err) {
      console.error(err);
      setStatus(`Failed to load: ${errorMessage(err)}`);
    }
  }

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) void handleFile(file);
  });

  modelOpacityInput.addEventListener("input", () => {
    viewer.setOpacity(Number(modelOpacityInput.value));
  });

  app.addEventListener("dragover", (event) => {
    event.preventDefault();
  });

  app.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer?.files[0];
    if (file) void handleFile(file);
  });

  playButton.addEventListener("click", () => {
    isPlaying = true;
    if (simulation === null) {
      // Stopped -> fresh start.
      void ensureCollisionFieldAndStart(false);
    }
    // Paused -> just resume in place; the frozen positions are untouched.
  });

  pauseButton.addEventListener("click", () => {
    isPlaying = false;
    if (simulation) {
      setStatus("Paused");
    }
  });

  stopButton.addEventListener("click", () => {
    isPlaying = false;
    points.visible = false;
    simulation = null;
    setStatus("Stopped");
  });

  applySourceButton.addEventListener("click", () => {
    const center: [number, number, number] = [
      (Number(sourceCenterXInput.value) || 0) * MM_TO_METERS,
      (Number(sourceCenterYInput.value) || 0) * MM_TO_METERS,
      (Number(sourceCenterZInput.value) || 0) * MM_TO_METERS,
    ];
    const size: [number, number, number] = [
      (Number(sourceSizeXInput.value) || 200) * MM_TO_METERS,
      (Number(sourceSizeYInput.value) || 200) * MM_TO_METERS,
      (Number(sourceSizeZInput.value) || 200) * MM_TO_METERS,
    ];
    const spacing = (Number(sourceSpacingInput.value) || 20) * MM_TO_METERS;
    const speed = Number(sourceSpeedInput.value) || 0;

    let dirX = Number(sourceDirXInput.value) || 0;
    let dirY = Number(sourceDirYInput.value) || 0;
    let dirZ = Number(sourceDirZInput.value) || 0;
    const dirLength = Math.hypot(dirX, dirY, dirZ);
    if (dirLength < 1e-9) {
      // Degenerate (all-zero) direction: fall back to straight down.
      dirX = 0;
      dirY = -1;
      dirZ = 0;
    } else {
      dirX /= dirLength;
      dirY /= dirLength;
      dirZ /= dirLength;
    }

    customWaterSource = {
      center,
      size,
      spacing,
      initialVelocity: [dirX * speed, dirY * speed, dirZ * speed],
    };

    setStatus("Fluid state set. Press Play to apply it.");

    if (simulation !== null) {
      void ensureCollisionFieldAndStart(false);
    }
  });

  clearSourceButton.addEventListener("click", () => {
    customWaterSource = null;
    populateSourceForm(computeDefaultWaterSource(obstacleBounds));
    setStatus("Fluid state cleared (back to the default placement)");

    if (simulation !== null) {
      void ensureCollisionFieldAndStart(false);
    }
  });

  let statusTimer = 0;
  viewer.setUpdateCallback((deltaSeconds) => {
    if (!isPlaying || !simulation) return;

    // Cap the delta fed into the simulation: a tab coming back from being
    // backgrounded can report a huge deltaSeconds, which would otherwise
    // burn through the substep cap instantly without visibly moving. Slow
    // motion is just this capped wall-clock delta scaled down before it
    // reaches the physics — the simulation itself is unaware of it.
    const timeScale = Math.max(Number(timeScaleInput.value) || 1, 0);
    simulation.update(Math.min(deltaSeconds, 0.1) * timeScale);

    const positionAttribute = particleGeometry.getAttribute("position") as THREE.BufferAttribute;
    positionAttribute.needsUpdate = true;

    statusTimer += deltaSeconds;
    if (statusTimer >= 0.2) {
      statusTimer = 0;
      const diagnostics = simulation.getDiagnostics();
      const obstacleNote = collisionField ? "STEP collision: on" : "STEP collision: off (box only)";
      setStatus(
        `Particles: ${simulation.particleCount} / ` +
          `Mean density: ${diagnostics.meanDensity.toFixed(1)} kg/m³ / ` +
          `Max speed: ${diagnostics.maxSpeed.toFixed(2)} m/s / ` +
          obstacleNote
      );
    }
  });
}

/** Fills the custom-water-source form fields (in mm) from a WaterBlockSource (in m) — used to seed it with the current default placement. */
function populateSourceForm(source: WaterBlockSource): void {
  sourceCenterXInput.value = (source.center[0] * METERS_TO_MM).toFixed(2);
  sourceCenterYInput.value = (source.center[1] * METERS_TO_MM).toFixed(2);
  sourceCenterZInput.value = (source.center[2] * METERS_TO_MM).toFixed(2);
  sourceSizeXInput.value = (source.size[0] * METERS_TO_MM).toFixed(2);
  sourceSizeYInput.value = (source.size[1] * METERS_TO_MM).toFixed(2);
  sourceSizeZInput.value = (source.size[2] * METERS_TO_MM).toFixed(2);
  sourceSpacingInput.value = (source.spacing * METERS_TO_MM).toFixed(2);

  const speed = Math.hypot(...source.initialVelocity);
  if (speed > 1e-9) {
    sourceDirXInput.value = (source.initialVelocity[0] / speed).toFixed(4);
    sourceDirYInput.value = (source.initialVelocity[1] / speed).toFixed(4);
    sourceDirZInput.value = (source.initialVelocity[2] / speed).toFixed(4);
  } else {
    sourceDirXInput.value = "0";
    sourceDirYInput.value = "-1";
    sourceDirZInput.value = "0";
  }
  sourceSpeedInput.value = speed.toFixed(4);
}

/**
 * STEP/CAD tools (SolidWorks, OnShape, Fusion360, ...) conventionally treat
 * Z as "up"; Three.js — and this app's physics, whose gravity is [0,-9.81,0]
 * — treats Y as "up". Applied once here (alongside the mm->m unit fix) so
 * rendering, collision, and physics all agree on which way is down; doing it
 * as a scene-graph rotation instead would leave the physics data pointing
 * the old way while only the rendering looked rotated.
 */
function transformMeshesFromStep(meshes: LoadedMesh[], scale: number): LoadedMesh[] {
  return meshes.map((mesh) => ({
    ...mesh,
    positions: applyZUpToYUp(mesh.positions, scale),
    normals: mesh.normals ? applyZUpToYUp(mesh.normals, 1) : null,
  }));
}

/** (x, y, z) -> (x*s, z*s, -y*s): a -90° rotation about X, plus a uniform scale. */
function applyZUpToYUp(vectors: Float32Array, scale: number): Float32Array {
  const out = new Float32Array(vectors.length);
  for (let i = 0; i < vectors.length; i += 3) {
    const x = vectors[i]!;
    const y = vectors[i + 1]!;
    const z = vectors[i + 2]!;
    out[i] = x * scale;
    out[i + 1] = z * scale;
    out[i + 2] = -y * scale;
  }
  return out;
}

function mergeMeshesForCollision(meshes: LoadedMesh[]): TriangleSoup {
  let totalVertices = 0;
  let totalIndices = 0;
  for (const mesh of meshes) {
    totalVertices += mesh.positions.length / 3;
    totalIndices += mesh.indices.length;
  }

  const positions = new Float32Array(totalVertices * 3);
  const indices = new Uint32Array(totalIndices);

  let vertexOffset = 0;
  let indexWriteOffset = 0;
  for (const mesh of meshes) {
    positions.set(mesh.positions, vertexOffset * 3);
    for (let i = 0; i < mesh.indices.length; i++) {
      indices[indexWriteOffset + i] = mesh.indices[i]! + vertexOffset;
    }
    vertexOffset += mesh.positions.length / 3;
    indexWriteOffset += mesh.indices.length;
  }

  return { positions, indices };
}

function computeMeshBounds(positions: Float32Array): ObstacleBounds {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  const count = positions.length / 3;
  for (let i = 0; i < count; i++) {
    for (let axis = 0; axis < 3; axis++) {
      const v = positions[i * 3 + axis]!;
      if (v < min[axis]!) min[axis] = v;
      if (v > max[axis]!) max[axis] = v;
    }
  }
  return { min, max };
}

main().catch((err: unknown) => {
  console.error(err);
  setStatus(`Initialization error: ${errorMessage(err)}`);
});

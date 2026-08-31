import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { LoadedMesh } from "../loader/types";

interface ModelPart {
  mesh: THREE.Mesh;
  material: THREE.MeshStandardMaterial;
  edges: THREE.LineSegments;
}

const EDGE_ANGLE_THRESHOLD_DEGREES = 20;
const DEFAULT_OPACITY = 0.15;

/**
 * Wraps a Three.js WebGPURenderer scene. Falls back to WebGL2 automatically
 * (handled internally by WebGPURenderer) on browsers without WebGPU.
 */
export class Viewer {
  readonly renderer: THREE.WebGPURenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly modelGroup: THREE.Group;
  private readonly canvas: HTMLCanvasElement;
  private readonly clock: THREE.Clock;
  private gridHelper: THREE.GridHelper;
  private updateCallback: ((deltaSeconds: number) => void) | null = null;
  private modelParts: ModelPart[] = [];
  /** Model surfaces are always edges + translucent; only the opacity is configurable. */
  private opacity = DEFAULT_OPACITY;
  private readonly axisGizmoScene: THREE.Scene;
  private readonly axisGizmoCamera: THREE.OrthographicCamera;

  private constructor(canvas: HTMLCanvasElement, renderer: THREE.WebGPURenderer) {
    this.canvas = canvas;
    this.renderer = renderer;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a1a);

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 10000);
    this.camera.position.set(2, 2, 2);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;

    this.modelGroup = new THREE.Group();
    this.scene.add(this.modelGroup);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2));
    const dirLight = new THREE.DirectionalLight(0xffffff, 2);
    dirLight.position.set(5, 10, 7);
    this.scene.add(dirLight);

    this.gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x2a2a2a);
    this.scene.add(this.gridHelper);

    this.clock = new THREE.Clock();

    this.axisGizmoScene = new THREE.Scene();
    this.axisGizmoCamera = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0.1, 10);
    buildAxisGizmo(this.axisGizmoScene);

    window.addEventListener("resize", () => this.handleResize());
    this.handleResize();

    this.renderer.setAnimationLoop(() => this.renderFrame());
  }

  static async create(canvas: HTMLCanvasElement): Promise<Viewer> {
    const renderer = new THREE.WebGPURenderer({ canvas, antialias: true });
    await renderer.init();
    return new Viewer(canvas, renderer);
  }

  /**
   * Registers a callback invoked once per rendered frame, before the camera
   * controls update and the scene renders. Used to drive things that need to
   * advance in lockstep with the render loop (e.g. a physics simulation)
   * without each caller managing its own requestAnimationFrame loop.
   * Pass null to stop.
   */
  setUpdateCallback(callback: ((deltaSeconds: number) => void) | null): void {
    this.updateCallback = callback;
  }

  /** Points the camera/controls at a world-space bounding box. */
  frameToBox(box: THREE.Box3): void {
    this.frameCamera(box);
  }

  /**
   * Repositions/resizes the floor grid to match a world-space bounding box
   * WITHOUT touching the camera — for callers (like restarting a running
   * simulation) that want the grid to keep reflecting the current domain
   * even while deliberately leaving the camera wherever the user left it.
   */
  updateGridForBox(box: THREE.Box3): void {
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) * 0.5 || 1;
    this.updateGridHelper(box, center, radius);
  }

  /** Replaces the currently displayed model and reframes the camera on it. */
  setMeshes(meshes: LoadedMesh[]): void {
    for (const part of this.modelParts) {
      part.mesh.geometry.dispose();
      part.material.dispose();
      part.edges.geometry.dispose();
      (part.edges.material as THREE.Material).dispose();
    }
    this.modelParts = [];
    this.modelGroup.clear();

    const bounds = new THREE.Box3();

    for (const mesh of meshes) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(mesh.positions, 3));
      if (mesh.normals) {
        geometry.setAttribute("normal", new THREE.BufferAttribute(mesh.normals, 3));
      } else {
        geometry.computeVertexNormals();
      }
      geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));

      const color = mesh.color
        ? new THREE.Color(mesh.color[0], mesh.color[1], mesh.color[2])
        : new THREE.Color(0x8899aa);

      const material = new THREE.MeshStandardMaterial({
        color,
        metalness: 0.1,
        roughness: 0.6,
        side: THREE.DoubleSide,
        // Avoids z-fighting between the solid surface and the edge-line
        // overlay, which are otherwise exactly coincident in depth.
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      });

      const threeMesh = new THREE.Mesh(geometry, material);

      // Edges (real feature edges, not a full tessellation wireframe) are
      // rendered as a separate sibling object layered over the translucent surface.
      const edgesGeometry = new THREE.EdgesGeometry(geometry, EDGE_ANGLE_THRESHOLD_DEGREES);
      const edges = new THREE.LineSegments(edgesGeometry, new THREE.LineBasicMaterial({ color: 0x000000 }));

      this.modelGroup.add(threeMesh);
      this.modelGroup.add(edges);
      this.modelParts.push({ mesh: threeMesh, material, edges });

      geometry.computeBoundingBox();
      if (geometry.boundingBox) {
        bounds.union(geometry.boundingBox);
      }
    }

    this.applyOpacity();
    this.frameCamera(bounds);
  }

  /** Sets the model surface opacity (0 = invisible, 1 = fully opaque). Edges always stay fully visible. */
  setOpacity(value: number): void {
    this.opacity = Math.min(Math.max(value, 0), 1);
    this.applyOpacity();
  }

  private applyOpacity(): void {
    for (const part of this.modelParts) {
      part.mesh.visible = true;
      part.edges.visible = true;
      part.material.transparent = true;
      part.material.opacity = this.opacity;
      part.material.depthWrite = false;
      part.material.needsUpdate = true;
    }
  }

  private frameCamera(box: THREE.Box3): void {
    if (box.isEmpty()) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) * 0.5 || 1;

    const distance = radius / Math.sin((this.camera.fov * Math.PI) / 360);
    const direction = new THREE.Vector3(1, 0.8, 1).normalize();

    this.camera.position.copy(center).addScaledVector(direction, distance * 1.5);
    this.camera.near = Math.max(distance / 100, 0.01);
    this.camera.far = distance * 100;
    this.camera.updateProjectionMatrix();

    this.controls.target.copy(center);
    this.controls.update();

    this.updateGridHelper(box, center, radius);
  }

  /**
   * The floor grid is a fixed number of world units by default, which looks
   * either invisibly small or absurdly huge depending on whatever content's
   * scale happens to be loaded (a 50mm STEP part vs. a multi-meter
   * assembly). Rebuilds it sized relative to, and positioned at the bottom
   * of, whatever box was just framed — the model's own bounds when just
   * viewing it (setMeshes), or the simulation domain when running (whose
   * floor is where water actually settles, padded below the model by an
   * amount that scales with the model's size — never a fixed offset).
   */
  private updateGridHelper(box: THREE.Box3, center: THREE.Vector3, radius: number): void {
    const size = Math.max(radius * 4, 1e-3);

    this.scene.remove(this.gridHelper);
    this.gridHelper.geometry.dispose();
    (this.gridHelper.material as THREE.Material).dispose();

    this.gridHelper = new THREE.GridHelper(size, 10, 0x444444, 0x2a2a2a);
    this.gridHelper.position.set(center.x, box.min.y, center.z);
    this.scene.add(this.gridHelper);
  }

  private renderFrame(): void {
    const deltaSeconds = this.clock.getDelta();
    this.updateCallback?.(deltaSeconds);
    this.controls.update();

    const width = this.canvas.clientWidth || window.innerWidth;
    const height = this.canvas.clientHeight || window.innerHeight;

    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, width, height);
    this.renderer.render(this.scene, this.camera);

    this.renderAxisGizmo(width, height);
  }

  /**
   * Renders a small CAD-style XYZ axis indicator into the top-right corner,
   * matching the main camera's orientation. Uses a second scene/camera and a
   * scissor+viewport rectangle rather than a full extra render target —
   * cheap, and the standard Three.js technique for picture-in-picture-style
   * overlays.
   */
  private renderAxisGizmo(canvasWidth: number, canvasHeight: number): void {
    const size = Math.max(60, Math.min(110, canvasWidth * 0.12, canvasHeight * 0.12));
    const margin = 12;
    const x = canvasWidth - size - margin;
    // WebGPURenderer's viewport/scissor origin is the top-left of the
    // canvas (unlike WebGLRenderer's bottom-left/OpenGL convention), so a
    // small y here means near the top.
    const y = margin;

    // Match the main camera's orientation (not its position/distance/zoom):
    // put the gizmo camera the same direction from the origin as the main
    // camera is from its orbit target, at a fixed distance.
    const direction = this.camera.position.clone().sub(this.controls.target);
    if (direction.lengthSq() < 1e-9) direction.set(0, 0, 1);
    direction.normalize();

    this.axisGizmoCamera.position.copy(direction).multiplyScalar(4);
    this.axisGizmoCamera.up.copy(this.camera.up);
    this.axisGizmoCamera.lookAt(0, 0, 0);

    this.renderer.setScissorTest(true);
    this.renderer.setScissor(x, y, size, size);
    this.renderer.setViewport(x, y, size, size);
    this.renderer.clearDepth();
    this.renderer.render(this.axisGizmoScene, this.axisGizmoCamera);
    this.renderer.setScissorTest(false);
  }

  private handleResize(): void {
    const width = this.canvas.clientWidth || window.innerWidth;
    const height = this.canvas.clientHeight || window.innerHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(width, height, false);
  }
}

/** Builds the X (red) / Y (green) / Z (blue) arrows + text labels for the axis gizmo. */
function buildAxisGizmo(scene: THREE.Scene): void {
  const axes: { dir: THREE.Vector3; color: number; label: string }[] = [
    { dir: new THREE.Vector3(1, 0, 0), color: 0xff5555, label: "X" },
    { dir: new THREE.Vector3(0, 1, 0), color: 0x55ff55, label: "Y" },
    { dir: new THREE.Vector3(0, 0, 1), color: 0x5599ff, label: "Z" },
  ];

  const length = 1;
  for (const axis of axes) {
    const arrow = new THREE.ArrowHelper(axis.dir, new THREE.Vector3(0, 0, 0), length, axis.color, 0.35, 0.22);
    scene.add(arrow);

    const label = createAxisLabelSprite(axis.label, axis.color);
    label.position.copy(axis.dir).multiplyScalar(length + 0.35);
    scene.add(label);
  }
}

function createAxisLabelSprite(text: string, color: number): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = `#${color.toString(16).padStart(6, "0")}`;
  ctx.font = "bold 44px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 32, 34);

  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, depthTest: false, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.setScalar(0.5);
  return sprite;
}

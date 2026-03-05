import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { loadTrie } from "./data/loadTrie";
import type { FilterState, TrieData } from "./types";
import { computeRadialLayout } from "./layout/radialLayout";
import type { RadialLayout } from "./layout/radialLayout";
import { computeLOD } from "./layout/lod";
import type { LODState } from "./layout/lod";
import { NodeRenderer } from "./rendering/NodeRenderer";
import { EdgeRenderer } from "./rendering/EdgeRenderer";
import { LabelRenderer } from "./rendering/LabelRenderer";
import { FilterPanel } from "./interaction/FilterPanel";
import { Picker } from "./interaction/Picker";
import { showTooltip, hideTooltip, setTooltipMetadata } from "./interaction/Tooltip";

async function main(): Promise<void> {
  const data: TrieData = await loadTrie();
  document.getElementById("loading")!.classList.add("hidden");

  // Pass metadata to tooltip for allophone context display
  setTooltipMetadata(data.metadata);

  // ── Renderer ───────────────────────────────────────────────────────────
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x06060c);
  document.body.appendChild(renderer.domElement);

  // ── Scene (no lighting — flat MeshBasicMaterial) ───────────────────────
  const scene = new THREE.Scene();

  // ── Camera + OrbitControls ─────────────────────────────────────────────
  const camera = new THREE.PerspectiveCamera(
    55,
    window.innerWidth / window.innerHeight,
    0.1,
    2000,
  );
  camera.position.set(80, 60, 100);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 10;
  controls.maxDistance = 800;
  controls.target.set(0, 0, 0);

  // ── Renderers ──────────────────────────────────────────────────────────
  const nodeRenderer = new NodeRenderer(data.metadata.nodeCount);
  nodeRenderer.setData(data.nodes);
  scene.add(nodeRenderer.mesh);
  scene.add(nodeRenderer.glowMesh);

  const edgeRenderer = new EdgeRenderer(data.metadata.edgeCount);
  edgeRenderer.setData(data.nodes, data.edges);
  scene.add(edgeRenderer.lines);
  scene.add(edgeRenderer.highways);

  const labelCanvas = document.getElementById("label-canvas") as HTMLCanvasElement;
  const labelRenderer = new LabelRenderer(labelCanvas);
  labelRenderer.setData(data.nodes);
  labelRenderer.resize(window.innerWidth, window.innerHeight);

  // ── Layout state ───────────────────────────────────────────────────────
  let currentLayout: RadialLayout;
  let currentLOD: LODState;
  let needsVisualUpdate = true;
  let labelNeedsUpdate = true;
  let lastLabelUpdate = 0;

  function recomputeLayout(filter: FilterState): void {
    currentLayout = computeRadialLayout(data.nodes, data.edges, filter);
    needsVisualUpdate = true;
    labelNeedsUpdate = true;
  }

  function updateVisuals(filter: FilterState): void {
    const camDist = camera.position.length();
    currentLOD = computeLOD(camDist, currentLayout.maxDepth);

    nodeRenderer.update(filter, currentLayout, currentLOD);
    edgeRenderer.update(filter, currentLayout, currentLOD);

    const stats = document.getElementById("stats")!;
    const langs = data.metadata.languages.join(", ");
    stats.textContent = `${nodeRenderer.mesh.count.toLocaleString()} nodes · ${data.metadata.totalWords.toLocaleString()} words · ${data.metadata.phonemeInventory.length} phonemes · ${langs}`;
  }

  // ── Filters ────────────────────────────────────────────────────────────
  function applyFilter(filter: FilterState): void {
    recomputeLayout(filter);
    updateVisuals(filter);
  }

  const filterPanel = new FilterPanel(data.metadata, applyFilter);
  applyFilter(filterPanel.getState());

  // ── Interaction ────────────────────────────────────────────────────────
  new Picker(camera, nodeRenderer, (node, x, y) => {
    if (node) {
      showTooltip(node, x, y, labelRenderer.getPath(node.id));
      labelRenderer.setHovered(node.id);
      document.body.style.cursor = "pointer";
    } else {
      hideTooltip();
      labelRenderer.setHovered(null);
      document.body.style.cursor = "";
    }
    labelNeedsUpdate = true;
  });

  // ── Resize ─────────────────────────────────────────────────────────────
  window.addEventListener("resize", () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    labelRenderer.resize(w, h);
    needsVisualUpdate = true;
    labelNeedsUpdate = true;
  });

  // ── Render loop ────────────────────────────────────────────────────────
  controls.addEventListener("change", () => {
    needsVisualUpdate = true;
    labelNeedsUpdate = true;
  });

  function animate(): void {
    requestAnimationFrame(animate);
    controls.update();

    if (needsVisualUpdate) {
      updateVisuals(filterPanel.getState());
      needsVisualUpdate = false;
    }

    const now = performance.now();
    if (labelNeedsUpdate && now - lastLabelUpdate > 200) {
      labelRenderer.update(filterPanel.getState(), currentLayout, currentLOD, camera);
      labelNeedsUpdate = false;
      lastLabelUpdate = now;
    }

    renderer.render(scene, camera);
  }

  animate();
}

main().catch(console.error);

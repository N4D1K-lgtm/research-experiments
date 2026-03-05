import * as THREE from "three";
import type { TrieNodeData, FilterState } from "../types";
import type { LayoutNode, RadialLayout } from "../layout/radialLayout";
import type { LODState } from "../layout/lod";
import { isNodeVisible, getNodeAlpha } from "../layout/lod";
import { blendNodeColor, hslToHex } from "../color/colorBlender";
import { nodeEntropy, incomingProb } from "../layout/entropy";

const MIN_SCALE = 0.12;
const MAX_SCALE = 1.8;
const GLOW_SCALE = 2.5;

/**
 * Node sizing encodes two things:
 * 1. Entropy (branching diversity) — high-entropy nodes are large hubs
 * 2. Transition probability from parent — high-prob nodes (highways) are larger
 *
 * Color encodes phonological role. Glow encodes terminal status.
 * Saturation encodes predictability: low-entropy (predictable) nodes are vivid,
 * high-entropy (uncertain) nodes are more washed-out.
 */
export class NodeRenderer {
  mesh: THREE.InstancedMesh;
  glowMesh: THREE.InstancedMesh;

  private nodeData: TrieNodeData[] = [];
  private nodeMap = new Map<number, TrieNodeData>();
  private dummy = new THREE.Object3D();
  private colorAttr: THREE.InstancedBufferAttribute;
  private glowColorAttr: THREE.InstancedBufferAttribute;

  instanceToNode: number[] = [];

  constructor(maxCount: number) {
    const geo = new THREE.IcosahedronGeometry(1, 2);
    const mat = new THREE.MeshBasicMaterial({ toneMapped: false });
    this.mesh = new THREE.InstancedMesh(geo, mat, maxCount);
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;

    const colors = new Float32Array(maxCount * 3);
    this.colorAttr = new THREE.InstancedBufferAttribute(colors, 3);
    this.mesh.instanceColor = this.colorAttr;

    const glowGeo = new THREE.IcosahedronGeometry(1, 1);
    const glowMat = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0.1,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      toneMapped: false,
    });
    this.glowMesh = new THREE.InstancedMesh(glowGeo, glowMat, maxCount);
    this.glowMesh.count = 0;
    this.glowMesh.frustumCulled = false;

    const glowColors = new Float32Array(maxCount * 3);
    this.glowColorAttr = new THREE.InstancedBufferAttribute(glowColors, 3);
    this.glowMesh.instanceColor = this.glowColorAttr;
  }

  setData(nodes: TrieNodeData[]): void {
    this.nodeData = nodes;
    this.nodeMap.clear();
    for (const n of nodes) this.nodeMap.set(n.id, n);
  }

  update(
    filter: FilterState,
    layout: RadialLayout,
    lod: LODState,
  ): void {
    this.instanceToNode = [];

    let idx = 0;
    let glowIdx = 0;
    const color = new THREE.Color();

    for (const [nodeId, layoutNode] of layout.nodes) {
      if (nodeId === 0) continue;

      const node = this.nodeMap.get(nodeId);
      if (!node) continue;

      if (node.depth > filter.maxDepth) continue;
      if (filter.terminalsOnly && !node.isTerminal) continue;
      if (filter.positionFilter.size > 0 && !filter.positionFilter.has(node.phonologicalPosition)) continue;

      const count = node.totalCount;
      if (count < filter.minFrequency || count === 0) continue;

      if (!isNodeVisible(layoutNode, lod)) continue;
      const alpha = getNodeAlpha(layoutNode, lod);
      if (alpha <= 0) continue;

      this.dummy.position.set(layoutNode.x, layoutNode.y, layoutNode.z);

      // ── Size: entropy (branching diversity) + incoming probability ──
      const entropy = nodeEntropy(node);
      const inProb = incomingProb(node, this.nodeMap);

      // Entropy component: high entropy = big hub (many valid continuations)
      // Max entropy for 20 children ≈ 4.3 bits. Scale to [0, 1].
      const entropyNorm = Math.min(1, entropy / 4.3);

      // Incoming prob component: high-probability branches are thicker
      // sqrt to compress range (0.3 at 9%, 0.7 at 50%)
      const probNorm = Math.sqrt(Math.min(1, inProb * 3));

      let s = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * (0.4 * entropyNorm + 0.35 * probNorm + 0.25 * Math.log10(Math.max(count, 1)) / 5);
      s = Math.min(MAX_SCALE, Math.max(MIN_SCALE, s));

      if (node.isTerminal) {
        s = Math.min(MAX_SCALE, s * 1.4);
      } else if (entropy === 0 && !node.isTerminal) {
        // Leaf non-terminal (dead end in grammar) — keep small
        s *= 0.5;
      }

      this.dummy.scale.setScalar(s);
      this.dummy.updateMatrix();
      this.mesh.setMatrixAt(idx, this.dummy.matrix);

      // ── Color: role hue + entropy modulates saturation ──
      const hsl = blendNodeColor(node);
      let l = hsl.l;
      let sat = hsl.s;

      // Low entropy (predictable path) → vivid/saturated
      // High entropy (decision point) → slightly washed, brighter
      if (entropy > 2.5) {
        // Decision hub: brighter, slightly less saturated
        l = Math.min(0.78, l + 0.08);
        sat = Math.max(0.25, sat - 0.1);
      } else if (entropy < 1.0 && node.childCount > 0) {
        // Funneling path: vivid color
        sat = Math.min(1.0, sat + 0.15);
      }

      // High-probability incoming edge → boost brightness (highway node)
      if (inProb > 0.15) {
        l = Math.min(0.82, l + inProb * 0.2);
      }

      const isMotifHighlighted = filter.highlightMotifs.size > 0 &&
        node.motifs?.some((m) => filter.highlightMotifs.has(m));

      if (node.isTerminal) {
        l = Math.min(0.88, l + 0.12);
        sat = Math.min(1.0, sat + 0.08);
      } else {
        l = Math.max(0.12, l * 0.55);
        sat = Math.max(0.08, sat * 0.55);
      }

      if (isMotifHighlighted) {
        l = Math.min(0.9, l + 0.2);
        sat = Math.min(1.0, sat + 0.2);
      } else if (filter.highlightMotifs.size > 0) {
        l *= 0.3;
        sat *= 0.3;
      }

      l *= alpha;

      color.setHex(hslToHex(hsl.h, sat, l));
      this.colorAttr.setXYZ(idx, color.r, color.g, color.b);

      this.instanceToNode.push(node.id);
      idx++;

      // Glow halo for terminals
      if (node.isTerminal) {
        this.dummy.scale.setScalar(s * GLOW_SCALE);
        this.dummy.updateMatrix();
        this.glowMesh.setMatrixAt(glowIdx, this.dummy.matrix);

        const glowL = Math.min(1.0, l * 1.5);
        color.setHex(hslToHex(hsl.h, sat, glowL));
        this.glowColorAttr.setXYZ(glowIdx, color.r, color.g, color.b);
        glowIdx++;
      }
    }

    this.mesh.count = idx;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.colorAttr.needsUpdate = true;

    this.glowMesh.count = glowIdx;
    this.glowMesh.instanceMatrix.needsUpdate = true;
    this.glowColorAttr.needsUpdate = true;
  }

  getNodeAtInstance(instanceId: number): TrieNodeData | null {
    const nodeId = this.instanceToNode[instanceId];
    return nodeId != null ? this.nodeMap.get(nodeId) ?? null : null;
  }
}

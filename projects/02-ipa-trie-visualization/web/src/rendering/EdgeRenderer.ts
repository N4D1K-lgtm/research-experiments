import * as THREE from "three";
import type { TrieNodeData, TrieEdge, FilterState } from "../types";
import type { RadialLayout } from "../layout/radialLayout";
import type { LODState } from "../layout/lod";
import { isNodeVisible } from "../layout/lod";
import { blendNodeColor, hslToHex } from "../color/colorBlender";

/** Threshold: edges above this probability get the "highway" treatment */
const HIGHWAY_THRESHOLD = 0.12;

/**
 * Two-pass edge rendering:
 *
 * 1. Base layer: all edges, dimmed proportionally to transition probability.
 *    Rare paths are barely visible threads.
 *
 * 2. Highway layer: only high-probability edges (>12%), drawn brighter with
 *    a separate additive mesh. These form the visible "grammar highways" —
 *    the paths the language actually uses heavily.
 *
 * Each bent edge = 2 line segments = 4 vertices in the LineSegments buffer.
 */
export class EdgeRenderer {
  /** Base edges (all transitions, dim) */
  lines: THREE.LineSegments;
  /** Highway edges (high-prob transitions, bright) */
  highways: THREE.LineSegments;

  private nodes: TrieNodeData[] = [];
  private nodeMap = new Map<number, TrieNodeData>();
  private edges: TrieEdge[] = [];

  // Base layer buffers
  private posBuffer: Float32Array;
  private colorBuffer: Float32Array;
  private maxSegments: number;

  // Highway layer buffers
  private hwPosBuffer: Float32Array;
  private hwColorBuffer: Float32Array;
  private hwMaxSegments: number;

  constructor(maxEdges: number) {
    // Base layer: 2 segments per edge
    this.maxSegments = maxEdges * 2;
    this.posBuffer = new Float32Array(this.maxSegments * 6);
    this.colorBuffer = new Float32Array(this.maxSegments * 6);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.posBuffer, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(this.colorBuffer, 3));
    geo.setDrawRange(0, 0);

    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.12,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.lines = new THREE.LineSegments(geo, mat);
    this.lines.frustumCulled = false;

    // Highway layer: at most ~30% of edges are highways
    this.hwMaxSegments = Math.ceil(maxEdges * 0.6);
    this.hwPosBuffer = new Float32Array(this.hwMaxSegments * 6);
    this.hwColorBuffer = new Float32Array(this.hwMaxSegments * 6);

    const hwGeo = new THREE.BufferGeometry();
    hwGeo.setAttribute("position", new THREE.BufferAttribute(this.hwPosBuffer, 3));
    hwGeo.setAttribute("color", new THREE.BufferAttribute(this.hwColorBuffer, 3));
    hwGeo.setDrawRange(0, 0);

    const hwMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      linewidth: 1, // note: linewidth >1 only works on some systems
    });

    this.highways = new THREE.LineSegments(hwGeo, hwMat);
    this.highways.frustumCulled = false;
  }

  setData(nodes: TrieNodeData[], edges: TrieEdge[]): void {
    this.nodes = nodes;
    this.edges = edges;
    this.nodeMap.clear();
    for (const n of nodes) this.nodeMap.set(n.id, n);
  }

  update(
    filter: FilterState,
    layout: RadialLayout,
    lod: LODState,
  ): void {
    let vertIdx = 0;
    let colorIdx = 0;
    let segmentCount = 0;

    let hwVertIdx = 0;
    let hwColorIdx = 0;
    let hwSegmentCount = 0;

    const tmpColor = new THREE.Color();

    for (const edge of this.edges) {
      if (segmentCount >= this.maxSegments) break;

      const srcLayout = layout.nodes.get(edge.source);
      const tgtLayout = layout.nodes.get(edge.target);
      if (!srcLayout || !tgtLayout) continue;

      if (!isNodeVisible(srcLayout, lod)) continue;
      if (!isNodeVisible(tgtLayout, lod)) continue;

      // Intermediate bend-point: lerp between parent's sphere and child
      // Short edges (high-prob) bend near parent (existing behavior).
      // Long edges (rare) pull the bend partway toward child for smoother curves.
      const cx = tgtLayout.x;
      const cy = tgtLayout.y;
      const cz = tgtLayout.z;
      const cLen = Math.sqrt(cx * cx + cy * cy + cz * cz);
      const invCLen = cLen > 0 ? 1 / cLen : 0;
      const dirX = cx * invCLen;
      const dirY = cy * invCLen;
      const dirZ = cz * invCLen;

      const parentR = srcLayout.sphereRadius;
      const edgeLen = tgtLayout.sphereRadius - parentR;
      // t=0 → bend at parent sphere (short edge), t→0.4 → bend partway to child (long edge)
      const bendT = Math.min(0.4, Math.max(0, (edgeLen - 20) / 200));
      const bendR = parentR + edgeLen * bendT;
      const midX = dirX * bendR;
      const midY = dirY * bendR;
      const midZ = dirZ * bendR;

      // Transition probability
      const tgtNode = this.nodeMap.get(edge.target);
      const srcNode = this.nodeMap.get(edge.source);

      let transitionProb = 0.5;
      if (srcNode?.transitionProbs && tgtNode) {
        transitionProb = srcNode.transitionProbs[tgtNode.phoneme] ?? 0.05;
      }

      // ── Base layer: brightness ∝ sqrt(probability) ──
      // sqrt compresses the range so even rare edges are faintly visible
      const probScale = Math.sqrt(Math.min(1, transitionProb * 4));

      if (tgtNode) {
        const hsl = blendNodeColor(tgtNode);
        tmpColor.setHex(hslToHex(hsl.h, hsl.s, hsl.l * 0.35 * probScale));
      } else {
        tmpColor.setRGB(0.05, 0.05, 0.05);
      }
      const cr = tmpColor.r;
      const cg = tmpColor.g;
      const cb = tmpColor.b;

      // Segment 1: parent → intermediate
      this.posBuffer[vertIdx++] = srcLayout.x;
      this.posBuffer[vertIdx++] = srcLayout.y;
      this.posBuffer[vertIdx++] = srcLayout.z;
      this.posBuffer[vertIdx++] = midX;
      this.posBuffer[vertIdx++] = midY;
      this.posBuffer[vertIdx++] = midZ;

      this.colorBuffer[colorIdx++] = cr * 0.5;
      this.colorBuffer[colorIdx++] = cg * 0.5;
      this.colorBuffer[colorIdx++] = cb * 0.5;
      this.colorBuffer[colorIdx++] = cr;
      this.colorBuffer[colorIdx++] = cg;
      this.colorBuffer[colorIdx++] = cb;

      // Segment 2: intermediate → child
      this.posBuffer[vertIdx++] = midX;
      this.posBuffer[vertIdx++] = midY;
      this.posBuffer[vertIdx++] = midZ;
      this.posBuffer[vertIdx++] = tgtLayout.x;
      this.posBuffer[vertIdx++] = tgtLayout.y;
      this.posBuffer[vertIdx++] = tgtLayout.z;

      this.colorBuffer[colorIdx++] = cr;
      this.colorBuffer[colorIdx++] = cg;
      this.colorBuffer[colorIdx++] = cb;
      this.colorBuffer[colorIdx++] = cr;
      this.colorBuffer[colorIdx++] = cg;
      this.colorBuffer[colorIdx++] = cb;

      segmentCount += 2;

      // ── Highway layer: only high-probability transitions ──
      if (transitionProb >= HIGHWAY_THRESHOLD && hwSegmentCount + 2 <= this.hwMaxSegments) {
        // Highway brightness: stronger scaling, reaches full brightness at ~35%
        const hwBright = Math.min(1, (transitionProb - HIGHWAY_THRESHOLD) / 0.25 * 0.7 + 0.3);

        if (tgtNode) {
          const hsl = blendNodeColor(tgtNode);
          tmpColor.setHex(hslToHex(hsl.h, Math.min(1, hsl.s + 0.15), Math.min(0.8, hsl.l * 0.7 * hwBright)));
        }
        const hr = tmpColor.r;
        const hg = tmpColor.g;
        const hb = tmpColor.b;

        // Same geometry as base, but brighter
        this.hwPosBuffer[hwVertIdx++] = srcLayout.x;
        this.hwPosBuffer[hwVertIdx++] = srcLayout.y;
        this.hwPosBuffer[hwVertIdx++] = srcLayout.z;
        this.hwPosBuffer[hwVertIdx++] = midX;
        this.hwPosBuffer[hwVertIdx++] = midY;
        this.hwPosBuffer[hwVertIdx++] = midZ;

        this.hwColorBuffer[hwColorIdx++] = hr * 0.6;
        this.hwColorBuffer[hwColorIdx++] = hg * 0.6;
        this.hwColorBuffer[hwColorIdx++] = hb * 0.6;
        this.hwColorBuffer[hwColorIdx++] = hr;
        this.hwColorBuffer[hwColorIdx++] = hg;
        this.hwColorBuffer[hwColorIdx++] = hb;

        this.hwPosBuffer[hwVertIdx++] = midX;
        this.hwPosBuffer[hwVertIdx++] = midY;
        this.hwPosBuffer[hwVertIdx++] = midZ;
        this.hwPosBuffer[hwVertIdx++] = tgtLayout.x;
        this.hwPosBuffer[hwVertIdx++] = tgtLayout.y;
        this.hwPosBuffer[hwVertIdx++] = tgtLayout.z;

        this.hwColorBuffer[hwColorIdx++] = hr;
        this.hwColorBuffer[hwColorIdx++] = hg;
        this.hwColorBuffer[hwColorIdx++] = hb;
        this.hwColorBuffer[hwColorIdx++] = hr;
        this.hwColorBuffer[hwColorIdx++] = hg;
        this.hwColorBuffer[hwColorIdx++] = hb;

        hwSegmentCount += 2;
      }
    }

    const geo = this.lines.geometry;
    (geo.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (geo.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    geo.setDrawRange(0, segmentCount * 2);

    const hwGeo = this.highways.geometry;
    (hwGeo.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (hwGeo.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    hwGeo.setDrawRange(0, hwSegmentCount * 2);
  }
}

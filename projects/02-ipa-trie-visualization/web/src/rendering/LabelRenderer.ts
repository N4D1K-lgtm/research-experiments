import * as THREE from "three";
import type { TrieNodeData, FilterState } from "../types";
import type { RadialLayout } from "../layout/radialLayout";
import type { LODState } from "../layout/lod";
import { isNodeVisible } from "../layout/lod";
import { getRoleColor } from "../color/languagePalette";
import { nodeEntropy } from "../layout/entropy";

const MAX_LABELS = 80;

/** Terminal nodes: English word as primary label */
const FONT_WORD_PRIMARY = "bold 13px 'Inter', 'Noto Sans', sans-serif";
/** Terminal nodes: IPA path as secondary label */
const FONT_IPA_SECONDARY = "500 10px 'Inter', 'Noto Sans', sans-serif";
/** Hub nodes: phoneme symbol */
const FONT_HUB = "bold 12px 'Inter', 'Noto Sans', sans-serif";
/** Motif annotation */
const FONT_MOTIF = "bold 9px 'Inter', 'Noto Sans', sans-serif";

/** Hub label thresholds */
const ENTROPY_HUB_THRESHOLD = 2.5; // bits
const HUB_MIN_CHILDREN = 5;

/** Distance fade: labels fully visible within this range, fade beyond */
const FADE_NEAR = 80;
const FADE_FAR = 350;

/**
 * HTML canvas overlay for crisp text labels at all zoom levels.
 *
 * Three-tier label strategy:
 * - Terminal nodes: English word (primary, white) + IPA path (secondary, role-colored)
 * - Hub nodes (entropy > 2.5, children > 5): phoneme symbol in role color
 * - Everything else: no label unless hovered
 */
export class LabelRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private nodeMap = new Map<number, TrieNodeData>();
  /** Cached full IPA path from root to each node (e.g. "kæt") */
  private pathCache = new Map<number, string>();
  private hoveredNodeId: number | null = null;
  private projVec = new THREE.Vector3();
  private camPos = new THREE.Vector3();

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  setData(nodes: TrieNodeData[]): void {
    this.nodeMap.clear();
    this.pathCache.clear();
    for (const n of nodes) this.nodeMap.set(n.id, n);
    // Build full paths by walking parentId to root
    for (const n of nodes) {
      if (n.depth === 0) {
        this.pathCache.set(n.id, "");
        continue;
      }
      const segments: string[] = [];
      let cur: TrieNodeData | undefined = n;
      while (cur && cur.depth > 0) {
        segments.push(cur.phoneme);
        cur = cur.parentId != null ? this.nodeMap.get(cur.parentId) : undefined;
      }
      segments.reverse();
      this.pathCache.set(n.id, segments.join(""));
    }
  }

  getPath(nodeId: number): string {
    return this.pathCache.get(nodeId) ?? "";
  }

  setHovered(nodeId: number | null): void {
    this.hoveredNodeId = nodeId;
  }

  resize(w: number, h: number): void {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  update(
    filter: FilterState,
    layout: RadialLayout,
    lod: LODState,
    camera: THREE.Camera,
  ): void {
    const ctx = this.ctx;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);

    // Get camera world position for distance-based fade
    this.camPos.setFromMatrixPosition(camera.matrixWorld);

    const enum LabelTier {
      Terminal = 0,
      Hub = 1,
      Hovered = 2,
    }

    interface Candidate {
      node: TrieNodeData;
      sx: number;
      sy: number;
      priority: number;
      tier: LabelTier;
      distanceFade: number;
    }

    const candidates: Candidate[] = [];

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

      // Project 3D to screen
      this.projVec.set(layoutNode.x, layoutNode.y, layoutNode.z);
      this.projVec.project(camera);

      // Behind camera
      if (this.projVec.z > 1) continue;

      const sx = (this.projVec.x + 1) / 2 * w;
      const sy = (1 - this.projVec.y) / 2 * h;

      if (sx < -50 || sx > w + 50 || sy < -50 || sy > h + 50) continue;

      // Camera distance fade
      const dx = layoutNode.x - this.camPos.x;
      const dy = layoutNode.y - this.camPos.y;
      const dz = layoutNode.z - this.camPos.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const distanceFade = 1 - Math.min(1, Math.max(0, (dist - FADE_NEAR) / (FADE_FAR - FADE_NEAR)));
      if (distanceFade <= 0.05) continue;

      // Hovered node always shows
      if (nodeId === this.hoveredNodeId) {
        candidates.push({ node, sx, sy, priority: Infinity, tier: LabelTier.Hovered, distanceFade: 1 });
        continue;
      }

      // Determine tier
      const isMotifHighlighted = filter.highlightMotifs.size > 0 &&
        node.motifs?.some((m) => filter.highlightMotifs.has(m));
      const motifBoost = isMotifHighlighted ? 1e5 : 0;

      if (node.isTerminal) {
        // Terminal: English word primary, IPA secondary
        candidates.push({
          node, sx, sy,
          priority: 1e6 + count + motifBoost,
          tier: LabelTier.Terminal,
          distanceFade,
        });
      } else {
        // Hub: high entropy + enough children
        const entropy = nodeEntropy(node);
        if (entropy > ENTROPY_HUB_THRESHOLD && node.childCount >= HUB_MIN_CHILDREN) {
          candidates.push({
            node, sx, sy,
            priority: 5e5 + count + motifBoost,
            tier: LabelTier.Hub,
            distanceFade,
          });
        }
        // Everything else: no label (skip)
      }
    }

    candidates.sort((a, b) => b.priority - a.priority);
    const selected = candidates.slice(0, MAX_LABELS);

    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";

    for (const { node, sx, sy, tier, distanceFade } of selected) {
      const roleColor = getRoleColor(node.phonologicalPosition);

      const isMotifHighlighted = filter.highlightMotifs.size > 0 &&
        node.motifs?.some((m) => filter.highlightMotifs.has(m));

      // Base alpha from distance fade
      let baseAlpha = distanceFade;
      if (isMotifHighlighted) {
        baseAlpha = 1.0;
      } else if (filter.highlightMotifs.size > 0) {
        baseAlpha *= 0.3;
      }

      if (tier === LabelTier.Terminal || tier === LabelTier.Hovered) {
        // ── Terminal: English word (primary) + IPA path (secondary) ──
        const englishWords = node.words?.["en_US"];
        if (englishWords && englishWords.length > 0) {
          // Primary: English word in white
          ctx.font = FONT_WORD_PRIMARY;
          ctx.fillStyle = "#e8e8f0";
          ctx.globalAlpha = Math.min(0.95, baseAlpha * 0.95);
          const wordLabel = englishWords.slice(0, 2).join(", ");
          const more = (node.terminalCounts?.["en_US"] ?? englishWords.length) > 2 ? " ..." : "";
          ctx.fillText(wordLabel + more, sx, sy - 10);

          // Secondary: IPA path in role color, smaller, below
          ctx.font = FONT_IPA_SECONDARY;
          ctx.fillStyle = roleColor;
          ctx.globalAlpha = Math.min(0.7, baseAlpha * 0.6);
          const fullPath = this.pathCache.get(node.id) ?? node.phoneme;
          ctx.fillText(`/${fullPath}/`, sx, sy + 4);
        } else {
          // No English words — fallback to IPA as primary
          ctx.font = FONT_WORD_PRIMARY;
          ctx.fillStyle = roleColor;
          ctx.globalAlpha = Math.min(0.9, baseAlpha * 0.85);
          const fullPath = this.pathCache.get(node.id) ?? node.phoneme;
          ctx.fillText(`/${fullPath}/`, sx, sy - 10);
        }
      } else if (tier === LabelTier.Hub) {
        // ── Hub: just the phoneme symbol ──
        ctx.font = FONT_HUB;
        ctx.fillStyle = roleColor;
        ctx.globalAlpha = Math.min(0.8, baseAlpha * 0.75);
        ctx.fillText(node.phoneme, sx, sy - 8);
      }

      // Motif annotation for highlighted nodes
      if (isMotifHighlighted && node.motifs) {
        const activeMotifs = node.motifs.filter((m) => filter.highlightMotifs.has(m));
        if (activeMotifs.length > 0) {
          ctx.font = FONT_MOTIF;
          ctx.fillStyle = "#ffcc00";
          ctx.globalAlpha = 0.8;
          ctx.fillText(activeMotifs[0], sx, sy - 24);
        }
      }
    }

    ctx.globalAlpha = 1.0;
  }
}

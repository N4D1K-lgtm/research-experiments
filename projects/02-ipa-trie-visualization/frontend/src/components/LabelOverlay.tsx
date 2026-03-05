import { useRef, useEffect, useCallback } from "react";
import { useThree, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { RenderNode, FilterState } from "../types/trie";
import type { RadialLayout } from "../utils/radialLayout";
import type { LODState } from "../utils/lod";
import { isNodeVisible } from "../utils/lod";
import { getRoleColor } from "../utils/languagePalette";
import { nodeEntropy } from "../utils/entropy";

const MAX_LABELS = 80;
const FONT_WORD_PRIMARY = "bold 13px 'Inter', 'Noto Sans', sans-serif";
const FONT_IPA_SECONDARY = "500 10px 'Inter', 'Noto Sans', sans-serif";
const FONT_HUB = "bold 12px 'Inter', 'Noto Sans', sans-serif";
const FONT_MOTIF = "bold 9px 'Inter', 'Noto Sans', sans-serif";
const ENTROPY_HUB_THRESHOLD = 2.5;
const HUB_MIN_CHILDREN = 5;
const FADE_NEAR = 80;
const FADE_FAR = 350;

interface Props {
  nodeMap: Map<number, RenderNode>;
  layout: RadialLayout | null;
  lod: LODState;
  filter: FilterState;
  hoveredNodeId: number | null;
}

export function LabelOverlay({ nodeMap, layout, lod, filter, hoveredNodeId }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { camera, size } = useThree();
  const pathCache = useRef(new Map<number, string>());
  const lastUpdateRef = useRef(0);
  const projVec = useRef(new THREE.Vector3());
  const camPos = useRef(new THREE.Vector3());

  // Build path cache
  useEffect(() => {
    pathCache.current.clear();
    for (const [id, node] of nodeMap) {
      if (node.depth === 0) {
        pathCache.current.set(id, "");
        continue;
      }
      const segments: string[] = [];
      let cur: RenderNode | undefined = node;
      while (cur && cur.depth > 0) {
        segments.push(cur.phoneme);
        cur = cur.parentId != null ? nodeMap.get(cur.parentId) : undefined;
      }
      segments.reverse();
      pathCache.current.set(id, segments.join(""));
    }
  }, [nodeMap]);

  // Create canvas
  useEffect(() => {
    const existing = document.getElementById("r3f-label-canvas") as HTMLCanvasElement;
    if (existing) {
      canvasRef.current = existing;
    } else {
      const canvas = document.createElement("canvas");
      canvas.id = "r3f-label-canvas";
      canvas.style.cssText =
        "position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:10";
      document.body.appendChild(canvas);
      canvasRef.current = canvas;
    }
    return () => {
      const el = document.getElementById("r3f-label-canvas");
      el?.remove();
    };
  }, []);

  // Resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.width * dpr;
    canvas.height = size.height * dpr;
    canvas.style.width = `${size.width}px`;
    canvas.style.height = `${size.height}px`;
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }, [size]);

  useFrame(() => {
    const now = performance.now();
    if (now - lastUpdateRef.current < 200) return;
    lastUpdateRef.current = now;

    const canvas = canvasRef.current;
    if (!canvas || !layout) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);

    camPos.current.setFromMatrixPosition(camera.matrixWorld);

    const enum LabelTier {
      Terminal = 0,
      Hub = 1,
      Hovered = 2,
    }

    interface Candidate {
      node: RenderNode;
      sx: number;
      sy: number;
      priority: number;
      tier: LabelTier;
      distanceFade: number;
    }

    const candidates: Candidate[] = [];

    for (const [nodeId, layoutNode] of layout.nodes) {
      if (nodeId === 0) continue;
      const node = nodeMap.get(nodeId);
      if (!node) continue;
      if (node.depth > filter.maxDepth) continue;
      if (filter.terminalsOnly && !node.isTerminal) continue;
      if (
        filter.positionFilter.size > 0 &&
        !filter.positionFilter.has(node.phonologicalPosition)
      )
        continue;
      if (node.totalCount < filter.minFrequency || node.totalCount === 0) continue;
      if (!isNodeVisible(layoutNode, lod)) continue;

      projVec.current.set(layoutNode.x, layoutNode.y, layoutNode.z);
      projVec.current.project(camera);
      if (projVec.current.z > 1) continue;

      const sx = ((projVec.current.x + 1) / 2) * w;
      const sy = ((1 - projVec.current.y) / 2) * h;
      if (sx < -50 || sx > w + 50 || sy < -50 || sy > h + 50) continue;

      const dx = layoutNode.x - camPos.current.x;
      const dy = layoutNode.y - camPos.current.y;
      const dz = layoutNode.z - camPos.current.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const distanceFade =
        1 - Math.min(1, Math.max(0, (dist - FADE_NEAR) / (FADE_FAR - FADE_NEAR)));
      if (distanceFade <= 0.05) continue;

      if (nodeId === hoveredNodeId) {
        candidates.push({
          node,
          sx,
          sy,
          priority: Infinity,
          tier: LabelTier.Hovered,
          distanceFade: 1,
        });
        continue;
      }

      const isMotifHighlighted =
        filter.highlightMotifs.size > 0 &&
        node.motifs?.some((m) => filter.highlightMotifs.has(m));
      const motifBoost = isMotifHighlighted ? 1e5 : 0;

      if (node.isTerminal) {
        candidates.push({
          node,
          sx,
          sy,
          priority: 1e6 + node.totalCount + motifBoost,
          tier: LabelTier.Terminal,
          distanceFade,
        });
      } else {
        const entropy = nodeEntropy(node);
        if (entropy > ENTROPY_HUB_THRESHOLD && node.childCount >= HUB_MIN_CHILDREN) {
          candidates.push({
            node,
            sx,
            sy,
            priority: 5e5 + node.totalCount + motifBoost,
            tier: LabelTier.Hub,
            distanceFade,
          });
        }
      }
    }

    candidates.sort((a, b) => b.priority - a.priority);
    const selected = candidates.slice(0, MAX_LABELS);

    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";

    for (const { node, sx, sy, tier, distanceFade } of selected) {
      const roleColor = getRoleColor(node.phonologicalPosition);

      const isMotifHighlighted =
        filter.highlightMotifs.size > 0 &&
        node.motifs?.some((m) => filter.highlightMotifs.has(m));

      let baseAlpha = distanceFade;
      if (isMotifHighlighted) {
        baseAlpha = 1.0;
      } else if (filter.highlightMotifs.size > 0) {
        baseAlpha *= 0.3;
      }

      if (tier === LabelTier.Terminal || tier === LabelTier.Hovered) {
        const englishWords = node.words?.["en_US"];
        if (englishWords && englishWords.length > 0) {
          ctx.font = FONT_WORD_PRIMARY;
          ctx.fillStyle = "#e8e8f0";
          ctx.globalAlpha = Math.min(0.95, baseAlpha * 0.95);
          const wordLabel = englishWords.slice(0, 2).join(", ");
          const more =
            (node.terminalCounts?.["en_US"] ?? englishWords.length) > 2 ? " ..." : "";
          ctx.fillText(wordLabel + more, sx, sy - 10);

          ctx.font = FONT_IPA_SECONDARY;
          ctx.fillStyle = roleColor;
          ctx.globalAlpha = Math.min(0.7, baseAlpha * 0.6);
          const fullPath = pathCache.current.get(node.id) ?? node.phoneme;
          ctx.fillText(`/${fullPath}/`, sx, sy + 4);
        } else {
          ctx.font = FONT_WORD_PRIMARY;
          ctx.fillStyle = roleColor;
          ctx.globalAlpha = Math.min(0.9, baseAlpha * 0.85);
          const fullPath = pathCache.current.get(node.id) ?? node.phoneme;
          ctx.fillText(`/${fullPath}/`, sx, sy - 10);
        }
      } else if (tier === LabelTier.Hub) {
        ctx.font = FONT_HUB;
        ctx.fillStyle = roleColor;
        ctx.globalAlpha = Math.min(0.8, baseAlpha * 0.75);
        ctx.fillText(node.phoneme, sx, sy - 8);
      }

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
  });

  return null;
}

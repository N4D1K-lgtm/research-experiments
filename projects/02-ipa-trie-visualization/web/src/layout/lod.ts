import type { LayoutNode } from "./radialLayout";

export interface LODState {
  visibleMaxDepth: number;
  fadeDepth: number;
}

/**
 * Compute visible depth from camera distance.
 * Closer camera → more depth levels visible.
 *
 * At the default camera distance (~140), show depth 5.
 * Zooming in reveals progressively deeper layers.
 * Zooming out collapses to depth 2-3.
 */
export function computeLOD(
  cameraDistance: number,
  maxDepth: number,
  baseDistance = 140,
): LODState {
  // Linear mapping: every halving of distance adds ~2 depth levels
  // At baseDistance: depth 5. At baseDistance/2: depth 7. At baseDistance*2: depth 3.
  const ratio = baseDistance / Math.max(cameraDistance, 1);
  const visibleMaxDepth = Math.min(
    maxDepth,
    Math.max(3, Math.round(5 + 3 * Math.log2(Math.max(ratio, 0.25)))),
  );
  return {
    visibleMaxDepth,
    fadeDepth: visibleMaxDepth,
  };
}

/**
 * Check if a node should be visible given LOD.
 * (Frustum culling is handled by Three.js; this is depth-only.)
 */
export function isNodeVisible(node: LayoutNode, lod: LODState): boolean {
  if (node.depth === 0) return true;
  return node.depth <= lod.visibleMaxDepth;
}

/**
 * Alpha for LOD fade-in at boundary depth.
 */
export function getNodeAlpha(node: LayoutNode, lod: LODState): number {
  if (node.depth < lod.fadeDepth) return 1.0;
  if (node.depth === lod.fadeDepth) return 0.6;
  return 0.0;
}

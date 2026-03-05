export interface LODState {
  visibleMaxDepth: number;
  fadeDepth: number;
}

export function computeLOD(
  cameraDistance: number,
  maxDepth: number,
  baseDistance = 140,
): LODState {
  const ratio = baseDistance / Math.max(cameraDistance, 1);
  const visibleMaxDepth = Math.min(
    maxDepth,
    Math.max(3, Math.round(5 + 3 * Math.log2(Math.max(ratio, 0.25)))),
  );
  return { visibleMaxDepth, fadeDepth: visibleMaxDepth };
}

export interface LayoutNode {
  id: number;
  depth: number;
  x: number;
  y: number;
  z: number;
  sphereRadius: number;
  dirX: number;
  dirY: number;
  dirZ: number;
  subtreeWeight: number;
  children: number[];
  parentId: number | null;
}

export function isNodeVisible(node: LayoutNode, lod: LODState): boolean {
  if (node.depth === 0) return true;
  return node.depth <= lod.visibleMaxDepth;
}

export function getNodeAlpha(node: LayoutNode, lod: LODState): number {
  if (node.depth < lod.fadeDepth) return 1.0;
  if (node.depth === lod.fadeDepth) return 0.6;
  return 0.0;
}

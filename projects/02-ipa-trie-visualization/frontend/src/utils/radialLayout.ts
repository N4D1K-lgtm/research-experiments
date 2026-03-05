import type { RenderNode, TrieEdge, FilterState } from "../types/trie";
import type { LayoutNode } from "./lod";

export interface RadialLayout {
  nodes: Map<number, LayoutNode>;
  maxDepth: number;
}

const BASE_EDGE = 40;
const DEPTH_DECAY = 0.88;
const PROB_SCALE = 2.5;
const MIN_EDGE = 8;
const MAX_EDGE = 120;
const ANGULAR_PAD = 0.005;
const PHI_MIN = Math.PI * 0.1;
const PHI_MAX = Math.PI * 0.9;
const PROB_BLEND = 0.6;
const WEIGHT_BLEND = 0.4;

function edgeLength(depth: number, prob: number): number {
  const raw =
    BASE_EDGE *
    Math.pow(DEPTH_DECAY, depth) *
    (1 / Math.sqrt(Math.max(0.001, prob * PROB_SCALE)));
  return Math.min(MAX_EDGE, Math.max(MIN_EDGE, raw));
}

function sphericalToCartesian(
  r: number,
  theta: number,
  phi: number,
): { x: number; y: number; z: number } {
  const sinPhi = Math.sin(phi);
  return {
    x: r * sinPhi * Math.cos(theta),
    y: r * Math.cos(phi),
    z: r * sinPhi * Math.sin(theta),
  };
}

function computeBlendedFractions(
  kidIds: number[],
  parentId: number,
  nodeMap: Map<number, RenderNode>,
  weights: Map<number, number>,
  _parentWeight: number,
): number[] {
  if (kidIds.length === 0) return [];
  const parent = nodeMap.get(parentId);
  const probs: number[] = [];
  const kidWeights: number[] = [];
  let totalProb = 0;
  let totalWeight = 0;

  for (const kidId of kidIds) {
    const kid = nodeMap.get(kidId);
    const w = weights.get(kidId) ?? 0;
    kidWeights.push(w);
    totalWeight += w;
    let p = 0.5 / kidIds.length;
    if (parent?.transitionProbs && kid) {
      p = parent.transitionProbs[kid.phoneme] ?? 0.05;
    }
    probs.push(p);
    totalProb += p;
  }

  const safeWeight = Math.max(totalWeight, 1);
  const safeProb = Math.max(totalProb, 0.001);
  const fractions: number[] = [];
  let totalBlend = 0;

  for (let i = 0; i < kidIds.length; i++) {
    const probFrac = probs[i] / safeProb;
    const weightFrac = kidWeights[i] / safeWeight;
    const blended = PROB_BLEND * probFrac + WEIGHT_BLEND * weightFrac;
    fractions.push(blended);
    totalBlend += blended;
  }

  if (totalBlend > 0) {
    for (let i = 0; i < fractions.length; i++) {
      fractions[i] /= totalBlend;
    }
  }
  return fractions;
}

export function computeRadialLayout(
  nodesArray: RenderNode[],
  edges: TrieEdge[],
  _filter: FilterState,
): RadialLayout {
  const nodeMap = new Map<number, RenderNode>();
  for (const n of nodesArray) nodeMap.set(n.id, n);

  // Build adjacency
  const adjacency = new Map<number, number[]>();
  for (const node of nodesArray) adjacency.set(node.id, []);
  for (const edge of edges) adjacency.get(edge.source)?.push(edge.target);

  // Compute subtree weights bottom-up
  const weights = new Map<number, number>();
  const order: number[] = [];
  const queue = [0];
  let head = 0;
  while (head < queue.length) {
    const id = queue[head++];
    order.push(id);
    const kids = adjacency.get(id);
    if (kids) for (const kid of kids) queue.push(kid);
  }

  for (let i = order.length - 1; i >= 0; i--) {
    const id = order[i];
    const node = nodeMap.get(id);
    let selfWeight = node ? Math.max(1, node.totalCount) : 0;
    let childWeight = 0;
    const kids = adjacency.get(id);
    if (kids) for (const kid of kids) childWeight += weights.get(kid) ?? 0;
    weights.set(id, selfWeight + childWeight);
  }

  const layoutNodes = new Map<number, LayoutNode>();
  let maxDepth = 0;

  layoutNodes.set(0, {
    id: 0,
    depth: 0,
    x: 0,
    y: 0,
    z: 0,
    sphereRadius: 0,
    dirX: 0,
    dirY: 1,
    dirZ: 0,
    subtreeWeight: weights.get(0) ?? 0,
    children: adjacency.get(0) ?? [],
    parentId: null,
  });

  interface QueueItem {
    id: number;
    thetaMin: number;
    thetaMax: number;
    phiMin: number;
    phiMax: number;
    parentRadius: number;
  }

  const bfsQueue: QueueItem[] = [];
  const rootKids = adjacency.get(0) ?? [];
  const rootWeight = weights.get(0) ?? 1;
  const activeRootKids = rootKids.filter((k) => (weights.get(k) ?? 0) > 0);
  const rootProbs = computeBlendedFractions(activeRootKids, 0, nodeMap, weights, rootWeight);

  let thetaOffset = 0;
  for (let i = 0; i < activeRootKids.length; i++) {
    const wedge = rootProbs[i] * Math.PI * 2;
    bfsQueue.push({
      id: activeRootKids[i],
      thetaMin: thetaOffset,
      thetaMax: thetaOffset + wedge,
      phiMin: PHI_MIN,
      phiMax: PHI_MAX,
      parentRadius: 0,
    });
    thetaOffset += wedge;
  }

  let bfsHead = 0;
  while (bfsHead < bfsQueue.length) {
    const { id, thetaMin, thetaMax, phiMin, phiMax, parentRadius } = bfsQueue[bfsHead++];
    const node = nodeMap.get(id);
    if (!node) continue;

    const depth = node.depth;
    if (depth > maxDepth) maxDepth = depth;

    let prob = 0.5;
    if (node.parentId != null) {
      const parent = nodeMap.get(node.parentId);
      if (parent?.transitionProbs) {
        prob = parent.transitionProbs[node.phoneme] ?? 0.05;
      }
    }

    const edge = edgeLength(depth, prob);
    const r = parentRadius + edge;
    const theta = (thetaMin + thetaMax) / 2;
    const phi = (phiMin + phiMax) / 2;
    const pos = sphericalToCartesian(r, theta, phi);

    const len = Math.sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
    const invLen = len > 0 ? 1 / len : 0;

    const kids = adjacency.get(id) ?? [];
    layoutNodes.set(id, {
      id,
      depth,
      x: pos.x,
      y: pos.y,
      z: pos.z,
      sphereRadius: r,
      dirX: pos.x * invLen,
      dirY: pos.y * invLen,
      dirZ: pos.z * invLen,
      subtreeWeight: weights.get(id) ?? 0,
      children: kids,
      parentId: node.parentId,
    });

    const activeKids = kids.filter((k) => (weights.get(k) ?? 0) > 0);
    if (activeKids.length === 0) continue;

    const myWeight = weights.get(id) ?? 1;
    const thetaRange = thetaMax - thetaMin;
    const phiRange = phiMax - phiMin;
    const fractions = computeBlendedFractions(activeKids, id, nodeMap, weights, myWeight);

    if (thetaRange >= phiRange) {
      const totalPad = activeKids.length > 1 ? ANGULAR_PAD * (activeKids.length - 1) : 0;
      const available = Math.max(0, thetaRange - totalPad);
      let offset = thetaMin;
      for (let i = 0; i < activeKids.length; i++) {
        const wedge = fractions[i] * available;
        bfsQueue.push({
          id: activeKids[i],
          thetaMin: offset,
          thetaMax: offset + wedge,
          phiMin,
          phiMax,
          parentRadius: r,
        });
        offset += wedge + ANGULAR_PAD;
      }
    } else {
      const totalPad = activeKids.length > 1 ? ANGULAR_PAD * (activeKids.length - 1) : 0;
      const available = Math.max(0, phiRange - totalPad);
      let offset = phiMin;
      for (let i = 0; i < activeKids.length; i++) {
        const wedge = fractions[i] * available;
        bfsQueue.push({
          id: activeKids[i],
          thetaMin,
          thetaMax,
          phiMin: offset,
          phiMax: offset + wedge,
          parentRadius: r,
        });
        offset += wedge + ANGULAR_PAD;
      }
    }
  }

  return { nodes: layoutNodes, maxDepth };
}

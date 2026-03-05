import type { TrieNodeData, TrieEdge, FilterState } from "../types";

export interface LayoutNode {
  id: number;
  depth: number;
  x: number;
  y: number;
  z: number;
  /** Distance from origin to this node (variable per-node, not per-shell) */
  sphereRadius: number;
  /** Normalized direction from origin to this node */
  dirX: number;
  dirY: number;
  dirZ: number;
  subtreeWeight: number;
  children: number[];
  parentId: number | null;
}

export interface RadialLayout {
  nodes: Map<number, LayoutNode>;
  maxDepth: number;
}

/** Probability-weighted edge length constants */
const BASE_EDGE = 40;
const DEPTH_DECAY = 0.88;
const PROB_SCALE = 2.5;
const MIN_EDGE = 8;
const MAX_EDGE = 120;

/** Angular layout constants */
const ANGULAR_PAD = 0.005; // radians padding between siblings
const PHI_MIN = Math.PI * 0.1; // avoid poles
const PHI_MAX = Math.PI * 0.9;

/** Blend ratio for angular allocation: 60% prob + 40% subtree weight */
const PROB_BLEND = 0.6;
const WEIGHT_BLEND = 0.4;

/**
 * Edge length: short for high-probability transitions, long for rare ones.
 * Deeper levels compress via exponential decay.
 */
function edgeLength(depth: number, prob: number): number {
  const raw = BASE_EDGE * Math.pow(DEPTH_DECAY, depth) * (1 / Math.sqrt(Math.max(0.001, prob * PROB_SCALE)));
  return Math.min(MAX_EDGE, Math.max(MIN_EDGE, raw));
}

/**
 * Build adjacency list (parent → children) from edges.
 */
function buildAdjacency(
  nodes: TrieNodeData[],
  edges: TrieEdge[],
): Map<number, number[]> {
  const children = new Map<number, number[]>();
  for (const node of nodes) {
    children.set(node.id, []);
  }
  for (const edge of edges) {
    children.get(edge.source)?.push(edge.target);
  }
  return children;
}

/**
 * Compute subtree weights bottom-up via reversed BFS.
 * Weight = sum of filtered counts for this node + all descendants.
 */
function computeSubtreeWeights(
  nodes: TrieNodeData[],
  adjacency: Map<number, number[]>,
  _filter: FilterState,
): Map<number, number> {
  const weights = new Map<number, number>();
  const nodeMap = new Map<number, TrieNodeData>();
  for (const n of nodes) nodeMap.set(n.id, n);

  // BFS level order
  const order: number[] = [];
  const queue = [0];
  let head = 0;
  while (head < queue.length) {
    const id = queue[head++];
    order.push(id);
    const kids = adjacency.get(id);
    if (kids) for (const kid of kids) queue.push(kid);
  }

  // Reverse: leaves first
  for (let i = order.length - 1; i >= 0; i--) {
    const id = order[i];
    const node = nodeMap.get(id);
    let selfWeight = node ? node.totalCount : 0;
    selfWeight = selfWeight > 0 ? Math.max(1, selfWeight) : 0;

    let childWeight = 0;
    const kids = adjacency.get(id);
    if (kids) {
      for (const kid of kids) childWeight += weights.get(kid) ?? 0;
    }
    weights.set(id, selfWeight + childWeight);
  }

  return weights;
}

/** Convert spherical (r, theta, phi) to Cartesian (x, y, z) */
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

/**
 * Compute 3D radial layout with probability-weighted edge lengths.
 *
 * Common paths (high transition probability) get short edges → dense compact clusters.
 * Rare paths get long edges → pushed to periphery.
 * Angular allocation blends 60% transition probability + 40% subtree weight,
 * so high-probability children occupy the central part of the angular patch.
 *
 * Still single-pass BFS, O(n).
 */
export function computeRadialLayout(
  nodes: TrieNodeData[],
  edges: TrieEdge[],
  filter: FilterState,
): RadialLayout {
  const adjacency = buildAdjacency(nodes, edges);
  const weights = computeSubtreeWeights(nodes, adjacency, filter);
  const nodeMap = new Map<number, TrieNodeData>();
  for (const n of nodes) nodeMap.set(n.id, n);

  const layoutNodes = new Map<number, LayoutNode>();
  let maxDepth = 0;

  // Root at origin
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

  // BFS with angular patches + parent radius
  interface QueueItem {
    id: number;
    thetaMin: number;
    thetaMax: number;
    phiMin: number;
    phiMax: number;
    parentRadius: number;
  }

  const bfsQueue: QueueItem[] = [];

  // Root's children: split full theta range, full phi range
  const rootKids = adjacency.get(0) ?? [];
  const rootWeight = weights.get(0) ?? 1;

  // Filter to kids with weight
  const activeRootKids = rootKids.filter((k) => (weights.get(k) ?? 0) > 0);

  // For root children, compute blended fractions (prob + weight)
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

  let head = 0;
  while (head < bfsQueue.length) {
    const { id, thetaMin, thetaMax, phiMin, phiMax, parentRadius } = bfsQueue[head++];
    const node = nodeMap.get(id);
    if (!node) continue;

    const depth = node.depth;
    if (depth > maxDepth) maxDepth = depth;

    // Compute transition probability from parent
    let prob = 0.5;
    if (node.parentId != null) {
      const parent = nodeMap.get(node.parentId);
      if (parent?.transitionProbs) {
        prob = parent.transitionProbs[node.phoneme] ?? 0.05;
      }
    }

    // Variable edge length based on depth and probability
    const edge = edgeLength(depth, prob);
    const r = parentRadius + edge;

    const theta = (thetaMin + thetaMax) / 2;
    const phi = (phiMin + phiMax) / 2;
    const pos = sphericalToCartesian(r, theta, phi);

    // Normalized direction from origin
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

    // Subdivide patch among children
    const activeKids = kids.filter((k) => (weights.get(k) ?? 0) > 0);
    if (activeKids.length === 0) continue;

    const myWeight = weights.get(id) ?? 1;
    const thetaRange = thetaMax - thetaMin;
    const phiRange = phiMax - phiMin;

    // Blended fractions: 60% prob + 40% weight
    const fractions = computeBlendedFractions(activeKids, id, nodeMap, weights, myWeight);

    if (thetaRange >= phiRange) {
      // Split along theta
      const totalPad =
        activeKids.length > 1 ? ANGULAR_PAD * (activeKids.length - 1) : 0;
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
      // Split along phi
      const totalPad =
        activeKids.length > 1 ? ANGULAR_PAD * (activeKids.length - 1) : 0;
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

/**
 * Compute blended angular fractions for a set of siblings.
 * 60% transition probability + 40% subtree weight.
 * High-probability children get more angular space (central corridor).
 */
function computeBlendedFractions(
  kidIds: number[],
  parentId: number,
  nodeMap: Map<number, TrieNodeData>,
  weights: Map<number, number>,
  parentWeight: number,
): number[] {
  if (kidIds.length === 0) return [];

  const parent = nodeMap.get(parentId);
  const fractions: number[] = [];

  let totalProb = 0;
  let totalWeight = 0;
  const probs: number[] = [];
  const kidWeights: number[] = [];

  for (const kidId of kidIds) {
    const kid = nodeMap.get(kidId);
    const w = weights.get(kidId) ?? 0;
    kidWeights.push(w);
    totalWeight += w;

    let p = 0.5 / kidIds.length; // fallback: uniform
    if (parent?.transitionProbs && kid) {
      p = parent.transitionProbs[kid.phoneme] ?? 0.05;
    }
    probs.push(p);
    totalProb += p;
  }

  // Normalize each component, then blend
  const safeWeight = Math.max(totalWeight, 1);
  const safeProb = Math.max(totalProb, 0.001);

  let totalBlend = 0;
  for (let i = 0; i < kidIds.length; i++) {
    const probFrac = probs[i] / safeProb;
    const weightFrac = kidWeights[i] / safeWeight;
    const blended = PROB_BLEND * probFrac + WEIGHT_BLEND * weightFrac;
    fractions.push(blended);
    totalBlend += blended;
  }

  // Renormalize to sum to 1
  if (totalBlend > 0) {
    for (let i = 0; i < fractions.length; i++) {
      fractions[i] /= totalBlend;
    }
  }

  return fractions;
}

import type { TrieNodeData } from "../types";

/**
 * Compute entropy of transition distribution in bits.
 * 0 = deterministic (single continuation), log2(n) = uniform over n children.
 */
export function nodeEntropy(node: TrieNodeData): number {
  if (!node.transitionProbs) return 0;
  const probs = Object.values(node.transitionProbs);
  if (probs.length <= 1) return 0;
  let h = 0;
  for (const p of probs) {
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}

/**
 * Get P(this_node | parent) — how probable is this branch from its parent.
 * Requires a lookup map to resolve parentId.
 */
export function incomingProb(
  node: TrieNodeData,
  nodeMap: Map<number, TrieNodeData>,
): number {
  if (node.parentId == null) return 1;
  const parent = nodeMap.get(node.parentId);
  if (!parent?.transitionProbs) return 0.5;
  return parent.transitionProbs[node.phoneme] ?? 0.05;
}

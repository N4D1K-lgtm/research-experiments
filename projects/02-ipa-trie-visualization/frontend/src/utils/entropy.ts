import type { RenderNode } from "../types/trie";

/** Shannon entropy of transition distribution in bits. */
export function nodeEntropy(node: RenderNode): number {
  const probs = Object.values(node.transitionProbs);
  if (probs.length <= 1) return 0;
  let h = 0;
  for (const p of probs) {
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}

/** P(this_node | parent) from parent's transition probs. */
export function incomingProb(
  node: RenderNode,
  nodeMap: Map<number, RenderNode>,
): number {
  if (node.parentId == null) return 1;
  const parent = nodeMap.get(node.parentId);
  if (!parent?.transitionProbs) return 0.5;
  return parent.transitionProbs[node.phoneme] ?? 0.05;
}

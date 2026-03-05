export type PhonologicalPosition = "onset" | "nucleus" | "coda" | "mixed";

export interface TrieNodeData {
  id: number;
  phoneme: string;
  depth: number;
  parentId: number | null;
  counts: { language: string; count: number }[];
  totalCount: number;
  position: { x: number; y: number; z: number };
  color: string;
  hsl: { h: number; s: number; l: number };
  phonologicalPosition: string;
  isTerminal: boolean;
  childCount: number;
  weight: number;
  transitionProbs: { phoneme: string; probability: number }[];
  allophones: string[];
  terminalCounts: { language: string; count: number }[];
  words: { language: string; words: string[] }[];
}

export interface TrieEdge {
  source: number;
  target: number;
}

export interface Motif {
  sequence: string[];
  count: number;
  label: string;
}

export interface TrieMetadata {
  languages: string[];
  nodeCount: number;
  edgeCount: number;
  maxDepth: number;
  totalWords: number;
  terminalNodes: number;
  phonemeInventory: string[];
  onsetInventory: string[];
  codaInventory: string[];
  motifs: Motif[];
}

export interface DepthStats {
  depth: number;
  nodes: number;
  terminals: number;
  avgBranch: number;
  avgEntropy: number;
  maxEntropy: number;
}

export interface LanguageInfo {
  code: string;
  name: string;
  family: string;
  typology: string;
  iso6393: string;
}

export interface FilterState {
  maxDepth: number;
  minFrequency: number;
  terminalsOnly: boolean;
  highlightMotifs: Set<string>;
  positionFilter: Set<PhonologicalPosition>;
}

// Transformed node for rendering (with transition probs as record for fast lookup)
export interface RenderNode {
  id: number;
  phoneme: string;
  depth: number;
  parentId: number | null;
  totalCount: number;
  position: { x: number; y: number; z: number };
  color: string;
  hsl: { h: number; s: number; l: number };
  phonologicalPosition: PhonologicalPosition;
  isTerminal: boolean;
  childCount: number;
  weight: number;
  transitionProbs: Record<string, number>;
  allophones: string[];
  terminalCounts: Record<string, number>;
  words: Record<string, string[]>;
  motifs?: string[];
}

/** Convert GraphQL node to render-friendly format */
export function toRenderNode(n: TrieNodeData): RenderNode {
  const transitionProbs: Record<string, number> = {};
  for (const tp of n.transitionProbs ?? []) {
    transitionProbs[tp.phoneme] = tp.probability;
  }
  const terminalCounts: Record<string, number> = {};
  for (const tc of n.terminalCounts ?? []) {
    terminalCounts[tc.language] = tc.count;
  }
  const words: Record<string, string[]> = {};
  for (const w of n.words ?? []) {
    words[w.language] = w.words;
  }
  return {
    id: n.id,
    phoneme: n.phoneme,
    depth: n.depth,
    parentId: n.parentId,
    totalCount: n.totalCount,
    position: n.position,
    color: n.color,
    hsl: n.hsl,
    phonologicalPosition: n.phonologicalPosition as PhonologicalPosition,
    isTerminal: n.isTerminal,
    childCount: n.childCount,
    weight: n.weight,
    transitionProbs,
    allophones: n.allophones ?? [],
    terminalCounts,
    words,
  };
}

export type PhonologicalPosition = "onset" | "nucleus" | "coda" | "mixed";

export interface TrieNodeData {
  id: number;
  phoneme: string;
  depth: number;
  parentId: number | null;
  counts: Record<string, number>;
  totalCount: number;
  position: { x: number; y: number; z: number };
  color: string;
  hsl: { h: number; s: number; l: number };
  /** Phonological role: onset, nucleus, coda, or mixed */
  phonologicalPosition: PhonologicalPosition;
  isTerminal: boolean;
  /** Number of child branches from this node */
  childCount: number;
  /** Branching probabilities: P(next_phoneme | this_path) */
  transitionProbs?: Record<string, number>;
  /** Motif labels this node participates in */
  motifs?: string[];
  /** Surface allophone forms that collapsed to this phoneme */
  allophones?: string[];
  /** Per-language count of words that terminate at this node */
  terminalCounts?: Record<string, number>;
  /** Sample words per language (capped at 5) */
  words?: Record<string, string[]>;
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

export interface PhonologicalMeta {
  phonemeInventory: string[];
  onsetInventory: string[];
  codaInventory: string[];
  motifs: Motif[];
  transitionMatrix: Record<string, Record<string, number>>;
  allophoneContexts: Record<string, { before: string[]; after: string[] }>;
}

export interface TrieMetadata extends PhonologicalMeta {
  languages: string[];
  nodeCount: number;
  edgeCount: number;
  maxDepth: number;
  totalWords: number;
  terminalNodes: number;
}

export interface TrieData {
  metadata: TrieMetadata;
  nodes: TrieNodeData[];
  edges: TrieEdge[];
}

export interface FilterState {
  maxDepth: number;
  minFrequency: number;
  terminalsOnly: boolean;
  /** Motif labels to highlight (empty = no highlight) */
  highlightMotifs: Set<string>;
  /** Phonological positions to show (empty = show all) */
  positionFilter: Set<PhonologicalPosition>;
}

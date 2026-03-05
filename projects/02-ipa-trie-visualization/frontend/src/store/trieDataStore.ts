import { create } from "zustand";
import type { RenderNode, TrieEdge, TrieMetadata } from "../types/trie";

interface TrieDataState {
  metadata: TrieMetadata | null;
  nodes: Map<number, RenderNode>;
  edges: TrieEdge[];
  loadedDepths: Set<number>;
  loading: boolean;
  error: string | null;

  setMetadata: (metadata: TrieMetadata) => void;
  addNodes: (nodes: RenderNode[]) => void;
  addEdges: (edges: TrieEdge[]) => void;
  markDepthLoaded: (depth: number) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useTrieDataStore = create<TrieDataState>((set, get) => ({
  metadata: null,
  nodes: new Map(),
  edges: [],
  loadedDepths: new Set(),
  loading: false,
  error: null,

  setMetadata: (metadata) => set({ metadata }),

  addNodes: (newNodes) =>
    set((state) => {
      const nodes = new Map(state.nodes);
      for (const n of newNodes) {
        nodes.set(n.id, n);
      }
      return { nodes };
    }),

  addEdges: (newEdges) =>
    set((state) => {
      // Deduplicate by source-target pair
      const existing = new Set(state.edges.map((e) => `${e.source}-${e.target}`));
      const unique = newEdges.filter((e) => !existing.has(`${e.source}-${e.target}`));
      return { edges: [...state.edges, ...unique] };
    }),

  markDepthLoaded: (depth) =>
    set((state) => {
      const loadedDepths = new Set(state.loadedDepths);
      loadedDepths.add(depth);
      return { loadedDepths };
    }),

  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
}));

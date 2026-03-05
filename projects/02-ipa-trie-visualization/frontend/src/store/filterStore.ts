import { create } from "zustand";
import type { FilterState, PhonologicalPosition } from "../types/trie";

interface FilterStoreState extends FilterState {
  setMaxDepth: (d: number) => void;
  setMinFrequency: (f: number) => void;
  setTerminalsOnly: (v: boolean) => void;
  toggleMotif: (label: string) => void;
  togglePosition: (pos: PhonologicalPosition) => void;
  setMaxDepthLimit: (d: number) => void;
}

export const useFilterStore = create<FilterStoreState>((set) => ({
  maxDepth: 12,
  minFrequency: 50,
  terminalsOnly: false,
  highlightMotifs: new Set<string>(),
  positionFilter: new Set<PhonologicalPosition>(),

  setMaxDepth: (d) => set({ maxDepth: d }),
  setMinFrequency: (f) => set({ minFrequency: f }),
  setTerminalsOnly: (v) => set({ terminalsOnly: v }),

  toggleMotif: (label) =>
    set((s) => {
      const next = new Set(s.highlightMotifs);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return { highlightMotifs: next };
    }),

  togglePosition: (pos) =>
    set((s) => {
      const next = new Set(s.positionFilter);
      if (next.has(pos)) next.delete(pos);
      else next.add(pos);
      return { positionFilter: next };
    }),

  setMaxDepthLimit: (d) => set({ maxDepth: d }),
}));

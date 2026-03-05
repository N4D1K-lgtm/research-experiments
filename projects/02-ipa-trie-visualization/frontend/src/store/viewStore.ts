import { create } from "zustand";
import type { RenderNode } from "../types/trie";

export type StudyMode = "explorer" | "analysis" | "phonology" | "crossling";

interface ViewState {
  mode: StudyMode;
  setMode: (mode: StudyMode) => void;
  selectedNode: RenderNode | null;
  setSelectedNode: (node: RenderNode | null) => void;
  detailPanelOpen: boolean;
  setDetailPanelOpen: (open: boolean) => void;
}

export const useViewStore = create<ViewState>((set) => ({
  mode: "explorer",
  setMode: (mode) => set({ mode }),
  selectedNode: null,
  setSelectedNode: (node) => set({ selectedNode: node, detailPanelOpen: node !== null }),
  detailPanelOpen: false,
  setDetailPanelOpen: (open) => set({ detailPanelOpen: open, selectedNode: open ? undefined : null }),
}));

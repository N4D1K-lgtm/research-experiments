import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import { useTrieDataStore } from "../../../store/trieDataStore";
import { useFilterStore } from "../../../store/filterStore";
import { useViewStore } from "../../../store/viewStore";
import { useLOD } from "../../../hooks/useLOD";
import { computeRadialLayout, type RadialLayout } from "../../../utils/radialLayout";
import type { RenderNode, FilterState } from "../../../types/trie";
import { COLORS, FONTS } from "../../../styles/theme";
import { Viewport } from "../../Viewport";
import { FilterPanel } from "../../FilterPanel";
import { Tooltip } from "../../Tooltip";
import { SearchPanel } from "../../SearchPanel";
import { NodeDetailPanel } from "../../NodeDetailPanel";
import { Prose, Insight } from "../shared";

export function ChapterExplorer() {
  const { metadata, nodes: nodeMap, edges } = useTrieDataStore();
  const filter = useFilterStore();
  const { setSelectedNode } = useViewStore();
  const maxDepth = metadata?.maxDepth ?? 12;
  const { lod, updateLOD } = useLOD(maxDepth);
  const [fullscreen, setFullscreen] = useState(false);
  const [layout, setLayout] = useState<RadialLayout | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Only mount the 3D canvas when the explorer section is visible
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { threshold: 0.05 },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    if (nodeMap.size === 0) return;
    const nodesArray = Array.from(nodeMap.values());
    const filterState: FilterState = {
      maxDepth: filter.maxDepth,
      minFrequency: filter.minFrequency,
      terminalsOnly: filter.terminalsOnly,
      highlightMotifs: filter.highlightMotifs,
      positionFilter: filter.positionFilter,
    };
    setLayout(computeRadialLayout(nodesArray, edges, filterState));
  }, [nodeMap, edges, filter.maxDepth, filter.minFrequency, filter.terminalsOnly]);

  const [hoveredNode, setHoveredNode] = useState<RenderNode | null>(null);
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0 });
  const handleHover = useCallback((node: RenderNode | null, x: number, y: number) => {
    setHoveredNode(node);
    setHoverPos({ x, y });
  }, []);
  const handleClick = useCallback((node: RenderNode | null) => {
    if (node) setSelectedNode(node);
  }, [setSelectedNode]);
  const handleNodeSelect = useCallback((node: RenderNode) => {
    setSelectedNode(node);
  }, [setSelectedNode]);

  const filterState: FilterState = useMemo(() => ({
    maxDepth: filter.maxDepth,
    minFrequency: filter.minFrequency,
    terminalsOnly: filter.terminalsOnly,
    highlightMotifs: filter.highlightMotifs,
    positionFilter: filter.positionFilter,
  }), [filter.maxDepth, filter.minFrequency, filter.terminalsOnly, filter.highlightMotifs, filter.positionFilter]);

  const pathCache = useMemo(() => {
    const cache = new Map<number, string>();
    for (const [id, node] of nodeMap) {
      if (node.depth === 0) { cache.set(id, ""); continue; }
      const segs: string[] = [];
      let cur: RenderNode | undefined = node;
      while (cur && cur.depth > 0) {
        segs.push(cur.phoneme);
        cur = cur.parentId != null ? nodeMap.get(cur.parentId) : undefined;
      }
      segs.reverse();
      cache.set(id, segs.join(""));
    }
    return cache;
  }, [nodeMap]);

  if (fullscreen) {
    return (
      <section id="explorer" style={{ position: "relative" }}>
        <div style={{ position: "fixed", inset: 0, zIndex: 200, background: COLORS.bg }}>
          <Viewport
            nodeMap={nodeMap}
            edges={edges}
            layout={layout}
            lod={lod}
            filter={filterState}
            onDistanceChange={updateLOD}
            onHover={handleHover}
            hoveredNodeId={hoveredNode?.id ?? null}
            onClick={handleClick}
          />
          {metadata && <FilterPanel metadata={metadata} />}
          {metadata && <SearchPanel onNodeSelect={handleNodeSelect} />}
          <Tooltip node={hoveredNode} x={hoverPos.x} y={hoverPos.y} pathCache={pathCache} />
          <NodeDetailPanel />

          {/* Exit fullscreen */}
          <button
            onClick={() => setFullscreen(false)}
            style={{
              position: "fixed",
              top: 16,
              left: 16,
              padding: "8px 14px",
              background: "rgba(8,8,16,0.85)",
              border: `1px solid ${COLORS.border}`,
              borderRadius: 8,
              color: COLORS.textDim,
              fontSize: 11,
              fontFamily: FONTS.mono,
              cursor: "pointer",
              backdropFilter: "blur(12px)",
              zIndex: 210,
            }}
          >
            ← Back to tutorial
          </button>

          {/* Stats */}
          <div style={{ position: "fixed", bottom: 16, left: 16, fontSize: 10, color: COLORS.textFaint, fontFamily: FONTS.mono, zIndex: 210 }}>
            {nodeMap.size.toLocaleString()} nodes · {metadata?.totalWords.toLocaleString()} words · {metadata?.phonemeInventory.length} phonemes · {metadata?.languages.join(", ")}
          </div>
        </div>
      </section>
    );
  }

  return (
    <section
      id="explorer"
      style={{
        minHeight: "100vh",
        padding: "80px 0 0",
        position: "relative",
      }}
    >
      <div style={{ maxWidth: 760, margin: "0 auto", padding: "0 40px" }}>
        <div style={{ marginBottom: 40 }}>
          <div style={{ fontSize: 11, fontFamily: FONTS.mono, color: COLORS.accent, textTransform: "uppercase", letterSpacing: 2, marginBottom: 8 }}>
            Chapter 8
          </div>
          <h2 style={{ fontSize: 32, fontWeight: 600, fontFamily: FONTS.mono, color: COLORS.textBright, lineHeight: 1.2, margin: 0 }}>
            Explore the Trie
          </h2>
          <p style={{ fontSize: 15, color: COLORS.textDim, lineHeight: 1.7, marginTop: 12, maxWidth: 560 }}>
            You've learned the concepts. Now explore the full {metadata?.nodeCount.toLocaleString() ?? "664,017"}-node
            trie yourself. Orbit, zoom, hover to inspect nodes, click for details.
          </p>
        </div>

        <Prose>
          The 3D view renders every node as an icosahedron. Size encodes a blend of entropy,
          transition probability, and frequency. Color is determined by phonological position.
          Terminal nodes (word endpoints) glow. Bright, short edges are high-probability transitions —
          the "highways" of phoneme space.
        </Prose>

        <Insight>
          Click any node to open the detail panel: see the full word list, transition probabilities,
          per-language frequency breakdown, allophones, and child nodes. This is the real data from
          the backend — every number comes from the trie built from{" "}
          {metadata?.totalWords?.toLocaleString() ?? "269,737"} pronunciations.
        </Insight>
      </div>

      {/* Embedded 3D viewport */}
      <div
        ref={containerRef}
        style={{
          position: "relative",
          height: "80vh",
          margin: "40px 0",
          borderTop: `1px solid ${COLORS.border}`,
          borderBottom: `1px solid ${COLORS.border}`,
          overflow: "hidden",
          background: COLORS.bg,
        }}
      >
        {isVisible && (
          <>
            <Viewport
              nodeMap={nodeMap}
              edges={edges}
              layout={layout}
              lod={lod}
              filter={filterState}
              onDistanceChange={updateLOD}
              onHover={handleHover}
              hoveredNodeId={hoveredNode?.id ?? null}
              onClick={handleClick}
            />
            <Tooltip node={hoveredNode} x={hoverPos.x} y={hoverPos.y} pathCache={pathCache} />
            <NodeDetailPanel />
          </>
        )}
        {!isVisible && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: COLORS.textFaint, fontFamily: FONTS.mono, fontSize: 12 }}>
            Scroll here to load 3D view
          </div>
        )}

        {/* Fullscreen button */}
        <button
          onClick={() => setFullscreen(true)}
          style={{
            position: "absolute",
            top: 16,
            right: 16,
            padding: "8px 14px",
            background: "rgba(8,8,16,0.85)",
            border: `1px solid ${COLORS.border}`,
            borderRadius: 8,
            color: COLORS.textDim,
            fontSize: 11,
            fontFamily: FONTS.mono,
            cursor: "pointer",
            backdropFilter: "blur(12px)",
            zIndex: 10,
          }}
        >
          Fullscreen ↗
        </button>
      </div>

      {/* Coda */}
      <div style={{ maxWidth: 760, margin: "0 auto", padding: "0 40px 120px" }}>
        <Prose>
          This is one trie built from {metadata?.languages.length ?? 12} languages. Every word,
          every phoneme, every transition — compressed into a single navigable structure. The
          patterns you see are not designed. They emerge from the statistical regularities of
          human speech.
        </Prose>

        <div
          style={{
            marginTop: 40,
            padding: "20px 24px",
            background: `${COLORS.accent}06`,
            border: `1px solid ${COLORS.accent}15`,
            borderRadius: 12,
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 13, color: COLORS.textDim, fontFamily: FONTS.sans, lineHeight: 1.8 }}>
            Built with Rust + SurrealDB + async-graphql + React + Three.js
            <br />
            Data from WikiPron, CMU Dict, and PHOIBLE
          </div>
        </div>
      </div>
    </section>
  );
}

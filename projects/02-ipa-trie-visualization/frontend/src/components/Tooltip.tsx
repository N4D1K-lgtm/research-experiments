import type { RenderNode } from "../types/trie";
import { getRoleColor, getRoleLabel } from "../utils/languagePalette";

interface Props {
  node: RenderNode | null;
  x: number;
  y: number;
  pathCache: Map<number, string>;
}

export function Tooltip({ node, x, y, pathCache }: Props) {
  if (!node) return null;

  const roleColor = getRoleColor(node.phonologicalPosition);
  const roleLabel = getRoleLabel(node.phonologicalPosition);
  const fullPath = pathCache.get(node.id) ?? node.phoneme;

  // Position tooltip
  const pad = 16;
  let tx = x + pad;
  let ty = y + pad;
  if (tx + 260 > window.innerWidth) tx = x - 260 - pad;
  if (ty + 300 > window.innerHeight) ty = y - 300 - pad;

  const transitions = Object.entries(node.transitionProbs).slice(0, 8);
  const totalChildren = Object.keys(node.transitionProbs).length;

  return (
    <div
      style={{
        position: "fixed",
        left: tx,
        top: ty,
        pointerEvents: "none",
        background: "rgba(8, 8, 16, 0.94)",
        border: "1px solid rgba(255, 255, 255, 0.08)",
        borderRadius: 10,
        padding: "12px 16px",
        fontSize: 12,
        lineHeight: 1.6,
        maxWidth: 260,
        zIndex: 100,
        backdropFilter: "blur(12px)",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
        color: "#d0d0d8",
        fontFamily: "'Inter', sans-serif",
      }}
    >
      {/* Phoneme header */}
      <div style={{ fontSize: 26, fontWeight: 600, color: roleColor, marginBottom: 2 }}>
        /{fullPath}/
        <span
          style={{
            display: "inline-block",
            fontSize: 8,
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: 0.8,
            padding: "1px 5px",
            borderRadius: 3,
            background: node.isTerminal ? "rgba(100,255,160,0.15)" : "rgba(255,255,255,0.06)",
            color: node.isTerminal ? "#6fda8e" : "#555",
            verticalAlign: "middle",
            marginLeft: 6,
          }}
        >
          {node.isTerminal ? "word" : "path"}
        </span>
        <span
          style={{
            display: "inline-block",
            fontSize: 8,
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: 0.8,
            padding: "1px 5px",
            borderRadius: 3,
            background: `${roleColor}22`,
            color: roleColor,
            verticalAlign: "middle",
            marginLeft: 4,
          }}
        >
          {roleLabel}
        </span>
      </div>

      {/* Meta */}
      <div style={{ color: "#666", fontSize: 10, textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 8 }}>
        depth {node.depth} · {node.totalCount.toLocaleString()} paths · {node.childCount} branches
      </div>

      {/* Allophones */}
      {node.allophones.length > 0 && (
        <div style={{ fontSize: 10, color: "#777", margin: "4px 0 6px" }}>
          allophones: {node.allophones.map((a) => (
            <span key={a} style={{ color: "#aaa", fontStyle: "italic", margin: "0 2px" }}>[{a}]</span>
          ))}
        </div>
      )}

      {/* Transitions */}
      {transitions.length > 0 && (
        <div style={{ marginTop: 8, paddingTop: 6, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
          <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 4 }}>
            Transitions ({totalChildren} paths)
          </div>
          {transitions.map(([phoneme, prob]) => (
            <div key={phoneme} style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 3, fontSize: 11 }}>
              <span style={{ minWidth: 30, color: "#999", fontFamily: "monospace" }}>/{phoneme}/</span>
              <div style={{ flex: 1, height: 3, background: "rgba(255,255,255,0.05)", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${Math.min(100, prob * 100)}%`, background: roleColor, borderRadius: 2 }} />
              </div>
              <span style={{ minWidth: 36, textAlign: "right", color: "#777", fontSize: 10 }}>
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Words */}
      {node.isTerminal && Object.keys(node.words).length > 0 && (
        <div style={{ marginTop: 8, paddingTop: 6, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
          {Object.entries(node.words).map(([lang, words]) => (
            <div key={lang} style={{ fontSize: 11, color: "#ccc", marginTop: 2 }}>
              {words.map((w, i) => (
                <span key={w}>
                  {i > 0 && ", "}
                  <em style={{ fontStyle: "normal", color: "#e8e8e8" }}>{w}</em>
                </span>
              ))}
              {(node.terminalCounts[lang] ?? words.length) > words.length && (
                <span style={{ color: "#555", fontSize: 9, marginLeft: 4 }}>
                  +{(node.terminalCounts[lang] ?? words.length) - words.length} more
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

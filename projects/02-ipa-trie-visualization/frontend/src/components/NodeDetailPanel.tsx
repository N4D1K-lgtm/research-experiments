import { useEffect, useState } from "react";
import { useQuery } from "urql";
import { NODE_DETAIL_QUERY } from "../graphql/queries";
import { useViewStore } from "../store/viewStore";
import { COLORS, FONTS, langColor, langLabel, hexAlpha } from "../styles/theme";
import { getRoleColor, getRoleLabel } from "../utils/languagePalette";
import { InfoTooltip } from "./InfoTooltip";

interface NodeChild {
  id: number;
  phoneme: string;
  totalCount: number;
  phonologicalPosition: string;
  isTerminal: boolean;
}

interface NodeDetailData {
  node: {
    id: number;
    phoneme: string;
    depth: number;
    parentId: number | null;
    totalCount: number;
    isTerminal: boolean;
    childCount: number;
    weight: number;
    phonologicalPosition: string;
    color: string;
    allophones: string[];
    transitionProbs: { phoneme: string; probability: number }[];
    terminalCounts: { language: string; count: number }[];
    words: { language: string; words: string[] }[];
    children: NodeChild[];
    parent: { id: number; phoneme: string; transitionProbs: { phoneme: string; probability: number }[] } | null;
  } | null;
}

export function NodeDetailPanel() {
  const { selectedNode, detailPanelOpen, setDetailPanelOpen } = useViewStore();
  const [animating, setAnimating] = useState(false);

  const [result] = useQuery<NodeDetailData>({
    query: NODE_DETAIL_QUERY,
    variables: { id: selectedNode?.id ?? -1 },
    pause: !selectedNode,
  });

  const node = result.data?.node ?? null;
  const renderNode = selectedNode;

  useEffect(() => {
    if (detailPanelOpen) {
      setAnimating(true);
      const t = setTimeout(() => setAnimating(false), 200);
      return () => clearTimeout(t);
    }
  }, [detailPanelOpen]);

  if (!detailPanelOpen || !renderNode) return null;

  const roleColor = getRoleColor(renderNode.phonologicalPosition);
  const roleLabel = getRoleLabel(renderNode.phonologicalPosition);
  const transitions = Object.entries(renderNode.transitionProbs)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 12);
  const totalPaths = Object.keys(renderNode.transitionProbs).length;

  // Build path from node's ancestry
  const pathSegments: string[] = [];
  // We only have the selected node's phoneme; the full path comes from pathCache in App
  // For now use what we have

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        right: 0,
        bottom: 0,
        width: 380,
        background: COLORS.bgPanel,
        borderLeft: `1px solid ${COLORS.border}`,
        backdropFilter: "blur(20px)",
        zIndex: 70,
        overflowY: "auto",
        fontFamily: FONTS.sans,
        color: COLORS.text,
        transform: animating ? "translateX(20px)" : "translateX(0)",
        opacity: animating ? 0 : 1,
        transition: "transform 0.2s ease, opacity 0.2s ease",
      }}
    >
      {/* Header */}
      <div
        style={{
          position: "sticky",
          top: 0,
          background: "rgba(6, 6, 12, 0.98)",
          borderBottom: `1px solid ${COLORS.border}`,
          padding: "16px 20px 14px",
          zIndex: 1,
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
        }}
      >
        <div>
          {/* Big phoneme */}
          <div
            style={{
              fontSize: 42,
              fontWeight: 600,
              fontFamily: FONTS.mono,
              color: roleColor,
              lineHeight: 1,
              marginBottom: 6,
            }}
          >
            /{renderNode.phoneme}/
          </div>

          {/* Badges */}
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <Badge
              label={renderNode.isTerminal ? "word endpoint" : "path node"}
              color={renderNode.isTerminal ? COLORS.green : COLORS.textFaint}
              bright={renderNode.isTerminal}
            />
            <Badge label={roleLabel} color={roleColor} />
            <Badge label={`depth ${renderNode.depth}`} color={COLORS.textDim} />
          </div>
        </div>

        <button
          onClick={() => setDetailPanelOpen(false)}
          style={{
            background: "none",
            border: "none",
            color: COLORS.textDim,
            fontSize: 18,
            cursor: "pointer",
            padding: "0 4px",
            fontFamily: FONTS.mono,
          }}
        >
          ×
        </button>
      </div>

      <div style={{ padding: "0 20px 20px" }}>
        {/* Stats row */}
        <Section
          title="Node Statistics"
          info="Aggregate counts for this phoneme position in the trie. Weight is the total number of word paths passing through this node across all languages."
        >
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
            <StatCard label="Paths" value={renderNode.totalCount.toLocaleString()} />
            <StatCard label="Branches" value={String(renderNode.childCount)} />
            <StatCard label="Weight" value={renderNode.weight.toLocaleString()} />
          </div>
        </Section>

        {/* Allophones */}
        {renderNode.allophones.length > 0 && (
          <Section
            title="Allophones"
            info="Phonetic variants of this phoneme that appear in different contexts. Allophones are distinct sounds that native speakers perceive as the same phoneme."
          >
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {renderNode.allophones.map((a) => (
                <span
                  key={a}
                  style={{
                    padding: "4px 10px",
                    background: `${roleColor}12`,
                    border: `1px solid ${roleColor}25`,
                    borderRadius: 6,
                    fontSize: 14,
                    fontFamily: FONTS.mono,
                    color: roleColor,
                    fontStyle: "italic",
                  }}
                >
                  [{a}]
                </span>
              ))}
            </div>
          </Section>
        )}

        {/* Transitions */}
        {transitions.length > 0 && (
          <Section
            title={`Transitions (${totalPaths} paths)`}
            info="Probability distribution of which phoneme follows this one. Higher bars mean that transition is more common in the corpus. This is the empirical bigram probability P(next | current)."
          >
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {transitions.map(([phoneme, prob]) => {
                const pColor = getRoleColor("mixed"); // default
                return (
                  <div
                    key={phoneme}
                    style={{ display: "flex", alignItems: "center", gap: 8 }}
                  >
                    <span
                      style={{
                        minWidth: 36,
                        fontFamily: FONTS.mono,
                        fontSize: 13,
                        fontWeight: 500,
                        color: COLORS.textBright,
                        textAlign: "right",
                      }}
                    >
                      /{phoneme}/
                    </span>
                    <div
                      style={{
                        flex: 1,
                        height: 6,
                        background: "rgba(255,255,255,0.04)",
                        borderRadius: 3,
                        overflow: "hidden",
                      }}
                    >
                      <div
                        style={{
                          height: "100%",
                          width: `${Math.min(100, prob * 100)}%`,
                          background: `linear-gradient(90deg, ${roleColor}88, ${roleColor})`,
                          borderRadius: 3,
                          transition: "width 0.3s ease",
                        }}
                      />
                    </div>
                    <span
                      style={{
                        minWidth: 44,
                        textAlign: "right",
                        fontFamily: FONTS.mono,
                        fontSize: 11,
                        color: COLORS.textDim,
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </Section>
        )}

        {/* Words per language */}
        {renderNode.isTerminal && Object.keys(renderNode.words).length > 0 && (
          <Section
            title="Words"
            info="Orthographic words from each language that end at this node in the trie. These share the same phonological path from root to this terminal."
          >
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {Object.entries(renderNode.words).map(([lang, words]) => {
                const count = renderNode.terminalCounts[lang] ?? words.length;
                return (
                  <div
                    key={lang}
                    style={{
                      background: `${langColor(lang)}08`,
                      border: `1px solid ${langColor(lang)}18`,
                      borderRadius: 10,
                      padding: "10px 14px",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        marginBottom: 6,
                      }}
                    >
                      <span
                        style={{
                          width: 7,
                          height: 7,
                          borderRadius: "50%",
                          background: langColor(lang),
                          flexShrink: 0,
                        }}
                      />
                      <span
                        style={{
                          fontSize: 11,
                          fontWeight: 600,
                          color: langColor(lang),
                          textTransform: "uppercase",
                          letterSpacing: 0.5,
                        }}
                      >
                        {langLabel(lang)}
                      </span>
                      <span style={{ fontSize: 9, color: COLORS.textFaint, marginLeft: "auto" }}>
                        {count.toLocaleString()} entries
                      </span>
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                      {words.slice(0, 12).map((w) => (
                        <span
                          key={w}
                          style={{
                            display: "inline-block",
                            padding: "3px 8px",
                            background: "rgba(255,255,255,0.04)",
                            borderRadius: 5,
                            fontSize: 13,
                            fontFamily: FONTS.mono,
                            color: COLORS.textBright,
                            fontWeight: 500,
                          }}
                        >
                          {w}
                        </span>
                      ))}
                      {words.length > 12 && (
                        <span
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            padding: "3px 8px",
                            fontSize: 10,
                            color: COLORS.textFaint,
                          }}
                        >
                          +{words.length - 12} more
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </Section>
        )}

        {/* Children preview */}
        {node && node.children.length > 0 && (
          <Section
            title={`Children (${node.children.length})`}
            info="Direct child nodes in the trie — each represents a phoneme that can follow the current one. Terminal children mark word endpoints."
          >
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {node.children.slice(0, 30).map((child) => (
                <span
                  key={child.id}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 4,
                    padding: "4px 8px",
                    background: child.isTerminal
                      ? "rgba(100,255,160,0.06)"
                      : "rgba(255,255,255,0.03)",
                    border: `1px solid ${getRoleColor(child.phonologicalPosition)}20`,
                    borderRadius: 6,
                    fontSize: 12,
                    fontFamily: FONTS.mono,
                    color: getRoleColor(child.phonologicalPosition),
                  }}
                >
                  /{child.phoneme}/
                  <span
                    style={{
                      fontSize: 9,
                      color: COLORS.textFaint,
                      fontFamily: FONTS.sans,
                    }}
                  >
                    {child.totalCount.toLocaleString()}
                  </span>
                  {child.isTerminal && (
                    <span style={{ width: 4, height: 4, borderRadius: "50%", background: COLORS.green }} />
                  )}
                </span>
              ))}
              {node.children.length > 30 && (
                <span style={{ fontSize: 10, color: COLORS.textFaint, padding: "4px 8px" }}>
                  +{node.children.length - 30} more
                </span>
              )}
            </div>
          </Section>
        )}

        {/* Per-language frequency breakdown */}
        {Object.keys(renderNode.terminalCounts).length > 0 && !renderNode.isTerminal && (
          <Section
            title="Frequency by Language"
            info="How many word paths pass through this node in each language's sub-trie. Languages with higher counts use this phoneme sequence more frequently."
          >
            {Object.entries(renderNode.terminalCounts)
              .sort(([, a], [, b]) => b - a)
              .map(([lang, count]) => {
                const maxCount = Math.max(
                  ...Object.values(renderNode.terminalCounts),
                );
                const pct = maxCount > 0 ? count / maxCount : 0;
                return (
                  <div
                    key={lang}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      marginBottom: 4,
                    }}
                  >
                    <span
                      style={{
                        width: 7,
                        height: 7,
                        borderRadius: "50%",
                        background: langColor(lang),
                        flexShrink: 0,
                      }}
                    />
                    <span
                      style={{
                        minWidth: 60,
                        fontSize: 11,
                        color: langColor(lang),
                        fontWeight: 500,
                      }}
                    >
                      {langLabel(lang)}
                    </span>
                    <div
                      style={{
                        flex: 1,
                        height: 4,
                        background: "rgba(255,255,255,0.04)",
                        borderRadius: 2,
                        overflow: "hidden",
                      }}
                    >
                      <div
                        style={{
                          height: "100%",
                          width: `${pct * 100}%`,
                          background: langColor(lang),
                          borderRadius: 2,
                          opacity: 0.6,
                        }}
                      />
                    </div>
                    <span
                      style={{
                        minWidth: 44,
                        textAlign: "right",
                        fontSize: 10,
                        color: COLORS.textDim,
                        fontFamily: FONTS.mono,
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {count.toLocaleString()}
                    </span>
                  </div>
                );
              })}
          </Section>
        )}
      </div>
    </div>
  );
}

function Section({
  title,
  info,
  children,
}: {
  title: string;
  info?: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ marginTop: 20 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          marginBottom: 10,
        }}
      >
        <h3
          style={{
            fontSize: 10,
            fontWeight: 600,
            color: COLORS.textDim,
            textTransform: "uppercase",
            letterSpacing: 1.2,
            margin: 0,
          }}
        >
          {title}
        </h3>
        {info && <InfoTooltip text={info} />}
      </div>
      {children}
    </div>
  );
}

function Badge({ label, color, bright }: { label: string; color: string; bright?: boolean }) {
  return (
    <span
      style={{
        display: "inline-block",
        fontSize: 8,
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: 0.8,
        padding: "2px 7px",
        borderRadius: 4,
        background: bright ? `${color}20` : `${color}10`,
        color,
        border: `1px solid ${color}20`,
      }}
    >
      {label}
    </span>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        background: COLORS.bgCard,
        borderRadius: 8,
        padding: "8px 12px",
        border: `1px solid ${COLORS.border}`,
      }}
    >
      <div
        style={{
          fontSize: 18,
          fontWeight: 600,
          fontFamily: FONTS.mono,
          color: COLORS.textBright,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </div>
      <div
        style={{
          fontSize: 9,
          color: COLORS.textDim,
          textTransform: "uppercase",
          letterSpacing: 0.5,
          marginTop: 2,
        }}
      >
        {label}
      </div>
    </div>
  );
}

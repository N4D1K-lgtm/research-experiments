import { useState } from "react";
import { useQuery } from "urql";
import { LANGUAGES_QUERY, CROSS_LINGUISTIC_STATS_QUERY } from "../graphql/queries";
import type { DepthStats, LanguageInfo } from "../types/trie";
import { COLORS, FONTS, langColor, langLabel, hexAlpha } from "../styles/theme";
import { InfoTooltip } from "./InfoTooltip";

interface CrossLingStat {
  language: string;
  stats: DepthStats[];
}

const FAMILY_ORDER = ["Germanic", "Romance", "Sino-Tibetan", "Japonic", "Semitic", "Uralic", "Turkic", "Indo-Aryan", "Bantu"];

export function CrossLingView() {
  const [langResult] = useQuery({ query: LANGUAGES_QUERY });
  const [crossResult] = useQuery({ query: CROSS_LINGUISTIC_STATS_QUERY });

  const languages: LanguageInfo[] = langResult.data?.languages ?? [];
  const crossStats: CrossLingStat[] = crossResult.data?.crossLinguisticStats ?? [];

  if (langResult.fetching || crossResult.fetching) {
    return <ViewLoading />;
  }

  // Group by family
  const families = new Map<string, LanguageInfo[]>();
  for (const lang of languages) {
    const list = families.get(lang.family) ?? [];
    list.push(lang);
    families.set(lang.family, list);
  }
  const sortedFamilies = Array.from(families.entries()).sort(
    ([a], [b]) => (FAMILY_ORDER.indexOf(a) === -1 ? 99 : FAMILY_ORDER.indexOf(a)) - (FAMILY_ORDER.indexOf(b) === -1 ? 99 : FAMILY_ORDER.indexOf(b)),
  );

  return (
    <div
      style={{
        position: "fixed",
        left: 56,
        top: 0,
        right: 0,
        bottom: 0,
        background: COLORS.bg,
        overflowY: "auto",
        fontFamily: FONTS.sans,
        color: COLORS.text,
      }}
    >
      <div style={{ maxWidth: 960, margin: "0 auto", padding: "40px 40px 60px" }}>
        <header style={{ marginBottom: 40 }}>
          <h1 style={{ fontSize: 28, fontWeight: 600, color: COLORS.textBright, fontFamily: FONTS.mono, marginBottom: 8 }}>
            Cross-Linguistic Comparison
          </h1>
          <p style={{ fontSize: 13, color: COLORS.textDim, lineHeight: 1.7, maxWidth: 640 }}>
            Compare phonological structure across {languages.length} languages from {families.size} language families.
            Each language builds its own sub-trie from its pronunciation data.
          </p>
        </header>

        {/* Convergence table */}
        <ConvergenceTable languages={languages} crossStats={crossStats} />

        {/* Per-family breakdown */}
        {sortedFamilies.map(([family, langs]) => (
          <FamilySection key={family} family={family} languages={langs} crossStats={crossStats} />
        ))}
      </div>
    </div>
  );
}

function ConvergenceTable({
  languages,
  crossStats,
}: {
  languages: LanguageInfo[];
  crossStats: CrossLingStat[];
}) {
  return (
    <section style={{ marginBottom: 48 }}>
      <SectionHeader
        title="Language Overview"
        info="Peak shell is the depth with the most nodes — it reflects where phonological branching is highest. Convergence depth is where branching factor drops below 2, meaning most paths have narrowed to few options."
      />

      <div
        style={{
          background: COLORS.bgCard,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 12,
          overflowX: "auto",
        }}
      >
        <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: FONTS.mono, fontSize: 11 }}>
          <thead>
            <tr>
              {["Language", "Family", "Typology", "Peak Shell", "Converge Depth", "Total Nodes"].map((h) => (
                <th
                  key={h}
                  style={{
                    padding: "12px 14px",
                    textAlign: h === "Language" ? "left" : "right",
                    fontWeight: 600,
                    color: COLORS.textDim,
                    borderBottom: `1px solid ${COLORS.border}`,
                    fontSize: 9,
                    textTransform: "uppercase",
                    letterSpacing: 0.8,
                    whiteSpace: "nowrap",
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {languages.map((lang) => {
              const stats = crossStats.find((cs) => cs.language === lang.code)?.stats ?? [];
              const peakShell = stats.reduce((best, s) => (s.nodes > best.nodes ? s : best), stats[0] ?? { depth: 0, nodes: 0 });
              const convergeDepth = stats.find((s) => s.depth > 1 && s.avgBranch < 2)?.depth ?? "—";
              const totalNodes = stats.reduce((s, d) => s + d.nodes, 0);

              return (
                <tr key={lang.code} style={{ borderBottom: `1px solid rgba(255,255,255,0.02)` }}>
                  <td style={{ padding: "8px 14px", textAlign: "left" }}>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                      <span style={{ width: 7, height: 7, borderRadius: "50%", background: langColor(lang.code) }} />
                      <span style={{ color: langColor(lang.code), fontWeight: 500 }}>{langLabel(lang.code)}</span>
                    </span>
                  </td>
                  <td style={tdStyle}>{lang.family}</td>
                  <td style={tdStyle}>{lang.typology}</td>
                  <td style={tdStyle}>
                    {peakShell.nodes.toLocaleString()}{" "}
                    <span style={{ color: COLORS.textFaint }}>(d={peakShell.depth})</span>
                  </td>
                  <td style={tdStyle}>{convergeDepth}</td>
                  <td style={tdStyle}>{totalNodes.toLocaleString()}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function FamilySection({
  family,
  languages,
  crossStats,
}: {
  family: string;
  languages: LanguageInfo[];
  crossStats: CrossLingStat[];
}) {
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <section style={{ marginBottom: 32 }}>
      <SectionHeader title={family} info={`Languages from the ${family} family in the corpus.`} />

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {languages.map((lang) => {
          const stats = crossStats.find((cs) => cs.language === lang.code)?.stats ?? [];
          const totalNodes = stats.reduce((s, d) => s + d.nodes, 0);
          const totalTerminals = stats.reduce((s, d) => s + d.terminals, 0);
          const maxDepth = stats.length > 0 ? stats[stats.length - 1].depth : 0;
          const isOpen = expanded === lang.code;

          // Find max for bar chart normalization
          const maxNodes = Math.max(...stats.map((s) => s.nodes), 1);

          return (
            <div
              key={lang.code}
              style={{
                background: COLORS.bgCard,
                border: `1px solid ${langColor(lang.code)}15`,
                borderRadius: 12,
                overflow: "hidden",
              }}
            >
              <button
                onClick={() => setExpanded(isOpen ? null : lang.code)}
                style={{
                  width: "100%",
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "12px 16px",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  fontFamily: FONTS.sans,
                  textAlign: "left",
                }}
              >
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: langColor(lang.code), flexShrink: 0 }} />
                <span style={{ color: langColor(lang.code), fontWeight: 600, fontSize: 14, flex: 1, fontFamily: FONTS.mono }}>
                  {langLabel(lang.code)}
                </span>
                <span style={{ fontSize: 9, color: COLORS.textFaint }}>{lang.typology}</span>
                <span style={{ fontSize: 11, color: COLORS.textDim, fontFamily: FONTS.mono, fontVariantNumeric: "tabular-nums" }}>
                  {totalNodes.toLocaleString()} nodes
                </span>
                <span style={{ color: COLORS.textFaint, fontSize: 14 }}>{isOpen ? "−" : "+"}</span>
              </button>

              {isOpen && stats.length > 0 && (
                <div style={{ padding: "0 16px 16px" }}>
                  {/* Stats row */}
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, marginBottom: 16 }}>
                    <MiniStat label="Nodes" value={totalNodes.toLocaleString()} />
                    <MiniStat label="Terminals" value={totalTerminals.toLocaleString()} />
                    <MiniStat label="Max Depth" value={String(maxDepth)} />
                    <MiniStat label="Peak Branch" value={Math.max(...stats.map((s) => s.avgBranch)).toFixed(1)} />
                  </div>

                  {/* Node distribution */}
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ fontSize: 9, color: COLORS.textFaint, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 6 }}>
                      Node count by depth
                    </div>
                    <div style={{ display: "flex", alignItems: "flex-end", gap: 1, height: 60 }}>
                      {stats.filter((s) => s.depth > 0).map((s) => (
                        <div
                          key={s.depth}
                          title={`Depth ${s.depth}: ${s.nodes.toLocaleString()} nodes`}
                          style={{
                            flex: 1,
                            height: `${Math.max(2, (s.nodes / maxNodes) * 100)}%`,
                            background: hexAlpha(langColor(lang.code), 0.5),
                            borderRadius: "2px 2px 0 0",
                          }}
                        />
                      ))}
                    </div>
                  </div>

                  {/* Entropy curve */}
                  <div>
                    <div style={{ fontSize: 9, color: COLORS.textFaint, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 6 }}>
                      Entropy by depth
                    </div>
                    <EntropyCurve stats={stats} color={langColor(lang.code)} />
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function EntropyCurve({ stats, color }: { stats: DepthStats[]; color: string }) {
  const filtered = stats.filter((s) => s.depth > 0);
  if (filtered.length < 2) return null;

  const maxE = Math.max(...filtered.map((s) => s.avgEntropy), 0.01);
  const maxD = Math.max(...filtered.map((s) => s.depth), 1);
  const W = 400, H = 50;

  const points = filtered.map((s) => {
    const x = (s.depth / maxD) * W;
    const y = H - (s.avgEntropy / maxE) * H;
    return `${x},${y}`;
  }).join(" ");

  return (
    <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={2}
        strokeOpacity={0.6}
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ background: "rgba(255,255,255,0.02)", borderRadius: 6, padding: "6px 10px", border: `1px solid ${COLORS.border}` }}>
      <div style={{ fontSize: 15, fontWeight: 600, fontFamily: FONTS.mono, color: COLORS.textBright, fontVariantNumeric: "tabular-nums" }}>
        {value}
      </div>
      <div style={{ fontSize: 8, color: COLORS.textFaint, textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</div>
    </div>
  );
}

function SectionHeader({ title, info }: { title: string; info?: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", marginBottom: 14 }}>
      <h2 style={{ fontSize: 16, fontWeight: 600, color: COLORS.textBright, fontFamily: FONTS.mono, margin: 0 }}>
        {title}
      </h2>
      {info && <InfoTooltip text={info} maxWidth={320} />}
    </div>
  );
}

function ViewLoading() {
  return (
    <div style={{ position: "fixed", left: 56, top: 0, right: 0, bottom: 0, display: "flex", alignItems: "center", justifyContent: "center", background: COLORS.bg }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ width: 24, height: 24, border: "2px solid rgba(255,255,255,0.08)", borderTopColor: "rgba(255,255,255,0.4)", borderRadius: "50%", animation: "spin 0.8s linear infinite", margin: "0 auto 12px" }} />
        <span style={{ fontSize: 12, color: COLORS.textFaint, fontFamily: FONTS.sans }}>Loading language data...</span>
      </div>
    </div>
  );
}

const tdStyle: React.CSSProperties = {
  padding: "8px 14px",
  textAlign: "right",
  fontVariantNumeric: "tabular-nums",
  color: "#888",
};

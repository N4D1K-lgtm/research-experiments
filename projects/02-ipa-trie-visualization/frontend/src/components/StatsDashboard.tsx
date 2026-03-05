import { useState } from "react";
import { useQuery } from "urql";
import {
  DEPTH_STATS_QUERY,
  LANGUAGES_QUERY,
  CROSS_LINGUISTIC_STATS_QUERY,
} from "../graphql/queries";
import type { TrieMetadata, DepthStats, LanguageInfo } from "../types/trie";

interface CrossLingStat {
  language: string;
  stats: DepthStats[];
}

interface Props {
  metadata: TrieMetadata;
}

const LANG_COLORS: Record<string, string> = {
  en_US: "#4ECDC4",
  fr_FR: "#FF6B6B",
  es_ES: "#FFE66D",
  de: "#C49BFF",
  nl: "#45B7D1",
  cmn: "#FF9F43",
  jpn: "#EE5A24",
  ara: "#A3CB38",
  fin: "#D980FA",
  tur: "#FDA7DF",
  hin: "#7EFFF5",
  swa: "#C4E538",
};

function langColor(code: string): string {
  return LANG_COLORS[code] ?? "#888";
}

export function StatsDashboard({ metadata }: Props) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"overview" | "depth" | "languages" | "inventory">("overview");

  const [depthResult] = useQuery({ query: DEPTH_STATS_QUERY, pause: !open });
  const [langResult] = useQuery({ query: LANGUAGES_QUERY, pause: !open });
  const [crossResult] = useQuery({ query: CROSS_LINGUISTIC_STATS_QUERY, pause: !open });

  const depthStats: DepthStats[] = depthResult.data?.depthStats ?? [];
  const languages: LanguageInfo[] = langResult.data?.languages ?? [];
  const crossStats: CrossLingStat[] = crossResult.data?.crossLinguisticStats ?? [];

  return (
    <>
      {/* Toggle button */}
      <button
        onClick={() => setOpen(!open)}
        style={{
          position: "fixed",
          bottom: 20,
          right: 20,
          zIndex: 60,
          background: open ? "rgba(255,255,255,0.12)" : "rgba(8, 8, 16, 0.85)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 10,
          padding: "8px 14px",
          color: "#aaa",
          fontSize: 11,
          fontFamily: "'Inter', sans-serif",
          cursor: "pointer",
          backdropFilter: "blur(12px)",
          letterSpacing: 0.3,
        }}
      >
        {open ? "Close Stats" : "Stats"}
      </button>

      {/* Panel */}
      {open && (
        <div
          style={{
            position: "fixed",
            top: 0,
            right: 0,
            bottom: 0,
            width: 420,
            background: "rgba(6, 6, 12, 0.96)",
            borderLeft: "1px solid rgba(255,255,255,0.06)",
            zIndex: 55,
            overflowY: "auto",
            fontFamily: "'Inter', sans-serif",
            color: "#d0d0d8",
            backdropFilter: "blur(20px)",
          }}
        >
          {/* Tabs */}
          <div
            style={{
              display: "flex",
              borderBottom: "1px solid rgba(255,255,255,0.06)",
              position: "sticky",
              top: 0,
              background: "rgba(6, 6, 12, 0.98)",
              zIndex: 1,
            }}
          >
            {(["overview", "depth", "languages", "inventory"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                style={{
                  flex: 1,
                  padding: "14px 0",
                  background: "none",
                  border: "none",
                  borderBottom: tab === t ? "2px solid rgba(255,255,255,0.4)" : "2px solid transparent",
                  color: tab === t ? "#e0e0e8" : "#555",
                  fontSize: 10,
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: 1,
                  cursor: "pointer",
                  fontFamily: "'Inter', sans-serif",
                }}
              >
                {t}
              </button>
            ))}
          </div>

          <div style={{ padding: "20px 24px" }}>
            {tab === "overview" && <OverviewTab metadata={metadata} depthStats={depthStats} languages={languages} />}
            {tab === "depth" && <DepthTab depthStats={depthStats} crossStats={crossStats} />}
            {tab === "languages" && <LanguagesTab languages={languages} crossStats={crossStats} />}
            {tab === "inventory" && <InventoryTab metadata={metadata} />}
          </div>
        </div>
      )}
    </>
  );
}

// ─── Overview ──────────────────────────────────────────────────────────────

function OverviewTab({
  metadata,
  depthStats,
  languages,
}: {
  metadata: TrieMetadata;
  depthStats: DepthStats[];
  languages: LanguageInfo[];
}) {
  const peakDepth = depthStats.reduce((max, d) => (d.nodes > (depthStats[max]?.nodes ?? 0) ? d.depth : max), 0);
  const totalTerminals = depthStats.reduce((sum, d) => sum + d.terminals, 0);
  const avgEntropy = depthStats.length > 0
    ? depthStats.reduce((s, d) => s + d.avgEntropy, 0) / depthStats.length
    : 0;

  return (
    <div>
      <SectionTitle>Trie Summary</SectionTitle>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 20 }}>
        <MetricCard label="Nodes" value={metadata.nodeCount.toLocaleString()} />
        <MetricCard label="Edges" value={metadata.edgeCount.toLocaleString()} />
        <MetricCard label="Words" value={metadata.totalWords.toLocaleString()} />
        <MetricCard label="Terminals" value={totalTerminals.toLocaleString()} />
        <MetricCard label="Max Depth" value={String(metadata.maxDepth)} />
        <MetricCard label="Peak Depth" value={String(peakDepth)} sub="most nodes" />
        <MetricCard label="Phonemes" value={String(metadata.phonemeInventory.length)} />
        <MetricCard label="Avg Entropy" value={avgEntropy.toFixed(2)} sub="bits" />
      </div>

      <SectionTitle>Languages ({metadata.languages.length})</SectionTitle>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 20 }}>
        {languages.map((l) => (
          <span
            key={l.code}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 5,
              padding: "3px 8px",
              background: "rgba(255,255,255,0.04)",
              borderRadius: 6,
              fontSize: 11,
              color: langColor(l.code),
              border: `1px solid ${langColor(l.code)}22`,
            }}
          >
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: langColor(l.code) }} />
            {l.name}
            <span style={{ fontSize: 9, color: "#555" }}>{l.family}</span>
          </span>
        ))}
      </div>

      {/* Mini depth chart */}
      {depthStats.length > 0 && (
        <>
          <SectionTitle>Node Distribution by Depth</SectionTitle>
          <MiniBarChart
            data={depthStats.filter((d) => d.depth > 0 && d.depth <= 30)}
            valueKey="nodes"
            labelKey="depth"
            color="rgba(78, 205, 196, 0.6)"
            height={100}
          />
        </>
      )}

      {/* Motifs */}
      {metadata.motifs.length > 0 && (
        <>
          <SectionTitle>Top Motifs</SectionTitle>
          <div style={{ fontSize: 11 }}>
            {metadata.motifs.slice(0, 8).map((m) => (
              <div
                key={m.label}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "4px 0",
                  borderBottom: "1px solid rgba(255,255,255,0.03)",
                }}
              >
                <span style={{ color: "#c9a539", fontFamily: "monospace" }}>/{m.label}/</span>
                <span style={{ color: "#555", fontVariantNumeric: "tabular-nums" }}>{m.count.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// ─── Depth Tab ──────────────────────────────────────────────────────────────

function DepthTab({
  depthStats,
  crossStats,
}: {
  depthStats: DepthStats[];
  crossStats: CrossLingStat[];
}) {
  const [metric, setMetric] = useState<"nodes" | "terminals" | "avgBranch" | "avgEntropy" | "maxEntropy">("nodes");
  const filtered = depthStats.filter((d) => d.depth > 0 && d.depth <= 30);

  return (
    <div>
      <SectionTitle>Depth Statistics</SectionTitle>

      {/* Metric selector */}
      <div style={{ display: "flex", gap: 4, marginBottom: 16, flexWrap: "wrap" }}>
        {(["nodes", "terminals", "avgBranch", "avgEntropy", "maxEntropy"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMetric(m)}
            style={{
              padding: "4px 10px",
              background: metric === m ? "rgba(255,255,255,0.1)" : "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: 6,
              color: metric === m ? "#ddd" : "#666",
              fontSize: 10,
              cursor: "pointer",
              fontFamily: "'Inter', sans-serif",
            }}
          >
            {m === "avgBranch" ? "branching" : m === "avgEntropy" ? "entropy" : m === "maxEntropy" ? "max entropy" : m}
          </button>
        ))}
      </div>

      {/* Global chart */}
      <MiniBarChart data={filtered} valueKey={metric} labelKey="depth" color="rgba(78, 205, 196, 0.6)" height={140} />

      {/* Per-language overlay for entropy/branching */}
      {(metric === "avgEntropy" || metric === "avgBranch") && crossStats.length > 0 && (
        <>
          <SectionTitle>Per-Language {metric === "avgEntropy" ? "Entropy" : "Branching"}</SectionTitle>
          <MultiLineChart crossStats={crossStats} metric={metric} maxDepth={30} height={180} />
        </>
      )}

      {/* Data table */}
      <SectionTitle>Raw Data</SectionTitle>
      <div style={{ fontSize: 10, overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "#555", textTransform: "uppercase", letterSpacing: 0.8 }}>
              <th style={thStyle}>Depth</th>
              <th style={thStyle}>Nodes</th>
              <th style={thStyle}>Terms</th>
              <th style={thStyle}>Branch</th>
              <th style={thStyle}>Entropy</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((d) => (
              <tr key={d.depth} style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
                <td style={tdStyle}>{d.depth}</td>
                <td style={tdStyle}>{d.nodes.toLocaleString()}</td>
                <td style={tdStyle}>{d.terminals.toLocaleString()}</td>
                <td style={tdStyle}>{d.avgBranch.toFixed(2)}</td>
                <td style={tdStyle}>{d.avgEntropy.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Languages Tab ──────────────────────────────────────────────────────────

function LanguagesTab({
  languages,
  crossStats,
}: {
  languages: LanguageInfo[];
  crossStats: CrossLingStat[];
}) {
  const [expanded, setExpanded] = useState<string | null>(null);

  // Group by family
  const families = new Map<string, LanguageInfo[]>();
  for (const lang of languages) {
    const list = families.get(lang.family) ?? [];
    list.push(lang);
    families.set(lang.family, list);
  }

  return (
    <div>
      {Array.from(families.entries()).map(([family, langs]) => (
        <div key={family} style={{ marginBottom: 20 }}>
          <SectionTitle>{family}</SectionTitle>
          {langs.map((lang) => {
            const stats = crossStats.find((s) => s.language === lang.code)?.stats ?? [];
            const totalNodes = stats.reduce((s, d) => s + d.nodes, 0);
            const totalTerminals = stats.reduce((s, d) => s + d.terminals, 0);
            const maxDepth = stats.length > 0 ? stats[stats.length - 1].depth : 0;
            const isOpen = expanded === lang.code;

            return (
              <div
                key={lang.code}
                style={{
                  marginBottom: 8,
                  background: "rgba(255,255,255,0.02)",
                  borderRadius: 10,
                  border: `1px solid ${langColor(lang.code)}15`,
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
                    padding: "10px 14px",
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                    fontFamily: "'Inter', sans-serif",
                    textAlign: "left",
                  }}
                >
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: langColor(lang.code), flexShrink: 0 }} />
                  <span style={{ color: langColor(lang.code), fontWeight: 600, fontSize: 13, flex: 1 }}>
                    {lang.name}
                  </span>
                  <span style={{ fontSize: 9, color: "#555" }}>{lang.typology}</span>
                  <span style={{ fontSize: 10, color: "#666", fontVariantNumeric: "tabular-nums" }}>
                    {totalNodes.toLocaleString()} nodes
                  </span>
                  <span style={{ color: "#444", fontSize: 12 }}>{isOpen ? "−" : "+"}</span>
                </button>

                {isOpen && stats.length > 0 && (
                  <div style={{ padding: "0 14px 14px" }}>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 12 }}>
                      <MetricCard label="Nodes" value={totalNodes.toLocaleString()} small />
                      <MetricCard label="Terminals" value={totalTerminals.toLocaleString()} small />
                      <MetricCard label="Max Depth" value={String(maxDepth)} small />
                    </div>

                    <div style={{ fontSize: 9, color: "#555", marginBottom: 4 }}>Node count by depth</div>
                    <MiniBarChart
                      data={stats.filter((d) => d.depth > 0)}
                      valueKey="nodes"
                      labelKey="depth"
                      color={langColor(lang.code) + "88"}
                      height={70}
                    />

                    <div style={{ fontSize: 9, color: "#555", marginTop: 10, marginBottom: 4 }}>Entropy by depth</div>
                    <MiniBarChart
                      data={stats.filter((d) => d.depth > 0)}
                      valueKey="avgEntropy"
                      labelKey="depth"
                      color={langColor(lang.code) + "66"}
                      height={50}
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

// ─── Inventory Tab ──────────────────────────────────────────────────────────

function InventoryTab({ metadata }: { metadata: TrieMetadata }) {
  return (
    <div>
      <SectionTitle>Full Phoneme Inventory ({metadata.phonemeInventory.length})</SectionTitle>
      <PhonemeGrid phonemes={metadata.phonemeInventory} />

      <SectionTitle>Onset Inventory ({metadata.onsetInventory.length})</SectionTitle>
      <PhonemeGrid phonemes={metadata.onsetInventory} color="#39a5c9" />

      <SectionTitle>Coda Inventory ({metadata.codaInventory.length})</SectionTitle>
      <PhonemeGrid phonemes={metadata.codaInventory} color="#9539c9" />
    </div>
  );
}

function PhonemeGrid({ phonemes, color = "#c9a539" }: { phonemes: string[]; color?: string }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 20 }}>
      {phonemes.map((p) => (
        <span
          key={p}
          style={{
            display: "inline-block",
            padding: "2px 6px",
            background: "rgba(255,255,255,0.03)",
            borderRadius: 4,
            fontSize: 12,
            fontFamily: "monospace",
            color,
            border: `1px solid ${color}22`,
          }}
        >
          {p}
        </span>
      ))}
    </div>
  );
}

// ─── Shared Components ──────────────────────────────────────────────────────

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h3
      style={{
        fontSize: 10,
        fontWeight: 600,
        color: "#555",
        textTransform: "uppercase",
        letterSpacing: 1.2,
        marginBottom: 10,
        marginTop: 20,
      }}
    >
      {children}
    </h3>
  );
}

function MetricCard({
  label,
  value,
  sub,
  small,
}: {
  label: string;
  value: string;
  sub?: string;
  small?: boolean;
}) {
  return (
    <div
      style={{
        background: "rgba(255,255,255,0.03)",
        borderRadius: 8,
        padding: small ? "6px 10px" : "10px 14px",
        border: "1px solid rgba(255,255,255,0.04)",
      }}
    >
      <div style={{ fontSize: small ? 14 : 18, fontWeight: 600, color: "#e0e0e8", fontVariantNumeric: "tabular-nums" }}>
        {value}
      </div>
      <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 0.5, marginTop: 2 }}>
        {label}
        {sub && <span style={{ marginLeft: 4, color: "#444" }}>{sub}</span>}
      </div>
    </div>
  );
}

function MiniBarChart({
  data,
  valueKey,
  labelKey,
  color,
  height,
}: {
  data: DepthStats[];
  valueKey: keyof DepthStats;
  labelKey: keyof DepthStats;
  color: string;
  height: number;
}) {
  if (data.length === 0) return null;
  const vals = data.map((d) => Number(d[valueKey]) || 0);
  const max = Math.max(...vals, 0.001);
  const barW = Math.max(2, Math.min(12, (372 - data.length) / data.length));

  return (
    <div style={{ position: "relative", height, display: "flex", alignItems: "flex-end", gap: 1, marginBottom: 8 }}>
      {data.map((d, i) => {
        const val = vals[i];
        const pct = val / max;
        return (
          <div
            key={i}
            title={`${String(d[labelKey])}: ${val % 1 !== 0 ? val.toFixed(3) : val.toLocaleString()}`}
            style={{
              flex: `0 0 ${barW}px`,
              height: `${Math.max(1, pct * 100)}%`,
              background: color,
              borderRadius: "2px 2px 0 0",
              transition: "height 0.3s ease",
            }}
          />
        );
      })}
    </div>
  );
}

function MultiLineChart({
  crossStats,
  metric,
  maxDepth,
  height,
}: {
  crossStats: CrossLingStat[];
  metric: "avgEntropy" | "avgBranch";
  maxDepth: number;
  height: number;
}) {
  // Find global max for Y axis
  let globalMax = 0;
  for (const cs of crossStats) {
    for (const s of cs.stats) {
      if (s.depth > 0 && s.depth <= maxDepth) {
        const val = s[metric];
        if (val > globalMax) globalMax = val;
      }
    }
  }
  if (globalMax === 0) return null;

  const w = 372;
  const pad = 2;
  const plotW = w - pad * 2;
  const plotH = height - 20;

  return (
    <div style={{ marginBottom: 16 }}>
      <svg width={w} height={height} style={{ display: "block" }}>
        {/* Grid lines */}
        {[0.25, 0.5, 0.75].map((frac) => (
          <line
            key={frac}
            x1={pad}
            x2={w - pad}
            y1={plotH * (1 - frac) + 10}
            y2={plotH * (1 - frac) + 10}
            stroke="rgba(255,255,255,0.04)"
          />
        ))}

        {/* Lines per language */}
        {crossStats.map((cs) => {
          const points = cs.stats
            .filter((s) => s.depth > 0 && s.depth <= maxDepth)
            .map((s) => {
              const x = pad + (s.depth / maxDepth) * plotW;
              const y = 10 + plotH * (1 - s[metric] / globalMax);
              return `${x},${y}`;
            });
          if (points.length < 2) return null;
          return (
            <polyline
              key={cs.language}
              points={points.join(" ")}
              fill="none"
              stroke={langColor(cs.language)}
              strokeWidth={1.5}
              strokeOpacity={0.7}
              strokeLinejoin="round"
            />
          );
        })}
      </svg>

      {/* Legend */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 6 }}>
        {crossStats.map((cs) => (
          <span key={cs.language} style={{ display: "inline-flex", alignItems: "center", gap: 3, fontSize: 9, color: "#666" }}>
            <span style={{ width: 10, height: 2, background: langColor(cs.language), borderRadius: 1 }} />
            {cs.language}
          </span>
        ))}
      </div>
    </div>
  );
}

const thStyle: React.CSSProperties = {
  padding: "6px 8px",
  textAlign: "right",
  fontWeight: 600,
  borderBottom: "1px solid rgba(255,255,255,0.06)",
};

const tdStyle: React.CSSProperties = {
  padding: "4px 8px",
  textAlign: "right",
  fontVariantNumeric: "tabular-nums",
  color: "#888",
};

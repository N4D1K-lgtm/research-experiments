import { useState, useMemo } from "react";
import { useQuery } from "urql";
import {
  DEPTH_STATS_QUERY,
  CROSS_LINGUISTIC_STATS_QUERY,
} from "../graphql/queries";
import type { DepthStats } from "../types/trie";
import { COLORS, FONTS, langColor, langLabel, hexAlpha } from "../styles/theme";
import { InfoTooltip } from "./InfoTooltip";

interface CrossLingStat {
  language: string;
  stats: DepthStats[];
}

export function AnalysisView() {
  const [depthResult] = useQuery({ query: DEPTH_STATS_QUERY });
  const [crossResult] = useQuery({ query: CROSS_LINGUISTIC_STATS_QUERY });

  const depthStats: DepthStats[] = depthResult.data?.depthStats ?? [];
  const crossStats: CrossLingStat[] = crossResult.data?.crossLinguisticStats ?? [];

  if (depthResult.fetching || crossResult.fetching) {
    return <ViewLoading label="Loading analysis data..." />;
  }

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
          <h1
            style={{
              fontSize: 28,
              fontWeight: 600,
              color: COLORS.textBright,
              fontFamily: FONTS.mono,
              marginBottom: 8,
            }}
          >
            Depth & Entropy Analysis
          </h1>
          <p style={{ fontSize: 13, color: COLORS.textDim, lineHeight: 1.7, maxWidth: 640 }}>
            How branching factor and information entropy evolve as phoneme sequences grow deeper.
            Each depth level represents one more phoneme in the pronunciation path.
          </p>
        </header>

        {/* Entropy + Branching dual-axis chart */}
        <EntropyBranchChart crossStats={crossStats} />

        {/* Convergence chart */}
        <ConvergenceChart crossStats={crossStats} />

        {/* Depth stats table */}
        <DepthTable depthStats={depthStats} />
      </div>
    </div>
  );
}

// ── Entropy + Branching Chart ────────────────────────────────────────────────

function EntropyBranchChart({ crossStats }: { crossStats: CrossLingStat[] }) {
  const [hovered, setHovered] = useState<{ lang: string; depth: number; entropy: number; branch: number } | null>(null);

  const { maxBranch, maxEntropy, maxDepth } = useMemo(() => {
    let mb = 0, me = 0, md = 0;
    for (const cs of crossStats) {
      for (const s of cs.stats) {
        mb = Math.max(mb, s.avgBranch);
        me = Math.max(me, s.avgEntropy);
        md = Math.max(md, s.depth);
      }
    }
    return {
      maxBranch: Math.ceil(mb + 1),
      maxEntropy: Math.ceil(me * 2) / 2 + 0.5,
      maxDepth: Math.min(md, 14),
    };
  }, [crossStats]);

  const W = 880, H = 340;
  const pad = { top: 40, right: 70, bottom: 50, left: 60 };
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  return (
    <section style={{ marginBottom: 48 }}>
      <SectionHeader
        title="Entropy & Branching by Depth"
        info="Branching factor (solid lines) shows how many phonemes typically follow at each depth. Entropy (dashed lines) measures the unpredictability of transitions in bits. Both tend to decrease at deeper levels as word paths narrow."
      />

      <div
        style={{
          background: COLORS.bgCard,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 12,
          padding: "20px 20px 16px",
          position: "relative",
        }}
      >
        <svg width={W} height={H} style={{ display: "block", width: "100%", height: "auto" }} viewBox={`0 0 ${W} ${H}`}>
          {/* Grid lines */}
          {[0.2, 0.4, 0.6, 0.8].map((frac) => (
            <line
              key={frac}
              x1={pad.left} x2={W - pad.right}
              y1={pad.top + ph * (1 - frac)}
              y2={pad.top + ph * (1 - frac)}
              stroke="rgba(255,255,255,0.04)"
              strokeWidth={0.5}
            />
          ))}

          {/* Per-language branching (solid) + entropy (dashed) */}
          {crossStats.map((cs) => {
            const points = cs.stats.filter((s) => s.depth > 0 && s.depth <= maxDepth);
            if (points.length < 2) return null;

            const branchLine = points.map((s) => {
              const x = pad.left + (s.depth / maxDepth) * pw;
              const y = pad.top + (1 - s.avgBranch / maxBranch) * ph;
              return `${x},${y}`;
            }).join(" ");

            const entropyLine = points.map((s) => {
              const x = pad.left + (s.depth / maxDepth) * pw;
              const y = pad.top + (1 - s.avgEntropy / maxEntropy) * ph;
              return `${x},${y}`;
            }).join(" ");

            return (
              <g key={cs.language}>
                {/* Branching — solid */}
                <polyline
                  points={branchLine}
                  fill="none"
                  stroke={langColor(cs.language)}
                  strokeWidth={2}
                  strokeOpacity={0.75}
                  strokeLinejoin="round"
                />
                {/* Entropy — dashed */}
                <polyline
                  points={entropyLine}
                  fill="none"
                  stroke={langColor(cs.language)}
                  strokeWidth={1.5}
                  strokeOpacity={0.35}
                  strokeDasharray="5,4"
                  strokeLinejoin="round"
                />
                {/* Interactive dots for branching */}
                {points.map((s) => {
                  const x = pad.left + (s.depth / maxDepth) * pw;
                  const y = pad.top + (1 - s.avgBranch / maxBranch) * ph;
                  return (
                    <circle
                      key={`${cs.language}-${s.depth}`}
                      cx={x} cy={y} r={3}
                      fill={langColor(cs.language)}
                      fillOpacity={0.5}
                      onMouseEnter={() => setHovered({ lang: cs.language, depth: s.depth, entropy: s.avgEntropy, branch: s.avgBranch })}
                      onMouseLeave={() => setHovered(null)}
                      style={{ cursor: "pointer" }}
                    />
                  );
                })}
              </g>
            );
          })}

          {/* X axis labels */}
          {Array.from({ length: Math.floor(maxDepth / 2) + 1 }, (_, i) => i * 2).map((d) => (
            <text
              key={d}
              x={pad.left + (d / maxDepth) * pw}
              y={H - pad.bottom + 22}
              fill={COLORS.textDim}
              fontSize={10}
              fontFamily={FONTS.mono}
              textAnchor="middle"
            >
              {d}
            </text>
          ))}
          <text
            x={pad.left + pw / 2}
            y={H - 6}
            fill={COLORS.textFaint}
            fontSize={10}
            fontFamily={FONTS.mono}
            textAnchor="middle"
          >
            phoneme depth
          </text>

          {/* Left Y axis — branching */}
          {[0, 0.25, 0.5, 0.75, 1].map((frac) => (
            <text
              key={`lb-${frac}`}
              x={pad.left - 10}
              y={pad.top + ph * (1 - frac) + 4}
              fill={COLORS.textDim}
              fontSize={9}
              fontFamily={FONTS.mono}
              textAnchor="end"
            >
              {(frac * maxBranch).toFixed(0)}
            </text>
          ))}
          <text
            x={14}
            y={pad.top + ph / 2}
            fill={COLORS.textFaint}
            fontSize={9}
            fontFamily={FONTS.mono}
            textAnchor="middle"
            transform={`rotate(-90, 14, ${pad.top + ph / 2})`}
          >
            avg branching (solid)
          </text>

          {/* Right Y axis — entropy */}
          {[0, 0.25, 0.5, 0.75, 1].map((frac) => (
            <text
              key={`re-${frac}`}
              x={W - pad.right + 10}
              y={pad.top + ph * (1 - frac) + 4}
              fill={COLORS.textFaint}
              fontSize={9}
              fontFamily={FONTS.mono}
              textAnchor="start"
            >
              {(frac * maxEntropy).toFixed(1)}
            </text>
          ))}
          <text
            x={W - 10}
            y={pad.top + ph / 2}
            fill={COLORS.textFaint}
            fontSize={9}
            fontFamily={FONTS.mono}
            textAnchor="middle"
            transform={`rotate(90, ${W - 10}, ${pad.top + ph / 2})`}
          >
            entropy bits (dashed)
          </text>

          {/* Hover tooltip */}
          {hovered && (
            <g>
              <rect
                x={pad.left + (hovered.depth / maxDepth) * pw + 8}
                y={pad.top + (1 - hovered.branch / maxBranch) * ph - 30}
                width={140}
                height={44}
                rx={6}
                fill="rgba(10,10,18,0.94)"
                stroke={COLORS.borderLight}
              />
              <text
                x={pad.left + (hovered.depth / maxDepth) * pw + 16}
                y={pad.top + (1 - hovered.branch / maxBranch) * ph - 14}
                fill={langColor(hovered.lang)}
                fontSize={10}
                fontFamily={FONTS.mono}
                fontWeight={600}
              >
                {langLabel(hovered.lang)} d={hovered.depth}
              </text>
              <text
                x={pad.left + (hovered.depth / maxDepth) * pw + 16}
                y={pad.top + (1 - hovered.branch / maxBranch) * ph + 2}
                fill={COLORS.textDim}
                fontSize={9}
                fontFamily={FONTS.mono}
              >
                branch: {hovered.branch.toFixed(2)} · H: {hovered.entropy.toFixed(3)}
              </text>
            </g>
          )}
        </svg>

        {/* Legend */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 12, paddingLeft: pad.left }}>
          {crossStats.map((cs) => (
            <span
              key={cs.language}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 5,
                fontSize: 10,
                color: COLORS.textDim,
              }}
            >
              <span
                style={{
                  width: 14,
                  height: 3,
                  background: langColor(cs.language),
                  borderRadius: 1,
                }}
              />
              {langLabel(cs.language)}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}

// ── Convergence Chart ────────────────────────────────────────────────────────

function ConvergenceChart({ crossStats }: { crossStats: CrossLingStat[] }) {
  const maxDepth = 14;

  const { dimCurves, maxDim } = useMemo(() => {
    const curves: Record<string, number[]> = {};
    let md = 0;
    for (const cs of crossStats) {
      const dims: number[] = [0];
      for (let d = 1; d <= maxDepth; d++) {
        const s = cs.stats.find((x) => x.depth === d);
        const logPop = s && s.nodes > 0 ? Math.log2(s.nodes) : dims[dims.length - 1];
        dims.push(logPop);
      }
      curves[cs.language] = dims;
      md = Math.max(md, ...dims);
    }
    return { dimCurves: curves, maxDim: Math.ceil(md / 5) * 5 + 2 };
  }, [crossStats]);

  const W = 880, H = 320;
  const pad = { top: 40, right: 24, bottom: 50, left: 60 };
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  // Convergence band
  const bandMin = 14, bandMax = 18;
  const yBandTop = pad.top + (1 - bandMax / maxDim) * ph;
  const yBandBot = pad.top + (1 - bandMin / maxDim) * ph;

  return (
    <section style={{ marginBottom: 48 }}>
      <SectionHeader
        title="Cross-Linguistic Convergence"
        info="log₂(shell population) measures addressing complexity — how many bits are needed to identify a node at each depth. Despite wildly different phonologies, all languages converge to a similar peak complexity around 2¹⁵-2¹⁸ distinct paths. This hints at shared cognitive or articulatory constraints."
      />

      <div
        style={{
          background: COLORS.bgCard,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 12,
          padding: "20px 20px 16px",
        }}
      >
        <svg width={W} height={H} style={{ display: "block", width: "100%", height: "auto" }} viewBox={`0 0 ${W} ${H}`}>
          {/* Grid */}
          {[0.2, 0.4, 0.6, 0.8].map((frac) => (
            <line
              key={frac}
              x1={pad.left} x2={W - pad.right}
              y1={pad.top + ph * (1 - frac)}
              y2={pad.top + ph * (1 - frac)}
              stroke="rgba(255,255,255,0.04)"
              strokeWidth={0.5}
            />
          ))}

          {/* Convergence band */}
          <rect
            x={pad.left} y={yBandTop}
            width={pw} height={yBandBot - yBandTop}
            fill={hexAlpha(COLORS.accent, 0.05)}
          />
          <line x1={pad.left} x2={W - pad.right} y1={yBandTop} y2={yBandTop}
            stroke={hexAlpha(COLORS.accent, 0.15)} strokeWidth={0.5} strokeDasharray="4,4" />
          <line x1={pad.left} x2={W - pad.right} y1={yBandBot} y2={yBandBot}
            stroke={hexAlpha(COLORS.accent, 0.15)} strokeWidth={0.5} strokeDasharray="4,4" />
          <text
            x={W - pad.right - 4}
            y={yBandTop + 14}
            fill={hexAlpha(COLORS.accent, 0.35)}
            fontSize={9}
            fontFamily={FONTS.mono}
            textAnchor="end"
          >
            peak complexity ~ 2^15-18 paths
          </text>

          {/* Lines */}
          {Object.entries(dimCurves).map(([lang, dims]) => {
            const points = dims.map((d, i) => {
              const x = pad.left + (i / maxDepth) * pw;
              const y = pad.top + (1 - d / maxDim) * ph;
              return `${x},${y}`;
            }).join(" ");

            return (
              <g key={lang}>
                <polyline
                  points={points}
                  fill="none"
                  stroke={langColor(lang)}
                  strokeWidth={2}
                  strokeOpacity={0.8}
                  strokeLinejoin="round"
                />
                {dims.map((d, i) => (
                  <circle
                    key={i}
                    cx={pad.left + (i / maxDepth) * pw}
                    cy={pad.top + (1 - d / maxDim) * ph}
                    r={2.5}
                    fill={langColor(lang)}
                    fillOpacity={0.5}
                  />
                ))}
              </g>
            );
          })}

          {/* Axes */}
          {Array.from({ length: Math.floor(maxDepth / 2) + 1 }, (_, i) => i * 2).map((d) => (
            <text
              key={d}
              x={pad.left + (d / maxDepth) * pw}
              y={H - pad.bottom + 22}
              fill={COLORS.textDim}
              fontSize={10}
              fontFamily={FONTS.mono}
              textAnchor="middle"
            >
              {d}
            </text>
          ))}
          <text x={pad.left + pw / 2} y={H - 6} fill={COLORS.textFaint} fontSize={10} fontFamily={FONTS.mono} textAnchor="middle">
            phoneme depth
          </text>

          {[0, 0.25, 0.5, 0.75, 1].map((frac) => (
            <text
              key={`y-${frac}`}
              x={pad.left - 10}
              y={pad.top + ph * (1 - frac) + 4}
              fill={COLORS.textDim}
              fontSize={9}
              fontFamily={FONTS.mono}
              textAnchor="end"
            >
              {(frac * maxDim).toFixed(0)}
            </text>
          ))}
          <text
            x={14}
            y={pad.top + ph / 2}
            fill={COLORS.textFaint}
            fontSize={9}
            fontFamily={FONTS.mono}
            textAnchor="middle"
            transform={`rotate(-90, 14, ${pad.top + ph / 2})`}
          >
            log₂(shell population)
          </text>
        </svg>

        {/* Legend */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 12, paddingLeft: pad.left }}>
          {crossStats.map((cs) => (
            <span
              key={cs.language}
              style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 10, color: COLORS.textDim }}
            >
              <span style={{ width: 14, height: 3, background: langColor(cs.language), borderRadius: 1 }} />
              {langLabel(cs.language)}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}

// ── Depth Table ────────────────────────────────────────────────────────────

function DepthTable({ depthStats }: { depthStats: DepthStats[] }) {
  const filtered = depthStats.filter((d) => d.depth > 0 && d.depth <= 30);

  return (
    <section>
      <SectionHeader
        title="Depth Statistics"
        info="Per-depth breakdown of the trie structure. Branching factor is the average number of children per node. Entropy measures transition unpredictability in bits — higher entropy means more evenly distributed transitions."
      />

      <div
        style={{
          background: COLORS.bgCard,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 12,
          padding: 16,
          overflowX: "auto",
        }}
      >
        <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: FONTS.mono, fontSize: 11 }}>
          <thead>
            <tr>
              {["Depth", "Nodes", "Terminals", "Branching", "Avg Entropy", "Max Entropy"].map((h) => (
                <th
                  key={h}
                  style={{
                    padding: "8px 12px",
                    textAlign: "right",
                    fontWeight: 600,
                    color: COLORS.textDim,
                    borderBottom: `1px solid ${COLORS.border}`,
                    fontSize: 9,
                    textTransform: "uppercase",
                    letterSpacing: 0.8,
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((d) => (
              <tr
                key={d.depth}
                style={{ borderBottom: `1px solid rgba(255,255,255,0.02)` }}
              >
                <td style={tdStyle}>{d.depth}</td>
                <td style={tdStyle}>{d.nodes.toLocaleString()}</td>
                <td style={tdStyle}>{d.terminals.toLocaleString()}</td>
                <td style={tdStyle}>{d.avgBranch.toFixed(2)}</td>
                <td style={tdStyle}>{d.avgEntropy.toFixed(3)}</td>
                <td style={tdStyle}>{d.maxEntropy.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ── Shared ──────────────────────────────────────────────────────────────────

function SectionHeader({ title, info }: { title: string; info?: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", marginBottom: 14 }}>
      <h2
        style={{
          fontSize: 16,
          fontWeight: 600,
          color: COLORS.textBright,
          fontFamily: FONTS.mono,
          margin: 0,
        }}
      >
        {title}
      </h2>
      {info && <InfoTooltip text={info} maxWidth={320} />}
    </div>
  );
}

function ViewLoading({ label }: { label: string }) {
  return (
    <div
      style={{
        position: "fixed",
        left: 56,
        top: 0,
        right: 0,
        bottom: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: COLORS.bg,
        fontFamily: FONTS.sans,
      }}
    >
      <div style={{ textAlign: "center" }}>
        <div
          style={{
            width: 24,
            height: 24,
            border: "2px solid rgba(255,255,255,0.08)",
            borderTopColor: "rgba(255,255,255,0.4)",
            borderRadius: "50%",
            animation: "spin 0.8s linear infinite",
            margin: "0 auto 12px",
          }}
        />
        <span style={{ fontSize: 12, color: COLORS.textFaint }}>{label}</span>
      </div>
    </div>
  );
}

const tdStyle: React.CSSProperties = {
  padding: "6px 12px",
  textAlign: "right",
  fontVariantNumeric: "tabular-nums",
  color: "#888",
};

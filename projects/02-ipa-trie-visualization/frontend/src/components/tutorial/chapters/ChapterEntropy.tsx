import { useState, useMemo } from "react";
import { useQuery } from "urql";
import { DEPTH_STATS_QUERY, CROSS_LINGUISTIC_STATS_QUERY } from "../../../graphql/queries";
import type { DepthStats } from "../../../types/trie";
import { COLORS, FONTS, langColor, langLabel, hexAlpha } from "../../../styles/theme";
import {
  Chapter, ChapterTitle, Prose, M, MathBlock, Insight, Definition,
  Widget, Tag, Divider,
} from "../shared";

interface CrossLingStat { language: string; stats: DepthStats[] }

export function ChapterEntropy() {
  const [depthResult] = useQuery({ query: DEPTH_STATS_QUERY });
  const [crossResult] = useQuery({ query: CROSS_LINGUISTIC_STATS_QUERY });
  const depthStats: DepthStats[] = depthResult.data?.depthStats ?? [];
  const crossStats: CrossLingStat[] = crossResult.data?.crossLinguisticStats ?? [];

  return (
    <Chapter id="entropy">
      <ChapterTitle
        number={5}
        title="Information & Entropy"
        subtitle="At each node in our trie, the next phoneme is uncertain. Shannon entropy measures exactly how uncertain — in bits."
      />

      <Prose>
        After the phoneme <M>/k/</M>, what comes next? In English, it might be <M>/æ/</M> (cat),
        <M>/ɔ/</M> (caught), <M>/aʊ/</M> (cow), or dozens of other phonemes. Some are far more
        likely than others. Entropy captures this: if all transitions are equally likely, entropy
        is high (maximum surprise). If one transition dominates, entropy is low (predictable).
      </Prose>

      <MathBlock label="Shannon Entropy">
        H(X) = −Σ p(xᵢ) · log₂ p(xᵢ)
      </MathBlock>

      <Definition term="Shannon Entropy">
        Measures the average information content (in bits) of a random variable. For a node with{" "}
        <M>n</M> possible next phonemes, each with probability <M>p(xᵢ)</M>:
        <ul style={{ margin: "8px 0 0 16px", lineHeight: 2 }}>
          <li><strong>H = 0</strong> → completely deterministic (only one possible next phoneme)</li>
          <li><strong>H = log₂(n)</strong> → maximum entropy (all transitions equally likely)</li>
          <li>Real nodes fall between these extremes</li>
        </ul>
      </Definition>

      <Divider />

      <Prose>
        Try it yourself. Drag the probability sliders below and watch how entropy changes.
        When one outcome dominates, entropy drops. When outcomes are balanced, entropy peaks.
      </Prose>

      <Widget label="Entropy Calculator" instructions="Drag sliders to set transition probabilities. Watch entropy respond.">
        <EntropyPlayground />
      </Widget>

      <Divider />

      <Prose>
        Now let's look at what entropy actually looks like across our trie. At shallow depths,
        many phonemes are possible — entropy is high. As we go deeper, paths narrow and entropy
        drops. But the <em>rate</em> of this drop varies dramatically between languages.
      </Prose>

      <Widget label="Live Data" instructions="This chart uses real data from the backend. Solid lines = branching factor, dashed = entropy.">
        <EntropyDepthChart crossStats={crossStats} />
      </Widget>

      <Prose>
        The dual-axis chart reveals a key relationship: <strong>branching factor</strong> (how many children
        a node has) and <strong>entropy</strong> (how uncertain the choice is) are related but not identical.
        A node can have many children but low entropy if one transition has 90% probability. Conversely,
        a node with just 3 children can have high entropy if each is equally likely.
      </Prose>

      <MathBlock label="Branching vs Entropy">
        H ≤ log₂(branching factor){" · "}
        Equality ⟺ uniform distribution
      </MathBlock>

      <Insight>
        Agglutinative languages like Finnish and Turkish show slower entropy decay — their morphological
        richness means the trie stays "surprised" at deeper levels. Isolating languages like Mandarin
        compress into shorter, bushier tries with rapid entropy collapse.
      </Insight>
    </Chapter>
  );
}

// ── Interactive Entropy Calculator ──────────────────────────────────────────

function EntropyPlayground() {
  const [probs, setProbs] = useState([0.4, 0.3, 0.2, 0.1]);
  const [labels] = useState(["a", "e", "i", "o"]);

  // Normalize
  const total = probs.reduce((s, p) => s + p, 0);
  const normalized = probs.map((p) => (total > 0 ? p / total : 1 / probs.length));

  // Entropy
  const entropy = normalized.reduce((h, p) => {
    if (p <= 0) return h;
    return h - p * Math.log2(p);
  }, 0);

  const maxEntropy = Math.log2(probs.length);
  const efficiency = maxEntropy > 0 ? entropy / maxEntropy : 0;

  const updateProb = (idx: number, val: number) => {
    const next = [...probs];
    next[idx] = Math.max(0.01, val);
    setProbs(next);
  };

  const setUniform = () => setProbs(probs.map(() => 1));
  const setSkewed = () => setProbs([0.85, 0.08, 0.05, 0.02]);

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
        <button onClick={setUniform} style={presetStyle}>Uniform</button>
        <button onClick={setSkewed} style={presetStyle}>Skewed</button>
        <button onClick={() => setProbs([0.5, 0.5, 0, 0])} style={presetStyle}>Binary</button>
        <button onClick={() => setProbs([1, 0, 0, 0])} style={presetStyle}>Deterministic</button>
      </div>

      {/* Sliders */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 20 }}>
        {probs.map((p, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontFamily: FONTS.mono, fontSize: 16, color: COLORS.accent, minWidth: 28 }}>
              /{labels[i]}/
            </span>
            <input
              type="range"
              min={0}
              max={100}
              value={p * 100}
              onChange={(e) => updateProb(i, Number(e.target.value) / 100)}
              style={{ flex: 1, accentColor: COLORS.accent }}
            />
            <span style={{ fontFamily: FONTS.mono, fontSize: 12, color: COLORS.textDim, minWidth: 42, textAlign: "right" }}>
              {(normalized[i] * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>

      {/* Result display */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 12,
          padding: "16px",
          background: `${COLORS.accent}06`,
          border: `1px solid ${COLORS.accent}18`,
          borderRadius: 10,
        }}
      >
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 28, fontWeight: 600, fontFamily: FONTS.mono, color: COLORS.textBright }}>
            {entropy.toFixed(3)}
          </div>
          <div style={{ fontSize: 9, color: COLORS.textDim, textTransform: "uppercase", letterSpacing: 0.8 }}>
            bits of entropy
          </div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 28, fontWeight: 600, fontFamily: FONTS.mono, color: COLORS.textBright }}>
            {maxEntropy.toFixed(3)}
          </div>
          <div style={{ fontSize: 9, color: COLORS.textDim, textTransform: "uppercase", letterSpacing: 0.8 }}>
            max possible
          </div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 28, fontWeight: 600, fontFamily: FONTS.mono, color: efficiency > 0.8 ? COLORS.accent : efficiency > 0.5 ? COLORS.yellow : COLORS.red }}>
            {(efficiency * 100).toFixed(0)}%
          </div>
          <div style={{ fontSize: 9, color: COLORS.textDim, textTransform: "uppercase", letterSpacing: 0.8 }}>
            efficiency
          </div>
        </div>
      </div>

      {/* Visual bar breakdown */}
      <div style={{ display: "flex", height: 8, borderRadius: 4, overflow: "hidden", marginTop: 12 }}>
        {normalized.map((p, i) => (
          <div
            key={i}
            style={{
              width: `${p * 100}%`,
              background: [COLORS.accent, COLORS.red, COLORS.yellow, COLORS.purple][i],
              transition: "width 0.2s ease",
            }}
          />
        ))}
      </div>
    </div>
  );
}

const presetStyle: React.CSSProperties = {
  padding: "4px 10px",
  background: "rgba(255,255,255,0.04)",
  border: `1px solid rgba(255,255,255,0.06)`,
  borderRadius: 5,
  color: "#888",
  fontSize: 10,
  fontFamily: "'Inter', sans-serif",
  cursor: "pointer",
};

// ── Entropy by Depth Chart ──────────────────────────────────────────────────

function EntropyDepthChart({ crossStats }: { crossStats: CrossLingStat[] }) {
  const { maxBranch, maxEntropy, maxDepth } = useMemo(() => {
    let mb = 0, me = 0, md = 0;
    for (const cs of crossStats) {
      for (const s of cs.stats) {
        mb = Math.max(mb, s.avgBranch);
        me = Math.max(me, s.avgEntropy);
        md = Math.max(md, s.depth);
      }
    }
    return { maxBranch: Math.ceil(mb + 1), maxEntropy: Math.ceil(me * 2) / 2 + 0.5, maxDepth: Math.min(md, 14) };
  }, [crossStats]);

  if (crossStats.length === 0) return <div style={{ color: COLORS.textFaint, fontSize: 12 }}>Loading...</div>;

  const W = 680, H = 300;
  const pad = { top: 36, right: 60, bottom: 44, left: 52 };
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  return (
    <div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
        {/* Grid */}
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1={pad.left} x2={W - pad.right} y1={pad.top + ph * (1 - f)} y2={pad.top + ph * (1 - f)} stroke="rgba(255,255,255,0.04)" strokeWidth={0.5} />
        ))}

        {/* Per-language lines */}
        {crossStats.map((cs) => {
          const pts = cs.stats.filter((s) => s.depth > 0 && s.depth <= maxDepth);
          if (pts.length < 2) return null;
          const branchLine = pts.map((s) => `${pad.left + (s.depth / maxDepth) * pw},${pad.top + (1 - s.avgBranch / maxBranch) * ph}`).join(" ");
          const entropyLine = pts.map((s) => `${pad.left + (s.depth / maxDepth) * pw},${pad.top + (1 - s.avgEntropy / maxEntropy) * ph}`).join(" ");
          return (
            <g key={cs.language}>
              <polyline points={branchLine} fill="none" stroke={langColor(cs.language)} strokeWidth={2} strokeOpacity={0.7} strokeLinejoin="round" />
              <polyline points={entropyLine} fill="none" stroke={langColor(cs.language)} strokeWidth={1.5} strokeOpacity={0.3} strokeDasharray="5,4" strokeLinejoin="round" />
            </g>
          );
        })}

        {/* X axis */}
        {Array.from({ length: Math.floor(maxDepth / 2) + 1 }, (_, i) => i * 2).map((d) => (
          <text key={d} x={pad.left + (d / maxDepth) * pw} y={H - pad.bottom + 18} fill={COLORS.textDim} fontSize={9} fontFamily={FONTS.mono} textAnchor="middle">{d}</text>
        ))}
        <text x={pad.left + pw / 2} y={H - 4} fill={COLORS.textFaint} fontSize={9} fontFamily={FONTS.mono} textAnchor="middle">phoneme depth</text>

        {/* Left Y */}
        {[0, 0.5, 1].map((f) => (
          <text key={`l${f}`} x={pad.left - 8} y={pad.top + ph * (1 - f) + 3} fill={COLORS.textDim} fontSize={8} fontFamily={FONTS.mono} textAnchor="end">{(f * maxBranch).toFixed(0)}</text>
        ))}
        <text x={10} y={pad.top + ph / 2} fill={COLORS.textFaint} fontSize={8} fontFamily={FONTS.mono} textAnchor="middle" transform={`rotate(-90,10,${pad.top + ph / 2})`}>branching (solid)</text>

        {/* Right Y */}
        {[0, 0.5, 1].map((f) => (
          <text key={`r${f}`} x={W - pad.right + 8} y={pad.top + ph * (1 - f) + 3} fill={COLORS.textFaint} fontSize={8} fontFamily={FONTS.mono} textAnchor="start">{(f * maxEntropy).toFixed(1)}</text>
        ))}
        <text x={W - 6} y={pad.top + ph / 2} fill={COLORS.textFaint} fontSize={8} fontFamily={FONTS.mono} textAnchor="middle" transform={`rotate(90,${W - 6},${pad.top + ph / 2})`}>entropy bits (dashed)</text>
      </svg>

      {/* Legend */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 8 }}>
        {crossStats.map((cs) => (
          <span key={cs.language} style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 9, color: COLORS.textDim }}>
            <span style={{ width: 12, height: 3, background: langColor(cs.language), borderRadius: 1 }} />
            {langLabel(cs.language)}
          </span>
        ))}
      </div>
    </div>
  );
}

import { useMemo } from "react";
import { useQuery } from "urql";
import { LANGUAGES_QUERY, CROSS_LINGUISTIC_STATS_QUERY } from "../../../graphql/queries";
import type { DepthStats, LanguageInfo } from "../../../types/trie";
import { COLORS, FONTS, langColor, langLabel, hexAlpha } from "../../../styles/theme";
import {
  Chapter, ChapterTitle, Prose, M, MathBlock, Insight, Definition,
  Widget, Tag, Divider,
} from "../shared";

interface CrossLingStat { language: string; stats: DepthStats[] }

export function ChapterCrossLinguistic() {
  const [langResult] = useQuery({ query: LANGUAGES_QUERY });
  const [crossResult] = useQuery({ query: CROSS_LINGUISTIC_STATS_QUERY });
  const languages: LanguageInfo[] = langResult.data?.languages ?? [];
  const crossStats: CrossLingStat[] = crossResult.data?.crossLinguisticStats ?? [];

  return (
    <Chapter id="crossling">
      <ChapterTitle
        number={7}
        title="The Universal Shape"
        subtitle="Despite wildly different phonologies, morphologies, and histories — all languages converge to the same structural shape. Why?"
      />

      <Prose>
        We have data from {languages.length} languages spanning {new Set(languages.map(l => l.family)).size} families:
        Germanic (English, German, Dutch), Romance (French, Spanish), Sino-Tibetan (Mandarin),
        Japonic (Japanese), Semitic (Arabic), Uralic (Finnish), Turkic (Turkish), Indo-Aryan (Hindi),
        and Bantu (Swahili). These languages differ in almost every possible dimension.
      </Prose>

      <Widget label="Language Families">
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {languages.map((l) => (
            <div
              key={l.code}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "6px 12px",
                background: `${langColor(l.code)}08`,
                border: `1px solid ${langColor(l.code)}20`,
                borderRadius: 8,
              }}
            >
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: langColor(l.code) }} />
              <span style={{ fontSize: 12, fontFamily: FONTS.mono, color: langColor(l.code), fontWeight: 500 }}>
                {langLabel(l.code)}
              </span>
              <span style={{ fontSize: 9, color: COLORS.textFaint }}>{l.family} · {l.typology}</span>
            </div>
          ))}
        </div>
      </Widget>

      <Prose>
        Yet when we measure the <em>addressing complexity</em> — log₂ of the shell population at
        each depth — all languages converge to a band around 2¹⁵ to 2¹⁸. This means roughly
        32,000 to 262,000 distinct paths at peak branching, regardless of whether the language is
        agglutinative, isolating, fusional, or polysynthetic.
      </Prose>

      <MathBlock label="Addressing Complexity">
        C(d) = log₂ |S_d|
      </MathBlock>

      <Prose>
        Where <M>|S_d|</M> is the number of nodes at depth <M>d</M>. This quantity measures
        how many bits you need to specify a node at that depth — the "addressing space" of the
        phonological structure.
      </Prose>

      <Divider />

      <Widget label="Convergence Chart" instructions="All languages peak around 15-18 bits of addressing complexity despite having very different phonologies.">
        <ConvergenceChart crossStats={crossStats} />
      </Widget>

      <Insight>
        This convergence is remarkable. It suggests that the capacity of human speech production
        and perception imposes a universal constraint on phonological structure — roughly 2¹⁶ to 2¹⁷
        distinguishable phoneme sequences at peak complexity, regardless of which phonemes a language
        uses or how its morphology works.
      </Insight>

      <Divider />

      <Prose>
        The table below shows the convergence depth for each language — the depth at which branching
        factor drops below 2 (meaning paths are narrowing rather than branching). Agglutinative
        languages (Finnish, Turkish) converge later because their rich morphology sustains branching
        deeper into the trie.
      </Prose>

      <Widget label="Convergence Table">
        <ConvergenceTable languages={languages} crossStats={crossStats} />
      </Widget>

      <Definition term="Why Convergence?">
        Three forces constrain the trie's shape:<br />
        <strong>1. Articulatory limits</strong> — the human vocal tract can only produce ~600 distinct
        phonemes, limiting branching factor.<br />
        <strong>2. Perceptual limits</strong> — listeners need to distinguish phonemes in real-time,
        putting an upper bound on inventory size.<br />
        <strong>3. Lexical pressure</strong> — languages need enough words to be useful (~50,000+),
        but not so many that they overflow working memory. This constrains the tree's total volume.
      </Definition>
    </Chapter>
  );
}

function ConvergenceChart({ crossStats }: { crossStats: CrossLingStat[] }) {
  const maxDepth = 14;

  const { dimCurves, maxDim } = useMemo(() => {
    const curves: Record<string, number[]> = {};
    let md = 0;
    for (const cs of crossStats) {
      const dims: number[] = [0];
      for (let d = 1; d <= maxDepth; d++) {
        const s = cs.stats.find((x) => x.depth === d);
        dims.push(s && s.nodes > 0 ? Math.log2(s.nodes) : dims[dims.length - 1]);
      }
      curves[cs.language] = dims;
      md = Math.max(md, ...dims);
    }
    return { dimCurves: curves, maxDim: Math.ceil(md / 5) * 5 + 2 };
  }, [crossStats]);

  if (crossStats.length === 0) return null;

  const W = 680, H = 300;
  const pad = { top: 36, right: 20, bottom: 44, left: 52 };
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  const bandMin = 14, bandMax = 18;
  const yTop = pad.top + (1 - bandMax / maxDim) * ph;
  const yBot = pad.top + (1 - bandMin / maxDim) * ph;

  return (
    <div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
        {/* Grid */}
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1={pad.left} x2={W - pad.right} y1={pad.top + ph * (1 - f)} y2={pad.top + ph * (1 - f)} stroke="rgba(255,255,255,0.04)" strokeWidth={0.5} />
        ))}

        {/* Convergence band */}
        <rect x={pad.left} y={yTop} width={pw} height={yBot - yTop} fill={hexAlpha(COLORS.accent, 0.05)} />
        <line x1={pad.left} x2={W - pad.right} y1={yTop} y2={yTop} stroke={hexAlpha(COLORS.accent, 0.15)} strokeWidth={0.5} strokeDasharray="4,4" />
        <line x1={pad.left} x2={W - pad.right} y1={yBot} y2={yBot} stroke={hexAlpha(COLORS.accent, 0.15)} strokeWidth={0.5} strokeDasharray="4,4" />
        <text x={W - pad.right - 4} y={yTop + 12} fill={hexAlpha(COLORS.accent, 0.3)} fontSize={8} fontFamily={FONTS.mono} textAnchor="end">
          convergence band ~ 2¹⁵-2¹⁸
        </text>

        {/* Lines */}
        {Object.entries(dimCurves).map(([lang, dims]) => (
          <g key={lang}>
            <polyline
              points={dims.map((d, i) => `${pad.left + (i / maxDepth) * pw},${pad.top + (1 - d / maxDim) * ph}`).join(" ")}
              fill="none" stroke={langColor(lang)} strokeWidth={2} strokeOpacity={0.8} strokeLinejoin="round"
            />
            {dims.map((d, i) => (
              <circle key={i} cx={pad.left + (i / maxDepth) * pw} cy={pad.top + (1 - d / maxDim) * ph} r={2} fill={langColor(lang)} fillOpacity={0.5} />
            ))}
          </g>
        ))}

        {/* Axes */}
        {Array.from({ length: Math.floor(maxDepth / 2) + 1 }, (_, i) => i * 2).map((d) => (
          <text key={d} x={pad.left + (d / maxDepth) * pw} y={H - pad.bottom + 18} fill={COLORS.textDim} fontSize={9} fontFamily={FONTS.mono} textAnchor="middle">{d}</text>
        ))}
        <text x={pad.left + pw / 2} y={H - 4} fill={COLORS.textFaint} fontSize={9} fontFamily={FONTS.mono} textAnchor="middle">phoneme depth</text>
        {[0, 0.5, 1].map((f) => (
          <text key={`y${f}`} x={pad.left - 8} y={pad.top + ph * (1 - f) + 3} fill={COLORS.textDim} fontSize={8} fontFamily={FONTS.mono} textAnchor="end">{(f * maxDim).toFixed(0)}</text>
        ))}
        <text x={10} y={pad.top + ph / 2} fill={COLORS.textFaint} fontSize={8} fontFamily={FONTS.mono} textAnchor="middle" transform={`rotate(-90,10,${pad.top + ph / 2})`}>log₂(shell population)</text>
      </svg>

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

function ConvergenceTable({ languages, crossStats }: { languages: LanguageInfo[]; crossStats: CrossLingStat[] }) {
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: FONTS.mono, fontSize: 11 }}>
      <thead>
        <tr>
          {["Language", "Family", "Typology", "Peak Shell", "Converge d"].map((h) => (
            <th key={h} style={{ padding: "8px 10px", textAlign: h === "Language" ? "left" : "right", fontWeight: 600, color: COLORS.textDim, borderBottom: `1px solid ${COLORS.border}`, fontSize: 9, textTransform: "uppercase", letterSpacing: 0.8 }}>
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {languages.map((lang) => {
          const stats = crossStats.find((cs) => cs.language === lang.code)?.stats ?? [];
          const peak = stats.reduce((best, s) => (s.nodes > best.nodes ? s : best), stats[0] ?? { depth: 0, nodes: 0 });
          const converge = stats.find((s) => s.depth > 1 && s.avgBranch < 2)?.depth ?? "—";
          return (
            <tr key={lang.code} style={{ borderBottom: `1px solid rgba(255,255,255,0.02)` }}>
              <td style={{ padding: "6px 10px", textAlign: "left" }}>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <span style={{ width: 6, height: 6, borderRadius: "50%", background: langColor(lang.code) }} />
                  <span style={{ color: langColor(lang.code), fontWeight: 500 }}>{langLabel(lang.code)}</span>
                </span>
              </td>
              <td style={{ padding: "6px 10px", textAlign: "right", color: "#888" }}>{lang.family}</td>
              <td style={{ padding: "6px 10px", textAlign: "right", color: "#888" }}>{lang.typology}</td>
              <td style={{ padding: "6px 10px", textAlign: "right", color: "#888" }}>
                {peak.nodes.toLocaleString()} <span style={{ color: COLORS.textFaint }}>(d={peak.depth})</span>
              </td>
              <td style={{ padding: "6px 10px", textAlign: "right", color: "#888" }}>{converge}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

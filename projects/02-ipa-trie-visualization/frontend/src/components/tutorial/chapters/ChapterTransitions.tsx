import { useState, useMemo } from "react";
import { useTrieDataStore } from "../../../store/trieDataStore";
import { COLORS, FONTS, hexAlpha } from "../../../styles/theme";
import { getRoleColor } from "../../../utils/languagePalette";
import {
  Chapter, ChapterTitle, Prose, M, MathBlock, Insight, Definition,
  Widget, Tag, Divider,
} from "../shared";

export function ChapterTransitions() {
  const nodes = useTrieDataStore((s) => s.nodes);

  // Build a transition matrix from depth-1 nodes
  const depth1Nodes = useMemo(() => {
    const d1: { phoneme: string; count: number; position: string; transitions: Record<string, number> }[] = [];
    for (const n of nodes.values()) {
      if (n.depth === 1 && Object.keys(n.transitionProbs).length > 0) {
        d1.push({
          phoneme: n.phoneme,
          count: n.totalCount,
          position: n.phonologicalPosition,
          transitions: n.transitionProbs,
        });
      }
    }
    d1.sort((a, b) => b.count - a.count);
    return d1;
  }, [nodes]);

  const [selectedPhoneme, setSelectedPhoneme] = useState<string | null>(null);
  const selected = depth1Nodes.find((n) => n.phoneme === selectedPhoneme) ?? depth1Nodes[0] ?? null;

  return (
    <Chapter id="transitions" dark>
      <ChapterTitle
        number={6}
        title="Transition Probabilities"
        subtitle="The trie isn't just a structure — it encodes a probability distribution. At every node, we can ask: given this phoneme, what's the probability of each next phoneme?"
      />

      <Prose>
        These transition probabilities are the empirical <em>bigram probabilities</em> of phoneme
        sequences. They're computed directly from the trie: for node <M>x</M>, the probability of
        transitioning to child <M>y</M> is simply the ratio of paths through <M>y</M> to total
        paths through <M>x</M>.
      </Prose>

      <MathBlock label="Transition Probability">
        P(y | x) = count(x → y) / count(x) = weight(y) / weight(x)
      </MathBlock>

      <Prose>
        This is a first-order Markov model: the probability of the next phoneme depends only on the
        current phoneme, not the full history. It's a simplification — real phonotactics involve
        longer-range dependencies — but it captures the dominant patterns.
      </Prose>

      <Definition term="Transition Matrix">
        A square matrix where entry <M>(i,j)</M> gives <M>P(phoneme_j | phoneme_i)</M>. Each row
        sums to 1. The matrix is typically sparse: most phoneme pairs have zero probability because
        that transition never occurs in the corpus. The non-zero entries reveal the phonotactic
        constraints of the language.
      </Definition>

      <Divider />

      <Prose>
        Select a phoneme below to see what follows it in the trie. The bar heights show the
        probability of each transition — the wider bars are the "highways" through phoneme space.
      </Prose>

      <Widget label="Transition Explorer" instructions="Click a phoneme to see its transition distribution">
        {/* Phoneme selector */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 20 }}>
          {depth1Nodes.slice(0, 24).map((n) => {
            const isSelected = selectedPhoneme === n.phoneme || (!selectedPhoneme && n === depth1Nodes[0]);
            return (
              <button
                key={n.phoneme}
                onClick={() => setSelectedPhoneme(n.phoneme)}
                style={{
                  padding: "5px 10px",
                  background: isSelected ? `${getRoleColor(n.position)}18` : "rgba(255,255,255,0.03)",
                  border: `1px solid ${isSelected ? getRoleColor(n.position) + "40" : COLORS.border}`,
                  borderRadius: 6,
                  color: isSelected ? getRoleColor(n.position) : COLORS.textDim,
                  fontSize: 14,
                  fontFamily: FONTS.mono,
                  fontWeight: isSelected ? 600 : 400,
                  cursor: "pointer",
                }}
              >
                /{n.phoneme}/
              </button>
            );
          })}
        </div>

        {/* Transition display */}
        {selected && <TransitionDisplay node={selected} />}
      </Widget>

      <Prose>
        Notice how transitions are highly non-uniform. After <M>/s/</M>, you're very likely to see
        a plosive (<M>/t/</M>, <M>/k/</M>, <M>/p/</M>) — these are the dominant "s-clusters" in
        Germanic languages. After a vowel, the distribution is much flatter. This non-uniformity is
        precisely what entropy measures.
      </Prose>

      <MathBlock label="Entropy from Transitions">
        H(node) = −Σ P(child | node) · log₂ P(child | node)
      </MathBlock>

      <Insight>
        In the 3D visualization, these probabilities directly control the layout. High-probability
        transitions get shorter edges (the child is pulled closer to the parent), creating visual
        "highways" — thick, bright paths that represent the most common phoneme sequences. Low-probability
        transitions create long, faint edges that reach out to the periphery.
      </Insight>
    </Chapter>
  );
}

function TransitionDisplay({ node }: { node: { phoneme: string; count: number; position: string; transitions: Record<string, number> } }) {
  const sorted = Object.entries(node.transitions)
    .sort(([, a], [, b]) => b - a);

  const top = sorted.slice(0, 16);
  const remaining = sorted.length - top.length;
  const roleColor = getRoleColor(node.position);

  // Compute entropy
  const entropy = sorted.reduce((h, [, p]) => {
    if (p <= 0) return h;
    return h - p * Math.log2(p);
  }, 0);
  const maxEntropy = Math.log2(sorted.length);

  return (
    <div>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 16 }}>
        <span style={{ fontSize: 32, fontFamily: FONTS.mono, fontWeight: 600, color: roleColor }}>
          /{node.phoneme}/
        </span>
        <span style={{ fontSize: 11, color: COLORS.textDim, fontFamily: FONTS.mono }}>
          {node.count.toLocaleString()} paths · {sorted.length} transitions · H = {entropy.toFixed(3)} bits
          {maxEntropy > 0 && ` (${((entropy / maxEntropy) * 100).toFixed(0)}% of max)`}
        </span>
      </div>

      {/* Bars */}
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        {top.map(([phoneme, prob]) => (
          <div key={phoneme} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ minWidth: 32, fontFamily: FONTS.mono, fontSize: 13, fontWeight: 500, color: COLORS.textBright, textAlign: "right" }}>
              /{phoneme}/
            </span>
            <div style={{ flex: 1, height: 8, background: "rgba(255,255,255,0.04)", borderRadius: 4, overflow: "hidden" }}>
              <div
                style={{
                  height: "100%",
                  width: `${Math.min(100, prob * 100)}%`,
                  background: `linear-gradient(90deg, ${roleColor}66, ${roleColor})`,
                  borderRadius: 4,
                  transition: "width 0.3s ease",
                }}
              />
            </div>
            <span style={{ minWidth: 48, textAlign: "right", fontFamily: FONTS.mono, fontSize: 11, color: COLORS.textDim, fontVariantNumeric: "tabular-nums" }}>
              {(prob * 100).toFixed(1)}%
            </span>
          </div>
        ))}
        {remaining > 0 && (
          <div style={{ fontSize: 10, color: COLORS.textFaint, fontFamily: FONTS.mono, paddingLeft: 40 }}>
            +{remaining} more transitions (each &lt; {(top[top.length - 1]?.[1] * 100).toFixed(1)}%)
          </div>
        )}
      </div>
    </div>
  );
}

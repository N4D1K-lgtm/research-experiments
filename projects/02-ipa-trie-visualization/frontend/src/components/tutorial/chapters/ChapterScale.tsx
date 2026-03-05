import { useQuery } from "urql";
import { DEPTH_STATS_QUERY } from "../../../graphql/queries";
import type { DepthStats } from "../../../types/trie";
import { useTrieDataStore } from "../../../store/trieDataStore";
import { COLORS, FONTS, hexAlpha } from "../../../styles/theme";
import {
  Chapter, ChapterTitle, Prose, M, MathBlock, Insight, Definition,
  Widget, Divider,
} from "../shared";

export function ChapterScale() {
  const metadata = useTrieDataStore((s) => s.metadata);
  const [depthResult] = useQuery({ query: DEPTH_STATS_QUERY });
  const depthStats: DepthStats[] = depthResult.data?.depthStats ?? [];
  const filtered = depthStats.filter((d) => d.depth > 0 && d.depth <= 30);
  const maxNodes = Math.max(...filtered.map((d) => d.nodes), 1);
  const peakDepth = filtered.reduce((best, d) => (d.nodes > best.nodes ? d : best), filtered[0] ?? { depth: 0, nodes: 0 });

  return (
    <Chapter id="scale">
      <ChapterTitle
        number={4}
        title="Scaling to Hundreds of Thousands"
        subtitle="Our toy trie held a handful of words. The real one holds every pronunciation from 12 languages — and its shape tells a story."
      />

      <Prose>
        The pipeline ingests pronunciation data from WikiPron, CMU Dict, and other sources across
        12 languages spanning 7 language families. Each word is tokenized into IPA phonemes and
        inserted into a single shared trie. The result: <M>{metadata?.nodeCount.toLocaleString() ?? "664,017"}</M> nodes,{" "}
        <M>{metadata?.totalWords?.toLocaleString() ?? "269,737"}</M> words.
      </Prose>

      <Definition term="Radial Layout">
        To visualize a trie this large, we use a <em>spherical radial layout</em>. The root sits at
        the center. Each depth level forms a concentric shell. A node's angular position is determined
        by a weighted blend of its transition probability (60%) and subtree weight (40%), so high-frequency
        paths occupy more angular space.

        <MathBlock label="Edge Length">
          ℓ(d) = L₀ · 0.88ᵈ / √(2.5 · p)
        </MathBlock>

        where <M>L₀ = 40</M> is the base edge length, <M>d</M> is depth, and <M>p</M> is the
        transition probability to this node. High-probability transitions produce shorter, tighter edges.
      </Definition>

      <Divider />

      <Prose>
        The distribution of nodes across depth levels reveals the trie's "shape". Most languages
        peak in node count around depth 3–5, where the tree branches most heavily. Beyond that,
        paths converge and the trie thins out.
      </Prose>

      <Widget label="Live Data" instructions="Each bar represents one depth level in the real trie. Height = node count.">
        <div style={{ marginBottom: 8, display: "flex", justifyContent: "space-between", fontSize: 10, color: COLORS.textDim, fontFamily: FONTS.mono }}>
          <span>depth 1</span>
          <span>depth {filtered[filtered.length - 1]?.depth ?? "?"}</span>
        </div>

        {/* Bar chart */}
        <div style={{ display: "flex", alignItems: "flex-end", gap: 1, height: 160 }}>
          {filtered.map((d) => {
            const pct = d.nodes / maxNodes;
            const isPeak = d.depth === peakDepth.depth;
            return (
              <div
                key={d.depth}
                title={`Depth ${d.depth}: ${d.nodes.toLocaleString()} nodes, ${d.terminals.toLocaleString()} terminals, branching ${d.avgBranch.toFixed(2)}`}
                style={{
                  flex: 1,
                  height: `${Math.max(1, pct * 100)}%`,
                  background: isPeak
                    ? `linear-gradient(to top, ${COLORS.accent}88, ${COLORS.accent})`
                    : `${COLORS.accent}55`,
                  borderRadius: "2px 2px 0 0",
                  position: "relative",
                  cursor: "help",
                }}
              >
                {isPeak && (
                  <div
                    style={{
                      position: "absolute",
                      top: -18,
                      left: "50%",
                      transform: "translateX(-50%)",
                      fontSize: 8,
                      fontFamily: FONTS.mono,
                      color: COLORS.accent,
                      whiteSpace: "nowrap",
                    }}
                  >
                    peak: d={d.depth}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Stats row */}
        <div style={{ display: "flex", gap: 20, marginTop: 16, fontSize: 10, fontFamily: FONTS.mono, color: COLORS.textDim }}>
          <span>peak at depth {peakDepth.depth}: {peakDepth.nodes.toLocaleString()} nodes</span>
          <span>total: {depthStats.reduce((s, d) => s + d.nodes, 0).toLocaleString()} nodes</span>
          <span>total terminals: {depthStats.reduce((s, d) => s + d.terminals, 0).toLocaleString()}</span>
        </div>
      </Widget>

      <Prose>
        Notice the asymmetry: the trie grows explosively in the first few levels (high branching),
        then tapers off. This shape is remarkably consistent across languages — a universal
        property that emerges from the statistics of phoneme sequencing, not from any language-specific rule.
      </Prose>

      <MathBlock label="Shell Population">
        |Sₐ| ≈ |S₁| · ∏ᵢ₌₂ᵈ bf(i)
      </MathBlock>

      <Prose>
        The population of shell <M>d</M> is roughly the product of branching factors at each
        preceding level. Since branching factors decay (from ~5 at depth 1 to ~1.5 by depth 6),
        the trie's growth is sub-exponential. It peaks, then declines — a fingerprint of the
        finite vocabulary of human speech.
      </Prose>

      <Insight>
        The 3D visualization at the end of this tutorial renders all{" "}
        {metadata?.nodeCount.toLocaleString() ?? "664,017"} nodes using instanced geometry. Node size
        encodes entropy (40%), incoming probability (35%), and frequency (25%). Terminal nodes (word
        endpoints) glow additively and scale 1.4× larger. The GPU renders all of this in real-time
        with LOD-based depth culling.
      </Insight>
    </Chapter>
  );
}

import { useState, useMemo } from "react";
import { COLORS, FONTS } from "../../../styles/theme";
import {
  Chapter, ChapterTitle, Prose, M, MathBlock, Insight, Definition,
  Widget, Divider,
} from "../shared";

// Mini trie data structure for the interactive builder
interface TNode {
  char: string;
  children: Map<string, TNode>;
  isWord: boolean;
  words: string[];
  depth: number;
}

function buildTrie(words: string[]): TNode {
  const root: TNode = { char: "·", children: new Map(), isWord: false, words: [], depth: 0 };
  for (const word of words) {
    let node = root;
    for (let i = 0; i < word.length; i++) {
      const ch = word[i];
      if (!node.children.has(ch)) {
        node.children.set(ch, { char: ch, children: new Map(), isWord: false, words: [], depth: i + 1 });
      }
      node = node.children.get(ch)!;
    }
    node.isWord = true;
    if (!node.words.includes(word)) node.words.push(word);
  }
  return root;
}

function countNodes(n: TNode): number {
  let c = 1;
  n.children.forEach((ch) => { c += countNodes(ch); });
  return c;
}

function maxDepth(n: TNode): number {
  let m = n.depth;
  n.children.forEach((ch) => { m = Math.max(m, maxDepth(ch)); });
  return m;
}

function countLeaves(n: TNode): number {
  if (n.children.size === 0) return 1;
  let c = 0;
  n.children.forEach((ch) => { c += countLeaves(ch); });
  return c;
}

// Layout computation
interface LayoutNode {
  node: TNode;
  x: number;
  y: number;
}

function layoutTrie(root: TNode, width: number, height: number): LayoutNode[] {
  const result: LayoutNode[] = [];
  const yStep = Math.min(60, (height - 60) / Math.max(1, maxDepth(root)));

  function layout(node: TNode, xMin: number, xMax: number, y: number) {
    const x = (xMin + xMax) / 2;
    result.push({ node, x, y });

    const children = Array.from(node.children.values());
    if (children.length === 0) return;
    const total = children.reduce((s, c) => s + countLeaves(c), 0);
    let cursor = xMin;
    for (const child of children) {
      const share = countLeaves(child) / total;
      layout(child, cursor, cursor + (xMax - xMin) * share, y + yStep);
      cursor += (xMax - xMin) * share;
    }
  }

  layout(root, 30, width - 30, 30);
  return result;
}

const WORD_SETS = [
  { label: "Basics", words: ["cat", "car", "cart", "care", "can", "cap"] },
  { label: "Phonemes", words: ["pat", "bat", "sat", "mat", "fat", "rat", "hat"] },
  { label: "Prefixes", words: ["un", "undo", "under", "unit", "until", "up", "upon", "upper"] },
  { label: "Custom", words: [] },
];

export function ChapterTrie() {
  const [activeSet, setActiveSet] = useState(0);
  const [customWords, setCustomWords] = useState("cat car cart care can cap");
  const [addWord, setAddWord] = useState("");

  const words = activeSet === 3
    ? customWords.split(/\s+/).filter(Boolean)
    : WORD_SETS[activeSet].words;

  const root = useMemo(() => buildTrie(words), [words]);
  const nodes = countNodes(root);
  const naiveNodes = words.reduce((s, w) => s + w.length, 0) + words.length; // each word stored separately
  const saved = naiveNodes - nodes;

  const W = 680, H = 320;
  const layoutNodes = useMemo(() => layoutTrie(root, W, H), [root]);

  return (
    <Chapter id="trie" dark>
      <ChapterTitle
        number={3}
        title="Building a Trie"
        subtitle="A trie (from 'retrieval') is a tree where shared prefixes are stored once. Every word is a path from root to a marked node."
      />

      <Prose>
        Imagine storing every English word as its own independent sequence. "cat", "car", and "cart"
        would each start with "c-a" — redundant. A trie eliminates this redundancy: the prefix
        "ca" exists once, and branches into "t", "r", "rt".
      </Prose>

      <Definition term="Prefix Trie">
        A tree data structure where each edge represents a single character (or phoneme). A node is
        labeled "terminal" if it marks the end of a complete word. The path from root to any terminal
        node spells out a complete word. Crucially, words sharing a common prefix share the same path
        up to the point where they diverge.
      </Definition>

      <MathBlock label="Space Savings">
        {"Naive: Σ|wᵢ| = "}{naiveNodes} nodes{" · "}{" "}
        {"Trie: "}{nodes} nodes{" · "}{" "}
        {"Saved: "}{saved} nodes ({((saved / naiveNodes) * 100).toFixed(0)}% compression)
      </MathBlock>

      <Divider />

      <Widget label="Trie Builder" instructions="Select a word set or type custom words. Watch how prefix sharing compresses the structure.">
        {/* Word set selector */}
        <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
          {WORD_SETS.map((ws, i) => (
            <button
              key={i}
              onClick={() => setActiveSet(i)}
              style={{
                padding: "5px 12px",
                background: activeSet === i ? "rgba(255,255,255,0.1)" : "rgba(255,255,255,0.03)",
                border: `1px solid ${activeSet === i ? COLORS.accent + "40" : COLORS.border}`,
                borderRadius: 6,
                color: activeSet === i ? COLORS.accent : COLORS.textDim,
                fontSize: 11,
                fontFamily: FONTS.mono,
                cursor: "pointer",
              }}
            >
              {ws.label}
            </button>
          ))}
        </div>

        {/* Custom word input */}
        {activeSet === 3 && (
          <input
            value={customWords}
            onChange={(e) => setCustomWords(e.target.value)}
            placeholder="Space-separated words..."
            style={{
              width: "100%",
              padding: "10px 14px",
              background: "rgba(255,255,255,0.04)",
              border: `1px solid ${COLORS.border}`,
              borderRadius: 8,
              color: COLORS.textBright,
              fontSize: 14,
              fontFamily: FONTS.mono,
              outline: "none",
              marginBottom: 14,
            }}
          />
        )}

        {/* Word chips */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 16 }}>
          {words.map((w) => (
            <span
              key={w}
              style={{
                padding: "3px 8px",
                background: `${COLORS.accent}10`,
                border: `1px solid ${COLORS.accent}20`,
                borderRadius: 5,
                fontSize: 12,
                fontFamily: FONTS.mono,
                color: COLORS.accent,
              }}
            >
              {w}
            </span>
          ))}
        </div>

        {/* Trie visualization */}
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
          {/* Edges */}
          {layoutNodes.map((ln) => {
            const parent = layoutNodes.find((p) =>
              p.node.children.has(ln.node.char) &&
              p.node.depth === ln.node.depth - 1 &&
              p.node.children.get(ln.node.char) === ln.node,
            );
            if (!parent) return null;
            // Curved edge
            const midY = (parent.y + ln.y) / 2;
            return (
              <path
                key={`e-${ln.node.char}-${ln.node.depth}-${ln.x}`}
                d={`M ${parent.x} ${parent.y} Q ${parent.x} ${midY} ${ln.x} ${ln.y}`}
                fill="none"
                stroke={ln.node.isWord ? `${COLORS.accent}66` : "rgba(255,255,255,0.1)"}
                strokeWidth={ln.node.isWord ? 1.5 : 0.8}
              />
            );
          })}

          {/* Nodes */}
          {layoutNodes.map((ln) => {
            const r = ln.node.depth === 0 ? 7 : ln.node.isWord ? 5 : 3.5;
            return (
              <g key={`n-${ln.node.char}-${ln.node.depth}-${ln.x}`}>
                <circle
                  cx={ln.x} cy={ln.y} r={r}
                  fill={
                    ln.node.depth === 0 ? COLORS.accent
                    : ln.node.isWord ? `${COLORS.accent}cc`
                    : "rgba(255,255,255,0.15)"
                  }
                />
                {/* Label above */}
                <text
                  x={ln.x} y={ln.y - r - 6}
                  fill={ln.node.isWord ? COLORS.textBright : "rgba(255,255,255,0.5)"}
                  fontSize={ln.node.isWord ? 13 : 11}
                  fontFamily={FONTS.mono}
                  fontWeight={ln.node.isWord ? 600 : 400}
                  textAnchor="middle"
                >
                  {ln.node.char}
                </text>
                {/* Word label below terminal nodes */}
                {ln.node.isWord && ln.node.words.length > 0 && (
                  <text
                    x={ln.x} y={ln.y + r + 12}
                    fill={`${COLORS.accent}77`}
                    fontSize={9}
                    fontFamily={FONTS.mono}
                    textAnchor="middle"
                  >
                    {ln.node.words.join(", ")}
                  </text>
                )}
              </g>
            );
          })}
        </svg>

        {/* Stats */}
        <div
          style={{
            marginTop: 12,
            padding: "10px 14px",
            background: "rgba(255,255,255,0.02)",
            borderRadius: 8,
            display: "flex",
            gap: 24,
            fontSize: 11,
            fontFamily: FONTS.mono,
            color: COLORS.textDim,
          }}
        >
          <span>{words.length} words</span>
          <span>{nodes} nodes</span>
          <span>max depth {maxDepth(root)}</span>
          <span style={{ color: COLORS.accent }}>saved {saved} nodes via prefix sharing</span>
        </div>
      </Widget>

      <Prose>
        The trie reveals structure: words with the same prefix cluster together. Adding "cart"
        to a trie containing "car" costs just one new node — the "t". The deeper you go,
        the more paths diverge and the tree fans out. But some paths converge too: multiple
        words may share not just prefixes but entire internal sequences.
      </Prose>

      <Insight>
        This is exactly what we'll do with phoneme sequences. Instead of letters, our trie stores
        IPA phonemes. Instead of English words, we insert pronunciations from{" "}
        <M>12 languages</M> simultaneously. The result is a single unified structure that captures
        the shared phonological patterns of human speech.
      </Insight>
    </Chapter>
  );
}

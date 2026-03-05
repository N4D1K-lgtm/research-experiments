import { useState } from "react";
import { useTrieDataStore } from "../../../store/trieDataStore";
import { COLORS, FONTS, hexAlpha } from "../../../styles/theme";
import {
  Chapter, ChapterTitle, Prose, M, MathBlock, Insight, Definition,
  Widget, Tag, Divider,
} from "../shared";

// IPA consonant features
const IPA_FEATURES: Record<string, { voiced: boolean; place: string; manner: string }> = {
  "p": { voiced: false, place: "Bilabial", manner: "Plosive" },
  "b": { voiced: true, place: "Bilabial", manner: "Plosive" },
  "t": { voiced: false, place: "Alveolar", manner: "Plosive" },
  "d": { voiced: true, place: "Alveolar", manner: "Plosive" },
  "k": { voiced: false, place: "Velar", manner: "Plosive" },
  "\u0261": { voiced: true, place: "Velar", manner: "Plosive" },
  "m": { voiced: true, place: "Bilabial", manner: "Nasal" },
  "n": { voiced: true, place: "Alveolar", manner: "Nasal" },
  "\u014B": { voiced: true, place: "Velar", manner: "Nasal" },
  "f": { voiced: false, place: "Labiodental", manner: "Fricative" },
  "v": { voiced: true, place: "Labiodental", manner: "Fricative" },
  "\u03B8": { voiced: false, place: "Dental", manner: "Fricative" },
  "\u00F0": { voiced: true, place: "Dental", manner: "Fricative" },
  "s": { voiced: false, place: "Alveolar", manner: "Fricative" },
  "z": { voiced: true, place: "Alveolar", manner: "Fricative" },
  "\u0283": { voiced: false, place: "Post-alv.", manner: "Fricative" },
  "\u0292": { voiced: true, place: "Post-alv.", manner: "Fricative" },
  "h": { voiced: false, place: "Glottal", manner: "Fricative" },
  "j": { voiced: true, place: "Palatal", manner: "Approximant" },
  "w": { voiced: true, place: "Labio-velar", manner: "Approximant" },
  "\u0279": { voiced: true, place: "Alveolar", manner: "Approximant" },
  "l": { voiced: true, place: "Alveolar", manner: "Lateral" },
};

const IPA_VOWELS: Record<string, { height: string; backness: string; rounded: boolean }> = {
  "i": { height: "Close", backness: "Front", rounded: false },
  "u": { height: "Close", backness: "Back", rounded: true },
  "\u026A": { height: "Near-close", backness: "Front", rounded: false },
  "\u028A": { height: "Near-close", backness: "Back", rounded: true },
  "e": { height: "Close-mid", backness: "Front", rounded: false },
  "o": { height: "Close-mid", backness: "Back", rounded: true },
  "\u0259": { height: "Mid", backness: "Central", rounded: false },
  "\u025B": { height: "Open-mid", backness: "Front", rounded: false },
  "\u0254": { height: "Open-mid", backness: "Back", rounded: true },
  "\u00E6": { height: "Near-open", backness: "Front", rounded: false },
  "a": { height: "Open", backness: "Front", rounded: false },
  "\u0251": { height: "Open", backness: "Back", rounded: false },
};

const PLACES = ["Bilabial", "Labiodental", "Dental", "Alveolar", "Post-alv.", "Palatal", "Labio-velar", "Velar", "Glottal"];
const MANNERS = ["Plosive", "Nasal", "Fricative", "Approximant", "Lateral"];

export function ChapterSound() {
  const metadata = useTrieDataStore((s) => s.metadata);
  const nodes = useTrieDataStore((s) => s.nodes);

  // Count phoneme frequencies from depth-1 nodes
  const phonemeCounts: Record<string, number> = {};
  for (const n of nodes.values()) {
    if (n.depth === 1) phonemeCounts[n.phoneme] = n.totalCount;
  }
  const inventory = new Set(metadata?.phonemeInventory ?? []);

  return (
    <Chapter id="sound" dark>
      <ChapterTitle
        number={1}
        title="What is a Sound?"
        subtitle="Before we can build a tree from speech, we need a precise way to describe speech sounds. That's what the International Phonetic Alphabet is for."
      />

      <Prose>
        Writing systems are a terrible model of pronunciation. The letter "c" in English can sound
        like <M>/k/</M> in "cat", <M>/s/</M> in "city", or <M>/tʃ/</M> in "cello". Meanwhile, the same
        sound <M>/k/</M> can be spelled "c", "k", "ck", "ch", or "q" depending on the word.
      </Prose>

      <Prose>
        The IPA solves this: one symbol per sound, one sound per symbol. Every symbol encodes
        exactly <em>how</em> the sound is produced — where in the mouth (place), how the air flows
        (manner), and whether the vocal folds vibrate (voicing).
      </Prose>

      <Definition term="Phoneme">
        The smallest unit of sound that distinguishes meaning in a language. Changing a phoneme
        changes the word: <M>/k/</M>at vs <M>/b/</M>at, ca<M>/t/</M> vs ca<M>/p/</M>.
        A phoneme is abstract — it may be realized as slightly different sounds (allophones) in
        different contexts.
      </Definition>

      <Divider />

      <Prose>
        The IPA organizes consonants in a grid. Each column is a <em>place of articulation</em> — where
        you constrict air flow. Each row is a <em>manner of articulation</em> — how you constrict it.
        Hover over any sound below to see its features.
      </Prose>

      <Widget label="Interactive" instructions="Hover over any phoneme to see its articulatory features and corpus frequency">
        <ConsonantChart inventory={inventory} phonemeCounts={phonemeCounts} />
      </Widget>

      <Prose>
        Notice the pattern: sounds come in <Tag color={COLORS.purple}>voiceless</Tag> and{" "}
        <Tag color={COLORS.accent}>voiced</Tag> pairs. <M>/p/</M> and <M>/b/</M> are made the same
        way — both bilabial plosives — but <M>/b/</M> vibrates the vocal folds. This minimal pairing
        is fundamental to how languages build their sound inventories.
      </Prose>

      <Insight>
        Our corpus contains {metadata?.phonemeInventory.length ?? "100+"} distinct phonemes across{" "}
        {metadata?.languages.length ?? 12} languages. No single language uses them all — English has
        about 44, Hawaiian about 13, and !Xóõ uses over 100 including clicks. But they all draw from
        the same articulatory space.
      </Insight>

      <Divider />

      <Prose>
        Vowels are different. Instead of a grid, they're plotted in a <em>trapezoid</em> — a 2D
        space defined by tongue height (open ↔ close) and backness (front ↔ back). Your mouth is
        literally a resonating chamber, and vowels are defined by the shape of that chamber.
      </Prose>

      <Widget label="Interactive" instructions="Hover over vowels. Circled vowels are rounded (lips protruded).">
        <VowelChart inventory={inventory} phonemeCounts={phonemeCounts} />
      </Widget>

      <Definition term="Phonological Position">
        In a syllable, phonemes play different roles:
        the <Tag color={COLORS.onset}>onset</Tag> comes first (consonants that begin the syllable),
        the <Tag color={COLORS.nucleus}>nucleus</Tag> is the core (usually a vowel),
        and the <Tag color={COLORS.coda}>coda</Tag> closes it (consonants at the end).
        "strong" → <Tag color={COLORS.onset}>/s/</Tag><Tag color={COLORS.onset}>/t/</Tag><Tag color={COLORS.onset}>/ɹ/</Tag><Tag color={COLORS.nucleus}>/ɔ/</Tag><Tag color={COLORS.coda}>/ŋ/</Tag>
      </Definition>

      <Prose>
        These positions matter because languages have strict rules about which phonemes can appear
        where. English allows complex onsets like <M>/str-/</M> but Japanese forbids most consonant
        clusters entirely. These constraints are what give each language its characteristic rhythm.
      </Prose>
    </Chapter>
  );
}

function ConsonantChart({ inventory, phonemeCounts }: { inventory: Set<string>; phonemeCounts: Record<string, number> }) {
  const [hovered, setHovered] = useState<string | null>(null);
  const hoveredInfo = hovered ? IPA_FEATURES[hovered] : null;

  const cellW = 68, cellH = 40;
  const padLeft = 80, padTop = 32;
  const W = padLeft + PLACES.length * cellW;
  const H = padTop + MANNERS.length * cellH + 16;

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
      {/* Grid */}
      {PLACES.map((_, i) => (
        <line key={`v${i}`} x1={padLeft + i * cellW} y1={padTop} x2={padLeft + i * cellW} y2={padTop + MANNERS.length * cellH}
          stroke={COLORS.border} strokeWidth={0.5} />
      ))}
      <line x1={padLeft + PLACES.length * cellW} y1={padTop} x2={padLeft + PLACES.length * cellW} y2={padTop + MANNERS.length * cellH} stroke={COLORS.border} strokeWidth={0.5} />
      {MANNERS.map((_, i) => (
        <line key={`h${i}`} x1={padLeft} y1={padTop + i * cellH} x2={padLeft + PLACES.length * cellW} y2={padTop + i * cellH} stroke={COLORS.border} strokeWidth={0.5} />
      ))}
      <line x1={padLeft} y1={padTop + MANNERS.length * cellH} x2={padLeft + PLACES.length * cellW} y2={padTop + MANNERS.length * cellH} stroke={COLORS.border} strokeWidth={0.5} />

      {/* Headers */}
      {PLACES.map((p, i) => (
        <text key={p} x={padLeft + (i + 0.5) * cellW} y={padTop - 10} fill={COLORS.textDim} fontSize={8} fontFamily={FONTS.mono} textAnchor="middle">{p}</text>
      ))}
      {MANNERS.map((m, i) => (
        <text key={m} x={padLeft - 6} y={padTop + (i + 0.5) * cellH + 3} fill={COLORS.textDim} fontSize={8} fontFamily={FONTS.mono} textAnchor="end">{m}</text>
      ))}

      {/* Phonemes */}
      {Object.entries(IPA_FEATURES).map(([sym, feat]) => {
        const pi = PLACES.indexOf(feat.place);
        const mi = MANNERS.indexOf(feat.manner);
        if (pi < 0 || mi < 0) return null;
        const voiceOff = feat.voiced ? cellW * 0.25 : -cellW * 0.25;
        const x = padLeft + (pi + 0.5) * cellW + voiceOff;
        const y = padTop + (mi + 0.5) * cellH;
        const inInv = inventory.has(sym);
        const isHov = hovered === sym;
        const count = phonemeCounts[sym];
        const freqAlpha = count ? Math.min(0.2, Math.log10(count + 1) * 0.04) : 0;
        const color = feat.voiced ? COLORS.accent : COLORS.purple;

        return (
          <g key={sym}>
            {inInv && count && <circle cx={x} cy={y} r={14} fill={hexAlpha(color, freqAlpha)} />}
            {isHov && <circle cx={x} cy={y} r={16} fill={hexAlpha(color, 0.2)} />}
            <text
              x={x} y={y + 1}
              fill={hexAlpha(color, inInv ? (isHov ? 1 : 0.85) : 0.2)}
              fontSize={isHov ? 16 : 13}
              fontFamily={FONTS.mono}
              fontWeight={isHov ? 600 : 400}
              textAnchor="middle"
              dominantBaseline="central"
              onMouseEnter={() => setHovered(sym)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: "pointer" }}
            >
              {sym}
            </text>
          </g>
        );
      })}

      {/* Hover info */}
      {hovered && hoveredInfo && (
        <g>
          <rect x={W - 190} y={6} width={180} height={phonemeCounts[hovered] ? 82 : 64} rx={8} fill="rgba(10,10,18,0.95)" stroke={COLORS.borderLight} />
          <text x={W - 176} y={28} fill={hoveredInfo.voiced ? COLORS.accent : COLORS.purple} fontSize={20} fontFamily={FONTS.mono} fontWeight={600}>/{hovered}/</text>
          <text x={W - 176} y={46} fill={COLORS.textDim} fontSize={9} fontFamily={FONTS.mono}>
            [{hoveredInfo.voiced ? "+voice" : "-voice"}] {hoveredInfo.place.toLowerCase()} {hoveredInfo.manner.toLowerCase()}
          </text>
          {phonemeCounts[hovered] && (
            <text x={W - 176} y={72} fill={COLORS.textFaint} fontSize={9} fontFamily={FONTS.mono}>
              {phonemeCounts[hovered].toLocaleString()} paths in trie
            </text>
          )}
        </g>
      )}
    </svg>
  );
}

function VowelChart({ inventory, phonemeCounts }: { inventory: Set<string>; phonemeCounts: Record<string, number> }) {
  const [hovered, setHovered] = useState<string | null>(null);

  const HEIGHTS = ["Close", "Near-close", "Close-mid", "Mid", "Open-mid", "Near-open", "Open"];
  const BACKS = ["Front", "Central", "Back"];

  const W = 380, H = 280;
  const pad = { top: 36, left: 50, right: 50, bottom: 20 };
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block", maxWidth: 420 }}>
      {/* Trapezoid */}
      <polygon
        points={`${pad.left + pw * 0.12},${pad.top} ${pad.left + pw * 0.5},${pad.top} ${pad.left + pw},${pad.top} ${pad.left + pw},${pad.top + ph} ${pad.left},${pad.top + ph}`}
        fill="none" stroke={COLORS.border} strokeWidth={0.5}
      />
      {/* Height labels */}
      {HEIGHTS.map((h, i) => (
        <text key={h} x={pad.left - 6} y={pad.top + (i / (HEIGHTS.length - 1)) * ph + 3} fill={COLORS.textFaint} fontSize={7} fontFamily={FONTS.mono} textAnchor="end">{h}</text>
      ))}
      {BACKS.map((b, i) => (
        <text key={b} x={pad.left + (i / (BACKS.length - 1)) * pw} y={pad.top - 10} fill={COLORS.textFaint} fontSize={8} fontFamily={FONTS.mono} textAnchor="middle">{b}</text>
      ))}

      {Object.entries(IPA_VOWELS).map(([sym, feat]) => {
        const hi = HEIGHTS.indexOf(feat.height);
        const bi = BACKS.indexOf(feat.backness);
        if (hi < 0 || bi < 0) return null;
        const yFrac = hi / (HEIGHTS.length - 1);
        const xBase = pad.left + (bi / (BACKS.length - 1)) * pw;
        const frontNarrow = (1 - yFrac) * pw * 0.12;
        const x = bi === 0 ? xBase + frontNarrow : xBase;
        const y = pad.top + yFrac * ph;
        const inInv = inventory.has(sym);
        const isHov = hovered === sym;

        return (
          <g key={sym}>
            {isHov && <circle cx={x} cy={y} r={14} fill={hexAlpha(COLORS.nucleus, 0.15)} />}
            {feat.rounded && <circle cx={x} cy={y} r={11} fill="none" stroke={hexAlpha(COLORS.nucleus, inInv ? 0.3 : 0.1)} strokeWidth={0.8} />}
            <text
              x={x} y={y + 1}
              fill={hexAlpha(COLORS.nucleus, inInv ? (isHov ? 1 : 0.85) : 0.2)}
              fontSize={isHov ? 16 : 13}
              fontFamily={FONTS.mono}
              fontWeight={isHov ? 600 : 400}
              textAnchor="middle"
              dominantBaseline="central"
              onMouseEnter={() => setHovered(sym)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: "pointer" }}
            >
              {sym}
            </text>
          </g>
        );
      })}

      {hovered && IPA_VOWELS[hovered] && (
        <g>
          <rect x={W - 160} y={H - 65} width={150} height={50} rx={8} fill="rgba(10,10,18,0.95)" stroke={COLORS.borderLight} />
          <text x={W - 148} y={H - 44} fill={COLORS.nucleus} fontSize={18} fontFamily={FONTS.mono}>/{hovered}/</text>
          <text x={W - 148} y={H - 26} fill={COLORS.textDim} fontSize={9} fontFamily={FONTS.mono}>
            {IPA_VOWELS[hovered].height.toLowerCase()} {IPA_VOWELS[hovered].backness.toLowerCase()} {IPA_VOWELS[hovered].rounded ? "rounded" : ""}
          </text>
        </g>
      )}
    </svg>
  );
}

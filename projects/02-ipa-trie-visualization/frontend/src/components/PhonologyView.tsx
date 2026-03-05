import { useState, useRef, useEffect, useCallback } from "react";
import { useTrieDataStore } from "../store/trieDataStore";
import { COLORS, FONTS, hexAlpha } from "../styles/theme";
import { InfoTooltip } from "./InfoTooltip";

// ── IPA Consonant Classification ──────────────────────────────────────────

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
  "\u0272": { voiced: true, place: "Palatal", manner: "Nasal" },
  "c": { voiced: false, place: "Palatal", manner: "Plosive" },
  "\u025F": { voiced: true, place: "Palatal", manner: "Plosive" },
  "x": { voiced: false, place: "Velar", manner: "Fricative" },
  "\u0263": { voiced: true, place: "Velar", manner: "Fricative" },
  "\u0280": { voiced: true, place: "Uvular", manner: "Trill" },
  "r": { voiced: true, place: "Alveolar", manner: "Trill" },
  "\u027E": { voiced: true, place: "Alveolar", manner: "Tap" },
  "q": { voiced: false, place: "Uvular", manner: "Plosive" },
  "\u0294": { voiced: false, place: "Glottal", manner: "Plosive" },
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
  "\u0252": { height: "Open", backness: "Back", rounded: true },
  "\u028C": { height: "Open-mid", backness: "Back", rounded: false },
  "\u025D": { height: "Mid", backness: "Central", rounded: false },
  "y": { height: "Close", backness: "Front", rounded: true },
  "\u00F8": { height: "Close-mid", backness: "Front", rounded: true },
  "\u0153": { height: "Open-mid", backness: "Front", rounded: true },
};

const PLACES = ["Bilabial", "Labiodental", "Dental", "Alveolar", "Post-alv.", "Palatal", "Labio-velar", "Velar", "Uvular", "Glottal"];
const MANNERS = ["Plosive", "Nasal", "Trill", "Tap", "Fricative", "Approximant", "Lateral"];

// ── Tokenizer ────────────────────────────────────────────────────────────

const COMBINING_START = 0x0300;
const COMBINING_END = 0x036F;
const IPA_DIACRITICS = new Set([
  "\u0329", "\u032F", "\u0325", "\u031F", "\u0320", "\u0303", "\u02D0",
  "\u02B0", "\u02B7", "\u02B2", "\u0361", "\u035C",
]);
const SUPRASEGMENTALS = new Set(["\u02C8", "\u02CC", ".", "\u203F", "|", "\u2016"]);
const VOWEL_SET = new Set(["a", "e", "i", "o", "u", "\u0259", "\u025B", "\u0254",
  "\u026A", "\u028A", "\u00E6", "\u0251", "\u0252", "\u028C", "\u025C",
  "\u0258", "\u0275", "\u0264", "\u026F", "\u0268", "\u0289", "y",
  "\u00F8", "\u0153", "\u0276", "\u0250", "\u025E", "\u0269", "\u025D"]);

function tokenize(ipa: string): string[] {
  const clean = ipa.replace(/[/\[\]]/g, "");
  const tokens: string[] = [];
  let i = 0;
  while (i < clean.length) {
    const ch = clean[i];
    if (SUPRASEGMENTALS.has(ch) || ch === " ") { i++; continue; }
    let token = ch;
    i++;
    while (i < clean.length) {
      const next = clean[i];
      const cp = next.codePointAt(0) || 0;
      if ((cp >= COMBINING_START && cp <= COMBINING_END) || IPA_DIACRITICS.has(next)) {
        token += next; i++;
      } else break;
    }
    if (token.trim()) tokens.push(token);
  }
  return tokens;
}

const ROLE_COLORS: Record<string, string> = { onset: COLORS.onset, nucleus: COLORS.nucleus, coda: COLORS.coda, mixed: COLORS.mixed };

export function PhonologyView() {
  const metadata = useTrieDataStore((s) => s.metadata);
  const nodes = useTrieDataStore((s) => s.nodes);

  // Count phoneme frequencies from depth-1 nodes
  const phonemeCounts: Record<string, number> = {};
  for (const n of nodes.values()) {
    if (n.depth === 1) phonemeCounts[n.phoneme] = n.totalCount;
  }

  const inventory = new Set(metadata?.phonemeInventory ?? []);
  const onsetInv = new Set(metadata?.onsetInventory ?? []);
  const codaInv = new Set(metadata?.codaInventory ?? []);

  function classify(token: string): string {
    const base = token[0];
    if (VOWEL_SET.has(base)) return "nucleus";
    if (onsetInv.has(base) && !codaInv.has(base)) return "onset";
    if (codaInv.has(base) && !onsetInv.has(base)) return "coda";
    if (onsetInv.has(base)) return "onset";
    return "onset";
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
          <h1 style={{ fontSize: 28, fontWeight: 600, color: COLORS.textBright, fontFamily: FONTS.mono, marginBottom: 8 }}>
            Phonological Feature Space
          </h1>
          <p style={{ fontSize: 13, color: COLORS.textDim, lineHeight: 1.7, maxWidth: 640 }}>
            The IPA organizes speech sounds by articulatory features. Consonants vary by place and manner of articulation;
            vowels by tongue height, backness, and lip rounding.
          </p>
        </header>

        {/* IPA Consonant Chart */}
        <IPAConsonantChart inventory={inventory} phonemeCounts={phonemeCounts} />

        {/* IPA Vowel Chart */}
        <IPAVowelChart inventory={inventory} phonemeCounts={phonemeCounts} />

        {/* Tokenizer */}
        <TokenizerWidget classify={classify} />

        {/* Inventory summary */}
        {metadata && (
          <section style={{ marginTop: 48 }}>
            <SectionHeader
              title="Phoneme Inventory"
              info="All phonemes attested in the corpus across all languages, categorized by their syllable position (onset = beginning, coda = end of syllable)."
            />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
              <InventoryGrid title="Full Inventory" phonemes={metadata.phonemeInventory} color={COLORS.accent} />
              <InventoryGrid title="Onset" phonemes={metadata.onsetInventory} color={COLORS.onset} />
              <InventoryGrid title="Coda" phonemes={metadata.codaInventory} color={COLORS.coda} />
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

// ── IPA Consonant Chart ────────────────────────────────────────────────────

function IPAConsonantChart({
  inventory,
  phonemeCounts,
}: {
  inventory: Set<string>;
  phonemeCounts: Record<string, number>;
}) {
  const [hovered, setHovered] = useState<string | null>(null);
  const hoveredInfo = hovered ? IPA_FEATURES[hovered] : null;

  const cellW = 74;
  const cellH = 44;
  const padLeft = 90;
  const padTop = 36;
  const W = padLeft + PLACES.length * cellW + 20;
  const H = padTop + MANNERS.length * cellH + 60;

  return (
    <section style={{ marginBottom: 48 }}>
      <SectionHeader
        title="IPA Consonant Chart"
        info="Consonants arranged by place of articulation (columns) and manner of articulation (rows). Voiceless sounds are on the left of each cell, voiced on the right. Brightness indicates frequency in the corpus."
      />

      <div
        style={{
          background: COLORS.bgCard,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 12,
          padding: 20,
          overflowX: "auto",
        }}
      >
        <svg width={W} height={H} style={{ display: "block" }} viewBox={`0 0 ${W} ${H}`}>
          {/* Grid */}
          {PLACES.map((_, i) => (
            <line key={`v${i}`} x1={padLeft + i * cellW} y1={padTop} x2={padLeft + i * cellW} y2={padTop + MANNERS.length * cellH} stroke={COLORS.border} strokeWidth={0.5} />
          ))}
          <line x1={padLeft + PLACES.length * cellW} y1={padTop} x2={padLeft + PLACES.length * cellW} y2={padTop + MANNERS.length * cellH} stroke={COLORS.border} strokeWidth={0.5} />
          {MANNERS.map((_, i) => (
            <line key={`h${i}`} x1={padLeft} y1={padTop + i * cellH} x2={padLeft + PLACES.length * cellW} y2={padTop + i * cellH} stroke={COLORS.border} strokeWidth={0.5} />
          ))}
          <line x1={padLeft} y1={padTop + MANNERS.length * cellH} x2={padLeft + PLACES.length * cellW} y2={padTop + MANNERS.length * cellH} stroke={COLORS.border} strokeWidth={0.5} />

          {/* Column headers */}
          {PLACES.map((p, i) => (
            <text key={p} x={padLeft + (i + 0.5) * cellW} y={padTop - 10} fill={COLORS.textDim} fontSize={9} fontFamily={FONTS.mono} textAnchor="middle">
              {p}
            </text>
          ))}

          {/* Row headers */}
          {MANNERS.map((m, i) => (
            <text key={m} x={padLeft - 8} y={padTop + (i + 0.5) * cellH + 4} fill={COLORS.textDim} fontSize={9} fontFamily={FONTS.mono} textAnchor="end">
              {m}
            </text>
          ))}

          {/* Phonemes */}
          {Object.entries(IPA_FEATURES).map(([sym, feat]) => {
            const pi = PLACES.indexOf(feat.place);
            const mi = MANNERS.indexOf(feat.manner);
            if (pi < 0 || mi < 0) return null;
            const voiceOffset = feat.voiced ? cellW * 0.28 : -cellW * 0.28;
            const x = padLeft + (pi + 0.5) * cellW + voiceOffset;
            const y = padTop + (mi + 0.5) * cellH;
            const inInv = inventory.has(sym);
            const isHov = hovered === sym;
            const count = phonemeCounts[sym];
            const freqAlpha = count ? Math.min(0.2, Math.log10(count + 1) * 0.04) : 0;

            return (
              <g key={sym}>
                {inInv && count && (
                  <circle cx={x} cy={y} r={16} fill={hexAlpha(feat.voiced ? COLORS.accent : COLORS.purple, freqAlpha)} />
                )}
                {isHov && (
                  <circle cx={x} cy={y} r={18} fill={hexAlpha(feat.voiced ? COLORS.accent : COLORS.purple, 0.2)} />
                )}
                <text
                  x={x} y={y + 1}
                  fill={hexAlpha(feat.voiced ? COLORS.accent : COLORS.purple, inInv ? (isHov ? 1 : 0.85) : 0.25)}
                  fontSize={isHov ? 18 : 14}
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

          {/* Hover detail box */}
          {hovered && hoveredInfo && (
            <g>
              <rect x={W - 210} y={10} width={196} height={phonemeCounts[hovered] ? 90 : 70} rx={8} fill="rgba(10,10,18,0.94)" stroke={COLORS.borderLight} />
              <text x={W - 196} y={35} fill={hoveredInfo.voiced ? COLORS.accent : COLORS.purple} fontSize={22} fontFamily={FONTS.mono}>
                /{hovered}/
              </text>
              <text x={W - 196} y={54} fill={COLORS.textDim} fontSize={10} fontFamily={FONTS.mono}>
                [{hoveredInfo.voiced ? "+voice" : "-voice"}] {hoveredInfo.place.toLowerCase()}
              </text>
              <text x={W - 196} y={70} fill={COLORS.textDim} fontSize={10} fontFamily={FONTS.mono}>
                {hoveredInfo.manner.toLowerCase()}
              </text>
              {phonemeCounts[hovered] && (
                <text x={W - 196} y={88} fill={COLORS.textFaint} fontSize={10} fontFamily={FONTS.mono}>
                  {phonemeCounts[hovered].toLocaleString()} occurrences
                </text>
              )}
            </g>
          )}

          {/* Legend */}
          <circle cx={padLeft} cy={H - 20} r={4} fill={COLORS.purple} fillOpacity={0.6} />
          <text x={padLeft + 8} y={H - 16} fill={COLORS.textDim} fontSize={10} fontFamily={FONTS.mono}>voiceless</text>
          <circle cx={padLeft + 90} cy={H - 20} r={4} fill={COLORS.accent} fillOpacity={0.6} />
          <text x={padLeft + 98} y={H - 16} fill={COLORS.textDim} fontSize={10} fontFamily={FONTS.mono}>voiced</text>
          <text x={padLeft + 200} y={H - 16} fill={COLORS.textFaint} fontSize={9} fontFamily={FONTS.mono}>
            brightness ∝ corpus frequency
          </text>
        </svg>
      </div>
    </section>
  );
}

// ── IPA Vowel Chart ──────────────────────────────────────────────────────

function IPAVowelChart({
  inventory,
  phonemeCounts,
}: {
  inventory: Set<string>;
  phonemeCounts: Record<string, number>;
}) {
  const [hovered, setHovered] = useState<string | null>(null);

  const HEIGHTS = ["Close", "Near-close", "Close-mid", "Mid", "Open-mid", "Near-open", "Open"];
  const BACKS = ["Front", "Central", "Back"];

  const W = 440, H = 340;
  const pad = { top: 40, left: 60, right: 60, bottom: 40 };
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  return (
    <section style={{ marginBottom: 48 }}>
      <SectionHeader
        title="IPA Vowel Trapezoid"
        info="Vowels arranged by tongue height (vertical) and backness (horizontal). Rounded vowels are shown with a ring. This mirrors the standard IPA vowel chart used in phonetics."
      />

      <div
        style={{
          background: COLORS.bgCard,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 12,
          padding: 20,
          display: "inline-block",
        }}
      >
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
          {/* Trapezoid outline */}
          <polygon
            points={`${pad.left + pw * 0.15},${pad.top} ${pad.left + pw * 0.5},${pad.top} ${pad.left + pw},${pad.top} ${pad.left + pw},${pad.top + ph} ${pad.left},${pad.top + ph}`}
            fill="none"
            stroke={COLORS.border}
            strokeWidth={0.5}
          />

          {/* Height labels */}
          {HEIGHTS.map((h, i) => {
            const y = pad.top + (i / (HEIGHTS.length - 1)) * ph;
            return (
              <text key={h} x={pad.left - 8} y={y + 4} fill={COLORS.textFaint} fontSize={8} fontFamily={FONTS.mono} textAnchor="end">
                {h}
              </text>
            );
          })}

          {/* Backness labels */}
          {BACKS.map((b, i) => {
            const x = pad.left + (i / (BACKS.length - 1)) * pw;
            return (
              <text key={b} x={x} y={pad.top - 12} fill={COLORS.textFaint} fontSize={9} fontFamily={FONTS.mono} textAnchor="middle">
                {b}
              </text>
            );
          })}

          {/* Vowels */}
          {Object.entries(IPA_VOWELS).map(([sym, feat]) => {
            const hi = HEIGHTS.indexOf(feat.height);
            const bi = BACKS.indexOf(feat.backness);
            if (hi < 0 || bi < 0) return null;

            // Trapezoid x offset: front is narrower at top
            const yFrac = hi / (HEIGHTS.length - 1);
            const xBase = pad.left + (bi / (BACKS.length - 1)) * pw;
            const frontNarrow = (1 - yFrac) * pw * 0.15;
            const x = bi === 0 ? xBase + frontNarrow : xBase;
            const y = pad.top + yFrac * ph;

            const inInv = inventory.has(sym);
            const isHov = hovered === sym;
            const count = phonemeCounts[sym];

            return (
              <g key={sym}>
                {isHov && <circle cx={x} cy={y} r={16} fill={hexAlpha(COLORS.nucleus, 0.15)} />}
                {feat.rounded && (
                  <circle cx={x} cy={y} r={12} fill="none" stroke={hexAlpha(COLORS.nucleus, inInv ? 0.3 : 0.1)} strokeWidth={1} />
                )}
                <text
                  x={x} y={y + 1}
                  fill={hexAlpha(COLORS.nucleus, inInv ? (isHov ? 1 : 0.85) : 0.25)}
                  fontSize={isHov ? 18 : 14}
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
          {hovered && IPA_VOWELS[hovered] && (
            <g>
              <rect x={W - 180} y={H - 80} width={160} height={60} rx={8} fill="rgba(10,10,18,0.94)" stroke={COLORS.borderLight} />
              <text x={W - 168} y={H - 58} fill={COLORS.nucleus} fontSize={20} fontFamily={FONTS.mono}>/{hovered}/</text>
              <text x={W - 168} y={H - 40} fill={COLORS.textDim} fontSize={10} fontFamily={FONTS.mono}>
                {IPA_VOWELS[hovered].height.toLowerCase()} {IPA_VOWELS[hovered].backness.toLowerCase()}
              </text>
              <text x={W - 168} y={H - 26} fill={COLORS.textFaint} fontSize={9} fontFamily={FONTS.mono}>
                {IPA_VOWELS[hovered].rounded ? "rounded" : "unrounded"}
                {phonemeCounts[hovered] ? ` · ${phonemeCounts[hovered].toLocaleString()} occ.` : ""}
              </text>
            </g>
          )}
        </svg>
      </div>
    </section>
  );
}

// ── Tokenizer Widget ────────────────────────────────────────────────────

function TokenizerWidget({ classify }: { classify: (token: string) => string }) {
  const [input, setInput] = useState("\u02C8k\u00E6t");
  const tokens = tokenize(input);

  const examples = [
    { label: "cat", ipa: "\u02C8k\u00E6t" },
    { label: "string", ipa: "\u02C8st\u0279\u026A\u014B" },
    { label: "beautiful", ipa: "\u02C8bju\u02D0t\u026Af\u028Al" },
    { label: "thought", ipa: "\u02C8\u03B8\u0254\u02D0t" },
  ];

  return (
    <section style={{ marginBottom: 48 }}>
      <SectionHeader
        title="IPA Tokenizer"
        info="Type or paste an IPA transcription to see how it gets tokenized into individual phonemes and classified by syllable position. The tokenizer handles combining diacritics and suprasegmental markers."
      />

      <div
        style={{
          background: COLORS.bgCard,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 12,
          padding: 20,
        }}
      >
        <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
          {examples.map((ex) => (
            <button
              key={ex.label}
              onClick={() => setInput(ex.ipa)}
              style={{
                padding: "5px 12px",
                background: input === ex.ipa ? "rgba(255,255,255,0.1)" : "rgba(255,255,255,0.03)",
                border: `1px solid ${input === ex.ipa ? COLORS.accent + "40" : COLORS.border}`,
                borderRadius: 6,
                color: input === ex.ipa ? COLORS.accent : COLORS.textDim,
                fontSize: 11,
                fontFamily: FONTS.mono,
                cursor: "pointer",
              }}
            >
              {ex.label}
            </button>
          ))}
        </div>

        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type IPA here..."
          style={{
            width: "100%",
            padding: "10px 14px",
            background: "rgba(255,255,255,0.04)",
            border: `1px solid ${COLORS.border}`,
            borderRadius: 8,
            color: COLORS.textBright,
            fontSize: 18,
            fontFamily: FONTS.mono,
            outline: "none",
            marginBottom: 16,
          }}
        />

        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center", minHeight: 40 }}>
          {tokens.length === 0 ? (
            <span style={{ color: COLORS.textFaint, fontSize: 12, fontFamily: FONTS.mono }}>
              Type IPA to see tokenization
            </span>
          ) : (
            tokens.map((t, i) => {
              const role = classify(t);
              const color = ROLE_COLORS[role] ?? COLORS.textDim;
              return (
                <span key={`${t}-${i}`} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <span
                    style={{
                      display: "inline-block",
                      padding: "6px 12px",
                      background: `${color}15`,
                      border: `1px solid ${color}30`,
                      borderRadius: 8,
                      fontSize: 18,
                      fontFamily: FONTS.mono,
                      fontWeight: 500,
                      color,
                    }}
                    title={`${role} — /${t}/`}
                  >
                    {t}
                  </span>
                  {i < tokens.length - 1 && (
                    <span style={{ color: COLORS.textFaint, fontSize: 12, fontFamily: FONTS.mono }}>→</span>
                  )}
                </span>
              );
            })
          )}
        </div>

        {tokens.length > 0 && (
          <div style={{ marginTop: 12, fontSize: 10, color: COLORS.textFaint, fontFamily: FONTS.mono }}>
            {tokens.length} tokens ·{" "}
            {tokens.filter((t) => classify(t) === "onset").length} onset ·{" "}
            {tokens.filter((t) => classify(t) === "nucleus").length} nucleus ·{" "}
            {tokens.filter((t) => classify(t) === "coda").length} coda
          </div>
        )}
      </div>
    </section>
  );
}

// ── Inventory Grid ──────────────────────────────────────────────────────

function InventoryGrid({ title, phonemes, color }: { title: string; phonemes: string[]; color: string }) {
  return (
    <div
      style={{
        background: COLORS.bgCard,
        border: `1px solid ${COLORS.border}`,
        borderRadius: 12,
        padding: 16,
      }}
    >
      <h4 style={{ fontSize: 10, color: COLORS.textDim, textTransform: "uppercase", letterSpacing: 1, marginBottom: 10, fontWeight: 600 }}>
        {title} ({phonemes.length})
      </h4>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
        {phonemes.map((p) => (
          <span
            key={p}
            style={{
              display: "inline-block",
              padding: "2px 6px",
              background: `${color}10`,
              borderRadius: 4,
              fontSize: 12,
              fontFamily: FONTS.mono,
              color,
              border: `1px solid ${color}20`,
            }}
          >
            {p}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Shared ──────────────────────────────────────────────────────────────

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

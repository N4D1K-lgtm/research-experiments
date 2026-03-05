import { useState } from "react";
import { useTrieDataStore } from "../../../store/trieDataStore";
import { COLORS, FONTS } from "../../../styles/theme";
import {
  Chapter, ChapterTitle, Prose, M, MathBlock, Insight, Definition,
  Widget, Tag, Divider,
} from "../shared";

// Tokenizer logic (same as pipeline)
const COMBINING_START = 0x0300, COMBINING_END = 0x036F;
const IPA_DIACRITICS = new Set(["\u0329","\u032F","\u0325","\u031F","\u0320","\u0303","\u02D0","\u02B0","\u02B7","\u02B2","\u0361","\u035C"]);
const SUPRASEGMENTALS = new Set(["\u02C8","\u02CC",".","\u203F","|","\u2016"]);
const VOWELS = new Set(["a","e","i","o","u","\u0259","\u025B","\u0254","\u026A","\u028A","\u00E6","\u0251","\u0252","\u028C","\u025D","y","\u00F8","\u0153"]);

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

function classify(base: string, onsets: Set<string>, codas: Set<string>): "onset" | "nucleus" | "coda" {
  if (VOWELS.has(base[0])) return "nucleus";
  if (onsets.has(base[0]) && !codas.has(base[0])) return "onset";
  if (codas.has(base[0]) && !onsets.has(base[0])) return "coda";
  return "onset";
}

const ROLE_COLORS: Record<string, string> = { onset: COLORS.onset, nucleus: COLORS.nucleus, coda: COLORS.coda };
const ROLE_NAMES: Record<string, string> = { onset: "onset", nucleus: "nucleus", coda: "coda" };

const EXAMPLES = [
  { word: "cat", ipa: "\u02C8k\u00E6t", lang: "English" },
  { word: "strength", ipa: "\u02C8st\u0279\u025B\u014B\u03B8", lang: "English" },
  { word: "beautiful", ipa: "\u02C8bju\u02D0t\u026Af\u028Al", lang: "English" },
  { word: "thought", ipa: "\u02C8\u03B8\u0254\u02D0t", lang: "English" },
  { word: "chat", ipa: "\u0283a", lang: "French" },
  { word: "Straße", ipa: "\u0283t\u0279a\u02D0s\u0259", lang: "German" },
];

export function ChapterTokenization() {
  const metadata = useTrieDataStore((s) => s.metadata);
  const onsets = new Set(metadata?.onsetInventory ?? []);
  const codas = new Set(metadata?.codaInventory ?? []);

  const [input, setInput] = useState("\u02C8st\u0279\u025B\u014B\u03B8");
  const tokens = tokenize(input);
  const [selectedExample, setSelectedExample] = useState(1);

  return (
    <Chapter id="tokenization">
      <ChapterTitle
        number={2}
        title="From Speech to Tokens"
        subtitle="A word's pronunciation is a sequence of phonemes. To build our trie, we need to break IPA strings into individual tokens — handling diacritics, combining marks, and suprasegmental features."
      />

      <Prose>
        Consider the English word "strength": /ˈstrɛŋθ/. This is <em>six</em> distinct phonemes, not
        eight characters. The tokenizer must recognize that <M>ˈ</M> is a stress marker (strip it),
        while <M>ŋ</M> is a single nasal consonant written with one Unicode character.
      </Prose>

      <MathBlock label="Tokenization">
        /ˈstrɛŋθ/ → [s] [t] [ɹ] [ɛ] [ŋ] [θ]
      </MathBlock>

      <Prose>
        The rules are subtle. IPA uses <em>combining diacritics</em> — characters that modify the
        preceding phoneme. A length mark <M>ː</M> after a vowel doesn't create a new token; it
        modifies the vowel into a long variant. Tie bars <M>◌͡◌</M> join two symbols into a single
        affricate. The tokenizer handles all of these by absorbing combining characters into their
        base phoneme.
      </Prose>

      <Definition term="Tokenization Algorithm">
        1. Strip suprasegmentals (stress marks, syllable boundaries).<br />
        2. For each base character, absorb any following combining diacritics (U+0300–U+036F) and
        IPA modifier characters (ː, ʰ, ʷ, etc.) into a single token.<br />
        3. Classify each token by its base character: vowels → <Tag color={COLORS.nucleus}>nucleus</Tag>,
        consonants → <Tag color={COLORS.onset}>onset</Tag> or <Tag color={COLORS.coda}>coda</Tag> based
        on the language's phonotactic inventory.
      </Definition>

      <Divider />

      <Widget label="Tokenizer Playground" instructions="Click an example or type your own IPA string">
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}>
          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              onClick={() => { setInput(ex.ipa); setSelectedExample(i); }}
              style={{
                padding: "5px 12px",
                background: selectedExample === i ? "rgba(255,255,255,0.1)" : "rgba(255,255,255,0.03)",
                border: `1px solid ${selectedExample === i ? COLORS.accent + "40" : COLORS.border}`,
                borderRadius: 6,
                color: selectedExample === i ? COLORS.accent : COLORS.textDim,
                fontSize: 11,
                fontFamily: FONTS.mono,
                cursor: "pointer",
              }}
            >
              {ex.word}
              <span style={{ fontSize: 8, color: COLORS.textFaint, marginLeft: 4 }}>{ex.lang}</span>
            </button>
          ))}
        </div>

        <input
          value={input}
          onChange={(e) => { setInput(e.target.value); setSelectedExample(-1); }}
          placeholder="Type or paste IPA..."
          style={{
            width: "100%",
            padding: "12px 16px",
            background: "rgba(255,255,255,0.04)",
            border: `1px solid ${COLORS.border}`,
            borderRadius: 8,
            color: COLORS.textBright,
            fontSize: 20,
            fontFamily: FONTS.mono,
            outline: "none",
            marginBottom: 20,
          }}
        />

        {/* Token display */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", minHeight: 48 }}>
          {tokens.length === 0 ? (
            <span style={{ color: COLORS.textFaint, fontSize: 13, fontFamily: FONTS.mono }}>
              Type IPA to see tokenization
            </span>
          ) : (
            tokens.map((t, i) => {
              const role = classify(t, onsets, codas);
              const color = ROLE_COLORS[role];
              return (
                <span key={`${t}-${i}`} style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                  <span
                    style={{
                      display: "inline-flex",
                      flexDirection: "column",
                      alignItems: "center",
                      padding: "8px 14px 6px",
                      background: `${color}12`,
                      border: `1px solid ${color}30`,
                      borderRadius: 10,
                      minWidth: 36,
                    }}
                  >
                    <span style={{ fontSize: 22, fontFamily: FONTS.mono, fontWeight: 500, color, lineHeight: 1 }}>
                      {t}
                    </span>
                    <span style={{ fontSize: 8, color: `${color}99`, textTransform: "uppercase", letterSpacing: 0.5, marginTop: 4 }}>
                      {ROLE_NAMES[role]}
                    </span>
                  </span>
                  {i < tokens.length - 1 && (
                    <span style={{ color: COLORS.textFaint, fontSize: 14, fontFamily: FONTS.mono }}>→</span>
                  )}
                </span>
              );
            })
          )}
        </div>

        {tokens.length > 0 && (
          <div style={{ marginTop: 16, padding: "10px 14px", background: "rgba(255,255,255,0.02)", borderRadius: 8, display: "flex", gap: 20, fontSize: 11, fontFamily: FONTS.mono, color: COLORS.textDim }}>
            <span>{tokens.length} tokens</span>
            <span style={{ color: COLORS.onset }}>{tokens.filter(t => classify(t, onsets, codas) === "onset").length} onset</span>
            <span style={{ color: COLORS.nucleus }}>{tokens.filter(t => classify(t, onsets, codas) === "nucleus").length} nucleus</span>
            <span style={{ color: COLORS.coda }}>{tokens.filter(t => classify(t, onsets, codas) === "coda").length} coda</span>
          </div>
        )}
      </Widget>

      <Prose>
        Each token becomes a node in our trie. The <em>sequence</em> of tokens defines a path
        from root to leaf. This is the key insight: every word is a walk through phoneme space,
        and words that start the same way share the beginning of their walk.
      </Prose>

      <Insight>
        The tokenization is language-aware. A phoneme classified as <Tag color={COLORS.onset}>onset</Tag> in
        English might be <Tag color={COLORS.coda}>coda</Tag>-only in Japanese. Our pipeline validates
        each token against the PHOIBLE database of phonological inventories for{" "}
        {metadata?.languages.length ?? 12} languages.
      </Insight>
    </Chapter>
  );
}

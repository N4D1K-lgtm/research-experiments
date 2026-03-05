#!/usr/bin/env python3
"""
Data-driven allophone detection via complementary distribution.

For each language, find pairs of sounds that NEVER contrast — i.e., no
minimal pair exists where swapping one for the other produces a different
valid word. These are allophone candidates.

Method:
  For every word, at every position, compute a "template" (the word with
  that position blanked out). If two sounds never co-occur in the same
  template, they're in complementary distribution → allophone candidates.

  Then group into equivalence classes and pick the most frequent member
  as the underlying phoneme.
"""

import sys
from collections import defaultdict
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from build_trie import tokenize_ipa, LANGUAGES, STRIP_TOKENS


def analyze_language(filepath: Path, lang: str) -> None:
    """Analyze one language for complementary distribution."""
    print(f"\n{'='*60}")
    print(f"  {lang} — {filepath.name}")
    print(f"{'='*60}")

    # 1. Tokenize all words (strip suprasegmentals)
    word_sequences: list[tuple[str, ...]] = []
    token_freq: dict[str, int] = defaultdict(int)

    for line in filepath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        for pron in parts[1].split(", "):
            raw = tokenize_ipa(pron)
            cleaned = tuple(t for t in raw if t not in STRIP_TOKENS)
            if not cleaned:
                continue
            word_sequences.append(cleaned)
            for t in cleaned:
                token_freq[t] += 1

    print(f"  {len(word_sequences)} word pronunciations")
    print(f"  {len(token_freq)} unique tokens")

    # 2. Build templates: for each word, blank out each position
    #    template -> set of tokens that appear at the blanked position
    templates: dict[tuple, set[str]] = defaultdict(set)

    for seq in word_sequences:
        for i in range(len(seq)):
            template = seq[:i] + ("_",) + seq[i+1:]
            templates[template].add(seq[i])

    # 3. Find all CONTRASTING pairs (share at least one template)
    contrasting: set[tuple[str, str]] = set()
    for template, tokens in templates.items():
        if len(tokens) < 2:
            continue
        token_list = sorted(tokens)
        for i in range(len(token_list)):
            for j in range(i + 1, len(token_list)):
                pair = (token_list[i], token_list[j])
                contrasting.add(pair)

    # 4. Find pairs in complementary distribution
    #    (both tokens exist but never contrast)
    all_tokens = sorted(token_freq.keys())
    MIN_FREQ = 50  # ignore very rare tokens

    frequent_tokens = [t for t in all_tokens if token_freq[t] >= MIN_FREQ]
    print(f"  {len(frequent_tokens)} tokens with freq >= {MIN_FREQ}")
    print(f"  {len(contrasting)} contrasting pairs found")

    complementary: list[tuple[str, str, int, int]] = []
    for i in range(len(frequent_tokens)):
        for j in range(i + 1, len(frequent_tokens)):
            a, b = frequent_tokens[i], frequent_tokens[j]
            if (a, b) not in contrasting:
                complementary.append((a, b, token_freq[a], token_freq[b]))

    print(f"  {len(complementary)} complementary pairs (no minimal pair found)")

    if not complementary:
        print("  No complementary distribution pairs found.")
        return

    # 5. Filter: prefer pairs that are phonetically plausible
    #    Heuristic: share a base character, or one is a modified form of the other
    def phonetic_similarity(a: str, b: str) -> float:
        """Rough phonetic similarity score (0-1)."""
        # Same base character (first char match)
        if a[0] == b[0]:
            return 0.9
        # One contains the other
        if a in b or b in a:
            return 0.7
        # Check if they share manner/place features via IPA categories
        # (very rough: same broad class)
        vowels = set("aeiouæɑɒəɛɜɪɔʊʌɤøœɵɶɐ")
        plosives = set("pbttdɖɟcɡkqʔ")
        fricatives = set("fvθðszʃʒɕʑçxɣχʁħʕhɦβɸ")
        nasals = set("mnɲŋɳɴ")
        liquids = set("lɫɭɹɻrɾʀʁ")

        for group in [vowels, plosives, fricatives, nasals, liquids]:
            if a[0] in group and b[0] in group:
                return 0.5
        return 0.0

    # Sort by phonetic similarity, then by combined frequency
    scored = []
    for a, b, fa, fb in complementary:
        sim = phonetic_similarity(a, b)
        scored.append((sim, fa + fb, a, b, fa, fb))

    scored.sort(key=lambda x: (-x[0], -x[1]))

    # Report
    print(f"\n  Complementary pairs (sorted by phonetic similarity):")
    print(f"  {'Token A':>10s}  {'freq':>7s}  {'Token B':>10s}  {'freq':>7s}  {'sim':>4s}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*10}  {'-'*7}  {'-'*4}")

    shown = 0
    for sim, combined_freq, a, b, fa, fb in scored:
        if sim < 0.1 and shown > 20:
            break
        # Show high-similarity pairs, or first 30
        if sim >= 0.3 or shown < 30:
            marker = " ***" if sim >= 0.7 else " **" if sim >= 0.4 else ""
            print(f"  {a:>10s}  {fa:>7d}  {b:>10s}  {fb:>7d}  {sim:>4.1f}{marker}")
            shown += 1

    # 6. Build suggested allophone map (high-confidence pairs only)
    print(f"\n  Suggested allophone map (similarity >= 0.5, both freq >= {MIN_FREQ}):")
    print(f"  {{")
    suggestions = []
    for sim, combined_freq, a, b, fa, fb in scored:
        if sim < 0.5:
            continue
        # The more frequent one is the phoneme, less frequent is the allophone
        if fa >= fb:
            phoneme, allophone = a, b
        else:
            phoneme, allophone = b, a
        suggestions.append((allophone, phoneme))
        print(f'      "{allophone}": "{phoneme}",  # freq {token_freq[allophone]} → {token_freq[phoneme]}, sim={sim:.1f}')
    print(f"  }}")

    return suggestions


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"

    all_suggestions: dict[str, list] = {}
    for lang in LANGUAGES:
        filepath = data_dir / f"{lang}.txt"
        if not filepath.exists():
            print(f"Skipping {lang}: file not found")
            continue
        result = analyze_language(filepath, lang)
        if result:
            all_suggestions[lang] = result

    # Print final summary map
    print(f"\n{'='*60}")
    print("  FINAL SUGGESTED ALLOPHONE_MAP")
    print(f"{'='*60}")
    print("ALLOPHONE_MAP = {")
    for lang in LANGUAGES:
        suggestions = all_suggestions.get(lang, [])
        print(f'    "{lang}": {{')
        for allophone, phoneme in suggestions:
            print(f'        "{allophone}": "{phoneme}",')
        print(f"    }},")
    print("}")


if __name__ == "__main__":
    main()

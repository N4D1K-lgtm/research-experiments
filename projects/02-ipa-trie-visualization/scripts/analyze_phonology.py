#!/usr/bin/env python3
"""
Phonological analysis of English IPA pronunciation data.
Uses the same tokenization/normalization as build_trie.py.
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict

# Add parent scripts dir so we can import from build_trie
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_trie import tokenize_ipa, normalize_phonemes

LANG = "en_US"
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / f"{LANG}.txt"


def load_phoneme_sequences() -> list[list[str]]:
    """Parse the data file and return list of normalized phoneme sequences."""
    sequences = []
    for line in DATA_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        ipa_field = parts[1]
        for pron in ipa_field.split(", "):
            raw_tokens = tokenize_ipa(pron)
            if not raw_tokens:
                continue
            phonemes, _ = normalize_phonemes(raw_tokens, LANG)
            if phonemes:
                sequences.append(phonemes)
    return sequences


def main():
    print(f"Loading {DATA_FILE}...")
    seqs = load_phoneme_sequences()
    print(f"Total pronunciation entries: {len(seqs)}\n")

    # ── 1. Unique phoneme inventory ──────────────────────────────────────
    all_phonemes = set()
    phoneme_freq = Counter()
    for seq in seqs:
        for ph in seq:
            all_phonemes.add(ph)
            phoneme_freq[ph] += 1

    print("=" * 60)
    print(f"1. UNIQUE PHONEME INVENTORY: {len(all_phonemes)} phonemes")
    print("=" * 60)
    print("\nAll phonemes (sorted by frequency):")
    for ph, count in phoneme_freq.most_common():
        print(f"  {ph:>4s}  {count:>7,d}  ({count/sum(phoneme_freq.values())*100:5.2f}%)")

    # ── 2. Top 30 bigrams ────────────────────────────────────────────────
    bigram_freq = Counter()
    for seq in seqs:
        for i in range(len(seq) - 1):
            bigram_freq[(seq[i], seq[i + 1])] += 1

    print("\n" + "=" * 60)
    print("2. TOP 30 BIGRAMS (phoneme transitions)")
    print("=" * 60)
    for (a, b), count in bigram_freq.most_common(30):
        print(f"  {a} -> {b:>4s}  {count:>6,d}")

    # ── 3. Word length distribution ──────────────────────────────────────
    length_dist = Counter(len(seq) for seq in seqs)

    print("\n" + "=" * 60)
    print("3. WORD LENGTH DISTRIBUTION (in phonemes)")
    print("=" * 60)
    total = len(seqs)
    cumulative = 0
    for length in sorted(length_dist.keys()):
        count = length_dist[length]
        cumulative += count
        bar = "#" * int(count / max(length_dist.values()) * 40)
        print(f"  {length:>2d} phonemes: {count:>6,d} words ({count/total*100:5.2f}%)  {bar}")
    print(f"\n  Mean length: {sum(len(s) for s in seqs) / len(seqs):.2f} phonemes")
    print(f"  Median length: {sorted(len(s) for s in seqs)[len(seqs)//2]} phonemes")
    print(f"  Max length: {max(len(s) for s in seqs)} phonemes")

    # ── 4. Onset and coda inventories ────────────────────────────────────
    onset_freq = Counter(seq[0] for seq in seqs)
    coda_freq = Counter(seq[-1] for seq in seqs)

    print("\n" + "=" * 60)
    print("4. ONSET INVENTORY (word-initial phonemes)")
    print("=" * 60)
    print(f"   {len(onset_freq)} unique onset phonemes\n")
    for ph, count in onset_freq.most_common():
        print(f"  {ph:>4s}  {count:>6,d}  ({count/total*100:5.2f}%)")

    print("\n" + "-" * 60)
    print("   CODA INVENTORY (word-final phonemes)")
    print("-" * 60)
    print(f"   {len(coda_freq)} unique coda phonemes\n")
    for ph, count in coda_freq.most_common():
        print(f"  {ph:>4s}  {count:>6,d}  ({count/total*100:5.2f}%)")

    # Onset-only and coda-only
    onset_only = set(onset_freq.keys()) - set(coda_freq.keys())
    coda_only = set(coda_freq.keys()) - set(onset_freq.keys())
    print(f"\n  Onset-only phonemes (never end a word): {sorted(onset_only)}")
    print(f"  Coda-only phonemes (never start a word): {sorted(coda_only)}")

    # ── 5. Top 20 trigrams ───────────────────────────────────────────────
    trigram_freq = Counter()
    for seq in seqs:
        for i in range(len(seq) - 2):
            trigram_freq[(seq[i], seq[i + 1], seq[i + 2])] += 1

    print("\n" + "=" * 60)
    print("5. TOP 20 TRIGRAMS (3-phoneme motifs)")
    print("=" * 60)
    for (a, b, c), count in trigram_freq.most_common(20):
        print(f"  {a} {b} {c}  {count:>6,d}")

    # ── 6. Branching factor (prefix sharing) ─────────────────────────────
    print("\n" + "=" * 60)
    print("6. BRANCHING FACTOR / PREFIX SHARING")
    print("=" * 60)

    for depth in [1, 2, 3]:
        prefix_counts = Counter()
        for seq in seqs:
            if len(seq) >= depth:
                prefix = tuple(seq[:depth])
                prefix_counts[prefix] += 1

        n_prefixes = len(prefix_counts)
        avg_words = len(seqs) / n_prefixes if n_prefixes else 0
        top10 = prefix_counts.most_common(10)

        print(f"\n  Depth {depth}: {n_prefixes} unique prefixes, "
              f"avg {avg_words:.1f} words/prefix")
        print(f"  Top 10 prefixes at depth {depth}:")
        for prefix, count in top10:
            label = " ".join(prefix)
            print(f"    {label:>12s}  {count:>6,d} words")

    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total pronunciations:  {len(seqs):,d}")
    print(f"  Phoneme inventory:     {len(all_phonemes)}")
    print(f"  Unique bigrams:        {len(bigram_freq)}")
    print(f"  Unique trigrams:       {len(trigram_freq)}")
    print(f"  Onset phonemes:        {len(onset_freq)}")
    print(f"  Coda phonemes:         {len(coda_freq)}")


if __name__ == "__main__":
    main()

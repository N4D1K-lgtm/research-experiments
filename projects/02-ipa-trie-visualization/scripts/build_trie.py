#!/usr/bin/env python3
"""
Build a phonological trie from IPA dictionary data.

Pipeline:
  parse TSVs -> tokenize IPA -> normalize (strip suprasegmentals,
  collapse allophones to phonemes) -> build trie with word terminals
  -> phonological analysis (transitions, positions, motifs, allophone contexts)
  -> layout (cone tree) -> color -> JSON
"""

import argparse
import json
import math
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ALL_LANGUAGES = ["en_US", "fr_FR", "es_ES", "de", "nl"]

GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))

# ── Phoneme classification ────────────────────────────────────────────────

# IPA vowels (monophthongs + common diphthong components)
IPA_VOWELS = set(
    "iɪeɛæaɑɒɔʌəɜɝɚuʊoøyɨʉɯɐœɶ"
)

def is_vowel(phoneme: str) -> bool:
    """Check if phoneme is a vowel (base character is in vowel set)."""
    if not phoneme:
        return False
    return phoneme[0] in IPA_VOWELS


# ── Vector helpers ─────────────────────────────────────────────────────────

Vec3 = tuple[float, float, float]


def vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vscale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def vcross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vnorm(a: Vec3) -> Vec3:
    le = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    if le < 1e-12:
        return (0.0, 0.0, 1.0)
    return (a[0] / le, a[1] / le, a[2] / le)


# ── IPA Tokenizer ──────────────────────────────────────────────────────────

COMBINING_CATEGORIES = {"Mn", "Mc", "Me"}

IPA_MODIFIERS = set(
    "ːˑˈˌ"
    "ʰʷʲˠˤ"
    "ⁿˡ"
    "̃̈"
    "˞"
    "ʼ"
)

IPA_SUPRASEGMENTALS = set("ˈˌ.‿|‖")
TIE_BARS = set("\u0361\u035C")


def is_combining(ch: str) -> bool:
    cat = unicodedata.category(ch)
    if cat in COMBINING_CATEGORIES:
        return True
    return ch in IPA_MODIFIERS


def tokenize_ipa(transcription: str) -> list[str]:
    s = transcription.strip()
    for wrapper in ("/", "[", "]"):
        s = s.strip(wrapper)
    s = s.strip()

    tokens: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in (" ", "\t"):
            i += 1
            continue
        if ch in IPA_SUPRASEGMENTALS:
            tokens.append(ch)
            i += 1
            continue

        phoneme = ch
        i += 1
        while i < len(s):
            nch = s[i]
            if nch in TIE_BARS:
                phoneme += nch
                i += 1
                if i < len(s):
                    phoneme += s[i]
                    i += 1
                continue
            if is_combining(nch):
                phoneme += nch
                i += 1
                continue
            if nch in ("ː", "ˑ"):
                phoneme += nch
                i += 1
                continue
            break
        tokens.append(phoneme)

    return tokens


# ── Phonological Normalization Layer ──────────────────────────────────────
#
# These rules map surface phonetic forms to underlying phonemic
# representations — the "phonological laws" that collapse allophonic
# variation into abstract phoneme identity.
#
# Data-driven validation: confirmed via complementary distribution
# analysis (find_allophones.py) against ipa-dict corpus.
#
# RULE 1 — Prosodic features: always strip (not part of phoneme identity)
#   Stress (ˈ ˌ), syllable boundaries (. ‿ | ‖)
#
# RULE 2 — Allophonic diacritics: strip (predictable from context)
#   Syllabic (̩ U+0329), non-syllabic (̯ U+032F),
#   voiceless sonorant (̥ U+0325), advanced/retracted (̟ U+031F, ̠ U+0320)
#   These were confirmed by data: n̩/n, ɪ̯/ɪ, ʊ̯/ʊ never contrast.
#
# RULE 3 — Phonemic features: KEEP (distinctive in ≥1 language)
#   Nasalization ̃ (French: bõ≠bo), length ː (German: iː≠ɪ),
#   aspiration ʰ, labialization ʷ, palatalization ʲ
#
# RULE 4 — Language-specific allophone rules (established phonology)
#   English: ɫ→l (dark L allophony), ɾ→t (flapping)
#   Spanish: β→b, ð→d, ɣ→ɡ (intervocalic lenition)
#   Data confirmed: these pairs never appear in same minimal-pair template.

# Suprasegmentals to strip entirely (prosodic, not segmental)
STRIP_TOKENS = {"ˈ", "ˌ", ".", "‿", "|", "‖"}

# Characters to strip from WITHIN tokens (stress marks absorbed by tokenizer)
STRIP_CHARS = {"ˈ", "ˌ"}

# Allophonic combining diacritics to strip
STRIP_COMBINING = {
    "\u0329",  # combining vertical line below (syllabic)
    "\u032F",  # combining inverted breve below (non-syllabic)
    "\u0325",  # combining ring below (voiceless sonorant)
    "\u031F",  # combining plus sign below (advanced tongue root)
    "\u0320",  # combining minus sign below (retracted tongue root)
}

# Per-language allophone → phoneme rules
ALLOPHONE_MAP: dict[str, dict[str, str]] = {
    "en_US": {
        "ɫ": "l",       # dark L → /l/ (complementary distribution confirmed)
        "ɾ": "t",       # alveolar flap → /t/ (intervocalic, confirmed)
    },
    "es_ES": {
        "β": "b",       # lenited → /b/ (intervocalic, confirmed)
        "ð": "d",       # lenited → /d/ (intervocalic, confirmed)
        "ɣ": "ɡ",       # lenited → /g/ (intervocalic, confirmed)
    },
    "de": {},
    "fr_FR": {},
    "nl": {},
}


def normalize_token(token: str, lang: str) -> str | None:
    """
    Apply phonological laws to map surface form → underlying phoneme.
    Returns None for tokens that should be stripped entirely.
    """
    # Rule 1: Strip standalone prosodic tokens
    if token in STRIP_TOKENS:
        return None

    # Rule 4: Language-specific allophone rules (exact match first)
    lang_map = ALLOPHONE_MAP.get(lang, {})
    if token in lang_map:
        return lang_map[token]

    # Rule 1b: Strip stress marks embedded within tokens
    # (tokenizer absorbs ˈ/ˌ as modifiers when they follow a base char)
    cleaned = "".join(ch for ch in token if ch not in STRIP_CHARS)

    # Rule 2: Strip allophonic combining diacritics
    cleaned = "".join(ch for ch in cleaned if ch not in STRIP_COMBINING)

    if not cleaned:
        return None

    # Re-check allophone map after stripping
    if cleaned in lang_map:
        return lang_map[cleaned]

    return cleaned


def normalize_phonemes(tokens: list[str], lang: str) -> tuple[list[str], list[str]]:
    """
    Normalize a token sequence through phonological laws.
    Returns (phonemes, original_surfaces) — normalized forms and
    pre-normalization tokens. Prosodic tokens removed from both.
    """
    phonemes = []
    surfaces = []
    for token in tokens:
        if token in STRIP_TOKENS:
            continue
        normalized = normalize_token(token, lang)
        if normalized is None:
            continue
        phonemes.append(normalized)
        surfaces.append(token)
    return phonemes, surfaces


# ── Trie Builder ───────────────────────────────────────────────────────────

MAX_SAMPLE_WORDS = 5  # max words to store per language per terminal node


class TrieNode:
    __slots__ = (
        "phoneme", "children", "counts", "node_id", "depth", "parent_id",
        "_weight", "allophones", "terminal_counts", "sample_words",
    )

    def __init__(self, phoneme: str, depth: int, parent_id: int | None = None):
        self.phoneme = phoneme
        self.depth = depth
        self.parent_id = parent_id
        self.children: dict[str, "TrieNode"] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.node_id: int = -1
        self._weight: int = 0
        # Set of surface allophone forms that mapped to this phoneme
        self.allophones: set[str] = set()
        # Word terminal tracking: how many words end here per language
        self.terminal_counts: dict[str, int] = defaultdict(int)
        # Sample words per language (capped)
        self.sample_words: dict[str, list[str]] = defaultdict(list)


def build_trie(data_dir: Path, languages: list[str]) -> TrieNode:
    root = TrieNode("ROOT", 0)
    for lang in languages:
        filepath = data_dir / f"{lang}.txt"
        if not filepath.exists():
            print(f"  Warning: {filepath} not found, skipping")
            continue

        word_count = 0
        allophone_collapses = 0
        for line in filepath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            word = parts[0]
            ipa_field = parts[1]
            for pron in ipa_field.split(", "):
                raw_tokens = tokenize_ipa(pron)
                if not raw_tokens:
                    continue

                phonemes, allophones = normalize_phonemes(raw_tokens, lang)
                if not phonemes:
                    continue

                # Count allophone collapses
                for ph, al in zip(phonemes, allophones):
                    if ph != al:
                        allophone_collapses += 1

                # Insert into trie
                node = root
                node.counts[lang] += 1
                for i, phoneme in enumerate(phonemes):
                    if phoneme not in node.children:
                        node.children[phoneme] = TrieNode(
                            phoneme, node.depth + 1, parent_id=node.node_id
                        )
                    node = node.children[phoneme]
                    node.counts[lang] += 1
                    # Track which allophone mapped to this phoneme
                    node.allophones.add(allophones[i])

                # Mark word terminal
                node.terminal_counts[lang] += 1
                if len(node.sample_words[lang]) < MAX_SAMPLE_WORDS:
                    node.sample_words[lang].append(word)

                word_count += 1
        print(f"  {lang}: {word_count} pronunciations, {allophone_collapses} allophone collapses")
    return root


def assign_ids(root: TrieNode) -> list[TrieNode]:
    nodes: list[TrieNode] = []
    queue = [root]
    next_id = 0
    while queue:
        node = queue.pop(0)
        node.node_id = next_id
        next_id += 1
        nodes.append(node)
        for child in node.children.values():
            child.parent_id = node.node_id
            queue.append(child)
    return nodes


def prune_trie(root: TrieNode, min_count: int) -> None:
    to_remove = []
    for phoneme, child in root.children.items():
        total = sum(child.counts.values())
        if total < min_count:
            to_remove.append(phoneme)
        else:
            prune_trie(child, min_count)
    for phoneme in to_remove:
        del root.children[phoneme]


# ── Phonological Analysis ──────────────────────────────────────────────────

def compute_transition_probs(node: TrieNode) -> dict[str, float]:
    """Compute P(next_phoneme | this_path) for each child."""
    if not node.children:
        return {}
    total = sum(sum(c.counts.values()) for c in node.children.values())
    if total == 0:
        return {}
    return {
        phoneme: round(sum(child.counts.values()) / total, 4)
        for phoneme, child in sorted(node.children.items(), key=lambda x: -sum(x[1].counts.values()))
    }


def classify_position(node: TrieNode, node_by_id: dict[int, "TrieNode"]) -> str:
    """
    Classify node's phonological position:
    onset = word-initial consonant (or consonant before first vowel)
    nucleus = vowel
    coda = consonant after last vowel (or word-final consonant)
    mixed = participates in multiple roles across different words
    """
    if node.depth == 0:
        return "mixed"

    phoneme = node.phoneme
    vowel = is_vowel(phoneme)

    if vowel:
        return "nucleus"

    # For consonants, check position context
    # If at depth 1, it's word-initial → onset
    if node.depth == 1:
        return "onset"

    # Walk up to find if we're before or after a vowel
    has_vowel_ancestor = False
    cur_id = node.parent_id
    while cur_id is not None and cur_id > 0:
        parent = node_by_id.get(cur_id)
        if parent is None:
            break
        if is_vowel(parent.phoneme):
            has_vowel_ancestor = True
            break
        cur_id = parent.parent_id

    # Check if any child paths lead through vowels
    has_vowel_child = any(is_vowel(ch) for ch in node.children.keys())

    if not has_vowel_ancestor and has_vowel_child:
        return "onset"
    elif has_vowel_ancestor and not has_vowel_child:
        return "coda"
    elif has_vowel_ancestor and has_vowel_child:
        return "mixed"
    else:
        # Consonant cluster — check depth heuristic
        if node.depth <= 3:
            return "onset"
        return "coda"


def detect_motifs(nodes: list[TrieNode], min_count: int = 100) -> list[dict[str, Any]]:
    """
    Find frequent sub-paths (n-grams, n=2..4) from the trie.
    Uses a node-id→parent lookup to build n-grams efficiently.
    """
    node_by_id = {n.node_id: n for n in nodes}
    ngram_counts: Counter[tuple[str, ...]] = Counter()

    for node in nodes:
        if node.depth < 2:
            continue
        count = sum(node.counts.values())
        if count == 0:
            continue

        # Build path suffix up to length 4 by walking parents
        path: list[str] = [node.phoneme]
        cur = node
        while len(path) < 4 and cur.parent_id is not None:
            parent = node_by_id.get(cur.parent_id)
            if parent is None or parent.depth == 0:
                break
            path.append(parent.phoneme)
            cur = parent
        path.reverse()

        # Count n-grams of length 2..4 ending at this node
        for n in range(2, min(5, len(path) + 1)):
            ngram = tuple(path[-n:])
            ngram_counts[ngram] += count

    # Filter and sort
    motifs = []
    for ngram, count in ngram_counts.most_common(300):
        if count < min_count:
            break
        motifs.append({
            "sequence": list(ngram),
            "count": count,
            "label": "".join(ngram),
        })

    return motifs[:200]


def build_transition_matrix(root: TrieNode) -> dict[str, dict[str, float]]:
    """Build global phoneme→phoneme transition probability matrix."""
    bigram_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def walk(node: TrieNode) -> None:
        count = sum(node.counts.values())
        for phoneme, child in node.children.items():
            child_count = sum(child.counts.values())
            if node.depth == 0:
                # From ROOT, just record first phoneme
                pass
            else:
                bigram_counts[node.phoneme][phoneme] += child_count
            walk(child)

    walk(root)

    matrix: dict[str, dict[str, float]] = {}
    for src, targets in sorted(bigram_counts.items()):
        total = sum(targets.values())
        if total == 0:
            continue
        matrix[src] = {
            tgt: round(cnt / total, 4)
            for tgt, cnt in sorted(targets.items(), key=lambda x: -x[1])
            if cnt / total >= 0.005  # Only include transitions with ≥0.5% probability
        }

    return matrix


def collect_allophone_contexts(data_dir: Path, lang: str) -> dict[str, dict[str, list[str]]]:
    """
    For each allophone→phoneme collapse, record the surrounding phoneme context.
    Returns: { allophone: { "before": [...], "after": [...] } }
    """
    filepath = data_dir / f"{lang}.txt"
    if not filepath.exists():
        return {}

    allophone_map = ALLOPHONE_MAP.get(lang, {})
    if not allophone_map:
        return {}

    contexts: dict[str, dict[str, set[str]]] = {}
    for allo in allophone_map:
        contexts[allo] = {"before": set(), "after": set()}

    for line in filepath.read_text(encoding="utf-8").splitlines():
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

            # Normalize but track original tokens
            normalized = []
            originals = []
            for token in raw_tokens:
                if token in STRIP_TOKENS:
                    continue
                norm = normalize_token(token, lang)
                if norm is None:
                    continue
                normalized.append(norm)
                originals.append(token)

            # Find allophones and record context
            for i, (orig, norm) in enumerate(zip(originals, normalized)):
                if orig in allophone_map:
                    before = normalized[i - 1] if i > 0 else "#"  # word boundary
                    after = normalized[i + 1] if i + 1 < len(normalized) else "#"
                    contexts[orig]["before"].add(before)
                    contexts[orig]["after"].add(after)

    # Convert sets to sorted lists, filter empty
    result: dict[str, dict[str, list[str]]] = {}
    for allo, ctx in contexts.items():
        before_list = sorted(ctx["before"])
        after_list = sorted(ctx["after"])
        if before_list or after_list:
            result[allo] = {
                "before": before_list,
                "after": after_list,
            }

    return result


def tag_node_motifs(nodes: list[TrieNode], motifs: list[dict[str, Any]]) -> dict[int, list[str]]:
    """
    For each node, find which motifs it participates in.
    Returns: { node_id: [motif_label, ...] }
    """
    node_by_id = {n.node_id: n for n in nodes}
    motif_seqs = [(tuple(m["sequence"]), m["label"]) for m in motifs[:50]]

    node_motifs: dict[int, list[str]] = {}

    for node in nodes:
        if node.depth < 2:
            continue

        # Build path suffix (up to max motif length)
        path: list[str] = [node.phoneme]
        path_ids: list[int] = [node.node_id]
        cur = node
        while len(path) < 4 and cur.parent_id is not None:
            parent = node_by_id.get(cur.parent_id)
            if parent is None or parent.depth == 0:
                break
            path.append(parent.phoneme)
            path_ids.append(parent.node_id)
            cur = parent
        path.reverse()
        path_ids.reverse()

        for seq, label in motif_seqs:
            seq_len = len(seq)
            if len(path) >= seq_len and tuple(path[-seq_len:]) == seq:
                for nid in path_ids[-seq_len:]:
                    if nid not in node_motifs:
                        node_motifs[nid] = []
                    if label not in node_motifs[nid]:
                        node_motifs[nid].append(label)

    return node_motifs


# ── Subtree weights ────────────────────────────────────────────────────────

def compute_weights(node: TrieNode) -> int:
    w = max(sum(node.counts.values()), 1)
    for child in node.children.values():
        w += compute_weights(child)
    node._weight = w
    return w


# ── Cone Tree Layout ──────────────────────────────────────────────────────

def layout_cone_tree(nodes: list[TrieNode], root: TrieNode) -> dict[int, Vec3]:
    """
    Recursive cone tree layout.
    Children positioned in a cone from parent along its outgoing direction.
    Angular wedges proportional to subtree weight. Edge lengths decay with depth.
    """
    positions: dict[int, Vec3] = {}
    directions: dict[int, Vec3] = {}

    positions[root.node_id] = (0.0, 0.0, 0.0)

    EDGE_BASE = 14.0
    DECAY = 0.84

    sys.setrecursionlimit(max(10000, len(nodes) + 100))

    def place_children(parent: TrieNode, parent_pos: Vec3, parent_dir: Vec3, depth: int) -> None:
        children = sorted(parent.children.values(), key=lambda c: -c._weight)
        if not children:
            return

        n = len(children)
        total_weight = sum(c._weight for c in children)
        if total_weight == 0:
            total_weight = 1

        edge_len = EDGE_BASE * (DECAY ** (depth - 1))

        if depth == 1:
            for i, child in enumerate(children):
                theta = math.acos(1 - 2 * (i + 0.5) / n)
                phi = GOLDEN_ANGLE * i
                direction: Vec3 = (
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                )
                child_pos = vadd(parent_pos, vscale(direction, edge_len))
                positions[child.node_id] = child_pos
                directions[child.node_id] = direction
                place_children(child, child_pos, direction, depth + 1)
        else:
            up = vnorm(parent_dir)
            if abs(up[1]) < 0.9:
                arb: Vec3 = (0.0, 1.0, 0.0)
            else:
                arb = (1.0, 0.0, 0.0)
            right = vnorm(vcross(up, arb))
            forward = vcross(right, up)

            cone_half = min(math.pi / 2.5, math.pi / 5 + 0.04 * n)
            cone_half *= max(0.5, 1.0 - depth * 0.03)

            if n == 1:
                child = children[0]
                offset_angle = (child.node_id * 2.399) % (2 * math.pi)
                deflection = 0.08
                child_dir = vnorm(vadd(
                    vscale(up, math.cos(deflection)),
                    vscale(vadd(
                        vscale(right, math.cos(offset_angle)),
                        vscale(forward, math.sin(offset_angle)),
                    ), math.sin(deflection)),
                ))
                child_pos = vadd(parent_pos, vscale(child_dir, edge_len))
                positions[child.node_id] = child_pos
                directions[child.node_id] = child_dir
                place_children(child, child_pos, child_dir, depth + 1)
            else:
                cumulative = 0.0
                for child in children:
                    frac = child._weight / total_weight
                    angle = cumulative + math.pi * frac
                    cumulative += 2 * math.pi * frac

                    perp = vadd(
                        vscale(right, math.cos(angle)),
                        vscale(forward, math.sin(angle)),
                    )
                    child_dir = vnorm(vadd(
                        vscale(up, math.cos(cone_half)),
                        vscale(perp, math.sin(cone_half)),
                    ))
                    child_pos = vadd(parent_pos, vscale(child_dir, edge_len))
                    positions[child.node_id] = child_pos
                    directions[child.node_id] = child_dir
                    place_children(child, child_pos, child_dir, depth + 1)

    place_children(root, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1)
    return positions


# ── Color (phonological role-based) ──────────────────────────────────────

# Phonological role hues
ROLE_HUES = {
    "onset": 200,    # cool blue/teal
    "nucleus": 45,   # warm amber/gold
    "coda": 280,     # muted purple
    "mixed": 150,    # neutral green
}


def role_hsl(position: str, total_count: int, is_terminal: bool) -> tuple[float, float, float]:
    """Compute HSL color based on phonological role."""
    hue = ROLE_HUES.get(position, 150)
    saturation = 0.65 if position != "mixed" else 0.35
    lightness = min(0.72, 0.32 + 0.18 * math.log10(max(total_count, 1)))
    if is_terminal:
        lightness = min(0.85, lightness + 0.15)
        saturation = min(1.0, saturation + 0.15)
    return (hue, saturation, lightness)


def hsl_to_hex(h: float, s: float, l: float) -> str:
    h = h / 360.0
    if s == 0:
        r = g = b = l
    else:
        def hue2rgb(p: float, q: float, t: float) -> float:
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1 / 6: return p + (q - p) * 6 * t
            if t < 1 / 2: return q
            if t < 2 / 3: return p + (q - p) * (2 / 3 - t) * 6
            return p
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue2rgb(p, q, h + 1 / 3)
        g = hue2rgb(p, q, h)
        b = hue2rgb(p, q, h - 1 / 3)

    return "#{:02x}{:02x}{:02x}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


# ── Serialization ──────────────────────────────────────────────────────────

def serialize(
    nodes: list[TrieNode],
    positions: dict[int, Vec3],
    languages: list[str],
    node_positions: dict[int, str],
    node_motifs: dict[int, list[str]],
    motifs: list[dict[str, Any]],
    transition_matrix: dict[str, dict[str, float]],
    allophone_contexts: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    max_depth = max(n.depth for n in nodes) if nodes else 0
    total_words = sum(nodes[0].counts.values()) if nodes else 0

    # Collect phoneme inventory from depth-1 nodes
    phoneme_inventory = sorted(set(n.phoneme for n in nodes if n.depth == 1))
    onset_inventory = sorted(set(
        n.phoneme for n in nodes
        if n.depth == 1 and not is_vowel(n.phoneme)
    ))
    coda_inventory: set[str] = set()
    for n in nodes:
        if sum(n.terminal_counts.values()) > 0 and not is_vowel(n.phoneme):
            coda_inventory.add(n.phoneme)
    coda_inventory_sorted = sorted(coda_inventory)

    json_nodes = []
    terminal_count = 0
    for node in nodes:
        pos = positions.get(node.node_id, (0, 0, 0))
        total_count = sum(node.counts.values())
        position = node_positions.get(node.node_id, "mixed")
        is_terminal = sum(node.terminal_counts.values()) > 0

        h, s, l = role_hsl(position, total_count, is_terminal)
        color = hsl_to_hex(h, s, l)

        if is_terminal:
            terminal_count += 1

        # Transition probabilities
        transition_probs = compute_transition_probs(node)

        entry: dict[str, Any] = {
            "id": node.node_id,
            "phoneme": node.phoneme,
            "depth": node.depth,
            "parentId": node.parent_id,
            "counts": dict(node.counts),
            "totalCount": total_count,
            "position": {"x": round(pos[0], 3), "y": round(pos[1], 3), "z": round(pos[2], 3)},
            "color": color,
            "hsl": {"h": round(h, 2), "s": round(s, 3), "l": round(l, 3)},
            "phonologicalPosition": position,
            "isTerminal": is_terminal,
            "childCount": len(node.children),
        }

        # Transition probabilities (only if has children)
        if transition_probs:
            entry["transitionProbs"] = transition_probs

        # Motifs
        motif_labels = node_motifs.get(node.node_id, [])
        if motif_labels:
            entry["motifs"] = motif_labels

        # Allophones: only include if there are surface forms different from the phoneme
        distinct_allophones = node.allophones - {node.phoneme}
        if distinct_allophones:
            entry["allophones"] = sorted(distinct_allophones)

        # Terminal data: word counts and sample words
        if is_terminal:
            entry["terminalCounts"] = dict(node.terminal_counts)
            # Flatten sample words, cap total
            words: dict[str, list[str]] = {}
            for lang, wlist in node.sample_words.items():
                if wlist:
                    words[lang] = wlist[:MAX_SAMPLE_WORDS]
            if words:
                entry["words"] = words

        json_nodes.append(entry)

    edges = []
    for node in nodes:
        if node.parent_id is not None:
            edges.append({"source": node.parent_id, "target": node.node_id})

    return {
        "metadata": {
            "languages": languages,
            "nodeCount": len(nodes),
            "edgeCount": len(edges),
            "maxDepth": max_depth,
            "totalWords": total_words,
            "terminalNodes": terminal_count,
            "phonemeInventory": phoneme_inventory,
            "onsetInventory": onset_inventory,
            "codaInventory": coda_inventory_sorted,
            "motifs": motifs[:100],
            "transitionMatrix": transition_matrix,
            "allophoneContexts": allophone_contexts,
        },
        "nodes": json_nodes,
        "edges": edges,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build IPA phoneme trie and export JSON")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent.parent / "output" / "trie.json")
    parser.add_argument("--min-count", type=int, default=0,
                        help="Prune nodes with total count below this threshold")
    parser.add_argument("--lang", type=str, default=None,
                        help="Process only this language (e.g. en_US). Default: all languages")
    args = parser.parse_args()

    languages = [args.lang] if args.lang else ALL_LANGUAGES

    print(f"Building IPA trie ({', '.join(languages)})...")
    root = build_trie(args.data_dir, languages)

    if args.min_count > 0:
        print(f"Pruning nodes with count < {args.min_count}...")
        prune_trie(root, args.min_count)

    print("Computing subtree weights...")
    compute_weights(root)

    print("Assigning IDs...")
    nodes = assign_ids(root)
    print(f"  {len(nodes)} nodes")

    print("Phonological analysis...")

    # Positional classification
    print("  Classifying phonological positions...")
    node_positions: dict[int, str] = {}
    node_id_map = {n.node_id: n for n in nodes}
    for node in nodes:
        node_positions[node.node_id] = classify_position(node, node_id_map)

    position_dist = Counter(node_positions.values())
    for pos, cnt in sorted(position_dist.items()):
        print(f"    {pos}: {cnt} nodes")

    # Motif detection
    print("  Detecting motifs...")
    motifs = detect_motifs(nodes, min_count=500)
    print(f"    Found {len(motifs)} motifs (count ≥ 500)")
    for m in motifs[:10]:
        print(f"      {''.join(m['sequence'])}: {m['count']:,}")

    # Tag nodes with motifs
    print("  Tagging nodes with motifs...")
    node_motifs = tag_node_motifs(nodes, motifs)
    tagged_count = len(node_motifs)
    print(f"    {tagged_count} nodes tagged with motifs")

    # Transition matrix
    print("  Building transition matrix...")
    transition_matrix = build_transition_matrix(root)
    print(f"    {len(transition_matrix)} source phonemes")

    # Allophone contexts
    print("  Collecting allophone contexts...")
    all_allophone_contexts: dict[str, dict[str, list[str]]] = {}
    for lang in languages:
        lang_contexts = collect_allophone_contexts(args.data_dir, lang)
        for allo, ctx in lang_contexts.items():
            all_allophone_contexts[allo] = ctx
    print(f"    {len(all_allophone_contexts)} allophones with context data")

    print("Computing cone tree layout...")
    positions = layout_cone_tree(nodes, root)

    print("Serializing...")
    data = serialize(
        nodes, positions, languages,
        node_positions, node_motifs, motifs,
        transition_matrix, all_allophone_contexts,
    )

    depths = defaultdict(int)
    for n in nodes:
        depths[n.depth] += 1
    terminals = sum(1 for n in nodes if sum(n.terminal_counts.values()) > 0)
    has_allophones = sum(1 for n in nodes if len(n.allophones - {n.phoneme}) > 0)

    print(f"\nSummary:")
    print(f"  Languages: {', '.join(languages)}")
    print(f"  Nodes: {data['metadata']['nodeCount']}")
    print(f"  Terminal nodes (word endpoints): {terminals}")
    print(f"  Nodes with allophone variants: {has_allophones}")
    print(f"  Edges: {data['metadata']['edgeCount']}")
    print(f"  Max depth: {data['metadata']['maxDepth']}")
    print(f"  Total words: {data['metadata']['totalWords']}")
    print(f"  Phoneme inventory: {len(data['metadata']['phonemeInventory'])} phonemes")
    print(f"  Motifs: {len(data['metadata']['motifs'])}")
    print(f"  Depth distribution:")
    for d in sorted(depths.keys()):
        print(f"    depth {d}: {depths[d]} nodes")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"\nOutput: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

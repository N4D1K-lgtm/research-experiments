#!/usr/bin/env python3
"""
Generate cross-linguistic stats JSON from IPA dictionary data.

For each language, builds a separate trie and computes per-depth statistics:
  depth, nodes, terminals, avgBranch, avgEntropy, maxEntropy

Uses the same tokenization and normalization pipeline as build_trie.py.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# Import shared tokenization/normalization from build_trie
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_trie import (
    ALL_LANGUAGES,
    tokenize_ipa,
    normalize_phonemes,
)


class TrieNode:
    __slots__ = ("children", "count", "is_terminal")

    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.count: int = 0
        self.is_terminal: bool = False


def build_lang_trie(data_dir: Path, lang: str) -> tuple[TrieNode, int]:
    """Build a trie for a single language. Returns (root, word_count)."""
    root = TrieNode()
    filepath = data_dir / f"{lang}.txt"
    if not filepath.exists():
        print(f"  Warning: {filepath} not found, skipping")
        return root, 0

    word_count = 0
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

            phonemes, _ = normalize_phonemes(raw_tokens, lang)
            if not phonemes:
                continue

            node = root
            node.count += 1
            for phoneme in phonemes:
                if phoneme not in node.children:
                    node.children[phoneme] = TrieNode()
                node = node.children[phoneme]
                node.count += 1

            node.is_terminal = True
            word_count += 1

    return root, word_count


def shannon_entropy(counts: list[int]) -> float:
    """Compute Shannon entropy in bits from a list of counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def compute_depth_stats(root: TrieNode) -> list[dict]:
    """Walk the trie and collect per-depth statistics."""
    # Gather all nodes by depth
    depth_nodes: dict[int, list[TrieNode]] = defaultdict(list)

    queue = [(root, 0)]
    while queue:
        node, depth = queue.pop(0)
        depth_nodes[depth].append(node)
        for child in node.children.values():
            queue.append((child, depth + 1))

    max_depth = max(depth_nodes.keys()) if depth_nodes else 0

    stats = []
    for d in range(max_depth + 1):
        nodes_at_depth = depth_nodes[d]
        num_nodes = len(nodes_at_depth)

        # Count terminals
        terminals = sum(1 for n in nodes_at_depth if n.is_terminal)

        # Branching factor and entropy: only for nodes that HAVE children
        branch_values = []
        entropy_values = []
        for n in nodes_at_depth:
            if n.children:
                branch_values.append(len(n.children))
                child_counts = [c.count for c in n.children.values()]
                entropy_values.append(shannon_entropy(child_counts))

        avg_branch = (
            round(sum(branch_values) / len(branch_values), 4)
            if branch_values
            else 0.0
        )
        avg_entropy = (
            round(sum(entropy_values) / len(entropy_values), 4)
            if entropy_values
            else 0.0
        )
        max_entropy = (
            round(max(entropy_values), 4)
            if entropy_values
            else 0.0
        )

        stats.append({
            "depth": d,
            "nodes": num_nodes,
            "terminals": terminals,
            "avgBranch": avg_branch,
            "avgEntropy": avg_entropy,
            "maxEntropy": max_entropy,
        })

    return stats


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    output_path = Path(__file__).resolve().parent.parent / "output" / "cross_linguistic_stats.json"

    result = {}

    for lang in ALL_LANGUAGES:
        print(f"Processing {lang}...")
        root, word_count = build_lang_trie(data_dir, lang)
        print(f"  {word_count} pronunciations, {len(root.children)} root children")

        stats = compute_depth_stats(root)
        result[lang] = stats

        # Print summary for verification
        if stats:
            d0 = stats[0]
            print(f"  depth 0: branch={d0['avgBranch']}, entropy={d0['avgEntropy']}")
            if len(stats) > 1:
                d1 = stats[1]
                print(f"  depth 1: nodes={d1['nodes']}, branch={d1['avgBranch']}, entropy={d1['avgEntropy']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nOutput: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()

//! Phonological motif (n-gram) detection.

use crate::trie::TrieNode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A detected phonological motif (frequent n-gram).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Motif {
    pub sequence: Vec<String>,
    pub count: u64,
    pub label: String,
}

/// Detect frequent n-grams (n=2..4) in the trie.
pub fn detect_motifs(root: &TrieNode, min_count: u64, max_results: usize) -> Vec<Motif> {
    let mut ngram_counts: HashMap<Vec<String>, u64> = HashMap::new();

    // Walk the trie collecting path n-grams
    collect_ngrams(root, &mut Vec::new(), &mut ngram_counts);

    // Filter and sort
    let mut motifs: Vec<Motif> = ngram_counts
        .into_iter()
        .filter(|(_, count)| *count >= min_count)
        .map(|(seq, count)| {
            let label = seq.join("");
            Motif {
                sequence: seq,
                count,
                label,
            }
        })
        .collect();

    motifs.sort_by(|a, b| b.count.cmp(&a.count));
    motifs.truncate(max_results);
    motifs
}

fn collect_ngrams(
    node: &TrieNode,
    path: &mut Vec<String>,
    counts: &mut HashMap<Vec<String>, u64>,
) {
    if !node.phoneme.is_empty() {
        path.push(node.phoneme.clone());

        let total = node.total_count();

        // Extract n-grams of length 2..4 from the end of the current path
        for n in 2..=4 {
            if path.len() >= n {
                let ngram: Vec<String> = path[path.len() - n..].to_vec();
                *counts.entry(ngram).or_default() += total;
            }
        }
    }

    for child in node.children.values() {
        collect_ngrams(child, path, counts);
    }

    if !node.phoneme.is_empty() {
        path.pop();
    }
}

/// Build a global transition matrix (phoneme → phoneme bigram probabilities).
pub fn transition_matrix(root: &TrieNode) -> HashMap<String, HashMap<String, f64>> {
    let mut bigram_counts: HashMap<String, HashMap<String, u64>> = HashMap::new();

    collect_bigrams(root, &mut Vec::new(), &mut bigram_counts);

    // Normalize to probabilities
    let mut matrix: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (from, to_counts) in &bigram_counts {
        let total: u64 = to_counts.values().sum();
        if total == 0 {
            continue;
        }
        let mut probs = HashMap::new();
        for (to, count) in to_counts {
            let prob = *count as f64 / total as f64;
            if prob >= 0.005 {
                // Only include transitions ≥ 0.5%
                probs.insert(to.clone(), (prob * 1000.0).round() / 1000.0);
            }
        }
        if !probs.is_empty() {
            matrix.insert(from.clone(), probs);
        }
    }

    matrix
}

fn collect_bigrams(
    node: &TrieNode,
    path: &mut Vec<String>,
    counts: &mut HashMap<String, HashMap<String, u64>>,
) {
    if !node.phoneme.is_empty() {
        path.push(node.phoneme.clone());

        if path.len() >= 2 {
            let from = &path[path.len() - 2];
            let to = &path[path.len() - 1];
            *counts
                .entry(from.clone())
                .or_default()
                .entry(to.clone())
                .or_default() += node.total_count();
        }
    }

    for child in node.children.values() {
        collect_bigrams(child, path, counts);
    }

    if !node.phoneme.is_empty() {
        path.pop();
    }
}

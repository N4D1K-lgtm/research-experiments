//! Trie statistical analysis: branching factors, entropy, depth stats.

use crate::trie::PhonologicalTrie;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-depth statistics for a language.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DepthStats {
    pub depth: u32,
    pub nodes: u32,
    pub terminals: u32,
    pub avg_branch: f64,
    pub avg_entropy: f64,
    pub max_entropy: f64,
}

/// Full analysis results for a trie.
pub struct TrieAnalysis {
    pub depth_stats: Vec<DepthStats>,
    pub phoneme_inventory: Vec<String>,
    pub onset_inventory: Vec<String>,
    pub coda_inventory: Vec<String>,
}

impl TrieAnalysis {
    /// Compute all statistics for a trie.
    pub fn compute(trie: &PhonologicalTrie) -> Self {
        let nodes = trie.nodes_bfs();
        let max_depth = trie.max_depth();

        let mut depth_stats = Vec::new();
        let mut phoneme_set = std::collections::HashSet::new();
        let mut onset_set = std::collections::HashSet::new();
        let mut coda_set = std::collections::HashSet::new();

        for d in 0..=max_depth {
            let nodes_at_depth: Vec<_> = nodes.iter().filter(|n| n.depth == d).collect();
            let node_count = nodes_at_depth.len() as u32;
            let terminal_count = nodes_at_depth.iter().filter(|n| n.is_terminal()).count() as u32;

            // Branching factor: average children per non-leaf node
            let branching_nodes: Vec<_> = nodes_at_depth
                .iter()
                .filter(|n| !n.children.is_empty())
                .collect();
            let avg_branch = if branching_nodes.is_empty() {
                0.0
            } else {
                branching_nodes.iter().map(|n| n.child_count() as f64).sum::<f64>()
                    / branching_nodes.len() as f64
            };

            // Shannon entropy per node
            let mut entropies: Vec<f64> = Vec::new();
            for node in &nodes_at_depth {
                if node.children.is_empty() {
                    continue;
                }
                let total: f64 = node.children.values().map(|c| c.total_count() as f64).sum();
                if total == 0.0 {
                    continue;
                }
                let entropy: f64 = node
                    .children
                    .values()
                    .map(|c| {
                        let p = c.total_count() as f64 / total;
                        if p > 0.0 {
                            -p * p.log2()
                        } else {
                            0.0
                        }
                    })
                    .sum();
                entropies.push(entropy);
            }

            let avg_entropy = if entropies.is_empty() {
                0.0
            } else {
                entropies.iter().sum::<f64>() / entropies.len() as f64
            };
            let max_entropy = entropies
                .iter()
                .copied()
                .fold(0.0_f64, f64::max);

            depth_stats.push(DepthStats {
                depth: d,
                nodes: node_count,
                terminals: terminal_count,
                avg_branch,
                avg_entropy,
                max_entropy,
            });

            // Collect inventories
            for node in &nodes_at_depth {
                if !node.phoneme.is_empty() {
                    phoneme_set.insert(node.phoneme.clone());
                    match node.role {
                        ipa_core::PhonologicalPosition::Onset => {
                            onset_set.insert(node.phoneme.clone());
                        }
                        ipa_core::PhonologicalPosition::Coda => {
                            coda_set.insert(node.phoneme.clone());
                        }
                        _ => {}
                    }
                }
            }
        }

        let mut phoneme_inventory: Vec<String> = phoneme_set.into_iter().collect();
        phoneme_inventory.sort();
        let mut onset_inventory: Vec<String> = onset_set.into_iter().collect();
        onset_inventory.sort();
        let mut coda_inventory: Vec<String> = coda_set.into_iter().collect();
        coda_inventory.sort();

        Self {
            depth_stats,
            phoneme_inventory,
            onset_inventory,
            coda_inventory,
        }
    }

    /// Compute cross-linguistic depth stats: per language, build separate trie stats.
    pub fn cross_linguistic_stats(
        trie: &PhonologicalTrie,
    ) -> HashMap<String, Vec<DepthStats>> {
        let nodes = trie.nodes_bfs();
        let max_depth = trie.max_depth();
        let mut result = HashMap::new();

        for lang in &trie.languages {
            let mut lang_stats = Vec::new();

            for d in 0..=max_depth {
                let nodes_at_depth: Vec<_> = nodes
                    .iter()
                    .filter(|n| n.depth == d && n.counts.get(lang).copied().unwrap_or(0) > 0)
                    .collect();

                let node_count = nodes_at_depth.len() as u32;
                let terminal_count = nodes_at_depth
                    .iter()
                    .filter(|n| n.terminal_counts.get(lang).copied().unwrap_or(0) > 0)
                    .count() as u32;

                let branching_nodes: Vec<_> = nodes_at_depth
                    .iter()
                    .filter(|n| {
                        n.children
                            .values()
                            .any(|c| c.counts.get(lang).copied().unwrap_or(0) > 0)
                    })
                    .collect();

                let avg_branch = if branching_nodes.is_empty() {
                    0.0
                } else {
                    branching_nodes
                        .iter()
                        .map(|n| {
                            n.children
                                .values()
                                .filter(|c| c.counts.get(lang).copied().unwrap_or(0) > 0)
                                .count() as f64
                        })
                        .sum::<f64>()
                        / branching_nodes.len() as f64
                };

                let mut entropies = Vec::new();
                for node in &branching_nodes {
                    let total: f64 = node
                        .children
                        .values()
                        .map(|c| c.counts.get(lang).copied().unwrap_or(0) as f64)
                        .sum();
                    if total == 0.0 {
                        continue;
                    }
                    let entropy: f64 = node
                        .children
                        .values()
                        .filter_map(|c| {
                            let count = c.counts.get(lang).copied().unwrap_or(0) as f64;
                            if count > 0.0 {
                                let p = count / total;
                                Some(-p * p.log2())
                            } else {
                                None
                            }
                        })
                        .sum();
                    entropies.push(entropy);
                }

                let avg_entropy = if entropies.is_empty() {
                    0.0
                } else {
                    entropies.iter().sum::<f64>() / entropies.len() as f64
                };
                let max_entropy = entropies.iter().copied().fold(0.0_f64, f64::max);

                if node_count > 0 {
                    lang_stats.push(DepthStats {
                        depth: d,
                        nodes: node_count,
                        terminals: terminal_count,
                        avg_branch,
                        avg_entropy,
                        max_entropy,
                    });
                }
            }

            result.insert(lang.clone(), lang_stats);
        }

        result
    }
}

use std::collections::HashMap;
use std::sync::Arc;

use ipa_core::{Language, PhonologicalPosition};
use ipa_trie::analysis::{DepthStats, TrieAnalysis};
use ipa_trie::motifs::{self, Motif};
use ipa_trie::trie::PhonologicalTrie;

/// A flattened node ready for GraphQL serving.
#[derive(Debug, Clone)]
pub struct FlatNode {
    pub id: u32,
    pub phoneme: String,
    pub depth: u32,
    pub parent_id: Option<u32>,
    pub child_ids: Vec<u32>,
    pub counts: HashMap<String, u64>,
    pub total_count: u64,
    pub position: [f64; 3],
    pub color: String,
    pub hsl: [f64; 3],
    pub role: PhonologicalPosition,
    pub is_terminal: bool,
    pub child_count: usize,
    pub weight: u64,
    pub transition_probs: HashMap<String, f64>,
    pub allophones: Vec<String>,
    pub terminal_counts: HashMap<String, u64>,
    pub sample_words: HashMap<String, Vec<String>>,
}

/// Metadata about the trie.
#[derive(Debug, Clone)]
pub struct TrieMetadataStore {
    pub languages: Vec<String>,
    pub node_count: u32,
    pub edge_count: u32,
    pub max_depth: u32,
    pub total_words: u64,
    pub terminal_nodes: u32,
    pub phoneme_inventory: Vec<String>,
    pub onset_inventory: Vec<String>,
    pub coda_inventory: Vec<String>,
    pub motifs: Vec<Motif>,
    pub transition_matrix: HashMap<String, HashMap<String, f64>>,
}

/// In-memory trie store with indexes for fast GraphQL lookups.
pub struct TrieStore {
    pub nodes: Vec<FlatNode>,
    pub node_by_id: HashMap<u32, usize>,
    pub nodes_by_depth: HashMap<u32, Vec<u32>>,
    pub edges: Vec<(u32, u32)>,
    pub phoneme_index: HashMap<String, Vec<u32>>,
    pub metadata: TrieMetadataStore,
    pub depth_stats: Vec<DepthStats>,
    pub cross_linguistic_stats: HashMap<String, Vec<DepthStats>>,
    pub language_details: Vec<Language>,
}

impl TrieStore {
    /// Build a TrieStore from a fully-constructed PhonologicalTrie.
    pub fn from_trie(trie: &PhonologicalTrie) -> Arc<Self> {
        let bfs_nodes = trie.nodes_bfs();
        let analysis = TrieAnalysis::compute(trie);
        let cross_stats = TrieAnalysis::cross_linguistic_stats(trie);
        let detected_motifs = motifs::detect_motifs(&trie.root, 500, 200);
        let trans_matrix = motifs::transition_matrix(&trie.root);

        let mut nodes = Vec::with_capacity(bfs_nodes.len());
        let mut node_by_id = HashMap::with_capacity(bfs_nodes.len());
        let mut nodes_by_depth: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut edges = Vec::new();
        let mut phoneme_index: HashMap<String, Vec<u32>> = HashMap::new();
        let mut total_words: u64 = 0;

        for (vec_idx, trie_node) in bfs_nodes.iter().enumerate() {
            let child_ids: Vec<u32> = trie_node.children.values().map(|c| c.node_id).collect();

            // Collect edges
            for &child_id in &child_ids {
                edges.push((trie_node.node_id, child_id));
            }

            total_words += trie_node.terminal_counts.values().sum::<u64>();

            // Index by phoneme
            if !trie_node.phoneme.is_empty() {
                phoneme_index
                    .entry(trie_node.phoneme.clone())
                    .or_default()
                    .push(trie_node.node_id);
            }

            // Index by depth
            nodes_by_depth
                .entry(trie_node.depth)
                .or_default()
                .push(trie_node.node_id);

            let flat = FlatNode {
                id: trie_node.node_id,
                phoneme: trie_node.phoneme.clone(),
                depth: trie_node.depth,
                parent_id: trie_node.parent_id,
                child_ids,
                counts: trie_node.counts.clone(),
                total_count: trie_node.total_count(),
                position: trie_node.position,
                color: trie_node.color.clone(),
                hsl: trie_node.hsl,
                role: trie_node.role,
                is_terminal: trie_node.is_terminal(),
                child_count: trie_node.child_count(),
                weight: trie_node.weight,
                transition_probs: trie_node.transition_probs.clone(),
                allophones: trie_node.allophones.clone(),
                terminal_counts: trie_node.terminal_counts.clone(),
                sample_words: trie_node.sample_words.clone(),
            };

            node_by_id.insert(flat.id, vec_idx);
            nodes.push(flat);
        }

        let metadata = TrieMetadataStore {
            languages: trie.languages.clone(),
            node_count: trie.node_count,
            edge_count: edges.len() as u32,
            max_depth: trie.max_depth(),
            total_words,
            terminal_nodes: trie.terminal_count(),
            phoneme_inventory: analysis.phoneme_inventory,
            onset_inventory: analysis.onset_inventory,
            coda_inventory: analysis.coda_inventory,
            motifs: detected_motifs,
            transition_matrix: trans_matrix,
        };

        let language_details = Language::all();

        Arc::new(Self {
            nodes,
            node_by_id,
            nodes_by_depth,
            edges,
            phoneme_index,
            metadata,
            depth_stats: analysis.depth_stats,
            cross_linguistic_stats: cross_stats,
            language_details,
        })
    }

    /// Look up a node by ID.
    pub fn get_node(&self, id: u32) -> Option<&FlatNode> {
        self.node_by_id.get(&id).map(|&idx| &self.nodes[idx])
    }

    /// Get nodes at a specific depth range.
    pub fn get_nodes_by_depth_range(&self, min_depth: u32, max_depth: u32) -> Vec<&FlatNode> {
        let mut result = Vec::new();
        for depth in min_depth..=max_depth {
            if let Some(ids) = self.nodes_by_depth.get(&depth) {
                for &id in ids {
                    if let Some(node) = self.get_node(id) {
                        result.push(node);
                    }
                }
            }
        }
        result
    }

    /// Search for nodes by phoneme string.
    pub fn search_phoneme(&self, phoneme: &str, limit: usize) -> Vec<&FlatNode> {
        let mut results = Vec::new();

        // Exact match first
        if let Some(ids) = self.phoneme_index.get(phoneme) {
            for &id in ids {
                if results.len() >= limit {
                    break;
                }
                if let Some(node) = self.get_node(id) {
                    results.push(node);
                }
            }
        }

        // If no exact match, try substring match on phoneme keys
        if results.is_empty() {
            for (key, ids) in &self.phoneme_index {
                if key.contains(phoneme) {
                    for &id in ids {
                        if results.len() >= limit {
                            return results;
                        }
                        if let Some(node) = self.get_node(id) {
                            results.push(node);
                        }
                    }
                }
            }
        }

        results
    }

    /// Get edges for a specific depth range.
    pub fn get_edges_by_depth_range(&self, min_depth: u32, max_depth: u32) -> Vec<(u32, u32)> {
        self.edges
            .iter()
            .filter(|(src, tgt)| {
                let src_depth = self.get_node(*src).map(|n| n.depth).unwrap_or(u32::MAX);
                let tgt_depth = self.get_node(*tgt).map(|n| n.depth).unwrap_or(u32::MAX);
                src_depth >= min_depth
                    && src_depth <= max_depth
                    && tgt_depth >= min_depth
                    && tgt_depth <= max_depth
            })
            .copied()
            .collect()
    }
}

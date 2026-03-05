//! Phonological trie construction from normalized pronunciation data.

use ipa_core::{token_is_vowel, PhonologicalPosition};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};

/// A node in the phonological trie.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrieNode {
    /// The IPA phoneme at this node (empty for root).
    pub phoneme: String,
    /// Children keyed by phoneme string.
    pub children: BTreeMap<String, TrieNode>,
    /// Per-language frequency counts.
    pub counts: HashMap<String, u64>,
    /// Assigned node ID (BFS order).
    pub node_id: u32,
    /// Depth from root.
    pub depth: u32,
    /// Parent node ID (None for root).
    pub parent_id: Option<u32>,
    /// Subtree weight for layout.
    pub weight: u64,
    /// Surface forms that collapsed to this phoneme.
    pub allophones: Vec<String>,
    /// Per-language count of words ending here.
    pub terminal_counts: HashMap<String, u64>,
    /// Sample words per language (up to 5 each).
    pub sample_words: HashMap<String, Vec<String>>,
    /// 3D position (assigned by layout).
    pub position: [f64; 3],
    /// HSL color string.
    pub color: String,
    /// HSL components.
    pub hsl: [f64; 3],
    /// Phonological position (onset/nucleus/coda/mixed).
    pub role: PhonologicalPosition,
    /// Transition probabilities to children.
    pub transition_probs: HashMap<String, f64>,
}

impl TrieNode {
    pub fn new_root() -> Self {
        Self {
            phoneme: String::new(),
            children: BTreeMap::new(),
            counts: HashMap::new(),
            node_id: 0,
            depth: 0,
            parent_id: None,
            weight: 0,
            allophones: Vec::new(),
            terminal_counts: HashMap::new(),
            sample_words: HashMap::new(),
            position: [0.0; 3],
            color: String::new(),
            hsl: [0.0; 3],
            role: PhonologicalPosition::Mixed,
            transition_probs: HashMap::new(),
        }
    }

    /// Total count across all languages.
    pub fn total_count(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Whether any words end at this node.
    pub fn is_terminal(&self) -> bool {
        !self.terminal_counts.is_empty()
    }

    /// Number of children.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }
}

/// The full phonological trie with metadata.
pub struct PhonologicalTrie {
    pub root: TrieNode,
    pub languages: Vec<String>,
    pub node_count: u32,
}

impl PhonologicalTrie {
    pub fn new() -> Self {
        Self {
            root: TrieNode::new_root(),
            languages: Vec::new(),
            node_count: 0,
        }
    }

    /// Insert a normalized pronunciation into the trie.
    pub fn insert(
        &mut self,
        phonemes: &[String],
        lang_code: &str,
        word: &str,
    ) {
        let mut node = &mut self.root;
        *node.counts.entry(lang_code.to_string()).or_default() += 1;

        for phoneme in phonemes {
            node = node
                .children
                .entry(phoneme.clone())
                .or_insert_with(|| {
                    let mut child = TrieNode::new_root();
                    child.phoneme = phoneme.clone();
                    child
                });
            *node.counts.entry(lang_code.to_string()).or_default() += 1;
        }

        // Mark terminal
        *node
            .terminal_counts
            .entry(lang_code.to_string())
            .or_default() += 1;

        // Store sample word (max 5 per language)
        let samples = node
            .sample_words
            .entry(lang_code.to_string())
            .or_default();
        if samples.len() < 5 && !samples.contains(&word.to_string()) {
            samples.push(word.to_string());
        }

        // Track language
        if !self.languages.contains(&lang_code.to_string()) {
            self.languages.push(lang_code.to_string());
        }
    }

    /// Prune nodes with total count below threshold.
    pub fn prune(&mut self, min_count: u64) {
        prune_recursive(&mut self.root, min_count);
    }

    /// Assign BFS node IDs and depth/parent references.
    pub fn assign_ids(&mut self) {
        let mut next_id: u32 = 0;

        self.root.node_id = next_id;
        self.root.depth = 0;
        self.root.parent_id = None;
        next_id += 1;

        assign_ids_recursive(&mut self.root, 0, None, &mut next_id);
        self.node_count = next_id;
    }

    /// Classify phonological positions for all nodes.
    pub fn classify_positions(&mut self) {
        classify_recursive(&mut self.root);
    }

    /// Compute transition probabilities for all nodes.
    pub fn compute_transition_probs(&mut self) {
        compute_probs_recursive(&mut self.root);
    }

    /// Compute subtree weights for layout.
    pub fn compute_weights(&mut self) {
        compute_weights_recursive(&mut self.root);
    }

    /// Assign colors based on phonological role.
    pub fn assign_colors(&mut self) {
        assign_colors_recursive(&mut self.root);
    }

    /// Collect all nodes in BFS order.
    pub fn nodes_bfs(&self) -> Vec<&TrieNode> {
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(&self.root);

        while let Some(node) = queue.pop_front() {
            result.push(node);
            for child in node.children.values() {
                queue.push_back(child);
            }
        }

        result
    }

    /// Get maximum depth of the trie.
    pub fn max_depth(&self) -> u32 {
        max_depth_recursive(&self.root)
    }

    /// Count terminal nodes.
    pub fn terminal_count(&self) -> u32 {
        let mut count = 0;
        for node in self.nodes_bfs() {
            if node.is_terminal() {
                count += 1;
            }
        }
        count
    }
}

fn prune_recursive(node: &mut TrieNode, min_count: u64) {
    node.children
        .retain(|_, child| child.total_count() >= min_count);
    for child in node.children.values_mut() {
        prune_recursive(child, min_count);
    }
}

fn assign_ids_recursive(
    node: &mut TrieNode,
    depth: u32,
    parent_id: Option<u32>,
    next_id: &mut u32,
) {
    node.depth = depth;
    node.parent_id = parent_id;
    let my_id = node.node_id;

    for child in node.children.values_mut() {
        child.node_id = *next_id;
        *next_id += 1;
        assign_ids_recursive(child, depth + 1, Some(my_id), next_id);
    }
}

fn classify_recursive(node: &mut TrieNode) {
    if !node.phoneme.is_empty() {
        node.role = classify_position(node);
    }
    for child in node.children.values_mut() {
        classify_recursive(child);
    }
}

fn classify_position(node: &TrieNode) -> PhonologicalPosition {
    if node.phoneme.is_empty() {
        return PhonologicalPosition::Mixed;
    }

    let is_vowel = token_is_vowel(&node.phoneme);

    if is_vowel {
        PhonologicalPosition::Nucleus
    } else if node.depth <= 1 {
        PhonologicalPosition::Onset
    } else if node.is_terminal() && !is_vowel {
        PhonologicalPosition::Coda
    } else {
        // Could be onset or coda depending on context
        // For simplicity, use depth heuristic: early = onset, late = coda
        PhonologicalPosition::Mixed
    }
}

fn compute_probs_recursive(node: &mut TrieNode) {
    let total: u64 = node.children.values().map(|c| c.total_count()).sum();
    if total > 0 {
        for (phoneme, child) in &node.children {
            let prob = child.total_count() as f64 / total as f64;
            node.transition_probs.insert(phoneme.clone(), prob);
        }
    }
    for child in node.children.values_mut() {
        compute_probs_recursive(child);
    }
}

fn compute_weights_recursive(node: &mut TrieNode) -> u64 {
    let mut weight = node.total_count();
    for child in node.children.values_mut() {
        weight += compute_weights_recursive(child);
    }
    node.weight = weight;
    weight
}

fn max_depth_recursive(node: &TrieNode) -> u32 {
    if node.children.is_empty() {
        return node.depth;
    }
    node.children
        .values()
        .map(|c| max_depth_recursive(c))
        .max()
        .unwrap_or(node.depth)
}

/// Role-based HSL color hues.
const ROLE_HUES: [(PhonologicalPosition, f64); 4] = [
    (PhonologicalPosition::Onset, 200.0),
    (PhonologicalPosition::Nucleus, 45.0),
    (PhonologicalPosition::Coda, 280.0),
    (PhonologicalPosition::Mixed, 150.0),
];

fn role_hue(role: PhonologicalPosition) -> f64 {
    for (r, h) in &ROLE_HUES {
        if *r == role {
            return *h;
        }
    }
    150.0
}

fn assign_colors_recursive(node: &mut TrieNode) {
    let hue = role_hue(node.role);
    let saturation = if node.role != PhonologicalPosition::Mixed {
        0.65
    } else {
        0.35
    };
    let total = node.total_count().max(1) as f64;
    let mut lightness = (0.32 + 0.18 * total.log10()).min(0.72);

    if node.is_terminal() {
        lightness = (lightness + 0.15).min(0.87);
    }

    node.hsl = [hue, saturation, lightness];
    node.color = hsl_to_hex(hue, saturation, lightness);

    for child in node.children.values_mut() {
        assign_colors_recursive(child);
    }
}

fn hsl_to_hex(h: f64, s: f64, l: f64) -> String {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let r = ((r1 + m) * 255.0).round() as u8;
    let g = ((g1 + m) * 255.0).round() as u8;
    let b = ((b1 + m) * 255.0).round() as u8;

    format!("#{r:02x}{g:02x}{b:02x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_count() {
        let mut trie = PhonologicalTrie::new();
        let phonemes: Vec<String> = vec!["p", "æ", "t"].into_iter().map(String::from).collect();
        trie.insert(&phonemes, "en_US", "pat");
        trie.insert(&phonemes, "en_US", "pat");

        assert_eq!(trie.root.total_count(), 2);
        assert_eq!(trie.root.children["p"].total_count(), 2);
        assert_eq!(trie.root.children["p"].children["æ"].total_count(), 2);
    }

    #[test]
    fn test_terminal_marking() {
        let mut trie = PhonologicalTrie::new();
        let phonemes: Vec<String> = vec!["k", "æ", "t"].into_iter().map(String::from).collect();
        trie.insert(&phonemes, "en_US", "cat");

        let t_node = &trie.root.children["k"].children["æ"].children["t"];
        assert!(t_node.is_terminal());
        assert_eq!(t_node.terminal_counts["en_US"], 1);
    }

    #[test]
    fn test_assign_ids() {
        let mut trie = PhonologicalTrie::new();
        let p1: Vec<String> = vec!["p", "æ", "t"].into_iter().map(String::from).collect();
        let p2: Vec<String> = vec!["k", "æ", "t"].into_iter().map(String::from).collect();
        trie.insert(&p1, "en_US", "pat");
        trie.insert(&p2, "en_US", "cat");
        trie.assign_ids();

        assert_eq!(trie.root.node_id, 0);
        assert!(trie.node_count > 0);
    }

    #[test]
    fn test_prune() {
        let mut trie = PhonologicalTrie::new();
        let p1: Vec<String> = vec!["p", "æ", "t"].into_iter().map(String::from).collect();
        trie.insert(&p1, "en_US", "pat");

        let p2: Vec<String> = vec!["z", "ɪ", "p"].into_iter().map(String::from).collect();
        for _ in 0..100 {
            trie.insert(&p2, "en_US", "zip");
        }

        trie.prune(10);
        assert!(!trie.root.children.contains_key("p")); // pruned
        assert!(trie.root.children.contains_key("z")); // kept
    }

    #[test]
    fn test_hsl_to_hex() {
        let hex = hsl_to_hex(0.0, 1.0, 0.5);
        assert_eq!(hex, "#ff0000");

        let hex = hsl_to_hex(120.0, 1.0, 0.5);
        assert_eq!(hex, "#00ff00");
    }
}

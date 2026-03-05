//! Export full trie.json matching the TypeScript TrieData interface.

use anyhow::Result;
use ipa_trie::analysis::TrieAnalysis;
use ipa_trie::motifs;
use ipa_trie::trie::PhonologicalTrie;
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

/// Matches TypeScript `TrieNodeData` interface.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TrieNodeData {
    id: u32,
    phoneme: String,
    depth: u32,
    parent_id: Option<u32>,
    counts: HashMap<String, u64>,
    total_count: u64,
    position: Position,
    color: String,
    hsl: HslColor,
    phonological_position: String,
    is_terminal: bool,
    child_count: usize,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    transition_probs: HashMap<String, f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    allophones: Vec<String>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    terminal_counts: HashMap<String, u64>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    words: HashMap<String, Vec<String>>,
}

#[derive(Serialize)]
struct Position {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Serialize)]
struct HslColor {
    h: f64,
    s: f64,
    l: f64,
}

#[derive(Serialize)]
struct TrieEdge {
    source: u32,
    target: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MotifData {
    sequence: Vec<String>,
    count: u64,
    label: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TrieMetadata {
    languages: Vec<String>,
    node_count: u32,
    edge_count: u32,
    max_depth: u32,
    total_words: u64,
    terminal_nodes: u32,
    phoneme_inventory: Vec<String>,
    onset_inventory: Vec<String>,
    coda_inventory: Vec<String>,
    motifs: Vec<MotifData>,
    transition_matrix: HashMap<String, HashMap<String, f64>>,
    allophone_contexts: HashMap<String, AllophoneContext>,
}

#[derive(Serialize)]
struct AllophoneContext {
    before: Vec<String>,
    after: Vec<String>,
}

#[derive(Serialize)]
struct TrieData {
    metadata: TrieMetadata,
    nodes: Vec<TrieNodeData>,
    edges: Vec<TrieEdge>,
}

/// Export the trie as trie.json.
pub fn export_trie(trie: &PhonologicalTrie, output_dir: &Path) -> Result<()> {
    let analysis = TrieAnalysis::compute(trie);
    let detected_motifs = motifs::detect_motifs(&trie.root, 500, 200);
    let trans_matrix = motifs::transition_matrix(&trie.root);

    let nodes_bfs = trie.nodes_bfs();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    let mut total_words: u64 = 0;

    for node in &nodes_bfs {
        total_words += node.terminal_counts.values().sum::<u64>();

        nodes.push(TrieNodeData {
            id: node.node_id,
            phoneme: node.phoneme.clone(),
            depth: node.depth,
            parent_id: node.parent_id,
            counts: node.counts.clone(),
            total_count: node.total_count(),
            position: Position {
                x: node.position[0],
                y: node.position[1],
                z: node.position[2],
            },
            color: node.color.clone(),
            hsl: HslColor {
                h: node.hsl[0],
                s: node.hsl[1],
                l: node.hsl[2],
            },
            phonological_position: node.role.to_string(),
            is_terminal: node.is_terminal(),
            child_count: node.child_count(),
            transition_probs: node.transition_probs.clone(),
            allophones: node.allophones.clone(),
            terminal_counts: node.terminal_counts.clone(),
            words: node.sample_words.clone(),
        });

        // Edges from parent to children
        for child in node.children.values() {
            edges.push(TrieEdge {
                source: node.node_id,
                target: child.node_id,
            });
        }
    }

    let motif_data: Vec<MotifData> = detected_motifs
        .into_iter()
        .map(|m| MotifData {
            sequence: m.sequence,
            count: m.count,
            label: m.label,
        })
        .collect();

    let data = TrieData {
        metadata: TrieMetadata {
            languages: trie.languages.clone(),
            node_count: trie.node_count,
            edge_count: edges.len() as u32,
            max_depth: trie.max_depth(),
            total_words,
            terminal_nodes: trie.terminal_count(),
            phoneme_inventory: analysis.phoneme_inventory,
            onset_inventory: analysis.onset_inventory,
            coda_inventory: analysis.coda_inventory,
            motifs: motif_data,
            transition_matrix: trans_matrix,
            allophone_contexts: HashMap::new(), // TODO: implement allophone context collection
        },
        nodes,
        edges,
    };

    let output_path = output_dir.join("trie.json");
    std::fs::create_dir_all(output_dir)?;
    let json = serde_json::to_string_pretty(&data)?;
    std::fs::write(&output_path, json)?;
    tracing::info!("Exported trie.json ({} nodes, {} edges)", trie.node_count, data.edges.len());

    Ok(())
}

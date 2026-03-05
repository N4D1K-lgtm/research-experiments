//! Export compact essay_data.json matching the TypeScript EssayData interface.

use anyhow::Result;
use ipa_trie::analysis::TrieAnalysis;
use ipa_trie::motifs;
use ipa_trie::trie::PhonologicalTrie;
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

/// Compact node format for the essay visualization.
#[derive(Serialize)]
struct CompactNode {
    id: u32,
    ph: String,
    d: u32,
    pid: Option<u32>,
    role: String,
    cnt: u64,
    term: bool,
    x: f64,
    y: f64,
    z: f64,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    tp: HashMap<String, f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    w: Vec<String>,
}

#[derive(Serialize)]
struct EssayMotif {
    seq: String,
    count: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EssayMeta {
    total_nodes: u32,
    total_edges: u32,
    total_words: u64,
    total_terminals: u32,
    phoneme_inventory: Vec<String>,
    onset_inventory: Vec<String>,
    coda_inventory: Vec<String>,
    max_depth: u32,
    motifs: Vec<EssayMotif>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EssayData {
    nodes: Vec<CompactNode>,
    meta: EssayMeta,
    transition_matrix: HashMap<String, HashMap<String, f64>>,
}

/// Export essay data (compact, depth ≤ max_depth).
pub fn export_essay_data(
    trie: &PhonologicalTrie,
    output_dir: &Path,
    max_depth: u32,
) -> Result<()> {
    let analysis = TrieAnalysis::compute(trie);
    let detected_motifs = motifs::detect_motifs(&trie.root, 500, 200);
    let trans_matrix = motifs::transition_matrix(&trie.root);

    let all_nodes = trie.nodes_bfs();
    let mut nodes = Vec::new();
    let mut edge_count = 0u32;
    let mut total_words = 0u64;

    for node in &all_nodes {
        if node.depth > max_depth {
            continue;
        }

        total_words += node.terminal_counts.values().sum::<u64>();

        // Flatten sample words
        let mut words: Vec<String> = Vec::new();
        for samples in node.sample_words.values() {
            for w in samples {
                if words.len() < 5 {
                    words.push(w.clone());
                }
            }
        }

        // Top transition probs (up to 5)
        let mut tp: Vec<(String, f64)> = node
            .transition_probs
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        tp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        tp.truncate(5);
        let tp_map: HashMap<String, f64> = tp.into_iter().collect();

        nodes.push(CompactNode {
            id: node.node_id,
            ph: node.phoneme.clone(),
            d: node.depth,
            pid: node.parent_id,
            role: node.role.short_code().to_string(),
            cnt: node.total_count(),
            term: node.is_terminal(),
            x: node.position[0],
            y: node.position[1],
            z: node.position[2],
            tp: tp_map,
            w: words,
        });

        // Count edges within depth limit
        for child in node.children.values() {
            if child.depth <= max_depth {
                edge_count += 1;
            }
        }
    }

    let essay_motifs: Vec<EssayMotif> = detected_motifs
        .into_iter()
        .map(|m| EssayMotif {
            seq: m.label,
            count: m.count,
        })
        .collect();

    let data = EssayData {
        nodes,
        meta: EssayMeta {
            total_nodes: trie.node_count,
            total_edges: edge_count,
            total_words,
            total_terminals: trie.terminal_count(),
            phoneme_inventory: analysis.phoneme_inventory,
            onset_inventory: analysis.onset_inventory,
            coda_inventory: analysis.coda_inventory,
            max_depth: trie.max_depth(),
            motifs: essay_motifs,
        },
        transition_matrix: trans_matrix,
    };

    let output_path = output_dir.join("essay_data.json");
    std::fs::create_dir_all(output_dir)?;
    let json = serde_json::to_string(&data)?;
    std::fs::write(&output_path, json)?;
    tracing::info!("Exported essay_data.json ({} nodes)", data.nodes.len());

    Ok(())
}

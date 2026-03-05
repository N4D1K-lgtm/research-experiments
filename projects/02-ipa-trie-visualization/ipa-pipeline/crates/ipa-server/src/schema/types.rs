use std::sync::Arc;

use async_graphql::*;
use ipa_core::PhonologicalPosition;

use crate::store::{FlatNode, TrieStore};

/// A node in the phonological trie.
pub struct TrieNodeGql {
    pub node: FlatNode,
}

#[Object]
impl TrieNodeGql {
    async fn id(&self) -> u32 {
        self.node.id
    }

    async fn phoneme(&self) -> &str {
        &self.node.phoneme
    }

    async fn depth(&self) -> u32 {
        self.node.depth
    }

    async fn parent_id(&self) -> Option<u32> {
        self.node.parent_id
    }

    async fn counts(&self) -> Vec<LanguageCount> {
        self.node
            .counts
            .iter()
            .map(|(lang, &count)| LanguageCount {
                language: lang.clone(),
                count,
            })
            .collect()
    }

    async fn total_count(&self) -> u64 {
        self.node.total_count
    }

    async fn position(&self) -> Position {
        Position {
            x: self.node.position[0],
            y: self.node.position[1],
            z: self.node.position[2],
        }
    }

    async fn color(&self) -> &str {
        &self.node.color
    }

    async fn hsl(&self) -> HslColor {
        HslColor {
            h: self.node.hsl[0],
            s: self.node.hsl[1],
            l: self.node.hsl[2],
        }
    }

    async fn phonological_position(&self) -> &str {
        match self.node.role {
            PhonologicalPosition::Onset => "onset",
            PhonologicalPosition::Nucleus => "nucleus",
            PhonologicalPosition::Coda => "coda",
            PhonologicalPosition::Mixed => "mixed",
        }
    }

    async fn is_terminal(&self) -> bool {
        self.node.is_terminal
    }

    async fn child_count(&self) -> i32 {
        self.node.child_count as i32
    }

    async fn weight(&self) -> u64 {
        self.node.weight
    }

    async fn transition_probs(&self) -> Vec<TransitionProb> {
        self.node
            .transition_probs
            .iter()
            .map(|(phoneme, &prob)| TransitionProb {
                phoneme: phoneme.clone(),
                probability: prob,
            })
            .collect()
    }

    async fn allophones(&self) -> &[String] {
        &self.node.allophones
    }

    async fn terminal_counts(&self) -> Vec<LanguageCount> {
        self.node
            .terminal_counts
            .iter()
            .map(|(lang, &count)| LanguageCount {
                language: lang.clone(),
                count,
            })
            .collect()
    }

    async fn words(&self) -> Vec<LanguageWords> {
        self.node
            .sample_words
            .iter()
            .map(|(lang, words)| LanguageWords {
                language: lang.clone(),
                words: words.clone(),
            })
            .collect()
    }

    /// Resolve parent node lazily.
    async fn parent(&self, ctx: &Context<'_>) -> Option<TrieNodeGql> {
        let store = ctx.data::<Arc<TrieStore>>().ok()?;
        let parent_id = self.node.parent_id?;
        store.get_node(parent_id).map(|n| TrieNodeGql { node: n.clone() })
    }

    /// Resolve child nodes lazily.
    async fn children(&self, ctx: &Context<'_>) -> Vec<TrieNodeGql> {
        let store = match ctx.data::<Arc<TrieStore>>() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        self.node
            .child_ids
            .iter()
            .filter_map(|&id| store.get_node(id).map(|n| TrieNodeGql { node: n.clone() }))
            .collect()
    }
}

/// 3D position.
#[derive(SimpleObject)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// HSL color components.
#[derive(SimpleObject)]
pub struct HslColor {
    pub h: f64,
    pub s: f64,
    pub l: f64,
}

/// Per-language count.
#[derive(SimpleObject)]
pub struct LanguageCount {
    pub language: String,
    pub count: u64,
}

/// Per-language word samples.
#[derive(SimpleObject)]
pub struct LanguageWords {
    pub language: String,
    pub words: Vec<String>,
}

/// Transition probability entry.
#[derive(SimpleObject)]
pub struct TransitionProb {
    pub phoneme: String,
    pub probability: f64,
}

/// An edge in the trie.
#[derive(SimpleObject)]
pub struct TrieEdge {
    pub source: u32,
    pub target: u32,
}

/// Trie metadata.
#[derive(SimpleObject)]
pub struct TrieMetadata {
    pub languages: Vec<String>,
    pub node_count: u32,
    pub edge_count: u32,
    pub max_depth: u32,
    pub total_words: u64,
    pub terminal_nodes: u32,
    pub phoneme_inventory: Vec<String>,
    pub onset_inventory: Vec<String>,
    pub coda_inventory: Vec<String>,
    pub motifs: Vec<MotifGql>,
}

/// A phonological motif.
#[derive(SimpleObject)]
pub struct MotifGql {
    pub sequence: Vec<String>,
    pub count: u64,
    pub label: String,
}

/// Paginated node connection.
#[derive(SimpleObject)]
pub struct NodeConnection {
    pub nodes: Vec<TrieNodeGql>,
    pub total_count: i32,
    pub has_more: bool,
}

/// Search result.
#[derive(SimpleObject)]
pub struct SearchResult {
    pub nodes: Vec<TrieNodeGql>,
    pub total_matches: i32,
}

/// Per-depth statistics.
#[derive(SimpleObject, Clone)]
pub struct DepthStatsGql {
    pub depth: u32,
    pub nodes: u32,
    pub terminals: u32,
    pub avg_branch: f64,
    pub avg_entropy: f64,
    pub max_entropy: f64,
}

/// Per-language depth statistics.
#[derive(SimpleObject)]
pub struct LanguageDepthStats {
    pub language: String,
    pub stats: Vec<DepthStatsGql>,
}

/// Language info.
#[derive(SimpleObject)]
pub struct LanguageGql {
    pub code: String,
    pub name: String,
    pub family: String,
    pub typology: String,
    pub iso639_3: String,
}

/// Transition matrix entry.
#[derive(SimpleObject)]
pub struct TransitionMatrixEntry {
    pub from_phoneme: String,
    pub transitions: Vec<TransitionProb>,
}

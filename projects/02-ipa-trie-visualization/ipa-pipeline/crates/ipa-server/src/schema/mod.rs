pub mod types;

use std::sync::Arc;

use async_graphql::*;

use crate::store::TrieStore;
use types::*;

pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// Get trie metadata (languages, counts, inventories).
    async fn metadata(&self, ctx: &Context<'_>) -> Result<TrieMetadata> {
        let store = ctx.data::<Arc<TrieStore>>()?;
        let meta = &store.metadata;
        Ok(TrieMetadata {
            languages: meta.languages.clone(),
            node_count: meta.node_count,
            edge_count: meta.edge_count,
            max_depth: meta.max_depth,
            total_words: meta.total_words,
            terminal_nodes: meta.terminal_nodes,
            phoneme_inventory: meta.phoneme_inventory.clone(),
            onset_inventory: meta.onset_inventory.clone(),
            coda_inventory: meta.coda_inventory.clone(),
            motifs: meta
                .motifs
                .iter()
                .map(|m| MotifGql {
                    sequence: m.sequence.clone(),
                    count: m.count,
                    label: m.label.clone(),
                })
                .collect(),
        })
    }

    /// Get a single node by ID.
    async fn node(&self, ctx: &Context<'_>, id: i32) -> Result<Option<TrieNodeGql>> {
        let store = ctx.data::<Arc<TrieStore>>()?;
        Ok(store
            .get_node(id as u32)
            .map(|n| TrieNodeGql { node: n.clone() }))
    }

    /// Get nodes within a depth range with pagination.
    async fn nodes_by_depth(
        &self,
        ctx: &Context<'_>,
        min_depth: i32,
        max_depth: i32,
        offset: Option<i32>,
        limit: Option<i32>,
    ) -> Result<NodeConnection> {
        let store = ctx.data::<Arc<TrieStore>>()?;
        let all_nodes = store.get_nodes_by_depth_range(min_depth as u32, max_depth as u32);
        let total = all_nodes.len() as i32;

        let off = offset.unwrap_or(0).max(0) as usize;
        let lim = limit.unwrap_or(5000).max(1) as usize;

        let page: Vec<TrieNodeGql> = all_nodes
            .into_iter()
            .skip(off)
            .take(lim)
            .map(|n| TrieNodeGql { node: n.clone() })
            .collect();

        let has_more = (off + page.len()) < total as usize;

        Ok(NodeConnection {
            nodes: page,
            total_count: total,
            has_more,
        })
    }

    /// Get children of a specific node.
    async fn children_of(&self, ctx: &Context<'_>, parent_id: i32) -> Result<Vec<TrieNodeGql>> {
        let store = ctx.data::<Arc<TrieStore>>()?;
        let parent = store
            .get_node(parent_id as u32)
            .ok_or_else(|| Error::new("Node not found"))?;

        Ok(parent
            .child_ids
            .iter()
            .filter_map(|&id| store.get_node(id).map(|n| TrieNodeGql { node: n.clone() }))
            .collect())
    }

    /// Get edges within a depth range.
    async fn edges_by_depth(
        &self,
        ctx: &Context<'_>,
        min_depth: i32,
        max_depth: i32,
    ) -> Result<Vec<TrieEdge>> {
        let store = ctx.data::<Arc<TrieStore>>()?;
        let edges = store.get_edges_by_depth_range(min_depth as u32, max_depth as u32);
        Ok(edges
            .into_iter()
            .map(|(source, target)| TrieEdge { source, target })
            .collect())
    }

    /// Search for nodes by phoneme.
    async fn search(
        &self,
        ctx: &Context<'_>,
        phoneme: String,
        limit: Option<i32>,
    ) -> Result<SearchResult> {
        let store = ctx.data::<Arc<TrieStore>>()?;
        let lim = limit.unwrap_or(50).max(1) as usize;
        let results = store.search_phoneme(&phoneme, lim);
        let total = results.len() as i32;

        Ok(SearchResult {
            nodes: results
                .into_iter()
                .map(|n| TrieNodeGql { node: n.clone() })
                .collect(),
            total_matches: total,
        })
    }

    /// Get per-depth statistics.
    async fn depth_stats(&self, ctx: &Context<'_>) -> Result<Vec<DepthStatsGql>> {
        let store = ctx.data::<Arc<TrieStore>>()?;
        Ok(store
            .depth_stats
            .iter()
            .map(|ds| DepthStatsGql {
                depth: ds.depth,
                nodes: ds.nodes,
                terminals: ds.terminals,
                avg_branch: ds.avg_branch,
                avg_entropy: ds.avg_entropy,
                max_entropy: ds.max_entropy,
            })
            .collect())
    }

    /// Get cross-linguistic depth statistics.
    async fn cross_linguistic_stats(
        &self,
        ctx: &Context<'_>,
        languages: Option<Vec<String>>,
    ) -> Result<Vec<LanguageDepthStats>> {
        let store = ctx.data::<Arc<TrieStore>>()?;
        let mut result = Vec::new();

        for (lang, stats) in &store.cross_linguistic_stats {
            if let Some(ref filter) = languages {
                if !filter.contains(lang) {
                    continue;
                }
            }
            result.push(LanguageDepthStats {
                language: lang.clone(),
                stats: stats
                    .iter()
                    .map(|ds| DepthStatsGql {
                        depth: ds.depth,
                        nodes: ds.nodes,
                        terminals: ds.terminals,
                        avg_branch: ds.avg_branch,
                        avg_entropy: ds.avg_entropy,
                        max_entropy: ds.max_entropy,
                    })
                    .collect(),
            });
        }

        Ok(result)
    }

    /// Get all supported languages.
    async fn languages(&self, ctx: &Context<'_>) -> Result<Vec<LanguageGql>> {
        let store = ctx.data::<Arc<TrieStore>>()?;
        Ok(store
            .language_details
            .iter()
            .filter(|l| store.metadata.languages.contains(&l.code))
            .map(|l| LanguageGql {
                code: l.code.clone(),
                name: l.name.clone(),
                family: l.family.clone(),
                typology: l.typology.clone(),
                iso639_3: l.iso639_3.clone(),
            })
            .collect())
    }

    /// Get the global transition matrix.
    async fn transition_matrix(&self, ctx: &Context<'_>) -> Result<Vec<TransitionMatrixEntry>> {
        let store = ctx.data::<Arc<TrieStore>>()?;
        Ok(store
            .metadata
            .transition_matrix
            .iter()
            .map(|(from, transitions)| TransitionMatrixEntry {
                from_phoneme: from.clone(),
                transitions: transitions
                    .iter()
                    .map(|(phoneme, &prob)| TransitionProb {
                        phoneme: phoneme.clone(),
                        probability: prob,
                    })
                    .collect(),
            })
            .collect())
    }
}

pub type IpaSchema = Schema<QueryRoot, EmptyMutation, EmptySubscription>;

pub fn build_schema(store: Arc<TrieStore>) -> IpaSchema {
    Schema::build(QueryRoot, EmptyMutation, EmptySubscription)
        .data(store)
        .finish()
}

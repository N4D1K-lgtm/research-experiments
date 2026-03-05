//! Export cross_linguistic_stats.json.

use anyhow::Result;
use ipa_trie::analysis::TrieAnalysis;
use ipa_trie::trie::PhonologicalTrie;
use std::path::Path;

/// Export cross-linguistic depth statistics per language.
pub fn export_cross_linguistic_stats(trie: &PhonologicalTrie, output_dir: &Path) -> Result<()> {
    let stats = TrieAnalysis::cross_linguistic_stats(trie);

    let output_path = output_dir.join("cross_linguistic_stats.json");
    std::fs::create_dir_all(output_dir)?;
    let json = serde_json::to_string_pretty(&stats)?;
    std::fs::write(&output_path, json)?;
    tracing::info!(
        "Exported cross_linguistic_stats.json ({} languages)",
        stats.len()
    );

    Ok(())
}

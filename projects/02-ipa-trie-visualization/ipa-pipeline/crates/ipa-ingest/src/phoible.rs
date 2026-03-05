//! PHOIBLE phonological inventory downloader and parser.

use anyhow::Result;
use ipa_core::phoneme_inventory::{InventoryStore, PhonemeInventory};
use std::path::{Path, PathBuf};

pub struct PhoibleSource {
    data_dir: PathBuf,
}

impl PhoibleSource {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
        }
    }

    /// Download phoible.csv.
    pub async fn download(&self) -> Result<PathBuf> {
        let url = "https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv";
        let output_path = self.data_dir.join("phoible").join("phoible.csv");

        if output_path.exists() {
            tracing::info!("Cached: phoible.csv");
            return Ok(output_path);
        }

        tracing::info!("Downloading PHOIBLE...");
        let client = reqwest::Client::new();
        let resp = client.get(url).send().await?;
        let bytes = resp.bytes().await?;

        std::fs::create_dir_all(output_path.parent().unwrap())?;
        std::fs::write(&output_path, &bytes)?;
        Ok(output_path)
    }

    /// Parse PHOIBLE CSV into per-language phoneme inventories.
    pub fn parse(&self, path: &Path, target_iso_codes: &[&str]) -> Result<InventoryStore> {
        let mut store = InventoryStore::new();
        let mut reader = csv::Reader::from_path(path)?;

        // Track inventories per language: use the first (lowest-numbered) inventory ID per language
        let mut seen_inventories: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();

        for result in reader.records() {
            let record = result?;
            let iso = record.get(3).unwrap_or("").trim(); // ISO6393 column

            if !target_iso_codes.contains(&iso) {
                continue;
            }

            let inventory_id: u64 = record.get(0).unwrap_or("0").trim().parse().unwrap_or(0);
            let phoneme = record.get(6).unwrap_or("").trim().to_string(); // Phoneme column
            let seg_class = record.get(8).unwrap_or("").trim(); // SegmentClass column

            // Only use the first inventory for each language (most conservative)
            let first_id = *seen_inventories
                .entry(iso.to_string())
                .or_insert(inventory_id);
            if inventory_id != first_id {
                continue;
            }

            if phoneme.is_empty() {
                continue;
            }

            let inventory = store
                .inventories
                .entry(iso.to_string())
                .or_insert_with(PhonemeInventory::default);

            // Marginal phonemes are typically marked but we'll treat all as valid
            let _ = seg_class; // Available for future use
            inventory.phonemes.insert(phoneme);
        }

        Ok(store)
    }
}

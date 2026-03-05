//! CMU Pronouncing Dictionary (IPA-converted) downloader and parser.

use crate::{DataSource, RawPronunciation};
use anyhow::Result;
use ipa_core::Language;
use std::path::{Path, PathBuf};

pub struct CmuDictSource {
    data_dir: PathBuf,
}

impl CmuDictSource {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
        }
    }

    /// Download the IPA-converted CMU dict.
    pub async fn download(&self) -> Result<PathBuf> {
        let url = "https://raw.githubusercontent.com/menelik3/cmudict-ipa/master/cmudict-0.7b-ipa.txt";
        let output_path = self.data_dir.join("cmu").join("cmudict-0.7b-ipa.txt");

        if output_path.exists() {
            tracing::info!("Cached: cmudict-0.7b-ipa.txt");
            return Ok(output_path);
        }

        tracing::info!("Downloading CMU Dict (IPA)...");
        let client = reqwest::Client::new();
        let resp = client.get(url).send().await?;
        let bytes = resp.bytes().await?;

        std::fs::create_dir_all(output_path.parent().unwrap())?;
        std::fs::write(&output_path, &bytes)?;
        Ok(output_path)
    }

    /// Parse CMU Dict IPA file.
    pub fn parse(&self, path: &Path) -> Result<Vec<RawPronunciation>> {
        let content = std::fs::read_to_string(path)?;
        let lang = Language::by_code("en_US").unwrap();
        let mut entries = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with(";;;") {
                continue;
            }

            let parts: Vec<&str> = line.splitn(2, '\t').collect();
            if parts.len() < 2 {
                continue;
            }

            let word = parts[0].trim().to_lowercase();
            let ipa = parts[1].trim().to_string();

            // Skip variant pronunciations (e.g., "WORD(2)")
            if word.contains('(') {
                continue;
            }

            entries.push(RawPronunciation {
                word,
                ipa,
                language: lang.clone(),
                source: DataSource::CmuDict,
            });
        }

        Ok(entries)
    }
}

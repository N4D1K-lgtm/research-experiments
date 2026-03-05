//! WikiPron data source downloader and parser.

use crate::{DataSource, RawPronunciation};
use anyhow::Result;
use ipa_core::Language;
use std::path::{Path, PathBuf};

/// WikiPron file naming convention mapping.
pub struct WikiPronSource {
    data_dir: PathBuf,
}

impl WikiPronSource {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
        }
    }

    /// Download WikiPron TSV files for the given languages.
    pub async fn download(&self, languages: &[Language]) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        let client = reqwest::Client::new();
        let base_url = "https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv";

        for lang in languages {
            let prefix = wikipron_prefix(&lang.iso639_3);
            if prefix.is_empty() {
                tracing::warn!("No WikiPron prefix for {}", lang.code);
                continue;
            }

            let suffix = if is_narrow_only(&lang.iso639_3) {
                "narrow"
            } else {
                "broad"
            };
            let filename = format!("{prefix}_{suffix}.tsv");
            let url = format!("{base_url}/{filename}");
            let output_path = self.data_dir.join("wikipron").join(&filename);

            if output_path.exists() {
                tracing::info!("Cached: {filename}");
                paths.push(output_path);
                continue;
            }

            tracing::info!("Downloading: {url}");
            let resp = client.get(&url).send().await?;

            if !resp.status().is_success() {
                tracing::warn!("Failed to download {filename}: {}", resp.status());
                continue;
            }

            let bytes = resp.bytes().await?;
            std::fs::create_dir_all(output_path.parent().unwrap())?;
            std::fs::write(&output_path, &bytes)?;
            paths.push(output_path);
        }

        Ok(paths)
    }

    /// Parse a WikiPron TSV file into raw pronunciations.
    pub fn parse(&self, path: &Path, language: &Language) -> Result<Vec<RawPronunciation>> {
        let content = std::fs::read_to_string(path)?;
        let mut entries = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 2 {
                continue;
            }

            let word = parts[0].trim().to_string();
            let ipa = parts[1].trim().to_string();

            // WikiPron uses space-segmented phonemes; we'll re-join and re-tokenize
            entries.push(RawPronunciation {
                word,
                ipa,
                language: language.clone(),
                source: DataSource::WikiPron,
            });
        }

        Ok(entries)
    }
}

/// Map ISO 639-3 codes to WikiPron filename prefixes.
/// Some languages only have narrow transcriptions (no broad).
fn wikipron_prefix(iso639_3: &str) -> &'static str {
    match iso639_3 {
        "eng" => "eng_latn_us",
        "fra" => "fra_latn",
        "spa" => "spa_latn_la",
        "deu" => "deu_latn",
        "nld" => "nld_latn",
        "cmn" => "zho_hani",
        "jpn" => "jpn_hira",
        "arb" => "ara_arab",
        "fin" => "fin_latn",
        "tur" => "tur_latn",
        "hin" => "hin_deva",
        "swh" => "swa_latn",
        _ => "",
    }
}

/// Whether this language only has narrow transcriptions on WikiPron.
fn is_narrow_only(iso639_3: &str) -> bool {
    matches!(iso639_3, "jpn")
}

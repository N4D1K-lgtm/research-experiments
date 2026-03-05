pub mod wikipron;
pub mod cmu;
pub mod phoible;
pub mod download;

use ipa_core::Language;
use serde::{Deserialize, Serialize};

/// A raw pronunciation entry from any data source, before validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawPronunciation {
    pub word: String,
    pub ipa: String,
    pub language: Language,
    pub source: DataSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataSource {
    WikiPron,
    CmuDict,
    Phoible,
}

impl std::fmt::Display for DataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataSource::WikiPron => write!(f, "wikipron"),
            DataSource::CmuDict => write!(f, "cmu"),
            DataSource::Phoible => write!(f, "phoible"),
        }
    }
}

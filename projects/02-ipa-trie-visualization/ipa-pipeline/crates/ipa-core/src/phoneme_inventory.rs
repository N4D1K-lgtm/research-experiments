//! Phoneme inventory validation using PHOIBLE data.

use std::collections::{HashMap, HashSet};

/// A phoneme inventory for a single language, sourced from PHOIBLE.
#[derive(Debug, Clone, Default)]
pub struct PhonemeInventory {
    /// Set of valid phoneme symbols for this language.
    pub phonemes: HashSet<String>,
    /// Marginal phonemes (rare/borrowed).
    pub marginal: HashSet<String>,
}

impl PhonemeInventory {
    /// Check if a phoneme is valid for this language.
    pub fn is_valid(&self, phoneme: &str) -> bool {
        self.phonemes.contains(phoneme)
    }

    /// Check if a phoneme is marginal (valid but rare).
    pub fn is_marginal(&self, phoneme: &str) -> bool {
        self.marginal.contains(phoneme)
    }

    /// Validate a sequence of phoneme tokens. Returns issues found.
    pub fn validate_tokens(&self, tokens: &[String]) -> Vec<String> {
        let mut issues = Vec::new();
        for token in tokens {
            if !self.is_valid(token) && !self.is_marginal(token) {
                issues.push(format!("Unknown phoneme: {token}"));
            }
        }
        issues
    }
}

/// Collection of phoneme inventories for all languages.
#[derive(Debug, Clone, Default)]
pub struct InventoryStore {
    /// Language ISO 639-3 code → inventory.
    pub inventories: HashMap<String, PhonemeInventory>,
}

impl InventoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, iso639_3: &str, inventory: PhonemeInventory) {
        self.inventories.insert(iso639_3.to_string(), inventory);
    }

    pub fn get(&self, iso639_3: &str) -> Option<&PhonemeInventory> {
        self.inventories.get(iso639_3)
    }

    /// Validate pronunciation tokens for a given language.
    pub fn validate(&self, iso639_3: &str, tokens: &[String]) -> Vec<String> {
        match self.get(iso639_3) {
            Some(inv) => inv.validate_tokens(tokens),
            None => vec![format!("No inventory for language: {iso639_3}")],
        }
    }
}

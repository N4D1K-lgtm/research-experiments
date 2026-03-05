//! SurrealDB database connection and operations.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::{Db, RocksDb};
use surrealdb::types::{RecordId, SurrealValue};

use crate::schema::SCHEMA;

/// Database wrapper around SurrealDB with RocksDB backend.
pub struct Database {
    pub db: Surreal<Db>,
}

/// Record returned from create/select operations.
#[derive(Debug, Serialize, Deserialize, SurrealValue)]
pub struct Record {
    pub id: RecordId,
}

/// Word record with its ID, for building the word→ID cache.
#[derive(Debug, Deserialize, SurrealValue)]
struct WordWithId {
    id: RecordId,
    orthography: String,
}

impl Database {
    /// Open or create a database at the given path.
    pub async fn open(path: &Path) -> Result<Self> {
        let db = Surreal::new::<RocksDb>(path.to_str().unwrap()).await?;
        db.use_ns("ipa").use_db("pipeline").await?;
        Ok(Self { db })
    }

    /// Initialize the schema (idempotent).
    pub async fn init_schema(&self) -> Result<()> {
        self.db.query(SCHEMA).await?;
        tracing::info!("Schema initialized");
        Ok(())
    }

    /// Insert a language record. Returns the record ID.
    pub async fn upsert_language(&self, lang: &ipa_core::Language) -> Result<RecordId> {
        let result: Option<Record> = self
            .db
            .query("SELECT * FROM language WHERE code = $code LIMIT 1")
            .bind(("code", lang.code.clone()))
            .await?
            .take(0)?;

        if let Some(existing) = result {
            return Ok(existing.id);
        }

        let result: Option<Record> = self
            .db
            .create("language")
            .content(LanguageRecord {
                code: lang.code.clone(),
                name: lang.name.clone(),
                family: lang.family.clone(),
                typology: lang.typology.clone(),
                iso639_3: lang.iso639_3.clone(),
            })
            .await?;

        result
            .map(|r| r.id)
            .ok_or_else(|| anyhow::anyhow!("Failed to create language record"))
    }

    /// Delete all pronunciations for a language.
    /// Call before re-ingesting to avoid duplicates on partial re-runs.
    pub async fn delete_pronunciations_for_language(
        &self,
        language_id: &RecordId,
    ) -> Result<()> {
        self.db
            .query("DELETE pronunciation WHERE word.language = $lang")
            .bind(("lang", language_id.clone()))
            .await?;
        Ok(())
    }

    /// Bulk insert all pronunciations for a language.
    ///
    /// Deletes existing pronunciations first to handle partial re-runs,
    /// then uses batch word upserts and bulk pronunciation inserts to minimize
    /// DB round-trips (from O(N*3) to O(chunks)).
    pub async fn bulk_insert_for_language(
        &self,
        language_id: &RecordId,
        entries: &[PronunciationInsert],
    ) -> Result<usize> {
        if entries.is_empty() {
            return Ok(0);
        }

        // 0. Delete any existing pronunciations for this language (idempotent re-runs)
        self.delete_pronunciations_for_language(language_id).await?;

        // 1. Collect unique words
        let unique_words: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            entries
                .iter()
                .filter(|e| seen.insert(e.word.clone()))
                .map(|e| e.word.clone())
                .collect()
        };

        // 2. Bulk upsert words using INSERT ... ON DUPLICATE KEY UPDATE
        //    Process in chunks to avoid overly large queries.
        const WORD_CHUNK: usize = 1000;
        for chunk in unique_words.chunks(WORD_CHUNK) {
            let mut query_str = String::from("INSERT INTO word [");
            for (i, _word) in chunk.iter().enumerate() {
                if i > 0 {
                    query_str.push(',');
                }
                query_str.push_str(&format!(
                    "{{ orthography: $w{i}, language: $lang }}"
                ));
            }
            query_str.push_str("] ON DUPLICATE KEY UPDATE id = id;");

            let mut q = self.db.query(&query_str).bind(("lang", language_id.clone()));
            for (i, word) in chunk.iter().enumerate() {
                q = q.bind((format!("w{i}"), word.clone()));
            }
            q.await?;
        }

        // 3. Fetch all word IDs for this language in one SELECT
        let word_records: Vec<WordWithId> = self
            .db
            .query("SELECT id, orthography FROM word WHERE language = $lang")
            .bind(("lang", language_id.clone()))
            .await?
            .take(0)?;

        let word_map: HashMap<String, RecordId> = word_records
            .into_iter()
            .map(|w| (w.orthography, w.id))
            .collect();

        // 4. Bulk insert pronunciations
        const PRON_CHUNK: usize = 5000;
        let mut total = 0;
        for chunk in entries.chunks(PRON_CHUNK) {
            let records: Vec<PronunciationRecord> = chunk
                .iter()
                .filter_map(|e| {
                    word_map.get(&e.word).map(|wid| PronunciationRecord {
                        word: wid.clone(),
                        ipa_raw: e.ipa_raw.clone(),
                        tokens: e.tokens.clone(),
                        normalized: e.normalized.clone(),
                        source: e.source.clone(),
                        valid: e.valid,
                        issues: e.issues.clone(),
                    })
                })
                .collect();
            let count = records.len();
            let _: Vec<Record> = self
                .db
                .insert("pronunciation")
                .content(records)
                .await?;
            total += count;
        }

        Ok(total)
    }
}

/// A pronunciation entry ready to insert into the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PronunciationInsert {
    pub word: String,
    pub ipa_raw: String,
    pub tokens: Vec<String>,
    pub normalized: Vec<String>,
    pub source: String,
    pub valid: bool,
    pub issues: Vec<String>,
    pub language_id: RecordId,
}

#[derive(Debug, Serialize, Deserialize, SurrealValue)]
struct LanguageRecord {
    code: String,
    name: String,
    family: String,
    typology: String,
    iso639_3: String,
}

#[derive(Debug, Serialize, Deserialize, SurrealValue)]
struct WordRecord {
    orthography: String,
    language: RecordId,
}

#[derive(Debug, Serialize, Deserialize, SurrealValue)]
struct PronunciationRecord {
    word: RecordId,
    ipa_raw: String,
    tokens: Vec<String>,
    normalized: Vec<String>,
    source: String,
    valid: bool,
    issues: Vec<String>,
}

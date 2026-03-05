//! Common database queries.

use anyhow::Result;
use serde::Deserialize;
use surrealdb::types::SurrealValue;

use crate::Database;

/// Pronunciation record from the database.
#[derive(Debug, Clone, Deserialize, SurrealValue)]
pub struct PronunciationQueryResult {
    pub word: String,
    pub normalized: Vec<String>,
}

/// Language count result.
#[derive(Debug, Deserialize, SurrealValue)]
pub struct LanguageCount {
    pub lang_code: String,
    pub total: u64,
}

#[derive(Debug, Deserialize, SurrealValue)]
struct CountResult {
    total: u64,
}

impl Database {
    /// Get all valid, normalized pronunciations for a language.
    pub async fn get_pronunciations_for_language(
        &self,
        lang_code: &str,
    ) -> Result<Vec<(String, Vec<String>)>> {
        let results: Vec<PronunciationQueryResult> = self
            .db
            .query(
                r#"
                SELECT
                    word.orthography AS word,
                    normalized
                FROM pronunciation
                WHERE valid = true
                    AND word.language.code = $lang_code
                ORDER BY word.orthography
                "#,
            )
            .bind(("lang_code", lang_code.to_string()))
            .await?
            .take(0)?;

        Ok(results
            .into_iter()
            .map(|r| (r.word, r.normalized))
            .collect())
    }

    /// Get count of pronunciations per language.
    pub async fn count_by_language(&self) -> Result<Vec<LanguageCount>> {
        let results: Vec<LanguageCount> = self
            .db
            .query(
                r#"
                SELECT
                    word.language.code AS lang_code,
                    count() AS total
                FROM pronunciation
                WHERE valid = true
                GROUP BY word.language.code
                "#,
            )
            .await?
            .take(0)?;

        Ok(results)
    }

    /// Get count of invalid pronunciations with issues.
    pub async fn count_invalid(&self) -> Result<u64> {
        let result: Option<CountResult> = self
            .db
            .query("SELECT count() AS total FROM pronunciation WHERE valid = false")
            .await?
            .take(0)?;

        Ok(result.map_or(0, |r| r.total))
    }
}

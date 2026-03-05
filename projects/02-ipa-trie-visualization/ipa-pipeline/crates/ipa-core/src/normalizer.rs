//! IPA normalization: maps surface phonetic forms to underlying phonemic representations
//! by collapsing allophonic variation.

use std::collections::HashMap;

/// Suprasegmental symbols to strip (Rule 1).
const PROSODIC_STRIP: &[char] = &['ˈ', 'ˌ', '.', '‿', '|', '‖'];

/// Allophonic combining diacritics to strip (Rule 2).
/// These are predictable from context and don't contribute to phonemic identity.
const ALLOPHONIC_DIACRITICS: &[char] = &[
    '\u{0329}', // ̩ syllabic
    '\u{032F}', // ̯ non-syllabic
    '\u{0325}', // ̥ voiceless
    '\u{031F}', // ̟ advanced tongue root
    '\u{0320}', // ̠ retracted tongue root
];

/// Phonemic features to KEEP (Rule 3).
/// These are contrastive and must be preserved.
const _PHONEMIC_FEATURES: &[char] = &[
    '\u{0303}', // ̃ nasalization
    'ː',        // length
    'ʰ',        // aspiration
    'ʷ',        // labialization
    'ʲ',        // palatalization
];

/// Language-specific allophone mappings (Rule 4).
fn allophone_map(lang: &str) -> HashMap<&'static str, &'static str> {
    match lang {
        "en_US" => HashMap::from([
            ("ɫ", "l"),  // Dark L (coda) → /l/
            ("ɾ", "t"),  // Alveolar flap (intervocalic) → /t/
        ]),
        "es_ES" => HashMap::from([
            ("β", "b"),  // Intervocalic lenition → /b/
            ("ð", "d"),  // Intervocalic lenition → /d/
            ("ɣ", "ɡ"),  // Intervocalic lenition → /g/
        ]),
        "de" | "fr_FR" | "nl" | "cmn" | "jpn" | "ara" | "fin" | "tur" | "hin" | "swa" => {
            HashMap::new()
        }
        _ => HashMap::new(),
    }
}

/// Normalize a single IPA token.
///
/// Returns `None` if the token is a standalone prosodic marker (should be dropped).
/// Returns `Some(normalized)` with the phonemic representation.
///
/// # Rules applied in order:
/// 1. Strip standalone prosodic features (ˈ, ˌ, ., ‿, |, ‖) → None
/// 2. Apply language-specific allophone rules (exact match)
/// 3. Strip embedded stress marks
/// 4. Strip allophonic combining diacritics (̩, ̯, ̥, ̟, ̠)
/// 5. Re-check allophone map after stripping
pub fn normalize_token(token: &str, lang: &str) -> Option<String> {
    // Rule 1: standalone prosodic features → drop entirely
    if token.len() <= 4 && token.chars().all(|c| PROSODIC_STRIP.contains(&c)) {
        return None;
    }

    let allophones = allophone_map(lang);

    // Rule 4 (first pass): exact allophone match
    if let Some(&replacement) = allophones.get(token) {
        return Some(replacement.to_string());
    }

    // Rule 3 (strip embedded stress marks) + Rule 2 (strip allophonic diacritics)
    let mut result = String::with_capacity(token.len());
    for c in token.chars() {
        // Skip embedded stress marks
        if PROSODIC_STRIP.contains(&c) {
            continue;
        }
        // Skip allophonic diacritics
        if ALLOPHONIC_DIACRITICS.contains(&c) {
            continue;
        }
        result.push(c);
    }

    if result.is_empty() {
        return None;
    }

    // Rule 4 (second pass): re-check after stripping diacritics
    if let Some(&replacement) = allophones.get(result.as_str()) {
        return Some(replacement.to_string());
    }

    Some(result)
}

/// Normalize a sequence of IPA tokens for a given language.
///
/// Filters out prosodic markers and applies all normalization rules.
pub fn normalize(tokens: &[String], lang: &str) -> Vec<String> {
    tokens
        .iter()
        .filter_map(|t| normalize_token(t, lang))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_stress() {
        assert_eq!(normalize_token("ˈ", "en_US"), None);
        assert_eq!(normalize_token("ˌ", "en_US"), None);
    }

    #[test]
    fn test_strip_syllable_boundary() {
        assert_eq!(normalize_token(".", "en_US"), None);
    }

    #[test]
    fn test_keep_nasalization() {
        assert_eq!(normalize_token("õ", "fr_FR"), Some("õ".to_string()));
    }

    #[test]
    fn test_keep_length() {
        assert_eq!(normalize_token("iː", "de"), Some("iː".to_string()));
    }

    #[test]
    fn test_keep_aspiration() {
        assert_eq!(normalize_token("pʰ", "en_US"), Some("pʰ".to_string()));
    }

    #[test]
    fn test_strip_syllabic() {
        // n̩ → n
        assert_eq!(normalize_token("n\u{0329}", "en_US"), Some("n".to_string()));
    }

    #[test]
    fn test_strip_voiceless_sonorant() {
        // n̥ → n
        assert_eq!(normalize_token("n\u{0325}", "en_US"), Some("n".to_string()));
    }

    #[test]
    fn test_english_dark_l() {
        assert_eq!(normalize_token("ɫ", "en_US"), Some("l".to_string()));
    }

    #[test]
    fn test_english_flap() {
        assert_eq!(normalize_token("ɾ", "en_US"), Some("t".to_string()));
    }

    #[test]
    fn test_spanish_lenition() {
        assert_eq!(normalize_token("β", "es_ES"), Some("b".to_string()));
        assert_eq!(normalize_token("ð", "es_ES"), Some("d".to_string()));
        assert_eq!(normalize_token("ɣ", "es_ES"), Some("ɡ".to_string()));
    }

    #[test]
    fn test_no_allophone_for_german() {
        assert_eq!(normalize_token("ɾ", "de"), Some("ɾ".to_string()));
    }

    #[test]
    fn test_full_normalize_sequence() {
        let tokens: Vec<String> = vec!["ˈ", "ɪ", "ŋ", "ɡ", "l", "ɪ", "ʃ"]
            .into_iter()
            .map(String::from)
            .collect();
        let result = normalize(&tokens, "en_US");
        assert_eq!(result, vec!["ɪ", "ŋ", "ɡ", "l", "ɪ", "ʃ"]);
    }

    #[test]
    fn test_plain_consonant_unchanged() {
        assert_eq!(normalize_token("p", "en_US"), Some("p".to_string()));
        assert_eq!(normalize_token("t", "en_US"), Some("t".to_string()));
        assert_eq!(normalize_token("k", "en_US"), Some("k".to_string()));
    }

    #[test]
    fn test_plain_vowel_unchanged() {
        assert_eq!(normalize_token("a", "en_US"), Some("a".to_string()));
        assert_eq!(normalize_token("ə", "en_US"), Some("ə".to_string()));
    }
}

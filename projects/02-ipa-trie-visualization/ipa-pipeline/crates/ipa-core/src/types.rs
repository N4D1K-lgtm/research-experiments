use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A supported language with metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Language {
    /// Internal code: "en_US", "fr_FR", etc.
    pub code: String,
    /// Display name: "English", "French", etc.
    pub name: String,
    /// Language family: "Germanic", "Romance", etc.
    pub family: String,
    /// Morphological typology: "Fusional", "Agglutinative", etc.
    pub typology: String,
    /// ISO 639-3 code: "eng", "fra", etc.
    pub iso639_3: String,
}

impl Language {
    pub fn all() -> Vec<Language> {
        vec![
            Language {
                code: "en_US".into(),
                name: "English".into(),
                family: "Germanic".into(),
                typology: "Fusional".into(),
                iso639_3: "eng".into(),
            },
            Language {
                code: "fr_FR".into(),
                name: "French".into(),
                family: "Romance".into(),
                typology: "Fusional".into(),
                iso639_3: "fra".into(),
            },
            Language {
                code: "es_ES".into(),
                name: "Spanish".into(),
                family: "Romance".into(),
                typology: "Fusional".into(),
                iso639_3: "spa".into(),
            },
            Language {
                code: "de".into(),
                name: "German".into(),
                family: "Germanic".into(),
                typology: "Fusional".into(),
                iso639_3: "deu".into(),
            },
            Language {
                code: "nl".into(),
                name: "Dutch".into(),
                family: "Germanic".into(),
                typology: "Fusional".into(),
                iso639_3: "nld".into(),
            },
            Language {
                code: "cmn".into(),
                name: "Mandarin Chinese".into(),
                family: "Sino-Tibetan".into(),
                typology: "Isolating".into(),
                iso639_3: "cmn".into(),
            },
            Language {
                code: "jpn".into(),
                name: "Japanese".into(),
                family: "Japonic".into(),
                typology: "Agglutinative".into(),
                iso639_3: "jpn".into(),
            },
            Language {
                code: "ara".into(),
                name: "Arabic".into(),
                family: "Semitic".into(),
                typology: "Fusional".into(),
                iso639_3: "arb".into(),
            },
            Language {
                code: "fin".into(),
                name: "Finnish".into(),
                family: "Uralic".into(),
                typology: "Agglutinative".into(),
                iso639_3: "fin".into(),
            },
            Language {
                code: "tur".into(),
                name: "Turkish".into(),
                family: "Turkic".into(),
                typology: "Agglutinative".into(),
                iso639_3: "tur".into(),
            },
            Language {
                code: "hin".into(),
                name: "Hindi".into(),
                family: "Indo-Aryan".into(),
                typology: "Fusional".into(),
                iso639_3: "hin".into(),
            },
            Language {
                code: "swa".into(),
                name: "Swahili".into(),
                family: "Bantu".into(),
                typology: "Agglutinative".into(),
                iso639_3: "swh".into(),
            },
        ]
    }

    pub fn by_code(code: &str) -> Option<Language> {
        Self::all().into_iter().find(|l| l.code == code)
    }

    pub fn by_iso639_3(iso: &str) -> Option<Language> {
        Self::all().into_iter().find(|l| l.iso639_3 == iso)
    }

    /// Maps WikiPron filename prefix (e.g., "eng_latn_us") to our language code.
    pub fn from_wikipron_prefix(prefix: &str) -> Option<Language> {
        let mapping: HashMap<&str, &str> = HashMap::from([
            ("eng_latn_us", "en_US"),
            ("fra_latn", "fr_FR"),
            ("spa_latn_la", "es_ES"),
            ("spa_latn_ca", "es_ES"),
            ("deu_latn", "de"),
            ("nld_latn", "nl"),
            ("zho_hani", "cmn"),
            ("jpn_hira", "jpn"),
            ("ara_arab", "ara"),
            ("fin_latn", "fin"),
            ("tur_latn", "tur"),
            ("hin_deva", "hin"),
            ("swa_latn", "swa"),
        ]);

        mapping
            .get(prefix)
            .and_then(|code| Self::by_code(code))
    }
}

/// Phonological position of a phoneme in syllable structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PhonologicalPosition {
    Onset,
    Nucleus,
    Coda,
    Mixed,
}

impl PhonologicalPosition {
    pub fn short_code(&self) -> &'static str {
        match self {
            PhonologicalPosition::Onset => "o",
            PhonologicalPosition::Nucleus => "n",
            PhonologicalPosition::Coda => "c",
            PhonologicalPosition::Mixed => "m",
        }
    }
}

impl std::fmt::Display for PhonologicalPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhonologicalPosition::Onset => write!(f, "onset"),
            PhonologicalPosition::Nucleus => write!(f, "nucleus"),
            PhonologicalPosition::Coda => write!(f, "coda"),
            PhonologicalPosition::Mixed => write!(f, "mixed"),
        }
    }
}

/// IPA vowel base characters (without diacritics).
pub const IPA_VOWELS: &[char] = &[
    'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
    'ɪ', 'ʏ', 'ʊ',
    'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
    'ə',
    'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
    'æ', 'ɐ',
    'a', 'ɶ', 'ɑ', 'ɒ',
];

/// Check if an IPA character is a vowel (base character, ignoring diacritics).
pub fn is_vowel(c: char) -> bool {
    IPA_VOWELS.contains(&c)
}

/// Check if the first character of a phoneme token is a vowel.
pub fn token_is_vowel(token: &str) -> bool {
    token.chars().next().is_some_and(|c| is_vowel(c))
}

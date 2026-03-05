//! IPA tokenizer: greedy left-to-right parser that handles combining diacritics,
//! IPA modifiers, tie bars, and suprasegmentals.

use unicode_normalization::UnicodeNormalization;

/// Suprasegmental symbols that are tokenized as standalone units.
const SUPRASEGMENTALS: &[char] = &['ˈ', 'ˌ', '.', '‿', '|', '‖'];

/// Modifier characters that attach to the preceding base character.
/// Note: ˈ and ˌ are suprasegmentals, NOT modifiers.
const MODIFIERS: &[char] = &[
    'ː', 'ˑ', 'ʰ', 'ʷ', 'ʲ', 'ˠ', 'ˤ', 'ⁿ', 'ˡ', 'ʼ', '˞',
];

/// Tie bar characters (U+0361, U+035C) that join two base characters.
const TIE_BARS: &[char] = &['\u{0361}', '\u{035C}'];

/// Check if a character is a combining mark (Unicode categories Mn, Mc, Me).
fn is_combining(c: char) -> bool {
    matches!(
        unicode_general_category::get_general_category(c),
        unicode_general_category::GeneralCategory::NonspacingMark
            | unicode_general_category::GeneralCategory::SpacingMark
            | unicode_general_category::GeneralCategory::EnclosingMark
    )
}

/// Check if a character is a suprasegmental.
fn is_suprasegmental(c: char) -> bool {
    SUPRASEGMENTALS.contains(&c)
}

/// Check if a character is a tie bar.
fn is_tie_bar(c: char) -> bool {
    TIE_BARS.contains(&c)
}

/// Tokenize an IPA transcription string into individual phoneme tokens.
///
/// Handles:
/// - Stripping of `/`, `[`, `]` wrappers
/// - Combining diacritics (U+0300–U+036F and others)
/// - IPA modifiers (ʰ, ʷ, ʲ, ː, etc.)
/// - Tie bars (U+0361, U+035C) joining two characters
/// - Suprasegmentals (ˈ, ˌ, ., ‿, |, ‖) as separate tokens
///
/// # Examples
/// ```
/// use ipa_core::tokenize_ipa;
/// let tokens = tokenize_ipa("/ˈɪŋɡlɪʃ/");
/// assert_eq!(tokens, vec!["ˈ", "ɪ", "ŋ", "ɡ", "l", "ɪ", "ʃ"]);
/// ```
pub fn tokenize_ipa(transcription: &str) -> Vec<String> {
    // NFC normalize first
    let input: String = transcription.nfc().collect();

    // Strip wrappers
    let input = input.trim();
    let input = input
        .strip_prefix('/')
        .or_else(|| input.strip_prefix('['))
        .unwrap_or(input);
    let input = input
        .strip_suffix('/')
        .or_else(|| input.strip_suffix(']'))
        .unwrap_or(input);
    let input = input.trim();

    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < len {
        let c = chars[i];

        // Skip whitespace
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // Suprasegmentals become standalone tokens
        if is_suprasegmental(c) {
            tokens.push(c.to_string());
            i += 1;
            continue;
        }

        // Start a new token with this base character
        let mut token = String::new();
        token.push(c);
        i += 1;

        // Greedily collect modifiers, combining diacritics, and tie-bar sequences
        while i < len {
            let next = chars[i];

            if is_tie_bar(next) {
                // Tie bar: consume it and the following character
                token.push(next);
                i += 1;
                if i < len {
                    token.push(chars[i]);
                    i += 1;
                }
            } else if is_combining(next) || MODIFIERS.contains(&next) {
                // Combining mark or modifier: attach to current token
                token.push(next);
                i += 1;
            } else {
                break;
            }
        }

        tokens.push(token);
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        assert_eq!(tokenize_ipa("pæt"), vec!["p", "æ", "t"]);
    }

    #[test]
    fn test_strip_slashes() {
        assert_eq!(tokenize_ipa("/pæt/"), vec!["p", "æ", "t"]);
    }

    #[test]
    fn test_strip_brackets() {
        assert_eq!(tokenize_ipa("[pæt]"), vec!["p", "æ", "t"]);
    }

    #[test]
    fn test_stress_marks() {
        assert_eq!(
            tokenize_ipa("/ˈɪŋɡlɪʃ/"),
            vec!["ˈ", "ɪ", "ŋ", "ɡ", "l", "ɪ", "ʃ"]
        );
    }

    #[test]
    fn test_aspiration() {
        assert_eq!(tokenize_ipa("pʰæt"), vec!["pʰ", "æ", "t"]);
    }

    #[test]
    fn test_long_vowel() {
        assert_eq!(tokenize_ipa("biːt"), vec!["b", "iː", "t"]);
    }

    #[test]
    fn test_nasalized_vowel() {
        assert_eq!(tokenize_ipa("bõ"), vec!["b", "õ"]);
    }

    #[test]
    fn test_tie_bar() {
        // t͡ʃ (voiceless postalveolar affricate)
        assert_eq!(tokenize_ipa("t͡ʃ"), vec!["t͡ʃ"]);
    }

    #[test]
    fn test_tie_bar_with_context() {
        assert_eq!(tokenize_ipa("t͡ʃæt"), vec!["t͡ʃ", "æ", "t"]);
    }

    #[test]
    fn test_syllabic_mark() {
        // n̩ (syllabic n, U+0329)
        assert_eq!(tokenize_ipa("bʌtn̩"), vec!["b", "ʌ", "t", "n̩"]);
    }

    #[test]
    fn test_multiple_diacritics() {
        // ãː (nasalized long vowel)
        assert_eq!(tokenize_ipa("ãː"), vec!["ãː"]);
    }

    #[test]
    fn test_labialization() {
        assert_eq!(tokenize_ipa("kʷa"), vec!["kʷ", "a"]);
    }

    #[test]
    fn test_palatalization() {
        assert_eq!(tokenize_ipa("nʲa"), vec!["nʲ", "a"]);
    }

    #[test]
    fn test_whitespace_ignored() {
        assert_eq!(tokenize_ipa("p æ t"), vec!["p", "æ", "t"]);
    }

    #[test]
    fn test_empty_input() {
        let result: Vec<String> = tokenize_ipa("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_secondary_stress() {
        assert_eq!(
            tokenize_ipa("ˌɪntəˈnæʃənəl"),
            vec!["ˌ", "ɪ", "n", "t", "ə", "ˈ", "n", "æ", "ʃ", "ə", "n", "ə", "l"]
        );
    }

    #[test]
    fn test_ejective() {
        assert_eq!(tokenize_ipa("pʼa"), vec!["pʼ", "a"]);
    }

    #[test]
    fn test_voiceless_sonorant() {
        // n̥ (voiceless n, U+0325)
        assert_eq!(tokenize_ipa("n̥a"), vec!["n̥", "a"]);
    }

    #[test]
    fn test_syllable_boundaries() {
        assert_eq!(
            tokenize_ipa("ˈsɪl.ə.bəl"),
            vec!["ˈ", "s", "ɪ", "l", ".", "ə", ".", "b", "ə", "l"]
        );
    }
}

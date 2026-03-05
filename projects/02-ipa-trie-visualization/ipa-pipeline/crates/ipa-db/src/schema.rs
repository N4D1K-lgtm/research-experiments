//! SurrealDB schema definitions and migrations.

/// SQL schema for the IPA pipeline database.
pub const SCHEMA: &str = r#"
-- Core tables
DEFINE TABLE language SCHEMAFULL;
DEFINE FIELD code       ON language TYPE string;
DEFINE FIELD name       ON language TYPE string;
DEFINE FIELD family     ON language TYPE string;
DEFINE FIELD typology   ON language TYPE string;
DEFINE FIELD iso639_3   ON language TYPE string;
DEFINE INDEX idx_lang_code ON language FIELDS code UNIQUE;

DEFINE TABLE phoneme SCHEMAFULL;
DEFINE FIELD symbol     ON phoneme TYPE string;
DEFINE FIELD features   ON phoneme TYPE option<object>;
DEFINE FIELD seg_class  ON phoneme TYPE option<string>;
DEFINE INDEX idx_phoneme_sym ON phoneme FIELDS symbol UNIQUE;

DEFINE TABLE inventory SCHEMAFULL;
DEFINE FIELD language   ON inventory TYPE record<language>;
DEFINE FIELD phoneme    ON inventory TYPE record<phoneme>;
DEFINE FIELD marginal   ON inventory TYPE bool DEFAULT false;
DEFINE FIELD source     ON inventory TYPE string;

DEFINE TABLE word SCHEMAFULL;
DEFINE FIELD orthography ON word TYPE string;
DEFINE FIELD language    ON word TYPE record<language>;
DEFINE INDEX idx_word_orth_lang ON word FIELDS orthography, language UNIQUE;

DEFINE TABLE pronunciation SCHEMAFULL;
DEFINE FIELD word       ON pronunciation TYPE record<word>;
DEFINE FIELD ipa_raw    ON pronunciation TYPE string;
DEFINE FIELD tokens     ON pronunciation TYPE array;
DEFINE FIELD normalized ON pronunciation TYPE array;
DEFINE FIELD source     ON pronunciation TYPE string;
DEFINE FIELD valid      ON pronunciation TYPE bool DEFAULT true;
DEFINE FIELD issues     ON pronunciation TYPE array DEFAULT [];

-- Trie structure
DEFINE TABLE trie_node SCHEMAFULL;
DEFINE FIELD phoneme     ON trie_node TYPE string;
DEFINE FIELD depth       ON trie_node TYPE int;
DEFINE FIELD is_terminal ON trie_node TYPE bool DEFAULT false;
DEFINE FIELD role        ON trie_node TYPE string;
DEFINE FIELD counts      ON trie_node TYPE option<object>;
DEFINE FIELD total_count ON trie_node TYPE int DEFAULT 0;
DEFINE FIELD position    ON trie_node TYPE option<object>;
DEFINE FIELD color       ON trie_node TYPE option<string>;

DEFINE TABLE child_of TYPE RELATION IN trie_node OUT trie_node ENFORCED;
DEFINE FIELD phoneme      ON child_of TYPE string;
DEFINE FIELD prob          ON child_of TYPE float;

-- Analysis tables
DEFINE TABLE depth_stats SCHEMAFULL;
DEFINE FIELD language    ON depth_stats TYPE record<language>;
DEFINE FIELD depth       ON depth_stats TYPE int;
DEFINE FIELD nodes       ON depth_stats TYPE int;
DEFINE FIELD terminals   ON depth_stats TYPE int;
DEFINE FIELD avg_branch  ON depth_stats TYPE float;
DEFINE FIELD avg_entropy ON depth_stats TYPE float;
DEFINE FIELD max_entropy ON depth_stats TYPE float;

DEFINE TABLE motif SCHEMAFULL;
DEFINE FIELD sequence    ON motif TYPE array;
DEFINE FIELD label       ON motif TYPE string;
DEFINE FIELD count       ON motif TYPE int;
DEFINE FIELD language    ON motif TYPE record<language>;
"#;

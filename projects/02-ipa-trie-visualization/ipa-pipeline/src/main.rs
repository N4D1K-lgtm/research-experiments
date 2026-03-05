use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

use ipa_core::{tokenize_ipa, normalize, Language};
use ipa_db::Database;
use ipa_db::store::PronunciationInsert;
use ipa_ingest::cmu::CmuDictSource;
use ipa_ingest::phoible::PhoibleSource;
use ipa_ingest::wikipron::WikiPronSource;
use ipa_trie::{ConeTreeLayout, PhonologicalTrie, TrieAnalysis};

#[derive(Parser)]
#[command(name = "ipa-pipeline")]
#[command(about = "IPA Trie Visualization data pipeline")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Data directory for downloads and cache
    #[arg(long, default_value = "data", global = true)]
    data_dir: PathBuf,

    /// Database path
    #[arg(long, default_value = "data/ipa.db", global = true)]
    db_path: PathBuf,
}

#[derive(Subcommand)]
enum Commands {
    /// Download data sources (WikiPron, CMU Dict, PHOIBLE)
    Download {
        /// Language codes to download (comma-separated, or "all")
        #[arg(long, default_value = "all")]
        lang: String,
        /// Data source to download
        #[arg(long, default_value = "all")]
        source: String,
    },
    /// Ingest downloaded data into database
    Ingest {
        #[arg(long, default_value = "all")]
        lang: String,
        /// Validate against PHOIBLE inventories
        #[arg(long)]
        validate: bool,
    },
    /// Build trie structures from ingested data
    Build {
        #[arg(long, default_value = "all")]
        lang: String,
        /// Minimum count threshold for pruning
        #[arg(long, default_value = "2")]
        min_count: u64,
    },
    /// Run analysis on built tries
    Analyze {
        /// Run cross-linguistic comparison
        #[arg(long)]
        cross_linguistic: bool,
        /// Detect phonological motifs
        #[arg(long)]
        motifs: bool,
        /// Compute entropy statistics
        #[arg(long)]
        entropy: bool,
    },
    /// Export data to JSON for web frontend
    Export {
        /// Export format (trie, essay, stats, all)
        #[arg(long, default_value = "all")]
        format: String,
        /// Output directory
        #[arg(long, default_value = "output")]
        output: String,
        /// Language codes (comma-separated, or "all")
        #[arg(long, default_value = "all")]
        lang: String,
        /// Minimum count threshold for pruning
        #[arg(long, default_value = "2")]
        min_count: u64,
        /// Max depth for essay data export
        #[arg(long, default_value = "6")]
        max_depth: u32,
    },
    /// Validate ingested data against phonological inventories
    Validate {
        #[arg(long, default_value = "all")]
        lang: String,
        /// Generate validation report
        #[arg(long)]
        report: bool,
    },
    /// Show pipeline statistics
    Stats {
        #[arg(long, default_value = "all")]
        lang: String,
    },
}

fn resolve_languages(lang_arg: &str) -> Vec<Language> {
    if lang_arg == "all" {
        Language::all()
    } else {
        lang_arg
            .split(',')
            .filter_map(|code| {
                let code = code.trim();
                Language::by_code(code).or_else(|| {
                    tracing::warn!("Unknown language code: {code}");
                    None
                })
            })
            .collect()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Download { lang, source } => {
            cmd_download(&cli.data_dir, &lang, &source).await
        }
        Commands::Ingest { lang, validate } => {
            cmd_ingest(&cli.data_dir, &cli.db_path, &lang, validate).await
        }
        Commands::Build { lang, min_count } => {
            cmd_build(&cli.db_path, &lang, min_count).await
        }
        Commands::Analyze {
            cross_linguistic,
            motifs,
            entropy,
        } => cmd_analyze(&cli.db_path, cross_linguistic, motifs, entropy).await,
        Commands::Export {
            format,
            output,
            lang,
            min_count,
            max_depth,
        } => cmd_export(&cli.db_path, &format, &output, &lang, min_count, max_depth).await,
        Commands::Validate { lang, report } => {
            cmd_validate(&cli.data_dir, &cli.db_path, &lang, report).await
        }
        Commands::Stats { lang } => cmd_stats(&cli.db_path, &lang).await,
    }
}

// ─── Download ────────────────────────────────────────────────────────────────

async fn cmd_download(data_dir: &Path, lang_arg: &str, source: &str) -> Result<()> {
    let languages = resolve_languages(lang_arg);
    tracing::info!("Downloading data for {} languages", languages.len());

    let do_all = source == "all";

    if do_all || source == "wikipron" {
        let wp = WikiPronSource::new(data_dir);
        let paths = wp.download(&languages).await?;
        tracing::info!("WikiPron: downloaded {} files", paths.len());
    }

    if do_all || source == "cmu" {
        let cmu = CmuDictSource::new(data_dir);
        let path = cmu.download().await?;
        tracing::info!("CMU Dict: {}", path.display());
    }

    if do_all || source == "phoible" {
        let phoible = PhoibleSource::new(data_dir);
        let path = phoible.download().await?;
        tracing::info!("PHOIBLE: {}", path.display());
    }

    tracing::info!("Download complete.");
    Ok(())
}

// ─── Ingest ──────────────────────────────────────────────────────────────────

async fn cmd_ingest(
    data_dir: &Path,
    db_path: &Path,
    lang_arg: &str,
    validate: bool,
) -> Result<()> {
    let languages = resolve_languages(lang_arg);
    let db = Database::open(db_path).await?;
    db.init_schema().await?;

    // Load PHOIBLE inventories if validating
    let inventory_store = if validate {
        let phoible = PhoibleSource::new(data_dir);
        let phoible_path = data_dir.join("phoible").join("phoible.csv");
        if phoible_path.exists() {
            let iso_codes: Vec<&str> = languages.iter().map(|l| l.iso639_3.as_str()).collect();
            let store = phoible.parse(&phoible_path, &iso_codes)?;
            tracing::info!(
                "Loaded PHOIBLE inventories for {} languages",
                store.inventories.len()
            );
            Some(store)
        } else {
            tracing::warn!("PHOIBLE data not found; run `download --source phoible` first");
            None
        }
    } else {
        None
    };

    for lang in &languages {
        // Upsert language record
        let lang_id = db.upsert_language(lang).await?;
        tracing::info!("Language: {} ({})", lang.name, lang.code);

        let mut all_entries: Vec<PronunciationInsert> = Vec::new();

        // CMU Dict (English only)
        if lang.code == "en_US" {
            let cmu_path = data_dir.join("cmu").join("cmudict-0.7b-ipa.txt");
            if cmu_path.exists() {
                let cmu = CmuDictSource::new(data_dir);
                let raw = cmu.parse(&cmu_path)?;
                tracing::info!("  CMU Dict: {} entries", raw.len());
                for entry in &raw {
                    let tokens = tokenize_ipa(&entry.ipa);
                    let normalized = normalize(&tokens, &lang.code);
                    let issues = if let Some(ref store) = inventory_store {
                        store.validate(&lang.iso639_3, &normalized)
                    } else {
                        vec![]
                    };
                    let valid = issues.is_empty();
                    all_entries.push(PronunciationInsert {
                        word: entry.word.clone(),
                        ipa_raw: entry.ipa.clone(),
                        tokens,
                        normalized,
                        source: "cmu".to_string(),
                        valid,
                        issues,
                        language_id: lang_id.clone(),
                    });
                }
            }
        }

        // WikiPron
        let wp = WikiPronSource::new(data_dir);
        let wp_dir = data_dir.join("wikipron");
        if wp_dir.exists() {
            // Find the matching file
            for entry in std::fs::read_dir(&wp_dir)? {
                let entry = entry?;
                let filename = entry.file_name().to_string_lossy().to_string();
                let prefix = if let Some(p) = filename.strip_suffix("_broad.tsv") {
                    p
                } else if let Some(p) = filename.strip_suffix("_narrow.tsv") {
                    p
                } else {
                    continue;
                };
                if let Some(matched_lang) = Language::from_wikipron_prefix(prefix) {
                    if matched_lang.code != lang.code {
                        continue;
                    }
                    let raw = wp.parse(&entry.path(), lang)?;
                    tracing::info!("  WikiPron: {} entries", raw.len());
                    for raw_entry in &raw {
                        let tokens = tokenize_ipa(&raw_entry.ipa);
                        let normalized = normalize(&tokens, &lang.code);
                        let issues = if let Some(ref store) = inventory_store {
                            store.validate(&lang.iso639_3, &normalized)
                        } else {
                            vec![]
                        };
                        let valid = issues.is_empty();
                        all_entries.push(PronunciationInsert {
                            word: raw_entry.word.clone(),
                            ipa_raw: raw_entry.ipa.clone(),
                            tokens,
                            normalized,
                            source: "wikipron".to_string(),
                            valid,
                            issues,
                            language_id: lang_id.clone(),
                        });
                    }
                }
            }
        }

        if all_entries.is_empty() {
            tracing::warn!("  No data found for {}", lang.code);
            continue;
        }

        tracing::info!(
            "  Bulk inserting {} pronunciations for {}...",
            all_entries.len(),
            lang.code
        );
        let total_inserted = db
            .bulk_insert_for_language(&lang_id, &all_entries)
            .await?;
        tracing::info!("  {}: {} pronunciations inserted", lang.code, total_inserted);
    }

    tracing::info!("Ingestion complete.");
    Ok(())
}

// ─── Build ───────────────────────────────────────────────────────────────────

async fn cmd_build(db_path: &Path, lang_arg: &str, min_count: u64) -> Result<()> {
    let languages = resolve_languages(lang_arg);
    let db = Database::open(db_path).await?;

    let mut trie = PhonologicalTrie::new();

    for lang in &languages {
        let pronunciations = db.get_pronunciations_for_language(&lang.code).await?;
        tracing::info!("{}: {} pronunciations", lang.code, pronunciations.len());

        for (word, phonemes) in &pronunciations {
            trie.insert(phonemes, &lang.code, word);
        }
    }

    tracing::info!("Raw trie: {} nodes", count_nodes(&trie));

    if min_count > 1 {
        trie.prune(min_count);
        tracing::info!("After pruning (min_count={}): {} nodes", min_count, count_nodes(&trie));
    }

    trie.assign_ids();
    trie.classify_positions();
    trie.compute_transition_probs();
    trie.compute_weights();
    trie.assign_colors();

    tracing::info!(
        "Built trie: {} nodes, max_depth={}, terminals={}",
        trie.node_count,
        trie.max_depth(),
        trie.terminal_count()
    );

    // Print depth stats
    let analysis = TrieAnalysis::compute(&trie);
    println!("\nDepth | Nodes | Terminals | Avg Branch | Avg Entropy");
    println!("------|-------|-----------|------------|------------");
    for ds in &analysis.depth_stats {
        println!(
            "{:5} | {:5} | {:9} | {:10.2} | {:11.3}",
            ds.depth, ds.nodes, ds.terminals, ds.avg_branch, ds.avg_entropy
        );
    }

    Ok(())
}

// ─── Analyze ─────────────────────────────────────────────────────────────────

async fn cmd_analyze(
    db_path: &Path,
    cross_linguistic: bool,
    motifs: bool,
    entropy: bool,
) -> Result<()> {
    let db = Database::open(db_path).await?;
    let trie = build_full_trie(&db).await?;

    if entropy || (!cross_linguistic && !motifs) {
        let analysis = TrieAnalysis::compute(&trie);
        println!("\n=== Depth Statistics ===");
        println!("Depth | Nodes | Terminals | Avg Branch | Avg Entropy | Max Entropy");
        println!("------|-------|-----------|------------|-------------|------------");
        for ds in &analysis.depth_stats {
            println!(
                "{:5} | {:5} | {:9} | {:10.2} | {:11.3} | {:11.3}",
                ds.depth, ds.nodes, ds.terminals, ds.avg_branch, ds.avg_entropy, ds.max_entropy
            );
        }
        println!("\nPhoneme inventory: {} phonemes", analysis.phoneme_inventory.len());
        println!("Onset inventory: {} phonemes", analysis.onset_inventory.len());
        println!("Coda inventory: {} phonemes", analysis.coda_inventory.len());
    }

    if motifs {
        let detected = ipa_trie::motifs::detect_motifs(&trie.root, 500, 50);
        println!("\n=== Top Motifs ===");
        for m in &detected {
            println!("  {:20} count={}", m.label, m.count);
        }
    }

    if cross_linguistic {
        let stats = TrieAnalysis::cross_linguistic_stats(&trie);
        println!("\n=== Cross-Linguistic Depth Stats ===");
        for (lang, lang_stats) in &stats {
            println!("\n  {lang}:");
            for ds in lang_stats {
                println!(
                    "    depth={} nodes={} terminals={} branch={:.2} entropy={:.3}",
                    ds.depth, ds.nodes, ds.terminals, ds.avg_branch, ds.avg_entropy
                );
            }
        }
    }

    Ok(())
}

// ─── Export ──────────────────────────────────────────────────────────────────

async fn cmd_export(
    db_path: &Path,
    format: &str,
    output_dir: &str,
    lang_arg: &str,
    min_count: u64,
    max_depth: u32,
) -> Result<()> {
    let db = Database::open(db_path).await?;
    let languages = resolve_languages(lang_arg);

    let mut trie = PhonologicalTrie::new();
    for lang in &languages {
        let pronunciations = db.get_pronunciations_for_language(&lang.code).await?;
        tracing::info!("{}: {} pronunciations", lang.code, pronunciations.len());
        for (word, phonemes) in &pronunciations {
            trie.insert(phonemes, &lang.code, word);
        }
    }

    if min_count > 1 {
        trie.prune(min_count);
    }
    trie.assign_ids();
    trie.classify_positions();
    trie.compute_transition_probs();
    trie.compute_weights();
    trie.assign_colors();
    ConeTreeLayout::apply(&mut trie.root);

    let output_path = Path::new(output_dir);
    let do_all = format == "all";

    if do_all || format == "trie" {
        ipa_export::export_trie(&trie, output_path)?;
    }

    if do_all || format == "essay" {
        ipa_export::export_essay_data(&trie, output_path, max_depth)?;
    }

    if do_all || format == "stats" {
        ipa_export::export_cross_linguistic_stats(&trie, output_path)?;
    }

    tracing::info!("Export complete → {}", output_dir);
    Ok(())
}

// ─── Validate ────────────────────────────────────────────────────────────────

async fn cmd_validate(
    data_dir: &Path,
    db_path: &Path,
    lang_arg: &str,
    report: bool,
) -> Result<()> {
    let languages = resolve_languages(lang_arg);
    let db = Database::open(db_path).await?;

    // Load PHOIBLE
    let phoible = PhoibleSource::new(data_dir);
    let phoible_path = data_dir.join("phoible").join("phoible.csv");
    if !phoible_path.exists() {
        anyhow::bail!("PHOIBLE data not found. Run `download --source phoible` first.");
    }
    let iso_codes: Vec<&str> = languages.iter().map(|l| l.iso639_3.as_str()).collect();
    let inventory_store = phoible.parse(&phoible_path, &iso_codes)?;

    let mut total_valid = 0u64;
    let mut total_invalid = 0u64;
    let mut unknown_phonemes: std::collections::HashMap<String, u64> = std::collections::HashMap::new();

    for lang in &languages {
        let pronunciations = db.get_pronunciations_for_language(&lang.code).await?;
        let mut lang_valid = 0u64;
        let mut lang_invalid = 0u64;

        for (_word, phonemes) in &pronunciations {
            let issues = inventory_store.validate(&lang.iso639_3, phonemes);
            if issues.is_empty() {
                lang_valid += 1;
            } else {
                lang_invalid += 1;
                for issue in &issues {
                    if let Some(phoneme) = issue.strip_prefix("Unknown phoneme: ") {
                        *unknown_phonemes.entry(phoneme.to_string()).or_default() += 1;
                    }
                }
            }
        }

        let total = lang_valid + lang_invalid;
        let pct = if total > 0 {
            lang_valid as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "{:6} | {:6} valid / {:6} total ({:.1}%)",
            lang.code, lang_valid, total, pct
        );
        total_valid += lang_valid;
        total_invalid += lang_invalid;
    }

    println!("\nOverall: {} valid, {} invalid", total_valid, total_invalid);

    if report && !unknown_phonemes.is_empty() {
        let mut sorted: Vec<_> = unknown_phonemes.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        println!("\nTop unknown phonemes:");
        for (phoneme, count) in sorted.iter().take(30) {
            println!("  {:10} × {}", phoneme, count);
        }
    }

    Ok(())
}

// ─── Stats ───────────────────────────────────────────────────────────────────

async fn cmd_stats(db_path: &Path, lang_arg: &str) -> Result<()> {
    let db = Database::open(db_path).await?;
    let _languages = resolve_languages(lang_arg);

    let counts = db.count_by_language().await?;
    let invalid = db.count_invalid().await?;

    println!("\n=== Pipeline Statistics ===\n");
    println!("{:6} | {:>8}", "Lang", "Count");
    println!("-------|----------");
    let mut total = 0u64;
    for lc in &counts {
        println!("{:6} | {:>8}", lc.lang_code, lc.total);
        total += lc.total;
    }
    println!("-------|----------");
    println!("{:6} | {:>8}", "Total", total);
    println!("\nInvalid entries: {}", invalid);

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn count_nodes(trie: &PhonologicalTrie) -> usize {
    trie.nodes_bfs().len()
}

/// Build a full trie from all valid pronunciations in the database.
async fn build_full_trie(db: &Database) -> Result<PhonologicalTrie> {
    let languages = Language::all();
    let mut trie = PhonologicalTrie::new();

    for lang in &languages {
        let pronunciations = db.get_pronunciations_for_language(&lang.code).await?;
        for (word, phonemes) in &pronunciations {
            trie.insert(phonemes, &lang.code, word);
        }
    }

    trie.assign_ids();
    trie.classify_positions();
    trie.compute_transition_probs();
    trie.compute_weights();
    trie.assign_colors();
    ConeTreeLayout::apply(&mut trie.root);

    Ok(trie)
}

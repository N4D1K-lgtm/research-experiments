use std::collections::HashSet;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use strange_loops::analysis::{analyze, reuse_analysis, SizeReport};
use strange_loops::db::Db;
use strange_loops::enumerate::ski_basis;
use strange_loops::reduce::{reduce, Library, ReductionResult, ReductionStatus};
use strange_loops::term::Term;

const FUEL: usize = 10_000;
const MAX_TERM_SIZE: usize = 50_000;
const DB_PATH: &str = "strange_loops.db";

// ── Experiment definitions ──────────────────────────────────────────

/// A naming experiment: one or more motifs added as new primitives.
struct NamingExperiment {
    /// Database experiment name
    db_name: &'static str,
    /// Motif labels (CL term strings), e.g. ["SS"] or ["SS", "S(SS)"]
    motif_descs: &'static [&'static str],
    /// Max term size to enumerate
    max_size: usize,
}

/// All single-motif experiments (k=4 basis, max size 6).
const SINGLE_MOTIFS: &[NamingExperiment] = &[
    // Original 5
    NamingExperiment { db_name: "extended_ski_ss",   motif_descs: &["SS"],    max_size: 6 },
    NamingExperiment { db_name: "extended_ski_sk",   motif_descs: &["SK"],    max_size: 6 },
    NamingExperiment { db_name: "extended_ski_si",   motif_descs: &["SI"],    max_size: 6 },
    NamingExperiment { db_name: "extended_ski_sss",  motif_descs: &["S(SS)"], max_size: 6 },
    NamingExperiment { db_name: "extended_ski_ks",   motif_descs: &["KS"],    max_size: 6 },
    // New 6
    NamingExperiment { db_name: "extended_ski_kk",   motif_descs: &["KK"],    max_size: 6 },
    NamingExperiment { db_name: "extended_ski_ki",   motif_descs: &["KI"],    max_size: 6 },
    NamingExperiment { db_name: "extended_ski_sii",  motif_descs: &["SII"],   max_size: 6 },
    NamingExperiment { db_name: "extended_ski_skk",  motif_descs: &["SKK"],   max_size: 6 },
    NamingExperiment { db_name: "extended_ski_sks",  motif_descs: &["S(KS)"], max_size: 6 },
    NamingExperiment { db_name: "extended_ski_ski2", motif_descs: &["S(KI)"], max_size: 6 },
];

/// Pair combination experiments (k=5 basis, max size 6).
const PAIR_COMBOS: &[NamingExperiment] = &[
    NamingExperiment { db_name: "combo_ss_sss",  motif_descs: &["SS", "S(SS)"], max_size: 6 },
    NamingExperiment { db_name: "combo_ss_ks",   motif_descs: &["SS", "KS"],    max_size: 6 },
    NamingExperiment { db_name: "combo_ss_sii",  motif_descs: &["SS", "SII"],   max_size: 6 },
    NamingExperiment { db_name: "combo_sss_sii", motif_descs: &["S(SS)", "SII"], max_size: 6 },
];

/// Triple combination experiments (k=6 basis, max size 5).
const TRIPLE_COMBOS: &[NamingExperiment] = &[
    NamingExperiment { db_name: "combo_ss_sss_sii",  motif_descs: &["SS", "S(SS)", "SII"], max_size: 5 },
    NamingExperiment { db_name: "combo_ss_sk_ks",     motif_descs: &["SS", "SK", "KS"],     max_size: 5 },
];

// ── Survey runner ───────────────────────────────────────────────────

/// Run a survey with database persistence and resume support.
fn run_survey(
    db: &Db,
    experiment_name: &str,
    basis: &[Term],
    lib: &Library,
    max_size: usize,
) -> Vec<SizeReport> {
    let basis_str: Vec<String> = basis.iter().map(|t| format!("{t}")).collect();
    let basis_json = format!("[{}]", basis_str.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(","));

    let exp_id = db
        .get_or_create_experiment(experiment_name, &basis_json, FUEL, MAX_TERM_SIZE)
        .expect("db: create experiment");

    println!("=== {experiment_name} (id={exp_id}) ===");
    println!("Basis: {{{}}}", basis_str.join(", "));
    println!("Fuel: {FUEL}, Max term size: {MAX_TERM_SIZE}");
    println!("Database: {DB_PATH}");
    println!();

    let mut prev_subs: HashSet<Term> = HashSet::new();
    let mut reports = Vec::new();

    for size in 1..=max_size {
        // Check if already computed
        if db.size_complete(exp_id, size).unwrap_or(false) {
            eprintln!("[{experiment_name}] Size {size}: already in database, skipping");
            // Still need prev_subs for motif counting at later sizes.
            let nfs = load_nfs_from_db(db, exp_id, size);
            for nf in &nfs {
                for sub in nf.subterms() {
                    prev_subs.insert(sub.clone());
                }
            }
            // Load the report for the summary
            if let Ok(db_reports) = db.load_reports(exp_id) {
                if let Some(r) = db_reports.iter().find(|r| r.size == size) {
                    reports.push(SizeReport {
                        size: r.size,
                        total_terms: r.total_terms,
                        normal_form_count: r.total_terms - r.divergent_count - r.explosive_count,
                        divergent_count: r.divergent_count,
                        explosive_count: r.explosive_count,
                        distinct_normal_forms: r.distinct_nfs,
                        compression_ratio: r.compression_ratio,
                        avg_steps: r.avg_steps,
                        max_steps: r.max_steps,
                        motif_count: r.motif_count,
                        top_subexpressions: vec![],
                    });
                }
            }
            continue;
        }

        // Enumerate
        let t0 = Instant::now();
        let terms = strange_loops::enumerate::enumerate_size(size, basis);
        let enum_time = t0.elapsed();

        // Reduce in parallel with progress bar
        let pb = ProgressBar::new(terms.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{prefix} [{bar:40.cyan/dim}] {pos}/{len} ({per_sec}, {eta})",
            )
            .unwrap()
            .progress_chars("=> "),
        );
        pb.set_prefix(format!(
            "[{experiment_name}] Size {size}/{max_size} (enum {:.1}s)",
            enum_time.as_secs_f64()
        ));

        let results: Vec<ReductionResult> = terms
            .par_iter()
            .map(|t| {
                let mut r = reduce(t, lib, FUEL, MAX_TERM_SIZE);
                // Discard the (potentially huge) intermediate term for non-normal-form
                // results — they're never used downstream and can consume GBs of RAM
                // when terms explode to MAX_TERM_SIZE nodes.
                if r.status != ReductionStatus::NormalForm {
                    r.term = Term::S; // placeholder
                }
                pb.inc(1);
                r
            })
            .collect();
        pb.finish_and_clear();
        let reduce_time = t0.elapsed();
        eprintln!(
            "[{experiment_name}] Size {size}: {} terms reduced in {:.1}s",
            terms.len(),
            reduce_time.as_secs_f64()
        );

        // Analyze and extract reuse candidates (both need &results)
        let (report, new_subs) = analyze(size, &results, &prev_subs);
        let reuse = reuse_analysis(&results, 10);

        // Build DB rows from terms+results, then drop both to free memory
        // before the (potentially slow) database writes.
        let db_rows: Vec<(String, Option<String>, usize, ReductionStatus)> = terms
            .iter()
            .zip(results.iter())
            .map(|(term, res)| {
                let term_str = format!("{term}");
                let nf_str = if res.status == ReductionStatus::NormalForm {
                    Some(format!("{}", res.term))
                } else {
                    None
                };
                (term_str, nf_str, res.steps, res.status)
            })
            .collect();
        drop(terms);
        drop(results);

        // Store reductions in database
        let t2 = Instant::now();
        let pb_db = ProgressBar::new_spinner();
        pb_db.set_style(
            ProgressStyle::with_template("{prefix} {spinner} {msg}")
                .unwrap(),
        );
        pb_db.set_prefix(format!("[{experiment_name}] Size {size}"));
        pb_db.set_message("writing to database...");
        pb_db.enable_steady_tick(std::time::Duration::from_millis(120));

        db.store_reductions(exp_id, size, &db_rows)
            .expect("db: store reductions");
        drop(db_rows);

        // Store report
        db.store_report(
            exp_id,
            size,
            report.total_terms,
            report.distinct_normal_forms,
            report.compression_ratio,
            report.divergent_count,
            report.explosive_count,
            report.motif_count,
            report.avg_steps,
            report.max_steps,
        )
        .expect("db: store report");

        // Store reuse candidates
        let motif_rows: Vec<(String, usize, usize)> = reuse
            .iter()
            .map(|(t, nf_count, savings)| (format!("{t}"), *nf_count, *savings))
            .collect();
        db.store_motifs(exp_id, size, &motif_rows)
            .expect("db: store motifs");

        pb_db.finish_and_clear();
        let store_time = t2.elapsed();
        eprintln!(
            "[{experiment_name}] Size {size}: stored in {:.1}s",
            store_time.as_secs_f64()
        );

        // Print report
        print_size_report(size, &report, &reuse);

        prev_subs = new_subs;
        reports.push(report);
    }

    reports
}

// ── Helpers ─────────────────────────────────────────────────────────

fn load_nfs_from_db(db: &Db, experiment_id: i64, size: usize) -> Vec<Term> {
    let mut stmt = db
        .conn
        .prepare(
            "SELECT DISTINCT normal_form FROM reductions
             WHERE experiment_id=?1 AND size=?2 AND normal_form IS NOT NULL",
        )
        .expect("db: prepare nf query");
    let rows = stmt
        .query_map(rusqlite::params![experiment_id, size as i64], |row| {
            row.get::<_, String>(0)
        })
        .expect("db: query nfs");

    let mut terms = Vec::new();
    for row in rows {
        if let Ok(s) = row {
            if let Some(t) = parse_term(&s) {
                terms.push(t);
            }
        }
    }
    terms
}

/// Simple recursive-descent parser for CL term display format.
/// Handles: S, K, I, #N, application (juxtaposition), parentheses.
fn parse_term(s: &str) -> Option<Term> {
    let bytes = s.as_bytes();
    let (term, rest) = parse_app(bytes)?;
    if rest.is_empty() {
        Some(term)
    } else {
        None
    }
}

fn parse_app(input: &[u8]) -> Option<(Term, &[u8])> {
    let (mut left, mut rest) = parse_atom(input)?;
    while let Some((right, new_rest)) = parse_atom(rest) {
        left = Term::app(left, right);
        rest = new_rest;
    }
    Some((left, rest))
}

fn parse_atom(input: &[u8]) -> Option<(Term, &[u8])> {
    if input.is_empty() {
        return None;
    }
    match input[0] {
        b'S' => Some((Term::S, &input[1..])),
        b'K' => Some((Term::K, &input[1..])),
        b'I' => Some((Term::I, &input[1..])),
        b'#' => {
            let mut i = 1;
            while i < input.len() && input[i].is_ascii_digit() {
                i += 1;
            }
            let n: usize = std::str::from_utf8(&input[1..i]).ok()?.parse().ok()?;
            Some((Term::Lib(n), &input[i..]))
        }
        b'(' => {
            let (term, rest) = parse_app(&input[1..])?;
            if rest.first() == Some(&b')') {
                Some((term, &rest[1..]))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Build a Term from a CL string like "SS", "S(SS)", "SII".
fn build_motif(desc: &str) -> Term {
    parse_term(desc).unwrap_or_else(|| panic!("Cannot parse motif: {desc}"))
}

fn print_size_report(size: usize, report: &SizeReport, reuse: &[(Term, usize, usize)]) {
    println!("--- Size {size} ---");
    println!("  Terms:       {}", report.total_terms);
    println!(
        "  Normal forms: {} ({:.1}%)",
        report.normal_form_count,
        if report.total_terms > 0 {
            100.0 * report.normal_form_count as f64 / report.total_terms as f64
        } else {
            0.0
        }
    );
    println!("  Divergent:   {}", report.divergent_count);
    println!("  Explosive:   {}", report.explosive_count);
    println!("  Distinct NFs: {}", report.distinct_normal_forms);
    println!(
        "  Compression: {:.4} ({}/{})",
        report.compression_ratio, report.distinct_normal_forms, report.total_terms
    );
    println!(
        "  Steps: avg={:.1}, max={}",
        report.avg_steps, report.max_steps
    );
    println!("  New motifs:  {}", report.motif_count);

    if !report.top_subexpressions.is_empty() {
        print!("  Top subs:    ");
        for (i, (t, count)) in report.top_subexpressions.iter().take(8).enumerate() {
            if i > 0 {
                print!("  ");
            }
            print!("{t}({count})");
        }
        println!();
    }

    if !reuse.is_empty() {
        println!("  Reuse candidates:");
        for (t, nf_count, savings) in reuse.iter().take(5) {
            println!("    {t:20} in {nf_count:4} NFs, savings={savings}");
        }
    }
    println!();
}

fn print_summary(label: &str, reports: &[SizeReport]) {
    println!("=== Summary: {label} ===");
    println!(
        "{:>5} | {:>9} | {:>8} | {:>7} | {:>5} | {:>6} | {:>6} | {:>9}",
        "Size", "Terms", "DistNFs", "Ratio", "Divg", "Expl", "Motifs", "AvgSteps"
    );
    println!("{}", "-".repeat(75));
    for r in reports {
        println!(
            "{:>5} | {:>9} | {:>8} | {:>7.4} | {:>5} | {:>6} | {:>6} | {:>9.2}",
            r.size,
            r.total_terms,
            r.distinct_normal_forms,
            r.compression_ratio,
            r.divergent_count,
            r.explosive_count,
            r.motif_count,
            r.avg_steps,
        );
    }
    println!();
}

/// Run a naming experiment (single, pair, or triple motif).
/// Returns (display_label, reports).
fn run_naming_experiment(
    db: &Db,
    exp: &NamingExperiment,
) -> (String, Vec<SizeReport>) {
    let motifs: Vec<Term> = exp.motif_descs.iter().map(|d| build_motif(d)).collect();
    let display_label = exp.motif_descs.join("+");

    println!(
        ">>> Naming experiment: adding {{{}}} as new primitive(s) <<<\n",
        display_label
    );

    // Build library with one entry per motif
    let mut lib = Library::new();
    for (desc, term) in exp.motif_descs.iter().zip(motifs.iter()) {
        lib.add_term(*desc, term);
    }

    // Build basis: S, K, I, #0, [#1, [#2]]
    let mut basis = vec![Term::S, Term::K, Term::I];
    for i in 0..motifs.len() {
        basis.push(Term::Lib(i));
    }

    let reports = run_survey(db, exp.db_name, &basis, &lib, exp.max_size);
    print_summary(&format!("Extended (SKI + {})", display_label), &reports);
    (display_label, reports)
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let db = Db::open(DB_PATH).expect("Failed to open database");

    println!("Strange Loops - A1: Recursion Phase Transitions");
    println!("================================================\n");

    // Phase 1: Baseline (SKI)
    let ski = ski_basis();
    let empty_lib = Library::new();
    let baseline = run_survey(&db, "baseline_ski", &ski, &empty_lib, 7);
    print_summary("Baseline (SKI)", &baseline);

    // Phase 2: Single-motif experiments (k=4, size 6)
    println!("\n{}", "=".repeat(60));
    println!("PHASE 2: Single-Motif Naming Experiments (k=4, max size 6)");
    println!("{}\n", "=".repeat(60));

    let mut all_results: Vec<(String, &str, Vec<SizeReport>)> = Vec::new();

    for exp in SINGLE_MOTIFS {
        let (label, reports) = run_naming_experiment(&db, exp);
        all_results.push((label, exp.db_name, reports));
    }

    // Phase 3: Pair combinations (k=5, size 6)
    println!("\n{}", "=".repeat(60));
    println!("PHASE 3: Pair Combination Experiments (k=5, max size 6)");
    println!("{}\n", "=".repeat(60));

    for exp in PAIR_COMBOS {
        let (label, reports) = run_naming_experiment(&db, exp);
        all_results.push((label, exp.db_name, reports));
    }

    // Phase 4: Triple combinations (k=6, size 5)
    println!("\n{}", "=".repeat(60));
    println!("PHASE 4: Triple Combination Experiments (k=6, max size 5)");
    println!("{}\n", "=".repeat(60));

    for exp in TRIPLE_COMBOS {
        let (label, reports) = run_naming_experiment(&db, exp);
        all_results.push((label, exp.db_name, reports));
    }

    // Final cross-experiment comparison
    print_all_experiments_comparison(&baseline, &all_results);

    println!("All results stored in {DB_PATH}");
    println!("Run `python3 analyze.py` for visualizations.");
}

/// Print a ranked summary of all experiments sorted by compression advantage.
fn print_all_experiments_comparison(
    baseline: &[SizeReport],
    experiments: &[(String, &str, Vec<SizeReport>)],
) {
    println!("\n{}", "=".repeat(70));
    println!("CROSS-EXPERIMENT COMPARISON (ALL)");
    println!("{}\n", "=".repeat(70));

    // Collect advantages at a common comparison size
    // Singles and pairs: compare at size 6; triples at size 5
    // We'll show both where available
    for compare_size in [6, 5] {
        if compare_size > baseline.len() {
            continue;
        }
        let b_ratio = baseline[compare_size - 1].compression_ratio;
        if b_ratio <= 0.0 {
            continue;
        }

        let mut ranked: Vec<(&str, &str, f64, f64, usize, f64, f64)> = Vec::new();

        for (label, db_name, reports) in experiments {
            if compare_size > reports.len() {
                continue;
            }
            let r = &reports[compare_size - 1];
            let advantage = r.compression_ratio / b_ratio;
            let divg_pct = if r.total_terms > 0 {
                100.0 * r.divergent_count as f64 / r.total_terms as f64
            } else { 0.0 };
            let expl_pct = if r.total_terms > 0 {
                100.0 * r.explosive_count as f64 / r.total_terms as f64
            } else { 0.0 };
            ranked.push((
                label.as_str(),
                db_name,
                r.compression_ratio,
                advantage,
                r.distinct_normal_forms,
                divg_pct,
                expl_pct,
            ));
        }
        ranked.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

        println!("=== Ranking at size {compare_size} (baseline rho={b_ratio:.4}) ===");
        println!(
            "{:>4} {:>16} {:>10} {:>10} {:>10} {:>7} {:>7}",
            "Rank", "Experiment", "Comp.Ratio", "Advantage", "Dist.NFs", "Divg%", "Expl%"
        );
        println!("{}", "-".repeat(75));
        for (i, (label, _, cr, adv, nfs, dp, ep)) in ranked.iter().enumerate() {
            println!(
                "{:>4} {:>16} {:>10.4} {:>9.3}x {:>10} {:>6.1}% {:>6.1}%",
                i + 1, label, cr, adv, nfs, dp, ep
            );
        }
        println!();
    }
}

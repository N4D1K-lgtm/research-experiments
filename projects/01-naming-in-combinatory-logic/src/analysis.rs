use std::collections::{HashMap, HashSet};

use crate::reduce::{ReductionResult, ReductionStatus};
use crate::term::Term;

#[derive(Debug)]
pub struct SizeReport {
    pub size: usize,
    pub total_terms: usize,
    pub normal_form_count: usize,
    pub divergent_count: usize,
    pub explosive_count: usize,
    pub distinct_normal_forms: usize,
    pub compression_ratio: f64,
    pub avg_steps: f64,
    pub max_steps: usize,
    pub motif_count: usize,
    /// Top sub-expressions by frequency: (term, count)
    pub top_subexpressions: Vec<(Term, usize)>,
}

/// Analyze reduction results for one size level.
///
/// `prev_subs`: set of all sub-expressions seen in normal forms at smaller sizes.
/// Returns the report and the updated cumulative sub-expression set.
pub fn analyze(
    size: usize,
    results: &[ReductionResult],
    prev_subs: &HashSet<Term>,
) -> (SizeReport, HashSet<Term>) {
    let total_terms = results.len();

    let mut normal_form_count = 0usize;
    let mut divergent_count = 0usize;
    let mut explosive_count = 0usize;
    let mut total_steps = 0usize;
    let mut max_steps = 0usize;

    let mut distinct_nfs: HashSet<Term> = HashSet::new();
    let mut sub_freq: HashMap<Term, usize> = HashMap::new();
    let mut all_subs: HashSet<Term> = prev_subs.clone();

    for r in results {
        match r.status {
            ReductionStatus::NormalForm => {
                normal_form_count += 1;
                total_steps += r.steps;
                if r.steps > max_steps {
                    max_steps = r.steps;
                }
                distinct_nfs.insert(r.term.clone());

                for sub in r.term.subterms() {
                    *sub_freq.entry(sub.clone()).or_insert(0) += 1;
                    all_subs.insert(sub.clone());
                }
            }
            ReductionStatus::OutOfFuel => {
                divergent_count += 1;
            }
            ReductionStatus::TermTooLarge => {
                explosive_count += 1;
            }
        }
    }

    // Motifs: sub-expressions that are NEW at this size
    let motif_count = all_subs.difference(prev_subs).count();

    // Sort sub-expressions by frequency (descending)
    let mut sorted_subs: Vec<(Term, usize)> = sub_freq.into_iter().collect();
    sorted_subs.sort_by(|a, b| b.1.cmp(&a.1));
    let top_subexpressions: Vec<(Term, usize)> = sorted_subs.into_iter().take(10).collect();

    let avg_steps = if normal_form_count > 0 {
        total_steps as f64 / normal_form_count as f64
    } else {
        0.0
    };

    let compression_ratio = if total_terms > 0 {
        distinct_nfs.len() as f64 / total_terms as f64
    } else {
        0.0
    };

    let report = SizeReport {
        size,
        total_terms,
        normal_form_count,
        divergent_count,
        explosive_count,
        distinct_normal_forms: distinct_nfs.len(),
        compression_ratio,
        avg_steps,
        max_steps,
        motif_count,
        top_subexpressions,
    };

    (report, all_subs)
}

/// Compute the "reuse value" of the top sub-expressions.
/// For each candidate sub-expression E, count how many distinct normal forms
/// contain it and estimate description-length savings if E were a primitive.
pub fn reuse_analysis(
    results: &[ReductionResult],
    top_k: usize,
) -> Vec<(Term, usize, usize)> {
    // (sub-expression, #NFs containing it, estimated size savings)
    let mut sub_freq: HashMap<Term, usize> = HashMap::new();

    let nfs: Vec<&Term> = results
        .iter()
        .filter(|r| r.status == ReductionStatus::NormalForm)
        .map(|r| &r.term)
        .collect();

    // Count in how many distinct NFs each sub-expression appears
    let mut nf_set: HashSet<&Term> = HashSet::new();
    let mut sub_nf_count: HashMap<Term, HashSet<usize>> = HashMap::new();

    for (i, nf) in nfs.iter().enumerate() {
        if !nf_set.insert(nf) {
            continue; // skip duplicate NFs
        }
        for sub in nf.subterms() {
            *sub_freq.entry(sub.clone()).or_insert(0) += 1;
            sub_nf_count.entry(sub.clone()).or_default().insert(i);
        }
    }

    let mut candidates: Vec<(Term, usize, usize)> = sub_freq
        .into_iter()
        .filter(|(t, _)| t.size() >= 2) // only non-atomic sub-expressions
        .map(|(t, count)| {
            let nf_count = sub_nf_count.get(&t).map_or(0, |s| s.len());
            let savings = count * (t.size() - 1); // each occurrence saves (size-1)
            (t, nf_count, savings)
        })
        .collect();

    candidates.sort_by(|a, b| b.2.cmp(&a.2)); // sort by savings descending
    candidates.truncate(top_k);
    candidates
}

# Structural Emergence Under Basis Extension in Combinatory Logic

Combinatory logic (S, K, I) is Turing complete with three primitives and three reduction rules. When you enumerate all terms at a given size and reduce them, the ratio of distinct outputs to total inputs drops exponentially. Most terms collapse into a small number of structures.

This project takes those frequently occurring sub-expressions, adds them as new primitives and re-enumerates to measure the effect on compression. Which sub-expressions slow the compression and which make it worse? What structural properties predict whether naming something helps?

## Approach

The reduction rules are:

```
I x      -> x
K x y    -> x
S x y z  -> x z (y z)
```

At each term size N, generate all possible terms (Catalan(N-1) * 3^N of them), reduce each to normal form with a step limit, and measure:

1. How many distinct normal forms there are vs total terms (compression ratio)
2. How sub-expression frequencies are distributed
3. Which sub-expressions are new at this size (motifs)
4. How much naming the common sub-expressions would shorten descriptions (reuse value)

Then select motifs, add each as a new primitive and re-run the survey.

## Running

```bash
cargo run --release
python analyze.py
```

## Files

- `A1_PLAN.md` - Full experimental plan
- `RESEARCH.md` - Broader research context
- `src/` - Rust implementation (enumeration, reduction, analysis, SQLite storage)
- `paper/` - LaTeX writeup
- `analyze.py` - Plotting and analysis
- `results.csv` - Raw results
- `plots/` - Generated figures

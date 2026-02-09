# Recursion Phase Transitions

A function that can call itself can represent unbounded computation. But abstraction, the ability to treat a pattern as a thing, seems to need something more: applying an operation to its own output and recognizing structure in the result. The question is whether there's a critical depth where this kicks in.

## Approach

Combinatory logic (S, K, I) is a good test case because it's minimal and self-application is native. The reduction rules are:

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

The hypothesis is that there's a critical size where the compression ratio drops sharply and a small number of structures start dominating. Those structures should correspond to known useful combinators like B, C, W.

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

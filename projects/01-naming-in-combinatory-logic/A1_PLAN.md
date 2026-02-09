# A1: Recursion Phase Transitions - Concrete Plan

## The Problem, Sharpened

We want to answer: **at what point does recursive self-application produce
structures that are more useful as building blocks than the process that
created them?**

But first we need to turn every vague word in that sentence into something
measurable.

---

## Phase 0: Formalize (Do This First, With Paper)

Before writing any code, you need to nail down three things:

### 0a. What formal system?

**Recommendation: Combinatory Logic (CL).**

Why CL over lambda calculus, string rewriting, or something custom?

- **Minimal**: three primitives (S, K, I) and one operation (application)
- **Self-application is native**: any term can be applied to any other.
  `SII` is literally the self-application combinator: `SII x → x x`
- **Turing complete**: so we KNOW abstraction can emerge. The question is when.
- **No variable binding**: unlike lambda calculus, no alpha-conversion or
  capture-avoidance headaches. Pure tree rewriting.
- **Well-studied**: we can validate our infrastructure against known results
  before exploring new territory.

The reduction rules are:

```
I x      → x
K x y    → x
S x y z  → x z (y z)
```

That's it. Everything else emerges from these three rules.

A term is either:
- A combinator: S, K, or I
- An application: (M N) where M and N are terms

Size of a term = number of combinators in it.
- Size 1: S, K, I
- Size 2: (S S), (S K), (S I), (K S), (K K), (K I), (I S), (I K), (I I)
- Size 3: all binary trees with 3 leaves, each leaf being S/K/I
- Size N: Catalan(N-1) * 3^N terms

### 0b. What are we measuring?

Four metrics, each capturing a different aspect of "emergent structure":

**Metric 1: Normal Form Compression Ratio**
- Generate all terms of size N
- Reduce each to normal form (with a step limit L for divergent terms)
- Count distinct normal forms: D(N)
- Ratio: D(N) / Total(N)
- If this ratio DROPS sharply at some N, many different terms are collapsing
  to the same structures. Those structures are "attractors" - natural
  abstractions.

**Metric 2: Sub-expression Frequency Distribution**
- Across all normal forms at size N, count every sub-expression
- Plot the frequency distribution
- Is it Zipfian? (A few sub-expressions dominate, long tail of rare ones)
- A Zipfian distribution would suggest a few structures are doing
  disproportionate "work" - these are proto-abstractions.

**Metric 3: Motif Emergence**
- A "motif" is a sub-expression that appears in normal forms at size N
  but does NOT appear in any term of size < N (as a term itself)
- These are genuinely NOVEL structures produced by the rewriting dynamics
- Count motifs at each size. Is there a size where motif count explodes?

**Metric 4: Compositional Reuse Value**
- For each common sub-expression E found at size N:
  - How many distinct normal forms at size N contain E?
  - If we "name" E and allow it as a new primitive, how much does the
    total description length of all normal forms at size N decrease?
- This directly measures: is E useful as a building block?
- The phase transition is when the reuse value of the best motif exceeds
  the cost of naming it.

### 0c. What are the specific hypotheses?

**H1**: The compression ratio D(N)/Total(N) decreases monotonically, but
with a sharp drop at some critical N*. Below N*, terms are "mostly distinct."
Above N*, they collapse into clusters around attractor structures.

**H2**: The sub-expression frequency distribution transitions from roughly
uniform (small N) to Zipfian (large N), and this transition happens at or
near N*.

**H3**: N* is SMALL. Probably 4-7. Because combinatory logic is expressive
enough that interesting things happen quickly.

**H4**: The motifs that emerge at N* are computationally meaningful - they
correspond to known combinators (like B = S(KS)K, C = S(S(KS)(S(KK)S))(KK),
W = SS(SK), etc.) or to structures with recognizable computational roles.

**H5 (the big one)**: Recursive generation produces these phase transitions;
iterative generation (applying a fixed rule repeatedly) does NOT, or does so
at a much larger N. This would confirm that recursion specifically (not just
repetition) is the precursor to abstraction.

---

## Phase 1: Build the Substrate (Rust)

### Project structure

```
strange-loops/
  Cargo.toml
  src/
    lib.rs            -- re-exports
    term.rs           -- Term enum, Display, Eq, Hash, size, sub-expressions
    reduce.rs         -- CL reduction engine (normal-order, with step limit)
    enumerate.rs      -- generate all terms of size N
    analysis.rs       -- the four metrics above
    main.rs           -- experiment runner, CSV output
```

### 1a. Term representation

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Term {
    S,
    K,
    I,
    App(Box<Term>, Box<Term>),
}
```

Key operations needed:
- `size(&self) -> usize` -- count of S/K/I leaves
- `subexpressions(&self) -> Vec<&Term>` -- all sub-terms
- `depth(&self) -> usize` -- tree depth
- Display as string (for debugging)
- Comparison and hashing (for deduplication)

### 1b. Reduction engine

Normal-order reduction (reduce leftmost outermost redex first):
- Match on term structure
- If `App(I, x)` → reduce to x
- If `App(App(K, x), y)` → reduce to x
- If `App(App(App(S, x), y), z)` → reduce to `App(App(x, z), App(y, z))`
- Otherwise, reduce sub-terms

CRITICAL: need a step limit. Many CL terms diverge (e.g., `SII(SII)`).
Use a fuel/gas parameter: maximum reduction steps before we declare "divergent."

Return type: `Result<Term, Divergent>`

Also track: number of steps to reach normal form. This is related to
Bennett's "logical depth" and is interesting in its own right.

### 1c. Enumeration

Generate all CL terms of size N. This is equivalent to generating all
full binary trees with N leaves, then assigning each leaf to S, K, or I.

Number of full binary trees with N leaves = Catalan(N-1)
Number of CL terms of size N = Catalan(N-1) * 3^N

| N | Catalan(N-1) | 3^N  | Total   |
|---|-------------|------|---------|
| 1 | 1           | 3    | 3       |
| 2 | 1           | 9    | 9       |
| 3 | 2           | 27   | 54      |
| 4 | 5           | 81   | 405     |
| 5 | 14          | 243  | 3,402   |
| 6 | 42          | 729  | 30,618  |
| 7 | 132         | 2187 | 288,684 |
| 8 | 429         | 6561 | 2,814,669 |

So sizes 1-7 are very tractable. Size 8 is a few million - still fine.
Size 9+ may need sampling rather than exhaustive enumeration.

---

## Phase 2: Build the Observatory

### 2a. Metric computations

For each size N:
1. Generate all terms
2. Reduce each (parallel - these are independent, use rayon)
3. Collect: (original_term, normal_form_or_divergent, step_count)
4. Compute:
   - D(N) = number of distinct normal forms
   - Compression ratio = D(N) / Total(N)
   - Sub-expression frequencies across all normal forms
   - Motifs (sub-expressions new at this size)
   - Reuse value of top motifs

### 2b. Output format

CSV files for each metric, plottable with any tool:
```
size, total_terms, distinct_normal_forms, compression_ratio, divergent_count, ...
```

Plus a separate file for the top-K sub-expressions at each size.

Don't bother with visualization in Rust - output data, plot with whatever
you prefer (gnuplot, Python/matplotlib, even a spreadsheet).

---

## Phase 3: Experiments

### Experiment 1: Baseline survey (FIRST THING TO RUN)
- Enumerate sizes 1 through 7 (maybe 8)
- Compute all four metrics
- Plot them
- Look for the phase transition
- This tells us if the hypothesis is even in the right ballpark

### Experiment 2: Iterative vs. Recursive comparison
- "Iterative": start with S, K, I. At each generation, apply every rule
  to every term from the PREVIOUS generation only
- "Recursive": at each generation, apply every rule to every term from
  ALL previous generations
- Compare metric trajectories
- Prediction: recursive shows phase transition earlier

### Experiment 3: Vary the base rules
- Try with just S and K (I is derivable: SKK → I)
- Try with S, K, and B (= S(KS)K, composition)
- Try with S, K, and W (= SS(SK), duplication)
- Does adding "natural abstractions" as primitives shift N*?
- If yes: abstraction begets abstraction (this is a big result)

### Experiment 4: The naming experiment
- Run the baseline survey
- At each size, find the top motif
- Add it as a new primitive (effectively, extend the combinator basis)
- Re-run the survey with the extended basis
- Does N* shift? Does a NEW phase transition appear at a higher level?
- This simulates the process of abstraction itself

---

## Phase 0.5: What to Read (Optional, Don't Get Lost)

If you want theoretical grounding (but don't let this delay building):

- **Combinatory Logic basics**: Hindley & Seldin, "Lambda-Calculus and
  Combinators" chapters 2-3. Just enough to understand S, K, I reduction.
- **Algorithmic Information Theory**: Li & Vitanyi, chapters 1-2. For the
  compression/complexity angle.
- **Logical Depth**: Bennett, "Logical Depth and Physical Complexity" (1988).
  Short paper, directly relevant to "step count to normal form" metric.
- **DreamCoder**: Ellis et al., "DreamCoder: Building Libraries of
  Compositional Abstractions" (2021). They do something related - learning
  abstractions from programs - but top-down with neural guidance. Your
  approach is bottom-up and exhaustive.

---

## Concrete First Session (What to Actually Do Tomorrow)

1. `cargo init` the project
2. Implement `Term` enum with `size()`, `Display`, `Eq`, `Hash`
3. Implement `reduce(term, fuel) -> Result<Term, Divergent>`
4. Implement `enumerate(n) -> Vec<Term>` for sizes 1-5
5. Write a quick `main` that enumerates size 1-5, reduces everything,
   and prints: total terms, distinct normal forms, divergent count
6. Look at the numbers. Do they surprise you?

That's maybe 200-300 lines of Rust and gives you your first real data point.
Everything after that is elaboration.

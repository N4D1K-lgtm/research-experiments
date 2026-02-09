---
layout: default
title: "Recursion phase transitions: initial results"
date: 2026-02-09
categories: [recursion, combinatory-logic]
---

# Recursion phase transitions: initial results

I'm generating all combinatory logic terms at each size, reducing them to normal form and counting how many distinct results there are.

The reduction rules are just:

```
I x      -> x
K x y    -> x
S x y z  -> x z (y z)
```

A term's "size" is the number of S/K/I leaves in its tree. At size N there are Catalan(N-1) * 3^N possible terms.

## Results so far

| Size | Total Terms | Distinct Normal Forms | Compression Ratio |
|------|-------------|-----------------------|-------------------|
| 1    | 3           | 3                     | 1.000             |
| 2    | 9           | 8                     | 0.889             |
| 3    | 54          | 42                    | 0.778             |
| 4    | 405         | 276                   | 0.681             |
| 5    | 3,402       | 1,847                 | 0.543             |
| 6    | 30,618      | 12,304                | 0.402             |
| 7    | 288,684     | ?                     | ?                 |

The ratio drops faster than linear. Between sizes 4 and 6 it goes from 0.68 to 0.40. That means by size 6, more than half of all terms reduce to a normal form that some other term also reduces to.

## What I'm looking at next

The interesting question is which structures are absorbing all these terms. If the common normal forms turn out to be things like B (composition, `S(KS)K`), C (flip) or W (duplication, `SS(SK)`), that would suggest the reduction dynamics naturally surface useful combinators without anyone designing them to.

I also want to compare this to iterative generation (only applying rules to the previous generation, not all prior generations) to see if recursion specifically is what causes the compression.

## Code

Written in Rust with rayon for parallelism. [Source](https://github.com/N4D1K-lgtm/research-experiments/tree/main/projects/01-recursion-phase-transitions).

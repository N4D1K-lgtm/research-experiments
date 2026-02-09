---
layout: default
title: "XX: [Title]"
date: YYYY-MM-DD
categories: [tag1, tag2, tag3]
---

# [Title]

[PDF version]({{ site.baseurl }}/assets/posts/XX/paper.pdf). [Source code](https://github.com/N4D1K-lgtm/research-experiments/tree/main/projects/XX-[project-slug]).

## Setup

<!-- What formal system / domain is this experiment in? State the primitives,
     rules, and core definitions. Use LaTeX for any formal notation. -->

<!-- What was enumerated or generated, and at what scale? State the parameter
     space and its growth rate. Link to the enumeration code. -->

<!-- How the generation works: the algorithmic approach (e.g. recursive
     partitioning, sampling) and any memoization. -->

<!-- What did you measure? List each metric (numbered) with a one-sentence
     description of what it captures. Give the formula where non-obvious. -->

<!-- Reduction / evaluation strategy: which evaluation order (normal-order,
     applicative, etc.) and why it matters. How terms are decomposed
     (e.g. spine decomposition). Link to the reduction code. Termination
     criteria: what limits apply, how divergent/explosive/failed terms are
     classified. -->

<!-- Data representation: how terms/objects are represented in memory.
     Call out any performance-critical choices (reference counting, caching,
     interning) and why they're needed. -->

<!-- Parallelism and memory: how work is distributed, any memory-bounding
     tricks (e.g. discarding intermediate results). -->

<!-- Storage and reproducibility: what's persisted (schema summary or link),
     resume support for interrupted runs. -->

## Baseline results

<!-- Primary data table: one row per parameter value, columns for each metric. -->

<!-- Key quantitative finding from the baseline (e.g. a fit, a scaling law).
     Include the equation and goodness of fit. -->

<!-- Figure: the main baseline plot. -->

### [Phenomenon 1]

<!-- Describe the first qualitative pattern visible in the data.
     Use a numbered sequence if there's a cascade or progression. -->

<!-- Figure if applicable. -->

### [Phenomenon 2]

<!-- Describe the second qualitative pattern (e.g. distribution shape,
     attractor structure). Include a table of top entries if relevant. -->

<!-- Figure if applicable. -->

### [Phenomenon 3]

<!-- Third pattern (e.g. reuse/frequency analysis). -->

<!-- Figure if applicable. -->

## The [main] experiment

<!-- Describe the experimental intervention: what was changed relative to
     baseline, how the parameter space was modified, and what was re-measured.
     Link to the relevant implementation (e.g. how new primitives are stored,
     how the parameter space changes). State how the new conditions relate
     to the baseline quantitatively (e.g. growth rate of the new space). -->

### [Sub-experiment] results

<!-- Results table for the intervention. -->

<!-- Figure: comparison across conditions. -->

<!-- Group the results into qualitative categories (winners, losers, neutral, etc.)
     with bold labels. For each group, state which conditions belong and why. -->

### What determines [key variable]

<!-- Distill the results into structural rules. Use a numbered list:
     1. Rule about one factor
     2. Rule about another factor
     3. Exception or edge case -->

<!-- Figure(s): classification / taxonomy plots. -->

## [Secondary experiments]

<!-- If you ran follow-up experiments (combinations, ablations, scaling tests),
     report them here with a table and brief interpretation. -->

<!-- Figure if applicable. -->

## Discussion

<!-- Interpret the main finding. What pattern holds across all conditions?
     What varies? What does this suggest about the underlying structure? -->

<!-- State the key open question raised by the results. -->

### Limitations

<!-- Bulleted list of caveats, each with a bold label and one-sentence
     explanation. Common categories:
     - Scale ceiling
     - Parameter sensitivity
     - Confounds
     - Simplifying assumptions -->

## What's next

<!-- Bulleted list of concrete follow-up experiments or analyses,
     each as a bold label with a one-sentence description. -->

## References

<!-- Cited works in author-date format. -->

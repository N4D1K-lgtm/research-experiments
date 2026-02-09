---
layout: default
title: Projects
permalink: /projects/
---

# Projects

## [01: Structural Emergence Under Basis Extension in Combinatory Logic]({{ site.baseurl }}/blog/2026/02/09/recursion-phase-transitions-initial-results/)

Combinatory logic (S, K, I) is Turing complete with three primitives and three reduction rules. When you enumerate all terms at a given size and reduce them, the ratio of distinct outputs to total inputs drops exponentially. Most terms collapse into a small number of structures.

The experiment is: take the sub-expressions that keep appearing in those structures, add them as new primitives and re-enumerate. Which ones slow the compression and which ones make it worse? What structural properties predict whether naming something helps?

[src](https://github.com/N4D1K-lgtm/research-experiments/tree/main/projects/01-naming-in-combinatory-logic/src) / [paper (pdf)]({{ site.baseurl }}/assets/posts/01/paper.pdf)

---

## Other Directions

These are outlined in the [research document](https://github.com/N4D1K-lgtm/research-experiments/blob/main/projects/01-naming-in-combinatory-logic/RESEARCH.md):

- Whether self-narration (generating descriptions of experience and feeding them back as training data) improves learning efficiency
- Knowledge representation using continuous spectra instead of categories
- Communication systems based on relative positioning in continuous space vs discrete symbols

---
layout: default
title: Home
---

# Research Experiments

Experiments at the intersection of communication, cognition and computation.

## Current Work

### [Structural Emergence Under Basis Extension in Combinatory Logic]({{ site.baseurl }}/blog/2026/02/09/recursion-phase-transitions-initial-results/)

Combinatory logic compresses as terms get larger: many syntactically distinct terms reduce to the same normal form. The question is what happens when you take the sub-expressions that keep showing up and name them as new primitives. Does naming the right thing slow the compression down, and what makes one sub-expression more worth naming than another?

[src](https://github.com/N4D1K-lgtm/research-experiments/tree/main/projects/01-naming-in-combinatory-logic/src) / [paper (pdf)]({{ site.baseurl }}/assets/posts/01/paper.pdf)

## Posts

<ul>
  {% for post in site.posts limit:5 %}
    <li>
      <span class="post-date">{{ post.date | date: "%b %-d, %Y" }}</span>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>

[All posts]({{ site.baseurl }}/blog)

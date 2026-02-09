---
layout: default
title: Home
---

# Research Experiments

Experiments on recursion, formal systems and cognition.

## Current

### [Recursion Phase Transitions]({{ site.baseurl }}/projects)

Recursion can represent unbounded computation, but abstraction seems to require something more specific: applying an operation to its own output and recognizing the resulting structure. I'm interested in whether there's a sharp transition point where this starts happening in minimal formal systems.

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

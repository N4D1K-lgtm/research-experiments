---
layout: default
title: Blog
permalink: /blog/
---

# Research Notes

Updates, insights, and progress from ongoing experiments.

<div class="posts">
  {% for post in site.posts %}
  <article class="post">
    <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
    <p class="post-date">{{ post.date | date: "%B %-d, %Y" }}</p>
    {% if post.excerpt %}
      <p>{{ post.excerpt | strip_html }}</p>
    {% endif %}
    <a href="{{ post.url | relative_url }}">Read more →</a>
  </article>
  {% endfor %}
</div>

---
layout: default
title: Blog
permalink: /blog/
---

# Writeups

<div class="posts">
  {% for post in site.posts %}
  <article class="post">
    <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
    <p class="post-date">{{ post.date | date: "%B %-d, %Y" }}</p>
  </article>
  {% endfor %}
</div>

---
layout: post
comments: true
title:  "Some Complex Terms in Deep Learning for beginners"
excerpt: "You may have heard about divergences, manifolds, norms etc in deep learning. In this post I will simplify all these terms and briefly look at some of thier common forms and applications found in deep learning literature."
date:   2019-04-08 18:03:00
mathjax: true
---

When I started my journey in machine learning I quite fumbled upon when I encountered terminologies like manifolds, divergences, norms etc. I had a vague and intutive understanding of them. In this post I would like to demistify them for others.

### Metric Space
In layman terms , it is a generalisation of notion of distance. Mathematically it can be defined as follows:
$$ (X,d)  $$ is called a metric space if $$ \exists $$ a mapping $$ d : X \times X \rightarrow \mathbb{R} $$ satisfying the following $$ \big(x,y\big) \rightarrow d\big(x,y\big) $$, d is said to be metric on X. Following holds $$ \forall x, y, z $$ :
 * $ d\big(x,y\big) \geq 0, d\big(x,y\big) = 0 \iff x=y $
 * $ d\big(x,y\big) = d\big(y,x\big) $
 * $ d\big(x,y\big) \leq d\big(x,z\big) + d\big(z,y\big)$

### Norms
Norms can be viewed as generalisation of notion of length. Let $V$ be a k-Vector Space. A map $ \text{$ \lVert $  $\rVert$} : V \rightarow \mathbb{R}$ is called a norm. 
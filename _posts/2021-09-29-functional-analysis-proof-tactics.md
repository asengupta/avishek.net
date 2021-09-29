---
title: "Functional Analysis and General Mathematical Proof Tactics"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics"]
draft: false
---

This article represents a collection of my ongoing notes on proof tactics I've found useful when I've been stuck trying to solve proof exercises. I aim to continue documenting these in as much detail as possible. For now, here is a sketch of some of them.

- Work backwards from what the exercise requires you to prove.
- For **Triangle Inequality** proofs, look for opportunities to break up $$\vert x-y \vert$$ into $$\vert x-z+z-y\vert \leq \vert x-z \vert + \vert z-y \vert $$.
- **Infimum (greatest lower bound) and supremum (least upper bound) of a set do not need to exist inside a set.** Thus, the infimum/supremum of two sets can be the same without those sets intersecting. Use infinite sequences with infimum/supremum outside of the set to compare against a set (which contains the infimum/supremum) to prove infimums/supremums equal without set intersections.
- Use variations of $$(x_k)=\sum\frac{1}{n}$$ to demonstrate **convergence of a sequence, but non-convergence of the corresponding series**.
  - Use multiples of $$(x_k)=\sum\frac{1}{n}$$, like $$(x_k)=\sum\frac{2^n}{n}$$ and break each term into $$\frac{1}{2^n}$$ number of $$\frac{1}{n}$$'s to demonstrate convergence of a sequence, but non-convergence of the corresponding series (or any multiple or power thereof).
- To rearrange terms in maximum expressions like $$\text{max}(a+b, c+d)$$, remember that: $$\text{max}(a+b, c+d) \leq \text{max}(a,c)+\text{max}(b,d)$$.
- To **convert sums into products** (or vice-versa), consider trying **Hölder's Inequality** (or the simpler **Cauchy-Schwarz Inequality**) on sums of indexed products:  
  $$x_1y_1+x_2y_2+...=\sum x_i y_i \leq {\left(\sum x_i^p\right)}^\frac{1}{p}{\left(\sum y_i^p\right)}^\frac{1}{p}$$
- If a constant term appears in the numerator of one term and the denominator of another, like $$\frac{x}{\lambda}+\lambda y$$, try to select $$\lambda=\sqrt{\frac{x}{y}}$$, so that you can get $$\sqrt{xy}+\sqrt{xy}=2\sqrt{xy}$$.
- Use loose equalities, i.e., if $$x+y \leq z, x,y,z \geq 0$$, then $$x \leq z$$.

---
title: "Functional Analysis Exercises 6 : Completion of Metric Spaces"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics", "Kreyszig"]
draft: false
---

This post lists solutions to the exercises in the **Completion of Metric Spaces section 1.6** of *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**. This is a work in progress, and proofs may be refined over time.

#### 1.6.1 Show that if a subspace $$Y$$ of a metric space consists of finitely manypoints, then $$Y$$ is complete.
**Proof:**

By definition, a limit point $$L$$ of the set $$Y$$ has at least one point $$x \neq L$$ within every neightbourhood $$\epsilon$$. Since $$Y$$ has a finite number of points, then it has no limit points, and thus (vacuously) contains all its limit points.

Thus, $$Y$$ is a complete metric subspace.

$$\blacksquare$$

---

#### 1.6.2 What is the completion of $$(X,d)$$, where $$X$$ is the set of all rational numbers and $$d(x,y)=\vert x-y \vert$$?

**Proof:**
The completion of $$(X,d)$$, where $$X$$ is the set of all rational numbers and $$d(x,y)=\vert x-y \vert$$, is $$\mathbb{R}$$, since every real number is the limit of a sequence of rational numbers.

$$\blacksquare$$

---

#### 1.6.3. What is the completion of a discrete metric space $$X$$? (Cf. 1.1-8.).

**Proof:**
A discrete metric space $$X$$ has no limit points, since no point in it has at least one point in every neighbourhood $$\epsilon$$. Thus, it vacuously contains all its limit points. Thus, the completion of the discrete metric space $$X$$ is itself.

$$\blacksquare$$

---

#### 1.6.4 If $$X_1$$ and $$X_2$$ are isometric and $$X_1$$ is complete, show that $$X_2$$ is complete.
**Proof:**

Assume $$x,y \in X_1$$, and let $$T:X_1 \rightarrow X_2$$. Since $$X_1$$ and $$X_2$$ are isometric, we have:

$$
d(x,y)=\bar{d}(Tx,Ty)
$$

We know that $$X_1$$ is complete: let us assume, for an arbitrary $$\epsilon$$, a point $$x_1 \in X_1$$ lying in the $$\epsilon$$-neighbourhood of a limit point $$x \in X_1$$. Then, we have:

$$
d(x,x_1)=d(Tx,Tx_1) < \epsilon
$$

Thus, for an arbitrary $$\epsilon$$, there is a point $$Tx \in X_2$$ which has a point $$Tx_1$$ in its $$\epsilon$$-neighbourhood as well. Thus, $$Tx$$ is a limit point of $$X_2$$ as well. Since $$Tx \in X_2$$ for all $$x \in X_1$$, $$X_2$$ contains all its limit points as well.

Thus, $$X_2$$ is complete.

$$\blacksquare$$

---

#### 1.6.5 (Homeomorphism) A homeomorphism is a continuous bijective mapping $$T: X \rightarrow Y$$ whose inverse is continuous; the metric spaces $$X$$ and $$Y$$ are then said to be homeomorphic. (a) Show that if $$X$$ and $$Y$$ are isometric, they are homeomorphic. (b) Illustrate with an example that a complete and an incomplete metric space may be homeomorphic.
**Proof:**

$$\blacksquare$$

---

#### 1.6.6 Show that $$C[0,1]$$ and $$C[a,b]$$ are isometric.
**Proof:**


$$\blacksquare$$

---

#### 1.6.7 If $$(X,d)$$ is complete, show that $$(X,\tilde{d})$$, where $$\tilde{d} = d/(l + d)$$, is complete.
**Proof:**

$$\blacksquare$$

---

#### 1.6.8 Show that in Prob. 7, completeness of $$(X,\tilde{d})$$ implies completeness of $$(X,d)$$.
**Proof:**


$$\blacksquare$$

---

#### 1.6.9  If $$(x_n)$$ and $$(x_n')$$ in $$(X,d)$$ are such that (1) holds and $$x_n \rightarrow l$$, show that $$(x_n')$$ converges and has the limit $$l$$.
**Proof:**


$$\blacksquare$$

---

#### 1.6.10  If $$(x_n)$$ and $$(x_n')$$ are convergent sequences in a metric space $$(X,d)$$ and have the same limit $$l$$, show that they satisfy (1).
**Proof:**

$$\blacksquare$$

---

#### 1.6.11   Show that (1) defines an equivalence relation on the set of all Cauchy sequences of elements of $$X$$.
**Proof:**


$$\blacksquare$$

---

#### 1.6.12   If $$(x_n)$$ is Cauchy in $$(X,d)$$ and $$(x_n')$$ in $$X$$ satisfies (1), show that $$(x_n')$$ is Cauchy in $$X$$.
**Proof:**


$$\blacksquare$$

---

#### 1.6.13   (pseudometric) A finite pseudometric on a set X is a function d: X x X ~ R satisfying (Ml), (M3), (M4), Sec. 1.1, and (M2*) d(x,x)=O. What is the difference between a metric and a pseudometric? Show that d(x, y) = 1{;1 - Till defines a pseudometric on the set of all ordered pairs of real numbers, where x = ({;1. {;2), y = (1)1. 1)2)' (We mention that some authors use the term semimetric instead of pseudometric.)

**Proof:**

$$\blacksquare$$

---

#### 1.6.14  Does d(x, y)= fIX(t)-y(t)1 dt define a metric or pseudometric on X if X is (i) the set of all real-valued continuous functions on [a, b], (ii) the set of all real-value Riemann integrable functions on [a, b]?
**Proof:**

$$\blacksquare$$

---

#### 1.6.15  If (X, d) is a pseudometric space, we call a set B(xo; r) = {x E X I d(x, xo) < r} (r>O) an open ball in X with center Xo and radius r. (Note that this is analogous to 1.3-1.) What are open balls of radius 1 in Prob. 13?

$$\blacksquare$$

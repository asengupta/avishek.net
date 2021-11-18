---
title: "Functional Analysis Exercises 9 : Further Properties of Normed Spaces"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics", "Kreyszig"]
draft: false
---

This post lists solutions to the exercises in the **Further Properties of Normed Spaces section 2.3** of *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**. This is a work in progress, and proofs may be refined over time.

### Notes
The requirements for a space to be a normed space are:

- **(N1)** **Nonnegativity**, i.e., $$\|x\| \geq 0, x \in X$$
- **(N2)** **Zero norm** implies **zero vector** and vice versa, i.e., $$\|x\|=0 \Leftrightarrow x=0, x \in X$$
- **(N3)** **Linearity** with respect to **scalar multiplication**, i.e., $$\|\alpha x\|=\vert \alpha \vert \|x\|, x \in X, \alpha \in \mathbb{R}$$
- **(N4)** **Triangle Inequality**, i.e., $$\|x+y\| \leq \|x\| + \|y\|, x,y \in X$$

#### 2.2.1. Show that $$c \in l^\infty$$ is a vector subspace of $$l^\infty$$ (cf. 1.5-3) and so is $$c_0$$, the space of all sequences of scalars converging to zero.

**Proof:**


$$\blacksquare$$

---

#### 2.2.2. Show that $$c_0$$ in Prob. 1 is a *closed* subspace of $$l^\infty$$, so that $$c_0$$ is complete by 1.5-2 and 1.4-7.

**Proof:**


$$\blacksquare$$

---

#### 2.2.3. In $$l^\infty$$, let $$Y$$ be the subset of all sequences with only finitely many nonzero terms. Show that $$Y$$ is a subspace of $$l^\infty$$ but not a closed subspace.

**Proof:**


$$\blacksquare$$

---

#### 2.2.4. (Continuity of vector space operations) Show that in a normed space $$X$$, vector addition and multiplication by scalars are continuous operations with respect to the norm; that is, the mappings defined by $$(x,y) \mapsto x+y$$ and $$(\alpha, x) \mapsto \alpha x$$ are continuous.

**Proof:**


$$\blacksquare$$

---

#### 2.2.5. Show that $$x_n \rightarrow x$$ and $$y_n \rightarrow y$$ implies $$x_n + y_n \rightarrow x + y$$. Show that $$\alpha_n \rightarrow \alpha$$ and $$x_n \rightarrow x$$ implies $$\alpha_n x_n \rightarrow \alpha x$$.

**Proof:**


$$\blacksquare$$

---

#### 2.2.6. Show that the closure $$\bar{Y}$$ of a subspace $$Y$$ of a normed space $$X$$ is again a vector subspace.

**Proof:**


$$\blacksquare$$

---

#### 2.2.7. (Absolute convergence) Show that convergence of $$\|y_1\| + \|y_2\| + \|y_3\| + \cdots$$ may not imply convergence of $$y_1 +y_2 + y_3 + \cdots$$. Hint. Consider $$Y$$ in Prob. 3 and $$(y_n)$$, where $$y_n = (\eta_j^{(n)}), \eta_n^{(n)} =1/n^2, \eta_j^{(n)} = 0$$ for all $$j \neq n$$.

**Proof:**


$$\blacksquare$$

---

#### 2.2.8. If in a normed space $$X$$, absolute convergence of any series always implies convergence of that series, show that $$X$$ is complete.

**Proof:**


$$\blacksquare$$

---

#### 2.2.9. Show that in a Banach space, an absolutely convergent series is convergent.

**Proof:**


$$\blacksquare$$

---

#### 2.2.10. (Schauder basis) Show that if a normed space has a Schauder basis, it is separable.

**Proof:**


$$\blacksquare$$

---

#### 2.2.11. Show that $$(e_n)$$, where $$en = (\delta_{nj})$$, is a Schauder basis for $$l^p$$, where $$1 \leq p< +\infty$$.

**Proof:**


$$\blacksquare$$

---

#### 2.2.12. (Seminorm) A seminorm on a vector space $$X$$ is a mapping $$p: X \rightarrow \mathbb{R}$$ satisfying **(N1)**, **(N3)**, **(N4)** in Sec. 2.2. (Some authors call this a pseudonorm.) Show that  

$$
p(0)= 0, \\ 
|p(y) - p(x)| \leq p(y-x).
$$

  **(Hence if $$p(x) = 0$$ implies $$x = 0$$, then $$p$$ is a norm.)**

**Proof:**


$$\blacksquare$$

---

#### 2.2.13. Show that in Prob. 12, the elements $$x \in X$$ such that $$p(x) = 0$$ form a subspace $$N$$ of $$X$$ and a norm on $$X/N$$ (cf. Prob. 14, Sec. 2.1) is defined by $${\|\hat{x}\|}_0=p(x)$$, where $$x \in \hat{x}$$ and $$\hat{x} \in X/N$$.

**Proof:**


$$\blacksquare$$

---

#### 2.2.14. (Quotient space) Let Y be a closed subspace of a normed space(X, 11·11). Show that a norm 11·110 on XIY (cf. Prob. 14, Sec. 2.1) is defined by  

$$
{\|\hat{x}\|}_0 = \text{inf }_{x \in \hat{x}} \|x\|
$$

**where $$\hat{x} \in X/Y$$, that is, $$\hat{x}$$ is any coset of $$Y$$.**

**Proof:**


$$\blacksquare$$

---

#### 2.2.15. (Product of normed spaces) If $$(X_1, {\|\bullet\|}_1)$$ and $$(X_2, {\|\bullet\|}_2)$$ are normed spaces, show that the product vector space $$X = X_1 \times X_2$$ (cf. Prob. 13, Sec. 2.1) becomes a normed space if we define

$$
\|x\|=\text{max }({\|x_1\|}_1, {\|x_2\|}_2)
$$

**Proof:**


$$\blacksquare$$

---

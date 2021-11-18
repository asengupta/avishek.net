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

#### 2.2.1. Show that $$c \subset l^\infty$$ is a vector subspace of $$l^\infty$$ (cf. 1.5-3) and so is $$c_0$$, the space of all sequences of scalars converging to zero.

**Proof:**

$$c \in l^\infty$$ is the space of all complex convergent sequences. Let $$x_n \rightarrow x$$ and $$y_n \rightarrow y$$, such that $$x_n,y_n \in \mathbb{C}$$.

Then, $$\forall \epsilon>0, \exists M, N \in \mathbb{N}$$, such that $$\|x_m-x\|<\epsilon$$ and $$\|y_n-y\|<\epsilon$$, for all $$m>M, n>N$$. Take $$N_0=\text{max }(M_0,N_0)$$, so that $$\|x_n-x\|<\epsilon$$ and $$\|y_n-y\|<\epsilon$$, for all $$n>N_0$$.

$$
\|\alpha x_n + \beta y_n - (\alpha x + \beta y)\| = \|\alpha x_n - \alpha x + \beta y_n - \beta y)\| \\
\|\alpha x_n + \beta y_n - (\alpha x + \beta y)\| \leq \|\alpha x_n - \alpha x\| + \|\beta y_n - \beta y\| \\
= |\alpha|\|x_n - x\| + |\beta|\|y_n - y\| < \alpha \epsilon + \beta \epsilon = (\alpha + \beta) \epsilon
$$

$$(\alpha + \beta) \epsilon$$ can be made as small as possible. In the limit $$n \rightarrow \infty$$, we thus get: $$\|\alpha x_n + \beta y_n - (\alpha x + \beta y)\| \rightarrow \alpha x + \beta y \in c \subset l^\infty$$.

Thus, $$c$$ is a vector subspace.

$$\blacksquare$$

For $$c_0$$, we have $$x=y=0$$, thus $$\alpha x + \beta y=0$$, and thus $$\alpha x_n + \beta y_n \rightarrow 0$$ as $$n \rightarrow \infty$$.

Thus, $$c_0$$ is a vector subspace.

$$\blacksquare$$

---

#### 2.2.2. Show that $$c_0$$ in Prob. 1 is a *closed* subspace of $$l^\infty$$, so that $$c_0$$ is complete by 1.5-2 and 1.4-7.

**Proof:**

Consider a Cauchy sequence $$(x_n) \in c_0$$. Then, $$\forall \epsilon>0, \exists N \in \mathbb{N}$$, such that $$d(x_m-x_n)<\epsilon$$, for all $$m,n>N$$.

This implies that:

$$
\text{sup }d(x_j^m, x_j^n)<\epsilon \\
d(x_j^m, x_j^n)<\epsilon
$$

Thus, for a fixed $$j$$, the sequence of scalars $$x_j^1, x_j^2, \cdots$$ is Cauchy. Since scalars are real numbers, and $$\mathbb{R}$$ is complete, the sequence converges, say to $$x_j$$, that is, $$\forall \epsilon>0, \exists M \in \mathbb{N}$$ such that $$\|x_j^m-x_j\|<\epsilon$$ for all $$m>M$$. This holds for every $$j$$, yielding a sequence $$x_1, x_2, x_3, \cdots$$.

Of course, $$x_1^m, x_2^m, x_3^m, \cdots$$ is also Cauchy since it is a convergent sequence (converging to zero), thus we have $$\forall \epsilon>0, \exists N \in \mathbb{N}$$ such that $$\|x_j^m-0\|<\epsilon$$ for all $$j>N$$
We have:

$$
\|x_j-0\| = \|x_j-x_j^m + x_j^m - 0\| \leq \|x_j-x_j^m\| + \|x_j^m - 0\| < \epsilon + \epsilon = 2 \epsilon
$$

Thus $$(x_j) \rightarrow 0$$, and $$(x_j) \in c_0$$. Since this holds for any arbitrary Cauchy sequence in $$c_0$$, it follows that $$c_0$$ contains all its limits, and is thus a closed subspace.

$$\blacksquare$$

---

#### 2.2.3. In $$l^\infty$$, let $$Y$$ be the subset of all sequences with only finitely many nonzero terms. Show that $$Y$$ is a subspace of $$l^\infty$$ but not a closed subspace.

**Proof:**

Let $$Y$$ be the subset of all sequences with only finitely many nonzero terms.

Let $$(x_n)=x_1, x_2, \cdots, x_m, 0, 0, \cdots$$ and let $$(y_n)=y_1, y_2, \cdots, y_n, 0, 0, \cdots$$. Without loss of generality, assume $$m<n$$.
It is clear that $$\delta(x_n)=\text{max }(x_1, x_2, \cdots, x_m) < \infty$$ and $$\delta(y_n)=\text{max }(y_1, y_2, \cdots, y_n) < \infty$$, thus $$(x_n), (y_n) \in YZ \subset l^\infty$$
Then, we have:

$$
\alpha(x_n) + \beta(y_n)=\alpha x_1 + \beta y_1, \alpha x_2 + \beta y_2, \cdots, \alpha x_m + \beta y_m, \beta y_{m+1}, \cdots, \beta y_n, 0, 0, \cdots \\
\Rightarrow \delta[\alpha(x_n) + \beta(y_n)]=\text{max }(\alpha x_1 + \beta y_1, \alpha x_2 + \beta y_2, \cdots, \alpha x_m + \beta y_m, \beta y_{m+1}, \cdots, \beta y_n) < \infty \\
\Rightarrow \alpha(x_n) + \beta(y_n) \in Y \subset l^\infty
$$

Thus $$Y$$ is a subspace of $$l^\infty$$.

Let there be a Cauchy sequence in $$Y$$, where
$$
y_n=\begin{cases}
1/j & \text{if } j \leq n \\
0 & \text{if } j > n
\end{cases}
$$

Assume $$m<n$$. Then $$d(x_m,x_n)=\text{sup } d(x_m^i, x_n^i)=\frac{1}{m+1}$$. Then as $$m \rightarrow \infty$$, $$\text {lim }_{m \rightarrow \infty} d(x_m,x_n) = 0$$, but this limit has more nonzero terms than any sequence in $$Y$$ and is thus not contained in $$Y$$. Thus, $$Y$$ is not complete.

$$\blacksquare$$

---

#### 2.2.4. (Continuity of vector space operations) Show that in a normed space $$X$$, vector addition and multiplication by scalars are continuous operations with respect to the norm; that is, the mappings defined by $$(x,y) \mapsto x+y$$ and $$(\alpha, x) \mapsto \alpha x$$ are continuous.

**Proof:**

Assume a normed space, with $$\|x-x_0\|<\epsilon$$ and $$\|y-y_0\|<\epsilon$$

Vector addition will be continuous if $$\forall \epsilon>0, \exists \delta>0$$, such that $$\|x-x_0\| < \delta, \|y-y_0\| < \delta \Rightarrow \|f(x,y)-f(x_0,y_0)\| < \epsilon$$.

Then, we have:

$$
\|x+y-(x_0+y_0)\|=\|(x-x_0)+(y-y_0)\| \leq \|x-x_0\| + \|y-y_0\| < 2 \epsilon
$$

$$2 \epsilon$$ can be made as small as needed, thus vector addition is continuous in normed space.

$$\blacksquare$$

Scalar multiplication will be continuous if $$\forall \epsilon>0, \exists \delta>0$$, such that $$\|x-x_0\| < \delta, \|\alpha - \alpha_0\| < \delta \Rightarrow \|\alpha x - \alpha x_0\| < \epsilon$$.

We'd like to express $$\alpha x-\alpha_0 x_0$$ using some combination of $$(\alpha-\alpha_0)$$ and $$(y-y_0)$$. As a preliminary test, let's see what terms fall out of the product $$(x-x_0)(\alpha-\alpha_0)$$.

Then, we have:

$$
(\alpha x-\alpha_0)(x-x_0)=\alpha x + \alpha_0 x_0 - \alpha_0 x - \alpha x_0
$$

Then we can write $$\alpha x - \alpha_0 x_0$$ as:

$$
\alpha x - \alpha_0 x_0 = \alpha x + \alpha_0 x_0 - \alpha_0 x - \alpha x_0 - \alpha_0 x_0 - \alpha_0 x_0 + \alpha_0 x + \alpha x_0 \\
= (\alpha - \alpha_0)(x-x_0) + \alpha_0 (x-x_0) + x_0(\alpha-\alpha_0)
$$

Therefore, we can write:

$$
\|\alpha x - \alpha_0 x_0\|=\|(\alpha - \alpha_0)(x-x_0) + \alpha_0 (x-x_0) + x_0(\alpha-\alpha_0)\| \\
\|\alpha x - \alpha_0 x_0\| \leq \|\alpha - \alpha_0\|\|x-x_0\| + |\alpha_0| \|x-x_0\| + |x_0|\|\alpha-\alpha_0\| < \epsilon^2 + |\alpha_0| \epsilon + |x_0| \epsilon
$$

The quantity on the RHS can be made as small as possible, and thus scalar multiplication is continuous in normed space.

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

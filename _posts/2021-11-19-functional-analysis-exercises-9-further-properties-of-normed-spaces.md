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

(See Above)

---

#### 2.2.6. Show that the closure $$\bar{Y}$$ of a subspace $$Y$$ of a normed space $$X$$ is again a vector subspace.

**Proof:**

Let $$x,y \in \bar{S}$$. Thus, $$\forall r>0$$, we have $$B(x,r) \cap S \neq \emptyset$$ and $$B(y,r) \cap S \neq \emptyset$$. Pick $$r<\epsilon$$, then $$\|x-x_0\|<\epsilon/2$$ and $$\|y-y_0\|<\epsilon/2$$.

Then, we have:

$$
\|\alpha x + \beta y - (\alpha x_0 + \beta y_0)\|=\|\alpha x - \alpha x_0 + \beta y - \beta y_0\| \\
\leq \|\alpha x - \alpha x_0\| + \|\beta y - \beta y_0\| < \epsilon/2 + \epsilon/2 = \epsilon
$$

This holds for every $$\epsilon>0$$, thus $$B(\alpha x + \beta y, r) \cap Y \neq \emptyset$$, since every open ball around it contains $$\alpha x_0 + \beta y_0 \in Y$$. Thus, $$\bar{Y}$$ is an vector subspace.

$$\blacksquare$$

---

#### 2.2.7. (Absolute convergence) Show that convergence of $$\|y_1\| + \|y_2\| + \|y_3\| + \cdots$$ may not imply convergence of $$y_1 +y_2 + y_3 + \cdots$$. Hint. Consider $$Y$$ in Prob. 3 and $$(y_n)$$, where $$y_n = (\eta_j^{(n)}), \eta_n^{(n)} =1/n^2, \eta_j^{(n)} = 0$$ for all $$j \neq n$$.

**Proof:**

We have:

$$
y_1=\frac{1}{2},0,0, \cdots \\
y_2=0,\frac{1}{2^2},0, \cdots \\
y_3=0,0,\frac{1}{2^3}, \cdots \\
\vdots \\
y_n=0,0,0, \cdots, \frac{1}{2^n}, \cdots \\
\vdots
$$

Correspondingly, the norms are:

$$
\|y_1\|=\frac{1}{2} \\
\|y_2\|=\frac{1}{2^2} \\
\|y_3\|=\frac{1}{2^3} \\
\|\vdots \\
\|y_n\|=\frac{1}{2^n} \\
\vdots
$$

Then $$\|y_1\| + \|y_2\| + \|y_3\| + \cdots$$ is a convergent series.

The partial sum $$s_n$$ is defined as:

$$
s_n=\frac{1}{2},\frac{1}{2^2}, \cdots, \frac{1}{2^n}, 0, 0, \cdots
$$

As $$n \rightarrow \infty$$, we get:

$$
\lim\limits_{n \rightarrow \infty} s_n=\frac{1}{2},\frac{1}{2^2}, \cdots
$$

Thus, this is the space of sequences with finite non-zero terms.

We have $$\|s_n-s_{m}\|=\text{sup }\vert s_{n(i)}-s_{n(i)}\vert$$ (note that $$s_n$$ is a sequence, being the sum of sequences). Assume $$m<n$$, then $$\|s_n-s_m\|=\frac{1}{2^{m+1}}$$. This can be made as small as possible, and thus $$(s_n)$$ is Cauchy.

Choose $$s=(\frac{1}{2^n})$$, and we have:

$$
\|s-s_n\|=\frac{1}{2^{n+1}}
$$

which goes to zero in the limit $$n \rightarrow \infty$$, thus $$s$$ is a limit of $$(s_n)$$.  However, $$s$$ has infinitely many terms, and thus is not in the space of sequences with finite non-zero terms.

Thus, $$y_1 +y_2 + y_3 + \cdots$$ does not converge even though $$\|y_1\| + \|y_2\| + \|y_3\| + \cdots$$ converges.

$$\blacksquare$$

---

#### 2.2.8. If in a normed space $$X$$, absolute convergence of any series always implies convergence of that series, show that $$X$$ is complete.

**Proof:**

Take a Cauchy sequence $$(x_n)$$. Pick $$N_k$$ such that $$\|x_m-x_n\|<\frac{1}{2^k}$$ for all $$m,n \geq N_k$$. Pick the corresponding $$y_k=x_{N_k}$$ from $$(x_n)$$. Then note that $$\|y_k-y_{k+1}\| < \frac{1}{2^k}$$.

Then $$\displaystyle \sum\limits_{k=1}^\infty \|y_{k+1}-y_k\| < \sum\limits_{k=1}^\infty \frac{1}{2^k} = 1$$. Thus, this series is absolutely convergent, and is by assumption, convergent. That is, $$\displaystyle \sum\limits_{k=1}^n y_{k+1}-y_k$$ is convergent, i.e., it converges to some element, say $$x$$.

Now, we have:

$$
\displaystyle \sum\limits_{k=1}^n y_{k+1}-y_k = \displaystyle \sum\limits_{k=1}^n x_{N_{k+1}}-x_{N_k}\\ 
=x_{N_{n+1}}-x_{N_1}
$$

In the limit of $$n \rightarrow \infty$$, this expression tends to $$x$$, that is:

$$
\lim\limits_{n \rightarrow \infty} x_{N_{n+1}}-x_{N_1} = x \\
\lim\limits_{n \rightarrow \infty} x_{N_{n+1}} = x + x_{N_1}
$$

Thus, this limit exists and since $$(x_n)$$ was an arbitrary Cauchy sequence, it converges to $$x$$. Thus $$X$$ is complete.

$$\blacksquare$$

---

#### 2.2.9. Show that in a Banach space, an absolutely convergent series is convergent.

**Proof:**

Let there be an absolutely convergent series $$\displaystyle \sum\limits_{i=1}^\infty \|x_k\|<\infty$$. Since it is convergent, it is also Cauchy, thus we have:

$$
\displaystyle \sum\limits_{i=m}^n |x_k|<\epsilon
$$

By the **Triangle Inequality**, we have:

$$
\displaystyle |\sum\limits_{i=m}^n x_k| \leq \sum\limits_{i=m}^n |x_k| \\
\displaystyle |\sum\limits_{i=m}^n x_k| = s_n - s_{m-1} < \epsilon
$$

Since the space is Banach, $$(s_n)$$ is a convergent sequence.

$$\blacksquare$$

---

#### 2.2.10. (Schauder basis) Show that if a normed space has a Schauder basis, it is separable.

**Proof:**

A Schauder basis of a space $$X$$ is a sequence $$(e_n)$$ such that $$\|x-(\alpha_1 e_1 + \alpha_2 e_2 + \alpha_3 e_3 + \cdots + \alpha_n e_n)\| \rightarrow 0, x \in X$$ as $$n \rightarrow \infty$$.

A space is separable if it has a countable subset which is dense in this space.

The partial sum of a Schauder basis is represented as $$s_n=\alpha_1 e_1 + \alpha_2 e_2 + \alpha_3 e_3 + \cdots + \alpha_n e_n$$.

This implies that $$\forall \epsilon > 0, \exists N$$ such that $$\|x-s_n\|<\epsilon$$ for all $$n>N$$. Thus every neighbourhood of $$x$$ has a Schauder representation.

Since $$\alpha_n \in \mathbb{R}$$, there exists a $$\beta_n \in \mathbb{Q}$$, such that $$\|\alpha_n-\beta_n\|<\epsilon$$.

**(Prove that $$Y=\sum\limits_{i=1}^n \beta_i e_i$$ is countable).**

Denote $$s_n'=\beta_1 e_1 + \beta_2 e_2 + \beta_3 e_3 + \cdots + \beta_n e_n$$, then we have:

$$
\|s_n-s_n'\|=\|(\alpha_1-\beta_1) e_1 + (\alpha_2-\beta_2) e_2 + (\alpha_3-\beta_3) e_3 + \cdots + (\alpha_n-\beta_n) e_n\| \\
\leq \|(\alpha_1-\beta_1) e_1\| + \|(\alpha_2-\beta_2) e_2\| + \|(\alpha_3-\beta_3) e_3\| + \cdots + \|(\alpha_n-\beta_n) e_n\| \\
= |\alpha_1-\beta_1| \|e_1\| + |\alpha_2-\beta_2| \|e_2\| + |\alpha_3-\beta_3| \|e_3\| + \cdots + |\alpha_n-\beta_n| \|e_n\| \\
= \epsilon \|e_1\| + \epsilon \|e_2\| + \epsilon \|e_3\| + \cdots + \epsilon \|e_n\| \\
= \epsilon (\|e_1\| + \|e_2\| +  \|e_3\| + \cdots + \|e_n\|)=K \epsilon
$$

(Note that even though $$K$$ depends upon how far the Schauder basis is expanded, for a fixed Schauder basis, a rational number can be chosen arbitrarily closer to the real number without resorting to going further along the Schauder basis).

$$
\|x-s_n'\| \leq \|x-s_n\| + \|s_n-s_n'\| < \epsilon + K \epsilon = (K+1) \epsilon
$$

This can be made as small as needed, and thus $$Y$$ (countable) is dense in this normed space, and hence the space is separable.

$$\blacksquare$$

---

#### 2.2.11. Show that $$(e_n)$$, where $$e_n = (\delta_{nj})$$, is a Schauder basis for $$l^p$$, where $$1 \leq p< +\infty$$.

**Proof:**

$$l^p$$ is the space of all bounded sequences. This implies that:

$$
\sum\limits_{i=1}^\infty {|x_i|}^p=K<\infty
$$

Equivalently,

$$
\lim\limits_{n \rightarrow \infty} \sum\limits_{i=1}^n {|x_i|}^p=K
$$

The norm is defined as $$
{\|x\|}_p={\left(\sum\limits_{i=1}^\infty {|x_i|}^p\right)}^{1/p}
$$

Assume the sequence is $$x_1, x_2, x_3, \cdots$$.
Then, we have:

$$
x_1 e_1=x_1, 0, 0, 0, \cdots \\
x_2 e_2=0, x_2, 0, 0, \cdots \\
x_3 e_3=0, x_2, 0, 0, \cdots \\
\vdots \\
x_n e_n=0, 0, 0, 0, \cdots, x_n, 0, 0, \cdots \\
$$

Then, we get:

$$
s_n=\sum\limits_{i=1}^n x_i e_i = x_1, x_2, \cdots, x_n, 0, 0, \cdots \\
x = s_n + \sum\limits_{i=n+1}^\infty x_i e_i \\
x-s_n = \sum\limits_{i=n+1}^\infty x_i e_i \\
\|x-s_n\| = {(\sum\limits_{i=n+1}^\infty {|x_i|}^p)}^{1/p}
$$

We know that:

$$
\sum\limits_{i=1}^\infty {|x_i|}^p=K=\sum\limits_{i=1}^n {|x_i|}^p + \sum\limits_{i=n+1}^\infty {|x_i|}^p
$$

Taking limits on both sides for $$n \rightarrow \infty$$, we get:

$$
\lim\limits_{n \rightarrow \infty} \underbrace{\sum\limits_{i=1}^n {|x_i|}^p}_\text{Partial Sum} + \lim\limits_{n \rightarrow \infty} \sum\limits_{i=n+1}^\infty {|x_i|}^p=K \\
K + \lim\limits_{n \rightarrow \infty} \sum\limits_{i=n+1}^\infty {|x_i|}^p = K \\
\lim\limits_{n \rightarrow \infty} \sum\limits_{i=n+1}^\infty {|x_i|}^p = 0 \\
$$

Thus $$\|x-s_n\| \rightarrow 0$$ as $$n \rightarrow \infty$$.

Thus, $$e_n = (\delta_{nj})$$ is a Schauder basis for $$l^p$$.

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

#### 2.2.14. (Quotient space) Let Y be a closed subspace of a normed space $$(X, \|\bullet\|)$$. Show that a norm $$\|\bullet\|_0$$ on $$X/Y$$ (cf. Prob. 14, Sec. 2.1) is defined by  

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

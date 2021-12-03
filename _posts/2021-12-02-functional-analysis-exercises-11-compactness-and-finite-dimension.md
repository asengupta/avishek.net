---
title: "Functional Analysis Exercises 11 : Compactness and Finite Dimension"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics", "Kreyszig"]
draft: false
---

This post lists solutions to the exercises in the **Compactness and Finite Dimension section 2.5** of *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**. This is a work in progress, and proofs may be refined over time.


#### 2.5.1. Show that $$\mathbb{R}^n$$ and $$\mathbb{C}^n$$ are not compact.

**Proof:**

$$\mathbb{R}^n$$ has a sequence $$x_n=1,2,3,4, \cdots$$ which does not have a convergent subsequence, since every $$d(x_m,x_n) \geq 1$$ for all $$m,n \in \mathbb{N}$$. The sequence exists in $$\mathbb{C}^n$$, implying that every sequence in that space does not have a convergent subsequence.

Thus $$\mathbb{R}^n$$ and $$\mathbb{C}^n$$ are not compact.

$$\blacksquare$$

---

#### 2.5.2. Show that a discrete metric space $$X$$ (cf. 1.1-8) consisting of infinitely many points is not compact.

**Proof:**

The discrete metric space $$X$$ has the distance metric as:

$$d(x,y)=\begin{cases}
0 & \text{if } x=y \\
1 & \text{if } x \neq y
\end{cases}$$

Assume any sequence $$(x_n) \subset X $$ which does not repeat its elements. Then $$d(x_m,x_n)=1$$ for all $$m,n: m \neq n, m,n \in \mathbb{N}$$. Thus, this sequence has no convergent subsequence. Thus the discrete metric space $$X$$ is not compact.

$$\blacksquare$$

---

#### 2.5.3. Give examples of compact and noncompact curves in the plane $$\mathbb{R}^2$$.

**Answer:**

$$y=x$$ is not compact because it is not bounded.  
$$y=\sin x, x \in [0, 2\pi]$$ is compact because $$[0,2\pi]$$ is compact and sine is a continuous function, and we know that continuous functions map compact sets to compact sets.

---

#### 2.5.4.  Show that for an infinite subset $$M$$ in Ihe space $$s$$ (cf. 2.2-8) to be compact, it is necessary that there arc numhers $$\gamma_1, \gamma_2, \cdots$$ such that for all $$x=(\xi_k(x)) \in M$$ we have $$\vert \xi_k(x) \vert \leq \gamma_k$$. (It can he shown that the condition is also sufficient for the compactness of $$M$$.)

**Proof:**

The metric of $$s$$ is defined by:

$$
d(x,y) = \sum\limits_{j=1}^\infty \frac{1}{2^j} \frac{|\xi_j-\eta_j|}{1+|\xi_j-\eta_j|}
$$


$$\xi_k(x)$$ extracts the $$k$$-th element of the sequence $$x$$.

Assume a sequence $$(p_n) \subset M$$, like so:

$$
p_1 = x^1_1, x^1_2, x^1_3, \cdots, x^1_k, \cdots \\
p_2 = x^2_1, x^2_2, x^2_3, \cdots, x^2_k, \cdots \\
\vdots \\
p_m = x^m_1, x^m_2, x^m_3, \cdots, x^m_k, \cdots \\
\vdots
$$

For fixed $$k$$, we have $$\vert \xi_k(p_m) \vert < \gamma_k$$. Thus $$x^m_k$$ is bounded for fixed $$k$$.

Set $$k=1$$, then $$x^1_1, x^2_1, x^3_1, \cdots$$ is bounded. Thus, by the **Bolzano-Weierstrauss Theorem**, this contains a convergent subsequence. Let this convergent subsequence converge to $$x_1$$. Let $$P_1$$ be the set of sequences which contain these subsequence entries. We can repeat the same exercise for $$k=2$$ and apply it to $$P_1$$ to get $$P_2$$, etc.

We finally get a subsequence of $$(p_n)$$ (call it $$p_{n_j}$$) where, for any given $$k$$, we have a convergent sequence $$(x^{n_j}_k)$$ converging to $$x_k$$. We construct a sequence out of these limits $$p=x_1, x_2, \cdots$$. For any $$p_m$$, we have:

$$
d(p_m, p) = \sum\limits_{j=1}^\infty \frac{1}{2^j} \frac{|x^m_j-x_j|}{1+|x^m_j-x_j|}=\sum\limits_{j=1}^\infty \frac{1}{2^j} \left( 1 - \frac{1}{1+|x^m_j-x_j|} \right)
$$

Take $$\epsilon > 0$$. Assume $$N_1 \in \mathbb{N}$$ such that $$\vert x^m_1-x_1 \vert$$, similarly for $$N_2$$, and so on. Now take $$N=\max{(N_1, N_2, \cdots)}$$. Then for $$m>N$$, we have:

$$
d(p_m, p) = \sum\limits_{j=1}^\infty \frac{1}{2^j} \left( 1 - \frac{1}{1+\epsilon} \right) \\
\lim_{\epsilon \rightarrow 0} d(p_m, p) = 0
$$

Thus, $$p$$ is the limit of the constructed subsequence $$p_{n_j}$$. Thus $$(p_n)$$ has a convergent subsequence. Since $$(p_n)$$ was arbitrary, $$M$$ is compact.

$$\blacksquare$$

---

#### 2.5.5.  (Local compactness) A metric space $$X$$ is said to be locally compact if every point of $$X$$ has a compact neighborhood. Show that $$\mathbb{R}$$ and $$\mathbb{C}$$ and, more generally, $$\mathbb{R}^n$$ and $$\mathbb{C}^n$$ are locally compact.

**Proof:**

Take $$x \in \mathbb{R}$$. Then, for any $$\epsilon>0$$, the set $$[x-\epsilon, x+\epsilon]$$ is closed and bounded and contains the $$\epsilon$$-neighbourhood, and is thus a compact neighbourhood of $$x$$. Thus $$\mathbb{R}$$ is locally compact.

Take $$x \in \mathbb{R}^n$$. For any $$\epsilon>0$$, the closed ball $$\bar{B}(x, \epsilon)$$ contains the open $$\epsilon$$-neighbourhood $$B(x,\epsilon)$$ and is thus a compact neighbourhood of $$x$$. Thus $$\mathbb{R}^n$$ is locally compact.

$$\blacksquare$$

---

#### 2.5.6. Show that a compact metric space $$X$$ is locally compact.

**Proof:**

The neighbourhood of a point $$x$$ is defined as a set which contains an open ball centered on $$x$$ (or open set containing $$x$$).

Since any open ball around $$x \in X$$ is also contained in the set $$X$$, $$X$$ is a neighbourhood of $$x$$. Since $$X$$ is compact, $$x$$ has a compact neighbourhood, implying that $$X$$ is locally compact.

$$\blacksquare$$

---

#### 2.5.7. If $$\dim Y < \infty$$ in Riesz's lemma 2.5-4, show that one can even choose $$\theta = 1$$.

**Proof:**

**(TODO)**

$$\blacksquare$$

---

#### 2.5.8. In Prob. 7, Sec. 2.4, show directly (without using 2.4-5) that there is an $$a > 0$$ such that $$a {\|x\|}_2 \leq \|x\|$$. (Use 2.5-7.)

**Proof:**

Norms are continuous. If $$f(x)=\frac{\|x\|}{ {\|x\|}_2}$$ is also continuous, assuming that $${\|x\|}_2 \neq 0$$. Assume a compact subset $$M \subset X$$; then $$f(M)$$ is also compact, and thus contains a maximum and a minimum value. Specifically, it contains a infimum, call it $$a$$. Then we have:

$$
f(x) = \frac{\|x\|}{ {\|x\|}_2} \geq a \\
{\|x\|} \geq a {\|x\|}_2
$$

If $${\|x\|}_2 =0$$, then obviously $${\|x\|} \geq a {\|x\|}_2 = 0$$.

$$\blacksquare$$

---

#### 2.5.9. If $$X$$ is a compact metric space and $$M \subset X$$ is closed, show that $$M$$ is compact.

**Proof:**

Every sequence in $$X$$ has a convergent subsequence.

Choose an arbitrary sequence $$(x_n) \subset M$$. Since $$(x_n) \subset X$$, it has a convergent subsequence $$(x_{n_k}$$, therefore this subsequence has a limit point $$x$$.

We note that $$(x_{n_k}) \subset M$$.

Since $$M$$ is closed, it contains all its limit points, thus $$x \in M$$. Thus $$M$$ contains the limit of this subsequence as well. Thus the sequence $$(x_n)$$ has a convergent subsequence in $$M$$. Since $$(x_n)$$ is arbitrary, all sequences in $$M$$ have convergent subsequences. Thus, $$M$$ is compact.

$$\blacksquare$$

---

#### 2.5.10. Let $$X$$ and $$Y$$ be metric spaces, $$X$$ compact, and $$T: X \rightarrow Y$$ bijective and continuous. Show that $$T$$ is a homeomorphism (cf. Prob. 5, Sec. 1.6).

**Proof:**

For $$T$$ to be a homeomorphism, its inverse should be continuous.

$$X$$ is compact and $$T$$ is continuous, thus $$T(X)$$ is compact.
Choose any sequence $$(x_n) \subset X$$. Since $$X$$ is compact, $$(x_n)$$ contains a convergent subsequence $$(x_{n_k})$$ which converges to $$x$$. Since continuous functions map convergent sequences to convergent sequences, $$(y_{n_k}) = T((x_{n_k}))$$ is a convergent sequence.

Now we define $$T^{-1}$$ as $$T^{-1}: (y_{n_k}) \mapsto (x_{n_k})$$. We know that if a function is continuous if and only if it maps convergent sequences to convergent sequences.

Since both $$(x_{n_k})$$ and $$(y_{n_k})$$ are convergent sequences, $$T^{-1}$$ is continuous at $$x$$.

Since $$(x_n)$$ was arbitrarily chosen, $$T$$ is a homeomorphism.

$$\blacksquare$$


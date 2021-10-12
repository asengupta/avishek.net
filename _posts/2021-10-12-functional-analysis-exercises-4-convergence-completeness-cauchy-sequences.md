---
title: "Functional Analysis Exercises 4 : Convergence, Cauchy Sequences, and Completeness"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics"]
draft: false
---

This post lists solutions to the exercises in the **Convergence, Cauchy Sequences, and Completeness section 1.4** of *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**. This is a work in progress, and proofs may be refined over time.

#### 1.4.1. (Subsequence) If a sequence $$(x)$$ in a metric space $$X$$ is convergent and has limit $$x$$, show that every subsequence $$(x_{n_k})$$ of $$(x_n)$$ is convergent and has the same limit $$x$$.

**Proof:**

Suppose $$(x_n)$$ is convergent and $$x_n \rightarrow x$$. Let $$(x_{n_k})$$ be a subsequence. Let $$x_{n_m}$$ correspond to $$x_i$$.

Since $$(x_n)$$ is convergent, $$\forall \epsilon>0, \exists N$$, such that $$d(x_n,x)<\epsilon$$ for $$n>N$$. Pick $$p \geq N$$, such that $$x_p$$ exists in $$(x_{n_k})$$ and is identified as $$x_{n_m}$$.

This implies that $$\forall \epsilon>0, \exists M$$, such that $$d(x_{n_m},x)<\epsilon$$.

The above is the definition of the convergence of a sequence to a limit. Thus, $$(x_{n_k})$$ converges to $$x$$.

Alternatively, you can prove that the limit of $$(x_{n_k})$$ is $$x$$ in the following manner.  
Suppose $$x_{n_k} \rightarrow y$$. Then, by the **Triangle Inequality**, we have:

$$
d(x,y) \leq d(x,x_p) + d(x_p,y) \\
\Rightarrow d(x,y) \leq d(x,x_p) + d(x_{n_m},y)
$$


Both $$d(x,x_p)$$ and $$d(x_{n_m},y)$$ can be made as small as possible, since $$(x_n)$$ and $$(x_{n_k})$$ converge, implying that $$d(x,y)$$ is smaller than any positive value. Thus:

$$
d(x,y) \leq d(x,x_p) + d(x_{n_m},y) \\
\Rightarrow d(x,y) < \epsilon_1 + \epsilon_2, \epsilon_1, \epsilon_2 > 0 \\
\Rightarrow d(x,y)=0
$$

Thus, $$x=y$$, i.e., $$x_{n_k}$$ has the same limit as $$(x_n)$$.

$$\blacksquare$$

---

#### 1.4.2. If $$(x_n)$$ is Cauchy and has a convergent subsequence, say, $$x_n \rightarrow x$$, show that $$(x_n)$$ is convergent with the limit $$x$$.

**Proof:**

Suppose $$(x_n)$$ is Cauchy. Let $$(x_{n_k})$$ be a subsequence. Let $$x_{n_m}$$ correspond to $$x_i$$.

- Since $$(x_{n_k})$$ is convergent, $$\forall \epsilon>0, \exists M$$, such that $$d(x_{n_k},x)<\epsilon$$ for $$n>M$$.
- Since $$(x_n)$$ is convergent, $$\forall \epsilon>0, \exists N$$, such that $$d(x_i,x_j)<\epsilon$$ for $$i,j>N$$.

Pick $$N_0=\text{max}(M,N)$$. Then the above two statements become:

- Since $$(x_{n_k})$$ is convergent, $$\forall \epsilon>0, \exists N_0$$, such that $$d(x_{n_i},x)<\epsilon$$ for $$i>N_0$$.
- Since $$(x_n)$$ is convergent, $$\forall \epsilon>0, \exists N_0$$, such that $$d(x_i,x_j)<\epsilon$$ for $$i,j>N_0$$.

(Note that we picked the same $$i$$ for both $$(x_n)$$ and $$(x_{n_k})$$ because any value greater than $$N_0$$ will fulfil the above conditions, so we might as well pick the same index. For any index $$i$$, we can use $$x_i$$ and $$x_{n_i}$$ interchangeably, since they index the same element in both the sequence and the subsequence.)

By the **Triangle Inequality**, we have:

$$
d(x_j,x) \leq d(x_j, x_i) + d(x_i,x) \\
\Rightarrow d(x_j,x) \leq d(x_j, x_i) + d(x_{n_i},x) \\
\Rightarrow d(x_j,x) \leq \epsilon + \epsilon \\
\Rightarrow d(x_j,x) < 2\epsilon
$$

$$\epsilon$$ can be made as small as possible, implying that $$d(x_j,x)$$ is smaller than any positive value. Thus:

$$
d(x_j,x)=0
$$

Hence, $$(x_n)$$ is convergent with the limit $$x$$.

$$\blacksquare$$

---

#### 1.4.3. Show that $$x_n \rightarrow x$$ if and only if for every neighborhood $$V$$ of $$x$$ there is an integer $$n_0$$ such that $$x_n \in V$$ for all $$n > n_0$$.

**Proof:**

Suppose that $$x_n \rightarrow x$$. Then $$\forall \epsilon>0, \exists N_0$$, such that $$d(x_n,x)<\epsilon$$ for all $$n>N_0$$. This implies that a neighbourhood $$V_\epsilon$$ of $$x$$ exists, which contains all $$x_{n>N_0}$$. Since there are an infinite number of values for $$\epsilon$$, it follows that this applies to every neighbourhood of $$x$$.

Conversely, suppose that for every neighborhood $$V$$ of $$x$$ there is an integer $$n_0$$ such that $$x_n \in V$$ for all $$n > n_0$$. Assume each neighbourhood has a size of $$\epsilon$$. Thus, $$x_n \in V$$ implies that $$d(x_n,x)<\epsilon$$. Then, we can restate this as the following: $$\forall \epsilon>0, \exists N_0$$, such that $$d(x_n,x)<\epsilon$$ for all $$n>N_0$$.

$$\blacksquare$$

---

#### 1.4.4. (Boundedness) Show that a Cauchy sequence is bounded.

**Proof:**

By definition, for a Cauchy sequence, we have: $$\forall \epsilon>0, \exists N_0$$, such that $$d(x_m,x_n)<\epsilon$$ for all $m,n>N_0$$.

Choose $$\epsilon=1$$. Then, assume the value of $$N_0$$ to be $$N_1$$. For any $$d(x_a,x_b)$$, we have:

- $$a<b \leq N_0$$: Then $$d(x_a,x_b) \leq a = max[d(x_a,x_0), d(x_a,x_1), \cdots, d(x_a,x_{N_1})]$$
- $$N_0<a<b$$: Then $$d(x_a,x_b) < \epsilon = 1$$
- $$a \leq N_0<b$$: By the **Triangle Inequality**, we have: $$d(x_a,x_b) \leq d(x_a,x_{N_1}) + d(x_{N_1}, x_b) < a + 1$$

Combining these upper bounds, we get: $$\text{sup } d(x_a,x_b) < a+1$$

$$\blacksquare$$

---

#### 1.4.5. Is boundedness of a sequence in a metric space sufficient for the sequence to be Cauchy? Convergent?

**Answer:**

Consider the discrete metric on $$\mathbb{R}$$. If we have a sequence $$(x_n)=0,1,0,1,\cdots$$, then the series is bounded because $$\text{sup } d(x_m, x_n)=1$$, but for $$\epsilon=\frac{1}{2}$$, there is no $$N$$ for which $$d(x_m,x_n)<\epsilon$$ for $$m,n>N$$. Thus, the sequence is not Cauchy, though it is bounded.

Convergence is sufficient for a sequence to be Cauchy. For convergence, we have the condition: if $$x_n \rightarrow x$$, $$\forall \epsilon>0, \exists N_0$$, such that $$d(x_n,x)<\epsilon$$ for all $$n>N_0$$.

Consider $$m,n>N_0$$. Then, by the **Triangle Inequality**, we have:

$$
d(x_m,x_n) \leq d(x_m,x) + d(x,x_n) < \epsilon + \epsilon = 2 \epsilon
$$

thus proving the Cauchy criterion.

---
#### 1.4.6. If $$(x_n)$$ and $$(y_n)$$ are Cauchy sequences in a metric space $$(X, d)$$, show that $$(a_n)$$, where $$a_n = d(x_n, y_n)$$, converges. Give illustrative examples.

**Proof:**

$$
d(x_m,x_n)<\epsilon \\
d(y_m,y_n)<\epsilon
$$

Then, we have:

$$
d(x_m,y_m) \leq d(x_m,x_n) + d(x_n,y_n) + d(y_n,y_m) \\
\Rightarrow d(x_m,y_m) - d(x_n,y_n) \leq d(x_m,x_n) + d(y_n,y_m)
\Rightarrow d(x_m,y_m) - d(x_n,y_n) < 2 \epsilon
$$

Similarly, we have:

$$
d(x_n,y_n) \leq d(x_n,x_m) + d(x_m,y_m) + d(y_m,y_n) \\
\Rightarrow d(x_n,y_n) - d(x_m,y_m) < 2 \epsilon
$$

The above inequalities imply that:
$$
\vert d(x_m,y_m) - d(x_n,y_n) \vert < 2 \epsilon \\
d[d(x_m,y_m) - d(x_n,y_n)] < 2 \epsilon
$$

This implies that $$a_n=d(x_n,y_n)$$ is Cauchy, and thus converges.

$$\blacksquare$$

---
#### 1.4.7. Give an indirect proof of Lemma 1.4-2(b).

---
#### 1.4.8. If $$d_1$$ and $$d_2$$ are metrics on the same set $$X$$ and there are positive numbers $$a$$ and $$b$$ such that for all $$x, y \in X$$, $$a.d_1(x,y) \leq d_2(x,y) \leq b.d_1(x,y)$$, show that the Cauchy sequences in $$(X, d_1)$$ and $$(X, d_2)$$ are the same.

---
#### 1.4.9. Using Prob. 8, show that the metric spaces in Probs. 13 to 15, Sec. 1.2, have the same Cauchy sequences.

---
#### 1.4.10. Using the completeness of $$\mathbb{R}$$, prove completeness of $$\mathbb{C}$$.


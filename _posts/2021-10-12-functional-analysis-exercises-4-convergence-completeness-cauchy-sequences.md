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


Both $$d(x,x_p)$$ and $$d(x_{n_m},y)$$ can be made as small as possible, since $$(x_n)$$ and $$(x_{n_k})$$ converge, implying that $$d(x,y)$$ is smaller than any positive value

$$
d(x,y) \leq d(x,x_p) + d(x_{n_m},y) \\
\Rightarrow d(x,y) < \epsilon_1 + \epsilon_2, \epsilon_1, \epsilon_2 > 0 \\
\Rightarrow d(x,y)=0
$$

Thus, $$x=y$$, i.e., $$x_{n_k}$$ has the same limit as $$(x_n)$$.

$$\blacksquare$$

---
#### 1.4.2. If $$(x_n)$$ is Cauchy and has a convergent subsequence, say, $$x_n \rightarrow x$$, show that $$(x_n)$$ is convergent with the limit $$x$$.

---
#### 1.4.3. Show that $$x_n \rightarrow x$$ if and only if for every neighborhood $$V$$ of $$x$$ there is an integer $$n_0$$ such that $$x_n \in V$$ for all $$n > n_0$$.

---
#### 1.4.4. (Boundedness) Show that a Cauchy sequence is bounded.

---
#### 1.4.5. Is boundedness of a sequence in a metric space sufficient for the sequence to be Cauchy? Convergent?

---
#### 1.4.6. If $$(x_n)$$ and $$(y_n)$$ are Cauchy sequences in a metric space $$(X, d)$$, show that $$(a_n)$$, where $$a_n = d(x_n, y_n)$$, converges. Give illustrative examples.

---
#### 1.4.7. Give an indirect proof of Lemma 1.4-2(b).

---
#### 1.4.8. If $$d_1$$ and $$d_2$$ are metrics on the same set $$X$$ and there are positive numbers $$a$$ and $$b$$ such that for all $$x, y \in X$$, $$a.d_1(x,y) \leq d_2(x,y) \leq b.d_1(x,y)$$, show that the Cauchy sequences in $$(X, d_1)$$ and $$(X, d_2)$$ are the same.

---
#### 1.4.9. Using Prob. 8, show that the metric spaces in Probs. 13 to 15, Sec. 1.2, have the same Cauchy sequences.

---
#### 1.4.10. Using the completeness of $$\mathbb{R}$$, prove completeness of $$\mathbb{C}$$.


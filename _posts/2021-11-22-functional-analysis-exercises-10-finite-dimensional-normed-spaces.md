---
title: "Functional Analysis Exercises 10 : Finite Dimensional Normed Spaces and Subspaces"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics", "Kreyszig"]
draft: false
---

This post lists solutions to the exercises in the **Finite Dimensional Normed Spaces and Subspaces section 2.4** of *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**. This is a work in progress, and proofs may be refined over time.

### Notes
The requirements for a space to be a normed space are:

- **(N1)** **Nonnegativity**, i.e., $$\|x\| \geq 0, x \in X$$
- **(N2)** **Zero norm** implies **zero vector** and vice versa, i.e., $$\|x\|=0 \Leftrightarrow x=0, x \in X$$
- **(N3)** **Linearity** with respect to **scalar multiplication**, i.e., $$\|\alpha x\|=\vert \alpha \vert \|x\|, x \in X, \alpha \in \mathbb{R}$$
- **(N4)** **Triangle Inequality**, i.e., $$\|x+y\| \leq \|x\| + \|y\|, x,y \in X$$

The requirements for a space to be a vector space are:

(Mnemonic: **ADD1 SIIA**)

#### Addition
- **(V1)** **Symmetry** implies $$x+y=y+x, x,y \in X$$.
- **(V2)** **Identity** implies **zero vector**, i.e., $$x+\theta=x, x,\theta \in X$$.
- **(V4)** **Inverse** implies $$x+(-x)=\theta, x,y,\theta \in X$$.
- **(V3)** **Associativity** implies $$(x+y)+z=x+(y+z), x,y,z \in X$$.

#### Multiplication
- **(V1)** **Associativity** implies $$x(yz)=(xy)z, x,y,z \in X$$.
- **(V2)** **Distributivity with respect to vector addition** implies $$\alpha(x+y)=\alpha x + \alpha y, x,y \in X, \alpha \in \mathbb{R}$$.
- **(V3)** **Distributivity with respect to scalar addition** implies $$(\alpha + \beta) x = \alpha x + \beta x, x \in X, \alpha, \beta \in \mathbb{R}$$.
- **(V4)** **Identity** implies, $$1x=x, x \in X$$.

#### 2.3.1. Give examples of subspaces of $$l^\infty$$ and $$l^2$$ which are not closed.

**Answer:**

$$l^\infty$$ is the space of all bounded sequences, i.e., $$\sum\limits_{i=1}^\infty \vert x_i \vert < \infty$$. The norm it is equipped with is $$\|(x_n)\|=\sup \vert x_i\vert $$.

The space of sequences with finitely many non-zero elements is an example of a subspace which is not closed.

In particular, the sequence defined as:

$$
(x_j^n)=\begin{cases}
\displaystyle\frac{1}{2^n} & \text{if } j \leq n \\
0 & \text{if } j > n
\end{cases}
$$

Assume $$m<n$$. Then, we have:

$$
\alpha x^{(m)} + \beta x^{(n)} \\
= \alpha (\frac{1}{2} , \frac{1}{2^2} , \cdots + \frac{1}{2^m}, , 0 , 0, \cdots) , \beta (\frac{1}{2} , \frac{1}{2^2} , \cdots , \frac{1}{2^m} , \frac{1}{2^{m+1}} , \cdots , \frac{1}{2^n} , 0 , 0, \cdots) \\
= (\alpha + \beta) \frac{1}{2} , (\alpha + \beta) \frac{1}{2^2} , \cdots , (\alpha + \beta) \frac{1}{2^m} , \frac{\beta}{2^{m+1}} , \cdots , \frac{\beta}{2^n}, 0 , 0, \cdots
$$

For the $$l^\infty$$ case, we have the norm as:

$$
{\|x-x^{(n)}\|}_\infty=\sup |x_j-x_j^{(n)}|=\frac{1}{2^{n+1}}
$$

where $$x=\frac{1}{2} , \frac{1}{2^2} , \cdots$$.

As $${\|x-x^{(n)}\|}+\infty \rightarrow 0$$ as $$n \rightarrow 0$$, $$x$$ is a limit of $$x^{(n)}$$. Thus, the limit exists, but it is not in this space, since $$x$$ has infinitely many terms.

In the case of $$l^2$$, the norm $${\|\bullet\|}_2$$ the partial sum $$s_n=\displaystyle\frac{1}{3}(1-\frac{1}{4^n})$$ (see Rough Work at the end to see how this was calculated).

Then $${\|x-x^{(n)}\|}_2=\displaystyle\frac{1}{3\cdot 4^n} \rightarrow 0$$ as $$n \rightarrow \infty$$ where $$x=\frac{1}{4} , \frac{1}{4^2} , \cdots$$. Thus, the limit exists, but it is not in this space, since $$x$$ again has infinitely many terms.

---

#### 2.3.2. What is the largest possible $$c$$ in (1) if $$X = \mathbb{R}^2$$ and $$x_1 = (1,0), x_2 = (0,1)$$? If $$X = \mathbb{R}^3$$ and $$x_1 = (1,0,0), x_2 = (0,1,0), x_3 = (0,0,1)$$?

**Answer:**

We have the identity:

$$
\|\alpha_1 e_1 + \alpha_2 e_2 + \cdots + \alpha_1 e_n\| \geq c(|\alpha_1| + |\alpha_2| + \cdots + |\alpha_n|)
$$

For $$\mathbb{R}^2$$, we have:

$$
\|\alpha_1 e_1 + \alpha_2 e_2\| \geq c(|\alpha_1| + |\alpha_2|) \\
{(\alpha_1^2 + \alpha_2^2)}^{1/2} \geq c(|\alpha_1| + |\alpha_2|)
$$

If $$\alpha_1=\alpha_2=0$$ gives us an equality above, let us pick an arbitrary small $$\alpha_1=\alpha_2=\epsilon$$, so that we get:

$$
{(\epsilon^2 + \epsilon^2)}^{1/2} \geq c(\epsilon + \epsilon) \\
2 \epsilon c \leq \sqrt 2 \epsilon \\
c \leq \frac{1}{\sqrt 2}
$$

For $$\mathbb{R}^3$$, we have:

$$
\|\alpha_1 e_1 + \alpha_2 e_2 + \alpha_3 e_3\| \geq c(|\alpha_1| + |\alpha_2| + |\alpha_3|) \\
{(\alpha_1^2 + \alpha_2^2 + \alpha_3^2)}^{1/2} \geq c(|\alpha_1| + |\alpha_2| + |\alpha_3|)
$$

If $$\alpha_1=\alpha_2=\alpha_3=0$$ gives us an equality above, let us pick an arbitrary small $$\alpha_1=\alpha_2=\alpha_3=\epsilon$$, so that we get:

$$
{(\epsilon^2 + \epsilon^2 + \epsilon^3)}^{1/2} \geq c(\epsilon + \epsilon + \epsilon) \\
3 \epsilon c \leq \sqrt 3 \epsilon \\
c \leq \frac{1}{\sqrt 3}
$$

---

#### 2.3.3. Show that in Def. 2.4-4 the axioms of an equivalence relation hold (cf. A1.4 in Appendix 1).

**Proof:**

The relation to demonstrate is an equivalence is the following:

$$
a{\|x\|}_2 \leq {\|x\|}_1 \leq b{\|x\|}_2
$$

**Reflexive**:
This is evident since:

$$
{\|x\|}_1 \leq {\|x\|}_1 \leq b{\|x\|}_1
$$

where $$a=b=1$$.

**Symmetric**:

We have:

$$
{\|x\|}_1 \leq b{\|x\|}_2 \\
(1/b){\|x\|}_1 \leq {\|x\|}_2 \\
{\|x\|}_2 \geq (1/b){\|x\|}_1
$$

$$
a{\|x\|}_2 \leq {\|x\|}_1 \\
{\|x\|}_2 \leq (1/a){\|x\|}_1
$$

Thus, we get:

$$
\frac{1}{b}{\|x\|}_1 \leq {\|x\|}_2 \leq \frac{1}{a}{\|x\|}_1
$$

**Transitive**:
Assume that:

$$
a_1{\|x\|}_2 \leq {\|x\|}_1 \leq b_1{\|x\|}_2 \\
a_2{\|x\|}_3 \leq {\|x\|}_2 \leq b_2{\|x\|}_3
$$

We have the following two identities:

$$
{\|x\|}_1 \leq b_1{\|x\|}_2 \\
(1/b_1){\|x\|}_1 \leq {\|x\|}_2 \\
(1/b_1){\|x\|}_1 \leq {\|x\|}_2 \leq b_2{\|x\|}_3 \\
{\|x\|}_1 \leq b_1b_2{\|x\|}_3
$$

$$
a_1{\|x\|}_2 \leq {\|x\|}_1 \\
{\|x\|}_2 \leq (1/a_1){\|x\|}_1 \\
a_2{\|x\|}_3 \leq {\|x\|}_2 \leq (1/a_1){\|x\|}_1 \\
a_1a_2{\|x\|}_3 \leq {\|x\|}_1 \\
$$

Putting the two inequalities above together, we get:

$$
a_1a_2{\|x\|}_3 \leq {\|x\|}_1 \leq b_1b_2{\|x\|}_3
$$

$$\blacksquare$$

---

#### 2.3.4. Show that equivalent norms on a vector space $$X$$ induce the same topology for $$X$$.

**Proof:**


$$\blacksquare$$

---

#### 2.3.5. If $$\|\bullet\|$$ and $${\|\bullet\|}_0$$ are equivalent norms on X, show that the Cauchy sequences in $$(X, \|\bullet\|)$$ and $$(X,{\|\bullet\|}_0)$$ are the same.

**Proof:**


$$\blacksquare$$

---

#### 2.3.6. Theorem 2.4-5 implies that $${\|\bullet\|}_2$$ and $${\|\bullet\|}_\infty$$ in Prob. 8, Sec. 2.2, are equivalent. Give a direct proof of this fact.

**Proof:**


$$\blacksquare$$

---

#### 2.3.7. Let $${\|\bullet\|}_2$$ be as in Prob. 8, Sec. 2.2, and let $$\|\bullet\|$$ be any norm on that vector space, call it $$X$$. Show directly (without using 2.4-5) that there is a b>0 such that $$\|\bullet\| \leq {\|\bullet\|}_2$$ for all $$x$$.

**Proof:**


$$\blacksquare$$

---

#### 2.3.8. Show that the norms $${\|\bullet\|}_1$$ and $${\|\bullet\|}_2$$ in Prob. 8, Sec. 2.2, satisfy $$\frac{1}{\sqrt{n}} {\|x\|}_1 \leq {\|\bullet\|}_2 \leq {\|x\|}_1$$.

**Proof:**


$$\blacksquare$$

---

#### 2.3.9. If two norms $$\|\bullet\|$$ and $${\|\bullet\|}_0$$ on a vector space $$X$$ are equivalent, show that (i) $$\|x_n - x\| \rightarrow 0$$ implies (ii) $${\|x_n - x\|}_0 \rightarrow 0$$ (and vice versa, of course).

**Proof:**


$$\blacksquare$$

---

#### 2.3.10. Show that all complex $$m \times n$$ matrices $$A = (\alpha_{jk})$$ with fixed $$m$$ and $$n$$ constitute an $$mn$$-dimensional vector space $$Z$$. Show that all norms on $$Z$$ are equivalent. What would be the analogues of $${\|\bullet\|}_1$$, $${\|\bullet\|}_2$$ and $${\|\bullet\|}_\infty$$ in Prob. 8, Sec. 2.2, for the present space $$Z$$?

**Proof:**


$$\blacksquare$$


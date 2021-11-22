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

Consider a Cauchy sequence $$(x_n)$$ in $$X$$. Then, we have, $$\forall \epsilon>0$$, $$\exists N$$, such that $$d(x_m,x_n) < \epsilon$$ for all $$m,n>N$$.

Let $$T:X \rightarrow Y$$. Since $$X$$ is isometric to $$Y$$, we have:

$$
d(x_m,x_n)=d(Tx_m,Tx_n) < \epsilon
$$

This implies that for every $$\epsilon>0$$, there exists a $$\delta>0$$, such that $$d(x_m,x_n) < \delta \Rightarrow d(Tx_m,Tx_n) < \epsilon$$. In this case $$\delta=\epsilon$$. Thus, $$T$$ is continuous at $$x_n$$.

The above argument can be used for $$T^{-1}$$ to prove that it is also continuous.

To prove injectivity, we note that $$x \neq y \Rightarrow d(x,y) \neq 0 \Rightarrow \Rightarrow d(Tx,Ty) \neq 0 \Rightarrow Tx \neq Ty$$.

To prove surjectivity, we pick a point $$y \in Y$$. Assume $$x_1 \in X$$. Then, by isometry we must have: d(y,Tx_1)=d(x, x_1), where $$x \in X$$. Thus, there is a corresponding preimage for every $$y \in Y$$.

$$\blacksquare$$

Consider the $$f:(0,1) \rightarrow \mathbb{R}$$ defined as $$f(x)=x$$. Then $$f(x)$$ and its inverse are continuous and bijective. $$(0,1)$$ is an incomplete metric space and $$\mathbb{R}$$ is complete.

---

#### 1.6.6 Show that $$C[0,1]$$ and $$C[a,b]$$ are isometric.
**Proof:**

We note that $$f(t)=\displaystyle\frac{t-a}{b-a}, a \neq b$$ is a mapping $$f: [a,b] \rightarrow [0,1]$$, and that $$f^{-1}(t)=a+(b-a)t$$ is a mapping $$f^{-1}: [0,1] \rightarrow [a,b]$$.

We note that $$f$$ and $$f^{-1}$$ are bijections.
The distance metric in $$C$$ is defined as $$d(x,y)=\sup\vert x(t) - y(t) \vert$$.

Define a mapping $$T:C_{t \in [0,1]}(t) \rightarrow C_{t \in [a,b]}(f^{-1}(t))$$

Think of $$C(f(t)$$ as the original function applied to $$[0,1]$$ even though the input $$t \in [a,b]$$. Then, practically we have $$C_{t \in [0,1]}(t)=C_{t \in [a,b]}(f(t))$$.

Then:

$$
d(Tx,Ty)=\sup_{[a,b]} |x(f(t)) - y(f(t))|=\sup_{[0,1]} |x(t) - y(t)|=d(x,y)
$$

Thus, $$T$$ preserves distances.

To prove injectivity, suppose $$Tx=Ty$$, then we have:

$$
d(Tx,Ty)=\sup_{[a,b]} |x(f(t)) - y(f(t))|=0 \\
\Rightarrow \sup_{[0,1]} |x(t) - y(t)|=d(x,y)=0 \\
\Rightarrow x=y
$$

For surjectivity, we note that for an arbitrary function $$y(f(t)) \in C[a,b]$$, we always have $$x(t) \in C[0,1]$$, since $$T^{-1}x=x(f^{-1}(f(t)))=x(t)$$.

$$\blacksquare$$

---

#### 1.6.7 If $$(X,d)$$ is complete, show that $$(X,\tilde{d})$$, where $$\tilde{d} = d/(l + d)$$, is complete.
**Proof:**

Let $$(x_n)$$ be a Cauchy sequence in $$(X,\bar{d})$$, so that we have, $$\forall \epsilon>0, \exists N$$, such that $$d(x_m,x_n) < \epsilon$$ for all $$m,n>N$$.

Then, we have:

$$
\frac{d(x_m,x_n)}{1+d(x_m,x_n)} < \epsilon \\
d(x_m,x_n) < \epsilon + \epsilon d(x_m,x_n) \\
d(x_m,x_n)(1-\epsilon) < \epsilon \\
d(x_m,x_n) < \frac{\epsilon}{1-\epsilon}
$$

Set $$\epsilon=\frac{1}{k}$$, so that we get:

$$
{d}(x_m,x_n) < \frac{1}{k-1} \\
$$

$$k$$ can be made as large as needed to make $$\epsilon$$ as small as needed. Thus, the sequence $$(x_n)$$ is Cauchy in $$(X,d)$$, and thus has a limit $$x$$, i.e., $$x_n \rightarrow x$$.

Then, $$d(x_n,x)<\epsilon$$.

$$
\bar{d}(x_n,x)<d(x_n,x)<\epsilon
$$

$$\blacksquare$$

---

#### 1.6.8 Show that in Prob. 7, completeness of $$(X,\tilde{d})$$ implies completeness of $$(X,d)$$.
**Proof:**
Suppose $$(X,\tilde{d})$$ is complete. Then, we have:



$$\blacksquare$$

---

#### 1.6.9  If $$(x_n)$$ and $$(x_n')$$ in $$(X,d)$$ are such that (1) holds and $$x_n \rightarrow l$$, show that $$(x_n')$$ converges and has the limit $$l$$.
**Proof:**


$$\blacksquare$$

---

#### 1.6.10  If $$(x_n)$$ and $$(x_n')$$ are convergent sequences in a metric space $$(X,d)$$ and have the same limit $$l$$, show that they satisfy (1).

(1) defines equivalence of two sequences as $$(x_n)\tilde(x_n') \Rightarrow \lim\limits_{n \rightarrow \infty} d(x_n,x_n')=0$$.

**Proof:**

Since $$(x_n)$$ and $$(x_n')$$ are convergent, we have, $$\forall \epsilon>0, \exists N_1, N_2$$ such that $$d(x_m,l)<\epsilon$$ and $$d(x_n',l)<\epsilon$$, for $$m>N_1, n>N_2$$. Choose $$N=\text{max}(N_1,N_2)$$, so that we have $$d(x_n,l)<\epsilon$$ and $$d(x_n',l)<\epsilon$$ for all $$n>N$$.

$$
d(x_n,x_n') \leq d(x_n,l) + d(l,x_n') < \epsilon+\epsilon=2 \epsilon \\
\Rightarrow \lim\limits_{n \rightarrow \infty} d(x_n,x_n') = 0
$$

$$\blacksquare$$

---

#### 1.6.11   Show that (1) defines an equivalence relation on the set of all Cauchy sequences of elements of $$X$$.

(1) defines equivalence of two sequences as $$(x_n)\tilde{}(x_n') \Rightarrow \lim\limits_{n \rightarrow \infty} d(x_n,x_n')=0$$.

**Proof:**

We will check for the following properties:

- Reflexive
- Symmetric
- Transitive

We know that $$d(x_n,x_n)=0$$ always because of the **Principle of Indiscernibles**. Thus, we get:

$$\lim\limits_{n \rightarrow \infty} d(x_n,x_n)=0$$

By the **Symmetry Property** of a distance metric, we know that $$d(x_n,x_n')=d(x_n',x_n)$$. Thus if we have $$\lim\limits_{n \rightarrow \infty} d(x_n,x_n')=0$$, then we also have:

$$\lim\limits_{n \rightarrow \infty} d(x_n',x_n)=0$$

By the **Triangle Inequality**, we have:

$$
d(x_n,z_n)<d(x_n,y_n)+d(y_n,z_n)
$$

Taking limits, we get:

$$
\lim\limits_{n \rightarrow \infty} d(x_n,z_n) \leq \lim\limits_{n \rightarrow \infty} d(x_n,y_n) + \lim\limits_{n \rightarrow \infty} d(y_n,z_n)
$$

If we have $$\lim\limits_{n \rightarrow \infty} d(x_n,y_n)=0$$ and $$\lim\limits_{n \rightarrow \infty} d(y_n,z_n)=0$$, we get:

$$
\lim\limits_{n \rightarrow \infty} d(x_n,z_n) \leq 0
$$

Since distances are always nonnegative, we have: $$\lim\limits_{n \rightarrow \infty} d(x_n,z_n) = 0$$.

$$\blacksquare$$

---

#### 1.6.12   If $$(x_n)$$ is Cauchy in $$(X,d)$$ and $$(x_n')$$ in $$X$$ satisfies (1), show that $$(x_n')$$ is Cauchy in $$X$$.
**Proof:**

Since $$(x_n)$$ is Cauchy, we have, $$\forall \epsilon>0, \exists N$$ such that $$d(x_m,x_n)<\epsilon$$ for $$m,n>N$$.

$$
d(x_m,x_n) < \epsilon
$$

We also have $$(x_n)$$ and $$(x_n')$$ being equivalent, so we can write:

$$
\lim\limits_{n \rightarrow \infty} d(x_n,x_n') = 0
$$

By the **Triangle Inequality**, we have:

$$
d(x_m',x_n') \leq d(x_m',x_m) + d(x_m,x_n) + d(x_n,x_n')
$$

Taking limits on both sides, we get:

$$
\lim\limits_{n \rightarrow \infty} d(x_m',x_n') \leq \underbrace{\lim\limits_{n \rightarrow \infty} d(x_m',x_m)}_\text{0 because equivalent} + \underbrace{\lim\limits_{n \rightarrow \infty} d(x_m,x_n)}_\text{0 because Cauchy} + \underbrace{\lim\limits_{n \rightarrow \infty} d(x_n,x_n')}_\text{0 because equivalent} \\
$$

Since distance metric has to be nonnegative, we conclude that:

$$
\lim\limits_{n \rightarrow \infty} d(x_m',x_n')=0
$$

$$\blacksquare$$

---

#### 1.6.13 (Pseudometric) A finite pseudometric on a set $$X$$ is a function $$d: X \times X \rightarrow R$$ satisfying (M1), (M3), (M4), Sec. 1.1, and (M2*) $$d(x,x)=0$$. What is the difference between a metric and a pseudometric? Show that $$d(x,y)=\vert \xi_1 - \eta_1\vert $$ defines a pseudometric on the set of all ordered pairs of real numbers, where $$x = (\xi_1,\xi_2), y = (\eta_1,\eta_2)$$. (We mention that some authors use the term semimetric instead of pseudometric.)

**Proof:**

**(M1)** We know that $$d(x,y)=\vert \xi_1 - \eta_1\vert $$ is always nonnegative, real-valued, and finite.  
**(M3)** Because $$d(x,y)=\vert \xi_1 - \eta_1\vert $$ has a modulus sign, we always have: $$\vert \xi_1 - \eta_1\vert = \vert \eta_1 - \xi_1\vert $$, and thus we have symmetry.  
**(M4)** We have: $$d(x,y)=\vert \xi_1 - \eta_1\vert = \vert \xi_1 - \kappa_1 + \kappa_1 - \eta_1\vert \leq \vert \xi_1 - \kappa_1 \vert + \vert \kappa_1 - \eta_1\vert$$. Thus, the **Triangle Inequaity** is shown.

**(Modified M2)** An example pair which satisfies this condition is $$(1,2)$$ and $$(1,3)$$. We see that any pair $$(\kappa,\xi)$$ and $$(\kappa,\eta)$$ will satisfy **(Modified M2)**. We see that if $$x_1=(\kappa,\xi)$$ and $$x_2=(\kappa,\eta)$$, then $$d(x,y)=\vert \kappa - \kappa\vert=0$$.

$$\blacksquare$$

---

#### 1.6.14 Does $$d(x,y)=\int\limits_a^b\vert x(t)-y(t)\vert dt$$ define a metric or pseudometric on $$X$$ if $$X$$ is (i) the set of all real-valued continuous functions on $$[a,b]$$, (ii) the set of all real-value Riemann integrable functions on $$[a,b]$$?
**Proof:**

**[TODO]**

$$\blacksquare$$

---

#### 1.6.15 If $$(X,d)$$ is a pseudometric space, we call a set $$B(x_0; r) = {x \in X : d(x,x_0) < r} (r>O)$$ an open ball in $$X$$ with center $$x_0$$ and radius $$r$$. (Note that this is analogous to 1.3-1.) What are open balls of radius $$1$$ in Prob. 13?

**Answer:**

The open ball in this case is a vertical rectangles with open width 2 centered at $$x_0$$.


---
title: "Solutions to Functional Analysis Exercises : Distance Metrics"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis"]
draft: false
---

This post lists solutions to many of the Functional Analysis exercises in *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**.

#### 1.1.2. Does $$d(x,y)={(x-y)}^2$$ define a metric on the set of all real numbers?
**Proof:**

For the distance metric $$d(x,y)={(x-y)}^2$$, we need to prove or disprove the **Triangle Inequality**:

$$
d(x,z) \leq d(x,y) + d(y,z)
$$

We start with the term $${(x-z)}^2$$, as follows:

$$
{(x-z)}^2 = {((x-y)+(y-z))}^2 \\
\Rightarrow {(x-z)}^2 = {(x-y)}^2 + {(y-z)}^2 + 2 (x-y) (y-z) \\
\Rightarrow d(x,z) = d(x,y) + d(y,z) + \underbrace{2 (x-y) (y-z)}_\text{positive or negative}
$$

When the term $$2 (x-y) (y-z)$$ is **positive**:

$$
d(x,z) > d(x,y) + d(y,z)
$$

When the term $$2 (x-y) (y-z)$$ is **negative** or **zero**:

$$
d(x,z) \leq d(x,y) + d(y,z)
$$

**Thus, the Triangle Inequality can only be satisfied for specific values of $$x$$, $$y$$, and $$z$$; hence $$d(x,y)={(x-y)}^2$$ is not a metric.**

$$\blacksquare$$

#### 1.1.3. Show that $$d(x,y)=\sqrt{|x-y|}$$ defines a metric on the set of all real numbers.

#### 1.1.5. Let $$d$$ be a metric on $$X$$. Determine all constants $$k$$ such that *(i)* $$kd$$, *(ii)* $$d+k$$ is a metric on $$X$$.
#### 1.1.6. Show that $$d$$ in 1.1-6 satisfies the triangle inequality.
#### 1.1.8. Show that another metric $$\bar{d}$$ on the set $$X$$ in 1.1-7 is defined by $$\bar{d}(x,y)=\int\limits_a^b |x(t) - y(t)| dt$$.
#### 1.1.9. Show that $$d$$ in 1.1-8 is a metric.
#### 1.1.10. **(Hamming Distance)** Let $$X$$ be the set of all ordered triples of zeros and ones. Show that $$X$$ consists of eight elements and a metric $$d$$ on $$X$$ is defined by $$d(x,y)=$$ number of places where $$x$$ and $$y$$ have different entries. (This space and similar spaces of $$n$$-tuples play a role in switching and automata theory and coding. $$d(x,y)$$ is called the *Hamming distance* between $$x$$ and $$y$$; cf. the paper by R. W. Hamming (1950) listed in Appendix 3.)
#### 1.1.12. **(Triangle inequality)** The triangle inequality has several useful consequences. For instance, using (1), show that $$|d(x,y) - d(z,w)| \leq d(x,z) + d(y,w)$$.
#### 1.1.13. Using the triangle inequality, show that $$|d(x,z) - d(y,z)| \leq d(x,y)$$.
#### 1.1.14. **(Axioms of a metric)** (M1) to (M4) could be replaced by other axioms (without changine the definition). For instance, show that (M3) and (M4) could be obtained from (M2) and $$d(x,y) \leq d(z,x) + d(z,y)$$.
#### 1.1.15. Show that nonnegativity of a metric follows from (M2)to (M4).

## Assorted Proofs

#### Prove that if $$S$$ is open, $$S'$$ is closed.

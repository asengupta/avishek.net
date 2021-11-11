---
title: "Functional Analysis Exercises 8 : Normed and Banach Spaces"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics", "Kreyszig"]
draft: false
---

This post lists solutions to the exercises in the **Normed Space, Banach Space section 2.2** of *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**. This is a work in progress, and proofs may be refined over time.

### Notes
The requirements for a space to be a normed space are:

- **(N1)** **Nonnegativity**, i.e., $$\|x\| \geq 0, x \in X$$
- **(N2)** **Zero norm** implies **zero vector** and vice versa, i.e., $$\|x\|=0 \Leftrightarrow x=0, x \in X$$
- **(N3)** **Linearity** with respect to **scalar multiplication**, i.e., $$\|\alpha x\|=\vert \alpha \vert \|x\|, x \in X, \alpha \in \mathbb{R}$$
- **(N4)** **Triangle Inequality**, i.e., $$\|x+y\| \leq \|x\| + \|y\|, x,y \in X$$

#### 2.2.1 Show that the norm $$\|x\|$$ of x is the distance from x to 0.
**Proof:**

$$\blacksquare$$

---

#### 2.2.2 Verify that the usual length of a vector in the plane or in three dimensional space has the properties (N1) to (N4) of a norm.
**Proof:**

$$\blacksquare$$

---

#### 2.2.3 Prove (2).
**Proof:**

$$\blacksquare$$

---

#### 2.2.4 Show that we may replace **(N2)** by $$\|x\|=0 \Rightarrow x=0$$ without altering the concept of a norm. Show that nonnegativity of a norm also follows from **(N3)** and **(N4)**.
**Proof:**

$$\blacksquare$$

---

#### 2.2.5 Show that (3) defines a norm.
**Proof:**

$$\blacksquare$$

---

#### 2.2.6 Let $$X$$ be the vector space of all ordered pairs $$x = (\xi_1, \xi_2), y = (\eta_1, \eta_2), \cdots$$ of real numbers. Show that norms on X are defined by

  $$
  {\|x\|}_1=|\eta_1| + |\eta_2| \\
  {\|x\|}_2={(\eta_1^2 + \eta_2^2)}^{1/2} \\
  {\|x\|}_\infty=\text{max } \{ |\xi_1|, |\xi_2| \}
  $$

**Proof:**

$$\blacksquare$$

---

#### 2.2.7 Verify that (4) satisfies (N1) to (N4).
**Proof:**

$$\blacksquare$$

---

#### 2.2.8 There are several norms of practical importance on the vector space of ordered n-tuples of numbers (cf. 2.2-2), notably those defined by  

  $$
  {\|x\|}_1=|\eta_1| + |\eta_2| + \cdots + |\eta_n| \\
  {\|x\|}_2={(|\eta_1|^p + |\eta_2|^p + \cdots + |\eta_n|^p)}^{1/p} \\
  {\|x\|}_\infty=\text{max } \{ |\xi_1|, |\xi_2|, \cdots, |\xi_n| \}
  $$
   
   **In each case, verify that (N1) to (N4) are satisfied.**

**Proof:**

$$\blacksquare$$

---

#### 2.2.9 Verify that (5) defines a norm.
**Proof:**

$$\blacksquare$$

---

#### 2.2.10 (Unit sphere) The sphere $$S(0; 1) = \{x \in X : \|x\| = 1\}$$ in a normed space $$X$$ is called the unit sphere. Show that for the norms in Prob. 6 and for the norm defined by the unit spheres look as shown in Fig. 16.

**Proof:**

$$\blacksquare$$

---

#### 2.2.11 (Convex set, segment) A subset $$A$$ of a vector space $$X$$ is said to be convex if $$x,y \in A$$ implies $$M=\{z \in X : z=\alpha x+(1-\alpha)y, 0\leq \alpha \leq 1\} \subset A$$. $$M$$ is called a closed segment with boundary points $$x$$ and $$y$$; any other Z E M is called an interior point of $$M$$. Show that the closed unit ball $$B(0; 1) =\{x \in X : \|x\| \leq 1\}$$ in a normed space X is convex.

**Proof:**

$$\blacksquare$$

---

#### 2.2.12 Using Prob. 11, show that does not define a norm on the vector space of all ordered pairs $$x = (\xi_1, \xi_2), \cdots$$ of real nwnbers. Sketch the curve $$\phi(x) = 1$$ and compare it with Fig. 18.

**Proof:**

$$\blacksquare$$

---

#### 2.2.13 Show that the discrete metric on a vector space $$X \neq \{0\}$$ cannot be obtained from a norm. (Cf. 1.1-8.)

**Proof:**

$$\blacksquare$$

---

#### 2.2.14 If $$d$$ is a metric on a vector space $$X \neq \{0\}$$ which is obtained from a norm, and $$d$$ is defined by $$\tilde{d}(x,x) = 0, \tilde{d}(x,y)=d(x,y)+1 (x \neq y)$$, show that $$d$$ cannot be obtained from a norm.

**Proof:**

$$\blacksquare$$

---

#### 2.2.15 (Bounded set) Show that a subset $$M$$ in a normed space $$X$$ is bounded if and only if there is a positive number $$c$$ such that $$\|x\| \leq  c$$ for every $$x \in M$$. (For the definition, see Prob. 6 in Sec. 1.2.)

**Proof:**

$$\blacksquare$$

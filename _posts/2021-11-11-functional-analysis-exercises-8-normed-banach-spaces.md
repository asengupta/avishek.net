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

We have:

$$
\|x\|=\|x + \theta|=\|x + \theta + (-\theta)\|=\|x + (-\theta) + \theta\| \\
\|x\| \leq \|x+(-\theta)\| + \|\theta\|
$$

We also have:

$$
\|x+(-\theta)\| \leq \|x\| + \|-\theta\| \\
\|x+(-\theta)\| \leq \|x\| + |-1|\|\theta\| \\
\|x+(-\theta)\| \leq \|x\| + \|\theta\| = \|x\|
$$

Thus, $$\|x\| \leq \|x+(-\theta)\|$$ and $$\|x\| \geq \|x+(-\theta)\|$$. Thus, $$\|x\| = \|x+(-\theta)\|$$, which is the distance between $$x$$ and $$\theta$$.

$$\blacksquare$$

---

#### 2.2.2 Verify that the usual length of a vector in the plane or in three dimensional space has the properties (N1) to (N4) of a norm.
**Proof:**

(Easy to prove. TODO)

$$\blacksquare$$

---

#### 2.2.3 Prove (2).

**Proof:**

We wish to prove the **Reverse Triangle Inequality**, which is:

$$
|\|y\| - \|x\|| \leq \|y-x\|
$$

We have:

$$
\|x\|=\|x-y+y\| \leq \|x-y\| + \|y\| = \|y-x\| + \|y\| \\
\|x\| - \|y\| = \|y-x\| \\
$$

We also have:

$$
\|y\|=\|y-x+x\| \leq \|y-x\| + \|x\| \\
\|y\| - \|x\| \leq \|y-x\|
$$

Then, we get:

$$
|\|y\| - \|x\|| \leq \|y-x\|
$$

$$\blacksquare$$

---

#### 2.2.4 Show that we may replace **(N2)** by $$\|x\|=0 \Rightarrow x=0$$ without altering the concept of a norm. Show that nonnegativity of a norm also follows from **(N3)** and **(N4)**.

**Proof:**

We have from **(N2)** the following:

$$
\|\alpha x\|=|\alpha|\|x\|
$$

Assuming that $$\alpha=0$$, and knowing that $$0x=\theta$$, we get:

$$
\|0 x\|=|0|\|x\| \\
\|\theta\|=0 \\
$$

Thus we conclude that $$x=\theta \Rightarrow \|\theta\|=0$$ from **(N2)**.

$$\blacksquare$$

We wish to prove that $$\|x\| \geq 0$$.

$$
\|x\|=\|x+x-x\| \leq \|x+x\| + \|-x\| = \|2x\| + \|x\| = 2\|x\| + \|x\| \\
2\|x\| + \|x\| \geq \|x\| \\
2\|x\| \geq 0 \\
\|x\| \geq 0 \\
$$

$$\blacksquare$$

---

#### 2.2.5 Show that (3) defines a norm.

**Proof:**

(3) defines the norm:
$$
{\|x\|}_2=\sqrt{(|\eta_1|^2 + |\eta_2|^2 + \cdots + |\eta_n|^2)}
$$

(Easy to prove. TODO)

$$\blacksquare$$

---

#### 2.2.6 Let $$X$$ be the vector space of all ordered pairs $$x = (\xi_1, \xi_2), y = (\eta_1, \eta_2), \cdots$$ of real numbers. Show that norms on X are defined by

  $$
  {\|x\|}_1=|\eta_1| + |\eta_2| \\
  {\|x\|}_2={(\eta_1^2 + \eta_2^2)}^{1/2} \\
  {\|x\|}_\infty=\text{max } \{ |\xi_1|, |\xi_2| \}
  $$

**Proof:**

(Easy to prove. TODO)

$$\blacksquare$$

---

#### 2.2.7 Verify that (4) satisfies (N1) to (N4).

**Proof:**

(Easy to prove. TODO)

$$\blacksquare$$

---

#### 2.2.8 There are several norms of practical importance on the vector space of ordered n-tuples of numbers (cf. 2.2-2), notably those defined by  

  $$
  {\|x\|}_1=|\eta_1| + |\eta_2| + \cdots + |\eta_n| \\
  {\|x\|}_p={(|\eta_1|^p + |\eta_2|^p + \cdots + |\eta_n|^p)}^{1/p} \\
  {\|x\|}_\infty=\text{max } \{ |\xi_1|, |\xi_2|, \cdots, |\xi_n| \}
  $$
   
   **In each case, verify that (N1) to (N4) are satisfied.**

**Proof:**

(Easy to prove. TODO)

The second result follows from **Minkowski's Inequality**.

$$\blacksquare$$

---

#### 2.2.9 Verify that (5) defines a norm.

**Proof:**

(Easy to prove. TODO)

$$\blacksquare$$

---

#### 2.2.10 (Unit sphere) The sphere $$S(0; 1) = \{x \in X : \|x\| = 1\}$$ in a normed space $$X$$ is called the unit sphere. Show that for the norms in Prob. 6 and for the norm defined by the unit spheres look as shown in Fig. 16.

**Answer:**

(Check diagram in book after your curve sketching)

---

#### 2.2.11 (Convex set, segment) A subset $$A$$ of a vector space $$X$$ is said to be convex if $$x,y \in A$$ implies $$M=\{z \in X : z=\alpha x+(1-\alpha)y, 0\leq \alpha \leq 1\} \subset A$$. $$M$$ is called a closed segment with boundary points $$x$$ and $$y$$; any other $$z \in M$$ is called an interior point of $$M$$. Show that the closed unit ball $$B(0; 1) =\{x \in X : \|x\| \leq 1\}$$ in a normed space X is convex.

**Proof:**

The norm of the point $$z=\alpha x+(1-\alpha)y$$ is:

$$
\|z\|=\|\alpha x+(1-\alpha)y\| \leq \|\alpha x\|+\|(1-\alpha)y\| \\
= \alpha \|x\| + (1-\alpha) \|y\|
$$

Since $$\|x\| \leq 1$$ and $$\|y\| \leq 1$$, we get:

$$
\|z\| \leq \alpha + (1-\alpha) = 1
$$

Thus $$z \in X$$, and thus the closed unit ball is convex.

$$\blacksquare$$

---

#### 2.2.12 Using Prob. 11, show that $$\phi(x)={(\sqrt{\vert\xi_1\vert} + \sqrt{\vert\xi_2\vert})}^2$$ does not define a norm on the vector space of all ordered pairs $$x = (\xi_1, \xi_2), \cdots$$ of real nwnbers. Sketch the curve $$\phi(x) = 1$$ and compare it with Fig. 18.

**Proof:**

We can see that $$(1,0)$$ and $$(0,1)$$ fall on the unit circle defined by this "norm". For it to be a valid norm, the unit ball must be convex. Thus all points $$z=\alpha x+(1-\alpha)y$$ must lie in the unit ball, i.e., $$\|z\|$ \leq 1$.

Set $$\alpha=\frac{1}{2}$$, we get $$z=(\frac{1}{2}, \frac{1}{2})$$.

However, using this norm gives us $$\|z\|={(\frac{1}{\sqrt{2}} + \frac{1}{\sqrt{2}})}^2=2$$, which implies it does not lie in the unit ball. Thus, this is not a valid norm.

$$\blacksquare$$

---

#### 2.2.13 Show that the discrete metric on a vector space $$X \neq \{0\}$$ cannot be obtained from a norm. (Cf. 1.1-8.)

**Proof:**

For any metric derived from a norm, it must be translation invariant, i.e.:

$$
d(x+a,y+a)=d(x,y), x,y,a \in X \\
d(\alpha x,\alpha y)=d(x,y), x,y \in X, \alpha \in \mathbb{R}
$$

The discrete metric is defined as:

$$
d(x,y)=\begin{cases}
0 & \text{if } x=y \\
1 & \text{if } x \neq y
\end{cases}
$$

Assume that $$x \neq y$$. Then $$\alpha x \neq \alpha y$$. Then $$d(\alpha x, \alpha y)=1 \neq \alpha d(x,y)$$.

Thus, the discrete metric cannot be derived from a norm.

$$\blacksquare$$

---

#### 2.2.14 If $$d$$ is a metric on a vector space $$X \neq \{0\}$$ which is obtained from a norm, and $$\tilde{d}$$ is defined by $$\tilde{d}(x,x) = 0, \tilde{d}(x,y)=d(x,y)+1 (x \neq y)$$, show that $$d$$ cannot be obtained from a norm.

**Proof:**

For any metric derived from a norm, it must be translation invariant, i.e.:

$$
d(x+a,y+a)=d(x,y), x,y,a \in X \\
d(\alpha x,\alpha y)=d(x,y), x,y \in X, \alpha \in \mathbb{R}
$$

Assume that $$x \neq y$$. Then $$\alpha x \neq \alpha y$$. Then $$\tilde{d}(\alpha x, \alpha y)=d(\alpha x, \alpha y) + 1 = \alpha d(x,y) + 1 \neq \alpha d(x,y) + \alpha = \alpha \tilde{d}(x,y)$$.

$$\blacksquare$$

---

#### 2.2.15 (Bounded set) Show that a subset $$M$$ in a normed space $$X$$ is bounded if and only if there is a positive number $$c$$ such that $$\|x\| \leq  c$$ for every $$x \in M$$. (For the definition, see Prob. 6 in Sec. 1.2.)

**Proof:**

A set is bounded if $$\delta(x,y)<\infty$$, where $$\delta(x,y)=\sup d(x,y)$$.

$$
(\Rightarrow)
$$
Assume that $$M$$ is bounded. Then $$\delta(x,y)=\sup d(x,y)<\infty$$. This implies that $$d(x,y) \leq c, c \in \mathbb{R}$$ for all $$x,y \in M$$. Set $$y=\theta$$ and note that $$d(x,\theta)=\|x\|$$, to get:

$$
d(x,\theta)=\|x\| \leq c
$$

$$\blacksquare$$

$$
(\Leftarrow)
$$
Assume that there is a positive number $$c$$ such that $$\|x\| \leq  c$$ for every $$x \in M$$.

Then $$\|x\| \leq c$$.

Using the **Triangle Inequality**, and noting that $$d(x, \theta)=\|x\|$$ and $$d(y, \theta)=\|y\|$$, we get:

$$
d(x,y) \leq d(x,\theta) + d(\theta, y) \\
d(x,y) \leq \|x\| + \|y\| \\
d(x,y) \leq c + c \\
d(x,y) \leq 2c \\
\Rightarrow \delta(x,y) = \sup d(x,y) \leq 2c < \infty
$$

Thus, $$M$$ is bounded.

$$\blacksquare$$

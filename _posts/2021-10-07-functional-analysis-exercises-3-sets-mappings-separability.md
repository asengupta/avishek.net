---
title: "Functional Analysis Exercises 3 : Sets, Continuous Mappings, and Separability"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics"]
draft: false
---

This post lists solutions to many of the exercises in the **Open Set, Closed Set, Neighbourhood section 1.3** of *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**. This is a work in progress, and proofs may be refined over time.

#### 1.3.1. Justify the terms "open ball" and "closed ball" by proving that (a) any open ball is an open set, (b) any closed ball is a closed set.
**Proof:**

The definition of an open ball is:

$$
B(x_0,r)=\{x: d(x,x_0)<r, x \in X\}
$$

This implies that every point in the open ball is separated from $$x_0$$ by some distance $$r-\epsilon$$, $$\epsilon > 0$$. Assume such a point $$p$$, then we have $$d(p,x_0)=r-\epsilon$$. Assume a point $$y$$ in an open ball of radius $$\epsilon$$ centered on $$p$$. We then have: $$d(y,p)<\epsilon$$. Then we can see that:

$$
d(y,x_0) \leq d(y,p) + d(p,x_0)<\epsilon + (r-\epsilon)=r \\
\Rightarrow d(y,x_0)<r
$$

Hence, we can have an open ball of some size $$\epsilon$$ around every point $$x$$ in the open ball.

Hence, an open ball is an open set.

$$\blacksquare$$

The definition of a closed ball is:

$$
\bar{B}(x_0,r)=\{x: d(x,x_0)\leq r, x \in X\}
$$

Then the complement of $$\bar{B}(x_0,r)$$ becomes:

$$
U=\bar{B}(x_0,r)'=\{x: d(x,x_0)>r, x \in X\}
$$

This implies that every point in the complement of the closed ball (call it $$U$$) is separated from $$x_0$$ by some distance $$r+\epsilon$$, $$\epsilon \geq 0$$. Assume such a point $$p$$, then we have $$d(p,x_0)=r+\epsilon$$. Assume a point $$y$$ in an open ball of radius $$\epsilon$$ centered on $$p$$. We then have: $$d(y,p)<\epsilon$$. Then we can see that:

$$
\require{cancel}
d(p,x_0) \leq d(p,y) + d(y,x_0) \\
\Rightarrow r+\cancel\epsilon < \cancel\epsilon + d(y,x_0)\\
\Rightarrow d(y,x_0) > r
$$

Thus, an open ball $$\{y: d(y,x)<\epsilon, y \in U\}$$ can exist around any $$x \in U$$. Thus, $$U$$ is an open set.

The complement of the open set $$U$$ is the open set $$\bar{B}(x_0,r)$$.
Hence, a closed ball is a closed set.

$$\blacksquare$$

#### 1.3.2. What is an open ball $$B(x_0;1)$$ on $$\mathbb{R}$$? In $$\mathbb{C}$$? (Cf. 1.1-5.) In $$\mathbb{C}[a,b]$$? (Cf. 1.1-7.) Explain Fig. 8.

**Answer:**

The open ball $$B(x_0;1)$$ on $$\mathbb{R}$$ is the open interval $$(x_0-1,x_0+1)$$.  
The open ball $$B(x_0;1)$$ on $$\mathbb{C}$$ is the unit disk centered at $$x_0$$, excluding its circumference.  
The open ball $$B(x_0;1)$$ in $$\mathbb{C}[a,b]$$ is the set of functions $$x(t)$$ which satisfy the condition $$\text{sup }\vert x_0(t)-x(t)\vert < 1$$.

#### 1.3.3. Consider $$C[0,2\pi]$$ and determine the smallest $$r$$ such that $$y \in \bar{B}(x;r)$$, where $$x(t)=\text{sin }t$$ and $$y(t)=\text{cos }t$$.

**Answer:**

The center of the ball is $$x=\text{cos } t$$. The point $$y=\text{cos } t$$ needs to be in this ball. This gives us the condition that the $$y$$ can at most be on the boundary of the ball. The radius of this minimal ball then becomes:

$$
r_{xy}(t)=x-y=\text{sin } t-\text{cos } t
$$

We need to minimise the above expression, thus differentiating $$r_{xy}$$ with respect to $$t$$ and setting it to zero, we get:

$$
\frac{dr_{xy}(t)}{t}=\text{cos } t+\text{sin } t=0 \\
\Rightarrow \text{tan } t = -1 \\
\Rightarrow t = -\frac{\pi}{4}
$$

Plugging this value back into that of $$r_{xy}$$, we get:

$$
\text{min } r_{xy}=-\sqrt{2} \\
|\text{min } r_{xy}|=\sqrt{2}
$$

#### 1.3.4. Show that any nonempty set $$A\subset (X,d)$$ is open if and only if it is a union of open balls.

#### 1.3.5. It is important to realise that certain sets may be open and closed at the same time. (a) Show that this is always the case for $$X$$ and $$\emptyset$$. (b) Show that in a discrete metric space $$X$$ (cf. 1.1-8), every subset is open and closed.

#### 1.3.6. If $$x_0$$ is an accumulation point of a set $$A \subset (X,d)$$, show that any neighbourhood of $$x_0$$ contains infinitely many points of $$A$$.

#### 1.3.7. Describe the closure of each of the following subsets:  
  **(a) The integers on $$\mathbb{R}$$**  
  **(b) the rational numbers on $$\mathbb{R}$$**  
  **(c) the complex numbers with rational real and imagin~ parts in $$\mathbb{C}$$, (d) the disk $${z: \vert z \vert < 1}\subset C$$.**  

#### 1.3.8. Show that the closure $$\bar{B(xo; r)}$$ of an open ball $$B(xo; r)$$ in a metric space can differ from the closed ball $$\bar{B}(xo; r)$$.

#### 1.3.9. Show that $$A \subset \bar{A}$$, $$\bar{\bar{A}} = \bar{A}$$, $$\bar{A \cup B} = \bar{A} \cup \bar{B}$$, $$\bar{A \cap B} \subset \bar{A} \cap \bar{B}$$.

#### 1.3.10. A point $$x$$ not belonging to a closed set $$M \subset (X, d)$$ always has a nonzero distance from $$M$$. To prove this, show that $$x \in \bar{A}$$ if and only if $$D(x, A) = 0$$ (cf. Prob. 10, Sec. 1.2); here $$A$$ is any nonempty subset of $$X$$.

#### 1.3.11. **(Boundary)** A boundary point $$x$$ of a set $$A \subset (X, d)$$ is a point of $$X$$ (which may or may not belong to $$A$$) such that every neighbourhood of $$x$$ contains points of $$A$$ as well as points not belonging to $$A$$; and the boundary (or frontier) of $$A$$ is the set of all boundary points of $$A$$. Describe the boundary of 
  **(a) the intervals $$(-1,1)$$, $$[-1,1)$$, $$[-1,1]$$ on $$\mathbb{R}$$**  
  **(b) the set of all rational numbers on $$\mathbb{R}$$**  
  **(c) the disks $${z : \vert z \vert < 1} \subset C$$ and $${z : \vert z \vert \leq 1} \subset C$$.**  

#### 1.3.12. **(Space $$B[a, b]$$)** Show that $$B[a, b]$$, $$a < b$$, is not separable. (Cf. 1.2-2.)

#### 1.3.13. Show that a metric space $$X$$ is separable if and only if $$X$$ has a countable subset $$Y$$ with the following property. For every $$ \epsilon > 0$$ and every $$x \in X$$ there is a $$y \in Y$$ such that $$d(x, y) < \epsilon$$.

#### 1.3.14. (Continuous mapping) Show that a mapping $$T: X \rightarrow Y$$ is continuous if and only if the inverse image of any closed set $$M \subset Y$$ is a closed set in X.

#### 1.3.15. Show that the image of an open set under a continuous mapping need not be open. 

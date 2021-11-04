---
title: "Functional Analysis Exercises 7 : Vector Spaces"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics", "Kreyszig"]
draft: false
---

This post lists solutions to the exercises in the **Vector Space section 2.1** of *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**. This is a work in progress, and proofs may be refined over time.

### Notes
The requirements for a space to be a vector space are:

- **(VA1)** Symmetric with respect to addition, i.e., $$x+y=y+x, x,y \in X$$
- **(VA2)** Existence of identity element, i.e., $$x+\theta=x, x,\theta \in X$$
- **(VA3)** Existence of inverse element, i.e., $$x+(-x)=\theta, x,\theta \in X$$
- **(VA4)** Associative with respect to addition, i.e., $$x+(y+z)-=(x+y)+z, x,y,z \in X$$

- **(VM1)** Associative with respect to scalar multiplication, i.e., $$\alpha (\beta x) = (\alpha \beta) x, x \in X, \alpha, \beta \in \mathbb{R}$$
- **(VM2)** Existence of identity element, i.e., $$\alpha_0 x=x, x \in X, \alpha_0=1 \in \mathbb{R}$$
- **(VM3)** Distributive with respect to addition of scalars, i.e., $$(\alpha + \beta) x=\alpha x + \beta x, x \in X, \alpha, \beta \in \mathbb{R}$$
- **(VM4)** Distributive with respect to addition of vectors, i.e., $$\alpha (x+y)=\alpha x + \alpha y, x,y \in X, \alpha \in \mathbb{R}$$


#### 2.1.1 Show that the set of all real numbers, with the usual addition and multiplication, constitutes a one-dimensional real vector space, and the set of all complex numbers constitutes a one-dimensional complex vector space.
**Proof:**

$$\blacksquare$$

---

#### 2.1.2 Prove (1) and (2).
**Proof:**

We have to prove:

**(1a)** $$0x=\theta$$  
**(1b)** $$\alpha \theta=\theta$$  
**(1c)** $$(-1) x=-x$$

where $$\alpha \in \mathbb{R}, x, \theta \in X$$

**(1a)** We have:

$$
\begin{array} {lr}
(0x) + (0x) = (0+0)x = 0x && \mathbf{\text{ (by (VM3))}} \\
(0x) + (0x) + (-(0x))= (0x) + (-(0x))  && \mathbf{\text{ (adding (0x) on both sides)}} \\
(0x) + \theta = \theta && \mathbf{\text{ (by (VA3))}} \\
0x = \theta && \mathbf{\text{ (by (VA2))}}
\end{array}
$$

$$\blacksquare$$

**(1b)** We have:

$$
\begin{array} {lr}
\alpha(0x)=(\alpha 0)x && \mathbf{\text{ (by (VM1))}} \\
(\alpha 0)x=0x=\theta
\end{array}
$$

$$\blacksquare$$

**(1c)** We have:

$$
\begin{array} {lr}
0x = \theta && \mathbf{\text{ (already proved)}} \\
(1-1)x = \theta \\
1x + (-1)x = \theta && \mathbf{\text{ (by (VM3))}} \\
x + (-1)x = \theta &&\mathbf{\text{ (by (VM2))}} \\
x + (-x) + (-1) x  = \theta + (-x) &&\mathbf{\text{ (adding (-x) on both sides)}} \\
\theta + (-1) x = \theta + (-x) && \mathbf{\text{ (by (VA3))}} \\
(-1) x + \theta = (-x) + \theta && \mathbf{\text{ (by (VA1))}} \\
(-1) x = (-x) && \mathbf{\text{ (by (VA2))}}
\end{array}
$$

$$\blacksquare$$

---

#### 2.1.3 Describe the span of $$M = {(1,1,1), (0,0,2)}$$ in $$\mathbb{R^1}$$.
**Answer:**

The span of $$M$$ is described by:

$$
\begin{bmatrix}
1 && 0 \\
1 && 0 \\
1 && 2
\end{bmatrix}
\bullet
\begin{bmatrix}
x \\
y
\end{bmatrix}
$$

Geometrically, this is a plane whose normal is perpendicular to both $$(1,1,1)$$ and $$(0,0,2)$$. Specifically, from the first perpendicularity with $$(0,0,0)$$:

$$
0x+0y+2z=0 \\
z=0
$$

From the second perpendicularity with $$(1,1,1)$$:

$$
x+y+z=0 \\
x+y=0 \text{ (since z=0)} \\
x=-y
$$

Choose $$x=1$$, then $$y=-1$$. Then, one choice of the normal vector is $$(1,-1,0)$$. The equation of the plane then becomes:

$$
x-y=0
$$

---

#### 2.1.4 Which of the following subsets of $$\mathbb{R}^3$$ constitute a subspace of $$\mathbb{R}^3$$? [Here, $$x = (\xi_1, \xi_2, \xi_3)$$.]
  **(a) All $$x$$ with $$\xi_1=\xi_2$$ and $$\xi_3=O$$.**
  **(b) All $$x$$ with $$\xi_1=\xi_2+1$$.**
  **(c) All $$x$$ with positive $$\xi_1$$, $$\xi_2$$, $$\xi_3$$.**
  **(d) All $$x$$ with $$\xi_1-\xi_2+\xi_3=k=\text{const}$$.**

**Proof:**

$$\blacksquare$$

---


#### 2.1.5 s. Show that $${x_1, \cdots, x_n}$$, where $$x_j(t) = t^j$$ , is a linearly independent set in the space $$C[a,b]$$.
**Proof:**

$$\blacksquare$$

---

#### 2.1.6 Show that in an $$n$$-dimensional vector space $$X$$, the representation of any $$x$$ as a linear combination of given basis vectors $$e_1, \cdots, e_n$$ is unique.
**Proof:**

$$\blacksquare$$

---

#### 2.1.7 Let $$\{e_1, \cdots, e_n\}$$ be a basis for a complex vector space $$X$$. Find a basis for $$X$$ regarded as a real vector space. What is the dimension of $$X$$ in either case?
**Proof:**

$$\blacksquare$$

---

#### 2.1.8 If $$M$$ is a linearly dependent set in a complex vector space $$X$$, is $$M$$ linearly dependent in $$X$$, regarded as a real vector space?
**Proof:**

$$\blacksquare$$

---

#### 2.1.9 On a fixed interval $$[a,b] \subset \mathbb{R}$$, consider the set $$X$$ consisting of all polynomials with real coefficients and of degree not exceeding a given $$n$$, and the polynomial $$x=0$$ (for which a degree is not defined in the usual discussion of degree). Show that $$X$$, with the usual addition and the usual mUltiplication by real numbers, is a real vector space of dimension $$n+1$$. Find a basis for $$X$$. Show that we can obtain a complex vector space $$\tilde{X}$$ in a similar fashion if we let those coefficients be complex. Is $$X$$ a subspace of $$\tilde{X}$$?
**Proof:**

$$\blacksquare$$

---

#### 2.1.10 If $$Y$$ and $$Z$$ are subspaces of a vector space $$X$$, show that $$Y\capZ$$ is a subspace of $$X$$, but $$Y\cupZ$$ need not be one. Give examples.
**Proof:**

$$\blacksquare$$

---

#### 2.1.11 If $$M \neq \emptyset$$ is any subset of a vector space $$$$, show that span $$M$$ is a subspace of $$X$$.
**Proof:**

$$\blacksquare$$

---

#### 2.1.12 Show that the set of all real two-rowed square matrices forms a vector space $$X$$. What is the zero vector in $$X$$? Determine dim $$X$$. Find a basis for $$X$$. Give examples of subspaces of X. Do the symmetric matrices $$x \in X$$ form a subspace? The singular matrices?
**Proof:**

$$\blacksquare$$

---

#### 2.1.13 (Product) Show that the Cartesian product $$X = X_1 \times X_2$$ of two vector spaces over the same field becomes a vector space if. we define the two algebraic operations by  
  $$(x_1. x_2) + (y_1, y_2) = (x_1 +y_1. x_2 + y_2)$$,  
  $\alpha(x_1, x_2) = (\alpha x_1, \alpha x_2)$$.

**Proof:**

$$\blacksquare$$

---

#### 2.1.14 (Quotient space, codimension) Let $$Y$$ be a subspace of a vector space $$X$$. The coset of an element $$x \in X$$ with respect to $$Y$$ is denoted by $$x + Y$$ and is defined to be the set (see Fig. 12)
  $$x+Y={v\vert v=x+y, \in Y}$$.
  
  **Show that the distinct cosets form a partition of X. Show that under algebraic operations defined by (see Figs. 13, 14)**
  
  $$
  (w+Y)+(x+Y)=(w+x)+Y
  \alpha(x+Y)=\alpha x+Y
  $$

**these cosets constitute the elements of a vector space. This space is called the quotient space (or sometimes factor space) of $$X$$ by $$Y$$ (or modulo $$Y$$) and is denoted by $$X/Y$$. Its dimension is called the codimension of $$Y$$ and is denoted by codim $$Y$$, that is, $$\text{codim } Y=\text{dim }(X/Y)$$.**

**Proof:**

$$\blacksquare$$

---

#### 2.1.15 Let $$X=\mathbb{R}^3$$ and $$Y=\{(\xi_1,0,0) \vert \xi \in \mathbb{R}\}$$. Find $$X/¥$$, $$X/X$$, $$X/{O}$$.

**Proof:**

$$\blacksquare$$

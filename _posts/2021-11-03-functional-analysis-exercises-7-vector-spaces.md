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
- **(VA4)** Associative with respect to addition, i.e., $$x+(y+z)=(x+y)+z, x,y,z \in X$$

- **(VM1)** Associative with respect to scalar multiplication, i.e., $$\alpha (\beta x) = (\alpha \beta) x, x \in X, \alpha, \beta \in \mathbb{R}$$
- **(VM2)** Existence of identity element, i.e., $$\alpha_0 x=x, x \in X, \alpha_0=1 \in \mathbb{R}$$
- **(VM3)** Distributive with respect to addition of scalars, i.e., $$(\alpha + \beta) x=\alpha x + \beta x, x \in X, \alpha, \beta \in \mathbb{R}$$
- **(VM4)** Distributive with respect to addition of vectors, i.e., $$\alpha (x+y)=\alpha x + \alpha y, x,y \in X, \alpha \in \mathbb{R}$$


#### 2.1.1 Show that the set of all real numbers, with the usual addition and multiplication, constitutes a one-dimensional real vector space, and the set of all complex numbers constitutes a one-dimensional complex vector space.
**Proof:**

Consider $$\mathbb{R}$$.

- **(VA1)** Symmetric with respect to addition, i.e., $$x+y=y+x, x,y \in \mathbb{R}$$
- **(VA2)** Existence of identity element, i.e., $$x+0=x, x,0 \in \mathbb{R}$$
- **(VA3)** Existence of inverse element, i.e., $$x+(-x)=0, x,0 \in \mathbb{R}$$
- **(VA4)** Associative with respect to addition, i.e., $$x+(y+z)=(x+y)+z, x,y,z \in \mathbb{R}$$

- **(VM1)** Associative with respect to scalar multiplication, i.e., $$\alpha (\beta x) = (\alpha \beta) x, x \in X, \alpha, \beta \in \mathbb{R}$$
- **(VM2)** Existence of identity element, i.e., $$1x=x, x \in X, 1 \in \mathbb{R}$$
- **(VM3)** Distributive with respect to addition of scalars, i.e., $$(\alpha + \beta) x=\alpha x + \beta x, x \in X, \alpha, \beta \in \mathbb{R}$$
- **(VM4)** Distributive with respect to addition of vectors, i.e., $$\alpha (x+y)=\alpha x + \alpha y, x,y \in X, \alpha \in \mathbb{R}$$

$$\blacksquare$$

Consider $$\mathbb{C}$$.

- **(VA1)** Symmetric with respect to addition, i.e., $$(a+ib)+(c+id)=(c+id)+(a+ib)=(a+c) + i(b+d), a+ib,c+id \in \mathbb{C}$$
- **(VA2)** Existence of identity element, i.e., $$(a+ib)+(0_0i)=a+ib, a+ib, c+id \in \mathbb{C}$$
- **(VA3)** Existence of inverse element, i.e., $$(a+ib)+(-a-ib)=0+0i, x,0 \in \mathbb{C}$$
- **(VA4)** Associative with respect to addition, i.e., $$x_1+ix_2+(y_1+iy_2+z_1+iz_2)  \\
  =(x_1+ix_2+y_1+iy_2)+z_1+iz_2  \\
  =(x_1+y_1+z_1)+i(x_2+y_2+z_2), x_1+ix_2,y_1+iy_2,z_1+iz_2 \in \mathbb{C}$$

- **(VM1)** Associative with respect to scalar multiplication, i.e., $$\alpha (\beta x) = (\alpha \beta) x, x \in X, \alpha, \beta \in \mathbb{R}$$
- **(VM2)** Existence of identity element, i.e., $$(1+0i)(a+ib)=a+ib, a+ib \in \mathbb{C}$$
- **(VM3)** Distributive with respect to addition of scalars, i.e., $$(\alpha + \beta) x=\alpha x + \beta x, x \in X, \alpha, \beta \in \mathbb{R}$$
- **(VM4)** Distributive with respect to addition of vectors, i.e., $$\alpha (x+y)=\alpha x + \alpha y, x,y \in X, \alpha \in \mathbb{R}$$

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
  **(a) All $$x$$ with $$\xi_1=\xi_2$$ and $$\xi_3=0$$.**  
  **(b) All $$x$$ with $$\xi_1=\xi_2+1$$.**  
  **(c) All $$x$$ with positive $$\xi_1$$, $$\xi_2$$, $$\xi_3$$.**  
  **(d) All $$x$$ with $$\xi_1-\xi_2+\xi_3=k=\text{const}$$.**

**Answer:**

For a subset to be a subspace, it needs to satisfy the following criterion:

$$
\alpha x + \beta y \in X, \alpha, \beta \in \mathbb{R}
$$

**(a)** Consider two arbitrary members $$x=(\xi, \xi, 0)$$ and $$y=(\eta, \eta, 0)$$ of the given subset (call it $$X$$).

Then, we have:

$$
\alpha x + \beta y=\alpha (\xi, \xi, 0) + \beta (\eta, \eta, 0) \\
= (\alpha\xi, \alpha\xi, \alpha 0) + (\beta\eta, \beta\eta, \beta 0) \\
= (\alpha\xi + \beta\eta, \alpha\xi + \beta\eta, \alpha 0 + \beta 0) \\
= (\alpha\xi + \beta\eta, \alpha\xi + \beta\eta, 0)  \in X
$$

Thus, $$X$$ is a subspace of $$\mathbb{R}^3$$.

**(b)** Consider two arbitrary members $$x=(\xi+1, \xi, 0)$$ and $$y=(\eta+1, \eta, 0)$$ of the given subset (call it $$X$$).

Then, we have:

$$
\require{cancel}
\alpha x + \beta y=\alpha (\xi+1, \xi, 0) + \beta (\eta+1, \eta, 0) \\
= (\alpha\xi + \alpha, \alpha\xi, \alpha 0) + (\beta\eta + \beta, \beta\eta, \beta 0) \\
= [(\alpha\xi + \beta\eta) + (\alpha + \beta), (\alpha\xi + \beta\eta), \alpha 0 + \beta 0] \\
= [(\alpha\xi + \beta\eta) + (\alpha + \beta), (\alpha\xi + \beta\eta), 0] \cancel{\in} X\\
$$

Thus, $$X$$ is not a subspace of $$\mathbb{R}^3$$.

**(c)** Consider two arbitrary members $$x=(\xi_1, \xi_2, \xi_3)$$ and $$y=(\eta_1, \eta_2, \eta_3)$$ of the given subset (call it $$X$$), with $$\xi_i,\eta_i \geq 0$$.

Then, we have:

$$
\alpha x + \beta y=\alpha (\xi_1, \xi_2, \xi_3) + \beta (\eta_1, \eta_2, \eta_3) \\
= (\alpha\xi_1, \alpha\xi_2, \alpha\xi_3) + (\beta\eta_1, \beta\eta_2, \beta\eta_3) \\
= (\alpha\xi_1 + \beta\eta_1, \alpha\xi_2 + \beta\eta_2, \alpha\xi_3 + \beta\eta_3)
$$

Choose any $$\xi_1>\eta_1$$, and $$\alpha=-1$$, $$\beta=1$$. Then, we have:

$$
\require{cancel}
\alpha\xi_1 + \beta\eta_1 = -\xi_1 + \eta_1 < 0 \cancel\in X
$$

Thus, $$X$$ is not a subspace of $$\mathbb{R}^3$$.

**(d)** Consider two arbitrary members $$x=(\xi_1, \xi_2, k-\xi_1+\xi_2)$$ and $$y=(\eta_1, \eta_2, k-\eta_1+\eta_2)$$ of the given subset (call it $$X$$).

Then we have:

$$
\require{cancel}
\alpha x + \beta y=\alpha (\xi_1, \xi_2, k-\xi_1+\xi_2) + \beta (\eta_1, \eta_2, k-\eta_1+\eta_2) \\
= [\alpha\xi_1, \alpha\xi_2, \alpha(k-\xi_1+\xi_2)] + [\beta\eta_1, \beta\eta_2, \beta(k-\eta_1+\eta_2)] \\
= [\alpha\xi_1 + \beta\eta_1, \alpha\xi_2 + \beta\eta_2, (\alpha + \beta) k - (\alpha\xi_1 + \beta\eta_1) + (\alpha\xi_2 + \beta\eta_2)] \cancel\in X
$$

Thus, $$X$$ is not a subspace of $$\mathbb{R}^3$$.

$$\blacksquare$$

---

#### 2.1.5 s. Show that $${x_1, \cdots, x_n}$$, where $$x_j(t) = t^j$$, is a linearly independent set in the space $$C[a,b]$$.
**Proof:**

A set $$\{x_1, x_2, \cdots, x_n\}$$ is linearly independent if:

$$
\alpha_1 x_1 + \alpha_2 x_2 + \cdots \alpha_n x_n = x_0
$$

only for all $$\alpha_i=0$$.

Any linear combination of the given set, call it $$X$$, is given by:

$$
f(t)=\alpha_1 t^1 + \alpha_2 t^2 + \cdots + \alpha_n t^n
$$

To prove that $$X$$ is linearly independent, we need to prove that that the zero vector $$f(t)=0=f_0(t)$$ is not possible for any combination of $$\alpha_i$$.

This is a polynomial of degree $$n$$. Fix all $$\alpha_i$$, with not all of them zero.  
By the **Fundamental Theorem of Algebra**, we know that it can have at most $$n$$ roots of this polynomial. Thus, there are at most $$n$$ values of $$t$$ for which $$f(t)=0$$. Thus, it is not zero for the remaining uncountable values of $$t \in [a,b]$$, therefore for an arbitrary combination of $$\alpha_i$$, $$f(t) \neq f_0(t)$$.

Thus, $$X$$ is a linearly independent set.

$$\blacksquare$$

---

#### 2.1.6 Show that in an $$n$$-dimensional vector space $$X$$, the representation of any $$x$$ as a linear combination of given basis vectors $$e_1, \cdots, e_n$$ is unique.
**Proof:**

Assume that $$x=\alpha_1 e_1 + \alpha_2 e_2 + \cdots + \alpha_n e_n$$.
Assume that $$x$$ can also be represented by a different linear combination $$\beta_i$$, such that:

$$x=\beta_1 e_1 + \beta_2 e_2 + \cdots + \beta_n e_n$$

Then we have:

$$
\alpha_1 e_1 + \alpha_2 e_2 + \cdots + \alpha_n e_n=\beta_1 e_1 + \beta_2 e_2 + \cdots + \beta_n e_n \\
(\alpha_1 e_1 + \alpha_2 e_2 + \cdots + \alpha_n e_n)+(-\beta_1 e_1) +(-\beta_2 e_2) + \cdots + (-\beta_n e_n)=\beta_1 e_1 + \beta_2 e_2 + \cdots + \beta_n e_n + (-\beta_1 e_1) +(-\beta_2 e_2) + \cdots + (-\beta_n e_n) \\
(\alpha_1 - \beta_1) e_1 + (\alpha_2 - \beta_2) e_2 + \cdots + (\alpha_n -\beta_n) e_n =(\beta_1 - \beta_1) e_1 + (\beta_2 - \beta_2) e_2 + \cdots + (\beta_n - \beta_n) e_n  \\
(\alpha_1 - \beta_1) e_1 + (\alpha_2 - \beta_2) e_2 + \cdots + (\alpha_n -\beta_n) e_n =0 e_1 + 0 e_2 + \cdots + 0 e_n  \\
$$

Then equating the coefficients on both sides, we get:

$$
\alpha_i-\beta_i=0 \\
\alpha_i=\beta_i
$$

Thus, the representation of any $$x$$ as a linear combination of given basis vectors $$e_1, \cdots, e_n$$ is unique.

$$\blacksquare$$

---

#### 2.1.7 Let $$\{e_1, \cdots, e_n\}$$ be a basis for a complex vector space $$X$$. Find a basis for $$X$$ regarded as a real vector space. What is the dimension of $$X$$ in either case?
**Answer:**

A complex vector space is a vector space whose field of scalars is the complex numbers. Then any vector $$x$$ in this complex vector space is representable as:

$$
x=(a_1 + ib_1) e_1 + (a_2 + ib_2) e_2 + \cdots + (a_n + ib_n) e_n \\
=(a_1 e_1 + a_2 e_2 + \cdots + a_n e_n) + (b_1 ie_1 + b_2 ie_2 + \cdots + b_n ie_n)
$$

The basis for $$X$$ regarded as a real vector space are:

$$
(e_1, e_2, \cdots, e_n, ie_1, ie_2, \cdots, ie_n)
$$

The dimension of the complex space is $$n$$.  
The dimension of the complex space regarded as a real vector space is $$2n$$.

---

#### 2.1.8 If $$M$$ is a linearly dependent set in a complex vector space $$X$$, is $$M$$ linearly dependent in $$X$$, regarded as a real vector space?
**Proof:**

Consider the set $$\{u=i,v=-1\}$$. This is a linearly dependent set because $$iv=u$$, since $$i.i=-1$$. But there is no $$\alpha \in \mathbb{C}$$ which gives $$\alpha u=v$$, i.e., $$\alpha i=-1$$.

Thus, $$X$$, regarded as a real vector space, is not necessarily dependent.

$$\blacksquare$$

---

#### 2.1.9 On a fixed interval $$[a,b] \subset \mathbb{R}$$, consider the set $$X$$ consisting of all polynomials with real coefficients and of degree not exceeding a given $$n$$, and the polynomial $$x=0$$ (for which a degree is not defined in the usual discussion of degree). Show that $$X$$, with the usual addition and the usual multiplication by real numbers, is a real vector space of dimension $$n+1$$. Find a basis for $$X$$. Show that we can obtain a complex vector space $$\tilde{X}$$ in a similar fashion if we let those coefficients be complex. Is $$X$$ a subspace of $$\tilde{X}$$?
**Proof:**

A polynomial not exceeding degree $$n$$ is given as: $$f(x)=\sum\limits_{i=0}^n \alpha_i x^i$$.

$$\blacksquare$$

---

#### 2.1.10 If $$Y$$ and $$Z$$ are subspaces of a vector space $$X$$, show that $$Y\cap Z$$ is a subspace of $$X$$, but $$Y\cup Z$$ need not be one. Give examples.
**Proof:**

We have, for all $$\alpha, \beta \in \mathbb{R}$$:

$$
\alpha y_1 + \beta y_2 \in Y \\
\alpha z_1 + \beta z_2 \in Z
$$

Assume $$x \in Y \cap Z$$.  
Then $$x \in Y$$ and $$x \in Z$$.  
Then $$\alpha x_1 + \beta x_2 \in Y$$ and $$\alpha x_1 + \beta x_2 \in Z$$.  
Then $$\alpha x_1 + \beta x_2 \in Y \cap Z$$

Thus, $$Y \cap Z$$ is a subspace of $$X$$.

$$\blacksquare$$

In $$\mathbb{R}^2$$, the vector space with the basis vector $$u=(1,0)$$ and the vector space with the basis vector $$v=(0,1)$$ gives two vector spaces $$Y$$ and $$Z$$. Choose $$\alpha=1$$, $$\beta=1$$, then $$\alpha u + \beta v = (1,1)$$ does not belong to $$Y \cup Z$$.

Thus, $$Y \cap Z$$ is not necessarily a subspace of $$X$$.

$$\blacksquare$$

---

#### 2.1.11 If $$M \neq \emptyset$$ is any subset of a vector space $$X$$, show that span $$M$$ is a subspace of $$X$$.
**Proof:**

Let $$M \subset X$$. Let $$e_1, e_2, \cdots, e_n \in M$$. Then, the span of $$M$$ is $$\sum\limits_{i=1}^n\alpha_i e_n$$.

Then, every $$x=\sum\limits_{i=1}^n\alpha_i e_n, x \in M$$. Pick any two arbitrary points in $$M$$, so that:

$$
x_1=\sum\limits_{i=1}^n\alpha_i e_i \\
x_2=\sum\limits_{i=1}^n\beta_i e_i \\
$$

Then, we get:

$$
ax_1+bx_2=\sum\limits_{i=1}^n a\alpha_i e_i + \sum\limits_{i=1}^n b\beta_i e_i \\
=\sum\limits_{i=1}^n (a\alpha_i+b\beta_i) e_i = \sum\limits_{i=1}^n k_i e_i \in M
$$

Thus, span $$M$$ is a subspace of $$X$$.

$$\blacksquare$$

---

#### 2.1.12 Show that the set of all real two-rowed square matrices forms a vector space $$X$$. What is the zero vector in $$X$$? Determine dim $$X$$. Find a basis for $$X$$. Give examples of subspaces of X. Do the symmetric matrices $$x \in X$$ form a subspace? The singular matrices?
**Proof:**

A real-valued $$2 \times 2$$ matrix is of the form:

$$
\begin{bmatrix}
a && b \\
c && d
\end{bmatrix}
$$

Assume $$x_1$$, $$x_2$$ as below:

$$
x_1=\begin{bmatrix}
a && b \\
c && d
\end{bmatrix}\\

x_2=\begin{bmatrix}
p && q \\
q && s
\end{bmatrix}
$$

Then, we have:

$$
\alpha x_1 + \beta x_2 = \alpha \begin{bmatrix}
a && b \\
c && d
\end{bmatrix}
+
\beta \begin{bmatrix}
p && q \\
r && s
\end{bmatrix} \\

= \begin{bmatrix}
\alpha a && \alpha b \\
\alpha c && \alpha d
\end{bmatrix}
+
\begin{bmatrix}
\beta p && \beta q \\
\beta r && \beta s
\end{bmatrix} \\

= \begin{bmatrix}
\alpha a + \beta p && \alpha b + \beta q \\
\alpha c + \beta r && \alpha d + \beta s
\end{bmatrix}
$$

This is also a $$2 \times 2$$ matrix, thus the set of all real two-rowed square matrices forms a vector space.

The zero vector of $$X$$ is $$\begin{bmatrix}
0 && 0 \\
0 && 0
\end{bmatrix}$$

The dimension of $$X$$ is 4. A possible basis for $$X$$ is:

$$
\begin{bmatrix}
1 && 0 \\
0 && 0
\end{bmatrix},
\begin{bmatrix}
0 && 1 \\
0 && 0
\end{bmatrix},
\begin{bmatrix}
0 && 0 \\
1 && 0
\end{bmatrix},
\begin{bmatrix}
0 && 0 \\
0 && 1
\end{bmatrix}
$$

An example of a subspace of $$X$$ is $$\begin{bmatrix}
k && 0 \\
0 && 0
\end{bmatrix}, k \in \mathbb{R}
$$.

Yes, the symmetric matrices form a subspace.
No, the singular matrices do not form a subspace. Here is a counter-example. Take $$x$$ and $$y$$ to be as follows:

$$
x=\begin{bmatrix}
2 && 6 \\
4 && 12
\end{bmatrix},
y=\begin{bmatrix}
1 && 1 \\
1 && 1
\end{bmatrix}
$$

The determinants for both $$x$$ and $$y$$ are zero, thus, they are singular.

Then, setting $$\alpha=1$$, $$\beta=1$$, we get:

$$
\alpha x + \beta = \begin{bmatrix}
2 && 6 \\
4 && 12
\end{bmatrix}
+
\begin{bmatrix}
1 && 1 \\
1 && 1
\end{bmatrix}=
\begin{bmatrix}
3 && 7 \\
5 && 13
\end{bmatrix}
$$

The determinant of the result is $$39-35=4 \neq 0$$, thus it is not a singular matrix.

$$\blacksquare$$

---

#### 2.1.13 (Product) Show that the Cartesian product $$X = X_1 \times X_2$$ of two vector spaces over the same field becomes a vector space if. we define the two algebraic operations by  
  $$(x_1. x_2) + (y_1, y_2) = (x_1 +y_1. x_2 + y_2)$$,  
  $$\alpha(x_1, x_2) = (\alpha x_1, \alpha x_2)$$.

**Proof:**

[Easy. TODO]

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

To prove that the cosets form a partition of $$X$$, we need to prove that an arbitrary element $$x \in X$$ belongs to one and only one coset.

Suppose $$x \in X$$ belongs to two cosets $$u+Y$$ and $$v+Y$$. Then, we have from the definition:

$$
x=v|v=u+y_1,y \in Y \\
x=v|v=v+y_2,y \in Y
$$

Then, we have:

$$
u+y_1=v+y_2 \\
u-v=y_2-y_1 \in Y\\
$$

Then $$u-v \in Y$$. Then $$u-v=y_0, y_0 \in Y$$.  
Then $$u=v+y_0$$, where $$y_0 \in Y$$. Then $$u \in v+Y$$. Since $$u \in u+Y$$, $$u+Y$$ and $$v+Y$$ are the same coset.

$$\blacksquare$$

[Easy to prove that it's a vector space. TODO]

---

#### 2.1.15 Let $$X=\mathbb{R}^3$$ and $$Y=\{(\xi_1,0,0) \vert \xi \in \mathbb{R}\}$$. Find $$X/Y$$, $$X/X$$, $$X/{0}$$.

**Answer:**

Loosely, the quotient space is the set of points which can translate cosets to cover the entire vector space.

For $$X/Y$$, we get the coset as the set of parallel subspaces along the vector $$(1,0,0)$$.

For $$X/X$$, any translation of the coset $$X$$ covers the entire vector space $$X$$. No translation also covers the entire space. This implies that $$x=\{0\}$$.

For $$X/\{0\}$$, we have $$x+\{0\}=\{v:v=x+0\}$$, i.e., each coset is the point itself. Thus the set of points required to partition $$X$$ into cosets is $$X$$ itself.

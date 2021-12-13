---
title: "Functional Analysis Exercises 12 : Linear Operators"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics", "Kreyszig"]
draft: false
---

This post lists solutions to the exercises in the **Linear Operators section 2.6** of *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**. This is a work in progress, and proofs may be refined over time.


#### 2.6.1. Show that the identity, zero, and differentiation operators are linear.

**Proof:**

The identity operator is $$I_x x=x$$. We have:

$$
I_x(\alpha x + \beta y) = \alpha x + \beta y = \alpha I_x x + \beta I_x y
$$

The zero operator is $$0x=0$$. We have:

$$
0(\alpha x + \beta y) = \alpha 0 + \beta 0 = \alpha 0x + \beta 0y
$$

The differentiation operator is $$Tx=x'(t)$$. We have:

$$
T(\alpha x(t) + \beta y(t)) = (\alpha x(t) + \beta y(t))' = \alpha x'(t) + \beta y'(t) = \alpha Tx + \beta Ty
$$

$$\blacksquare$$

---

#### 2.6.2. Show that the operators $$T_1, \cdots , T_4$$ from $$\mathbb{R}^2$$ into $$\mathbb{R}^2$$ defined by

$$
(\xi_1, \xi_2) \mapsto (\xi_1, 0) \\
(\xi_1, \xi_2) \mapsto (0, \xi_2) \\
(\xi_1, \xi_2) \mapsto (\xi_2, \xi_1) \\
(\xi_1, \xi_2) \mapsto (\gamma\xi_1, \gamma\xi_2)
$$

**respectively, are linear, and interpret these operators geometrically.**

**Proof:**

We take $$x=(x_1,x_2)$$ and $$y=(y_1,y_2)$$.

$$
T_1(x_1,x_2) = (x_1,0)
$$

We have:

$$
T_1(\alpha x + \beta y) = T[(\alpha x_1, \alpha x_2) + (\beta y_1, \beta y_2)] \\
= T_1(\alpha x_1 + \beta y_1, \alpha x_2 + \beta y_2) = (\alpha x_1 + \beta y_1, 0) \\
= (\alpha x_1, 0) + (\beta y_1, 0) = \alpha T_1 x + \beta T_1 y
$$

$$\blacksquare$$

We take $$x=(x_1,x_2)$$ and $$y=(y_1,y_2)$$.

$$
T_2(x_1,x_2) = (0,x_2)
$$

We have:

$$
T_2(\alpha x + \beta y) = T[(\alpha x_1, \alpha x_2) + (\beta y_1, \beta y_2)] \\
= T_2(\alpha x_1 + \beta y_1, \alpha x_2 + \beta y_2) = (0, \alpha x_2 + \beta y_2) \\
= (0, \alpha x_2) + (0, \beta y_2) = \alpha T_2 x + \beta T_2 y
$$

$$\blacksquare$$

We take $$x=(x_1,x_2)$$ and $$y=(y_1,y_2)$$.

$$
T_3(x_1,x_2) = (x_2,x_1)
$$

We have:

$$
T_3(\alpha x + \beta y) = T[(\alpha x_1, \alpha x_2) + (\beta y_1, \beta y_2)] \\
= T_3(\alpha x_1 + \beta y_1, \alpha x_2 + \beta y_2) = (\alpha x_2 + \beta y_2, \alpha x_1 + \beta y_1)
= \alpha (x_2, x_1) + \beta (y_2, y_1) = \alpha T_3 x + \beta T_3 y
$$

$$\blacksquare$$

We take $$x=(x_1,x_2)$$ and $$y=(y_1,y_2)$$.

$$
T_4(x_1,x_2) = (\gamma x_1, \gamma x_2)
$$

We have:

$$
T_4(\alpha x + \beta y) = T[(\alpha x_1, \alpha x_2) + (\beta y_1, \beta y_2)] \\
= T_4(\alpha x_1 + \beta y_1, \alpha x_2 + \beta y_2) = (\gamma \alpha x_1 + \gamma \beta y_1, \gamma \alpha x_2 + \gamma \beta y_2) \\
= \alpha \gamma (x_1, x_2) + \beta \gamma (y_1, y_2) = \alpha T_4 x + \beta T_4 y
$$

$$\blacksquare$$

---

#### 2.6.3. What are the domain, range and null space of $$T_1, T_2, T_3$$ in Prob. 2?

**Answer:**

The operators are:

$$
T_1(x_1,x_2) = (x_1,0) \\
T_2(x_1,x_2) = (0, x_2) \\
T_3(x_1,x_2) = (x_2, x_1)
$$

For **$$T_1$$**, we have:

Domain is $$\mathbb{R}^2$$.  
Range is $$((-\infty,0), (+\infty,0))$$.  
Null Space is $$((0,-\infty), (0,+\infty))$$.

For **$$T_2$$**, we have:

Domain is $$\mathbb{R}^2$$.  
Range is $$((0,-\infty), (0,+\infty))$$.
Null Space is $$((-\infty,0), (+\infty,0))$$.

For **$$T_2$$**, we have:

Domain is $$\mathbb{R}^2$$.  
Range is $$\mathbb{R}^2$$.  
Null Space is $$(0,0)$$.

---

#### 2.6.4.  What is the null space of $$T_4$$ in Prob. 2? Of $$T_1$$ and $$T_2$$ in 2.6-7? Of $$T$$ in 2.6-4?

**Answer:**

The null space of $$T_4(x_1,x_2)=(\gamma x_1, \gamma x_2)$$ is $$(0,0)$$.

The null space of $$T_1$$, which is the cross product $$T_1: \mathbb{R}^3 \rightarrow \mathbb{R}^3$$ with a fixed vector $$a=(a_1, a_2, a_3)$$ is $$(ka_1, ka_2, ka_3)$$, with $$k \in \mathbb{R}$$.

The null space of $$T_2=\langle x,a \rangle$$ with $$a=(a_1,a_2,a_3)$$ as a constant vector is the plane defined by $$a_1 x + a_2 y + a_3 z = 0$$

The null space of $$T: x(t) \mapsto x'(t)$$ is the space of functions $$x(t)=c$$, where $$c \in \mathbb{R}$$.

---

#### 2.6.5. Let $$T: X \rightarrow Y$$ be a linear operator. Show that the image of a subspace $$V$$ of $$X$$ is a vector space, and so is the inverse image of a subspace $$W$$ of $$Y$$.

**Proof:**

Let $$x,y \in V$$, and $$x',y' \in \text{Im } V$$. Then we have:

$$
\alpha x' + \beta y' = \alpha Tx + \beta Ty = T(\alpha x + \beta y)
$$

We know that $$\alpha x + \beta y \in V$$, therefore $$T(\alpha x + \beta y) \in \text{Im } V$$. Thus $$\alpha x' + \beta y' \in \text{Im } V$$. Thus $$\text{Im } V$$ is a vector subspace of $$V$$ and is thus a vector space.

$$\blacksquare$$

Let $$x,y \in \text{Im } W$$, and $$x',y' \in W$$. Then we have:

$$
\alpha x + \beta y = \alpha T^{-1} x' + \beta T^{-1} y' = T^{-1}(\alpha x' + \beta y')
$$

We know that $$\alpha x' + \beta y' \in W$$, thus $$T(\alpha x' + \beta y') \in \text{Im } W$$. Thus $$\alpha x + \beta y \in \text{Im } W$$. Thus $$\text{Im } W$$ is a vector subspace of $$W$$ and is thus a vector space.

$$\blacksquare$$

---

#### 2.6.6. If the product (the composite) of two linear operators exists, show that it is linear.

**Proof:**

$$
Tx = (T_1 \circ T_2) x = T_1(T_2 x)
$$

Thus for $$z=\alpha x + \beta y$$, we have:

$$
Tz = (T_1 \circ T_2) z = T_1(T_2 z) = T_1(T_2 (\alpha x + \beta y)) \\
T_1(\alpha T_2 x + \beta T_2 y) = T_1(\alpha T_2 x) + T_1(\beta T_2 y) \\
= \alpha (T_1 (T_2 x)) + \beta (T_1 (T_2 y)) \\
= \alpha (T_1 \circ T_2) x + \beta (T_1 \circ T_2) y
= \alpha T x + \beta T y
$$

$$\blacksquare$$

---

#### 2.6.7. (Commutativity) Let $$X$$ be any vector space and $$S: X \rightarrow X$$ and $$T: X \rightarrow X$$ any operators. $$S$$ and $$T$$ are said to commute if $$ST = TS$$, that is, $$(ST)x = (TS)x$$ for all $$x \in X$$. Do $$T_1$$ and $$T_3$$, in Prob. 2 commute?

**Proof:**

We have the following operators:

$$
T_1(x_1,x_2)=(x_1,0) \\
T_2(x_1,x_2)=(0,x_2)
$$

We have :

$$
(T_1 T_2) (x_1,x_2) = T_1(T_2(x_1,x_2)) = T_1(0,x_2) = (0,0) \\
(T_2 T_1) (x_1,x_2) = T_2(T_1(x_1,x_2)) = T_2(x_1,0) = (0,0)
$$

Thus $$T_1, T_2$$ commute.

$$\blacksquare$$

---

#### 2.6.8. Write the operators in Prob. 2 using $$2 \times 2$$ matrices.

**Answer:**

The operators are:

$$
(\xi_1, \xi_2) \mapsto (\xi_1, 0) \\
(\xi_1, \xi_2) \mapsto (0, \xi_2) \\
(\xi_1, \xi_2) \mapsto (\xi_2, \xi_1) \\
(\xi_1, \xi_2) \mapsto (\gamma\xi_1, \gamma\xi_2)
$$

$$(\xi_1, \xi_2) \mapsto (\xi_1, 0)$$ can be represented as:

$$
\begin{bmatrix}
1 && 0 \\
0 && 0
\end{bmatrix}
$$

$$(\xi_1, \xi_2) \mapsto (0, \xi_2)$$ can be represented as:

$$
\begin{bmatrix}
0 && 0 \\
0 && 1
\end{bmatrix}
$$

$$(\xi_1, \xi_2) \mapsto (\xi_2, \xi_1)$$ can be represented as:

$$
\begin{bmatrix}
0 && 1 \\
1 && 0
\end{bmatrix}
$$

$$(\xi_1, \xi_2) \mapsto (\gamma\xi_1, \gamma\xi_2)$$ can be represented as:

$$
\begin{bmatrix}
\gamma && 0 \\
0 && \gamma
\end{bmatrix}
$$

---

#### 2.6.9. In 2.6-8, write $$y = Ax$$ in terms of components, show that $$T$$ is linear and give examples.

**Proof:**

Let $$A$$ be an arbitrary $$m \times n$$ matrix, and $$x$$ be an $$n \times 1$$ vector. Then we have $$T:x \mapsto Ax$$

Then we have $$y$$ as an $$m \times 1$$ vector, and $$y_k = \sum\limits_{i=1}^n A_{ki} x_i$$.

Let $$z=\alpha x + \beta y$$. Then we have: $$z_i=\alpha x_i + \beta y_i$$.

Then we have, for $$y=A(\alpha x + \beta y)$$:

$$
y_k = \sum\limits_{i=1}^n A_{ki} (\alpha x_i + \beta y_i) \\
= \sum\limits_{i=1}^n (\alpha A_{ki} x_i + \beta A_{ki} y_i) \\
=  \alpha \sum\limits_{i=1}^n A_{ki} x_i + \beta \sum\limits_{i=1}^n A_{ki} y_i
$$

Thus $$y=\alpha Ax + \beta Ay = \alpha Tx + \beta Ty$$ and $$T$$ is linear.

$$\blacksquare$$

---

#### 2.6.10. Formulate the condition in 2.6-10(a) in terms of the null space of $$T$$.

**Answer:**

The condition is:

The inverse $$T^{-1}: \mathcal{R}(T) \rightarrow \mathcal{D}(T)$$ exists if and only if $$Tx=0 \Rightarrow x=0$$.

Reformulated in terms of the null space of $$T$$, it states the inverse $$T^{-1}: \mathcal{R}(T) \rightarrow \mathcal{D}(T)$$ exists if and only if the null space of $$T$$ consists of only the zero vector.

---

#### 2.6.11. Let $$X$$ be the vector space of all complex $$2 \times 2$$ matrices and define $$T: X \rightarrow X$$ by $$Tx = bx$$, where $$b \in X$$ is fixed and $$bx$$ denotes the usual product of matrices. Show that $$T$$ is linear. Under what condition does $$T^{-1}$$ exist?

**Proof:**

We have $$Tx=bx=c$$, so we can write each individual element of the result $$c_{ij}$$ as:

$$
c_{ij}= \sum\limits_{k=1}^2 b_{ik} x_{kj}
$$

We can thus write, for $$z=\alpha x + \beta y$$:

$$
{(Tz)}_{ij} = \sum\limits_{k=1}^2 b_{ik} (\alpha x_{kj} + \beta y_{kj}) \\
= \alpha \sum\limits_{k=1}^2 b_{ik} x_{kj} + \beta \sum\limits_{k=1}^2 b_{ik} y_{kj} \\
= \alpha {(Tx)}_{ij} + \beta {(Ty)}_{ij} \\
\Rightarrow T(\alpha x + \beta y) = \alpha Tx + \beta Ty
$$

$$T^{-1}$$ exists when the determinant $$ad-bc \neq 0$$.

Alternatively, the two column vectors in the $$2 \times 2$$ matrix should be linearly independent.

$$\blacksquare$$

---

#### 2.6.12. Does the inverse of $$T$$ in 2.6-4 (differentiation operator) exist?

**Proof:**

For the inverse to exist, we need the operator to be injective, that is:

$$
x_1 \neq x_2 \Rightarrow Tx_1 \neq Tx_2 \\
\text{or, } Tx_1 = Tx_2 \Rightarrow x_1 = x_2
$$

As a counter-example, take:

$$
x_1(t) = x + 2 \\
x_2(t) = x + 3
$$

Then, applying the differentiation operator on them, we get:

$$
Tx_1(t) = x_1'(t) = 1 \\
Tx_2(t) = x_2'(t) = 1
$$

Thus $$Tx_1=Tx_2$$, but $$x_1 \neq x_2$$. Thus, the inverse of $$T$$ does not exist.

$$\blacksquare$$

---

#### 2.6.13. Let $$T: \mathcal{D}(T) \rightarrow Y$$ be a linear operator whose inverse exists. If $$\{x_1, \cdots, x_n\}$$ is a linearly independent set in $$\mathcal{D}(T)$$, show that the set $$\{Tx_1, \cdots, Tx_n\}$$ is linearly independent.

**Proof:**

$$\{x_1, \cdots, x_n\}$$ is a linearly independent set. Thus, $$\sum\limits_{i=1}^n \alpha_i x_i = 0 \Rightarrow \alpha_i=0$$.

Assume then that $$\sum\limits_{i=1}^n \alpha_i Tx_i = 0$$. Then we have:

$$
\sum\limits_{i=1}^n \alpha_i Tx_i = T(\sum\limits_{i=1}^n \alpha_i x_i) = 0
$$

Because $$T0=0$$, we then have:

$$
T(\sum\limits_{i=1}^n \alpha_i x_i) = 0 = T0
\Rightarrow \sum\limits_{i=1}^n \alpha_i x_i = 0
$$

But this implies that all $$\alpha_i = 0$$.  
Thus $$\sum\limits_{i=1}^n \alpha_i Tx_i = 0 \Rightarrow \alpha_i = 0$$

$$\blacksquare$$

---

#### 2.6.14. Let $$T: X \rightarrow Y$$ be a linear operator and $$\dim X = \dim Y = n < \infty$$. Show that $$\mathcal{R}(T) = Y$$ if and only if $$T^{-1}$$ exists.

**Proof:**


$$\blacksquare$$

---

#### 2.6.15. Consider the vector space $$X$$ of all real-valued functions which are defined on $$\mathbb{R}$$ and have derivatives of all orders everywhere on $$\mathbb{R}$$. Define $$T: X \rightarrow X$$ by $$y(t) = Tx(t) = x'(t)$$. Show that $$\mathcal{R}(T)$$ is all of $$X$$ but $$T^{-1}$$ does not exist. Compare with Prob. 14 and comment.

**Proof:**


$$\blacksquare$$

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



---

#### 2.6.5. Let $$T: X \rightarrow Y$$ be a linear operator. Show that the image of a subspace $$V$$ of $$X$$ is a vector space, and so is the inverse image of a subspace $$W$$ of $$Y$$.

**Proof:**

$$\blacksquare$$

---

#### 2.6.6. If the product (the composite) of two linear operators exists, show that it is linear.

**Proof:**


$$\blacksquare$$

---

#### 2.6.7. (Commutativity) Let $$X$$ be any vector space and $$S: X \rightarrow X$$ and $$T: X \rightarrow X$$ any operators. $$S$$ and $$T$$ are said to commute if $$ST = TS$$, that is, $$(ST)x = (TS)x$$ for all $$x \in X$$. Do $$T_1$$ and $$T_3$$, in Prob. 2 commute?

**Proof:**

$$\blacksquare$$

---

#### 2.6.8. Write the operators in Prob. 2 using $$2 \times 2$$ matrices.

**Answer:**



---

#### 2.6.9. In 2.6-8, write $$y = Ax$$ in terms of components, show that $$T$$ is linear and give examples.

**Proof:**


$$\blacksquare$$

---

#### 2.6.10. Formulate the condition in 2.6-1O(a) in terms of the null space of $$T$$.

**Answer:**



---

#### 2.6.11. Let $$X$$ be the vector space of all complex $$2 \times 2$$ matrices and define $$T: X \rightarrow X$$ by $$Tx = bx$$, where $$b \in X$$ is fixed and $$bx$$ denotes the usual product of matrices. Show that $$T$$ is linear. Under what condition does $$T^{-1}$$ exist?

**Proof:**


$$\blacksquare$$

---

#### 2.6.12. Does the inverse of $$T$$ in 2.6-4 exist?

**Proof:**


$$\blacksquare$$

---

#### 2.6.13. Let $$T: \mathcal{D}(T) \rightarrow Y$$ be a linear operator whose inverse exists. If $$\{x_1, \cdots, x_n\}$$ is a linearly independent set in $$\mathcal{R}(T)$$, show that the set $$\{Tx_1, \cdots, Tx_n\}$$ is linearly independent.

**Proof:**


$$\blacksquare$$

---

#### 2.6.14. Let $$T: X \rightarrow Y$$ be a linear operator and $$\dim X = \dim Y = n < \infty$$. Show that $$\mathcal{R}(T) = Y$$ if and only if $$T^{-1}$$ exists.

**Proof:**


$$\blacksquare$$

---

#### 2.6.15. Consider the vector space $$X$$ of all real-valued functions which are defined on $$\mathbb{R}$$ and have derivatives of all orders everywhere on $$\mathbb{R}$$. Define $$T: X \rightarrow X$$ by $$y(t) = Tx(t) = x'(t)$$. Show that $$\mathcal{R}(T)$$ is all of $$X$$ but $$T^{-1}$$ does not exist. Compare with Prob. 14 and comment.

**Proof:**


$$\blacksquare$$

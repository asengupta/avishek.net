---
title: "Intuitions about the Orthogonality of Matrix Subspaces"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Linear Algebra", "Theory"]
---

This is the easiest way I've been able to explain to myself around the orthogonality of matrix spaces. The argument will essentially be based on the geometry of planes which extends naturally to hyperplanes.

Some quick definitions first:

- **Column Space of $$A$$**: The space spanned by a set of linearly independent column vectors.
- **Null Space of $$A$$**: The space spanned by the set of vectors which satisfy the condition $$Ax=0$$. For a non-empty null space, this implies the following equivalent statements:
    - There exists some non-zero combination of the column vectors of $$A$$ which results in the zero vector.
    - There exists at least one vector which gets transformed by matrix $$A$$ into the zero vector.

- **Row Space of $$A$$**: The space spanned by a set of linearly independent row vectors.
- **Left Null Space of $$A$$**: The space spanned by the set of vectors which satisfy the condition $$A^Tx=0$$. For a non-empty left null space, this implies the following equivalent statements:
    - There exists some non-zero combination of the row vectors of $$A$$ (i.e., column vectors of $$A^T$$) which results in the zero vector.
    - There exists at least one vector which gets transformed by matrix $$A^T$$ into the zero vector.

The important point is that any argument we make around the column space and null space of $$A$$ applies exactly to the row space and left null space of $$A^T$$, and vice versa.

For purposes of this discussion, I'll pick a matrix which already has linearly independent column and row vectors.

$$
A=
\begin{bmatrix}
a_{11} && a_{12} && ... && a_{1N} \\
a_{21} && a_{22} && ... && a_{2N} \\
a_{31} && a_{32} && ... && a_{3N} \\
\vdots && \vdots && \vdots && \vdots \\
a_{M1} && a_{M2} && ... && a_{MN} \\
\end{bmatrix}
$$

Let's consider the non-zero null space of $$A$$ and pick a vector from that space. Let that vector be $$x_O=(x_{O1}, x_{O2}, x_{O3}, ..., x_{ON})$$.

$$
A=
\begin{bmatrix}
a_{11} && a_{12} && ... && a_{1N} \\
a_{21} && a_{22} && ... && a_{2N} \\
a_{31} && a_{32} && ... && a_{3N} \\
\vdots && \vdots && \vdots && \vdots \\
a_{M1} && a_{M2} && ... && a_{MN} \\
\end{bmatrix}
\begin{bmatrix}
x_{O1} \\
x_{O1} \\
x_{O1} \\
\vdots \\
x_{O1} \\
\end{bmatrix}
=
\begin{bmatrix}
a_{11}x_{O1} + a_{12}x_{O2} + a_{13}x_{O3} + ... + a_{1N}x_{ON} \\
a_{21}x_{O1} + a_{22}x_{O2} + a_{23}x_{O3} + ... + a_{2N}x_{ON} \\
a_{31}x_{O1} + a_{32}x_{O2} + a_{33}x_{O3} + ... + a_{3N}x_{ON} \\
\vdots \\
a_{M1}x_{O1} + a_{M2}x_{O2} + a_{M3}x_{O3} + ... + a_{MN}x_{ON} \\
\end{bmatrix}
= 0
$$

Let's take the equation of the first row:

$$a_{11}x_{O1} + a_{12}x_{O2} + a_{13}x_{O3} + ... + a_{1N}x_{ON}=0$$

This represents a hyperplane:
$$\mathbf{a_{11}x + a_{12}x + a_{13}x + ... + a_{1N}x=0}$$ with the normal vector $$\mathbf{\hat{n}=(a_{11}, a_{12}, a_{13}, ..., a_{1N})}$$. **Note that $$\hat{n}$$ is also one of the row vectors which spans A's row space.**

By the basic definition of hyperplanes and normal vectors (for a quick refresher, see [Vectors, Normals, and Hyperplanes]({% post_url 2021-03-29-vectors-normals-hyperplanes %})), we can say that:

- The vector $$x_O$$ is orthogonal to the normal vector $$\hat{n}$$, i.e., $$\mathbf{x_O\perp \hat{n}}$$. Equivalently, $$\mathbf{x_O\cdot \hat{n}=0}$$ (the dot product is zero). This condition is true for every vector $$x_O$$ in $$A$$'s null space. 
- Thus A's null space is orthogonal to the first row vector of $$A$$.

This argument can be extended to all row vectors in $$A$$, proving that $$A$$'s null space is orthogonal to every row vector in $$A$$. By the property of linearity, this implies that $$A$$'s null space is orthogonal to $$A$$'s row space, i.e., $$\mathbf{N(A)\perp R(A)}$$.

Now, apply the same argument for $$A^T$$, i.e., the null space of $$A^T$$ is orthogonal to $$A^T$$'s row space, i.e., $$\mathbf{N(A^T)\perp R(A^T)}$$. But, we already know that:

$$\mathbf{R(A^T)=C(A)}$$: The row space of $$A^T$$ is the column space of $$A$$.
$$\mathbf{N(A^T)=LN(A)}$$: The null space of $$A^T$$ is the left null space of $$A$$.

Thus, the left null space of $$A$$ is orthogonal to the column space of $$A$$.

To summarise:
- **The left null space of $$A$$ is orthogonal to the column space of $$A$$.**
- **The null space of $$A$$ is orthogonal to the row space of $$A$$.**


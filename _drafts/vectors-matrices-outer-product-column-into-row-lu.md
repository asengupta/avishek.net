---
title: "Matrix Outer Product: Columns-into-Rows and the LU Factorisation"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Linear Algebra", "Theory"]
---
We will discuss the Column-into-Rows computation technique for matrix outer products. This will lead us to one of the important factorisations (the LU Decomposition) that are used computationally when solving systems of equations, or computing matrix inverses.

## The Building Block
Let's start with the building block which we'll extend to the computation of the matrix outer product. That component is the multiplication of a column vector ($$M\times 1$$, left side) with a row vector ($$1\times N$$, right side).
Without doing any computation, we can immediately say that the resulting matrix is $$M\times N$$. Taking a concrete example here:

$$
\begin{bmatrix}
a_{11} \\
a_{21}
\end{bmatrix}
\begin{bmatrix}
b_{11} && b_{12} \\
\end{bmatrix}=
\begin{bmatrix}
a_{11}b_{11} && a_{11}b_{12} \\
a_{21}b_{11} && a_{11}b_{22} \\
\end{bmatrix}
$$

Note that this specific calculation itself can be done using the column picture/row picture/value-wise computation.

## Extension to all matrices
We will extend this computation to the outer product of two general matrices, A ($$M\times N$$) and B ($$N\times P$$). Here, $$a_1$$, $$a_2$$, ..., $$a_N$$ are the $$N$$ columns of $$A$$, and $$b_1$$, $$b_2$$, ..., $$b_N$$ are the $$N$$ rows of $$B$$.

$$
\begin{bmatrix}
| && | && | && |\\
a_1 && a_2 && ... && a_N\\
| && | && | && |\\
\end{bmatrix}
\begin{bmatrix}
--- && b_1 && --- \\
--- && b_2 && --- \\
--- && b_N && --- \\
\end{bmatrix}\\
=\mathbf{a_1b_1+a_2b_2+...+a_Nb_N}
$$

Each product in the result $$a_1b_1$$, $$a_2b_2$$, etc. is an $$M\times P$$ matrix, and the sum of them is obviously $$M\times P$$ as well.
Alright, so this is a way of computing the outer product of matrices. That's great, but let's look at why this is useful. That application is the LU Decomposition technique, and the high-level intuition behind it is sketched below.

- A single matrix will be expressed as the sum of the product of two matrices $${\ell}_1 u_1$$, $${\ell}_2 u_2$$, ..., $${\ell}_N u_N$$.
- We will go in the reverse direction and express this sum as a product of two matrices, $$\mathbf{L}$$ and $$\mathbf{U}$$.
- Thus, we will have expressed $$A$$ as $$\mathbf{A=LU}$$, which is essentially a factorisation of A.
- We will also consider the special nature of $$\mathbf{L}$$ and $$\mathbf{U}$$ as part of this discussion.

## LU Factorisation: Procedure

The matrix $$L$$ is a **lower diagonal matrix**, meaning that the all the elements below the diagonal are zero. Here is an example of a lower diagonal matrix:

$$
\begin{bmatrix}
1 && 4 && 10 && 2\\
\mathbf{0} && 5 && 30 && 13\\
\mathbf{0} && \mathbf{0} && 3 && 34\\
\mathbf{0} && \mathbf{0} && \mathbf{0} && 44\\
\end{bmatrix}\\
$$

As you will see, the other factor in the $$\mathbf{LU}$$ combination, $$U$$ will be an **upper diagonal matrix**, i.e., all its elements above its diagonal are zero.

We start with a basic 
## Conclusion
This rule of swapping the order of the outer product will also apply to when we are calculating inverses.

---
title: "Matrix Outer Product: Columns-into-Rows and the LU Factorisation"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Linear Algebra", "Theory"]
---
We will discuss the Column-into-Rows computation technique for matrix outer products. This will lead us to one of the important factorisations (the LU Decomposition) that is used computationally when solving systems of equations, or computing matrix inverses.

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

- A single matrix will be expressed as the sum of the product of two matrices $${\ell}_1 u_1$$, $${\ell}_2 u_2$$, ..., $${\ell}_N u_N$$. It is important to note that we are dealing with an $$N\times N$$ matrix, i.e., a square matrix.
- We will go in the reverse direction and express this sum as a product of two matrices, $$\mathbf{L}$$ and $$\mathbf{U}$$.
- Thus, we will have expressed $$A$$ as $$\mathbf{A=LU}$$, which is essentially a factorisation of A.
- We will also consider the special nature of $$\mathbf{L}$$ and $$\mathbf{U}$$ as part of this discussion.

## LU Factorisation: Procedure

The final matrix $$L$$ is a **lower diagonal matrix**, meaning that the all the elements above the diagonal are zero. Here is an example of a lower diagonal matrix:

$$
\begin{bmatrix}
1 && \mathbf{0} && \mathbf{0} && \mathbf{0}\\
5 && 6 && \mathbf{0} && \mathbf{0}\\
7 && 8 && 3 && \mathbf{0}\\
8 && 12 && 8 && 44\\
\end{bmatrix}\\
$$

As you will see, the other factor in the $$\mathbf{LU}$$ combination, $$U$$ will be an **upper diagonal matrix**, i.e., all its elements above its diagonal are zero.

The decomposition technique is the same as high school students follow when solving systems of equations. We will do the same, but in a more structured manner.

**Note on Terminology**: This method for calculating the **LU** is formally known as the **Gaussian Elimination** method.

The steps are as follows:
1. Subtract the first row from all the rows to force all the elements in the first column to become zero. Choose an appropriate multiplier for each row which allows you to do this.
2. This amounts to subtracting a matrix whose:
    - First row is 1 times the first row of A
    - Second row is x times the first row of A
    - ...and so on
3. Express the subtracting matrix as a product of two vectors.
4. Repeat this until A has all zeroes.
5. Aggregate the sum of matrix products (that represents the original matrix) as a single product of two matrices, $$L$$ and $$U$$.

Let's look at the general case.

1. Force the first column of the matrix to become all zeroes, by subtracting suitable scaled versions of the first row from every row. Express the thing you've subtracted as product of a column vector and a row vector, like we discussed above, since that is our building block.

$$
\begin{bmatrix}
a_{11} && a_{12} && ... && a_{1N}\\
a_{21} && a_{22} && ... && a_{2N}\\
a_{31} && a_{32} && ... && a_{3N}\\
...    && ...    && ..  && ...\\
a_{N1} && a_{N2} && ... && a_{NN}\\
\end{bmatrix}\\
=
\begin{bmatrix}
1\\
\ell_{12}\\
.\\
.\\
.\\
\ell_{1N}\\
\end{bmatrix}
\begin{bmatrix}
a_{11} && a_{12} && ... && a_{1N}\\
\end{bmatrix}
+
\begin{bmatrix}
\mathbf{0} && \mathbf{0} && ... && \mathbf{0}\\
\mathbf{0} && a'_{22} && ... && a'_{2N}\\
\mathbf{0} && a'_{32} && ... && a'_{3N}\\
...    && ...    && ..  && ...\\
\mathbf{0} && a'_{N2} && ... && a'_{NN}\\
\end{bmatrix}\\
$$

2. Call the first term on the RHS, as $$\ell_1u_1$$, that is:

$$
\ell_1u_1+
\begin{bmatrix}
\mathbf{0} && \mathbf{0} && ... && \mathbf{0}\\
\mathbf{0} && a'_{22} && ... && a'_{2N}\\
\mathbf{0} && a'_{32} && ... && a'_{3N}\\
...    && ...    && ..  && ...\\
\mathbf{0} && a'_{N2} && ... && a'_{NN}\\
\end{bmatrix}\\
$$

3. Now subtract the second row from all the rows below it (with an appropriate multiplier) to make all the numbers in the second column, zero, that is:

$$
\begin{bmatrix}
\mathbf{0} && \mathbf{0} && ... && \mathbf{0}\\
\mathbf{0} && a'_{22} && ... && a'_{2N}\\
\mathbf{0} && a'_{32} && ... && a'_{3N}\\
...    && ...    && ..  && ...\\
\mathbf{0} && a'_{N2} && ... && a'_{NN}\\
\end{bmatrix}\\
=
\ell_1u_1+
\begin{bmatrix}
0\\
1\\
\ell_{23}\\
.\\
.\\
\ell_{2N}\\
\end{bmatrix}
\begin{bmatrix}
0 && a'_{22} && a'_{23} && ... && a_{1N}\\
\end{bmatrix}
+
\begin{bmatrix}
\mathbf{0} && \mathbf{0} && ... && \mathbf{0}\\
\mathbf{0} && \mathbf{0} && ... && \mathbf{0}\\
\mathbf{0} && \mathbf{0} && ... && a''_{3N}\\
...    && ...    && ..  && ...\\
\mathbf{0} && \mathbf{0} && ... && a''_{NN}\\
\end{bmatrix}\\
$$

4. Call the first term on the LHS as $$\ell_2u_2$$, so that:

$$
\begin{bmatrix}
\mathbf{0} && \mathbf{0} && ... && \mathbf{0}\\
\mathbf{0} && a'_{22} && ... && a'_{2N}\\
\mathbf{0} && a'_{32} && ... && a'_{3N}\\
...    && ...    && ..  && ...\\
\mathbf{0} && a'_{N2} && ... && a'_{NN}\\
\end{bmatrix}\\
=
\ell_1u_1+
\ell_2u_2+
\begin{bmatrix}
\mathbf{0} && \mathbf{0} && ... && \mathbf{0}\\
\mathbf{0} && \mathbf{0} && ... && \mathbf{0}\\
\mathbf{0} && \mathbf{0} && ... && a''_{3N}\\
...    && ...    && ..  && ...\\
\mathbf{0} && \mathbf{0} && ... && a''_{NN}\\
\end{bmatrix}\\
$$

I hope you can see the pattern, we are gradually aiming to reduce all the elements of A to zero, while extracting all the $$\ell u$$ factors as a sum.
Doing this will ultimately give us:

$$
\begin{bmatrix}
\mathbf{0} && \mathbf{0} && ... && \mathbf{0}\\
\mathbf{0} && a'_{22} && ... && a'_{2N}\\
\mathbf{0} && a'_{32} && ... && a'_{3N}\\
...    && ...    && ..  && ...\\
\mathbf{0} && a'_{N2} && ... && a'_{NN}\\
\end{bmatrix}\\
=
\mathbf{
\ell_1u_1+
\ell_2u_2+...
\ell_Nu_N}
$$

where all $$\ell$$'s are column vectors and all $$u$$'s are row vectors.

If you remember the general pattern of outer product using the columns-into-rows approach, you can rewrite this entire sum as a product of two vectors. **That is, for $$\ell_1 u_1$$, $$\ell_1$$ becomes the first column of $$L$$ and $$u_1$$ becomes the first row of $$U$$, and so on.**

$$
\begin{bmatrix}
a_{11} && a_{12} && ... && a_{1N}\\
a_{21} && a_{22} && ... && a_{2N}\\
a_{31} && a_{32} && ... && a_{3N}\\
...    && ...    && ..  && ...\\
a_{N1} && a_{N2} && ... && a_{NN}\\
\end{bmatrix}\\
=
\begin{bmatrix}
1 && \mathbf{0} && \mathbf{0} && ... && \mathbf{0}\\
\ell_{12} && 1 && \mathbf{0} && ... && \mathbf{0}\\
\ell_{13} && \ell_{23} && 1 && ... && \mathbf{0}\\
\ell_{14} && \ell_{24} && \ell_{34} && ... && \mathbf{0}\\
...    && ...    && ..  && ... && ...\\
\ell_{1N} && \ell_{2N} && \ell_{3N} && .. && 1\\
\end{bmatrix}
\begin{bmatrix}
x_{11} && x_{12} && x_{13} && ... && x_{1N}\\
\mathbf{0} && x_{22} && x_{23} && ... && x_{2N}\\
\mathbf{0} && \mathbf{0} && x_{33} && ... && x_{3N}\\
\mathbf{0} && \mathbf{0} && \mathbf{0} && ... && x_{3N}\\
...    && ...    && ..  && ... && ...\\
\mathbf{0} && \mathbf{0} && \mathbf{0} && .. && x_{NN}\\
\end{bmatrix}\\
$$

which is the form that we wanted, namely:

$$A=LU$$

## Implications for Machine Learning
The LU factorisation will mostly be seen in lower level matrix computational techniques. Below are some examples.

- For a system of equations given by $$Ax=b$$, the LU decomposition technique can be used to solve systems of linear equations repeatedly for different values of $$b$$ without doing the entire process of Gaussian Elimination every time for a different value of $$b$$.
- Matlab uses LU decomposition to calculate inverse matrices.
- The LU decomposition technique can make calculating determinants easier. We will speak of determinants at a later point.


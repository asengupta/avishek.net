---
title: "The Cholesky and LDL' Factorisations"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Theory", "Linear Algebra"]
---
This article discusses a set of two useful (and closely related) matrix factorisations: the Cholesky and the $$LDL^T$$ factorisations. Both of them find various uses: the Cholesky factorisation particularly is used when solving large systems of linear equations.

The Cholesky factorisation decomposes a positive definite matrix into the following form:

$$
A=LL^T
$$

where $$L$$ is a lower triangular matrix.
The $$LDL^T$$ factorisation as its name suggests, decomposes a positive definite matrix into the following form:

$$
A=LDL^T
$$

where $$D$$ is a diagonal matrix, and $$L$$ is a lower triangular matrix which has 1 in all its diagonal elements.

## Cholesky Factorisation

We will derive expressions for the Cholesky method by working backwards from the desired form of the factors. We will look at the $$3\times 3$$ case to reinforce the pattern.

$$
A= \begin{bmatrix}
a_{11} && a_{12} && a_{13} \\
a_{21} && a_{22} && a_{23} \\
a_{31} && a_{32} && a_{33} \\
\end{bmatrix}
\\
L=
\begin{bmatrix}
L_{11} && 0 && 0 \\
L_{21} && L_{22} && 0\\
L_{31} && L_{32} && a_{33}\\
\end{bmatrix}
$$

Since we want $$A=LL^T$$, we can write out $$LL^T$$ as:

$$
LL^T=
\begin{bmatrix}
L_{11} && 0 && 0 \\
L_{21} && L_{22} && 0\\
L_{31} && L_{32} && L_{33}\\
\end{bmatrix}
\cdot
\begin{bmatrix} 
L_{11} && L_{21} && L_{31} \\
0 && L_{22} && L_{32}\\
0 && 0 && L_{33}\\
\end{bmatrix}
=
\begin{bmatrix}
{L_{11}}^2   && L_{11}L_{21}   && L_{11}L_{31} \\
L_{21}L_{11} && {L_{21}}^2 + {L_{22}}^2 && L_{21}L_{31} + L_{22}L_{32}\\
L_{31}L_{11} && L_{31}L_{21} + L_{32}L_{22} && {L_{31}}^2 + {L_{32}}^2 + {L_{33}}^2\\
\end{bmatrix}
$$

The product of a matrix and its transpose is always symmetric, so we can ignore the upper right triangular portion of the above result, when computing the elements. Thus, we have the following equality:

$$
A= \begin{bmatrix}
A_{11} && A_{12} && A_{13} \\
A_{21} && A_{22} && A_{23} \\
A_{31} && A_{32} && A_{33} \\
\end{bmatrix}
=
\begin{bmatrix}
{L_{11}}^2   && -   && - \\
L_{21}L_{11} && {L_{21}}^2 + {L_{22}}^2 && -\\
L_{31}L_{11} && L_{31}L_{21} + L_{32}L_{22} && {L_{31}}^2 + {L_{32}}^2 + {L_{33}}^2\\
\end{bmatrix}
$$

The element $$L_{11}$$ is the easiest to compute; equating the terms gives us:

$$
L_{11}=\sqrt{a_{11}}
$$

Now let us consider the diagonal elements; the pattern suggests the follow the form:

$$
A_{ii}=\sum_{k=1}^i{L_{ik}}^2 \\
={L_{ii}}^2 + \sum_{k=1}^{i-1}{L_{ik}}^2 \\
\Rightarrow L_{ii}=\sqrt{A_{ii} - \sum_{k=1}^{i-1}{L_{ik}}^2} \\
$$

Now let us consider the non-diagonal elements in the lower triangular section of the matrix. The pattern suggests the following form. You can convince yourself by computing the results for $$a_{21}$$, $$a_{31}$$, and $$a_{32}$$.

$$
A_{ij}=\sum_{k=1}^j L_{ik}L_{jk} \\
= L_{ij}L_{jj} + \sum_{k=1}^{j-1} L_{ik}L_{jk} \\
\Rightarrow L_{ij}=\frac{1}{L_{jj}}\cdot \left( A_{ij} - \sum_{k=1}^{j-1} L_{ik}L_{jk} \right)
$$

We consider how the two equations above can help us compute this. For this illustration, we pick the Cholesky-Crout algorithm, which proceeds to find the elements of $$L$$, column by column. The same concept works if you compute row by row (Cholesky–Banachiewicz algorithm).

## Example Computation (Cholesky-Crout)
Note, at each stage, we've bolded the terms which are already known from one of the previous steps. Otherwise, we would not be able to compute the result of the current step. All the terms on the right hand side in each step should be known quantities.

### First Column
The first column is easy to compute.

$$
L_{11}=\sqrt{a_{11}} \\
L_{21}=\frac{\mathbf{A_{21}}}{\mathbf{L_{11}}} \\
L_{31}=\frac{\mathbf{A_{31}}}{\mathbf{L_{11}}} \\
...
$$

### Second Column
The first element in the lower triangular section of the second column is the diagonal element (remember we're only considering the lower triangular section, since the upper triangular section will be the mirror image, because $$LL^T is symmetric$$). For this, we can apply the diagonal element formula, like so:

$$
L_{22}=\sqrt{\mathbf{A_{22}} - {\mathbf{L_{21}}}^2} \\
L_{32}=\frac{1}{\mathbf{L_{22}}}\cdot\sqrt{\mathbf{A_{32}} - \mathbf{L_{31}}\mathbf{L_{21}}} \\
$$

### Third Column
The first element in the lower triangular section of the third column is the last diagonal element $$L_{33}$$. We can again apply the diagonal element formula, like so:

$$
L_{33}=\sqrt{\mathbf{A_{33}} - ({\mathbf{L_{31}}}^2 + {\mathbf{L_{32}}}^2)} \\
$$

## $$LDL^T$$ Factorisation

Let's look at the $$LDL^T$$ factorisation. This is very similar to the Cholesky factorisation, except for the fact that it avoids the need to compute square roots for every term computation. The form that the $$LDL^T$$ factorisation takes is:

$$
A=LDL^T
$$

where $$L$$ is lower triangular with all its diagonal elements set to 1, and $$D$$ is a diagonal matrix.

Let us take the $$3\times 3$$ matrix as an example again, and we will follow the same approach as we did with the Cholesky factorisation.

$$
\begin{bmatrix}
A_{11} && A_{12} && A_{13} \\
A_{21} && A_{22} && A_{23} \\
A_{31} && A_{32} && A_{33} \\
\end{bmatrix}
=
\begin{bmatrix}
1 && 0 && 0 \\
L_{21} && 1 && 0\\
L_{31} && L_{32} && 1\\
\end{bmatrix}
\cdot
\begin{bmatrix}
D_{11} && 0 && 0 \\
0 && D_{22} && 0\\
0 && 0 && D_{33}\\
\end{bmatrix}
\cdot
\begin{bmatrix}
1 && L_{21} && L_{31} \\
0 && 1 && L_{32}\\
0 && 0 && 1\\
\end{bmatrix}
$$

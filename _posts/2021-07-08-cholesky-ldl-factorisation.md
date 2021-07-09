---
title: "The Cholesky and LDL* Factorisations"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Theory", "Linear Algebra"]
---
This article discusses a set of two useful (and closely related) factorisations for **positive-definite matrices**: the **Cholesky** and the **$$LDL^T$$** factorisations. Both of them find various uses: the Cholesky factorisation particularly is used when **solving large systems of linear equations**.

**NOTE**: By definition, **a positive-definite matrix is symmetric**.

## Factorisation Forms

- The **Cholesky factorisation** decomposes a positive definite matrix into the following form:  
  
  $$ \mathbf{A=LL^T}$$  
  
  where $$A$$ is **positive-definite**, and $$L$$ is a **lower triangular matrix**.
  

- The **$$LDL^T$$ factorisation** as its name suggests, decomposes a **positive definite matrix** into the following form:  
  
  $$ \mathbf{A=LDL^T} $$  
  
  where $$A$$ is **positive-definite**, $$D$$ is a **diagonal matrix**, and $$L$$ is a **lower triangular matrix** which has **1 in all its diagonal elements**.

## Cholesky Factorisation

We will derive expressions for the **Cholesky** method by working backwards from the desired form of the factors. We will look at the $$3\times 3$$ case to reinforce the pattern.

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

**The product of a matrix and its transpose is always symmetric**, so we can ignore the upper right triangular portion of the above result, when computing the elements. Thus, we have the following equality:

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
\Rightarrow \mathbf{L_{ii}=\sqrt{A_{ii} - \sum_{k=1}^{i-1}{L_{ik}}^2}} \\
$$

Now let us consider the non-diagonal elements in the lower triangular section of the matrix. The pattern suggests the following form. You can convince yourself by computing the results for $$a_{21}$$, $$a_{31}$$, and $$a_{32}$$.

$$
A_{ij}=\sum_{k=1}^j L_{ik}L_{jk} \\
= L_{ij}L_{jj} + \sum_{k=1}^{j-1} L_{ik}L_{jk} \\
\Rightarrow \mathbf{L_{ij}=\frac{1}{L_{jj}}\cdot \left( A_{ij} - \sum_{k=1}^{j-1} L_{ik}L_{jk} \right)}
$$

We consider how the two equations above can help us compute this. For this illustration, we pick the **Cholesky-Crout** algorithm, which proceeds to find the elements of $$L$$, **column by column**. The same concept works if you compute **row by row** (**Cholesky–Banachiewicz algorithm**).

## Example Computation (Cholesky-Crout)
Note, at each stage, **we've bolded the terms which are already known from one of the previous steps**. Otherwise, we would not be able to compute the result of the current step. All the terms on the right hand side in each step should be known quantities.

### First Column
The first column is easy to compute.

$$
L_{11}=\sqrt{\mathbf{a_{11}}} \\
L_{21}=\frac{\mathbf{A_{21}}}{\mathbf{L_{11}}} \\
L_{31}=\frac{\mathbf{A_{31}}}{\mathbf{L_{11}}} \\
...
$$

### Second Column
The first element in the lower triangular section of the second column is the diagonal element (remember we're only considering the lower triangular section, since the upper triangular section will be the mirror image, because $$LL^T$$ is symmetric). For this, we can apply the **diagonal element formula**, like so:

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

Let's look at the **$$LDL^T$$** factorisation. This is very similar to the **Cholesky** factorisation, except for the fact that **it avoids the need to compute square roots for every term computation**. The form that the $$LDL^T$$ factorisation takes is:

$$
A=LDL^T
$$

where $$A$$ is a **positive-definite matrix**, $$L$$ is lower triangular with all its diagonal elements set to 1, and $$D$$ is a diagonal matrix.

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

Multiplying out the right hand side gives us the following:

$$
\begin{bmatrix}
D_{11} && 0 && 0 \\
L_{21}D_{11} && D_{22} && 0\\
L_{31}D_{11} && L_{32}D_{22} && D_{33}\\
\end{bmatrix}
\cdot
\begin{bmatrix}
1 && L_{21} && L_{31} \\
0 && 1 && L_{32}\\
0 && 0 && 1\\
\end{bmatrix} \\
=
\begin{bmatrix}
D_{11} && L_{21}D_{11} && L_{31}D_{11} \\
L_{21}D_{11} && {L_{21}}^2D_{11} + D_{22} && L_{31}L_{21}D_{11} + L_{32}D_{22}\\
L_{31}D_{11} && L_{31}L_{21}D_{11} + L_{32}D_{22} && {L_{31}}^2D_{11} + {L_{32}}^2D_{22} + D_{33} \\
\end{bmatrix}

$$

This suggests the following pattern for the diagonal elements.

$$
A_{ii}=D_{ii} + \sum_{k=1}^{i-1}{L_{ik}}^2D_{kk} \\
\Rightarrow \mathbf{D_{ii} = A_{ii} - \sum_{k=1}^{i-1}{L_{ik}}^2D_{kk}}
$$

For the off-diagonal elements, the following pattern is suggested.

$$
A_{ij}=L_{ij}D_{jj} + \sum_{k=1}^{j-1}L_{ik}L_{jk}D_{kk} \\
\Rightarrow \mathbf{L_{ij} = \frac{1}{D_{jj}}\cdot \left( A_{ij} - \sum_{k=1}^{j-1}L_{ik}L_{jk}D_{kk}\right)}
$$

The example computation for the $$LDL^T$$ is not shown here, but it proceeds in exactly the same way as the Cholesky example computation above.

**The important thing to note is that these equations work because every element we are computing, depends only on other elements in the matrix which are above and to the left of that particular element.**

Since we begin from the top left and proceed column-wise, we know all the factors needed to compute any element. Further, the **symmetry of the matrix** allows us to do the same thing **row-wise** instead.

## Implications: Forward and Backward Substitution

Assume we have a system of linear equations, contrived to be arranged like so:

$$
x_1=c_1 \\
2x_1+3x_2=c_2 \\
4x_1-5x_2+6x_3=c_3 \\
2x_1+7x_2-8x_3+3x_4=c_3 \\
$$

How would you find $$x_1$$, $$x_2$$, $$x_3$$, $$x_4$$?
In this case, it is very easy, because **you can always start at the top**, knowing what $$x_1$$ is, substitute it into the second equation, get $$x_2$$, plug $$x_1$$ and $$x_2$$ into the third equation, and so on. No tiresome Gaussian Elimination is required, because the equations are set up to allow for the solutions to be arrived at very quickly. This is called solution by **Forward Substitution**.

In the same vein, consider the following system of equations:

$$
2x_1+7x_2-8x_3+3x_4=c_3 \\
\hspace{1.2cm}4x_2-5x_3+6x_4=c_3 \\
\hspace{2.4cm}2x_3+3x_4=c_2 \\
\hspace{3.7cm}x_4=c_1 \\
$$

How would you solve the above system? Very easy, in the same way as **Forward Substitution**, except in this case, you'd be working backwards from the bottom. This is **Backward Substitution**, and if you have a system of equations arranged in either of the above configurations, the solution is usually very direct.

If these equations were converted into matrix form, you see immediately that the **forward substitution form is an lower triangular matrix**, like so:

$$
\begin{bmatrix}
1 && 0 && 0 && 0\\
2 && 3 && 0 && 0\\
4 && 5 && 6 && 0\\
2 && 7 &&8 && 3\\
\end{bmatrix}
\cdot X =
\begin{bmatrix}
c_1 \\
c_2 \\
c_3 \\
c_4 \\
\end{bmatrix} \\
$$

Similarly, the **backward substitution** form is a **upper triangular matrix**, like so:

$$
\begin{bmatrix}
2 && 7 && 8 && 3\\
0 && 4 && 5 && 6\\
0 && 0 && 2 && 3\\
0 && 0 && 0 && 1\\
\end{bmatrix}
\cdot X =
\begin{bmatrix}
c_4 \\
c_3 \\
c_2 \\
c_1 \\
\end{bmatrix}
$$

This is why the **Cholesky** and **$$LDL^T$$** factorisations are so useful; once the original system is recast into one of these forms, solution of the system of linear equations proceeds very directly.

## Applications
### 1. Solutions to Linear Equations
**Cholesky factorisation** is used in solving large systems of linear equations, because we can exploit the **lower triangular** nature of $$L$$ and the **upper triangular** nature of $$L^T$$.

$$
AX=B
\Rightarrow LL^TX=B
$$

If we now set $$Y=L^TX$$, then we can write:
$$
LY=B
$$

This can be solved very simply using **forward substitution**, because $$L$$ is lower triangular. Once we have computed $$Y$$, we solve the following system which we used for substitution, i.e.:

$$
L^TX=Y
$$

This is also a very easy computation, since $$L^T$$ is upper triangular, and thus $$X$$ can be solved using **backward substitution**.

It is also important to note that one of the aims of these factorisation algorithms is to also ensure that the **decomposed factors are as sparse as possible**. To this end, there is usually a step before the actual factorisation where the initial matrix is reconfigured (swapping columns and/or rows) based on certain metrics to ensure that the factors end up being as sparse as possible. Such algorithms are called **Minimum Degree Algorithms**.

### 2. Interior Point Method Algorithms in Linear Programming
In **Linear Programming** solvers (like **GLPK**), **Cholesky factorisation** is used as part of the **Interior Point Method** solver in each step. Again, the primary use is to **solve a system of linear equations**, but this is an example of where it fits in a larger real-world context.

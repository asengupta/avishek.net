---
title: "Kernel Functions: Radial Basis Function kernel and the Representer Theorem"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Kernels", "Theory", "Functional Analysis", "Linear Algebra"]
draft: false
---
This article uses the previous mathematical groundwork to delve into a non-trivial Reproducing Kernel Hilbert Space (RKHS, in short), as well as discuss why the particular kernel form makes potentially intractable infinite-dimensional Machine Learning problems tractable. We do this by discussing the **Representer Theorem**. We will also introduce some simple examples of Reproducing Kernel Hilbert Spaces, including a simplification of the frequently-used Radial Basis Function kernel.

The specific posts discussing the background are:

- [Kernel Functions: Kernel Functions with Mercer's Theorem]({% post_url 2021-07-21-kernel-functions-mercers-theorem %})
- [Kernel Functions: Kernel Functions with Reproducing Kernel Hilbert Spaces]({% post_url 2021-07-20-kernel-functions-rkhs %})
- [Kernel Functions: Functional Analysis and Linear Algebra Preliminaries]({% post_url 2021-07-17-kernel-functions-functional-analysis-preliminaries %})
- [Functional Analysis: Norms, Linear Functionals, and Operators]({% post_url 2021-07-19-functional-analysis-results-for-operators %})
- [Functional and Real Analysis Notes]({% post_url 2021-07-18-notes-on-convergence-continuity %})

We will discuss the following:

- Polynomial Kernels
- A simplified version of the popular Radial Basis Function kernel
- Empirical Risk Minimisation
- The Representer Theorem

## Polynomial Kernels

Polynomial kernels are probably the simplest to understand; they are not necessarily used in practice (except for Natural Language Processing, source: Wikipedia), but are a good stepping-stone to understanding the infinite-dimensional scenarios.

### Homogenous Polynomial Kernels (No Constant Term)

We start with an example of a simplified form of the polynomial kernel, without constants. These are called **Homogenous Kernels**.

This example is specific for vectors in $$\mathbb{R}^2$$; we will be lifting them to $$\mathbb{R}^3$$. The kernel function is $$\kappa(x,y)={\langle x,y\rangle}^2$$. Incidentally, the kernel function will work with any $$\mathbb{R}^n$$ domain of vectors, but for illustration purposes, we will stick to $$\mathbb{R}^2$$.

Assume the vectors are $$X=\begin{bmatrix}X_1 \\ X_2\end{bmatrix}$$ and $$Y=\begin{bmatrix}Y_1 \\ Y_2\end{bmatrix}$$, and the inner product is the usual inner product defined in Euclidean space, i.e., $$\langle x,y\rangle=x^Ty$$.

$$
\kappa(x,y)={\langle x,y\rangle}^2\\
={(X_1Y_1+X_2Y_2)}^2={(X_1Y_1)}^2+{(X_2Y_2)}^2+2X_1Y_1X_2Y_2
$$

We'd like to know what mapping function $$\phi(x)$$ (or equivalently, $$\kappa(x, \bullet)$$) projects our original points $$X$$ and $$Y$$ to $$\mathbb{R}^3$$, so that the inner product in this new vector space is equal to the evaluation of the kernel function $$\kappa(x,y)$$.

Because I know the answer already so we will write it straight away.

$$
\phi(x)=\kappa(x,\bullet)=\begin{bmatrix}{Yx_1}^2 \\
{x_2}^2 \\ \sqrt 2x_1x_2 \end{bmatrix}
$$

You can verify yourself that this satisfies the requirement of $$\kappa(x,y)=\langle\phi(x)\phi(y)\rangle$$.

$$
\langle\phi(x)\phi(y)\rangle=\begin{bmatrix}{X_1}^2 && {X_2}^2 && \sqrt 2X_1X_2 \end{bmatrix}\cdot \begin{bmatrix}{Y_1}^2 \\
{Y_2}^2 \\ \sqrt 2Y_1Y_2 \end{bmatrix}\\
= {(X_1Y_1+X_2Y_2)}^2={(X_1Y_1)}^2+{(X_2Y_2)}^2+2X_1Y_1X_2Y_2 \\
={\langle x,y\rangle}^2=\kappa(x,y)
$$

### General Polynomial Kernel
The kernel function described above is a special case of the Generalised Polynomial Kernel form, which is of the form (again assuming the Euclidean version of inner product $$x^Ty$$):

$$
\kappa(x,y)={(x^Ty+c)}^d
$$

where $$c\in\mathbb{R}$$ and $$d\in\mathbb{N}$$ to. **Note that $$d$$ does not represent the number of dimensions that the vector will be lifted to.**

Let's look at a slight generalisation for a vector $$\mathbb{R}^n$$ for $$d=2$$, using the application of the **Binomial Theorem**.

$$
\kappa(x,y)={(x^Ty+c)}^2={\left(\sum_{i=1}^nx_iy_i+c\right)}^2 \\
= {\left(\sum_{i=1}^nx_iy_i\right)}^2 + 2c\left(\sum_{i=1}^nx_iy_i\right) + c^2 \\
= \sum_{i=1}^n{(x_iy_i)}^2 + 2\sum_{i=1}^{n-1}\sum_{j+1}^n (x_ix_j)(y_iy_j) + 2c\left(\sum_{i=1}^nx_iy_i\right) + c^2 \\
= \sum_{i=1}^n{(x_iy_i)}^2 + \sum_{i=1}^{n-1}\sum_{j+1}^n (\sqrt{2}x_ix_j)(\sqrt{2}y_iy_j) + \left(\sum_{i=1}^n\sqrt{2c}x_i\sqrt{2c}y_i\right) + c^2 \\
$$

Through visual inspection (or by expanding the above and noting the pattern of the polynomial), we can infer the mapping $$\phi$$ as;

$$
\phi(x)=\begin{bmatrix}
{x_1}^2 \\
{x_1}^2 \\
\vdots \\
{x_n}^2 \\
\sqrt{2}x_1x_2 \\
\sqrt{2}x_1x_3 \\
\vdots \\
\sqrt{2}x_1x_n \\
\sqrt{2}x_2x_3 \\
\sqrt{2}x_2x_4 \\
\vdots \\
\sqrt{2}x_2x_n \\
\vdots \\
\sqrt{2}x_{n-1}x_n \\
\sqrt{2c}x_1 \\
\sqrt{2c}x_2 \\
\vdots \\
\sqrt{2c}x_n \\
c
\end{bmatrix}
$$

Similar mappings for higher values of $$d$$ can be inferred by applying the **Multinomial Theorem**.


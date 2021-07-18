---
title: "Kernel Functions: Reproducing Kernel Hilbert Spaces"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Kernels", "Theory", "Functional Analysis", "Linear Algebra"]
draft: true
---
This article uses the groundwork laid in [Kernel Functions: Functional Analysis and Linear Algebra Preliminaries]({% post_url 2021-07-17-kernel-functions-functional-analysis-preliminaries %}) to discuss the construction of **Reproducing Kernel Hilbert Spaces**, which allows certain functions (called **Kernel Functions**) to be a valid representation of an **inner product** in (potentially) higher-dimensional space. This construction will allow us to perform the necessary higher-dimensional computations, without projecting every point in our data set into higher dimensions, explicitly, in the case of **Non-Linear Support Vector Machines**, which will be discussed in an upcoming article.

This construction, it is to be noted, is not unique to Support Vector Machines, and applies to the general class of techniques in Machine Learning, called **Kernel Methods**. 

## Inner Product and the Gram Matrix
With this intuition, we turn to a common operation in many Machine Learning algorithms: the **Inner Product**. 

For a quick refresher of **Inner Product** (specifically, the **Dot Product** ), check out [Dot Product: Algebraic and Geometric Equivalence]({% post_url 2021-04-11-dot-product-algebraic-geometric-equivalence %}).

We ask the following question: **when is a function on a pair of original vectors also the inner product of those vectors projected into a higher dimensional space?** If we can answer this question, then we can circumvent the process of projecting a pair of vectors into higher-dimensional space, and then computing their inner products, and apply a single linear functional which gives us the inner products in higher dimensional space.

Assume that such a linear functional exists. We call this the **kernel function** $$\kappa(x,y)$$. Now recall the notion of function currying from programming, where specifying a subset of arguments to a function, yields a new function with the already-passed-in parameters fixed, and the rest of the parameters still available to specify.

If we specify one of the parameters of $$\kappa(x,y)$$, say $$y=Y$$, this yields a new function with $$y$$ fixed to $$Y$$ and $$x$$ still available to specify. We use the common notation used for common functions, by putting a dot in the place of the unspecified variables. We write it like so:

$$\kappa(x,y)=\kappa(\bullet,\mathbf{Y})$$

Mathematically, we can consider this as a mapping function $$\Phi_Y$$ which maps $$\mathbb{R}^n$$ (which is $$x$$) to the space of functions which operate on $$\mathbb{R}^n$$. We represent this function space as $$\mathcal{F}(\mathbb{R}^n)$$

$$
\Phi_Y:\mathbb{R}^n\rightarrow \mathcal{F}(\mathbb{R}^n)
$$

An example of a kernel function would be:

$$
\kappa(x,y)=x_1y_1+x_2y_2
$$

where $$x=\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$
and $$y=\begin{bmatrix}
y_1 \\
y_2
\end{bmatrix}
$$.

Setting $$y=\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$, we get:

$$
\kappa(x, \begin{bmatrix}
1 \\
1
\end{bmatrix}
)=\kappa(\bullet, \begin{bmatrix}
1 \\
1
\end{bmatrix}
)=\Phi_{[1 \hspace{1mm} 1]^T}
$$

where the expression for $$\Phi_{[1 \hspace{1mm} 1]^T}$$ is:

$$
\Phi_{[1 \hspace{1mm} 1]^T}(x)=x_1+x_2
$$


## Metric Spaces
A set equipped with a distance metric is a metric space. A distance metric is defined as a function which defines the "distance" between two members of a set. The definition of what "distance" consititutes, varies based on the sort of metric space, and the application. A distance metric $$d(x,y)$$ must satisfy the following conditions.

- $$d(x,y)>0, \forall x\neq y$$
- $$d(x,y)=0 \Rightarrow x=y$$
- $$d(x,y)=d(y,x)$$
- $$d(x,z)\leq d(x,y)+d(y,z)$$

## Normed Spaces
A vector space equipped with a norm, not neceessarily complete
## Banach Spaces
A vector space equipped with a norm, which has been completed
## Hilbert Spaces
A Banach Space equipped with an inner product
## Hilbert Spaces of Functions
## Riesz Representation Theorem
A link between **Functional Analysis** and **Linear Algebra**
## Properties of Kernel Functions
Symmetric, Positive and Semi-Definite
## Reproducing Kernel Hilbert Spaces : Construction
Proofs of properties of RKHS Inner Product
## Alternative Formulation : Mercer's Theorem

---
title: "Kernel Functions: Reproducing Kernel Hilbert Spaces"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Kernels", "Theory", "Functional Analysis", "Linear Algebra"]
draft: true
---
This article uses the groundwork laid in [Kernel Functions: Functional Analysis and Linear Algebra Preliminaries]({% post_url 2021-07-17-kernel-functions-functional-analysis-preliminaries %}) to discuss the construction of **Reproducing Kernel Hilbert Spaces**. We originally asked the following question: **what can we say about a function on a pair of input vectors which also ends up being the inner product of those vectors projected onto a higher dimensional space?**

If we can answer this question, then we can circumvent the process of projecting a pair of vectors into higher-dimensional space, and then computing their inner products; we can simply apply a single function which gives us the inner products in higher dimensional space.

![Kernel Function Shortcut](/assets/images/kernel-function-shortcut-diagram.jpg)


## Proof by Construction

Mathematically, we are looking for a few things:
- A mapping function $$\phi(x)$$: The mapping function projects our input into a higher-dimensional space.
- A definition of an inner product operation $$\langle\bullet, \bullet\rangle$$: We already know (or think we know) that this inner product is the same as our intuitive understanding of an inner product, but we'll see.
- A kernel function $$\kappa(x,y)$$ 
We will alter our original question slightly to give some motivation for this proof. We ask the following:
- Is there a Hilbert space where the kernel function we choose is a valid inner product operation?
- If so, what does the projecting function
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

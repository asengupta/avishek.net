---
title: "Kernel Functions: Reproducing Kernel Hilbert Spaces"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Kernels", "Theory", "Functional Analysis", "Linear Algebra"]
draft: false
---
This article uses the previous mathematicsl groundwork to discuss the construction of **Reproducing Kernel Hilbert Spaces**. We'll make several assumptions that have been proved and discussed in those articles. There are multiple ways of discussing Kernel Functions, like the **Moore–Aronszajn Theorem** and **Mercer's Theorem**. We may discuss some of those approaches in the future, but here we will focus on the constructive approach here to characterise **Kernel Functions**.

The two specific posts discussing the background are:

- [Kernel Functions: Functional Analysis and Linear Algebra Preliminaries]({% post_url 2021-07-17-kernel-functions-functional-analysis-preliminaries %})
- [Functional Analysis: Results for Operators]({% post_url 2021-07-19-functional-analysis-results-for-operators %})

We originally asked the following question: **what can we say about a function on a pair of input vectors which also ends up being the inner product of those vectors projected onto a higher dimensional space?**

If we can answer this question, then we can circumvent the process of projecting a pair of vectors into higher-dimensional space, and then computing their inner products; we can simply apply a single function which gives us the inner products in higher dimensional space.

![Kernel Function Shortcut](/assets/images/kernel-function-shortcut-diagram.jpg)

## Proof by Construction

Mathematically, we are looking for a few things:
- A mapping function $$\phi(x)$$: The mapping function projects our input into a higher-dimensional space.
- A definition of an inner product operation $$\langle\bullet, \bullet\rangle$$: We already know (or think we know) that this inner product is the same as our intuitive understanding of an inner product, but we'll see.
- A kernel function $$\kappa(x,y)$$ which performs both of the above operations in one shot, i.e., $$\kappa(x,y)=\langle\Phi(x)\cdot\Phi(y)\rangle$$.

There are three properties that a valid inner product operation must satisfy: we discussed them in [Kernel Functions: Functional Analysis and Linear Algebra Preliminaries]({% post_url 2021-07-17-kernel-functions-functional-analysis-preliminaries %}). This implies that $$\kappa$$ must satisfy these properties. I reproduce them below for reference.

- **Positive Definite**: $$\kappa(x,x)>0$$ if $$x\neq 0$$
- **Principle of Indiscernibles**: $$x=0$$ if $$\kappa(x,y)=0$$
- **Symmetric**: $$\kappa(x,y)=\kappa(y,x)$$
- **Linear**:
    - $$\kappa(\alpha x,y)=\alpha\kappa(x,y), \alpha\in\mathbb{R}$$
    - $$\kappa(x+y,z)=\kappa(x,z)+\kappa(y,z)$$

We will alter our original question slightly to give some motivation for this proof. We ask the following:
- Is there a Hilbert space where the kernel function $$\kappa$$ we choose is a valid inner product operation?
- If so, what does the projecting function $$\Phi$$ look like?

Now, recall the criterion for positive semi-definiteness for a Gram matrix of $$n$$ data points. Translated into polynomial form, it looked like this:

$$
\sum_{j=1}^n\sum_{i=1}^n\alpha_i\alpha_j\kappa(x_j, x_i) \geq 0
$$

You can see immediately, that for $$n=1$$, and $$\alpha_1=1$$, the above simplifies to $$\kappa(x_1,y_1)=\kappa(x,y)\geq 0$$. This immediately gives us a hint about functions for which positive semi-definite matrices can be constructed out of all data points (without making any assumptions about those data points); they are good candidates for kernel functions. Note that the above simplification results from having a single data point in our data set.

Furthermore, if $$\kappa$$ is symmetric, the Gram matrix will be symmetric because $$G_{ij}=\kappa(x_i, x_j)=\kappa(x_j, x_i)=G_{ji}$$.

So there is definitely strong evidence of positive semi-definite functions being kernel functions, but we need to add a little more rigour to our intuition. For example, we still have no idea what the mapping function $$\Phi$$ should look like. 

## Linear Combinations of Positive Semi-definite Matrices

There are two more points from the above exploratory analysis that is worth discussing, because we will be using it in our proof.

- The sum of positive semi-definite matrices is also positive semi-definite. Take two positive semi-definite matrices $$A$$ and $$B$$. Then, by definition of positive semi-definiteness, $$v^TAv\geq 0$$ and $$v^TBv\geq 0$$. So if we write:

  $$
  v^T(A+B)v=(v^TA+v^TB)v=\underbrace{\underbrace{v^TAv}_{\geq 0}+\underbrace{v^TBv}_{\geq 0}}_{\geq 0}
  $$

- Positive semi-definite matrices scaled by non-negative scalars $$\alpha\geq 0, \alpha\in\mathbb{R}$$ are also positive semi-definite because:

  $$
  v^T(\underbrace{\alpha}_{\geq 0} A)v= \underbrace{\underbrace{\alpha}_{\geq 0}\underbrace{v^TAv}_{\geq 0}}_{\geq 0}  $$

The practical implication, as we will see, is that non-negative linear combinations of kernel functions are also kernel functions. In practice, the proof will use a set of kernel functions defined by a particular set of data points as basis vectors and use those to define a vector space of kernel functions. Any function in this space will then necessarily be a kernel function as well because of the implications we just discussed.

## Evaluation Functionals

The Evaluation Functional is an interesting function: it takes another function as an input, and applies a specific argument to that function. As an example, if we have a function, like so:

$$
f(x)=2x+3
$$

We can define an evaluation functional called $$\delta_3(f)$$ such that:

$$
\delta_3(f)=f(3)=2.3+3=9
$$

## Continuity and Boundedness of Evaluation Functional
Here we will treat the Evaluation Functional in its functional form (the "formula view", if you like). Is the graph of the Evaluation Functional continuous. We can prove that if a linear functional is bounded, then it is also continuous. In this case, we will prove that the Evaluation Functional is bounded in the function space $$\mathcal{H}$$.

## Connecting the Evaluation Functional and the Riesz Representation Theorem

We have already discussed that the Riesz Representation Theorem applies to bounded linear functionals. Here we connect that concept with the Evaluation Functional.

For every $$x\in X$$, we have a corresponding evaluation functional $$\delta_x$$. In other words, there exists a mapping $$\Phi:x\mapsto\delta_x$$

By the Riesz Representation Theorm, the application of the evaluation functional onto any function $$f\in\mathcal{H}$$ is equivalent to the inner product of a vector $$K_x$$ and $$f$$. We can write this down as:

$$\delta_x(f)={\langle K_x,f\rangle}_H=f(x)$$

This leads to the Reproducing Kernel property. If we assume that the vector corresponding to the evaluation functional is the Kernel Function, then we can take another Kernel Function, say $$K_y$$, and can write:

$$
\delta_x(K_y)=\langle K_x, K_y \rangle=K_y(x)=K(x,y)
$$

Now we know that a kernel function takes in two vectors as input. How are we to define $$K_x$$, $$K_y$$ such that they take in one value. The next section discusses the construction of the mapping.

## Function Currying
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

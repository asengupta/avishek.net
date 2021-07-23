---
title: "Functional Analysis: Results for Operators"
author: avishek
usemathjax: true
tags: ["Mathematics", "Theory", "Operator Theory", "Functional Analysis"]
draft: false
---
This article expands the groundwork laid in [Kernel Functions: Functional Analysis and Linear Algebra Preliminaries]({% post_url 2021-07-17-kernel-functions-functional-analysis-preliminaries %}) to discuss some more properties and proofs for some of the properties of functions that we will use in the construction of **Reproducing Kernel Hilbert Spaces**. 

## Vector and Operator Norms

## Linearity
Here we will treat the Evaluation Functional as an infinite dimensional vector, and apply it on a function $$f$$. This amounts to taking the inner product with $$f$$. If we treat the function $$f$$ as the vector, and the values of the corresponding values of the Evaluation Functional as coefficients of a function, then it is definitely a linear functional. All evaluation functionals are linear functionals.

## Continuity and Boundedness
Here we will treat the Evaluation Functional in its functional form (the "formula view", if you like). Is the graph of the Evaluation Functional continuous. We can prove that if a linear functional is bounded, then it is also continuous. In this case, we will prove that the Evaluation Functional is bounded in the function space $$\mathcal{H}$$.

Boundedness is defined as follows:

$$
\|Tx\|\leq C\|x\| \text{  for } C>0, C\in\mathbb{R}
$$

Continuity is defined as follows:

$$
f\rightarrow f_0 \Rightarrow Tf\rightarrow Tf_0
$$


## Riesz Representation Theorem

Informally, the Riesz Representation Theorem states that every continuous (and therefore, bounded) linear functional $$f$$ applied on a vector $$x$$ in a vector space $$X$$ is equivalent to the inner product of a unique vector (corresponding to $$f$$) with $$x$$.

We'll show a motivating example in finite-dimensional space $$\mathbb{R}^2$$. Assume we have a vector $$\vec{z}=\begin{bmatrix}1\\2\end{bmatrix}$$, and a function $$f(x)=2\vec{x}_1+3\vec{x}_2$$.

Then there exists a vector $$\vec{v}=\begin{bmatrix}2\\3\end{bmatrix}$$ corresponding to $$f$$ such that $$f(\vec{z})=\langle\vec{v},\vec{z}\rangle$$. Indeed, if we evaluate both sides, we get:

$$
f(\vec{z})=2.1+3.2=8 \\
\langle\vec{v},\vec{z}\rangle=2.1+3.2=8
$$

Since the evaluation functional can be pro

## Metric Spaces
A set equipped with a distance metric is a metric space. A distance metric is defined as a function which defines the "distance" between two members of a set. The definition of what "distance" consititutes, varies based on the sort of metric space, and the application. A distance metric $$d(x,y)$$ must satisfy the following conditions.

- $$d(x,y)>0, \forall x\neq y$$
- $$d(x,y)=0 \Rightarrow x=y$$
- $$d(x,y)=d(y,x)$$
- $$d(x,z)\leq d(x,y)+d(y,z)$$

## Alternative Formulation : Mercer's Theorem

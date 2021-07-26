---
title: "Kernel Functions: Understanding the Evaluation Functional"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Kernels", "Theory", "Functional Analysis"]
draft: true
---
## Evaluation Functional and the Riesz Representation Theorem

We have already discussed that the Riesz Representation Theorem applies to bounded linear functionals. Here we connect that concept with the Evaluation Functional.

For every $$x\in X$$, we have a corresponding evaluation functional $$\delta_x$$. In other words, there exists a mapping $$\Phi:x\mapsto\delta_x$$

By the Riesz Representation Theorm, the application of the evaluation functional onto any function $$f\in\mathcal{H}$$ is equivalent to the inner product of a vector $$K_x$$ and $$f$$. We can write this down as:

$$\delta_x(f)={\langle K_x,f\rangle}_H=f(x)$$

This leads to the Reproducing Kernel property. If we assume that the vector corresponding to the evaluation functional is the partially-evaluated Kernel Function, then we can take another partially-evaluated Kernel Function, say $$K_y$$, and can write:

$$
\delta_x(K_y)=\langle K_x, K_y \rangle=K_y(x)=K(x,y)
$$

where

$$
K_x=\kappa(x,\bullet) \\
K_y=\kappa(y,\bullet)
$$

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

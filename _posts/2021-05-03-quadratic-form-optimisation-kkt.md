---
title: "Quadratic Optimisation, Lagrangian Duals, and the Karush-Kuhn-Tucker Conditions"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Quadratic Optimisation", "Linear Algebra", "Optimisation", "Theory"]
---

This article continues the original discussion on **Quadratic Optimisation**, where we considered **Principal Components Analysis** as a motivation. Here, we extend the **Lagrangian Multipliers** approach, which in its current form, admits only equality constraints. We will extend it to allow constraints which can be expressed as inequalities.

This applies to the general class of **Convex Optimisation**, so it will automatically apply to **Quadratic Programming** problems. As we will see, this will lead to some cases where constraints are not activated versus cases where they are not.

As a result, integrating both cases lead to the **Karush-Kuhn-Tucker conditions**. This will be useful for finding the solution to determining the maximal margin hyperplane in **Support Vector Machines**, because the constraints (the supporting hyperplanes) are expressed in terms of inequalities.

We will touch upon the **Saddle Point Theorem** for the Lagrangian dual, but not delve too deep into it. **Convex Optimisation** is a vast topic, and there are very good books which treat the subject in a lot more detail.

## Lagrangian

We have already seen in [Vector Calculus: Lagrange Multipliers, Manifolds, and the Implicit Function Theorem]({% post_url 2021-04-24-vector-calculus-lagrange-multipliers %}) that the gradient vector of a function can be expressed as a linear combination of the gradient vectors of the constraint manifolds.

$$ \mathbf{
Df=\lambda_1 Dh_1(U,V)+\lambda_2 Dh_2(U,V)+\lambda_3 Dh_3(U,V)+...+\lambda_n Dh_n(U,V)
} \\
\\
$$

We can rewrite this as:

$$ \mathbf{
Df(x,\lambda)=\sum_{i=1}^n\lambda_i.Dg_i(x)
} \\
\Rightarrow f(x,\lambda)=\sum_{i=1}^n\lambda_i.g_i(x)
$$

where $$x=(U,V)$$. We will not consider the pivotal and non-pivotal variables separately in this discussion.

So far, we have been very general about what kinds of functions $$f$$ and $$g_i$$ look like. Since we will be focusing a lot more on Convex Optimisation going forward, we will qualify the forms of these functions.

We will first consider the Lagrangian form of a function. The Lagrangian form is simply restating the Lagrange Multiplier form as a function $$L(X,\lambda)$$, like so:

$$
L(x,\lambda)=f(x,\lambda)+\sum_{i=1}^n\lambda_i.g_i(x)
$$

We have simply moved all the terms of the Lagrangian formulation onto one side and denoted it by $$L(x,\lambda)$$.

We
## Exploring the Properties of the Lagrangian

## Affine Functions as Constraints
We will now 
### Convex Functions
### Affine Sets and Affine Functions
## Geometric Intuition of Convex Optimisation
### Active Constraints
### Inactive Constraints
## Max-Min Inequality
### Duality Gap
## Karush-Kuhn-Tucker Conditions
## Saddle Point Theorem

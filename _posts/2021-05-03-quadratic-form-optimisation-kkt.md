---
title: "Quadratic Optimisation, Lagrangian Duals, and the Karush-Kuhn-Tucker Conditions"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Quadratic Optimisation", "Linear Algebra", "Optimisation", "Theory"]
draft: true
---

This article continues the original discussion on **Quadratic Optimisation**, where we considered **Principal Components Analysis** as a motivation. Here, we extend the **Lagrangian Multipliers** approach, which in its current form, admits only equality constraints. We will extend it to allow constraints which can be expressed as inequalities. This applies to the general class of **Convex Optimisation**, so it will automatically apply to **Quadratic Programming** problems.

As we will see, this will lead to some cases where constraints are not activated versus cases where they are not.

As a result, integrating both cases lead to the **Karush-Kuhn-Tucker conditions**. This will be useful for finding the solution to determining the maximal margin hyperplane in **Support Vector Machines**, because the constraints (the supporting hyperplanes) are expressed in terms of inequalities.

We will touch upon the **Saddle Point Theorem** for the Lagrangian dual, but not delve too deep into it. **Convex Optimisation** is a vast topic, and there are very good books which treat the subject in a lot more detail.

## Max-Min Inequality
## Geometric Intuition of Convex Optimisation
### Active Constraints
### Inactive Constraints
## Karush-Kuhn-Tucker Conditions
## Saddle Point Theorem

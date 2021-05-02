---
title: "Quadratic Optimisation and the Karush-Kuhn-Tucker Conditions: Part Two"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Quadratic Optimisation", "Linear Algebra", "Optimisation", "Theory"]
draft: true
---

This article continues the original discussion on **Quadratic Optimisation**, where we considered **Principal Components Analysis** as a motivation. Here, we extend the **Lagrangian Multipliers** approach, which in its current form, admits only equality constraints. We will extend it to allow constraints which can be expressed as inequalities.

As we will see, this will lead to some cases where constraints are not activated versus cases where they are not.

As a result, we will see how integrating both cases lead to the **Karush-Kuhn-Tucker conditions**. This will be useful for finding the solution to determining the maximal margin hyperplane in **Support Vector Machines**, because the constraints (the supporting hyperplanes) are expressed in terms of inequalities.


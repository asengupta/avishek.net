---
title: "Kernel Functions: Example and the Representer Theorem"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Kernels", "Theory", "Functional Analysis", "Linear Algebra"]
draft: true
---
This article uses the previous mathematical groundwork to delve into a non-trivial Reproducing Kernel Hilbert Space (RKHS, in short), as well as discuss why the particular kernel form makes potentially intractable infinite-dimensional Machine Learning problems tractable. We do this by discussing the **Representer Theorem**.

The specific posts discussing the background are:

- [Kernel Functions: Kernel Functions with Mercer's Theorem]({% post_url 2021-07-21-kernel-functions-mercers-theorem %})
- [Kernel Functions: Kernel Functions with Reproducing Kernel Hilbert Spaces]({% post_url 2021-07-20-kernel-functions-rkhs %})
- [Kernel Functions: Functional Analysis and Linear Algebra Preliminaries]({% post_url 2021-07-17-kernel-functions-functional-analysis-preliminaries %})
- [Functional Analysis: Norms, Linear Functionals, and Operators]({% post_url 2021-07-19-functional-analysis-results-for-operators %})
- [Functional and Real Analysis Notes]({% post_url 2021-07-18-notes-on-convergence-continuity %})

We will discuss the following:

- An Example of a Non-Trivial RKHS
- Empirical Risk Minimisation
- The Representer Theorem

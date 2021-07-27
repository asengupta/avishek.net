---
title: "Kernel Functions with Mercer's Theorem"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Kernels", "Theory", "Functional Analysis"]
draft: true
---
This article takes a second perspective to **Kernel Functions** using **Mercer's Theorem**. We discussed this theorem in [Functional Analysis: Norms, Operators, and Some Theorems]({% post_url 2021-07-19-functional-analysis-results-for-operators %}). We will see that Mercer's Theorem applies somewhat more directly to the characterisation of Kernel Functions, and there is no need for an elaborate construction, like we do for **Reproducing Kernel Hilbert Spaces**.

The specific posts discussing the background are:

- [Kernel Functions: Functional Analysis and Linear Algebra Preliminaries]({% post_url 2021-07-17-kernel-functions-functional-analysis-preliminaries %})
- [Functional Analysis: Norms, Linear Functionals, and Operators]({% post_url 2021-07-19-functional-analysis-results-for-operators %})
- [Functional and Real Analysis Notes]({% post_url 2021-07-18-notes-on-convergence-continuity %})

It is also advisable (though not necessary) to review - [Kernel Functions with Reproducing Kernel Hilbert Spaces]({% post_url 2021-07-20-kernel-functions-rkhs %}) to contrast and compare that approach with the one shown here.

Recall what Mercer's Theorem states:

$$
\kappa(x,y)=\sum_{i=1}^\infty \lambda_i \psi_i(x)\psi_i(y)
$$

where \kappa(x,y) is a positive semi-definite function and $$\psi_i(\bullet)$$ is the $$i$$th eigenfunction. Note that this implies that there are an infinite number of eigenfunctions.

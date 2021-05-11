---
title: "Support Vector Machines from First Principles: Non-Linear SVMs and the Kernel Trick"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Support Vector Machines", "Kernels", "Theory"]
draft: true
---

The Support Vector Machine we have discussed only works for linearly separable data. Real-world data sets are seldom linearly separable. In this
We will focus on the advantages of projecting the linearly-inseparable data into higher dimensions, and why that might lead to a new problem which can solved using linear separation techniques using **Support Vector Machines**.

We'll then discuss the computational disadvantages of doing this in practice, and look at some of the theory, namely the **Kernel Trick**, which allows us to perform the necessary higher-dimensional computations, without projecting every point in our data set into higher dimensions, explicitly.


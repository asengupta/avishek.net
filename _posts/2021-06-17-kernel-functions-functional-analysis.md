---
title: "Kernel Functions: Results from Functional Analysis"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Kernels", "Theory", "Functional Analysis"]
draft: true
---

This article discusses an important construction called **Reproducing Kernel Hilbert Spaces**, which allows the Kernel function to be a valid representation of an inner product in (potentially) higher-dimensional space, from **Functional Analysis**. This construction will allow us to perform the necessary higher-dimensional computations, without projecting every point in our data set into higher dimensions, explicitly, in the case of **Non-Linear Support Vector Machines**, which will be discussed in the upcoming article.

This construction, it is to be noted, is not unique to Support Vector Machines, and applies to the general class of techniques in Machine Learning, called **Kernel Methods**.

As usual, there is a whole raft of mathematical machinery that we'll have to define, to understand some of these concepts. Most of them can be intuitively related to familiar notions of $$\mathbb{R}^n$$ spaces, and we'll use motivating examples to connect the mathematical machinery to the engineer's intuition.

## Mathematical Preliminaries

## Metric Spaces
A set equipped with a distance metric
## Banach Spaces
A metric space equipped with a norm
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

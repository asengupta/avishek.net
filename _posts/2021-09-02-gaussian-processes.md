---
title: "Gaussian Processes"
author: avishek
usemathjax: true
tags: ["Theory", "Gaussian Processes", "Probability", "Machine Learning"]
draft: false
---

Continuing from the roadmap set out in [Road to Gaussian Processes]({% post_url 2021-04-17-road-to-gaussian-processes %}), we begin with the geometry of the central object which underlies this Machine Learning Technique, the **Multivariate Gaussian Distribution**. We will study its form to build up some geometric intuition around its interpretation.

To do this, we will cover the material in two phases.

The first pass will build the intuition necessary to understand Gaussian Processes and how they relate to regression, and how they model uncertainty during interpolation and extrapolation.

The second pass will delve into the mathematical underpinnings necessary to appreciate the technique more rigorously. Specifically, the following material will be covered:

- Schur Complements and Diagonalisation of Partitioned Matrices
- Conditioned Distributions as Gaussians
- Sampling from Multivariate Gaussian Distributions
- Generalising Discrete Covariance Matrices to Kernels

# First Pass: Intuition
- A single sampled vector from an $$n$$-dimensional Multivariate Gaussian represents one possible data set.
- The covariance matrix represents how correlated each dimension is to each other.
- Conditioning a Multivariate Gaussian distribution is equivalent to setting a specific dimension to a specific value (which is usually a point in the test data set).

# Second Pass: Mathematical Underpinnings
This pass will delve into the mathematical underpinnings necessary to appreciate the technique more rigorously.

## Schur Complements and Diagonalisation of Partitioned Matrices
## Conditioned Distributions as Gaussians
## Sampling from Multivariate Gaussian Distributions
## Generalising Discrete Covariance Matrices to Kernels

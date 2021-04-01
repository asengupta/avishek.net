---
title: "Ways of interpreting a matrix"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Linear Algebra", "Theory"]
---

These are a few ways to look at a matrix.
- A matrix is a collection of column vectors.
- A matrix is a collection of row vectors.
- A matrix is a linear transformation, with its column vectors being the new basis.

## Intuition:

- A matrix $$A$$ cannot be inverted (i.e., it does not have a unique inverse) if any of its column vectors are linearly dependent on the others.
    - This is because, then there will always be a non-zero vector solution which will lose all of its components to zero; and there is no way to reverse that operation to recover the original vector.

![A Single Linearly Dependent Vector results in a non-invertible matrix](/assets/even-one-linear-dependence-causes-non-invertible-matrix.jpg)


---
title: "Quick Intuitions about Common Ways of Looking at Matrix Multiplications"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Linear Algebra", "Theory"]
---
We consider the more frequently utilised viewpoints of matrix multiplication, and relate it to one or more applications where using a certain viewpoint is more useful. These are the viewpoints we will consider.

- Linear Combination of Columns
- Linear Combination of Rows
- Linear Transformation
- Sum of Columns into Rows
- Dot Product of Rows and Columns
- Block Matrix Multiplication

## Linear Combination of Columns
This is the most common, and probably one of the most useful, ways of looking at matrix multiplication. This is because the concept of linear combinations of columns is a fundamental way of determining linear independence (or linear dependence), which then informs us about many things, including:

- Dimensionality of the column and row space
- Dimensionality of the null space and left null space
- Uniqueness of solutions
- Invertibility of matrix

![Linear Combination of Columns](/assets/images/linear-combination-matrix-multiplication.jpg)

## Linear Combination of Rows
Most commonly used interpretation when defining and working with vector subspaces, and checking linear independence
## Sum of Columns into Rows
Common form of treating a matrix when performing **LU** Decomposition
## Linear Transformation
Very common way of perceiving matrix multiplication in computer graphics, as well as when considering change of basis.

## Dot Product of Rows and Columns
Common form of treating matrices when doing proofs where the transpose invariance property of symmetric matrices is utilised, i.e., $$A^T=A$$.
## Block Matrix Multiplication
![Block Matrix Multiplication](/assets/images/block-matrix-multiplication.jpg)
Used when proofs involve properties of a larger matrix composed of submatrices.

### Recursive Calculation
The block matrix calculation can be extended to be recursive.
![Recursive Block Matrix Multiplication](/assets/images/recursive-block-matrix-multiplication.png)

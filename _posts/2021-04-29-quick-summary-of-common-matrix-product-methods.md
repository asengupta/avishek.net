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

This is obviously the most commonly used interpretation when defining and working with vector subspaces, as well.

![Linear Combination of Columns](/assets/images/linear-combination-matrix-multiplication.jpg)

## Linear Combination of Rows
There's not much more to say about the linear combinations of rows. However, any deduction about the row rank of a matrix from looking at its row vectors automatically applies to the column rank as well, so it is useful in situations where you find looking at rows easier than columns.

## Sum of Columns into Rows
The product of a column of the left matrix and a row of the right matrix gives a matrix of the same dimensions as the final result. Thus, each product results in one "layer" of the final result. Subsequent "layers" are added on through summation. Thus, product looks like so:

Thus, for $$A\in\mathbb{R}^{m\times n}$$ and $$B\in\mathbb{R}^{n\times p}$$

$$
AB=C_{A1}R_{B1}+C_{A2}R_{B2}+C_{A3}R_{B3}+...+C_{An}R_{Bn}
$$

This is a common form of treating a matrix when performing **LU Decomposition**. See [Matrix Outer Product: Columns-into-Rows and the LU Factorisation]({% post_url 2021-04-02-vectors-matrices-outer-product-column-into-row-lu %}) for an extended explanation of the **LU Factorisation**.

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

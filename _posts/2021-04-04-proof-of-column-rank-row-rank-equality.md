---
title: "Matrix Rank and Some Results"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Linear Algebra", "Theory"]
---

I'd like to introduce some basic results about the rank of a matrix. Simply put, the rank of a matrix is the number of independent vectors in a matrix. Note that I didn't say whether these are column vectors or row vectors; that's because of the following section which will narrow down the specific cases (we will also prove that these numbers are equal for any matrix).

A matrix is **full rank** if 1) all of its column vectors are linearly independent, and 2) all of its row vectors are linearly independent.
A matrix is **full column rank** if all of its column vectors are linearly independent.
A matrix is **full row rank** if all of its row vectors are linearly independent.

For an $$M\times N$$ matrix, if the rank of a matrix is less than than the smaller of M, N, i.e., $$min(M,N)$$, then we call it **degenerate**, **rank-deficient**, **singular**, etc. This has implications for whether a matrix is invertible or not, namely that **a degenerate matrix is not invertible.** See [Assorted Intuitions about Matrices]({% post_url 2021-04-03-matrix-intuitions %}) for a quick intuition.

Note here, that I didn't specify whether the rank implied column rank or row rank. As we shall see in a moment, we will prove that the column rank of a matrix always equals its row rank.

## Proof of Equality of Column Rank and Row Rank of a Matrix

Before getting into the proof, let's state an obvious fact (or maybe not so obvious, but at least it should follow from our definition of matrix multiplication).

The fact is that multiplying a matrx A, which has some column rank **c** and row rank **r** (just to be super general about ranks), cannot alter its column or row rank. Can you see why? It is because we understand that matrix multiplication is essentially both 1) a linear combination of column vectors, and 2) a linear combination of row vectors. **A linear combination of a set of vectors cannot create a linearly independent vector.**

That's like trying to combine the $$(1,0,0)$$ vector and the $$(0,1,0)$$ to create a $$(0,0,1)$$; you just cannot do it.

With that out of the way, let's consider a matrix, any matrix with column rank **c** and row rank **r**. We want to determine a relation between these two ranks.

We should be able to express this matrix as a linear combination of its **c** column vectors. It would look like this:

$$
A=
\begin{bmatrix}
| && | && | && ... && | \\
bc_1 && bc_2 && bc_3 && ... && bc_c \\
| && | && | && ... && | \\
\end{bmatrix}
\begin{bmatrix}
--- r_1 --- \\
--- r_2 --- \\
--- r_3 --- \\
... \\
--- r_c --- \\
\end{bmatrix}\\
$$

The only assumption I've made in the identity above is that $$bc_1$$, $$bc_2$$, etc. are linearly independent; **there are no assumptions about any of the rows $$r_1$$, $$r_2$$, etc.**

However, let us look at this same identity, but from the point of view of a linear combination of the row vectors $$r_2$$, $$r_2$$, ..., $$r_c$$. How many row vectors are there? **c** row vectors, of course, since by the rules of matrix multiplication, if the left matrix has **c** columns, the right matrix needs **c** rows.

This implies that the **row rank of the right matrix is at most c**. It can be less than **c**, since we have not made any assumptions about its row rank, but we now have an upper bound on the row rank of this matrix. By extension, the matrix **A**'s row rank cannot exceed **c** either. That is:

$$r\leq c$$

Now, we apply the same argument, but this time, we take the **r** linearly independent row vectors, from which we can get:

$$c\leq r$$

The only scenario which satisfies both of these above conditions is when $$\mathbf{r=c}$$.

**The column rank of a matrix always equals its row rank.**
It is important to note that this rule holds for every matrix. Let's quickly talk of the implications of this for general matrix multiplication. From here on out, we will not distinguish between row rank and column rank, because the values are the same. We will simply refer to it as a matrix's rank.

Let's assume **A** is a matrix of rank $$R_A$$ and matrix **B** has a rank of $$R_B$$. When we multiply them, it results in a matrix C with rank $$R_C$$. How is $$R_C$$ related to $$R_A$$ and $$R_B$$?

Well, simply based on the argument in the proof we just looked at, where we were multiplying two matrices, we can write:

$$
R_C\leq R_A \\
R_C\leq R_B
$$

This simply implies that
$$\mathbf{R_C=min(R_A, R_B)}$$

That is, **the rank of the matrix product is equal to the smaller of the ranks of the two multiplying matrices.**

It should follow automatically, that the **ranks of $$A^TA$$, $$AA^T$$ are always equal to the rank of $$A$$**.

## Notes
- The rank can be obtained from the row echelon form (or reduced row echelon form) of a matrix.

---
title: "Matrix Outer Product: Linear Combinations of Vectors"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Linear Algebra", "Theory"]
---
Matrix multiplication is a fundamental operation in almost any Machine Learning proof, statement, or computation. Much insight may be gleaned by looking at different ways of looking matrix multiplication. In this post, we will look at one (and possibly the most important) interpretation: namely, the **linear combination of vectors**.

In fact, the geometric abstraction of this operation allows us to infer many properties that might be obscured if we were treating matrix multiplication as simple sums of products of numbers.

To begin with, I'll state the one fact that holds true no matter how you performing matrix multiplication. Even if you forget everything you read in this article, remember this thing:

**Matrix multiplication is a linear combination of a set of vectors.**

Let's begin with a simple two-dimensional vector, like so:
$$
A_1=\begin{bmatrix}
2 \\
3 \\
\end{bmatrix}
$$

Let's introduce a $$1\times 1$$ vector $$x_1=\begin{bmatrix}
                                        2 \\
                                        \end{bmatrix}$$

We multiply them together, like so:
$$Y=A_1x_1
= \begin{bmatrix}
2 \\
3 \\
\end{bmatrix}.\begin{bmatrix}
              2 \\
              \end{bmatrix}
=\begin{bmatrix}
 4 \\
 6 \\
 \end{bmatrix}
$$

This is nothing special, simply a scaling of the $$\begin{bmatrix}2 && 3\end{bmatrix}^T$$ matrix.

## 1. Linear Combination of Column Vectors
Let's take the next step. We will add one more column vector to A, and add a number to $$x_1$$.

$$A_2=\begin{bmatrix}
2 && 5\\
3 && 10\\
\end{bmatrix}

v_2=\begin{bmatrix}
 2 \\
 3 \\
 \end{bmatrix}
$$

In this approach, we are only correlating columns on both sides. Consider the following diagram to see how this combination works.

![Column Vector Matrix Multiplication](/assets/Project%20125.png)

What you are really doing is this: you are considering the **weighted sum of all the column vectors** of $$A_2$$ (Remember, in this picture, $$A_2$$ is just a bunch of column vectors).

What are these weights? Each value in the column in $$v_2$$ is a weight. Each weight scales a column vector, and these weighted vectors are added together to form a single column.

I hope it becomes obvious that this implies that the number of column vectors in $$A_2$$ (the number of columns in $$A_2$$) must equal the number of values in $$v_2$$'s column (the number of rows in $$v_2$$), because there has to exist a one-to-one correspondence between them in order for this operation to be possible.


This results in a **linear combination of the column vectors** in $$A_2$$. A linear combination of two vectors is of the form $$\alpha x + \beta y$$, where $$\alpha$$ and $$\beta$$ are simple scalars (numbers). What you are really doing is either squashing or stretching some vectors by some factor (this can be a negative number), and then adding them together. That's what a linear combination essentially means.

To put it more concretely in this example, your computation is as follows:

$$A_2=\left[2.\begin{bmatrix}
2 \\
3 \\
\end{bmatrix} + 3.\begin{bmatrix}
                5 \\
                10 \\
                \end{bmatrix}\right]=
                \begin{bmatrix}
                                19 \\
                                36 \\
                                \end{bmatrix}
$$

Let's extend to the more general case, where we add another column to $$v_2$$, so now that we have:

$$A_3=\begin{bmatrix}
2 && 5\\
3 && 10\\
\end{bmatrix}

v_3=\begin{bmatrix}
 2 & -2\\
 3 & -4\\
 \end{bmatrix}
$$

How do we extend what we already know to this new case? Very simple: each column in $$v_3$$ results in a corresponding column in the final result, and each output column is computed exactly the same. You are still linearly combining all the column vectors in $$A_3$$, but the set of weights you're using depends upon which column of $$v_3$$ you are using for computation. Thus, this new computation is as follows:

$$A_2=\left[\left(2.\begin{bmatrix}
2 \\
3 \\
\end{bmatrix} + 3.\begin{bmatrix}
                5 \\
                10 \\
                \end{bmatrix}\right) \left(-2.\begin{bmatrix}
                                    2 \\
                                    3 \\
                                    \end{bmatrix} + (-4).\begin{bmatrix}
                                                    5 \\
                                                    10 \\
                                                    \end{bmatrix}\right)\right]
\\
=\begin{bmatrix}
                                19 && -24\\
                                36 && -46\\
                                \end{bmatrix}
$$

This concept, as usual, can be easily extended to any set of matrices.
### The Geometric Interpretation
Let's consider the geometric interpretation. If you have followed the verbal explanation so far, the geometric interpretation should be pretty straightforward to comprehend.

![Linear Combinations of Vectors](/assets/vector-linear-combination.png)

The vector $$\begin{bmatrix}2 && 3\end{bmatrix}^T$$ is stretched by a factor of 2, to become $$\begin{bmatrix}4 && 6\end{bmatrix}^T$$.
Similarly, the vector $$\begin{bmatrix}5 && 10\end{bmatrix}^T$$ is stretched by a factor of 3, to become $$\begin{bmatrix}15 && 30\end{bmatrix}^T$$.
The sum of these vectors is $$\begin{bmatrix}19 && 36\end{bmatrix}^T$$, as indicated by the red arrow in the diagram above.

**Quick Aside**: The column vectors which are linearly combined, come from the left side of the expression. The weights come from the right side. This is important to know, since the row column interpretation (which we will study next) inverts the order.

## Relevance to Machine Learning
In addition to vectors (and by extension, matrices) being used to frame almost every Machine Learning/Statistics problem, these are some examples of how they are used:

- Many Machine Learning problems related to prediction boil down to **determining a hyperplane that best captures the trend of the data**, subject to certain assumptions (eg: Linear Models/Generalised Linear Models).
- Many Machine Learning problems related to classification, boil down to **finding the optimal dividing hyperplane between two different classes of data** (eg: Support Vector Machines).
- Relationships between vectors give us important information about the space that they define (more on this in Vector Subspaces). This in turn can help us infer information certain important properties of a matrix (invertibility, eigenvectors, etc.). This can directly tell us whether certain Machine Learning processes can be applied or not.

---
title: "Statistics from Linear Algebra: Notes"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Theory", "Statistics", "Linear Algebra"]
draft: true
---

This article covers a smattering of statistical procedures which can be derived from Linear Algebra without recourse to Calculus. Much of the intuition is from the book **The Geometry of Multivariate Statistics** by *Thomas D. Wickens*. If you have ever wondered: "How is the formula for the F-Test derived?", I'd urge you to take a deeper look at this book.

We will discuss the following relations:

- Mean as the Projection Coefficient onto the All-Ones Vector
- Variance as a Projection onto the Null Space of the All-Ones Vector
- Pearson's Correlation Coefficient as the Cosine of Angle between Two Vectors
- Coefficient of Projection as the Linear Regression Coefficient

## Mean as the Projection Coefficient onto the All-Ones Vector
Assume $$n$$ scalar observations which form a vector $$x\in\mathbb{R}^n$$. Then the mean of this set of observations is the projection of $$x$$ onto the all-ones vector given by $$Q\in\mathbb{R}^n$$.

Recall that the projection coefficient of a vector $$b$$ onto a vector $$a$$ is given by:

$$
t=\frac{a^Tb}{a^Ta}
$$

In this case, $$a=Q$$ and $$b=x$$. Note that $$Q^TQ$$ evaluates to $$n$$, so we get:

$$
t=\frac{Q^Tb}{Q^TQ} \\
t=\frac{1}{n}
\left(
\begin{bmatrix}
1 && 1 && \cdots && 1
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{bmatrix}
\right) \\
= \frac{x_1+x_2+\cdots+x_n}{n} \\
= \frac{1}{n}\sum_{i=1}^n x_i
$$

which is the formula for the mean.

## Variance as a Projection onto the Null Space of the All-Ones Vector
## Pearson's Correlation Coefficient as the Cosine of Angle between Two Vectors
## Coefficient of Projection as the Linear Regression Coefficient

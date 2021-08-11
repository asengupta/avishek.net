---
title: "Statistics from Linear Algebra: Notes"
author: avishek
usemathjax: true
tags: ["Theory", "Statistics", "Linear Algebra"]
draft: false
---

This article covers a smattering of statistical procedures which can be derived from Linear Algebra without recourse to Calculus. Much of this stems from me attempting to understand links between statistical concepts and **Linear Algebra**. This post is a work in progress and will receive new additions over time.

We will discuss the following relations:

- Mean as the Projection Coefficient onto the Model Vector
- Variance as the Averaged Projection onto the Null Space of the Model Vector
- Pearson's Correlation Coefficient as the Cosine of Angle between Two Vectors
- Coefficient of Projection as the Linear Regression Coefficient

## Model Vector and Error Space

We will introduce some terms and intuition to aid the discussions on the above statistical concepts.

## Mean as the Projection Coefficient onto the Model Vector
Assume $$n$$ scalar observations which form a vector $$x\in\mathbb{R}^n$$. Then the mean of this set of observations is the projection of $$x$$ onto the All-Ones Vector given by $$\nu_1\in\mathbb{R}^n$$. We will refer to $$\nu_1$$ as the **Model Vector**, going forward.

The **Model Vector** looks like:

$$
\begin{bmatrix}
1 \\ 1 \\ \vdots \\ 1 
\end{bmatrix}
\Leftarrow n\text{ entries}
$$

Recall that the projection coefficient of a vector $$b$$ onto a vector $$a$$ is given by:

$$
t=\frac{a^Tb}{a^Ta}
$$

The situation is as below.

![Geometric Interpretation of Mean](/assets/images/mean-geometric-interpretation.png)

In this case, $$a=\nu_1$$ and $$b=x$$. Note that $$\nu_1^T\nu_1=n$$, so we get:

$$
t=\frac{\nu_1^Tb}{\nu_1^T\nu_1} \\
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

## Variance as the Averaged Projection onto the Null Space of the Model Vector
![Geometric Interpretation of Variance](/assets/images/mean-variance-geometric-interpretation.png)


## Pearson's Correlation Coefficient as the Cosine of Angle between Two Vectors
## Coefficient of Projection as the Linear Regression Coefficient

## Footnotes
Much of the intuition is from the books listed under [References](#references). If you have ever wondered: "How is the formula for the F-Test derived?", or "Why do we use $$n-1$$ in the denominator for sample variance instead of $$n$$ without seeing **Bessel's Correction** in every explanation?" I'd urge you to take a deeper look at these books.

## References:
- 
- **The Geometry of Multivariate Statistics** : *Thomas D. Wickens*
- **Statistical Methods: The Geometric Approach** : *David J. Saville*, *Graham R. Wood*

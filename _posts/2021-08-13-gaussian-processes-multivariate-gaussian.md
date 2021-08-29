---
title: "Gaussian Processes: The Multivariate Gaussian Distribution"
author: avishek
usemathjax: true
tags: ["Theory", "Gaussian Processes", "Probability"]
draft: false
---

Continuing from the roadmap set out in [Road to Gaussian Processes]({% post_url 2021-04-17-road-to-gaussian-processes %}), we begin with the core ideas which underlie this Machine Learning Technique, the Gaussian distribution, specifically the **Multivariate Gaussian** distribution.

- Algebraic form of $$n$$-dimensional ellipsoid
- Projection as Change of Basis

## Algebraic form of $$n$$-dimensional ellipsoid

The standard form of an ellipsoid in $$\mathbb{R}^2$$ is:

$$
\frac{ {(x-x_0)}^2}{a^2} + \frac{ {(y-y_0)}^2}{b^2}=C
$$

Generally, for an $$n$$-dimensional ellipsoid, the standard form is:

$$
\sum_{i=1}^n \frac{ {(x_i-\mu_i)}^2}{\lambda_i^2}=C
$$

Let us denote $$X=\begin{bmatrix}x_1\\x_2\\ \vdots\\ x_n\end{bmatrix}$$, $$\mu=\begin{bmatrix}\mu_1\\\mu_2\\ \vdots\\ \mu_n\end{bmatrix}$$ and

$$D=\begin{bmatrix}
\lambda_1 && 0 && 0 && \cdots && 0 \\
0 && \lambda_2 && 0 && \cdots && 0 \\
0 && 0 && \lambda_3 && \cdots && 0 \\
\vdots && \vdots && \vdots && \ddots && \vdots \\
0 && 0 && 0 && \cdots && \lambda_n \\
\end{bmatrix}
$$

Then, we can see that:

$$D^{-1}=\begin{bmatrix}
\frac{1}{\lambda_1} && 0 && 0 && \cdots && 0 \\
0 && \frac{1}{\lambda_2} && 0 && \cdots && 0 \\
0 && 0 && \frac{1}{\lambda_3} && \cdots && 0 \\
\vdots && \vdots && \vdots && \ddots && \vdots \\
0 && 0 && 0 && \cdots && \frac{1}{\lambda_n} \\
\end{bmatrix}
$$

Then, we can rewrite the standard form of the $$n$$-dimensional ellipsoid as:

$$
{\|D^{-1}(X-\mu)\|}^2=C \\
$$

Alternatively, we can write:

$$
\begin{equation}
{[D^{-1}(X-\mu)]}^T [D^{-1}(X-\mu)]=C
\label{eq:algebraic_n_ellipsoid}
\end{equation}
$$

This can be easily verified by expanding out the terms. like so:

$$
D^{-1}(X-\mu)=\begin{bmatrix}
\frac{1}{\lambda_1} && 0 && 0 && \cdots && 0 \\
0 && \frac{1}{\lambda_2} && 0 && \cdots && 0 \\
0 && 0 && \frac{1}{\lambda_3} && \cdots && 0 \\
\vdots && \vdots && \vdots && \ddots && \vdots \\
0 && 0 && 0 && \cdots && \frac{1}{\lambda_n} \\
\end{bmatrix} \bullet
\begin{bmatrix}x_1-\mu_1\\x_2-\mu_2\\ \vdots\\ x_n-\mu_n\end{bmatrix}
=
\begin{bmatrix}\frac{x_1-\mu_1}{\lambda_1}\\\frac{x_2-\mu_2}{\lambda_2}\\ \vdots\\ \frac{x_n-\mu_n}{\lambda_n}\end{bmatrix}
$$

Then, expanding out $$\eqref{eq:algebraic_n_ellipsoid}$$, we get:

$$
{[D^{-1}(X-\mu)]}^T [D^{-1}(X-\mu)]=C \\
\Rightarrow \begin{bmatrix}\frac{x_1-\mu_1}{\lambda_1} && \frac{x_2-\mu_2}{\lambda_2} && \cdots && \frac{x_n-\mu_n}{\lambda_n}\end{bmatrix}
\bullet
\begin{bmatrix}\frac{x_1-\mu_1}{\lambda_1}\\\frac{x_2-\mu_2}{\lambda_2}\\ \vdots\\ \frac{x_n-\mu_n}{\lambda_n}\end{bmatrix}\\
= \sum_{i=1}^n \frac{ {(x_i-\mu_i)}^2}{\lambda_i^2}=C
$$

## Projection as Change of Basis

We have already discussed projections of vectors onto other vectors in several places (for example, in [Gram-Schmidt Orthogonalisation]({% post_url 2021-05-27-gram-scmidt-orthogonalisation %})). We can look at vector projection through a different lens, namely as a change in coordinate system.


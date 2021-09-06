---
title: "Gaussian Processes: Theory"
author: avishek
usemathjax: true
tags: ["Theory", "Gaussian Processes", "Probability", "Machine Learning"]
draft: false
---

Continuing from the roadmap set out in [Road to Gaussian Processes]({% post_url 2021-04-17-road-to-gaussian-processes %}), we begin with the geometry of the central object which underlies this Machine Learning Technique, the **Multivariate Gaussian Distribution**. We will study its form to build up some geometric intuition around its interpretation.

To do this, we will cover the material in two phases.

The second pass will delve into the mathematical underpinnings necessary to appreciate the technique more rigorously. Specifically, the following material will be covered:

- Schur Complements and Diagonalisation of Partitioned Matrices
- Conditioned Distributions as Gaussians
- Sampling from Multivariate Gaussian Distributions
- Generalising Discrete Covariance Matrices to Kernels


This pass will delve into the mathematical underpinnings necessary to appreciate the technique more rigorously.

## Bivariate Gaussian Distribution: Brute Force Derivation

Before moving to the generalised method of proof, I felt it instructive to look at the Bivariate Gaussian Distribution. Apart from the fact that the results for the mean and covariance applies to the more general case, it is helpful to see the mechanics of the exercise starting with the eigendecomposed form of the covariance matrix, before introducing Schur Complements for the more general case.

To that end, we define the Bivariate Gaussian Distribution as:

$$
\begin{equation}
P(x,y)=K_0\cdot \text{exp}\left( -\frac{1}{2} {(X-\mu)}^T\Sigma^{-1}(X-\mu)\right)
\label{eq:bivariate-joint-gaussian}
\end{equation}
$$

where $$X$$, $$\mu$$, and $$\Sigma$$ are defined as follows:

$$
X =
\begin{bmatrix}
x \\
y
\end{bmatrix} \\

\mu =
\begin{bmatrix}
\mu_x \\
\mu_y
\end{bmatrix} \\
$$

$$
\begin{equation}
\Sigma =
\begin{bmatrix}
\Sigma_{11} && \Sigma_{12} \\
\Sigma_{21} && \Sigma_{22} \\
\end{bmatrix}
\label{eq:covariance-matrix}
\end{equation}
$$

Note that $$\Sigma$$ is necessarily symmetric. Furthermore, the covariance matrix can be decomposed into orthonormal eigenvectors like so:

$$
\begin{equation}
\Sigma=
\begin{bmatrix}
a && -b \\
b && a \\
\end{bmatrix}
\cdot
\begin{bmatrix}
\lambda_1 && 0 \\
0 && \lambda_2 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
a && b \\
-b && a \\
\end{bmatrix}
\label{eq:covariance-eigendecomposition}
\end{equation}
$$

where $$\sqrt{a^2+b^2}=1$$.
Then, multiplying out the expression in $$\eqref{eq:covariance-eigendecomposition}$$ and equating the result to the terms in $$\eqref{eq:covariance-matrix}$$ gives us the following:

$$
\Sigma=
\begin{bmatrix}
\Sigma_{11} && \Sigma_{12} \\
\Sigma_{21} && \Sigma_{22} \\
\end{bmatrix}
=
\begin{bmatrix}
\lambda_1 a^2 + \lambda_2 b^2 && ab(\lambda_1-\lambda_2) \\
ab(\lambda_1-\lambda_2) && \lambda_1 b^2 + \lambda_2 a^2
\end{bmatrix} \\

\Sigma_{11}=\lambda_1 a^2 + \lambda_2 b^2 \\
\Sigma_{12}=\Sigma_{21}=ab(\lambda_1-\lambda_2) \\
\Sigma_{22}=\lambda_1 b^2 + \lambda_2 a^2
$$

Consequently, we can also write $$\Sigma^{-1}$$ as:

$$
\begin{equation}
\Sigma^{-1}=
\begin{bmatrix}
a && -b \\
b && a \\
\end{bmatrix}
\cdot
\begin{bmatrix}
\frac{1}{\lambda_1} && 0 \\
0 && \frac{1}{\lambda_2} \\
\end{bmatrix}
\cdot
\begin{bmatrix}
a && b \\
-b && a \\
\end{bmatrix}
\label{eq:inverse-covariance}
\end{equation}
$$

Let us condition on $$y$$, i.e., we set $$y=y_0$$, and derive the resulting one-dimensional probability distribution in $$x$$. The conditioning will then look like:

$$
\begin{equation}
P(x|y=y_0)=\frac{P(x,y)}{P(y=y_0)}
\label{eq:conditional-distribution}
\end{equation}
$$

We have already defined $$P(x,y)$$ in $$\eqref{eq:bivariate-joint-gaussian}$$. The other expression $$P(y=y_0)$$ then looks like this:

$$
\begin{equation}
P(y=y_0)=K_1\cdot \text{exp}\left(-\frac{1}{2}\cdot\frac{ {(y_0-\mu_y)}^2}{\Sigma_{22}}\right) \\
P(y=y_0)=K_1\cdot \text{exp}\left(-\frac{1}{2}\cdot\frac{ {(y_0-\mu_y)}^2}{\lambda_1 b^2 + \lambda_2 a^2}\right)
\label{eq:y-condition}
\end{equation}
$$

Putting $$\eqref{eq:y-condition}$$ and $$\eqref{eq:bivariate-joint-gaussian}$$ into $$\eqref{eq:conditional-distribution}$$, we get:

$$
\begin{equation}
P(x|y=y_0)=\frac{K_0\cdot \text{exp}\left( -\frac{1}{2} {(X-\mu)}^T\Sigma^{-1}(X-\mu)\right)}{K_1\cdot \text{exp}\left(-\frac{1}{2}\cdot\frac{ {(y_0-\mu_y)}^2}{\lambda_1 b^2 + \lambda_2 a^2}\right)} \\
=\frac{K_0}{K_1}\cdot\text{exp}\left[-\frac{1}{2}\left(
\begin{bmatrix}
x-\mu_x && y_0-\mu_y
\end{bmatrix}
\cdot
\begin{bmatrix}
a && -b \\
b && a \\
\end{bmatrix}
\cdot
\begin{bmatrix}
\frac{1}{\lambda_1} && 0 \\
0 && \frac{1}{\lambda_2} \\
\end{bmatrix}
\cdot
\begin{bmatrix}
a && b \\
-b && a \\
\end{bmatrix}
\cdot
\begin{bmatrix}
x-\mu_x \\
y_0-\mu_y
\end{bmatrix}

-

\frac{ {(y_0-\mu_y)}^2}{\lambda_1 b^2 + \lambda_2 a^2}
\right)
\right] \\

=\frac{K_0}{K_1}\cdot\text{exp}\left[-\frac{1}{2}\left(
\underbrace{
\begin{bmatrix}
x-\mu_x && y_0-\mu_y
\end{bmatrix}
\cdot
\begin{bmatrix}
\frac{a^2}{\lambda_1}+\frac{b^2}{\lambda_2} && ab(\frac{1}{\lambda_1}-\frac{1}{\lambda_2}) \\
ab(\frac{1}{\lambda_1}-\frac{1}{\lambda_2}) && \frac{b^2}{\lambda_1}+\frac{a^2}{\lambda_2} \\
\end{bmatrix}
\cdot
\begin{bmatrix}
x-\mu_x \\
y_0-\mu_y
\end{bmatrix}
}_A

-

\frac{ {(y_0-\mu_y)}^2}{\lambda_1 b^2 + \lambda_2 a^2}
\right)
\right]
\label{eq:conditional-distribution-derivation}
\end{equation}
$$


Let us simplify $$A$$, where:

$$
A = \begin{bmatrix}
x-\mu_x && y_0-\mu_y
\end{bmatrix}
\cdot
\begin{bmatrix}
\frac{a^2}{\lambda_1}+\frac{b^2}{\lambda_2} && ab(\frac{1}{\lambda_1}-\frac{1}{\lambda_2}) \\
ab(\frac{1}{\lambda_1}-\frac{1}{\lambda_2}) && \frac{b^2}{\lambda_1}+\frac{a^2}{\lambda_2} \\
\end{bmatrix}
\cdot
\begin{bmatrix}
x-\mu_x \\
y_0-\mu_y
\end{bmatrix}
$$

Multiplying everything out in $$A$$, we get:

$$
A = \frac{ {(x-\mu_x)}^2(\lambda_2 a^2 + \lambda_1 b^2) - 2(x-\mu_x)(y_0-\mu_y)ab(\lambda_1 - \lambda_2) + {(y_0-\mu_y)}^2 (\lambda_2 b^2 + \lambda_1 a^2)}{\lambda_1\lambda_2} \\
= \frac {\lambda_2 a^2 + \lambda_1 b^2}{\lambda_1\lambda_2}\left[ \underbrace{ {(x-\mu_x)}^2 - \frac{2(x-\mu_x)(y_0-\mu_y)ab(\lambda_1 - \lambda_2)}{\lambda_2 a^2 + \lambda_1 b^2} }_{\text{Complete the Square}} + \frac{ {(y_0-\mu_y)}^2 (\lambda_2 b^2 + \lambda_1 a^2)}{\lambda_2 a^2 + \lambda_1 b^2}\right]
$$

Completing the square above as indicated gives us:

$$
A = \frac {\lambda_2 a^2 + \lambda_1 b^2}{\lambda_1\lambda_2} \left[{\left(x-\mu_x-\frac{(y_0-\mu_y)ab(\lambda_1-\lambda_2)}{\lambda_2 a^2 + \lambda_1 b^2}\right)}^2 - \frac { {(y_0-\mu_y)}^2 a^2 b^2 {(\lambda_1-\lambda_2)}^2 }{ {(\lambda_2 a^2 + \lambda_1 b^2)}^2} + \frac{ {(y_0-\mu_y)}^2 (\lambda_2 b^2 + \lambda_1 a^2)}{\lambda_2 a^2 + \lambda_1 b^2}\right] \\
= \frac {\lambda_2 a^2 + \lambda_1 b^2}{\lambda_1\lambda_2} \left[{\left(x-\mu_x-\frac{(y_0-\mu_y)ab(\lambda_1-\lambda_2)}{\lambda_2 a^2 + \lambda_1 b^2}\right)}^2 + \frac { {(y_0-\mu_y)}^2 {(a^2+b^2)}^2 \lambda_1 \lambda_2 }{ {(\lambda_2 a^2 + \lambda_1 b^2)}^2} \right]
$$

However, remember that the $$\sqrt{a^2+b^2}=1$$, thus we get:

$$
A= \frac {\lambda_2 a^2 + \lambda_1 b^2}{\lambda_1\lambda_2} \left[{\left(x-\mu_x-\frac{(y_0-\mu_y)ab(\lambda_1-\lambda_2)}{\lambda_2 a^2 + \lambda_1 b^2}\right)}^2 + \frac { {(y_0-\mu_y)}^2 \lambda_1 \lambda_2 }{ {(\lambda_2 a^2 + \lambda_1 b^2)}^2} \right] \\
= \frac {\lambda_2 a^2 + \lambda_1 b^2}{\lambda_1\lambda_2} {\left(x-\mu_x-\frac{(y_0-\mu_y)ab(\lambda_1-\lambda_2)}{\lambda_2 a^2 + \lambda_1 b^2}\right)}^2 + \frac { {(y_0-\mu_y)}^2}{ \lambda_2 a^2 + \lambda_1 b^2}
$$

Substituting the value of $$A$$ back into $$\eqref{eq:conditional-distribution-derivation}$$, we get:

$$
\require{cancel}
P(x|y=y_0)=\frac{K_0}{K_1}\cdot \text{exp}\left( -\frac{1}{2} \frac {\lambda_2 a^2 + \lambda_1 b^2}{\lambda_1\lambda_2} {\left(x-\mu_x-\frac{(y_0-\mu_y)ab(\lambda_1-\lambda_2)}{\lambda_2 a^2 + \lambda_1 b^2}\right)}^2\right)\cdot\text{exp}\left(\cancel{\frac { {(y_0-\mu_y)}^2}{ \lambda_2 a^2 + \lambda_1 b^2}} - \cancel{\frac { {(y_0-\mu_y)}^2}{ \lambda_2 a^2 + \lambda_1 b^2}}\right) \\
=\frac{K_0}{K_1}\cdot \text{exp}\left( -\frac{1}{2} \frac {\lambda_2 a^2 + \lambda_1 b^2}{\lambda_1\lambda_2} {\left(x-\mu_x-\frac{(y_0-\mu_y)ab(\lambda_1-\lambda_2)}{\lambda_2 a^2 + \lambda_1 b^2}\right)}^2\right) \\
=\frac{K_0}{K_1}\cdot \text{exp}\left( -\frac{1}{2} \frac {\lambda_2 a^2 + \lambda_1 b^2}{\lambda_1\lambda_2} {\left(x-\mu_x-\frac{(y_0-\mu_y)\Sigma_{12}}{\Sigma_{22}}\right)}^2\right) \\
P(x|y=y_0)=\frac{K_0}{K_1}\cdot \text{exp}\left( -\frac{1}{2} {\left(\frac {\lambda_1\lambda_2}{\lambda_2 a^2 + \lambda_1 b^2}\right)}^{-1} {\left(x-\left[\mu_x+\frac{(y_0-\mu_y)\Sigma_{12}}{\Sigma_{22}}\right]\right)}^2\right)
$$

The final expression gives us a one-dimensional Gaussian with mean and covariance as: 

$$
\begin{equation}
\mu_{x|y=y_0}=\mu_x+\frac{(y_0-\mu_y)\Sigma_{12}}{\Sigma_{22}}
\label{eq:bivariate-gaussian-mean}
\end{equation}
$$

$$
\begin{equation}
\Sigma_{x|y=y_0}=\frac {\lambda_1\lambda_2}{\lambda_2 a^2 + \lambda_1 b^2}
\label{eq:bivariate-gaussian-covariance}
\end{equation}
$$

More generally, as we will see (and you can verify yourself, especially the covariance expression), the mean and variance of the conditional probability are:

$$
\begin{equation}
\mu_{x|y=y_0}=\mu_x+(y_0-\mu_y){\Sigma_{22}}^{-1}\Sigma_{12}
\label{eq:multivariate-gaussian-mean}
\end{equation}
$$

$$
\begin{equation}
\Sigma_{x|y=y_0}=\Sigma_{11}-\Sigma_{12}{\Sigma_{12}}^{-1}\Sigma_{21}
\label{eq:multivariate-gaussian-covariance}
\end{equation}
$$

You can verify that evaluation the expression $$\eqref{eq:multivariate-gaussian-covariance}$$ indeed yields the expression for the Conditional Covariance of the Bivariate Gaussian in $$\eqref{eq:bivariate-gaussian-covariance}$$.

As you will have also noticed, the derivation quickly becomes very complicated, and a general approach is needed to scale the proof to higher dimensions. This is the focus of the next topic.

## Schur Complements and Diagonalisation of Partitioned Matrices
## Conditioned Distributions as Gaussians
## Sampling from Multivariate Gaussian Distributions
## Generalising Discrete Covariance Matrices to Kernels

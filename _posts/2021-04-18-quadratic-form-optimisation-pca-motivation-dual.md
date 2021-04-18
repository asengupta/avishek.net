---
title: "Quadratic Form of Matrices, Principal Component Analysis as a Quadratic Optimisation Problem"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Quadratic Optimisation", "Linear Algebra", "Principal Components Analysis", "Optimisation", "Theory"]
---

This article presents the intuition behind the **Quadratic Form of a Matrix**, as well as its optimisation counterpart, **Quadratic Optimisation**. **Principal Components Analysis** is presented here, not in its own right, but as an application of these two concepts. PCA proper will be presented in another article where we will discuss **eigendecomposition**, **eigenvalues**, and **eigenvectors**.

One of the aims of this article is also to introduce PCA in a different context than eigenvectors, to show that there is more than one way to view a particular topic.

Parts of this should also serve as preliminaries for a couple of other topics:
- The **Quadratic Form** discussion will come in useful when talking of **Gaussian Processes**.
- The **Quadratic Optimisation** discussion will come in useful in deriving various results about **Support Vector Machines**.
- We will be better prepared for when I discuss **Principal Components Analysis** and **Singular Value Decomposition**, after this article.

## Preliminary: Computing the Dot Product of a set of vectors

This is not really a new concept, merely a reiteration of something that we will use as part of our optimisation-inspired derivation for Principal Components Analysis. We already know that the outer product of two matrices $$A$$ and $$B$$, let's call it $$C$$, has $$C_ik$$ as the dot product of the $$i$$th row of $$A$$ and the $$j$$th column of $$B$$.

Specifically **if we have a row vector $$A$$ and a column vector $$B$$, the normal vector outer product operation will yield the dot product $$A\cdot B$$**.

Generalising, instead of $$A$$ being a single row vector, let it be a set of row vectors, i.e., a matrix $$O\times F$$ (O=number of observations, F=number of variables, this will become relevant in a bit).

Then, the result of $$C=AB$$ (where $$B$$ is a $$F\times 1$$ vector), will give us a $$O\times 1$$ vector in which the $$i$$th entry will be the dot product of the $$i$$the row vector of $$A$$ and the vector $$B$$.


Geometrically, this gives us the projections of all the row vectors of $$A$$ onto $$B$$. This diagram explains the operation.

## Preliminary: Variance
Variance essentially measures the degree of spread of a set of data around a mean value. This is also the same variance that is used to characterise a Gaussian Distribution. Given a set of data $$x_i\in\mathbb{R}, i\in[1,N]$$ with a mean $$\mu$$, variance is the average squared distance from the mean, i.e.,

$$
\sigma^2=\frac{\sum_{i=1}^{N}{\|x_i-\mu\|}^2_2}{N}
$$

where the subscript 2 indicates the **L2 Norm** or **Euclidean Norm**, which is nothing other than our basic distance formula.
Let us simplify; we can always center the mean so that $$\mu=0$$, in which case, the above variance identity reduces to:

$$
\sigma^2=\frac{\sum_{i=1}^{N}{\|x_i\|}^2}{N}
$$

If $$x_i$$ are entries in a $$N\times 1$$ vector $$X$$, the above can be re-expressed as:

$$
\sigma^2=\frac{1}{N}\cdot X^TX
$$

## Principal Components Analysis as an Optimisation Problem
Let's look at PCA and how we can express it, starting from one of its definitions. The intuition behind PCA will be explained more when we get to eigendecomposition; for the moment, we restrict ourselves to a more mechanical interpretation of PCA.

### 1. The Setup
From the Wikipedia definition of PCA:

The first principal component can equivalently be defined as a direction that maximizes the variance of the projected data. The $$i$$th principal component can be taken as a direction orthogonal to the first $$i-1$$th principal components that maximizes the variance of the projected data.

Alright, there's a lot to unpack here, but I want to simplify and say very informally that **PCA involves finding vectors which maximise the variance of a data set, when those vectors are used as basis vectors**.

Consider a matrix $$X$$, with one observation (data point) per row, each column representing a particular feature of this observation. Let there be $$O$$ observations and $$F$$ features. Thus, $$X$$ is a $$O\times F$$ matrix.

**We'd like to project these data onto a vector such that the variance of these projected points onto that vector is maximised.** There might be multiple vectors like this, but for the moment, let's stick with finding one of them.

This picture explains the idea. As you can see, we can pick any random vector, and project our data onto it. These projections are scaled versions of this vector. We'd like the spread of these projections to be as dispersed as possible. For example $$V_1$$ doesn't seem to provide much spread (variance) for our data, whereas $$V_2$$ provides a much larger variance.

### 2. Variance of Projections

$$X$$ is a $$O\times F$$ matrix, $$V$$ is a $$F\times 1$$ vector, so the projection is a $$O\times 1$$ vector, the $$i$$th entry corresponding to the projection (a single number) of the $$i$$th row vector in $$X$$ onto $$V$$.

$$
W=XV
$$

Variance is thus:

$$
W^TW={(XV)}^TXV \\
\Rightarrow W^TW=V^TX^TXV
\Rightarrow W^TW=V^T\Sigma V
$$

We ignore the division by $$N$$, because that is merely a scaling factor which will not really affect framing our optimisation problem.
It is important to note that **$$\Sigma$$ is a symmetric matrix**, because it is a product of a matrix and its transpose.

### 2.1 Aside: The product of a matrix and its transpose is symmetric - Proof

$$A$$ is an $$M\times N$$ matrix. Consider the product of $$A^T$$ and $$A$$:

$$
C_{ik}=\sum_{j=1}^N A^T_{ij}A_{jk} \\
C_{ki}=\sum_{j=1}^N A^T_{kj}A_{ji}
$$

By definition of transpose, we have:
$$
A^T_{ij}=A_{ji} \\
A^T_{kj}=A_{jk}
$$

Substituting the above in the first identity above, we get:

$$
C_{ik}=\sum_{j=1}^N A_{ji}A^T_{kj} \\
\Rightarrow C_{ik}=\sum_{j=1}^N A^T_{kj}A_{ji}=C_{ki} \\
$$

Thus, $$C$$ is symmetric.

### 3. Maximise Variance
We thus need to find a vector $$V$$ which maximises the expression $$V^TX^TXV$$. We express this as:

**Maximise $$V^T\Sigma V$$ \\
Subject to: $$V^TV=1$$**

We will have more to say about the constraint $$V^TV=1$$ in a bit, when we elaborate on the Quadratic Form of a matrix, but for the moment, understand that the definition of Principal Components Analysis is about finding a set of vectors which maximise the variance of a data set when those vectors are used as the basis.

Of course, there is a lot of be said about eigendecomposition, but take particular note of the form of the cost function. Very generally, it is $$xQx^T$$ or $$x^TQx$$ (depending upon how you have structured your data). This form arises quite often in general optimisation problems, and obviously, is very important in Linear Algebra. This belongs to the class of optimisation called **Quadratic Optimisation**.

We will explore optimisation in a little bit, but let us focus on the **Quadratic Form of a Matrix** first.

## Quadratic Form of a Matrix

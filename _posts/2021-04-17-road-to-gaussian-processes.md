---
title: "Road to Gaussian Processes"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Gaussian Processes", "Theory"]
---

This article aims to start the road towards a theoretical intuition behind **Gaussian Processes**, another Machine Learning technique based on **Bayes' Rule**. However, there is a raft of material that I needed to understand and relearn before fully appreciating some of the underpinnings of this technique.

I'd like to do some high level dives into some of the topics I believe will help practictioners go a little deeper than "it's just a Gaussian Process of many variables".

## Theory Track for Gaussian Processes
The map below shows the rough order in which the preliminary material will be presented.

{% mermaid %}
graph TD;
quad[Quadratic Form of Matrix]-->chol[Cholesky Factorisation];
tri[Triangular Matrices]-->chol[Cholesky Factorisation];
det[Determinants]-->chol[Cholesky Factorisation];
jac[Jacobian]-->jaclin[Jacobian of Linear Transformations]
cov[Covariance Matrix]-->mvn[Multivariate Gaussian]
chol[Cholesky Factorisation]-->mvn[Multivariate Gaussian]
mvn[Multivariate Gaussian]-->mvnlin[MVN as Linearly Transformed Sums of Uncorrelated Random Variables]
crv[Change of Random Variable]-->mvnlin[MVN as Linearly Transformed Sums of Uncorrelated Random Variables]
jaclin[Jacobian of Linear Transformations]-->mvnlin[MVN as Linearly Transformed Sums of Uncorrelated Random Variables]
diffeq[Difference Equations]-->diffmat[Difference Matrix]-->gp[Gaussian Processes]
mvnlin[MVN as Linearly Transformed Sums of Uncorrelated Random Variables]-->Conditioning
mvnlin[MVN as Linearly Transformed Sums of Uncorrelated Random Variables]-->Marginalisation
Conditioning-->gp[Gaussian Processes]
Marginalisation-->gp[Gaussian Processes]
style chol fill:#006f00,stroke:#000,stroke-width:2px,color:#fff
style mvn fill:#006fff,stroke:#000,stroke-width:2px,color:#fff
style gp fill:#8f0f00,stroke:#000,stroke-width:2px,color:#fff
{% endmermaid %}

Let's survey the topics and their relevance quickly in this article.

- To understand the composite nature of **Multivariate Gaussian Distributions**, you'll need to be able to see it as a combination of simple, uncorrelated, Gaussian distributions of one random variable. This requires expressing our original random variables as a different set of random variables. This will necessitate understanding how **Changes of N-dimensional Random Variables** are expressed, which in turn require a basic understanding of **Jacobians**.
-To recast the resulting expression into a familiar Gaussian form, we will need to express the **Covariance Matrix** as a Cholesky-decomposed product. **Cholesky Factorisation** works on **positive definite symmetric matrices**, thus an understanding of the **Quadratic Forms of Matrices** is also needed.
- To actually use a **Gaussian Process** for prediction, it will need to be both **marginalised** and **conditioned** to extract out probability information about the prediction. The formulation of how the mean and variance of an MVN looks like under these operations will require making an assumption about the prior in terms of its neighbouring points. Hence, the **Difference Matrix** formulation.


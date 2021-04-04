---
title: "Linear Regression: Assumptions and Results using the Maximum Likelihood Estimator"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Linear Regression", "Maximum Likelihod Estimator", "Theory", "Probability"]
---

It is instructive 

$$
P(x_1)=\frac{1}{\sqrt{2\pi\sigma^2}}.e^{-\frac{ {(x-\mu)}^2}{2\sigma^2}} \\
P(X)=P(x_1)P(x_2)...P(x_n) \\
P(x)=\prod_{i=1}^{N}P(x_i)
$$

Taking the log on both sides (base e), we get:

$$
log_e P(x)=\sum_{i=1}^{N}log_e P(x_i) \\ 
log_e P(x)=\sum_{i=1}^{N}log_e \frac{1}{\sqrt{2\pi\sigma^2}}.e^{-\frac{ {(x_i-\mu)}^2}{2\sigma^2}} \\
log_e P(x)=\sum_{i=1}^{N}log_e \frac{1}{\sqrt{2\pi\sigma^2}} + \sum_{i=1}^{N}log_e e^{-\frac{ {(x_i-\mu)}^2}{2\sigma^2}} \\
log_e P(x)=-\frac{1}{2}\sum_{i=1}^{N}log_e 2\pi\sigma^2 + \sum_{i=1}^{N}log_e e^{-\frac{ {(x_i-\mu)}^2}{2\sigma^2}} \\
log_e P(x)=-\frac{1}{2}\sum_{i=1}^{N}log_e 2\pi -\frac{1}{2}\sum_{i=1}^{N}log_e \sigma^2 + \sum_{i=1}^{N}log_e e^{-\frac{ {(x_i-\mu)}^2}{2\sigma^2}} \\
$$
Dropping the first term on the right side, since it is a constant, we get:

$$
log_e P(x)\propto -\frac{1}{2}\sum_{i=1}^{N}\frac{ {(x_i-\mu)}^2}{\sigma^2} -\sum_{i=1}^{N}log_e \sigma \\
L(x)=-\frac{1}{2}\sum_{i=1}^{N}\frac{ {(x_i-\mu)}^2}{\sigma^2} -N.log_e \sigma \\
$$

Thus, our problem of finding the best values for $$\mu$$ and $$\sigma$$ boils down to maximising the above expression $$L(x)$$.

Since this is an equation in two variables, let's take the partial differential with respect to each variable, while treating the other as a constant.

### Derivation of the mean $$\mu$$

$$
\frac{\partial {L(x)}}{\partial\mu}=\frac{1}{\sigma^2}\sum_{i=1}^{N}(x_i-\mu)
$$

Setting this partial derivative to 0, we get:

$$
\frac{1}{\sigma^2}\sum_{i=1}^{N}(x_i-\mu)=0 \\
\sum_{i=1}^{N}(x_i-\mu)=0 \\
\sum_{i=1}^{N}x_i- N\mu=0 \\
\mu=\frac{1}{N}\sum_{i=1}^{N}x_i
$$

The above is the definition of the arithmetical mean, essentially the average value of all the observations.

### Derivation of the variance $$\sigma$$

$$
\frac{\partial {L(x)}}{\partial\sigma}=\frac{1}{\sigma^3}\sum_{i=1}^{N}{(x_i-\mu)}^2 -\frac{N}{\sigma} \\
\frac{\partial {L(x)}}{\partial\sigma}=\frac{1}{\sigma}\left(\frac{1}{\sigma^2}\sum_{i=1}^{N}{(x_i-\mu)}^2 -N\right) \\
$$

Setting this partial derivative to 0, we get:

$$
\frac{1}{\sigma^2}\sum_{i=1}^{N}{(x_i-\mu)}^2 -N=0 \\
N\sigma^2=\sum_{i=1}^{N}{(x_i-\mu)}^2 \\
\sigma^2=\frac{1}{N}\sum_{i=1}^{N}{(x_i-\mu)}^2
$$

The above is the definition of variance of a Gaussian distribution.

---
title: "A Quick Note on Proving the Triangle Inequality on a Derived Distance Metric using Monotonicity"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics", "Kreyszig"]
draft: false
---

This is a quick note on proving the **Triangle Inequality** criterion of the following claim:

**If $$d(x,y)$$ is a distance metric, then $$\bar{d}(x,y)=\frac{d(x,y)}{1+d(x,y)}$$ is also a valid distance metric.**

The four criteria for satisfying a distance metric are:

- **(M1)** $$0 \leq d(x,y)<\infty, d(x,y)\in \mathbb{R}$$
- **(M2)** $$d(x,y)=0$$ if and only if $$x=y$$
- **(M3)** $$d(x,y)=d(y,x)$$
- **(M4)** $$d(x,z) \leq d(x,y) + d(y,z)$$

**(M1)** to **(M3)** follow quite readily. Let us look at proving **(M4)**.

Observing the form of $$\bar{d}(x,y)$$, let us assume the function $$f(t)=\displaystyle\frac{t}{1+t}$$. Differentiating with respect to $$t$$, we get:

$$
\frac{df(t)}{dt}=\frac{1}{1+t} - \frac{t}{ {(1+t)}^2} \\
= \frac{1}{ {(1+t)}^2}
$$

This shows that $$f(t)$$ is monotonically increasing. A function $$f(x)$$ is monotonically increasing if for $$x_1 \leq x_2$$, we have $$f(x_1) \leq f(x_2)$$. Since our $$f(t)$$ is monotonically increasing, we can write that for $$t_1 \leq t_2$$:

$$
\begin{equation}
f(t_1) \leq f(t_2) \\
\Rightarrow \frac{t_1}{1+t_1} \leq \frac{t_2}{1+t_2}
\label{eq:1}
\end{equation}
$$

Set $$t_1=d(x,y)$$ and $$t_2=d(x,z) + d(z,y)$$. We can immediately see that $$t_1 \leq t_2$$. Thus substituting these values into $$\eqref{eq:1}$$, we get:

$$
\begin{equation}
\frac{d(x,y)}{1+d(x,y)} \leq \frac{d(x,z) + d(z,y)}{1+d(x,z) + d(z,y)} \\
= \frac{d(x,z)}{1+d(x,z) + d(z,y)} + \frac{d(z,y)}{1+d(x,z) + d(z,y)}
\label{eq:2}
\end{equation}
$$

We see that:

$$
\displaystyle\frac{d(x,z)}{1+d(x,z) + d(z,y)} \leq \frac{d(x,z)}{1+d(x,z)} \\
\displaystyle\frac{d(z,y)}{1+d(x,z) + d(z,y)} \leq \frac{d(z,y)}{1+d(z,y)}
$$

Thus we can rewrite the above inequality in $$\eqref{eq:2}$$ as:

$$
\frac{d(x,y)}{1+d(x,y)} \leq \frac{d(x,z)}{1+d(x,z)} + \frac{d(z,y)}{1+d(z,y)} \\
\Rightarrow \bar{d(x,y)} \leq \bar{d}(x,z) + \bar{d}(z,y)
$$

thus, proving the **Triangle Inequality**.

This is the central idea in proving that the distance metric in a sequence space of all bounded and unbounded complex numbers **(Kreyszig 1.2-1)** has a metric defined by:

$$
d(x,y)=\displaystyle\sum_{j=1}^\infty \frac{1}{2^j} \frac{\vert \eta_j - \theta_j\vert}{1 + \vert \eta_j - \theta_j\vert}
$$

where $$x=(\eta_j)$$ and $$y=(\theta_j)$$.

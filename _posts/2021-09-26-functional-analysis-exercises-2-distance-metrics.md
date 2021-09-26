---
title: "Functional Analysis Exercises 2 : Distance Metrics"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics"]
draft: false
---

This post lists solutions to many of the exercises in the **Distance Metrics section 1.2** of *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**. This also includes proofs of **Hölder's Inequality** and **Minkowski's Inequality**.

#### 1.2.1. Show that in 1.2-1 we can obtain another metric by replacing $$\frac{1}{2^i}$$ with $$\mu_i>0$$ such that $$\sum\mu_i$$ converges.

#### 1.2.2. Using (6), show that the geometric mean of two positive numbers does not exceed the arithmetic mean.


#### 1.2.3. Show that the Cauchy-Schwarz inequality (11) implies
  $${(|\xi| + \cdots + |\xi|)}^2 \leq n ({|\xi|}^2 + \cdots + {|\xi|}^2)$$.
#### 1.2.4. (Space $$\ell^p$$) Find a sequence which converges to 0, but is not in any space $$\ell^p$$, where $$1\leq p<+\infty$$.
#### 1.2.5. Find a sequence $$x$$ which is in $$\ell^p$$ with p>1 but $$\require{cancel} x\cancel{\in}\ell^1$$.
#### 1.2.6. **(Diameter, bounded set)** The diameter $$\delta(A)$$ of a nonempty set A in a  metric space $$(X, d)$$ is defined to be $$\delta(A) = \text{sup} d(x,y)$$. A is said to be bounded if $$\delta(A)<\infty$$. Show that $$A\subset B$$ implies $$\delta(A)\leq \delta(B)$$.
#### 1.2.7. Show that $$\delta(A)=0$$ *(cf. Prob. 6)* if and only if A consists of a single point.
#### 1.2.8. **(Distance between sets)** The distance $$D(A,B)$$ between two nonempty subsets $$A$$ and $$B$$ of a metric space $$(X, d)$$ is defined to be:

$$D(A,B) = \text{inf } d(a, b)$$.

#### Show that $$D$$ does not define a metric on the power set of $$X$$. (For this reason we use another symbol, $$D$$, but one that still reminds us of $$d$$.)

#### 1.2.9. If An $$B \cap P$$, show that $$D(A,B) = 0$$ in Prob. 8. What about the converse?

#### 1.2.10. The distance $$D(x,B)$$ from a point $$x$$ to a non-empty subset $$B$$ of $$(X,d)$$ is defined to be

$$D(x,B)= \text{inf } d(x, b)$$

#### in agreement with Prob. 8. Show that for any $$x,y\in X$$,

$$
|D(x,B) - D(y,B)| \leq d(x, y)
$$.

#### 1.2.11. If $$(X,d)$$ is any metric space, show that another metric on $$X$$ is defined by

$$
\bar{d}(x,y)=\frac{d(x,y)}{1+d(x,y)}
$$

#### and $$X$$ is bounded in the metric $$\bar{d}$$.

#### 1.2.12. Show that the union of two bounded sets A and B in a metric space is a bounded set. (Definition in Prob. 6.)

#### 1.2.13. **(Product of metric spaces)** The Cartesian product $$X = X_1 \times X_2$$ of two    metric spaces $$(X_1,d_1)$$ and $$(X_2,d_2)$$ can be made into a metric space $$(X,d)$$ in many ways. For instance, show that a metric $$d$$ is defined by

$$
\bar{d}(x,y)=d_1(x_1,y_1) + d_1(x_2,y_2)
$$

#### where $$x=(x_1,x_2)$$, $$y=(y_1,y_2)$$.

#### 1.2.14. Show that another metric on $$X$$ in Prob. 13 is defined by

$$
\bar{d}(x,y)=\sqrt{ {d_1(x_1,y_1)}^2 + {d_1(x_2,y_2)}^2}
$$

#### 1.2.15. Show that a third metric on $$X$$ in Prob. 13 is defined by

$$
\bar{d}(x,y)=max[d_1(x_1,y_1), d_1(x_2,y_2)]
$$

## Young's Inequality

We begin with the idea of conjugate exponents, which we call $$p$$ and $$q$$, related like so:

$$
\frac{1}{p} + \frac{1}{q} = 1 \\
\Rightarrow q + p = pq \\
\Rightarrow q + p = pq \\
\Rightarrow  p(q - 1)= q \\
\Rightarrow  p=\frac{q}{q-1} \\
\Rightarrow  1-p=1-\frac{q}{q-1} \\
\Rightarrow  1-p=\frac{q-1-q}{q-1}
$$

$$
\begin{equation}
\Rightarrow  1-p=\frac{1}{1-q}
\label{eq:conjugate-exponents-property-1}
\end{equation}
$$

where $$p>1$$. We now use these exponents in the following identity:

$$
u=t^{p-1}
$$

We also record the following implication.

$$
t=u^{\frac{1}{p-1}} \\
\Rightarrow t=u^{q-1}
$$

For $$p=2$$, we get $$u=t$$, which is the equation of the straight line $$x=y$$.
The above follows from $$\eqref{eq:conjugate-exponents-property-1}$$.

Let $$\alpha, \beta >0: \alpha, \beta \in \mathbb{R}$$, then in $$\mathbb{R}^2$$, $$\alpha\beta$$ describes the area of a rectangle. Let us plot the graph of $$u=t^{p-1}$$ for $$p=2$$, so that $$u=t$$.

If we integrate $$\displaystyle\int\limits_0^\alpha u.dt=\int\limits_0^\alpha t^{p-1}.dt$$, and $$\displaystyle\int\limits_0^\alpha t.du=\int\limits_0^\alpha u^{q-1}.du$$, we can compute an area like this:

$$
\begin{align}
S &= \int\limits_0^\alpha t^{p-1}.dt + \int\limits_0^\alpha u^{q-1}.du \\
&= \frac{\alpha^2}{2} + \frac{\beta^2}{2}
\end{align}
$$

![Linear Conjugate Exponents](/assets/images/linear-conjugate-exponents.png)

You will notice that regardless of the choice of $$\alpha, \beta$$, there will always be a small portion of $$S$$ which is bigger than $$\alpha\beta$$. The only situation in which there is no extra area is when $$\alpha=\beta$$. Therefore, we can say that:

$$
\alpha \beta \leq \frac{\alpha^2}{2} + \frac{\beta^2}{2}
$$

This less-than-or-equal relation carries over to other values of $$p$$ where $$u=t^{p-1}$$ is an exponential graph. The following two graphs illustrate how this is always true.

![Nonlinear Conjugate Exponents - Alpha Larger](/assets/images/nonlinear-conjugate-exponents-alpha-larger.png)

![Nonlinear Conjugate Exponents - Beta Larger](/assets/images/nonlinear-conjugate-exponents-beta-larger.png)

Thus, we can conclude that:

$$
\begin{equation}
\alpha \beta \leq \frac{\alpha^p}{p} + \frac{\beta^q}{q}
\label{eq:youngs-inequality}
\end{equation}
$$

$$\eqref{eq:youngs-inequality}$$ is called **Young's Equality**, and we will use it to prove **Hölder's Inequality** next.

## Hölder's Inequality

We look at $$\ell^p$$ spaces now. Briefly recapping, $$\ell^p$$ spaces are spaces of sequences. A sequence $$\xi=\{\xi_1, \xi_2, \cdots\} \in\ell^p$$ must satisfy the following condition:

$$
\sum\limits_{i=1}^\infty{|\xi_i|}^p<\infty
$$

The norm $$\|\bullet\|$$ in $$\ell^p$$ spaces is usually defined as:

$$
\|\bullet\|={\left(\sum\limits_{i=1}^\infty{|\xi_i|}^p\right)}^{\frac{1}{p}}
$$

which induces the distance metric between two sequences $$\xi, \eta \in \ell_p$$:

$$
d(\xi,\eta)={\left(\sum\limits_{i=1}^\infty{|\xi_i-\eta_i|}^p\right)}^{\frac{1}{p}}
$$

Pick any $$\xi,\eta\in\ell^p$$.

Let us pick any two corresponding terms in $$\xi$$ and $$\eta$$, and let $$\alpha=\vert\xi_i\vert$$ and $$\beta_i=\vert\eta\vert$$, since we cannot guarantee these terms will be positive.

Then, from **Young's Inequality** $$\eqref{eq:youngs-inequality}$$, we get:


$$
|\xi_i||\eta_i| \leq \frac{ {|\xi_i|}^p}{p} + \frac{ {|\eta_i|}^q}{q} \\
\Rightarrow |\xi_i\eta_i| \leq \frac{ {|\xi_i|}^p}{p} + \frac{ {|\eta_i|}^q}{q}
$$

Summing over all $$i$$, we get:

$$
\begin{equation}
\displaystyle\sum\limits_{i=1}^\infty|\xi_i\eta_i| \leq \frac{\sum\limits_{i=1}^\infty{|\xi_i|}^p}{p} + \frac{\sum\limits_{i=1}^\infty{|\eta_i|}^q}{q}
\label{eq:holders-youngs-inequality-application}
\end{equation}
$$

What we'd like to do is prove that: $$\displaystyle\sum\limits_{i=1}^\infty\vert\xi_i\eta_i\vert \leq 1$$. The only identity immediately available is the conjugate exponent identity, namely:

$$
\frac{1}{p} + \frac{1}{q} = 1
$$

In order to be able to set the RHS of $$\eqref{eq:holders-youngs-inequality-application}$$, we need the following condition:

$$
\begin{equation}
\displaystyle\sum\limits_{i=1}^\infty{ {\vert\xi_i\vert}^p}=\displaystyle\sum_{i=1}^\infty{ {\vert\eta_i\vert}^q}=1
\label{eq:holders-unity-condition}
\end{equation}
$$

Let us then assume $$\eqref{eq:holders-unity-condition}$$. Then, $$\eqref{eq:holders-youngs-inequality-application}$$ becomes:

$$
\begin{equation}
\displaystyle\sum\limits_{i=1}^\infty|\xi_i\eta_i| \leq 1
\label{eq:holders-less-than-unity-condition}
\end{equation}
$$

We need to determine wha sort of $$\xi$$ and $$\eta$$ can satisfy this condition. Let's take $$\xi$$ as an example. We have:

$$
\xi=(\xi_1, \xi_2, \cdots)
$$

Remember the norm for $$\ell^p$$ spaces? Here it is again:

$$
\|\bullet\|={\left(\sum\limits_{i=1}^\infty{|\xi_i|}^p\right)}^{\frac{1}{p}}
$$

If we divide each term in $$\xi$$ by its norm:

$$
\begin{equation}
\xi_i = \frac{\bar\xi_i}{ {\left(\sum\limits_{i=1}^\infty{|\bar{\xi_i}|}^p\right)}^{\frac{1}{p}}}
\label{eq:holders-sequence-scale-factor}
\end{equation}
$$

Then, from $$\eqref{eq:holders-unity-condition}$$, we get:

$$
\displaystyle\sum\limits_{i=1}^\infty{ {|\xi_i|}^p}=\frac{1}{ \sum\limits_{i=1}^\infty{|\bar{\xi_i}|}^p} \left({|\bar{\xi_1}|}^p + {|\bar{\xi_2}|}^p + \cdots\right) \\
=\frac{\sum\limits_{i=1}^\infty{|\bar{\xi_i}|}^p}{\sum\limits_{i=1}^\infty{|\bar{\xi_i}|}^p}
= 1
$$

which satisfies condition $$\eqref{eq:holders-unity-condition}$$, regardless of which sequence $$\bar{\xi}$$ we choose from $$\ell^p$$.

Applying the same logic to $$\eta_i$$, and substituting $$\eqref{eq:holders-sequence-scale-factor}$$ into $$\eqref{eq:holders-less-than-unity-condition}$$, we get:

$$
\displaystyle\sum\limits_{i=1}^\infty |\bar\xi_i \bar\eta_i| \leq {\left(\displaystyle\sum\limits_{i=1}^\infty{ {|\bar\xi_i|}^p}\right)}^{\frac{1}{p}} \bullet {\left(\displaystyle\sum\limits_{i=1}^\infty{ {|\bar\eta_i|}^q}\right)}^{\frac{1}{q}}
$$

Removing the overbars from $$\bar\xi_i$$ $$\bar\eta_i$$ to indicate any two sequences in an $$\ell^p$$ space, we get:

$$
\begin{equation}
\displaystyle\sum\limits_{i=1}^\infty |\xi_i \eta_i| \leq {\left(\displaystyle\sum\limits_{i=1}^\infty{ {|\xi_i|}^p}\right)}^{\frac{1}{p}} \bullet {\left(\displaystyle\sum\limits_{i=1}^\infty{ {|\eta_i|}^q}\right)}^{\frac{1}{q}}
\label{eq:holders-inequality}
\end{equation}
$$

The result $$\eqref{eq:holders-inequality}$$ is referred to as **Hölder's Inequality**. This will be used to prove **Minkowski's Inequality** next.

## Minkowski's Inequality


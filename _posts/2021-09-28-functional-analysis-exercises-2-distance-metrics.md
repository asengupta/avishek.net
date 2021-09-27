---
title: "Functional Analysis Exercises 2 : Distance Metrics"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics"]
draft: false
---

This post lists solutions to many of the exercises in the **Distance Metrics section 1.2** of *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**.

#### 1.2.1. Show that in 1.2-1 we can obtain another metric by replacing $$\frac{1}{2^i}$$ with $$\mu_i>0$$ such that $$\sum\mu_i$$ converges.

**Proof**: The space referred to is the space of all bounded and unbounded sequences.

The candidate metric is defined as:

$$
d(x,y)=\displaystyle\sum_{i=1}^\infty \mu_i\frac{|x_i-y_i|}{1+|x_i-y_i|}
$$

**(M1)** $$d(x,y)$$ is bounded, non-negative, and real.

We know that $$\displaystyle\sum_{i=1}^\infty \mu_i$$ converges, thus if we prove $$\lambda_i \mu_i<mu_i$$, we will have proved that $$\displaystyle\sum_{i=1}^\infty \lambda_i \mu_i$$ converges, and is thus real, bounded.

Indeed, if we examine $$d(x,y)$$, we can make the following observation:

$$
d(x,y)=\displaystyle\sum_{i=1}^\infty \mu_i\underbrace{\frac{|x_i-y_i|}{1+|x_i-y_i|}}_{0\leq\lambda<1}
$$

Thus, $$d(x,y)$$ is bounded and real. $$d(x,y)$$ is also nonnegative because $$0\leq\lambda<1$$ and $$\mu_i>0$$.

**(M2)** This is evident since:

$$
d(x,x)=\displaystyle\sum_{i=1}^\infty \mu_i\frac{|x_i-x_i|}{1+|x_i-x_i|}=0
$$

**(M3)** This is easily seen since the modulus sign guarantees that:$$\vert x_i-y_i\vert=\vert y_i-x_i \vert$$, and thus $$d(x,y)=d(y,x)$$.

**(M4)**

For convenience of notation, let us denote use the following notation:

$$
A=|x_i-y_i| \\
B=|y_i-z_i| \\
C=|z_i-x_i|
$$

We'd like to prove that:

$$
\require{cancel}
\frac{A}{1+A} \leq \frac{B}{1+B} + \frac{C}{1+C} \\
= \frac{B+C+2BC}{(1+B)(1+C)} \\
\Rightarrow A(1+B)(1+C) \leq (B+C+2BC)(1+A) \\
\Rightarrow A+\cancel{CA}+\cancel{AB}+\cancel{ABC} \leq B+C+2BC+\cancel{AB}+\cancel{CA}+\cancel{2}ABC \\
\Rightarrow A \leq B+C+2BC+ABC \\
\Rightarrow |x_i-y_i| \leq |x_i-z_i|+|z_i-y_i|+2BC+ABC
$$

Thus, we need to prove that:

$$
|x_i-y_i| \leq |x_i-z_i|+|z_i-y_i|+2BC+ABC
$$

where $$A,B,C \geq 0$$.

We already know from the **Triangle Inequality** that:

$$
\begin{align*}
|x_i-y_i| &= |x_i-z_i+z_i-y_i| \\
|x_i-y_i| &\leq |x_i-z_i|+|z_i-y_i| \\
\Rightarrow |x_i-y_i| &\leq |x_i-z_i|+|z_i-y_i|+2BC+ABC
\end{align*}
$$

Thus $$d(x,y)$$ is a metric.

$$\blacksquare$$

#### 1.2.2. Using (6), show that the geometric mean of two positive numbers does not exceed the arithmetic mean.

**Proof**:
From the identity involving conjugate exponents, we know that:

$$
\alpha \beta \leq \frac{\alpha^p}{p} + \frac{\beta^q}{q} \\
\Rightarrow 2\alpha \beta \leq \frac{\alpha^p}{p} + \frac{\beta^q}{q} + \alpha \beta
$$

Set $$p=2$$, then we get $$q=2$$, so that we get:

$$
2\alpha \beta \leq \frac{\alpha^2}{2} + \frac{\beta^2}{2} + \alpha \beta \\
\Rightarrow 4\alpha \beta \leq \alpha^2 + \beta^2 + 2\alpha \beta \\
\Rightarrow \alpha \beta \leq {\left(\frac{\alpha + \beta}{2}\right)}^2 \\
\Rightarrow \sqrt{\alpha \beta} \leq \frac{\alpha + \beta}{2}
$$

This proves that the Geometric Mean of two numbers cannot exceed their Arithmetic Mean.

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

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

## Hölder's Inequality

## Minkowski's Inequality


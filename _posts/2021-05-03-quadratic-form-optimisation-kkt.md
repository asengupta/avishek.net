---
title: "Quadratic Optimisation, Lagrangian Dual, and the Karush-Kuhn-Tucker Conditions"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Quadratic Optimisation", "Linear Algebra", "Optimisation", "Theory"]
---

This article continues the original discussion on **Quadratic Optimisation**, where we considered **Principal Components Analysis** as a motivation. Here, we extend the **Lagrangian Multipliers** approach, which in its current form, admits only equality constraints. We will extend it to allow constraints which can be expressed as inequalities.

Much of this discussion applies to the general class of **Convex Optimisation**; however, I will be constraining the form of the problem slightly to simplify discussion. We will need to develop some basic mathematical results in order to fully appreciate the implications of the **Karush-Kuhn-Tucker Theorem** (also called the **Saddle Point Theorem**).

**Convex Optimisation** solves problems framed using the following standard form:

Minimise (with respect to $$x$$), $$\mathbf{f(x)}$$

subject to:

$$\mathbf{g_i(x)\leq 0, i=1,...,n}$$ \\
$$\mathbf{h_i(x)=0, i=1,...,m}$$

where:

- $$\mathbf{f(x)}$$ is a **convex** function 
- $$\mathbf{g_i(x)}$$ are **convex** functions
- $$\mathbf{h_i(x)}$$ are **affine** functions.

For **Quadratic Optimisation**, the extra constraint that is imposed is: $$g_i(x)$$ is are also affine functions. Therefore, all of our constraints are essentially linear.

Furthermore, for this discussion, I omit $$h_i(x)$$ for clarity; any **equality constraints can always be converted into inequality constraints**, and become part of $$g_i(x)$$.

Thus, this is the reframing of the **Quadratic Optimisation** problem for the purposes of this discussion.

Minimise (with respect to $$x$$), $$\mathbf{f(x)}$$

subject to: $$\mathbf{g_i(x)\leq 0, i=1,...,n}$$

where:

- $$\mathbf{f(x)}$$ is a **convex function**
- $$\mathbf{g_i(x)}$$ are **affine functions**

## Preliminary: Affine Sets
Take any two vectors $$\vec{v_1}$$ and $$\vec{v_2}$$. All the vectors (or points, if you so prefer) along the line joining the tips of $$\vec{v_1}$$ and $$\vec{v_2}$$ obviously lie on a straight line. Thus, we can represent any vector along this line segment as:

$$
\vec{v}=\vec{v_1}+\theta(\vec{v_1}-\vec{v_2}) \\
=\theta \vec{v_1}+(1-\theta)\vec{v_2}
$$

We say that all these vectors (including $$\vec{v_1}$$ and $$\vec{v_2}$$) form an **affine set**. More generally, a vector is a member of an affine set if it satisfies the following definition.

$$
\vec{v}=\theta_{1} \vec{v_1}+\theta_{2} \vec{v_2}+...+\theta_{n} \vec{v_n} \\
\theta_1+\theta_2+...+\theta_n=1
$$

In words, a vector is an **affine combination** of $$n$$ vectors if the **coefficients of the linear combinations of those vectors sum to one**.

## Preliminary: Convex and Concave Functions
The layman's explanation of a convex function is that it is a bowl-shaped function. However, let us state this mathematically: we say a function is convex, if the graph of that function lies below every point on a line connecting any two points on that function.

If $$(x_1, f(x_1))$$ and $$(x_2, f(x_2))$$ are two points on a function $$f(x)$$, then $$f(x)$$ is **convex** if:

$$
\mathbf{f(\theta x_1+(1-\theta x_2))\leq \theta f(x_1)+(1-\theta)f(x_2)}
$$

Consider a point $$P$$ on the line connecting $$[x_1, f(x_1)]$$ and $$[x_2, f(x_2)]$$, its coordinate on that line is $$[\theta x_1+(1-\theta) x_2, \theta f(x_1)+(1-\theta) f(x_2)]$$. The corresponding point on the graph is $$[\theta x_1+(1-\theta) x_2, f([\theta x_1+(1-\theta) x_2)]$$.

The same condition, but inverted, can be applied to define a concave function. A function $$f(x)$$ is **concave** iff:

$$
\mathbf{f(\theta x_1+(1-\theta x_2))\geq \theta f(x_1)+(1-\theta)f(x_2)}
$$

## Preliminary: Affine Functions
An function $$f(x)$$ is an **affine function** iff:

$$\mathbf{f(\theta x_1+(1-\theta) x_2)=f(\theta x_1)+f((1-\theta) x_2)}$$

Let's take a simple function $$f(x)=Ax+C$$ where $$x$$ is a vector. $$A$$ is a transformation matrix, and $$C$$ is a constant vector. Then, for two vectors $$\vec{v_1}$$ and $$\vec{v_2}$$, we have:

$$
f(\theta \vec{v_1}+(1-\theta) \vec{v_2})=A.[\theta \vec{v_1}+(1-\theta) \vec{v_2}]+C \\
=A\theta \vec{v_1}+A(1-\theta) \vec{v_2}+(\theta+1-\theta)C \\
=A\theta \vec{v_1}+A(1-\theta) \vec{v_2}+\theta C+(1-\theta)C \\
=[\theta A\vec{v_1}+\theta C]+[(1-\theta) A\vec{v_2}+(1-\theta)C]\\
=\theta[A\vec{v_1}+C]+(1-\theta)[A\vec{v_2}+C]\\
=\theta f(\vec{v_1})+(1-\theta) f(\vec{v_2})
$$

Thus all mappings of the form $$\mathbf{f(x)=Ax+C}$$ are **affine functions**.

We may draw another interesting conclusion: **affine functions are both convex and concave**. This is because affine functions satisfy the equality conditions for both convexity and concavity: **an affine set on an affine function lies fully on the function itself**.

## Preliminary: Pointwise Infimum and Supremum
The **infimum** of two functions $$f_1(x)$$ and $$f_2(x)$$ is defined as thus:

$$
inf(f_1, f_2)=min\{f_1(x), f_2(x)\}
$$

The **supremum** of two functions $$f_1(x)$$ and $$f_2(x)$$ is defined as thus:

$$
sup(f_1, f_2)=max\{f_1(x), f_2(x)\}
$$

We'll prove an interesting result that will prove useful when exploring the shape of the **Lagrangian of the objective function**, namely that **the infimum of any set of concave functions is a concave function**.

Let there be a chord $$C_1$$ connecting (x_1, f_1(x_1)) and (x_2, f_1(x_2)) for a concave function $$f_1(x)$$.
Let there be a chord $$C_2$$ connecting (x_1, f_2(x_1)) and (x_2, f_2(x_2)) for a concave function $$f_2(x)$$.

Let us fix two points with the same x-coordinate $$x_0$$, one lying on $$C_1$$, the other on $$C_2$$. Assume these points are respectively represented as:

$$
P_1=\alpha_1 x_1+\beta_1 x_2, \alpha_1+\beta_1=1 \\
P_2=\alpha_2 x_1+\beta_2 x_2, \alpha_2+\beta_2=1
$$

Then, by the definition of a **concave function** (see above), we can write for $$f_1$$ and $$f_2$$:

$$
f_1(\alpha_1 x_1+\beta_1 x_2)\geq \alpha_1 f_1(x_1)+\beta_1 f_1(x_2) \\
f_2(\alpha_2 x_1+\beta_2 x_2)\geq \alpha_2 f_1(x_1)+\beta_2 f_1(x_2)
$$

If we assume that $$inf(f_1(\alpha_1 x_1+\beta_1 x_2), f_2(\alpha_2 x_1+\beta_2 x_2))=f_1(\alpha_1 x_1+\beta_1 x_2)$$, this implies that:

$$
f_2(\alpha_2 x_1+\beta_2 x_2) \geq f_1(\alpha_1 x_1+\beta_1 x_2)
$$

This gives us the two inequalities:

$$
f_2(\alpha_2 x_1+\beta_2 x_2)\geq \alpha_1 f_1(x_1)+\beta_1 f_1(x_2) \\
f_2(\alpha_2 x_1+\beta_2 x_2)\geq \alpha_2 f_2(x_1)+\beta_2 f_2(x_2)
$$

The second inequality, we already know; it's the combination of the first and second inequality which tells us that the graph of the function $$f_2(\alpha_2 x_1+\beta_2 x_2)$$ is higher than the two points $$P_0$$, $$P_1$$ on the respective chords $$C_1$$, $$C_2$$ of $$f_1$$, $$f_2$$.

Since this is a general result for any two chords in $$f_1$$ and $$f_2$$, we can conclude that **the infimum of two convex functions is also a convex function**. This can be extended to an arbitrary set of arbitrary concave functions.

Using very similar arguments, we can also prove that **the supremum of an arbitrary set of convex functions is also a convex function**.

The other straightforward conclusion is that **the infimum of any set of affine functions is always concave, because affine functions are concave** (they are also convex, but we cannot draw any conclusions about the infimum of convex functions).

## Lagrangian
We now have the machinery to explore the Lagrangian Dual in some detail.

We have already seen in [Vector Calculus: Lagrange Multipliers, Manifolds, and the Implicit Function Theorem]({% post_url 2021-04-24-vector-calculus-lagrange-multipliers %}) that the gradient vector of a function can be expressed as a **linear combination of the gradient vectors** of the constraint manifolds.

$$ \mathbf{
Df=\lambda_1 Dh_1(U,V)+\lambda_2 Dh_2(U,V)+\lambda_3 Dh_3(U,V)+...+\lambda_n Dh_n(U,V)
} \\
\\
$$

We can rewrite this as:

$$ \mathbf{
Df(x,\lambda)=\sum_{i=1}^n\lambda_i.Dg_i(x)
} \\
\Rightarrow f(x,\lambda)=\sum_{i=1}^n\lambda_i.g_i(x)
$$

where $$x=(U,V)$$. We will not consider the pivotal and non-pivotal variables separately in this discussion.

We will first consider the **Lagrangian** form of a function. The Lagrangian form is simply restating the Lagrange Multiplier form as a function $$L(X,\lambda)$$, like so:

$$
L(x,\lambda)=f(x,\lambda)+\sum_{i=1}^n\lambda_i.g_i(x)
$$

We have simply moved all the terms of the Lagrangian formulation onto one side and denoted it by $$L(x,\lambda)$$.


## Exploring the Properties of the Lagrangian

## Affine Functions as Constraints
## Geometric Intuition of Convex Optimisation
### Active Constraints
### Inactive Constraints
## Max-Min Inequality
### Duality Gap
## Karush-Kuhn-Tucker Conditions
## Saddle Point Theorem

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

![Affine Set](/assets/images/affine-set.png)

In words, a vector is an **affine combination** of $$n$$ vectors if the **coefficients of the linear combinations of those vectors sum to one**.

## Preliminary: Convex and Concave Functions
The layman's explanation of a convex function is that it is a bowl-shaped function. However, let us state this mathematically: we say a function is convex, if the graph of that function lies below every point on a line connecting any two points on that function.

![Convex Function](/assets/images/convex-function.png)

If $$(x_1, f(x_1))$$ and $$(x_2, f(x_2))$$ are two points on a function $$f(x)$$, then $$f(x)$$ is **convex** iff:

$$
\mathbf{f(\theta x_1+(1-\theta x_2))\leq \theta f(x_1)+(1-\theta)f(x_2)}
$$

Consider a point $$P$$ on the line connecting $$[x_1, f(x_1)]$$ and $$[x_2, f(x_2)]$$, its coordinate on that line is $$[\theta x_1+(1-\theta) x_2, \theta f(x_1)+(1-\theta) f(x_2)]$$. The corresponding point on the graph is $$[\theta x_1+(1-\theta) x_2, f([\theta x_1+(1-\theta) x_2)]$$.

![Concave Function](/assets/images/concave-function.png)
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

## Preliminary: Some Inequality Proofs

### Result 1

If $$a\geq b$$, and $$c\geq d$$, then:

$$min(a,c)\geq min(b,d)$$

The proof goes like this, we can define the following inequalities in terms of the $$min$$ function:

$$
\begin{eqnarray}
a \geq min(a,c) \label{eq:1} \\
c \geq min(a,c) \label{eq:2} \\
b \geq min(b,d) \label{eq:3} \\
d \geq min(b,d) \label{eq:4} \\
\end{eqnarray}
$$

Then, the identities $$a \geq b$$ and $$\eqref{eq:3}$$ imply:

$$ a \geq b \geq min(b,d)$$

Similarly, the identities $$c \geq d$$ and $$\eqref{eq:4}$$ imply that:

$$ c \geq d \geq min(b,d)$$

Therefore, regardless of our choice from $$a$$, $$c$$ from the function $$min(a,c)$$, the result will always be greater than $$min(b,d)$$. Thus we write:

$$ \begin{equation} \mathbf{min(a,c) \geq min(b,d)} \label{ineq:1}\end{equation}$$

### Result 2

Here we prove that:

$$
min(a+b, c+d) \geq min(a,c)+min(b,d)
$$

Here we take a similar approach, noting that:

$$
a \geq min(a,c) \\
c \geq min(a,c) \\
b \geq min(b,d) \\
d \geq min(b,d) \\
$$

Therefore, if we compute $$a+b$$ and $$c+d$$, we can write:

$$
a+b \geq min(a,c)+min(b,d) \\
c+d \geq min(a,c)+min(b,d)
$$

Therefore, regardless of our choice from $$a+b$$, $$c+d$$ from the function $$min(a+b,c+d)$$, the result will always be greater than $$min(a,c)+min(b,d)$$. Thus we write:

$$
\begin{equation}\mathbf{min(a+b, c+d) \geq min(a,c)+min(b,d)} \label{ineq:2} \end{equation}
$$

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

![Concave Infimum](/assets/images/concave-infimum.png)

Let there be a chord $$C_1$$ connecting (x_1, f_1(x_1)) and (x_2, f_1(x_2)) for a concave function $$f_1(x)$$.
Let there be a chord $$C_2$$ connecting (x_1, f_2(x_1)) and (x_2, f_2(x_2)) for a concave function $$f_2(x)$$.

Let us fix two arbitrary x-coordinates $$x_1$$ and $$x_2$$. Then, by the definition of a **concave function** (see above), we can write for $$f_1$$ and $$f_2$$:

$$
f_1(\alpha x_1+\beta x_2)\geq \alpha f_1(x_1)+\beta f_1(x_2) \\
f_2(\alpha x_1+\beta x_2)\geq \alpha f_2(x_1)+\beta f_2(x_2)
$$

where $$\alpha+\beta=1$$. Let us define the infimum function as:

$$\mathbf{inf(x)=min\{f_1(x), f_2(x)\}}$$

Then:

$$
inf(\alpha x_1+\beta x_2)=min\{ f_1(\alpha x_1+\beta x_2), f_2(\alpha x_1+\beta x_2)\} \\
\geq min\{ \alpha f_1(x_1)+\beta f_1(x_2), \alpha f_2(x_1)+\beta f_2(x_2)\} \hspace{4mm} .......(from \eqref{ineq:1})\\
\geq \alpha.min\{f_1(x_1),f_2(x_1)\} + \beta.min\{f_1(x_2),f_2(x_2)\} \hspace{4mm} .......(from \eqref{ineq:2})\\
= \mathbf{\alpha.inf(x_1) + \beta.inf(x_2)}
$$

Thus, we can summarise:

$$
\begin{equation}
\mathbf{inf(\alpha x_1+\beta x_2) \geq \alpha.inf(x_1) + \beta.inf(x_2)}
\end{equation}
$$

which is the form of an **concave function**, and thus we can conclude that $$inf(x)$$ is a concave function if all of its component functions are concave.

Since this is a general result for any two coordinates $$x_1,x_2:x_1,x_2 \neq 0$$, we can conclude that **the infimum of two concave functions is also a concave function**. This can be extended to an arbitrary set of arbitrary concave functions.

Using very similar arguments, we can also prove that **the supremum of an arbitrary set of convex functions is also a convex function**.

The other straightforward conclusion is that **the infimum of any set of affine functions is always concave, because affine functions are concave** (they are also convex, but we cannot draw any general conclusions about the infimum of convex functions).

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

![Shape of Lagrangian for a Convex Objective Function](/assets/images/lagrangian-shape.png)
![Shape of Lagrangian for a Convex Objective Function](/assets/images/quadratic-surface-no-cross-term-saddle.png)


## Exploring the Properties of the Lagrangian

## Geometric Intuition of Convex Optimisation
### Active Constraints
### Inactive Constraints
## Max-Min Inequality
### Duality Gap
## Karush-Kuhn-Tucker or Saddle Point Theorem

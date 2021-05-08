---
title: "Quadratic Optimisation: Lagrangian Dual, and the Karush-Kuhn-Tucker Conditions"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Quadratic Optimisation", "Linear Algebra", "Optimisation", "Theory"]
---

This article concludes the (very abbreviated) theoretical background required to understand **Quadratic Optimisation**. Here, we extend the **Lagrangian Multipliers** approach, which in its current form, admits only equality constraints. We will extend it to allow constraints which can be expressed as inequalities.

Much of this discussion applies to the general class of **Convex Optimisation**; however, I will be constraining the form of the problem slightly to simplify discussion. We have already developed most of the basic mathematical results (see [Quadratic Optimisation Concepts]({% post_url 2021-05-08-quadratic-optimisation-theory %})) in order to fully appreciate the implications of the **Karush-Kuhn-Tucker Theorem** (also called the **Saddle Point Theorem**).

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

For this discussion, I'll omit the equality constraints $$h_i(x)$$ for clarity; any **equality constraints can always be converted into inequality constraints**, and become part of $$g_i(x)$$.

Thus, this is the reframing of the **Quadratic Optimisation** problem for the purposes of this discussion.

Minimise (with respect to $$x$$), $$\mathbf{f(x)}$$

subject to: $$\mathbf{g_i(x)\leq 0, i=1,...,n}$$

where:

- $$\mathbf{f(x)}$$ is a **convex function**
- $$\mathbf{g_i(x)}$$ are **affine functions**

We have already seen in [Vector Calculus: Lagrange Multipliers, Manifolds, and the Implicit Function Theorem]({% post_url 2021-04-24-vector-calculus-lagrange-multipliers %}) that the gradient vector of a function can be expressed as a **linear combination of the gradient vectors** of the constraint manifolds.

$$ \mathbf{
Df=\lambda_1 Dh_1(U,V)+\lambda_2 Dh_2(U,V)+\lambda_3 Dh_3(U,V)+...+\lambda_n Dh_n(U,V)
}
$$

We can rewrite this as:

$$ \mathbf{
Df(x)=\sum_{i=1}^n\lambda_i.Dg_i(x)
} \\
\Rightarrow f(x)=\sum_{i=1}^n\lambda_i.g_i(x)
$$

where $$x=(U,V)$$. We will not consider the pivotal and non-pivotal variables separately in this discussion.

In this original formulation, we expressed the gradient vector as a linear combination of the gradient vector(s) of the constraint manifold(s).
We can bring everything over to one side and flip the signs of the Lagrangian Multipliers to get the following:

$$
Df(x)+\sum_{i=1}^n\lambda_i.Dg_i(x)=0
$$

Since the derivatives in this case represent the gradient vectors, we can rewrite the above as:

$$
\mathbf{
\begin{equation}
\nabla f(x)+\sum_{i=1}^n\lambda_i.\nabla g_i(x)=0
\label{eq:kkt-1}
\end{equation}
}
$$

This expresses the fact that the **gradient vector of the tangent space must be opposite (and obviously parallel) to the direction of the gradient vector of the objective function**. All it really amounts to is a **change in the sign** of the multipliers $$\lambda_i$$; we do this so that the **Lagrange multiplier terms act as penalties** when the constraints $$g_i(x)$$ are violated. We will see this in action when we explore the properties of the Lagrangian in the next few sections.

The identity $$\eqref{eq:kkt-1}$$ is the **Stationarity Condition**, one of the **Karush-Kuhn-Tucker Conditions**.

## Lagrangian
We now have the machinery to explore the **Lagrangian Dual** in some detail. Before proceeding with this section, let us restate the **Quadratic Optimisation** problem in a more simplified form first.

We will first consider the **Lagrangian** of a function. The Lagrangian form is simply restating the Lagrange Multiplier form as a function $$L(X,\lambda)$$, like so:

$$
L(x,\lambda)=f(x)+\sum_{i=1}^n\lambda_i.g_i(x)\text{  such that }\lambda_i\geq 0
$$

**Note**: The $$\mathbf{\lambda_i \geq 0}$$ is an important constraint that will help us formulate the **Karush-Kuhn-Tucker Conditions**.

We have simply moved all the terms of the Lagrangian formulation onto one side and denoted it by $$L(x,\lambda)$$.

Note that differentiating with respect to $$x$$ and setting it to zero, will get us back to the usual **Vector Calculus**-motivated definition, i.e.:

$$
D_xL=
\mathbf{
\nabla f-{[\nabla G]}^T\lambda
}
$$

where $$G$$ represents $$n$$ constraint functions, $$\lambda$$ represents the $$n$$ Lagrange multipliers, and $$f$$ is the objective function.

## Exploring the Properties of the Lagrangian
### 1. The Primal Optimisation Problem
We will now explore the properties of the Lagrangian, both analytically, as well as geometrically.

Remembering the definition of the supremum of a function, we find the supremum of the Lagrangian with respect to $$\lambda$$ (that is, to find the supremum in each case, we vary the value of $$\lambda$$) to be the following:

$$
sup_\lambda L(x,\lambda)=\begin{cases}
f(x) & \text{if } g_i(x)\leq 0 \\
\infty & \text{if } g_i(x)>0
\end{cases}
$$

Remember that $$\mathbf{\lambda \geq 0}$$.

Thus, for the first case, if $$g_i(x) \leq 0$$, the best we can do is set $$\lambda=0$$, since any other non-negative value will not be the supremum.

In the second case, if $$g(x)>0$$, the supremum of the function can be as high as we like as long as we keep increasing the value of $$\lambda$$. Thus, we can simply set it to $$\infty$$, and the corresponding supremum becomes $$\infty$$.

We can see that the function $$sup_\lambda L(x)$$ incorporates the constraints $$g_i(x)$$ directly, there is a penalty of $$\infty$$ for any constraint which is violated. Therefore, the original problem of minimising $$f(x)$$ can be equivalently stated as:

$$
\text{Minimise (w.r.t. x)  }sup_\lambda L(x,\lambda) \\
\text{where } L(x,\lambda)=f(x)+\sum_{i=1}^n\lambda_i.g_i(x)
$$

Equivalently, we say:

$$
\text{Find }\mathbf{inf_x\text{  }sup_\lambda L(x,\lambda)} \\
\text{where } L(x,\lambda)=f(x)+\sum_{i=1}^n\lambda_i.g_i(x)
$$

This is referred to in the mathematical optimisation field as the **primal optimisation problem**.

Difficulty in Solving the Primal Optimisation Problem
The main difficulty in solving the primal problem arises from the fact that the number of variables to optimise is stuff

### 2. The Dual Optimisation Problem
We already know from the [Max-Min Inequality]({% post_url 2021-05-08-quadratic-optimisation-theory %}) that:

$$
\mathbf{\text{sup}_y \text{ inf}_x f(x,y)\leq \text{inf}_x \text{ sup}_y f(x,y)} \text{   }\forall x,y\in\mathbb{R}
$$

Since this is a general statement about any $$f(x,y)$$, we can apply this inequality to the Primal Optimisation Problem, i.e.:

$$
\text{sup}_\lambda \text{ inf}_x L(x,\lambda) \leq \text{inf}_x \text{ sup}_\lambda L(x,\lambda)
$$

The right side is the **Primal Optimisation Problem**, and the right side is known as the **Dual Optimisation Problem**, and in this case, the **Lagrangian Dual**.

To understand the fuss about the **Lagrangian Dual**, we will begin with the more restrictive case where equality holds for the **Max-Min Inequality**, and later discuss the more general case and its implications. For this first part, we will assume that:

$$
\text{sup}_\lambda \text{ inf}_x L(x,\lambda) = \text{inf}_x \text{ sup}_\lambda L(x,\lambda)
$$

Let's look at a motivating example. This is the graph of the Lagrangian for the following problem:

$$
\text{Minimise}_x f(x)=x^2 \\
\text{subject to: } x \leq 0
$$

The Lagrangian in this case is given by:

$$L(x,\lambda)=x^2+\lambda x$$

This is the corresponding graph of $$L(x,\lambda)$$.

![Shape of Lagrangian for a Convex Objective Function](/assets/images/lagrangian-shape.png)

Let us summarise a few properties of this graph.

- **The function is convex in $$x$$**: Assume $$\lambda=C$$ is a constant, then the function has the form $$\mathbf{x^2+Cx}$$ which is a family of parabolas. **A parabola is a convex function**, thus the result follows.
- **The function is concave in $$\lambda$$**: Assume that $$x=C$$ and $$x^2=K$$ are constants, then the function has the form $$\mathbf{C\lambda+K}$$, which is the general form of **affine functions**. **Affine functions are both convex and concave**, but we will be drawing more conclusions based on their concave nature, so we will simply say that **the Lagrangian is concave in $$\lambda$$**. Thus, **the Lagrangian is also a family of concave functions**.
- As a direct consequence of the Lagrangian being a family of concave functions, we can say that **the pointwise infimum of the Lagrangian is a concave function**. We established this result in [Quadratic Optimisation Concepts]({% post_url 2021-05-08-quadratic-optimisation-theory %}). This result is irrespective of the shape of the Lagrangian in the direction of $$x$$. This is important because the

![Shape of Lagrangian for a Convex Objective Function](/assets/images/lagrangian-saddle.png)

## Geometric Intuition of Convex Optimisation
## Karush-Kuhn-Tucker or Saddle Point Theorem
### Weak Duality
### Strong Duality
### Duality Gap



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

We will talk of this simplification when we explore the shape of the Lagrangian.

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

## Preliminary: Infimum and Supremum
The **infimum** of a function $$f(x)$$ is defined as:

$$
\mathbf{inf_x(f(x))=M | M<f(x) \forall x}
$$

The infimum is defined for all functions even if the minimum does not exist for a function, and is equal to the mimimum if it does exist.

The supremum of a function $$f(x)$$ is defined as:

$$
\mathbf{sup_x(f(x))=M | M>f(x) \forall x}
$$

The **supremum** is defined for all functions even if the maximum does not exist for a function, and is equal to the maximum if it does exist.

## Preliminary: Pointwise Infimum and Pointwise Supremum
The **pointwise infimum** of two functions $$f_1(x)$$ and $$f_2(x)$$ is defined as thus:

$$
pinf(f_1, f_2)=min\{f_1(x), f_2(x)\}
$$

The **pointwise supremum** of two functions $$f_1(x)$$ and $$f_2(x)$$ is defined as thus:

$$
psup(f_1, f_2)=max\{f_1(x), f_2(x)\}
$$

We'll prove an interesting result that will prove useful when exploring the shape of the **Lagrangian of the objective function**, namely that **the pointwise infimum of any set of concave functions is a concave function**.

![Concave Pointwise Infimum](/assets/images/concave-infimum.png)

Let there be a chord $$C_1$$ connecting (x_1, f_1(x_1)) and (x_2, f_1(x_2)) for a concave function $$f_1(x)$$.
Let there be a chord $$C_2$$ connecting (x_1, f_2(x_1)) and (x_2, f_2(x_2)) for a concave function $$f_2(x)$$.

Let us fix two arbitrary x-coordinates $$x_1$$ and $$x_2$$. Then, by the definition of a **concave function** (see above), we can write for $$f_1$$ and $$f_2$$:

$$
f_1(\alpha x_1+\beta x_2)\geq \alpha f_1(x_1)+\beta f_1(x_2) \\
f_2(\alpha x_1+\beta x_2)\geq \alpha f_2(x_1)+\beta f_2(x_2)
$$

where $$\alpha+\beta=1$$. Let us define the **pointwise infimum** function as:

$$\mathbf{pinf(x)=min\{f_1(x), f_2(x)\}}$$

Then:

$$
pinf(\alpha x_1+\beta x_2)=min\{ f_1(\alpha x_1+\beta x_2), f_2(\alpha x_1+\beta x_2)\} \\
\geq min\{ \alpha f_1(x_1)+\beta f_1(x_2), \alpha f_2(x_1)+\beta f_2(x_2)\} \hspace{4mm} .......(from \eqref{ineq:1})\\
\geq \alpha.min\{f_1(x_1),f_2(x_1)\} + \beta.min\{f_1(x_2),f_2(x_2)\} \hspace{4mm} .......(from \eqref{ineq:2})\\
= \mathbf{\alpha.pinf(x_1) + \beta.pinf(x_2)}
$$

Thus, we can summarise:

$$
\begin{equation}
\mathbf{pinf(\alpha x_1+\beta x_2) \geq \alpha.pinf(x_1) + \beta.pinf(x_2)}
\end{equation}
$$

which is the form of an **concave function**, and thus we can conclude that $$pinf(x)$$ is a concave function if all of its component functions are concave.

Since this is a general result for any two coordinates $$x_1,x_2:x_1,x_2 \neq 0$$, we can conclude that **the pointwise infimum of two concave functions is also a concave function**. This can be extended to an arbitrary set of arbitrary concave functions.

Using very similar arguments, we can also prove that **the pointwise supremum of an arbitrary set of convex functions is also a convex function**.

The other straightforward conclusion is that **the pointwise infimum of any set of affine functions is always concave, because affine functions are concave** (they are also convex, but we cannot draw any general conclusions about the pointwise infimum of convex functions).

**Note**: The **pointwise infimum** and **pointwise supremum** have different definitions from the **infimum** and **supremum**, respectively.

## Active and Inactive Constraints
In Quadratic Programming, $$g_i(x)|i=1,2,...,n$$ represent the constraint functions. An important concept to get an intuition is about the difference between dealing with pure equality constraints and inequality cnstraints.

The diagram below shows an example where all the constraints are equality constraints.

![Equality Coonstraints](/assets/images/optimisation-equality-constraints.png)

There are two points to note.

- All equality constraints are expressed in the form $$g_i(x)=0$$ and they all must be satisfied simultaneously.
- All equality constraints, being affine must be tangent to the objective function surface, since only then can the gradient vector of the solution be expressed as the Lagrangian combination of these tangent spaces.

The situation changes when inequality constraints are involved. Here is another rough diagram to demonstrate. The y-coordinate represents the image of the objective function $$f(x)$$. The x-coordinate represents the image of the constraint function $$g(x)$$, i.e., the different values $$g(x)$$ can take for different values of $$x$$.

The equality condition in this case maps to the y-axis, since that corresponds to $$g(x)=0$$. However, we're dealing with inequality constraints here, namely, $$g(x) \leq 0$$; thus the viable space of solutions for $$f(x)$$ are all to the left of the y-axis.

As you can see, since $$g(x)\leq 0$$, the solution is not required to touch the level set of the constraint manifold corresponding to zero. Such solutions might not be the optimal solutions (we will see why in a moment), but they are viable solutions nevertheless.

We now draw two example solution spaces with two different shapes.

![Active Constraint in Optimisation](/assets/images/optimisation-active-constraint.png)

In the first figure, the global minimum of $$f(x)$$ violates the constraint since it lies in the $$g(x)>0$$. Thus, we cannot pick that; we must pick minimum $$f(x)$$ that does not violate the constraint $$g(x)\leq 0$$. This point in the diagram lies on the y-axis, i.e., on $$g(x)=0$$. The constraint $g(x)\leq 0$$ in this scenario is considered an **active constraint**.

![Inactive Constraint in Optimisation](/assets/images/optimisation-inactive-constraint.png)

Contrast this with the diagram above. Here, the shape of the solution space is different. The minimum $$f(x)$$ lies within the $$g(x)\leq 0$$ zone. This means that even if we minimise $$f(x)$$ without regard to the constraint $$g(x)\leq 0$$, we'll still get the minimum solution which still satisfies the constraint. In this scenario, we call $$g(x)\leq 0$$ an **inactive constraint**.

We note the 
## Lagrangian
We now have the machinery to explore the **Lagrangian Dual** in some detail. Before proceeding with this section, let us restate the **Quadratic Optimisation** problem in a more simplified form first.

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

We will first consider the **Lagrangian** of a function. The Lagrangian form is simply restating the Lagrange Multiplier form as a function $$L(X,\lambda)$$, like so:

$$
L(x,\lambda)=f(x,\lambda)+\sum_{i=1}^n\lambda_i.g_i(x)
$$

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

Remembering the definition of the supremum of a function, we find the supremum of the Lagrangian with respect to \lambda (that is, to find the supremum in each case, we vary the value of \lambda) to be the following:

$$
sup_\lambda L(x,\lambda)=\begin{cases}
f(x) & \text{if } g_i(x)\leq 0 \\
\infty & \text{if } g_i(x)>0
\end{cases}
$$

We can see that the function $$sup_\lambda L(x)$$ incorporates the constraints $$g_i(x)$$ directly, there is a penalty of $$\infty$$ for any constraint which is violated. Therefore, the original problem of minimising $$f(x)$$ can be equivalently stated as:

$$
\text{Minimise (w.r.t. x)  }sup_\lambda L(x,\lambda) \\
\text{where } L(x,\lambda)=f(x,\lambda)+\sum_{i=1}^n\lambda_i.g_i(x)
$$

Equivalently, we say:

$$
\text{Find }\mathbf{inf_x\text{  }sup_\lambda L(x,\lambda)} \\
\text{where } L(x,\lambda)=f(x,\lambda)+\sum_{i=1}^n\lambda_i.g_i(x)
$$

This is referred to in the mathematical optimisation field as the **primal optimisation problem**.

Difficulty in Solving the Primal Optimisation Problem

### 2. The Dual Optimisation Problem
## 3. Max-Min Inequality
## Geometric Intuition of Convex Optimisation
## Karush-Kuhn-Tucker or Saddle Point Theorem
### Weak Duality
### Strong Duality
### Duality Gap

![Shape of Lagrangian for a Convex Objective Function](/assets/images/lagrangian-shape.png)
![Shape of Lagrangian for a Convex Objective Function](/assets/images/quadratic-surface-no-cross-term-saddle.png)



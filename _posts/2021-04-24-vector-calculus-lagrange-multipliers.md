---
title: "Vector Calculus: Lagrange Multipliers, Manifolds, and the Implicit Function Theorem"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Optimisation", "Vector Calculus", "Lagrange Multipliers", "Theory"]
---
In this article, we finally put all our understanding of **Vector Calculus** to use by showing why and how **Lagrange Multipliers** work. We will be focusing on several important ideas, but the most important one is around the **linearisation of spaces at a local level**, which might not be smooth globally. The **Implicit Function Theorem** will provide a strong statement around the conditions necessary to satisfy this.

We will then look at **critical points**, and how constraining them to a manifold naturally leads to the condition that the **normal vector of the curve to be optimised, must also be normal to the tangent space of the manifold**.

We will then restate this in terms of **Lagrange multipliers**.

**Note**: This article pulls a lot of understanding together, so be sure to have understood the material in [Vector Calculus: Graphs, Level Sets, and Linear Manifolds]({%post_url 2021-04-20-vector-calculus-simple-manifolds%}), before delving into this article, or you might get thoroughly confused.

- Implicit Function Theorem
- Constrained Critical Points
- Lagrangian Formulation without Paramterisation

We start with two important ideas that we reviewed.

- Function Composition and Functions as Objects
- Chain Rule
- Constraints are not necessarily linear

## Functions are Objects
Let us assume that:

$$
f(x,y)=xy \\
y=g(x)=x^2
$$

then the notation $$F=f\circ g$$ represents function composition, where represents a function $$F$$ which has the same output as $$f(x,g(x))$$ (in this example). Also note that $$F$$ is a function of only $$x$$. In texts, $$f(x,y)$$ is written as $$f(x,g(x))$$ and is equivalent to the above form.

Do not let the fact that there is a function in the parameter of $$f$$ confuse you; treat it as you would any other variable. If you know th actual expression $$f(x,g)$$, you can differentiate with respect to $$g$$ if needed. After all, $$g(x)$$ is essentially $$y$$.
Writing it as $$f(x,g(x))$$ is notational shorthand for expressing that y is not really a free variable, it is expressed in terms of $$x$$. Moreover $$D_xF(x)$$ is the same thing as writing $$Df(x,g(x))$$.

Now, if we wanted to find $$D_xf(x,g(x))$$, it is trivial to see that substituting $$x^2$$ for $$y$$ in $$f(x,y)$$ gives us $$f(x)=x^3$$, therefore:

$$
D_xf(x,g(x))=3x^2
$$

However, let's look at the Chain Rule of differentiation for the above $$f(x,y)$$, because in our proofs, the actual form of $$f(x,y)$$ and $$g(x)$$ will not be available, and thus we will have to use the Chain Rule to express any results. For multivariable calculus, you may write:

$$
D_xf(x,g)=\frac{\partial f(x,g)}{\partial x} \\
=\frac{df(x, g)}{d{[x\hspace{3mm} g]}^T}.\frac{d{[x\hspace{3mm} g]}^T}{dx} \\
$$

Let's denote $$\Phi (x)=\begin  {bmatrix}x \\ g\end{bmatrix}$$.
$${[x \hspace{3mm} g]}^T$$ is a vector so we are partially differentiating $$f(x,y)$$. In our example above, we can write for the first term, and substitute:

$$
\frac{f(x,g)}{d[x \hspace{3mm} g]}=\left[\frac{\partial f(x,g)}{\partial x} \hspace{3mm} \frac{\partial f(x,g)}{\partial g} \right] \\
\Rightarrow D_xf(x,y)=\left[\frac{\partial f(x,g)}{\partial x} \hspace{3mm} \frac{\partial f(x,g)}{\partial g} \right].\frac{d\Phi (x)}{dx}
$$

For the second term, we may write:

$$
\frac{d{[x \hspace{3mm} g]}^T}{dx}=\begin{bmatrix}
1 \\
\frac{dg}{dx}
\end{bmatrix} \\
\Rightarrow
D_xf(x,g)=\left[\frac{\partial f(x,g)}{\partial x} \hspace{3mm} \frac{\partial f(x,g)}{\partial g} \right].\begin{bmatrix}
1 \\
\frac{dg}{dx}
\end{bmatrix} \\
= \frac{\partial f(x,g)}{\partial x} + \frac{\partial f(x,g)}{\partial g}\frac{dg}{dx}
$$

Only now do we need to unroll $$g(x)$$ to look at the specific form this derivative takes. We can write the following:

$$
\frac{dg}{dx}=2x \\
\frac{\partial f(x,g)}{\partial x}=g \\
\frac{\partial f(x,g)}{\partial g}=x
$$

Substituting these values back into the expression for $$D_xF(x)$$, we get:

$$
D_xf(x,g)=g+x.2x=g(x)+2x^2 \\
= x^2+2x^2 \\
= 3x^2
$$

Yes, that was a complicated way of computing the same result we got earlier, but I want you to see the mechanics involved in applying the Chain Rule. More importantly, let us revisit the intermediate expression we used, namely:

$$
D_xf(x,g)=\left[\frac{\partial f(x,g)}{\partial x} \hspace{3mm} \frac{\partial f(x,g)}{\partial g} \right].\frac{d\Phi (x)}{dx}
$$

If you notice carefully, **the expression $$\left[\frac{\partial f(x,g)}{\partial x} \hspace{3mm} \frac{\partial f(x,g)}{\partial g} \right]$$ exactly represents the gradient operator $$\nabla f(x,g(x))$$**.

Furthermore, look at $$\frac {d\Phi (x)}{dx}=\begin{bmatrix}1 \\ \frac{dg}{dx}\end{bmatrix}$$. We can recognise this as the parametric form of the line $$\mathbf{y=g'(x).x}$$. This is because any vector on the tangent can be represented as $$\begin{bmatrix}1 \\ \frac{dg}{dx}\end{bmatrix}.t$$. As a consequence, **this is the tangent space of $$g(x)$$**. If we represent $$\frac {d\Phi (x)}{dx}$$ as $$T_x$$ (the tangent space), we can rewrite the identity as:

**$$
D_xf(x,g)=\nabla f(x,g(x)).T_x
$$**

**Important Note**: Note that $$T_x$$ is **not** the normal vector to the tangent, but the actual vector along the tangent.

If we have a point $$P$$ which satisfies $$g(x)$$, i.e., has the coordinates $$(x_0, g(x_0))$$, then the following holds:

**$$
D_xf(P)=\nabla f(P).T_x
$$**

## Implicit Function Theorem
Functions come in many shapes and sizes. They aren't always necessarily linear. However, that does not mean that analysis of these nonlinear functions is intractable. Calculus makes the mostly nonlinear world around us, and tells us that we can treat any curve or surface, in any dimension, as linear if we only zoom into it close enough. It basically asks us to pretend that a complicated curve (and the corresponding function) is a linear function. This approximation is grossly wrong at larger scales, but gets better and better the more we zoom in.

In optimisation, the constraints aren't always going to be linear.
Every level set is a graph, i.e., for every equation, you can find the graph of a function where $$n-k$$ variables in the level set equation may be expressed as functions of $$k$$ independent variables, i.e., such a mapping exists locally. We could not have made the assumption of the existence of this mapping for the general case of nonlinear constraints without the Implicit Function Theorem.

## Constrained Critical Points
---SOME TEXT---

## Lagrangian Reformulation without Paramterisation
---SOME TEXT---

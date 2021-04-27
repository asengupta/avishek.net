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

- Constrained Critical Points
- Lagrangian Formulation without Paramterisation
- Implicit Function Theorem

We start with two important ideas that we reviewed.

- Function Composition 
- Chain Rule and Functions as Objects
- Constraints are not necessarily linear

## Function Composition and Functions as Objects
Let us assume that:

$$
f(x,y)=xy \\
y=g(x)=x^2
$$

Then the notation $$\mathbf{F=f\circ g}$$ represents function composition, where represents a function $$F$$ which has the same output as $$f(x,g(x))$$ (in this example). Also note that $$F$$ is a function of only $$x$$. In texts, $$f(x,y)$$ is written as $$f(x,g(x))$$ and is equivalent to the above form.

Do not let the fact that there is a function in the parameter of $$f$$ confuse you; treat it as you would any other variable. If you know the actual expression $$f(x,g)$$, you can differentiate with respect to $$g$$ if needed. After all, $$g(x)$$ is essentially $$y$$.

Writing it as $$f(x,g(x))$$ is notational shorthand for expressing that $$y$$ is not really a free variable, it is expressed in terms of $$x$$. Moreover $$D_xF(x)$$ is the same thing as writing $$Df(x,g(x))$$.

We can borrow our intuition of **function pipelines in programming** to make sense of this: an input $$x$$ enters $$g(x)$$, comes out as some output, which is then fed to the $$y$$ parameter of $$f(x,y)$$. The $$x$$ parameter is already available, so it gets applied for free (actually, while programming, you can't say things like "gets applied for free", you actually have to do the necessary plumbing to allow $$x$$ to reach $$f(x,y)$$).

**Note that the composite function $$F(x)$$ takes in only one input, $$x$$.** This is because the first function that is applied is $$g(x)$$. You do not need to specify a $$y$$ -- in fact, you should not -- because the value of $$y$$ is constrained to be (in this instance) $$x^2$$.

You can do **symbolic manipulation** to directly substitute $$g(x)$$ into $$F(x)$$ to get $$F(x)=x.x^2=x^3$$. Either way, that's how function composition works. We introduce this because it will be used to build in the constraints for our optimisation problem, and we will see how **taking derivatives of composed functions translates to the dot product of linear transformations**.

## Gradients and Tangent Spaces
Let us continue with the above example. Assume that:

$$
f(x,y)=xy \\
y=g(x)=x^2
$$

Then the notation $$F=f\circ g$$ represents function composition, where $$F(x)=f(x,g(x))$$, as we have already stated.

Now, if we wanted to find $$D_xf(x,g(x))$$, it is trivial to see that substituting $$x^2$$ for $$y$$ in $$f(x,y)$$ gives us $$f(x)=x^3$$, therefore:

$$
D_xf(x,g(x))=3x^2
$$

However, let's look at the **Chain Rule** of differentiation for the above $$f(x,y)$$, because in our proofs, the actual form of $$f(x,y)$$ and $$g(x)$$ will not be available, and thus we will have to use the Chain Rule to express any results. We have only **one free variable** for this composite function, i.e., $$x$$, so we may write:

$$
D_xf(x,g)=\frac{\partial f(x,g)}{\partial x} \\
=\frac{df(x, g)}{d{[x\hspace{3mm} g]}^T}.\frac{d{[x\hspace{3mm} g]}^T}{dx} \\
$$

Let's denote $$\Phi (x)=\begin  {bmatrix}x \\ g\end{bmatrix}$$.
$${[x \hspace{3mm} g]}^T$$ is a vector so we are partially differentiating $$f(x,y)$$. In our example above, we can write for the **first term**, and substitute:

$$
\frac{f(x,g)}{d[x \hspace{3mm} g]}=\left[\frac{\partial f(x,g)}{\partial x} \hspace{3mm} \frac{\partial f(x,g)}{\partial g} \right] \\
\Rightarrow D_xf(x,y)=\left[\frac{\partial f(x,g)}{\partial x} \hspace{3mm} \frac{\partial f(x,g)}{\partial g} \right].\frac{d\Phi (x)}{dx}
$$

For the **second term**, we may write:

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

Yes, that was a complicated way of computing the same result we got earlier, but I want you to see the mechanics involved in applying the **Chain Rule** in the case of partial derivatives. More importantly, let us revisit the intermediate expression we used, namely:

$$
D_xf(x,g)=\left[\frac{\partial f(x,g)}{\partial x} \hspace{3mm} \frac{\partial f(x,g)}{\partial g} \right].\frac{d\Phi (x)}{dx}
$$

If you notice carefully, **the expression $$\left[\frac{\partial f(x,g)}{\partial x} \hspace{3mm} \frac{\partial f(x,g)}{\partial g} \right]$$ exactly represents the gradient operator $$\nabla f(x,g(x))$$**.

Furthermore, look at $$\frac {d\Phi (x)}{dx}=\begin{bmatrix}1 \\ \frac{dg}{dx}\end{bmatrix}$$. We can recognise this as the **parametric form of the line $$\mathbf{y=g'(x).x}$$**. This is because any vector on the tangent can be represented as $$\begin{bmatrix}1 \\ \frac{dg}{dx}\end{bmatrix}.t$$.

As a consequence, **this is the tangent space of $$g(x)$$**. If we represent $$\frac {d\Phi (x)}{dx}$$ as $$T_x$$ (the tangent space), we can rewrite the identity as:

**$$
D_xf(x,g)=\nabla f(x,g(x)).T_x
$$**

**Important Note**: Note that $$T_x$$ is **not** the normal vector to the tangent, but the actual vector along the tangent.

If we have a point $$P$$ which satisfies $$g(x)$$, i.e., has the coordinates $$(x_0, g(x_0))$$, then the following holds:

**$$
D_xf(P)=\nabla f(P).T_x
$$**

**The above expression represents the dot product of the gradient vector and the tangent vector at a point P which exists on the curve of the function defined by $$f(x,y)=xy$$ and satisfies the constraint $$g(x)=x^2$$.**

The above simple two-dimensional case will serve as our starting point. We now generalise in two conceptual directions.

## Generalisation to Multiple Constraints
Recall what we spoke about systems of linear equations in [Vector Calculus: Graphs, Level Sets, and Linear Manifolds]({%post_url 2021-04-20-vector-calculus-simple-manifolds%}). Specifically, if there are $$n$$ variables, and $$n-k$$ equations, we can parametrically specify $$n-k$$ variables in terms of the other $$k$$ free variables.

We can generalise the above dot product derivation for this general case. Before we do this, in order to avoid the ugly-looking $$n-k$$ expression, we restate the above as:

If there are $$N$$ variables, and $$n$$ equations, we can parametrically specify $$n$$ variables in terms of the other $$m$$ free variables. Also, obviously, $$m+n=N$$.

Now we write:

$$
\Phi=\begin{bmatrix}
u_1 \\
u_2 \\
u_3 \\
\vdots \\
u_m \\
\phi_1(u_1, u_2, u_3, ..., u_m) \\
\phi_2(u_1, u_2, u_3, ..., u_m) \\
\phi_3(u_1, u_2, u_3, ..., u_m) \\
\vdots \\
\phi_n(u_1, u_2, u_3, ..., u_m)
\end{bmatrix}

=\begin{bmatrix}
u_1 \\
u_2 \\
u_3 \\
\vdots \\
u_m \\
\phi_1 \\
\phi_2 \\
\phi_3 \\
\vdots \\
\phi_n
\end{bmatrix} \\

f(u_1, u_2, u_3, ..., u_m, v_1, v_2, v_3, ..., v_n)
$$

You should be able to recognise both of the above as the higher-dimensional analogs of the simple two-dimensional case we looked at earlier.

$$
F(u_1, u_2, u_3, ..., u_m)=f \circ G \\
D_UF=\frac{\partial f}{\partial (u_1, u_2, u_3, ..., u_m)} \\
=\frac{\partial f}{\partial (u_1, u_2, u_3, ..., u_m, \phi_1, \phi_2, \phi_3, ..., \phi_n)}.\frac{\partial G}{(u_1, u_2, u_3, ..., u_m)}
$$

The first one immediately reduces to $$D_uf$$, where $$u=(u_1, u_2, u_3, ..., u_m)$$. Let's look at the second term, because that is going to take the derivative of $$G$$, which is no longer a simple function, but a matrix of functions.

$$
\frac{\partial G}{(u_1, u_2, u_3, ..., u_m)}=
\begin{bmatrix}
\frac{\partial u_1}{\partial u_1} && \frac{\partial u_1}{\partial u_2} && \frac{\partial u_1}{\partial u_3} && ... && \frac{\partial u_1}{\partial u_m} \\
\frac{\partial u_2}{\partial u_1} && \frac{\partial u_2}{\partial u_2} && \frac{\partial u_2}{\partial u_3} && ... && \frac{\partial u_2}{\partial u_m} \\
\frac{\partial u_3}{\partial u_1} && \frac{\partial u_3}{\partial u_2} && \frac{\partial u_3}{\partial u_3} && ... && \frac{\partial u_3}{\partial u_m} \\
\vdots && \vdots && \vdots && \vdots && \vdots \\
\frac{\partial u_m}{\partial u_1} && \frac{\partial u_m}{\partial u_2} && \frac{\partial u_m}{\partial u_3} && ... && \frac{\partial u_m}{\partial u_m} \\
\\
\frac{\partial \phi_1}{\partial u_1} && \frac{\partial \phi_1}{\partial u_2} && \frac{\partial \phi_1}{\partial u_3} && ... && \frac{\partial \phi_1}{\partial u_m} \\
\frac{\partial \phi_2}{\partial u_1} && \frac{\partial \phi_2}{\partial u_2} && \frac{\partial \phi_2}{\partial u_3} && ... && \frac{\partial \phi_2}{\partial u_m} \\
\frac{\partial \phi_3}{\partial u_1} && \frac{\partial \phi_3}{\partial u_2} && \frac{\partial \phi_3}{\partial u_3} && ... && \frac{\partial \phi_3}{\partial u_m} \\
\vdots && \vdots && \vdots && \vdots && \vdots \\
\frac{\partial \phi_n}{\partial u_1} && \frac{\partial \phi_n}{\partial u_2} && \frac{\partial \phi_n}{\partial u_3} && ... && \frac{\partial \phi_n}{\partial u_m} 
\end{bmatrix} \\
$$

Yes, that is a lot of partial derivatives. But, as you can guess, most of this will be dramatically simplified. Note the first $$m$$ rows: all but one column in each of those $$m$$ rows will become zero. This simplifies to:

$$
\frac{\partial G}{(u_1, u_2, u_3, ..., u_m)}=
\begin{bmatrix}
1 && 0 && 0 && ... && 0 \\
0 && 1 && 0 && ... && 0 \\
0 && 0 && 1 && ... && 0 \\
\vdots && \vdots && \vdots && \vdots && \vdots \\
0 && 0 && 0 && ... && 1 \\
\\
\frac{\partial \phi_1}{\partial u_1} && \frac{\partial \phi_1}{\partial u_2} && \frac{\partial \phi_1}{\partial u_3} && ... && \frac{\partial \phi_1}{\partial u_m} \\
\frac{\partial \phi_2}{\partial u_1} && \frac{\partial \phi_2}{\partial u_2} && \frac{\partial \phi_2}{\partial u_3} && ... && \frac{\partial \phi_2}{\partial u_m} \\
\frac{\partial \phi_3}{\partial u_1} && \frac{\partial \phi_3}{\partial u_2} && \frac{\partial \phi_3}{\partial u_3} && ... && \frac{\partial \phi_3}{\partial u_m} \\
\vdots && \vdots && \vdots && \vdots && \vdots \\
\frac{\partial \phi_n}{\partial u_1} && \frac{\partial \phi_n}{\partial u_2} && \frac{\partial \phi_n}{\partial u_3} && ... && \frac{\partial \phi_n}{\partial u_m}
\end{bmatrix} \\
$$

To simplify notation further, we can collapse a lot of the above:
$$
\frac{\partial G}{\partial U}=
\begin{bmatrix}
I_{m \times m} \\
{\phi'(U)}_{n \times m}
\end{bmatrix}
$$

Plugging this back into the equation for $$D_UF$$, we get:

$$
\mathbf{
D_UF=D_{(U,V)}f_{1 \times (m+n)}.\begin{bmatrix}
I_{m \times m} \\
{\phi'(U)}_{n \times m}
\end{bmatrix} \\
= \nabla f.T_X
}
$$

where $$\mathbf{
T_X=\begin{bmatrix}
I_{m \times m} \\
{\phi'(U)}_{n \times m}
\end{bmatrix}
}$$

This is still in the same form as the simple case that we described above. The above resolves to $$1 \times m$$ matrix.

## Generalisation to Nonlinear Functions

## Implicit Function Theorem
Functions come in many shapes and sizes. They aren't always necessarily linear. However, that does not mean that analysis of these nonlinear functions is intractable. Calculus makes the mostly nonlinear world around us, and tells us that we can treat any curve or surface, in any dimension, as linear if we only zoom into it close enough. It basically asks us to pretend that a complicated curve (and the corresponding function) is a linear function. This approximation is grossly wrong at larger scales, but gets better and better the more we zoom in.

In optimisation, the constraints aren't always going to be linear.
Every level set is a graph, i.e., for every equation, you can find the graph of a function where $$n-k$$ variables in the level set equation may be expressed as functions of $$k$$ independent variables, i.e., such a mapping exists locally. We could not have made the assumption of the existence of this mapping for the general case of nonlinear constraints without the Implicit Function Theorem.

## Lagrangian Reformulation without Paramterisation
---SOME TEXT---

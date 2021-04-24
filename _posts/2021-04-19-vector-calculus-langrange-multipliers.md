---
title: "Vector Calculus: Manifolds, Constraints, and Lagrange Multipliers"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Optimisation", "Vector Calculus", "Lagrange Multipliers", "Theory"]
---

In this article, we take a detour to understand the mathematical intuition behind **Constrained Optimisation**, and more specifically the method of **Lagrangian multipliers**. We have been discussing **Linear Algebra**, specifically matrices, for quite a bit now. **Optimisation theory**, and **Quadratic Optimisation** as well, relies heavily on **Vector Calculus** for many of its results and proofs.

Most of the rules for single variable calculus translate over to vectors calculus, but now we are dealing with **vector-valued functions** and **partial differentials**.

This article will introduce the building blocks that we will need to reach our destination of understanding Lagrangians, with slightly more rigour than a couple of contour plots. **Please note that this is in no way an exhaustive introduction to Vector Calculus, only the concepts necessary to progress towards the stated goal are introduced.** If you're interested in studying the topic furhtre, there is a wealth of material to be found.

We will motivate most of the theory by illustrating the two-dimensional case in pictures, but understand that in actuality, we will often deal with **higher-dimensional vector spaces**.

Before we delve into the material proper, let's look at the big picture approach that you will go through as part of this.

- Required Review Material
- Orthogonal Complements
- Graphs and Level Sets
- Jacobians
- Tangent Spaces and Directional Derivatives
- Implicit Function Theorem
- Parameterisation in an underdetermined system of Linear Equations
- Constrained Critical Points
- Lagrangian Formulation without Paramterisation

## Required Review Material
- [Matrix Intuitions]({% post_url 2021-04-03-matrix-intuitions %})
- [Matrix Rank and Some Results]({% post_url 2021-04-04-proof-of-column-rank-row-rank-equality %})
- [Intuitions about the Orthogonality of Matrix Subspaces]({% post_url 2021-04-02-matrix-subspaces-intuitions %})

## Linear Algebra: Quick Recall and Some Identities
The reason we revisit this topic is because I want to introduce some new notation notation, and talk about a couple of properties you may or may not be aware of.

## Orthogonal Complements
We have already met **Orthogonal Complements** in [Intuitions about Matrix Subspaces]({% post_url 2021-04-02-matrix-subspaces-intuitions %}), when we were talking about **column spaces**, **row spaces**, **null spaces**, and **left null spaces**. Recalling quickly, the **column space** and the **left null space** are mutually orthogonal complements, and the **row space** and the **null space** are mutually orthogonal complements.

The other fact worth recalling is that the **column rank of a matrix is equal to its row rank**. Since the **rank of a matrix determines the dimensionality of a vector subspace** (1 if it is a line, 2 if it's a plane, and so on), it follows that the **column space and row space of a matrix have the same dimensionality**. Note that this dimensionality is not dependent on dimensionality of the vector space that the row/column space is embedded in, for example, a 1D subspace (a line) can exist in a three-dimensional vector space.

One (obvious) fact is that the dimensionality of the ambient vector space $$V$$ is equal to or larger than the dimensionality of its row space/column space/null space/left null space of $$A$$. That is:

$$dim(V)\geq dim(S) \| S\in {C(A), R(A), LN(A), N(A)}$$

We will make more precise statements about these relationships in the next section on $$Rank-Nullity Theorem$$.

## Rank Nullity Theorem
The **Rank Nullity** Theorem states that the sum of the dimensionality of the column space (rank) and that of its orthogonal complement, **left null space**, (**nullity**) is equal to the **dimension of the vector space they are embedded in**.
By the same token, the the dimensionality of the **row space** (**rank**) and that of its orthogonal complement, the **null space**, (**nullity**) is equal to the **dimension of the vector space they are embedded in**.

Mathematically, this implies:

$$
\mathbf{dim(C(A))+dim(LN(A))=dim(V) \\
dim(R(A))+dim(N(A))=dim(V)}
$$

where $$V$$ is the embedding space. This always ends up as the number of basis vectors required to uniquely identify a vector in $$V$$. To take a motivating example:

- A vector $$U=(1,2,3)$$ requires three basis vectors ($$(1,0,0), (0,1,0), (0,0,1)$$) to specify it completely in $${\mathbb{R}}^3$$. Note that this is choice of basis vectors is not unique; I simply picked the **standard basis vectors** to illustrate the point. Thus $$dim(U)$$ in this case is 3.
- The dimensionality of $$V=(1,2,3)$$ is 1, since it is a vector. The column space of this "matrix" $$\begin{bmatrix} 1 \\ 2 \end{bmatrix}$$ is basically a straight line extending infinitely long in both directions of this vector.
- **To fully cover the ambient vector space $$V={\mathbb{R}}^3$$**, you need a vector space which is a plane, which is a two-dimensional subspace. This is mechanically deducible from the **Rank-Nullity Theorem** (3-1=2), but you can also intuit that **the entire 3D space $$V$$ can be covered by taking a plane and translating it infinitely forwards and backwards along the vector $$U$$**.
- The plane we would like to pick is the orthogonal complement of $$V$$, i.e., the plane $$P=x+2y+3z=0$$. Can you see why it's an orthogonal complement? Taking the dot product of the plane $$P$$ and vector $$V$$ gives us zero.
- It is also clear that to represent this plane in the form of a vector subspace (i.e., matrix form), we need **two linearly independent 3D column vectors**. This will automatically imply that the rank of this matrix is 2, which validates the conclusion that we drew earlier from the **Rank-Nullity Theorem**.

### Subset containership of Orthogonal Complements
This rule is pretty simple, it states the following:

**If $$V\subset U$$, then $$V^{\perp}\supset U^{\perp}$$**

We illustrate this with a simple motivating example, but without proof.
If $$\vec{U}=(1,0,0)$$, then its orthogonal complement $$U^{\perp}$$ is the plane $$x=0$$, as shown below:
![Vector U and Plane U-Perp](/assets/images/vector-u-and-its-orthogonal-complement-plane-u-perp.png)

If $$V$$ is the plane $$z=0$$, then its orthogonal complement is $$\vec{V^{\perp}}=(0,0,1)$$, as shown below:
![Plane V and Vector V-Perp](/assets/images/plane-v-and-its-orthogonal-complement-vector-v-perp.png)

Now, the relation $$\mathbf{V\subset U}$$ clearly holds, since the vector $$\vec{U}=(1,0,0)$$ exists in the plane $$V=z=0$$, as shown below:

![Plane V Contains Vector U](/assets/images/plane-v-containing-vector-u.png)

Similarly, the relation $$\mathbf{V^{\perp}\supset U^{\perp}}$$ clearly holds, since $$U^{\perp}=x=0$$ contains the vector $$\vec{V^{\perp}}=(0,0,1)$$, as shown below:

![Plane U-Perp Contains Vector V-Perp](/assets/images/plane-u-perp-containing-vector-v-perp.png)

This validates the **Subset Rule**.

## Graphs and Level Sets
Optimisation problems boil down to maximising (or minimising) a function of (usually) multiple variables. The form of the function has a bearing on how easy or hard the solution might be. This section introduces some terminology that further discussion will refer to, and it is best to not admit any ambiguity in some of these terms.

### Graph of a Function
Consider the function $$f(x)=x^2$$. By itself, it represents a single curve. This is a function of variable. However, consider what the function $$g(x)=(x,f(x))$$ looks like. This notation might seem unfamiliar -- after all, isn't a function supposed to output a single value? -- but we have already dealt with functions that return multiple values; we just bundled them up in matrices. So it is the case in this scenario: $$g(x)$$ takes in a value in $$\mathbb{R}$$ and returns a matrix (usually a column vector, for the sake of consistency) of the form
$$\begin{bmatrix}
x \\
f(x)
\end{bmatrix}
$$. Later on, when we introduce vector-valued functions, $$x$$ and $$f(x)$$ will themselves be column vectors in their own right.

### Level Set of a Function
Consider the function $$f(x_1,x_2)=x_1^2-x_2$$. This is a function of two independent variables $$x_1$$ and $$x_2$$. Note that this function is very similar to $$y=x^2$$ (which can be rewritten as $$x^2-y=0$$), except there is no constraint on the output value. If we wish to observe the shape of this function, we will need to look at it in 3D: two dimensions for the independent variables, and the third for the output of the function $$f(x_1,x_2)$$.



---SOME TEXT---

## Jacobians
---SOME TEXT---

## Tangent Spaces and Directional Derivatives
---SOME TEXT---

## Implicit Function Theorem
---SOME TEXT---

## Parameterisation in an underdetermined system of Linear Equations
---SOME TEXT---

## Constrained Critical Points
---SOME TEXT---

## Lagrangian Reformulation without Paramterisation
---SOME TEXT---

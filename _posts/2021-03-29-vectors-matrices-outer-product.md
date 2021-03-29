---
title: "Vectors and Matrices: Different Interpretations"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Linear Algebra"]
---

Matrix multiplication is an important topic, both from the point of view of intuition and computation, and this is not an exaggeration. You're not simply multiplying and adding rows and columns of numbers (well, you are, but that is just the mechanics of obtaining the result). A lot of insights may be gleaned from looking at this basic operation from multiple viewpoints. I'll discuss some of these viewpoints, and will show the geometric picture associated with them.

## Vectors
But even before discussing the outer product operation (matrix multiplication), it is very instructive to look at a single **vector** (a special case of a matrix).

The convention for defining vectors you will find in every textbook/paper is as a column of numbers (basically a **column vector**). We'd write a vector $$v{1}$$ as:

$$
v{1}=\begin{bmatrix}
x{1} \\
x{2} \\
\end{bmatrix}
$$

Vectors implicitly originate from the origin (in this case, $$(0,0)$$). We take

$$
v{2}=\begin{bmatrix}
x{11} && x{12} \\
x{21} && x{22} \\
\end{bmatrix}
$$

You can look at $$v{2}$$ in two ways:

A set of two column vectors, i.e.,
$$
\begin{bmatrix}
x{11} \\
x{21} \\
\end{bmatrix}
$$,
$$
\begin{bmatrix}
x{12} \\
x{22} \\
\end{bmatrix}
$$

or, a set of two row vectors, i.e.,
$$
\begin{bmatrix}
x{11} && x{12} \\
\end{bmatrix}
$$,
$$
\begin{bmatrix}
x{21} && x{22} \\
\end{bmatrix}
$$

The **column vector** picture is usually more prevalent. Usually, if a column vector needs to be turned into a row vector (for purposes of multiplication if, say, you are trying to create a symmetric matrix), you'd still consider it as a column vector, and use the **transpose** operator, written as $$v^T$$.

## Lines, Planes, and Hyperplanes
Let's talk a little bit about lines, surfaces, and their higher dimensional variants (hyperplanes) and their normal vectors.

Here's a line $$6x+4y=0$$. I've also drawn its normal vector which is $$(6,4)$$. Now, note the direct correlation between the coefficients of the line equation and the normal vector. In fact, this is a very general rule, and we will see the reason for this right now.
![Line and its Normal Vector](/assets/line-and-normal-vector.png)

**Quick Aside**: Why is $$(6,4)$$ the normal vector? This is because any point on the line $$(6x+4y=0)$$ forms a vector with the origin, which is perpendicular to the normal vector (which obviously translates to the entire line being perpendicular to the $$(6,4)$$ vector). This is shown below.

![Line and its Normal Vector Relationships](/assets/line-and-normal-vector-relationships.png)

We haven't talked about the dot product yet (that comes later), but allow me to note a painfully obvious fact, any point on the line $$6x+4y=0$$, satisfies that equation. Indeed, you can see this clearly if we take $$(2,-3)$$ as an example, and write:

$$6.(2)+4.(-3)=0$$

Let us interpret this simple calculation in another way. This is like taking two vectors $$(6,4)$$ and $$(2,-3)$$ and multiplying their individual components, and summing up the results. $$(2,-3)$$ is obviously the vector we chose, and $$(6,4)$$ is another vector, which in this case, is...our normal vector.

This operation of multiplying individual components of vectors, and summing them up, is the **dot product** operation. It is not obvious that taking the dot product of perpendicular vectors will always result in 0, but we will prove it later, and it is true in the general case.

Indeed, an alternate definition of a line is the **set of all vectors which are perpendicular to the normal vector** (in this case, $$(6,4)$$). This is why a line (and as we will see, a plane, and hyperplanes) can be characterised by a single vector.

**Quick Aside**: There are alternate ways to express a line (or plane or hyperplane) using vectors which involve the **column space** interpretation, but I defer that discussion to the Vector Subspaces topic.

The dot product operation is denoted as $$A\cdot B$$, and I'll defer more discussion on the dot product to its specific topic.



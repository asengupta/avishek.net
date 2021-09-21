---
title: "Real Analysis Proofs #1"
author: avishek
usemathjax: true
tags: ["Real Analysis", "Mathematics", "Proof", "Pure Mathematics"]
---

Since I'm currently self-studying **Real Analysis**, I'll be listing down proofs I either initially had trouble understanding, or enjoyed proving, here. These are very mathematical posts, and are for personal documentation, mostly.

## Recursive Definitions
Source: **Analysis 1** by *Terence Tao*

## Definitions
- A natural number is any element in the set $$\mathbb{N}:=\{0,1,2,3,...\}$$.

## Peano Axioms Used

1. $$0$$ is a natural number.
2. If $$n$$ is a natural number, $$\mathbf{n++}$$ is also a natural number.
3. $$0$$ is not the successor to any natural number, i.e., $$n++ \neq 0, \forall n\in\mathbb{N}$$.
4. Different natural numbers must have different successors. If $$m\neq n$$, then $$m++ \neq n++$$. Conversely, if $$m++ \neq n++$$, then $$m=n$$.

## Proposition
Suppose there exists a function $$f_n:\mathbb{R}\rightarrow\mathbb{R}$$. Let $$c\in\mathbb{N}$$. Then we can assign a unique natural number $$a_n$$ for each natural number $$n$$, such that $$a_0=c$$, and $$a_{n++}=f_n(a_n) \forall n\in\mathbb{N}$$.

### Proof by Induction
**For zero**

Let $$0$$ be assigned $$a_0=c$$.
Then, $$a_{0++}=f_0(a_0)$$. Since $$0$$ is never a successor to any natural number by Axiom (3), $$a_0$$ will not recur as for $$a_{0++}$$.

**For $$n$$**

From Axiom (4), we can infer that:

$$
n++\neq n,n-1,n-2,...,1,0 \\
\Rightarrow a_{n++}\neq a_n,a_{n-1},a_{n-2},...,a_1,a_0
$$

Thus, $$a_{n++}$$ is unique in the set $$\{a_0,a_1,a_2,...,a_n,a_{n++}\}$$.

**For $$n++$$**

By extension, for $$(n++)++$$, we can write:

$$
(n++)++\neq n++,n,n-1,n-2,...,1,0 \\
\Rightarrow a_{(n++)++}\neq a_{n++},a_n,a_{n-1},a_{n-2},...,a_1,a_0
$$

Thus, $$a_{(n++)++}$$ is unique in the set $$\{a_0,a_1,a_2,...,a_n,a_{n++},a_{(n++)++}\}$$. Thus, we can assign a unique natural number $$a_{(n++)++}$$ such that $$a_{(n++)++}=f_{n++}(a_{n++})$$.

$$\blacksquare$$

## Proof of Existence of Real Cube Roots

Let $$r\in\mathbb{N}$$
For the case of $$r=0$$, the cube root is $$0$$.

Consider the set $$\mathbb{S}=\{x:x^3\leq r, x\in \mathbb{R}, r\in \mathbb{R}\}$$.

This set is non-empty because $0\in\mathbb{S}$. It is also bounded by $$r+1$$ because $$(r+1)^3=r^3+3r^2+3r+1>r$$.

Therefore, by the **Completeness Axiom**, $$\mathbb{S}$$ has a least upper bound. Denote this least upper bound by $$x$$.

By the **Trichotomy property**, these are the possible cases:

- **Case 1**: $$x^3<r$$
- **Case 2**: $$x^3>r$$
- **Case 3**: $$x^3=r$$.

**Case 1**: Assume that: $$\mathbb{x^3<r}$$

Then, by our definition of $$\mathbb{S}$$, **$$x\in\mathbb{S}$$ and is its least upper bound**, i.e., **there are no elements in $$\mathbb{S}$$ which are greater than $$x$$**.

If the cube of the least upper bound $$x$$ is less than $$r$$, then it is enough to show that there exists a $$x+\delta:\delta>0$$ whose cube is also less than $$r$$.

Assume that $$0<\delta<1$$. There can exist $$\delta>1$$, but that would restrict the choice of upper bounds we have to play about with:

Then, we'd like to find a $$0<\delta<1$$ such that $$(x+\delta)^3<r$$. This gives us:

$$
(x+\delta)^3<r \\
x^3+\delta^3+3x^2\delta+3x\delta^2<r \\
(x^3-r)+\delta^3+3x^2\delta+3x\delta^2<0
$$

We know that $$3x\delta^2<3x\delta$$ is a positive quantity, and note that $$\delta^3<\delta$$, thus we can say:

$$
(x^3-r)+\delta^3+3x^2\delta+3x\delta^2<(x^3-r)+\delta+3x^2\delta+3x\delta
$$

Then, it is enough to prove that:

$$(x^3-r)+\delta+3x^2\delta+3x\delta<0$$

With some algebraic manipulation, we get:

$$
(x^3-r)+\delta+3x^2\delta+3x\delta<0 \\
\Rightarrow \delta(1+3x^2+3x)<r-x^3 \\
\Rightarrow \delta<\frac{r-x^3}{1+3x^2+3x}
$$

If we assume $$\delta=\frac{1}{k}:k\in\mathbb{N}$$, then we can say:

$$
k>\frac{1+3x^2+3x}{r-x^3}: k\in\mathbb{N}
$$

Since the **Archimedean property** states that natural numbers have no upper bound, $$k$$ must exist.
This means, we have proven that there is a $$k\in\mathbb{N}$$, for which there exists a cube root $$(x+\frac{1}{k})$$ which is larger than $$x$$, such that $$(x+\frac{1}{k})^3<r$$. **This implies that $$(x+\frac{1}{k})$$ exists in $$\mathbb{S}$$**. However, this contradicts our assumption that no element greater than $$x$$ exists in $$\mathbb{S}$$.

**Thus, the statement $$x^3<r$$ is false.**

**Case 2**: Assume that: $$\mathbb{x^3>r}$$

If the cube of the least upper bound $$x$$ is greater than $$r$$, then it is enough to show that there exists a $$x-\delta:\delta>0$$ whose cube is also greater than $$r$$.

Assume that $$0<\delta<1$$. There can exist $$\delta>1$$, but that would restrict the choice of upper bounds we have to play about with:

Then, we'd like to find a $$0<\delta<1$$ such that $$(x+\delta)^3>r$$. This gives us:

$$
(x-\delta)^3>r \\
x^3-\delta^3-3x^2\delta+3x\delta^2>r \\
(x^3-r)-\delta^3-3x^2\delta+3x\delta^2>0 \\
$$

Again note that since $$\delta^3<\delta$$, and $$3x\delta^2$$ is positive, we can write:

$$
(x^3-r)-\delta^3-3x^2\delta+3x\delta^2>(x^3-r)-\delta-3x^2\delta \\
$$

Thus it is enough to prove that:

$$
(x^3-r)-\delta-3x^2\delta>0
$$

Some algebraic manipulation gives us:

$$
(1+3x^2)\delta<x^3-r \\
\Rightarrow \delta<\frac{x^3-r}{1+3x^2}
$$

If we assume $$\delta=\frac{1}{k}:k\in\mathbb{N}$$, then we can say:

$$
k>\frac{1+3x^2}{x^3-r}: k\in\mathbb{N}
$$

Since the **Archimedean property** states that natural numbers have no upper bound, $$k$$ must exist.
This means, we have proven that there is a $$k\in\mathbb{N}$$, for which there exists a cube root $$(x-\frac{1}{k})$$ which is smaller than $$x$$, such that $$(x-\frac{1}{k})^3<r$$. **Thus, $$(x+\frac{1}{k})$$ is a least upper bound for $$\mathbb{S}$$**; however, this contradicts our assumtion that $$x$$ is the least upper bound.

**Thus, the statement $$x^3>r$$ is false.**

Thus, the only possibility is that **Case 3** is true, i.e., $$x^3=r$$, thus implying the existence of real cube roots of real numbers.

$$\blacksquare$$

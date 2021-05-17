---
title: "Real Analysis Proofs"
author: avishek
usemathjax: true
tags: ["Real Analysis", "Mathematics", "Proof"]
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

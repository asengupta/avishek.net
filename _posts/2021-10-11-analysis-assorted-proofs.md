---
title: "Assorted Analysis Proofs"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Analysis", "Pure Mathematics"]
draft: false
---

This post lists assorted proofs from **Analysis**, without any particular theme.

#### Prove that if $$S$$ is open, $$S'$$ is closed.

**Proof:**

We claim that if $$S$$ is open, $$S'$$ is closed.
Thus, we'd like to prove that for a sequence $$(x_k) \in S'$$:

$$
\text{lim}_{k \rightarrow \infty} (x_k)= x_0 \in S'
$$

We will prove this by contradiction.

Assume that $$\require{cancel} x_0 \notin S'$$. Then, $$x_0 \in S$$.

Since $$S$$ is open, there exists an $$r>0$$, such that $$d(x_0,p)<r$$; that is, there exists an $$r$$-neighbourhood around $$x_0$$ in $$S$$.

Choose $$\epsilon<r$$, then there exists $$N \in \mathbb{N}$$, such that for all $$k>N$$, $$d(x_k, x_0)<\epsilon<r$$.

Thus, there exist $$x_k$$'s in the $$r$$-neighbourhood of $$x_0$$. **Thus, for $$k>N$$, $$x_k \in S$$, which contradicts our initial assumption that $$(x_k) \in S'$$.**

Thus, $$x_0 \in S'$$.
Since $$(x_k)$$ is an arbitrary sequence in $$S'$$, $$S'$$ contains the limit points of all sequences within it.

**Hence $$S'$$ is closed.**

$$\blacksquare$$

#### Let $$x,y \in \mathbb{R}$$. If $$y-x>1$$, then show there exists $$z \in \mathbb{Z}$$ such that $$x<z<y$$.

**Proof:**

Consider the set $$U=\{u:u<y, u \in \mathbb{Z}\}$$.

Since $$U$$ is bounded from above by $$y$$, it has a least upper bound, call it $$U_\text{sup}$$.

We note that $$y-U_\text{sup}<1$$. This is because if $$y-U_\text{sup} > 1$$, then $$y-(U_\text{sup}+1) > 0$$, implying the $$U_\text{sup}$$ is not the largest $$x \in U$$ which satisfies $$x<y$$, which is a contradiction.

Thus, we can write:

$$
y-U_\text{sup}<1 \\
\Rightarrow U_\text{sup}-y>-1
$$

Adding the above identity to $$y-x>1$$, we get:

$$
U_\text{sup}-x>0 \\
\Rightarrow U_\text{sup}>x \\
x<U_\text{sup}<y
$$

Thus, we have found an $$z \in \mathbb{Z}$$ which satisfies $$x<z<y$$.

You can prove the same thing by assuming $$V=\{v:v>x, x \in \mathbb{Z}\}$$ and taking $$\text{inf } V$$, and performing a similar procedure.

$$\blacksquare$$

#### Let $$Y$$ be a subset of $$X$$. Show that $$x \in \bar{Y} \Leftrightarrow B(x,r) \cap Y \neq \emptyset, \forall r>0$$.

**Proof:**

This can be proved using the contrapositive which states that:

$$
x \notin \bar{Y} \Leftrightarrow \exists r>0, B(x,r) \cap Y = \emptyset
$$

Assume that $$x \notin \bar{Y}$$. Then $$x \in X \setminus \bar{Y}$$. Since $$\bar{Y}$$ is closed, $$X \setminus \bar{Y}$$ is open. Then, there is an open ball of radius $$r>0$$, such that $$B(x,r) \subset X \setminus \bar{Y}$$. Since $$\subset X \setminus \bar{Y} \cap \bar{Y} = \emptyset$$, this implies that $$B(x,r) \cap \bar{Y} = \emptyset$$. Since $$Y \subseteq \bar{Y}$$, we get $$B(x,r) \cap Y = \emptyset$$.

Assume that $$\exists r>0, B(x,r) \cap Y = \emptyset$$. Then $$X \setminus B(x,r)$$ is a closed set and contains $$\bar{Y}$$. Since $$x \in B(x,r)$$, $$\bar{Y} \subset X \setminus B(x,r)$$, and $$B(x,r) \cap X \setminus B(x,r) = \emptyset$$, we have $$x \notin X \setminus B(x,r)$$, and consequently, $$x \notin \bar{Y}$$.

$$\blacksquare$$

---
title: "Functional Analysis Exercises : Distance Metrics"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics"]
draft: false
---

This post lists solutions to many of the exercises in the **Distance Metrics section 1.1** of *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**, as well as some assorted ones.

#### 1.1.2. Does $$d(x,y)={(x-y)}^2$$ define a metric on the set of all real numbers?
**Proof:**

For the distance metric $$d(x,y)={(x-y)}^2$$, we need to prove or disprove the **Triangle Inequality**:

$$
d(x,z) \leq d(x,y) + d(y,z)
$$

We start with the term $${(x-z)}^2$$, as follows:

$$
{(x-z)}^2 = {((x-y)+(y-z))}^2 \\
\Rightarrow {(x-z)}^2 = {(x-y)}^2 + {(y-z)}^2 + 2 (x-y) (y-z) \\
\Rightarrow d(x,z) = d(x,y) + d(y,z) + \underbrace{2 (x-y) (y-z)}_\text{positive or negative}
$$

When the term $$2 (x-y) (y-z)$$ is **positive**:

$$
d(x,z) > d(x,y) + d(y,z)
$$

When the term $$2 (x-y) (y-z)$$ is **negative** or **zero**:

$$
d(x,z) \leq d(x,y) + d(y,z)
$$

**Thus, the *Triangle Inequality* can only be satisfied for specific values of $$x$$, $$y$$, and $$z$$; hence $$d(x,y)={(x-y)}^2$$ is not a valid distance metric.**

$$\blacksquare$$

#### 1.1.3. Show that $$d(x,y)=\sqrt{|x-y|}$$ defines a metric on the set of all real numbers.
**Proof:**

For the distance metric $$d(x,y)=\sqrt{\vert x-y \vert}$$, we need to prove the **Triangle Inequality**:

$$
d(x,z) \leq d(x,y) + d(y,z)
$$

We start with the basic **Triangle Inequality** for $$\mathbb{R}$$:

$$
|x-z| \leq |x-y| + |y-z|
$$

Adding and subtracting $$2 \sqrt{\vert x-y \vert \vert y-z \vert}$$ on the RHS we get:

$$
|x-z| \leq |x-y| + |y-z| + 2 \sqrt{\vert x-y \vert \vert y-x \vert} - 2 \sqrt{\vert x-y \vert \vert y-z \vert} \\
\Rightarrow |x-z| \leq {(\sqrt{\vert x-y \vert} + \sqrt{\vert y-z \vert})}^2 - \underbrace{2 \sqrt{\vert x-y \vert \vert y-z \vert}}_\text{positive}
$$

Setting $$C=2 \sqrt{\vert x-y \vert \vert y-z \vert}>0$$, we get:

$$
|x-z| \leq {\left(\sqrt{\vert x-y \vert} + \sqrt{\vert y-z \vert}\right)}^2 - C \\
\Rightarrow {\left(\sqrt{\vert x-z \vert}\right)}^2 \leq {\left(\sqrt{\vert x-y \vert} + \sqrt{\vert y-z \vert}\right)}^2 - C \\
\Rightarrow {\left(\sqrt{\vert x-z \vert}\right)}^2 + C \leq {\left(\sqrt{\vert x-y \vert} + \sqrt{\vert y-z \vert}\right)}^2 \\
\Rightarrow {\left(\sqrt{\vert x-z \vert}\right)}^2 \leq {\left(\sqrt{\vert x-y \vert} + \sqrt{\vert y-z \vert}\right)}^2 \\
\Rightarrow \sqrt{\vert x-z \vert} \leq \sqrt{\vert x-y \vert} + \sqrt{\vert y-z \vert} \\
\Rightarrow d(x,z) \leq d(x,y) + d(y,z)
$$

**Hence, this proves the *Triangle Inequality*, and consequently, $$d(x,y)=\sqrt{\vert x-y \vert}$$ is a valid distance metric.**

$$\blacksquare$$

#### 1.1.5. Let $$d$$ be a metric on $$X$$. Determine all constants $$k$$ such that *(i)* $$kd$$, *(ii)* $$d+k$$ is a metric on $$X$$.

**(i) Proof:**

Let $$\bar{d}(x,y)=kd(x,y)$$ be a candidate metric on $$X$$. For it to be a valid distance metric, it must satisfy the four axioms of a metric, i.e.:

- $$\bar{d}(x,y)>0$$ if $$x \neq y$$: For this to hold for $$x \neq y$$, we must have $$k>0, k \in \mathbb{R}$$.
- $$\bar{d}(x,y)=0$$ if and only if $$x=y$$: For this to hold, we can have $$k \in \mathbb{R}$$.
- $$\bar{d}(x,y)=\bar{d}(y,x)$$: For this to hold, we can have $$k \in \mathbb{R}$$.
- $$\bar{d}(x,z) \leq \bar{d}(x,y) + \bar{d}(y,z)$$: For this, if we multiply the original, valid Triangle Inequality by $$k$$ on both sides, we have the following:
  
  $$
  kd(x,z) \leq k[d(x,y) + d(y,z)] \\
  \Rightarrow kd(x,z) \leq kd(x,y) + kd(y,z) \\
  \Rightarrow \bar{d}(x,z) \leq \bar{d}(x,y) + \bar{d}(y,z)
  $$

  proving that the **Triangle Inequality** holds for $$\bar{d}$$ for any $$k \in \mathbb{R}$$.

Putting all of these together, we get the condition that $$k>0, k \in \mathbb{R}$$.

**(ii) Proof:**

Let $$\bar{d}(x,y)=d(x,y) + k$$ be a candidate metric on $$X$$. For it to be a valid distance metric, it must satisfy the four axioms of a metric, i.e.:

- $$\bar{d}(x,y)>0$$ if $$x \neq y$$: For this to hold for $$x \neq y$$, we must have $$k>0, k \in \mathbb{R}$$.
- $$\bar{d}(x,y)=0$$ if and only if $$x=y$$: For this to hold, we must have $$d(x,y)+k=0$$. Since $$d(x,y)=0$$ already, $$k=0, k \in \mathbb{R}$$.
- $$\bar{d}(x,y)=\bar{d}(y,x)$$: For this to hold, we can have $$k \in \mathbb{R}$$.
- $$\bar{d}(x,z) \leq \bar{d}(x,y) + \bar{d}(y,z)$$: For this, we can note the following:
  
  $$
  d(x,z) + k \leq k[d(x,y) + d(y,z)] + 2k \\
  \Rightarrow d(x,z) + k \leq [d(x,y) + k] + [d(y,z) + k] \\
  \Rightarrow \bar{d}(x,z) \leq \bar{d}(x,y) + \bar{d}(y,z)
  $$

  proving that the **Triangle Inequality** holds for $$\bar{d}$$ for any $$k \in \mathbb{R}$$.

Putting all of these together, we get the condition that $$k=0, k \in \mathbb{R}$$.

#### 1.1.6. Show that $$d$$ in 1.1-6 satisfies the triangle inequality.
[TODO]

#### 1.1.8. Show that another metric $$\bar{d}$$ on the set $$X$$ in 1.1-7 is defined by $$\bar{d}(x,y)=\int\limits_a^b |x(t) - y(t)| dt$$.
[TODO]

#### 1.1.9. Show that $$d$$ in 1.1-8 is a metric.
[TODO]

#### 1.1.10. **(Hamming Distance)** Let $$X$$ be the set of all ordered triples of zeros and ones. Show that $$X$$ consists of eight elements and a metric $$d$$ on $$X$$ is defined by $$d(x,y)=$$ number of places where $$x$$ and $$y$$ have different entries. (This space and similar spaces of $$n$$-tuples play a role in switching and automata theory and coding. $$d(x,y)$$ is called the *Hamming distance* between $$x$$ and $$y$$; cf. the paper by R. W. Hamming (1950) listed in Appendix 3.)

**Proof:**

The set of all ordered triples of zeros and ones, forms a sequence of numbers $$X=[0,7], x_i \in X, x_i \in \mathbb{Z}$$ in their binary form. Thus, it is evident that the number of triples is $$2^3=8$$.

Let $$a,b,c \in {0,1}$$, then each number in this set, can be represented as $$x_i=a+2b+4c$$. Then the suggested distance metric is:

$$
d(x,y)=|a_x-a_y|+|b_x-b_y|+|c_x-c_y|
$$

Let $$x_1$$, $$x_2$$, $$x_3$$ be defined as follows:

$$
x=a_x+2b_x+4c_x \\
y=a_y+2b_y+4c_y \\
z=a_z+2b_z+4c_z
$$

$$
\begin{align}
d(x,z) &= |a_x-a_z|+|b_x-b_z|+|c_x-c_z| \\
&=|a_x-a_y+a_y-a_z|+|b_x-b_y+b_y-b_z|+|c_x-c_y+c_y-c_z|
\end{align}
$$

Now we have the following inequalities:

$$
|a_x-a_y+a_y-a_z| \leq |a_x-a_y|+|a_y-a_z| \\
|b_x-b_y+b_y-b_z| \leq |b_x-b_y|+|b_y-b_z| \\
|c_x-c_y+c_y-c_z| \leq |c_x-c_y|+|c_y-c_z|
$$

Summing up these inequalities, and noting that the LHS resolves to $$d(x,z)$$, we get:

$$
d(x,z) \leq |a_x-a_y|+|a_y-a_z| + |b_x-b_y|+|b_y-b_z| + |c_x-c_y|+|c_y-c_z| \\
\Rightarrow d(x,z) \leq (|a_x-a_y|+|b_x-b_y|+|c_x-c_y|) + (|a_y-a_z|+|b_y-b_z|+|c_y-c_z|) \\
\Rightarrow d(x,z) \leq d(x,y) + d(y,z)
$$

**Thus, this proves the Triangle Inequality for the Hamming Distance as a metric.**

$$\blacksquare$$

#### 1.1.12. **(Triangle inequality)** The triangle inequality has several useful consequences. For instance, using the generalised triangle inequality, show that $$|d(x,y) - d(z,w)| \leq d(x,z) + d(y,w)$$.

We write the two following triangle inequalities. One involves $$x$$, $$y$$, $$z$$. The other one involves $$w$$, $$y$$, $$z$$.

$$
d(x,y) \leq d(x,z) + d(z,y)
$$

$$
\begin{equation}
d(z,y) \leq d(z,w) + d(w,y)
\label{eq:1-1-12-1}
\end{equation}
$$

Adding $$d(x,z)$$ to $$\eqref{eq:1-1-12-1}$$, we get:

$$
d(x,z) + d(z,y) \leq d(x,z) + d(z,w) + d(w,y)
$$

Thus, we have:

$$
d(x,y) \leq d(x,z) + d(z,y) \leq  d(x,z) + d(z,w) + d(w,y) \\
\Rightarrow d(x,y) \leq  d(x,z) + d(z,w) + d(w,y) \\
\Rightarrow d(x,y) \leq  d(x,z) + d(z,w) + d(y,w)  \text{(by the Symmetry property of a Distance Metric)}
$$

$$
\begin{equation}
d(x,y) - d(z,w) \leq  d(x,z) + d(w,y)
\label{eq:1-1-12-abs-1}
\end{equation}
$$

We write the two following triangle inequalities. One involves $$x$$, $$z$$, $$w$$. The other one involves $$x$$, $$y$$, $$w$$.

$$
d(z,w) \leq d(z,x) + d(x,w)
$$

$$
\begin{equation}
d(x,w) \leq d(x,y) + d(y,w)
\label{eq:1-1-12-2}
\end{equation}
$$

Adding $$d(z,x)$$ to $$\eqref{eq:1-1-12-2}$$, we get:

$$
d(z,x) + d(x,w) \leq d(z,x) + d(x,y) + d(y,w)
$$

Thus, we have:

$$
d(z,w) \leq d(z,x) + d(x,w) \leq  d(z,x) + d(x,y) + d(y,w) \\
\Rightarrow d(z,w) \leq  d(z,x) + d(x,y) + d(y,w) \\
\Rightarrow d(z,w) \leq  d(x,z) + d(x,y) + d(y,w)  \text{(by the Symmetry property of a Distance Metric)}
$$

$$
\begin{equation}
d(z,w) - d(x,y) \leq  d(z,x) + d(y,w)
\label{eq:1-1-12-abs-2}
\end{equation}
$$

Summarising $$\eqref{eq:1-1-12-abs-1}$$ and $$\eqref{eq:1-1-12-abs-2}$$, we get:

$$
d(x,y) - d(z,w) \leq  d(x,z) + d(y,w) \\
d(z,w) - d(x,y) \leq  d(x,z) + d(y,w) \\
\Rightarrow \mathbf{ |d(x,y) - d(z,w)| \leq d(x,z) + d(y,w) }
$$

$$\blacksquare$$

#### 1.1.13. Using the triangle inequality, show that $$|d(x,z) - d(y,z)| \leq d(x,y)$$.

**Proof:**

We have to show that:

$$
|d(x,z) - d(y,z)| \leq d(x,y)
$$

We write the following Triangle Inequality:

$$
\begin{equation}
d(x,z) \leq d(x,y) + d(y,z) \\
\Rightarrow d(x,z) - d(y,z) \leq d(x,y)
\label{eq:1-1-13-abs-1}
\end{equation}
$$

The other Triangle Inequality we write is:

$$
\begin{equation}
d(y,z) \leq d(y,x) + d(x,z) \\
\Rightarrow d(y,z) - d(x,z) \leq d(y,x) \\
\Rightarrow d(y,z) - d(x,z) \leq d(x,y) \text{(by the Symmetry property of a Distance Metric)}
\label{eq:1-1-13-abs-2}
\end{equation}
$$

Summarising the results of $$\eqref{eq:1-1-13-abs-1}$$ and $$\eqref{eq:1-1-13-abs-2}$$, we get:

$$
d(x,z) - d(y,z) \leq d(x,y) \\
d(y,z) - d(x,z) \leq d(x,y) \\
\Rightarrow \mathbf{|d(x,z) - d(y,z)| \leq d(x,y)}
$$

$$\blacksquare$$

#### 1.1.14. **(Axioms of a metric)** (M1) to (M4) could be replaced by other axioms (without changine the definition). For instance, show that (M3) and (M4) could be obtained from (M2) and $$d(x,y) \leq d(z,x) + d(z,y)$$.

For reference, the axioms **(M1)** to **(M4)** are as follows:

- **(M1)** $$0 \leq d(x,y)<\infty, d(x,y)\in \mathbb{R}$$
- **(M2)** $$d(x,y)=0$$ if and only if $$x=y$$
- **(M3)** $$d(x,y)=d(y,x)$$
- **(M4)** $$d(x,z) \leq d(x,y) + d(y,z)$$

**Proof for (M3):**

The allowed assumptions are:

- **(A1)** $$d(x,y)=0$$ if and only if $$x=y$$
- **(A2)** $$d(x,y) \leq d(z,x) + d(z,y)$$


Set $$z=y$$ in **(A2)**, so that we get:

$$
\require{cancel}
\begin{equation}
d(x,y) \leq d(y,x) + \cancel{d(y,y)} \text{ (by (A1))} \\
\Rightarrow d(x,y) \leq d(y,x)
\label{eq:1-1-14-abs-1}
\end{equation}
$$

From **(A2)**, we get $$d(y,x)$$ as:

$$
\begin{equation}
d(y,x) \leq d(z,y) + d(z,x)
\label{eq:1-1-14-y-x}
\end{equation}
$$

Set $$z=x$$ in $$\eqref{eq:1-1-14-y-x}$$ again, so that we get:

$$
\begin{equation}
d(y,x) \leq d(x,y) + \cancel{d(x,x)} \text{ (by (A1))} \\
\Rightarrow d(y,x) \leq d(x,y) \\
\Rightarrow d(x,y) \geq d(y,x)
\label{eq:1-1-14-abs-2}
\end{equation}
$$

Summarising the results of $$\eqref{eq:1-1-14-abs-1}$$ and $$\eqref{eq:1-1-14-abs-2}$$, we get:

$$
d(x,y) \leq d(y,x) \\
d(x,y) \geq d(y,x)
$$

This implies that:

$$
\begin{equation}
d(x,y)=d(y,x)
\label{eq:1-1-14-symmetry}
\end{equation}
$$

$$\blacksquare$$

**Proof for (M4):**

**(M4)** should immediately follow from $$\eqref{eq:1-1-14-symmetry}$$, since:

$$
d(x,y) \leq d(z,x) + d(z,y)
\Rightarrow d(x,y) \leq d(x,z) + d(z,y)
$$

$$\blacksquare$$

#### 1.1.15. Show that nonnegativity of a metric follows from (M2)to (M4).

**Proof:**

We have to prove that: $$d(x,y) \geq 0$$ follows from **(M2)** and **(M4)**.

- **(M2)** $$d(x,y)=0$$ if and only if $$x=y$$
- **(M3)** $$d(x,y)=d(y,x)$$
- **(M4)** $$d(x,z) \leq d(x,y) + d(y,z)$$

From the Triangle Inequality **(M4)**, we have:

$$
d(x,y) \leq d(x,z) + d(z,y)
$$

Set $$x=y$$, then we get:

$$
d(y,y) \leq d(y,z) + d(z,y) \\
\Rightarrow \underbrace{\cancel{d(y,y)}}_\text{by (M2)} \leq d(y,z) + \underbrace{d(y,z)}_\text{by (M3)} \\
\Rightarrow 2d(y,z) \geq 0 \\
\Rightarrow d(y,z) \geq 0
$$

This proves **(M1)**.

$$\blacksquare$$

## Assorted Proofs

#### Prove that if $$S$$ is open, $$S'$$ is closed.

**Proof:**

We claim that if $$S$$ is open, $$S'$$ is closed.
Thus, we'd like to prove that for a sequence $$(x_k) \in S'$$:

$$
\text{lim}_{k \rightarrow \infty} (x_k)= x_0 \in S'
$$

We will prove this by contradiction.

Assume that $$x_0 \cancel{\in} S'$$. Then, $$x_0 \in S$$.

Since $$S$$ is open, there exists an $$r>0$$, such that $$d(x_0,p)<r$$; that is, there exists an $$r$$-neighbourhood around $$x_0$$ in $$S$$.

Choose $$\epsilon<r$$, then there exists $$N \in \mathbb{N}$$, such that for all $$k>N$$, $$d(x_k, x_0)<\epsilon<r$$.

Thus, there exist $$x_k$$'s in the $$r$$-neighbourhood of $$x_0$$. **Thus, for $$k>N$$, $$x_k \in S$$, which contradicts our initial assumption that $$(x_k) \in S'$$.**

Thus, x_0 \in S'.
Since $$(x_k)$$ is an arbitrary sequence in $$S'$$, $$S'$$ contains the limit points of all sequences within it.

**Hence $$S'$$ is closed.**

$$\blacksquare$$

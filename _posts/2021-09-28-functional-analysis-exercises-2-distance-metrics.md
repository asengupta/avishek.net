---
title: "Functional Analysis Exercises 2 : Distance Metrics"
author: avishek
usemathjax: true
tags: ["Mathematics", "Proof", "Functional Analysis", "Pure Mathematics"]
draft: false
---

This post lists solutions to many of the exercises in the **Distance Metrics section 1.2** of *Erwin Kreyszig's* **Introductory Functional Analysis with Applications**.

For reference, the axioms **(M1)** to **(M4)** for a distance metric are as follows:

- **(M1)** $$0 \leq d(x,y)<\infty, d(x,y)\in \mathbb{R}$$
- **(M2)** $$d(x,y)=0$$ if and only if $$x=y$$
- **(M3)** $$d(x,y)=d(y,x)$$
- **(M4)** $$d(x,z) \leq d(x,y) + d(y,z)$$

#### 1.2.1. Show that in 1.2-1 we can obtain another metric by replacing $$\frac{1}{2^i}$$ with $$\mu_i>0$$ such that $$\sum\mu_i$$ converges.

**Proof**: The space referred to is the space of all bounded and unbounded sequences.

The candidate metric is defined as:

$$
d(x,y)=\displaystyle\sum_{i=1}^\infty \mu_i\frac{|x_i-y_i|}{1+|x_i-y_i|}
$$

**(M1)** $$d(x,y)$$ is bounded, non-negative, and real.

We know that $$\displaystyle\sum_{i=1}^\infty \mu_i$$ converges, thus if we prove $$\lambda_i \mu_i<mu_i$$, we will have proved that $$\displaystyle\sum_{i=1}^\infty \lambda_i \mu_i$$ converges, and is thus real, bounded.

Indeed, if we examine $$d(x,y)$$, we can make the following observation:

$$
d(x,y)=\displaystyle\sum_{i=1}^\infty \mu_i\underbrace{\frac{|x_i-y_i|}{1+|x_i-y_i|}}_{0\leq\lambda<1}
$$

Thus, $$d(x,y)$$ is bounded and real. $$d(x,y)$$ is also nonnegative because $$0\leq\lambda<1$$ and $$\mu_i>0$$.

**(M2)** $$d(x,y)=0$$ if and only if $$x=y$$.

This is evident since:

$$
d(x,x)=\displaystyle\sum_{i=1}^\infty \mu_i\frac{|x_i-x_i|}{1+|x_i-x_i|}=0
$$

**(M3)** $$d(x,y)=d(y,x)$$

This is easily seen since the modulus sign guarantees that:$$\vert x_i-y_i\vert=\vert y_i-x_i \vert$$, and thus $$d(x,y)=d(y,x)$$.

**(M4)** $$d(x,z) \leq d(x,y) + d(y,z)$$

For convenience of notation, let us denote use the following notation:

$$
A=|x_i-y_i| \\
B=|y_i-z_i| \\
C=|z_i-y_i|
$$

We'd like to prove that:

$$
\require{cancel}
\frac{A}{1+A} \leq \frac{B}{1+B} + \frac{C}{1+C} \\
= \frac{B+C+2BC}{(1+B)(1+C)} \\
\Rightarrow A(1+B)(1+C) \leq (B+C+2BC)(1+A) \\
\Rightarrow A+\cancel{CA}+\cancel{AB}+\cancel{ABC} \leq B+C+2BC+\cancel{AB}+\cancel{CA}+\cancel{2}ABC \\
\Rightarrow A \leq B+C+2BC+ABC \\
\Rightarrow |x_i-y_i| \leq |x_i-z_i|+|z_i-y_i|+2BC+ABC
$$

Thus, we need to prove that:

$$
|x_i-y_i| \leq |x_i-z_i|+|z_i-y_i|+2BC+ABC
$$

where $$A,B,C \geq 0$$.

We already know from the **Triangle Inequality** that:

$$
\begin{align*}
|x_i-y_i| &= |x_i-z_i+z_i-y_i| \\
|x_i-y_i| &\leq |x_i-z_i|+|z_i-y_i| \\
\Rightarrow |x_i-y_i| &\leq |x_i-z_i|+|z_i-y_i|+2BC+ABC
\end{align*}
$$

Thus, we have:

$$
\frac{A}{1+A} \leq \frac{B}{1+B} + \frac{C}{1+C}
$$

Multiplying throughout by $$\mu_i$$, and summing over $$i$$, we have:

$$
\displaystyle\sum_{i=1}^\infty\mu_i\frac{A}{1+A} \leq \sum_{i=1}^\infty\mu_i\frac{B}{1+B} + \sum_{i=1}^\infty\mu_i\frac{C}{1+C} \\
\Rightarrow d(x,y) \leq d(x,z) + d(z,y)
$$

Thus $$d(x,y)$$ is a metric.

$$\blacksquare$$

#### 1.2.2. Using (6), show that the geometric mean of two positive numbers does not exceed the arithmetic mean.

**Proof**:
From the identity involving conjugate exponents, we know that:

$$
\alpha \beta \leq \frac{\alpha^p}{p} + \frac{\beta^q}{q} \\
\Rightarrow 2\alpha \beta \leq \frac{\alpha^p}{p} + \frac{\beta^q}{q} + \alpha \beta
$$

Set $$p=2$$, then we get $$q=2$$, so that we get:

$$
2\alpha \beta \leq \frac{\alpha^2}{2} + \frac{\beta^2}{2} + \alpha \beta \\
\Rightarrow 4\alpha \beta \leq \alpha^2 + \beta^2 + 2\alpha \beta \\
\Rightarrow \alpha \beta \leq {\left(\frac{\alpha + \beta}{2}\right)}^2 \\
\Rightarrow \sqrt{\alpha \beta} \leq \frac{\alpha + \beta}{2}
$$

This proves that the Geometric Mean of two numbers cannot exceed their Arithmetic Mean.

$$\blacksquare$$

#### 1.2.3. Show that the Cauchy-Schwarz inequality (11) implies
  $${(|\xi_1| + \cdots + |\xi_n|)}^2 \leq n ({|\xi_1|}^2 + \cdots + {|\xi_n|}^2)$$.

**Proof**: The Cauchy-Schwarz Inequality is :

$$
\displaystyle\sum_{i=1}^n |x_i y_i| \leq {\left( \displaystyle\sum_{i=1}^n {|x_i|}^2 \right)}^\frac{1}{2}
\bullet
{\left( \displaystyle\sum_{i=1}^n {|y_i|}^2 \right)}^\frac{1}{2}
$$

Set $$y_i=1$$, so that we get:

$$
\displaystyle\sum_{i=1}^n |x_i| \leq {\left( \displaystyle\sum_{i=1}^n {|x_i|}^2 \right)}^\frac{1}{2} \\
\Rightarrow {\left(\displaystyle\sum_{i=1}^n |x_i|\right)}^2 \leq \displaystyle\sum_{i=1}^n {|x_i|}^2 \\
\Rightarrow {(|x_1| + \cdots + |x_n|)}^2 \leq n ({|x_1|}^2 + \cdots + {|x_n|}^2)
$$

$$\blacksquare$$

#### 1.2.4. (Space $$\ell^p$$) Find a sequence which converges to 0, but is not in any space $$\ell^p$$, where $$1\leq p<+\infty$$.

**Answer:**

$$(x_k)=\frac{1}{n}$$ is the key to creating sequences which converge, but whose corresponding series are not summable. The issue is that you cannot just use $$(x_k)=\frac{1}{n}$$ because for $$p>1$$, we have $$S_k=\displaystyle\sum {\left(\frac{1}{n}\right)}^p$$ will converge, and obviously we do not want that.
The trick is to introduce a numerator which will clearly show divergence of the series (using the Ratio Test for example), but then write the series as the **numerator** number of $$\frac{1}{n}$$ terms.

Let us take a simple example: consider the sequence $$(x_k)=\frac{n+1}{n}$$. The series is then written as:

$$
x_k=\{2, \frac{3}{2}, \frac{4}{3}, \frac{5}{4}, \frac{6}{5}, \cdots \}
$$

But the above could also be written like so:

$$
y_k=\{\underbrace{\frac{1}{1}, \frac{1}{1}}_{\text{Two }1}, \underbrace{\frac{1}{2}, \frac{1}{2}, \frac{1}{2}}_{\text{Three }\frac{1}{2}}, \underbrace{\frac{1}{3}, \frac{1}{3}, \frac{1}{3}, \frac{1}{3}}_{\text{Four }\frac{1}{3}}, \underbrace{\frac{1}{4}, \frac{1}{4}, \frac{1}{4}, \frac{1}{4}, \frac{1}{4}}_{\text{Five }\frac{1}{4}}, \cdots \}
$$

This shows that the limit of this sequence is zero, even though the "parent" series is clearly not summable. Breaking up the terms of a divergent series to create a new sequence which has a finite limit is the key idea here.

However, in the above example, the choice of numerator $$n+1$$ does not pass the Ratio Test for divergence, because $$\frac{x_{k+1}}{x_k}=\frac{n^2+2n}{n^2+2n+1}<1$$. We will thus need a bigger numerator. An additive factor will not do, because for all $$K>0$$, we will get $$\frac{x_{k+1}}{x_k}=\frac{2n}{n+1}>1$$, for $$n>1$$, which is always going to be the case for our series..

An exponential factor like $$2^n$$ will work as the numberator, because then we have $$\frac{x_{k+1}}{x_k}=\frac{n^2+2n}{n^2+2n+1}<1$$
**Note:** $$(n+1)$$ is not the only choice for a numerator. Anything that is clearly bigger than the denominator should do. $$2^n$$, for example is also a valid choice. The choice will affect how many $$\frac{1}{n}$$ terms will appear from each term in the original series.

$$
z_k=\{\underbrace{\frac{1}{1}, \frac{1}{1}}_{\text{Two }1}, \underbrace{\frac{1}{2}, \frac{1}{2}, \frac{1}{2}, \frac{1}{2}}_{\text{Four }\frac{1}{2}}, \underbrace{\frac{1}{3}, \frac{1}{3}, \frac{1}{3}, \frac{1}{3}, \frac{1}{3}, \frac{1}{3}, \frac{1}{3}, \frac{1}{3}}_{\text{Eight }\frac{1}{3}}, \cdots \}
$$

$$z_k$$ is an example of a sequence which converges to zero, but whose series diverges for all $$p \geq 1$$.

#### 1.2.5. Find a sequence $$x$$ which is in $$\ell^p$$ with p>1 but $$\require{cancel} x\cancel{\in}\ell^1$$.

**Answer:**

$$(x_i)=\frac{1}{i}$$ converges to 0.
$$\displaystyle\sum_{i=1}^\infty{(\frac{1}{i})}^p$$ diverges to $$\infty$$ for $$p\leq 1$$, and thus violates the condition for a sequence in $$\ell^p$$ spaces, i.e. $$\displaystyle\sum_{i=1}^\infty{\vert\xi_i\vert}^p<\infty$$ for $$p \leq 1$$.

#### 1.2.6. **(Diameter, bounded set)** The diameter $$\delta(A)$$ of a nonempty set A in a  metric space $$(X, d)$$ is defined to be $$\delta(A) = \text{sup } d(x,y)$$. A is said to be bounded if $$\delta(A)<\infty$$. Show that $$A\subset B$$ implies $$\delta(A)\leq \delta(B)$$.

**Proof:**

By the definition of set membership, we can say that if $$x,y \in A$$, and $$A\subset B$$, then $$x,y \in B$$.

Consider the set of all distances in $$A$$ and $$B$$, like so:

$$
\Delta_A={d(x,y): x,y \in A} \\
\Delta_B={d(u,v): u,v \in B}
$$

Since $$x,y\in A \Rightarrow x,y \in B$$, all $$d(x,y):x,y \in A$$ must exist in \Delta_B. Thus, we have:

$$
\Delta_A \subset \Delta_B
$$

Thus, $$\delta(A)=\text{sup } \Delta_A$$ exists in $$\Delta_B$$, i.e.,

$$
\begin{equation}
\delta(A) \in \Delta_B
\label{eq:diameter-A-in-DeltaB}
\end{equation}
$$

By the definition of the diameter of a bounded set, we have:

$$
\begin{equation}
\delta_B=\text{sup } \Delta_B
\label{eq:diameter-B-sup-DeltaB}
\end{equation}
$$

Putting $$\eqref{eq:diameter-A-in-DeltaB}$$ and $$\eqref{eq:diameter-B-sup-DeltaB}$$ together, implies that:

$$
\delta(A) \leq \delta(B)
$$

**Verbal Reasoning**: $$\delta(A)$$ is *a* member of $$B$$, while $$\delta(B)$$ is the least upper bound of $$B$$, thus $$\delta(A)$$ has to be less than or, at the most, equal to this least upper bound $$\delta(B)$$.

$$\blacksquare$$

#### 1.2.7. Show that $$\delta(A)=0$$ *(cf. Prob. 6)* if and only if A consists of a single point.

**Proof:**

Assume $$A=\{x\}$$.  
Then the set $$\Delta_A={d(x,x)}$$.  
Therefore $$\delta(A)=\text{sup} \Delta_A=d(x,x)$$.

By the definition of a distance metric, $$d(x,x)=0$$.

Thus, we get:

$$
\delta(A)=0
$$

$$\blacksquare$$

For the "only if" side of implication, Let $$\delta(A)=\text{sup } \Delta_A=0$$.

By the definition of a distance metric, $$d(x,y)\geq 0$$.  
Thus, all other elements of $$\Delta_A$$ have to be zero. This implies there is only one element in $$\Delta_A={0}$$.  
Therefore, distances between all points $$x \in A$$ must be zero.  

By the definition of a distance metric, $$d(x,y)=0$$ if $$x=y$$.

Therefore, all points must be equal to each other, i.e., there is only one point in $$A$$.

$$\blacksquare$$

#### 1.2.8. **(Distance between sets)** The distance $$D(A,B)$$ between two nonempty subsets $$A$$ and $$B$$ of a metric space $$(X, d)$$ is defined to be:


$$D(A,B) = \text{inf } d(a, b)$$.

#### Show that $$D$$ does not define a metric on the power set of $$X$$. (For this reason we use another symbol, $$D$$, but one that still reminds us of $$d$$.)

**Proof**
Consider $$A={3,4}$$, and $$B={4,5}$$.  
Then, we have:

$$
P(A)=\{\emptyset,\{3\}, \{4\}\, \{3,4\}\} \\
P(B)=\{\emptyset,\{4\}, \{5\}\, \{4,5\}\}
$$

Then $$D(P(A),P(B))=\text{inf } d(a,b)=0$$

However, $$P(A) \neq P(B)$$.

Thus, property **(M1)** of the distance metric is violated, and $$D$$ is not a valid distance metric on the power set of $$X$$.

$$\blacksquare$$

#### 1.2.9. If An $$A \cap B \neq \emptyset$$, show that $$D(A,B) = 0$$ in Prob. 8. What about the converse?

**Proof**

Let $$\Delta_{XY}=\{d(x,y):x \in A\, y \in B\}$$

Let $$A \cap B \neq \emptyset$$, then there exists at least one element $$p\in A,B$$.

Then, $$d(p,p) \in \Delta_{XY}$$ and $$d(p,p)=0$$.

Since a distance metric must be nonnegative, $$D(A,B)=\text{inf }\Delta_{XY}=0$$

$$\blacksquare$$

The converse is not true. To see why this is not true, remember that the infimum/supremum of a set does *not* have to belong to that set. Therefore, two sets can have the same infimum/supremum while still having their intersection be the null set.

Thus, consider the set formed by the sequence $$\{\frac{1}{2^n}\}$$. Then $$X=\{\frac{1}{2^n}\}$$ The infimum (and the limit point) of $$X$$ is $$0$$.

Now consider a second set $$Y={0}$$.

Then, let $$\Delta_{XY}$$ is the set of all distances between $$X$$ and $$Y$$, defined as:

$$
\begin{align*}
\Delta_{XY}&=\{d(x,y):x \in A, y \in B\} \\
&=\Bigl\{\frac{1}{2} - 0, \frac{1}{2^2} - 0, \frac{1}{2^3} - 0, \frac{1}{2^4} - 0, ...\Bigr\} \\
&=\Bigl\{\frac{1}{2}, \frac{1}{2^2}, \frac{1}{2^3}, \frac{1}{2^4}, ...\Bigr\}
\end{align*}
$$

We know then that:

$$
\text{lim }_{n\rightarrow\infty}\frac{1}{2^n}=0 \\
\Rightarrow \text{lim }_{n\rightarrow\infty}\Delta_{XY}=0 \\
$$

In this case, $$\Delta_{XY}$$ has no Least Upper Bound. Then, $$D{X,Y}$$ is:

$$
D(X,Y)=\text{inf }\Delta_{XY}=0 \\
$$

This is a case where $$D(X,y)=0$$, even though $$X\cap Y=\emptyset$$.  
Thus, we see that:

$$
\require{cancel}
D(X,Y)=0 \cancel\Rightarrow X\cap Y \neq \emptyset
$$

$$\blacksquare$$

#### 1.2.10. The distance $$D(x,B)$$ from a point $$x$$ to a non-empty subset $$B$$ of $$(X,d)$$ is defined to be

$$D(x,B)= \text{inf } d(x, b)$$

#### in agreement with Prob. 8. Show that for any $$x,y\in X$$,

$$
|D(x,B) - D(y,B)| \leq d(x, y)
$$.

**Proof:**

We have, from the **Triangle Inequality**,

$$
\begin{equation}
d(x,b) \leq d(x,y) + d(y,b) \\
\Rightarrow d(x,b)-d(y,b) \leq d(x,y)
\label{eq:1-2-10-1}
\end{equation}
$$

We also have:

$$
\begin{equation}
d(y,b) \leq d(y,x) + d(x,b) \\
\Rightarrow d(y,b) \leq d(x,y) + d(x,b) \\
\Rightarrow d(y,b)-d(x,b) \leq d(x,y)
\label{eq:1-2-10-2}
\end{equation}
$$

From the results of $$\eqref{eq:1-2-10-1}$$ and $$\eqref{eq:1-2-10-2}$$, we get:

$$
|d(x,b)-d(y,b)| \leq d(x,y)
$$

This is true for all $$d(x,b)$$ and $$d(y,b)$$, including their infimums, i.e.:

$$
|\text{inf } d(x,b)-\text{inf } d(y,b)| \leq d(x,y) \\
\Rightarrow |D(x,B)-D(y,B)| \leq d(x,y)
$$

$$\blacksquare$$

#### 1.2.11. If $$(X,d)$$ is any metric space, show that another metric on $$X$$ is defined by

$$
\bar{d}(x,y)=\frac{d(x,y)}{1+d(x,y)}
$$

#### and $$X$$ is bounded in the metric $$\bar{d}$$.

**Proof:**

The candidate metric is defined as:

$$
\bar{d}(x,y)=\frac{d(x,y)}{1+d(x,y)}
$$

**(M1)** $$0 \leq d(x,y)<\infty, d(x,y)\in \mathbb{R}$$

$$d(x,y)$$ is already a metric. Thus, $$d(x,y)$$ is nonnegative and bounded. Then \bar{d}(x,y) is also nonegative and bounded, by its definition.

**(M2)** $$d(x,y)=0$$ if and only if $$x=y$$

This is evident if we set $$d(x,y)=0$$ in the definition of $$\bar{d}(x,y)$$.

**(M3)** $$d(x,y)=d(y,x)$$

Since $$d(x,y)$$ is symmetric, substituting $$d(y,x)$$ in the definition of \bar{d}(x,y) shows that it is symmetric also.

**(M4)** $$d(x,z) \leq d(x,y) + d(y,z)$$

For convenience of notation, let us denote use the following notation:

$$
A=d(x,y) \\
B=d(x,z) \\
C=d(z,y)
$$

We'd like to prove that:

$$
\require{cancel}
\frac{A}{1+A} \leq \frac{B}{1+B} + \frac{C}{1+C} \\
= \frac{B+C+2BC}{(1+B)(1+C)} \\
\Rightarrow A(1+B)(1+C) \leq (B+C+2BC)(1+A) \\
\Rightarrow A+\cancel{CA}+\cancel{AB}+\cancel{ABC} \leq B+C+2BC+\cancel{AB}+\cancel{CA}+\cancel{2}ABC \\
\Rightarrow A \leq B+C+2BC+ABC \\
\Rightarrow d(x,y) \leq d(x,z)+d(z,y)+2BC+ABC
$$

Thus, we need to prove that:

$$
d(x,y) \leq d(x,z)+d(z,y)+2BC+ABC
$$

where $$A,B,C \geq 0$$.

We already know from the **Triangle Inequality** that:

$$
\begin{align*}
d(x,y) &\leq d(x,z)+d(z,y) \\
\Rightarrow d(x,y) &\leq d(x,z)+d(z,y)+2BC+ABC
\end{align*}
$$

Thus, we have:

$$
\frac{A}{1+A} \leq \frac{B}{1+B} + \frac{C}{1+C}
$$

Multiplying throughout by $$\mu_i$$, and summing over $$i$$, we have:

$$
\displaystyle\sum_{i=1}^\infty\mu_i\frac{A}{1+A} \leq \sum_{i=1}^\infty\mu_i\frac{B}{1+B} + \sum_{i=1}^\infty\mu_i\frac{C}{1+C} \\
\Rightarrow \bar{d}(x,y) \leq \bar{d}(x,z) + \bar{d}(z,y)
$$

Thus $$d(x,y)$$ is a metric.

$$\blacksquare$$

#### 1.2.12. Show that the union of two bounded sets A and B in a metric space is a bounded set. (Definition in Prob. 6.)

Let us define three sets of metrics:

$$
\Delta_A=\{d(x,y):x,y \in A\} \\ 
\Delta_B=\{d(x,y):x,y \in B\} \\ 
\Delta_{AB}=\{d(x,y):x \in A,y \in B\} \\ 
$$

Verbally,
- $$\Delta_A$$ is the set of all distances between points in $$A$$. $$\Delta_A$$ is bounded because $$A$$ is bounded by definition, therefore $$\delta(A)=\text{sup }\Delta_A<\infty$$.
- $$\Delta_B$$ is the set of all distances between points in $$B$$. $$\Delta_B$$ is bounded because $$B$$ is bounded by definition, therefore $$\delta(B)=\text{sup }\Delta_B<\infty$$.
- $$\Delta_{AB}$$ is the set of all distances between points in $$A$$ and points in $$B$$.

Then the set of all distances between points in $$C=A \cup B$$ is $$\Delta_C=\Delta_A \cap \Delta_B \cap \Delta_{AB}$$.

By definition, diameter of $$C$$ is:

$$
\delta(A \cup B) = \text{inf } \Delta_C
$$

From the axioms of the metric $$d$$, we note that $$\forall d(x,y) \in \Delta_A, \Delta_B, \Delta_{AB}, d(x,y)<\infty$$.

Then, we can deduce the following:

$$
\delta(A \cup B) = \text{inf } \Delta_C = \text{inf } \{d(x,y): d(x,y) \in \Delta_A, \Delta_B, \Delta_{AB}\} \\
\Rightarrow \delta(A \cup B) < \infty
$$

Hence, the union of two bounded sets is bounded.

$$\blacksquare$$

#### 1.2.13. **(Product of metric spaces)** The Cartesian product $$X = X_1 \times X_2$$ of two    metric spaces $$(X_1,d_1)$$ and $$(X_2,d_2)$$ can be made into a metric space $$(X,d)$$ in many ways. For instance, show that a metric $$d$$ is defined by

$$
\bar{d}(x,y)=d_1(x_1,y_1) + d_2(x_2,y_2)
$$

#### where $$x=(x_1,x_2)$$, $$y=(y_1,y_2)$$.

$$Proof:$$

The candidate distance metric is:

$$
\bar{d}(x,y)=d_1(x_1,y_1) + d_2(x_2,y_2)
$$

where $$d_1$$ and $$d_2$$ are already valid distance metrics.

**(M1)** $$0 \leq \bar{d}(x,y)<\infty, \bar{d}(x,y)\in \mathbb{R}$$

Because $$d_1$$ and $$d_2$$ are already valid metrics. Thus $$0<d_1(x,y)<\infty$$ and $$0<d_2(x,y)<\infty$$.

$$
\therefore 0<\bar{d}(x,y)<\infty
$$

Thus, \bar{d}(x,y) is nonegative and bounded, by its definition.

**(M2)** $$\bar{d}(x,y)=0$$ if and only if $$x=y$$

This is evident if we set $$x=y$$, then we have $$d_1(x,x)=d_2(x,x)=0$$ and thus $$\bar{d}(x,y)=0$$.

**(M3)** $$\bar{d}(x,y)=\bar{d}(y,x)$$

Since $$d_1(x,y)$$ and $$d_2(x,y)$$ are symmetric, substituting $$d_1(y,x)$$ and $$d_2(y,x)$$ in the definition of \bar{d}(x,y) shows that it is symmetric also.

**(M4)** $$\bar{d}(x,z) \leq \bar{d}(x,y) + \bar{d}(y,z)$$

$$
d_1(x_1,y_1) \leq d_1(x_1,z_1) + d_1(z_1, y_1) \\
d_2(x_2,y_2) \leq d_2(x_2,z_2) + d_2(z_2, y_2) \\
$$

Summing the above two identities, we see:

$$
d_1(x_1,y_1) + d_2(x_2,y_2) \leq d_1(x_1,z_1) + d_1(z_1, y_1) + d_2(x_2,z_2) + d_2(z_2, y_2) \\
\Rightarrow \underbrace{d_1(x_1,y_1) + d_2(x_2,y_2)}_{\bar{d}(x_2,z_2)} \leq \underbrace{d_1(x_1,z_1) + d_2(x_2,z_2)}_{\bar{d}(x,z)} + \underbrace{d_1(z_1, y_1) + d_2(z_2, y_2)}_{\bar{d}(z,y)} \\
\Rightarrow \bar{d}(x,y) \leq d_1(x,z) + d_2(z,y)
$$

Hence, $$\bar{d}$$ is a valid distance metric.

$$\blacksquare$$

#### 1.2.14. Show that another metric on $$X$$ in Prob. 13 is defined by

$$
\bar{d}(x,y)=\sqrt{ {d_1(x_1,y_1)}^2 + {d_1(x_2,y_2)}^2}
$$

#### 1.2.15. Show that a third metric on $$X$$ in Prob. 13 is defined by

$$
\bar{d}(x,y)=max[d_1(x_1,y_1), d_1(x_2,y_2)]
$$

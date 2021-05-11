---
title: "Support Vector Machines from First Principles: Linear SVMs"
author: avishek
usemathjax: true
tags: ["Machine Learning", "Support Vector Machines", "Optimisation", "Theory"]
---

We have looked at how **Lagrangian Multipliers** and how they help build constraints as part of the function that we wish to optimise. Their relevance in **Support Vector Machines** is how the constraints about the classifier margin (i.e., the supporting hyperplanes) is incorporated in the search for the **optimal hyperplane**.

We introduced the first part of the problem in [Support Vector Machines from First Principles: Part One]({% post_url 2021-04-14-support-vector-machines-derivations %}). We then took a detour through **Vector Calculus** and ***Constrained Quadratic Optimisation** to build our mathematical understanding for the succeeeding analysis.

The necessary background material for understanding this article is covered in:

- [Support Vector Machines from First Principles: Part One]({% post_url 2021-04-14-support-vector-machines-derivations %})
- Vector Calculus Background
    - [Vector Calculus: Graphs, Level Sets, and Constraint Manifolds]({% post_url 2021-04-20-vector-calculus-simple-manifolds %})
    - [Vector Calculus: Lagrange Multipliers]({% post_url 2021-04-24-vector-calculus-lagrange-multipliers %})
    - [Vector Calculus: Implicit Function Theorem and Inverse Function Theorem]({% post_url 2021-04-29-inverse-function-theorem-implicit-function-theorem %}) (**Note**: This covers more theoretical background)
- Quadratic Form and Motivating Problem
  - [Quadratic Optimisation: PCA as Motivation]({% post_url 2021-04-19-quadratic-form-optimisation-pca-motivation-part-one %})
  - [Conclusion to Quadratic Optimisation: PCA as Motivation]({% post_url 2021-04-28-quadratic-optimisation-pca-lagrange-multipliers %})
- Quadratic Optimisation
  - [Quadratic Optimisation: Mathematical Background]({% post_url 2021-05-08-quadratic-optimisation-theory %})
  - [Quadratic Optimisation: Karush-Kuhn-Tucker Conditions]({% post_url 2021-05-10-quadratic-form-optimisation-kkt %})
  
Before we proceed with the calculations, I'll restate the original problem again.

## 1. Support Vector Machine Problem Statement
For a set of data $$x_i, i\in[1,N]$$, if we assume that data is divided into two classes (-1,+1), we can write the constraint equations as:

$$
\mathbf{m_{max}=max \frac{2k}{\|N\|}}
$$

subject to the following constraints/;

$$
\mathbf{
N^Tx_i\geq b+k, \forall x_i|y_i=+1 \\
N^Tx_i\leq b-k, \forall x_i|y_i=-1
}
$$

We are also given a set of training examples $$x_i, i=1,2,...,n$$ which are already labelled either **+1** or **-1**. The important assumption here is that these training data points are linearly separable, i.e., there exists a hyperplane which divides the two categories, such that no point is misclassified. Our task is to find this hyperplane with the maximum possible margin, which will be defined by its supporting hyperplanes.

## 2. Restatement of the Support Vector Machine Problem Statement
Remembering the standard form of a Quadratic Programming problem, we want the objective function to be a minimisation problem, as well as a quadratic problem.

Furthermore, we'd like to set the constant $$k=1$$, and rewrite $$N$$ with $$w$$. Thus, the objective function may be rewritten as:

$$
\mathbf{min f(x)=\frac{w^Tw}{2}}
$$

since squaring $$w$$ does not affect the outcome of the minimisation problem.

We have two constraints; we'd like to rewrite them in the form $$g(x)\leq 0$$. Thus, we get:

$$
-(w^Tx_i-b)+1\leq 0, \forall x_i|y_i=+1\\
w^Tx_i-b+1\leq 0, \forall x_i|y_i=-1
$$


You will notice that they differ only in the sign of $$(w^Tx_i-b)$$, which is dependent on the reverse sign of $$y_i$$. We can collapse these two inequalities into a single one by using $$y_i$$ as a determinant of the sign.

$$
g_i(x)=\sum_{i=1}^n-y_i(w^Tx_i-b)+1\leq 0, \forall x_i|y_i\in\{-1,+1\}
$$

The Lagrangian then is:

$$
\mathbf{
L(w,\lambda,b)=f(x)+\lambda_i g_i(x)} \\
L(w,\lambda,b)=\frac{w^Tw}{2}+\sum_{i=1}^n\lambda_i [-y_i(w^Tx_i-b)+1] \\
\mathbf{
L(w,\lambda,b)=\frac{w^Tw}{2}-\sum_{i=1}^n\lambda_i [y_i(w^Tx_i-b)-1]
}
$$

for all $$x_i$$ such that $$\lambda_i\geq 0$$, $$g_i(x)\leq 0$$, and $$y_i\in\{-1,+1\}$$.

We have already assumed the Primal and Dual Feasibility Conditions above. The Dual Optimisation Problem is then:

$$
\text{max}_\lambda\hspace{4mm}\text{min}_{w,b} \hspace{4mm} L(w,\lambda,b)
$$

$$
\begin{equation}
\text{max}_\lambda\hspace{4mm}\text{min}_{w,b} \hspace{4mm} \frac{w^Tw}{2}-\sum_{i=1}^n\lambda_i [y_i(w^Tx_i-b)-1] \label{eq:lagrangian}
\end{equation}
$$

Note that the only constraints that will be activated will be the ones which are for points lying on the supporting hyperplanes.
Let's see what the KKT Stationarity Condition gives us.

$$
\frac{\partial L}{\partial w}=w-\sum_{i=1}^n \lambda_ix_iy_i
$$

Setting this partial differential to zero, we get:

$$
\begin{equation}
w^\ast=\sum_{i=1}^n \lambda_ix_iy_i \label{eq:weight}
\end{equation}
$$

If we denote $$w^\ast$$ as the optimal solution for $$w$$. Similarly, differentiating with respect to $$b$$, and setting it to zero, we get:

$$
\frac{\partial L}{\partial b}=0 \\
\Rightarrow \begin{equation}
\sum_{i=1}^n \lambda_iy_i=0 \label{eq:b-constraint}
\end{equation}
$$

This doesn't give us an expression for $$b$$ but does give us a specific condition that needs to be fulfilled by any point which lies on the supporting hyperplane.

Let us simplify the $$\eqref{eq:lagrangian}$$ in light of these new identities. We write:

$$
L(\lambda,w^\ast,b^\ast)=\frac{w^Tw}{2}+\sum_{i=1}^n\lambda_i [y_i(w^Tx_i-b)-1] \\
=\frac{w^Tw}{2}+\sum_{i=1}^n\lambda_i y_i w^Tx_i- \sum_{i=1}^n\lambda_i y_ib + \sum_{i=1}^n\lambda_i
$$

The term $$\sum_{i=1}^n\lambda_i y_ib$$ vanishes because of $$\eqref{eq:b-constraint}$$, so we get:

$$
L(\lambda,w^\ast,b^\ast)=\frac{w^Tw}{2}+\sum_{i=1}^n\lambda_i y_i w^Tx_i + \sum_{i=1}^n\lambda_i
$$

Applying the identity $$\eqref{eq:weight}$$ to this result, we get:

$$
L(\lambda,w^\ast,b^\ast)=\frac{1}{2} \sum_{i=1}^n\sum_{j=1}^n\lambda_i\lambda_jy_iy_jx_ix_j - \sum_{i=1}^n\sum_{j=1}^n\lambda_i\lambda_jy_iy_jx_ix_j + \sum_{i=1}^n \lambda_i \\
\mathbf{
L(\lambda,w^\ast,b^\ast)=\sum_{i=1}^n \lambda_i - \frac{1}{2} \sum_{i=1}^n\sum_{j=1}^n\lambda_i\lambda_jy_iy_jx_ix_j
}
$$

Thus, $$\lambda_i$$ can be solved by optimising $$L(\lambda,w^\ast,b^\ast)$$, that is:

$$
\lambda^\ast=\text{arginf}_\lambda L(\lambda,w^\ast,b^\ast) \\
\mathbf{
\lambda^\ast=\text{arginf}_\lambda \left[\sum_{i=1}^n \lambda_i - \frac{1}{2} \sum_{i=1}^n\sum_{j=1}^n\lambda_i\lambda_jy_iy_jx_ix_j\right]
}
$$

### Solving for $$b$$
We saw earlier that differentiating the Lagrangian with respect to $$b$$ gave us the constraint $$\eqref{eq:b-constraint}$$ but not a direct expression for $$b$$. Let us make the following observations:

- We already know $$w^\ast$$. Thus, we know the separating hyperplane through the origin, though we do not know $$b$$. In two dimensions, this would be the equivalent of the y-intercept.
- For the points labelled $$+1$$, the minimum value you get by plugging $$x_i$$ into $$w^\ast x$$ is definitely a point on the (as yet undetermined) positive supporting hyperplane $$H^+$$. You can have multiple points which achieve this minimum value; all of those points lie on $$H^+$$, which is obviously parallel to $$f(x)=w^\ast x$$.
- For the points labelled $$-1$$, the maximum value you get by plugging $$x_i$$ into $$w^\ast x$$ is definitely a point on the (as yet undetermined) negative supporting hyperplane $$H^-$$. You can have multiple points which achieve this maximum value; all of those points lie on $$H^-$$, which is obviously parallel to $$f(x)=w^\ast x$$.

Therefore, we may find $$b^+$$ and $$b^-$$ by finding:

- $$H^+$$ is the hyperplane with "slope" $$w^\ast$$ and passing through the point $$x^+$$ which gives the minimum value (positive or negative) for $$f(x)=w^\ast x$$. There may be multiple points like $$x^+$$; pick any one. $$H^+$$ will have y-intercept $$b^+$$.
- $$H^-$$ is the hyperplane with "slope" $$w^\ast$$ and passing through the point $$x^-$$ which gives the maximum value (positive or negative) for $$f(x)=w^\ast x$$. There may be multiple points like $$x^-$$; pick any one. $$H^-$$ will have y-intercept $$b^-$$.

$$H^+$$ and $$H^-$$ are the supporting hyperplanes. The situation is shown below.

We already saw in [Support Vector Machines from First Principles: Part One]({% post_url 2021-04-14-support-vector-machines-derivations %}) that the separating hyperplane $$H_0$$ lies midway between $$H^+$$ and $$H^-$$, implying that $$b^\ast$$ is the mean of $$b^+$$ and $$b^-$$. Thus, we get:

$$
\begin{equation}
b^\ast=\frac{b^++b^-}{2} \label{eq:b}
\end{equation}
$$

Note that at the end of our calculation, we will have arrived at ($$\lambda^\ast$$, $$w^\ast$$, $$b^\ast$$) as the optimal solution for the Lagrangian. Recall that by our assumptions of Quadratic Optimisation, this Lagrangian is a concave-convex function, and thus the primal and the dual optimum solutions coincide. In effect, this is the same solution that we'd have gotten if we'd solved the original optimisation problem.

Once the training has completed, categorising a new point from a test set, is done simply by finding:

$$
y_t=sgn[w^\ast x_t+b^\ast]
$$

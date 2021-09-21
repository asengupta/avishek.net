---
title: "Mercer's Theorem: Mathematical Preliminaries "
author: avishek
usemathjax: true
tags: ["Mathematics", "Theory", "Functional Analysis", "Mercer's Theorem"]
draft: true
---
This article lays the ground for taking a second perspective to **Kernel Functions** using **Mercer's Theorem**. We discussed this theorem in [Functional Analysis: Norms, Operators, and Some Theorems]({% post_url 2021-07-19-functional-analysis-results-for-operators %}) briefly. We will see that Mercer's Theorem applies somewhat more directly to the characterisation of Kernel Functions, and there is no need for an elaborate construction, like we do for **Reproducing Kernel Hilbert Spaces**. Before we do that, this post will lay out the mathematical concepts necessary for understanding the proof behind Mercer's Theorem.

The specific posts discussing the background are:

- [Kernel Functions: Functional Analysis and Linear Algebra Preliminaries]({% post_url 2021-07-17-kernel-functions-functional-analysis-preliminaries %})
- [Functional Analysis: Norms, Linear Functionals, and Operators]({% post_url 2021-07-19-functional-analysis-results-for-operators %})
- [Functional and Real Analysis Notes]({% post_url 2021-07-18-notes-on-convergence-continuity %})

It is also advisable (though not necessary) to review [Kernel Functions with Reproducing Kernel Hilbert Spaces]({% post_url 2021-07-20-kernel-functions-rkhs %}) to contrast and compare that approach with the one shown here.

## Mercer's Theorem Learning Roadmap

Here is the roadmap for understanding the concepts relating to Mercer's Theorem.

{% mermaid %}
graph TD;
sequences[Sequences]
function_sequences[Function Sequences]
compact_set[Compact Set]
compact_operators[Compact Operators]
linear_operators[Linear Operators]
uniform_convergence[Uniform Convergence]
banach_space[Banach Space]
hilbert_space[Hilbert Space]
lp_space[Lp Space]
jordan_canonical_form[Jordan Canonical Form]
relatively_compact_subspace[Relatively Compact Subpace]
spectral_theorem[Spectral Theorem of Compact Operators]
arzela_ascoli_theorem[Arzelà-Ascoli Theorem]
mercer_theorem[Mercer's Theorem]

sequences-->function_sequences-->uniform_convergence-->banach_space-->hilbert_space
compact_set-->relatively_compact_subspace
relatively_compact_subspace-->compact_operators
linear_operators-->compact_operators
hilbert_space-->lp_space-->compact_operators
compact_operators-->spectral_theorem
jordan_canonical_form-->spectral_theorem
spectral_theorem-->mercer_theorem
function_sequences-->arzela_ascoli_theorem
arzela_ascoli_theorem-->mercer_theorem

style compact_set fill:#006f00,stroke:#000,stroke-width:2px,color:#fff
style spectral_theorem fill:#006fff,stroke:#000,stroke-width:2px,color:#fff
style mercer_theorem fill:#8f0f00,stroke:#000,stroke-width:2px,color:#fff
style arzela_ascoli_theorem fill:#8f0ff0,stroke:#000,stroke-width:2px,color:#fff
{% endmermaid %}

## Sequences and Boundedness
## Cauchy Sequences
## Function Sequences
## Open and Closed Sets
## Compact Sets and Relatively Compact Subspaces
## Compact Operators
## Jordan Canonical Form
## Arzelà-Ascoli Theorem

## Older Stuff (will be deleted)
Recall what Mercer's Theorem states:

$$
\kappa(x,y)=\sum_{i=1}^\infty \lambda_i \psi_i(x)\psi_i(y)
$$

where $$\kappa(x,y)$$ is a positive semi-definite function and $$\psi_i(\bullet)$$ is the $$i$$th eigenfunction. Note that this implies that there are an infinite number of eigenfunctions.

## Evaluation Functionals

The Evaluation Functional is an interesting function: it takes another function as an input, and applies a specific argument to that function. As an example, if we have a function, like so:

$$
f(x)=2x+3
$$

We can define an evaluation functional called $$\delta_3(f)$$ such that:

$$
\delta_3(f)=f(3)=2.3+3=9
$$

## Continuity and Boundedness of Evaluation Functional
Here we will treat the Evaluation Functional in its functional form (the "formula view", if you like). Is the graph of the Evaluation Functional continuous. We can prove that if a linear functional is bounded, then it is also continuous. In this case, we will prove that the Evaluation Functional is bounded in the function space $$\mathcal{H}$$.


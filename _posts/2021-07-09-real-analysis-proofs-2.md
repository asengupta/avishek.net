---
title: "Real Analysis: Patterns for Proving Irrationality of Square Roots"
author: avishek
usemathjax: true
tags: ["Real Analysis", "Mathematics", "Proof"]
---
ld
Continuing on my journey through **Real Analysis**, we will focus here on common **proof patterns** which apply to **irrational square roots**.
These patterns apply to the following sort of proof exercises:

- Prove that $$\sqrt 2$$ is irrational.
- Prove that $$\sqrt 3$$ is irrational.
- Prove that $$\sqrt 7$$ is irrational.
- Prove that $$\sqrt 12$$ is irrational.
- ...etc.

The proofs are all based on **Proof by Contradiction**. **Thus, the starting point is always to assume that the square root of the number in question, say $$\sqrt 7$$, is indeed a rational number**, implying that it can be expressed as $$\frac{p}{q}$$, where $$p,q\in\mathbb{N}$$.

Let's take a specific example which will demonstrate the first proof pattern:

## Prove that $$\sqrt 2$$ is irrational

### Proof
We assume that $$\sqrt 2$$ is rational. Therefore, we can express it as a ratio of two integers $$\frac{p}{q}:p,q\in\mathbb{N}$$, which have no common factors between them. Thus, we may write:

$$
\frac{p^2}{q^2}=2 \
\Rightarrow p^2=2q^2
$$

The easiest templatised way to get started in all situations is to make a quick **truth table**, to narrow down the **feasibility of $$p, q$$ being odd and/or even**.

| p    | q    | Feasible? |  Reason |
|------|------|-----------|---------|
| Even | Even | False     | By definition |
| Even | Odd  | True      |  |
| Odd  | Even | False     | $$2q^2$$ is even, so $$p^2$$ must be even, thus $$p$$ must be even|
| Odd  | Odd  | False     |  $$2q^2$$ is even, so $$p^2$$ cannot be odd, thus $$p$$ cannot be odd|

Thus, the only valid option is $$p$$ even and $$q$$ odd.

At this point we'll show two different ways of arriving at a contradiction. The first one is the 'classic' proof.

#### <u>Proof Pattern 1</u>
Since $$p$$ is even, we set $$p=2k, k\in\mathbb{N}$$, so that:
$$
{(2k)}^2=2q^2 \\
\Rightarrow 4k^2=2q^2 \\
\Rightarrow 2k^2=q^2
$$

This implies that $$q^2$$ is even, therefore $$q$$ has to be even.
This leads us to a contradiction, since $$p$$ and $$q$$ cannot both be even.
The second contradiction this leads us to is that the truth table shows that $$q$$ is odd, but we concluded that $$q$$ is even.

Therefore, $$\sqrt 2$$ is not a rational number.
$$\blacksquare$$

**NOTE:** In the above proof, **we do not exploit the fact that $$q$$ is odd**. We only show by some algebraic manipulation that $$q$$ is even, which contradicts two of the facts deduced from the truth table.

#### <u>Proof Pattern 2</u>
Since we have determined $$p$$ even and $$q$$ odd, we set $$p=2m, m\in\mathbb{N}$$, and $$q=2n+1, n\in\mathbb{N}$$. Then, substituting these expressions, we get:

$$
p^2=2q^2 \\
\Rightarrow {(2m)}^2=2{(2n+1)}^2 \\
\Rightarrow 4m^2=2(4n^2+4n+1) \\
\Rightarrow \underbrace{2m^2}_{even}=\underbrace{\underbrace{4n^2+4n}_{even}+1}_{odd}
$$

This leads us to a contradiction, because the left hand side is even, but the right hand side is odd.
Therefore, $$\sqrt 2$$ is not a rational number.
$$\blacksquare$$

**NOTE**: In the above proof, **we do exploit the fact that $$q$$ is odd**, because we restate it as n odd number, to arrive at the contradiction.

In some cases, you will not need to use all the facts in the truth table, in some proofs you will have to.

Let us apply this template to another similar problem to show another pattern.

## Prove that $$\sqrt 3$$ is irrational

### Proof
We assume that $$\sqrt 3$$ is rational. Therefore, we can express it as a ratio of two integers $$\frac{p}{q}:p,q\in\mathbb{N}$$, which have no common factors between them. Thus:

$$
\frac{p^2}{q^2}=3 \\
\Rightarrow p^2=3q^2
$$

As usual, we make a quick truth table, to narrow down the feasibility of $$p, q$$ being odd and/or even.

| p    | q    | Feasible? |  Reason |
|------|------|-----------|---------|
| Even | Even | False     | By definition |
| Even | Odd  | False     | $$3q^2$$ is odd, so $$p^2$$ cannot be even, thus $$p$$ cannot be even |
| Odd  | Even | False     | $$12q^2$$ is even, so $$p^2$$ cannot be odd, thus $$p$$ cannot be odd|
| Odd  | Odd  | True      | |

Thus, the only valid option is $$p$$ odd and $$q$$ odd.

#### <u>Proof Pattern 2</u>
You cannot arrive at a contradiction using **Proof Pattern 1** like we did above, at this point. Instead, we express both $$p$$ and $$q$$ as odd numbers. We set $$p=2m+1$$ and $$q=2n+1$$, so that:
$$
{(2m+1)}^2=3{(2n+1)}^2 \\
\Rightarrow 4m^2+4m+1=3(4n^2+4n+1) \\
\Rightarrow 4m^2+4m+1=12n^2+12n+3 \\
\Rightarrow 4m^2+4m=12n^2+12n+2 \\
\Rightarrow \underbrace{2(m^2+m)}_{even}=\underbrace{\underbrace{6(n^2+n)}_{even}+1}_{odd} \\
$$

Thus, we arrive at a contradiction, where the right side is even, but the left side is odd.

Therefore, $$\sqrt 3$$ is not a rational number.
$$\blacksquare$$

#### <u>Proof Pattern 3</u>
We don't express $$p$$ and $$q$$ as odd numbers in this pattern. We do some simple algebraic manipulation on the original form, like so:

$$
\frac{p^2}{q^2}=3 \\
\Rightarrow p^2=3q^2 \\
\Rightarrow p^2-q^2=2q^2 \\
\Rightarrow \underbrace{(p+q)(p-q)}_{both\ even\ or\ both\ odd}=\underbrace{2q^2}_{even}
$$

The left side has to be even, by the above condition. Now, we note that $$p+q$$ and $$p-q$$ can be either both even, or both odd, but never one even, one odd. Why? This is easily verified:

- **$$p$$ even, $$q$$ even**: $$p+q=2(m+n), p-q=2(m-n)$$
- **$$p$$ even, $$q$$ odd**: $$p+q=2(m+n)+1, p-q=2(m-n)-1$$
- **$$p$$ odd, $$q$$ even**: $$p+q=2(m+n)+1, p-q=2(m-n)+1$$
- **$$p$$ odd, $$q$$ odd**: $$p+q=2(m+n+1), p-q=2(m-n)$$

Thus, the only way the right hand side condition can hold is if both $$p+q$$ and $$p-q$$ are even. Write $$p+q=2x$$ and $$p-q=2y$$, so that:

$$
(p+q)(p-q)=2q^2 \\
\Rightarrow 4xy=2q^2 \\
\Rightarrow q^2=2xy
$$

This implies that $$q$$ is even. But this contradicts our fact from the truth table that $$q$$ is odd.

Therefore, $$\sqrt 3$$ is not a rational number.
$$\blacksquare$$

**NOTE**: This proof pattern does not recur often, I've only been able to successfully apply it for the proof of irrationality of $$\sqrt 3$$

The final pattern we'd like to write down, reduces a problem to a simpler one.

## Prove that $$\sqrt 12$$ is irrational

### Proof
We assume that $$\sqrt 12$$ is rational. Therefore, we can express it as a ratio of two integers $$\frac{p}{q}:p,q\in\mathbb{N}$$, which have no common factors between them. Thus:

$$
\frac{p^2}{q^2}=12 \\
\Rightarrow p^2=12q^2
$$

As usual, we make a quick truth table, to narrow down the feasibility of $$p, q$$ being odd and/or even.

| p    | q    | Feasible? |  Reason |
|------|------|-----------|---------|
| Even | Even | False     | By definition |
| Even | Odd  | True      |  |
| Odd  | Even | False     | $$12q^2$$ is even, so $$p^2$$ cannot be odd, thus $$p$$ cannot be odd|
| Odd  | Odd  | False     | $$12q^2$$ is even, so $$p^2$$ cannot be odd, thus $$p$$ cannot be odd|

Thus, the only valid option is $$p$$ even and $$q$$ odd.

#### <u>Proof Pattern 4</u>
We only express $$p$$ as an even number, initially. We set $$p=2m$$, so that

$$
{(2m)}^2=12q^2 \\
4m^2=12q^2 \\
m^2=3q^2
$$

**NOTE**: The above problem now reduces to proving that $$\sqrt 3$$ is irrational. You'll see that expanding $$q$$ intially as an odd number will not get you anywhere.

All proofs of irrationality of square roots of appropriate numbers can (probably) be derived using the patterns shown above, unless I chance upon a very special case.

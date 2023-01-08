---
title: "Tests increase our Knowledge of the System: A Proof from Probability"
author: avishek
usemathjax: true
tags: ["Proof", "Tests", "Software Engineering", "Probability"]
draft: false 
---

**Note:** This is a post from July 13, 2011, rescued from my old blog. This is only for archival purposes, and is reproduced verbatim, but I make no claims about its rigour, though it does still seem plausible.

This was an old proof that was up on my old blog, but since I’m no longer posting to that, I’m reposting it here for posterity. Also, rewriting the equations in LaTeX, now that I have installed a plugin for that.

I present a simple mathematical device to prove that tests improve our understanding of code. It does not really matter if this is code written by the test author himself or is legacy. To do this, some simplification of the situation is necessary.

We assume $$X$$ is the unit of code under consideration. $$X$$ may be a function, a class or a compiled binary. The only restriction on $$X$$ is that it can accept inputs and produce measurable outputs.

Without loss of generality, we may assume that the $$X$$’s output consists of n bits. If complicated structures like objects are present in the result, they may simply be decomposed into bits and laid out in a convenient order to fit this model. This assumption exists to simplify quantizing the output space only.

We also assume that unique inputs yield unique outputs, but this assumption does not affect the fundamental conclusion.

Let us define the probabilities:

$$P\left(A\right)$$ = Probability that X uses the correct algorithm = p \\
$$P(B)$$ = Probability of test $$T_1$$ passing (getting the correct output) for a given input $$I_1 = \frac{1}{2^n}$$ \\
$$P(B|A)$$ = Probability of test $$T_1$$ passing (getting the correct output) for a given input $$I_1$$, given $$X$$ uses the correct algorithm = 1

Therefore, using Bayes’ Theorem:

$$P\left(A|B\right)$$ 
= Probability that $$X$$ uses the correct algorithm given test passes for a given input $$I_1$$

We can thus write:

$$
P\left(B|A\right).P\left(A\right)/P\left(B\right) = p.2^n \\
P\left(1\right)=p.2^n
$$

**Note that after writing one test, the probability of X using the correct algorithm has increased (n>=1) by a factor of $$2^n$$.**

Let us now write another test with input $$I_2$$. Note that I assume that $$T_1$$ passing does not affect any probabilities other than the updated probability of $$X$$ using the correct algorithm, i.e., the tests are statistically independent (I believe that’s the term used :-) .


$$P\left(A\right)$$
= Probability that X uses the correct algorithm = $$p.2^n$$ \\
$$P\left(B\right)$$
= Probability of test T2 passing (getting the correct output) for a given input $$I_2$$ = $$\frac{1}{2^n}$$ \\
$$P\left(B|A\right)$$
= Probability of test $$T_2$$ passing (getting the correct output) for a given input $$I_2$$, given $$X$$ uses the correct algorithm = 1

Therefore, using Bayes’ Theorem:

$$P\left(A|B\right)$$
= Probability that $$X$$ uses the correct algorithm given test passes for a given input $$I_2$$ 

$$
P(A|B)= P\left(B|A\right).P\left(A\right)/P\left(B\right) = p.2^n.2^n \\
P\left(2\right)=p.2^n.2^n
$$

After having written $$t$$ tests, we may write:

$$
P\left(t\right)=p.2^{nt}
$$

**$$t$$ tests, therefore, increase the probability (or our knowledge, in very rough terms) of X being implemented correctly, by a factor of 2^nt.**

Probability is inversely correlated with entropy; thus, we have also reduced the entropy of the system.
It might be useful to state that $$I$$ use the term ‘test’ in a broad sense. The test may range from an automated unit test to human verification.

It turns out that it should be possible to determine $$p$$. Note that $$t$$ is the number of **statistically independent tests** that we can write. This implies that $$t$$ has a fixed upper bound. Thus:

$$T$$ = total number of statistically independent tests for $$X$$ \\
$$p.2^{nT} <= 1$$ \\
$$p <= \frac{1}{2^{nT}}$$


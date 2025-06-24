---
title: "Automated Hypothesis Verification using LLMs and Hierarchical Bayes Models"
author: avishek
usemathjax: false
tags: ["Hierarchical Bayes", "Large Language Models", "Reasoning"]
draft: true
---

_This post has not been written or edited by AI._

## Abstract
We look at how Hierarchical Bayes Models can be used with simple Beta-Bernoulli prior/posterior updates to recursively decompose a hypothesis into sub-hypotheses to form an inference tree. The beliefs of these sub-hypothese are updated based on the strength of the evidence gathered. These beliefs are propagated upwards through the inference to indicate the aggregate confidence of the original root hypothesis.

## Motivation
For the last year or so, I've been heavily involved in building reverse engineering tools dealing with legacy code. This legacy code includes the usual suspects (COBOL, HLASM), but can also include code written in more "modern" stacks, like Java, C#, etc. Much of this tooling is driven through LLMs (isn't everything these days :-) ?).

However, these efforts have also forced some deeper introspection on my part about how humans deal with comprehending legacy code. There are several studies on models of human comprehension of code (both novices and experts), but for the purposes of this post, I will restrict myself to my own (obviously incomplete) mental model of how I resolve uncertainty when attempting to validate or invalidate a hypothesis.

This could be a hypothesis about anything, for example:

- This routine does not modify this variable.
- This piece of code branches off unconditionally, but does return to the point of origin.
- This memory addresses represents a particular quantity in the domain
- ... and so on

Most of the time, we (I?) look for **signals** which strengthen or weaken my belief in the hypothesis. Some studies call these signals **beacons**. The result of aggregating all these signals gives me a rough idea of how valid my hypothesis is. It is important to note that this is a sliding scale from "This is definitely false" to "This is definitely true", and other values like "I'm still not sure" in between.

This seems to be a good fit for Bayesian reasoning.

Let's talk of decomposition. Whenever I have a hypothesis, I'm subconsciously breaking it down into smaller hypotheses that I can prove/disprove. Then, I would go and gather evidence for/against these smaller hypotheses, and go back and assess my confidence in my original hypothesis. Essentially, we can think of this as building an **inference tree**, like so:

![Hypothesis Decomposition](/assets/images/inductor-hypothesis-decomposition.png)

The question is: is this something reproducible through LLMs and Bayesian techniques?
This sort of hierarchical modelling is something found in **Hierarchical Bayes Models**.


## How do we validate a hypothesis?

- Propose a hypothesis (User / LLM)
- Decompose the hypothesis with initial levels of belief (LLM)
- Gather evidence for sub-hypotheses (LLM tool use - tools are deterministic)
- Propagate beliefs upward based on evidence to original hypothesis (Deterministic)

The flow would look something like the following:

![Hypothesis Belief Aggregation](/assets/images/inductor-belief-aggregation.png)

## Aggregating beliefs: the Beta-Bernoulli conjugate

## Gathering evidence: MCP tools

## Motivating example

![Inductor Step 1](/assets/images/inductor-step-01.png)
![Inductor Step 2](/assets/images/inductor-step-02.png)
![Inductor Step 3](/assets/images/inductor-step-03.png)
...and so on, until the leaves are reached.

![Inductor Step 8](/assets/images/inductor-step-08.png)
![Inductor Step 9](/assets/images/inductor-step-09.png)
![Inductor Prior and Posterior](/assets/images/inductor-before-after.png)

## Architecture

![Overall Architecture](/assets/images/inductor-macro-structure.png)
![Hypothesis Decomposer](/assets/images/inductor-hypothesis-decomposer-langgraph.png)
![Hypothesis Validator](/assets/images/inductor-hypothesis-validator-langgraph.png)

## References
-[Inductor](https://github.com/asengupta/inductor)

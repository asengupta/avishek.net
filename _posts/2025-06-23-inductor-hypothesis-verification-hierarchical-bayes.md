---
title: "Automated Hypothesis Verification using LLMs and Hierarchical Bayes Models"
author: avishek
usemathjax: true
tags: ["Hierarchical Bayes", "Large Language Models", "Reasoning"]
draft: true
---

_This post has not been written or edited by AI._

## Abstract
We look at how Hierarchical Bayes Models can be used with simple Beta-Bernoulli prior/posterior updates to recursively decompose a hypothesis into sub-hypotheses to form an inference tree. The beliefs of these sub-hypothese are updated based on the strength of the evidence gathered. These beliefs are propagated upwards through the inference to indicate the aggregate confidence of the original root hypothesis.

## How do we validate a hypothesis?

- Propose a hypothesis (User / LLM)
- Decompose the hypothesis with initial levels of belief (LLM)
- Gather evidence for sub-hypotheses (LLM tool use - tools are deterministic)
- Propagate beliefs based on evidence to original hypothesis (Deterministic)
- Inspired by me trying to replicate parts of my own mental process when attempting to reverse engineer unfamiliar code :-)

## Decomposing hypotheses into inference trees

![Hypothesis Decomposition](/assets/images/inductor-hypothesis-decomposition.png)

## Aggregating beliefs: the Beta-Bernoulli conjugate

![Hypothesis Belief Aggregation](/assets/images/inductor-belief-aggregation.png)

## Gathering evidence: MCP tools

## Motivating example

![Inductor Step 1](/assets/images/inductor-step-01.png)
![Inductor Step 2](/assets/images/inductor-step-02.png)
![Inductor Step 3](/assets/images/inductor-step-03.png)
![Inductor Step 4](/assets/images/inductor-step-04.png)
![Inductor Step 5](/assets/images/inductor-step-05.png)
![Inductor Step 6](/assets/images/inductor-step-06.png)
![Inductor Step 7](/assets/images/inductor-step-07.png)
![Inductor Step 8](/assets/images/inductor-step-08.png)
![Inductor Step 9](/assets/images/inductor-step-09.png)
![Inductor Prior and Posterior](/assets/images/inductor-before-after.png)

## Architecture

![Overall Architecture](/assets/images/inductor-macro-structure.png)
![Hypothesis Decomposer](/assets/images/inductor-hypothesis-decomposer-langgraph.png)
![Hypothesis Validator](/assets/images/inductor-hypothesis-validator-langgraph.png)

## References
-[Inductor](https://github.com/asengupta/inductor)

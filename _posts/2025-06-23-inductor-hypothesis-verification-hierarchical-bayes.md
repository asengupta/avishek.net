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

## References
-[Inductor](https://github.com/asengupta/inductor)

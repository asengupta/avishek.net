---
title: "Automated Hypothesis Verification using LLMs and Hierarchical Bayes Models"
author: avishek
usemathjax: true
tags: ["Hierarchical Bayes", "Large Language Models", "Reasoning"]
draft: false
---

_This post has not been written or edited by AI._

## Abstract
We look at how a Hierarchical Bayes-like models can be used with simple Beta-Bernoulli prior/posterior updates to recursively decompose a hypothesis into sub-hypotheses to form an inference tree. The beliefs of these sub-hypotheses are updated based on the strength of the evidence gathered. These beliefs are propagated upwards through the inference to indicate the aggregate confidence of the original root hypothesis.

## Motivation
For the last year or so, I've been heavily involved in building reverse engineering tools dealing with legacy code. This legacy code includes the usual suspects (COBOL, HLASM), but can also include code written in more "modern" stacks, like Java, C#, etc. Much of this tooling is driven through LLMs (isn't everything these days :-) ?).

However, these efforts have also forced some deeper introspection on my part about how humans deal with comprehending legacy code. There are several studies on models of human comprehension of code (both novices and experts), but for the purposes of this post, I will restrict myself to my own (obviously incomplete) mental model of how I resolve uncertainty when attempting to validate or invalidate a hypothesis.

This could be a hypothesis about anything, for example:

- This routine does not modify this variable.
- This piece of code branches off unconditionally, but does return to the point of origin.
- This memory addresses represents a particular quantity in the domain
- ... and so on

Most of the time, we (I?) look for **signals** which strengthen or weaken my belief in the hypothesis. Some studies call these signals **beacons**. The result of aggregating all these signals gives me a rough idea of how valid my hypothesis is. It is important to note that this is a sliding scale from "This is definitely false" to "This is definitely true", and other values like "I'm still not sure" in between.

This seems to be a good fit for Bayesian reasoning. For the purposes of this experiment, I adopted a simple approach which is analogous to using a Hierarchical Bayes Model with a Beta-Bernoulli conjugate for prior-posterior belief calculations (more on that [here](#analogy-with-hierarchical-bayes-the-beta-bernoulli-conjugate))

Let's talk of decomposition. Whenever I have a hypothesis, I'm subconsciously breaking it down into smaller hypotheses that I can prove/disprove. Then, I would go and gather evidence for/against these smaller hypotheses, and go back and assess my confidence in my original hypothesis. Essentially, we can think of this as building an **inference tree**, like so:

![Hypothesis Decomposition](/assets/images/inductor-hypothesis-decomposition.png)

The question is: is this something reproducible through LLMs and Bayes-like techniques?
This sort of hierarchical modelling is something found in **Hierarchical Bayes Models**. We will use something similar, but much simpler. We will simply sum up weighted combinations of the evidences for and against the corresponding sub-hypotheses.

## How do we validate a hypothesis?

- Propose a hypothesis (User / LLM)
- Decompose the hypothesis with initial levels of belief (LLM) like so:
  - At every level of decomposition, ask the LLM to decide whether any more decomposition into sub-hypotheses is required. If decomposition is not required, the sub-hypothesis ends with a set of leaf evidence nodes. These are the pieces of evidence that will be gathered to determine the strength of the sub-hypothesis.
- Gather evidence for sub-hypotheses: This is where LLMs can use tools (exposed via MCP tools) to pick the best tool(s) to gather specific pieces of evidence. The LLM decides whether the evidence supports the sub-hypothesis or not.
- Propagate **beliefs** upward based on summing the (potentially weighted) counts of for/against evidence, all the way up to the original hypothesis (Deterministic)
- The final weighted counts of the for/against evidence at the root provides us a degree of belief in the root hypothesis.

The flow would look something like the following:

![Hypothesis Belief Aggregation](/assets/images/inductor-belief-aggregation.png)

In this case, we can simply sum aggregate the counts of the evidences into two buckets:

- Evidence supporting the sub-hypothesis
- Evidence not supporting the sub-hypothesis

## Motivating example

Here, I took a simple HLASM program, ran it through Tape/Z to parse its structure, and exposed its various functionalities to the Langgraph system through an MCP server. Examples of the functionalities exposed were:

- Regex search
- Cyclomatic complexity
- Code in specific sections (labels)
- ...etc.

The hypothesis that I asked it to verify was that the program uses a lot of registers. This is shown in the screenshot below.
![Inductor Step 1](/assets/images/inductor-step-01.png)

Beyond this point, the **Hypothesis Decomposer** component of Inductor started recursively decomposing this hypothesis into an inference tree, as show by the progression of screenshots below (I intentionally limited the number of branches at each step to 2 for speed of demonstration):

![Inductor Step 2](/assets/images/inductor-step-02.png)
![Inductor Step 3](/assets/images/inductor-step-03.png)

...and so on, until the leaves of evidence are reached.

![Inductor Step 8](/assets/images/inductor-step-08.png)
![Inductor Step 9](/assets/images/inductor-step-09.png)

At this point, the inference tree has been built, and the **Hypothesis Validator** component goes into action, starting to collect evidence, and aggregating the strengths up the hierarchy of the tree. The result is an updated strength of the root hypothesis. As shown in the screenshot below, the original strength was 0.5 (equally likely to be true or false), and the posterior strength came down to 0.4, indicating a weakened belief in the root hypothesis.

![Inductor Prior and Posterior](/assets/images/inductor-before-after.png)

## Architecture

![Overall Architecture](/assets/images/inductor-macro-structure.png)
![Hypothesis Decomposer](/assets/images/inductor-hypothesis-decomposer-langgraph.png)
![Hypothesis Validator](/assets/images/inductor-hypothesis-validator-langgraph.png)

## Current Limitations

- All sub-hypotheses are assumed to be independent of each other. This is often not true. Overlapping, dependent sub-hypotheses need causal connections between them which would normally affect belief propagation.
- The Beta-Bernoulli conjugate calculations have been chosen for their simplicity, and aren't necessarily the best fit to represent the prior and posterior. More sophisticated probability distribution modelling using MCMC, etc. should ideally be done.

## Analogy with Hierarchical Bayes: the Beta-Bernoulli conjugate

The Beta distribution is a customisable probability distribution, whose shape can be controlled by two parameters $$\alpha$$ and $$\beta$$. The formula for the distribution is given as:

$$
f(x;\alpha,\beta) = k.x^{\alpha-1}.{(1-x)}^{\beta-1}
$$

where $$k$$ is a constant chosen such that the area under the probability distribution sums to 1.
The shapes of the Beta distribution for different values of $$\alpha$$ and $$\beta$$ are shown below (taken from Wikipedia). As you can see, the shape can vary widely depending upon the parameter combination.

![Beta Distribution shapes](/assets/images/beta-distribution.png)

## References
-[Inductor](https://github.com/asengupta/inductor)

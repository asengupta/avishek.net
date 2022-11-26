---
title: "The No-Questions Asked Guide to PyTorch : Part 1"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Neural Networks"]
draft: true
---

Programming guides are probably the first posts to become obsolete, as APIs are updated. Regardless, we will look at building simple neural networks in PyTorch; we won't be starting from models with a million parameters, however. We will proceed from the basics, starting with a single neuron, talk a little about the tensor notation and how that relates to our usual mathematical notation of representing everything with column vectors, and scale up from there.

We will assume you have already installed Python and PyTorch; we'll not get into those details.

### One Neuron, One Input, No Bias, No Activation Function

This is the simplest setup one can imagine. We've stripped away every possible piece we could: the resulting thing is a neuron -- barely. At this point, we are literally solving a multiplication problem: it could not get any simpler.

```python
{% include_relative code/simple-linear-nn.py %}
```

### One Neuron, One Input, No Bias, RelU Activation Function

```python
{% include_relative code/simple-linear-nn-relu.py %}
```
If you run the above code a few times, you will find that, in some runs, there are no updates to the weights at all. The loss essential stays constant. This is because the output of the neuron is less than zero, which clamps the output to zero, and thus kills the gradient in the backpropagation process. Why does this happen? To understand that, let's break down the backpropagation equation for this simple model.

TODO: Demonstrate ReLU breaking backpropagation

### One Neuron, One Input, No Bias, Leaky RelU Activation Function

To rectify the above situation, we will use what is called the **Leaky ReLU**. This adds a small, but nonzero, gradient to the original ReLU function if the input is less than zero. This is demonstrated below:

```python
{% include_relative code/simple-linear-nn-leaky-relu.py %}
```

### One Neuron, One Input with Bias, Leaky RelU Activation Function

We now add the bias input to the neuron. This is as simple as setting ```bias=True``` in the above code, so that we get the following:

```python
{% include_relative code/simple-linear-nn-bias-leaky-relu.py %}
```

### One Neuron, Multiple Inputs with Bias, Leaky RelU Activation Function

```python
{% include_relative code/simple-linear-nn-multiple-inputs-bias-leaky-relu.py %}
```

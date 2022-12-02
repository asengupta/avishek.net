---
title: "The No-Questions Asked Guide to PyTorch : Transformers"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Transformers"]
draft: false
---

It may seem strange that I'm jumping from implementing a simple neural network into **Transformers**. I will return to building up the foundations of neural networks soon enough: for the moment, let's look at **Transformers**.

There are several excellent guides to understanding the original Transformers architecture; I will use [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar to guide this implementation.

One thing about this guide is that it does not start with a polished walkthrough of the finished code. Rather, I build it in stages, experimenting with PyTorch API's, adding/modifying/deleting code as I go along. The idea is two-fold: one, to give you a sense of what goes behind implementing a paper incrementally, and second, to demonstrate that progress while writing code is not linear.

We will start with following the simplified block diagram as is presented on the site.

![Encoder-Decoder Block Diagram](/assets/images/transformer-encoder-decoder-block-diagram.png)

We will start without any PyTorch dependencies, and almost blindly build an object model based on this diagram. The paper says that there are 6 encoders stacked on top of each other, so we will build a simple linked encoder graph. Most of this code will be probably thrown away later, but at this point we are feeling out the structure of the solution without worrying too much about the details: experimenting with the details will come later. Here's the code:

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-001.py' %}
```

Replicating the decoder block diagram is almost as easy: simply steal code from the encoder. We will probably not get to the decoder for a while, but this is a exercise to get a lay of the land before getting lost in the weeds.

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-002.py' %}
```

At this point, we will want to start taking a peek at the insides of the encoder and the decoder. The breakdown looks like as below:

![Encoder-Decoder Breakdown](/assets/images/transformer-encoder-decoder-breakdown.png)

We want to build these blocks, but building the Self-Attention layer is going to take a while, and we don't want to wait to build out the scaffolding. Thus, what we will do is build all these blocks and assume they are all vanilla Feedforward neural networks. It does not matter that this is not the actual picture: we're mostly interested in filling the blanks. We can always go back and replace parts as we see fit.

Thus, even the Self-Attention layers are also represented as Feedforward neural networks, and the code is as below:

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-003.py' %}
```

To be honest, a lot of this is pretty dirty code; there are magic numbers, most of the object variables are not used. That's alright. This is also the first time we start including PyTorch dependencies.
We've created a ```Sequential``` stack to house our layers, but there is not much to say about it, since it is essentially a placeholder for the real layers to be built in and incorporated.

### Introducing the Query-Key-Value triad function

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-004.py' %}
```

### Applying the Query-Key-Value function to a single word

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-005.py' %}
```

### Building the Softmax Scores for a single word

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-006.py' %}
```

### Building the Attention Score for a single word

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-007.py' %}
```

### Applying the Query-Key-Value function to a multiple words

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-009.py' %}
```

### Building the Attention Scores for multiple words

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-010.py' %}
```

### Encapsulating Attention Score calculation into a custom layer

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-011.py' %}
```

### Incorporating the Self Attention layer into the Encoder

This is also the point at which we build the multi-head attention block by running the input through eight attention blocks.

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-012.py' %}
```

### Projecting the Attention Outputs back into original word width

At this point, we are ready to pass the output into the Feedforward neural network.

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-013.py' %}
```


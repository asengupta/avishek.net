---
title: "The No-Questions Asked Guide to PyTorch : Transformers, Part 1"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Transformers"]
draft: false
---

It may seem strange that I'm jumping from implementing a simple neural network into **Transformers**. I will return to building up the foundations of neural networks soon enough: for the moment, let's build a **Transformer** using PyTorch.

The original paper is [Attention is All You Need](https://arxiv.org/abs/1706.03762). However, there are several excellent guides to understanding the original Transformers architecture; I will use [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar to guide this implementation. Many of the diagrams of the Transformer internals are gratefully borrowed from this site.

As is obligatory, we reproduce the original Transformer architecture before diving into the depths of its implementation.

![Transformer Architecture](/assets/images/transformers-architecture.png)

One thing about this guide is that it does not start with a polished walkthrough of the finished code. Rather, we build it in stages, experimenting with PyTorch API's, adding/modifying/deleting code as I go along. The idea is two-fold: one, to give you a sense of what goes behind implementing a paper incrementally, and second, to demonstrate that progress while writing code is not necessarily linear.

Also, apart from some stock functionality like ```Linear``` and ```LayerNorm```, we won't be using any Transformer-specific layers available in PyTorch, like ```MultiheadAttention```. This is so we can gain a better understanding of the attention mechanism by building it ourselves.

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

Our initial aim is similar to the mathematics problems on dimensional analysis we used to solve in school: namely, we want to get the dimensions of our inputs and outputs correct and parameterisable. We will start with one word, and midway, scale to supporting multiple words.

### Introducing the Query-Key-Value triad function

This step is pretty simple: we will introduce the function which returns us the Key-Query-Value triad for a given word. Remember that we are not worried about the actual calculations right now; we will only worry ourselves about getting the dimensions right. Since the original paper mentions scaling the input to 64 dimensions, our function will simply return three 64-dimensional tensors filled with ones.

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-004.py' %}
```

### Applying the Query-Key-Value function to a single word

We can immediately move to the next logical step: actually multiplying our 512-dimensional input by $$W_Q$$, $$W_K$$ and $$W_V$$ matrices. Remember we still want to get out three 64-dimensional vectors, thus the sizes of these matrices will be $$512 \times 64$$. This is also when we actually multiply the input with these vectors in our ```qkv()``` function.

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-005.py' %}
```

### Building the Attention Score for a single word

At this point, we are ready have the three vectors $$Q$$, $$K$$, and $$V$$. We are about to implement part of the following calculation, except that it is for a single word:

$$
Z = \displaystyle\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \times V
$$

The single word version of the above calculation for the $$i$$-th word can be written as:

$$
Z_i = \displaystyle\sum_{j=1}^N \text{Softmax} \left(\frac {Q_i K_j}{\sqrt{d_K}}\right) V_j
$$

In the current case, we only have one word in total, so the above simply reduces to:

$$
Z_1 = \displaystyle\text{Softmax} \left(\frac {Q_1 K_1}{\sqrt{d_K}}\right) V_1
$$

![Softmax on Attention Score on a Single Word](/assets/images/self-attention-softmax.png)

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-007.py' %}
```

### Building the Attention Scores for multiple words

Now, we are ready to go a little more production-strength. Instead of dealing with individual words, we will stack them in a $$N \times 512$$ tensor ($$N$$ being the number of words), and build the attention scores of all of these words, using the matrix version of the calculation, like we noted in the previous section. Specifically:

$$
Z = \displaystyle\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \times V
$$

I think it is instructive to understand why this matrix multiplication works. To unpack this, let us discard the scaling factor ($$\sqrt{d_k}$$) and the Softmax function, leaving us with the core of the calculation:

$$
Z = Q K^T V
$$

Let us take a concrete example of 2 words. Then, $$Q$$, $$K$$, and $$V$$ are all $$2 \times 64$$ tensors. Specifically, let us write:

$$
Q = \begin{bmatrix}
Q_1 \\
Q_2
\end{bmatrix}

K = \begin{bmatrix}
K_1 \\
K_2
\end{bmatrix}

V = \begin{bmatrix}
V_1 \\
V_2
\end{bmatrix}
$$

where $$Q_1$$, $$Q_2$$ (queries for the two words), $$K_1$$, and $$K_2$$ (keys for the two words), $$V_1$$, and $$V_2$$ (values for the two words)are 64-dimensional tensors. Then we have:

$$
QK^T = \begin{bmatrix}
Q_1 K_1 && Q_1 K_2 \\
Q_2 K_1 && Q_2 K_2 \\
\end{bmatrix}
$$

and then we have:

$$
QK^T V = \begin{bmatrix}
Q_1 K_1 && Q_1 K_2 \\
Q_2 K_1 && Q_2 K_2
\end{bmatrix}
\begin{bmatrix}
V_1 \\
V_2
\end{bmatrix}
$$

We want to treat these product as the weighted combinations of the rows of $$V$$, thus we get:

$$
QK^T V = \begin{bmatrix}
Q_1 K_1 V_1 + Q_1 K_2 V_2 \\
Q_2 K_1 V_1 + Q_2 K_2 V_2
\end{bmatrix}
=
\begin{bmatrix}
\text{Attention Score of Word 1} \\
\text{Attention Score of Word 2} \\
\end{bmatrix}
$$

The first row is the (simplified) attention score of the first word, and the second one that of the second word. The scaling and the Softmax just gives us the linearly transformed version of the above.

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-010.py' %}
```

### Encapsulating Attention Score calculation into a custom layer

We are well on our way to building out a functional (at least where input/output sizes are concerned) encoder layer. This step has some basic refactoring. Now that we know our calculations are ready, let us move into a custom ```SelfAttentionLayer```.

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-011.py' %}
```

### Incorporating the Self Attention layer into the Encoder

The magic numbers are getting pretty ugly; let's start moving them into variables for readability. This is also the point at which we build the multi-head attention block by running the input through eight attention blocks. Notice how much little of the original code we have really touched; that will be getting replaced very soon.

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-012.py' %}
```

### Projecting the Attention Outputs back into original word width

We are mostly done with building the Attention part of the Encoder; we'd like to get started on the Feedforward Neural Network. However, to do that, the output of the Attention component needs to projected back into 512-dimensional space (the original word width).
This is achieved by multiplying the output ($$N \times (64*8)$$) by  $$W_O$$ ($$(64*8) \times 512$$).

At this point, we are ready to pass the output into the Feedforward neural network.

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-013.py' %}
```

### Adding Feedforward Neural Network

We'd already built a FFNN in one of our first iterations; now we need to hook it up to the output of the Attention layer. Each word needs to be run through the same FFNN. Each word width is 512, thus the number of inputs to the FFNN is 512. The output needs to be of the same dimension. For this iteration, we will not worry about the ReLU and the hidden layer; a single layer will suffice to demonstrate that the Encoder in its current form can give a dimensionally-correct output.

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-014.py' %}
```

### Fixing Feedforward Neural Network architecture

We'd like to now tune the FFNN architecture in line with the paper. The paper notes that there is one hidden layer consisting of 2048 units, followed by a ReLU layer. Thus, we have the following architecture:

- 512 outputs fully connected to 2048 units
- 2048 units passing their outputs through a ReLU layer
- ReLU layer fully connected to a layer of 512 units (remember, our output needs to be 512-dimensional)

This is described in the paper as:

$$
FFN(x) = \text{max}(0,xW_1 + b_1) W_2 + b_2
$$

This neural network needs to be applied as many times as we have words. So if we have 4 words, our word matrix is represented as a $$4 \times 512$$ matrix; and each row (of size 512) of this matrix needs to be passed into the FFNN. That's 4 rows in this case.

The FFNN is wrapped in a ```Sequential``` module, and uses a Leaky ReLU.

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-015.py' %}
```

### Adding Add-and-Norm Layer with Residual Connections

We still need to add the residual connections which are added and normed to the outputs of both the Multihead Attention block and the FFNN. This is simply a matter of element-wise adding of the input and passing the result through a ```LayerNorm``` layer.

![Residual Connection with Layer Norm](/assets/images/transformer-residual-layer-norm.png)

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-016.py' %}
```

### Refactoring, Stacking Encoders, and Placeholder for Positional Encoding

This step involves a lot of refactoring and cleaning up. Specifically, reordering and cleaning up parameters happens here. Default parameters are added as well. This is also the first time we touch the function to stack six Encoders. Our original code is pretty much useless at this point; thus we simply build a ```Sequential``` container of ```Encoder``` layers. We obviously verify that it outputs a $$2 \times 512$$ vector.

There is still one part of the Encoder we haven't fleshed out fully: the positional encoding. For the moment, we add a placeholder function ```positionally_encoded()``` which we will implement fully going forward.

```python
{% include_absolute '/code/pytorch-learn/transformer/history/transformer-017.py' %}
```
This concludes Part 1 of implementing Transformers using PyTorch. There are a lot of loose ends which we will continue to address in the sequels. The demonstration of the incremental build-up should give you a fair idea of how you can go about implementing models from scratch in PyTorch.

### References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

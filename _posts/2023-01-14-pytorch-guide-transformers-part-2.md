---
title: "The No-Questions Asked Guide to PyTorch : Transformers, Part 2"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Transformers"]
draft: false
---

We continue looking at the **Transformer** architecture from where we left from [Part 1](2022-11-29-pytorch-guide-transformers-part-1). When we'd stopped, we'd set up the Encoder stack, but had stopped short of adding positional encoding, and starting work on the Decoder stack. In this post, we will focus on setting up the training cycle.

Specifically, we will cover:

- Positional Encoding
- Decoder stack, including the masked multi-head attention mechanism
- Set up the basic training regime via Teacher Forcing

We will also lay out the dimensional analysis a little more clearly, and add necessary unit tests to verify intended functionality. The code is available [here](https://github.com/asengupta/transformers-paper-implementation).

### Positional Encoding

![Position Encoding zoomed out](/assets/images/transformers-positional-encoding-full.png)

![Position Encoding zoomed in](/assets/images/transformers-positional-encoding-zoomed.png)

```python
{% include_code https://raw.githubusercontent.com/asengupta/transformers-paper-implementation/main/transformer.py!184!205%}
```

## Building the Decoder stack

We now proceed to build the **Decoder** proper.

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformers Explained Visually](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34)


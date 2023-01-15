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

You can see the code for visualising the positional encoding [here](https://github.com/asengupta/transformers-paper-implementation/blob/main/positional-encoding.py).

Both images show the encoding map at different levels of zoom.

![Position Encoding zoomed out](/assets/images/transformers-positional-encoding-full.png)

![Position Encoding zoomed in](/assets/images/transformers-positional-encoding-zoomed.png)

The code in the main Transformer implementation which implements the positional embedding is shown below.
```python
{% include_code https://raw.githubusercontent.com/asengupta/transformers-paper-implementation/main/transformer.py!223!242%}
```

## Building the Decoder stack
## Notes on the Code

- The last word in the output is added to the output buffer, during inference.
- The encoder output is injected directly into the sublayer of every Decoder. To build up the chain of Decoders in PyTorch, so that we can put the full stack inside a Sequential block, we simply inject the encoder output to the root Decoder, and have it output the encoder output (together with the actual Decoder output) as part of the Decoder's actual output to make it easy for the next Decoder in the stack to consume the Encoder and Decoder outputs.

We now proceed to build the **Decoder** proper.

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformers Explained Visually: Part 1](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)
- [Transformers Explained Visually: Part 2](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34)

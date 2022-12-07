---
title: "Plenoxels and Neural Radiance Fields using PyTorch: Part 2"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Neural Radiance Fields", "Machine Vision"]
draft: false
---

We continue looking at [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131). We set up a very contrived opacity model in our last post, which gave us an image like this:

![Very Basic Volumetric Rendering of Cube](/assets/images/basic-volumetric-rendering-cube.png)

It looks very chunky; this is due to the finite resolution of the voxel world. More importantly, it does not incorporate a proper volumetric rendering model, which we will discuss in this post.

A point about performance and readability is in order here. The implementation that we are building is not going to be very fast, and the code is certainly not as optimised as it should be. However, the focus is on the explainability of the process, and how that translates into code. We may certainly delve into utilising the GPU more effectively at some point, but let us get the basics of the paper sorted first before attempting that.

Another point about cleanliness; much of the intermediate code on the way to a full implementation is going to be somewhat messy, and is presented as-is, without removing commented code, the odd ```print()``` statement and any number of variables which might not end up getting used. I've also always had a Python console open for really quick experiments with the PyTorch API, before lifting an expression from there and planting it in the code proper.

In this article, we will discuss the following:

- Building the Voxel Data Structure: First Pass
- Incorporating the Volumetric Rendering Model
- Incorporating Trilinear Interpolation
- Fixing Volumetric Rendering Artifacts

### Building the Voxel Data Structure: First Pass

In our final model, we'd like to store opacity and the spherical harmonic coefficients per voxel. Opacity will be denoted by $$\sigma \in [0,1]$$ and is as straightforward as it sounds. Let's talk quickly about how we should encode colour. Each colour channel (R,G,B) will have its own set of 9 spherical harmonic coefficients; this gives us 27 numbers to store. Add $$\sigma$$ to this, and each voxel is essentially represented by 28 numbers.

For the purposes of our initial experiments, we will not worry about the spherical harmonics, and simply represent colour using RGB triplets ranging from 0.0 to 1.0. We will still have opacity $$\sigma$$. Add the three numbers necessary to represent a voxel's position in 3D space, and we have 3+1+3=7 numbers to completely characterise a voxel.

We will begin by implementing a very simple and naive data structure to represent our world: a $$(i \times j \times k)$$ tensor cube. Indexing this structure using ```world[x,y,z]``` helps us locate a particular voxel in space. Each entry in this world will thus contain a tensor containing 4 numbers, the opacity $$\sigma$$ and the RGB triplet.

In this article, we will not make full use of the RGB triplet; it will mostly be there for demonstration purposes before being replaced by the proper spherical harmonics model going forward.

Addressing a voxel in a position which does not exist in this world will give us $$[0,0,0,0]$$. 

### Incorporating the Volumetric Rendering Model

We will take a bit of a pause to understand the optical model involved in volumetric rendering since it is essential to the actual rendering and the subsequent calculation of the loss. This article follows [Optical Model for Volumetric Rendering](https://www.youtube.com/watch?v=hiaHlTLN9TE) quite a bit. Solving differential equations is involved, but for the most part, you should be able to skip to the results, if you are not interested in the gory details.

Let's take the transmittance equation:

$$
\begin{equation}
\frac{dC}{ds} = \sigma(s)c(s) - \sigma(s)C(s)
\label{eq:transmittance}
\end{equation}
$$

Let's solve it fully, so I can show you the process, even though it's a very ordinary first-order differential equation.

Assume $$C=uv$$. Then, we can write by the Chain Rule of Differentiation:

$$
\frac{dC(s)}{ds} = u\frac{dv}{ds} + v\frac{du}{ds}
$$

Let's get equation $$\eqref{eq:transmittance}$$ into the canonical form $$Y'(x) + P(x)Y = Q(x)$$.

$$
\begin{equation}
\frac{dC(s)}{ds} + \sigma(s)C(s) = \sigma(s)c(s)
\label{eq:transmittance-canonical}
\end{equation}
$$

The integrating factor is thus:

$$
I(t) = e^{\int_0^s P(t) dt} \\
= e^{\int_0^s \sigma(t) dt}
$$

Multiplying both sides of $$\eqref{eq:transmittance-canonical}$$ by the integrating factor, we get:

$$
\frac{dC(s)}{ds}.e^{\int_0^s \sigma(t) dt} + \sigma(s)C(s).e^{\int_0^s \sigma(t) dt} = \sigma(s)c(s).e^{\int_0^s \sigma(t) dt}
$$

We note that for the left side:

$$
\frac{d[C(s).e^{\int_0^s \sigma(t) dt}]}{ds} = \frac{dC(s)}{ds}.e^{\int_0^s \sigma(t) dt} + \sigma(s)C(s).e^{\int_0^s \sigma(t) dt}
$$

So we can essentially write:

$$
\frac{d[C(s).e^{\int_0^s \sigma(t) dt}]}{ds} = \sigma(s)c(s).e^{\int_0^s \sigma(t) dt} \\
\int_0^D \frac{d[C(s).e^{\int_0^s \sigma(t) dt}]}{ds} ds = \int_0^D \sigma(s)c(s).e^{\int_0^s \sigma(t) dt} ds \\
$$

Let's discount the constant factor of integration for the moment, and assume that $$C(0) = 0$$, we then have:

$$
C(D).e^{\int_0^D \sigma(t) dt} = \int_0^D \sigma(s)c(s).e^{\int_0^s \sigma(t) dt} ds \\
C(D) = \int_0^D \sigma(s)c(s).e^{\int_0^s \sigma(t) dt}.e^{-\int_0^D \sigma(t) dt} ds \\
C(D) = \int_0^D \sigma(s)c(s).e^{-\int_s^D \sigma(t) dt} ds \\
$$

A very nice thing happens if $$c(s)$$ happens to be constant. Let us assume it is $$c$$. Then the above equation becomes:

$$
C(D) = c \int_0^D \sigma(s).e^{-\int_s^D \sigma(t) dt} ds \\
$$

Note that from the Fundamental Theorem of Calculus (see [here](https://math.stackexchange.com/questions/1047523/derivative-of-integral-with-x-as-the-lower-limit)), we can write:

$$
\frac{d[e^{-\int_s^D \sigma(t) dt}]}{ds}=e^{-\int_s^D \sigma(t) dt}.\frac{d[-\int_s^D \sigma(t) dt]}{ds} \\
= - e^{-\int_s^D \sigma(t) dt}.\frac{d[\int_s^D \sigma(t) dt]}{ds} \\
= - e^{-\int_s^D \sigma(t) dt}(\sigma(D) \frac{dD}{ds} - \sigma(s) \frac{ds}{ds}) \\
= - e^{-\int_s^D \sigma(t) dt}(\sigma(D).0 - \sigma(s).1) \\
= - e^{-\int_s^D \sigma(t) dt}(- \sigma(s)) \\
= \sigma(s).e^{-\int_s^D \sigma(t) dt} \\
$$

Then, we can rewrite $$C(D)$$ as:

$$
C(D) = c \int_0^D \frac{d[e^{-\int_s^D \sigma(t) dt}]}{ds} ds \\
= c \left[ e^{-\int_D^D \sigma(t) dt} - e^{-\int_0^D \sigma(t) dt} \right] \\
= c \left[ e^0 - e^{-\int_0^D \sigma(t) dt} \right] \\
C(D) = c \left[ 1 - e^{-\int_0^D \sigma(t) dt} \right]
$$

```python
{% include_absolute '/code/pytorch-learn/plenoxels/volumetric-rendering.py' %}
```

![Volumetric Sheath Cube](/assets/images/volumetric-sheath-cube-01.png)
![Volumetric Rendering Cube without Trilinear Interpolation](/assets/images/volumetric-rendering-cube-without-trilinear-interpolation.png)

### Incorporating Trilinear Interpolation

```python
{% include_absolute '/code/pytorch-learn/plenoxels/volumetric-rendering-with-trilinear-interpolation.py' %}
```
![Volumetric Rendering with Trilinear Interpolation with Artifacts](/assets/images/volumteric-rendering-cube-trilinear-interpolation-with-artifacts.png)

### Fixing Volumetric Rendering Artifacts

![Volumetric Rendering with Trilinear Interpolation](/assets/images/volumetric-rendering-cube-trilinear-interpolation.png)

### References

- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [Plenoxels Explained](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks)
- [Direct Volume Rendering](https://www.youtube.com/watch?v=hiaHlTLN9TE)
- [Optical Models for Volumetric Rendering](https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf)
- [Common Artifacts in Volume Rendering](https://arxiv.org/abs/2109.13704)
- [Trilinear Interpolation](https://en.wikipedia.org/wiki/Trilinear_interpolation)

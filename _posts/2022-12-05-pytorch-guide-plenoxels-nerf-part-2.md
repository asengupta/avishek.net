---
title: "Plenoxels and Neural Radiance Fields using PyTorch: Part 2"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Neural Radiance Fields", "Machine Vision"]
draft: false
---

We continue looking at [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131). We start off with understanding Spherical Harmonics, and why we want to use them to capture colour information based on angular direction in 3D space.

### Spherical Harmonics

Spherical harmonics essentially define a series (possibly infinite) of functions defined over a sphere, that is to say, the harmonics are parameterised by the angular coordinates of a 3D polar coordinate system. It's essentially like the Fourier Transform, in that the larger the degree of the harmonic, the more accurately a signal on a sphere can be encoded in the series.

$$
f(\theta, \phi) = \sum_{l=0}^N \sum_{m=-l}^l C^m_l Y^m_l (\theta, \phi)
$$

The paper expands only to the degree of 2. This implies that that the number of terms in the above series is $$1+3+5=9$$.

The basis functions for degree 2 are listed below:

$$
Y^0_0 = \frac{1}{2} \sqrt {\frac{1}{\pi}} \\
Y^{-1}_1 = \frac{1}{2} \sqrt {\frac{3}{\pi}} y \\
Y^0_1 = \frac{1}{2} \sqrt {\frac{3}{\pi}} z \\
Y^1_1 = \frac{1}{2} \sqrt {\frac{3}{\pi}} x \\
Y^{-2}_2 = \frac{1}{2} \sqrt {\frac{15}{\pi}} xy \\
Y^{-1}_2 = \frac{1}{2} \sqrt {\frac{15}{\pi}} yz \\
Y^0_2 = \frac{1}{4} \sqrt {\frac{5}{\pi}} (3z^2 - 1) \\
Y^1_2 = \frac{1}{2} \sqrt {\frac{15}{\pi}} xz \\
Y^2_2 = \frac{1}{4} \sqrt {\frac{15}{\pi}} (x^2 - y^2)
$$

where $$x$$, $$y$$, and $$z$$ are defined as:

$$
x = \text{sin } \theta . \text{cos } \phi \\
y = \text{sin } \theta . \text{sin } \phi \\
z = \text{cos } \theta
$$

Here is an example of an example 2nd order harmonic with some randomly chosen coefficients. The colours have been modified based on the spherical harmonic value at that specific spherical angle.

![Random Spherical Harmonic](/assets/images/random-spherical-harmonic.png)


```python
{% include_absolute '/code/pytorch-learn/plenoxels/spherical-harmonics.py' %}
```

```python
{% include_absolute '/code/pytorch-learn/plenoxels/volumetric-rendering.py' %}
```

```python
{% include_absolute '/code/pytorch-learn/plenoxels/volumetric-rendering-with-trilinear-interpolation.py' %}
```
![Volumetric Rendering with Trilinear Interpolation](/assets/images/volumetric-rendering-cube-trilinear-interpolation.png)


### Building the Voxel Data Structure: First Pass

We will store opacity and the spherical harmonic coefficients. Opacity will be denoted by $$\sigma \in [0,1]$$ and is as straightforward as it sounds. Let's talk quickly about how we should encode colour. Each colour channel (R,G,B) will have its own set of 9 spherical harmonic coefficients; this gives us 27 numbers to store. Add $$\sigma$$ to this, and each voxel is essentially represented by 28 numbers.

### Incorporating the Volumetric Rendering Model

We will take a bit of a pause to understand the optical model involved in volumetric rendering since it is essential to the actual rendering and the subsequent calculation of the loss. This article follows [Optical Model for Volumetric Rendering](https://www.youtube.com/watch?v=hiaHlTLN9TE) quite a bit. Solving differential equations is involved, but for the most part, you should be able to skip to the results, if you are not interested in the gory details.

### Incorporating Trilinear Interpolation

### Example Render using Spherical Harmonics and Opacity


### Calculating Loss

### Incorporating Training Images

### Optimising using RMSProp

### Reconstruction Results

### References

- [Spherical Harmonics](https://patapom.com/blog/SHPortal/#fn:2)
- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [Plenoxels Explained](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks)
- [Optical Model for Volumetric Rendering](https://www.youtube.com/watch?v=hiaHlTLN9TE)
- [Common Artifacts in Volume Rendering](https://arxiv.org/abs/2109.13704)

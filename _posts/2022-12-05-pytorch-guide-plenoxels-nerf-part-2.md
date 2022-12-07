---
title: "Plenoxels and Neural Radiance Fields using PyTorch: Part 2"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Neural Radiance Fields", "Machine Vision"]
draft: false
---

We continue looking at [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131).

### Building the Voxel Data Structure: First Pass

We will store opacity and the spherical harmonic coefficients. Opacity will be denoted by $$\sigma \in [0,1]$$ and is as straightforward as it sounds. Let's talk quickly about how we should encode colour. Each colour channel (R,G,B) will have its own set of 9 spherical harmonic coefficients; this gives us 27 numbers to store. Add $$\sigma$$ to this, and each voxel is essentially represented by 28 numbers.

### Incorporating the Volumetric Rendering Model

We will take a bit of a pause to understand the optical model involved in volumetric rendering since it is essential to the actual rendering and the subsequent calculation of the loss. This article follows [Optical Model for Volumetric Rendering](https://www.youtube.com/watch?v=hiaHlTLN9TE) quite a bit. Solving differential equations is involved, but for the most part, you should be able to skip to the results, if you are not interested in the gory details.

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
- [Optical Model for Volumetric Rendering](https://www.youtube.com/watch?v=hiaHlTLN9TE)
- [Common Artifacts in Volume Rendering](https://arxiv.org/abs/2109.13704)
- [Trilinear Interpolation](https://en.wikipedia.org/wiki/Trilinear_interpolation)

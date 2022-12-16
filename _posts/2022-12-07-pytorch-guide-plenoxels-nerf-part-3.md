---
title: "Plenoxels and Neural Radiance Fields using PyTorch: Part 3"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Neural Radiance Fields", "Machine Vision"]
draft: false
---

This is part of a series of posts breaking down the paper [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131), and providing (hopefully) well-annotated source code to aid in understanding.

- [Part 1]({% post_url 2022-12-04-pytorch-guide-plenoxels-nerf-part-1 %})
- [Part 2]({% post_url 2022-12-05-pytorch-guide-plenoxels-nerf-part-2 %})
- [Part 3 (this one)]({% post_url 2022-12-07-pytorch-guide-plenoxels-nerf-part-3 %})
- [Part 4]({% post_url 2022-12-07-pytorch-guide-plenoxels-nerf-part-4 %})

We continue looking at [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131). We start off with understanding **Spherical Harmonics**, and why we want to use them to capture colour information based on angular direction in 3D space.

### Spherical Harmonics

Spherical harmonics essentially define a series (theoretically infinite) of functions defined over a sphere, that is to say, the function values are parameterised by the angles of a spherical coordinate system. It's essentially like the Fourier Transform, in that the larger the degree of the harmonic, the more accurately a signal on a sphere can be encoded in the series.

$$
f(\theta, \phi) = \sum_{l=0}^N \sum_{m=-l}^l C^m_l Y^m_l (\theta, \phi)
$$

Here, $$N$$ is the order, $$C^m_l$$ are the coefficients; this results in a linear combination of the basis functions $$Y^m_l$$, which represents the target function. How accurately the target is described depends upon how far the series is expanded. The paper expands only to degree 2. This implies that that the number of terms in the above series is $$1+3+5=9$$.

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

Here is an example of an example 2nd order harmonic with some randomly chosen coefficients. The colours have been modified based on the spherical harmonic value at that specific viewing angle.

![Random Spherical Harmonic](/assets/images/random-spherical-harmonic.png)

```python
{% include_absolute '/code/pytorch-learn/plenoxels/spherical-harmonics.py' %}
```
### Incorporating Spherical Harmonics

Let's incorporate spherical harmonics into our rendering model. We have reached the point where each voxel in the world needs to be represented by a full-blown tensor of opacity and three set of 9 harmonic coefficients, hence 28 numbers.

Also, we are no longer calculating a single density; we are calculating the intensity for each colour channel. It is the same calculation, however the colour will now be calculated using the spherical harmonic functions.

```python
{% include_absolute '/code/pytorch-learn/plenoxels/volumetric-rendering-with-trilinear-interpolation-higher-sampling-rate-spherical-harmonics.py' %}
```
We also take this opportunity to do some obvious optimisations, like precomputing the spherical harmonic constants.

### Example Renders using Spherical Harmonics

Just to recap how our volume renderer has progressed, here is an evolution sequence.

**Non-rendered voxel grid**
![Voxel Cube](/assets/images/voxel-cube.png)
**Simple density-based voxel rendering**
![Very Basic Volumetric Rendering of Cube](/assets/images/basic-volumetric-rendering-cube.png)
**Volumentric rendering without Linear Interpolation**
![Volumetric Rendering Cube without Trilinear Interpolation](/assets/images/volumetric-rendering-cube-without-trilinear-interpolation.png)
**Volumentric rendering with Linear Interpolation**
![Volumetric Rendering with Trilinear Interpolation](/assets/images/volumetric-rendering-cube-trilinear-interpolation.png)
**Volumentric rendering with Linear Interpolation and Spherical Harmonics**
![Volumetric Rendering using Trilinear Interpolation with Spherical Harmonic-1](/assets/images/volumetric-rendering-trilinear-interpolation-spherical-harmonics.png)
![Volumetric Rendering using Trilinear Interpolation with Spherical Harmonic-2](/assets/images/volumetric-rendering-trilinear-interpolation-spherical-harmonics-box-in-box.png)

Our volumetric renderer is mostly ready; there might be some smaller changes that we will keep doing. However, the sequels will start looking at the actual reason we are looking at this paper, namely, the optimisation of the spherical harmonics to capture the voxel geometry of the scene. This will require us to first calculate loss, and that is where we will start next time.

### References

- [Spherical Harmonics](https://patapom.com/blog/SHPortal/#fn:2)
- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [Plenoxels Explained](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks)

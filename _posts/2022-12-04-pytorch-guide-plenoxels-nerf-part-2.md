---
title: "Plenoxels and Neural Radiance Fields using PyTorch: Part 2"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Neural Radiance Fields", "Machine Vision"]
draft: true
---

### Building the Voxel Data Structure

We will store opacity and the spherical harmonic coefficients.

### Incorporating the Volumetric Rendering Model

We will take a bit of a pause to understand the optical model involved in volumetric rendering since it is essential to the actual rendering and the subsequent calculation of the loss. This article follows [Optical Model for Volumetric Rendering](https://www.youtube.com/watch?v=hiaHlTLN9TE) quite a bit. Solving differential equations is involved, but for the most part, you should be able to skip to the results, if you are not interested in the gory details.

### Calculating Loss

### Incorporating Training Images

### Incorporating Trilinear Interpolation

### Optimising using RMSProp

### References

- [Camera Matrix - Kris Kitani](https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf)
- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [Plenoxels Explained](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks)
- [Optical Model for Volumetric Rendering](https://www.youtube.com/watch?v=hiaHlTLN9TE)

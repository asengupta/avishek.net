---
title: "Plenoxels and Neural Radiance Fields using PyTorch: Building a Volume Renderer"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Neural Radiance Fields", "Machine Vision"]
draft: false
---

The relevant paper is [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131). We will also use [this explanation](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks) to understand some parts of this paper a little better.

Before we get into the implementations of the paper proper, we will need a game plan. This game plan will include some theoretical background that we will have to go through to implement parts of this paper. The theoretical background will include:

- Camera Model
- Volumetric Rendering Model
- Spherical Harmonics
- Regularisation

![Voxel Cube](/assets/images/voxel-cube.png)
![Very Basic Volumetric Rendering of Cube](/assets/images/basic-volumetric-rendering-cube.png)

The following code plays around with the pinhole camera model and sets up a very basic (maybe even contrived) volumetric rendering model.

```python
{% include_absolute '/code/pytorch-learn/plenoxels/camera2.py' %}
```
